import numpy as np
import jax.numpy as jnp
from numba import njit
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.fortran_src.fortran_core.math as fmath

"""
JAX implementation of fortran_core.physics
"""

@jit
def planck_f_lr(T,nul,nur):
    # calculate planck function between two frequencies
    B_nu = 0.0
    # Take mean using Boole's method
    nu_large = jnp.maximum(nul,nur)
    nu_small = jnp.minimum(nul,nur)
    nu1 = nu_small
    nu2 = nu_small+1.0*(nu_large-nu_small)/4.0
    nu3 = nu_small+2.0*(nu_large-nu_small)/4.0
    nu4 = nu_small+3.0*(nu_large-nu_small)/4.0
    nu5 = nu_large
    diff_nu = nu2-nu1
    B_nu = B_nu + 1.0/90.0*( \
        7.0* 2.0*cst.h*nu1**3.0/cst.c**2.0/(jnp.exp(cst.h*nu1/(cst.kB*T))-1.0) +\
        32.0*2.0*cst.h*nu2**3.0/cst.c**2.0/(jnp.exp(cst.h*nu2/(cst.kB*T))-1.0) +\
        12.0*2.0*cst.h*nu3**3.0/cst.c**2.0/(jnp.exp(cst.h*nu3/(cst.kB*T))-1.0) +\
        32.0*2.0*cst.h*nu4**3.0/cst.c**2.0/(jnp.exp(cst.h*nu4/(cst.kB*T))-1.0) +\
        7.0* 2.0*cst.h*nu5**3.0/cst.c**2.0/(jnp.exp(cst.h*nu5/(cst.kB*T))-1.0))
    return B_nu



@jit
def calc_photon_destruction_prob(total_opacity_in,
                                 continuum_opacity_scat_emis):
    """_summary_

    Args:
        gravity (float): Surface gravity in cgs (cm/s^2)
        pressure (ndarray):1D array of pressures in pascal
        total_opacity_in (ndarray): kappa, has shape (g_len, freq_len, press_len)
                                    This is the sum of all species' opacities, at 
                                    every c-k g coordinate, frequency and pressure.
        continuum_opacity_scat_emis (ndarray): The continuum opacity, which is constant 
                                                in g, for every frequency and pressure.
                                                This includes clouds, Rayleigh and CIA
                                                opacities.

    Returns:
        photon_destruction_prob: probability that the photon DOESN'T leave the atmospheric layer
    """
    total_opacity = total_opacity_in[:,:,0,:]
    g_len = total_opacity_in.shape[0]
    photon_destruction_prob = jnp.ones(total_opacity.shape)
    for i_g in range(g_len):
        total_opacity = total_opacity.at[i_g].set(total_opacity[i_g] + continuum_opacity_scat_emis)
        photon_destruction_prob = photon_destruction_prob.at[i_g].set(continuum_opacity_scat_emis/total_opacity[i_g])
    photon_destruction_prob = 1 - photon_destruction_prob
    return photon_destruction_prob

@jit
def transmission_g_tot_ck_scat(gravity,
                               pressure,
                               total_opacity_in,
                               continuum_opacity_scattering_emis):
    """_summary_

    Args:
        gravity (_type_): _description_
        pressure (_type_): _description_
        total_opacity_in (_type_): _description_
        continuum_opacity_scattering_emis (_type_): _description_

    Returns:
        _type_: _description_
    """
    photon_destruction_prob = calc_photon_destruction_prob(total_opacity_in,
                                                           continuum_opacity_scattering_emis)
    transmission = compute_optical_depths(gravity,
                                       pressure,
                                       total_opacity_in)
    return transmission, photon_destruction_prob

@jit
def compute_rosseland_opacities_core(clouds_final_absorption_opacities,
                                     frequencies_bin_edges,
                                     temperature,
                                     weights_gauss):
    """Calculate the Rosseland mean opacity as a function of temperature.

    Args:
        clouds_final_absorption_opacities (DeviceArray): Absorption opacities with shape (N_g, N_frequency_edges - 1, N_temperature).
        frequencies_bin_edges (DeviceArray): Frequency bin edges with shape (N_frequency_edges,).
        temperature (DeviceArray): Temperature with shape (N_temperature,).
        weights_gauss (DeviceArray): Gauss weights with shape (N_g,).

    Returns:
        DeviceArray: Rosseland opacity with shape (N_temperature,).
    """
    diff_freqs = frequencies_bin_edges[:-1] - frequencies_bin_edges[1:]
    B_nu_dT = compute_planck_function_temperature_derivative(temperature, frequencies_bin_edges)

    # Precompute for efficiency
    inv_absorption_opacities = 1.0 / clouds_final_absorption_opacities

    # Use einsum for efficient contraction
    weighted_inv_opacities = jnp.einsum('g,gfn->fn', weights_gauss, inv_absorption_opacities)

    kappa_rosse = jnp.sum(B_nu_dT * weighted_inv_opacities * diff_freqs[:, None], axis=0)
    numerator = jnp.sum(B_nu_dT * diff_freqs[:, None], axis=0)

    return numerator / kappa_rosse


def compute_star_planck_function_integral():
    # TODO
    return 

def compute_rosseland_opacity(opacities, 
                              temperature, 
                              weights_gauss, 
                              frequencies_bin_edges, 
                              scattering_in_emission, 
                              continuum_opacities_scattering):
    """_summary_

    Args:
        total_opacity (_type_): _description_
        temperature (_type_): _description_
        w_gauss (_type_): _description_
        border_freqs (_type_): _description_
        do_scat_emis (_type_): _description_
        continuum_opacity_scattering_emis (_type_): _description_

    Returns:
        _type_: _description_
    """

    if scattering_in_emission:
        opacities_use = opacities + continuum_opacities_scattering[None,:,:]
    else:
        opacities_use = opacities

    kappa_rosse = compute_rosseland_opacities_core(opacities_use, 
                                                   frequencies_bin_edges, 
                                                   temperature,
                                                   weights_gauss)
    return kappa_rosse
rosseland_opacity = jit(compute_rosseland_opacity,static_argnums=(4))

def _planck_opacity(total_opacity, 
                      temperature, 
                      w_gauss, 
                      border_freqs,
                      do_scat_emis, 
                      continuum_opacity_scattering_emis):
    """_summary_

    Args:
        total_opacity (_type_): _description_
        temperature (_type_): _description_
        w_gauss (_type_): _description_
        border_freqs (_type_): _description_
        do_scat_emis (_type_): _description_
        continuum_opacity_scattering_emis (_type_): _description_

    Returns:
        _type_: _description_
    """
    if do_scat_emis:
        total_opacity_use = total_opacity + continuum_opacity_scattering_emis[None,:,:]
    else:
        total_opacity_use = total_opacity
    kappa_planck = compute_planck_opacities(total_opacity_use[:,:,:], 
                                   border_freqs, 
                                   temperature,
                                   w_gauss)
    return kappa_planck
planck_opacity = jit(_planck_opacity,static_argnums=(4))

@jit
def compute_planck_opacities(total_opacity,
                    border_freqs,
                    temperature,
                    w_gauss):
    """_summary_

    Args:
        total_opacity (_type_): _description_
        border_freqs (_type_): _description_
        temperature (_type_): _description_
        w_gauss (_type_): _description_

    Returns:
        _type_: _description_
    """
    diff_freqs = jnp.diff(border_freqs)
    B_nu = star_planck(temperature,border_freqs)

    kappa_planck= jnp.sum(B_nu *jnp.sum(w_gauss[:,None,None]/total_opacity[:,:-1,:],axis=0) *diff_freqs[:,None],axis=0)
    norm = jnp.sum(B_nu * diff_freqs[:,None], axis = 0)
    return kappa_planck / norm


@jit
def planck_f(T,nu):
    """
    Calculates B_ν https://en.wikipedia.org/wiki/Planck%27s_law
    For every temperature and frequency provided.

    Args:
        T (ndarray): 1D array of temperature in Kelvin
        nu (ndarray): 1D array of frequencies in Hertz

    Returns:
        B_nu (ndarray): (len(nu),len(T)) matrix of the spectral radiance (ie flux) as calculated from 
                        Planck's law for each frequency and temperature.
    """
    T_use = jnp.tile(T,(nu.shape[0],1))
    buffer = 2.0*cst.h*nu[:,None]**3/cst.c**2
    B_nu = buffer / (jnp.exp(cst.h*nu[:,None]/(cst.kB*T_use))-1.0)
    return B_nu

@jit
def compute_planck_function_temperature_derivative(T, frequency_bin_edges):
    """
    Calculates the gradient of B_ν (Planck's Law) with respect to temperature for 
    each temperature and frequency provided using JAX.

    Args:
        T (DeviceArray): 1D JAX array of temperatures in Kelvin.
        frequency_bin_edges (DeviceArray): 1D JAX array of frequency bin edges in Hertz.

    Returns:
        B_nu_dT (DeviceArray): (len(T), len(nu)) JAX array of the derivative of the spectral radiance.
    """
    nu = (frequency_bin_edges[:-1] + frequency_bin_edges[1:]) * 0.5  # Calculate midpoints directly

    # Constants for efficiency (avoid repeated calculations within the loop)
    h_over_kb = cst.h / cst.kB
    two_h_over_c2 = 2.0 * cst.h**2 / cst.c**2
    
    # Vectorized calculation of exp_terms and denominator for all T and nu
    exp_arg = (h_over_kb * jnp.outer(0.5/T, nu))
    exp_term_pos = jnp.exp(exp_arg) 
    exp_term_neg = jnp.exp(-exp_arg)

    denominator = (exp_term_pos - exp_term_neg)**2 * jnp.outer(cst.kB * T**2, jnp.ones_like(nu)) 

    # Calculate B_nu_dT using outer product for broadcasting, then transpose
    B_nu_dT = (two_h_over_c2 * jnp.outer(jnp.ones_like(T), nu**4) / denominator).T  # Correct broadcasting
    return B_nu_dT

def compute_planck_function_integral():
    #TODO
    return

@jit
def star_planck(T,nu):
    """
    Calculates B_ν https://en.wikipedia.org/wiki/Planck%27s_law
    For every temperature and frequency provided. 
    Approximates the integral using Boole's method

    Args:
        T (ndarray): 1D array of temperature in Kelvin
        nu (ndarray): 1D array of frequencies in Hertz

    Returns:
        B_nu (ndarray): (len(nu),len(T)) matrix of the spectral radiance (ie flux) as calculated from 
                        Planck's law for each frequency and temperature.
    """
    frequency_len = nu.shape[0]
    bnus = []
    # Take mean using Boole's method
    for i in range(frequency_len-1):
        nu_large = jnp.max(nu[i:i+1])
        nu_small = jnp.min(nu[i:i+1])
        nu1 = nu_small
        nu2 = nu_small+1.0*(nu_large-nu_small)/4.0
        nu3 = nu_small+2.0*(nu_large-nu_small)/4.0
        nu4 = nu_small+3.0*(nu_large-nu_small)/4.0
        nu5 = nu_large
        val = 1.0/90.0*\
            ((7.0* 2.0*cst.h*nu1**3)/(cst.c**2.0*(jnp.exp(cst.h*nu1/(cst.kB*T))-1.0)) +\
            (32.0*2.0*cst.h*nu2**3)/(cst.c**2.0*(jnp.exp(cst.h*nu2/(cst.kB*T))-1.0)) +\
            (12.0*2.0*cst.h*nu3**3)/(cst.c**2.0*(jnp.exp(cst.h*nu3/(cst.kB*T))-1.0) )+\
            (32.0*2.0*cst.h*nu4**3)/(cst.c**2.0*(jnp.exp(cst.h*nu4/(cst.kB*T))-1.0)) +\
            (7.0* 2.0*cst.h*nu5**3)/(cst.c**2.0*(jnp.exp(cst.h*nu5/(cst.kB*T))-1.0)))
        bnus.append(val)
    B_nu = jnp.array(bnus)
    return B_nu