"""
JAX implementation of fortran_core.physics
"""
import jax.numpy as jnp
import warnings
from jax import jit, vmap, lax, debug
from functools import partial

import petitRADTRANS.physical_constants as cst
# import petitRADTRANS.fortran_radtrans_core import math as fmath

@jit
def compute_planck_function_integral_old(T, nul, nur):
    # TODO this might not be used anymore.
    # calculate planck function between two frequencies
    B_nu = 0.0
    # Take mean using Boole's method
    nu_large = jnp.maximum(nul, nur)
    nu_small = jnp.minimum(nul, nur)
    nu1 = nu_small
    nu2 = nu_small+1.0*(nu_large-nu_small)/4.0
    nu3 = nu_small+2.0*(nu_large-nu_small)/4.0
    nu4 = nu_small+3.0*(nu_large-nu_small)/4.0
    nu5 = nu_large
    diff_nu = nu2-nu1
    B_nu = B_nu + 1.0/90.0*(
        7.0 * 2.0 * cst.h * nu1**3.0 / cst.c**2.0 / (jnp.exp(cst.h * nu1 / (cst.kB * T)) - 1.0) +
        32.0 * 2.0 * cst.h * nu2**3.0 / cst.c**2.0 / (jnp.exp(cst.h * nu2 / (cst.kB * T)) - 1.0) +
        12.0 * 2.0 * cst.h * nu3**3.0 / cst.c**2.0 / (jnp.exp(cst.h * nu3 / (cst.kB * T)) - 1.0) +
        32.0 * 2.0 * cst.h * nu4**3.0 / cst.c**2.0 / (jnp.exp(cst.h * nu4 / (cst.kB * T)) - 1.0) +
        7.0 * 2.0 * cst.h * nu5**3.0 / cst.c**2.0 / (jnp.exp(cst.h * nu5 / (cst.kB * T)) - 1.0))
    return B_nu


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


def compute_star_planck_function_integral(temperatures, frequencies):
    """
    Not clear what this function is actually doing...
    TODO check what the difference between the different integral functions is
    """
    return compute_planck_function_integral(temperatures, frequencies)


@partial(jit, static_argnums=(4,))
def compute_rosseland_opacities(
        opacities,
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
        opacities_use = opacities + continuum_opacities_scattering[None, :, :]
    else:
        opacities_use = opacities

    kappa_rosse = compute_rosseland_opacities_core(opacities_use,
                                                   frequencies_bin_edges,
                                                   temperature,
                                                   weights_gauss)
    return kappa_rosse


@jit
def compute_planck_function(T, nu):
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
    T_use = jnp.tile(T, (nu.shape[0], 1))
    buffer = 2.0 * cst.h*nu[:, None]**3 / cst.c**2
    B_nu = buffer / (jnp.exp(cst.h * nu[:, None] / (cst.kB * T_use)) - 1.0)
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
    B_nu_dT = (two_h_over_c2 * jnp.outer(jnp.ones_like(T), nu**4) / denominator).T
    return B_nu_dT


def compute_planck_function_integral(temperatures, frequencies):
    """
    Compute mean using Boole's method
    TODO: possibly a faster integration method?
    """
    def reverse_frequencies(temperatures, frequencies):
        warnings.warn("Frequencies are not in descending order. This may lead to incorrect results.")
        return _compute_planck_function_integral(temperatures, jnp.flip(frequencies))
    return lax.cond(
        frequencies[1] < frequencies[0],
        _compute_planck_function_integral,
        reverse_frequencies,
        temperatures,
        frequencies
    )


@jit
def _compute_planck_function_integral(temperatures, frequencies):
    cst_c2 = cst.c ** 2.0
    cst_h_c2 = cst.h / cst_c2
    cst_h_k = cst.h / cst.kB
    factor1 = 1.0 / 90.0
    factor2 = 14.0 * cst_h_c2  # (7 * 2)
    factor3 = 64.0 * cst_h_c2  # (32 * 2)
    factor4 = 24.0 * cst_h_c2  # (12 * 2)

    planck_flux = jnp.zeros((temperatures.shape[0], frequencies.shape[0] - 1))

    cst_h_kt = cst_h_k / temperatures

    nu_large = frequencies[:-1]
    nu_small = frequencies[1:]
    diff_nu = nu_large - nu_small
    nu2 = nu_small + diff_nu * 0.25
    nu3 = nu_small + diff_nu * 0.5
    nu4 = nu_small + diff_nu * 0.75

    planck_flux = (
        factor1
        * (factor2 * nu_small[None, :] ** 3.0 / (jnp.exp(cst_h_kt[:, None] * nu_small[None, :]) - 1.0)
            + factor3 * nu2[None, :] ** 3.0 / (jnp.exp(cst_h_kt[:, None] * nu2[None, :]) - 1.0)
            + factor4 * nu3[None, :] ** 3.0 / (jnp.exp(cst_h_kt[:, None] * nu3[None, :]) - 1.0)
            + factor3 * nu4[None, :] ** 3.0 / (jnp.exp(cst_h_kt[:, None] * nu4[None, :]) - 1.0)
            + factor2 * nu_large[None, :] ** 3.0 / (jnp.exp(cst_h_kt[:, None] * nu_large[None, :]) - 1.0)))
    return planck_flux


@jit
def star_planck(T, nu):
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
    # Take mean using Boole's method implemented via lax.scan for JAX friendliness
    def _body(carry, i):
        nu_segment = nu[i:i+1]
        nu_large = jnp.max(nu_segment)
        nu_small = jnp.min(nu_segment)
        nu1 = nu_small
        nu2 = nu_small + 1.0 * (nu_large - nu_small) / 4.0
        nu3 = nu_small + 2.0 * (nu_large - nu_small) / 4.0
        nu4 = nu_small + 3.0 * (nu_large - nu_small) / 4.0
        nu5 = nu_large
        val = 1.0 / 90.0 * (
            (7.0 * 2.0 * cst.h * nu1**3) / (cst.c**2.0 * (jnp.exp(cst.h * nu1 / (cst.kB * T)) - 1.0)) +
            (32.0 * 2.0 * cst.h * nu2**3) / (cst.c**2.0 * (jnp.exp(cst.h * nu2 / (cst.kB * T)) - 1.0)) +
            (12.0 * 2.0 * cst.h * nu3**3) / (cst.c**2.0 * (jnp.exp(cst.h * nu3 / (cst.kB * T)) - 1.0)) +
            (32.0 * 2.0 * cst.h * nu4**3) / (cst.c**2.0 * (jnp.exp(cst.h * nu4 / (cst.kB * T)) - 1.0)) +
            (7.0 * 2.0 * cst.h * nu5**3) / (cst.c**2.0 * (jnp.exp(cst.h * nu5 / (cst.kB * T)) - 1.0))
        )
        return carry, val

    indices = jnp.arange(frequency_len - 1)
    _, bnus = lax.scan(_body, None, indices)
    B_nu = bnus
    return B_nu


def compute_pressure_hydrostatic_equilibrium(
            mean_molar_masses, reference_gravity, planet_radius, p0, temperature, radii, rk4=True):
    p0 = p0 * 1e6
    vs = 1. / radii
    r_pl_sq = planet_radius ** 2

    def integrand(press):
        temp = temperature(press)
        mu = mean_molar_masses(press / 1e6, temp)
        if not jnp.isscalar(mu):
            mu = mu[0]

        integral = mu * cst.amu * reference_gravity * r_pl_sq / cst.kB / temp
        return integral

    pressures = [p0]
    dvs = jnp.diff(vs)
    press1 = p0
    chi1 = jnp.log(p0)
    for dv in dvs:
        k1 = integrand(press1)

        if rk4:
            chi2 = chi1 + 0.5 * dv * k1
            press2 = jnp.exp(chi2)
            k2 = integrand(press2)
            chi3 = chi1 + 0.5 * dv * k2
            press3 = jnp.exp(chi3)
            k3 = integrand(press3)
            chi4 = chi1 + dv * k3
            press4 = jnp.exp(chi4)
            k4 = integrand(press4)
            chi1 = chi1 + 1. / 6. * (k1 + 2 * k2 + 2 * k3 + k4) * dv
        else:
            chi1 = chi1 + dv * k1

        press1 = jnp.exp(chi1)
        pressures.append(press1)

    return jnp.array(pressures) / 1e6