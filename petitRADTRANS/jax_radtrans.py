import numpy as np
import jax.numpy as jnp
from numba import njit
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst


pressures = 1e6*np.logspace(-6,3,20)
grav = 10**4
total_kap = np.ones((16,50,2,20))

@jit
def compute_optical_depths(reference_gravity,pressures,opacities):
    """_summary_

    Args:
        reference_gravity (float): Surface gravity in cgs (cm/s^2)
        pressure (ndarray): 1D array of pressures in pascal
        opacities (ndarray): kappa, has shape (g_len, freq_len, press_len)
                                 This is the sum of all species' opacities, at 
                                 every c-k g coordinate, frequency and pressure.

    Returns:
        transmission: Tau, a 1D array of transmission coefficients for g, 
                      frequency and for every pressure level in the atmosphere.
    """
    optical_depth = jnp.zeros(opacities.shape)
    for i_struc in range(1,opacities.shape[3]):
        optical_depth = optical_depth.at[i_struc].set(optical_depth[i_struc-1] +\
                       (opacities[i_struc]+opacities[i_struc-1])\
                        /2.0/reference_gravity*(pressures[i_struc]-pressures[i_struc-1]))
    return optical_depth
@jit
def compute_species_optical_depths(reference_gravity, pressures, opacities):
    """
    Calculate optical depths with 2nd order accuracy for clouds using JAX.

    Args:
        reference_gravity (DeviceArray): Reference gravity (scalar).
        pressures (DeviceArray): Pressures at each layer with shape (n_layers,).
        opacities (DeviceArray): Opacities with shape (n_g, n_frequencies, n_species, n_layers).

    Returns:
        DeviceArray: Optical depths with shape (n_g, n_frequencies, n_species, n_layers).
    """

    n_layers = pressures.shape[0]

    # Calculate pressure differences
    pressure_diffs = pressures[1:] - pressures[:-1]  

    # Calculate average opacities between layers (vectorized)
    avg_opacities = (opacities[..., 1:] + opacities[..., :-1]) / 2.0  

    # Calculate optical depth increments (vectorized)
    d_optical_depths = avg_opacities * pressure_diffs[None, None, None, :] / reference_gravity 

    # Integrate optical depths cumulatively (vectorized)
    optical_depths = jnp.concatenate(
        [
            jnp.zeros_like(opacities[..., :1]),  # Initial optical depth at the top is zero
            jnp.cumsum(d_optical_depths, axis=-1) 
        ],
        axis=-1
    )

    return optical_depths

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
    kappa_planck = calc_planck_opa(total_opacity_use[:,:,:], 
                                   border_freqs, 
                                   temperature,
                                   w_gauss)
    return kappa_planck
planck_opacity = jit(_planck_opacity,static_argnums=(4))

@jit
def calc_planck_opa(total_opacity,
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
def flux_ck_contribution(frequency,
                         temperature,
                         scat_angle,
                         w_gauss_scat_angle,
                         w_gauss):
    """_summary_

    Args:
        frequency (_type_): _description_
        transmission (_type_): _description_
        temperature (_type_): _description_
        scat_angle (_type_): mu, the cosine of the scattering angle
        w_gauss_scat_angle (_type_): _description_
        w_gauss (_type_): _description_

    Returns:
        _type_: _description_
    """

    struct_shape = (frequency.shape[0],temperature.shape[0])
    flux = jnp.zeros(struct_shape[0])
    contr_em = jnp.zeros(struct_shape)

    for i_scat_angle in range(scat_angle.shape[0]):
        # Transmissions for a given incidence angle
        transm_mu = jnp.exp(-transmission/scat_angle[i_scat_angle])

        # Flux contribution from given scat_angle-angle

        #mean_transm = np.zeros((struct_shape[0],transmission.shape[2],struct_shape[1])
        mean_transm = jnp.sum(transm_mu* w_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        r = planck_f(temperature,frequency)
        # Do the actual radiative transport
        flux_mu = jnp.sum((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:])/2.0,axis=1)      
        flux_mu = jnp.add(flux_mu,r[:,struct_shape[1]]*transm_all[:,struct_shape[1]])

        contr_em = contr_em.at[:,:-1].add((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:]) *\
                                           scat_angle[i_scat_angle]*w_gauss_scat_angle[i_scat_angle])     
        contr_em = contr_em.at[:,-1].add(2.0*r[:,struct_shape[1]] * transm_all[:,struct_shape[1]] *\
                                          scat_angle[i_scat_angle]*w_gauss_scat_angle[i_scat_angle])

        # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        flux = jnp.add(flux, flux_mu/2.0*scat_angle[i_scat_angle]*w_gauss_scat_angle[i_scat_angle])

    #Normalization
    flux = flux*4.0*jnp.pi
    contr_em = contr_em.T
    contr_em = jnp.divide(contr_em,jnp.sum(contr_em,axis=0))
    return flux, contr_em

@jit
def flux_ck(frequency,
            temperature,
            w_gauss,
            scat_angle,
            w_gauss_scat_angle,
            optical_depths):
    """_summary_

    Args:
        frequency (ndarray): shape (len_freq)
        transmission (ndarray): shape (len_g, len_freq, len_spec, len_temp))
        temperature (ndarray): shape (len_temp)
        scat_angle (ndarray): shape (len_scatter)
        w_gauss_scat_angle (ndarray): shape (len_scatter)
        w_gauss (ndarray): shape (len_scatter)

    Returns:
        flux: shape (len_freq)
    """
    struct_shape = (frequency.shape[0],temperature.shape[0])
    flux = jnp.zeros(struct_shape[0])
    for i_scat_angle in range(scat_angle.shape[0]):
        # Transmissions for a given incidence angle
        transm_mu= jnp.exp(-optical_depths/scat_angle[i_scat_angle])
        mean_transm = jnp.sum(transm_mu* w_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        r = planck_f(temperature,frequency)
        # Do the actual radiative transport
        flux_mu = jnp.sum((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:])*0.5,axis=1)      
        flux_mu = jnp.add(flux_mu,r[:,struct_shape[1]]*transm_all[:,struct_shape[1]])
        #print(flux_mu.shape)
        # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        flux = jnp.add(flux,flux_mu*0.5*scat_angle[i_scat_angle]*w_gauss_scat_angle[i_scat_angle])
    #Normalization
    return flux*4.0*jnp.pi, None

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





def feautrier_rad_trans(border_freqs,
                        tau_approx_scat,
                        temperature,
                        mu,
                        w_gauss_mu,
                        w_gauss_ck,
                        photon_destruct_in,
                        surf_refl,
                        surf_emi,
                        I_star_0,
                        geom,
                        mu_star):
    #tau_approx_scat = (N_g,freq_len_bounded,struc_len)
    # new = (freq_len_bounded,struc_len,N_g)
    # Lengths of arrays
    freq_len_bounded = border_freqs.shape[0] - 1
    struc_len = temperature.shape[0] 
    g_len = w_gauss_ck.shape[0]
    N_mu = mu.shape[0]

    # Definitions
    #I_plus_surface = jnp.zeros((N_mu,g_len,freq_len_bounded))
    GCM_read = False
    iter_scat = 5
    I_plus_surface = jnp.zeros((freq_len_bounded,g_len,N_mu))
    source_planet_scat_n = jnp.zeros((freq_len_bounded,struc_len,g_len))
    source_planet_scat_n1 = jnp.zeros((freq_len_bounded,struc_len,g_len))
    source_planet_scat_n2 = jnp.zeros((freq_len_bounded,struc_len,g_len))
    source_planet_scat_n3 = jnp.zeros((freq_len_bounded,struc_len,g_len))

    flux = jnp.zeros(freq_len_bounded)
    source = jnp.zeros((freq_len_bounded,struc_len,g_len))

    photon_destruct = photon_destruct_in

    if  geom ==0: 
        I_star_calc = 0.5* jnp.abs(I_star_0[:,None,None,None])*jnp.exp(-tau_approx_scat[:,:,None,:]/mu[None,None,:,None])
        J_star_ini = jnp.sum(0.5*I_star_calc*w_gauss_mu[None,None,:,None],axis=2)
    elif geom ==1:
        I_star_calc = 0.25* jnp.abs(I_star_0[:,None,None,None])*jnp.exp(-tau_approx_scat[:,:,None,:]/mu[None,None,:,None])
        J_star_ini = jnp.sum(0.5*I_star_calc*w_gauss_mu[None,None,:,None],axis=2)
    elif geom==2:
        J_star_ini = jnp.abs(I_star_0/4.*jnp.exp(-tau_approx_scat[i,:,:]/mu_star))
    else:
        print('Invalid geometry')

    r = vmap(planck_f_lr,in_axes=(None,0,0))(temperature,border_freqs[:-1],border_freqs[1:])
    plancks = r
    source = photon_destruct[:,:,:]*plancks[:,:,None] +  (1.0-photon_destruct[:,:,:])*J_star_ini[:,:,:]
    photon_destruct = jnp.where(photon_destruct_in<1e-10,1e-10,photon_destruct)

    conv_val = 1.0
    i_iter_scat = 0
    state = (i_iter_scat,
            flux,
            source,
            temperature,
            border_freqs,
            plancks,
            photon_destruct,
            J_star_ini,
            tau_approx_scat,
            surf_emi,
            surf_refl,
            mu,
            w_gauss_mu,
            I_star_calc,
            I_plus_surface,
            mu_star,
            w_gauss_ck,
            source_planet_scat_n,
            source_planet_scat_n1,
            source_planet_scat_n2,
            source_planet_scat_n3,
            conv_val)
    
    state = lax.while_loop(cond_fun, single_scat_iteration, state)

    i_iter_scat,flux,source,temperature,border_freqs,plancks,photon_destruct,\
    J_star_ini,tau_approx_scat,surf_emi,surf_refl,mu,w_gauss_mu,\
    I_star_calc,I_plus_surface,mu_star,w_gauss_ck,source_planet_scat_n,source_planet_scat_n1,\
        source_planet_scat_n2,source_planet_scat_n3,conv_val = state
    
    # Calculate the contribution function.
    # Copied from flux_ck, here using "source" as the source function
    # (before it was the Planck function).
    struct_shape = (freq_len_bounded,temperature.shape[0])
    contr_em = jnp.zeros(struct_shape)

    # Let's ignore the contribution function for now.
    if False:
        for i_mu in range(mu.shape[0]):
            # Transmissions for a given incidence angle
            transm_mu = jnp.exp(-transmission/mu[i_mu])

            # Flux contribution from given mu-angle

            #mean_transm = np.zeros((struct_shape[0],transmission.shape[2],struct_shape[1])
            mean_transm = jnp.sum(transm_mu* w_gauss[:, None, None, None], axis=0)
            transm_all = jnp.prod(mean_transm, axis=1)
            r = jnp.sum(source*w_gauss_ck[:,None,None,None],axis = 0)
            #r = planck_f(temperature,frequency)
            # Do the actual radiative transport
            contr_em = contr_em.at[:,:-1].add((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:]) *\
                                            mu[i_mu]*w_gauss_mu[i_mu])     
            contr_em = contr_em.at[:,-1].add(2.0*r[:,struct_shape[1]] * transm_all[:,struct_shape[1]] *\
                                            mu[i_mu]*w_gauss_mu[i_mu])

            # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        #Normalization
        contr_em = contr_em.T
        contr_em = jnp.divide(contr_em,jnp.sum(contr_em,axis=0))
    return flux, source
py_feautrier_rad_trans = jit(feautrier_rad_trans, static_argnames=('geom'))

def cond_fun(state):
    i_iter_scat,flux,source,temperature,border_freqs,plancks,photon_destruct,\
    J_star_ini,tau_approx_scat,surf_emi,surf_refl,mu,w_gauss_mu,\
    I_star_calc,I_plus_surface,mu_star,w_gauss_ck,source_planet_scat_n,source_planet_scat_n1,\
        source_planet_scat_n2,source_planet_scat_n3,conv_val = state
    return (conv_val > 1e-3 ) & (i_iter_scat < 9)

@jit
def single_scat_iteration(state):
    i_iter_scat,flux,source,temperature,border_freqs,plancks,photon_destruct,\
    J_star_ini,tau_approx_scat,surf_emi,surf_refl,mu,w_gauss_mu,\
    I_star_calc,I_plus_surface,mu_star,w_gauss_ck,source_planet_scat_n,source_planet_scat_n1,\
        source_planet_scat_n2,source_planet_scat_n3,conv_val = state

    freq_len_bounded=tau_approx_scat.shape[0]
    struc_len = tau_approx_scat.shape[1]
    g_len = tau_approx_scat.shape[2]
    N_mu = mu.shape[0]

    geom = 1
    I_minus = 0.0
    inv_del_tau_min = 1e10
    GCM_read = False

    a = jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    b = jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    c = jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    I_J=jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    I_H=jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    
    lambda_loc = jnp.zeros((freq_len_bounded,struc_len,g_len,N_mu))
    J_planet_scat = jnp.zeros((freq_len_bounded,struc_len,g_len))
    I_GCM = jnp.zeros((N_mu,freq_len_bounded))

    flux_old = flux
 
    r = lax.cond(i_iter_scat == 0,
                lambda : jnp.repeat(plancks[:, :, jnp.newaxis], 16, axis=2),
                lambda : source)
    
    f1 = 2.0*mu[None,None,None,:]/(tau_approx_scat[:,1:,:,None]-tau_approx_scat[:,:-1,:,None])
    f2 = mu[None,None,None,:]/(tau_approx_scat[:,2:,:,None]-tau_approx_scat[:,1:-1,:,None])
    f3 = mu[None,None,None,:]/(tau_approx_scat[:,1:-1,:,None]-tau_approx_scat[:,:-2,:,None])

    ############### EMISSION ONLY TERM ################
    print(surf_emi.shape,I_plus_surface.shape,plancks.shape,mu.shape,w_gauss_mu.shape)
    I_minus = surf_emi[:,None]*plancks[:,-1,None] \
        + surf_refl[:,None] * 2.0 * jnp.sum(I_plus_surface[:, :, :] * mu[None,None,:] * w_gauss_mu[None,None,:],axis=2)
    
    ############### SURFACE SCATTERING ################
    print(I_minus.shape,I_star_calc.shape)
    if  geom != 2:
            I_minus = I_minus + surf_refl[:,None] \
                * 2.0 * jnp.sum(I_star_calc[:, -1,:,:] * mu[None,:,None] * w_gauss_mu[None,:,None],axis=1)
    else:
        I_minus = I_minus + surf_refl[:,None] *J_star_ini[:,-1,:] * 4.0 * mu_star
    
    f1 = jnp.where(f1 > 0.5*inv_del_tau_min, 0.5*inv_del_tau_min, f1)
    f1 = jnp.where(f1 != f1, 0.5*inv_del_tau_min, f1)
    f2 = jnp.where(f2 > inv_del_tau_min, inv_del_tau_min, f2)
    f2 = jnp.where(f2 != f2, inv_del_tau_min, f2)
    f3 = jnp.where(f3 > inv_del_tau_min, inv_del_tau_min, f3)
    f3 = jnp.where(f3 != f3, inv_del_tau_min, f3)
    
    a = a.at[:,0,:,:].set(0.0)
    b = b.at[:,0,:,:].set(1.0 + (2.0 * f1[:,0,:,:] * (1.0 + f1[:,0,:,:])))
    c = c.at[:,0,:,:].set(-2.0*f1[:,0,:,:]**2.0)

    a = a.at[:,1:-1,:,:].set(-f1[:,1:,:,:]*f3[:,:,:,:])
    b = b.at[:,1:-1,:,:].set(1.0 + (f1[:,1:,:,:]*(f2[:,:,:,:]+f3[:,:,:,:])))
    c = c.at[:,1:-1,:,:].set(-f1[:,1:,:,:]*f2[:,:,:,:])

    a = a.at[:,-1,:,:].set(0.0)
    b = b.at[:,-1,:,:].set(1.0)
    c = c.at[:,-1,:,:].set(0.0)

    lambda_loc = lambda_loc.at[:,0,:,:].add(w_gauss_mu[None,None,:]/(1.0 + 2.0 * f1[:,0,:,:] * (1.0 + f1[:,0,:,:])))
    lambda_loc = lambda_loc.at[:,1:-1,:,:].add(w_gauss_mu[None,None,:]/(b[:,1:-1,:,:]))
    lambda_loc = lambda_loc.at[:,-1,:,:].add(w_gauss_mu[None,None,:]/(1.0 + 2.0*f1[:,-1,:,:]**2.0))

    I_plus = I_plus_surface 
    I_J= wrap_tridag(a, b, c, r, I_plus, I_minus)
    I_J = jnp.transpose(I_J,axes=(0,3,2,1))
    I_H = -I_J 
    deriv1 = f1[:,1:,:,:]*(I_J[:,2:,:,:]-I_J[:,1:-1,:,:])
    deriv2 = f2[:,:,:,:]*(I_J[:,1:-1,:,:]-I_J[:,:-2,:,:])
    I_H = I_H.at[:,1:-1,:,:].set(-(deriv1+deriv2)*0.5)
    I_plus_surface = I_plus_surface.at[:, :, :].set(I_J[:,-1,:,:]  - deriv1[:,-1,:,:])
    I_H = I_H.at[:,-1,:,:].set(0.0)

    J_bol_g = jnp.sum(I_J * w_gauss_mu[None,None,None,:],axis=3)
    flux = jnp.sum(jnp.sum(-I_H[:,0,:,:]*mu[None,None,:]*\
                                           4.0 *jnp.pi * w_gauss_ck[None,:,None] * w_gauss_mu[None,None,:],axis=2),axis=1)

    # Save angle-dependent surface flux
    #if GCM_read:
    #    for j in range(N_mu):
    #        I_GCM = I_GCM.at[:,j].add( - 2.0*I_H[1,j]*w_gauss_ck[None,l])
    J_planet_scat = J_planet_scat.at[:,:,:].set(J_bol_g)

    photon_destruct = jnp.where(photon_destruct < 1e-10, 1e-10, photon_destruct)
    temp = vmap(planck_f_lr,in_axes = (None,0,0))(temperature[1:-1],border_freqs[:-1],border_freqs[1:])
    r = r.at[:,1:-1,:].set(jnp.repeat(temp[:, :, jnp.newaxis], 16, axis=2))
    source = source.at[:,:,:].set(jnp.sum((photon_destruct[:,:,:,None]*r[:,:,:,None]+\
                                            (1.0-photon_destruct[:,:,:,None])*(J_star_ini[:,:,:,None]+\
                                            J_planet_scat[:,:,:,None]-\
                                            lambda_loc[:,:,:,:]*source[:,:,:,None]))/ \
                                            (1.0-(1.0-photon_destruct[:,:,:,None])*lambda_loc[:,:,:,:]),axis=3))
    #for i in range(freq_len_bounded):
    #   #r = r.at[i,1:struc_len-1].set(planck_f_lr(temperature[1:struc_len-1],border_freqs[i],border_freqs[i+1]))
    #    source = source.at[i,:,:].set(jnp.sum((photon_destruct[i,:,:,None]*r[i,:,None,None]+\
    #                                            (1.0-photon_destruct[i,:,:,None])* \
    #                                            (J_star_ini[i,:,:,None]+\
    #                                            J_planet_scat[i,:,:,None]-\
    #                                            lambda_loc[i,:,:,:]*\
    #                                            source[i,:,:,None])) / \
    #                                            (1.0-(1.0-photon_destruct[i,:,:,None])*lambda_loc[i,:,:,:]),axis=2))
    source_planet_scat_n3 = source_planet_scat_n2
    source_planet_scat_n2 = source_planet_scat_n1
    source_planet_scat_n1 = source_planet_scat_n
    source_planet_scat_n  = source

    source = lax.cond(i_iter_scat%4 ==0,
                     lambda: NG_source_approx(source_planet_scat_n,
                                              source_planet_scat_n1,
                                              source_planet_scat_n2,
                                              source_planet_scat_n3,
                                              source),
                     lambda : source)
    
    conv_val = jnp.max(jnp.abs((flux-flux_old)/flux))
    i_iter_scat += 1

    state = (i_iter_scat,flux,source,temperature,border_freqs,plancks,photon_destruct,\
    J_star_ini,tau_approx_scat,surf_emi,surf_refl,mu,w_gauss_mu,\
    I_star_calc,I_plus_surface,mu_star,w_gauss_ck,source_planet_scat_n,source_planet_scat_n1,\
    source_planet_scat_n2,source_planet_scat_n3,conv_val)

    return state

@jit
def wrap_tridag(a, b, c, r, I_plus, I_minus):
    # Set initial value of r
    # freq, struct, glen, nmu
    r_updated = jnp.zeros((r.shape[0],r.shape[1],r.shape[2],I_plus.shape[2]))
    r_updated = r_updated.at[:,:,:,:].set(r[:,:,:,None])
    r_updated = r_updated.at[:,-1,:,:].set(0.5 * (I_plus + I_minus[:, :, None]))
    # Calculate I_J and I_H using vectorized computation
    I_J = vmap(vmap(vmap(wrap_callback, 
                    in_axes=(1, 1, 1, 1)),
                    in_axes=(2, 2, 2, 2)), 
                    in_axes=(0, 0, 0, 0))(a, b, c, r_updated)
    return I_J

def wrap_callback(a, b, c, r):
    result_shape = ShapeDtypeStruct(r.shape, r.dtype)
    return pure_callback(fs.tridag_own, result_shape, a,b,c,r)

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
def NG_source_approx(source_n,source_n1,source_n2,source_n3,source):
    #source_buff = jnp.zeros((freq_len_p_1,struc_len,N_g))
    Q1 = source_n - (2.0*source_n1) + source_n2
    Q2 = source_n - source_n1 - source_n2 + source_n3
    Q3 = source_n - source_n1
    Q1 = Q1.at[:,0,:].set(0.0)
    Q2 = Q2.at[:,0,:].set(0.0)
    Q3 = Q3.at[:,0,:].set(0.0)
    A1 = jnp.sum(Q1*Q1,axis = 1)
    A2 = jnp.sum(Q2*Q1,axis = 1)
    B1 = jnp.sum(Q1*Q2,axis = 1)
    B2 = jnp.sum(Q2*Q2,axis = 1)
    C1 = jnp.sum(Q1*Q3,axis = 1)
    C2 = jnp.sum(Q2*Q3,axis = 1)

    a = (C1*B2-C2*B1)/(A1*B2-A2*B1)
    b = (C2*A1-C1*A2)/(A1*B2-A2*B1)

    temp_buff = (1.0-a[:,None,:]-b[:,None,:])*source_n + a[:,None,:]*source_n1 + b[:,None,:]*source_n2
    temp_buff = jnp.where(temp_buff <= 0.0, 0.0, temp_buff)
    source = temp_buff
    return source

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
# https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
@jit 
def tridag_own(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(jnp.array, (a, b, c, d)) # copy arrays
    ac,bc,cc,dc = lax.fori_loop(1,nf,internal_tridag,(ac,bc,cc,dc))     
    xc = bc
    xc = xc.at[-1].set(dc[-1]/bc[-1])
    xcr = jnp.flip(xc)
    bcr = jnp.flip(bc)
    ccr = jnp.flip(cc)
    dcr = jnp.flip(dc)
    xcr,bcr,ccr,dcr = lax.fori_loop(0,nf-1,internal_tridag_reverse,(xcr,bcr,ccr,dcr))
    return jnp.flip(xcr)

def internal_tridag_reverse(il,val):
    xc,bc,cc,dc = val
    xc = xc.at[il].set((dc[il]-cc[il]*xc[il-1])/bc[il])
    return (xc,bc,cc,dc)

def internal_tridag(it,val):
    ac,bc,cc,dc = val
    mc = ac[it]/bc[it-1]
    bc = bc.at[it].add(- mc*cc[it-1])
    dc = dc.at[it].add(- mc*dc[it-1])
    return (ac,bc,cc,dc)
