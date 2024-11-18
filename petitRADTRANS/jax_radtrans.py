import numpy as np
import jax.numpy as jnp
from numba import njit
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.fortran_src.fortran_core.math as fmath
import petitRADTRANS.jax_physics as jphys
import petitRADTRANS.jax_math as jmath

"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""

# Testing arrays
#pressures = 1e6*np.logspace(-6,3,20)
#grav = 10**4
#total_kap = np.ones((16,50,2,20))

def combine_ck_opacities():
    # TODO
    return

@jit
def flux_ck_contribution(frequencies, 
                         temperatures, 
                         weights_gauss,
                         emission_cos_angles,
                         emission_cos_angles_weights,
                         optical_depths):
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

    struct_shape = (frequencies.shape[0],temperatures.shape[0])
    flux = jnp.zeros(struct_shape[0])
    contr_em = jnp.zeros(struct_shape)

    for i_scat_angle in range(emission_cos_angles.shape[0]):
        # Transmissions for a given incidence angle
        transm_mu = jnp.exp(-optical_depths/emission_cos_angles[i_scat_angle])

        # Flux contribution from given scat_angle-angle

        #mean_transm = np.zeros((struct_shape[0],transmission.shape[2],struct_shape[1])
        mean_transm = jnp.sum(transm_mu* weights_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        r = jphys.planck_f(temperatures,frequencies)
        # Do the actual radiative transport
        flux_mu = jnp.sum((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:])/2.0,axis=1)      
        flux_mu = jnp.add(flux_mu,r[:,struct_shape[1]]*transm_all[:,struct_shape[1]])

        contr_em = contr_em.at[:,:-1].add((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:]) *\
                                           emission_cos_angles[i_scat_angle]*emission_cos_angles_weights[i_scat_angle])     
        contr_em = contr_em.at[:,-1].add(2.0*r[:,struct_shape[1]] * transm_all[:,struct_shape[1]] *\
                                          emission_cos_angles[i_scat_angle]*emission_cos_angles_weights[i_scat_angle])

        # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        flux = jnp.add(flux, flux_mu/2.0*emission_cos_angles[i_scat_angle]*emission_cos_angles_weights[i_scat_angle])

    #Normalization
    flux = flux*4.0*jnp.pi
    contr_em = contr_em.T
    contr_em = jnp.divide(contr_em,jnp.sum(contr_em,axis=0))
    return flux, contr_em

@jit
def compute_ck_flux(
    frequency,
    temperature,
    w_gauss,
    emission_angle,
    w_gauss_emission_angle,
    optical_depths):
    """_summary_

    Args:
        frequency (ndarray): shape (len_freq)
        transmission (ndarray): shape (len_g, len_freq, len_spec, len_temp))
        temperature (ndarray): shape (len_temp)
        emission_angle (ndarray): shape (len_scatter)
        w_gauss_emission_angle (ndarray): shape (len_scatter)
        w_gauss (ndarray): shape (len_scatter)

    Returns:
        flux: shape (len_freq)
    """
    struct_shape = (frequency.shape[0],temperature.shape[0])
    flux = jnp.zeros(struct_shape[0])
    for i_emission_angle in range(emission_angle.shape[0]):
        # Transmissions for a given incidence angle
        transm_mu= jnp.exp(-optical_depths/emission_angle[i_emission_angle])
        mean_transm = jnp.sum(transm_mu* w_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        r = jphys.planck_f(temperature,frequency)
        # Do the actual radiative transport
        flux_mu = jnp.sum((r[:,:-1]+r[:,1:])*(transm_all[:,0:-1]-transm_all[:,1:])*0.5,axis=1)      
        flux_mu = jnp.add(flux_mu,r[:,struct_shape[1]]*transm_all[:,struct_shape[1]])
        #print(flux_mu.shape)
        # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        flux = jnp.add(flux,flux_mu*0.5*emission_angle[i_emission_angle]*w_gauss_emission_angle[i_emission_angle])
    #Normalization
    # TODO Contribution function
    return flux*4.0*jnp.pi, None

def compute_cloud_hansen_opacities():
    #TODO
    return


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


def compute_cloud_particles_mean_radius():
    # TODO
    return


def compute_cloud_particles_mean_radius_hansen():
    # TODO
    return


def compute_feautrier_radiative_transfer(border_freqs,
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
    # TODO: finish adding physics to this function
    # Very incomplete

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

    r = vmap(jphys.planck_f_lr,in_axes=(None,0,0))(temperature,border_freqs[:-1],border_freqs[1:])
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
py_compute_feautrier_radiative_transfer = jit(compute_feautrier_radiative_transfer, static_argnames=('geom'))

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
    I_J= jmath.wrap_tridag(a, b, c, r, I_plus, I_minus)
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
    temp = vmap(jphys.planck_f_lr,in_axes = (None,0,0))(temperature[1:-1],border_freqs[:-1],border_freqs[1:])
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

@jit
def compute_optical_depths(
    reference_gravity,
    pressures,
    opacities,
    scattering_in_emission,
    continuum_opacities_scattering):
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

    # TODO add scattering_in_emission check
    # May want to write separate function rather than using lax.cond
    
    optical_depth = jnp.zeros(opacities.shape)
    for i_struc in range(1,opacities.shape[3]):
        optical_depth = optical_depth.at[i_struc].set(optical_depth[i_struc-1] +\
                       (opacities[i_struc]+opacities[i_struc-1])\
                        /2.0/reference_gravity*(pressures[i_struc]-pressures[i_struc-1]))
    return optical_depth


@jit
def compute_photon_destruction_prob(total_opacity_in, continuum_opacity_scat_emis):
    """_summary_
    # TODO this might not be used anymore - calculated in Radtrans?
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
def compute_planck_opacities_core(
    total_opacity,
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
    B_nu = jphys.star_planck(temperature,border_freqs)

    kappa_planck= jnp.sum(B_nu *jnp.sum(w_gauss[:,None,None]/total_opacity[:,:-1,:],axis=0) *diff_freqs[:,None],axis=0)
    norm = jnp.sum(B_nu * diff_freqs[:,None], axis = 0)
    return kappa_planck / norm

def _compute_planck_opacities(
    total_opacity, 
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
    kappa_planck = compute_planck_opacities_core(total_opacity_use[:,:,:], 
                                   border_freqs, 
                                   temperature,
                                   w_gauss)
    return kappa_planck
compute_planck_opacities = jit(_compute_planck_opacities,static_argnums=(4))

def _compute_radius_hydrostatic_equilibrium(
        pressures, 
        reference_gravity, 
        atmosphere_densities,
        reference_pressure, 
        planet_radius,
        variable_gravity
    ):
    n_layers = pressures.shape[0]

    inv_rho = 1.0 / atmosphere_densities
    radius_hydrostatic_equilibrium = jnp.zeros(n_layers)
    R0 = 0.0

    if variable_gravity:
        coefficient = 1.0 / reference_gravity / planet_radius ** 2.0
    else:
        coefficient = -1.0 / reference_gravity

    # Calculate radius with vertically varying gravity, set up such that at P=P0, i.e. R=planet_radius
    # the planet has the predefined scalar gravity value
    """
    radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium.at[-2].set(radius_hydrostatic_equilibrium[-1]\
            - (inv_rho[-2] + inv_rho(-1))\
            * (pressures[-1] - pressures[-2]) * coefficient * 0.5
    )
        do i_str = n_layers - 2, 1, -1
            radius_hydrostatic_equilibrium(i_str) = radius_hydrostatic_equilibrium(i_str + 1) - integ_parab(&
                pressures(i_str), &
                pressures(i_str + 1), &
                pressures(i_str + 2), &
                inv_rho(i_str), &
                inv_rho(i_str + 1), &
                inv_rho(i_str + 2), &
                pressures(i_str), pressures(i_str+1) &
            ) * coefficient
        end do

        ! Find R0
        do i_str = n_layers - 1, 1, -1
            if ((pressures(i_str + 1) > reference_pressure) .and. (pressures(i_str) <= reference_pressure)) then
                if (i_str <= n_layers - 2) then
                    R0 = radius_hydrostatic_equilibrium(i_str + 1) - integ_parab(&
                        pressures(i_str), &
                        pressures(i_str + 1), &
                        pressures(i_str + 2), &
                        inv_rho(i_str), &
                        inv_rho(i_str + 1), &
                        inv_rho(i_str + 2), &
                        reference_pressure, &
                        pressures(i_str + 1) &
                    ) * coefficient

                    exit
                else
                    R0 = radius_hydrostatic_equilibrium(i_str + 1) - (inv_rho(i_str) + inv_rho(i_str+1)) &
                        * (pressures(i_str + 1) - reference_pressure) * coefficient * 0.5d0

                    exit
                end if
            end if
        end do

        if (variable_gravity) then
            R0 = 1d0 / planet_radius - R0
            radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
            radius_hydrostatic_equilibrium = 1d0 / radius_hydrostatic_equilibrium
        else
            R0 = planet_radius - R0
            radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
        end if

        contains
            function integ_parab(x, y, z, fx, fy, fz, a, b)
                ! Function to calc higher order integ.
                implicit none
                ! I/O
                double precision :: x, y, z, fx, fy, fz, a, b
                double precision :: integ_parab
                ! Internal
                double precision :: c1, c2, c3

                c3 = ((fz - fy) / (z - y) - (fz - fx) / (z - x)) / (y - x)
                c2 = (fz - fx) / (z - x) - c3 * (z + x)
                c1 = fx - c2 * x - c3 * x ** 2d0

                integ_parab = c1 * (b - a) + c2 * (b ** 2d0 - a ** 2d0) / 2d0 + c3 * (b ** 3d0 - a ** 3d0) / 3d0
            end function integ_parab"""
    return radius_hydrostatic_equilibrium
compute_radius_hydrostatic_equilibrium = jit(_compute_radius_hydrostatic_equilibrium,static_argnums=(5))


def compute_rosseland_opacities():
    # TODO
    # Unclear if necessary - check jax_physics (compute_rosseland_opacity)
    return

def compute_transit_radii():
    # TODO
    return

def compute_transmission_spectrum_contribution():
    # TODO
    return

def interpolate_cloud_opacities():
    # TODO
    return

@jit
def transmission_g_tot_ck_scat(gravity,
                               pressure,
                               total_opacity_in,
                               continuum_opacity_scattering_emis):
    """
    # TODO: what is the new name

    Args:
        gravity (_type_): _description_
        pressure (_type_): _description_
        total_opacity_in (_type_): _description_
        continuum_opacity_scattering_emis (_type_): _description_

    Returns:
        _type_: _description_
    """
    photon_destruction_prob = compute_photon_destruction_prob(total_opacity_in,
                                                           continuum_opacity_scattering_emis)
    transmission = compute_optical_depths(gravity,
                                       pressure,
                                       total_opacity_in)
    return transmission, photon_destruction_prob
