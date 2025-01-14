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
def m_approximate_ng_source(source_n, source_n1, source_n2, source_n3, source,
                            n_g, n_frequencies_bin_edges, n_layers):
    """Approximates the source term."""

    huge_q = jnp.sqrt(jnp.finfo(jnp.float64).max)
    tiniest = jnp.finfo(jnp.float64).tiny

    def freq_loop(i_freq, carry):
        source = carry

        def ng_loop(i_ng, carry):
            source = carry

            tn = source_n[i_ng, i_freq, :]
            tn1 = source_n1[i_ng, i_freq, :]
            tn2 = source_n2[i_ng, i_freq, :]
            tn3 = source_n3[i_ng, i_freq, :]

            Q1 = tn - 2.0 * tn1 + tn2
            Q2 = tn - tn1 - tn2 + tn3
            Q3 = tn - tn1

            Q1 = Q1.at[0].set(0.0) # equivalent to Q1(1) = 0d0
            Q2 = Q2.at[0].set(0.0) # equivalent to Q2(1) = 0d0
            Q3 = Q3.at[0].set(0.0) # equivalent to Q3(1) = 0d0


            A1 = jnp.minimum(jnp.sum(Q1 * Q1), huge_q)
            A2 = jnp.minimum(jnp.sum(Q2 * Q1), huge_q)
            B1 = jnp.minimum(jnp.sum(Q1 * Q2), huge_q)
            B2 = jnp.minimum(jnp.sum(Q2 * Q2), huge_q)
            C1 = jnp.minimum(jnp.sum(Q1 * Q3), huge_q)
            C2 = jnp.minimum(jnp.sum(Q2 * Q3), huge_q)

            AB_denominator = A1 * B2 - A2 * B1

            def calculate_source(carry):
                a, b, source = carry
                temp_buff = jnp.maximum((1.0 - a - b) * tn + a * tn1 + b * tn2, 0.0)
                source = source.at[i_ng, i_freq, :].set(temp_buff)
                return a, b, source

            def use_original_source(carry):
                a, b, source = carry
                source = source.at[i_ng, i_freq, :].set(source[i_ng, i_freq, :])
                return a, b, source

            a = (C1 * B2 - C2 * B1) / AB_denominator
            b = (C2 * A1 - C1 * A2) / AB_denominator

            calculate_condition = (jnp.abs(A1) >= 1e-250) & \
                                    (jnp.abs(A2) >= 1e-250) & \
                                    (jnp.abs(B1) >= 1e-250) & \
                                    (jnp.abs(B2) >= 1e-250) & \
                                    (jnp.abs(C1) >= 1e-250) & \
                                    (jnp.abs(C2) >= 1e-250)

            a, b, source = jax.lax.cond(AB_denominator >= tiniest and calculate_condition, calculate_source, use_original_source, (a, b, source))

            return source

        source = jax.lax.fori_loop(0, n_g, ng_loop, source)

        return source

    source = jax.lax.fori_loop(0, n_frequencies_bin_edges - 1, freq_loop, source)

    return source


def compute_feautrier_radiative_transfer(
        frequencies_bin_edges,
        temperatures,
        weights_gauss,
        emission_cos_angles,
        emission_cos_angles_weights,
        optical_depths,
        photon_destruction_probabilities,
        emission_geometry,
        stellar_intensity,
        star_irradiation_cos_angle,
        reflectances,
        emissivities,
        return_contribution):

    n_frequencies_bin_edges = frequencies_bin_edges.shape[0]
    n_layers = temperatures.shape[0]
    n_angles = emission_cos_angles.shape[0]
    n_g = weights_gauss.shape[0]

    iter_scat = 1000
    tiniest = jnp.finfo(jnp.float64).tiny
    pi_4 = 4.0 * jnp.pi

    n_iterations_min = 9
    convergence_threshold = 1e-3

    inv_del_tau_min = 1e10
    inv_del_tau_min_half = inv_del_tau_min * 0.5

    I_plus_surface = jnp.zeros((n_angles, n_g, n_frequencies_bin_edges-1))
    I_minus = 0.0

    I_J = jnp.zeros((n_layers, n_angles))
    I_H = jnp.zeros((n_layers, n_angles))
    source = jnp.zeros((n_layers, n_g, n_frequencies_bin_edges - 1))
    J_planet_scat = jnp.zeros((n_layers, n_g, n_frequencies_bin_edges - 1))
    photon_destruction_probabilities_ = jnp.transpose(photon_destruction_probabilities, (1, 2, 0))
    J_star_ini = jnp.zeros((n_g, n_layers, n_frequencies_bin_edges-1))
    I_star_calc = jnp.zeros((n_g, n_layers, n_angles, n_frequencies_bin_edges-1))
    I_surface_no_scattering = jnp.zeros((n_g, n_frequencies_bin_edges - 1))
    surf_refl_2 = 2.0 * reflectances
    mu_weight = emission_cos_angles * emission_cos_angles_weights

    b = jnp.ones((n_layers, n_angles))
    c = jnp.zeros((n_layers, n_angles))
    a = jnp.zeros((n_layers, n_angles))
    lambda_loc = jnp.zeros((n_layers, n_g, n_frequencies_bin_edges - 1))

    source_planet_scat_n = jnp.zeros(n_layers)
    source_planet_scat_n1 = jnp.zeros(n_layers)
    source_planet_scat_n2 = jnp.zeros(n_layers)
    source_planet_scat_n3 = jnp.zeros(n_layers)
    mu_double = emission_cos_angles * 2.0

    flux = jnp.zeros(n_frequencies_bin_edges - 1)
    emission_contribution = jnp.zeros((n_layers, n_frequencies_bin_edges - 1))
    planck = jnp.zeros((n_layers, n_frequencies_bin_edges - 1))

    for k in range(n_frequencies_bin_edges - 1):
        for i in range(n_layers):
            planck = planck.at[i, k].set(jphys.compute_planck_function_integral(
                frequencies_bin_edges[k],
                frequencies_bin_edges[k+1],
                temperatures[i])
                )

    # Main radiative transfer loop (simplified structure)
    for i_freq in range(n_frequencies_bin_edges - 1):
        for i_g in range(n_g):
            for i_lay in range(n_layers):
                source = source.at[i_lay, i_g, i_freq].set(photon_destruction_probabilities_[i_freq, i_lay, i_g] *
                                                           planck[i_lay, i_freq])

    geometry_coefficient = 0.0
    if emission_geometry == "dayside_ave":
        geometry_coefficient = 0.5
    if emission_geometry == "planetary_ave":
        geometry_coefficient = 0.25
    if emission_geometry == "non-isotropic":
        geometry_coefficient = -1.5

    if geometry_coefficient > 0:
        def inner_loop(i, J_star_ini_i):
            def inner_inner_loop(i_mu, J_star_ini_i_mu):
                I_star_calc_i_mu = (geometry_coefficient *
                                    jnp.abs(stellar_intensity[i]) *
                                    jnp.exp(-optical_depths[:, i, :] /
                                            emission_cos_angles[i_mu]))
                return J_star_ini_i_mu + 0.5 * I_star_calc_i_mu * emission_cos_angles_weights[i_mu]
            J_star_ini_i = jax.lax.fori_loop(0, n_angles, inner_inner_loop, J_star_ini_i)
            return J_star_ini_i
        J_star_ini = jax.lax.fori_loop(0, n_frequencies_bin_edges - 1, inner_loop, J_star_ini)

    else:
        def inner_loop(i, J_star_ini_i):
            return jnp.abs(stellar_intensity[i] /
                           4.0 *
                           jnp.exp(-optical_depths[:, i, :] /
                                   star_irradiation_cos_angle))
     
        def update_J_star_ini(i, J_star_ini):
            J_star_ini = J_star_ini.at[:, :, i].set(inner_loop(i, J_star_ini[:, :, i]))
            return J_star_ini

        J_star_ini = jax.lax.fori_loop(0, n_frequencies_bin_edges - 1, update_J_star_ini, J_star_ini)

    planck = jphys.compute_planck_function_integral(temperatures, frequencies_bin_edges)

    def update_surface_intensity(i, I_surface_no_scattering):
        I_surface_emission = emissivities[i] * planck[n_layers - 1, i]

        def update_scattering(l, I_surface_no_scattering_l):
            if emission_geometry.strip().lower() != 'non-isotropic':
                I_surface_reflection = surf_refl_2[i] * jnp.sum(I_star_calc[l, n_layers - 1, :, i] * mu_weight)
            else:
                I_surface_reflection = (reflectances[i] *
                                        J_star_ini[l, n_layers - 1, i] *
                                        4.0 *
                                        star_irradiation_cos_angle)

            return I_surface_no_scattering_l.at[l].set(I_surface_emission + I_surface_reflection)

        I_surface_no_scattering_i = jax.lax.fori_loop(0, n_g, update_scattering, jnp.zeros(n_g))
        return I_surface_no_scattering.at[i].set(I_surface_no_scattering_i)
    
    I_surface_no_scattering = \
        jax.lax.fori_loop(0, n_frequencies_bin_edges - 1, update_surface_intensity, I_surface_no_scattering)
    
    source = (photon_destruction_probabilities_ *
              planck +
              (1.0 - photon_destruction_probabilities_) *
              J_star_ini)
    dtau_1_level = jnp.zeros(n_layers - 1)
    dtau_2_level = jnp.zeros(n_layers - 2)
    dtau_1_level = dtau_1_level.at[:, 0].set(optical_depths[:, i, 1] - optical_depths[:, i, 0])
    dtau_1_level = jnp.where(jnp.abs(dtau_1_level[0]) < tiniest, tiniest, dtau_1_level.at[0].get())

    # Main loop starts here
    for i in range(1, n_frequencies_bin_edges):
        def process_g(l, results_g):
            flux_tmp = 0.0

            def process_k(k, dtau_arrays):
                dtau_1_level, dtau_2_level = dtau_arrays
                dtau_1_level = dtau_1_level.at[l, k].set(optical_depths[l, i, k + 1] - optical_depths[l, i, k])
                dtau_1_level = jnp.where(jnp.abs(dtau_1_level[l, k]) < tiniest,
                                         tiniest,
                                         dtau_1_level.at[l, k].get())

                dtau_2_level = dtau_2_level.at[l, k - 1].set(optical_depths[l, i, k + 1] - optical_depths[l, i, k - 1])
                dtau_2_level =\
                    jnp.where(jnp.abs(dtau_2_level[l, k - 1]) < tiniest,
                              tiniest,
                              dtau_2_level.at[l, k - 1].get())
                return dtau_1_level, dtau_2_level

            dtau_1_level, dtau_2_level = jax.lax.fori_loop(1, n_layers - 1, process_k, (dtau_1_level, dtau_2_level))

            def process_j(j, results_j):
                b, c, a, lambda_loc, f2_struct, f3_struct = results_j

                # First level (top)
                f1 = emission_cos_angles[j] / dtau_1_level[l, 0]
                f1 = jnp.where(f1 > inv_del_tau_min, inv_del_tau_min, f1)

                b = b.at[0, j].set(1.0 + 2.0 * f1 * (1.0 + f1))
                c = c.at[0, j].set(-2.0 * f1**2.0)
                a = a.at[0, j].set(0.0)

                lambda_loc = lambda_loc.at[0, l, i].set(lambda_loc[0, l, i] +
                                                        emission_cos_angles_weights[j] /
                                                        (1.0 + 2.0 * f1 * (1.0 + f1))
                                                        )

                # Mid levels
                def process_k_inner(k, results_k):
                    b, c, a, lambda_loc, f2_struct, f3_struct = results_k
                    f1 = mu_double[j] / dtau_2_level[l, k - 1]
                    f2 = emission_cos_angles[j] / dtau_1_level[l, k]
                    f3 = emission_cos_angles[j] / dtau_1_level[l, k - 1]

                    f1 = jnp.where(f1 > inv_del_tau_min_half, inv_del_tau_min_half, f1)
                    f2 = jnp.where(f2 > inv_del_tau_min, inv_del_tau_min, f2)
                    f3 = jnp.where(f3 > inv_del_tau_min, inv_del_tau_min, f3)

                    b = b.at[k, j].set(1.0 + f1 * (f2 + f3))
                    c = c.at[k, j].set(-f1 * f2)
                    a = a.at[k, j].set(-f1 * f3)

                    f2_struct = f2_struct.at[k, j].set(f2)
                    f3_struct = f3_struct.at[k, j].set(f3)

                    lambda_loc = lambda_loc.at[k, l, i].set(lambda_loc[k, l, i] +
                                                            emission_cos_angles_weights[j] /
                                                            (1.0 + f1 * (f2 + f3)))
                    return b, c, a, lambda_loc, f2_struct, f3_struct

                b, c, a, lambda_loc, f2_struct, f3_struct = \
                    jax.lax.fori_loop(1, n_layers - 1, process_k_inner, (b, c, a, lambda_loc, f2_struct, f3_struct))

                # Last level (surface)
                f1 = emission_cos_angles[j] / dtau_1_level[l,n_layers - 2]  # Corrected index here
                f1 = jnp.where(f1 > inv_del_tau_min, inv_del_tau_min, f1)
                lambda_loc = lambda_loc.at[n_layers - 1, l, i].set(lambda_loc[n_layers - 1, l, i] +
                                                                   emission_cos_angles_weights[j] /
                                                                   (1.0 + 2.0 * f1**2.0))  # Corrected index here

                return b, c, a, lambda_loc, f2_struct, f3_struct
            b, c, a, lambda_loc, f2_struct, f3_struct =\
                jax.lax.fori_loop(0, n_angles, process_j, (b, c, a, lambda_loc, f2_struct, f3_struct))

            def scattering_iteration(i_iter_scat, results_iter):
                (i_iter_scat,
                 conv_val,
                 r,
                 b,
                 c,
                 a,
                 I_plus_surface,
                 I_H,
                 I_J,
                 source,
                 J_planet_scat,
                 flux_tmp,
                 flux_tmp_old,
                 source_planet_scat_n,
                 source_planet_scat_n1,
                 source_planet_scat_n2,
                 source_planet_scat_n3) = results_iter

                def angle_loop(j, results_angle):
                    r, b, c, a, I_plus_surface, I_J, I_H = results_angle
                    I_minus = (I_surface_no_scattering[l, i] +
                               surf_refl_2[i] *
                               jnp.sum(I_plus_surface[:, l, i] * mu_weight))
                    r = r.at[n_layers-1].set(0.5 * (I_plus_surface[j, l, i] + I_minus))
                    I_J = I_J.at[:, j].set(jmath.solve_tridiagonal_system(a[:, j], b[:, j], c[:, j], r))
                    I_H = I_H.at[0, j].set(-I_J[0, j])

                    def k_loop(k, results_k):
                        I_H, I_J, I_plus_surface = results_k
                        deriv1 = f2_struct[k, j] * (I_J[k + 1, j] - I_J[k, j])
                        deriv2 = f3_struct[k, j] * (I_J[k, j] - I_J[k - 1, j])
                        I_H = I_H.at[k, j].set(-(deriv1 + deriv2) * 0.5)

                        I_plus_surface = jnp.where(k == n_layers - 2, 
                                                   I_plus_surface.at[j,l,i].set(I_J[n_layers - 1, j] - deriv1), 
                                                   I_plus_surface)
                        return I_H, I_J, I_plus_surface
                    
                    I_H, I_J, I_plus_surface = jax.lax.fori_loop(1, n_layers - 1, k_loop, (I_H, I_J, I_plus_surface))
                    I_H = I_H.at[n_layers-1, j].set(0.0)
                    return r, b, c, a, I_plus_surface, I_J, I_H
                r, b, c, a, I_plus_surface, I_J, I_H =\
                    jax.lax.fori_loop(0, n_angles, angle_loop, (r, b, c, a, I_plus_surface, I_J, I_H))
                
                flux_tmp_old = flux_tmp
                flux_tmp = 0.0

                def flux_loop(j, flux_tmp):
                    return flux_tmp - I_H[0, j] * mu_weight[j]
                
                flux_tmp = jax.lax.fori_loop(0, n_angles, flux_loop, flux_tmp)

                conv_val = jnp.abs(1.0 - flux_tmp_old / flux_tmp)

                def cond_fun(loop_carry_over):
                    i_iter_scat, conv_val = loop_carry_over
                    return (conv_val >= convergence_threshold) | (i_iter_scat <= n_iterations_min)

                def body_fun(loop_carry_over):
                    i_iter_scat, conv_val, r, I_H, I_J, source, J_planet_scat, flux_tmp, flux_tmp_old, source_planet_scat_n, source_planet_scat_n1, source_planet_scat_n2, source_planet_scat_n3 = loop_carry_over

                    J_bol_g = 0.0

                    def j_bol_loop(j, J_bol_g):
                        return J_bol_g + I_J[:, j] * emission_cos_angles_weights[j]
                    
                    J_bol_g = jax.lax.fori_loop(0, n_angles, j_bol_loop, J_bol_g)

                    J_planet_scat = J_planet_scat.at[:, l, i].set(J_bol_g)

                    source = source.at[:, l, i].set(
                        (photon_destruction_probabilities_[:, l, i] * planck[:, i] +
                        (1.0 - photon_destruction_probabilities_[:, l, i]) *
                        (J_star_ini[l, :, i] + J_planet_scat[:, l, i] - lambda_loc[:, l, i] * source[:, l, i])) /
                        (1.0 - (1.0 - photon_destruction_probabilities_[:, l, i]) * lambda_loc[:, l, i])
                    )

                    source_planet_scat_n3 = source_planet_scat_n2
                    source_planet_scat_n2 = source_planet_scat_n1
                    source_planet_scat_n1 = source_planet_scat_n

                    source_planet_scat_n = jnp.where(i_iter_scat % 4 == 0, source[:, l, i], source_planet_scat_n)

                    source = jnp.where(i_iter_scat % 4 == 0, 
                                       m_approximate_ng_source(source_planet_scat_n,
                                                               source_planet_scat_n1,
                                                               source_planet_scat_n2,
                                                               source_planet_scat_n3,
                                                               source[:, l, i]),
                                       source[:, l, i])

                    r = r.at[:].set(source[:, l, i])

                    return (i_iter_scat + 1,
                            conv_val,
                            r,
                            I_H,
                            I_J,
                            source,
                            J_planet_scat,
                            flux_tmp,
                            flux_tmp_old,
                            source_planet_scat_n,
                            source_planet_scat_n1,
                            source_planet_scat_n2,
                            source_planet_scat_n3)
                
                (i_iter_scat,
                 conv_val,
                 r,
                 b,
                 c,
                 a,
                 I_plus_surface,
                 I_H,
                 I_J,
                 source,
                 J_planet_scat,
                 flux_tmp,
                 flux_tmp_old,
                 source_planet_scat_n,
                 source_planet_scat_n1,
                 source_planet_scat_n2,
                 source_planet_scat_n3) = jax.lax.while_loop(cond_fun,
                                                             body_fun,
                                                             (i_iter_scat,
                                                              conv_val,
                                                              r,
                                                              I_H,
                                                              I_J,
                                                              source,
                                                              J_planet_scat,
                                                              flux_tmp,
                                                              flux_tmp_old,
                                                              source_planet_scat_n,
                                                              source_planet_scat_n1,
                                                              source_planet_scat_n2,
                                                              source_planet_scat_n3))

                return (i_iter_scat,
                        conv_val,
                        r,
                        b,
                        c,
                        a,
                        I_plus_surface,
                        I_H,
                        I_J,
                        source,
                        J_planet_scat,
                        flux_tmp,
                        flux_tmp_old,
                        source_planet_scat_n,
                        source_planet_scat_n1,
                        source_planet_scat_n2,
                        source_planet_scat_n3)
            
            (i_iter_scat,
             conv_val,
             r,
             b,
             c,
             a,
             I_plus_surface,
             I_H,
             I_J,
             source,
             J_planet_scat,
             flux_tmp,
             flux_tmp_old,
             source_planet_scat_n,
             source_planet_scat_n1,
             source_planet_scat_n2,
             source_planet_scat_n3) = \
                jax.lax.fori_loop(0,
                                  iter_scat,
                                  scattering_iteration,
                                  (i_iter_scat,
                                   conv_val,
                                   r,
                                   b,
                                   c,
                                   a,
                                   I_plus_surface,
                                   I_H,
                                   I_J,
                                   source,
                                   J_planet_scat,
                                   flux_tmp,
                                   flux_tmp_old,
                                   source_planet_scat_n,
                                   source_planet_scat_n1,
                                   source_planet_scat_n2,
                                   source_planet_scat_n3))
            
            def flux_calculation(results_flux):
                flux = results_flux

                def flux_loop(j, flux):
                    return flux - I_H[0, j] * pi_4 * weights_gauss[l] * mu_weight[j]

                flux = jax.lax.fori_loop(0, n_angles, flux_loop, flux)
                return flux

            # Example usage (inside the larger function, after the scattering iteration):
            flux = flux.at[i].set(flux_calculation(flux[i]))

            return flux

        flux = jax.lax.fori_loop(0, n_g, process_g, (flux))



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

Python

import jax
import jax.numpy as jnp
from jax import jit

@jit
def integ_parab(x, y, z, fx, fy, fz, a, b):
    """Function to calculate higher order integral."""
    c3 = ((fz - fy) / (z - y) - (fz - fx) / (z - x)) / (y - x)
    c2 = (fz - fx) / (z - x) - c3 * (z + x)
    c1 = fx - c2 * x - c3 * x**2
    return c1 * (b - a) + c2 * (b**2 - a**2) / 2 + c3 * (b**3 - a**3) / 3

@jit
def compute_radius_hydrostatic_equilibrium(pressures, reference_gravity, atmosphere_densities,
                                          reference_pressure, planet_radius, variable_gravity,
                                          n_layers):
    """Calculate the radius at hydrostatic equilibrium from the pressure grid."""

    inv_rho = 1.0 / atmosphere_densities

    if variable_gravity:
        coefficient = 1.0 / reference_gravity / planet_radius**2
    else:
        coefficient = -1.0 / reference_gravity

    radius_hydrostatic_equilibrium = jnp.zeros(n_layers)

    radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium.at[n_layers - 2].set(
        radius_hydrostatic_equilibrium[n_layers - 1] - (inv_rho[n_layers - 2] + inv_rho[n_layers - 1]) *
        (pressures[n_layers - 1] - pressures[n_layers - 2]) * coefficient * 0.5
    )

    def body(i, val):
      radius_hydrostatic_equilibrium = val[0]
      radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium.at[i].set(
          radius_hydrostatic_equilibrium[i + 1] - integ_parab(
              pressures[i], pressures[i + 1], pressures[i + 2],
              inv_rho[i], inv_rho[i + 1], inv_rho[i + 2],
              pressures[i], pressures[i + 1]
          ) * coefficient
      )
      return (radius_hydrostatic_equilibrium,)

    radius_hydrostatic_equilibrium, = jax.lax.fori_loop(n_layers - 3, 0, body, (radius_hydrostatic_equilibrium,))

    def find_r0_body(i, val):
        R0 = val[0]
        found = val[1]
        if (pressures[i + 1] > reference_pressure) and (pressures[i] <= reference_pressure) and (not found):
            if i <= n_layers - 3:
                R0 = radius_hydrostatic_equilibrium[i + 1] - integ_parab(
                    pressures[i], pressures[i + 1], pressures[i + 2],
                    inv_rho[i], inv_rho[i + 1], inv_rho[i + 2],
                    reference_pressure, pressures[i + 1]
                ) * coefficient
            else:
                R0 = radius_hydrostatic_equilibrium[i + 1] - (inv_rho[i] + inv_rho[i + 1]) * \
                    (pressures[i + 1] - reference_pressure) * coefficient * 0.5
            found = True
        return (R0, found)
    
    R0, _ = jax.lax.fori_loop(n_layers - 2, 0, find_r0_body, (0.0, False))

    if variable_gravity:
        R0 = 1.0 / planet_radius - R0
        radius_hydrostatic_equilibrium = 1.0 / (radius_hydrostatic_equilibrium + R0)
    else:
        R0 = planet_radius - R0
        radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0

    return radius_hydrostatic_equilibrium


def compute_rosseland_opacities():
    # TODO
    # Unclear if necessary - check jax_physics (compute_rosseland_opacity)
    return

@jit
def compute_transit_radii(opacities, temperatures, pressures, reference_gravity, mean_molar_masses,
                          reference_pressure, planet_radius, weights_gauss,
                          scattering_in_transmission, continuum_opacities_scattering, variable_gravity,
                          n_frequencies, n_layers, n_g, n_species):
    """Calculate the transmission spectrum."""

    P0_cgs = reference_pressure * 1e6

    rho = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures, reference_gravity, rho, P0_cgs, planet_radius, variable_gravity, n_layers
    )

    opacities_ = jnp.where(opacities < 0, 0.0, opacities)

    alpha_t2 = jnp.zeros((n_g, n_frequencies, n_species, n_layers - 1))
    alpha_t2 = alpha_t2.at[:, :, :, :-1].set(opacities_[:, :, :, :-1] * rho[:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
    alpha_t2 = alpha_t2.at[:, :, :, :-2].add(opacities_[:, :, :, 1:-1] * rho[1:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :])
    alpha_t2 = alpha_t2.at[:, :, :, -1].set(alpha_t2[:, :, :, -1] + opacities_[:, :, :, -1] * rho[-1])

    if scattering_in_transmission:
        alpha_t2_scat = jnp.zeros((n_frequencies, n_layers - 1))
        alpha_t2_scat = alpha_t2_scat.at[:, :-1].set(continuum_opacities_scattering[:, :-1] * rho[:-1])
        alpha_t2_scat = alpha_t2_scat.at[:, :-2].add(continuum_opacities_scattering[:, 1:-1] * rho[1:-1])
        alpha_t2_scat = alpha_t2_scat.at[:, -1].set(alpha_t2_scat[:, -1] + continuum_opacities_scattering[:, -1] * rho[-1])
    else:
        alpha_t2_scat = jnp.zeros((n_frequencies, n_layers - 1))

    radius_squared = radius_hydrostatic_equilibrium**2
    radius_squared2 = jnp.zeros((n_layers, n_layers))

    for i_str in range(n_layers):
        for j_str in range(i_str):
            radius_squared2 = radius_squared2.at[j_str, i_str].set(radius_squared[j_str] - radius_squared[i_str])
    radius_squared2 = jnp.sqrt(radius_squared2)
    radius_squared2 = radius_squared2.at[:-1, :].set(radius_squared2[:-1, :] - radius_squared2[1:, :])

    t_graze = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
    t_graze_scat = jnp.zeros((n_frequencies, n_layers))

    for i_str in range(1, n_layers):
        for j_str in range(i_str):
            t_graze = t_graze.at[:, :, :, i_str].add(alpha_t2[:, :, :, j_str] * radius_squared2[j_str, i_str])
            t_graze_scat = t_graze_scat.at[:, i_str].add(alpha_t2_scat[:, j_str] * radius_squared2[j_str, i_str])

    t_graze = jnp.exp(-t_graze)
    t_graze_scat = jnp.exp(-t_graze_scat)

    t_graze_wlen_int = jnp.ones((n_frequencies, n_layers))
    for i_str in range(1, n_layers):
        for i_spec in range(n_species):
            for i_freq in range(n_frequencies):
                t_graze_wlen_int =\
                      t_graze_wlen_int.at[i_freq, i_str].set(t_graze_wlen_int[i_freq, i_str] * 
                                                             jnp.sum(t_graze[:, i_freq, i_spec, i_str] * 
                                                                     weights_gauss))
                if scattering_in_transmission and (i_spec == 0):
                    t_graze_wlen_int = \
                        t_graze_wlen_int.at[i_freq, i_str].set(t_graze_wlen_int[i_freq, i_str] *
                                                               t_graze_scat[i_freq, i_str])

    t_graze_wlen_int = 1.0 - t_graze_wlen_int

    t_graze_wlen_int = t_graze_wlen_int * radius_hydrostatic_equilibrium[jnp.newaxis, :]

    d_radius = jnp.zeros(n_layers)
    d_radius = d_radius.at[1:].set(radius_hydrostatic_equilibrium[:-1] - radius_hydrostatic_equilibrium[1:])

    t_graze_wlen_int_1 = jnp.zeros((n_frequencies, n_layers))
    t_graze_wlen_int_2 = jnp.zeros((n_frequencies, n_layers))

    t_graze_wlen_int_1 = t_graze_wlen_int_1.at[:, 1:].set(t_graze_wlen_int[:, :-1] * d_radius[1:])
    t_graze_wlen_int_2 = t_graze_wlen_int_2.at[:, 1:].set(t_graze_wlen_int[:, 1:] * d_radius[1:])

    t_graze_wlen_int = t_graze_wlen_int_1 + t_graze_wlen_int_2
    t_graze_wlen_int_t = t_graze_wlen_int.T

    transit_radii = jnp.zeros(n_frequencies)
    for i_freq in range(n_frequencies):
        transit_radii = transit_radii.at[i_freq].set(jnp.sum(transit_radii[i_freq] + t_graze_wlen_int_t[:, i_freq]))

    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1]**2)

    return transit_radii, radius_hydrostatic_equilibrium

def compute_transmission_spectrum_contribution():
    # TODO
    return

@jit
def integrate_opacities(intp_ind, n_cloud_wavelengths, n_layers, kappa, lambd, l_bord1, l_bord2):
    """Calculate the integral of a linearly interpolated function kappa."""

    kappa_integ = kappa[intp_ind, :] * (l_bord2 - l_bord1) + \
                  (kappa[intp_ind + 1, :] - kappa[intp_ind, :]) / \
                  (lambd[intp_ind + 1] - lambd[intp_ind]) * \
                  0.5 * ((l_bord2 - lambd[intp_ind])**2 - (l_bord1 - lambd[intp_ind])**2)
    return kappa_integ

@jit
def interpolate_cloud_opacities(cloud_absorption_opacities, cloud_scattering_opacities,
                                cloud_scattering_reduction_factor, cloud_wavelengths,
                                frequencies_bin_edges,
                                n_cloud_wavelengths, n_layers, n_frequencies_bins):
    """Interpolate cloud opacities to actual radiative transfer wavelength grid."""

    clouds_final_absorption_opacities = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_anisotropic_extinctions = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_isotropic_extinctions = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_scattering_reduction_factor_out = jnp.zeros((n_frequencies_bins - 1, n_layers))

    HIT_border_lamb = cst.c / frequencies_bin_edges

    kappa_tot_integ = jnp.zeros((n_frequencies_bins - 1, n_layers))
    kappa_tot_scat_integ = jnp.zeros((n_frequencies_bins - 1, n_layers))

    def loop_body(HIT_i_lamb, carry):
        (clouds_final_absorption_opacities, 
         cloud_anisotropic_extinctions, 
         cloud_isotropic_extinctions, 
         cloud_scattering_reduction_factor_out) = carry

        intp_index_small_min = jnp.clip(
            int((jnp.log10(HIT_border_lamb[HIT_i_lamb]) - jnp.log10(cloud_wavelengths[0])) /
                jnp.log10(cloud_wavelengths[-1] / cloud_wavelengths[0]) * (n_cloud_wavelengths - 1) + 1),
                1,
                n_cloud_wavelengths - 1
                )
        intp_index_small_max = jnp.clip(
            int((jnp.log10(HIT_border_lamb[HIT_i_lamb + 1]) - jnp.log10(cloud_wavelengths[0])) /
                jnp.log10(cloud_wavelengths[-1] / cloud_wavelengths[0]) * (n_cloud_wavelengths - 1) + 1), 
                1, 
                n_cloud_wavelengths - 1
                )

        kappa_integ = jnp.zeros(n_layers)
        kappa_scat_integ = jnp.zeros(n_layers)
        red_fac_aniso_integ = jnp.zeros(n_layers)

        def integrate_section(start_index, end_index):
            nonlocal kappa_integ, kappa_scat_integ, red_fac_aniso_integ
            kappa_integ += integrate_opacities(
                start_index -1 , 
                n_cloud_wavelengths, 
                n_layers, 
                cloud_absorption_opacities,
                cloud_wavelengths, 
                HIT_border_lamb[HIT_i_lamb] if start_index==intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index==intp_index_small_max else cloud_wavelengths[end_index]
                )
            kappa_scat_integ += integrate_opacities(
                start_index -1, 
                n_cloud_wavelengths, 
                n_layers, 
                cloud_scattering_opacities,
                cloud_wavelengths, 
                HIT_border_lamb[HIT_i_lamb] if start_index==intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index==intp_index_small_max else cloud_wavelengths[end_index]
                )
            red_fac_aniso_integ += integrate_opacities(
                start_index -1, 
                n_cloud_wavelengths, 
                n_layers, 
                cloud_scattering_reduction_factor,
                cloud_wavelengths, 
                HIT_border_lamb[HIT_i_lamb] if start_index==intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index==intp_index_small_max else cloud_wavelengths[end_index]
                )
            return 

        def inner_loop_body(new_small_ind, _):
            integrate_section(new_small_ind, new_small_ind + 1)
            return new_small_ind + 1, None

        if (intp_index_small_max - intp_index_small_min) == 0:
            integrate_section(intp_index_small_min, intp_index_small_max+1)
        elif (intp_index_small_max - intp_index_small_min) == 1:
            integrate_section(intp_index_small_min, intp_index_small_min+1)
            integrate_section(intp_index_small_max, intp_index_small_max+1)
        else:
            integrate_section(intp_index_small_min, intp_index_small_min+1)
            _, _ = jax.lax.while_loop(
                lambda val: val[0] < intp_index_small_max, 
                inner_loop_body, 
                (intp_index_small_min + 1, None)
                )
            
            integrate_section(intp_index_small_max, intp_index_small_max+1)

        kappa_integ = kappa_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        kappa_scat_integ = kappa_scat_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        red_fac_aniso_integ = red_fac_aniso_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])

        kappa_tot_integ = kappa_tot_integ.at[HIT_i_lamb, :].set(kappa_integ)
        kappa_tot_scat_integ = kappa_tot_scat_integ.at[HIT_i_lamb, :].set(kappa_scat_integ)

        clouds_final_absorption_opacities = clouds_final_absorption_opacities.at[HIT_i_lamb, :].add(kappa_integ)
        cloud_anisotropic_extinctions =\
            cloud_anisotropic_extinctions.at[HIT_i_lamb, :].add(kappa_integ + kappa_scat_integ * red_fac_aniso_integ)
        cloud_isotropic_extinctions =\
            cloud_isotropic_extinctions.at[HIT_i_lamb, :].add(kappa_integ + kappa_scat_integ)
        cloud_scattering_reduction_factor_out =\
            cloud_scattering_reduction_factor_out.at[HIT_i_lamb, :].add(red_fac_aniso_integ)

        return (clouds_final_absorption_opacities, 
                cloud_anisotropic_extinctions, 
                cloud_isotropic_extinctions, 
                cloud_scattering_reduction_factor_out
        )

    carry = (clouds_final_absorption_opacities, 
             cloud_anisotropic_extinctions, 
             cloud_isotropic_extinctions, 
             cloud_scattering_reduction_factor_out)
    
    (clouds_final_absorption_opacities, 
     cloud_anisotropic_extinctions, 
     cloud_isotropic_extinctions, 
     cloud_scattering_reduction_factor_out) = jax.lax.fori_loop(0, n_frequencies_bins - 1, loop_body, carry)

    return (clouds_final_absorption_opacities, 
            cloud_anisotropic_extinctions, 
            cloud_isotropic_extinctions, 
            cloud_scattering_reduction_factor_out)


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
