"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback, debug


import numpy as np
from functools import partial
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.fortran_radtrans_core import math as fmath
import petitRADTRANS.jax_physics as jphys
import petitRADTRANS.jax_math as jmath

CST_PI = jnp.pi  # Or the specific precision from Fortran
TINIEST = jnp.finfo(jnp.float64).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float64).max)


@jit
def combine_ck_opacities(opacities_fortran_order, g_gauss, weights_gauss):
    """
    JAX implementation of the combine_ck_opacities Fortran subroutine.

    Args:
        opacities_fortran_order (jax.Array): Opacities with shape (n_g, n_frequencies, n_species, n_layers).
        g_gauss (jax.Array): Gaussian quadrature g-values, shape (n_g,).
        weights_gauss (jax.Array): Gaussian quadrature weights, shape (n_g,).

    Returns:
        jax.Array: Combined opacities, shape (n_g, n_frequencies, n_layers).
    """
    n_g, n_frequencies, n_species, n_layers = opacities_fortran_order.shape

    # Determine the computation dtype from g_gauss for consistency
    # This assumes g_gauss, weights_gauss, and opacities_fortran_order share the same precision needs.
    # JAX's x64 mode should be enabled upstream if float64 is desired.
    # e.g., from jax import config; config.update("jax_enable_x64", True)
    dtype = g_gauss.dtype

    threshold_coefficient = 1e-2  # Fortran: 1d-2 (Python float, JAX handles promotion)
    #n_sample = n_g * n_g

    # Threshold calculation: threshold(n_frequencies, n_layers)
    threshold = jnp.max(opacities_fortran_order[0, :, :, :], axis=1) * threshold_coefficient
    # threshold shape is (n_frequencies, n_layers)

    # g_presort calculation: flat array of weights_gauss(i) * weights_gauss(j)
    g_presort_flat = (weights_gauss[:, None] * weights_gauss[None, :]).ravel()
    # g_presort_flat shape: (n_g*n_g,) which is n_sample

    # Permute opacities for easier vmap/scan: (layers, frequencies, species, n_g)
    opacities_perm = jnp.transpose(opacities_fortran_order, (3, 1, 2, 0))
    # Permute threshold to (layers, frequencies)
    threshold_perm = jnp.transpose(threshold, (1, 0))

    def combine_one_species_lf(opac_out_current_lf, spec_opac_lf, threshold_lf):
        """
        Combines opacities of one species with the current accumulated opacities
        for a single layer and frequency.
        opac_out_current_lf: shape (n_g,)
        spec_opac_lf: shape (n_g,)
        threshold_lf: scalar
        """

        def perform_combination():
            k_out_combined_flat = (opac_out_current_lf[:, None] + spec_opac_lf[None, :]).ravel()

            sampled_opa_weights_0 = k_out_combined_flat
            sampled_opa_weights_1 = g_presort_flat

            # OPTIMIZATION: Use jax.lax.sort
            # Sort by opacities (sampled_opa_weights_0 are keys)
            k_sorted, weights_sorted = lax.sort(
                operand=[sampled_opa_weights_0, sampled_opa_weights_1],
                dimension=0,  # Sort along the 0-th axis of these 1D arrays
                num_keys=1    # Sort based on the first operand (sampled_opa_weights_0)
            )

            weights_normalized = weights_sorted / jnp.sum(weights_sorted)

            cum_sum_weights_normalized = jnp.cumsum(weights_normalized)
            # OPTIMIZATION: Explicit dtype for literals
            prev_cum_sum = jnp.concatenate(
                [jnp.array([0.0], dtype=dtype),
                 cum_sum_weights_normalized[:-1]]
                 )
            g_x_vals = weights_normalized * 0.5 + prev_cum_sum
            g_x_coords = jnp.concatenate([g_x_vals, jnp.array([1.0], dtype=dtype)])

            k_y_intermediate = k_sorted.at[0].set(opac_out_current_lf[0] + spec_opac_lf[0])
            k_y_coords = jnp.concatenate([
                k_y_intermediate,
                jnp.array([opac_out_current_lf[-1] + spec_opac_lf[-1]], dtype=dtype)
            ])

            new_opac_out_lf = jnp.interp(g_gauss, g_x_coords, k_y_coords)
            return new_opac_out_lf

        def skip_combination():
            return opac_out_current_lf

        return lax.cond(spec_opac_lf[-1] < threshold_lf,
                        skip_combination,
                        perform_combination)

    def process_layer_freq(opacities_all_species_lf, threshold_lf):
        opac_out_initial_lf = opacities_all_species_lf[0, :]

        # This Python if is fine for JIT as n_species is static for a given trace
        if n_species == 1:
            return opac_out_initial_lf

        def species_scan_body(current_combined_opac_lf, next_species_opac_lf):
            updated_opac_lf = combine_one_species_lf(
                current_combined_opac_lf,
                next_species_opac_lf,
                threshold_lf
            )
            return updated_opac_lf, None

        remaining_species_opac_lf = opacities_all_species_lf[1:, :]

        final_combined_opac_lf, _ = lax.scan(
            species_scan_body,
            opac_out_initial_lf,
            remaining_species_opac_lf
        )
        return final_combined_opac_lf

    opacities_out_perm = vmap(  # vmap over layers
        vmap(process_layer_freq, in_axes=(0, 0)),  # vmap over frequencies
        in_axes=(0, 0)
    )(opacities_perm, threshold_perm)

    opacities_out_final = jnp.transpose(opacities_out_perm, (2, 1, 0))

    return opacities_out_final


@jit
def calculate_ck_emission_contribution(
        frequencies,
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

    struct_shape = (frequencies.shape[0], temperatures.shape[0])
    flux = jnp.zeros(struct_shape[0])
    contr_em = jnp.zeros(struct_shape)

    for i_scat_angle in range(emission_cos_angles.shape[0]):
        # Transmissions for a given incidence angle
        transm_mu = jnp.exp(-optical_depths/emission_cos_angles[i_scat_angle])

        # Flux contribution from given scat_angle-angle

        # mean_transm = np.zeros((struct_shape[0],transmission.shape[2],struct_shape[1])
        mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        r = jphys.compute_planck_function(temperatures, frequencies)

        # Do the actual radiative transport
        flux_mu = jnp.sum(
            (r[:, :-1]+r[:, 1:])
            * (transm_all[:, 0:-1]-transm_all[:, 1:])
            / 2.0,
            axis=1
            )
        flux_mu = jnp.add(
            flux_mu,
            r[:, struct_shape[1]] * transm_all[:, struct_shape[1]]
            )
        contr_em = contr_em.at[:, :-1].add(
            (r[:, :-1]+r[:, 1:])
            * (transm_all[:, 0:-1]-transm_all[:, 1:])
            * emission_cos_angles[i_scat_angle]
            * emission_cos_angles_weights[i_scat_angle]
            )
        contr_em = contr_em.at[:, -1].add(
            2.0
            * r[:, struct_shape[1]]
            * transm_all[:, struct_shape[1]]
            * emission_cos_angles[i_scat_angle]
            * emission_cos_angles_weights[i_scat_angle]
            )

        # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
        flux = jnp.add(
            flux,
            flux_mu
            / 2.0
            * emission_cos_angles[i_scat_angle]
            * emission_cos_angles_weights[i_scat_angle]
            )

    # Normalization
    flux = flux * 4.0 * jnp.pi
    contr_em = contr_em.T
    contr_em = jnp.divide(contr_em, jnp.sum(contr_em, axis=0))
    return flux, contr_em


@partial(jit, static_argnames=("return_contribution"))
def compute_ck_flux(
        frequencies,
        temperatures,
        weights_gauss,
        emission_cos_angles,
        emission_cos_angles_weights,
        optical_depths,
        return_contribution):
    """_summary_

    Args:
        frequencies (ndarray): shape (len_freq)
        transmission (ndarray): shape (len_g, len_freq, len_spec, len_temp))
        temperatures (ndarray): shape (len_temp)
        emission_cos_angles (ndarray): shape (len_scatter)
        emission_cos_angles_weights (ndarray): shape (len_scatter)
        weights_gauss (ndarray): shape (len_scatter)

    Returns:
        flux: shape (len_freq)
    """
    def calculate_flux(
            frequencies,
            temperatures,
            weights_gauss,
            emission_cos_angles,
            emission_cos_angles_weights,
            optical_depths
            ):
        struct_shape = (frequencies.shape[0], temperatures.shape[0])
        contribution = jnp.zeros((struct_shape[1], struct_shape[0]))
        flux = jnp.zeros(struct_shape[0])

        for i_mu in range(emission_cos_angles.shape[0]):
            # Transmissions for a given incidence angle
            transm_mu = jnp.exp(-optical_depths/emission_cos_angles[i_mu])
            mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)
            transm_all = jnp.prod(mean_transm, axis=1)

            r = jphys.compute_planck_function(temperatures, frequencies)
            # Do the actual radiative transport
            flux_mu = jnp.sum(
                (r[:, :-1]+r[:, 1:])
                * (transm_all[:, 0:-1]-transm_all[:, 1:])
                * 0.5,
                axis=1
                )
            flux_mu = jnp.add(flux_mu, r[:, struct_shape[1]]*transm_all[:, struct_shape[1]])
            # print(flux_mu.shape)
            # angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
            flux = jnp.add(flux,
                           (
                            flux_mu
                            * 0.5
                            * emission_cos_angles[i_mu]
                            * emission_cos_angles_weights[i_mu]
                            ))
        return flux * 4.0 * jnp.pi, contribution
        
    # Check if we should calculate the contribution function
    flux, contribution = lax.cond(
        return_contribution,
        calculate_ck_emission_contribution,
        calculate_flux,
        frequencies,
        temperatures,
        weights_gauss,
        emission_cos_angles,
        emission_cos_angles_weights,
        optical_depths
    )
    return flux, contribution


def compute_cloud_hansen_opacities():
    # TODO
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


def solve_tridiagonal_fortran(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d calling the Fortran implementation of the tridag solver.
    Args:
        a (jax.Array): Lower diagonal, shape (N,). a[i] is coeff of x_i for eq i+1.
        b(jax.Array): Main diagonal, shape (N,).
        c (jax.Array): Upper diagonal, shape (N,). c[i] is coeff of x_{i+2} for eq i+1.
        d (jax.Array): Right-hand side, shape (N,).
    Returns:
        jax.Array: Solution x, shape (N,).
    """
    result_shape = ShapeDtypeStruct(d.shape, d.dtype)
    return pure_callback(fmath.solve_tridiagonal_system, result_shape, a, b, c, d)


@jit
def solve_tridiagonal_system_jax(a_lower_diag, b_main_diag, c_upper_diag, r_rhs):
    """
    Solves a tridiagonal system Ax = r_rhs using Thomas algorithm or jax.scipy.
    Args:
        a_lower_diag (jax.Array): Lower diagonal, shape (N-1,). a[i] is coeff of x_i for eq i+1.
        b_main_diag (jax.Array): Main diagonal, shape (N,).
        c_upper_diag (jax.Array): Upper diagonal, shape (N-1,). c[i] is coeff of x_{i+2} for eq i+1.
        r_rhs (jax.Array): Right-hand side, shape (N,).
    Returns:
        jax.Array: Solution x, shape (N,).
    """

    # return solve_tridiagonal_fortran(a_lower_diag, b_main_diag, c_upper_diag, r_rhs)
    return solve_tridiagonal_fortran(jnp.concatenate([jnp.array([0]), a_lower_diag]),
                                     b_main_diag,
                                     jnp.concatenate([c_upper_diag, jnp.array([0])]),
                                     r_rhs)


@partial(jit)
def m_approximate_ng_source_layer_jax(tn, tn1, tn2, tn3, current_source_layer):
    """
    Ng acceleration for a single layer's source function history.
    Inputs are all shape (n_layers,).
    """
    Q1 = tn - 2.0 * tn1 + tn2
    Q2 = tn - tn1 - tn2 + tn3
    Q3 = tn - tn1

    # Boundary condition from Fortran (Q1[0]=0 etc.)
    Q1 = Q1.at[0].set(0.0)
    Q2 = Q2.at[0].set(0.0)
    Q3 = Q3.at[0].set(0.0)

    A1 = jnp.minimum(jnp.sum(Q1 * Q1), HUGE_SQRT)
    A2 = jnp.minimum(jnp.sum(Q2 * Q1), HUGE_SQRT)
    B1 = jnp.minimum(jnp.sum(Q1 * Q2), HUGE_SQRT)
    B2 = jnp.minimum(jnp.sum(Q2 * Q2), HUGE_SQRT)
    C1 = jnp.minimum(jnp.sum(Q1 * Q3), HUGE_SQRT)
    C2 = jnp.minimum(jnp.sum(Q2 * Q3), HUGE_SQRT)

    AB_denominator = A1 * B2 - A2 * B1

    cond_skip_ng = (AB_denominator < TINIEST) | \
                   (jnp.abs(A1) < 1e-250) | (jnp.abs(A2) < 1e-250) | \
                   (jnp.abs(B1) < 1e-250) | (jnp.abs(B2) < 1e-250) | \
                   (jnp.abs(C1) < 1e-250) | (jnp.abs(C2) < 1e-250)

    def ng_update_func():
        a_ng = (C1 * B2 - C2 * B1) / AB_denominator
        b_ng = (C2 * A1 - C1 * A2) / AB_denominator
        return jnp.maximum((1.0 - a_ng - b_ng) * tn + a_ng * tn1 + b_ng * tn2, 0.0)

    def no_ng_update_func():
        return current_source_layer  # Or tn, depending on Fortran logic for `source`

    return lax.cond(cond_skip_ng, no_ng_update_func, ng_update_func)


# Vmap over (g, freq) dimensions
# Input source arrays will be (n_g, n_freq, n_layers)
vmapped_m_approximate_ng_source = vmap(
    vmap(
        m_approximate_ng_source_layer_jax,
        in_axes=(0, 0, 0, 0, 0)),  # vmap over freq
    in_axes=(0, 0, 0, 0, 0))   # vmap over g


# --- Main Feautrier Radiative Transfer JAX Function ---
@partial(jit, static_argnames=(
    "emission_geometry_str", "return_contribution",
    "iter_scat_max", "n_iterations_min_convergence"
))
def compute_feautrier_radiative_transfer(
    frequencies_bin_edges,  # (n_frequencies_bin_edges,)
    temperatures,           # (n_layers,)
    weights_gauss,          # (n_g,)
    emission_cos_angles,    # (n_angles,)
    emission_cos_angles_weights,  # (n_angles,)
    optical_depths_in,      # (n_g, n_freq_bins, n_layers)
    photon_destruction_probabilities_in,  # (n_g, n_freq_bins, n_layers)
    stellar_intensity,      # (n_freq_bins,)
    star_irradiation_cos_angle,  # scalar
    reflectances,           # (n_freq_bins,)
    emissivities,           # (n_freq_bins,)
    # Static parameters for JIT compilation
    emission_geometry_str: str,
    return_contribution: bool,
    iter_scat_max: int = 1000,
    n_iterations_min_convergence: int = 9,  # Renamed from n_iterations_min
    convergence_threshold: float = 1e-3
):
    """Computes the atmospheric emission spectrum using the Feautrier method.

    This function solves the radiative transfer equation for a plane-parallel
    atmosphere, including thermal emission, scattering of planetary thermal
    radiation, scattering of incident stellar radiation, and surface emission
    and reflection. It uses an iterative approach (Accelerated Lambda Iteration
    combined with Ng acceleration) to converge on the source function and
    ultimately the outgoing flux.

    Args:
        frequencies_bin_edges (jax.Array): Edges of the frequency bins in Hz.
            Shape: (n_frequencies_bin_edges,).
        temperatures (jax.Array): Temperature profile of the atmosphere in K.
            Shape: (n_layers,).
        weights_gauss (jax.Array): Gaussian quadrature weights for k-distribution
            integration. Shape: (n_g,).
        emission_cos_angles (jax.Array): Cosines of the emission angles (mu)
            for angular quadrature. Shape: (n_angles,).
        emission_cos_angles_weights (jax.Array): Weights for the emission angle
            quadrature. Shape: (n_angles,).
        optical_depths_in (jax.Array): Layer-integrated optical depths.
            Shape: (n_g, n_freq_bins, n_layers).
        photon_destruction_probabilities_in (jax.Array): Photon destruction
            probability (1 - single scattering albedo) for each g-point,
            frequency bin, and layer. Shape: (n_g, n_freq_bins, n_layers).
        stellar_intensity (jax.Array): Incident stellar flux at the top of the
            atmosphere in erg/s/cm^2/Hz. Shape: (n_freq_bins,).
        star_irradiation_cos_angle (float): Cosine of the stellar irradiation
            angle (mu_star), used for 'non-isotropic' geometry.
        reflectances (jax.Array): Surface reflectance (frequency-dependent).
            Shape: (n_freq_bins,).
        emissivities (jax.Array): Surface emissivity (frequency-dependent).
            Shape: (n_freq_bins,).
        n_layers (int): Static. Number of atmospheric layers.
        n_angles (int): Static. Number of emission angles for quadrature.
        n_g (int): Static. Number of g-ordinates in the k-distribution.
        emission_geometry_str (str): Static. Specifies the emission geometry.
            Options: 'dayside_ave', 'planetary_ave', 'non-isotropic'.
        return_contribution (bool): Static. If True, attempts to calculate the
            emission contribution function (currently a placeholder).
        iter_scat_max (int): Static. Maximum number of scattering iterations.
        n_iterations_min_convergence (int): Static. Minimum number of iterations
            before convergence is checked.
        convergence_threshold (float): Static. Relative change threshold for
            convergence of the iterative solution.

    Returns:
        Tuple[jax.Array, jax.Array]:
            - flux_f (jax.Array): Calculated outgoing flux at the top of the
              atmosphere for each frequency bin, in erg/s/cm^2/Hz.
              Shape: (n_freq_bins,).
            - emission_contribution_lf (jax.Array): Emission contribution
              function (currently a placeholder, returns zeros).
              Shape: (n_layers, n_freq_bins).
    """

    # --- Constants and Precomputations ---
    n_g, n_freq_bins, n_layers = photon_destruction_probabilities_in.shape
    n_angles = emission_cos_angles.shape[0]
    pi_4 = 4.0 * CST_PI
    inv_del_tau_min = 1e10  # From Fortran
    inv_del_tau_min_half = inv_del_tau_min * 0.5

    # Reshape photon_destruction_probabilities: (n_g, n_freq, n_layers) -> (n_layers, n_g, n_freq)
    photon_destruction_probabilities_gfl = photon_destruction_probabilities_in  # (g,f,l)

    mu_double = emission_cos_angles * 2.0
    surf_refl_2 = 2.0 * reflectances
    mu_weight_a = emission_cos_angles * emission_cos_angles_weights  # Renamed for clarity (a,)

    # --- 1. Stellar Attenuation (J_star_ini) ---
    # J_star_ini: (n_g, n_freq_bins, n_layers)
    if emission_geometry_str == 'dayside_ave':
        geometry_coefficient = 0.5
    elif emission_geometry_str == 'planetary_ave':
        geometry_coefficient = 0.25
    elif emission_geometry_str == 'non-isotropic':
        geometry_coefficient = -1.0  # Special flag
    else:
        raise ValueError(f"Unknown emission_geometry: {emission_geometry_str}")

    if geometry_coefficient > 0:
        # optical_depths_in: (g, f, l), emission_cos_angles: (a)
        # stellar_intensity: (f)
        # exp_term: (g, f, a, l) = exp(- opt_depths(g,f,l) / mu(a) )
        exp_term_gfal = jnp.exp(-optical_depths_in[:, :, None, :]
                                / emission_cos_angles[None, None, :, None])
        I_star_calc_gfal = (geometry_coefficient
                            * jnp.abs(stellar_intensity[None, :, None, None])
                            * exp_term_gfal
                            )
        # J_star_ini (g,f,l) = 0.5 * sum_a (I_star_calc_gfal * mu_weight(a))
        J_star_ini_gfl = 0.5 * jnp.sum(I_star_calc_gfal
                                       * emission_cos_angles_weights[None, None, :, None],
                                       axis=2)
    else:  # non-isotropic
        exp_term_gfl = jnp.exp(-optical_depths_in / star_irradiation_cos_angle)
        J_star_ini_gfl = jnp.abs(stellar_intensity[None, :, None] / 4.0 * exp_term_gfl)

    # Clamp photon destruction probabilities
    photon_destruction_probabilities_gfl = jnp.maximum(photon_destruction_probabilities_gfl, 1e-10)

    # --- 2. Planck Function Integral ---
    # planck_integral_lf: (n_layers, n_freq_bins)
    planck_integral_lf = jphys.compute_planck_function_integral(temperatures, frequencies_bin_edges)
    planck_integral_fl = jnp.transpose(planck_integral_lf, (1, 0))  # (f,l) for broadcast to (g,f,l)

    # --- 3. Surface Terms ---
    # I_surface_no_scattering_gf: (n_g, n_freq_bins)
    I_surface_emission_f = emissivities * planck_integral_lf[n_layers-1, :]  # (f,)

    if geometry_coefficient > 0:
        # I_star_calc_gfal was (g,f,a,l). Need I_star_calc at surface: I_star_calc_gfa_surf (g,f,a)
        I_star_calc_gfa_surf = I_star_calc_gfal[:, :, :, n_layers-1]
        I_surface_reflection_gf = surf_refl_2[None, :] * jnp.sum(  # sum over angles
            I_star_calc_gfa_surf * mu_weight_a[None, None, :], axis=2
        )
    else:  # non-isotropic
        I_surface_reflection_gf = (
            reflectances[None, :]
            * J_star_ini_gfl[:, :, n_layers-1]
            * 4.0 * star_irradiation_cos_angle
            )

    I_surface_no_scattering_gf = I_surface_emission_f[None, :] + I_surface_reflection_gf

    # --- 4. Initial Source Function & Precompute Feautrier Coefficients ---
    # S_gfl: (g, f, l)
    # planck_integral_fl is (f,l), broadcast to (g,f,l) using [None, :, :]
    # photon_destruction_probabilities_gfl is (g,f,l)
    initial_source_gfl = photon_destruction_probabilities_gfl * planck_integral_fl[None, :, :] + \
        (1.0 - photon_destruction_probabilities_gfl) * J_star_ini_gfl

    # Delta taus for Feautrier coefficients
    # dtau1_gfl: (n_g, n_freq, n_layers-1), optical_depths_in(g,f,l)
    dtau1_gfl = optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1]
    dtau1_gfl = jnp.where((jnp.abs(dtau1_gfl) < TINIEST), TINIEST, dtau1_gfl)

    # dtau2_gfl: (n_g, n_freq, n_layers-2)
    dtau2_gfl = optical_depths_in[:, :, 2:] - optical_depths_in[:, :, :-2]
    dtau2_gfl = jnp.where((jnp.abs(dtau2_gfl) < TINIEST), TINIEST, dtau2_gfl)

    # Feautrier coefficients (b_main_diags_gfal, a_lower_diags_gfalm1, c_upper_diags_gfalm1)
    # Shapes: (g, f, a, l) for main, (g, f, a, l-1) for off-diagonals

    # Top layer (l=0)
    f1_top_gfa = jnp.clip(emission_cos_angles[None, None, :, None] /
                          dtau1_gfl[:, :, None, 0, None],
                          -inv_del_tau_min,
                          inv_del_tau_min)
    b_gfal_0 = 1.0 + 2.0 * f1_top_gfa * (1.0 + f1_top_gfa)
    c_gfal_0 = -2.0 * f1_top_gfa**2
    # a_gfal_0 is 0

    # Mid layers (l=1 to n_layers-2)
    # f1_mid_gfalm1: (g,f,a,l-2), using dtau2 for layers 1 to n_layers-2
    f1_mid_gfalm1 = jnp.clip(mu_double[None, None, :, None] /
                             dtau2_gfl[:, :, None, :],
                             - inv_del_tau_min_half,
                             inv_del_tau_min_half)
    # f2_mid_gfalm1: (g,f,a,l-2), using dtau1 for layers 1 to n_layers-2 
    # (dtau1 indices 1 to n_layers-2)
    f2_mid_gfalm1 = jnp.clip(emission_cos_angles[None, None, :, None] /
                             dtau1_gfl[:, :, None, 1:],
                             -inv_del_tau_min,
                             inv_del_tau_min)
    # f3_mid_gfalm1: (g,f,a,l-2), using dtau1 for layers 0 to n_layers-3 
    # (dtau1 indices 0 to n_layers-3)
    f3_mid_gfalm1 = jnp.clip(emission_cos_angles[None, None, :, None] /
                             dtau1_gfl[:, :, None, :-1],
                             -inv_del_tau_min,
                             inv_del_tau_min)

    b_gfal_mid = 1.0 + f1_mid_gfalm1 * (f2_mid_gfalm1 + f3_mid_gfalm1)
    c_gfal_mid = -f1_mid_gfalm1 * f2_mid_gfalm1  # Upper diagonal part
    a_gfal_mid = -f1_mid_gfalm1 * f3_mid_gfalm1  # Lower diagonal part

    # Assemble full diagonal arrays for tridiagonal solver
    # b_main_diags_gfal: (g,f,a,l)
    _b = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers))
    _b = _b.at[:, :, :, 0].set(b_gfal_0[..., 0])
    _b = _b.at[:, :, :, 1:n_layers-1].set(b_gfal_mid)
    _b = _b.at[:, :, :, n_layers-1].set(1.0)  # Surface b=1
    b_main_diags_gfal = _b

    # c_upper_diags_gfalm1: (g,f,a,l-1)
    _c = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers - 1))
    _c = _c.at[:, :, :, 0].set(c_gfal_0[..., 0])
    _c = _c.at[:, :, :, 1:n_layers-1].set(c_gfal_mid)  # c_gfal_mid is shape (g,f,a,l-2)
    # This covers upper diags for rows 1 to n_layers-2
    c_upper_diags_gfalm1 = _c

    # a_lower_diags_gfalm1: (g,f,a,l-1)
    _a = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers - 1))
    # a_gfal_mid is shape (g,f,a,l-2), covers lower diags for rows 1 to n_layers-2
    _a = _a.at[:, :, :, :n_layers-2].set(a_gfal_mid)
    a_lower_diags_gfalm1 = _a

    # Lambda_loc for ALI (Accelerated Lambda Iteration)
    # lambda_loc_gfl: (n_g, n_freq_bins, n_layers)
    lambda_loc_contrib_top_gfa = mu_weight_a[None, None, :, None] / b_gfal_0
    lambda_loc_top_gf = jnp.sum(lambda_loc_contrib_top_gfa, axis=2)  # sum over angles

    lambda_loc_contrib_mid_gfalm1 = mu_weight_a[None, None, :, None] / b_gfal_mid
    lambda_loc_mid_gflm2 = jnp.sum(lambda_loc_contrib_mid_gfalm1, axis=2)

    f1_surf_gfa = jnp.clip(emission_cos_angles[None, None, :, None]
                           / dtau1_gfl[:, :, None, n_layers-2, None],
                           -inv_del_tau_min,
                           inv_del_tau_min)
    lambda_loc_contrib_surf_gfa = mu_weight_a[None, None, :, None] / (1.0 + 2.0 * f1_surf_gfa**2)
    lambda_loc_surf_gf = jnp.sum(lambda_loc_contrib_surf_gfa, axis=2)

    _l = jnp.zeros((n_g, n_freq_bins, n_layers))
    _l = _l.at[:, :, 0].set(lambda_loc_top_gf[..., 0])
    _l = _l.at[:, :, 1:n_layers-1].set(lambda_loc_mid_gflm2)
    _l = _l.at[:, :, n_layers-1].set(lambda_loc_surf_gf[..., 0])
    lambda_loc_gfl = _l  # (g,f,l)

    # --- 5. Scattering Iteration Loop ---
    # State: (current_source_gfl, current_I_plus_surface_gfa, I_H_gfal_prev, J_planet_scat_gfl_prev,
    #         s_n_gfl, s_n1_gfl, s_n2_gfl, s_n3_gfl, iter_idx, converged_flag)
    @jit
    def scattering_loop_body_func(loop_state_prev):
        (current_source_gfl, current_I_plus_surface_gfa, I_H_gfal_prev,
         s_n_gfl, s_n1_gfl, s_n2_gfl, s_n3_gfl, iter_idx, converged_flag) = loop_state_prev

        # --- 5a. Solve Feautrier System for each (g,f,a) ---
        # RHS for tridiagonal system: r_gfal (g,f,a,l)
        # Source part: current_source_gfl (g,f,l)
        r_gfal = jnp.repeat(current_source_gfl[:, :, None, :], n_angles, axis=2)  # (g,f,a,l)

        reflection_term_gf = surf_refl_2[None, :] *\
            jnp.sum(current_I_plus_surface_gfa * mu_weight_a[None, None, :], axis=2) # sum over angles
        I_minus_boundary_gf = I_surface_no_scattering_gf + reflection_term_gf

        r_gfal_surface = 0.5 * (current_I_plus_surface_gfa + I_minus_boundary_gf[:, :, None])  # (g,f,a)
        r_gfal = r_gfal.at[:, :, :, n_layers-1].set(r_gfal_surface)

        # Vectorize solver: solve_tridiagonal_system_jax(a_lower(l-1), b_main(l), c_upper(l-1), r_rhs(l))
        # Inputs are (g,f,a,dims). Vmap over (g,f,a).
        # I_J_gfal: (g,f,a,l)
        I_J_gfal = vmap(vmap(vmap(solve_tridiagonal_system_jax, in_axes=(0, 0, 0, 0))))(
            a_lower_diags_gfalm1, b_main_diags_gfal, c_upper_diags_gfalm1, r_gfal
        )

        # --- 5b. Calculate I_H_gfal_current and new I_plus_surface_gfa ---
        I_H_gfal_current = jnp.zeros_like(I_J_gfal)  # Renamed
        I_H_gfal_current = I_H_gfal_current.at[:, :, :, 0].set(-I_J_gfal[:, :, :, 0])  # Top layer

        # Mid layers for I_H
        # deriv1_gfalm1: (g,f,a,l-2) using f2_mid and I_J for layers 1 to n_layers-2
        deriv1_gfalm1 = f2_mid_gfalm1 * (I_J_gfal[:, :, :, 2:n_layers]
                                         - I_J_gfal[:, :, :, 1:n_layers-1]
                                         )
        # deriv2_gfalm1: (g,f,a,l-2) using f3_mid and I_J for layers 0 to n_layers-3
        deriv2_gfalm1 = f3_mid_gfalm1 * (I_J_gfal[:, :, :, 1:n_layers-1]
                                         - I_J_gfal[:, :, :, 0:n_layers-2]
                                         )
        I_H_gfal_current = I_H_gfal_current.at[:, :, :, 1:n_layers-1].set(
            -(deriv1_gfalm1 + deriv2_gfalm1) * 0.5
            )

        # New I_plus_surface_gfa (g,f,a) for next iteration
        # I_plus_surface = I_J_surface - deriv1_surface 
        #   (deriv1 for k=n_layers-1, which is last element of deriv1_gfalm1)
        new_I_plus_surface_gfa = I_J_gfal[:, :, :, n_layers-1] - deriv1_gfalm1[:, :, :, n_layers-2]

        # --- 5c. Update Source Function (ALI + Ng) ---
        # J_bol_g_gfl (g,f,l) = sum_a (I_J_gfal * mu_w_a)
        J_bol_g_gfl = jnp.sum(I_J_gfal * mu_weight_a[None, None, :, None], axis=2)
        J_planet_scat_gfl_current = J_bol_g_gfl  # (g,f,l)

        # ALI update for source_lgf
        # photon_destruction_probabilities_gfl (g,f,l), planck_integral_fl (f,l)
        # J_star_ini_gfl (g,f,l), lambda_loc_gfl (g,f,l), current_source_gfl (g,f,l)
        num_gfl = photon_destruction_probabilities_gfl * planck_integral_fl[None, :, :] + \
            (1.0 - photon_destruction_probabilities_gfl) * \
            (J_star_ini_gfl + J_planet_scat_gfl_current - lambda_loc_gfl * current_source_gfl)
        den_gfl = 1.0 - (1.0 - photon_destruction_probabilities_gfl) * lambda_loc_gfl
        source_after_ali_gfl = num_gfl / jnp.maximum(den_gfl, TINIEST)

        # Ng Acceleration History Update
        next_s_n3_gfl, next_s_n2_gfl, next_s_n1_gfl = s_n2_gfl, s_n1_gfl, s_n_gfl
        next_s_n_gfl = source_after_ali_gfl  # Store pre-Ng source for history

        # Apply Ng (iter_idx is 0-based)
        # Fortran: mod(i_iter_scat, 4) == 0. Here: (iter_idx + 1) % 4 == 0
        def ng_accel_func():
            # vmapped_m_approximate_ng_source expects (g,f,l) inputs
            accelerated_s_gfl = vmapped_m_approximate_ng_source(
                next_s_n_gfl, next_s_n1_gfl, next_s_n2_gfl, next_s_n3_gfl, source_after_ali_gfl
            )
            return accelerated_s_gfl

        final_updated_source_gfl = lax.cond(
            (iter_idx > 0) & ((iter_idx + 1) % 4 == 0),  # iter_idx is 0 to iter_scat_max-1
            ng_accel_func,
            lambda: source_after_ali_gfl  # No Ng
        )

        # Convergence check based on relative change in total I_H at the top boundary
        total_I_H_top_prev = jnp.sum(I_H_gfal_prev[:, :, :, 0], axis=(0, 2)) # Sum over g and angles
        total_I_H_top_current = jnp.sum(I_H_gfal_current[:, :, :, 0], axis=(0, 2))
        max_rel_change = jnp.max(1 - total_I_H_top_prev / total_I_H_top_current)
        new_converged_flag = (iter_idx > n_iterations_min_convergence) &\
            (max_rel_change < convergence_threshold)

        # Update loop state
        next_loop_state = (final_updated_source_gfl, new_I_plus_surface_gfa, I_H_gfal_current,
                           next_s_n_gfl, next_s_n1_gfl, next_s_n2_gfl, next_s_n3_gfl,
                           iter_idx + 1, new_converged_flag)

        return next_loop_state  # keep returning I_H for final flux

    def convergence_condition_func(loop_state):
        # Note: the requested condition had iter_idx > iter_scat_max for *continuation*.  
        # Here we want the loop to *stop* when converged OR max iterations reached.
        _, _, _, _, _, _, _, iter_idx, converged_flag = loop_state
        return (iter_idx <= iter_scat_max) & (jnp.logical_not(converged_flag))
        # Stopping condition: loop runs WHILE (not converged AND iter_idx <= iter_scat_max)

    # Initial state for while_loop (using (g,f,l) and (g,f,a) layouts)
    initial_iter_state = (initial_source_gfl,                       # current_source_gfl
                          jnp.zeros((n_g, n_freq_bins, n_angles)),  # current_I_plus_surface_gfa
                          jnp.zeros((n_g, n_freq_bins, n_angles, n_layers)),  # I_H_gfal_prev
                          jnp.zeros_like(initial_source_gfl),       # s_n_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n1_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n2_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n3_gfl
                          0,                                       # iter_idx
                          False)                                   # converged_flag

    final_loop_state = lax.while_loop(convergence_condition_func,
                                      scattering_loop_body_func,
                                      initial_iter_state)

    # Extract final values from the loop state.
    # The I_H_gfal_prev and J_planet_scat_gfl_prev in the final_loop_state
    # are the I_H_gfal_current and J_planet_scat_gfl_current from the last completed iteration.
    _, _, I_H_gfal_final, _, _, _, _, _, _ = final_loop_state

    # --- 6. Calculate Flux ---
    # flux_f (n_freq_bins) = sum_g sum_a (-I_H_gfal(g,f,a,l=0) * pi_4 * w_gauss_g * mu_w_a)
    # I_H_gfal_final is (g,f,a,l). We need I_H at top (l=0).
    I_H_top_gfa = I_H_gfal_final[:, :, :, 0]  # (g,f,a)
    flux_f = jnp.sum(
        -I_H_top_gfa * pi_4 * weights_gauss[:, None, None] * mu_weight_a[None, None, :],
        axis=(0, 2)  # Sum over g and angles
    )

    # --- 7. Contribution Function (if return_contribution) ---
    emission_contribution_lf = jnp.zeros((n_layers, n_freq_bins))
    if return_contribution:
        # Logic similar to Fortran:
        # transm_mu_gfl = exp(-optical_depths_in / emission_cos_angles)
        # transm_all_fl = sum_g (transm_mu_gfl * weights_gauss_g)
        # r_source_fl = sum_g (final_source_lgf (transposed to g,f,l) * weights_gauss_g)
        # Loop over angles i_mu:
        #   contrib_lf +=
        # (r_source_fl(l) + r_source_fl(l+1)) * (transm_all_fl(l) - transm_all_fl(l+1)) * mu * mu_w
        #   Handle surface term carefully.
        # Normalize contrib_lf.
        # This part needs careful translation of the loops and sums.
        pass  # Placeholder for detailed implementation

    return flux_f, emission_contribution_lf


@partial(jit, static_argnames=("scattering_in_emission"))
def compute_optical_depths(
        reference_gravity,
        pressures,
        opacities,
        scattering_in_emission=False,
        continuum_opacities_scattering=None):
    """_summary_

    Args:
        reference_gravity (float): Surface gravity in cgs (cm/s^2)
        pressure (jax.Array): 1D array of pressures in pascal
        opacities (jax.Array): kappa, has shape (g_len, freq_len, press_len)
                                 This is the sum of all species' opacities, at
                                 every c-k g coordinate, frequency and pressure.
        scattering_in_emission (bool): If True, includes continuum scattering 
                            opacities in the calculation of optical depths. 
        continuum_opacities_scattering (jax.Array): The continuum opacity, which
                            has the same shape as the opacities array.

    Returns:
        optical_depths (jax.Array): Tau, a 1D array of optical depths for g,
                    frequency and for every pressure level in the atmosphere.
        photon_destruction_probabilities (jax.Array): probability that a photon
                    interacts with the atmosphere.
    """

    def scattering_function():
        photon_destruction_probabilities = jnp.ones(opacities[:, :, 0, :].shape)
        _opacities = opacities.copy()
        for i_g in range(opacities.shape[0]):
            _opacities = _opacities.at[i_g, :, 0, :].add(
                continuum_opacities_scattering
            )
            photon_destruction_probabilities = photon_destruction_probabilities.at[i_g, :, :].set(
                continuum_opacities_scattering / _opacities[i_g, :, 0, :]
            )

        photon_destruction_probabilities = 1 - photon_destruction_probabilities
        return photon_destruction_probabilities, _opacities

    def false_function():
        return jnp.ones(opacities[:, :, 0, :].shape), opacities

    photon_destruction_probabilities, _opacities = lax.cond(
        scattering_in_emission,
        scattering_function,
        false_function
    )
    optical_depth = jnp.zeros(opacities.shape)
    for i_struc in range(1, opacities.shape[3]):
        optical_depth = optical_depth.at[i_struc].set(
            optical_depth[i_struc-1]
            + (_opacities[i_struc]+_opacities[i_struc-1])
            / 2.0
            / reference_gravity*(pressures[i_struc]-pressures[i_struc-1]))
    return optical_depth, photon_destruction_probabilities


@jit
def compute_photon_destruction_prob(total_opacity_in, continuum_opacity_scat_emis):
    """_summary_
    # TODO this might not be used anymore - calculated in compute_optical_depths.
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
    total_opacity = total_opacity_in[:, :, 0, :]
    g_len = total_opacity_in.shape[0]
    photon_destruction_prob = jnp.ones(total_opacity.shape)
    for i_g in range(g_len):
        total_opacity = total_opacity.at[i_g].set(total_opacity[i_g] + continuum_opacity_scat_emis)
        photon_destruction_prob = photon_destruction_prob.at[i_g].set(continuum_opacity_scat_emis/total_opacity[i_g])
    photon_destruction_prob = 1 - photon_destruction_prob
    return photon_destruction_prob


@jit
def compute_planck_opacities_core(
        opacities,
        temperatures,
        weights_gauss,
        frequencies_bin_edges,
        ):
    """_summary_

    Args:
        opacities (_type_): _description_
        temperatures (_type_): _description_
        weights_gauss (_type_): _description_
        frequencies_bin_edges (_type_): _description_

    Returns:
        _type_: _description_
    """
    diff_freqs = jnp.diff(frequencies_bin_edges)
    B_nu = jphys.star_planck(temperatures, frequencies_bin_edges)

    kappa_planck = jnp.sum(B_nu
                           * jnp.sum(weights_gauss[:, None, None]/opacities[:, :, :], axis=0)
                           * diff_freqs[:, None],
                           axis=0)
    norm = jnp.sum(B_nu * diff_freqs[:, None], axis=0)
    return kappa_planck / norm


@partial(jit, static_argnames=("scattering_in_emission"))
def compute_planck_opacities(
        opacities,
        temperatures,
        weights_gauss,
        frequencies_bin_edges,
        scattering_in_emission,
        continuum_opacities_scattering):
    """_summary_

    Args:
        opacities (jax.Array): line opacities, has shape (g_len, freq_len, press_len)
        temperatures (jax.Array): temperatures, has shape (press_len)
        weights_gauss (jax.Array): weights
        frequencies_bin_edges (jax.Array):edges of the frequency grid
        scattering_in_emission (bool): compute with continuum opacities
        continuum_opacities_scattering (jax.Array): continuum opacities

    Returns:
        kappa_planck (jax.Array): _description_
    """
    total_opacity_use = lax.cond(
        scattering_in_emission,
        lambda: opacities + continuum_opacities_scattering[None, :, :],
        lambda: opacities
    )
   
    kappa_planck = compute_planck_opacities_core(
        total_opacity_use[:, :, :],
        temperatures,
        weights_gauss,
        frequencies_bin_edges)
    return kappa_planck


@jit
def integ_parab(x, y, z, fx, fy, fz, a, b):
    """Function to calculate higher order integral."""
    c3 = ((fz - fy) / (z - y) - (fz - fx) / (z - x)) / (y - x)
    c2 = (fz - fx) / (z - x) - c3 * (z + x)
    c1 = fx - c2 * x - c3 * x**2
    return c1 * (b - a) + c2 * (b**2 - a**2) / 2 + c3 * (b**3 - a**3) / 3


@partial(jit, static_argnames=("variable_gravity"))
def compute_radius_hydrostatic_equilibrium(
        pressures,
        reference_gravity,
        atmosphere_densities,
        reference_pressure,
        planet_radius,
        variable_gravity):
    """Calculate the radius at hydrostatic equilibrium from the pressure grid."""
    n_layers = pressures.shape[0]
    inv_rho = 1.0 / (atmosphere_densities + TINIEST)
    dtype = pressures.dtype

    coefficient = lax.cond(
        variable_gravity,
        lambda: 1.0 / (reference_gravity * (planet_radius**2) + TINIEST),
        lambda: -1.0 / (reference_gravity + TINIEST)
    )

    radius_hydrostatic_equilibrium = jnp.zeros(n_layers, dtype=dtype)

    radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium.at[n_layers - 2].set(
        radius_hydrostatic_equilibrium[n_layers - 1]
        - (inv_rho[n_layers - 2] + inv_rho[n_layers - 1])
        * (pressures[n_layers - 1] - pressures[n_layers - 2])
        * coefficient
        * 0.5
    )

    def condition(val):
        i = val[0]
        return i >= 0
    
    def body(val):
        i = val[0]
        radius_hydrostatic_equilibrium = val[1]
        radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium.at[i].set(
            radius_hydrostatic_equilibrium[i + 1]
            - integ_parab(
                pressures[i],
                pressures[i + 1],
                pressures[i + 2],
                inv_rho[i],
                inv_rho[i + 1],
                inv_rho[i + 2],
                pressures[i],
                pressures[i + 1]
            )
            * coefficient
        )
        return (i-1, radius_hydrostatic_equilibrium,)
    _, radius_hydrostatic_equilibrium = lax.while_loop(
        condition,
        body,
        (n_layers - 3, radius_hydrostatic_equilibrium,)
        )

    reference_index = jnp.argmin(jnp.abs(pressures - reference_pressure))
    reference_index = lax.cond(
        pressures[reference_index] <= reference_pressure,
        lambda: reference_index,
        lambda: reference_index - 1
    )

    layer_condition = (reference_index <= n_layers - 3)
    
    def layer_true():
        return radius_hydrostatic_equilibrium[reference_index + 1] - integ_parab(
                pressures[reference_index],
                pressures[reference_index + 1],
                pressures[reference_index + 2],
                inv_rho[reference_index],
                inv_rho[reference_index + 1],
                inv_rho[reference_index + 2],
                reference_pressure,
                pressures[reference_index + 1]
            ) * coefficient

    def layer_false():
        return (radius_hydrostatic_equilibrium[reference_index + 1]
                - (inv_rho[reference_index] + inv_rho[reference_index + 1])
                * (pressures[reference_index + 1] - reference_pressure)
                * coefficient
                * 0.5)

    R0 = lax.cond(
            layer_condition,
            layer_true,
            layer_false
        )

    def variable_gravity_radius(R0, radius_hydrostatic_equilibrium):
        R0 = (1.0 / planet_radius) - R0
        radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
        radius_hydrostatic_equilibrium = 1.0 / (radius_hydrostatic_equilibrium + TINIEST)
        return radius_hydrostatic_equilibrium

    def constant_gravity_radius(R0, radius_hydrostatic_equilibrium):
        R0 = planet_radius - R0
        radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
        return radius_hydrostatic_equilibrium

    return lax.cond(variable_gravity,
                    variable_gravity_radius,
                    constant_gravity_radius,
                    R0,
                    radius_hydrostatic_equilibrium)


@partial(jit, static_argnames=["scattering_in_transmission", "variable_gravity"])
def compute_transit_radii(
        opacities,
        temperatures,
        pressures,
        reference_gravity,
        mean_molar_masses,
        reference_pressure,
        planet_radius,
        weights_gauss,
        scattering_in_transmission,
        continuum_opacities_scattering,
        variable_gravity):
    """Calculate the transmission spectrum."""
    n_g, n_frequencies, n_species, n_layers = opacities.shape
    P0_cgs = reference_pressure * 1e6

    rho = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures, reference_gravity, rho, P0_cgs, planet_radius, variable_gravity
    )

    opacities_ = jnp.where(opacities < 0, 0.0, opacities)

    alpha_t2 = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
    alpha_t2 = alpha_t2.at[:, :, :, :-1].set(
        opacities_[:, :, :, :-1]
        * rho[:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )
    alpha_t2 = alpha_t2.at[:, :, :, :-2].add(
        opacities_[:, :, :, 1:-1]
        * rho[1:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        )
    alpha_t2 = alpha_t2.at[:, :, :, -1].set(
        alpha_t2[:, :, :, -1]
        + opacities_[:, :, :, -1]
        * rho[-1]
        )

    def transmission_scattering():
        alpha_t2_scat = jnp.zeros((n_frequencies, n_layers))
        alpha_t2_scat = alpha_t2_scat.at[:, :-1].set(
            continuum_opacities_scattering[:, :-1]
            * rho[:-1]
            )
        alpha_t2_scat = alpha_t2_scat.at[:, 1:-1].add(
            continuum_opacities_scattering[:, 1:-1]
            * rho[1:-1]
            )
        alpha_t2_scat = alpha_t2_scat.at[:, -1].set(
            alpha_t2_scat[:, -1]
            + continuum_opacities_scattering[:, -1] 
            * rho[-1]
            )
        return alpha_t2_scat

    def no_transmission_scattering():
        return jnp.zeros((n_frequencies, n_layers))

    alpha_t2_scat = lax.cond(
        scattering_in_transmission,
        transmission_scattering,
        no_transmission_scattering
    )

    radius_squared = radius_hydrostatic_equilibrium**2
    radius_squared2 = jnp.zeros((n_layers, n_layers))

    for i_str in range(n_layers):
        for j_str in range(i_str):
            radius_squared2 = radius_squared2.at[j_str, i_str].set(
                radius_squared[j_str] - radius_squared[i_str]
                )
    radius_squared2 = jnp.sqrt(radius_squared2)
    radius_squared2 = radius_squared2.at[:-1, :].set(
        radius_squared2[:-1, :] - radius_squared2[1:, :]
        )

    t_graze = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
    t_graze_scat = jnp.zeros((n_frequencies, n_layers))

    for i_str in range(1, n_layers):
        for j_str in range(i_str):
            t_graze = t_graze.at[:, :, :, i_str].add(
                alpha_t2[:, :, :, j_str] * radius_squared2[j_str, i_str]
                )
            t_graze_scat = t_graze_scat.at[:, i_str].add(
                alpha_t2_scat[:, j_str] * radius_squared2[j_str, i_str]
                )

    t_graze = jnp.exp(-t_graze)
    t_graze_scat = jnp.exp(-t_graze_scat)

    t_graze_wlen_int = jnp.ones((n_frequencies, n_layers))
    for i_str in range(1, n_layers):
        for i_spec in range(n_species):
            for i_freq in range(n_frequencies):
                t_graze_wlen_int =\
                      t_graze_wlen_int.at[i_freq, i_str].set(
                          t_graze_wlen_int[i_freq, i_str]
                          * jnp.sum(t_graze[:, i_freq, i_spec, i_str] * weights_gauss)
                          )
                if scattering_in_transmission and (i_spec == 0):
                    t_graze_wlen_int = t_graze_wlen_int.at[i_freq, i_str].set(
                        t_graze_wlen_int[i_freq, i_str] * t_graze_scat[i_freq, i_str]
                        )

    t_graze_wlen_int = 1.0 - t_graze_wlen_int

    t_graze_wlen_int = t_graze_wlen_int * radius_hydrostatic_equilibrium[jnp.newaxis, :]

    d_radius = jnp.zeros(n_layers)
    d_radius = d_radius.at[1:].set(
        radius_hydrostatic_equilibrium[:-1] - radius_hydrostatic_equilibrium[1:]
        )

    t_graze_wlen_int_1 = jnp.zeros((n_frequencies, n_layers))
    t_graze_wlen_int_2 = jnp.zeros((n_frequencies, n_layers))

    t_graze_wlen_int_1 = t_graze_wlen_int_1.at[:, 1:].set(t_graze_wlen_int[:, :-1] * d_radius[1:])
    t_graze_wlen_int_2 = t_graze_wlen_int_2.at[:, 1:].set(t_graze_wlen_int[:, 1:] * d_radius[1:])

    t_graze_wlen_int = t_graze_wlen_int_1 + t_graze_wlen_int_2
    t_graze_wlen_int_t = t_graze_wlen_int.T

    transit_radii = jnp.zeros(n_frequencies)
    for i_freq in range(n_frequencies):
        transit_radii = transit_radii.at[i_freq].set(
            jnp.sum(transit_radii[i_freq] + t_graze_wlen_int_t[:, i_freq])
            )

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
            int((jnp.log10(HIT_border_lamb[HIT_i_lamb]) - jnp.log10(cloud_wavelengths[0]))
                / jnp.log10(cloud_wavelengths[-1] / cloud_wavelengths[0])
                * (n_cloud_wavelengths - 1) + 1),
            1, n_cloud_wavelengths - 1
            )
        intp_index_small_max = jnp.clip(
            int((jnp.log10(HIT_border_lamb[HIT_i_lamb + 1]) - jnp.log10(cloud_wavelengths[0]))
                / jnp.log10(cloud_wavelengths[-1] / cloud_wavelengths[0])
                * (n_cloud_wavelengths - 1) + 1),
            1, n_cloud_wavelengths - 1
            )

        kappa_integ = jnp.zeros(n_layers)
        kappa_scat_integ = jnp.zeros(n_layers)
        red_fac_aniso_integ = jnp.zeros(n_layers)

        def integrate_section(start_index, end_index):
            nonlocal kappa_integ, kappa_scat_integ, red_fac_aniso_integ
            kappa_integ += integrate_opacities(
                start_index - 1,
                n_cloud_wavelengths,
                n_layers,
                cloud_absorption_opacities,
                cloud_wavelengths,
                HIT_border_lamb[HIT_i_lamb] if start_index == intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index == intp_index_small_max else cloud_wavelengths[end_index]
                )
            kappa_scat_integ += integrate_opacities(
                start_index - 1,
                n_cloud_wavelengths,
                n_layers,
                cloud_scattering_opacities,
                cloud_wavelengths,
                HIT_border_lamb[HIT_i_lamb] if start_index == intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index == intp_index_small_max else cloud_wavelengths[end_index]
                )
            red_fac_aniso_integ += integrate_opacities(
                start_index - 1,
                n_cloud_wavelengths,
                n_layers,
                cloud_scattering_reduction_factor,
                cloud_wavelengths,
                HIT_border_lamb[HIT_i_lamb] if start_index == intp_index_small_min else cloud_wavelengths[start_index-1],
                HIT_border_lamb[HIT_i_lamb + 1] if end_index == intp_index_small_max else cloud_wavelengths[end_index]
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
            _, _ = lax.while_loop(
                lambda val: val[0] < intp_index_small_max, 
                inner_loop_body, 
                (intp_index_small_min + 1, None)
                )

            integrate_section(intp_index_small_max, intp_index_small_max+1)

        kappa_integ = kappa_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        kappa_scat_integ = kappa_scat_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        red_fac_aniso_integ = red_fac_aniso_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])

        # kappa_tot_integ = kappa_tot_integ.at[HIT_i_lamb, :].set(kappa_integ)
        # kappa_tot_scat_integ = kappa_tot_scat_integ.at[HIT_i_lamb, :].set(kappa_scat_integ)

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
     cloud_scattering_reduction_factor_out) = lax.fori_loop(0, n_frequencies_bins - 1, loop_body, carry)

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
    photon_destruction_prob = compute_photon_destruction_prob(
        total_opacity_in,
        continuum_opacity_scattering_emis
        )
    transmission = compute_optical_depths(
        gravity,
        pressure,
        total_opacity_in
        )
    return transmission, photon_destruction_prob
