"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""
import sys
import jax.numpy as jnp
from jax import config, jit, vmap, lax, ShapeDtypeStruct, pure_callback, debug
from jax.typing import ArrayLike
from interpax import interp1d

import numpy as np
from functools import partial
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.radtrans_core.physics_core as physics_core
import petitRADTRANS.radtrans_core.linalg as linalg

CST_PI = jnp.pi  # Or the specific precision from Fortran
TINIEST = jnp.finfo(jnp.float32).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float32).max)
if config.jax_enable_x64:
    TINIEST = jnp.finfo(jnp.float64).tiny
    HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float64).max)


_UNROLL_SPECIES_LIMIT = 8


@jit
def combine_ck_opacities(opacities_fortran_order, g_gauss, weights_gauss,
                         input_layout='ng_f_s_l'):
    """
    Combines correlated-k opacities across species.

    Parameters
    ----------
    opacities_fortran_order : jax.Array
        Opacities sorted in ascending g-order along the n_g axis.
        Shape depends on input_layout (see below).
    g_gauss : jax.Array, shape (n_g,)
        Gaussian quadrature g-values, sorted ascending 0 -> 1.
    weights_gauss : jax.Array, shape (n_g,)
        Gaussian quadrature weights.
    input_layout : str, optional
        'ng_f_s_l'  (default) — shape (n_g, n_frequencies, n_species, n_layers).
            A transpose + copy is performed internally to reach (f*l, s, n_g).
        'fl_s_ng'               — shape (n_frequencies*n_layers, n_species, n_g).
            No transpose or copy. Use this layout if the caller already holds
            data in this order.

    Returns
    -------
    jax.Array, shape (n_g, n_frequencies, n_layers)
    """

    if input_layout == 'fl_s_ng':
        opacities_flat = opacities_fortran_order
        n_fl, n_species, n_g = opacities_flat.shape

        threshold_flat = (
            jnp.max(opacities_flat[:, :, 0], axis=1) * 1e-2
        )
        n_frequencies = n_fl
        n_layers = 1
    else:
        n_g, n_frequencies, n_species, n_layers = opacities_fortran_order.shape
        n_fl = n_frequencies * n_layers

        threshold_flat = (
            jnp.max(opacities_fortran_order[0, :, :, :], axis=1) * 1e-2
        ).reshape(n_fl)  # (f*l,)

        # Permute (n_g, f, s, l) -> (f, l, s, n_g) then flatten to (f*l, s, n_g).
        # The non-trivial permutation makes the view non-contiguous, so the
        # subsequent reshape forces a copy. Unavoidable with this input layout.
        opacities_flat = jnp.transpose(
            opacities_fortran_order, (1, 3, 2, 0)
        ).reshape(n_fl, n_species, n_g)

    # ------------------------------------------------------------------
    # Precompute constants shared across all (f*l, species) calls.
    # ------------------------------------------------------------------
    # weights_use: divide first 8 weights by 3, matching Fortran
    weights_use = weights_gauss.at[:8].divide(3.0)

    # Outer product of corrected weights, shape (n_g^2,).
    g_presort_flat = (weights_use[:, None] * weights_use[None, :]).reshape(-1)
    g_presort_sum = jnp.sum(g_presort_flat)  # scalar
    n_sample  = n_g * n_g
    _idx      = jnp.arange(n_sample)
    _pin_mask = (_idx == 0) | (_idx == n_sample - 1)   # (n_g^2,) bool, constant

    # ------------------------------------------------------------------
    # Core combination for a single (f*l) cell.
    # ------------------------------------------------------------------

    def _perform_combination(opac_out_lf, spec_opac_lf):
        """
        Combine opac_out_lf with spec_opac_lf for one cell.
        """
        # Outer sum of all g-point pairs -> (n_g^2,)
        k_combined_flat = (opac_out_lf[:, None] + spec_opac_lf[None, :]).reshape(-1)

        # Co-sort k and weights in a single radix pass.
        k_sorted, weights_sorted = lax.sort(
            operand=(k_combined_flat, g_presort_flat),
            num_keys=1,
        )
        weights_normalised = weights_sorted / g_presort_sum

        # Cumulative-sum midpoints.
        right_edges = jnp.cumsum(weights_normalised)          # (n_g^2,)
        g_x_vals    = right_edges - weights_normalised * 0.5  # (n_g^2,)
        k_min_val  = opac_out_lf[0]  + spec_opac_lf[0]   # scalar
        k_max_val  = opac_out_lf[-1] + spec_opac_lf[-1]  # scalar
        pin_vals   = jnp.where(_idx == 0, k_min_val, k_max_val)  # (n_g^2,)
        k_pinned   = jnp.where(_pin_mask, pin_vals, k_sorted)    # (n_g^2,) — one alloc

        return jnp.interp(g_gauss, g_x_vals, k_pinned, right=k_max_val)

    def process_one_cell(opacities_all_species_lf, threshold_lf):
        """
        Combine all species for a single (f*l) cell.
        """
        initial = opacities_all_species_lf[0, :]

        if n_species == 1:
            return initial

        if n_species <= _UNROLL_SPECIES_LIMIT:
            current = initial
            for s in range(1, n_species):
                spec = opacities_all_species_lf[s, :]
                current = lax.cond(
                    spec[-1] < threshold_lf,
                    lambda: current,
                    lambda: _perform_combination(current, spec),
                )
            return current

        # Fallback for large n_species.
        def scan_body(current_opac, spec_opac):
            return lax.cond(
                spec_opac[-1] < threshold_lf,
                lambda: current_opac,
                lambda: _perform_combination(current_opac, spec_opac),
            ), None

        final, _ = lax.scan(scan_body, initial, opacities_all_species_lf[1:, :])
        return final

    opacities_out_flat = vmap(process_one_cell)(
        opacities_flat, threshold_flat
    )  # (f*l, n_g)

    # Reshape and transpose back to (n_g, f, l).
    return jnp.transpose(
        opacities_out_flat.reshape(n_frequencies, n_layers, n_g),
        (2, 0, 1),
    )


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

    # Vectorize over emission angles using vmap: compute per-angle contributions then sum.
    r = physics_core.compute_planck_function(temperatures, frequencies)

    def _angle_contrib(mu, mu_w):
        transm_mu = jnp.exp(-optical_depths / mu)
        mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        # Radiative transport contribution for this angle
        flux_mu = jnp.sum((r[:, :-1] + r[:, 1:]) * (transm_all[:, 0:-1] - transm_all[:, 1:]) / 2.0,
                          axis=1)
        flux_mu = flux_mu + r[:, struct_shape[1]] * transm_all[:, struct_shape[1]]

        contr_local = jnp.zeros(struct_shape)
        contr_local = contr_local.at[:, :-1].set(
            (r[:, :-1] + r[:, 1:]) * (transm_all[:, 0:-1] - transm_all[:, 1:]) * mu * mu_w
        )
        contr_local = contr_local.at[:, -1].set(
            2.0 * r[:, struct_shape[1]] * transm_all[:, struct_shape[1]] * mu * mu_w
        )

        flux_contrib = flux_mu / 2.0 * mu * mu_w
        return flux_contrib, contr_local

    vmapped = vmap(_angle_contrib, in_axes=(0, 0))
    fluxs_per_angle, contrs_per_angle = vmapped(emission_cos_angles, emission_cos_angles_weights)

    # Sum over angles
    flux = jnp.sum(fluxs_per_angle, axis=0)
    contr_em = jnp.sum(contrs_per_angle, axis=0)

    # Normalization
    flux = flux * 4.0 * jnp.pi
    contr_em = contr_em.T
    contr_em = jnp.divide(contr_em, jnp.sum(contr_em, axis=0))
    return flux, contr_em


@jit
def compute_ck_flux(
        frequencies: ArrayLike,
        temperatures: ArrayLike,
        weights_gauss: ArrayLike,
        emission_cos_angles: ArrayLike,
        emission_cos_angles_weights: ArrayLike,
        optical_depths: ArrayLike,
        return_contribution: bool = False):
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

        # Vectorize over emission angles using vmap: compute per-angle flux then sum.
        r = physics_core.compute_planck_function(temperatures, frequencies)

        def _angle_contrib(mu, mu_w):
            transm_mu = jnp.exp(-optical_depths / mu)
            mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)
            transm_all = jnp.prod(mean_transm, axis=1)

            flux_mu = jnp.sum((r[:, :-1] + r[:, 1:]) * (transm_all[:, 0:-1] - transm_all[:, 1:]) * 0.5,
                              axis=1)
            flux_mu = flux_mu + r[:, struct_shape[1]] * transm_all[:, struct_shape[1]]

            return flux_mu * 0.5 * mu * mu_w

        vmapped = vmap(_angle_contrib, in_axes=(0, 0))
        fluxs_per_angle = vmapped(emission_cos_angles, emission_cos_angles_weights)
        flux = jnp.sum(fluxs_per_angle, axis=0)
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


# ---------------------------------------------------------------------------
# Ng acceleration — fully batched, no vmap
# ---------------------------------------------------------------------------

def _ng_accelerated_batched(tn, tn1, tn2, tn3, current_source):
    """
    Ng acceleration, fully vectorised over batch.

    All inputs/output: (batch, n_layers).
    """
    Q1 = tn - 2.0 * tn1 + tn2
    Q2 = tn - tn1 - tn2 + tn3
    Q3 = tn - tn1

    # Zero boundary on first element
    Q1 = Q1.at[:, 0].set(0.0)
    Q2 = Q2.at[:, 0].set(0.0)
    Q3 = Q3.at[:, 0].set(0.0)

    # O8: 5 explicit dot products instead of a full 3x3 Gram matmul.
    # Only G[0,0], G[1,0], G[1,1], G[0,2], G[1,2] are used; the other
    # four entries of the symmetric matrix are never read.
    A1 = jnp.minimum(jnp.einsum('bl,bl->b', Q1, Q1), HUGE_SQRT)
    A2 = jnp.minimum(jnp.einsum('bl,bl->b', Q2, Q1), HUGE_SQRT)
    B2 = jnp.minimum(jnp.einsum('bl,bl->b', Q2, Q2), HUGE_SQRT)
    C1 = jnp.minimum(jnp.einsum('bl,bl->b', Q1, Q3), HUGE_SQRT)
    C2 = jnp.minimum(jnp.einsum('bl,bl->b', Q2, Q3), HUGE_SQRT)

    AB_denom = A1 * B2 - A2 * A2

    skip = (
        (AB_denom < TINIEST)
        | (jnp.abs(A1) < 1e-250)
        | (jnp.abs(A2) < 1e-250)
        | (jnp.abs(B2) < 1e-250)
        | (jnp.abs(C1) < 1e-250)
        | (jnp.abs(C2) < 1e-250)
    )  # (batch,)

    safe_denom = jnp.where(AB_denom == 0.0, 1.0, AB_denom)
    a_ng = (C1 * B2 - C2 * A2) / safe_denom
    b_ng = (C2 * A1 - C1 * A2) / safe_denom

    accelerated = jnp.maximum(
        (1.0 - a_ng - b_ng)[:, None] * tn
        + a_ng[:, None] * tn1
        + b_ng[:, None] * tn2,
        0.0,
    )  # (batch, l)

    # In Fortran m_approximate_ng_source, when the early `return` fires, the
    # output argument `source` (5th arg) keeps whatever value it had on entry —
    return jnp.where(skip[:, None], current_source, accelerated)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

@jit(static_argnames=(
    "emission_geometry_str",
    "adaptive_feautrier_iterations",
    "return_contribution",
    "iter_scat_max",
))
def compute_feautrier_radiative_transfer(
        frequencies_bin_edges,               # (n_freq_bin_edges,)
        temperatures,                        # (n_layers,)
        weights_gauss,                       # (n_g,)
        emission_cos_angles,                 # (n_angles,)
        emission_cos_angles_weights,         # (n_angles,)
        optical_depths_in,                   # (n_g, n_freq_bins, n_layers)
        photon_destruction_probabilities_in, # (n_g, n_freq_bins, n_layers)
        stellar_intensity,                   # (n_freq_bins,)
        star_irradiation_cos_angle,          # scalar
        reflectances,                        # (n_freq_bins,)
        emissivities,                        # (n_freq_bins,)
        adaptive_feautrier_iterations: bool = True,
        emission_geometry_str: str = "dayside_ave",
        return_contribution: bool = False,
        iter_scat_max: int = 100,
        convergence_threshold: float = 1e-2,
):
    """Computes the atmospheric emission spectrum using the Feautrier method.
    """

    # ------------------------------------------------------------------
    # 0. Shapes and scalar constants
    # ------------------------------------------------------------------
    n_g, n_freq_bins, n_layers = photon_destruction_probabilities_in.shape
    n_angles = emission_cos_angles.shape[0]
    n_batch  = n_g * n_freq_bins * n_angles
    pi_4     = 4.0 * jnp.pi
    inv_dtau  = 1e10
    inv_dtau2 = inv_dtau * 0.5

    mu_double   = emission_cos_angles * 2.0
    surf_refl_2 = 2.0 * reflectances
    mu_weight_a = emission_cos_angles * emission_cos_angles_weights   # (a,)

    # ------------------------------------------------------------------
    # 1. Geometry coefficient — Python level, never traced
    # ------------------------------------------------------------------
    if emission_geometry_str == 'dayside_ave':
        geom = 0.5
    elif emission_geometry_str == 'planetary_ave':
        geom = 0.25
    elif emission_geometry_str == 'non-isotropic':
        geom = -1.0
    else:
        raise ValueError(f"Unknown emission_geometry: {emission_geometry_str}")

    # ------------------------------------------------------------------
    # 2. Photon destruction probabilities — clamped
    # ------------------------------------------------------------------
    pdp_gfl = jnp.maximum(photon_destruction_probabilities_in, 1e-10)

    # ------------------------------------------------------------------
    # 3. Stellar attenuation  J_star_ini_gfl (g, f, l)
    # ------------------------------------------------------------------
    if geom > 0:
        # exp_over_mu: (g, f, a, l)
        exp_over_mu_gfal = jnp.exp(
            -optical_depths_in[:, :, None, :]
            / emission_cos_angles[None, None, :, None]
        )

        J_star_ini_gfl = (
            0.5 * geom
            * jnp.abs(stellar_intensity)[None, :, None, None]
            * emission_cos_angles_weights[None, None, :, None]
            * exp_over_mu_gfal
        ).sum(axis=2)  # (g, f, l)

        # Surface reflection — slice exp at surface before releasing
        exp_surf_gfa = exp_over_mu_gfal[:, :, :, n_layers - 1]   # (g, f, a)
        del exp_over_mu_gfal

        I_surface_reflection_gf = (
            surf_refl_2[None, :]
            * (
                geom
                * jnp.abs(stellar_intensity)[None, :, None]
                * exp_surf_gfa
                * mu_weight_a[None, None, :]
            ).sum(axis=2)
        )  # (g, f)

    else:  # non-isotropic
        exp_gfl = jnp.exp(-optical_depths_in / star_irradiation_cos_angle)
        J_star_ini_gfl = jnp.abs(stellar_intensity[None, :, None] / 4.0 * exp_gfl)
        I_surface_reflection_gf = (
            reflectances[None, :]
            * J_star_ini_gfl[:, :, n_layers - 1]
            * 4.0 * star_irradiation_cos_angle
        )

    # ------------------------------------------------------------------
    # 4. Planck function
    # ------------------------------------------------------------------
    planck_integral_lf = physics_core.compute_planck_function_integral(
        temperatures, frequencies_bin_edges
    )  # (l, f)
    planck_integral_fl = planck_integral_lf.T   # (f, l) — zero-copy view

    # ------------------------------------------------------------------
    # 5. Surface boundary terms
    # ------------------------------------------------------------------
    I_surface_emission_f    = emissivities * planck_integral_lf[n_layers - 1, :]
    I_surface_no_scatter_gf = I_surface_emission_f[None, :] + I_surface_reflection_gf

    # ------------------------------------------------------------------
    # 6. Per-(g,f) minimum iteration count  (Fix 2)
    # ------------------------------------------------------------------
    if adaptive_feautrier_iterations:
        n_iter_min_gf = jnp.clip(
            1 + jnp.max(1.0 / pdp_gfl, axis=2),   # max over layers → (g, f)
            4, 9
        ).astype(jnp.int32)
    else:
        n_iter_min_gf = jnp.full((n_g, n_freq_bins), 9, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # 7. initial carry source = bare Planck (matches Fortran's
    #    `r = planck(:,i)` set before the iteration loop).
    # ------------------------------------------------------------------
    planck_broadcast_gfl = jnp.broadcast_to(
        planck_integral_fl[None, :, :], (n_g, n_freq_bins, n_layers)
    )  # (g, f, l)

    # ------------------------------------------------------------------
    # 8. Delta-taus
    # ------------------------------------------------------------------
    dtau1 = optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1]   # (g,f,l-1)
    dtau1 = jnp.where(jnp.abs(dtau1) < TINIEST, TINIEST, dtau1)

    dtau2 = optical_depths_in[:, :, 2:] - optical_depths_in[:, :, :-2]   # (g,f,l-2)
    dtau2 = jnp.where(jnp.abs(dtau2) < TINIEST, TINIEST, dtau2)

    # ------------------------------------------------------------------
    # 9. Feautrier tridiagonal coefficients
    # ------------------------------------------------------------------
    # top row (l=0)
    f1_top = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, 0, None],
        -inv_dtau, inv_dtau,
    )  # (g, f, a, 1)
    b_top = 1.0 + 2.0 * f1_top * (1.0 + f1_top)
    c_top = -2.0 * f1_top ** 2
    a_top = jnp.zeros_like(b_top)

    # mid rows (l=1 … n_layers-2)
    f1_mid = jnp.clip(
        mu_double[None, None, :, None] / dtau2[:, :, None, :],
        -inv_dtau2, inv_dtau2,
    )  # (g, f, a, l-2)
    f2_mid = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, 1:],
        -inv_dtau, inv_dtau,
    )  # (g, f, a, l-2)
    f3_mid = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, :-1],
        -inv_dtau, inv_dtau,
    )  # (g, f, a, l-2)
    b_mid = 1.0 + f1_mid * (f2_mid + f3_mid)
    c_mid = -f1_mid * f2_mid
    a_mid = -f1_mid * f3_mid

    a_flat = jnp.pad(
        jnp.concatenate([a_top, a_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
    ).reshape(n_batch, n_layers)                        # padded with 0
    b_flat = jnp.pad(
        jnp.concatenate([b_top, b_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
        constant_values=1.0,
    ).reshape(n_batch, n_layers)                        # padded with 1
    c_flat = jnp.pad(
        jnp.concatenate([c_top, c_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
    ).reshape(n_batch, n_layers)                        # padded with 0

    # ------------------------------------------------------------------
    # 10. Lambda_loc for ALI  (g, f, l)
    # ------------------------------------------------------------------
    lam_top = jnp.sum(
        mu_weight_a[None, None, :, None] / b_top, axis=2
    )[..., 0]  # (g, f)

    lam_mid = jnp.sum(
        mu_weight_a[None, None, :, None] / b_mid, axis=2
    )  # (g, f, l-2)

    f1_surf = jnp.clip(
        emission_cos_angles[None, None, :, None]
        / dtau1[:, :, None, n_layers - 2, None],
        -inv_dtau, inv_dtau,
    )  # (g, f, a, 1)
    lam_surf = jnp.sum(
        mu_weight_a[None, None, :, None] / (1.0 + 2.0 * f1_surf ** 2), axis=2
    )[..., 0]  # (g, f)

    lambda_loc_gfl = jnp.concatenate(
        [lam_top[:, :, None], lam_mid, lam_surf[:, :, None]], axis=-1
    )  # (g, f, l)

    # ------------------------------------------------------------------
    # 11. loop-invariant ALI intermediates precomputed once.
    # ------------------------------------------------------------------
    one_minus_pdp_gfl  = 1.0 - pdp_gfl                                    # (g,f,l)
    pdp_times_planck   = pdp_gfl * planck_integral_fl[None, :, :]          # (g,f,l)
    ali_inv_denom_gfl  = 1.0 / jnp.maximum(
        1.0 - one_minus_pdp_gfl * lambda_loc_gfl, TINIEST
    )                                                                       # (g,f,l)

    # ------------------------------------------------------------------
    # 12. Inner Feautrier solve
    # ------------------------------------------------------------------
    def _feautrier_solve(source_gfl, I_plus_surface_gfa, I_minus_boundary_gf):
        """
        Returns I_J_gfal (g,f,a,l), I_H_top (g,f,a), new_I_plus_surface (g,f,a).
        """
        # O3: repeat (g*f, l-1) → (g*f*a, l-1) via a single XLA repeat op.
        r_inner_flat = jnp.repeat(
            source_gfl[:, :, :n_layers - 1].reshape(n_g * n_freq_bins, n_layers - 1),
            n_angles,
            axis=0,
        )  # (batch, l-1)

        r_surf_flat = (
            0.5 * (I_plus_surface_gfa + I_minus_boundary_gf[:, :, None])
        ).reshape(n_batch, 1)  # (batch, 1)

        r_flat = jnp.concatenate([r_inner_flat, r_surf_flat], axis=-1)  # (batch, l)

        I_J_flat = linalg.solve_tridiagonal_thomas_batched(a_flat, b_flat, c_flat, r_flat)
        I_J_gfal = I_J_flat.reshape(n_g, n_freq_bins, n_angles, n_layers)

        I_H_top = -I_J_gfal[:, :, :, 0]   # (g, f, a)

        d1_surf = f2_mid[:, :, :, -1] * (
            I_J_gfal[:, :, :, n_layers - 1] - I_J_gfal[:, :, :, n_layers - 2]
        )
        new_I_plus_surface = I_J_gfal[:, :, :, n_layers - 1] - d1_surf  # (g, f, a)

        return I_J_gfal, I_H_top, new_I_plus_surface

    # ------------------------------------------------------------------
    # 13. Scattering iteration loop
    #
    # Carry:
    #   source_gfl       (g, f, l)    — interior RHS / current source
    #   I_plus_surface   (g, f, a)    — upward intensity at surface
    #   flux_gf_prev     (g, f)       — O5: replaces I_H_top_prev (g,f,a)
    #   I_minus_gf       (g, f)       — surface boundary for contribution fn
    #   ng_history       (4, g, f, l) — O1: ring-buffer replaces 4 arrays
    #   ng_idx           int32        — O1: ring-buffer write pointer
    #   iter_idx         int32
    #   converged_gf     (g, f) bool  — per-(g,f) convergence latch
    # ------------------------------------------------------------------

    def _surface_reflection(I_plus_gfa):
        return surf_refl_2[None, :] * jnp.sum(
            I_plus_gfa * mu_weight_a[None, None, :], axis=2
        )  # (g, f)

    def loop_body(carry):
        (source_gfl, I_plus_surface_gfa, flux_gf_prev, I_minus_gf,
         ng_history, ng_idx,
         iter_idx, converged_gf) = carry

        # 13a. Solve
        I_minus_gf_new = I_surface_no_scatter_gf + _surface_reflection(I_plus_surface_gfa)
        I_J_gfal, I_H_top, new_I_plus_surface = _feautrier_solve(
            source_gfl, I_plus_surface_gfa, I_minus_gf_new
        )

        # 13b. J_bol (g, f, l) = angular quadrature of mean intensity
        J_bol_gfl = jnp.sum(I_J_gfal * mu_weight_a[None, None, :, None], axis=2)

        # 13c. ALI source update using precomputed loop-invariant terms (O4).
        source_ali = (
            pdp_times_planck
            + one_minus_pdp_gfl * (J_star_ini_gfl + J_bol_gfl - lambda_loc_gfl * source_gfl)
        ) * ali_inv_denom_gfl

        # 13d. ring-buffer Ng history update.
        # Write source_ali into slot ng_idx%4; advance the pointer.
        # No array shifting — only one slice is written per iteration.
        ng_history = ng_history.at[ng_idx % 4].set(source_ali)
        next_ng_idx = ng_idx + 1

        # 13e. Ng via jnp.where instead of lax.cond.
        # Ng is always computed; blended in only every 4th iteration.
        # read history from ring-buffer by relative index.
        tn = ng_history[(ng_idx) % 4]   # most recent  = source_ali
        tn1 = ng_history[(ng_idx - 1) % 4]
        tn2 = ng_history[(ng_idx - 2) % 4]
        tn3 = ng_history[(ng_idx - 3) % 4]

        B = n_g * n_freq_bins
        ng_result = _ng_accelerated_batched(
            tn.reshape(B, n_layers),
            tn1.reshape(B, n_layers),
            tn2.reshape(B, n_layers),
            tn3.reshape(B, n_layers),
            source_ali.reshape(B, n_layers),   # post-ALI fallback
        ).reshape(n_g, n_freq_bins, n_layers)

        do_ng = (iter_idx > 0) & ((iter_idx + 1) % 4 == 0)   # scalar bool
        final_source = jnp.where(do_ng, ng_result, source_ali)

        # 13f. Convergence check.
        flux_gf_cur = -jnp.einsum('gfa,a->gf', I_H_top, mu_weight_a)  # (g, f)

        conv_val_gf = jnp.abs(
            1.0 - flux_gf_prev / jnp.where(
                jnp.abs(flux_gf_cur) < 1e-100, 1e-100, flux_gf_cur
            )
        )  # (g, f)

        # Per-(g,f) latch: once a pair converges it stays converged.
        newly_converged = (iter_idx > n_iter_min_gf) & (conv_val_gf < convergence_threshold)
        new_converged_gf = converged_gf | newly_converged

        next_carry = (
            final_source, new_I_plus_surface, flux_gf_cur, I_minus_gf_new,
            ng_history, next_ng_idx,
            iter_idx + 1, new_converged_gf,
        )
        return next_carry

    def continue_cond(carry):
        *_, iter_idx, converged_gf = carry
        return (iter_idx <= iter_scat_max) & jnp.logical_not(jnp.all(converged_gf))

    initial_carry = (
        planck_broadcast_gfl,                                    # Fix B: planck as first RHS
        jnp.zeros((n_g, n_freq_bins, n_angles)),                 # I_plus_surface_gfa
        jnp.zeros((n_g, n_freq_bins)),                           # O5: flux_gf_prev
        jnp.zeros((n_g, n_freq_bins)),                           # I_minus_gf
        jnp.zeros((4, n_g, n_freq_bins, n_layers)),              # O1: Ng ring-buffer
        jnp.int32(0),                                            # O1: ng_idx
        jnp.int32(0),                                            # iter_idx
        jnp.zeros((n_g, n_freq_bins), dtype=jnp.bool_),          # converged_gf
    )

    final_carry = lax.while_loop(continue_cond, loop_body, initial_carry)

    # iteration — I_H_top already reduced over angles. Use it directly.
    (final_source_gfl, _, flux_gf_final, I_minus_gf_final,
     _, _, _, _) = final_carry

    # ------------------------------------------------------------------
    # 14. Flux  (n_freq_bins,)
    # ------------------------------------------------------------------
    flux_f = pi_4 * jnp.einsum('gf,g->f', flux_gf_final, weights_gauss)

    # ------------------------------------------------------------------
    # 15. Contribution function (optional)
    # ------------------------------------------------------------------
    emission_contribution_lf = jnp.zeros((n_layers, n_freq_bins))

    def compute_contribution():
        # G-integrated source: (l, f)
        r_src_lf = jnp.einsum('g,gfl->lf', weights_gauss, final_source_gfl)

        # G-integrated I_minus: (f,)
        I_minus_f = jnp.einsum('g,gf->f', weights_gauss, I_minus_gf_final)

        # This replaces n_angles separate exp(-tau/mu) calls inside fori_loop.
        transm_gfla = jnp.exp(
            -optical_depths_in[:, :, :, None]
            / emission_cos_angles[None, None, None, :]
        )  # (g, f, l, a)
        transm_lfa = jnp.einsum('g,gfla->lfa', weights_gauss, transm_gfla)  # (l, f, a)
        del transm_gfla

        def i_mu_body(i_mu, acc):
            mu_val        = emission_cos_angles[i_mu]
            mu_weight_val = emission_cos_angles_weights[i_mu]

            # O6: no exp() here — just slice the precomputed array.
            transm_lf = transm_lfa[:, :, i_mu]   # (l, f)

            contrib_mid = (
                (r_src_lf[:-1, :] + r_src_lf[1:, :])
                * (transm_lf[:-1, :] - transm_lf[1:, :])
                * mu_val * mu_weight_val
            )  # (l-1, f)

            contrib_last = (
                2.0 * I_minus_f
                * transm_lf[n_layers - 1, :]
                * mu_val * mu_weight_val
            )  # (f,)

            acc = acc.at[:-1, :].add(contrib_mid)
            acc = acc.at[n_layers-1, :].add(contrib_last)
            return acc

        accumulated = lax.fori_loop(
            0, n_angles, i_mu_body, jnp.zeros((n_layers, n_freq_bins))
        )
        total = jnp.sum(accumulated, axis=0, keepdims=True)
        return jnp.where(jnp.abs(total) < 1e-100, 0.0, accumulated / total)

    emission_contribution_lf = lax.cond(
        return_contribution,
        compute_contribution,
        lambda: emission_contribution_lf,
    )

    return flux_f, emission_contribution_lf


@jit
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
        # Add continuum scattering to the first species (assumed at index 0)
        _opacities = opacities.at[:, :, 0, :].add(continuum_opacities_scattering)
        # Compute photon destruction probabilities for all g, freq, press
        total_opac = _opacities[:, :, 0, :]
        photon_destruction_probabilities = continuum_opacities_scattering / total_opac
        photon_destruction_probabilities = 1.0 - photon_destruction_probabilities
        return photon_destruction_probabilities, _opacities

    def false_function():
        return jnp.ones(opacities[:, :, 0, :].shape), opacities

    photon_destruction_probabilities, _opacities = lax.cond(
        scattering_in_emission,
        scattering_function,
        false_function
    )

    photon_destruction_probabilities = jnp.clip(photon_destruction_probabilities, 0.0, 1.0)

    # Vectorized optical depth calculation
    pressure_diffs = pressures[1:] - pressures[:-1]  # (n_layers-1,)
    avg_opacities = (_opacities[..., 1:] + _opacities[..., :-1]) / 2.0  # (..., n_layers-1)
    d_optical_depths = avg_opacities * pressure_diffs / reference_gravity  # (..., n_layers-1)
    # Pad with zeros at the start for cumulative sum
    d_optical_depths_padded = jnp.pad(d_optical_depths, [(0, 0)] * (d_optical_depths.ndim - 1) + [(1, 0)])
    optical_depths = jnp.cumsum(d_optical_depths_padded, axis=-1)

    return optical_depths, photon_destruction_probabilities


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
    total_opacity = total_opacity_in[:, :, 0, :]

    # Vectorized per-g computation: add continuum opacity and compute destruction probability
    def _per_g(opacity_g):
        tot = opacity_g + continuum_opacity_scat_emis
        return continuum_opacity_scat_emis / tot

    photon_destruction_prob = vmap(_per_g)(total_opacity)
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
    B_nu = physics_core.star_planck(temperatures, frequencies_bin_edges)

    kappa_planck = jnp.sum(B_nu
                           * jnp.sum(weights_gauss[:, None, None]/opacities[:, :, :], axis=0)
                           * diff_freqs[:, None],
                           axis=0)
    norm = jnp.sum(B_nu * diff_freqs[:, None], axis=0)
    return kappa_planck / norm


@jit
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
        variable_gravity=True):
    """
    Calculate the radius at hydrostatic equilibrium for a planetary atmosphere.

    This function integrates the hydrostatic equilibrium equation to compute the radius
    at each atmospheric layer, given the pressure grid, atmospheric density profile,
    reference gravity, reference pressure, and planet radius.

    Args:
        pressures (jax.Array): 1D array of atmospheric pressures in cgs units (dyne/cm^2).
                               Shape: (n_layers,).
        reference_gravity (float): Reference gravity in cgs units (cm/s^2).
        atmosphere_densities (jax.Array): 1D array of atmospheric densities in cgs units (g/cm^3).
                                          Shape: (n_layers,).
        reference_pressure (float): Reference pressure in cgs units (dyne/cm^2).
        planet_radius (float): Planet radius in cm.
        variable_gravity (bool, optional): If True, gravity varies as 1/r^2 with radius.
                                           If False, gravity is constant throughout the atmosphere.

    Returns:
        jax.Array: 1D array of radius values at each atmospheric layer in cm.
                   Shape: (n_layers,).
    """

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


@partial(jit, static_argnames=("variable_gravity"))
def compute_transit_radii(
        opacities, continuum_opacities_scattering, pressures, temperatures, weights_gauss, mean_molar_masses,
        reference_gravity, reference_pressure, planet_radius, variable_gravity=True
        ):
    """Calculate the planetary transmission spectrum.
        # TODO complete docstring
        Args:
            opacities:
            continuum_opacities_scattering:
            pressures:
            temperatures:
            weights_gauss:
            mean_molar_masses:
                Mean molecular weight in units of amu.
                (1-d numpy array, same length as pressure array).
            reference_gravity (float):
                Atmospheric gravitational acceleration at reference pressure and radius in units of
                dyne/cm^2
            reference_pressure (float):
                Reference pressure in bar.
            planet_radius (float):
                Planet radius in cm.
            variable_gravity (bool):
                If True, gravity in the atmosphere will vary proportional to 1/r^2, where r is the planet
                radius.
            copy_opacities (bool):
                If True, a new array will be created for the transparencies, distinct from the opacities. This
                increases memory usage. Not needed anymore with JAX, arrays are immutable.

        Returns:
            * transmission radius in cm (1-d numpy array, as many elements as wavelengths)
            * planet radius as function of atmospheric pressure (1-d numpy array, as many elements as atmospheric
            layers)
    """

    # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
    # (temperature, pressure, mmw), gravity, reference pressure and radius.
    pressures_cgs = pressures * 1e6  # bar to cgs
    reference_pressure_cgs = reference_pressure * 1e6  # bar to cgs

    atmospheric_densities = pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures
    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures=pressures_cgs,
        reference_gravity=reference_gravity,
        atmosphere_densities=atmospheric_densities,
        reference_pressure=reference_pressure_cgs,
        planet_radius=planet_radius,
        variable_gravity=variable_gravity
    )

    is_negative_condition = radius_hydrostatic_equilibrium < 0.
    first_non_negative_idx = jnp.argmax(radius_hydrostatic_equilibrium >= 0.)  # Index of first True
    replacement_value = radius_hydrostatic_equilibrium[first_non_negative_idx]
    radius_hydrostatic_equilibrium = jnp.where(is_negative_condition, replacement_value, radius_hydrostatic_equilibrium)

    # Calculate the density
    rho = (pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures)

    # Bring continuum scattering opacities in right shape for matrix operations later.
    # Reminder: when calling this function, continuum absorption opacities have already
    # been added to line_struc_kappas.
    continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape(
        (1, continuum_opacities_scattering.shape[0], 1, pressures.size)
        )

    transparencies = opacities*rho
    transparencies = transparencies.at[:, :, :1, :].add(continuum_opa_scat_reshaped * rho)

    # Calculate average mean free path between neighboring layers for later integration
    # Factor 1/2 is omitted because it cancels with effective planet area integration below.
    transparencies = transparencies.at[:, :, :, 1:].add(transparencies[:, :, :, :-1])

    # Prepare matrix for delta path lengths during optical depth integration
    diff_s = jnp.zeros((pressures.size, pressures.size))

    # Calculate matrix of delta path lengths
    r_ik = (radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.
            - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.)
    r_ik = jnp.where(r_ik < 0., 0., r_ik)
    r_ik = jnp.sqrt(r_ik)
    diff_s = diff_s.at[1:, 1:].set(- r_ik[1:, 1:] + r_ik[1:, :-1])

    # Calculate optical depths
    transparencies = jnp.einsum('ijkl,ml', transparencies, diff_s)

    # Delete unnecessary intermediate array
    del diff_s

    # Calculate transmittances (exp(-optical_depths), out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.exp(-1 * transparencies)

    # Integrate over g if c-k
    transparencies = lax.cond(
        weights_gauss.size == 1,
        lambda: transparencies[0],
        lambda: jnp.einsum('ijkl,i', transparencies, weights_gauss)
        )

    transparencies = jnp.swapaxes(transparencies, 1, 0)  # (species, frequencies, levels)

    transparencies = lax.cond(
        transparencies.shape[0] == 1,
        lambda: transparencies[0],
        lambda: jnp.prod(transparencies, axis=0)
    )

    # Calculate transparencies (1 - transmittances)
    transparencies *= -1
    transparencies += 1

    # Annulus radius increments
    annulus_radius_increments = -jnp.diff(radius_hydrostatic_equilibrium)

    # Integrate effective area, 2 pi omitted:
    # - 2 cancels with 1/2 of average inverse mean free path above
    # - pi cancels when calculating the radius from the area below
    transparencies *= radius_hydrostatic_equilibrium
    transparencies = transparencies.at[:, 1:].add(transparencies[:, :-1])
    transparencies = transparencies[:, 1:]  # get rid of unnecessary layer
    transparencies *= annulus_radius_increments
    transit_radii = jnp.sum(transparencies, axis=1)  # (frequencies)

    # Transform area to transmission radius
    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.)

    return transit_radii, radius_hydrostatic_equilibrium


@partial(jit, static_argnames=("variable_gravity"))
def compute_transit_radii_summed(
        opacities, continuum_opacities_scattering, pressures, temperatures, weights_gauss, mean_molar_masses,
        reference_gravity, reference_pressure, planet_radius, variable_gravity=True
        ):
    """Calculate the planetary transmission spectrum.
        # TODO complete docstring
        Args:
            opacities:
            continuum_opacities_scattering:
            pressures:
            temperatures:
            weights_gauss:
            mean_molar_masses:
                Mean molecular weight in units of amu.
                (1-d numpy array, same length as pressure array).
            reference_gravity (float):
                Atmospheric gravitational acceleration at reference pressure and radius in units of
                dyne/cm^2
            reference_pressure (float):
                Reference pressure in bar.
            planet_radius (float):
                Planet radius in cm.
            variable_gravity (bool):
                If True, gravity in the atmosphere will vary proportional to 1/r^2, where r is the planet
                radius.
            summed_species_opacities (bool):
                If True, it is assumed that the given opacities contains the summed opacities of all species.
                This is the case in line-by-line.
            copy_opacities (bool):
                If True, a new array will be created for the transparencies, distinct from the opacities. This
                increases memory usage. Not needed anymore with JAX, arrays are immutable.

        Returns:
            * transmission radius in cm (1-d numpy array, as many elements as wavelengths)
            * planet radius as function of atmospheric pressure (1-d numpy array, as many elements as atmospheric
            layers)
    """
    # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
    # (temperature, pressure, mmw), gravity, reference pressure and radius.
    pressures_cgs = pressures * 1e6  # bar to cgs
    reference_pressure_cgs = reference_pressure * 1e6  # bar to cgs

    atmospheric_densities = pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures
    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures=pressures_cgs,
        reference_gravity=reference_gravity,
        atmosphere_densities=atmospheric_densities,
        reference_pressure=reference_pressure_cgs,
        planet_radius=planet_radius,
        variable_gravity=variable_gravity
    )

    is_negative_condition = radius_hydrostatic_equilibrium < 0.
    first_non_negative_idx = jnp.argmax(radius_hydrostatic_equilibrium >= 0.)  # Index of first True
    replacement_value = radius_hydrostatic_equilibrium[first_non_negative_idx]
    radius_hydrostatic_equilibrium = jnp.where(is_negative_condition, replacement_value, radius_hydrostatic_equilibrium)

    # Calculate the density
    rho = (pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures)

    # Bring continuum scattering opacities in right shape for matrix operations later.
    # Reminder: when calling this function, continuum absorption opacities have already
    # been added to line_struc_kappas.
    continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape(
        (1, continuum_opacities_scattering.shape[0], 1, pressures.size)
        )

    transparencies = opacities[:, :, :1, :] * rho
    transparencies += continuum_opa_scat_reshaped * rho

    # Calculate average mean free path between neighboring layers for later integration
    # Factor 1/2 is omitted because it cancels with effective planet area integration below.
    transparencies = transparencies.at[:, :, :, 1:].add(transparencies[:, :, :, :-1])

    # Prepare matrix for delta path lengths during optical depth integration
    diff_s = jnp.zeros((pressures.size, pressures.size))

    # Calculate matrix of delta path lengths
    r_ik = (radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.
            - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.)
    r_ik = jnp.where(r_ik < 0., 0., r_ik)
    r_ik = jnp.sqrt(r_ik)
    diff_s = diff_s.at[1:, 1:].set(- r_ik[1:, 1:] + r_ik[1:, :-1])

    # Calculate optical depths (out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.einsum('ijkl,ml', transparencies, diff_s)

    # Delete unnecessary intermediate array
    del diff_s

    # Calculate transmittances (exp(-optical_depths), out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.exp(-1 * transparencies)

    transparencies = lax.cond(
        weights_gauss.size == 1,
        lambda: transparencies[0],
        lambda: jnp.einsum('ijkl,i', transparencies, weights_gauss, optimize=True)
        )

    transparencies = jnp.swapaxes(transparencies, 1, 0)  # (species, frequencies, levels)

    transparencies = lax.cond(
        transparencies.shape[0] == 1,
        lambda: transparencies[0],
        lambda: jnp.prod(transparencies, axis=0)
        )

    # Calculate transparencies (1 - transmittances)
    transparencies *= -1
    transparencies += 1

    # Annulus radius increments
    annulus_radius_increments = -jnp.diff(radius_hydrostatic_equilibrium)

    # Integrate effective area, 2 pi omitted:
    # - 2 cancels with 1/2 of average inverse mean free path above
    # - pi cancels when calculating the radius from the area below
    transparencies *= radius_hydrostatic_equilibrium
    transparencies = transparencies.at[:, 1:].add(transparencies[:, :-1])
    transparencies = transparencies[:, 1:]  # get rid of unnecessary layer
    transparencies *= annulus_radius_increments
    transit_radii = jnp.sum(transparencies, axis=1)  # (frequencies)

    # Transform area to transmission radius
    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.)

    return transit_radii, radius_hydrostatic_equilibrium


@partial(jit, static_argnames=["scattering_in_transmission", "variable_gravity"])
def compute_transit_radii_fortran(
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


@partial(jit, static_argnames=["variable_gravity", "scattering_in_transmission"] )
def compute_transmission_spectrum_contribution(
    opacities,  # (n_g, n_frequencies, n_species, n_layers)
    temperatures,  # (n_layers)
    pressures,  # (n_layers)
    reference_gravity,
    mean_molar_masses,  # (n_layers)
    reference_pressure,  # scalar, in bar
    planet_radius,  # scalar, in cm
    weights_gauss,  # (n_g)
    transit_radii_squared,  # (n_frequencies) -> R_eff^2 with all layers
    continuum_opacities_scattering,  # (n_frequencies, n_layers)
    scattering_in_transmission=True,  # boolean
    variable_gravity=True,  # boolean
):
    """
    JAX version of the Fortran subroutine compute_transmission_spectrum_contribution.
    Calculates the contribution of each atmospheric layer to the transmission spectrum.
    Uses lax.cond for conditional operations based on scattering_in_transmission.
    """
    n_g = opacities.shape[0]
    n_frequencies = opacities.shape[1]
    n_species = opacities.shape[2]
    n_layers = opacities.shape[3]

    P0_cgs = reference_pressure * 1e6
    pressures_cgs = pressures * 1e6
    rho = (mean_molar_masses * cst.amu * pressures_cgs) / (cst.kB * temperatures)

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures_cgs, reference_gravity, rho, P0_cgs, planet_radius,
        variable_gravity
    )
    R_sq = radius_hydrostatic_equilibrium**2

    def loop_body_leave_out_layer(i_leave_str_idx, _):
        current_opacities = opacities.at[..., i_leave_str_idx].set(0.0)

        # --- Conditionally set current_continuum_scattering for the left-out layer ---
        def set_continuum_scattering_true_branch(operands):
            _cos, _idx = operands
            return _cos.at[:, _idx].set(0.0)

        def set_continuum_scattering_false_branch(operands):
            _cos, _ = operands
            return _cos # Return original if no scattering

        current_continuum_scattering_eff = lax.cond(
            scattering_in_transmission,
            set_continuum_scattering_true_branch,
            set_continuum_scattering_false_branch,
            (continuum_opacities_scattering, i_leave_str_idx)
        )
        # If not scattering_in_transmission, continuum_opacities_scattering might not be used
        # or could be zeros. If it's zeros, the above is fine.
        # If it's non-zero but scattering_in_transmission is False, we should ensure it's effectively zeroed.
        # A safer approach if continuum_opacities_scattering might be non-zero when flag is False:
        current_continuum_scattering_for_calc = lax.cond(
            scattering_in_transmission,
            lambda cos_in: cos_in.at[:, i_leave_str_idx].set(0.0),
            lambda cos_in: jnp.zeros_like(cos_in), # Ensure it's zero if flag is false
            continuum_opacities_scattering
        )


        rho_exp = rho[None, None, None, :]
        alpha_t2_current = (current_opacities[..., :-1] * rho_exp[..., :-1] +
                            current_opacities[..., 1:] * rho_exp[..., 1:])

        # --- Conditionally calculate alpha_t2_scat_current ---
        alpha_t2_scat_operands = (rho, current_continuum_scattering_for_calc, n_frequencies, n_layers)

        def true_alpha_t2_scat_branch(ops):
            _rho, _ccs, _, _ = ops
            _rho_exp_scat = _rho[None, :]
            return (_ccs[:, :-1] * _rho_exp_scat[:, :-1] +
                    _ccs[:, 1:] * _rho_exp_scat[:, 1:])

        def false_alpha_t2_scat_branch(ops):
            #_, _, _n_freq, _n_layers_ops = ops  # Renamed to avoid conflict
            return jnp.zeros((n_frequencies, n_layers - 1))

        alpha_t2_scat_current = lax.cond(
            scattering_in_transmission,
            true_alpha_t2_scat_branch,
            false_alpha_t2_scat_branch,
            alpha_t2_scat_operands
        )

        t_graze_current_init = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
        t_graze_scat_current_init = jnp.zeros((n_frequencies, n_layers))

        def i_str_loop_body_for_t_graze(i_str_f_minus_2, current_t_grazes_tuple):
            current_t_graze_accum, current_t_graze_scat_accum = current_t_grazes_tuple
            i_str_f = i_str_f_minus_2 + 2
            R_impact_sq_curr = R_sq[i_str_f - 1]

            def j_str_loop_body_for_t_graze(j_str_f_minus_1, grazes_sum_for_i_str_tuple):
                t_graze_sum_gas, t_graze_sum_scat = grazes_sum_for_i_str_tuple
                j_str_f = j_str_f_minus_1 + 1
                idx_j_alpha = j_str_f - 1

                s_outer = jnp.sqrt(jnp.maximum(0., R_sq[j_str_f - 1] - R_impact_sq_curr))
                s_inner = lax.cond(
                    (j_str_f + 1) == i_str_f,
                    lambda _: 0.0,
                    lambda _: jnp.sqrt(jnp.maximum(0., R_sq[j_str_f] - R_impact_sq_curr)),
                    operand=None
                )
                path_in_shell = s_outer - s_inner

                t_graze_sum_gas += alpha_t2_current[..., idx_j_alpha] * path_in_shell

                # --- Conditionally update t_graze_scat_sum ---
                def update_t_graze_scat_sum_true(ops):
                    _tgs_sum, _alpha_val, _path = ops
                    return _tgs_sum + _alpha_val * _path

                def update_t_graze_scat_sum_false(ops):
                    _tgs_sum, _, _ = ops
                    return _tgs_sum

                t_graze_sum_scat_ops = (t_graze_sum_scat, alpha_t2_scat_current[:, idx_j_alpha], path_in_shell)
                t_graze_sum_scat = lax.cond(
                    scattering_in_transmission,
                    update_t_graze_scat_sum_true,
                    update_t_graze_scat_sum_false,
                    t_graze_sum_scat_ops
                )
                return t_graze_sum_gas, t_graze_sum_scat

            initial_sums_for_i_str_gas = jnp.zeros_like(alpha_t2_current[..., 0])
            initial_sums_for_i_str_scat = jnp.zeros_like(alpha_t2_scat_current[:, 0]) # Always init, even if not used

            final_t_graze_sum_gas, final_t_graze_sum_scat = lax.fori_loop(
                0, i_str_f - 1, j_str_loop_body_for_t_graze,
                (initial_sums_for_i_str_gas, initial_sums_for_i_str_scat)
            )

            current_t_graze_accum = current_t_graze_accum.at[..., i_str_f - 1].set(final_t_graze_sum_gas)

            # --- Conditionally update current_t_graze_scat_accum ---
            def update_scat_accum_true(ops):
                _acc, _val, _idx = ops
                return _acc.at[:, _idx].set(_val)

            def update_scat_accum_false(ops):
                _acc, _, _ = ops
                return _acc

            current_t_graze_scat_accum_ops = (current_t_graze_scat_accum, final_t_graze_sum_scat, i_str_f - 1)
            current_t_graze_scat_accum = lax.cond(
                scattering_in_transmission,
                update_scat_accum_true,
                update_scat_accum_false,
                current_t_graze_scat_accum_ops
            )
            return current_t_graze_accum, current_t_graze_scat_accum

        t_graze_current, t_graze_scat_current = lax.fori_loop(
            0, n_layers - 1, i_str_loop_body_for_t_graze,
            (t_graze_current_init, t_graze_scat_current_init)
        )

        exp_t_graze = jnp.exp(-t_graze_current)
        T_gas_spec_contrib = jnp.sum(exp_t_graze * weights_gauss[:, None, None, None], axis=0)
        T_gas_total = jnp.prod(T_gas_spec_contrib, axis=1)

        # --- Conditionally calculate T_scat_eff ---
        # T_gas_total is shape (n_frequencies, n_layers)
        # t_graze_scat_current is shape (n_frequencies, n_layers)
        def compute_T_scat_eff_true(ops):
            _t_graze_scat, _ = ops
            return jnp.exp(-_t_graze_scat)

        def compute_T_scat_eff_false(ops):
            _, _T_gas_total_template = ops
            return jnp.ones_like(_T_gas_total_template)

        T_scat_eff_ops = (t_graze_scat_current, T_gas_total)
        T_scat_eff = lax.cond(
            scattering_in_transmission,
            compute_T_scat_eff_true,
            compute_T_scat_eff_false,
            T_scat_eff_ops
        )

        T_layer_freq = T_gas_total * T_scat_eff
        t_graze_wlen_int_temp = T_layer_freq.T

        y_val = (1.0 - t_graze_wlen_int_temp) * radius_hydrostatic_equilibrium[:, None]
        delta_R = radius_hydrostatic_equilibrium[:-1] - radius_hydrostatic_equilibrium[1:]
        integrand_terms = (y_val[:-1, :] + y_val[1:, :]) * delta_R[:, None]
        integral_val = jnp.sum(integrand_terms, axis=0)
        transit_radii_temp_sq = integral_val + R_sq[-1]

        contrib_for_this_layer = transit_radii_squared - transit_radii_temp_sq
        return contrib_for_this_layer

    def accumulate_contributions_loop(i_leave_str_idx, contributions_array):
        contrib_for_this_layer = loop_body_leave_out_layer(i_leave_str_idx, None)
        contributions_array = contributions_array.at[i_leave_str_idx, :].set(contrib_for_this_layer)
        return contributions_array

    initial_contributions_array = jnp.zeros((n_layers, n_frequencies))
    transmission_contribution_raw = lax.fori_loop(
        0, n_layers, accumulate_contributions_loop, initial_contributions_array
    )

    sum_contrib = jnp.sum(transmission_contribution_raw, axis=0, keepdims=True)

    transmission_contribution_final = jnp.where(
        sum_contrib == 0,
        0.0,
        transmission_contribution_raw / sum_contrib
    )

    return transmission_contribution_final, radius_hydrostatic_equilibrium


@jit
def integrate_opacities(intp_ind, kappa, lambd, l_bord1, l_bord2):
    """Calculate the integral of a linearly interpolated function kappa."""

    kappa_integ = kappa[intp_ind, :] * (l_bord2 - l_bord1) + \
        (kappa[intp_ind + 1, :] - kappa[intp_ind, :]) / \
        (lambd[intp_ind + 1] - lambd[intp_ind]) * \
        0.5 * ((l_bord2 - lambd[intp_ind])**2 - (l_bord1 - lambd[intp_ind])**2)
    return kappa_integ


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
