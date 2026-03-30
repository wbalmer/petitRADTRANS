"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""

from typing import Any, Optional
import jax.numpy as jnp
from jax import Array, config, jit, vmap, lax, debug
from jax.typing import ArrayLike
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


def build_ck_combination_state(weights_gauss: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Precompute the pairwise quadrature weights used for c-k mixing.

    Parameters
    ----------
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Pre-normalized pairwise g-combination weights (sum to 1) and their
        original sum before normalization.
    """
    # C4: Use weights_gauss directly — matches Fortran, which does not apply
    # the /3 adjustment to the outer-product weights.
    g_combination_weights = (weights_gauss[:, None] * weights_gauss[None, :]).reshape(
        -1
    )
    g_combination_weights_sum = jnp.sum(g_combination_weights)
    # C1: Pre-normalize so the inner loop avoids a per-call vector division.
    g_combination_weights = g_combination_weights / g_combination_weights_sum

    return g_combination_weights, g_combination_weights_sum


@jit(static_argnames=("input_layout",))
def combine_ck_opacities(
    opacities_fortran_order: ArrayLike,
    g_gauss: ArrayLike,
    weights_gauss: Optional[ArrayLike] = None,
    line_species_mass_fractions: Optional[ArrayLike] = None,
    continuum_opacities: Optional[ArrayLike] = None,
    g_combination_weights: Optional[ArrayLike] = None,
    g_combination_weights_sum: Optional[ArrayLike] = None,
    input_layout: str = "ng_f_s_l",
) -> ArrayLike:
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

    if g_combination_weights is None or g_combination_weights_sum is None:
        if weights_gauss is None:
            raise ValueError(
                "weights_gauss must be provided when precomputed c-k state is unavailable"
            )
        g_combination_weights, g_combination_weights_sum = build_ck_combination_state(
            weights_gauss
        )

    if line_species_mass_fractions is None or continuum_opacities is None:
        raise ValueError(
            "line_species_mass_fractions and continuum_opacities are required"
        )

    if input_layout == "fl_s_ng":
        opacities_flat = opacities_fortran_order
        n_fl, n_species, n_g = opacities_flat.shape
        n_frequencies, n_layers = continuum_opacities.shape
    else:
        n_g, n_frequencies, n_species, n_layers = opacities_fortran_order.shape
        n_fl = n_frequencies * n_layers

        # Permute (n_g, f, s, l) -> (f, l, s, n_g) then flatten to (f*l, s, n_g).
        # The non-trivial permutation makes the view non-contiguous, so the
        # subsequent reshape forces a copy. Unavoidable with this input layout.
        opacities_flat = jnp.transpose(opacities_fortran_order, (1, 3, 2, 0)).reshape(
            n_fl, n_species, n_g
        )

    abundance_weights_flat = jnp.broadcast_to(
        line_species_mass_fractions[None, :, :],
        (n_frequencies, n_layers, n_species),
    ).reshape(n_fl, n_species)
    continuum_flat = continuum_opacities.reshape(n_fl)

    # ------------------------------------------------------------------
    # Precompute constants shared across all (f*l, species) calls.
    # ------------------------------------------------------------------
    n_sample = n_g * n_g

    # ------------------------------------------------------------------
    # Core batched combination across all (f×l) cells.
    # Uses a single batched sort, cumsum, and searchsorted instead of
    # vmap over per-cell sort+interp, reducing kernel dispatch overhead.
    # ------------------------------------------------------------------

    def _perform_combination_batched(
        current: ArrayLike,
        spec_s: ArrayLike,
    ) -> ArrayLike:
        """Combine current (n_fl, n_g) with spec_s (n_fl, n_g) for all cells."""
        n_fl_local = current.shape[0]

        # Outer sum of all g-point pairs → (n_fl, n_g²)
        k_combined = (current[:, :, None] + spec_s[:, None, :]).reshape(
            n_fl_local, n_sample
        )

        # Sort keys only; reconstruct weight order via permutation indices.
        # Avoids co-sorting a tiled weight array (halves sort data movement).
        perm = jnp.argsort(k_combined, axis=-1)
        k_sorted = jnp.take_along_axis(k_combined, perm, axis=-1)
        weights_sorted = g_combination_weights[perm]  # 1D→2D broadcast gather

        # Cumulative-sum midpoints (batched along last axis).
        right_edges = jnp.cumsum(weights_sorted, axis=-1)
        g_x_vals = right_edges - weights_sorted * 0.5

        # Pin endpoints.
        k_sorted = k_sorted.at[:, 0].set(current[:, 0] + spec_s[:, 0])
        k_sorted = k_sorted.at[:, n_sample - 1].set(current[:, -1] + spec_s[:, -1])

        # Batched binary search — replaces vmap(searchsorted).
        # 9 iterations covers 2^9 = 512 > n_sample = 256.
        g_target = g_gauss[None, :]  # (1, n_g)
        rows = jnp.arange(n_fl_local)[:, None]  # (n_fl, 1)
        lo = jnp.zeros((n_fl_local, n_g), dtype=jnp.int32)
        hi = jnp.full((n_fl_local, n_g), n_sample - 1, dtype=jnp.int32)

        for _ in range(9):
            mid = (lo + hi) >> 1
            go_right = g_x_vals[rows, mid] < g_target
            lo = jnp.where(go_right, mid + 1, lo)
            hi = jnp.where(go_right, hi, mid)

        idx = jnp.clip(lo, 1, n_sample - 1)
        idx_lo = idx - 1
        x0 = g_x_vals[rows, idx_lo]
        x1 = g_x_vals[rows, idx]
        y0 = k_sorted[rows, idx_lo]
        y1 = k_sorted[rows, idx]
        t = jnp.clip((g_target - x0) / jnp.maximum(x1 - x0, TINIEST), 0.0, 1.0)
        return y0 + t * (y1 - y0)

    # ------------------------------------------------------------------
    # C2 + C3: Vectorized threshold & species loop.
    # ------------------------------------------------------------------

    # Weight all species by abundance  (n_fl, n_species, n_g)
    weighted_all = opacities_flat * abundance_weights_flat[:, :, None]

    # Initialize with first species + continuum  (n_fl, n_g)
    current = weighted_all[:, 0, :] + continuum_flat[:, None]

    if n_species > 1:
        # C2: Vectorized threshold matching Fortran structure.
        threshold = jnp.max(weighted_all[:, :, -1], axis=1) * 1e-2  # (n_fl,)
        active_mask = weighted_all[:, :, -1] >= threshold[:, None]  # (n_fl, n_species)

        # Batched species loop — no vmap, single sort kernel per species.
        for s in range(1, n_species):
            spec_s = weighted_all[:, s, :]  # (n_fl, n_g)
            combined = _perform_combination_batched(current, spec_s)  # (n_fl, n_g)
            current = jnp.where(active_mask[:, s : s + 1], combined, current)

    # Reshape and transpose back to (n_g, f, l).
    return jnp.transpose(
        current.reshape(n_frequencies, n_layers, n_g),
        (2, 0, 1),
    )


@jit
def calculate_ck_emission_contribution(
    frequencies: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    emission_cos_angles: ArrayLike,
    emission_cos_angles_weights: ArrayLike,
    optical_depths: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute emission flux and normalized contribution functions for c-k opacities.

    Parameters
    ----------
    frequencies : jax.Array, shape (n_frequencies,)
        Frequency grid on which the emergent flux is evaluated.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    emission_cos_angles : jax.Array, shape (n_angles,)
        Cosines of the emission angles used for angular integration.
    emission_cos_angles_weights : jax.Array, shape (n_angles,)
        Quadrature weights associated with ``emission_cos_angles``.
    optical_depths : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Per-species optical depth profile for each g point and frequency.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Emergent flux with shape ``(n_frequencies,)`` and a layer-by-frequency
        contribution function with shape ``(n_layers, n_frequencies)`` normalized
        to sum to one at each frequency.
    """

    struct_shape = (frequencies.shape[0], temperatures.shape[0])
    flux = jnp.zeros(struct_shape[0])
    contr_em = jnp.zeros(struct_shape)

    # Vectorize over emission angles using vmap: compute per-angle contributions then sum.
    r = physics_core.compute_planck_function(temperatures, frequencies)

    def _angle_contrib(mu: ArrayLike, mu_w: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        transm_mu = jnp.exp(-optical_depths / mu)
        mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)
        transm_all = jnp.prod(mean_transm, axis=1)

        # Radiative transport contribution for this angle
        flux_mu = jnp.sum(
            (r[:, :-1] + r[:, 1:]) * (transm_all[:, 0:-1] - transm_all[:, 1:]) / 2.0,
            axis=1,
        )
        flux_mu = flux_mu + r[:, struct_shape[1]] * transm_all[:, struct_shape[1]]

        contr_local = jnp.zeros(struct_shape)
        contr_local = contr_local.at[:, :-1].set(
            (r[:, :-1] + r[:, 1:])
            * (transm_all[:, 0:-1] - transm_all[:, 1:])
            * mu
            * mu_w
        )
        contr_local = contr_local.at[:, -1].set(
            2.0 * r[:, struct_shape[1]] * transm_all[:, struct_shape[1]] * mu * mu_w
        )

        flux_contrib = flux_mu / 2.0 * mu * mu_w
        return flux_contrib, contr_local

    vmapped = vmap(_angle_contrib, in_axes=(0, 0))
    fluxs_per_angle, contrs_per_angle = vmapped(
        emission_cos_angles, emission_cos_angles_weights
    )

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
    return_contribution: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """Integrate the emission flux for a correlated-k atmosphere.

    Parameters
    ----------
    frequencies : ArrayLike, shape (n_frequencies,)
        Frequency grid on which the flux is returned.
    temperatures : ArrayLike, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : ArrayLike, shape (n_g,)
        Correlated-k quadrature weights.
    emission_cos_angles : ArrayLike, shape (n_angles,)
        Cosines of the emission angles used to integrate over solid angle.
    emission_cos_angles_weights : ArrayLike, shape (n_angles,)
        Quadrature weights associated with ``emission_cos_angles``.
    optical_depths : ArrayLike, shape (n_g, n_frequencies, n_species, n_layers)
        Per-species optical depths for the emitting atmosphere.
    return_contribution : bool, optional
        If ``True``, also compute the normalized emission contribution function.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Emergent flux with shape ``(n_frequencies,)`` and either a normalized
        contribution function of shape ``(n_layers, n_frequencies)`` or a zero
        array with the same shape when ``return_contribution`` is ``False``.
    """

    def calculate_flux(
        frequencies: ArrayLike,
        temperatures: ArrayLike,
        weights_gauss: ArrayLike,
        emission_cos_angles: ArrayLike,
        emission_cos_angles_weights: ArrayLike,
        optical_depths: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        struct_shape = (frequencies.shape[0], temperatures.shape[0])
        contribution = jnp.zeros((struct_shape[1], struct_shape[0]))
        flux = jnp.zeros(struct_shape[0])

        # Vectorize over emission angles using vmap: compute per-angle flux then sum.
        r = physics_core.compute_planck_function(temperatures, frequencies)

        def _angle_contrib(mu: ArrayLike, mu_w: ArrayLike) -> ArrayLike:
            transm_mu = jnp.exp(-optical_depths / mu)
            mean_transm = jnp.sum(
                transm_mu * weights_gauss[:, None, None, None], axis=0
            )
            transm_all = jnp.prod(mean_transm, axis=1)

            flux_mu = jnp.sum(
                (r[:, :-1] + r[:, 1:])
                * (transm_all[:, 0:-1] - transm_all[:, 1:])
                * 0.5,
                axis=1,
            )
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
        optical_depths,
    )
    return flux, contribution


@jit
def compute_species_optical_depths(
    reference_gravity: ArrayLike,
    pressures: ArrayLike,
    opacities: ArrayLike,
) -> ArrayLike:
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
    d_optical_depths = (
        avg_opacities * pressure_diffs[None, None, None, :] / reference_gravity
    )

    # Integrate optical depths cumulatively (vectorized)
    optical_depths = jnp.concatenate(
        [
            jnp.zeros_like(
                opacities[..., :1]
            ),  # Initial optical depth at the top is zero
            jnp.cumsum(d_optical_depths, axis=-1),
        ],
        axis=-1,
    )

    return optical_depths


# ---------------------------------------------------------------------------
# Ng acceleration — fully batched, no vmap
# ---------------------------------------------------------------------------


def _ng_accelerated_batched(
    tn: ArrayLike,
    tn1: ArrayLike,
    tn2: ArrayLike,
    tn3: ArrayLike,
    current_source: ArrayLike,
) -> ArrayLike:
    """Apply one fully batched Ng acceleration update.

    Parameters
    ----------
    tn, tn1, tn2, tn3 : jax.Array, shape (batch, n_layers)
        Consecutive source-function iterates ordered from newest to oldest.
    current_source : jax.Array, shape (batch, n_layers)
        Non-accelerated source update to keep when the Ng solve is ill-conditioned.

    Returns
    -------
    jax.Array, shape (batch, n_layers)
        Accelerated source function, or ``current_source`` for rows that fail the
        numerical safeguards.
    """
    Q1 = tn - 2.0 * tn1 + tn2
    Q2 = tn - tn1 - tn2 + tn3
    Q3 = tn - tn1

    # Zero boundary on first element
    Q1 = Q1.at[:, 0].set(0.0)
    Q2 = Q2.at[:, 0].set(0.0)
    Q3 = Q3.at[:, 0].set(0.0)

    # 5 explicit dot products instead of a full 3x3 Gram matmul.
    # Only G[0,0], G[1,0], G[1,1], G[0,2], G[1,2] are used; the other
    # four entries of the symmetric matrix are never read.
    A1 = jnp.minimum(jnp.einsum("bl,bl->b", Q1, Q1), HUGE_SQRT)
    A2 = jnp.minimum(jnp.einsum("bl,bl->b", Q2, Q1), HUGE_SQRT)
    B2 = jnp.minimum(jnp.einsum("bl,bl->b", Q2, Q2), HUGE_SQRT)
    C1 = jnp.minimum(jnp.einsum("bl,bl->b", Q1, Q3), HUGE_SQRT)
    C2 = jnp.minimum(jnp.einsum("bl,bl->b", Q2, Q3), HUGE_SQRT)

    AB_denom = A1 * B2 - A2 * A2  # B1==A2

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
        (1.0 - a_ng - b_ng)[:, None] * tn + a_ng[:, None] * tn1 + b_ng[:, None] * tn2,
        0.0,
    )  # (batch, l)

    # In Fortran m_approximate_ng_source, when the early `return` fires, the
    # output argument `source` (5th arg) keeps whatever value it had on entry —
    return jnp.where(skip[:, None], current_source, accelerated)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


@jit(
    static_argnames=(
        "irradiation_geometry_str",
        "adaptive_feautrier_iterations",
        "return_contribution",
        "iter_scat_max",
    )
)
def _compute_feautrier_radiative_transfer_chunk(
    frequencies_bin_edges: ArrayLike,  # (n_freq_bin_edges,)
    temperatures: ArrayLike,  # (n_layers,)
    weights_gauss: ArrayLike,  # (n_g,)
    emission_cos_angles: ArrayLike,  # (n_angles,)
    emission_cos_angles_weights: ArrayLike,  # (n_angles,)
    optical_depths_in: ArrayLike,  # (n_g, n_freq_bins, n_layers)
    photon_destruction_probabilities_in: ArrayLike,  # (n_g, n_freq_bins, n_layers)
    stellar_intensity: ArrayLike,  # (n_freq_bins,)
    star_irradiation_cos_angle: float,  # scalar
    reflectances: ArrayLike,  # (n_freq_bins,)
    emissivities: ArrayLike,  # (n_freq_bins,)
    adaptive_feautrier_iterations: bool = True,
    irradiation_geometry_str: str = "dayside_ave",
    return_contribution: bool = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
) -> tuple[ArrayLike, ArrayLike]:
    """Solve the scattering emission problem for one frequency chunk.

    Parameters
    ----------
    frequencies_bin_edges : jax.Array, shape (n_freq_bins + 1,)
        Bin-edge frequency grid used for the Planck integral.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    emission_cos_angles : jax.Array, shape (n_angles,)
        Cosines of the viewing angles used by the Feautrier solver.
    emission_cos_angles_weights : jax.Array, shape (n_angles,)
        Quadrature weights associated with ``emission_cos_angles``.
    optical_depths_in : jax.Array, shape (n_g, n_freq_bins, n_layers)
        Total optical depths for each g point, frequency bin, and layer.
    photon_destruction_probabilities_in : jax.Array, shape (n_g, n_freq_bins, n_layers)
        Local thermalization probabilities, clipped internally away from zero.
    stellar_intensity : jax.Array, shape (n_freq_bins,)
        Incident stellar intensity at the top of the atmosphere.
    star_irradiation_cos_angle : float
        Cosine of the stellar irradiation angle for non-isotropic illumination.
    reflectances : jax.Array, shape (n_freq_bins,)
        Surface or lower-boundary reflectance.
    emissivities : jax.Array, shape (n_freq_bins,)
        Surface or lower-boundary emissivity.
    adaptive_feautrier_iterations : bool, optional
        Adapt the minimum number of scattering iterations to the local photon
        destruction probability.
    irradiation_geometry_str : {"dayside_ave", "planetary_ave", "non-isotropic"}, optional
        Illumination geometry used for the stellar source term.
    return_contribution : bool, optional
        If ``True``, compute the normalized emission contribution function.
    iter_scat_max : int, optional
        Hard iteration cap for the scattering loop.
    convergence_threshold : float, optional
        Relative flux convergence criterion used per ``(g, frequency)`` pair.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Emergent flux with shape ``(n_freq_bins,)`` and a normalized emission
        contribution array with shape ``(n_layers, n_freq_bins)``.
    """

    # ------------------------------------------------------------------
    # 0. Shapes and scalar constants
    # ------------------------------------------------------------------
    n_g, n_freq_bins, n_layers = photon_destruction_probabilities_in.shape
    n_angles = emission_cos_angles.shape[0]
    n_batch = n_g * n_freq_bins * n_angles
    pi_4 = 4.0 * jnp.pi
    inv_dtau = 1e10
    inv_dtau2 = inv_dtau * 0.5

    mu_double = emission_cos_angles * 2.0
    surf_refl_2 = 2.0 * reflectances
    angle_weight_a = emission_cos_angles_weights
    mu_weight_a = emission_cos_angles * emission_cos_angles_weights

    # ------------------------------------------------------------------
    # 1. Geometry coefficient — Python level, never traced
    # ------------------------------------------------------------------
    if irradiation_geometry_str == "dayside_ave":
        geom = 0.5
    elif irradiation_geometry_str == "planetary_ave":
        geom = 0.25
    elif irradiation_geometry_str == "non-isotropic":
        geom = -1.0
    else:
        raise ValueError(f"Unknown irradiation_geometry: {irradiation_geometry_str}")

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
            -optical_depths_in[:, :, None, :] / emission_cos_angles[None, None, :, None]
        )

        J_star_ini_gfl = (
            0.5
            * geom
            * jnp.abs(stellar_intensity)[None, :, None, None]
            * emission_cos_angles_weights[None, None, :, None]
            * exp_over_mu_gfal
        ).sum(
            axis=2
        )  # (g, f, l)

        # Surface reflection — slice exp at surface before releasing
        exp_surf_gfa = exp_over_mu_gfal[:, :, :, n_layers - 1]  # (g, f, a)
        del exp_over_mu_gfal

        I_surface_reflection_gf = surf_refl_2[None, :] * (
            geom
            * jnp.abs(stellar_intensity)[None, :, None]
            * exp_surf_gfa
            * mu_weight_a[None, None, :]
        ).sum(
            axis=2
        )  # (g, f)

    else:  # non-isotropic
        exp_gfl = jnp.exp(-optical_depths_in / star_irradiation_cos_angle)
        J_star_ini_gfl = jnp.abs(stellar_intensity[None, :, None] / 4.0 * exp_gfl)
        I_surface_reflection_gf = (
            reflectances[None, :]
            * J_star_ini_gfl[:, :, n_layers - 1]
            * 4.0
            * star_irradiation_cos_angle
        )

    # ------------------------------------------------------------------
    # 4. Planck function
    # ------------------------------------------------------------------
    planck_integral_lf = physics_core.compute_planck_function_integral(
        temperatures, frequencies_bin_edges
    )  # (l, f)
    planck_integral_fl = planck_integral_lf.T  # (f, l) — zero-copy view

    # ------------------------------------------------------------------
    # 5. Surface boundary terms
    # ------------------------------------------------------------------
    I_surface_emission_f = emissivities * planck_integral_lf[n_layers - 1, :]
    I_surface_no_scatter_gf = I_surface_emission_f[None, :] + I_surface_reflection_gf

    # ------------------------------------------------------------------
    # 6. Per-(g,f) minimum iteration count  (Fix 2)
    # ------------------------------------------------------------------
    if adaptive_feautrier_iterations:
        n_iter_min_gf = jnp.clip(
            1 + jnp.max(1.0 / pdp_gfl, axis=2), 4, 9  # max over layers → (g, f)
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
    dtau1 = optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1]  # (g,f,l-1)
    dtau1 = jnp.where(jnp.abs(dtau1) < TINIEST, TINIEST, dtau1)

    dtau2 = optical_depths_in[:, :, 2:] - optical_depths_in[:, :, :-2]  # (g,f,l-2)
    dtau2 = jnp.where(jnp.abs(dtau2) < TINIEST, TINIEST, dtau2)

    # ------------------------------------------------------------------
    # 9. Feautrier tridiagonal coefficients
    # ------------------------------------------------------------------
    # top row (l=0)
    f1_top = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, 0, None],
        -inv_dtau,
        inv_dtau,
    )  # (g, f, a, 1)
    b_top = 1.0 + 2.0 * f1_top * (1.0 + f1_top)
    c_top = -2.0 * f1_top**2
    a_top = jnp.zeros_like(b_top)

    # mid rows (l=1 … n_layers-2)
    f1_mid = jnp.clip(
        mu_double[None, None, :, None] / dtau2[:, :, None, :],
        -inv_dtau2,
        inv_dtau2,
    )  # (g, f, a, l-2)
    f2_mid = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, 1:],
        -inv_dtau,
        inv_dtau,
    )  # (g, f, a, l-2)
    f3_mid = jnp.clip(
        emission_cos_angles[None, None, :, None] / dtau1[:, :, None, :-1],
        -inv_dtau,
        inv_dtau,
    )  # (g, f, a, l-2)
    b_mid = 1.0 + f1_mid * (f2_mid + f3_mid)
    c_mid = -f1_mid * f2_mid
    a_mid = -f1_mid * f3_mid

    a_flat = jnp.pad(
        jnp.concatenate([a_top, a_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
    ).reshape(
        n_batch, n_layers
    )  # padded with 0
    b_flat = jnp.pad(
        jnp.concatenate([b_top, b_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
        constant_values=1.0,
    ).reshape(
        n_batch, n_layers
    )  # padded with 1
    c_flat = jnp.pad(
        jnp.concatenate([c_top, c_mid], axis=-1),
        ((0, 0), (0, 0), (0, 0), (0, 1)),
    ).reshape(
        n_batch, n_layers
    )  # padded with 0

    # ------------------------------------------------------------------
    # 10. Lambda_loc for ALI  (g, f, l)
    # ------------------------------------------------------------------
    lam_top = jnp.sum(angle_weight_a[None, None, :, None] / b_top, axis=2)[
        ..., 0
    ]  # (g, f)

    lam_mid = jnp.sum(
        angle_weight_a[None, None, :, None] / b_mid, axis=2
    )  # (g, f, l-2)

    f1_surf = jnp.clip(
        emission_cos_angles[None, None, :, None]
        / dtau1[:, :, None, n_layers - 2, None],
        -inv_dtau,
        inv_dtau,
    )  # (g, f, a, 1)
    lam_surf = jnp.sum(
        angle_weight_a[None, None, :, None] / (1.0 + 2.0 * f1_surf**2), axis=2
    )[
        ..., 0
    ]  # (g, f)

    lambda_loc_gfl = jnp.concatenate(
        [lam_top[:, :, None], lam_mid, lam_surf[:, :, None]], axis=-1
    )  # (g, f, l)

    # ------------------------------------------------------------------
    # 11. loop-invariant ALI intermediates precomputed once.
    # ------------------------------------------------------------------
    one_minus_pdp_gfl = 1.0 - pdp_gfl  # (g,f,l)
    pdp_times_planck = pdp_gfl * planck_integral_fl[None, :, :]  # (g,f,l)
    ali_inv_denom_gfl = 1.0 / jnp.maximum(
        1.0 - one_minus_pdp_gfl * lambda_loc_gfl, TINIEST
    )  # (g,f,l)

    # ------------------------------------------------------------------
    # 12. Inner Feautrier solve
    # ------------------------------------------------------------------
    def _feautrier_solve(
        source_gfl: ArrayLike,
        I_plus_surface_gfa: ArrayLike,
        I_minus_boundary_gf: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns J_bol_gfl (g,f,l), I_H_top (g,f,a), new_I_plus_surface (g,f,a).
        """
        r_surf_gfa = 0.5 * (
            I_plus_surface_gfa + I_minus_boundary_gf[:, :, None]
        )  # (g, f, a)

        I_H_top, x_last_gfa, x_nm2_gfa, J_bol_gfl = linalg.solve_feautrier_tridiagonal(
            a_flat,
            b_flat,
            c_flat,
            source_gfl[:, :, : n_layers - 1],
            r_surf_gfa,
            angle_weight_a,
            n_g,
            n_freq_bins,
            n_angles,
            n_layers,
        )

        d1_surf = f2_mid[:, :, :, -1] * (x_last_gfa - x_nm2_gfa)
        new_I_plus_surface = x_last_gfa - d1_surf  # (g, f, a)

        return J_bol_gfl, I_H_top, new_I_plus_surface

    # ------------------------------------------------------------------
    # 13. Scattering iteration loop with a fixed iteration count.
    #
    # Reverse-mode autodiff cannot differentiate the original data-dependent
    # while-loop. Run a static number of iterations and freeze per-(g, f)
    # state once a pair converges.
    # ------------------------------------------------------------------

    def _surface_reflection(I_plus_gfa: ArrayLike) -> ArrayLike:
        return surf_refl_2[None, :] * jnp.sum(
            I_plus_gfa * mu_weight_a[None, None, :], axis=2
        )  # (g, f)

    def _scan_step(
        carry: tuple[ArrayLike, ...],
        iter_idx: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ...], None]:
        (
            source_gfl,
            I_plus_surface_gfa,
            flux_gf_prev,
            I_minus_gf,
            ng_history,
            converged_gf,
        ) = carry

        active_gf = jnp.logical_not(converged_gf)
        active_gfl = active_gf[:, :, None]
        active_gfa = active_gf[:, :, None]

        def _advance(state: tuple[ArrayLike, ...]) -> tuple[ArrayLike, ...]:
            (
                current_source_gfl,
                current_I_plus_surface_gfa,
                current_flux_gf_prev,
                current_I_minus_gf,
                current_ng_history,
                current_converged_gf,
            ) = state

            I_minus_gf_new = I_surface_no_scatter_gf + _surface_reflection(
                current_I_plus_surface_gfa
            )
            J_bol_gfl, I_H_top, new_I_plus_surface = _feautrier_solve(
                current_source_gfl,
                current_I_plus_surface_gfa,
                I_minus_gf_new,
            )

            source_ali = (
                pdp_times_planck
                + one_minus_pdp_gfl
                * (J_star_ini_gfl + J_bol_gfl - lambda_loc_gfl * current_source_gfl)
            ) * ali_inv_denom_gfl

            slot_index = iter_idx % 4
            updated_slot = jnp.where(
                active_gfl, source_ali, current_ng_history[slot_index]
            )
            current_ng_history = current_ng_history.at[slot_index].set(updated_slot)

            tn = current_ng_history[slot_index]
            tn1 = current_ng_history[(slot_index - 1) % 4]
            tn2 = current_ng_history[(slot_index - 2) % 4]
            tn3 = current_ng_history[(slot_index - 3) % 4]

            B = n_g * n_freq_bins
            ng_result = _ng_accelerated_batched(
                tn.reshape(B, n_layers),
                tn1.reshape(B, n_layers),
                tn2.reshape(B, n_layers),
                tn3.reshape(B, n_layers),
                source_ali.reshape(B, n_layers),
            ).reshape(n_g, n_freq_bins, n_layers)

            do_ng = (iter_idx > 0) & ((iter_idx + 1) % 4 == 0)
            next_source_gfl = jnp.where(do_ng, ng_result, source_ali)
            next_source_gfl = jnp.where(active_gfl, next_source_gfl, current_source_gfl)

            flux_gf_cur = -jnp.einsum("gfa,a->gf", I_H_top, mu_weight_a)
            conv_val_gf = jnp.abs(
                1.0
                - current_flux_gf_prev
                / jnp.where(
                    jnp.abs(flux_gf_cur) < 1e-100,
                    1e-100,
                    flux_gf_cur,
                )
            )
            newly_converged = (iter_idx > n_iter_min_gf) & (
                conv_val_gf < convergence_threshold
            )
            next_converged_gf = current_converged_gf | newly_converged

            next_I_plus_surface_gfa = jnp.where(
                active_gfa,
                new_I_plus_surface,
                current_I_plus_surface_gfa,
            )
            next_flux_gf_prev = jnp.where(active_gf, flux_gf_cur, current_flux_gf_prev)
            next_I_minus_gf = jnp.where(active_gf, I_minus_gf_new, current_I_minus_gf)

            return (
                next_source_gfl,
                next_I_plus_surface_gfa,
                next_flux_gf_prev,
                next_I_minus_gf,
                current_ng_history,
                next_converged_gf,
            )

        next_carry = lax.cond(
            jnp.any(active_gf),
            _advance,
            lambda state: state,
            carry,
        )
        return next_carry, None

    initial_carry = (
        planck_broadcast_gfl,
        jnp.zeros((n_g, n_freq_bins, n_angles)),
        jnp.zeros((n_g, n_freq_bins)),
        jnp.zeros((n_g, n_freq_bins)),
        jnp.zeros((4, n_g, n_freq_bins, n_layers)),
        jnp.zeros((n_g, n_freq_bins), dtype=jnp.bool_),
    )

    final_carry, _ = lax.scan(
        _scan_step,
        initial_carry,
        jnp.arange(iter_scat_max + 1, dtype=jnp.int32),
    )

    (final_source_gfl, _, flux_gf_final, I_minus_gf_final, _, _) = final_carry

    # ------------------------------------------------------------------
    # 14. Flux  (n_freq_bins,)
    # ------------------------------------------------------------------
    flux_f = pi_4 * jnp.einsum("gf,g->f", flux_gf_final, weights_gauss)

    # ------------------------------------------------------------------
    # 15. Contribution function (optional)
    # ------------------------------------------------------------------
    emission_contribution_lf = jnp.zeros((n_layers, n_freq_bins))

    def compute_contribution() -> ArrayLike:
        # G-integrated source: (l, f)
        r_src_lf = jnp.einsum("g,gfl->lf", weights_gauss, final_source_gfl)

        # G-integrated I_minus: (f,)
        I_minus_f = jnp.einsum("g,gf->f", weights_gauss, I_minus_gf_final)

        # This replaces n_angles separate exp(-tau/mu) calls inside fori_loop.
        transm_gfla = jnp.exp(
            -optical_depths_in[:, :, :, None] / emission_cos_angles[None, None, None, :]
        )  # (g, f, l, a)
        transm_lfa = jnp.einsum("g,gfla->lfa", weights_gauss, transm_gfla)  # (l, f, a)
        del transm_gfla

        def i_mu_body(i_mu: int, acc: ArrayLike) -> ArrayLike:
            mu_val = emission_cos_angles[i_mu]
            mu_weight_val = emission_cos_angles_weights[i_mu]

            # no exp() here — just slice the precomputed array.
            transm_lf = transm_lfa[:, :, i_mu]  # (l, f)

            contrib_mid = (
                (r_src_lf[:-1, :] + r_src_lf[1:, :])
                * (transm_lf[:-1, :] - transm_lf[1:, :])
                * mu_val
                * mu_weight_val
            )  # (l-1, f)

            contrib_last = (
                2.0 * I_minus_f * transm_lf[n_layers - 1, :] * mu_val * mu_weight_val
            )  # (f,)

            acc = acc.at[:-1, :].add(contrib_mid)
            acc = acc.at[n_layers - 1, :].add(contrib_last)
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


@jit(
    static_argnames=(
        "irradiation_geometry_str",
        "adaptive_feautrier_iterations",
        "return_contribution",
        "iter_scat_max",
        "chunk_feautrier",
        "feautrier_chunk_size",
    )
)
def compute_feautrier_radiative_transfer(
    frequencies_bin_edges: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    emission_cos_angles: ArrayLike,
    emission_cos_angles_weights: ArrayLike,
    optical_depths_in: ArrayLike,
    photon_destruction_probabilities_in: ArrayLike,
    stellar_intensity: ArrayLike,
    star_irradiation_cos_angle: float,
    reflectances: ArrayLike,
    emissivities: ArrayLike,
    adaptive_feautrier_iterations: bool = True,
    irradiation_geometry_str: str = "dayside_ave",
    return_contribution: bool = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
    chunk_feautrier: bool = True,
    feautrier_chunk_size: int = 1024,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute emission flux with the Feautrier scattering solver.

    Parameters
    ----------
    frequencies_bin_edges : jax.Array, shape (n_freq_bins + 1,)
        Bin-edge frequency grid used for the Planck integral.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    emission_cos_angles : jax.Array, shape (n_angles,)
        Cosines of the viewing angles used by the Feautrier solver.
    emission_cos_angles_weights : jax.Array, shape (n_angles,)
        Quadrature weights associated with ``emission_cos_angles``.
    optical_depths_in : jax.Array, shape (n_g, n_freq_bins, n_layers)
        Total optical depths for each g point, frequency bin, and layer.
    photon_destruction_probabilities_in : jax.Array, shape (n_g, n_freq_bins, n_layers)
        Local thermalization probabilities.
    stellar_intensity : jax.Array, shape (n_freq_bins,)
        Incident stellar intensity at the top of the atmosphere.
    star_irradiation_cos_angle : float
        Cosine of the stellar irradiation angle for non-isotropic illumination.
    reflectances : jax.Array, shape (n_freq_bins,)
        Surface or lower-boundary reflectance.
    emissivities : jax.Array, shape (n_freq_bins,)
        Surface or lower-boundary emissivity.
    adaptive_feautrier_iterations : bool, optional
        Adapt the minimum number of scattering iterations to the local photon
        destruction probability.
    irradiation_geometry_str : {"dayside_ave", "planetary_ave", "non-isotropic"}, optional
        Illumination geometry used for the stellar source term.
    return_contribution : bool, optional
        If ``True``, compute the normalized emission contribution function.
    iter_scat_max : int, optional
        Hard iteration cap for the scattering loop.
    convergence_threshold : float, optional
        Relative flux convergence criterion used per ``(g, frequency)`` pair.
    chunk_feautrier : bool, optional
        If ``True``, split the frequency axis into chunks before calling the core
        solver to reduce peak memory use.
    feautrier_chunk_size : int, optional
        Number of frequency bins per chunk when chunking is enabled.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Emergent flux with shape ``(n_freq_bins,)`` and a normalized emission
        contribution array with shape ``(n_layers, n_freq_bins)``.
    """
    n_g, n_freq_bins, n_layers = photon_destruction_probabilities_in.shape

    if (not chunk_feautrier) or feautrier_chunk_size >= n_freq_bins:
        return _compute_feautrier_radiative_transfer_chunk(
            frequencies_bin_edges=frequencies_bin_edges,
            temperatures=temperatures,
            weights_gauss=weights_gauss,
            emission_cos_angles=emission_cos_angles,
            emission_cos_angles_weights=emission_cos_angles_weights,
            optical_depths_in=optical_depths_in,
            photon_destruction_probabilities_in=photon_destruction_probabilities_in,
            stellar_intensity=stellar_intensity,
            star_irradiation_cos_angle=star_irradiation_cos_angle,
            reflectances=reflectances,
            emissivities=emissivities,
            adaptive_feautrier_iterations=adaptive_feautrier_iterations,
            irradiation_geometry_str=irradiation_geometry_str,
            return_contribution=return_contribution,
            iter_scat_max=iter_scat_max,
            convergence_threshold=convergence_threshold,
        )

    if feautrier_chunk_size <= 0:
        raise ValueError(
            f"feautrier_chunk_size must be > 0, but was {feautrier_chunk_size}"
        )

    n_chunks = (n_freq_bins + feautrier_chunk_size - 1) // feautrier_chunk_size
    padded_n_freq_bins = n_chunks * feautrier_chunk_size
    pad_freq_bins = padded_n_freq_bins - n_freq_bins

    optical_depths_padded = jnp.pad(
        optical_depths_in, ((0, 0), (0, pad_freq_bins), (0, 0))
    )
    photon_destruction_probabilities_padded = jnp.pad(
        photon_destruction_probabilities_in,
        ((0, 0), (0, pad_freq_bins), (0, 0)),
    )
    stellar_intensity_padded = jnp.pad(stellar_intensity, (0, pad_freq_bins))
    reflectances_padded = jnp.pad(reflectances, (0, pad_freq_bins))
    emissivities_padded = jnp.pad(emissivities, (0, pad_freq_bins))

    if pad_freq_bins > 0:
        last_bin_width = frequencies_bin_edges[-1] - frequencies_bin_edges[-2]
        extra_frequency_edges = frequencies_bin_edges[-1] + last_bin_width * jnp.arange(
            1, pad_freq_bins + 1
        )
        frequencies_bin_edges_padded = jnp.concatenate(
            (frequencies_bin_edges, extra_frequency_edges)
        )
    else:
        frequencies_bin_edges_padded = frequencies_bin_edges

    optical_depths_chunks = jnp.moveaxis(
        optical_depths_padded.reshape(n_g, n_chunks, feautrier_chunk_size, n_layers),
        1,
        0,
    )
    photon_destruction_probabilities_chunks = jnp.moveaxis(
        photon_destruction_probabilities_padded.reshape(
            n_g, n_chunks, feautrier_chunk_size, n_layers
        ),
        1,
        0,
    )
    stellar_intensity_chunks = stellar_intensity_padded.reshape(
        n_chunks, feautrier_chunk_size
    )
    reflectances_chunks = reflectances_padded.reshape(n_chunks, feautrier_chunk_size)
    emissivities_chunks = emissivities_padded.reshape(n_chunks, feautrier_chunk_size)
    # Vectorised sliding-window construction — avoids Python for-loop
    # that would be unrolled at trace time.
    _chunk_starts = jnp.arange(n_chunks) * feautrier_chunk_size
    frequencies_bin_edges_chunks = vmap(
        lambda s: lax.dynamic_slice(
            frequencies_bin_edges_padded, (s,), (feautrier_chunk_size + 1,)
        )
    )(_chunk_starts)

    def _chunk_body(
        _: None, chunk_inputs: tuple[ArrayLike, ...]
    ) -> tuple[None, tuple[ArrayLike, ArrayLike]]:
        (
            chunk_frequency_bin_edges,
            chunk_optical_depths,
            chunk_photon_destruction_probabilities,
            chunk_stellar_intensity,
            chunk_reflectances,
            chunk_emissivities,
        ) = chunk_inputs

        chunk_flux, chunk_emission_contribution = (
            _compute_feautrier_radiative_transfer_chunk(
                frequencies_bin_edges=chunk_frequency_bin_edges,
                temperatures=temperatures,
                weights_gauss=weights_gauss,
                emission_cos_angles=emission_cos_angles,
                emission_cos_angles_weights=emission_cos_angles_weights,
                optical_depths_in=chunk_optical_depths,
                photon_destruction_probabilities_in=chunk_photon_destruction_probabilities,
                stellar_intensity=chunk_stellar_intensity,
                star_irradiation_cos_angle=star_irradiation_cos_angle,
                reflectances=chunk_reflectances,
                emissivities=chunk_emissivities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                irradiation_geometry_str=irradiation_geometry_str,
                return_contribution=return_contribution,
                iter_scat_max=iter_scat_max,
                convergence_threshold=convergence_threshold,
            )
        )

        return None, (chunk_flux, chunk_emission_contribution)

    _, (flux_chunks, emission_contribution_chunks) = lax.scan(
        _chunk_body,
        None,
        (
            frequencies_bin_edges_chunks,
            optical_depths_chunks,
            photon_destruction_probabilities_chunks,
            stellar_intensity_chunks,
            reflectances_chunks,
            emissivities_chunks,
        ),
    )

    flux_f = flux_chunks.reshape(padded_n_freq_bins)[:n_freq_bins]
    emission_contribution_lf = jnp.transpose(
        emission_contribution_chunks, (1, 0, 2)
    ).reshape(n_layers, padded_n_freq_bins,)[:, :n_freq_bins]

    return flux_f, emission_contribution_lf


@jit
def compute_optical_depths(
    reference_gravity: ArrayLike,
    pressures: ArrayLike,
    opacities: ArrayLike,
    scattering_in_emission: bool = False,
    continuum_opacities_scattering: Optional[ArrayLike] = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Integrate total optical depth and photon destruction probabilities.

    Parameters
    ----------
    reference_gravity : float
        Surface gravity in cgs units.
    pressures : jax.Array, shape (n_layers,)
        Pressure grid in cgs units, ordered from top to bottom of the atmosphere.
    opacities : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Absorptive opacity contribution for each g point, frequency, species, and layer.
    scattering_in_emission : bool, optional
        If ``True``, include ``continuum_opacities_scattering`` in the total opacity
        used by the emission solver.
    continuum_opacities_scattering : jax.Array, shape (n_g or 1, n_frequencies, n_layers) or (n_frequencies, n_layers), optional
        Scattering opacity added to the first species channel when emission scattering
        is enabled.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Optical depths with shape ``(n_g, n_frequencies, n_species, n_layers)`` and
        photon destruction probabilities with shape ``(n_g, n_frequencies, n_layers)``.
    """

    def scattering_function() -> tuple[ArrayLike, ArrayLike]:
        # Add continuum scattering to the first species (assumed at index 0)
        _opacities = opacities.at[:, :, 0, :].add(continuum_opacities_scattering)
        # Compute photon destruction probabilities for all g, freq, press
        absorptive_opac = opacities[:, :, 0, :]
        total_opac = _opacities[:, :, 0, :]
        safe_total_opac = jnp.where(total_opac > TINIEST, total_opac, 1.0)
        photon_destruction_probabilities = absorptive_opac / safe_total_opac
        photon_destruction_probabilities = jnp.where(
            total_opac > TINIEST,
            photon_destruction_probabilities,
            1.0,
        )
        return photon_destruction_probabilities, _opacities

    def false_function() -> tuple[ArrayLike, ArrayLike]:
        return jnp.ones(opacities[:, :, 0, :].shape), opacities

    photon_destruction_probabilities, _opacities = lax.cond(
        scattering_in_emission, scattering_function, false_function
    )

    photon_destruction_probabilities = jnp.clip(
        photon_destruction_probabilities, 0.0, 1.0
    )

    # Vectorized optical depth calculation
    pressure_diffs = pressures[1:] - pressures[:-1]  # (n_layers-1,)
    avg_opacities = (
        _opacities[..., 1:] + _opacities[..., :-1]
    ) / 2.0  # (..., n_layers-1)
    d_optical_depths = (
        avg_opacities * pressure_diffs / reference_gravity
    )  # (..., n_layers-1)
    # Pad with zeros at the start for cumulative sum
    d_optical_depths_padded = jnp.pad(
        d_optical_depths, [(0, 0)] * (d_optical_depths.ndim - 1) + [(1, 0)]
    )
    optical_depths = jnp.cumsum(d_optical_depths_padded, axis=-1)

    return optical_depths, photon_destruction_probabilities


@jit
def compute_photon_destruction_prob(
    total_opacity_in: ArrayLike,
    continuum_opacity_scat_emis: ArrayLike,
) -> ArrayLike:
    """Compute thermalization probabilities from total and scattering opacities.

    Parameters
    ----------
    total_opacity_in : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Total absorptive opacity cube. Only the first species slot is inspected by
        this legacy helper, matching the historical Fortran interface.
    continuum_opacity_scat_emis : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in emission calculations.

    Returns
    -------
    jax.Array, shape (n_g, n_frequencies, n_layers)
        Photon destruction probability, equal to the absorptive fraction of the
        total extinction.
    """
    total_opacity = total_opacity_in[:, :, 0, :]

    # Vectorized per-g computation: add continuum opacity and compute destruction probability
    def _per_g(opacity_g: ArrayLike) -> ArrayLike:
        tot = opacity_g + continuum_opacity_scat_emis
        safe_tot = jnp.where(tot > TINIEST, tot, 1.0)
        destruction_prob = opacity_g / safe_tot
        return jnp.where(tot > TINIEST, destruction_prob, 1.0)

    photon_destruction_prob = vmap(_per_g)(total_opacity)
    return photon_destruction_prob


@jit
def compute_planck_opacities_core(
    opacities: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    frequencies_bin_edges: ArrayLike,
) -> ArrayLike:
    """Compute Planck-mean opacities from a total opacity cube.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, n_layers)
        Total opacity for each g point, frequency bin, and layer.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    frequencies_bin_edges : jax.Array, shape (n_frequencies + 1,)
        Frequency bin edges used for the Planck integral.

    Returns
    -------
    jax.Array, shape (n_layers,)
        Planck-mean opacity for each atmospheric layer.
    """
    diff_freqs = jnp.diff(frequencies_bin_edges)
    B_nu = physics_core.star_planck(temperatures, frequencies_bin_edges)

    kappa_planck = jnp.sum(
        B_nu
        * jnp.sum(weights_gauss[:, None, None] / opacities[:, :, :], axis=0)
        * diff_freqs[:, None],
        axis=0,
    )
    norm = jnp.sum(B_nu * diff_freqs[:, None], axis=0)
    return kappa_planck / norm


@jit
def compute_planck_opacities(
    opacities: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    frequencies_bin_edges: ArrayLike,
    scattering_in_emission: bool,
    continuum_opacities_scattering: ArrayLike,
) -> ArrayLike:
    """Compute Planck-mean opacities with optional continuum scattering.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, n_layers)
        Absorptive line opacity cube.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    frequencies_bin_edges : jax.Array, shape (n_frequencies + 1,)
        Frequency bin edges used for the Planck integral.
    scattering_in_emission : bool
        If ``True``, add continuum scattering before taking the Planck mean.
    continuum_opacities_scattering : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity.

    Returns
    -------
    jax.Array, shape (n_layers,)
        Planck-mean opacity for each atmospheric layer.
    """
    total_opacity_use = lax.cond(
        scattering_in_emission,
        lambda: opacities + continuum_opacities_scattering[None, :, :],
        lambda: opacities,
    )

    kappa_planck = compute_planck_opacities_core(
        total_opacity_use[:, :, :], temperatures, weights_gauss, frequencies_bin_edges
    )
    return kappa_planck


@jit
def integ_parab(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    fx: ArrayLike,
    fy: ArrayLike,
    fz: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
) -> ArrayLike:
    """Integrate the quadratic passing through three support points.

    Parameters
    ----------
    x, y, z : float
        Abscissae of the support points.
    fx, fy, fz : float
        Function values at ``x``, ``y``, and ``z``.
    a, b : float
        Integration bounds.

    Returns
    -------
    float
        Integral of the interpolating parabola between ``a`` and ``b``.
    """
    c3 = ((fz - fy) / (z - y) - (fz - fx) / (z - x)) / (y - x)
    c2 = (fz - fx) / (z - x) - c3 * (z + x)
    c1 = fx - c2 * x - c3 * x**2
    return c1 * (b - a) + c2 * (b**2 - a**2) / 2 + c3 * (b**3 - a**3) / 3


@jit
def compute_hydrostatic_interval_integrals(
    pressures: ArrayLike,
    inv_rho: ArrayLike,
) -> ArrayLike:
    """Compute per-layer hydrostatic integrals over the pressure grid."""
    core_integrals = integ_parab(
        pressures[:-2],
        pressures[1:-1],
        pressures[2:],
        inv_rho[:-2],
        inv_rho[1:-1],
        inv_rho[2:],
        pressures[:-2],
        pressures[1:-1],
    )
    tail_integral = (inv_rho[-2] + inv_rho[-1]) * (pressures[-1] - pressures[-2]) * 0.5
    return jnp.concatenate((core_integrals, tail_integral[None]))


@jit(static_argnames=("variable_gravity"))
def compute_radius_hydrostatic_equilibrium(
    pressures: ArrayLike,
    reference_gravity: ArrayLike,
    atmosphere_densities: ArrayLike,
    reference_pressure: ArrayLike,
    planet_radius: ArrayLike,
    variable_gravity: bool = True,
) -> ArrayLike:
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

    if variable_gravity:
        coefficient = 1.0 / (reference_gravity * (planet_radius**2) + TINIEST)
    else:
        coefficient = -1.0 / (reference_gravity + TINIEST)

    interval_integrals = compute_hydrostatic_interval_integrals(
        pressures=pressures,
        inv_rho=inv_rho,
    )
    radius_hydrostatic_equilibrium = (
        -coefficient
        * jnp.cumsum(
            interval_integrals[::-1],
            dtype=dtype,
        )[::-1]
    )
    radius_hydrostatic_equilibrium = jnp.concatenate(
        (radius_hydrostatic_equilibrium, jnp.zeros(1, dtype=dtype))
    )

    reference_layer_mask = (pressures[:-1] <= reference_pressure) & (
        pressures[1:] > reference_pressure
    )
    reference_index = jnp.argmax(reference_layer_mask)

    def compute_reference_radius_offset() -> ArrayLike:

        def layer_true() -> ArrayLike:
            return (
                radius_hydrostatic_equilibrium[reference_index + 1]
                - integ_parab(
                    pressures[reference_index],
                    pressures[reference_index + 1],
                    pressures[reference_index + 2],
                    inv_rho[reference_index],
                    inv_rho[reference_index + 1],
                    inv_rho[reference_index + 2],
                    reference_pressure,
                    pressures[reference_index + 1],
                )
                * coefficient
            )

        def layer_false() -> ArrayLike:
            return radius_hydrostatic_equilibrium[reference_index + 1] - (
                (inv_rho[reference_index] + inv_rho[reference_index + 1])
                * (pressures[reference_index + 1] - reference_pressure)
                * coefficient
                * 0.5
            )

        return lax.cond(reference_index <= n_layers - 3, layer_true, layer_false)

    R0 = lax.cond(
        jnp.any(reference_layer_mask),
        compute_reference_radius_offset,
        lambda: jnp.zeros((), dtype=dtype),
    )

    if variable_gravity:
        R0 = (1.0 / planet_radius) - R0
        radius_hydrostatic_equilibrium = radius_hydrostatic_equilibrium + R0
        return 1.0 / (radius_hydrostatic_equilibrium + TINIEST)

    R0 = planet_radius - R0
    return radius_hydrostatic_equilibrium + R0


@jit(static_argnames=("variable_gravity"))
def compute_transit_radii(
    opacities: ArrayLike,
    continuum_opacities_scattering: ArrayLike,
    pressures: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    mean_molar_masses: ArrayLike,
    reference_gravity: ArrayLike,
    reference_pressure: ArrayLike,
    planet_radius: ArrayLike,
    variable_gravity: bool = True,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute transmission radii from per-species opacities.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Absorptive opacities. Continuum absorption is assumed to have already been
        added to the first species slot by the caller.
    continuum_opacities_scattering : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in transmission.
    pressures : jax.Array, shape (n_layers,)
        Pressure grid in bar.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    mean_molar_masses : jax.Array, shape (n_layers,)
        Mean molecular mass in amu.
    reference_gravity : float
        Gravity at ``reference_pressure`` in cgs units.
    reference_pressure : float
        Reference pressure in bar where ``planet_radius`` is defined.
    planet_radius : float
        Planet radius at ``reference_pressure`` in cm.
    variable_gravity : bool, optional
        If ``True``, vary gravity as $1/r^2$ when reconstructing the hydrostatic radius profile.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Effective transit radii with shape ``(n_frequencies,)`` and the hydrostatic
        radius profile with shape ``(n_layers,)``.
    """

    def _safe_sqrt_positive(values: ArrayLike) -> ArrayLike:
        positive_values = values > 0.0
        safe_values = jnp.where(
            positive_values,
            values,
            lax.stop_gradient(jnp.ones_like(values)),
        )
        return jnp.where(positive_values, jnp.sqrt(safe_values), 0.0)

    # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
    # (temperature, pressure, mmw), gravity, reference pressure and radius.
    pressures_cgs = pressures * 1e6  # bar to cgs
    reference_pressure_cgs = reference_pressure * 1e6  # bar to cgs

    atmospheric_densities = (
        pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures
    )
    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures=pressures_cgs,
        reference_gravity=reference_gravity,
        atmosphere_densities=atmospheric_densities,
        reference_pressure=reference_pressure_cgs,
        planet_radius=planet_radius,
        variable_gravity=variable_gravity,
    )

    is_negative_condition = radius_hydrostatic_equilibrium < 0.0
    first_non_negative_idx = jnp.argmax(
        radius_hydrostatic_equilibrium >= 0.0
    )  # Index of first True
    replacement_value = radius_hydrostatic_equilibrium[first_non_negative_idx]
    radius_hydrostatic_equilibrium = jnp.where(
        is_negative_condition, replacement_value, radius_hydrostatic_equilibrium
    )

    # Calculate the density
    rho = pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures

    # Bring continuum scattering opacities in right shape for matrix operations later.
    # Reminder: when calling this function, continuum absorption opacities have already
    # been added to line_struc_kappas.
    continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape(
        (1, continuum_opacities_scattering.shape[0], 1, pressures.size)
    )

    transparencies = opacities * rho
    transparencies = transparencies.at[:, :, :1, :].add(
        continuum_opa_scat_reshaped * rho
    )

    # Calculate average mean free path between neighboring layers for later integration
    # Factor 1/2 is omitted because it cancels with effective planet area integration below.
    transparencies = transparencies.at[:, :, :, 1:].add(transparencies[:, :, :, :-1])

    # Prepare matrix for delta path lengths during optical depth integration
    diff_s = jnp.zeros((pressures.size, pressures.size))

    # Calculate matrix of delta path lengths
    r_ik = (
        radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.0
        - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.0
    )
    r_ik = _safe_sqrt_positive(r_ik)
    diff_s = diff_s.at[1:, 1:].set(-r_ik[1:, 1:] + r_ik[1:, :-1])

    # Calculate optical depths
    transparencies = jnp.einsum("ijkl,ml", transparencies, diff_s)

    # Delete unnecessary intermediate array
    del diff_s

    # Calculate transmittances (exp(-optical_depths), out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.exp(-1 * transparencies)

    # Integrate over g if c-k
    transparencies = lax.cond(
        weights_gauss.size == 1,
        lambda: transparencies[0],
        lambda: jnp.einsum("ijkl,i", transparencies, weights_gauss),
    )

    transparencies = jnp.swapaxes(
        transparencies, 1, 0
    )  # (species, frequencies, levels)

    transparencies = lax.cond(
        transparencies.shape[0] == 1,
        lambda: transparencies[0],
        lambda: jnp.prod(transparencies, axis=0),
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
    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.0)

    return transit_radii, radius_hydrostatic_equilibrium


@jit(static_argnames=("variable_gravity"))
def compute_transit_radii_summed(
    opacities: ArrayLike,
    continuum_opacities_scattering: ArrayLike,
    pressures: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    mean_molar_masses: ArrayLike,
    reference_gravity: ArrayLike,
    reference_pressure: ArrayLike,
    planet_radius: ArrayLike,
    variable_gravity: bool = True,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute transmission radii when species opacities are already summed.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, 1 or n_species, n_layers)
        Opacity cube whose first species slot already contains the summed absorption
        opacity used for transmission.
    continuum_opacities_scattering : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in transmission.
    pressures : jax.Array, shape (n_layers,)
        Pressure grid in bar.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    mean_molar_masses : jax.Array, shape (n_layers,)
        Mean molecular mass in amu.
    reference_gravity : float
        Gravity at ``reference_pressure`` in cgs units.
    reference_pressure : float
        Reference pressure in bar where ``planet_radius`` is defined.
    planet_radius : float
        Planet radius at ``reference_pressure`` in cm.
    variable_gravity : bool, optional
        If ``True``, vary gravity as $1/r^2$ when reconstructing the hydrostatic radius profile.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Effective transit radii with shape ``(n_frequencies,)`` and the hydrostatic
        radius profile with shape ``(n_layers,)``.
    """

    def _safe_sqrt_positive(values: ArrayLike) -> ArrayLike:
        positive_values = values > 0.0
        safe_values = jnp.where(
            positive_values,
            values,
            lax.stop_gradient(jnp.ones_like(values)),
        )
        return jnp.where(positive_values, jnp.sqrt(safe_values), 0.0)

    # Calculate planetary radius in hydrostatic equilibrium, using the atmospheric structure
    # (temperature, pressure, mmw), gravity, reference pressure and radius.
    pressures_cgs = pressures * 1e6  # bar to cgs
    reference_pressure_cgs = reference_pressure * 1e6  # bar to cgs

    atmospheric_densities = (
        pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures
    )
    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures=pressures_cgs,
        reference_gravity=reference_gravity,
        atmosphere_densities=atmospheric_densities,
        reference_pressure=reference_pressure_cgs,
        planet_radius=planet_radius,
        variable_gravity=variable_gravity,
    )

    is_negative_condition = radius_hydrostatic_equilibrium < 0.0
    first_non_negative_idx = jnp.argmax(
        radius_hydrostatic_equilibrium >= 0.0
    )  # Index of first True
    replacement_value = radius_hydrostatic_equilibrium[first_non_negative_idx]
    radius_hydrostatic_equilibrium = jnp.where(
        is_negative_condition, replacement_value, radius_hydrostatic_equilibrium
    )

    # Calculate the density
    rho = pressures_cgs * mean_molar_masses * cst.amu / cst.kB / temperatures

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
    r_ik = (
        radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.0
        - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.0
    )
    r_ik = _safe_sqrt_positive(r_ik)
    diff_s = diff_s.at[1:, 1:].set(-r_ik[1:, 1:] + r_ik[1:, :-1])

    # Calculate optical depths (out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.einsum("ijkl,ml", transparencies, diff_s)

    # Delete unnecessary intermediate array
    del diff_s

    # Calculate transmittances (exp(-optical_depths), out is used for in-place calculations to reduce memory usage)
    transparencies = jnp.exp(-1 * transparencies)

    transparencies = lax.cond(
        weights_gauss.size == 1,
        lambda: transparencies[0],
        lambda: jnp.einsum("ijkl,i", transparencies, weights_gauss, optimize=True),
    )

    transparencies = jnp.swapaxes(
        transparencies, 1, 0
    )  # (species, frequencies, levels)

    transparencies = lax.cond(
        transparencies.shape[0] == 1,
        lambda: transparencies[0],
        lambda: jnp.prod(transparencies, axis=0),
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
    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.0)

    return transit_radii, radius_hydrostatic_equilibrium


@jit(static_argnames=["scattering_in_transmission", "variable_gravity"])
def compute_transit_radii_fortran(
    opacities: ArrayLike,
    temperatures: ArrayLike,
    pressures: ArrayLike,
    reference_gravity: ArrayLike,
    mean_molar_masses: ArrayLike,
    reference_pressure: ArrayLike,
    planet_radius: ArrayLike,
    weights_gauss: ArrayLike,
    scattering_in_transmission: bool,
    continuum_opacities_scattering: ArrayLike,
    variable_gravity: bool,
) -> tuple[ArrayLike, ArrayLike]:
    """Mirror the historical Fortran transmission-radius implementation.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Absorptive opacity cube.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    pressures : jax.Array, shape (n_layers,)
        Pressure grid in cgs units.
    reference_gravity : float
        Gravity at the reference radius in cgs units.
    mean_molar_masses : jax.Array, shape (n_layers,)
        Mean molecular mass in amu.
    reference_pressure : float
        Reference pressure in bar where ``planet_radius`` is defined.
    planet_radius : float
        Planet radius at ``reference_pressure`` in cm.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    scattering_in_transmission : bool
        Whether to include continuum scattering in the grazing optical depth.
    continuum_opacities_scattering : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in transmission.
    variable_gravity : bool
        If ``True``, vary gravity as $1/r^2$ when reconstructing the hydrostatic radius profile.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Effective transit radii with shape ``(n_frequencies,)`` and the hydrostatic
        radius profile with shape ``(n_layers,)``.
    """
    n_g, n_frequencies, n_species, n_layers = opacities.shape
    P0_cgs = reference_pressure * 1e6

    rho = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures, reference_gravity, rho, P0_cgs, planet_radius, variable_gravity
    )

    opacities_ = jnp.where(opacities < 0, 0.0, opacities)

    alpha_t2 = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
    alpha_t2 = alpha_t2.at[:, :, :, :-1].set(
        opacities_[:, :, :, :-1] * rho[:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    )
    alpha_t2 = alpha_t2.at[:, :, :, :-2].add(
        opacities_[:, :, :, 1:-1] * rho[1:-1][jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    )
    alpha_t2 = alpha_t2.at[:, :, :, -1].set(
        alpha_t2[:, :, :, -1] + opacities_[:, :, :, -1] * rho[-1]
    )

    def transmission_scattering() -> ArrayLike:
        alpha_t2_scat = jnp.zeros((n_frequencies, n_layers))
        alpha_t2_scat = alpha_t2_scat.at[:, :-1].set(
            continuum_opacities_scattering[:, :-1] * rho[:-1]
        )
        alpha_t2_scat = alpha_t2_scat.at[:, 1:-1].add(
            continuum_opacities_scattering[:, 1:-1] * rho[1:-1]
        )
        alpha_t2_scat = alpha_t2_scat.at[:, -1].set(
            alpha_t2_scat[:, -1] + continuum_opacities_scattering[:, -1] * rho[-1]
        )
        return alpha_t2_scat

    def no_transmission_scattering() -> ArrayLike:
        return jnp.zeros((n_frequencies, n_layers))

    alpha_t2_scat = lax.cond(
        scattering_in_transmission, transmission_scattering, no_transmission_scattering
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
                t_graze_wlen_int = t_graze_wlen_int.at[i_freq, i_str].set(
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

    t_graze_wlen_int_1 = t_graze_wlen_int_1.at[:, 1:].set(
        t_graze_wlen_int[:, :-1] * d_radius[1:]
    )
    t_graze_wlen_int_2 = t_graze_wlen_int_2.at[:, 1:].set(
        t_graze_wlen_int[:, 1:] * d_radius[1:]
    )

    t_graze_wlen_int = t_graze_wlen_int_1 + t_graze_wlen_int_2
    t_graze_wlen_int_t = t_graze_wlen_int.T

    transit_radii = jnp.zeros(n_frequencies)
    for i_freq in range(n_frequencies):
        transit_radii = transit_radii.at[i_freq].set(
            jnp.sum(transit_radii[i_freq] + t_graze_wlen_int_t[:, i_freq])
        )

    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2)

    return transit_radii, radius_hydrostatic_equilibrium


@jit(static_argnames=["variable_gravity", "scattering_in_transmission"])
def compute_transmission_spectrum_contribution(
    opacities: ArrayLike,  # (n_g, n_frequencies, n_species, n_layers)
    temperatures: ArrayLike,  # (n_layers)
    pressures: ArrayLike,  # (n_layers)
    reference_gravity: ArrayLike,
    mean_molar_masses: ArrayLike,  # (n_layers)
    reference_pressure: ArrayLike,  # scalar, in bar
    planet_radius: ArrayLike,  # scalar, in cm
    weights_gauss: ArrayLike,  # (n_g)
    transit_radii_squared: ArrayLike,  # (n_frequencies) -> R_eff^2 with all layers
    continuum_opacities_scattering: ArrayLike,  # (n_frequencies, n_layers)
    scattering_in_transmission: bool = True,  # boolean
    variable_gravity: bool = True,  # boolean
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate normalized layer contributions to the transmission spectrum.

    Parameters
    ----------
    opacities : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Absorptive opacity cube.
    temperatures : jax.Array, shape (n_layers,)
        Atmospheric temperature profile.
    pressures : jax.Array, shape (n_layers,)
        Pressure grid in bar.
    reference_gravity : float
        Gravity at the reference radius in cgs units.
    mean_molar_masses : jax.Array, shape (n_layers,)
        Mean molecular mass in amu.
    reference_pressure : float
        Reference pressure in bar where ``planet_radius`` is defined.
    planet_radius : float
        Planet radius at ``reference_pressure`` in cm.
    weights_gauss : jax.Array, shape (n_g,)
        Correlated-k quadrature weights.
    transit_radii_squared : jax.Array, shape (n_frequencies,)
        Baseline effective transit radius squared computed with all layers included.
    continuum_opacities_scattering : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in transmission.
    scattering_in_transmission : bool, optional
        Whether to include continuum scattering in the grazing optical depth.
    variable_gravity : bool, optional
        If ``True``, vary gravity as $1/r^2$ when reconstructing the hydrostatic radius profile.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Normalized layer-by-frequency contribution array with shape
        ``(n_layers, n_frequencies)`` and the hydrostatic radius profile with shape
        ``(n_layers,)``.
    """
    n_g = opacities.shape[0]
    n_frequencies = opacities.shape[1]
    n_species = opacities.shape[2]
    n_layers = opacities.shape[3]

    P0_cgs = reference_pressure * 1e6
    pressures_cgs = pressures * 1e6
    rho = (mean_molar_masses * cst.amu * pressures_cgs) / (cst.kB * temperatures)

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures_cgs, reference_gravity, rho, P0_cgs, planet_radius, variable_gravity
    )
    R_sq = radius_hydrostatic_equilibrium**2

    def loop_body_leave_out_layer(i_leave_str_idx: int, _: Any) -> ArrayLike:
        current_opacities = opacities.at[..., i_leave_str_idx].set(0.0)

        # --- Conditionally set current_continuum_scattering for the left-out layer ---
        def set_continuum_scattering_true_branch(
            operands: tuple[ArrayLike, int],
        ) -> ArrayLike:
            _cos, _idx = operands
            return _cos.at[:, _idx].set(0.0)

        def set_continuum_scattering_false_branch(
            operands: tuple[ArrayLike, int],
        ) -> ArrayLike:
            _cos, _ = operands
            return _cos  # Return original if no scattering

        current_continuum_scattering_eff = lax.cond(
            scattering_in_transmission,
            set_continuum_scattering_true_branch,
            set_continuum_scattering_false_branch,
            (continuum_opacities_scattering, i_leave_str_idx),
        )
        # If not scattering_in_transmission, continuum_opacities_scattering might not be used
        # or could be zeros. If it's zeros, the above is fine.
        # If it's non-zero but scattering_in_transmission is False, we should ensure it's effectively zeroed.
        # A safer approach if continuum_opacities_scattering might be non-zero when flag is False:
        current_continuum_scattering_for_calc = lax.cond(
            scattering_in_transmission,
            lambda cos_in: cos_in.at[:, i_leave_str_idx].set(0.0),
            lambda cos_in: jnp.zeros_like(cos_in),  # Ensure it's zero if flag is false
            continuum_opacities_scattering,
        )

        rho_exp = rho[None, None, None, :]
        alpha_t2_current = (
            current_opacities[..., :-1] * rho_exp[..., :-1]
            + current_opacities[..., 1:] * rho_exp[..., 1:]
        )

        # --- Conditionally calculate alpha_t2_scat_current ---
        alpha_t2_scat_operands = (
            rho,
            current_continuum_scattering_for_calc,
            n_frequencies,
            n_layers,
        )

        def true_alpha_t2_scat_branch(
            ops: tuple[ArrayLike, ArrayLike, int, int],
        ) -> ArrayLike:
            _rho, _ccs, _, _ = ops
            _rho_exp_scat = _rho[None, :]
            return (
                _ccs[:, :-1] * _rho_exp_scat[:, :-1]
                + _ccs[:, 1:] * _rho_exp_scat[:, 1:]
            )

        def false_alpha_t2_scat_branch(
            ops: tuple[ArrayLike, ArrayLike, int, int],
        ) -> ArrayLike:
            # _, _, _n_freq, _n_layers_ops = ops  # Renamed to avoid conflict
            return jnp.zeros((n_frequencies, n_layers - 1))

        alpha_t2_scat_current = lax.cond(
            scattering_in_transmission,
            true_alpha_t2_scat_branch,
            false_alpha_t2_scat_branch,
            alpha_t2_scat_operands,
        )

        t_graze_current_init = jnp.zeros((n_g, n_frequencies, n_species, n_layers))
        t_graze_scat_current_init = jnp.zeros((n_frequencies, n_layers))

        def i_str_loop_body_for_t_graze(
            i_str_f_minus_2: int,
            current_t_grazes_tuple: tuple[ArrayLike, ArrayLike],
        ) -> tuple[ArrayLike, ArrayLike]:
            current_t_graze_accum, current_t_graze_scat_accum = current_t_grazes_tuple
            i_str_f = i_str_f_minus_2 + 2
            R_impact_sq_curr = R_sq[i_str_f - 1]

            def j_str_loop_body_for_t_graze(
                j_str_f_minus_1: int,
                grazes_sum_for_i_str_tuple: tuple[ArrayLike, ArrayLike],
            ) -> tuple[ArrayLike, ArrayLike]:
                t_graze_sum_gas, t_graze_sum_scat = grazes_sum_for_i_str_tuple
                j_str_f = j_str_f_minus_1 + 1
                idx_j_alpha = j_str_f - 1

                s_outer = jnp.sqrt(
                    jnp.maximum(0.0, R_sq[j_str_f - 1] - R_impact_sq_curr)
                )
                s_inner = lax.cond(
                    (j_str_f + 1) == i_str_f,
                    lambda _: 0.0,
                    lambda _: jnp.sqrt(
                        jnp.maximum(0.0, R_sq[j_str_f] - R_impact_sq_curr)
                    ),
                    operand=None,
                )
                path_in_shell = s_outer - s_inner

                t_graze_sum_gas += alpha_t2_current[..., idx_j_alpha] * path_in_shell

                # --- Conditionally update t_graze_scat_sum ---
                def update_t_graze_scat_sum_true(
                    ops: tuple[ArrayLike, ArrayLike, ArrayLike],
                ) -> ArrayLike:
                    _tgs_sum, _alpha_val, _path = ops
                    return _tgs_sum + _alpha_val * _path

                def update_t_graze_scat_sum_false(
                    ops: tuple[ArrayLike, ArrayLike, ArrayLike],
                ) -> ArrayLike:
                    _tgs_sum, _, _ = ops
                    return _tgs_sum

                t_graze_sum_scat_ops = (
                    t_graze_sum_scat,
                    alpha_t2_scat_current[:, idx_j_alpha],
                    path_in_shell,
                )
                t_graze_sum_scat = lax.cond(
                    scattering_in_transmission,
                    update_t_graze_scat_sum_true,
                    update_t_graze_scat_sum_false,
                    t_graze_sum_scat_ops,
                )
                return t_graze_sum_gas, t_graze_sum_scat

            initial_sums_for_i_str_gas = jnp.zeros_like(alpha_t2_current[..., 0])
            initial_sums_for_i_str_scat = jnp.zeros_like(
                alpha_t2_scat_current[:, 0]
            )  # Always init, even if not used

            final_t_graze_sum_gas, final_t_graze_sum_scat = lax.fori_loop(
                0,
                i_str_f - 1,
                j_str_loop_body_for_t_graze,
                (initial_sums_for_i_str_gas, initial_sums_for_i_str_scat),
            )

            current_t_graze_accum = current_t_graze_accum.at[..., i_str_f - 1].set(
                final_t_graze_sum_gas
            )

            # --- Conditionally update current_t_graze_scat_accum ---
            def update_scat_accum_true(
                ops: tuple[ArrayLike, ArrayLike, int],
            ) -> ArrayLike:
                _acc, _val, _idx = ops
                return _acc.at[:, _idx].set(_val)

            def update_scat_accum_false(
                ops: tuple[ArrayLike, ArrayLike, int],
            ) -> ArrayLike:
                _acc, _, _ = ops
                return _acc

            current_t_graze_scat_accum_ops = (
                current_t_graze_scat_accum,
                final_t_graze_sum_scat,
                i_str_f - 1,
            )
            current_t_graze_scat_accum = lax.cond(
                scattering_in_transmission,
                update_scat_accum_true,
                update_scat_accum_false,
                current_t_graze_scat_accum_ops,
            )
            return current_t_graze_accum, current_t_graze_scat_accum

        t_graze_current, t_graze_scat_current = lax.fori_loop(
            0,
            n_layers - 1,
            i_str_loop_body_for_t_graze,
            (t_graze_current_init, t_graze_scat_current_init),
        )

        exp_t_graze = jnp.exp(-t_graze_current)
        T_gas_spec_contrib = jnp.sum(
            exp_t_graze * weights_gauss[:, None, None, None], axis=0
        )
        T_gas_total = jnp.prod(T_gas_spec_contrib, axis=1)

        # --- Conditionally calculate T_scat_eff ---
        # T_gas_total is shape (n_frequencies, n_layers)
        # t_graze_scat_current is shape (n_frequencies, n_layers)
        def compute_T_scat_eff_true(ops: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
            _t_graze_scat, _ = ops
            return jnp.exp(-_t_graze_scat)

        def compute_T_scat_eff_false(ops: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
            _, _T_gas_total_template = ops
            return jnp.ones_like(_T_gas_total_template)

        T_scat_eff_ops = (t_graze_scat_current, T_gas_total)
        T_scat_eff = lax.cond(
            scattering_in_transmission,
            compute_T_scat_eff_true,
            compute_T_scat_eff_false,
            T_scat_eff_ops,
        )

        T_layer_freq = T_gas_total * T_scat_eff
        t_graze_wlen_int_temp = T_layer_freq.T

        y_val = (1.0 - t_graze_wlen_int_temp) * radius_hydrostatic_equilibrium[:, None]
        delta_R = (
            radius_hydrostatic_equilibrium[:-1] - radius_hydrostatic_equilibrium[1:]
        )
        integrand_terms = (y_val[:-1, :] + y_val[1:, :]) * delta_R[:, None]
        integral_val = jnp.sum(integrand_terms, axis=0)
        transit_radii_temp_sq = integral_val + R_sq[-1]

        contrib_for_this_layer = transit_radii_squared - transit_radii_temp_sq
        return contrib_for_this_layer

    def accumulate_contributions_loop(
        i_leave_str_idx: int, contributions_array: ArrayLike
    ) -> ArrayLike:
        contrib_for_this_layer = loop_body_leave_out_layer(i_leave_str_idx, None)
        contributions_array = contributions_array.at[i_leave_str_idx, :].set(
            contrib_for_this_layer
        )
        return contributions_array

    initial_contributions_array = jnp.zeros((n_layers, n_frequencies))
    transmission_contribution_raw = lax.fori_loop(
        0, n_layers, accumulate_contributions_loop, initial_contributions_array
    )

    sum_contrib = jnp.sum(transmission_contribution_raw, axis=0, keepdims=True)

    transmission_contribution_final = jnp.where(
        sum_contrib == 0, 0.0, transmission_contribution_raw / sum_contrib
    )

    return transmission_contribution_final, radius_hydrostatic_equilibrium


@jit
def integrate_opacities(
    intp_ind: int,
    kappa: ArrayLike,
    lambd: ArrayLike,
    l_bord1: ArrayLike,
    l_bord2: ArrayLike,
) -> ArrayLike:
    """Integrate a linearly interpolated opacity segment.

    Parameters
    ----------
    intp_ind : int
        Index of the wavelength interval containing the integration bounds.
    kappa : jax.Array, shape (n_lambda, ...)
        Opacity values tabulated on ``lambd``.
    lambd : jax.Array, shape (n_lambda,)
        Monotonic wavelength grid.
    l_bord1, l_bord2 : float
        Lower and upper integration bounds inside interval ``intp_ind``.

    Returns
    -------
    jax.Array
        Integrated opacity over ``[l_bord1, l_bord2]`` for the trailing dimensions of ``kappa``.
    """

    kappa_integ = kappa[intp_ind, :] * (l_bord2 - l_bord1) + (
        kappa[intp_ind + 1, :] - kappa[intp_ind, :]
    ) / (lambd[intp_ind + 1] - lambd[intp_ind]) * 0.5 * (
        (l_bord2 - lambd[intp_ind]) ** 2 - (l_bord1 - lambd[intp_ind]) ** 2
    )
    return kappa_integ


@jit
def transmission_g_tot_ck_scat(
    gravity: ArrayLike,
    pressure: ArrayLike,
    total_opacity_in: ArrayLike,
    continuum_opacity_scattering_emis: ArrayLike,
) -> tuple[tuple[ArrayLike, ArrayLike], ArrayLike]:
    """Legacy wrapper returning emission optical depths and destruction probabilities.

    Parameters
    ----------
    gravity : float
        Surface gravity in cgs units.
    pressure : jax.Array, shape (n_layers,)
        Pressure grid in cgs units.
    total_opacity_in : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
        Total absorptive opacity cube.
    continuum_opacity_scattering_emis : jax.Array, shape (n_frequencies, n_layers)
        Continuum scattering opacity used in emission.

    Returns
    -------
    tuple[tuple[jax.Array, jax.Array], jax.Array]
        The direct output of :func:`compute_optical_depths` and the photon destruction
        probability returned again for compatibility with older call sites.
    """
    photon_destruction_prob = compute_photon_destruction_prob(
        total_opacity_in, continuum_opacity_scattering_emis
    )
    transmission = compute_optical_depths(gravity, pressure, total_opacity_in)
    return transmission, photon_destruction_prob
