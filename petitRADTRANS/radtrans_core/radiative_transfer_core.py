"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""

from functools import partial
from typing import Literal, Optional, cast, overload

import jax
import jax.numpy as jnp
from jax import custom_vjp, jit, lax, pure_callback, vmap
from jax.typing import ArrayLike
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.radtrans_core.physics_core as physics_core
import petitRADTRANS.radtrans_core.linalg as linalg
from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

CST_PI = jnp.pi  # Or the specific precision from Fortran

TINIEST = jnp.finfo(FEAUTRIER_JAX_DTYPE).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(FEAUTRIER_JAX_DTYPE).max)
WORKING_TINIEST = jnp.finfo(WORKING_JAX_DTYPE).tiny

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


def _prepare_ck_opacity_inputs(
    opacities_fortran_order: ArrayLike,
    line_species_mass_fractions: ArrayLike,
    continuum_opacities: ArrayLike,
    input_layout: str,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, int, int, int, int]:
    """Prepare flattened c-k inputs shared by the CPU and GPU implementations."""
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

    return (
        opacities_flat,
        abundance_weights_flat,
        continuum_flat,
        n_frequencies,
        n_layers,
        n_species,
        n_g,
    )


def _finalize_ck_opacity_shape(
    current: ArrayLike,
    n_frequencies: int,
    n_layers: int,
    n_g: int,
) -> ArrayLike:
    """Reshape flattened c-k output back to Fortran-compatible layout."""
    return jnp.transpose(
        current.reshape(n_frequencies, n_layers, n_g),
        (2, 0, 1),
    )


@jit(static_argnames=("input_layout",))
def _combine_ck_opacities_jax(
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

    (
        opacities_flat,
        abundance_weights_flat,
        continuum_flat,
        n_frequencies,
        n_layers,
        n_species,
        n_g,
    ) = (
        _prepare_ck_opacity_inputs(
            opacities_fortran_order,
            line_species_mass_fractions,
            continuum_opacities,
            input_layout,
        )
    )

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
        t = jnp.clip(
            (g_target - x0) / jnp.maximum(x1 - x0, jnp.sqrt(WORKING_TINIEST)),
            0.0,
            1.0,
        )
        return y0 + t * (y1 - y0)

    # ------------------------------------------------------------------
    # C2 + C3: Vectorized threshold & species loop.
    # ------------------------------------------------------------------

    def _weight_species(species_index: int) -> ArrayLike:
        return opacities_flat[:, species_index, :] * abundance_weights_flat[
            :, species_index : species_index + 1
        ]

    # Initialize with first species + continuum  (n_fl, n_g)
    current = _weight_species(0) + continuum_flat[:, None]

    if n_species > 1:
        # C2: Vectorized threshold matching Fortran structure.
        # Fortran uses g=1 (first/weakest g-point) to compute the threshold,
        # then checks g=n_g (last/strongest) against it. The continuum has
        # already been added to species 0 in the Fortran input, so fold it
        # into the threshold probe here as well.
        threshold_input = (opacities_flat[:, :, 0] * abundance_weights_flat).at[
            :, 0
        ].add(continuum_flat)
        threshold = jnp.max(threshold_input, axis=1) * 1e-2  # (n_fl,)
        active_mask = (
            opacities_flat[:, :, -1] * abundance_weights_flat
        ) >= threshold[:, None]  # (n_fl, n_species)

        # Batched species loop — no vmap, single sort kernel per species.
        for s in range(1, n_species):
            spec_s = _weight_species(s)  # (n_fl, n_g)
            combined = _perform_combination_batched(current, spec_s)  # (n_fl, n_g)
            current = jnp.where(active_mask[:, s : s + 1], combined, current)

    return _finalize_ck_opacity_shape(
        current,
        n_frequencies,
        n_layers,
        n_g,
    )


# ---------------------------------------------------------------------------
# Fortran CPU forward path + custom VJP wrapper
# ---------------------------------------------------------------------------


def _prepare_fortran_ck_input(
    opacities_fortran_order: ArrayLike,
    line_species_mass_fractions: ArrayLike,
    continuum_opacities: ArrayLike,
    input_layout: str,
) -> ArrayLike:
    """Build mass-fraction-weighted opacities with continuum in (n_g, n_f, n_s, n_l) layout."""
    if input_layout == "fl_s_ng":
        n_f, n_l = continuum_opacities.shape
        _, n_s, n_g = opacities_fortran_order.shape
        opas = jnp.transpose(
            opacities_fortran_order.reshape(n_f, n_l, n_s, n_g),
            (3, 0, 2, 1),
        )
    else:
        opas = opacities_fortran_order

    # Apply mass-fraction weights: (n_l, n_s).T -> (n_s, n_l) -> broadcast to (n_g, n_f, n_s, n_l)
    modified = opas * line_species_mass_fractions.T[None, None, :, :]
    # Add continuum uniformly across g-points for species 0
    modified = modified.at[:, :, 0, :].set(
        modified[:, :, 0, :] + continuum_opacities[None, :, :]
    )
    return modified


def _run_fortran_ck(modified_opas, g_gauss_arr, weights_gauss_arr):
    """pure_callback target: invoke Fortran combine_ck_opacities on concrete arrays.

    Returns float64 when inputs are float64 (x64 enabled path), or matches the
    input dtype when x64 is disabled (float32 path). The declared ShapeDtypeStruct
    in _combine_ck_opacities_fortran_callback always matches this return dtype.
    """
    import numpy as np
    from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore

    return_dtype = np.asarray(modified_opas).dtype
    m = np.asfortranarray(np.asarray(modified_opas, dtype=np.float64))
    g = np.ascontiguousarray(np.asarray(g_gauss_arr, dtype=np.float64))
    w = np.ascontiguousarray(np.asarray(weights_gauss_arr, dtype=np.float64))
    result = fcore.combine_ck_opacities(m, g, w)
    return np.array(result, dtype=return_dtype, order="C", copy=True)


def _combine_ck_opacities_fortran_callback(
    modified: ArrayLike,
    g_gauss: ArrayLike,
    weights_gauss: ArrayLike,
) -> ArrayLike:
    n_g, n_f, _, n_l = modified.shape
    input_dtype = modified.dtype
    # When x64 is disabled, jax.ShapeDtypeStruct cannot declare float64.
    # In that case we pass float32 inputs and declare float32 output; the
    # Fortran subroutine still runs in double precision internally, and
    # _run_fortran_ck returns the dtype that matches what was declared here.
    if jax.config.jax_enable_x64:
        callback_dtype = jnp.float64
        callback_inputs = (
            jnp.asarray(modified, dtype=jnp.float64),
            jnp.asarray(g_gauss, dtype=jnp.float64),
            jnp.asarray(weights_gauss, dtype=jnp.float64),
        )
    else:
        callback_dtype = input_dtype
        callback_inputs = (
            jnp.asarray(modified, dtype=input_dtype),
            jnp.asarray(g_gauss, dtype=input_dtype),
            jnp.asarray(weights_gauss, dtype=input_dtype),
        )
    result = pure_callback(
        _run_fortran_ck,
        jax.ShapeDtypeStruct((n_g, n_f, n_l), callback_dtype),
        *callback_inputs,
    )
    return result.astype(input_dtype)


def _use_fortran_ck_callback() -> bool:
    """Use the Fortran callback only on CPU backends."""
    return jax.default_backend() == "cpu"


@partial(custom_vjp, nondiff_argnums=(7,))
def _combine_ck_opacities_fortran(
    opacities_fortran_order: ArrayLike,
    g_gauss: ArrayLike,
    weights_gauss: ArrayLike,
    line_species_mass_fractions: ArrayLike,
    continuum_opacities: ArrayLike,
    g_combination_weights: ArrayLike,
    g_combination_weights_sum: ArrayLike,
    input_layout: str,
) -> ArrayLike:
    """Correlated-k mixing: Fortran forward, JAX backward."""
    modified = _prepare_fortran_ck_input(
        opacities_fortran_order,
        line_species_mass_fractions,
        continuum_opacities,
        input_layout,
    )
    return _combine_ck_opacities_fortran_callback(
        modified, g_gauss, weights_gauss
    )


def _combine_ck_fortran_fwd(
    opacities_fortran_order,
    g_gauss,
    weights_gauss,
    line_species_mass_fractions,
    continuum_opacities,
    g_combination_weights,
    g_combination_weights_sum,
    input_layout,
):
    result = _combine_ck_opacities_fortran(
        opacities_fortran_order,
        g_gauss,
        weights_gauss,
        line_species_mass_fractions,
        continuum_opacities,
        g_combination_weights,
        g_combination_weights_sum,
        input_layout,
    )
    residuals = (
        opacities_fortran_order,
        g_gauss,
        weights_gauss,
        line_species_mass_fractions,
        continuum_opacities,
        g_combination_weights,
        g_combination_weights_sum,
    )
    return result, residuals


def _combine_ck_fortran_bwd(input_layout, residuals, g):
    _, vjp_fn = jax.vjp(
        partial(_combine_ck_opacities_jax, input_layout=input_layout),
        *residuals,
    )
    return vjp_fn(g)


_combine_ck_opacities_fortran.defvjp(_combine_ck_fortran_fwd, _combine_ck_fortran_bwd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    """Combine correlated-k opacities across species.

    Uses the Fortran implementation on CPU for the forward pass (via
    ``pure_callback``) and the JAX implementation for reverse-mode
    differentiation. Non-CPU backends use the pure JAX implementation for both
    forward and backward execution.
    """
    if g_combination_weights is None or g_combination_weights_sum is None:
        if weights_gauss is None:
            raise ValueError(
                "weights_gauss must be provided when precomputed c-k state is unavailable"
            )
        g_combination_weights, g_combination_weights_sum = build_ck_combination_state(
            weights_gauss
        )

    if weights_gauss is None:
        raise ValueError("weights_gauss is required for the Fortran callback")

    if line_species_mass_fractions is None or continuum_opacities is None:
        raise ValueError(
            "line_species_mass_fractions and continuum_opacities are required"
        )

    if _use_fortran_ck_callback():
        return _combine_ck_opacities_fortran(
            opacities_fortran_order,
            g_gauss,
            weights_gauss,
            line_species_mass_fractions,
            continuum_opacities,
            g_combination_weights,
            g_combination_weights_sum,
            input_layout,
        )

    return _combine_ck_opacities_jax(
        opacities_fortran_order,
        g_gauss,
        weights_gauss=weights_gauss,
        line_species_mass_fractions=line_species_mass_fractions,
        continuum_opacities=continuum_opacities,
        g_combination_weights=g_combination_weights,
        g_combination_weights_sum=g_combination_weights_sum,
        input_layout=input_layout,
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


@jit(static_argnames=("return_contribution",))
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
    if return_contribution:
        flux, contribution = calculate_ck_emission_contribution(
            frequencies,
            temperatures,
            weights_gauss,
            emission_cos_angles,
            emission_cos_angles_weights,
            optical_depths,
        )
    else:
        flux, contribution = calculate_flux(
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
        opacities (DeviceArray): Opacities with shape (..., n_layers).

    Returns:
        DeviceArray: Optical depths with the same leading shape as ``opacities``.
    """
    opacities_work = jnp.asarray(opacities, dtype=WORKING_JAX_DTYPE)
    pressures_work = jnp.asarray(pressures, dtype=WORKING_JAX_DTYPE)
    reference_gravity_work = jnp.asarray(reference_gravity, dtype=WORKING_JAX_DTYPE)

    # Calculate pressure differences
    pressure_diffs = pressures_work[1:] - pressures_work[:-1]
    pressure_diffs = jnp.reshape(
        pressure_diffs,
        (1,) * (opacities_work.ndim - 1) + (pressure_diffs.shape[0],),
    )

    # Calculate average opacities between layers (vectorized)
    avg_opacities = (opacities_work[..., 1:] + opacities_work[..., :-1]) / 2.0

    # Calculate optical depth increments (vectorized)
    d_optical_depths = avg_opacities * pressure_diffs / reference_gravity_work

    # Integrate optical depths cumulatively (vectorized)
    optical_depths = jnp.concatenate(
        [
            jnp.zeros_like(
                opacities_work[..., :1]
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

    # Solve the 2x2 Ng normal equations scale-invariantly so that
    # ill-conditioned rows -- which the final jnp.where discards anyway -- cannot
    # poison the reverse-mode gradient. A1=Q1.Q1>=0, B2=Q2.Q2>=0, A2=Q1.Q2, and
    # AB_denom=A1*B2-A2**2>=0 (Cauchy-Schwarz). The original code used an
    # absolute TINIEST threshold and divided by the raw determinant; a merely
    # tiny (not exactly zero) denominator then made a_ng/b_ng and especially
    # their gradient (~num/denom**2) overflow to inf, and the discarded branch
    # turned that into NaN via 0*inf -- which NaN'd the entire warm-start
    # gradient at the prior median. Normalising by the Gram scale puts the
    # determinant in [0, 1] so a fixed *relative* threshold detects near-singular
    # rows regardless of the source magnitude; the scale floor (sqrt of the
    # smallest normal) keeps 1/scale and its gradient finite; and skipped rows
    # get a trivial finite system so the dead branch is finite in value AND
    # gradient.
    gram_scale = jnp.maximum(A1, B2)
    inv_scale = 1.0 / jnp.maximum(gram_scale, jnp.sqrt(TINIEST))
    A1n = A1 * inv_scale
    A2n = A2 * inv_scale
    B2n = B2 * inv_scale
    C1n = C1 * inv_scale
    C2n = C2 * inv_scale
    AB_denom_n = A1n * B2n - A2n * A2n  # in [0, 1]

    skip = (gram_scale < TINIEST) | (AB_denom_n < 1e-6)  # (batch,)

    safe_denom = jnp.where(skip, 1.0, AB_denom_n)
    a_ng = jnp.where(skip, 0.0, C1n * B2n - C2n * A2n) / safe_denom
    b_ng = jnp.where(skip, 0.0, C2n * A1n - C1n * A2n) / safe_denom

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


@overload
def _compute_feautrier_radiative_transfer_chunk(
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
    return_solver_diagnostics: Literal[False] = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...


@overload
def _compute_feautrier_radiative_transfer_chunk(
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
    return_solver_diagnostics: Literal[True] = True,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]: ...


@jit(
    static_argnames=(
        "irradiation_geometry_str",
        "adaptive_feautrier_iterations",
        "return_contribution",
        "return_solver_diagnostics",
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
    return_solver_diagnostics: bool = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
) -> (
    tuple[ArrayLike, ArrayLike, ArrayLike]
    | tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]
):
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
    # Solution A: dtype-aware floor.  In float32 the old floor of 1e-10
    # is below machine epsilon relative to 1.0 (eps32 ~ 1.2e-7), so
    # 1 - (1 - 1e-10) collapses to zero, making the ALI denominator
    # insensitive to the photon destruction probability in scattering-
    # dominated cells.  Using 10*eps ensures the floor is always
    # representable as a correction to 1.0 in whatever dtype is active.
    _pdp_floor = (
        jnp.finfo(FEAUTRIER_JAX_DTYPE).eps * 10
    )  # ~1.2e-6 float32, ~2.2e-15 float64
    pdp_gfl = jnp.maximum(photon_destruction_probabilities_in, _pdp_floor)

    # ------------------------------------------------------------------
    # 3. Stellar attenuation  J_star_ini_gfl (g, f, l)
    # ------------------------------------------------------------------
    if geom > 0:
        stellar_intensity_abs_f = jnp.abs(stellar_intensity)

        # Accumulate each angle contribution in sequence to avoid a full
        # (g, f, angle, layer) exponential cube.
        def _accumulate_stellar_angle(
            carry: tuple[ArrayLike, ArrayLike],
            angle_inputs: tuple[ArrayLike, ArrayLike, ArrayLike],
        ) -> tuple[tuple[ArrayLike, ArrayLike], None]:
            j_star_accum_gfl, surface_reflection_accum_gf = carry
            mu, angle_weight, mu_weight = angle_inputs

            exp_over_mu_gfl = jnp.exp(-optical_depths_in / mu)
            j_star_accum_gfl = j_star_accum_gfl + angle_weight * exp_over_mu_gfl
            surface_reflection_accum_gf = (
                surface_reflection_accum_gf
                + mu_weight * exp_over_mu_gfl[:, :, n_layers - 1]
            )

            return (
                j_star_accum_gfl,
                surface_reflection_accum_gf,
            ), None

        (j_star_angle_sum_gfl, surface_reflection_angle_sum_gf), _ = lax.scan(
            _accumulate_stellar_angle,
            (
                jnp.zeros_like(optical_depths_in),
                jnp.zeros((n_g, n_freq_bins), dtype=optical_depths_in.dtype),
            ),
            (
                emission_cos_angles,
                emission_cos_angles_weights,
                mu_weight_a,
            ),
        )

        J_star_ini_gfl = (
            0.5 * geom * stellar_intensity_abs_f[None, :, None] * j_star_angle_sum_gfl
        )

        I_surface_reflection_gf = (
            surf_refl_2[None, :]
            * geom
            * stellar_intensity_abs_f[None, :]
            * surface_reflection_angle_sum_gf
        )

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
    # Floor |delta-tau| at sqrt(tiny), not tiny. The Feautrier coefficients below
    # divide by these (f1 = clip(mu/dtau, ...)); with a floor of `tiny` the
    # clipped forward value is finite but its gradient, -mu/dtau**2, overflows
    # because dtau**2 underflows to 0 -- and the clip then turns that into
    # 0*inf = NaN, which propagated into the whole log-posterior gradient. A
    # sqrt(tiny) floor keeps dtau**2 in the normal range (finite gradient); such
    # steps are physically zero and clipped regardless, so the forward is
    # unchanged.
    _dtau_floor = jnp.sqrt(TINIEST)
    dtau1 = optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1]  # (g,f,l-1)
    dtau1 = jnp.where(jnp.abs(dtau1) < _dtau_floor, _dtau_floor, dtau1)

    dtau2 = optical_depths_in[:, :, 2:] - optical_depths_in[:, :, :-2]  # (g,f,l-2)
    dtau2 = jnp.where(jnp.abs(dtau2) < _dtau_floor, _dtau_floor, dtau2)

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

    # Guard the diagonal before dividing: b_mid = 1 + f1*(f2+f3) can pass
    # through ~0 at non-monotonic / extreme optical depths, and 1/b_mid then has
    # a gradient -1/b_mid**2 that overflows (b_mid**2 underflows), feeding a NaN
    # into lambda_loc -> the ALI source. Floor |b_mid| at sqrt(tiny) (forward
    # unchanged for any non-degenerate cell).
    _b_floor = jnp.sqrt(TINIEST)
    b_mid_safe = jnp.where(jnp.abs(b_mid) < _b_floor, _b_floor, b_mid)
    lam_mid = jnp.sum(
        angle_weight_a[None, None, :, None] / b_mid_safe, axis=2
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
    # Solution B: rewrite the ALI denominator to avoid catastrophic
    # cancellation near 1.0.  The identity
    #   1 - (1 - eps) * lambda  =  eps + (1 - eps) * (1 - lambda)
    # keeps both addends positive and O(eps) or O(1-lambda), so neither
    # term is lost to rounding regardless of the floating-point precision.
    ali_denom_gfl = pdp_gfl + one_minus_pdp_gfl * (1.0 - lambda_loc_gfl)  # (g,f,l)
    ali_inv_denom_gfl = 1.0 / jnp.maximum(ali_denom_gfl, TINIEST)  # (g,f,l)

    # ------------------------------------------------------------------
    # 12. Inner Feautrier solve
    # ------------------------------------------------------------------
    # Pre-factor the tridiagonal LHS once (CPU only; returns None on GPU).
    # Subsequent solves inside the scan loop skip the LHS forward elimination.
    thomas_factors = linalg.factor_feautrier(
        a_flat, b_flat, c_flat, n_g, n_freq_bins, n_angles, n_layers
    )

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

        I_H_top, x_last_gfa, x_nm2_gfa, J_bol_gfl = linalg.solve_feautrier_prefactored(
            thomas_factors,
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
    # 13. Scattering iteration control.
    #
    # Reverse-mode autodiff cannot differentiate the original data-dependent
    # while-loop. Keep the adaptive path on a static scan budget, but split
    # out a dedicated fixed-iteration kernel and a conservative direct solve
    # for columns with no atmospheric scattering and a non-reflective lower
    # boundary.
    # ------------------------------------------------------------------

    zero_source_gfl = jnp.zeros((n_g, n_freq_bins, n_layers))
    initial_I_plus_surface_gfa = jnp.zeros((n_g, n_freq_bins, n_angles))
    initial_flux_gf = jnp.zeros((n_g, n_freq_bins))
    initial_I_H_top = jnp.zeros((n_g, n_freq_bins, n_angles))
    initial_I_minus_gf = (
        jnp.zeros((n_g, n_freq_bins))
        if return_contribution
        else jnp.zeros((1,), dtype=planck_broadcast_gfl.dtype)
    )
    configured_iter_budget = jnp.array(iter_scat_max + 1, dtype=jnp.int32)
    fixed_iteration_count = min(iter_scat_max + 1, 9)

    no_scattering_pdp_tol = _pdp_floor
    no_atmospheric_scattering = jnp.all(one_minus_pdp_gfl <= no_scattering_pdp_tol)
    non_reflective_lower_boundary = jnp.all(jnp.abs(reflectances) <= TINIEST)
    use_direct_no_scattering_solve = (
        no_atmospheric_scattering & non_reflective_lower_boundary
    )

    def _surface_reflection(I_plus_gfa: ArrayLike) -> ArrayLike:
        return surf_refl_2[None, :] * jnp.sum(
            I_plus_gfa * mu_weight_a[None, None, :], axis=2
        )  # (g, f)

    def _advance_state(
        state: tuple[ArrayLike, ...],
        iter_idx: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ...], ArrayLike]:
        (
            current_source_gfl,
            current_I_plus_surface_gfa,
            current_flux_gf_prev,
            current_I_minus_gf,
            _current_I_H_top,
            current_hist0_gfl,
            current_hist1_gfl,
            current_hist2_gfl,
            current_hist3_gfl,
            _current_all_converged,
            _solve_count,
            _noop_count,
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

        def _slot0(_: None):
            next_hist0 = source_ali
            return (
                next_hist0,
                current_hist1_gfl,
                current_hist2_gfl,
                current_hist3_gfl,
                next_hist0,
                current_hist3_gfl,
                current_hist2_gfl,
                current_hist1_gfl,
            )

        def _slot1(_: None):
            next_hist1 = source_ali
            return (
                current_hist0_gfl,
                next_hist1,
                current_hist2_gfl,
                current_hist3_gfl,
                next_hist1,
                current_hist0_gfl,
                current_hist3_gfl,
                current_hist2_gfl,
            )

        def _slot2(_: None):
            next_hist2 = source_ali
            return (
                current_hist0_gfl,
                current_hist1_gfl,
                next_hist2,
                current_hist3_gfl,
                next_hist2,
                current_hist1_gfl,
                current_hist0_gfl,
                current_hist3_gfl,
            )

        def _slot3(_: None):
            next_hist3 = source_ali
            return (
                current_hist0_gfl,
                current_hist1_gfl,
                current_hist2_gfl,
                next_hist3,
                next_hist3,
                current_hist2_gfl,
                current_hist1_gfl,
                current_hist0_gfl,
            )

        (
            next_hist0_gfl,
            next_hist1_gfl,
            next_hist2_gfl,
            next_hist3_gfl,
            tn,
            tn1,
            tn2,
            tn3,
        ) = lax.switch(slot_index, (_slot0, _slot1, _slot2, _slot3), operand=None)

        do_ng = (iter_idx > 0) & ((iter_idx + 1) % 4 == 0)

        def _apply_ng(_: None) -> ArrayLike:
            batch_size = n_g * n_freq_bins
            return _ng_accelerated_batched(
                tn.reshape(batch_size, n_layers),
                tn1.reshape(batch_size, n_layers),
                tn2.reshape(batch_size, n_layers),
                tn3.reshape(batch_size, n_layers),
                source_ali.reshape(batch_size, n_layers),
            ).reshape(n_g, n_freq_bins, n_layers)

        next_source_gfl = lax.cond(
            do_ng,
            _apply_ng,
            lambda _: source_ali,
            operand=None,
        )
        # Ng acceleration is a forward-only convergence accelerator: its internal
        # 2x2 least-squares solve spans an enormous dynamic range at extreme
        # atmospheres and yields NaN reverse-mode gradients (0*inf from the
        # discarded ill-conditioned rows) that silently zeroed the warm-start
        # gradient at the prior median. Route the gradient through the
        # numerically clean ALI source update instead, keeping the accelerated
        # value only in the forward pass. At the fixed point the accelerated and
        # ALI sources coincide, so dS*/dtheta is unchanged; elsewhere this is the
        # standard "treat the accelerator as a constant" approximation.
        next_source_gfl = source_ali + lax.stop_gradient(next_source_gfl - source_ali)

        flux_gf_cur = -jnp.einsum("gfa,a->gf", I_H_top, mu_weight_a)
        conv_val_gf = jnp.abs(
            1.0
            - current_flux_gf_prev
            / jnp.where(
                jnp.abs(flux_gf_cur) < jnp.sqrt(TINIEST),
                jnp.sqrt(TINIEST),
                flux_gf_cur,
            )
        )

        if return_contribution:
            next_I_minus_gf = I_minus_gf_new
        else:
            next_I_minus_gf = current_I_minus_gf

        next_state = (
            next_source_gfl,
            new_I_plus_surface,
            flux_gf_cur,
            next_I_minus_gf,
            I_H_top,
            next_hist0_gfl,
            next_hist1_gfl,
            next_hist2_gfl,
            next_hist3_gfl,
            _current_all_converged,
            _solve_count,
            _noop_count,
        )
        return next_state, conv_val_gf

    initial_carry = (
        planck_broadcast_gfl,
        initial_I_plus_surface_gfa,
        initial_flux_gf,
        initial_I_minus_gf,
        initial_I_H_top,
        zero_source_gfl,
        zero_source_gfl,
        zero_source_gfl,
        zero_source_gfl,
        jnp.array(False, dtype=jnp.bool_),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    def _run_no_scattering_direct(
        _: None,
    ) -> tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
    ]:
        direct_I_minus_gf = I_surface_no_scatter_gf
        _, direct_I_H_top, _ = _feautrier_solve(
            planck_broadcast_gfl,
            initial_I_plus_surface_gfa,
            direct_I_minus_gf,
        )
        flux_gf_direct = -jnp.einsum("gfa,a->gf", direct_I_H_top, mu_weight_a)
        final_I_minus_gf = (
            direct_I_minus_gf if return_contribution else initial_I_minus_gf
        )

        return (
            planck_broadcast_gfl,
            flux_gf_direct,
            final_I_minus_gf,
            direct_I_H_top,
            jnp.array(True, dtype=jnp.bool_),
            jnp.array(1, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32),
            configured_iter_budget,
        )

    def _run_fixed_iterations(
        _: None,
    ) -> tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
    ]:
        def _fixed_loop_body(
            iter_idx: int,
            state: tuple[ArrayLike, ...],
        ) -> tuple[ArrayLike, ...]:
            next_state, _ = _advance_state(
                state, jnp.asarray(iter_idx, dtype=jnp.int32)
            )
            return (
                next_state[0],
                next_state[1],
                next_state[2],
                next_state[3],
                next_state[4],
                next_state[5],
                next_state[6],
                next_state[7],
                next_state[8],
                jnp.array(False, dtype=jnp.bool_),
                state[10] + jnp.array(1, dtype=jnp.int32),
                state[11],
            )

        final_carry = lax.fori_loop(
            0,
            fixed_iteration_count,
            _fixed_loop_body,
            initial_carry,
        )

        return (
            final_carry[0],
            final_carry[2],
            final_carry[3],
            final_carry[4],
            jnp.array(fixed_iteration_count == 9, dtype=jnp.bool_),
            final_carry[10],
            final_carry[11],
            jnp.array(fixed_iteration_count, dtype=jnp.int32),
            configured_iter_budget,
        )

    def _adaptive_scan_step(
        carry: tuple[ArrayLike, ...],
        iter_idx: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ...], None]:
        def _advance(state: tuple[ArrayLike, ...]) -> tuple[ArrayLike, ...]:
            next_state, conv_val_gf = _advance_state(state, iter_idx)
            completed_iterations = iter_idx + jnp.array(1, dtype=jnp.int32)
            next_all_converged = jnp.all(
                (completed_iterations >= n_iter_min_gf)
                & (conv_val_gf < convergence_threshold)
            )
            return (
                next_state[0],
                next_state[1],
                next_state[2],
                next_state[3],
                next_state[4],
                next_state[5],
                next_state[6],
                next_state[7],
                next_state[8],
                next_all_converged,
                state[10] + jnp.array(1, dtype=jnp.int32),
                state[11],
            )

        def _skip(state: tuple[ArrayLike, ...]) -> tuple[ArrayLike, ...]:
            return (
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                state[5],
                state[6],
                state[7],
                state[8],
                state[9],
                state[10],
                state[11] + jnp.array(1, dtype=jnp.int32),
            )

        next_carry = lax.cond(
            jnp.logical_not(carry[9]),
            _advance,
            _skip,
            carry,
        )
        return next_carry, None

    def _run_adaptive_iterations(
        _: None,
    ) -> tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
    ]:
        final_carry, _ = lax.scan(
            _adaptive_scan_step,
            initial_carry,
            jnp.arange(iter_scat_max + 1, dtype=jnp.int32),
        )

        return (
            final_carry[0],
            final_carry[2],
            final_carry[3],
            final_carry[4],
            final_carry[9],
            final_carry[10],
            final_carry[11],
            configured_iter_budget,
            configured_iter_budget,
        )

    loop_runner = (
        _run_adaptive_iterations
        if adaptive_feautrier_iterations
        else _run_fixed_iterations
    )
    (
        final_source_gfl,
        flux_gf_final,
        I_minus_gf_final,
        I_H_top,
        final_all_converged,
        solve_count,
        noop_count,
        iter_budget,
        configured_iter_budget_out,
    ) = lax.cond(
        use_direct_no_scattering_solve,
        _run_no_scattering_direct,
        loop_runner,
        operand=None,
    )

    # ------------------------------------------------------------------
    # 14. Flux  (n_freq_bins,)
    # ------------------------------------------------------------------
    flux_f = pi_4 * jnp.einsum("gf,g->f", flux_gf_final, weights_gauss)
    angle_dependent_intensities = 2.0 * jnp.einsum(
        "gfa,g->fa", I_H_top, weights_gauss
    )  # (f, a)

    # ------------------------------------------------------------------
    # 15. Contribution function (optional)
    # ------------------------------------------------------------------
    def compute_contribution() -> ArrayLike:
        weights_gauss_work = jnp.asarray(weights_gauss, dtype=WORKING_JAX_DTYPE)
        final_source_work = jnp.asarray(final_source_gfl, dtype=WORKING_JAX_DTYPE)
        i_minus_work = jnp.asarray(I_minus_gf_final, dtype=WORKING_JAX_DTYPE)
        optical_depths_work = jnp.asarray(optical_depths_in, dtype=WORKING_JAX_DTYPE)
        emission_cos_angles_work = jnp.asarray(
            emission_cos_angles, dtype=WORKING_JAX_DTYPE
        )
        emission_cos_angles_weights_work = jnp.asarray(
            emission_cos_angles_weights, dtype=WORKING_JAX_DTYPE
        )

        # G-integrated source: (l, f)
        r_src_lf = jnp.einsum("g,gfl->lf", weights_gauss_work, final_source_work)

        # G-integrated I_minus: (f,)
        I_minus_f = jnp.einsum("g,gf->f", weights_gauss_work, i_minus_work)

        # This replaces n_angles separate exp(-tau/mu) calls inside fori_loop.
        transm_gfla = jnp.exp(
            -optical_depths_work[:, :, :, None]
            / emission_cos_angles_work[None, None, None, :]
        )  # (g, f, l, a)
        transm_lfa = jnp.einsum("g,gfla->lfa", weights_gauss_work, transm_gfla)  # (l, f, a)
        del transm_gfla
        def i_mu_body(i_mu: int, acc: ArrayLike) -> ArrayLike:
            mu_val = emission_cos_angles_work[i_mu]
            mu_weight_val = emission_cos_angles_weights_work[i_mu]

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
            0,
            n_angles,
            i_mu_body,
            jnp.zeros((n_layers, n_freq_bins), dtype=WORKING_JAX_DTYPE),
        )
        total = jnp.sum(accumulated, axis=0, keepdims=True)
        return jnp.where(
            jnp.abs(total) < WORKING_TINIEST,
            jnp.zeros_like(accumulated),
            accumulated / total,
        )

    if return_contribution:
        emission_contribution_lf = compute_contribution()
    else:
        emission_contribution_lf = None

    solver_diagnostics = {
        "total_solve_count": solve_count,
        "total_noop_count": noop_count,
        "iter_scat_budget": iter_budget,
        "configured_iter_scat_budget": configured_iter_budget_out,
        "converged_before_cap": final_all_converged,
    }

    if return_solver_diagnostics:
        return (
            flux_f,
            emission_contribution_lf,
            angle_dependent_intensities,
            solver_diagnostics,
        )

    return flux_f, emission_contribution_lf, angle_dependent_intensities


@overload
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
    return_solver_diagnostics: Literal[False] = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
    chunk_feautrier: bool = True,
    feautrier_chunk_size: int = 1536,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]: ...


@overload
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
    return_solver_diagnostics: Literal[True] = True,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
    chunk_feautrier: bool = True,
    feautrier_chunk_size: int = 1536,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]: ...


@jit(
    static_argnames=(
        "irradiation_geometry_str",
        "adaptive_feautrier_iterations",
        "return_contribution",
        "return_solver_diagnostics",
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
    return_solver_diagnostics: bool = False,
    iter_scat_max: int = 100,
    convergence_threshold: float = 1e-2,
    chunk_feautrier: bool = True,
    feautrier_chunk_size: int = 1536,
) -> (
    tuple[ArrayLike, ArrayLike, ArrayLike]
    | tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]
):
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
    n_angles = emission_cos_angles.shape[0]

    if (not chunk_feautrier) or feautrier_chunk_size >= n_freq_bins:
        return _compute_feautrier_radiative_transfer_chunk(
            frequencies_bin_edges=frequencies_bin_edges,
            temperatures=temperatures,
            weights_gauss=weights_gauss,
            emission_cos_angles=emission_cos_angles,
            emission_cos_angles_weights=emission_cos_angles_weights,
            optical_depths_in=optical_depths_in.astype(FEAUTRIER_JAX_DTYPE),
            photon_destruction_probabilities_in=photon_destruction_probabilities_in.astype(
                FEAUTRIER_JAX_DTYPE
            ),
            stellar_intensity=stellar_intensity,
            star_irradiation_cos_angle=star_irradiation_cos_angle,
            reflectances=reflectances,
            emissivities=emissivities,
            adaptive_feautrier_iterations=adaptive_feautrier_iterations,
            irradiation_geometry_str=irradiation_geometry_str,
            return_contribution=return_contribution,
            return_solver_diagnostics=return_solver_diagnostics,
            iter_scat_max=iter_scat_max,
            convergence_threshold=convergence_threshold,
        )

    if feautrier_chunk_size <= 0:
        raise ValueError(
            f"feautrier_chunk_size must be > 0, but was {feautrier_chunk_size}"
        )

    n_chunks = (n_freq_bins + feautrier_chunk_size - 1) // feautrier_chunk_size

    # Frequency-chunk start indices. The final chunk's start is clamped so that
    # every chunk reads a full ``feautrier_chunk_size`` window that stays within
    # bounds; the last chunk therefore overlaps the previous one. Overlapping
    # bins are recomputed identically and overwritten during reassembly, so the
    # result is exact while avoiding any full-resolution padded copy of the
    # large optical-depth / photon-destruction arrays.
    # Use the canonical integer dtype (matches the Python int literals used as
    # the other dynamic_slice start indices, in both x32 and x64 modes).
    chunk_starts = jnp.minimum(
        jnp.arange(n_chunks) * feautrier_chunk_size,
        n_freq_bins - feautrier_chunk_size,
    )

    def _extract_chunk(start):
        # Slice a single frequency window directly out of the unpadded inputs.
        # The large optical-depth / photon-destruction arrays are passed in the
        # compact WORKING dtype and upcast to FEAUTRIER_JAX_DTYPE one chunk at a
        # time, so their full-resolution float64 copies are never materialised.
        chunk_optical_depths = lax.dynamic_slice(
            optical_depths_in,
            (0, start, 0),
            (n_g, feautrier_chunk_size, n_layers),
        ).astype(FEAUTRIER_JAX_DTYPE)
        chunk_photon_destruction_probabilities = lax.dynamic_slice(
            photon_destruction_probabilities_in,
            (0, start, 0),
            (n_g, feautrier_chunk_size, n_layers),
        ).astype(FEAUTRIER_JAX_DTYPE)
        chunk_stellar_intensity = lax.dynamic_slice(
            stellar_intensity, (start,), (feautrier_chunk_size,)
        )
        chunk_reflectances = lax.dynamic_slice(
            reflectances, (start,), (feautrier_chunk_size,)
        )
        chunk_emissivities = lax.dynamic_slice(
            emissivities, (start,), (feautrier_chunk_size,)
        )
        chunk_frequency_bin_edges = lax.dynamic_slice(
            frequencies_bin_edges, (start,), (feautrier_chunk_size + 1,)
        )
        return (
            chunk_frequency_bin_edges,
            chunk_optical_depths,
            chunk_photon_destruction_probabilities,
            chunk_stellar_intensity,
            chunk_reflectances,
            chunk_emissivities,
        )

    def _scatter_chunks(chunks, out_shape, freq_axis):
        # Reassemble per-chunk outputs (stacked along axis 0 by lax.scan) into a
        # single ``out_shape`` array by placing each chunk at its (clamped)
        # frequency offset along ``freq_axis``. ``freq_axis`` is the position of
        # the (length feautrier_chunk_size) frequency dimension inside each
        # per-chunk array. The overlapping tail of the final chunk overwrites
        # identical recomputed values, so the result is exact.
        output = jnp.zeros(out_shape, dtype=chunks.dtype)

        def _place(i, acc):
            return lax.dynamic_update_slice_in_dim(
                acc, chunks[i], chunk_starts[i], axis=freq_axis
            )

        return lax.fori_loop(0, n_chunks, _place, output)

    if return_solver_diagnostics:
        if return_contribution:

            def _chunk_body(
                _: None, chunk_inputs: tuple[ArrayLike, ...]
            ) -> tuple[None, tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]]:
                (
                    chunk_frequency_bin_edges,
                    chunk_optical_depths,
                    chunk_photon_destruction_probabilities,
                    chunk_stellar_intensity,
                    chunk_reflectances,
                    chunk_emissivities,
                ) = _extract_chunk(chunk_inputs)

                (
                    chunk_flux,
                    chunk_emission_contribution,
                    chunk_angle_dependent_intensities,
                    chunk_solver_diagnostics,
                ) = _compute_feautrier_radiative_transfer_chunk(
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
                    return_contribution=True,
                    return_solver_diagnostics=True,
                    iter_scat_max=iter_scat_max,
                    convergence_threshold=convergence_threshold,
                )

                return None, (
                    chunk_flux,
                    chunk_emission_contribution,
                    chunk_angle_dependent_intensities,
                    chunk_solver_diagnostics,
                )

            _, (
                flux_chunks,
                emission_contribution_chunks,
                angle_dependent_intensities_chunks,
                solver_diagnostics_chunks,
            ) = lax.scan(
                _chunk_body,
                None,
                chunk_starts,
            )
        else:

            def _chunk_body(
                _: None, chunk_inputs: tuple[ArrayLike, ...]
            ) -> tuple[None, tuple[ArrayLike, ArrayLike, dict[str, ArrayLike]]]:
                (
                    chunk_frequency_bin_edges,
                    chunk_optical_depths,
                    chunk_photon_destruction_probabilities,
                    chunk_stellar_intensity,
                    chunk_reflectances,
                    chunk_emissivities,
                ) = _extract_chunk(chunk_inputs)

                (
                    chunk_flux,
                    _,
                    chunk_angle_dependent_intensities,
                    chunk_solver_diagnostics,
                ) = _compute_feautrier_radiative_transfer_chunk(
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
                    return_contribution=False,
                    return_solver_diagnostics=True,
                    iter_scat_max=iter_scat_max,
                    convergence_threshold=convergence_threshold,
                )

                return None, (
                    chunk_flux,
                    chunk_angle_dependent_intensities,
                    chunk_solver_diagnostics,
                )

            _, (
                flux_chunks,
                angle_dependent_intensities_chunks,
                solver_diagnostics_chunks,
            ) = lax.scan(
                _chunk_body,
                None,
                chunk_starts,
            )
    else:
        if return_contribution:

            def _chunk_body(
                _: None, chunk_inputs: tuple[ArrayLike, ...]
            ) -> tuple[None, tuple[ArrayLike, ArrayLike, ArrayLike]]:
                (
                    chunk_frequency_bin_edges,
                    chunk_optical_depths,
                    chunk_photon_destruction_probabilities,
                    chunk_stellar_intensity,
                    chunk_reflectances,
                    chunk_emissivities,
                ) = _extract_chunk(chunk_inputs)

                chunk_result = cast(
                    tuple[ArrayLike, ArrayLike, ArrayLike],
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
                        return_contribution=True,
                        return_solver_diagnostics=False,
                        iter_scat_max=iter_scat_max,
                        convergence_threshold=convergence_threshold,
                    ),
                )

                chunk_flux = chunk_result[0]
                chunk_emission_contribution = chunk_result[1]
                chunk_angle_dependent_intensities = chunk_result[2]

                return None, (
                    chunk_flux,
                    chunk_emission_contribution,
                    chunk_angle_dependent_intensities,
                )

            _, (
                flux_chunks,
                emission_contribution_chunks,
                angle_dependent_intensities_chunks,
            ) = lax.scan(
                _chunk_body,
                None,
                chunk_starts,
            )
        else:

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
                ) = _extract_chunk(chunk_inputs)

                chunk_flux, _, chunk_angle_dependent_intensities = (
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
                        return_contribution=False,
                        return_solver_diagnostics=False,
                        iter_scat_max=iter_scat_max,
                        convergence_threshold=convergence_threshold,
                    )
                )

                return None, (
                    chunk_flux,
                    chunk_angle_dependent_intensities,
                )

            _, (
                flux_chunks,
                angle_dependent_intensities_chunks,
            ) = lax.scan(
                _chunk_body,
                None,
                chunk_starts,
            )

    # flux per chunk: (cs,)            -> (n_freq_bins,),           freq axis 0
    # angle per chunk: (cs, n_angles)  -> (n_freq_bins, n_angles),  freq axis 0
    # contribution per chunk: (n_layers, cs) -> (n_layers, n_freq_bins), freq axis 1
    flux_f = _scatter_chunks(flux_chunks, (n_freq_bins,), 0)
    angle_dependent_intensities = _scatter_chunks(
        angle_dependent_intensities_chunks, (n_freq_bins, n_angles), 0
    )

    if return_contribution:
        emission_contribution_lf = _scatter_chunks(
            emission_contribution_chunks, (n_layers, n_freq_bins), 1
        )
    else:
        emission_contribution_lf = None

    if return_solver_diagnostics:
        solver_diagnostics = {
            "total_solve_count": jnp.sum(
                solver_diagnostics_chunks["total_solve_count"]
            ),
            "total_noop_count": jnp.sum(solver_diagnostics_chunks["total_noop_count"]),
            "iter_scat_budget": jnp.sum(solver_diagnostics_chunks["iter_scat_budget"]),
            "configured_iter_scat_budget": jnp.sum(
                solver_diagnostics_chunks["configured_iter_scat_budget"]
            ),
            "converged_before_cap": jnp.all(
                solver_diagnostics_chunks["converged_before_cap"]
            ),
            "n_chunks": jnp.array(n_chunks, dtype=jnp.int32),
        }
        return (
            flux_f,
            emission_contribution_lf,
            angle_dependent_intensities,
            solver_diagnostics,
        )

    return flux_f, emission_contribution_lf, angle_dependent_intensities


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

    opacities_work = jnp.asarray(opacities, dtype=WORKING_JAX_DTYPE)
    pressures_work = jnp.asarray(pressures, dtype=WORKING_JAX_DTYPE)
    reference_gravity_work = jnp.asarray(reference_gravity, dtype=WORKING_JAX_DTYPE)
    if continuum_opacities_scattering is None:
        continuum_opacities_scattering_work = jnp.zeros(
            opacities_work[:, :, 0, :].shape, dtype=WORKING_JAX_DTYPE
        )
    else:
        continuum_opacities_scattering_work = jnp.asarray(
            continuum_opacities_scattering, dtype=WORKING_JAX_DTYPE
        )

    def scattering_function() -> tuple[ArrayLike, ArrayLike]:
        # Add continuum scattering to the first species (assumed at index 0)
        _opacities = opacities_work.at[:, :, 0, :].add(
            continuum_opacities_scattering_work
        )
        # Compute photon destruction probabilities for all g, freq, press
        absorptive_opac = opacities_work[:, :, 0, :]
        total_opac = _opacities[:, :, 0, :]
        # Floor the denominator at sqrt(tiny), not tiny. absorptive/total has
        # gradient ~ -absorptive/total**2, and for total just above WORKING_TINIEST
        # the square underflows to 0 -> inf gradient, which the outer jnp.where
        # then turns into 0*inf = NaN. Because this photon-destruction
        # probability feeds the scattering source at every (g, f, layer), that
        # single NaN propagated into essentially every parameter's gradient at
        # the prior median. sqrt(tiny) keeps total**2 normal; cells with such
        # negligible opacity are physically irrelevant, so the forward is
        # unchanged in practice.
        safe_total_opac = jnp.maximum(
            total_opac, jnp.sqrt(WORKING_TINIEST)
        )
        photon_destruction_probabilities = absorptive_opac / safe_total_opac
        photon_destruction_probabilities = jnp.where(
            total_opac > WORKING_TINIEST,
            photon_destruction_probabilities,
            jnp.ones_like(total_opac),
        )
        return photon_destruction_probabilities, _opacities

    def false_function() -> tuple[ArrayLike, ArrayLike]:
        return (
            jnp.ones(opacities_work[:, :, 0, :].shape, dtype=WORKING_JAX_DTYPE),
            opacities_work,
        )

    photon_destruction_probabilities, _opacities = lax.cond(
        scattering_in_emission, scattering_function, false_function
    )

    photon_destruction_probabilities = jnp.clip(
        photon_destruction_probabilities, 0.0, 1.0
    )

    # Vectorized optical depth calculation
    pressure_diffs = pressures_work[1:] - pressures_work[:-1]  # (n_layers-1,)
    avg_opacities = (
        _opacities[..., 1:] + _opacities[..., :-1]
    ) / 2.0  # (..., n_layers-1)
    d_optical_depths = (
        avg_opacities * pressure_diffs / reference_gravity_work
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
        # Floor at sqrt(tiny): opacity_g/tot has gradient ~ -opacity_g/tot**2,
        # which overflows (tot**2 underflows) for tot just above tiny and is then
        # turned into 0*inf = NaN by the jnp.where. See scattering_function.
        safe_tot = jnp.maximum(tot, jnp.sqrt(TINIEST))
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
            return jnp.asarray(
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
                * coefficient,
                dtype=dtype,
            )

        def layer_false() -> ArrayLike:
            return jnp.asarray(
                radius_hydrostatic_equilibrium[reference_index + 1]
                - (
                    (inv_rho[reference_index] + inv_rho[reference_index + 1])
                    * (pressures[reference_index + 1] - reference_pressure)
                    * coefficient
                    * 0.5
                ),
                dtype=dtype,
            )

        return jnp.asarray(
            lax.cond(reference_index <= n_layers - 3, layer_true, layer_false),
            dtype=dtype,
        )

    R0 = lax.cond(
        jnp.any(reference_layer_mask),
        compute_reference_radius_offset,
        lambda: jnp.asarray(0.0, dtype=dtype),
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

    rho = atmospheric_densities
    rho_work = jnp.asarray(rho, dtype=WORKING_JAX_DTYPE)
    opacities_work = jnp.asarray(opacities, dtype=WORKING_JAX_DTYPE)
    continuum_opacities_scattering_work = jnp.asarray(
        continuum_opacities_scattering, dtype=WORKING_JAX_DTYPE
    )
    weights_gauss_work = jnp.asarray(weights_gauss, dtype=WORKING_JAX_DTYPE)

    # Build adjacent-layer extinction pairs directly so we do not retain both
    # the full opacity*rho cube and a second paired copy at the same time.
    pair_continuum_transparencies = (
        continuum_opacities_scattering_work[:, :-1] * rho_work[:-1][None, :]
        + continuum_opacities_scattering_work[:, 1:] * rho_work[1:][None, :]
    )

    # Factor 1/2 is omitted because it cancels with effective planet area integration below.
    pair_transparencies = (
        opacities_work[..., :-1] * rho_work[:-1][None, None, None, :]
        + opacities_work[..., 1:] * rho_work[1:][None, None, None, :]
    )
    pair_transparencies = pair_transparencies.at[:, :, :1, :].add(
        pair_continuum_transparencies[None, :, None, :]
    )

    # Only the active shell-path block contributes; avoid building the padded
    # (n_layers, n_layers) matrix whose first row and column are identically zero.
    radius_squared = radius_hydrostatic_equilibrium**2.0
    impact_radius_squared = radius_squared[1:, None]
    upper_shell_boundaries = radius_squared[:-1][None, :]
    lower_shell_boundaries = radius_squared[1:][None, :]
    diff_s_active = _safe_sqrt_positive(
        upper_shell_boundaries - impact_radius_squared
    ) - _safe_sqrt_positive(lower_shell_boundaries - impact_radius_squared)
    diff_s_active = jnp.asarray(diff_s_active, dtype=WORKING_JAX_DTYPE)

    optical_depths_active = jnp.einsum(
        "gfsl,ml->gfsm", pair_transparencies, diff_s_active, optimize=True
    )
    transparencies = jnp.exp(
        -jnp.pad(optical_depths_active, ((0, 0), (0, 0), (0, 0), (1, 0)))
    )

    if weights_gauss.size == 1:
        transparencies = transparencies[0]
    else:
        transparencies = jnp.einsum(
            "gfsl,g->fsl", transparencies, weights_gauss_work, optimize=True
        )

    if transparencies.shape[1] == 1:
        transparencies = transparencies[:, 0, :]
    else:
        transparencies = jnp.prod(transparencies, axis=1)

    # Calculate transparencies (1 - transmittances)
    transparencies = 1.0 - transparencies

    # Annulus radius increments
    annulus_radius_increments = -jnp.diff(radius_hydrostatic_equilibrium)

    # Integrate effective area, 2 pi omitted:
    # - 2 cancels with 1/2 of average inverse mean free path above
    # - pi cancels when calculating the radius from the area below
    annulus_terms = (
        transparencies[:, 1:] * radius_hydrostatic_equilibrium[None, 1:]
        + transparencies[:, :-1] * radius_hydrostatic_equilibrium[None, :-1]
    )
    transit_radii = jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)

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

    rho = atmospheric_densities
    rho_work = jnp.asarray(rho, dtype=WORKING_JAX_DTYPE)
    opacities_work = jnp.asarray(opacities[:, :, :1, :], dtype=WORKING_JAX_DTYPE)
    continuum_opacities_scattering_work = jnp.asarray(
        continuum_opacities_scattering, dtype=WORKING_JAX_DTYPE
    )
    weights_gauss_work = jnp.asarray(weights_gauss, dtype=WORKING_JAX_DTYPE)

    pair_continuum_transparencies = (
        continuum_opacities_scattering_work[:, :-1] * rho_work[:-1][None, :]
        + continuum_opacities_scattering_work[:, 1:] * rho_work[1:][None, :]
    )

    # Calculate average mean free path between neighboring layers for later integration.
    # Factor 1/2 is omitted because it cancels with effective planet area integration below.
    pair_transparencies = (
        opacities_work[..., :-1] * rho_work[:-1][None, None, None, :]
        + opacities_work[..., 1:] * rho_work[1:][None, None, None, :]
        + pair_continuum_transparencies[None, :, None, :]
    )

    # Only the active shell-path block contributes; avoid building the padded
    # (n_layers, n_layers) matrix whose first row and column are identically zero.
    radius_squared = radius_hydrostatic_equilibrium**2.0
    impact_radius_squared = radius_squared[1:, None]
    upper_shell_boundaries = radius_squared[:-1][None, :]
    lower_shell_boundaries = radius_squared[1:][None, :]
    diff_s_active = _safe_sqrt_positive(
        upper_shell_boundaries - impact_radius_squared
    ) - _safe_sqrt_positive(lower_shell_boundaries - impact_radius_squared)
    diff_s_active = jnp.asarray(diff_s_active, dtype=WORKING_JAX_DTYPE)

    optical_depths_active = jnp.einsum(
        "gfsl,ml->gfsm", pair_transparencies, diff_s_active, optimize=True
    )
    transparencies = jnp.exp(
        -jnp.pad(optical_depths_active, ((0, 0), (0, 0), (0, 0), (1, 0)))
    )

    if weights_gauss.size == 1:
        transparencies = transparencies[0]
    else:
        transparencies = jnp.einsum(
            "gfsl,g->fsl", transparencies, weights_gauss_work, optimize=True
        )

    transparencies = transparencies[:, 0, :]

    # Calculate transparencies (1 - transmittances)
    transparencies = 1.0 - transparencies

    # Annulus radius increments
    annulus_radius_increments = -jnp.diff(radius_hydrostatic_equilibrium)

    # Integrate effective area, 2 pi omitted:
    # - 2 cancels with 1/2 of average inverse mean free path above
    # - pi cancels when calculating the radius from the area below
    annulus_terms = (
        transparencies[:, 1:] * radius_hydrostatic_equilibrium[None, 1:]
        + transparencies[:, :-1] * radius_hydrostatic_equilibrium[None, :-1]
    )
    transit_radii = jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)

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
    n_layers = opacities.shape[3]

    def _safe_sqrt_positive(values: ArrayLike) -> ArrayLike:
        positive_values = values > 0.0
        safe_values = jnp.where(
            positive_values,
            values,
            lax.stop_gradient(jnp.ones_like(values)),
        )
        return jnp.where(positive_values, jnp.sqrt(safe_values), 0.0)

    P0_cgs = reference_pressure * 1e6
    pressures_cgs = pressures * 1e6
    rho = (mean_molar_masses * cst.amu * pressures_cgs) / (cst.kB * temperatures)

    radius_hydrostatic_equilibrium = compute_radius_hydrostatic_equilibrium(
        pressures_cgs, reference_gravity, rho, P0_cgs, planet_radius, variable_gravity
    )
    radius_squared = radius_hydrostatic_equilibrium**2

    rho_work = jnp.asarray(rho, dtype=WORKING_JAX_DTYPE)
    opacities_work = jnp.asarray(opacities, dtype=WORKING_JAX_DTYPE)
    continuum_opacities_scattering_work = jnp.asarray(
        continuum_opacities_scattering, dtype=WORKING_JAX_DTYPE
    )
    weights_gauss_work = jnp.asarray(weights_gauss, dtype=WORKING_JAX_DTYPE)

    pair_opacity_times_rho = (
        opacities_work[..., :-1] * rho_work[:-1][None, None, None, :]
        + opacities_work[..., 1:] * rho_work[1:][None, None, None, :]
    )

    impact_radius_squared = radius_squared[1:, None]
    upper_shell_boundaries = radius_squared[:-1][None, :]
    lower_shell_boundaries = radius_squared[1:][None, :]
    diff_s_active = _safe_sqrt_positive(
        upper_shell_boundaries - impact_radius_squared
    ) - _safe_sqrt_positive(lower_shell_boundaries - impact_radius_squared)
    diff_s_active = jnp.asarray(diff_s_active, dtype=WORKING_JAX_DTYPE)

    gas_optical_depths_active = jnp.einsum(
        "gfsl,ml->gfsm", pair_opacity_times_rho, diff_s_active, optimize=True
    )

    if scattering_in_transmission:
        pair_continuum_times_rho = (
            continuum_opacities_scattering_work[:, :-1] * rho_work[:-1][None, :]
            + continuum_opacities_scattering_work[:, 1:] * rho_work[1:][None, :]
        )
        scattering_optical_depths_active = jnp.einsum(
            "fl,ml->fm", pair_continuum_times_rho, diff_s_active, optimize=True
        )
    else:
        scattering_optical_depths_active = jnp.zeros(
            (opacities.shape[1], n_layers - 1), dtype=opacities.dtype
        )

    # Only the two shells adjacent to a removed layer change, so precompute the
    # geometry once and apply a per-layer rank-1 optical-depth correction.
    layer_path_weights = jnp.pad(diff_s_active, ((0, 0), (1, 0))) + jnp.pad(
        diff_s_active, ((0, 0), (0, 1))
    )
    layer_path_weights_by_layer = jnp.swapaxes(layer_path_weights, 0, 1)
    weights_gauss_expanded = weights_gauss_work[:, None, None, None]
    annulus_radius_increments = (
        radius_hydrostatic_equilibrium[:-1] - radius_hydrostatic_equilibrium[1:]
    )
    top_of_atmosphere_transmission = jnp.ones(
        (opacities.shape[1], 1), dtype=WORKING_JAX_DTYPE
    )

    def _leave_one_out_contribution(i_leave_str_idx: ArrayLike) -> ArrayLike:
        path_by_layer = layer_path_weights_by_layer[i_leave_str_idx]
        rho_by_layer = rho_work[i_leave_str_idx]
        gas_opacity_times_rho = opacities_work[..., i_leave_str_idx] * rho_by_layer
        gas_optical_depths_leave_one_out = gas_optical_depths_active - (
            gas_opacity_times_rho[..., None] * path_by_layer
        )

        gas_transmission_active = jnp.exp(-gas_optical_depths_leave_one_out)
        gas_transmission_by_species = jnp.sum(
            gas_transmission_active * weights_gauss_expanded,
            axis=0,
        )
        total_transmission_active = jnp.prod(gas_transmission_by_species, axis=1)

        if scattering_in_transmission:
            continuum_opacity_times_rho = (
                continuum_opacities_scattering_work[:, i_leave_str_idx] * rho_by_layer
            )
            scattering_optical_depths_leave_one_out = (
                scattering_optical_depths_active
                - continuum_opacity_times_rho[:, None] * path_by_layer
            )
            total_transmission_active = total_transmission_active * jnp.exp(
                -scattering_optical_depths_leave_one_out
            )

        total_transmission = jnp.concatenate(
            (top_of_atmosphere_transmission, total_transmission_active),
            axis=1,
        )
        effective_area_fraction = 1.0 - total_transmission
        annulus_terms = (
            effective_area_fraction[:, 1:] * radius_hydrostatic_equilibrium[None, 1:]
            + effective_area_fraction[:, :-1]
            * radius_hydrostatic_equilibrium[None, :-1]
        )
        transit_radii_leave_one_out_squared = (
            jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)
            + radius_squared[-1]
        )
        return transit_radii_squared - transit_radii_leave_one_out_squared

    transmission_contribution_raw = lax.map(
        _leave_one_out_contribution,
        jnp.arange(n_layers, dtype=jnp.int32),
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
