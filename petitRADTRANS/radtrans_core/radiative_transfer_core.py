"""
JAX implementation of fortran_core.core
- rayleigh scattering moved to jax_rayleigh_scattering
"""

from functools import partial
from typing import Literal, Optional, cast, overload

import jax
import jax.numpy as jnp
from jax import custom_vjp, jit, lax, vmap
from jax.typing import ArrayLike
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.radtrans_core.physics_core as physics_core
import petitRADTRANS.radtrans_core.linalg as linalg
import petitRADTRANS.radtrans_core.ck_mixing_ffi as ck_mixing_ffi
from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

CST_PI = jnp.pi  # Or the specific precision from Fortran

TINIEST = jnp.finfo(FEAUTRIER_JAX_DTYPE).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(FEAUTRIER_JAX_DTYPE).max)
WORKING_TINIEST = jnp.finfo(WORKING_JAX_DTYPE).tiny

_UNROLL_SPECIES_LIMIT = 8

# Number of (frequency*layer) cells processed per chunk in the pure-JAX c-k
# mixing kernel. The per-species combination materialises several (chunk, n_g**2)
# intermediates; chunking caps their peak size so the full-resolution
# (n_freq*n_layers, n_g**2) sort buffers are never resident at once. A
# non-positive value (or a chunk >= the number of cells) disables chunking.
#
# Default OFF. GPU benchmarking showed chunking lowers peak memory by ~1.7-2.3x
# (e.g. xl from 6.4 to 3.7 GB) but costs ~5-36% runtime, and the remaining peak
# is dominated by the input opacity array and its transpose (which this chunking
# does not touch), not the sort buffers it caps. It is therefore an opt-in tool
# for memory-constrained runs (large frequency grids / heavily vmapped
# retrievals near the device memory limit) rather than a default. Enable via the
# ``ck_chunk_size`` argument (e.g. 16384-65536); larger chunks regress speed
# less but cap memory less.
DEFAULT_CK_CHUNK_SIZE = 0

# Single source of truth for the Feautrier chunking defaults. Both values are
# static (compile-time) arguments of the flux kernels: every distinct
# (chunk_feautrier, feautrier_chunk_size) pair compiles its own kernel, and the
# chunk size also selects the linear solver variant (see linalg._feautrier_use_thomas).
# Entry points keep their own fallbacks (memory constraints differ between
# forward modelling and retrievals).
DEFAULT_CHUNK_FEAUTRIER = True
DEFAULT_FEAUTRIER_CHUNK_SIZE = 1536


def resolve_feautrier_chunking(
    chunk_feautrier=None,
    feautrier_chunk_size=None,
    fallback_chunk_feautrier=DEFAULT_CHUNK_FEAUTRIER,
    fallback_chunk_size=DEFAULT_FEAUTRIER_CHUNK_SIZE,
):
    """Resolve the Feautrier chunking settings.

    Precedence: explicit argument > caller-supplied fallback. Values are
    coerced to plain bool/int so the static jit cache keys do not split on
    equivalent representations (e.g. 1536 vs numpy.int64(1536)).
    """
    if chunk_feautrier is None:
        chunk_feautrier = fallback_chunk_feautrier

    if feautrier_chunk_size is None:
        feautrier_chunk_size = fallback_chunk_size

    return bool(chunk_feautrier), int(feautrier_chunk_size)


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


@jit(static_argnames=("input_layout", "ck_chunk_size"))
def _combine_ck_opacities_jax(
    opacities_fortran_order: ArrayLike,
    g_gauss: ArrayLike,
    weights_gauss: Optional[ArrayLike] = None,
    line_species_mass_fractions: Optional[ArrayLike] = None,
    continuum_opacities: Optional[ArrayLike] = None,
    g_combination_weights: Optional[ArrayLike] = None,
    g_combination_weights_sum: Optional[ArrayLike] = None,
    input_layout: str = "ng_f_s_l",
    ck_chunk_size: int = DEFAULT_CK_CHUNK_SIZE,
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
    ck_chunk_size : int, optional
        Number of (frequency*layer) cells combined per chunk. The per-species
        combination materialises several (chunk, n_g**2) intermediates; chunking
        bounds their peak size. A non-positive value, or a chunk that is not
        smaller than the number of cells, disables chunking and processes every
        cell at once. Each (f, l) cell is combined independently, so chunking is
        numerically exact.

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
    # C2 + C3: Vectorized threshold & species loop, evaluated on a chunk of
    # (frequency*layer) cells. Each cell is combined independently, so this is
    # applied either to every cell at once or one chunk at a time (below).
    # ------------------------------------------------------------------

    def _combine_species_chunk(
        opacities_chunk: ArrayLike,  # (chunk, n_species, n_g)
        abundance_weights_chunk: ArrayLike,  # (chunk, n_species)
        continuum_chunk: ArrayLike,  # (chunk,)
    ) -> ArrayLike:
        """Mix all species for one chunk of cells → (chunk, n_g)."""

        def _weight_species(species_index: int) -> ArrayLike:
            return opacities_chunk[:, species_index, :] * abundance_weights_chunk[
                :, species_index : species_index + 1
            ]

        # Initialize with first species + continuum  (chunk, n_g)
        current_chunk = _weight_species(0) + continuum_chunk[:, None]

        if n_species > 1:
            # C2: Vectorized threshold matching Fortran structure.
            # Fortran uses g=1 (first/weakest g-point) to compute the threshold,
            # then checks g=n_g (last/strongest) against it. The continuum has
            # already been added to species 0 in the Fortran input, so fold it
            # into the threshold probe here as well.
            threshold_input = (
                opacities_chunk[:, :, 0] * abundance_weights_chunk
            ).at[:, 0].add(continuum_chunk)
            threshold = jnp.max(threshold_input, axis=1) * 1e-2  # (chunk,)
            active_mask = (
                opacities_chunk[:, :, -1] * abundance_weights_chunk
            ) >= threshold[:, None]  # (chunk, n_species)

            # Batched species loop — no vmap, single sort kernel per species.
            for s in range(1, n_species):
                spec_s = _weight_species(s)  # (chunk, n_g)
                combined = _perform_combination_batched(
                    current_chunk, spec_s
                )  # (chunk, n_g)
                current_chunk = jnp.where(
                    active_mask[:, s : s + 1], combined, current_chunk
                )

        return current_chunk

    # ------------------------------------------------------------------
    # Frequency/layer chunking. The per-species combination materialises
    # several (cells, n_g**2) sort buffers; processing the cell axis in tiles
    # keeps only one tile's buffers resident at a time. The final tile's start
    # is clamped so every tile reads a full window (the overlap recomputes
    # identical per-cell values and is overwritten on reassembly), mirroring the
    # Feautrier chunking and avoiding a padded full-resolution copy.
    # ------------------------------------------------------------------
    n_fl = opacities_flat.shape[0]

    if ck_chunk_size <= 0 or ck_chunk_size >= n_fl:
        current = _combine_species_chunk(
            opacities_flat, abundance_weights_flat, continuum_flat
        )
    else:
        chunk = ck_chunk_size
        n_chunks = (n_fl + chunk - 1) // chunk
        chunk_starts = jnp.minimum(jnp.arange(n_chunks) * chunk, n_fl - chunk)

        def _chunk_body(
            _: None, start: ArrayLike
        ) -> tuple[None, ArrayLike]:
            opacities_chunk = lax.dynamic_slice(
                opacities_flat, (start, 0, 0), (chunk, n_species, n_g)
            )
            abundance_weights_chunk = lax.dynamic_slice(
                abundance_weights_flat, (start, 0), (chunk, n_species)
            )
            continuum_chunk = lax.dynamic_slice(continuum_flat, (start,), (chunk,))
            return None, _combine_species_chunk(
                opacities_chunk, abundance_weights_chunk, continuum_chunk
            )

        # current_chunks: (n_chunks, chunk, n_g) — only n_g wide, so this stacked
        # output is cheap; the large (chunk, n_g**2) buffers live and die inside
        # each scan step.
        _, current_chunks = lax.scan(_chunk_body, None, chunk_starts)

        current = jnp.zeros((n_fl, n_g), dtype=current_chunks.dtype)

        def _place(i: ArrayLike, acc: ArrayLike) -> ArrayLike:
            return lax.dynamic_update_slice_in_dim(
                acc, current_chunks[i], chunk_starts[i], axis=0
            )

        current = lax.fori_loop(0, n_chunks, _place, current)

    return _finalize_ck_opacity_shape(
        current,
        n_frequencies,
        n_layers,
        n_g,
    )


# ---------------------------------------------------------------------------
# Fortran CPU forward path + custom VJP wrapper
# ---------------------------------------------------------------------------


def _use_ck_mixing_ffi() -> bool:
    """Use the native FFI mixing kernel only on CPU backends."""
    return jax.default_backend() == "cpu"


@partial(custom_vjp, nondiff_argnums=(7,))
def _combine_ck_opacities_ffi(
    opacities_fortran_order: ArrayLike,
    g_gauss: ArrayLike,
    weights_gauss: ArrayLike,
    line_species_mass_fractions: ArrayLike,
    continuum_opacities: ArrayLike,
    g_combination_weights: ArrayLike,
    g_combination_weights_sum: ArrayLike,
    input_layout: str,
) -> ArrayLike:
    """Correlated-k mixing: native FFI forward, JAX backward.

    The forward pass runs the batched C++ kernel
    (petitRADTRANS/cpp_src/prt_ck_mixing_ffi.cc) as a typed XLA custom call;
    it reproduces the random-overlap mixing of the original Fortran
    implementation, parallelized over (frequency, layer) cells on the XLA
    intra-op thread pool.
    """
    (
        opacities_flat,
        abundance_weights_flat,
        continuum_flat,
        n_frequencies,
        n_layers,
        _,
        n_g,
    ) = _prepare_ck_opacity_inputs(
        opacities_fortran_order,
        line_species_mass_fractions,
        continuum_opacities,
        input_layout=input_layout,
    )
    weighted_opacities = opacities_flat * abundance_weights_flat[:, :, None]

    mixed_flat = ck_mixing_ffi.combine_ck_mixing_ffi(
        weighted_opacities,
        continuum_flat,
        g_gauss,
        weights_gauss,
    )

    return _finalize_ck_opacity_shape(mixed_flat, n_frequencies, n_layers, n_g)


def _combine_ck_ffi_fwd(
    opacities_fortran_order,
    g_gauss,
    weights_gauss,
    line_species_mass_fractions,
    continuum_opacities,
    g_combination_weights,
    g_combination_weights_sum,
    input_layout,
):
    result = _combine_ck_opacities_ffi(
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


def _combine_ck_ffi_bwd(input_layout, residuals, g):
    _, vjp_fn = jax.vjp(
        partial(_combine_ck_opacities_jax, input_layout=input_layout),
        *residuals,
    )
    return vjp_fn(g)


_combine_ck_opacities_ffi.defvjp(_combine_ck_ffi_fwd, _combine_ck_ffi_bwd)


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
    ck_chunk_size: int = DEFAULT_CK_CHUNK_SIZE,
) -> ArrayLike:
    """Combine correlated-k opacities across species.

    Uses the native FFI kernel on CPU for the forward pass (a typed XLA
    custom call) and the JAX implementation for reverse-mode
    differentiation. Non-CPU backends use the pure JAX implementation for both
    forward and backward execution.

    ``ck_chunk_size`` only affects the pure-JAX path (non-CPU backends, or the
    reverse-mode VJP of the FFI forward pass): it caps the peak memory of the
    per-species combination by processing the (frequency, layer) cells in
    chunks. A non-positive value (the default) disables chunking.
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
        raise ValueError("weights_gauss is required for the c-k mixing kernel")

    if line_species_mass_fractions is None or continuum_opacities is None:
        raise ValueError(
            "line_species_mass_fractions and continuum_opacities are required"
        )

    if _use_ck_mixing_ffi():
        return _combine_ck_opacities_ffi(
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
        ck_chunk_size=ck_chunk_size,
    )


def _compute_ck_angle_transmission(
    optical_depths: ArrayLike,
    weights_gauss: ArrayLike,
    mu: ArrayLike,
) -> ArrayLike:
    """Total transmission along one emission angle.

    The correlated-k transmission is the g-quadrature average of
    ``exp(-tau / mu)``, multiplied over species (random-overlap assumption).

    Parameters
    ----------
    optical_depths : jax.Array, shape (n_g, n_frequencies, n_species, n_layers)
    weights_gauss : jax.Array, shape (n_g,)
    mu : jax.Array, scalar
        Cosine of the emission angle.

    Returns
    -------
    jax.Array, shape (n_frequencies, n_layers)
    """
    transm_mu = jnp.exp(-optical_depths / mu)
    mean_transm = jnp.sum(transm_mu * weights_gauss[:, None, None, None], axis=0)

    return jnp.prod(mean_transm, axis=1)


def _compute_ck_layer_emission_terms(
    planck: ArrayLike,
    transm_all: ArrayLike,
) -> ArrayLike:
    """Trapezoidal per-layer emission terms, shared by the flux and contribution.

    Parameters
    ----------
    planck : jax.Array, shape (n_frequencies, n_layers)
    transm_all : jax.Array, shape (n_frequencies, n_layers)

    Returns
    -------
    jax.Array, shape (n_frequencies, n_layers - 1)
    """
    return (planck[:, :-1] + planck[:, 1:]) * (
        transm_all[:, :-1] - transm_all[:, 1:]
    )


def _compute_ck_angle_flux(
    layer_terms: ArrayLike,
    planck: ArrayLike,
    transm_all: ArrayLike,
) -> ArrayLike:
    """Emergent flux along one angle, before the ``mu * mu_weight`` weighting.

    The trailing term closes the integral with the deepest layer. Note the
    original code indexed column ``n_layers`` here, one past the last valid
    column; that worked only because JAX clamps out-of-bounds static indices to
    the last element, so it is written explicitly as ``-1``.

    Returns
    -------
    jax.Array, shape (n_frequencies,)
    """
    return (
        jnp.sum(layer_terms * 0.5, axis=1) + planck[:, -1] * transm_all[:, -1]
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

    # Vectorize over emission angles using vmap: compute per-angle contributions then sum.
    r = physics_core.compute_planck_function(temperatures, frequencies)

    def _angle_contrib(mu: ArrayLike, mu_w: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        transm_all = _compute_ck_angle_transmission(optical_depths, weights_gauss, mu)
        layer_terms = _compute_ck_layer_emission_terms(r, transm_all)

        # Radiative transport contribution for this angle
        flux_mu = _compute_ck_angle_flux(layer_terms, r, transm_all)

        contr_local = jnp.zeros(struct_shape)
        contr_local = contr_local.at[:, :-1].set(layer_terms * mu * mu_w)
        contr_local = contr_local.at[:, -1].set(
            2.0 * r[:, -1] * transm_all[:, -1] * mu * mu_w
        )

        flux_contrib = flux_mu * 0.5 * mu * mu_w
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
    # Floor the denominator: a frequency column that sums to zero (no emission
    # anywhere, e.g. a fully opaque column) would otherwise normalize to NaN.
    contr_em_sum = jnp.sum(contr_em, axis=0)
    contr_em = contr_em / jnp.maximum(
        contr_em_sum, jnp.finfo(contr_em.dtype).tiny
    )
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

        # Vectorize over emission angles using vmap: compute per-angle flux then sum.
        r = physics_core.compute_planck_function(temperatures, frequencies)

        def _angle_contrib(mu: ArrayLike, mu_w: ArrayLike) -> ArrayLike:
            transm_all = _compute_ck_angle_transmission(
                optical_depths, weights_gauss, mu
            )
            layer_terms = _compute_ck_layer_emission_terms(r, transm_all)
            flux_mu = _compute_ck_angle_flux(layer_terms, r, transm_all)

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
    tn, tn1, tn2, tn3 : jax.Array, shape (n_layers, batch)
        Consecutive source-function iterates ordered from newest to oldest,
        in layer-major layout.
    current_source : jax.Array, shape (n_layers, batch)
        Non-accelerated source update to keep when the Ng solve is ill-conditioned.

    Returns
    -------
    jax.Array, shape (n_layers, batch)
        Accelerated source function, or ``current_source`` for columns that fail
        the numerical safeguards.
    """
    Q1 = tn - 2.0 * tn1 + tn2
    Q2 = tn - tn1 - tn2 + tn3
    Q3 = tn - tn1

    # Zero boundary on first layer
    Q1 = Q1.at[0, :].set(0.0)
    Q2 = Q2.at[0, :].set(0.0)
    Q3 = Q3.at[0, :].set(0.0)

    # 5 explicit dot products instead of a full 3x3 Gram matmul.
    # Only G[0,0], G[1,0], G[1,1], G[0,2], G[1,2] are used; the other
    # four entries of the symmetric matrix are never read.
    A1 = jnp.minimum(jnp.einsum("lb,lb->b", Q1, Q1), HUGE_SQRT)
    A2 = jnp.minimum(jnp.einsum("lb,lb->b", Q2, Q1), HUGE_SQRT)
    B2 = jnp.minimum(jnp.einsum("lb,lb->b", Q2, Q2), HUGE_SQRT)
    C1 = jnp.minimum(jnp.einsum("lb,lb->b", Q1, Q3), HUGE_SQRT)
    C2 = jnp.minimum(jnp.einsum("lb,lb->b", Q2, Q3), HUGE_SQRT)

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
        (1.0 - a_ng - b_ng)[None, :] * tn + a_ng[None, :] * tn1 + b_ng[None, :] * tn2,
        0.0,
    )  # (l, batch)

    # In Fortran m_approximate_ng_source, when the early `return` fires, the
    # output argument `source` (5th arg) keeps whatever value it had on entry —
    return jnp.where(skip[None, :], current_source, accelerated)


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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, dict[str, ArrayLike]]: ...


@jit(
    static_argnames=(
        "irradiation_geometry_str",
        "adaptive_feautrier_iterations",
        "return_contribution",
        "return_solver_diagnostics",
        "iter_scat_max",
        "thomas_min_batch",
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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
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
    # All layer-indexed arrays below use the layer-major ("lm") layout
    # (n_layers, n_g * n_freq_bins): every per-layer step of the Feautrier
    # sweeps then reads and writes contiguous rows, and the tridiagonal
    # solver consumes and emits the source function without any transposes
    # inside the scattering iteration.
    n_g, n_freq_bins, n_layers = photon_destruction_probabilities_in.shape
    n_angles = emission_cos_angles.shape[0]
    n_gf = n_g * n_freq_bins
    pi_4 = 4.0 * jnp.pi

    surf_refl_2_gf = jnp.broadcast_to(
        2.0 * reflectances[None, :], (n_g, n_freq_bins)
    ).reshape(n_gf)
    angle_weight_a = emission_cos_angles_weights
    mu_weight_a = emission_cos_angles * emission_cos_angles_weights

    optical_depths_lm = optical_depths_in.reshape(n_gf, n_layers).T  # (l, gf)

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
    pdp_lm = jnp.maximum(
        photon_destruction_probabilities_in.reshape(n_gf, n_layers).T, _pdp_floor
    )  # (l, gf)

    # ------------------------------------------------------------------
    # 3. Stellar attenuation  J_star_ini_lm (l, gf)
    # ------------------------------------------------------------------
    if geom > 0:
        stellar_intensity_abs_gf = jnp.broadcast_to(
            jnp.abs(stellar_intensity)[None, :], (n_g, n_freq_bins)
        ).reshape(n_gf)

        # Accumulate each angle contribution in sequence to avoid a full
        # (layer, gf, angle) exponential cube.
        def _accumulate_stellar_angle(
            carry: tuple[ArrayLike, ArrayLike],
            angle_inputs: tuple[ArrayLike, ArrayLike, ArrayLike],
        ) -> tuple[tuple[ArrayLike, ArrayLike], None]:
            j_star_accum_lm, surface_reflection_accum_gf = carry
            mu, angle_weight, mu_weight = angle_inputs

            exp_over_mu_lm = jnp.exp(-optical_depths_lm / mu)
            j_star_accum_lm = j_star_accum_lm + angle_weight * exp_over_mu_lm
            surface_reflection_accum_gf = (
                surface_reflection_accum_gf
                + mu_weight * exp_over_mu_lm[n_layers - 1]
            )

            return (
                j_star_accum_lm,
                surface_reflection_accum_gf,
            ), None

        (j_star_angle_sum_lm, surface_reflection_angle_sum_gf), _ = lax.scan(
            _accumulate_stellar_angle,
            (
                jnp.zeros_like(optical_depths_lm),
                jnp.zeros(n_gf, dtype=optical_depths_lm.dtype),
            ),
            (
                emission_cos_angles,
                emission_cos_angles_weights,
                mu_weight_a,
            ),
        )

        J_star_ini_lm = (
            0.5 * geom * stellar_intensity_abs_gf[None, :] * j_star_angle_sum_lm
        )

        I_surface_reflection_gf = (
            surf_refl_2_gf
            * geom
            * stellar_intensity_abs_gf
            * surface_reflection_angle_sum_gf
        )

    else:  # non-isotropic
        stellar_intensity_gf = jnp.broadcast_to(
            stellar_intensity[None, :], (n_g, n_freq_bins)
        ).reshape(n_gf)
        reflectances_gf = jnp.broadcast_to(
            reflectances[None, :], (n_g, n_freq_bins)
        ).reshape(n_gf)
        exp_lm = jnp.exp(-optical_depths_lm / star_irradiation_cos_angle)
        J_star_ini_lm = jnp.abs(stellar_intensity_gf[None, :] / 4.0 * exp_lm)
        I_surface_reflection_gf = (
            reflectances_gf
            * J_star_ini_lm[n_layers - 1]
            * 4.0
            * star_irradiation_cos_angle
        )

    # ------------------------------------------------------------------
    # 4. Planck function
    # ------------------------------------------------------------------
    planck_integral_lf = physics_core.compute_planck_function_integral(
        temperatures, frequencies_bin_edges
    )  # (l, f)
    planck_lm = jnp.broadcast_to(
        planck_integral_lf[:, None, :], (n_layers, n_g, n_freq_bins)
    ).reshape(n_layers, n_gf)  # (l, gf)

    # ------------------------------------------------------------------
    # 5. Surface boundary terms
    # ------------------------------------------------------------------
    emissivities_gf = jnp.broadcast_to(
        emissivities[None, :], (n_g, n_freq_bins)
    ).reshape(n_gf)
    I_surface_no_scatter_gf = (
        emissivities_gf * planck_lm[n_layers - 1] + I_surface_reflection_gf
    )  # (gf,)

    # ------------------------------------------------------------------
    # 6. Per-(g,f) minimum iteration count  (Fix 2)
    # ------------------------------------------------------------------
    if adaptive_feautrier_iterations:
        n_iter_min_gf = jnp.clip(
            1 + jnp.max(1.0 / pdp_lm, axis=0), 4, 9  # max over layers → (gf,)
        ).astype(jnp.int32)
    else:
        n_iter_min_gf = jnp.full(n_gf, 9, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # 7. initial carry source = bare Planck (matches Fortran's
    #    `r = planck(:,i)` set before the iteration loop).
    # ------------------------------------------------------------------
    # planck_lm doubles as the initial source iterate.

    # ------------------------------------------------------------------
    # 8. Delta-taus
    # ------------------------------------------------------------------
    # Floor |delta-tau| at sqrt(tiny), not tiny. The Feautrier coefficients
    # divide by these (f1 = clip(mu/dtau, ...)); with a floor of `tiny` the
    # clipped forward value is finite but its gradient, -mu/dtau**2, overflows
    # because dtau**2 underflows to 0 -- and the clip then turns that into
    # 0*inf = NaN, which propagated into the whole log-posterior gradient. A
    # sqrt(tiny) floor keeps dtau**2 in the normal range (finite gradient); such
    # steps are physically zero and clipped regardless, so the forward is
    # unchanged.
    _dtau_floor = jnp.sqrt(TINIEST)
    dtau1_lm = optical_depths_lm[1:] - optical_depths_lm[:-1]  # (l-1, gf)
    dtau1_lm = jnp.where(jnp.abs(dtau1_lm) < _dtau_floor, _dtau_floor, dtau1_lm)

    dtau2_lm = optical_depths_lm[2:] - optical_depths_lm[:-2]  # (l-2, gf)
    dtau2_lm = jnp.where(jnp.abs(dtau2_lm) < _dtau_floor, _dtau_floor, dtau2_lm)

    # ------------------------------------------------------------------
    # 9-12. Feautrier system preparation and inner solve
    # ------------------------------------------------------------------
    # Coefficient construction, Thomas pre-factoring and the Lambda diagonal
    # are fused into a single scan over layers (CPU), so the
    # (gf, n_angles, n_layers)-sized tridiagonal coefficient arrays are never
    # materialised. ``f_surface_ga`` is the clipped surface stencil
    # mu/dtau1[n_layers-2] used for the emergent-intensity derivative.
    solver_state, lambda_loc_lm, f_surface_ga = linalg.prepare_feautrier_system_lm(
        dtau1_lm,
        dtau2_lm,
        emission_cos_angles,
        angle_weight_a,
        n_gf,
        n_angles,
        n_layers,
        thomas_min_batch=thomas_min_batch,
    )

    # loop-invariant ALI intermediates precomputed once.
    one_minus_pdp_lm = 1.0 - pdp_lm  # (l, gf)
    pdp_times_planck_lm = pdp_lm * planck_lm  # (l, gf)
    # Solution B: rewrite the ALI denominator to avoid catastrophic
    # cancellation near 1.0.  The identity
    #   1 - (1 - eps) * lambda  =  eps + (1 - eps) * (1 - lambda)
    # keeps both addends positive and O(eps) or O(1-lambda), so neither
    # term is lost to rounding regardless of the floating-point precision.
    ali_denom_lm = pdp_lm + one_minus_pdp_lm * (1.0 - lambda_loc_lm)  # (l, gf)
    ali_inv_denom_lm = 1.0 / jnp.maximum(ali_denom_lm, TINIEST)  # (l, gf)

    def _feautrier_solve(
        source_lm: ArrayLike,
        I_plus_surface_ga: ArrayLike,
        I_minus_boundary_gf: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Returns J_bol_lm (l, gf), I_H_top (gf, a), new_I_plus_surface (gf, a).
        """
        r_surf_ga = 0.5 * (
            I_plus_surface_ga + I_minus_boundary_gf[:, None]
        )  # (gf, a)

        I_H_top, x_last_ga, x_nm2_ga, J_bol_lm = linalg.solve_feautrier_system_lm(
            solver_state,
            source_lm,
            r_surf_ga,
            angle_weight_a,
            n_gf,
            n_angles,
            n_layers,
            thomas_min_batch=thomas_min_batch,
        )

        d1_surf = f_surface_ga * (x_last_ga - x_nm2_ga)
        new_I_plus_surface = x_last_ga - d1_surf  # (gf, a)

        return J_bol_lm, I_H_top, new_I_plus_surface

    # ------------------------------------------------------------------
    # 13. Scattering iteration control.
    #
    # Reverse-mode autodiff cannot differentiate the original data-dependent
    # while-loop. Keep the adaptive path on a static scan budget, but split
    # out a dedicated fixed-iteration kernel and a conservative direct solve
    # for columns with no atmospheric scattering and a non-reflective lower
    # boundary.
    # ------------------------------------------------------------------

    zero_source_lm = jnp.zeros((n_layers, n_gf))
    initial_I_plus_surface_ga = jnp.zeros((n_gf, n_angles))
    initial_flux_gf = jnp.zeros(n_gf)
    initial_I_H_top = jnp.zeros((n_gf, n_angles))
    initial_I_minus_gf = (
        jnp.zeros(n_gf)
        if return_contribution
        else jnp.zeros((1,), dtype=planck_lm.dtype)
    )
    configured_iter_budget = jnp.array(iter_scat_max + 1, dtype=jnp.int32)
    fixed_iteration_count = min(iter_scat_max + 1, 9)

    no_scattering_pdp_tol = _pdp_floor
    no_atmospheric_scattering = jnp.all(one_minus_pdp_lm <= no_scattering_pdp_tol)
    non_reflective_lower_boundary = jnp.all(jnp.abs(reflectances) <= TINIEST)
    use_direct_no_scattering_solve = (
        no_atmospheric_scattering & non_reflective_lower_boundary
    )

    def _surface_reflection(I_plus_ga: ArrayLike) -> ArrayLike:
        return surf_refl_2_gf * jnp.sum(
            I_plus_ga * mu_weight_a[None, :], axis=1
        )  # (gf,)

    def _advance_state(
        state: tuple[ArrayLike, ...],
        iter_idx: ArrayLike,
    ) -> tuple[tuple[ArrayLike, ...], ArrayLike]:
        (
            current_source_lm,
            current_I_plus_surface_ga,
            current_flux_gf_prev,
            current_I_minus_gf,
            _current_I_H_top,
            current_hist0_lm,
            current_hist1_lm,
            current_hist2_lm,
            current_hist3_lm,
            _current_all_converged,
            _solve_count,
            _noop_count,
        ) = state

        I_minus_gf_new = I_surface_no_scatter_gf + _surface_reflection(
            current_I_plus_surface_ga
        )
        J_bol_lm, I_H_top, new_I_plus_surface = _feautrier_solve(
            current_source_lm,
            current_I_plus_surface_ga,
            I_minus_gf_new,
        )
        source_ali = (
            pdp_times_planck_lm
            + one_minus_pdp_lm
            * (J_star_ini_lm + J_bol_lm - lambda_loc_lm * current_source_lm)
        ) * ali_inv_denom_lm

        slot_index = iter_idx % 4

        def _slot0(_: None):
            next_hist0 = source_ali
            return (
                next_hist0,
                current_hist1_lm,
                current_hist2_lm,
                current_hist3_lm,
                next_hist0,
                current_hist3_lm,
                current_hist2_lm,
                current_hist1_lm,
            )

        def _slot1(_: None):
            next_hist1 = source_ali
            return (
                current_hist0_lm,
                next_hist1,
                current_hist2_lm,
                current_hist3_lm,
                next_hist1,
                current_hist0_lm,
                current_hist3_lm,
                current_hist2_lm,
            )

        def _slot2(_: None):
            next_hist2 = source_ali
            return (
                current_hist0_lm,
                current_hist1_lm,
                next_hist2,
                current_hist3_lm,
                next_hist2,
                current_hist1_lm,
                current_hist0_lm,
                current_hist3_lm,
            )

        def _slot3(_: None):
            next_hist3 = source_ali
            return (
                current_hist0_lm,
                current_hist1_lm,
                current_hist2_lm,
                next_hist3,
                next_hist3,
                current_hist2_lm,
                current_hist1_lm,
                current_hist0_lm,
            )

        # When iter_idx is a Python int (fully unrolled fixed-iteration path),
        # the slot rotation and the Ng trigger resolve at trace time, so the
        # history bookkeeping becomes static dataflow with no buffer shuffling.
        static_iter = isinstance(iter_idx, int)
        slot_branches = (_slot0, _slot1, _slot2, _slot3)
        if static_iter:
            slot_outputs = slot_branches[slot_index](None)
        else:
            slot_outputs = lax.switch(slot_index, slot_branches, operand=None)
        (
            next_hist0_lm,
            next_hist1_lm,
            next_hist2_lm,
            next_hist3_lm,
            tn,
            tn1,
            tn2,
            tn3,
        ) = slot_outputs

        do_ng = (iter_idx > 0) & ((iter_idx + 1) % 4 == 0)

        def _apply_ng(_: None) -> ArrayLike:
            return _ng_accelerated_batched(tn, tn1, tn2, tn3, source_ali)

        if static_iter:
            next_source_lm = _apply_ng(None) if do_ng else source_ali
        else:
            next_source_lm = lax.cond(
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
        next_source_lm = source_ali + lax.stop_gradient(next_source_lm - source_ali)

        flux_gf_cur = -jnp.einsum("ba,a->b", I_H_top, mu_weight_a)  # (gf,)
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
            next_source_lm,
            new_I_plus_surface,
            flux_gf_cur,
            next_I_minus_gf,
            I_H_top,
            next_hist0_lm,
            next_hist1_lm,
            next_hist2_lm,
            next_hist3_lm,
            _current_all_converged,
            _solve_count,
            _noop_count,
        )
        return next_state, conv_val_gf

    initial_carry = (
        planck_lm,
        initial_I_plus_surface_ga,
        initial_flux_gf,
        initial_I_minus_gf,
        initial_I_H_top,
        zero_source_lm,
        zero_source_lm,
        zero_source_lm,
        zero_source_lm,
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
            planck_lm,
            initial_I_plus_surface_ga,
            direct_I_minus_gf,
        )
        flux_gf_direct = -jnp.einsum("ba,a->b", direct_I_H_top, mu_weight_a)
        final_I_minus_gf = (
            direct_I_minus_gf if return_contribution else initial_I_minus_gf
        )

        return (
            planck_lm,
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
        # The iteration count is a static Python int, so unroll the loop at
        # trace time: the per-iteration history-slot rotation and Ng trigger
        # in _advance_state then fold into static dataflow (no lax.switch /
        # lax.cond buffer shuffling inside a loop body).
        final_carry = initial_carry
        for iter_idx in range(fixed_iteration_count):
            next_state, _ = _advance_state(final_carry, iter_idx)
            final_carry = (
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
                final_carry[10] + jnp.array(1, dtype=jnp.int32),
                final_carry[11],
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
        final_source_lm,
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
    flux_f = pi_4 * jnp.einsum(
        "gf,g->f", flux_gf_final.reshape(n_g, n_freq_bins), weights_gauss
    )
    angle_dependent_intensities = 2.0 * jnp.einsum(
        "gfa,g->fa", I_H_top.reshape(n_g, n_freq_bins, n_angles), weights_gauss
    )  # (f, a)

    # ------------------------------------------------------------------
    # 15. Contribution function (optional)
    # ------------------------------------------------------------------
    def compute_contribution() -> ArrayLike:
        weights_gauss_work = jnp.asarray(weights_gauss, dtype=WORKING_JAX_DTYPE)
        final_source_work = jnp.asarray(
            final_source_lm.reshape(n_layers, n_g, n_freq_bins),
            dtype=WORKING_JAX_DTYPE,
        )
        i_minus_work = jnp.asarray(
            I_minus_gf_final.reshape(n_g, n_freq_bins), dtype=WORKING_JAX_DTYPE
        )
        optical_depths_work = jnp.asarray(optical_depths_in, dtype=WORKING_JAX_DTYPE)
        emission_cos_angles_work = jnp.asarray(
            emission_cos_angles, dtype=WORKING_JAX_DTYPE
        )
        emission_cos_angles_weights_work = jnp.asarray(
            emission_cos_angles_weights, dtype=WORKING_JAX_DTYPE
        )

        # G-integrated source: (l, f)
        r_src_lf = jnp.einsum("g,lgf->lf", weights_gauss_work, final_source_work)

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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
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
        "thomas_min_batch",
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
    thomas_min_batch: int = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
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
    thomas_min_batch : int, optional
        Batch size (``n_gf * n_angles``) above which the Thomas-factorisation
        solver path is used over PCR on GPU. See :func:`linalg._feautrier_use_thomas`.

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
            thomas_min_batch=thomas_min_batch,
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
                    thomas_min_batch=thomas_min_batch,
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
                    thomas_min_batch=thomas_min_batch,
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
                        thomas_min_batch=thomas_min_batch,
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
                        thomas_min_batch=thomas_min_batch,
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
    continuum_opacities_scattering : jax.Array, optional
        Shape (n_g or 1, n_frequencies, n_layers) or (n_frequencies, n_layers).
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

    # Calculate optical depths
    optical_depths_active = jnp.einsum(
        "gfsl,ml->gfsm", pair_transparencies, diff_s_active, optimize=True
    )
    # Calculate transmittances (exp(-optical_depths)
    transparencies = jnp.exp(
        -jnp.pad(optical_depths_active, ((0, 0), (0, 0), (0, 0), (1, 0)))
    )

    if weights_gauss.size == 1:
        transparencies = transparencies[0] # (frequencies, species, layers)
    else:
        # Integrate over correlated-k's g-coordinate
        transparencies = jnp.einsum(
            "gfsl,g->fsl", transparencies, weights_gauss_work, optimize=True # (frequencies, species, layers)
        )

    # get rid of species dimension
    # remove axis 1 (species) if only single species
    if transparencies.shape[1] == 1:
        transparencies = transparencies[:, 0, :] # (frequencies, layers)
    # combine transmission of several species (exp(-tau1) * exp(-tau2)...)
    else:
        transparencies = jnp.prod(transparencies, axis=1)  # (frequencies, layers)

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
    transit_radii = jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1) # (frequencies)

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
    continuum_opacities_scattering: ArrayLike,  # (n_frequencies, n_layers)
    scattering_in_transmission: bool = True,  # boolean
    variable_gravity: bool = True,  # boolean
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate normalized layer contributions to the transmission spectrum.

    The baseline (all-layers) transit radii are computed internally with the
    same working-precision numerics as the leave-one-out evaluations. Using the
    externally computed transit radii from the main transmission path would add
    a constant, layer-independent offset (main path vs. this function's working
    precision) to every leave-one-out difference; that offset dwarfs the true
    per-layer signal and flattens the normalized contribution toward
    ``1 / n_layers``.

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

    def _transit_radii_squared_from_optical_depths(
        gas_optical_depths: ArrayLike,
        scattering_optical_depths: ArrayLike,
    ) -> ArrayLike:
        gas_transmission = jnp.exp(-gas_optical_depths)
        gas_transmission_by_species = jnp.sum(
            gas_transmission * weights_gauss_expanded,
            axis=0,
        )
        total_transmission_active = jnp.prod(gas_transmission_by_species, axis=1)

        if scattering_in_transmission:
            total_transmission_active = total_transmission_active * jnp.exp(
                -scattering_optical_depths
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
        return (
            jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)
            + radius_squared[-1]
        )

    transit_radii_baseline_squared = _transit_radii_squared_from_optical_depths(
        gas_optical_depths_active,
        scattering_optical_depths_active,
    )

    def _leave_one_out_contribution(i_leave_str_idx: ArrayLike) -> ArrayLike:
        path_by_layer = layer_path_weights_by_layer[i_leave_str_idx]
        rho_by_layer = rho_work[i_leave_str_idx]
        gas_opacity_times_rho = opacities_work[..., i_leave_str_idx] * rho_by_layer
        gas_optical_depths_leave_one_out = gas_optical_depths_active - (
            gas_opacity_times_rho[..., None] * path_by_layer
        )

        if scattering_in_transmission:
            continuum_opacity_times_rho = (
                continuum_opacities_scattering_work[:, i_leave_str_idx] * rho_by_layer
            )
            scattering_optical_depths_leave_one_out = (
                scattering_optical_depths_active
                - continuum_opacity_times_rho[:, None] * path_by_layer
            )
        else:
            scattering_optical_depths_leave_one_out = scattering_optical_depths_active

        transit_radii_leave_one_out_squared = (
            _transit_radii_squared_from_optical_depths(
                gas_optical_depths_leave_one_out,
                scattering_optical_depths_leave_one_out,
            )
        )
        return transit_radii_baseline_squared - transit_radii_leave_one_out_squared

    transmission_contribution_raw = lax.map(
        _leave_one_out_contribution,
        jnp.arange(n_layers, dtype=jnp.int32),
    )

    sum_contrib = jnp.sum(transmission_contribution_raw, axis=0, keepdims=True)

    transmission_contribution_final = jnp.where(
        sum_contrib == 0, 0.0, transmission_contribution_raw/sum_contrib
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
