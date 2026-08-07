import math
from functools import partial

from jax import default_backend
import jax.numpy as jnp
from jax import lax, jit

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE

TINIEST = jnp.finfo(FEAUTRIER_JAX_DTYPE).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(FEAUTRIER_JAX_DTYPE).max)
# Gradient-safe denominator floor. A reciprocal 1/x has gradient -1/x**2, which
# overflows to inf once x**2 underflows to 0 -- i.e. for x in roughly
# [TINIEST, sqrt(TINIEST)]. Flooring denominators at sqrt(TINIEST) keeps x**2 in
# the normal range so the reverse-mode gradient stays finite; such denominators
# are physically zero either way, so the forward result is unchanged. (Flooring
# at TINIEST itself, as the code originally did, leaves the gradient NaN.)
GRAD_SAFE_DENOM = jnp.sqrt(TINIEST)

# Detect GPU once at import time to choose optimal tridiagonal solver.
_USE_GPU = default_backend() != "cpu"

# Batch threshold (n_gf * n_angles) above which the Thomas-factorisation
# Feautrier path is preferred over parallel cyclic reduction (PCR) on GPU.
#
# Conventional wisdom prefers PCR on GPU because Thomas is sequential over
# layers. But GPU benchmarking (testjaxprt/benchmarking/benchmark_gpu_kernels.py)
# shows the opposite once the batch is large. The Thomas factorisation feeds an
# associative-scan solve (O(log n_layers) depth, O(n_layers) work) that is the
# fastest solver measured at every size — e.g. at xl (batch ~9.6e4, 300 layers)
# the assoc solve is ~6x faster than PCR (which runs ceil(log2(n_layers))
# full-array sweeps with heavy memory traffic) and ~1.8x faster than the
# sequential Thomas scan. The solve is the per-scattering-iteration hot loop, so
# this dominates the emission cost. PCR is retained only at small batch, where
# its near-free vectorised prepare wins the full prepare+solve loop (the Thomas
# factorisation cannot be amortised over enough solves). Frequency chunking does
# not change n_layers, so it cannot move this crossover in PCR's favour.
#
# The full-loop crossover on an A100 sits below batch ~24k for scattering
# (multiple solves); 16384 routes production-scale and medium batches to the
# assoc path while leaving tiny batches on PCR. Tunable via the
# ``thomas_min_batch`` argument (huge value forces PCR everywhere; 0 forces the
# Thomas-factor/assoc path everywhere).
DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH = 16384


def _feautrier_use_thomas(
    n_gf: int, n_angles: int, thomas_min_batch: int = DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH
) -> bool:
    """Whether to use the Thomas-factorisation Feautrier path over PCR.

    The Thomas factorisation (``prepare``) feeds either the sequential scan solve
    on CPU or the associative-scan solve on GPU; PCR is the alternative. Always
    true on CPU. On GPU, true when the batch ``n_gf * n_angles`` is large enough
    that the assoc solve's speed outweighs the Thomas factorisation's higher
    one-time prepare cost. ``n_gf`` and ``n_angles`` are static, so this resolves
    at trace time and keeps the ``prepare``/``solve`` choice consistent (their
    solver states are not interchangeable).
    """
    if not _USE_GPU:
        return True
    return (n_gf * n_angles) >= thomas_min_batch


@jit
def solve_tridiagonal_pcr(a, b, c, d):
    """
    Solves a batch of tridiagonal systems using the parallel cyclic reduction (PCR) algorithm.

    This implementation is based on a parallel CUDA implementation and is suitable for GPU execution.
    For the method to be numerically stable, the matrix should be diagonally dominant.

    The shapes of the input arrays should be (..., N), where (...) is one or more batch dimensions
    and N is the size of each tridiagonal system.

    Source:
    https://github.com/tanim72/15418-final-project

    Args:
        a (jnp.ndarray): The lower diagonal of the matrix A, shape (..., N).
                         a[..., 0] should be 0.
        b (jnp.ndarray): The main diagonal of the matrix A, shape (..., N).
        c (jnp.ndarray): The upper diagonal of the matrix A, shape (..., N).
                         c[..., -1] should be 0.
        d (jnp.ndarray): The right-hand side of the equation, shape (..., N).

    Returns:
        jnp.ndarray: The solution of the system, shape (..., N).
    """
    original_shape = a.shape
    N = original_shape[-1]

    if N == 0:
        return jnp.zeros_like(d)

    # Flatten batch dimensions to (M, N)
    M = a.size // N
    a_flat = a.reshape(M, N)
    b_flat = b.reshape(M, N)
    c_flat = c.reshape(M, N)
    d_flat = d.reshape(M, N)

    # --- Define shift operations for accessing neighbors ---
    def shift_right(x, s):
        """Shifts the array to the right, padding with zeros. Accesses x[i-s]."""
        rolled = jnp.roll(x, shift=s, axis=-1)
        mask = jnp.arange(x.shape[-1]) < s
        return jnp.where(mask, 0.0, rolled)

    def shift_left(x, s):
        """Shifts the array to the left, padding with zeros. Accesses x[i+s]."""
        rolled = jnp.roll(x, shift=-s, axis=-1)
        mask = jnp.arange(x.shape[-1]) >= x.shape[-1] - s
        return jnp.where(mask, 0.0, rolled)

    # PCR is a loop that doubles the stride at each iteration
    def pcr_body_fun(val):
        """One step of the parallel cyclic reduction."""
        a_pcr, b_pcr, c_pcr, d_pcr, stride = val

        # --- Get values from neighbors at i-stride and i+stride ---
        a_left = shift_right(a_pcr, stride)
        b_left = shift_right(b_pcr, stride)
        c_left = shift_right(c_pcr, stride)
        d_left = shift_right(d_pcr, stride)

        a_right = shift_left(a_pcr, stride)
        b_right = shift_left(b_pcr, stride)
        c_right = shift_left(c_pcr, stride)
        d_right = shift_left(d_pcr, stride)

        # --- Calculate reduction factors alpha and gamma ---
        # Guard the division so neither the forward value nor its reverse-mode
        # gradient sees a divide-by-zero. The boundary shifts pad b_left[...,0]
        # and b_right[...,-1] with exact zeros; a plain
        # ``jnp.where(b != 0, x / b, 0.0)`` still evaluates ``x / 0 -> inf`` in
        # the un-taken branch, and reverse-mode then turns the masked-out
        # ``0 * inf`` into NaN (the forward result is unaffected, which is why
        # this only surfaced once the GPU PCR path was differentiated). Dividing
        # by a denominator that is 1.0 wherever the entry is masked keeps both
        # passes finite; the jnp.where still selects 0.0 at those entries.
        safe_b_left = jnp.where(b_left != 0, b_left, 1.0)
        safe_b_right = jnp.where(b_right != 0, b_right, 1.0)
        alpha = jnp.where(b_left != 0, -a_pcr / safe_b_left, 0.0)
        gamma = jnp.where(b_right != 0, -c_pcr / safe_b_right, 0.0)

        # --- Update the diagonals and the right-hand side ---
        # The update corresponds to climinating off-diagonal elements
        b_new = b_pcr + alpha * c_left + gamma * a_right
        d_new = d_pcr + alpha * d_left + gamma * d_right
        a_new = alpha * a_left
        c_new = gamma * c_right

        return a_new, b_new, c_new, d_new, stride * 2

    # --- Run the reduction loop ---
    # The stride doubles each sweep until it reaches N, so the reduction always
    # takes exactly ceil(log2(N)) sweeps. N is the (static) trailing array
    # dimension, so we unroll those sweeps as a Python loop rather than
    # lax.while_loop: while_loop is not reverse-mode differentiable (it raises
    # "does not work for lax.while_loop ... with dynamic start/stop values"
    # under grad), whereas this static unroll of ~log2(N) sweeps is. That is
    # what lets the GPU Feautrier solve participate in the retrieval gradient.
    n_sweeps = 0 if N <= 1 else int(math.ceil(math.log2(N)))
    val = (a_flat, b_flat, c_flat, d_flat, 1)
    for _ in range(n_sweeps):
        val = pcr_body_fun(val)
    a_final, b_final, c_final, d_final, _ = val

    # --- Final solve ---
    # After reduction, the matrix is diagonal (a_final and c_final are zero)
    # The solution is simply d / b
    # Add epsilon to avoid division by zero in case of singular matrices.
    x_flat = d_final / (b_final + 1e-12)

    # Reshape solution back to original batch dimensions
    return x_flat.reshape(original_shape)


# ---------------------------------------------------------------------------
# Layer-major ("lm") Feautrier system interface.
#
# These functions keep every layer-indexed array in (n_layers, n_gf) /
# (n_layers, batch) layout so that all per-layer scan steps read and write
# contiguous rows. The coefficient construction is fused into the
# pre-factoring scan, so the (n_gf, n_angles, n_layers)-sized tridiagonal
# coefficient arrays of the gf-major path are never materialised on CPU.
# ---------------------------------------------------------------------------

# Clip bounds for the Feautrier finite-difference coefficients, matching the
# original gf-major construction in radiative_transfer_core.
_FEAUTRIER_INV_DTAU_CAP = 1e10
_FEAUTRIER_INV_DTAU2_CAP = _FEAUTRIER_INV_DTAU_CAP * 0.5


@partial(jit, static_argnames=("n_gf", "n_angles", "n_layers"))
def _prepare_feautrier_system_lm_cpu(
    dtau1_lm,
    dtau2_lm,
    emission_cos_angles,
    emission_cos_angles_weights,
    n_gf,
    n_angles,
    n_layers,
):
    """Fused coefficient construction + Thomas pre-factoring + Lambda diagonal.

    One scan over layers builds the Feautrier tridiagonal rows on the fly from
    the delta-optical-depths, immediately eliminates them (Thomas forward
    elimination), and accumulates the angle-quadrature Lambda diagonal, so the
    full (batch, n_layers) coefficient arrays are never materialised.

    The per-row formulas are exactly those of the original gf-major
    construction: row 0 uses the one-sided top boundary stencil, rows
    1..n_layers-2 the centered stencil, and row n_layers-1 is the identity
    (the surface boundary value is imposed through the RHS).

    Parameters
    ----------
    dtau1_lm : jax.Array, shape (n_layers - 1, n_gf)
        First-neighbor optical-depth differences, floored away from zero.
    dtau2_lm : jax.Array, shape (n_layers - 2, n_gf)
        Second-neighbor optical-depth differences, floored away from zero.
    emission_cos_angles : jax.Array, shape (n_angles,)
    emission_cos_angles_weights : jax.Array, shape (n_angles,)
    n_gf, n_angles, n_layers : int
        Static shape parameters; n_layers must be >= 3.

    Returns
    -------
    factors : tuple of jax.Array, each shape (n_layers, batch)
        ``(c_prime, inv_denom, a_times_inv_denom)`` Thomas factors, with
        batch = n_gf * n_angles (angle fastest).
    lambda_loc_lm : jax.Array, shape (n_layers, n_gf)
        Angle-quadrature diagonal of the approximate Lambda operator.
    f_surface_ga : jax.Array, shape (n_gf, n_angles)
        ``mu / dtau1[n_layers - 2]`` (clipped), used for the surface
        derivative of the emergent Feautrier variable.
    """
    n = n_layers
    batch = n_gf * n_angles
    dtype = dtau1_lm.dtype

    mu = emission_cos_angles[None, :]  # (1, n_angles)
    mu2 = (emission_cos_angles * 2.0)[None, :]
    w = emission_cos_angles_weights[None, :]

    def step(carry, k):
        c_prev, f_prev = carry

        dtau1_k = lax.dynamic_index_in_dim(
            dtau1_lm, jnp.minimum(k, n - 2), keepdims=False
        )  # (n_gf,)
        dtau2_km1 = lax.dynamic_index_in_dim(
            dtau2_lm, jnp.clip(k - 1, 0, n - 3), keepdims=False
        )  # (n_gf,)

        f2 = jnp.clip(
            mu / dtau1_k[:, None], -_FEAUTRIER_INV_DTAU_CAP, _FEAUTRIER_INV_DTAU_CAP
        )  # (n_gf, n_angles)
        f1 = jnp.clip(
            mu2 / dtau2_km1[:, None], -_FEAUTRIER_INV_DTAU2_CAP, _FEAUTRIER_INV_DTAU2_CAP
        )  # (n_gf, n_angles)

        is_top = k == 0
        is_surf = k == n - 1

        b_top = 1.0 + 2.0 * f2 * (1.0 + f2)
        b_mid = 1.0 + f1 * (f2 + f_prev)
        b_k = jnp.where(is_top, b_top, jnp.where(is_surf, 1.0, b_mid))
        c_k_row = jnp.where(is_top, -2.0 * f2**2, jnp.where(is_surf, 0.0, -f1 * f2))
        a_k_row = jnp.where(is_top | is_surf, 0.0, -f1 * f_prev)

        # Lambda diagonal row. The mid-row diagonal is floored away from zero
        # before the reciprocal so its reverse-mode gradient stays finite (see
        # GRAD_SAFE_DENOM). At k == n-1, f2 holds mu/dtau1[n-2] = the surface
        # stencil coefficient.
        b_mid_safe = jnp.where(
            jnp.abs(b_mid) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, b_mid
        )
        lam_branch = jnp.where(
            is_top, b_top, jnp.where(is_surf, 1.0 + 2.0 * f2**2, b_mid_safe)
        )
        lam_k = jnp.sum(w / lam_branch, axis=1)  # (n_gf,)

        # Thomas forward elimination of the freshly built row.
        denom = b_k.reshape(batch) - a_k_row.reshape(batch) * c_prev
        safe_den = jnp.where(jnp.abs(denom) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom)
        inv_den = 1.0 / safe_den
        c_k = c_k_row.reshape(batch) * inv_den
        a_inv_k = a_k_row.reshape(batch) * inv_den

        return (c_k, f2), (c_k, inv_den, a_inv_k, lam_k)

    (_, f_surface_ga), (c_prime, inv_denom, a_times_inv_denom, lambda_loc_lm) = (
        lax.scan(
            step,
            (
                jnp.zeros(batch, dtype=dtype),
                jnp.zeros((n_gf, n_angles), dtype=dtype),
            ),
            jnp.arange(n),
        )
    )

    return (c_prime, inv_denom, a_times_inv_denom), lambda_loc_lm, f_surface_ga


@partial(jit, static_argnames=("n_gf", "n_angles", "n_layers"))
def _prepare_feautrier_system_lm_gpu(
    dtau1_lm,
    dtau2_lm,
    emission_cos_angles,
    emission_cos_angles_weights,
    n_gf,
    n_angles,
    n_layers,
):
    """GPU variant: materialise the tridiagonal rows for the PCR solver.

    Same formulas as :func:`_prepare_feautrier_system_lm_cpu`, but built
    vectorised (PCR has no separable factorisation step, so the coefficient
    arrays must exist). Returns ``(a, b, c)`` in the (batch, n_layers) layout
    expected by :func:`solve_tridiagonal_pcr` as the solver state.
    """
    n = n_layers
    batch = n_gf * n_angles

    mu = emission_cos_angles[None, None, :]
    mu2 = (emission_cos_angles * 2.0)[None, None, :]
    w = emission_cos_angles_weights[None, :]

    f2_all = jnp.clip(
        mu / dtau1_lm[:, :, None], -_FEAUTRIER_INV_DTAU_CAP, _FEAUTRIER_INV_DTAU_CAP
    )  # (n-1, n_gf, a)
    f1_all = jnp.clip(
        mu2 / dtau2_lm[:, :, None], -_FEAUTRIER_INV_DTAU2_CAP, _FEAUTRIER_INV_DTAU2_CAP
    )  # (n-2, n_gf, a)

    b_top = 1.0 + 2.0 * f2_all[0] * (1.0 + f2_all[0])  # (n_gf, a)
    b_mid = 1.0 + f1_all * (f2_all[1:] + f2_all[:-1])  # (n-2, n_gf, a)
    c_top = -2.0 * f2_all[0] ** 2
    c_mid = -f1_all * f2_all[1:]
    a_mid = -f1_all * f2_all[:-1]

    pad_row = jnp.zeros((1, n_gf, n_angles), dtype=dtau1_lm.dtype)
    one_row = jnp.ones((1, n_gf, n_angles), dtype=dtau1_lm.dtype)
    a_lm = jnp.concatenate([pad_row, a_mid, pad_row], axis=0)
    b_lm = jnp.concatenate([b_top[None], b_mid, one_row], axis=0)
    c_lm = jnp.concatenate([c_top[None], c_mid, pad_row], axis=0)

    b_mid_safe = jnp.where(jnp.abs(b_mid) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, b_mid)
    lam_top = jnp.sum(w / b_top, axis=1)  # (n_gf,)
    lam_mid = jnp.sum(w[None] / b_mid_safe, axis=2)  # (n-2, n_gf)
    lam_surf = jnp.sum(w / (1.0 + 2.0 * f2_all[-1] ** 2), axis=1)  # (n_gf,)
    lambda_loc_lm = jnp.concatenate(
        [lam_top[None], lam_mid, lam_surf[None]], axis=0
    )  # (n, n_gf)

    state = (
        a_lm.reshape(n, batch).T,
        b_lm.reshape(n, batch).T,
        c_lm.reshape(n, batch).T,
    )

    return state, lambda_loc_lm, f2_all[-1]


def prepare_feautrier_system_lm(
    dtau1_lm,
    dtau2_lm,
    emission_cos_angles,
    emission_cos_angles_weights,
    n_gf,
    n_angles,
    n_layers,
    thomas_min_batch: int = DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
):
    """Build the per-chunk Feautrier solver state, Lambda diagonal and surface
    stencil coefficient from layer-major delta-optical-depths.

    Dispatches to the Thomas fused scan (CPU always, and GPU at large batch) or
    to the materialising PCR preparation (GPU at small batch); the returned
    state is consumed by :func:`solve_feautrier_system_lm`, which must make the
    same choice. See :func:`_feautrier_use_thomas`.
    """
    prepare = (
        _prepare_feautrier_system_lm_cpu
        if _feautrier_use_thomas(n_gf, n_angles, thomas_min_batch)
        else _prepare_feautrier_system_lm_gpu
    )
    return prepare(
        dtau1_lm,
        dtau2_lm,
        emission_cos_angles,
        emission_cos_angles_weights,
        n_gf,
        n_angles,
        n_layers,
    )


@partial(jit, static_argnames=("n_gf", "n_angles", "n_layers"))
def _solve_feautrier_system_lm_cpu(
    factors,
    source_lm,
    r_surf_ga,
    angle_weight_a,
    n_gf,
    n_angles,
    n_layers,
):
    """Layer-major prefactored Thomas solve.

    Parameters
    ----------
    factors : tuple of jax.Array, each (n_layers, batch)
        Thomas factors from :func:`_prepare_feautrier_system_lm_cpu`.
    source_lm : jax.Array, shape (n_layers, n_gf)
        Angle-independent source function; only rows 0 .. n_layers-2 enter the
        RHS (the surface row is ``r_surf_ga``).
    r_surf_ga : jax.Array, shape (n_gf, n_angles)
        Surface boundary RHS.
    angle_weight_a : jax.Array, shape (n_angles,)
        Angular quadrature weights for the mean-intensity accumulation.

    Returns
    -------
    I_H_top : jax.Array, shape (n_gf, n_angles)
    x_last : jax.Array, shape (n_gf, n_angles)
    x_nm2 : jax.Array, shape (n_gf, n_angles)
    J_bol_lm : jax.Array, shape (n_layers, n_gf)
    """
    c_prime, inv_denom, a_times_inv_denom = factors
    n = n_layers
    batch = n_gf * n_angles

    def _weighted_gf(x_k):
        return jnp.sum(
            x_k.reshape(n_gf, n_angles) * angle_weight_a[None, :],
            axis=1,
        )

    # RHS forward sweep over rows 0 .. n-2 as one uniform scan: the
    # zero-initialised carry makes row 0 exact because a_times_inv_denom[0] is
    # zero (Feautrier padding).
    def fwd_rhs_step(d_prev, k):
        rhs_k = jnp.broadcast_to(
            source_lm[k][:, None], (n_gf, n_angles)
        ).reshape(batch)
        d_k = rhs_k * inv_denom[k] - a_times_inv_denom[k] * d_prev
        return d_k, d_k

    d_nm2, d_upto_nm2 = lax.scan(
        fwd_rhs_step,
        jnp.zeros(batch, dtype=inv_denom.dtype),
        jnp.arange(n - 1),
    )  # d_upto_nm2: (n-1, batch)

    # Surface row
    x_last_flat = (
        r_surf_ga.reshape(batch) * inv_denom[n - 1]
        - a_times_inv_denom[n - 1] * d_nm2
    )

    # Backward substitution; the reverse scan emits rows in natural layer
    # order, avoiding a flip copy.
    x_nm2_flat = d_upto_nm2[n - 2, :] - c_prime[n - 2, :] * x_last_flat
    weighted_last = _weighted_gf(x_last_flat)
    weighted_nm2 = _weighted_gf(x_nm2_flat)

    def bwd_step(x_next, k):
        x_k = d_upto_nm2[k, :] - c_prime[k, :] * x_next
        return x_k, _weighted_gf(x_k)

    x0_flat, weighted_rest = lax.scan(
        bwd_step,
        x_nm2_flat,
        jnp.arange(n - 2),
        reverse=True,
    )
    J_bol_lm = jnp.concatenate(
        [weighted_rest, weighted_nm2[None, :], weighted_last[None, :]],
        axis=0,
    )  # (n, n_gf)

    return (
        (-x0_flat).reshape(n_gf, n_angles),
        x_last_flat.reshape(n_gf, n_angles),
        x_nm2_flat.reshape(n_gf, n_angles),
        J_bol_lm,
    )


# ---------------------------------------------------------------------------
# Experimental: prefactored Thomas solve with the two layer-axis recurrences
# expressed as parallel (associative) scans instead of sequential lax.scan.
#
# Both Thomas RHS sweeps are first-order affine recurrences y_k = A_k y_{k-1} +
# B_k. An affine map x -> A x + B composes associatively:
#   (A2,B2) o (A1,B1) = (A2 A1, A2 B1 + B2)
# so lax.associative_scan over (A,B) yields every prefix in O(log n_layers)
# depth (vs the n_layers sequential steps of lax.scan), at O(n_layers) work. On
# GPU this trades sequential-step launch/sync overhead for extra parallel work;
# whether it wins is what the benchmark measures.
# ---------------------------------------------------------------------------


def _affine_compose(earlier, later):
    """Compose two affine maps applied earlier-then-later.

    Each operand is a pytree ``(A, B)`` representing ``x -> A x + B``. Returns
    the composition ``later ∘ earlier``. Used as the associative operator for
    :func:`lax.associative_scan`, whose inclusive prefix at position ``k`` is the
    map taking the initial value to ``y_k``.
    """
    A_e, B_e = earlier
    A_l, B_l = later
    return (A_l * A_e, A_l * B_e + B_l)


@partial(jit, static_argnames=("n_gf", "n_angles", "n_layers"))
def _solve_feautrier_system_lm_assoc(
    factors,
    source_lm,
    r_surf_ga,
    angle_weight_a,
    n_gf,
    n_angles,
    n_layers,
):
    """Associative-scan variant of :func:`_solve_feautrier_system_lm_cpu`.

    Identical interface, factors and result; the forward RHS sweep and backward
    substitution are evaluated with :func:`lax.associative_scan` (O(log n_layers)
    depth) rather than sequential ``lax.scan``. The full per-layer solution is
    materialised once to form the angle-weighted ``J_bol`` (the fused
    sequential solve avoids this, trading memory for the parallel depth).
    """
    c_prime, inv_denom, a_times_inv_denom = factors
    n = n_layers
    batch = n_gf * n_angles

    if n == 1:
        x_last_flat = r_surf_ga.reshape(batch) * inv_denom[0]
        J_bol_lm = jnp.sum(
            x_last_flat.reshape(n_gf, n_angles) * angle_weight_a[None, :], axis=1
        )[None, :]
        return (
            (-x_last_flat).reshape(n_gf, n_angles),
            x_last_flat.reshape(n_gf, n_angles),
            x_last_flat.reshape(n_gf, n_angles),
            J_bol_lm,
        )

    # RHS for inner rows 0 .. n-2, broadcast over angles → (n-1, batch).
    rhs_inner = jnp.broadcast_to(
        source_lm[: n - 1, :, None], (n - 1, n_gf, n_angles)
    ).reshape(n - 1, batch)

    # Forward sweep: d_k = rhs_k*inv_denom_k - a_times_inv_denom_k*d_{k-1},
    # d_{-1}=0. a_times_inv_denom[0]=0 (Feautrier padding) makes row 0 exact.
    A_fwd = -a_times_inv_denom[: n - 1]  # (n-1, batch)
    B_fwd = rhs_inner * inv_denom[: n - 1]  # (n-1, batch)
    _, d_upto_nm2 = lax.associative_scan(_affine_compose, (A_fwd, B_fwd), axis=0)
    d_nm2 = d_upto_nm2[n - 2]

    # Surface row.
    x_last_flat = (
        r_surf_ga.reshape(batch) * inv_denom[n - 1]
        - a_times_inv_denom[n - 1] * d_nm2
    )

    # Backward substitution: x_k = d_k - c_prime_k * x_{k+1}, x_{n-1}=x_last.
    # Reverse the row order so it becomes a forward affine recurrence seeded by
    # x_last, run the prefix scan, then map the seed through each prefix.
    A_bwd = (-c_prime[: n - 1])[::-1]  # (n-1, batch), reversed
    B_bwd = d_upto_nm2[::-1]  # (n-1, batch), reversed
    A_cum, B_cum = lax.associative_scan(_affine_compose, (A_bwd, B_bwd), axis=0)
    x_rev = A_cum * x_last_flat[None, :] + B_cum  # X_j = x_{n-2-j}
    x_inner = x_rev[::-1]  # (n-1, batch): rows 0 .. n-2

    x0_flat = x_inner[0]
    x_nm2_flat = x_inner[n - 2]

    # Angle-weighted J_bol over the full layer stack (rows 0..n-2 + surface).
    x_full = jnp.concatenate([x_inner, x_last_flat[None, :]], axis=0)  # (n, batch)
    J_bol_lm = jnp.sum(
        x_full.reshape(n, n_gf, n_angles) * angle_weight_a[None, None, :], axis=2
    )  # (n, n_gf)

    return (
        (-x0_flat).reshape(n_gf, n_angles),
        x_last_flat.reshape(n_gf, n_angles),
        x_nm2_flat.reshape(n_gf, n_angles),
        J_bol_lm,
    )


@partial(jit, static_argnames=("n_gf", "n_angles", "n_layers"))
def _solve_feautrier_system_lm_gpu(
    state,
    source_lm,
    r_surf_ga,
    angle_weight_a,
    n_gf,
    n_angles,
    n_layers,
):
    """Layer-major Feautrier solve through the PCR kernel (GPU)."""
    a, b, c = state
    n = n_layers
    batch = n_gf * n_angles

    d_inner = (
        jnp.broadcast_to(
            source_lm[: n - 1, :, None], (n - 1, n_gf, n_angles)
        )
        .reshape(n - 1, batch)
        .T
    )  # (batch, n-1)
    d_full = jnp.concatenate([d_inner, r_surf_ga.reshape(batch, 1)], axis=1)

    x_full = solve_tridiagonal_pcr(a, b, c, d_full)  # (batch, n)

    x_gfan = x_full.reshape(n_gf, n_angles, n)
    J_bol_lm = jnp.sum(
        x_gfan * angle_weight_a[None, :, None], axis=1
    ).T  # (n, n_gf)

    return (
        (-x_full[:, 0]).reshape(n_gf, n_angles),
        x_full[:, n - 1].reshape(n_gf, n_angles),
        x_full[:, n - 2].reshape(n_gf, n_angles),
        J_bol_lm,
    )


def solve_feautrier_system_lm(
    state,
    source_lm,
    r_surf_ga,
    angle_weight_a,
    n_gf,
    n_angles,
    n_layers,
    thomas_min_batch: int = DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
):
    """Solve the prepared layer-major Feautrier system for one source iterate.

    ``state`` must come from :func:`prepare_feautrier_system_lm` with the same
    batch (the Thomas-factor and PCR-coefficient states are not interchangeable);
    the shared :func:`_feautrier_use_thomas` decision guarantees this. Returns
    ``(I_H_top, x_last, x_nm2, J_bol_lm)`` with the angular arrays shaped
    (n_gf, n_angles) and ``J_bol_lm`` shaped (n_layers, n_gf).

    Solver choice (see GPU benchmarking):
    - Thomas-factor path (CPU always, GPU at large batch): on GPU the
      associative-scan solve is fastest at every measured size (e.g. ~6x faster
      than PCR and ~1.8x faster than the sequential Thomas scan at xl), so it is
      used there; on CPU the sequential scan stays best (the parallel scan's
      O(n_layers log n_layers) extra work does not pay off without massive
      device parallelism).
    - PCR path (GPU at small batch): kept because its near-free vectorised
      ``prepare`` wins the full prepare+solve loop when the batch is too small to
      amortise the Thomas factorisation.
    """
    if _feautrier_use_thomas(n_gf, n_angles, thomas_min_batch):
        solve = (
            _solve_feautrier_system_lm_assoc
            if _USE_GPU
            else _solve_feautrier_system_lm_cpu
        )
    else:
        solve = _solve_feautrier_system_lm_gpu
    return solve(
        state,
        source_lm,
        r_surf_ga,
        angle_weight_a,
        n_gf,
        n_angles,
        n_layers,
    )


@jit
def linear_fit(x, y):
    """
    Calculate slope and y-axis intercept of x,y data, assuming zero error on data.
    Translated from Fortran linear_fit.

    Args:
        x: 1D array of x values.
        y: 1D array of y values.

    Returns:
        Tuple (a, b) where a is the y-intercept and b is the slope.
    """
    ndata = x.shape[0]
    sum_x = jnp.sum(x)
    sum_y = jnp.sum(y)
    sum_xy = jnp.sum(x * y)
    sum_x_sq = jnp.sum(x**2)

    # Fortran: b = (sum(x)*sum(y)/dble(ndata) - sum(x*y))/ &
    #             (sum(x)**2d0/dble(ndata) - sum(x**2d0))
    denominator = sum_x**2 / ndata - sum_x_sq
    # Handle potential division by zero if all x values are the same
    b = lax.cond(
        jnp.abs(denominator) < 1e-15,  # Use a small tolerance
        lambda: 0.0,  # If denominator is zero, slope is effectively zero
        lambda: (sum_x * sum_y / ndata - sum_xy) / denominator,
    )

    # Fortran: a = sum(y-b*x)/dble(ndata)
    a = jnp.sum(y - b * x) / ndata

    return a, b
