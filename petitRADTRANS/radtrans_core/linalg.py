import math
from functools import partial

from jax import default_backend
import jax.numpy as jnp
from jax import lax, jit

from petitRADTRANS.jax_dtype import JAX_DTYPE

TINIEST = jnp.finfo(JAX_DTYPE).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(JAX_DTYPE).max)
# Gradient-safe denominator floor. A reciprocal 1/x has gradient -1/x**2, which
# overflows to inf once x**2 underflows to 0 -- i.e. for x in roughly
# [TINIEST, sqrt(TINIEST)]. Flooring denominators at sqrt(TINIEST) keeps x**2 in
# the normal range so the reverse-mode gradient stays finite; such denominators
# are physically zero either way, so the forward result is unchanged. (Flooring
# at TINIEST itself, as the code originally did, leaves the gradient NaN.)
GRAD_SAFE_DENOM = jnp.sqrt(TINIEST)

# Detect GPU once at import time to choose optimal tridiagonal solver.
_USE_GPU = default_backend() != "cpu"


@jit
def solve_tridiagonal_thomas(a, b, c, d):
    """
    Solves a tridiagonal system of equations using the Thomas algorithm.
    The system is of the form Ax = d, where A is a tridiagonal matrix.
    The inputs are Jax arrays.
    The implementation is a Jax port of the naive implementation of the Thomas algorithm.
    See https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    WARNING: This implementation is not stable for all matrices. It is recommended to use it only for
    diagonally dominant matrices.
    See https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Numerical_stability

    Args:
        a (jnp.ndarray): The lower diagonal of the matrix A.
        b (jnp.ndarray): The main diagonal of the matrix A.
        c (jnp.ndarray): The upper diagonal of the matrix A.
        d (jnp.ndarray): The right-hand side of the equation.

    Returns:
        jnp.ndarray: The solution of the system.
    """
    n = b.shape[-1]
    c_p = jnp.zeros_like(c)
    d_p = jnp.zeros_like(d)

    # Forward sweep
    c_p = c_p.at[..., 0].set(c[..., 0] / b[..., 0])
    d_p = d_p.at[..., 0].set(d[..., 0] / b[..., 0])

    def forward_sweep(i, val):
        c_p, d_p = val
        denominator = b[..., i] - a[..., i] * c_p[..., i - 1]

        # if jnp.any(jnp.abs(denominator) < 1e-10):
        #     raise ValueError("The matrix is singular.")

        c_p = c_p.at[..., i].set(c[..., i] / denominator)
        d_p = d_p.at[..., i].set(
            (d[..., i] - a[..., i] * d_p[..., i - 1]) / denominator
        )

        return c_p, d_p

    c_p, d_p = lax.fori_loop(1, n - 1, forward_sweep, (c_p, d_p))

    # Handle the last element
    denominator = b[..., n - 1] - a[..., n - 1] * c_p[..., n - 2]
    d_p = d_p.at[..., n - 1].set(
        (d[..., n - 1] - a[..., n - 1] * d_p[..., n - 2]) / denominator
    )

    # Backward substitution
    x = jnp.zeros_like(d)
    x = x.at[..., -1].set(d_p[..., -1])

    def backward_substitution(i, x):
        x = x.at[..., i].set(d_p[..., i] - c_p[..., i] * x[..., i + 1])
        return x

    x = lax.fori_loop(0, n - 1, lambda i, x: backward_substitution(n - 2 - i, x), x)

    return jnp.clip(x, max=HUGE_SQRT)


@jit
def solve_tridiagonal_thomas_batched(a, b, c, d):
    """
    Solves n independent tridiagonal systems simultaneously.

    Parameters
    ----------
    a, b, c, d : (batch, n_layers)
        Lower diagonal (a[:,0] unused), main diagonal,
        upper diagonal (c[:,-1] unused), RHS.

    Returns
    -------
    x : (batch, n_layers)
    """
    n = b.shape[1]

    # ---- forward sweep: initialise with row 0 ----
    c0 = c[:, 0] / b[:, 0]  # (batch,)
    d0 = d[:, 0] / b[:, 0]  # (batch,)

    def fwd_step(carry, k):
        c_prev, d_prev = carry
        denom = b[:, k] - a[:, k] * c_prev
        safe_den = jnp.where(jnp.abs(denom) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom)
        c_k = c[:, k] / safe_den
        d_k = (d[:, k] - a[:, k] * d_prev) / safe_den
        return (c_k, d_k), (c_k, d_k)

    _, (c_rest, d_rest) = lax.scan(fwd_step, (c0, d0), jnp.arange(1, n))
    # prepend layer-0 results  →  (n, batch) each
    c_all = jnp.concatenate([c0[None, :], c_rest], axis=0)
    d_all = jnp.concatenate([d0[None, :], d_rest], axis=0)

    # ---- back substitution ----
    x_last = d_all[n - 1, :]  # (batch,)

    def bwd_step(x_next, k):
        x_k = d_all[k, :] - c_all[k, :] * x_next
        return x_k, x_k

    _, x_rest = lax.scan(bwd_step, x_last, jnp.arange(n - 2, -1, -1))
    # x_rest is (n-1, batch) in reverse order; rebuild full solution
    x_all = jnp.concatenate([x_rest[::-1, :], x_last[None, :]], axis=0)  # (n, batch)

    return x_all.T  # (batch, n)


@partial(jit, static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def factor_feautrier_thomas(
    a,
    b,
    c,
    n_g,
    n_freq_bins,
    n_angles,
    n_layers,
):
    """Pre-factor the LHS of the Feautrier tridiagonal system.

    Performs the forward elimination that depends only on (a, b, c),
    caching the modified upper diagonal and reciprocal denominators so
    that subsequent solves with different RHS vectors only need the
    cheaper RHS-only forward sweep and backward substitution.

    The elimination runs as a single uniform scan over all rows; the
    Feautrier coefficient padding (``a[:, 0] == 0`` and ``c[:, -1] == 0``)
    makes the boundary rows fall out of the generic recurrence with a
    zero-initialised carry, so no per-row stitching (concatenates) is
    needed. Callers must respect that padding convention.

    Parameters
    ----------
    a, b, c : jax.Array, shape (batch, n_layers)
        Tridiagonal coefficients where batch = n_g * n_freq_bins * n_angles.
        ``a[:, 0]`` and ``c[:, -1]`` must be zero (Feautrier padding).
    n_g, n_freq_bins, n_angles, n_layers : int
        Static shape parameters.

    Returns
    -------
    c_prime : jax.Array, shape (n_layers, batch)
        Modified upper diagonal from Thomas forward elimination.
        ``c_prime[n_layers-1]`` is zero (from the ``c`` padding).
    inv_denom : jax.Array, shape (n_layers, batch)
        Reciprocal of the elimination denominators.  Includes the surface
        row at index ``n_layers - 1``.
    a_times_inv_denom : jax.Array, shape (n_layers, batch)
        ``a[:, k] * inv_denom[k]`` pre-multiplied for the RHS sweep.
        ``a_times_inv_denom[0]`` is zero (from the ``a`` padding).
    """
    batch = n_g * n_freq_bins * n_angles

    # Transpose once to layer-major so each scan step reads contiguous rows
    # (a[:, k] slices would stride by n_layers on every step).
    a_lm = a.T  # (n_layers, batch)
    b_lm = b.T
    c_lm = c.T

    def fwd_step(c_prev, abc_k):
        a_k, b_k, c_k_row = abc_k
        denom = b_k - a_k * c_prev
        safe_den = jnp.where(jnp.abs(denom) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom)
        inv_den = 1.0 / safe_den
        c_k = c_k_row * inv_den
        a_inv_k = a_k * inv_den
        return c_k, (c_k, inv_den, a_inv_k)

    _, (c_prime, inv_denom, a_times_inv_denom) = lax.scan(
        fwd_step,
        jnp.zeros(batch, dtype=a.dtype),
        (a_lm, b_lm, c_lm),
    )

    return c_prime, inv_denom, a_times_inv_denom


@partial(jit, static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def solve_feautrier_thomas_prefactored(
    c_prime,
    inv_denom,
    a_times_inv_denom,
    d_inner,
    d_surf,
    angle_weight_a,
    n_g,
    n_freq_bins,
    n_angles,
    n_layers,
):
    """Solve the Feautrier tridiagonal system using pre-factored LHS arrays.

    Uses the output of :func:`factor_feautrier_thomas` to skip the LHS
    forward elimination, performing only the RHS forward sweep and backward
    substitution.

    Parameters
    ----------
    c_prime : jax.Array, shape (n_layers, batch)
        Modified upper diagonal from :func:`factor_feautrier_thomas`.
    inv_denom : jax.Array, shape (n_layers, batch)
        Reciprocal denominators from :func:`factor_feautrier_thomas`.
    a_times_inv_denom : jax.Array, shape (n_layers, batch)
        Pre-multiplied ``a * inv_denom`` from :func:`factor_feautrier_thomas`.
    d_inner : jax.Array, shape (n_g, n_freq_bins, n_layers - 1)
        Angle-independent RHS entries for layers 0 .. n_layers-2.
    d_surf : jax.Array, shape (n_g, n_freq_bins, n_angles)
        Boundary RHS entry for the surface layer.
    angle_weight_a : jax.Array, shape (n_angles,)
        Angular quadrature weights for mean-intensity accumulation.
    n_g, n_freq_bins, n_angles, n_layers : int
        Static shape parameters.

    Returns
    -------
    I_H_top : jax.Array, shape (n_g, n_freq_bins, n_angles)
    x_last : jax.Array, shape (n_g, n_freq_bins, n_angles)
    x_nm2 : jax.Array, shape (n_g, n_freq_bins, n_angles)
    J_bol_gfl : jax.Array, shape (n_g, n_freq_bins, n_layers)
    """
    n = n_layers
    n_gf = n_g * n_freq_bins
    batch = n_gf * n_angles

    def _broadcast_rhs_layer(layer_idx):
        return jnp.broadcast_to(
            d_inner[:, :, layer_idx][:, :, None],
            (n_g, n_freq_bins, n_angles),
        ).reshape(batch)

    def _weighted_gf(x_k):
        return jnp.sum(
            x_k.reshape(n_gf, n_angles) * angle_weight_a[None, :],
            axis=1,
        )

    if n == 1:
        x_last_flat = d_surf.reshape(batch) * inv_denom[0]
        weighted_layers = _weighted_gf(x_last_flat)[None, :]
        return (
            (-x_last_flat).reshape(n_g, n_freq_bins, n_angles),
            x_last_flat.reshape(n_g, n_freq_bins, n_angles),
            x_last_flat.reshape(n_g, n_freq_bins, n_angles),
            weighted_layers.T.reshape(n_g, n_freq_bins, n),
        )

    # RHS-only forward sweep using cached factors. A zero-initialised carry
    # makes row 0 fall out of the generic recurrence (a_times_inv_denom[0] is
    # zero by the Feautrier padding convention), so the sweep is one uniform
    # scan and its stacked output is used directly — no concatenate.
    def fwd_rhs_step(d_prev, k):
        rhs_k = _broadcast_rhs_layer(k)
        d_k = rhs_k * inv_denom[k] - a_times_inv_denom[k] * d_prev
        return d_k, d_k

    d_nm2, d_upto_nm2 = lax.scan(
        fwd_rhs_step,
        jnp.zeros(batch, dtype=inv_denom.dtype),
        jnp.arange(n - 1),
    )  # d_upto_nm2: (n-1, batch)

    # Surface row
    x_last_flat = (
        d_surf.reshape(batch) * inv_denom[n - 1] - a_times_inv_denom[n - 1] * d_nm2
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
    weighted_layers = jnp.concatenate(
        [
            weighted_rest,
            weighted_nm2[None, :],
            weighted_last[None, :],
        ],
        axis=0,
    )

    I_H_top = (-x0_flat).reshape(n_g, n_freq_bins, n_angles)
    x_last = x_last_flat.reshape(n_g, n_freq_bins, n_angles)
    x_nm2 = x_nm2_flat.reshape(n_g, n_freq_bins, n_angles)
    J_bol_gfl = weighted_layers.T.reshape(n_g, n_freq_bins, n)

    return I_H_top, x_last, x_nm2, J_bol_gfl


@partial(jit, static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def solve_tridiagonal_thomas_batched_feautrier_fused(
    a,
    b,
    c,
    d_inner,
    d_surf,
    angle_weight_a,
    n_g,
    n_freq_bins,
    n_angles,
    n_layers,
):
    """
    Solves Feautrier tridiagonal systems without materialising the full
    (g, f, a, l) solution tensor.

    Parameters
    ----------
    a, b, c : (batch, n_layers)
        Tridiagonal coefficients where batch = n_g * n_freq_bins * n_angles.
    d_inner : (n_g, n_freq_bins, n_layers - 1)
        Angle-independent RHS entries for layers 0 .. n_layers-2.
    d_surf : (n_g, n_freq_bins, n_angles)
        Boundary RHS entry for the surface layer.
    angle_weight_a : (n_angles,)
        Angular quadrature weights for mean-intensity accumulation.

    Returns
    -------
    I_H_top : (n_g, n_freq_bins, n_angles)
    x_last : (n_g, n_freq_bins, n_angles)
    x_nm2 : (n_g, n_freq_bins, n_angles)
    J_bol_gfl : (n_g, n_freq_bins, n_layers)
    """
    n = n_layers
    n_gf = n_g * n_freq_bins
    batch = n_gf * n_angles

    def _broadcast_rhs_layer(layer_idx):
        return jnp.broadcast_to(
            d_inner[:, :, layer_idx][:, :, None],
            (n_g, n_freq_bins, n_angles),
        ).reshape(batch)

    def _weighted_gf(x_k):
        return jnp.sum(
            x_k.reshape(n_gf, n_angles) * angle_weight_a[None, :],
            axis=1,
        )

    c0 = c[:, 0] / b[:, 0]
    d0 = _broadcast_rhs_layer(0) / b[:, 0]

    if n > 2:

        def fwd_step(carry, k):
            c_prev, d_prev = carry
            denom = b[:, k] - a[:, k] * c_prev
            safe_den = jnp.where(jnp.abs(denom) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom)
            rhs_k = _broadcast_rhs_layer(k)
            c_k = c[:, k] / safe_den
            d_k = (rhs_k - a[:, k] * d_prev) / safe_den
            return (c_k, d_k), (c_k, d_k)

        _, (c_mid, d_mid) = lax.scan(
            fwd_step,
            (c0, d0),
            jnp.arange(1, n - 1),
        )

        c_nm2 = c_mid[-1, :]
        d_nm2 = d_mid[-1, :]
        c_upto_nm2 = jnp.concatenate([c0[None, :], c_mid], axis=0)
        d_upto_nm2 = jnp.concatenate([d0[None, :], d_mid], axis=0)
    else:
        c_nm2 = c0
        d_nm2 = d0
        c_upto_nm2 = c0[None, :]
        d_upto_nm2 = d0[None, :]

    denom_last = b[:, n - 1] - a[:, n - 1] * c_nm2
    safe_denom_last = jnp.where(
        jnp.abs(denom_last) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom_last
    )
    x_last_flat = (d_surf.reshape(batch) - a[:, n - 1] * d_nm2) / safe_denom_last

    if n == 1:
        weighted_layers = _weighted_gf(x_last_flat)[None, :]
        return (
            (-x_last_flat).reshape(n_g, n_freq_bins, n_angles),
            x_last_flat.reshape(n_g, n_freq_bins, n_angles),
            x_last_flat.reshape(n_g, n_freq_bins, n_angles),
            weighted_layers.T.reshape(n_g, n_freq_bins, n),
        )

    x_nm2_flat = d_upto_nm2[n - 2, :] - c_upto_nm2[n - 2, :] * x_last_flat
    weighted_last = _weighted_gf(x_last_flat)
    weighted_nm2 = _weighted_gf(x_nm2_flat)

    if n > 2:

        def bwd_step(x_next, k):
            x_k = d_upto_nm2[k, :] - c_upto_nm2[k, :] * x_next
            return x_k, _weighted_gf(x_k)

        x0_flat, weighted_rest_desc = lax.scan(
            bwd_step,
            x_nm2_flat,
            jnp.arange(n - 3, -1, -1),
        )
        weighted_layers = jnp.concatenate(
            [
                weighted_rest_desc[::-1, :],
                weighted_nm2[None, :],
                weighted_last[None, :],
            ],
            axis=0,
        )
    else:
        x0_flat = x_nm2_flat
        weighted_layers = jnp.stack([weighted_nm2, weighted_last], axis=0)

    I_H_top = (-x0_flat).reshape(n_g, n_freq_bins, n_angles)
    x_last = x_last_flat.reshape(n_g, n_freq_bins, n_angles)
    x_nm2 = x_nm2_flat.reshape(n_g, n_freq_bins, n_angles)
    J_bol_gfl = weighted_layers.T.reshape(n_g, n_freq_bins, n)

    return I_H_top, x_last, x_nm2, J_bol_gfl


# @jit(static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def solve_tridiagonal_thomas_batched_fused(
    a, b, c, d, mu_weight_flat, n_g, n_freq_bins, n_angles, n_layers
):
    """
    Solves n_batch independent tridiagonal systems simultaneously and fuses
    three downstream post-processing steps into the backward substitution pass,
    eliminating the full (batch, n_layers) solution array. More memory efficient,
    but slowe on CPU.

    Parameters
    ----------
    a, b, c, d : (batch, n_layers)
        Lower diagonal (a[:,0] unused), main diagonal,
        upper diagonal (c[:,-1] unused), RHS.
    mu_weight_flat : (batch,)
        Per-batch-element quadrature weight for J_bol accumulation.
        Equals mu_weight_a[angle_index] for each batch element, tiled over
        (g, f) so that mu_weight_flat[g*n_freq_bins*n_angles + f*n_angles + a]
        = mu_weight_a[a].
    n_g, n_freq_bins : int
        Used to reshape the J_bol accumulator.

    Returns
    -------
    I_H_top_flat      : (batch,)   = -x[0],   the top-boundary intensity.
    x_nm1_flat        : (batch,)   = x[n-1],  solution at the surface layer.
    x_nm2_flat        : (batch,)   = x[n-2],  solution at the penultimate layer.
    J_bol_gfl         : (n_g, n_freq_bins, n_layers)
                        Angle-quadrature-weighted mean intensity per (g,f,l):
                        J_bol_gfl[g,f,l] = sum_a( x[g,f,a,l] * mu_weight_a[a] ).

    Design
    ------
    Forward sweep  : identical to the original solver.
    Backward pass  :
      Rather than emitting x[k] at every scan step (which would materialise the
      full (n-1, batch) solution), we carry only what is needed:
        • x_next      (batch,)        — rolling solution value, used to compute x[k]
        • J_bol_acc   (n_g*n_freq_bins, n_layers) — per-(g,f,l) weighted partial sum;
                      updated at each step via a dynamic .at[:, k].add() scatter.

      x[n-1] and x[n-2] are extracted before the scan; x[0] is the final
      value of x_next in the carry after the scan.

    Memory saved vs. the original solver + downstream uses
    -------------------------------------------------------
      Eliminated : I_J_flat  (batch, n_layers)  — full Thomas solution
                   x_rest    (n-1, batch)        — backward scan output
                   I_J_gfal  (g, f, a, l)        — reshape of I_J_flat
                   Total ≈ 2 × n_g × n_freq_bins × n_angles × n_layers float64
      Added      : J_bol_acc (n_g*n_freq_bins, n_layers) in backward scan carry
                   ≈ n_g × n_freq_bins × n_layers float64  (factor n_angles smaller)
      Net saving ≈ (2*n_angles − 1) × n_g × n_freq_bins × n_layers float64.
    """
    # n_g, n_freq_bins, n_angles, n_layers are all static (concrete Python ints).
    # We never call b.shape for n or derive n_angles from a traced value, so
    # jnp.zeros and jnp.arange receive concrete arguments throughout.
    n = n_layers
    n_gf = n_g * n_freq_bins

    # ----------------------------------------------------------------
    # Forward sweep — identical to the original solver.
    # Produces c_all (n, batch) and d_all (n, batch).
    # ----------------------------------------------------------------
    c0 = c[:, 0] / b[:, 0]  # (batch,)
    d0 = d[:, 0] / b[:, 0]  # (batch,)

    def fwd_step(carry, k):
        c_prev, d_prev = carry
        denom = b[:, k] - a[:, k] * c_prev
        safe_den = jnp.where(jnp.abs(denom) < GRAD_SAFE_DENOM, GRAD_SAFE_DENOM, denom)
        c_k = c[:, k] / safe_den
        d_k = (d[:, k] - a[:, k] * d_prev) / safe_den
        return (c_k, d_k), (c_k, d_k)

    _, (c_rest, d_rest) = lax.scan(fwd_step, (c0, d0), jnp.arange(1, n))
    # c_all, d_all: (n, batch)
    c_all = jnp.concatenate([c0[None, :], c_rest], axis=0)
    d_all = jnp.concatenate([d0[None, :], d_rest], axis=0)

    # ----------------------------------------------------------------
    # Backward substitution — fused with J_bol accumulation.
    #
    # Extract x[n-1] and x[n-2] before the scan so they can be
    # returned without being emitted as scan outputs.
    # ----------------------------------------------------------------

    # Helper: weighted reduction over the angle axis for J_bol accumulation.
    # x_k: (batch,) = (n_gf, n_angles) flattened in C-order.
    # weighted_k: (n_gf,) = sum over angles of x_k * mu_weight_flat.
    def _weighted_gf(x_k):
        # reshape (n_gf * n_angles,) → (n_gf, n_angles), then dot with mu_weight_a
        return (
            x_k.reshape(n_gf, n_angles) * mu_weight_flat.reshape(n_gf, n_angles)
        ).sum(axis=1)
        # shape (n_gf,)

    x_last = d_all[n - 1, :]  # x[n-1], (batch,)
    x_nm2 = d_all[n - 2, :] - c_all[n - 2, :] * x_last  # x[n-2], (batch,)

    # Seed J_bol accumulator with contributions from layers n-1 and n-2.
    J_bol_acc = jnp.zeros((n_gf, n))  # (n_gf, n_layers)
    J_bol_acc = J_bol_acc.at[:, n - 1].set(_weighted_gf(x_last))
    J_bol_acc = J_bol_acc.at[:, n - 2].set(_weighted_gf(x_nm2))

    # Backward scan over k = n-3 down to 0.
    # Carry: (x_next, J_bol_acc)
    # xs:    layer indices k in descending order
    # No scan output (ys=None) — only the carry is used.
    def bwd_step(carry, k):
        x_next, J_bol = carry
        x_k = d_all[k, :] - c_all[k, :] * x_next  # (batch,)
        J_bol = J_bol.at[:, k].add(_weighted_gf(x_k))  # dynamic scatter
        return (x_k, J_bol), None

    (x_0, J_bol_acc), _ = lax.scan(
        bwd_step,
        (x_nm2, J_bol_acc),
        jnp.arange(n - 3, -1, -1),
    )

    # ----------------------------------------------------------------
    # Assemble outputs — no I_J_flat or I_J_gfal ever materialised.
    # ----------------------------------------------------------------
    I_H_top_flat = -x_0  # (batch,)
    J_bol_gfl = J_bol_acc.reshape(n_g, n_freq_bins, n)  # (g, f, l)

    return I_H_top_flat, x_last, x_nm2, J_bol_gfl


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


@partial(jit, static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def solve_tridiagonal_pcr_batched_feautrier_fused(
    a,
    b,
    c,
    d_inner,
    d_surf,
    angle_weight_a,
    n_g,
    n_freq_bins,
    n_angles,
    n_layers,
):
    """PCR-based Feautrier tridiagonal solver optimised for GPU.

    Mirrors the interface of ``solve_tridiagonal_thomas_batched_feautrier_fused``
    but uses the O(log N) parallel cyclic reduction algorithm internally.

    Parameters
    ----------
    a, b, c : (batch, n_layers)
        Tridiagonal coefficients where batch = n_g * n_freq_bins * n_angles.
    d_inner : (n_g, n_freq_bins, n_layers - 1)
        Angle-independent RHS entries for layers 0 .. n_layers-2.
    d_surf : (n_g, n_freq_bins, n_angles)
        Boundary RHS entry for the surface layer.
    angle_weight_a : (n_angles,)
        Angular quadrature weights for mean-intensity accumulation.
    n_g, n_freq_bins, n_angles, n_layers : int
        Static shape parameters.

    Returns
    -------
    I_H_top : (n_g, n_freq_bins, n_angles)
    x_last : (n_g, n_freq_bins, n_angles)
    x_nm2 : (n_g, n_freq_bins, n_angles)
    J_bol_gfl : (n_g, n_freq_bins, n_layers)
    """
    n = n_layers
    n_gf = n_g * n_freq_bins
    batch = n_gf * n_angles

    # Build the full RHS (batch, n_layers) by broadcasting d_inner over
    # angles and placing d_surf at the surface layer.
    d_inner_broadcast = (
        jnp.broadcast_to(
            d_inner[:, :, :, None],  # (g, f, l-1, 1)
            (n_g, n_freq_bins, n_layers - 1, n_angles),
        )
        .transpose(0, 1, 3, 2)
        .reshape(batch, n_layers - 1)
    )  # (batch, l-1)
    d_surf_flat = d_surf.reshape(batch, 1)  # (batch, 1)
    d_full = jnp.concatenate([d_inner_broadcast, d_surf_flat], axis=1)  # (batch, n)

    # Solve with PCR
    x_full = solve_tridiagonal_pcr(a, b, c, d_full)  # (batch, n)

    # Extract outputs matching the Thomas fused solver interface
    x_last_flat = x_full[:, n - 1]  # (batch,)
    x_nm2_flat = x_full[:, n - 2] if n > 1 else x_last_flat
    x0_flat = x_full[:, 0]  # (batch,)

    # J_bol: angle-weighted mean intensity per (g, f, l)
    x_gfal = x_full.reshape(n_gf, n_angles, n)  # (gf, a, l)
    J_bol_gfl = jnp.sum(x_gfal * angle_weight_a[None, :, None], axis=1).reshape(
        n_g, n_freq_bins, n
    )  # (g, f, l)

    I_H_top = (-x0_flat).reshape(n_g, n_freq_bins, n_angles)
    x_last = x_last_flat.reshape(n_g, n_freq_bins, n_angles)
    x_nm2 = x_nm2_flat.reshape(n_g, n_freq_bins, n_angles)

    return I_H_top, x_last, x_nm2, J_bol_gfl


def solve_feautrier_tridiagonal(
    a, b, c, d_inner, d_surf, angle_weight_a, n_g, n_freq_bins, n_angles, n_layers
):
    """Dispatch to the optimal Feautrier tridiagonal solver for the current device.

    Uses the sequential Thomas algorithm on CPU (good sequential throughput)
    and the parallel cyclic reduction algorithm on GPU (O(log N) depth).

    Parameters are identical to ``solve_tridiagonal_thomas_batched_feautrier_fused``.
    """
    if _USE_GPU:
        return solve_tridiagonal_pcr_batched_feautrier_fused(
            a,
            b,
            c,
            d_inner,
            d_surf,
            angle_weight_a,
            n_g,
            n_freq_bins,
            n_angles,
            n_layers,
        )
    return solve_tridiagonal_thomas_batched_feautrier_fused(
        a,
        b,
        c,
        d_inner,
        d_surf,
        angle_weight_a,
        n_g,
        n_freq_bins,
        n_angles,
        n_layers,
    )


def factor_feautrier(a, b, c, n_g, n_freq_bins, n_angles, n_layers):
    """Pre-factor the Feautrier tridiagonal LHS.

    On CPU, delegates to :func:`factor_feautrier_thomas`.
    On GPU, returns ``None`` (PCR has no separable factorisation step).

    Returns
    -------
    factors : tuple or None
        ``(c_prime, inv_denom, a_times_inv_denom)`` on CPU, ``None`` on GPU.
    """
    if _USE_GPU:
        return None
    return factor_feautrier_thomas(a, b, c, n_g, n_freq_bins, n_angles, n_layers)


def solve_feautrier_prefactored(
    factors,
    a,
    b,
    c,
    d_inner,
    d_surf,
    angle_weight_a,
    n_g,
    n_freq_bins,
    n_angles,
    n_layers,
):
    """Solve the Feautrier system, using pre-factored LHS when available.

    On CPU (``factors is not None``), uses :func:`solve_feautrier_thomas_prefactored`
    which skips the LHS forward elimination.  On GPU (``factors is None``), falls
    back to the full PCR solver.

    Parameters
    ----------
    factors : tuple or None
        Output of :func:`factor_feautrier`.  ``None`` triggers the GPU path.
    a, b, c, d_inner, d_surf, angle_weight_a, n_g, n_freq_bins, n_angles, n_layers
        Same as :func:`solve_feautrier_tridiagonal`.
    """
    if factors is not None:
        c_prime, inv_denom, a_times_inv_denom = factors
        return solve_feautrier_thomas_prefactored(
            c_prime,
            inv_denom,
            a_times_inv_denom,
            d_inner,
            d_surf,
            angle_weight_a,
            n_g,
            n_freq_bins,
            n_angles,
            n_layers,
        )
    return solve_tridiagonal_pcr_batched_feautrier_fused(
        a,
        b,
        c,
        d_inner,
        d_surf,
        angle_weight_a,
        n_g,
        n_freq_bins,
        n_angles,
        n_layers,
    )


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
):
    """Build the per-chunk Feautrier solver state, Lambda diagonal and surface
    stencil coefficient from layer-major delta-optical-depths.

    Dispatches to the fused scan on CPU and to the materialising PCR
    preparation on GPU; the returned state is consumed by
    :func:`solve_feautrier_system_lm`.
    """
    prepare = (
        _prepare_feautrier_system_lm_gpu if _USE_GPU
        else _prepare_feautrier_system_lm_cpu
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
):
    """Solve the prepared layer-major Feautrier system for one source iterate.

    ``state`` must come from :func:`prepare_feautrier_system_lm` on the same
    backend. Returns ``(I_H_top, x_last, x_nm2, J_bol_lm)`` with the angular
    arrays shaped (n_gf, n_angles) and ``J_bol_lm`` shaped (n_layers, n_gf).
    """
    solve = (
        _solve_feautrier_system_lm_gpu if _USE_GPU else _solve_feautrier_system_lm_cpu
    )
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
