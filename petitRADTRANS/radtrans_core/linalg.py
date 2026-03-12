import jax.numpy as jnp
from jax import lax, jit, config

TINIEST = jnp.finfo(jnp.float32).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float32).max)
if config.jax_enable_x64:
    TINIEST = jnp.finfo(jnp.float64).tiny
    HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float64).max)


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
        d_p = d_p.at[..., i].set((d[..., i] - a[..., i] * d_p[..., i - 1]) / denominator)

        return c_p, d_p

    c_p, d_p = lax.fori_loop(1, n - 1, forward_sweep, (c_p, d_p))

    # Handle the last element
    denominator = b[..., n - 1] - a[..., n - 1] * c_p[..., n - 2]
    d_p = d_p.at[..., n - 1].set((d[..., n - 1] - a[..., n - 1] * d_p[..., n - 2]) / denominator)

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
    c0 = c[:, 0] / b[:, 0]   # (batch,)
    d0 = d[:, 0] / b[:, 0]   # (batch,)

    def fwd_step(carry, k):
        c_prev, d_prev = carry
        denom    = b[:, k] - a[:, k] * c_prev
        safe_den = jnp.where(jnp.abs(denom) < TINIEST, TINIEST, denom)
        c_k = c[:, k] / safe_den
        d_k = (d[:, k] - a[:, k] * d_prev) / safe_den
        return (c_k, d_k), (c_k, d_k)

    _, (c_rest, d_rest) = lax.scan(fwd_step, (c0, d0), jnp.arange(1, n))
    # prepend layer-0 results  →  (n, batch) each
    c_all = jnp.concatenate([c0[None, :], c_rest], axis=0)
    d_all = jnp.concatenate([d0[None, :], d_rest], axis=0)

    # ---- back substitution ----
    x_last = d_all[n - 1, :]   # (batch,)

    def bwd_step(x_next, k):
        x_k = d_all[k, :] - c_all[k, :] * x_next
        return x_k, x_k

    _, x_rest = lax.scan(bwd_step, x_last, jnp.arange(n - 2, -1, -1))
    # x_rest is (n-1, batch) in reverse order; rebuild full solution
    x_all = jnp.concatenate([x_rest[::-1, :], x_last[None, :]], axis=0)  # (n, batch)

    return x_all.T   # (batch, n)


#@jit(static_argnames=("n_g", "n_freq_bins", "n_angles", "n_layers"))
def solve_tridiagonal_thomas_batched_fused(a, b, c, d, mu_weight_flat, n_g, n_freq_bins, n_angles, n_layers):
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
    n    = n_layers
    n_gf = n_g * n_freq_bins

    # ----------------------------------------------------------------
    # Forward sweep — identical to the original solver.
    # Produces c_all (n, batch) and d_all (n, batch).
    # ----------------------------------------------------------------
    c0 = c[:, 0] / b[:, 0]   # (batch,)
    d0 = d[:, 0] / b[:, 0]   # (batch,)

    def fwd_step(carry, k):
        c_prev, d_prev = carry
        denom    = b[:, k] - a[:, k] * c_prev
        safe_den = jnp.where(jnp.abs(denom) < TINIEST, TINIEST, denom)
        c_k      = c[:, k] / safe_den
        d_k      = (d[:, k] - a[:, k] * d_prev) / safe_den
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
        return (x_k.reshape(n_gf, n_angles) * mu_weight_flat.reshape(n_gf, n_angles)).sum(axis=1)
        # shape (n_gf,)

    x_last = d_all[n - 1, :]                              # x[n-1], (batch,)
    x_nm2  = d_all[n - 2, :] - c_all[n - 2, :] * x_last  # x[n-2], (batch,)

    # Seed J_bol accumulator with contributions from layers n-1 and n-2.
    J_bol_acc = jnp.zeros((n_gf, n))                      # (n_gf, n_layers)
    J_bol_acc = J_bol_acc.at[:, n - 1].set(_weighted_gf(x_last))
    J_bol_acc = J_bol_acc.at[:, n - 2].set(_weighted_gf(x_nm2))

    # Backward scan over k = n-3 down to 0.
    # Carry: (x_next, J_bol_acc)
    # xs:    layer indices k in descending order
    # No scan output (ys=None) — only the carry is used.
    def bwd_step(carry, k):
        x_next, J_bol = carry
        x_k    = d_all[k, :] - c_all[k, :] * x_next      # (batch,)
        J_bol  = J_bol.at[:, k].add(_weighted_gf(x_k))    # dynamic scatter
        return (x_k, J_bol), None

    (x_0, J_bol_acc), _ = lax.scan(
        bwd_step,
        (x_nm2, J_bol_acc),
        jnp.arange(n - 3, -1, -1),
    )

    # ----------------------------------------------------------------
    # Assemble outputs — no I_J_flat or I_J_gfal ever materialised.
    # ----------------------------------------------------------------
    I_H_top_flat = -x_0                                    # (batch,)
    J_bol_gfl    = J_bol_acc.reshape(n_g, n_freq_bins, n)  # (g, f, l)

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
    initial_val = (a_flat, b_flat, c_flat, d_flat, 1)

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
        # Use jnp.where to avoid division by zero if b_left/b_right are zero
        # This is safe for diagonally-dominant systems.
        alpha = jnp.where(b_left != 0, -a_pcr / b_left, 0.)
        gamma = jnp.where(b_right != 0, -c_pcr / b_right, 0.)

        # --- Update the diagonals and the right-hand side ---
        # The update corresponds to climinating off-diagonal elements
        b_new = b_pcr + alpha * c_left + gamma * a_right
        d_new = d_pcr + alpha * d_left + gamma * d_right
        a_new = alpha * a_left
        c_new = gamma * c_right

        return a_new, b_new, c_new, d_new, stride * 2

    def pcr_cond_fun(val):
        """Loop until stride is equal to or greater than N."""
        *_, stride = val
        return stride < N

    # --- Run the reduction loop ---
    a_final, b_final, c_final, d_final, _ = lax.while_loop(
        pcr_cond_fun, pcr_body_fun, initial_val
    )

    # --- Final solve ---
    # After reduction, the matrix is diagonal (a_final and c_final are zero)
    # The solution is simply d / b
    # Add epsilon to avoid division by zero in case of singular matrices.
    x_flat = d_final / (b_final + 1e-12)

    # Reshape solution back to original batch dimensions
    return x_flat.reshape(original_shape)
