import jax.numpy as jnp
from jax import lax, jit

HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float32).max)

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
