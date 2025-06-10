"""
JAX implementation of fortran_core.math
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst
from  petitRADTRANS.fortran_radtrans_core import math as fmath


@jit
def wrap_tridag(a, b, c, r, I_plus, I_minus):
    # Set initial value of r
    # freq, struct, glen, nmu
    r_updated = jnp.zeros((r.shape[0], r.shape[1], r.shape[2], I_plus.shape[2]))
    r_updated = r_updated.at[:, :, :, :].set(r[:, :, :, None])
    r_updated = r_updated.at[:, -1, :, :].set(0.5 * (I_plus + I_minus[:, :, None]))
    # Calculate I_J and I_H using vectorized computation
    I_J = vmap(vmap(vmap(
                wrap_callback,
                in_axes=(1, 1, 1, 1)),
            in_axes=(2, 2, 2, 2)),
        in_axes=(0, 0, 0, 0))(a, b, c, r_updated)
    return I_J

def wrap_callback(a, b, c, r):
    result_shape = ShapeDtypeStruct(r.shape, r.dtype)
    return pure_callback(fmath.tridag_own, result_shape, a, b, c, r)


@jit
def solve_tridiagonal_system(a, b, c, d):
    '''
    Solves a tridiagonal system of linear equations Ax = d using the Thomas algorithm (TDMA).

    Args:
        a (jax.numpy.ndarray): The lower diagonal of matrix A. If length N, a[0] is ignored.
        b (jax.numpy.ndarray): The main diagonal of matrix A. Length N.
        c (jax.numpy.ndarray): The upper diagonal of matrix A. If length N, c[N-1] is ignored.
        d (jax.numpy.ndarray): The right-hand side vector. Length N.

    Returns:
        jax.numpy.ndarray: The solution vector x. Length N.

    Notes:
    - Assumes a, b, c, d are JAX arrays of equal length N.
    - For the matrix A:
        - A_ii = b_i
        - A_{i,i-1} = a_i  (for i = 1 to N-1, effectively using a[1]...a[N-1] from input `a`)
        - A_{i,i+1} = c_i  (for i = 0 to N-2, effectively using c[0]...c[N-2] from input `c`)
    - Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d)  # number of equations
    # Ensure inputs are JAX arrays. If they are already, this is a no-op.
    ac, bc_fwd, cc, dc_fwd = map(jnp.array, (a, b, c, d))

    # Forward elimination
    # Original ac and cc are passed through, bc_fwd and dc_fwd are modified.
    _, bc_fwd, _, dc_fwd = lax.fori_loop(
        1,
        nf,
        _thomas_forward_elimination_body,
        (ac, bc_fwd, cc, dc_fwd)  # cc is original upper diagonal
    )

    # Backward substitution
    x = jnp.zeros(nf, dtype=d.dtype)
    x = x.at[-1].set(dc_fwd[-1] / bc_fwd[-1])

    # Loop from nf-2 down to 0
    # The loop variable `i` goes from 0 to nf-2.
    # We calculate x[nf-2-i] in each iteration.
    x = lax.fori_loop(
        0,
        nf - 1,  # Iterates nf-1 times, for i = 0, ..., nf-2
        lambda i, x_carry: _thomas_backward_substitution_body(i, x_carry, nf, cc, bc_fwd, dc_fwd),
        x
    )
    return x


def _thomas_forward_elimination_body(it, val):
    ac, bc, cc, dc = val
    # it ranges from 1 to nf-1
    # ac[it] is a_i in standard notation (for row i)
    # bc[it-1] is the modified b_{i-1}
    # cc[it-1] is c_{i-1} in standard notation (for row i-1)
    mc = ac[it] / bc[it-1]
    bc_new_it = bc[it] - mc * cc[it-1]
    dc_new_it = dc[it] - mc * dc[it-1]
    bc = bc.at[it].set(bc_new_it)
    dc = dc.at[it].set(dc_new_it)
    return (ac, bc, cc, dc)


def _thomas_backward_substitution_body(i, x_carry, nf, cc, bc_fwd, dc_fwd):
    # i iterates from 0 to nf-2 (for a total of nf-1 iterations)
    # We are calculating x[idx] where idx goes from nf-2 down to 0.
    idx = nf - 2 - i
    # x_carry[idx+1] is x_{idx+1} which has been computed in the previous step (or initialized for x[nf-1])
    # cc[idx] is the original upper diagonal element c_idx
    # bc_fwd[idx] is the modified main diagonal element from forward elimination
    # dc_fwd[idx] is the modified RHS element from forward elimination
    x_val = (dc_fwd[idx] - cc[idx] * x_carry[idx+1]) / bc_fwd[idx]
    x_carry = x_carry.at[idx].set(x_val)
    return x_carry
