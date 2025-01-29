import numpy as np
import jax.numpy as jnp
from numba import njit
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst
from  petitRADTRANS.fortran_radtrans_core import math as fmath

"""
JAX implementation of fortran_core.math
"""

@jit
def wrap_tridag(a, b, c, r, I_plus, I_minus):
    # Set initial value of r
    # freq, struct, glen, nmu
    r_updated = jnp.zeros((r.shape[0],r.shape[1],r.shape[2],I_plus.shape[2]))
    r_updated = r_updated.at[:,:,:,:].set(r[:,:,:,None])
    r_updated = r_updated.at[:,-1,:,:].set(0.5 * (I_plus + I_minus[:, :, None]))
    # Calculate I_J and I_H using vectorized computation
    I_J = vmap(vmap(vmap(wrap_callback, 
                    in_axes=(1, 1, 1, 1)),
                    in_axes=(2, 2, 2, 2)), 
                    in_axes=(0, 0, 0, 0))(a, b, c, r_updated)
    return I_J

def wrap_callback(a, b, c, r):
    result_shape = ShapeDtypeStruct(r.shape, r.dtype)
    return pure_callback(fmath.tridag_own, result_shape, a,b,c,r)

@jit 
def tridag_own(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(jnp.array, (a, b, c, d)) # copy arrays
    ac,bc,cc,dc = lax.fori_loop(1,nf,internal_tridag,(ac,bc,cc,dc))     
    xc = bc
    xc = xc.at[-1].set(dc[-1]/bc[-1])
    xcr = jnp.flip(xc)
    bcr = jnp.flip(bc)
    ccr = jnp.flip(cc)
    dcr = jnp.flip(dc)
    xcr,bcr,ccr,dcr = lax.fori_loop(0,nf-1,internal_tridag_reverse,(xcr,bcr,ccr,dcr))
    return jnp.flip(xcr)

def internal_tridag_reverse(il,val):
    xc,bc,cc,dc = val
    xc = xc.at[il].set((dc[il]-cc[il]*xc[il-1])/bc[il])
    return (xc,bc,cc,dc)

def internal_tridag(it,val):
    ac,bc,cc,dc = val
    mc = ac[it]/bc[it-1]
    bc = bc.at[it].add(- mc*cc[it-1])
    dc = dc.at[it].add(- mc*dc[it-1])
    return (ac,bc,cc,dc)