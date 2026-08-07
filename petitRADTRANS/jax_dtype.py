"""Single source of truth for petitRADTRANS JAX floating-point dtypes.

Import ``WORKING_JAX_DTYPE`` for memory-sensitive storage and routine working
arrays (here we use single precision).
Import ``FEAUTRIER_JAX_DTYPE`` for the Feautrier radiative transfer solver and
other numerically sensitive functions that need double precision to work.
"""


from jax import config
import jax.numpy as jnp

WORKING_JAX_DTYPE = jnp.float32

FEAUTRIER_JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    FEAUTRIER_JAX_DTYPE = jnp.float64
