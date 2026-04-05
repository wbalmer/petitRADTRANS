"""Single source of truth for the active JAX floating-point dtype.

Import ``JAX_DTYPE`` from this module everywhere in the petitRADTRANS codebase
instead of repeating the ``jax_enable_x64`` check in each file.

When JAX x64 mode is **disabled** (the default on all accelerators) the dtype
is ``jnp.float32``.  When ``jax_enable_x64=True`` the dtype is ``jnp.float64``.
"""

from jax import config
import jax.numpy as jnp

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64
