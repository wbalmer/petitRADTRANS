"""Single source of truth for petitRADTRANS JAX floating-point dtypes.

Import ``WORKING_JAX_DTYPE`` for memory-sensitive storage and routine working
arrays. Import ``FEAUTRIER_JAX_DTYPE`` for the Feautrier scattering solver and
other numerically sensitive kernels that should honor ``jax_enable_x64``.

``JAX_DTYPE`` is kept as a backward-compatible alias for
``FEAUTRIER_JAX_DTYPE`` while the codebase is migrated to explicit per-surface
dtype choices.
"""

from jax import config
import jax.numpy as jnp

WORKING_JAX_DTYPE = jnp.float32

FEAUTRIER_JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    FEAUTRIER_JAX_DTYPE = jnp.float64

JAX_DTYPE = FEAUTRIER_JAX_DTYPE
