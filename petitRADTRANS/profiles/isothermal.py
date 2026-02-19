import jax.numpy as jnp
from jax import jit, config
from jax.typing import ArrayLike

def temperature_profile_function_isothermal(pressures: ArrayLike, temperature: float) -> ArrayLike:
    pressures = jnp.asarray(pressures)
    temperature = jnp.asarray(temperature)
    return jnp.ones_like(pressures) * temperature