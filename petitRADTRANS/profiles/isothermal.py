import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def temperature_profile_function_isothermal(pressures: ArrayLike, temperature: float) -> Array:
    """
    Returns an isothermal temperature profile.

    Args:
        pressures: An array of pressures. The shape is preserved for the output temperatures.
        temperature: The constant temperature value for the profile.

    Returns:
        An array of temperatures with the same shape as `pressures`, all set to the input `temperature`.
    """
    pressures = jnp.asarray(pressures)
    temperature = jnp.asarray(temperature)
    return jnp.ones_like(pressures) * temperature
