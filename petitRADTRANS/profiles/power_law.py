import jax.numpy as jnp
from jax import jit, config, Array
from jax.typing import ArrayLike


def power_law_temperature_profile(pressures: ArrayLike, alpha: float, T0: float) -> Array:
    """
    Compute a power law profile for temperature; log(T) = a*log(P) + b.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        pressures (array-like): An array or list of pressure data points.
        alpha (float): power law exponent (how steep is the profile)
        T0 (float): multiplicative factor (offsets the profile)

    Returns:
        temperatures (array): temperature values for each pressure value
    """
    temperatures = T0*(pressures/pressures[0])**alpha
    return temperatures