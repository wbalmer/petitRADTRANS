import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from interpax import interp1d


def cubic_spline_temperature_profile(
    pressures: ArrayLike,
    temperature_points: ArrayLike,
    gamma: float,
    nnodes: int = 0
) -> tuple[Array, float]:
    """
    Compute a cubic spline profile for temperature based on pressure points.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        pressures: An array of pressure data points.
        temperature_points: An array of temperature data points.
        gamma: A parameter controlling the curvature of the spline.
        nnodes: Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        A tuple containing:
            - interpolated_temperatures: Interpolated temperature values based on the cubic spline.
            - prior: Curvature prior value calculated for the spline.
    """
    # TODO benchmark interp1d vs PchipInterpolator for cubic spline interpolation in terms of speed and accuracy
    cs = interp1d(
        x=jnp.linspace(
            jnp.log10(pressures[0]),
            jnp.log10(pressures[-1]),
            nnodes + 2
        ),
        y=temperature_points,
        method='monotonic'
    )

    interpolated_temperatures: ArrayLike = cs(jnp.log10(pressures))
    prior: float = temperature_curvature_prior(interpolated_temperatures, gamma)

    return interpolated_temperatures, prior


def linear_spline_temperature_profile(
    pressures: ArrayLike,
    temperature_points: ArrayLike,
    gamma: float,
    nnodes: int = 0
) -> tuple[ArrayLike, float]:
    """
    Compute a linear spline profile for temperature based on pressure points.

    This function computes a linear spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        pressures: An array of pressure data points.
        temperature_points: An array of temperature data points.
        gamma: A parameter controlling the curvature of the spline.
        nnodes: Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        A tuple containing:
            - interpolated_temperatures: Interpolated temperature values based on the linear spline.
            - prior: Curvature prior value calculated for the spline.
    """
    interpolated_temperatures = jnp.interp(
        jnp.log10(pressures),
        jnp.linspace(
            jnp.log10(pressures[0]),
            jnp.log10(pressures[-1]),
            int(nnodes) + 2
        ),
        temperature_points
    )

    prior = temperature_curvature_prior(interpolated_temperatures, gamma)

    return interpolated_temperatures, prior


def temperature_curvature_prior(temperatures: ArrayLike, gamma: float) -> float:
    """
    Compute a curvature prior for a temperature-pressure profile.

    This function calculates a curvature prior for a temperature-pressure profile,
    penalizing deviations from a smooth, low-curvature profile, based on Line 2015.

    Args:
        temperatures: An array of temperature data points.
        gamma: The curvature penalization factor.

    Returns:
        The curvature prior value.
    """
    weighted_temp_prior = -0.5 * jnp.sum((temperatures[2:] - 2 * temperatures[1:-1] + temperatures[:-2]) ** 2) / gamma
    weighted_temp_prior -= 0.5 * jnp.log(2 * jnp.pi * gamma)

    return weighted_temp_prior
