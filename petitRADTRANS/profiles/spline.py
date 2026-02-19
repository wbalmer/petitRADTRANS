import jax.numpy as jnp
from jax import jit, config
from jax.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from interpax import interp1d


def cubic_spline_profile(
    press: ArrayLike,
    temperature_points: ArrayLike,
    gamma: float,
    nnodes: int = 0
) -> tuple[ArrayLike, float]:
    """
    Compute a cubic spline profile for temperature based on pressure points.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the cubic spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """

    # TODO benchmark interp1d vs PchipInterpolator for cubic spline interpolation in terms of speed and accuracy
    cs = interp1d(
        x=jnp.linspace(
            jnp.log10(press[0]),
            jnp.log10(press[-1]),
            nnodes + 2
        ),
        y=temperature_points,
        method='monotonic'
    )

    interpolated_temps: ArrayLike = cs(jnp.log10(press))
    prior: float = temperature_curvature_prior(press, interpolated_temps, gamma)

    return interpolated_temps, prior


def linear_spline_profile(
    press: ArrayLike,
    temperature_points: ArrayLike,
    gamma: float,
    nnodes: int = 0
) -> tuple[ArrayLike, float]:
    """
    Compute a linear spline profile for temperature based on pressure points.

    This function computes a linear spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the linear spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """
    interpolated_temps = jnp.interp(
        jnp.log10(press),
        jnp.linspace(
            jnp.log10(press[0]),
            jnp.log10(press[-1]),
            int(nnodes) + 2
        ),
        temperature_points
    )

    prior = temperature_curvature_prior(press, interpolated_temps, gamma)

    return interpolated_temps, prior


def temperature_curvature_prior(press, temps, gamma):
    """
    Compute a curvature prior for a temperature-pressure profile.

    This function calculates a curvature prior for a temperature-pressure profile,
    penalizing deviations from a smooth, low-curvature profile, based on Line 2015

    Args:
        press (array-like): An array or list of pressure data points.
        temps (array-like): An array or list of temperature data points.
        gamma (float): The curvature penalization factor.

    Returns:
        float: The curvature prior value.
    """
    weighted_temp_prior = -0.5 * jnp.sum((temps[2:] - 2 * temps[1:-1] + temps[:-2]) ** 2) / gamma
    weighted_temp_prior -= 0.5 * jnp.log(2 * jnp.pi * gamma)

    return weighted_temp_prior