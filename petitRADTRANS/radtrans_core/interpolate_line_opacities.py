"""
JAX implementation of fortran_inputs.interpolate_line_opacities
"""
import jax.numpy as jnp
from jax import lax, vmap, jit, config

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64


@jit
def interpolate_line_opacities(
    pressures,
    temperatures,
    temperature_pressure_grid_input,
    opacity_grid,
    temperature_grid_size,
    pressure_grid_size,
):
    """
    Interpolate an opacity grid at a given temperature profile.

    Parameters:
    pressures : jax.numpy.ndarray
        Array of pressures for each layer.
    temperatures : jax.numpy.ndarray
        Array of temperatures for each layer.
    temperature_pressure_grid_input : jax.numpy.ndarray
        Temperature-pressure grid for interpolation.
    opacity_grid : jax.numpy.ndarray
        Opacity grid for interpolation.

    Returns:
    interpolated_opacities : jax.numpy.ndarray
        Interpolated opacities.
    """

    n_pressures = temperature_pressure_grid_input.shape[1]
    pressure_grid = temperature_pressure_grid_input[0, :, 1]
    temperature_grid = temperature_pressure_grid_input[:, 0, 0]
    temperature_pressure_grid = temperature_pressure_grid_input.reshape(-1, 2)

    # Find temperature and pressure indices
    i_temperatures = jnp.searchsorted(temperature_grid, temperatures) - 1
    i_pressures = jnp.searchsorted(pressure_grid, pressures) - 1

    # Clamp indices to valid ranges
    i_temperatures = jnp.clip(i_temperatures, 0, temperature_grid_size - 2)
    i_pressures = jnp.clip(i_pressures, 0, pressure_grid_size - 2)

    # Calculate index offsets
    j_t0_p0 = i_temperatures * n_pressures + i_pressures
    j_t0_p1 = j_t0_p0 + 1
    j_t1_p0 = (i_temperatures + 1) * n_pressures + i_pressures
    j_t1_p1 = j_t1_p0 + 1

    # Gather required opacity grid values
    y_p_low_t0 = opacity_grid[..., j_t0_p0]
    y_p_high_t0 = opacity_grid[..., j_t0_p1]
    y_p_low_t1 = opacity_grid[..., j_t1_p0]
    y_p_high_t1 = opacity_grid[..., j_t1_p1]

    # Calculate interpolation factors
    d_pressures_t0 = temperature_pressure_grid[j_t0_p1, 1] - temperature_pressure_grid[j_t0_p0, 1]
    d_pressures_t1 = temperature_pressure_grid[j_t1_p1, 1] - temperature_pressure_grid[j_t1_p0, 1]
    d_temperatures = temperature_pressure_grid[j_t1_p0, 0] - temperature_pressure_grid[j_t0_p0, 0]

    # Handle division by zero
    dx_0 = jnp.where(d_pressures_t0 == 0.0,
                     0.0,
                     (pressures - temperature_pressure_grid[j_t0_p0, 1]) / d_pressures_t0)
    dx_1 = jnp.where(d_pressures_t1 == 0.0,
                     0.0,
                     (pressures - temperature_pressure_grid[j_t1_p0, 1]) / d_pressures_t1)
    t_factor = jnp.where(d_temperatures == 0,
                         0,
                         (temperatures - temperature_pressure_grid[j_t0_p0, 0]) / d_temperatures)

    # Interpolate along pressure
    y_t_lows = y_p_low_t0 + dx_0 * (y_p_high_t0 - y_p_low_t0)
    y_t_lows = jnp.where(pressures > jnp.max(temperature_pressure_grid[:, 1]),
                         y_p_high_t0,
                         y_t_lows)
    y_t_lows = jnp.where(pressures < jnp.min(temperature_pressure_grid[:, 1]),
                         y_p_low_t0,
                         y_t_lows)

    y_t_highs = y_p_low_t1 + dx_1 * (y_p_high_t1 - y_p_low_t1)
    y_t_highs = jnp.where(pressures > jnp.max(temperature_pressure_grid[:,1]),
                          y_p_high_t1,
                          y_t_highs)
    y_t_highs = jnp.where(pressures < jnp.min(temperature_pressure_grid[:,1]),
                          y_p_low_t1,
                          y_t_highs)

    # Interpolate along temperature
    interpolated_opacities = y_t_lows + t_factor[None, None, :] * (y_t_highs - y_t_lows)
    interpolated_opacities = jnp.where(temperatures > jnp.max(temperature_pressure_grid[:, 0]),
                                       y_t_highs,
                                       interpolated_opacities)
    interpolated_opacities = jnp.where(temperatures < jnp.min(temperature_pressure_grid[:, 0]),
                                       y_t_lows,
                                       interpolated_opacities)
    return interpolated_opacities


@jit
def _interpolate_stacked_species_opacities(
    pressures,
    temperatures,
    line_opacities_grid,
    line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size,
    line_opacities_pressure_grid_size,
):
    interpolated_species_opacities = vmap(
        lambda opacity_grid, temperature_pressure_grid, temperature_grid_size, pressure_grid_size: (
            interpolate_line_opacities(
                pressures,
                temperatures,
                temperature_pressure_grid,
                opacity_grid,
                temperature_grid_size,
                pressure_grid_size,
            )
        ),
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )(
        line_opacities_grid,
        line_opacities_temperature_pressure_grid,
        line_opacities_temperature_grid_size,
        line_opacities_pressure_grid_size,
    )

    return jnp.moveaxis(interpolated_species_opacities, 0, 2)


def interpolate_species_opacities(
    pressures, temperatures, n_g, n_frequencies, line_opacities_grid, line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size, line_opacities_pressure_grid_size
):
    # Interpolate line opacities to given temperature structure.
    n_layers = pressures.size
    if isinstance(line_opacities_grid, dict):
        n_line_species = len(line_opacities_grid)

        if n_line_species > 0:
            line_opacities = jnp.zeros(
                (n_g, n_frequencies, n_line_species, n_layers), dtype=JAX_DTYPE)

            for i, species in enumerate(line_opacities_grid):
                line_opacities_tp_grid = \
                    line_opacities_temperature_pressure_grid[species].reshape(
                        line_opacities_temperature_grid_size[species],
                        line_opacities_pressure_grid_size[species],
                        2)
                line_opacities = line_opacities.at[:, :, i, :].set(interpolate_line_opacities(
                    pressures,
                    temperatures,
                    line_opacities_tp_grid,
                    line_opacities_grid[species],
                    line_opacities_temperature_grid_size[species],
                    line_opacities_pressure_grid_size[species],
                ))
        else:
            line_opacities = jnp.zeros(
                (n_g, n_frequencies, 1, n_layers), dtype=JAX_DTYPE)

        return line_opacities

    n_line_species = line_opacities_grid.shape[0]

    if n_line_species > 0:
        return _interpolate_stacked_species_opacities(
            pressures=pressures,
            temperatures=temperatures,
            line_opacities_grid=line_opacities_grid,
            line_opacities_temperature_pressure_grid=line_opacities_temperature_pressure_grid,
            line_opacities_temperature_grid_size=line_opacities_temperature_grid_size,
            line_opacities_pressure_grid_size=line_opacities_pressure_grid_size,
        )

    return jnp.zeros(
        (n_g, n_frequencies, 1, n_layers), dtype=JAX_DTYPE)
