"""
JAX implementation of fortran_inputs.interpolate_line_opacities
"""

import jax.numpy as jnp
from jax import lax, vmap, jit

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE

# Opacity-cube memory layouts shared with radiative_transfer_core.
LAYOUT_NG_F_S_L = "ng_f_s_l"  # (n_g, n_frequencies, n_species, n_layers)
LAYOUT_FL_S_NG = "fl_s_ng"  # (n_frequencies * n_layers, n_species, n_g)


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

    # Grid bounds, hoisted: each was otherwise recomputed at every clamp below.
    pressure_min = jnp.min(temperature_pressure_grid[:, 1])
    pressure_max = jnp.max(temperature_pressure_grid[:, 1])
    temperature_min = jnp.min(temperature_pressure_grid[:, 0])
    temperature_max = jnp.max(temperature_pressure_grid[:, 0])

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
    d_pressures_t0 = (
        temperature_pressure_grid[j_t0_p1, 1] - temperature_pressure_grid[j_t0_p0, 1]
    )
    d_pressures_t1 = (
        temperature_pressure_grid[j_t1_p1, 1] - temperature_pressure_grid[j_t1_p0, 1]
    )
    d_temperatures = (
        temperature_pressure_grid[j_t1_p0, 0] - temperature_pressure_grid[j_t0_p0, 0]
    )

    # Handle division by zero
    dx_0 = jnp.where(
        d_pressures_t0 == 0.0,
        0.0,
        (pressures - temperature_pressure_grid[j_t0_p0, 1]) / d_pressures_t0,
    )
    dx_1 = jnp.where(
        d_pressures_t1 == 0.0,
        0.0,
        (pressures - temperature_pressure_grid[j_t1_p0, 1]) / d_pressures_t1,
    )
    t_factor = jnp.where(
        d_temperatures == 0,
        0,
        (temperatures - temperature_pressure_grid[j_t0_p0, 0]) / d_temperatures,
    )

    # Interpolate along pressure
    y_t_lows = y_p_low_t0 + dx_0 * (y_p_high_t0 - y_p_low_t0)
    y_t_lows = jnp.where(pressures > pressure_max, y_p_high_t0, y_t_lows)
    y_t_lows = jnp.where(pressures < pressure_min, y_p_low_t0, y_t_lows)

    y_t_highs = y_p_low_t1 + dx_1 * (y_p_high_t1 - y_p_low_t1)
    y_t_highs = jnp.where(pressures > pressure_max, y_p_high_t1, y_t_highs)
    y_t_highs = jnp.where(pressures < pressure_min, y_p_low_t1, y_t_highs)

    # Interpolate along temperature
    interpolated_opacities = y_t_lows + t_factor[None, None, :] * (y_t_highs - y_t_lows)
    interpolated_opacities = jnp.where(
        temperatures > temperature_max, y_t_highs, interpolated_opacities
    )
    interpolated_opacities = jnp.where(
        temperatures < temperature_min, y_t_lows, interpolated_opacities
    )

    return interpolated_opacities


@jit
def compute_line_opacity_interpolation_nodes(
    pressures,
    temperatures,
    temperature_pressure_grid,
    temperature_grid_size,
    pressure_grid_size,
    node_offsets,
):
    """Locate the atmospheric layers in the temperature-pressure grid of each line species.

    The line opacities of all species are stored in one table, whose last axis runs over the
    temperature-pressure points ("nodes") of all species, one species after the other, without padding
    (see ``Radtrans._set_packed_line_opacities``). The opacity of a species at a layer is the bilinear
    interpolation of its opacities at the 4 nodes surrounding the layer, i.e. the sum of these 4 opacities
    weighted by the values returned here. Outside the grid of a species, the opacity at the closest
    boundary of the grid is used (constant extrapolation), which corresponds to weights of 0 or 1.

    Writing the interpolation as a weighted sum, rather than as nested linear interpolations
    ``low + x * (high - low)``, avoids the cancellation that the latter suffers from when ``high`` is much
    smaller than ``low``, and lets other per-layer factors (the mass fractions) be folded into the weights.

    Args:
        pressures: (cgs) pressures of the atmospheric layers, of shape (n_layers,)
        temperatures: (K) temperatures of the atmospheric layers, of shape (n_layers,)
        temperature_pressure_grid:
            Temperatures and pressures of the grid of each species, of shape
            (n_species, max_n_temperatures, max_n_pressures, 2), padded by repeating the last valid value.
        temperature_grid_size: number of temperatures of the grid of each species, of shape (n_species,)
        pressure_grid_size: number of pressures of the grid of each species, of shape (n_species,)
        node_offsets: index of the first node of each species in the opacity table, of shape (n_species,)

    Returns:
        node_indices: int32 array of shape (4, n_species, n_layers), indices of the nodes in the table
        node_weights: array of shape (4, n_species, n_layers), in the working precision
    """

    def _locate(grid, values, grid_size):
        # Index of the lower node of the interval containing each value, and position within it
        i_low = jnp.clip(
            jnp.searchsorted(grid, values) - 1, 0, jnp.maximum(grid_size - 2, 0)
        )
        i_high = jnp.minimum(i_low + 1, grid_size - 1)
        low = grid[i_low]
        interval = grid[i_high] - low
        degenerate = interval == 0.0
        position = jnp.where(
            degenerate, 0.0, (values - low) / jnp.where(degenerate, 1.0, interval)
        )

        # Constant extrapolation outside the grid
        return i_low, i_high, jnp.clip(position, 0.0, 1.0)

    def _species_nodes(grid, n_temperatures, n_pressures, offset):
        i_t0, i_t1, y = _locate(grid[:, 0, 0], temperatures, n_temperatures)
        i_p0, i_p1, x = _locate(grid[0, :, 1], pressures, n_pressures)

        indices = offset + jnp.stack(
            (
                i_t0 * n_pressures + i_p0,
                i_t0 * n_pressures + i_p1,
                i_t1 * n_pressures + i_p0,
                i_t1 * n_pressures + i_p1,
            )
        )
        weights = jnp.stack(
            (
                (1.0 - y) * (1.0 - x),
                (1.0 - y) * x,
                y * (1.0 - x),
                y * x,
            )
        )

        return indices.astype(jnp.int32), weights

    node_indices, node_weights = vmap(_species_nodes, out_axes=1)(
        temperature_pressure_grid,
        temperature_grid_size,
        pressure_grid_size,
        node_offsets,
    )

    # The positions are evaluated in the precision of the pressures and temperatures. The opacities
    # are stored in the working precision: weights of a higher precision would only promote the
    # interpolated opacities to that precision, doubling their memory footprint for no benefit.
    return node_indices, node_weights.astype(WORKING_JAX_DTYPE)


def _weighted_nodes_sum(opacity_grid, node_indices, node_weights):
    """Sum the opacities at the 4 nodes surrounding each layer, weighted by ``node_weights``.

    Args:
        opacity_grid: packed opacity table, of shape (n_g, n_frequencies, n_nodes)
        node_indices: int array of shape (4, ...), indices of the nodes along the last axis of the table
        node_weights: array of shape (4, ...)

    Returns:
        Array of shape (n_g, n_frequencies, ...).
    """
    return (
        opacity_grid[:, :, node_indices[0]] * node_weights[0]
        + opacity_grid[:, :, node_indices[1]] * node_weights[1]
        + opacity_grid[:, :, node_indices[2]] * node_weights[2]
        + opacity_grid[:, :, node_indices[3]] * node_weights[3]
    )


def _weight_nodes(node_weights, line_species_mass_fractions):
    """Fold the mass fractions, of shape (n_layers, n_species), into the weights (4, n_species, n_layers)."""
    if line_species_mass_fractions is None:
        return node_weights

    return node_weights * jnp.asarray(
        line_species_mass_fractions, dtype=WORKING_JAX_DTYPE
    ).T[None, :, :]


@jit(static_argnames=("output_layout",))
def _interpolate_packed_species_opacities(
    pressures,
    temperatures,
    line_opacities_grid,
    line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size,
    line_opacities_pressure_grid_size,
    line_opacities_node_offsets,
    line_species_mass_fractions=None,
    output_layout=LAYOUT_NG_F_S_L,
):
    node_indices, node_weights = compute_line_opacity_interpolation_nodes(
        pressures,
        temperatures,
        line_opacities_temperature_pressure_grid,
        line_opacities_temperature_grid_size,
        line_opacities_pressure_grid_size,
        line_opacities_node_offsets,
    )
    node_weights = _weight_nodes(node_weights, line_species_mass_fractions)

    if output_layout == LAYOUT_FL_S_NG:
        # (n_g, n_frequencies, n_layers, n_species) to (n_frequencies * n_layers, n_species, n_g)
        opacities = _weighted_nodes_sum(
            line_opacities_grid,
            jnp.swapaxes(node_indices, 1, 2),
            jnp.swapaxes(node_weights, 1, 2),
        )

        return jnp.transpose(opacities, (1, 2, 3, 0)).reshape(
            -1, opacities.shape[3], opacities.shape[0]
        )

    # (n_g, n_frequencies, n_species, n_layers)
    return _weighted_nodes_sum(line_opacities_grid, node_indices, node_weights)


@jit
def _interpolate_packed_species_opacities_summed(
    pressures,
    temperatures,
    line_opacities_grid,
    line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size,
    line_opacities_pressure_grid_size,
    line_opacities_node_offsets,
    line_species_mass_fractions,
):
    node_indices, node_weights = compute_line_opacity_interpolation_nodes(
        pressures,
        temperatures,
        line_opacities_temperature_pressure_grid,
        line_opacities_temperature_grid_size,
        line_opacities_pressure_grid_size,
        line_opacities_node_offsets,
    )
    node_weights = _weight_nodes(node_weights, line_species_mass_fractions)

    total_opacities = jnp.zeros(
        (
            line_opacities_grid.shape[0],
            line_opacities_grid.shape[1],
            pressures.size,
        ),
        dtype=WORKING_JAX_DTYPE,
    )

    # Accumulate one species at a time, so that the opacities of the individual species
    # (n_g, n_frequencies, n_species, n_layers) are never held in memory at once
    def _accumulate_species(species_index, total):
        return total + _weighted_nodes_sum(
            line_opacities_grid,
            node_indices[:, species_index, :],
            node_weights[:, species_index, :],
        )

    total_opacities = lax.fori_loop(
        0,
        node_indices.shape[1],
        _accumulate_species,
        total_opacities,
    )

    return total_opacities[:, :, None, :]


def interpolate_species_opacities(
    pressures,
    temperatures,
    n_g,
    n_frequencies,
    line_opacities_grid,
    line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size,
    line_opacities_pressure_grid_size,
    output_layout=LAYOUT_NG_F_S_L,
    line_opacities_node_offsets=None,
    line_species_mass_fractions=None,
):
    """Interpolate the line opacities of each species to the given pressures and temperatures.

    The line opacities can be given in two forms:
        - "packed" (the runtime layout of ``Radtrans``): ``line_opacities_grid`` is a single array of shape
          (n_g, n_frequencies, n_nodes), whose last axis runs over the temperature-pressure points of all
          species, the points of a species starting at ``line_opacities_node_offsets``. The grids are given
          as arrays with one row per species (see :func:`compute_line_opacity_interpolation_nodes`).
        - per species: ``line_opacities_grid`` and the grids are dictionaries with the species as keys.

    Args:
        pressures: (cgs) pressures of the atmospheric layers, of shape (n_layers,)
        temperatures: (K) temperatures of the atmospheric layers, of shape (n_layers,)
        n_g: number of g-samples (1 in line-by-line mode)
        n_frequencies: number of frequencies
        line_opacities_grid: the opacity table(s), see above
        line_opacities_temperature_pressure_grid: the temperature-pressure grid(s) of the species
        line_opacities_temperature_grid_size: the number of temperatures of the grid(s)
        line_opacities_pressure_grid_size: the number of pressures of the grid(s)
        output_layout:
            Layout of the output (packed form only), ``'ng_f_s_l'`` or ``'fl_s_ng'``.
        line_opacities_node_offsets:
            Packed form only: index of the first temperature-pressure point of each species in the table.
        line_species_mass_fractions:
            Optional mass fractions of the line species, of shape (n_layers, n_species) (packed form only).
            If given, the opacity of each species is multiplied by its mass fraction. Doing so here costs
            nothing, whereas weighting the interpolated opacities afterward requires an additional pass
            over (and copy of) the largest array of the calculation.

    Returns:
        The opacities in the working precision, of shape (n_g, n_frequencies, n_species, n_layers), or
        (n_frequencies * n_layers, n_species, n_g) if ``output_layout`` is ``'fl_s_ng'``.
    """
    # Interpolate line opacities to given temperature structure.
    n_layers = pressures.size
    if isinstance(line_opacities_grid, dict):
        n_line_species = len(line_opacities_grid)

        if n_line_species > 0:
            line_opacities = jnp.zeros(
                (n_g, n_frequencies, n_line_species, n_layers),
                dtype=WORKING_JAX_DTYPE,
            )

            for i, species in enumerate(line_opacities_grid):
                line_opacities_tp_grid = line_opacities_temperature_pressure_grid[
                    species
                ].reshape(
                    line_opacities_temperature_grid_size[species],
                    line_opacities_pressure_grid_size[species],
                    2,
                )
                line_opacities = line_opacities.at[:, :, i, :].set(
                    interpolate_line_opacities(
                        pressures,
                        temperatures,
                        line_opacities_tp_grid,
                        line_opacities_grid[species],
                        line_opacities_temperature_grid_size[species],
                        line_opacities_pressure_grid_size[species],
                    )
                )
        else:
            line_opacities = jnp.zeros(
                (n_g, n_frequencies, 1, n_layers), dtype=WORKING_JAX_DTYPE
            )

        return line_opacities

    if line_opacities_node_offsets is None:
        raise ValueError(
            "line_opacities_node_offsets is required when the line opacities are given as a packed table"
        )

    n_line_species = line_opacities_node_offsets.shape[0]

    if n_line_species > 0:
        return _interpolate_packed_species_opacities(
            pressures=pressures,
            temperatures=temperatures,
            line_opacities_grid=line_opacities_grid,
            line_opacities_temperature_pressure_grid=line_opacities_temperature_pressure_grid,
            line_opacities_temperature_grid_size=line_opacities_temperature_grid_size,
            line_opacities_pressure_grid_size=line_opacities_pressure_grid_size,
            line_opacities_node_offsets=line_opacities_node_offsets,
            line_species_mass_fractions=line_species_mass_fractions,
            output_layout=output_layout,
        )

    return jnp.zeros((n_g, n_frequencies, 1, n_layers), dtype=WORKING_JAX_DTYPE)


def interpolate_species_opacities_summed(
    pressures,
    temperatures,
    n_g,
    n_frequencies,
    line_opacities_grid,
    line_opacities_temperature_pressure_grid,
    line_opacities_temperature_grid_size,
    line_opacities_pressure_grid_size,
    line_opacities_node_offsets,
    line_species_mass_fractions,
):
    """Abundance-weighted sum of the interpolated line opacities.

    Unlike :func:`interpolate_species_opacities`, this accepts only the packed
    (array) opacity-grid form, not the per-species dict form.

    Returns:
        The summed opacities in the working precision, of shape (n_g, n_frequencies, 1, n_layers).
    """
    n_layers = pressures.size

    if line_opacities_node_offsets.shape[0] > 0:
        return _interpolate_packed_species_opacities_summed(
            pressures=pressures,
            temperatures=temperatures,
            line_opacities_grid=line_opacities_grid,
            line_opacities_temperature_pressure_grid=line_opacities_temperature_pressure_grid,
            line_opacities_temperature_grid_size=line_opacities_temperature_grid_size,
            line_opacities_pressure_grid_size=line_opacities_pressure_grid_size,
            line_opacities_node_offsets=line_opacities_node_offsets,
            line_species_mass_fractions=line_species_mass_fractions,
        )

    return jnp.zeros((n_g, n_frequencies, 1, n_layers), dtype=WORKING_JAX_DTYPE)
