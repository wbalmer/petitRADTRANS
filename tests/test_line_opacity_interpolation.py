"""Test the interpolation of the line opacities in their packed runtime layout.

The opacities of all line species are stored in one array, whose last axis runs over the
temperature-pressure points of the species, one species after the other, each species keeping its own
grid. The interpolation in this layout is compared with the interpolation of the table of each species
on its own.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from .context import petitRADTRANS  # noqa: F401

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE
from petitRADTRANS.radtrans_core.interpolate_line_opacities import (
    interpolate_line_opacities,
    interpolate_species_opacities,
    interpolate_species_opacities_summed,
)

_N_G = 4
_N_FREQUENCIES = 6

# (temperatures, pressures) of the grid of each species: different sizes and ranges
_GRIDS = (
    (np.array([100.0, 300.0, 900.0, 2700.0]), np.array([1e0, 1e3, 1e6])),
    (np.array([500.0, 1500.0]), np.array([1e-2, 1e1, 1e3, 1e5, 1e8])),
    (np.array([200.0, 400.0, 800.0]), np.array([1e2, 1e4, 1e6])),
)

# Layers inside, on the nodes of, and outside the grids
_PRESSURES = np.array([1e-4, 1e0, 3.7e1, 1e3, 4.2e4, 1e6, 3e7, 1e9])
_TEMPERATURES = np.array([50.0, 100.0, 432.0, 900.0, 1234.0, 2700.0, 310.0, 5000.0])


def _build_tables(seed=0):
    """Per-species opacity tables spanning several orders of magnitude, and their packed layout."""
    rng = np.random.default_rng(seed)
    n_species = len(_GRIDS)
    max_n_temperatures = max(temperatures.size for temperatures, _ in _GRIDS)
    max_n_pressures = max(pressures.size for _, pressures in _GRIDS)

    opacities = []
    grids = []
    grids_stacked = np.zeros((n_species, max_n_temperatures, max_n_pressures, 2))

    for i, (temperatures, pressures) in enumerate(_GRIDS):
        n_nodes = temperatures.size * pressures.size
        opacities.append(
            (10.0 ** rng.uniform(-8.0, 4.0, (_N_G, _N_FREQUENCIES, n_nodes))).astype(np.float32)
        )
        grid = np.stack(np.meshgrid(temperatures, pressures, indexing="ij"), axis=-1)
        grids.append(grid)

        # Padded as Radtrans does it: the last temperature and pressure are repeated
        grids_stacked[i] = grid[
            np.minimum(np.arange(max_n_temperatures), temperatures.size - 1)[:, None],
            np.minimum(np.arange(max_n_pressures), pressures.size - 1)[None, :],
        ]

    n_temperatures = np.array([temperatures.size for temperatures, _ in _GRIDS], dtype=np.int32)
    n_pressures = np.array([pressures.size for _, pressures in _GRIDS], dtype=np.int32)
    node_offsets = np.concatenate(([0], np.cumsum(n_temperatures * n_pressures)[:-1])).astype(np.int32)

    packed = dict(
        line_opacities_grid=jnp.asarray(np.concatenate(opacities, axis=-1)),
        line_opacities_temperature_pressure_grid=jnp.asarray(grids_stacked, dtype=WORKING_JAX_DTYPE),
        line_opacities_temperature_grid_size=jnp.asarray(n_temperatures),
        line_opacities_pressure_grid_size=jnp.asarray(n_pressures),
        line_opacities_node_offsets=jnp.asarray(node_offsets),
    )

    return opacities, grids, packed


def _locate(grid, values):
    """Lower node of the interval of the grid containing each value, and clamped position within it."""
    i_low = np.clip(np.searchsorted(grid, values) - 1, 0, grid.size - 2)
    position = (values - grid[i_low]) / (grid[i_low + 1] - grid[i_low])

    return i_low, np.clip(position, 0.0, 1.0)  # constant extrapolation outside the grid


def _interpolate_species_separately(opacities, grids):
    """Reference: bilinear interpolation of the table of each species, in double precision.

    Returns:
        Array of shape (g, frequency, species, layer).
    """
    interpolated = []

    for opacity, (temperatures, pressures) in zip(opacities, _GRIDS):
        table = opacity.astype(np.float64).reshape(
            (_N_G, _N_FREQUENCIES, temperatures.size, pressures.size)
        )
        i_t, y = _locate(temperatures, _TEMPERATURES)
        i_p, x = _locate(pressures, _PRESSURES)

        interpolated.append(
            (1.0 - y) * (1.0 - x) * table[:, :, i_t, i_p]
            + (1.0 - y) * x * table[:, :, i_t, i_p + 1]
            + y * (1.0 - x) * table[:, :, i_t + 1, i_p]
            + y * x * table[:, :, i_t + 1, i_p + 1]
        )

    return np.stack(interpolated, axis=2)


def _interpolate_packed(packed, **kwargs):
    return interpolate_species_opacities(
        pressures=jnp.asarray(_PRESSURES),
        temperatures=jnp.asarray(_TEMPERATURES),
        n_g=_N_G,
        n_frequencies=_N_FREQUENCIES,
        **packed,
        **kwargs,
    )


def test_packed_interpolation_matches_the_interpolation_of_each_species():
    opacities, grids, packed = _build_tables()

    interpolated = _interpolate_packed(packed)

    assert interpolated.shape == (_N_G, _N_FREQUENCIES, len(_GRIDS), _PRESSURES.size)
    np.testing.assert_allclose(
        interpolated, _interpolate_species_separately(opacities, grids), rtol=1e-5
    )


def test_packed_interpolation_keeps_opacities_much_smaller_than_their_neighbours():
    # Interpolating as low + x * (high - low) returns 0 on a node whose opacity is more than ~1e7 times
    # smaller than its neighbour's, in single precision. The weighted sum of the nodes does not.
    opacities, grids, packed = _build_tables()
    table = np.array(packed["line_opacities_grid"])
    table[:, :, 0] = 1e3  # species 0, first node (1e0, 100 K)
    table[:, :, 1] = 1e-6  # species 0, second node (1e3, 100 K)
    packed["line_opacities_grid"] = jnp.asarray(table)

    # Layer at (1e3, 100 K), exactly on the second node
    interpolated = interpolate_species_opacities(
        pressures=jnp.asarray([1e3]),
        temperatures=jnp.asarray([100.0]),
        n_g=_N_G,
        n_frequencies=_N_FREQUENCIES,
        **packed,
    )
    nested = interpolate_line_opacities(
        jnp.asarray([1e3], dtype=WORKING_JAX_DTYPE),
        jnp.asarray([100.0], dtype=WORKING_JAX_DTYPE),
        jnp.asarray(grids[0], dtype=WORKING_JAX_DTYPE),
        jnp.asarray(table[:, :, :12]),
        grids[0].shape[0],
        grids[0].shape[1],
    )

    np.testing.assert_array_equal(interpolated[:, :, 0, 0], np.float32(1e-6))
    assert np.all(np.asarray(nested) == 0.0)  # what this guards against


def test_packed_interpolation_is_done_in_the_working_precision():
    # The opacities are stored in the working precision: pressures and temperatures of a higher
    # precision (64-bit JAX) must not promote the interpolated opacities, the largest array of a
    # spectral calculation
    _, _, packed = _build_tables()

    assert _interpolate_packed(packed).dtype == WORKING_JAX_DTYPE
    assert _interpolate_packed(packed, output_layout="fl_s_ng").dtype == WORKING_JAX_DTYPE


def test_packed_interpolation_is_exact_on_the_nodes_and_constant_outside_the_grid():
    opacities, _, packed = _build_tables()
    interpolated = np.asarray(_interpolate_packed(packed))

    # Species 0, grid of 4 temperatures by 3 pressures: node index = i_temperature * 3 + i_pressure
    table = opacities[0]
    # Layer 1: (1e0, 100 K), the first node
    np.testing.assert_array_equal(interpolated[:, :, 0, 1], table[:, :, 0])
    # Layer 3: (1e3, 900 K), node (2, 1)
    np.testing.assert_array_equal(interpolated[:, :, 0, 3], table[:, :, 2 * 3 + 1])
    # Layer 5: (1e6, 2700 K), the last node
    np.testing.assert_array_equal(interpolated[:, :, 0, 5], table[:, :, 3 * 3 + 2])
    # Layer 0: (1e-4, 50 K), below the grid in pressure and temperature: first node
    np.testing.assert_array_equal(interpolated[:, :, 0, 0], table[:, :, 0])
    # Layer 7: (1e9, 5000 K), above the grid in pressure and temperature: last node
    np.testing.assert_array_equal(interpolated[:, :, 0, 7], table[:, :, 3 * 3 + 2])

    # A species does not read the nodes of its neighbours: species 1, last node of its 2 by 5 grid
    np.testing.assert_array_equal(interpolated[:, :, 1, 7], opacities[1][:, :, 1 * 5 + 4])


def test_packed_interpolation_layouts_hold_the_same_opacities():
    _, _, packed = _build_tables()

    interpolated = np.asarray(_interpolate_packed(packed))
    flat = np.asarray(_interpolate_packed(packed, output_layout="fl_s_ng"))

    assert flat.shape == (_N_FREQUENCIES * _PRESSURES.size, len(_GRIDS), _N_G)
    # (g, frequency, species, layer) to (frequency * layer, species, g)
    np.testing.assert_array_equal(
        flat,
        np.transpose(interpolated, (1, 3, 2, 0)).reshape(flat.shape),
    )


@pytest.mark.parametrize("output_layout", ["ng_f_s_l", "fl_s_ng"])
def test_packed_interpolation_weights_the_opacities_by_the_mass_fractions(output_layout):
    _, _, packed = _build_tables()
    rng = np.random.default_rng(1)
    mass_fractions = 10.0 ** rng.uniform(-6.0, 0.0, (_PRESSURES.size, len(_GRIDS)))

    unweighted = np.asarray(_interpolate_packed(packed, output_layout=output_layout), dtype=np.float64)
    weighted = _interpolate_packed(
        packed,
        output_layout=output_layout,
        line_species_mass_fractions=jnp.asarray(mass_fractions),
    )

    if output_layout == "ng_f_s_l":
        expected = unweighted * mass_fractions.T[None, None, :, :]
    else:
        expected = unweighted * np.tile(mass_fractions, (_N_FREQUENCIES, 1))[:, :, None]

    assert weighted.dtype == WORKING_JAX_DTYPE
    np.testing.assert_allclose(weighted, expected, rtol=1e-5)


def test_summed_packed_interpolation_is_the_sum_of_the_weighted_species():
    _, _, packed = _build_tables()
    rng = np.random.default_rng(2)
    mass_fractions = jnp.asarray(10.0 ** rng.uniform(-3.0, 0.0, (_PRESSURES.size, len(_GRIDS))))

    weighted = np.asarray(
        _interpolate_packed(packed, line_species_mass_fractions=mass_fractions), dtype=np.float64
    )
    summed = interpolate_species_opacities_summed(
        pressures=jnp.asarray(_PRESSURES),
        temperatures=jnp.asarray(_TEMPERATURES),
        n_g=_N_G,
        n_frequencies=_N_FREQUENCIES,
        line_species_mass_fractions=mass_fractions,
        **packed,
    )

    assert summed.shape == (_N_G, _N_FREQUENCIES, 1, _PRESSURES.size)
    assert summed.dtype == WORKING_JAX_DTYPE
    np.testing.assert_allclose(summed[:, :, 0, :], np.sum(weighted, axis=2), rtol=1e-5)


def test_packed_interpolation_requires_the_node_offsets():
    _, _, packed = _build_tables()
    del packed["line_opacities_node_offsets"]

    with pytest.raises(ValueError, match="line_opacities_node_offsets"):
        _interpolate_packed(packed)
