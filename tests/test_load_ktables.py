"""Test the loading of HDF5 correlated-k tables onto the Radtrans frequency grid.
"""
from jax import config

config.update("jax_enable_x64", True)

import os  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from .context import petitRADTRANS  # noqa: E402,F401
from .test_correlated_k_binning import _G_SAMPLES, _synthetic_kcoeff, _write_ktable  # noqa: E402

import petitRADTRANS.opacities.load as load  # noqa: E402

_FILE_NAME = "1H2-16O__Test.R1000_0.9-1.25mu.ktable.petitRADTRANS.h5"


def _load(file, frequencies):
    kcoeff_shape = _synthetic_kcoeff().shape

    return load.load_hdf5_ktables(
        file_path_hdf5=file,
        frequencies=frequencies,
        g_size=_G_SAMPLES.size,
        temperature_pressure_grid_size=kcoeff_shape[0] * kcoeff_shape[1],
        frequency_grid_misalignment_tolerance=1e-2,
        return_host_array=True,
    )


@pytest.fixture
def ktable_file(tmp_path):
    return _write_ktable(os.path.join(tmp_path, _FILE_NAME), _synthetic_kcoeff())


def _file_frequencies(file):
    import h5py

    with h5py.File(file, "r") as f:
        return petitRADTRANS.physical_constants.c * f["bin_centers"][:][::-1]


def test_load_hdf5_ktables_selects_the_frequencies_of_the_model(ktable_file):
    file_frequencies = _file_frequencies(ktable_file)
    opacities = _load(ktable_file, file_frequencies[2:8])

    kcoeff = _synthetic_kcoeff()
    n_wavenumbers = kcoeff.shape[2]
    # Opacities are per mass, ordered by decreasing wavenumber, and of shape (g, frequency, temperature-pressure)
    expected = kcoeff[:, :, n_wavenumbers - 3: n_wavenumbers - 9: -1, :] / (18.0 * petitRADTRANS.physical_constants.amu)
    expected = np.swapaxes(expected, 0, 1).reshape((-1, 6, _G_SAMPLES.size)).swapaxes(0, 2)

    assert opacities.shape == (_G_SAMPLES.size, 6, kcoeff.shape[0] * kcoeff.shape[1])
    np.testing.assert_allclose(opacities, expected, rtol=1e-6)


@pytest.mark.parametrize("shift", [1.0 + 1e-6, 1.0 - 1e-6])
def test_load_hdf5_ktables_realigns_a_grid_misaligned_by_one_bin(ktable_file, shift):
    # A slightly shifted model grid drops one frequency at an edge of the intersection with the file grid
    file_frequencies = _file_frequencies(ktable_file)

    aligned = _load(ktable_file, file_frequencies[2:8])
    misaligned = _load(ktable_file, file_frequencies[2:8] * shift)

    np.testing.assert_array_equal(misaligned, aligned)


def test_load_hdf5_ktables_rejects_a_misalignment_beyond_the_tolerance(ktable_file):
    file_frequencies = _file_frequencies(ktable_file)

    with pytest.raises(ValueError, match="misalignment"):
        _load(ktable_file, file_frequencies[2:8] * (1.0 + 1e-4))
