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


def test_load_hdf5_ktables_fills_the_frequencies_missing_from_the_file_with_zeros(ktable_file):
    file_frequencies = _file_frequencies(ktable_file)
    # Extend the model grid beyond the low-frequency end of the file, keeping its spacing
    ratio = file_frequencies[-1] / file_frequencies[-2]
    extension = file_frequencies[-1] * ratio ** np.arange(1, 4)
    frequencies = np.concatenate((file_frequencies[-5:], extension))

    opacities = _load(ktable_file, frequencies)
    covered = _load(ktable_file, file_frequencies[-5:])

    np.testing.assert_array_equal(opacities[:, :5, :], covered)
    assert np.all(opacities[:, 5:, :] == 0.0)
    assert np.all(covered > 0.0)


def test_load_hdf5_ktables_writes_into_the_slot_of_a_shared_table(ktable_file):
    # Radtrans stores all species in one array, each species loading its opacities into its own slot
    file_frequencies = _file_frequencies(ktable_file)
    frequencies = file_frequencies[2:8]
    reference = _load(ktable_file, frequencies)
    n_g, n_frequencies, n_nodes = reference.shape

    table = np.full((n_g, n_frequencies, n_nodes + 7), -1.0, dtype=reference.dtype)
    slot = table[:, :, 3:3 + n_nodes]

    returned = load.load_hdf5_ktables(
        file_path_hdf5=ktable_file,
        frequencies=frequencies,
        g_size=n_g,
        temperature_pressure_grid_size=n_nodes,
        frequency_grid_misalignment_tolerance=1e-2,
        out=slot,
    )

    assert returned is slot
    np.testing.assert_array_equal(slot, reference)
    # The neighbouring slots are left untouched
    assert np.all(table[:, :, :3] == -1.0)
    assert np.all(table[:, :, 3 + n_nodes:] == -1.0)

    with pytest.raises(ValueError, match="out must be of shape"):
        load.load_hdf5_ktables(
            file_path_hdf5=ktable_file,
            frequencies=frequencies,
            g_size=n_g,
            temperature_pressure_grid_size=n_nodes,
            frequency_grid_misalignment_tolerance=1e-2,
            out=table,
        )


@pytest.mark.parametrize("block_bytes, reorder_block_bytes", [(1, 1), (700, 64), (4096, 100000)])
def test_load_hdf5_ktables_does_not_depend_on_the_size_of_the_read_blocks(
    ktable_file, monkeypatch, block_bytes, reorder_block_bytes
):
    # The table is read through a staging buffer, in blocks of pressures, or of temperatures at one
    # pressure if a pressure is larger than the buffer: the result must not depend on the blocks
    file_frequencies = _file_frequencies(ktable_file)
    reference = _load(ktable_file, file_frequencies[1:9])

    monkeypatch.setattr(load, "_OPACITY_READ_BLOCK_BYTES", block_bytes)
    monkeypatch.setattr(load, "_OPACITY_REORDER_BLOCK_BYTES", reorder_block_bytes)

    np.testing.assert_array_equal(_load(ktable_file, file_frequencies[1:9]), reference)


def test_load_hdf5_ktables_reads_compressed_chunked_tables(tmp_path):
    import h5py

    kcoeff = _synthetic_kcoeff()
    file = _write_ktable(os.path.join(tmp_path, _FILE_NAME), kcoeff)
    reference = _load(file, _file_frequencies(file)[2:8])

    # Same table, stored in compressed chunks that the read blocks do not line up with
    with h5py.File(file, "r+") as f:
        del f["kcoeff"]
        f.create_dataset("kcoeff", data=kcoeff, chunks=(2, 1, 5, 3), compression="gzip")

    np.testing.assert_array_equal(_load(file, _file_frequencies(file)[2:8]), reference)


def test_host_buffer_is_handed_over_to_jax_without_changing_it():
    table = load.empty_aligned((3, 5, 7), np.float32)

    assert table.shape == (3, 5, 7)
    assert table.dtype == np.float32
    assert table.ctypes.data % load._HOST_BUFFER_ALIGNMENT == 0
    assert table.flags.c_contiguous

    table[...] = np.arange(table.size, dtype=np.float32).reshape(table.shape)
    expected = table.copy()
    device_table = load.transfer_to_device(table)

    # The buffer may now be shared with JAX: it must not be modified anymore
    assert not table.flags.writeable
    np.testing.assert_array_equal(np.asarray(device_table), expected)
