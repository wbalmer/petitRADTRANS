import os
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import h5py
import warnings

from petitRADTRANS.opacities.opacities import CorrelatedKOpacity, Opacity, _get_mpi_context
from petitRADTRANS.opacities.registry import (
    _get_default_rebinning_wavelength_range as _registry_get_default_rebinning_wavelength_range,
)
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.utils import intersection_indices
from petitRADTRANS.utils import LockedDict
from petitRADTRANS.validation import _handle_grid_misalignment


from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=WORKING_JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=WORKING_JAX_DTYPE)

# Size of the staging buffer through which line opacity tables are read from disk. The tables are
# stored in double precision and in a different axis order than the one used at runtime, so they are
# converted block by block: only this much memory is needed on top of the final table.
_OPACITY_READ_BLOCK_BYTES = 32 * 1024 * 1024

# Size of the pieces in which a staging buffer is reordered into the runtime layout. A piece and the
# part of the table it is written to must fit in the CPU caches for the reordering to be efficient.
_OPACITY_REORDER_BLOCK_BYTES = 2 * 1024 * 1024

# Byte alignment of the host buffer holding the loaded line opacities. On CPU, JAX adopts a
# sufficiently aligned NumPy buffer without copying it (see transfer_to_device).
_HOST_BUFFER_ALIGNMENT = 64


def empty_aligned(shape, dtype, alignment=_HOST_BUFFER_ALIGNMENT):
    """Allocate an uninitialized C-contiguous NumPy array whose data is aligned to ``alignment`` bytes.

    NumPy only guarantees the alignment of the array's dtype. JAX's CPU backend can adopt a host
    buffer without copying it only if the buffer is aligned for vectorized access, which a plain
    ``np.empty`` does not guarantee on every platform.

    Args:
        shape: shape of the array
        dtype: dtype of the array
        alignment: required alignment of the first element, in bytes (must be a multiple of the itemsize)

    Returns:
        An uninitialized array of the requested shape and dtype. It is a view of a larger byte buffer,
        which it keeps alive.
    """
    dtype = np.dtype(dtype)
    n_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    buffer = np.empty(n_bytes + alignment, dtype=np.uint8)
    offset = (-buffer.ctypes.data) % alignment

    return buffer[offset:offset + n_bytes].view(dtype).reshape(shape)


def transfer_to_device(host_array):
    """Hand a host (NumPy) array over to JAX with at most one copy.

    On CPU, an aligned host buffer is adopted in place, so that no second copy of the array ever
    exists. On an accelerator, the array is copied to the device once, and the host buffer is released
    as soon as the caller drops its own reference to it.

    The JAX array may share its memory with ``host_array``: the caller must not modify ``host_array``
    afterwards. To guard against this, it is made read-only.

    Args:
        host_array: the NumPy array to transfer

    Returns:
        The JAX array, on the default device.
    """
    host_array.flags.writeable = False

    return jax.device_put(host_array)


def _read_opacity_table_into(
    dataset,
    source_indices,
    destination,
    destination_indices,
    n_temperatures,
    n_pressures,
    scale,
    clip_negative,
    block_bytes=None,
    reorder_block_bytes=None,
):
    """Read spectral columns of an HDF5 line opacity table into its runtime layout, block by block.

    The table is converted to the destination dtype by HDF5 while it is read, clipped (optional) and
    scaled exactly as the full-array operations it replaces: the scaling is evaluated in double
    precision on the single precision values, then rounded to the destination dtype.

    The file is read in blocks of pressures (and of temperatures, if a single pressure is larger than
    the staging buffer), i.e. along its slowest-varying axes, so that it is accessed sequentially.

    Args:
        dataset:
            HDF5 dataset of shape (n_pressures, n_temperatures, n_spectral_points) (line-by-line) or
            (n_pressures, n_temperatures, n_spectral_points, n_g) (correlated-k).
        source_indices:
            Indices of the spectral points of the dataset to read, in the order of the destination.
        destination:
            Array of shape (n_g, n_frequencies, n_temperatures * n_pressures), filled in place. The
            temperature-pressure index is i_temperature * n_pressures + i_pressure. May be a view.
        destination_indices:
            Indices along the frequency axis of the destination receiving the columns ``source_indices``.
        n_temperatures: number of temperatures of the table
        n_pressures: number of pressures of the table
        scale: factor by which the table is multiplied (e.g. to convert cross-sections to opacities)
        clip_negative: if True, negative values are set to 0 before scaling
        block_bytes:
            Approximate size of the staging buffer, in bytes (default: _OPACITY_READ_BLOCK_BYTES).
        reorder_block_bytes:
            Approximate size of the pieces in which the staging buffer is reordered, in bytes
            (default: _OPACITY_REORDER_BLOCK_BYTES).
    """
    if block_bytes is None:
        block_bytes = _OPACITY_READ_BLOCK_BYTES

    if reorder_block_bytes is None:
        reorder_block_bytes = _OPACITY_REORDER_BLOCK_BYTES

    source_indices = np.asarray(source_indices, dtype=np.int64)
    destination_indices = np.asarray(destination_indices, dtype=np.int64)

    if source_indices.size != destination_indices.size:
        raise ValueError(
            f"frequencies size mismatch: {source_indices.size} spectral points selected in the opacity file "
            f"cannot be stored into {destination_indices.size} frequencies\n"
            f"This may be caused by loading opacities of different resolving power"
        )

    has_g = dataset.ndim == 4
    n_g = dataset.shape[3] if has_g else 1
    n_columns = source_indices.size
    staging_dtype = destination.dtype

    if dataset.shape[0] != n_pressures or dataset.shape[1] != n_temperatures:
        raise ValueError(
            f"opacity table of shape {dataset.shape} is inconsistent with its pressure-temperature grid "
            f"({n_pressures} pressures, {n_temperatures} temperatures)"
        )

    if destination.shape[0] != n_g or destination.shape[2] != n_temperatures * n_pressures:
        raise ValueError(
            f"cannot store an opacity table with {n_g} g-samples and {n_temperatures * n_pressures} "
            f"pressure-temperature points into an array of shape {destination.shape}"
        )

    # Splitting an axis never requires a copy, so this is a view of the destination
    destination_4d = destination.reshape(
        (n_g, destination.shape[1], n_temperatures, n_pressures)
    )

    if not np.may_share_memory(destination_4d, destination):
        raise RuntimeError("the opacity table destination could not be reshaped in place")

    if n_columns == 0:
        return

    # Spectral selection in the file. The requested columns are regularly spaced (consecutive, or
    # every n-th one when the opacities are down-sampled), in increasing or decreasing order: they
    # are read with a regular (strided) selection, so that only the requested columns are read.
    source_steps = np.diff(source_indices)
    regular = n_columns == 1 or (source_steps[0] != 0 and np.all(source_steps == source_steps[0]))

    if regular:
        source_step = abs(int(source_steps[0])) if n_columns > 1 else 1
        reverse_columns = n_columns > 1 and source_steps[0] < 0
        first_column = int(source_indices.min())
        spectral_selection = slice(
            first_column, first_column + (n_columns - 1) * source_step + 1, source_step
        )
        n_read_columns = n_columns
        take_columns = None
    else:  # irregular spacing: never the case for petitRADTRANS tables
        first_column = int(source_indices.min())
        spectral_selection = slice(first_column, int(source_indices.max()) + 1)
        n_read_columns = spectral_selection.stop - spectral_selection.start
        reverse_columns = False
        take_columns = source_indices - first_column

    # Frequencies of the destination receiving the columns, as a slice if possible
    if np.all(np.diff(destination_indices) == 1):
        destination_columns = slice(
            int(destination_indices[0]), int(destination_indices[-1]) + 1
        )
    else:
        destination_columns = None

    # Blocks of pressures, or of temperatures at one pressure if a pressure does not fit the staging buffer
    bytes_per_temperature = n_read_columns * n_g * staging_dtype.itemsize
    bytes_per_pressure = n_temperatures * bytes_per_temperature
    pressures_per_block = int(min(n_pressures, max(1, block_bytes // bytes_per_pressure)))
    temperatures_per_block = n_temperatures

    if bytes_per_pressure > block_bytes:
        temperatures_per_block = int(max(1, block_bytes // bytes_per_temperature))

    # Read whole HDF5 chunks: otherwise a compressed chunk would be decompressed once per block it overlaps
    if dataset.chunks is not None:
        pressures_per_block = int(
            min(n_pressures, -(-pressures_per_block // dataset.chunks[0]) * dataset.chunks[0])
        )

        if temperatures_per_block < n_temperatures:
            temperatures_per_block = int(
                min(n_temperatures, -(-temperatures_per_block // dataset.chunks[1]) * dataset.chunks[1])
            )

    staging_shape = (pressures_per_block, temperatures_per_block, n_read_columns)

    if has_g:
        staging_shape += (n_g,)

    staging = np.empty(staging_shape, dtype=staging_dtype)
    trailing = (slice(None),) if has_g else ()

    for pressure_start in range(0, n_pressures, pressures_per_block):
        pressure_stop = min(pressure_start + pressures_per_block, n_pressures)
        n_block_pressures = pressure_stop - pressure_start

        for temperature_start in range(0, n_temperatures, temperatures_per_block):
            temperature_stop = min(temperature_start + temperatures_per_block, n_temperatures)
            n_block_temperatures = temperature_stop - temperature_start

            # HDF5 converts the stored (double precision) values to the staging dtype while reading
            dataset.read_direct(
                staging,
                source_sel=(
                    slice(pressure_start, pressure_stop),
                    slice(temperature_start, temperature_stop),
                    spectral_selection,
                ) + trailing,
                dest_sel=(
                    slice(0, n_block_pressures),
                    slice(0, n_block_temperatures),
                    slice(None),
                ) + trailing,
            )
            block = staging[:n_block_pressures, :n_block_temperatures]

            if clip_negative:
                np.maximum(block, 0.0, out=block)

            # Columns in the order of the destination
            if take_columns is not None:
                block = np.take(block, take_columns, axis=2)
            elif reverse_columns:
                block = block[:, :, ::-1]

            # (pressure, temperature, column[, g]) to (g, column, temperature, pressure)
            if has_g:
                block = np.transpose(block, (3, 2, 1, 0))
            else:
                block = np.transpose(block, (2, 1, 0))[np.newaxis]

            # Reorder a few columns at a time, so that the work stays within the CPU caches
            columns_per_piece = int(
                max(
                    1,
                    reorder_block_bytes
                    // (n_g * n_block_temperatures * n_block_pressures * staging_dtype.itemsize),
                )
            )

            for column_start in range(0, n_columns, columns_per_piece):
                column_stop = min(column_start + columns_per_piece, n_columns)

                # Scale in double precision, in the destination's memory order, then round on assignment
                scaled = np.multiply(
                    block[:, column_start:column_stop], scale, dtype=np.float64, order="C"
                )

                if destination_columns is not None:
                    piece_columns = slice(
                        destination_columns.start + column_start,
                        destination_columns.start + column_stop,
                    )
                else:
                    piece_columns = destination_indices[column_start:column_stop]

                destination_4d[
                    :,
                    piece_columns,
                    temperature_start:temperature_stop,
                    pressure_start:pressure_stop,
                ] = scaled


def load_cia_opacity(file_path_hdf5, return_radtrans_opacities=True):
    """Load a CIA opacity file.

    Args:
        file_path_hdf5 (str):
            Absolute file path to hdf5 CIA table.
        return_radtrans_opacities (bool, optional):
            True by default. If False, the arrays as
            directly read from the HDF5 table are returned.

    Returns:
            """
    with h5py.File(file_path_hdf5, "r") as f:
        if not return_radtrans_opacities:
            return (
                f["DOI"][()][0].decode("utf-8"),
                f["mol_name"][:],
                f["wavenumbers"][:],
                f["alpha"][:],
                f["t"][:],
            )

        wavelengths = 1 / f["wavenumbers"][:]  # cm-1 to cm
        wavelengths = wavelengths[::-1]  # correct ordering

        species = f["mol_name"][:]
        species = [s.decode("utf-8") for s in species]

        return (
            species,
            jnp.prod(f["mol_mass"][:], dtype=WORKING_JAX_DTYPE),  # weight
            jnp.asarray(wavelengths, dtype=WORKING_JAX_DTYPE),
            jnp.asarray(f["t"][:], dtype=WORKING_JAX_DTYPE),  # temperatures
            jnp.asarray(
                np.transpose(f["alpha"][:])[::-1, :],
                dtype=WORKING_JAX_DTYPE,
            ),  # alpha (wavelength, temperature), correct ordering
        )


def load_cloud_opacity(file_path_hdf5, return_radtrans_opacities=True):
    """Load a cloud opacity file."""
    host_dtype = np.dtype(FEAUTRIER_JAX_DTYPE)

    with h5py.File(file_path_hdf5, "r") as f:
        if not return_radtrans_opacities:
            return (
                f["wavenumbers"][:],
                f["particle_radius_bins"][:],
                f["particles_radii"][:],
                f["particles_density"][()],
                f["absorption_opacities"][:],
                f["scattering_opacities"][:],
                f["asymmetry_parameters"][:],
            )

        cloud_wavelengths = 1 / f["wavenumbers"][:]  # cm-1 to cm
        cloud_wavelengths = cloud_wavelengths[::-1]  # correct ordering

        return (
            np.asarray(cloud_wavelengths, dtype=host_dtype),
            np.asarray(f["particle_radius_bins"][:], dtype=host_dtype),
            np.asarray(f["particles_radii"][:], dtype=host_dtype),
            np.asarray(f["particles_density"][()], dtype=host_dtype),
            np.asarray(f["absorption_opacities"][:], dtype=host_dtype),
            np.asarray(f["scattering_opacities"][:], dtype=host_dtype),
            np.asarray(f["asymmetry_parameters"][:], dtype=host_dtype),
        )


def load_hdf5_ktables(
    file_path_hdf5,
    frequencies,
    g_size,
    temperature_pressure_grid_size,
    frequency_grid_misalignment_tolerance,
    return_radtrans_opacities=True,
    return_host_array=False,
    out=None,
):
    """Load k-coefficient tables in HDF5 format, based on the ExoMolOP setup.

    The table is read in blocks and converted on the fly to the working precision and to the runtime
    layout (g, frequency, temperature-pressure), so that no full-size intermediate copy is made.

    Args:
        file_path_hdf5: path to the k-table file
        frequencies: the Radtrans frequency grid, in decreasing order
        g_size: number of g-samples of the k-table
        temperature_pressure_grid_size: number of temperature-pressure points of the k-table
        frequency_grid_misalignment_tolerance:
            Tolerated misalignment between the file and the Radtrans frequency grids, relative to the
            resolving power of the file.
        return_radtrans_opacities: if False, the arrays as directly read from the HDF5 table are returned
        return_host_array: if True, return a NumPy array instead of a JAX array
        out:
            Optional NumPy array of shape (g_size, frequencies.size, temperature_pressure_grid_size) into
            which the opacities are written. It may be a view of a larger array (e.g., the slot of a
            species in a table shared by several species). If given, it is also the returned array.

    Returns:
        The opacities in cm2/g, of shape (g_size, frequencies.size, temperature_pressure_grid_size), in the
        working precision. The temperature-pressure index is i_temperature * n_pressures + i_pressure.
        Frequencies not covered by the file are set to 0.
    """
    # TODO other opacity spurces are loaded in load.py, so load_hdf5_ktables should
    #   do the same!
    import h5py

    with h5py.File(file_path_hdf5, "r") as f:
        if not return_radtrans_opacities:
            return (
                f["DOI"][()][0].decode("utf-8"),
                f["bin_centers"][:],
                f["bin_edges"][:],
                f["kcoeff"][:],
                f["mol_mass"][()][0],
                f["mol_name"][()][0].decode("utf-8"),
                f["p"][:],
                f["t"][:],
                f["samples"][:],
                f["weights"][:],
            )

        bin_centers = f["bin_centers"][:]
        n_wavelengths = len(bin_centers)
        host_dtype = np.dtype(WORKING_JAX_DTYPE)
        _frequencies = cst.c * bin_centers[::-1]
        n_temperatures = len(f["t"][:])
        n_pressures = len(f["p"][:])

        indices_opacities, indices_file = (
            get_frequency_grids_intersection_indices(
                frequencies=frequencies,
                file_frequencies=_frequencies,
                tolerate_grid_misalignment=True,
            )
        )

        if (
            np.nonzero(indices_file)[0].size
            != np.nonzero(indices_opacities)[0].size
        ):
            grid_misalignment: bool = False

            if np.nonzero(indices_file)[0].size == frequencies.size - 1:
                indices_file = _handle_grid_misalignment(
                    indices=indices_file,
                    frequencies_reference=frequencies,
                    frequencies_test=_frequencies,
                )
                grid_misalignment = True
            elif (
                np.nonzero(indices_opacities)[0].size
                == np.nonzero(indices_file)[0].size - 1
            ):
                indices_opacities = _handle_grid_misalignment(
                    indices=indices_opacities,
                    frequencies_reference=_frequencies,
                    frequencies_test=frequencies,
                )
                grid_misalignment = True

            if grid_misalignment:
                max_relative_difference = np.max(
                    np.abs(
                        frequencies[indices_opacities] / _frequencies[indices_file]
                        - 1
                    )
                )
                spectral_info = Opacity.split_species_spectral_info(
                    os.path.basename(file_path_hdf5)
                )[1]
                resolving_power = Opacity.split_spectral_info(spectral_info)[0]
                resolving_power = Opacity.get_resolving_power_from_string(
                    resolving_power
                )

                if (
                    max_relative_difference
                    > frequency_grid_misalignment_tolerance / resolving_power
                ):
                    raise ValueError(
                        f"maximum frequency grid misalignment from file ({max_relative_difference}) "
                        f"is larger than tolerated "
                        f"({frequency_grid_misalignment_tolerance / resolving_power}); "
                        f"expect inaccuracies in spectral features location\n"
                        f"To remove this warning, ensure that your correlated-k opacity files "
                        f"have similar wavenumber grids"
                    )

        selected_file_indices = np.nonzero(indices_file)[0]
        # Invert index order since frequency arrays were inverted above,
        # but opacity file is still in its original order.
        selection_start = n_wavelengths - int(selected_file_indices[-1]) - 1
        selection_stop = n_wavelengths - int(selected_file_indices[0])

        # Divide by mass to convert cross-sections to opacities
        mol_mass_inv = float(np.ravel(1 / (f["mol_mass"][()] * cst.amu))[0])

        n_frequencies = int(np.size(frequencies))
        table_shape = (g_size, n_frequencies, temperature_pressure_grid_size)

        if temperature_pressure_grid_size != n_pressures * n_temperatures:
            raise ValueError(
                f"temperature_pressure_grid_size ({temperature_pressure_grid_size}) does not match the "
                f"pressure-temperature grid of file '{file_path_hdf5}' "
                f"({n_pressures} pressures x {n_temperatures} temperatures)"
            )

        if out is None:
            line_opacities_grid = np.empty(table_shape, dtype=host_dtype)
        else:
            if tuple(out.shape) != table_shape:
                raise ValueError(
                    f"out must be of shape {table_shape}, but is of shape {tuple(out.shape)}"
                )

            line_opacities_grid = out

        # The line opacities grid has shape (g_size, frequencies.size, n_pressures * n_temperatures),
        # with wavelengths in ascending order: the file columns are read in reverse order.
        # Only fill in those values where the data exists in the opacity input file.
        covered_frequencies = np.asarray(indices_opacities, dtype=bool)

        if not np.all(covered_frequencies):
            line_opacities_grid[:, ~covered_frequencies, :] = 0.0

        _read_opacity_table_into(
            dataset=f["kcoeff"],
            source_indices=np.arange(selection_stop - 1, selection_start - 1, -1),
            destination=line_opacities_grid,
            destination_indices=np.nonzero(covered_frequencies)[0],
            n_temperatures=n_temperatures,
            n_pressures=n_pressures,
            scale=mol_mass_inv,
            clip_negative=True,
        )

    if return_host_array or out is not None:
        return line_opacities_grid

    return jnp.asarray(line_opacities_grid, dtype=WORKING_JAX_DTYPE)


def load_hdf5_line_opacity_table(
    file_path_hdf5,
    frequencies,
    line_by_line_opacity_sampling=1,
    return_radtrans_opacities=True,
    out=None,
):
    """Load line-by-line opacities into a host-side ``(g, frequency, n_temperature*n_pressure)`` table.

    The returned array stays on the host as a NumPy array so callers can build
    the final runtime layout first and only convert to JAX once.

    The table is read in blocks and converted on the fly to the working precision and to the runtime
    layout, so that no full-size intermediate copy is made.

    Args:
        file_path_hdf5: path to the line-by-line opacity file
        frequencies: the Radtrans frequency grid, in decreasing order
        line_by_line_opacity_sampling: only every line_by_line_opacity_sampling-th point of the file is loaded
        return_radtrans_opacities: if False, the arrays as directly read from the HDF5 table are returned
        out:
            Optional NumPy array of shape (1, frequencies.size, n_temperatures * n_pressures) into which the
            opacities are written. It may be a view of a larger array (e.g., the slot of a species in a
            table shared by several species). If given, it is also the returned array.

    Returns:
        The opacities in cm2/g, of shape (1, frequencies.size, n_temperatures * n_pressures), in the working
        precision. The temperature-pressure index is i_temperature * n_pressures + i_pressure.
        Frequencies not covered by the file are set to 0.
    """
    frequencies_relative_tolerance = (
        1e-12  # allow for a small relative deviation from the wavenumber grid
    )
    host_dtype = np.dtype(WORKING_JAX_DTYPE)

    with h5py.File(file_path_hdf5, "r") as f:
        if not return_radtrans_opacities:
            return (
                f["DOI"][()][0].decode("utf-8"),
                f["bin_edges"][:],
                f["xsecarr"][:],
                f["mol_mass"][()][0],
                f["mol_name"][()][0].decode("utf-8"),
                f["p"][:],
                f["t"][:],
            )

        frequency_grid = cst.c * f["bin_edges"][:]  # cm-1 to s-1
        frequencies = np.asarray(frequencies)

        selection = np.nonzero(
            np.logical_and(
                np.greater_equal(
                    frequency_grid,
                    np.min(frequencies) * (1 - frequencies_relative_tolerance),
                ),
                np.less_equal(
                    frequency_grid,
                    np.max(frequencies) * (1 + frequencies_relative_tolerance),
                ),
            )
        )[0]
        selection_start = int(selection[0])
        selection_stop = int(selection[-1]) + 1

        if line_by_line_opacity_sampling > 1:
            # Expand the low-frequency edge so the down-sampled, reversed grid
            # still covers the requested wavelength upper bound, then align the
            # positive-step slice so that it lands on the same samples as the
            # historical "reverse then stride" implementation.
            expanded_start = max(
                0,
                selection_start - (line_by_line_opacity_sampling - 1),
            )
            selection_start = expanded_start + (
                (selection_stop - 1 - expanded_start) % line_by_line_opacity_sampling
            )

        # Indices of the selected spectral points in the file, in the order of the Radtrans frequency
        # grid (decreasing frequencies, hence reversed), and the frequencies they are stored into
        source_indices = np.arange(
            selection_start, selection_stop, line_by_line_opacity_sampling
        )[::-1]
        destination_indices = np.arange(source_indices.size)

        if source_indices.size != frequencies.size:
            file_frequencies = frequency_grid[source_indices]

            if (
                file_frequencies[-1] == file_frequencies[-2]
            ):  # handle duplicated last wavenumber
                file_frequencies = file_frequencies[:-1]
                source_indices = source_indices[:-1]

            indices_opacities, indices_file = get_frequency_grids_intersection_indices(
                frequencies=frequencies,
                file_frequencies=file_frequencies,
                tolerate_grid_misalignment=False,
            )

            source_indices = source_indices[indices_file]
            destination_indices = np.nonzero(indices_opacities)[0]

        n_pressures, n_temperatures = f["xsecarr"].shape[:2]
        table_shape = (1, frequencies.size, n_temperatures * n_pressures)

        if out is None:
            line_opacities_grid = np.empty(table_shape, dtype=host_dtype)
        else:
            if tuple(out.shape) != table_shape:
                raise ValueError(
                    f"out must be of shape {table_shape}, but is of shape {tuple(out.shape)}"
                )

            line_opacities_grid = out

        # Only fill in those values where the data exists in the opacity input file
        if destination_indices.size != frequencies.size:
            covered_frequencies = np.zeros(frequencies.size, dtype=bool)
            covered_frequencies[destination_indices] = True
            line_opacities_grid[:, ~covered_frequencies, :] = 0.0

        # Divide by mass to convert cross-sections to opacities
        mol_mass_inv = float(np.ravel(1 / (f["mol_mass"][()] * cst.amu))[0])

        _read_opacity_table_into(
            dataset=f["xsecarr"],
            source_indices=source_indices,
            destination=line_opacities_grid,
            destination_indices=destination_indices,
            n_temperatures=n_temperatures,
            n_pressures=n_pressures,
            scale=mol_mass_inv,
            clip_negative=False,
        )

    return line_opacities_grid


def get_line_opacity_file(path_input_data, species, category, disable_mpi=False):
    hdf5_file = None

    # Handle bin-down of correlated-k opacities
    if category == CorrelatedKOpacity.get_default_category():
        # Get the targeted resolving power
        _species, _ = Opacity.split_species_spectral_info(species)
        target_resolving_power, range_filename = Opacity.find_spectral_information(
            species
        )

        if "R" in target_resolving_power:
            target_resolving_power = CorrelatedKOpacity.get_resolving_power_from_string(
                target_resolving_power
            )

        if target_resolving_power == "":
            target_resolving_power = int(
                CorrelatedKOpacity.get_default_resolving_power()
            )

        # Bin down the opacities if necessary
        if target_resolving_power < int(
            CorrelatedKOpacity.get_default_resolving_power()
        ):
            # Try to find the binned-down file first
            _hdf5_file = CorrelatedKOpacity.find(
                path_input_data=path_input_data,
                category=category,
                species=species,
                find_all=True,  # output an empty list if no matching file found
                search_online=False,  # don't search online since this is not the standard resolving power
            )

            if (
                len(_hdf5_file) == 0
            ):  # no matching file found, make the file using exo_k, then find it
                print("Binned-down file not found, starting rebinning...")

                # Rebin from the file with the default resolving power, which should always exist
                default_spectral_sampling = (
                    CorrelatedKOpacity.get_resolving_power_string(
                        resolving_power=int(
                            CorrelatedKOpacity.get_default_resolving_power()
                        )
                    )
                )

                # Build a "default" species using the provided information and the default spectral sampling
                default_species = Opacity.join_species_all_info(
                    species_name=_species,  # contains already all non-spectral info
                    natural_abundance="",
                    charge="",
                    cloud_info="",
                    source="",
                    spectral_info="",
                    spectral_sampling=default_spectral_sampling,
                    wavelength_range=range_filename,
                )

                # Find the HDF5 file of the species with the default spectral sampling
                print(
                    f"Searching for the file of species '{_species}' "
                    f"with default spectral sampling ({default_spectral_sampling})...",
                )
                default_hdf5_file = CorrelatedKOpacity.find(
                    path_input_data=path_input_data,
                    category=category,
                    species=default_species,
                    find_all=False,
                    search_online=True,
                )
                print(
                    " File found.",
                )

                state = rebin_ck_line_opacities(
                    input_file=default_hdf5_file,
                    target_resolving_power=target_resolving_power,
                    wavenumber_grid=None,
                    rewrite=False,
                    disable_mpi=disable_mpi,
                )

                if state == -1:
                    raise RuntimeError(
                        "unable to perform binning down, please install exo_k"
                    )

                hdf5_file = None
            elif isinstance(_hdf5_file, str):  # exactly one matching file found, use it
                hdf5_file = _hdf5_file
            else:  # more than one file exists, try to find the correct file
                hdf5_file = None

    if hdf5_file is None:
        hdf5_file = Opacity.find(
            path_input_data=path_input_data,
            category=category,
            species=species,
            find_all=False,
            search_online=True,
        )

    return hdf5_file


def load_line_opacities_pressure_temperature_grid(hdf5_file):
    """Load line opacities temperature grids."""
    with h5py.File(hdf5_file, "r") as f:
        pressure_grid = f["p"][:]
        temperature_grid = f["t"][:]

    temperature_mesh, pressure_mesh = np.meshgrid(
        temperature_grid,
        pressure_grid * 1e6,
        indexing="ij",
    )
    # The operation below will finally return a jnp array
    # with shape (N_pressures * N_temperatures, 2) with this content:
    # T0, P0
    # T0, P1,
    # ...
    # TN_temperatures, PN_pressures
    line_opacities_temperature_pressure_grid = (
        np.stack(
            (temperature_mesh, pressure_mesh),
            axis=-1,
        )
        .reshape(-1, 2)
        .astype(np.dtype(WORKING_JAX_DTYPE), copy=False)
    )
    line_opacities_temperature_grid_size = temperature_grid.size
    line_opacities_pressure_grid_size = pressure_grid.size

    return (
        line_opacities_temperature_pressure_grid,
        line_opacities_temperature_grid_size,
        line_opacities_pressure_grid_size,
    )


def get_frequency_grids_intersection_indices(
    frequencies: ArrayLike,
    file_frequencies: ArrayLike,
    tolerate_grid_misalignment: bool = False,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Get the indices to fill frequencies from a file within the Radtrans frequency grid.

    For example, if the Radtrans frequency grid is [0.1, ..., 0.3, ..., 3][::-1] and some loaded opacity frequency grid
    is in the interval [0.3, ..., 3, ..., 28][::-1], then the output indices will be the indices corresponding to
    [0.3, ..., 3][::-1] on the Radtrans frequency grid, and on the file frequency grid.
    The [::-1] denotes changing the array from increasing to decreasing order, since intersection_indices
    only works for decreasing order arrays.

    Args:
        frequencies: the Radtrans frequency grid
        file_frequencies: the file frequency grid
        tolerate_grid_misalignment: if True, do not raise an error if the intersection indices sizes differ by at
        most 1.

    Returns:
        The indices of the intersection between both grids, for the Radtrans grid and the file grid0
    """
    indices_opacities, indices_file = intersection_indices(
        array1=frequencies, array2=file_frequencies
    )

    if np.nonzero(indices_opacities)[0].size != np.nonzero(indices_file)[0].size:
        if (
            not tolerate_grid_misalignment
            or np.abs(
                np.nonzero(indices_opacities)[0].size - np.nonzero(indices_file)[0].size
            )
            != 1
        ):
            raise ValueError(
                f"frequencies size mismatch: value frequency array of size {np.nonzero(indices_opacities)[0].size} "
                f"cannot be broadcast to indexing result of size {np.nonzero(indices_file)[0].size}\n"
                f"This may be caused by loading opacities of different resolving power"
            )

    return indices_opacities, indices_file


def get_opacity_sources_dict():
    """Returns an empy locked dictionary structure that can be used
    to store all opacity species participating in a pRT flux or
    transmission spectrum call.
    """

    return LockedDict.build_and_lock(
        {
            "init": LockedDict.build_and_lock(
                {
                    "line_species": (),
                    "gas_continuum_contributors": (),
                    "rayleigh_species": (),
                    "cloud_species": (),
                }
            ),
            "runtime": LockedDict.build_and_lock(
                {
                    "opaque_cloud_top_pressure": None,
                    "power_law": LockedDict.build_and_lock(
                        {
                            "power_law_opacity_350nm": None,
                            "power_law_opacity_coefficient": None,
                        }
                    ),
                    "gray_opacity": None,
                    "cloud_photosphere_median_optical_depth": None,
                    "additional_absorption_opacities_function": None,
                    "additional_scattering_opacities_function": None,
                    "stellar_intensities": None,
                }
            ),
        }
    )


def _get_default_rebinning_wavelength_range():
    return _registry_get_default_rebinning_wavelength_range()


def rebin_ck_line_opacities(
    input_file,
    target_resolving_power,
    wavenumber_grid=None,
    rewrite=False,
    disable_mpi=False,
):
    _, comm, rank = _get_mpi_context(disable_mpi=disable_mpi)

    try:
        import exo_k
    except ImportError:
        # Only raise a warning to give a chance to download the binned-down file
        warnings.warn(
            "binning down of opacities requires exo_k to be installed, no binning down has been performed"
        )
        return -1

    if (
        rank == 0
    ):  # prevent race condition when writing the binned down file during multi-processes execution
        if wavenumber_grid is None:
            # Define own wavenumber grid, make sure that log spacing is constant everywhere
            wavelengths_boundaries = _get_default_rebinning_wavelength_range()

            n_spectral_points = int(
                target_resolving_power
                * np.log(wavelengths_boundaries[1] / wavelengths_boundaries[0])
                + 1
            )

            wavenumber_grid = np.logspace(
                np.log10(1 / wavelengths_boundaries[1] * 1e4),
                np.log10(1 / wavelengths_boundaries[0] * 1e4),
                n_spectral_points,
            )

        # Output files
        output_file = input_file.replace(
            CorrelatedKOpacity.join_species_all_info(
                "",
                spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                    CorrelatedKOpacity.get_default_resolving_power()
                ),
            ),
            CorrelatedKOpacity.join_species_all_info(
                "",
                spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                    target_resolving_power
                ),
            ),
        )

        if os.path.isfile(output_file) and not rewrite:
            print(f"File '{output_file}' already exists, skipping re-binning...")
        else:
            print(f"Rebinning file '{input_file}' to R = {target_resolving_power}...")
            # Use Exo-k to rebin to low-res
            tab = exo_k.Ktable(filename=input_file, remove_zeros=True)
            tab.bin_down(wavenumber_grid)
            print(" Done.")

            print(f" Writing binned down file '{output_file}'...", end=" ")
            tab.write_hdf5(output_file)
            print("Done.")

            print(
                f"Successfully binned down k-table into '{output_file}' (R = {target_resolving_power})"
            )
    elif not isinstance(rank, int):
        raise TypeError(
            f"MPI rank must be of type '{int.__name__}', not {type(rank)}\n"
            f"This is likely a code error and not a user error, you may report this to "
            f"https://gitlab.com/mauricemolli/petitRADTRANS/-/issues"
        )

    if comm is not None:  # wait for the main process to finish the binning down
        comm.barrier()

    return 0  # execution without error


def rebin_multiple_ck_line_opacities(
    target_resolving_power, paths=None, species=None, rewrite=False, disable_mpi=False
):
    if species is None:
        species = []

    if paths is None:
        paths = []

    # Define own wavenumber grid, make sure that log spacing is constant everywhere
    wavelengths_boundaries = _get_default_rebinning_wavelength_range()
    n_spectral_points = int(
        target_resolving_power
        * np.log(wavelengths_boundaries[1] / wavelengths_boundaries[0])
        + 1
    )

    wavenumber_grid = np.logspace(
        np.log10(1 / wavelengths_boundaries[1] * 1e4),
        np.log10(1 / wavelengths_boundaries[0] * 1e4),
        n_spectral_points,
    )

    success = False

    # Do the rebinning, loop through species
    for i, s in enumerate(species):
        # Output files
        hdf5_opacity_file_input = paths[i]

        state = rebin_ck_line_opacities(
            input_file=hdf5_opacity_file_input,
            target_resolving_power=target_resolving_power,
            wavenumber_grid=wavenumber_grid,
            rewrite=rewrite,
            disable_mpi=disable_mpi,
        )

        if state is None:
            success = True
        elif state == -1:
            success = False
            break

    if success:
        print("Successfully binned down all k-tables\n")
