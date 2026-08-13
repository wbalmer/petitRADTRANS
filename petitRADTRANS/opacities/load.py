import os
import numpy as np
import numpy.typing as npt
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


from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=WORKING_JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=WORKING_JAX_DTYPE)


def load_cia_opacity(file_path_hdf5, return_radtrans_opacities=True):
    """Load a CIA opacity file."""
    host_dtype = np.dtype(WORKING_JAX_DTYPE)

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
            np.prod(f["mol_mass"][:]).astype(host_dtype, copy=False),  # weight
            np.asarray(wavelengths, dtype=host_dtype),
            np.asarray(f["t"][:], dtype=host_dtype),  # temperatures
            np.asarray(
                np.transpose(f["alpha"][:])[::-1, :],
                dtype=host_dtype,
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


def load_hdf5_line_opacity_table(
    file_path_hdf5,
    frequencies,
    line_by_line_opacity_sampling=1,
    return_radtrans_opacities=True,
):
    """Load line-by-line opacities into a host-side ``(g, frequency, n_temperature*n_pressure)`` table.

    The returned array stays on the host as a NumPy array so callers can build
    the final stacked runtime layout first and only convert to JAX once.
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

        line_opacities_grid = np.asarray(
            f["xsecarr"][
                :,
                :,
                selection_start:selection_stop:line_by_line_opacity_sampling,
            ],
            dtype=host_dtype,
        )

        # Divide by mass to convert cross-sections to opacities
        mol_mass_inv = 1 / (f["mol_mass"][()] * cst.amu)
        line_opacities_grid *= mol_mass_inv

    line_opacities_grid = line_opacities_grid[:, :, ::-1]

    if line_opacities_grid.shape[-1] != frequencies.size:
        frequency_grid = frequency_grid[
            selection_start:selection_stop:line_by_line_opacity_sampling
        ][::-1]

        if (
            frequency_grid[-1] == frequency_grid[-2]
        ):  # handle duplicated last wavenumber
            frequency_grid = frequency_grid[:-1]
            line_opacities_grid = line_opacities_grid[:, :, :-1]

        indices_opacities, indices_file = get_frequency_grids_intersection_indices(
            frequencies=frequencies,
            file_frequencies=frequency_grid,
            tolerate_grid_misalignment=False,
        )

        # Fill opacity array
        _line_opacities_grid = np.zeros(
            (
                line_opacities_grid.shape[0],
                line_opacities_grid.shape[1],
                frequencies.size,
            ),
            dtype=host_dtype,
        )
        _line_opacities_grid[:, :, indices_opacities] = line_opacities_grid[
            :, :, indices_file
        ]
    else:
        _line_opacities_grid = line_opacities_grid

    _line_opacities_grid = np.swapaxes(_line_opacities_grid, 0, 1)  # (t, p, wvl)
    _line_opacities_grid = _line_opacities_grid.reshape(
        (
            _line_opacities_grid.shape[0] * _line_opacities_grid.shape[1],
            _line_opacities_grid.shape[2],
        )
    )  # (tp, wvl)
    _line_opacities_grid = np.swapaxes(_line_opacities_grid, 0, 1)  # (wvl, tp)
    _line_opacities_grid = _line_opacities_grid[np.newaxis, :, :]  # (g, wvl, tp)

    return _line_opacities_grid


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
