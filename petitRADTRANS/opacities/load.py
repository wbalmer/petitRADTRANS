import os
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jax import config
from jax.typing import ArrayLike
import h5py
import warnings

from petitRADTRANS.opacities.opacities import CorrelatedKOpacity, Opacity
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.utils import intersection_indices
from petitRADTRANS.utils import LockedDict

MPI = None
rank = 0
comm = None
load_mpi = True

if load_mpi:
    # MPI Multiprocessing
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        MPI = None
        comm = None
        rank = 0


JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=JAX_DTYPE)


def load_cia_opacity(file_path_hdf5, return_radtrans_opacities=True):
    """Load a CIA opacity file."""
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
            jnp.prod(f["mol_mass"][:]),  # weight
            wavelengths,
            f["t"][:],  # temperatures
            jnp.transpose(f["alpha"][:])[
                ::-1, :
            ],  # alpha (wavelength, temperature), correct ordering
        )


def load_cloud_opacity(file_path_hdf5, return_radtrans_opacities=True):
    """Load a cloud opacity file."""
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
            cloud_wavelengths,
            f["particle_radius_bins"][:],
            f["particles_radii"][:],
            f["particles_density"][()],
            f["absorption_opacities"][:],
            f["scattering_opacities"][:],
            f["asymmetry_parameters"][:],
        )


def load_hdf5_line_opacity_table(
    file_path_hdf5,
    frequencies,
    line_by_line_opacity_sampling=1,
    return_radtrans_opacities=True,
):
    """Load opacities (cm2.g-1) tables in HDF5 format, based on petitRADTRANS pseudo-ExoMol setup."""
    frequencies_relative_tolerance = (
        1e-12  # allow for a small relative deviation from the wavenumber grid
    )

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

        selection = jnp.nonzero(
            jnp.logical_and(
                jnp.greater_equal(
                    frequency_grid,
                    jnp.min(frequencies) * (1 - frequencies_relative_tolerance),
                ),
                jnp.less_equal(
                    frequency_grid,
                    jnp.max(frequencies) * (1 + frequencies_relative_tolerance),
                ),
            )
        )[0]
        selection = jnp.array([selection[0], selection[-1]])

        if line_by_line_opacity_sampling > 1:
            # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
            selection = selection.at[0].subtract(
                line_by_line_opacity_sampling - 1
            )  # array is ordered by increasing wavenumber

        line_opacities_grid = f["xsecarr"][:, :, selection[0] : selection[-1] + 1]

        # Divide by mass to convert cross-sections to opacities
        mol_mass_inv = 1 / (f["mol_mass"][()] * cst.amu)
        line_opacities_grid *= mol_mass_inv

    line_opacities_grid = line_opacities_grid[:, :, ::-1]

    if line_by_line_opacity_sampling > 1:
        line_opacities_grid = line_opacities_grid[:, :, ::line_by_line_opacity_sampling]

    if line_opacities_grid.shape[-1] != frequencies.size:
        frequency_grid = frequency_grid[selection[0] : selection[-1] + 1]
        frequency_grid = frequency_grid[::-1]

        if line_by_line_opacity_sampling > 1:
            frequency_grid = frequency_grid[::line_by_line_opacity_sampling]

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
        _line_opacities_grid = jnp.zeros(
            (
                line_opacities_grid.shape[0],
                line_opacities_grid.shape[1],
                frequencies.size,
            ),
            dtype=JAX_DTYPE,
        )
        _line_opacities_grid = _line_opacities_grid.at[:, :, indices_opacities].set(
            line_opacities_grid[:, :, indices_file]
        )
        del line_opacities_grid
    else:
        _line_opacities_grid = line_opacities_grid

    _line_opacities_grid = jnp.swapaxes(_line_opacities_grid, 0, 1)  # (t, p, wvl)
    _line_opacities_grid = _line_opacities_grid.reshape(
        (
            _line_opacities_grid.shape[0] * _line_opacities_grid.shape[1],
            _line_opacities_grid.shape[2],
        )
    )  # (tp, wvl)
    _line_opacities_grid = jnp.swapaxes(_line_opacities_grid, 0, 1)  # (wvl, tp)
    _line_opacities_grid = _line_opacities_grid[
        jnp.newaxis, :, jnp.newaxis, :
    ]  # (g, wvl, species, tp)

    return _line_opacities_grid


def get_line_opacity_file(path_input_data, species, category):
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

    ret_val = jnp.zeros(
        (temperature_grid.size * pressure_grid.size, 2), dtype=JAX_DTYPE
    )

    for i_t in range(temperature_grid.size):
        for i_p in range(pressure_grid.size):
            ret_val = ret_val.at[i_t * pressure_grid.size + i_p, 1].set(
                pressure_grid[i_p] * 1e6
            )  # bar to cgs
            ret_val = ret_val.at[i_t * pressure_grid.size + i_p, 0].set(
                temperature_grid[i_t]
            )

    line_opacities_temperature_pressure_grid = ret_val
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

    For example, if the Radtrans frequency grid is [0.1, ..., 0.3, ..., 3] and some loaded opacity frequency grid
    is in the interval [0.3, ..., 3, ..., 28], then the output indices will be the indices corresponding to
    [0.3, ..., 3] on the Radtrans frequency grid, and on the file frequency grid.

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
    return np.array([0.1, 251.0])  # um


def rebin_ck_line_opacities(
    input_file, target_resolving_power, wavenumber_grid=None, rewrite=False
):
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
    target_resolving_power, paths=None, species=None, rewrite=False
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
        )

        if state is None:
            success = True
        elif state == -1:
            success = False
            break

    if success:
        print("Successfully binned down all k-tables\n")
