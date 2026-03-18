"""Stores the Radtrans object.
"""
import copy
import os
import sys
import warnings
from typing import Any, Callable

import h5py
import jax.numpy as jnp
from jax import jit, vmap, lax, config, tree_util, debug
from jax.typing import ArrayLike

import numpy as np
import numpy.typing as npt
from interpax import interp1d

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.radtrans_core.interpolate_line_opacities import interpolate_species_opacities
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.radtrans_core.spectrum_utils import compute_bins_edges, rebin_spectrum
from petitRADTRANS.opacities.opacities import CIAOpacity, CloudOpacity, CorrelatedKOpacity, LineByLineOpacity, Opacity
from petitRADTRANS.opacities import cia, clouds, continuum, hminus, load, rayleigh, rosseland
from petitRADTRANS.opacities.optical_depths import compute_optical_depths
from petitRADTRANS.stellar_spectra.phoenix import compute_star_spectrum
from petitRADTRANS.physics import (
    flux_hz2flux_cm, frequency2wavelength, hz2um, um2hz, wavelength2frequency
)
from petitRADTRANS.utils import LockedDict, tuple_init
from petitRADTRANS.validation import (
    _check_anisotropic_cloud_scattering,
    _check_line_opacity_mode,
    _check_input_data_file_existence,
    _check_path_input_data,
    _check_pressures,
    _check_return_clear_spectrum_relevance,
    _check_wavelength_boundaries,
    _handle_grid_misalignment
    )

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=JAX_DTYPE)

def _noop_opacity_fn(wavelengths, pressures):
    """Return zero opacity matrix with shape (n_wavelengths, n_pressures)."""
    return jnp.zeros((wavelengths.size, pressures.size), dtype=JAX_DTYPE)

class Radtrans:
    r"""Calculate spectra using a given set of opacities.

    Args:
        pressures (Optional):
            (bar) Array defining the pressure grid to be used for the spectral calculations.
        wavelength_boundaries (Optional):
            List containing left and right border of wavelength region to be considered, in micron. If nothing else
            is specified, it will be equal to ``[0.05, 300]``, hence using the full petitRADTRANS wavelength range
            (0.11 to 250 microns for ``'c-k'`` mode, 0.3 to 30 microns for the ``'lbl'`` mode). The larger the
            range the longer the computation time.
        line_species (Optional):
            List of strings, denoting which line absorber species to include.
        gas_continuum_contributors (Optional):
            List of strings, denoting which continuum absorber species to include.
        rayleigh_species (Optional):
            List of strings, denoting which Rayleigh scattering species to include.
        cloud_species (Optional):
            List of strings, denoting which cloud opacity species to include.
        line_opacity_mode (Optional[string]):
            If equal to ``'c-k'``: use low-resolution mode, at :math:`\\lambda/\\Delta \\lambda = 1000`, with the
            correlated-k assumption. if equal to ``'lbl'``: use high-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line treatment.
        line_by_line_opacity_sampling (Optional[int]):
            If ``mode = 'lbl'``, then this will only load every line_by_line_opacity_sampling-nth point of the
            high-resolution opacities. This can be used to save time and RAM.
            The user should verify whether this leads to solutions which are identical to the results of the
            non down-sampled :math:`10^6` resolution.
        scattering_in_emission (Optional[bool]):
            Will be ``False`` by default.
            If ``True`` scattering will be included in the emission spectral calculations. Note that this increases
            the runtime of pRT!
        emission_angle_grid (Optional):
            Array defining the cosines of the angle grid to be used for the emission spectrum calculations,
            and their weights. The array is of shape (2, n_angles), with emission_angle_grid[0] being the cosines of
            the angles, and emission_angle_grid[1] being the weights.
            A dictionary of array with keys 'cos_angles' and 'weights' can also be used.
            If None, a default set of values and weights are used.
        anisotropic_cloud_scattering (Optional[bool, str]):
            If True, anisotropic cloud scattering opacities are used for the spectral calculations.
            If False, isotropic cloud scattering opacities are used for the spectral calculations.
            If 'auto' (recommended), anisotropic_cloud_scattering is set to True for emission spectrum
            calculations, and to False for transmission spectrum calculations.
        path_input_data (Optional[str]):
            Path to the input_data folder, containing the files to be loaded by petitRADTRANS.
    """
    __dat_opacity_files_warning_message = (
        "loading opacities from .dat files is discouraged, the HDF5 format offer better performances at for a lower "
        "memory usage\n\n"
        "Converting petitRADTRANS .dat opacity files into HDF5 can be done by executing:\n"
        ">>> from petitRADTRANS.__file_conversion import convert_all\n"
        ">>> convert_all()\n\n"
        "Alternatively, the petitRADTRANS HDF5 files can be downloaded "
        "(see https://petitradtrans.readthedocs.io/en/latest/content/available_opacities.html)"
    )

    __line_opacity_property_setting_warning_message = (
        "setting a Radtrans line opacity property should be avoided\n"
        "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
        "(they need to be updated for consistency)\n"
        "It is recommended to create a new Radtrans instance instead"
    )

    __property_setting_warning_message = (
        "setting a Radtrans property directly is not recommended\n"
        "Create a new Radtrans instance (recommended) "
        "or re-do all the setup steps necessary for the modification to be taken into account"
    )

    _frequency_grid_misalignment_tolerance: float = 2e-3

    def __init__(
        self,
        pressures: ArrayLike = None,  # NOTE: How many layers
        wavelength_boundaries: ArrayLike = None,
        line_species: list[str] | tuple[str] = None,
        gas_continuum_contributors: list[str] | tuple[str] = None,
        rayleigh_species: list[str] | tuple[str] = None,
        cloud_species: list[str] | tuple[str] = None,
        line_opacity_mode: str = 'c-k',
        line_by_line_opacity_sampling: int = 1,  # TODO: Replace with generic spectral resolution parameter.
        scattering_in_emission: bool = False,  # TODO: Default to True
        emission_angle_grid: ArrayLike = None,  # NOTE: Probably doesn't need to be a main parameter
        anisotropic_cloud_scattering: bool = 'auto',  # NOTE: Probably doesn't need to be a main parameter
        path_input_data: str = None
    ):
        # Inputs checks
        _check_line_opacity_mode(line_opacity_mode)
        _check_anisotropic_cloud_scattering(anisotropic_cloud_scattering)

        # Initialize properties
        if pressures is None:
            debug.print("pressure was not set, initializing one layer at 1 bar")
            pressures = jnp.array([1.0], dtype=JAX_DTYPE)  # bar

        self._pressures = jnp.array(pressures) * 1e6  # bar to cgs  # TODO pressure could be spectral function argument
        _check_pressures(pressures)

        self._line_species = tuple_init(line_species)
        self._gas_continuum_contributors = tuple_init(gas_continuum_contributors)
        self._rayleigh_species = tuple_init(rayleigh_species)
        self._cloud_species = tuple_init(cloud_species)

        # Storing basenames at initialization to avoid recomputing them (hard to JIT compile)
        # TODO: is basename correct? What about isotopes? Enforce consistent naming.
        #self._species_basenames = {}
        #for species in self._line_species:
        #    self._species_basenames[species.split('.R')[0]] = (Opacity.get_species_base_name(species))
        #for species in self._rayleigh_species:
        #    self._species_basenames[species] = (Opacity.get_species_base_name(species))
        #for species in self._cloud_species:
        #    self._species_basenames[species] = (Opacity.get_species_base_name(species))

        self._line_opacity_mode = line_opacity_mode
        self._line_by_line_opacity_sampling = line_by_line_opacity_sampling
        self._scattering_in_emission = scattering_in_emission

        if wavelength_boundaries is None:
            self._wavelength_boundaries = jnp.array([0.05, 300.], dtype=JAX_DTYPE)  # um
        else:
            _check_wavelength_boundaries(wavelength_boundaries)
            self._wavelength_boundaries = wavelength_boundaries

        self._anisotropic_cloud_scattering = anisotropic_cloud_scattering

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()
        self._path_input_data = path_input_data

        # Initialize non-settable properties
        self.__scattering_in_transmission = False
        self.__sum_opacities = False
        self.__absorber_present = not (
                len(self._line_species) == 0
                and len(self._gas_continuum_contributors) == 0
                and len(self._rayleigh_species) == 0
                and len(self._cloud_species) == 0
        )

        # Initialize line parameters
        self._frequencies, self._frequency_bins_edges = self._init_frequency_grid()

        # Initialize loaded line opacities variables
        self._lines_loaded_opacities = LockedDict.build_and_lock(
            {
                'temperature_pressure_grid': {},
                'temperature_grid_size': {},
                'pressure_grid_size': {},
                'opacity_grid': {},
                'temperature_pressure_grid_stacked': jnp.zeros((0, 1, 1, 2), dtype=JAX_DTYPE),
                'temperature_grid_size_array': jnp.zeros(0, dtype=jnp.int32),
                'pressure_grid_size_array': jnp.zeros(0, dtype=jnp.int32),
                'opacity_grid_stacked': jnp.zeros((0, 1, 1, 1), dtype=JAX_DTYPE),
                'ck_g_combination_weights': jnp.ones(1, dtype=JAX_DTYPE),
                'ck_g_combination_weights_sum': jnp.array(1.0, dtype=JAX_DTYPE),
                'g_gauss': jnp.array(jnp.ones(1)),
                'weights_gauss': jnp.array(jnp.ones(1))
            }
        )

        # Initialize cia opacities
        self._cias_loaded_opacities = cia.init_cia_loaded_opacities(
            cia_contributors=cia.get_cia_contributors(self._gas_continuum_contributors)
        )

        # Initialize loaded cloud opacities variables
        # Use JAX-friendly empty sentinel arrays as defaults so jitted code sees arrays, not None
        # TODO: revert to default to None
        empty_wavelength_shape = 3
        empty_n_clouds_shape = 1
        empty_n_particles_radii = 3
        empty_3d = jnp.zeros(
                    (empty_n_particles_radii,
                     empty_wavelength_shape,
                     empty_n_clouds_shape), dtype=JAX_DTYPE)
        self._clouds_loaded_opacities = LockedDict.build_and_lock(
            {
                'wavelengths': jnp.zeros(empty_wavelength_shape, dtype=JAX_DTYPE),
                'absorption_opacities': empty_3d,
                'scattering_opacities': empty_3d,
                'particles_radii': jnp.zeros(empty_n_particles_radii, dtype=JAX_DTYPE),
                'particles_radii_bins': jnp.zeros(empty_n_particles_radii+1, dtype=JAX_DTYPE),
                'particles_densities': jnp.zeros(empty_n_clouds_shape, dtype=JAX_DTYPE),
                'particles_asymmetry_parameters': empty_3d,
            }
        )

        # Initialize the angle (mu) grid for the emission spectral calculations
        self._emission_angle_grid = LockedDict.build_and_lock(
            {
                'cos_angles': None,
                'weights': None
            }
        )

        if emission_angle_grid is None:
            self._emission_angle_grid['cos_angles'] = jnp.array([
                0.1127016654,
                0.5,
                0.8872983346
            ])

            self._emission_angle_grid['weights'] = jnp.array([
                0.2777777778,
                0.4444444444,
                0.2777777778
            ])
        else:
            if isinstance(emission_angle_grid, dict):
                self._emission_angle_grid['cos_angles'] = emission_angle_grid['cos_angles']
                self._emission_angle_grid['weights'] = emission_angle_grid['weights']
            elif hasattr(emission_angle_grid, '__iter__'):
                self._emission_angle_grid['cos_angles'] = jnp.array(emission_angle_grid[0], copy=True)
                self._emission_angle_grid['weights'] = jnp.array(emission_angle_grid[1], copy=True)

                if self._emission_angle_grid['cos_angles'].size != self._emission_angle_grid['weights'].size:
                    raise ValueError(f"emission_cos_angle_grid_weights must be an array of the same size than "
                                     f"emission_cos_angle_grid ({self._emission_angle_grid['cos_angles'].size}), "
                                     f"but is of size {self._emission_angle_grid['weights'].size}")
            else:
                raise ValueError(f"emission_angle_grid must be a dictionary or an iterable of shape (2, n_angles), but "
                                 f"is of type {type(emission_angle_grid)}")

        # Load all opacities
        self.load_all_opacities()

    def __getattr__(self, name):
        """Override of the object base __getattr__ method, in order to hint towards pRT3 names when pRT2 names are used.
        """
        # TODO remove in version 4.0.0
        base_message = f"'{self.__class__.__name__}' object has no attribute '{name}'"

        def _handle_deprecated_attributes(suggested_name):
            raise AttributeError(f"{base_message}. Maybe you meant {suggested_name}?")

        if name == 'calc_flux':
            _handle_deprecated_attributes(suggested_name=self.calculate_flux.__name__)
        elif name == 'calc_transm':
            _handle_deprecated_attributes(suggested_name=self.calculate_transit_radii.__name__)

        return super().__getattribute__(name)

    @property
    def anisotropic_cloud_scattering(self):
        return self._anisotropic_cloud_scattering

    @anisotropic_cloud_scattering.setter
    def anisotropic_cloud_scattering(self, mode: str):
        debug.print(self.__property_setting_warning_message)
        _check_anisotropic_cloud_scattering(mode)
        self._anisotropic_cloud_scattering = mode

    @property
    def cias_loaded_opacities(self):
        return self._cias_loaded_opacities

    @cias_loaded_opacities.setter
    def cias_loaded_opacities(self, dictionary: dict[str, dict[str, ArrayLike]]):
        debug.print(
            "setting the Radtrans CIA opacity property should be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key1, value1 in dictionary.items():
            for key2, value2 in value1.items():
                self._cias_loaded_opacities[key1][key2] = value2

    @property
    def clouds_loaded_opacities(self):
        return self._clouds_loaded_opacities

    @clouds_loaded_opacities.setter
    def clouds_loaded_opacities(self, dictionary: dict[str, ArrayLike]):
        debug.print(
            "setting the Radtrans cloud opacity property should be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key, value in dictionary.items():
            self._clouds_loaded_opacities[key] = value

    @property
    def cloud_species(self):
        return self._cloud_species

    @cloud_species.setter
    def cloud_species(self, species: list):
        debug.print(self.__property_setting_warning_message)
        self._cloud_species = species

    @property
    def emission_angle_grid(self):
        return self._emission_angle_grid

    @emission_angle_grid.setter
    def emission_angle_grid(self, dictionary: dict[str, ArrayLike]):
        debug.print(self.__property_setting_warning_message)

        for key, value in dictionary.items():
            self._emission_angle_grid[key] = value

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, array: ArrayLike):
        debug.print(
            "setting frequencies directly should be avoided\n"
            "This property is loaded from the opacity data in the input_data directory "
            "and is inter-dependent with other line opacities parameters, that also need to be updated\n"
            "It is recommended to create a new Radtrans instance with e.g. different wavelength boundaries instead"
        )
        self._frequencies = array

    @property
    def frequency_bins_edges(self):
        return self._frequency_bins_edges

    @frequency_bins_edges.setter
    def frequency_bins_edges(self, array: ArrayLike):
        debug.print(self.__line_opacity_property_setting_warning_message)
        self._frequency_bins_edges = array

    @property
    def gas_continuum_contributors(self):
        return self._gas_continuum_contributors

    @gas_continuum_contributors.setter
    def gas_continuum_contributors(self, species: list[str]):
        debug.print(self.__property_setting_warning_message)
        self._gas_continuum_contributors = species

    @property
    def line_by_line_opacity_sampling(self):
        return self._line_by_line_opacity_sampling

    @line_by_line_opacity_sampling.setter
    def line_by_line_opacity_sampling(self, value: int):
        debug.print(self.__property_setting_warning_message)
        self._line_by_line_opacity_sampling = value

    @property
    def lines_loaded_opacities(self):
        return self._lines_loaded_opacities

    @lines_loaded_opacities.setter
    def lines_loaded_opacities(self, dictionary: dict[str, ArrayLike]):
        debug.print(self.__line_opacity_property_setting_warning_message)

        for key, value in dictionary.items():
            self._lines_loaded_opacities[key] = value

    @property
    def line_opacity_mode(self):
        return self._line_opacity_mode

    @line_opacity_mode.setter
    def line_opacity_mode(self, mode):
        debug.print(self.__property_setting_warning_message)
        _check_line_opacity_mode(mode)
        self._line_opacity_mode = mode

    @property
    def line_species(self):
        return tuple(self._line_species)

    @line_species.setter
    def line_species(self, species: list | tuple):
        debug.print(self.__property_setting_warning_message)
        self._line_species = tuple(species)

    @property
    def path_input_data(self):
        return self._path_input_data

    @path_input_data.setter
    def path_input_data(self, path: str):
        debug.print(self.__property_setting_warning_message)
        _check_path_input_data(path)
        self._path_input_data = path

    @property
    def pressures(self) -> ArrayLike:
        # TODO pressures doesn't need to be a property
        return self._pressures

    @pressures.setter
    def pressures(self, array):
        debug.print(self.__property_setting_warning_message)
        _check_pressures(array)
        self._pressures = array

    @property
    def rayleigh_species(self):
        return self._rayleigh_species

    @rayleigh_species.setter
    def rayleigh_species(self, species: list):
        debug.print(self.__property_setting_warning_message)
        self._rayleigh_species = species

    @property
    def scattering_in_emission(self):
        return self._scattering_in_emission

    @scattering_in_emission.setter
    def scattering_in_emission(self, value: bool):
        debug.print(self.__property_setting_warning_message)
        self._scattering_in_emission = value

    @property
    def wavelength_boundaries(self):
        return self._wavelength_boundaries

    @wavelength_boundaries.setter
    def wavelength_boundaries(self, array: ArrayLike):
        debug.print(self.__property_setting_warning_message)
        _check_wavelength_boundaries(array)
        self._wavelength_boundaries = array

    def _clouds_have_effect(self, mass_fractions, cloud_fraction):
        """Check if the clouds have any effect, i.e. if the cloud species MMR is greater than 0.

        Args:
            mass_fractions: atmospheric mass mixing ratios
            cloud_fraction: cloud coverage fraction
        """

        # TODO revert to clearer pythonic statements
        output = lax.cond(
            cloud_fraction == 0,
            lambda: False,
            lambda: True
        )

        def _clouds_have_effect_internal(mass_fractions):
            add_cloud_opacity = False
            if len(self._cloud_species) > 0:
                for cloud_species in self._cloud_species:
                    add_cloud_opacity = lax.cond(
                        jnp.any(mass_fractions[cloud_species] > 0),
                        lambda: True,
                        lambda: False)
            return add_cloud_opacity

        return lax.cond(
            output,
            lambda: _clouds_have_effect_internal(mass_fractions),
            lambda: False
        )

    def _get_complete_coverage_cloud_species(self, complete_coverage_clouds):
        if complete_coverage_clouds is None:
            return ()

        return tuple(
            species for species in self._cloud_species
            if species in complete_coverage_clouds
        )

    def _get_partial_coverage_cloud_species(self, complete_coverage_clouds):
        complete_coverage_cloud_species = set(
            self._get_complete_coverage_cloud_species(complete_coverage_clouds)
        )

        return tuple(
            species for species in self._cloud_species
            if species not in complete_coverage_cloud_species
        )

    @staticmethod
    def _compute_additional_continuum_opacities(
        frequencies,
        pressures,
        additional_absorption_opacities_function,
        additional_scattering_opacities_function,
        cloud_photosphere_median_optical_depth
    ):
        additional_absorption_opacities = None
        additional_scattering_opacities = None
        cloud_absorption_opacities = None
        cloud_scattering_opacities = None

        # Add optional absorption opacity from outside
        if additional_absorption_opacities_function is None:
            if cloud_photosphere_median_optical_depth is not None:
                cloud_absorption_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
        else:
            additional_absorption_opacities = additional_absorption_opacities_function(
                hz2um(frequencies),
                pressures * 1e-6  # cgs to bar
            )

            if cloud_photosphere_median_optical_depth is not None:  # the additional opacities are treated as clouds
                # This assumes a single cloud model that is given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                cloud_absorption_opacities = additional_absorption_opacities

        # Add optional scattering opacity from outside
        if additional_scattering_opacities_function is None:
            if cloud_photosphere_median_optical_depth is not None:
                cloud_scattering_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
        else:
            additional_scattering_opacities = additional_scattering_opacities_function(
                hz2um(frequencies),
                pressures * 1e-6  # cgs to bar
            )

            if cloud_photosphere_median_optical_depth is not None:  # the additional opacities are treated as clouds
                # This assumes a single cloud model that is given by the parametrized opacities from
                # give_absorption_opacity and give_scattering_opacity
                cloud_scattering_opacities = additional_scattering_opacities

        return (
            additional_absorption_opacities, additional_scattering_opacities,
            cloud_absorption_opacities, cloud_scattering_opacities
        )

    def _init_spectral_function(
        self, is_emission, reference_gravity, complete_coverage_clouds, return_clear_spectrum,
        mass_fractions, cloud_fraction, opaque_cloud_top_pressure
    ):
        """Initializations and checks common to calculate_flux and calculate_transit_radii."""

        def handle_invalid_reference_gravity(reference_gravity):
            if reference_gravity <= 0:
                raise ValueError(
                    f"reference gravity must be > 0, but was {reference_gravity}"
                )
        debug.callback(handle_invalid_reference_gravity, reference_gravity)

        if complete_coverage_clouds is None:
            complete_coverage_clouds = ()

        spectrum_clear = None

        # Sanity checking and debugging messaging
        _check_return_clear_spectrum_relevance(
            return_clear_spectrum=return_clear_spectrum,
            clouds_have_effect=self._clouds_have_effect(mass_fractions, cloud_fraction),
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_fraction=cloud_fraction,
        )

        sum_opacities = self._set_sum_opacities(
            emission=is_emission,
            line_opacity_mode=self._line_opacity_mode,
            scattering_in_emission=self._scattering_in_emission)

        if self._anisotropic_cloud_scattering == 'auto':
            anisotropic_cloud_scattering = is_emission
        else:
            anisotropic_cloud_scattering = self._anisotropic_cloud_scattering

        if not anisotropic_cloud_scattering and is_emission:
            debug.print(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {anisotropic_cloud_scattering}; "
                          f"set it to True or 'auto' to disable this warning")
        elif anisotropic_cloud_scattering and not is_emission:
            debug.print(f"anisotropic cloud scattering is not recommended for transmission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {anisotropic_cloud_scattering}; "
                          f"set it to False or 'auto' to disable this warning")

        return (anisotropic_cloud_scattering, complete_coverage_clouds,
                spectrum_clear, sum_opacities)

    @staticmethod
    def _sanitize_frequency_surface_input(values, frequencies, default_value, quantity_name):
        if values is None:
            return jnp.full_like(frequencies, default_value, dtype=JAX_DTYPE)

        values = jnp.array(values, dtype=JAX_DTYPE)

        if jnp.ndim(values) == 0:
            return values * jnp.ones_like(frequencies, dtype=JAX_DTYPE)

        def _check_input_size(_values, _frequencies):
            if jnp.size(_values) != _frequencies.size:
                raise ValueError(
                    f"{quantity_name} must be a scalar or of the same size than frequencies "
                    f"({_frequencies.size}), but is of size {jnp.size(_values)}"
                )

        debug.callback(_check_input_size, values, frequencies)

        return values

    def _sanitize_emission_surface_inputs(
        self,
        star_irradiation_angle,
        stellar_intensities,
        star_effective_temperature,
        star_radius,
        orbit_semi_major_axis,
        reflectances,
        emissivities,
    ):
        star_irradiation_cos_angle = jnp.cos(jnp.deg2rad(star_irradiation_angle))
        star_irradiation_cos_angle = jnp.where(star_irradiation_cos_angle <= 0.0, 1e-8, star_irradiation_cos_angle)

        if stellar_intensities is None:
            stellar_intensities = jnp.zeros_like(self._frequencies, dtype=JAX_DTYPE)

            if (
                self._scattering_in_emission
                and star_effective_temperature is not None
                and orbit_semi_major_axis is not None
            ):
                stellar_intensities = compute_star_spectrum(
                    star_effective_temperature=star_effective_temperature,
                    orbit_semi_major_axis=orbit_semi_major_axis,
                    frequencies=self._frequencies,
                    star_radius=star_radius,
                )
        else:
            stellar_intensities = jnp.array(stellar_intensities, dtype=JAX_DTYPE)

        reflectances = self._sanitize_frequency_surface_input(
            values=reflectances,
            frequencies=self._frequencies,
            default_value=0.0,
            quantity_name='reflectance',
        )
        emissivities = self._sanitize_frequency_surface_input(
            values=emissivities,
            frequencies=self._frequencies,
            default_value=1.0,
            quantity_name='emissivity',
        )

        return {
            'star_irradiation_cos_angle': star_irradiation_cos_angle,
            'stellar_intensities': stellar_intensities,
            'reflectances': reflectances,
            'emissivities': emissivities,
        }

    @staticmethod
    def _normalize_optional_mapping(mapping):
        if mapping is None:
            return None

        normalized_mapping = {}

        for key, value in list(mapping.items()):
            normalized_mapping[key] = jnp.array(value, dtype=JAX_DTYPE)

        return normalized_mapping

    @staticmethod
    def _normalize_optional_scalar(value, missing_value):
        if value is None:
            return missing_value

        return jnp.array(value, dtype=JAX_DTYPE)

    def _sanitize_flux_opacity_inputs(
        self,
        complete_coverage_clouds,
        cloud_particles_mean_radii,
        cloud_hansen_a,
        cloud_hansen_b,
        clouds_particles_porosity_factor,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        opaque_cloud_top_pressure,
        cloud_particle_radius_distribution_std,
        power_law_opacity_350nm,
        power_law_opacity_coefficient,
        gray_opacity,
        cloud_photosphere_median_optical_depth,
        cloud_particle_number_density_grid,
        additional_absorption_opacities_function,
        additional_scattering_opacities_function,
        target_particle_radii,
        target_pressure,
    ):
        if cloud_f_sed is None:
            cloud_f_sed_norm = {}
        elif isinstance(cloud_f_sed, dict):
            cloud_f_sed_norm = {key: jnp.array(value, dtype=JAX_DTYPE) for key, value in cloud_f_sed.items()}
        else:
            cloud_f_sed_norm = {species: jnp.array(cloud_f_sed, dtype=JAX_DTYPE) for species in self._cloud_species}

        clouds_particles_porosity_factor_norm = {species: _MISSING_NEG_INF for species in self._cloud_species}
        if clouds_particles_porosity_factor is not None:
            for key, value in list(clouds_particles_porosity_factor.items()):
                clouds_particles_porosity_factor_norm[key] = jnp.array(value, dtype=JAX_DTYPE)

        if eddy_diffusion_coefficients is None:
            eddy_diffusion_coefficients_norm = None
        elif isinstance(eddy_diffusion_coefficients, dict):
            eddy_diffusion_coefficients_norm = {
                key: jnp.array(value, dtype=JAX_DTYPE)
                for key, value in eddy_diffusion_coefficients.items()
            }
        else:
            eddy_diffusion_coefficients_norm = jnp.array(eddy_diffusion_coefficients, dtype=JAX_DTYPE)

        return {
            'complete_coverage_clouds': () if complete_coverage_clouds is None else tuple(complete_coverage_clouds),
            'cloud_particles_mean_radii': self._normalize_optional_mapping(cloud_particles_mean_radii),
            'cloud_hansen_a': self._normalize_optional_mapping(cloud_hansen_a),
            'cloud_hansen_b': self._normalize_optional_mapping(cloud_hansen_b),
            'clouds_particles_porosity_factor': clouds_particles_porosity_factor_norm,
            'cloud_f_sed': cloud_f_sed_norm,
            'eddy_diffusion_coefficients': eddy_diffusion_coefficients_norm,
            'opaque_cloud_top_pressure': self._normalize_optional_scalar(opaque_cloud_top_pressure, _MISSING_FLOAT),
            'cloud_particle_radius_distribution_std': self._normalize_optional_scalar(
                cloud_particle_radius_distribution_std,
                _MISSING_FLOAT,
            ),
            'power_law_opacity_350nm': self._normalize_optional_scalar(power_law_opacity_350nm, _MISSING_FLOAT),
            'power_law_opacity_coefficient': self._normalize_optional_scalar(
                power_law_opacity_coefficient,
                _MISSING_FLOAT,
            ),
            'gray_opacity': (
                jnp.array(0.0, dtype=JAX_DTYPE)
                if gray_opacity is None
                else jnp.array(gray_opacity, dtype=JAX_DTYPE)
            ),
            'cloud_photosphere_median_optical_depth': self._normalize_optional_scalar(
                cloud_photosphere_median_optical_depth,
                _MISSING_FLOAT,
            ),
            'cloud_particle_number_density_grid': self._normalize_optional_mapping(
                cloud_particle_number_density_grid
            ),
            'additional_absorption_opacities_function': (
                additional_absorption_opacities_function
                if additional_absorption_opacities_function is not None
                else _noop_opacity_fn
            ),
            'additional_scattering_opacities_function': (
                additional_scattering_opacities_function
                if additional_scattering_opacities_function is not None
                else _noop_opacity_fn
            ),
            'target_particle_radii': (
                None if target_particle_radii is None else jnp.array(target_particle_radii, dtype=JAX_DTYPE)
            ),
            'target_pressure': (
                None if target_pressure is None else jnp.array(target_pressure, dtype=JAX_DTYPE)
            ),
        }

    def _stack_loaded_line_opacities(self):
        n_line_species = len(self._line_species)

        (self._lines_loaded_opacities['ck_g_combination_weights'],
         self._lines_loaded_opacities['ck_g_combination_weights_sum']) = (
            radiative_transfer_core.build_ck_combination_state(
                self._lines_loaded_opacities['weights_gauss']
            )
        )

        if n_line_species == 0:
            self._lines_loaded_opacities['temperature_pressure_grid_stacked'] = jnp.zeros((0, 1, 1, 2), dtype=JAX_DTYPE)
            self._lines_loaded_opacities['temperature_grid_size_array'] = jnp.zeros(0, dtype=jnp.int32)
            self._lines_loaded_opacities['pressure_grid_size_array'] = jnp.zeros(0, dtype=jnp.int32)
            self._lines_loaded_opacities['opacity_grid_stacked'] = jnp.zeros((0, 1, self._frequencies.size, 1), dtype=JAX_DTYPE)
            return

        temperature_grid_sizes = np.array(
            [self._lines_loaded_opacities['temperature_grid_size'][species] for species in self._line_species],
            dtype=np.int32,
        )
        pressure_grid_sizes = np.array(
            [self._lines_loaded_opacities['pressure_grid_size'][species] for species in self._line_species],
            dtype=np.int32,
        )
        max_temperature_grid_size = int(np.max(temperature_grid_sizes))
        max_pressure_grid_size = int(np.max(pressure_grid_sizes))
        max_temperature_pressure_grid_size = int(np.max(temperature_grid_sizes * pressure_grid_sizes))
        n_g = int(self._lines_loaded_opacities['g_gauss'].size)

        temperature_pressure_grid_stacked = np.zeros(
            (n_line_species, max_temperature_grid_size, max_pressure_grid_size, 2),
            dtype=np.asarray(self._pressures).dtype,
        )
        opacity_grid_stacked = np.zeros(
            (n_line_species, n_g, self._frequencies.size, max_temperature_pressure_grid_size),
            dtype=np.asarray(self._frequencies).dtype,
        )

        for i_species, species in enumerate(self._line_species):
            temperature_grid_size = int(temperature_grid_sizes[i_species])
            pressure_grid_size = int(pressure_grid_sizes[i_species])
            temperature_pressure_grid = np.asarray(
                self._lines_loaded_opacities['temperature_pressure_grid'][species]
            ).reshape(temperature_grid_size, pressure_grid_size, 2)
            opacity_grid = np.asarray(self._lines_loaded_opacities['opacity_grid'][species])

            if temperature_grid_size < max_temperature_grid_size or pressure_grid_size < max_pressure_grid_size:
                temperature_pressure_grid = np.pad(
                    temperature_pressure_grid,
                    (
                        (0, max_temperature_grid_size - temperature_grid_size),
                        (0, max_pressure_grid_size - pressure_grid_size),
                        (0, 0),
                    ),
                    mode='edge',
                )

            if opacity_grid.shape[-1] < max_temperature_pressure_grid_size:
                opacity_grid = np.pad(
                    opacity_grid,
                    ((0, 0), (0, 0), (0, max_temperature_pressure_grid_size - opacity_grid.shape[-1])),
                    mode='edge',
                )

            temperature_pressure_grid_stacked[i_species] = temperature_pressure_grid
            opacity_grid_stacked[i_species] = opacity_grid

        self._lines_loaded_opacities['temperature_pressure_grid_stacked'] = jnp.array(
            temperature_pressure_grid_stacked,
            dtype=JAX_DTYPE,
        )
        self._lines_loaded_opacities['temperature_grid_size_array'] = jnp.array(
            temperature_grid_sizes,
            dtype=jnp.int32,
        )
        self._lines_loaded_opacities['pressure_grid_size_array'] = jnp.array(
            pressure_grid_sizes,
            dtype=jnp.int32,
        )
        self._lines_loaded_opacities['opacity_grid_stacked'] = jnp.array(
            opacity_grid_stacked,
            dtype=JAX_DTYPE,
        )

    @staticmethod
    def _set_sum_opacities(emission, line_opacity_mode, scattering_in_emission):
        sum_opacities = False  # False in c-k emission if no scattering, and always False in c-k transmission
        if line_opacity_mode == 'c-k':
            if emission and scattering_in_emission:
                sum_opacities = True
        elif line_opacity_mode == 'lbl':  # always True in lbl mode
            sum_opacities = True
        return sum_opacities

    def _calculate_flux(
        self, temperatures, reference_gravity, opacities, continuum_opacities_scattering,
        emission_geometry, star_irradiation_cos_angle, stellar_intensity, reflectances, emissivities,
        cloud_f_sed, photospheric_cloud_optical_depths, cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
        adaptive_feautrier_iterations, return_contribution=False,
        sum_opacities=None,
        chunk_feautrier=False, feautrier_chunk_size=1024,
    ):
        """Compute the emission flux for the already-prepared opacity state.

        This helper converts opacities into optical depths, chooses the
        appropriate radiative-transfer solver, and optionally evaluates the
        emission contribution function.

        Args:
            temperatures:
                Atmospheric temperature profile in K with one entry per layer.
            reference_gravity:
                Reference gravity in cgs units used in the optical-depth
                calculation.
            opacities:
                Total absorption opacity array prepared by
                ``_calculate_opacities``.
            continuum_opacities_scattering:
                Continuum scattering opacity array with frequency and layer
                dimensions.
            emission_geometry:
                Irradiation geometry passed to the emission solver.
            star_irradiation_cos_angle:
                Cosine of the stellar irradiation angle.
            stellar_intensity:
                Incident stellar intensity spectrum on the model frequency grid.
            reflectances:
                Surface reflectance spectrum.
            emissivities:
                Surface emissivity spectrum.
            cloud_f_sed:
                Cloud settling parameter forwarded to the optical-depth helper
                when photospheric cloud scaling is requested.
            photospheric_cloud_optical_depths:
                Optional target cloud optical depth used by the photospheric
                cloud parametrization.
            cloud_photosphere_wavelength_boundaries:
                Optional wavelength interval used when evaluating the
                photospheric cloud optical depth.
            cloud_anisotropic_scattering_opacities:
                Cloud anisotropic scattering opacities used when scattering in
                emission is enabled.
            cloud_absorption_opacities:
                Cloud absorption opacity contribution used by the optical-depth
                calculation.
            adaptive_feautrier_iterations:
                Whether to adapt the number of scattering iterations in the
                Feautrier solver.
            return_contribution:
                If ``True``, also return the emission contribution function.
            chunk_feautrier:
                If ``True``, evaluate the Feautrier solver in frequency chunks.
            feautrier_chunk_size:
                Number of frequency bins per chunk when ``chunk_feautrier`` is
                enabled.

        Returns:
            Tuple ``(flux, emission_contribution, optical_depths,
            relative_cloud_scaling_factor)`` on the internal frequency grid.
        """

        if sum_opacities is None:
            sum_opacities = self.__sum_opacities

        optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor = (
            compute_optical_depths(
                pressures=self._pressures,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                sum_opacities=sum_opacities,
                scattering_in_emission=self._scattering_in_emission,
                absorber_present=self.__absorber_present,
                # Custom cloud parameters
                frequencies=self._frequencies,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
                cloud_f_sed=cloud_f_sed,
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=cloud_absorption_opacities,
                photospheric_cloud_optical_depths=photospheric_cloud_optical_depths
            )
        )
        # TODO jit to cond
        if self._scattering_in_emission:
            # TODO investigate bug with scattering and low VMR near surface
            # Only use 0 index for species because for lbl or test_ck_shuffle_comp = True
            # everything has been moved into the 0th index
            flux, emission_contribution = radiative_transfer_core.compute_feautrier_radiative_transfer(
                frequencies_bin_edges=self._frequency_bins_edges,
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                emission_cos_angles=self._emission_angle_grid['cos_angles'],
                emission_cos_angles_weights=self._emission_angle_grid['weights'],
                optical_depths_in=optical_depths[:, :, 0, :],
                photon_destruction_probabilities_in=photon_destruction_probabilities,
                stellar_intensity=stellar_intensity,
                star_irradiation_cos_angle=star_irradiation_cos_angle,
                reflectances=reflectances,
                emissivities=emissivities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                emission_geometry_str=emission_geometry,
                return_contribution=return_contribution,
                chunk_feautrier=chunk_feautrier,
                feautrier_chunk_size=feautrier_chunk_size,
            )
        else:
            # TODO jit to cond
            if self._line_opacity_mode == 'lbl' and len(self._line_species) > 1:
                _optical_depths = optical_depths[:, :, :1, :]
            else:
                _optical_depths = optical_depths.view()

            flux, emission_contribution = radiative_transfer_core.compute_ck_flux(
                self._frequencies,
                temperatures,
                self._lines_loaded_opacities['weights_gauss'],
                self._emission_angle_grid['cos_angles'],
                self._emission_angle_grid['weights'],
                _optical_depths,
                return_contribution
            )
        return flux, emission_contribution, optical_depths, relative_cloud_scaling_factor

    def _maybe_calculate_photosphere_radius(
        self,
        planet_radius,
        return_photosphere_radius,
        temperatures,
        mean_molar_masses,
        reference_gravity,
        opacities,
        continuum_opacities_scattering,
        cloud_f_sed,
        cloud_photosphere_median_optical_depth,
        cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        optical_depths,
    ):
        if planet_radius is None or not return_photosphere_radius:
            return None

        return self.calculate_photosphere_radius(
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            planet_radius=planet_radius,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            cloud_f_sed=cloud_f_sed,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            optical_depths=optical_depths,
        )

    def _calculate_radius_hydrostatic_equilibrium(
        self,
        temperatures,
        mean_molar_masses,
        reference_gravity,
        planet_radius,
    ):
        atmospheric_densities = self._pressures * mean_molar_masses * cst.amu / cst.kB / temperatures

        return radiative_transfer_core.compute_radius_hydrostatic_equilibrium(
            pressures=self._pressures,
            reference_gravity=reference_gravity,
            atmosphere_densities=atmospheric_densities,
            reference_pressure=self._pressures[-1],
            planet_radius=planet_radius,
        )

    def _get_photosphere_optical_depths(
        self,
        optical_depths,
        reference_gravity,
        opacities,
        continuum_opacities_scattering,
        cloud_f_sed,
        cloud_photosphere_median_optical_depth,
        cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        sum_opacities,
    ):
        if optical_depths is not None:
            return optical_depths

        optical_depths, _, _ = compute_optical_depths(
            pressures=self._pressures,
            reference_gravity=reference_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            scattering_in_emission=self._scattering_in_emission,
            sum_opacities=sum_opacities,
            absorber_present=self.__absorber_present,
            frequencies=self._frequencies,
            weights_gauss=self._lines_loaded_opacities['weights_gauss'],
            cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
            cloud_f_sed=cloud_f_sed,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
        )
        return optical_depths

    def _interpolate_photosphere_radius_profile(
        self,
        optical_depths,
        radius_hydrostatic_equilibrium,
    ):
        # TODO replace interp1d
        radius_interp = interp1d(self._pressures, radius_hydrostatic_equilibrium)
        photosphere_radius = jnp.zeros(self._frequencies.size, dtype=JAX_DTYPE)
        weights_gauss_reshape = self._lines_loaded_opacities['weights_gauss'].reshape(
            len(self._lines_loaded_opacities['weights_gauss']), 1
        )

        # TODO vmap or fori_loop
        # TODO replace interp1d
        for i_freq in range(self._frequencies.size):
            tau_p = jnp.sum(weights_gauss_reshape * optical_depths[:, i_freq, 0, :], axis=0)
            pressures_tau_p = interp1d(tau_p, self._pressures)
            if jnp.max(tau_p) > 2. / 3.:
                photosphere_radius = photosphere_radius.at[i_freq].set(radius_interp(pressures_tau_p(2 / 3)))
            else:
                debug.print('Atmosphere is too transparent for calculating the "photospheric '
                            'radius" at tau = 2/3. Returning the radius at maximum pressure instead!')
                photosphere_radius = photosphere_radius.at[i_freq].set(radius_hydrostatic_equilibrium[-1])

        return photosphere_radius

    @jit(static_argnames=[
        'self',
        'cloud_particles_radius_distribution',
        'complete_coverage_clouds',
        'additional_absorption_opacities_function',
        'additional_scattering_opacities_function',
        'sum_opacities',
        'anisotropic_cloud_scattering'
        ])
    def _calculate_opacities(
        self, temperatures, mass_fractions, mean_molar_masses, reference_gravity,
        opaque_cloud_top_pressure=None,
        cloud_particles_mean_radii=None, cloud_particle_radius_distribution_std=None,
        cloud_particles_radius_distribution="lognormal",
        cloud_particle_number_density_grid=None,
        cloud_hansen_a=None, cloud_hansen_b=None,
        clouds_particles_porosity_factor=None,
        cloud_f_sed=None, eddy_diffusion_coefficients=None,
        haze_factor=1.0, power_law_opacity_350nm=None, power_law_opacity_coefficient=None,
        gray_opacity=None, cloud_photosphere_median_optical_depth=None,
        cloud_fraction=1.0, complete_coverage_clouds=None,
        return_cloud_contribution=False,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        compute_clear_opacities=False,
        sum_opacities=None,
        anisotropic_cloud_scattering=None,
        target_particle_radii=None,
        target_pressure=None
    ):
        """Build line, continuum, and cloud opacity products for a spectrum calculation.

        The method interpolates line opacities to the current atmospheric
        structure, computes gas and cloud continuum contributions, combines
        them according to the active line-opacity mode, and optionally builds a
        second clear-sky opacity set for partial-cloud calculations.

        Args:
            temperatures:
                Atmospheric temperature profile in K.
            mass_fractions:
                Mapping from species name to mass-fraction profile.
            mean_molar_masses:
                Mean molecular mass profile in amu.
            reference_gravity:
                Reference gravity in cgs units.
            opaque_cloud_top_pressure:
                Optional pressure in bar where an opaque cloud deck is imposed.
            cloud_particles_mean_radii:
                Optional cloud particle mean radii keyed by cloud species.
            cloud_particle_radius_distribution_std:
                Width of the log-normal particle size distribution.
            cloud_particles_radius_distribution:
                Cloud particle size distribution model, typically
                ``"lognormal"`` or ``"hansen"``.
            cloud_particle_number_density_grid:
                Optional per-species particle number density grid.
            cloud_hansen_a:
                Optional Hansen ``a`` parameter profiles keyed by cloud species.
            cloud_hansen_b:
                Optional Hansen ``b`` parameter profiles keyed by cloud species.
            clouds_particles_porosity_factor:
                Optional porosity factors keyed by cloud species.
            cloud_f_sed:
                Cloud settling parameter, either scalar or per-species mapping.
            eddy_diffusion_coefficients:
                Optional eddy diffusion coefficient profile or mapping.
            haze_factor:
                Scalar multiplier for Rayleigh scattering.
            power_law_opacity_350nm:
                Optional reference scattering opacity at 0.35 micron.
            power_law_opacity_coefficient:
                Wavelength exponent for the parametrized scattering opacity.
            gray_opacity:
                Optional gray absorption opacity added everywhere.
            cloud_photosphere_median_optical_depth:
                Optional target median cloud optical depth near the photosphere.
            cloud_fraction:
                Cloud coverage fraction for mixed cloudy and clear columns.
            complete_coverage_clouds:
                Cloud species that should always be treated as fully covering.
            return_cloud_contribution:
                If ``True``, propagate cloud contribution products required by
                downstream diagnostics.
            additional_absorption_opacities_function:
                Optional user-supplied absorption opacity callback.
            additional_scattering_opacities_function:
                Optional user-supplied scattering opacity callback.
            compute_clear_opacities:
                If ``True``, also construct clear-column opacities.
            target_particle_radii:
                Optional cloud particle radii grid to which cloud opacities
                should be evaluated or interpolated.
            target_pressure:
                Optional pressure grid associated with ``target_particle_radii``.

        Returns:
            Tuple containing cloudy opacities, cloudy scattering opacities,
            cloud-only opacity diagnostics, particle radii products, a flag for
            scattering in transmission, and optional clear-column opacity
            products.
        """
        # TODO: how to do JAX sanity checks
        #if jnp.any(jnp.less_equal(temperatures, 0)):
        #    debug.print(f"temperature array ('temperatures') contains negative or null values ({temperatures})")

        #for species, mass_fraction in mass_fractions.items():
        #    if jnp.any(jnp.less(mass_fraction, 0)):
        #        debug.print(f"mass fractions for species '{species}' ('mass_fractions') "
        #                      f"contains negative values ({mass_fraction})")

        #if jnp.any(jnp.less(mean_molar_masses, 0)):
        #    debug.print(f"mean molar mass array ('mean_molar_masses') contains negative values ({mean_molar_masses})")

        #if cloud_fraction < 0 or cloud_fraction > 1:
        #    raise ValueError(f"cloud coverage fraction must be between 0 and 1, but was {cloud_fraction}")

        if complete_coverage_clouds is None:
            complete_coverage_clouds = ()
        complete_coverage_clouds = tuple(complete_coverage_clouds)

        if sum_opacities is None:
            sum_opacities = self.__sum_opacities

        if anisotropic_cloud_scattering is None:
            anisotropic_cloud_scattering = self._anisotropic_cloud_scattering

        # Initialization
        if len(self._line_species) > 0:
            line_species_mass_fractions = jnp.zeros(
                (self._pressures.size, len(self._line_species)), dtype=JAX_DTYPE)
        else:
            # If there are no specified line species then we need at least an array to contain the continuum opacities
            line_species_mass_fractions = jnp.zeros(
                (self._pressures.size, 1), dtype=JAX_DTYPE)

        # Fill line mass fraction dictionary with provided mass fraction dictionary
        for i_spec, species in enumerate(self._line_species):
            # Check if user provided the detailed line absorber name or if line absorber name should be matched exactly
            if species in mass_fractions:
                line_species_mass_fractions = \
                    line_species_mass_fractions.at[:, i_spec].set(mass_fractions[species])
            else:
                default_species, spectral_info = Opacity.split_species_spectral_info(species)

                if default_species in mass_fractions:
                    line_species_mass_fractions = \
                        line_species_mass_fractions.at[:, i_spec].set(mass_fractions[default_species])
                else:
                    raise ValueError(f"line species '{species}' not found in mass fractions dict "
                                     f"(listed species: {list(mass_fractions.keys())}")

        # Continuum opacities
        continuum_opacities = continuum.compute_continuum_opacities(
            frequencies=self._frequencies,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=self._pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            cias_loaded_opacities=self._cias_loaded_opacities,
            gas_continuum_contributors=self._gas_continuum_contributors,
            gray_opacity=gray_opacity,
            #species_basenames=self._species_basenames
        )
        continuum_opacities_scattering, scattering_in_transmission = (
            continuum.compute_continuum_scattering_opacities(
                frequencies=self._frequencies,
                pressures=self._pressures,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                rayleigh_species=self._rayleigh_species,
                haze_factor=haze_factor,
                #species_basenames=self._species_basenames
            )
        )

        complete_coverage_cloud_species = self._get_complete_coverage_cloud_species(complete_coverage_clouds)

        (cloudy_cloud_continuum_opacities, cloudy_cloud_continuum_opacities_scattering,
         cloud_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, _cloud_particles_mean_radii,
         clear_cloud_continuum_opacities, clear_cloud_continuum_opacities_scattering,
         clear_cloud_scattering_opacities, clear_cloud_absorption_opacities,
         cloud_scattering_in_transmission) = clouds.compute_cloud_continuum_opacities(
            frequencies=self._frequencies,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=self._pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            scattering_in_transmission=scattering_in_transmission,
            sum_opacities=sum_opacities,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            cloud_species=self._cloud_species,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            clouds_loaded_opacities=self._clouds_loaded_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            complete_coverage_clouds=complete_coverage_cloud_species,
            return_cloud_contribution=return_cloud_contribution,
            compute_clear_opacities=compute_clear_opacities,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        # TODO jitable for ValueError (jax.debug.callback)
        # TODO update conditional to not use None, instead check for _MISSING_FLOAT
        # Check if photospheric_cloud_optical_depths is used with a single cloud model
        # Combining cloud opacities from different models is currently not supported with the
        # photospheric_cloud_optical_depths parameter
        """if len(self._cloud_species) > 0 and cloud_photosphere_median_optical_depth is not None:
            if (additional_absorption_opacities_function is not None
                    or additional_scattering_opacities_function is not None):
                raise ValueError(
                    "The photospheric_cloud_optical_depths can only be used in combination with "
                    "a single cloud model. "
                    "Either use a physical cloud model by choosing cloud_species "
                    "or use parametrized cloud opacities with "
                    "the give_absorption_opacity and give_scattering_opacity parameters."
                )
        """
        # Additional opacities
        """if not self._clouds_have_effect(mass_fractions, cloud_fraction):
            (additional_absorption_opacities, additional_scattering_opacities,
             _cloud_absorption_opacities, _cloud_scattering_opacities) = self._compute_additional_continuum_opacities(
                frequencies=self._frequencies,
                pressures=self._pressures,
                additional_absorption_opacities_function=additional_absorption_opacities_function,
                additional_scattering_opacities_function=additional_scattering_opacities_function,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth
            )
        else:"""
        #additional_scattering_opacities = None
        #additional_absorption_opacities = None
        #_cloud_absorption_opacities = None
        #_cloud_scattering_opacities = None

        #cloud_absorption_opacities = _cloud_absorption_opacities
        #cloud_scattering_opacities = _cloud_scattering_opacities
        """if len(complete_coverage_clouds) > 0:
            # cloud_absorption_opacities is a list
            complete_coverage_cloud_absorption_opacities = _cloud_absorption_opacities
            complete_coverage_cloud_scattering_opacities = _cloud_scattering_opacities
        else:
            cloud_absorption_opacities = _cloud_absorption_opacities
            cloud_scattering_opacities = _cloud_scattering_opacities
        """
        #del _cloud_absorption_opacities
        #del _cloud_scattering_opacities

        #_opacities_first_species = None

        # Line opacities (calculating them just before calculating the combined opacities reduces peak memory usage)
        ck_flattened_layout = self._line_opacity_mode == 'c-k' and sum_opacities
        line_opacities = interpolate_species_opacities(
            pressures=self._pressures,
            temperatures=temperatures,
            n_g=self._lines_loaded_opacities['g_gauss'].size,
            n_frequencies=self._frequencies.size,
            line_opacities_grid=self._lines_loaded_opacities['opacity_grid_stacked'],
            line_opacities_temperature_pressure_grid=self._lines_loaded_opacities['temperature_pressure_grid_stacked'],
            line_opacities_temperature_grid_size=self._lines_loaded_opacities['temperature_grid_size_array'],
            line_opacities_pressure_grid_size=self._lines_loaded_opacities['pressure_grid_size_array'],
            output_layout='fl_s_ng' if ck_flattened_layout else 'ng_f_s_l'
        )

        cloudy_continuum_opacities = continuum_opacities + cloudy_cloud_continuum_opacities
        cloudy_continuum_opacities_scattering = (
            continuum_opacities_scattering + cloudy_cloud_continuum_opacities_scattering
        )

        # Combine line opacities with continuum opacities, combined opacities are stored in the 1st species slot
        opacities = continuum.compute_combined_opacities(
            opacities=line_opacities,
            continuum_opacities=cloudy_continuum_opacities,
            line_species=self._line_species,
            line_species_mass_fractions=line_species_mass_fractions,
            line_opacity_mode=self._line_opacity_mode,
            sum_opacities=sum_opacities,
            g_gauss=self._lines_loaded_opacities['g_gauss'],
            weights_gauss=self._lines_loaded_opacities['weights_gauss'],
            g_combination_weights=self._lines_loaded_opacities['ck_g_combination_weights'],
            g_combination_weights_sum=self._lines_loaded_opacities['ck_g_combination_weights_sum'],
            input_layout='fl_s_ng' if ck_flattened_layout else 'ng_f_s_l'
        )

        def _build_clear_branch(_line_opacities):
            clear_continuum_opacities = continuum_opacities + clear_cloud_continuum_opacities
            clear_continuum_opacities_scattering = (
                continuum_opacities_scattering + clear_cloud_continuum_opacities_scattering
            )
            clear_opacities = continuum.compute_combined_opacities(
                opacities=_line_opacities,
                continuum_opacities=clear_continuum_opacities,
                line_species=self._line_species,
                line_species_mass_fractions=line_species_mass_fractions,
                line_opacity_mode=self._line_opacity_mode,
                sum_opacities=sum_opacities,
                g_gauss=self._lines_loaded_opacities['g_gauss'],
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                g_combination_weights=self._lines_loaded_opacities['ck_g_combination_weights'],
                g_combination_weights_sum=self._lines_loaded_opacities['ck_g_combination_weights_sum'],
                input_layout='fl_s_ng' if ck_flattened_layout else 'ng_f_s_l'
            )
            return clear_opacities, clear_continuum_opacities_scattering

        clear_opacities, clear_continuum_opacities_scattering = lax.cond(
            compute_clear_opacities,
            _build_clear_branch,
            lambda _line_opacities: (
                jnp.zeros_like(opacities),
                jnp.zeros_like(continuum_opacities_scattering),
            ),
            line_opacities,
        )

        return (
            opacities, cloudy_continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities, cloud_opacities,
            _cloud_particles_mean_radii,
            cloud_scattering_in_transmission,
            clear_opacities, clear_continuum_opacities_scattering,
            clear_cloud_scattering_opacities, clear_cloud_absorption_opacities
        )

    def _calculate_transit_radii(
        self, temperatures, mean_molar_masses, reference_gravity,
        reference_pressure, planet_radius, variable_gravity,
        opacities, continuum_opacities_scattering, copy_opacities, return_contributions
    ):
        """Compute transmission-spectrum radii for the prepared opacity state.

        Args:
            temperatures:
                Atmospheric temperature profile in K.
            mean_molar_masses:
                Mean molecular mass profile in amu.
            reference_gravity:
                Reference gravity in cgs units at ``reference_pressure``.
            reference_pressure:
                Pressure in bar where ``planet_radius`` is defined.
            planet_radius:
                Planet radius in cm at ``reference_pressure``.
            variable_gravity:
                If ``True``, use a $1/r^2$ gravity profile in the transmission
                calculation.
            opacities:
                Total opacity array produced by ``_calculate_opacities``.
            continuum_opacities_scattering:
                Continuum scattering opacity array.
            copy_opacities:
                Retained for API compatibility with older transmission
                contribution workflows.
            return_contributions:
                Retained for compatibility with the transmission contribution
                interface. The current implementation returns a zero-filled
                placeholder contribution array.

        Returns:
            Tuple ``(transit_radii, radius_hydrostatic_equilibrium,
            transmission_contribution)``.
        """
        #if copy_opacities:
        transmission_contribution = jnp.zeros(
            (jnp.size(self._pressures), self._frequencies.size), dtype=JAX_DTYPE)
        #else:
        #    transmission_contribution = None

        # Calculate the transmission spectrum
        """if self.__sum_opacities and len(self._line_species) > 1:
            transit_radii, radius_hydrostatic_equilibrium = radiative_transfer_core.compute_transit_radii_summed(
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                pressures=self._pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity
            )"""

        """if return_contributions:
                # ORDER FOR TRANSMISSION CONTRIBUTION ARGS
                #opacities,
                #temperatures,
                #pressures * 1e-6,
                #reference_gravity,
                #mean_molar_masses,
                #reference_pressure,
                #planet_radius,
                #weights_gauss,
                #transit_radii ** 2,
                #continuum_opacities_scattering,
                #scattering_in_transmission,
                #variable_gravity
            transmission_contribution, radius_hydrostatic_equilibrium = (
                radiative_transfer_core.compute_transmission_spectrum_contribution(
                    transit_radii=transit_radii,
                    opacities=opacities[:, :, :1, :],
                    temperatures=temperatures,
                    pressures=self._pressures,
                    reference_gravity=reference_gravity,
                    mean_molar_masses=mean_molar_masses,
                    reference_pressure=reference_pressure,
                    planet_radius=planet_radius,
                    weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                    scattering_in_transmission=self.__scattering_in_transmission,
                    continuum_opacities_scattering=continuum_opacities_scattering,
                    variable_gravity=variable_gravity
                )
        )"""
        #else:
        transit_radii, radius_hydrostatic_equilibrium = radiative_transfer_core.compute_transit_radii(
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            pressures=self._pressures * 1e-6,  # cgs to bar
            temperatures=temperatures,
            weights_gauss=self._lines_loaded_opacities['weights_gauss'],
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity
        )

        """    if return_contributions:
                transmission_contribution, radius_hydrostatic_equilibrium = (
                    radiative_transfer_core.compute_transmission_spectrum_contribution(
                        transit_radii=transit_radii,
                        opacities=opacities,
                        temperatures=temperatures,
                        pressures=self._pressures,
                        reference_gravity=reference_gravity,
                        mean_molar_masses=mean_molar_masses,
                        reference_pressure=reference_pressure,
                        planet_radius=planet_radius,
                        weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                        scattering_in_transmission=self.__scattering_in_transmission,
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        variable_gravity=variable_gravity
                    )
                )
        """
        return transit_radii, radius_hydrostatic_equilibrium, transmission_contribution

    def _init_frequency_grid(self):
        """Initialize the Radtrans frequency grid, used to calculate the spectra.
        The frequency grid comes from the requested opacity files, in the following priority order:
            1. lines,
            2. CIA,
            3. clouds.
        If not opacities are provided, the mean of the wavelength boundaries is used. The frequency grid in that case
        has only 1 element.

        Returns:
            frequencies:
                (Hz) frequencies (center of bin) of the line opacities, also use for spectral calculations, of size N
            frequency_bins_edges:
                (Hz) edges of the frequencies bins, of size N+1
                for correlated-k only, number of points used to sample the g-space (1 in the case lbl is used)
        """
        if len(self._line_species) > 0:
            frequencies, frequency_bins_edges = self._init_frequency_grid_from_lines()
        elif len(self._gas_continuum_contributors) > 0:
            hdf5_file = cia.get_cia_aliases(self._gas_continuum_contributors[0])
            hdf5_file = CIAOpacity.find(
                path_input_data=self._path_input_data,
                species=hdf5_file
            )

            with h5py.File(hdf5_file, 'r') as f:
                frequency_grid = cst.c * f['wavenumbers'][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=1
            )
        elif len(self._cloud_species) > 0:
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[0])
            hdf5_file = CloudOpacity.find(
                path_input_data=self._path_input_data,
                species=hdf5_file
            )

            with h5py.File(hdf5_file, 'r') as f:
                frequency_grid = cst.c * f['wavenumbers'][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=1
            )
        else:
            debug.print("no opacity source given (lines, CIA, or clouds), "
                          "setting frequency grid using the mean of wavelength boundaries (1 element)")
            frequency_bins_edges = jnp.zeros(2, dtype=JAX_DTYPE)
            frequency_bins_edges = frequency_bins_edges.at[0].set(wavelength2frequency(self._wavelength_boundaries[1] * 1e-4))  # um to cm
            frequency_bins_edges = frequency_bins_edges.at[1].set(wavelength2frequency(self._wavelength_boundaries[0] * 1e-4))  # um to cm
            frequencies = jnp.mean(frequency_bins_edges)

        return frequencies, frequency_bins_edges

    @staticmethod
    def _init_frequency_grid_from_frequency_grid(frequency_grid, wavelength_boundaries, sampling=1):
        # Get frequency boundaries
        frequency_min = um2hz(wavelength_boundaries[1])
        frequency_max = um2hz(wavelength_boundaries[0])

        # Check if the requested wavelengths boundaries are within the file boundaries
        bad_boundaries = False

        if frequency_min < frequency_grid[0]:
            bad_boundaries = True

        if frequency_max > frequency_grid[-1]:
            bad_boundaries = True

        if bad_boundaries:
            raise ValueError(f"Requested wavelength interval "
                             f"({wavelength_boundaries[0]}--{wavelength_boundaries[1]}) "
                             f"is out of opacities table wavelength grid "
                             f"({hz2um(frequency_grid[-1])}--{hz2um(frequency_grid[0])})")

        # Get the freq. corresponding to the requested boundaries, with the request fully within the selection
        selection = jnp.nonzero(jnp.logical_and(
            jnp.greater_equal(frequency_grid, frequency_min),
            jnp.less_equal(frequency_grid, frequency_max)
        ))[0]
        selection = jnp.array([selection[0], selection[-1]])

        if frequency_grid[selection[0]] > frequency_min:
            selection = selection.at[0].add(-1)

        if frequency_grid[selection[-1]] < frequency_max:
            selection = selection.at[-1].add(1)

        if sampling > 1:
            # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
            selection = selection.at[0].subtract(sampling - 1)

        frequencies = frequency_grid[selection[0]:selection[-1] + 1]
        frequencies = frequencies[::-1]

        # Down-sample frequency grid in lbl mode if requested
        if sampling > 1:
            frequencies = frequencies[::sampling]

        frequency_bins_edges = jnp.array(
            wavelength2frequency(compute_bins_edges(frequency2wavelength(frequencies))), dtype=JAX_DTYPE
        )

        return frequencies, frequency_bins_edges

    def _init_frequency_grid_from_lines(self):
        if self._line_opacity_mode == 'c-k':  # correlated-k
            # Get dimensions of molecular opacity arrays for a given P-T point, they define the resolution
            # Use the first entry of self.line_species for this, if given
            opacities_file = load.get_line_opacity_file(
                path_input_data=self._path_input_data,
                species=self._line_species[0],
                category='correlated_k_opacities'
            )

            with h5py.File(opacities_file, 'r') as f:
                frequency_bins_edges = cst.c * f['bin_edges'][:][::-1]

            wavelengths = hz2um(frequency_bins_edges)
        elif self._line_opacity_mode == 'lbl':  # line-by-line
            # Load the wavelength grid
            opacities_file = LineByLineOpacity.find(
                path_input_data=self._path_input_data,
                species=self._line_species[0]
            )

            with h5py.File(opacities_file, 'r') as f:
                frequency_grid = cst.c * f['bin_edges'][:]  # cm-1 to Hz

            if frequency_grid[0] == frequency_grid[1]:  # handle duplicated last wavenumber
                frequency_grid = frequency_grid[1:]

            wavelengths = hz2um(frequency_grid[::-1])
        else:
            raise ValueError(f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'")

        # Extend the wavelength range if user requests larger range than what first line opa species contains
        if wavelengths[-1] < self._wavelength_boundaries[1]:
            delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[-1]

            add_high = 10 ** jnp.arange(
                jnp.log10(wavelengths[-1]),
                jnp.log10(self._wavelength_boundaries[-1]) + delta_log_wavelength,
                delta_log_wavelength
            )[1:]
            wavelengths = jnp.concatenate((wavelengths, add_high))

        if wavelengths[0] > self._wavelength_boundaries[0]:
            delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[0]

            if delta_log_wavelength == 0:  # handle duplicate first wavelength
                delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[1]

            add_low = 1e1 ** (-np.arange(
                -jnp.log10(wavelengths[0]),
                -jnp.log10(self._wavelength_boundaries[0]) + delta_log_wavelength,
                delta_log_wavelength
            )[1:][::-1])
            wavelengths = jnp.concatenate((add_low, wavelengths))

        # Get back to frequencies
        if self._line_opacity_mode == 'c-k':
            frequency_bins_edges = um2hz(wavelengths)
            frequencies = (frequency_bins_edges[1:] + frequency_bins_edges[:-1]) * 0.5

            # Cut the wavelength range if user requests smaller range than what first line opa species contains
            indices_within_boundaries = jnp.nonzero(jnp.logical_and(
                jnp.greater(frequency2wavelength(frequencies), self._wavelength_boundaries[0] * 1e-4),  # um to cm
                jnp.less(frequency2wavelength(frequencies), self._wavelength_boundaries[1] * 1e-4)  # um to cm
            ))[0]

            frequencies = jnp.array(frequencies[indices_within_boundaries], dtype=JAX_DTYPE)

            # Get the corresponding frequencies bin edges, +2 is to catch the upper bin edge
            frequency_bins_edges = jnp.array(
                frequency_bins_edges[indices_within_boundaries[0]:indices_within_boundaries[-1] + 2], dtype=JAX_DTYPE
            )
        elif self._line_opacity_mode == 'lbl':
            frequency_grid = um2hz(wavelengths[::-1])
            frequencies, frequency_bins_edges = self._init_frequency_grid_from_frequency_grid(
                frequency_grid=frequency_grid,
                wavelength_boundaries=self._wavelength_boundaries,
                sampling=self._line_by_line_opacity_sampling
            )
        else:
            raise ValueError(f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'")

        return frequencies, frequency_bins_edges

    def _tree_flatten(self):
        children = (
            self._pressures,
            self._wavelength_boundaries,
            self._frequencies,
            self._frequency_bins_edges,
            self._lines_loaded_opacities,
            self._cias_loaded_opacities,
            self._clouds_loaded_opacities,
            self._emission_angle_grid,
        )
        aux_data = {
            'line_species': self._line_species,
            'gas_continuum_contributors': self._gas_continuum_contributors,
            'rayleigh_species': self._rayleigh_species,
            'cloud_species': self._cloud_species,
            'line_opacity_mode': self._line_opacity_mode,
            'line_by_line_opacity_sampling': self._line_by_line_opacity_sampling,
            'scattering_in_emission': self._scattering_in_emission,
            'anisotropic_cloud_scattering': self._anisotropic_cloud_scattering,
            'path_input_data': self._path_input_data,
            'scattering_in_transmission': self.__scattering_in_transmission,
            'sum_opacities': self.__sum_opacities,
            'absorber_present': self.__absorber_present,
        }
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        instance = cls.__new__(cls)
        (
            instance._pressures,
            instance._wavelength_boundaries,
            instance._frequencies,
            instance._frequency_bins_edges,
            instance._lines_loaded_opacities,
            instance._cias_loaded_opacities,
            instance._clouds_loaded_opacities,
            instance._emission_angle_grid,
        ) = children
        instance._line_species = aux_data['line_species']
        instance._gas_continuum_contributors = aux_data['gas_continuum_contributors']
        instance._rayleigh_species = aux_data['rayleigh_species']
        instance._cloud_species = aux_data['cloud_species']
        instance._line_opacity_mode = aux_data['line_opacity_mode']
        instance._line_by_line_opacity_sampling = aux_data['line_by_line_opacity_sampling']
        instance._scattering_in_emission = aux_data['scattering_in_emission']
        instance._anisotropic_cloud_scattering = aux_data['anisotropic_cloud_scattering']
        instance._path_input_data = aux_data['path_input_data']
        instance._Radtrans__scattering_in_transmission = aux_data['scattering_in_transmission']
        instance._Radtrans__sum_opacities = aux_data['sum_opacities']
        instance._Radtrans__absorber_present = aux_data['absorber_present']
        return instance

    def calculate_contribution_spectra(
        self, mode,
        opaque_cloud_top_pressure=None,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        stellar_intensities=None,
        **kwargs
    ):
        """Calculate total and per-opacity-source spectra for attribution studies.

        The method evaluates the requested spectral mode once with all active
        opacity sources and then repeats the calculation for each individual
        initialization-time or runtime opacity source, returning a dictionary of
        spectra keyed by contribution name.

        Args:
            mode:
                Either ``"emission"`` or ``"transmission"``.
            opaque_cloud_top_pressure:
                Optional opaque cloud deck pressure in bar.
            power_law_opacity_350nm:
                Optional reference scattering opacity for the parametrized power
                law cloud model.
            power_law_opacity_coefficient:
                Wavelength exponent for the parametrized scattering opacity.
            gray_opacity:
                Optional gray absorption opacity.
            cloud_photosphere_median_optical_depth:
                Optional target cloud optical depth near the photosphere.
            additional_absorption_opacities_function:
                Optional user-provided absorption opacity callback.
            additional_scattering_opacities_function:
                Optional user-provided scattering opacity callback.
            stellar_intensities:
                Optional stellar intensity spectrum used in emission mode.
            **kwargs:
                Additional keyword arguments forwarded to
                ``calculate_flux`` or ``calculate_transit_radii``.

        Returns:
            Nested dictionary containing the total spectrum under ``"Total"``
            and contribution spectra grouped by opacity source.
        """
        # Store opacity sources values
        opacity_sources = load.get_opacity_sources_dict()
        opacity_sources.update({
            'init': {
                'line_species': self.line_species,
                'gas_continuum_contributors': self.gas_continuum_contributors,
                'cloud_species': self.cloud_species,
                'rayleigh_species': self.rayleigh_species
            },
            'runtime': {
                'opaque_cloud_top_pressure': opaque_cloud_top_pressure,
                'power_law': {
                    'power_law_opacity_350nm': power_law_opacity_350nm,
                    'power_law_opacity_coefficient': power_law_opacity_coefficient,
                },
                'gray_opacity': gray_opacity,
                'cloud_photosphere_median_optical_depth': cloud_photosphere_median_optical_depth,
                'additional_absorption_opacities_function': additional_absorption_opacities_function,
                'additional_scattering_opacities_function': additional_scattering_opacities_function,
                'stellar_intensities': stellar_intensities
            }
        })

        # Default values for opacity sources
        opacity_source_default = load.get_opacity_sources_dict()

        # Initialize spectral mode
        if mode == 'emission':
            spectral_function = 'calculate_flux'
        elif mode == 'transmission':
            spectral_function = 'calculate_transit_radii'

            # Remove spectral parameters that are not implemented in transmission
            del opacity_sources['runtime']['cloud_photosphere_median_optical_depth']
            del opacity_sources['runtime']['stellar_intensities']
        else:
            raise ValueError(f"mode '{mode}' is not implemented, "
                             f"available modes are 'emission'|'transmission'")

        # Initialize opacity contributions dict
        opacity_contributions = {}

        # Total spectrum
        spectral_parameters = {
            copy.deepcopy(k): copy.deepcopy(v)
            for k, v in kwargs.items()
        }

        for key, value in opacity_sources['runtime'].items():
            if key == 'power_law':
                spectral_parameters['power_law_opacity_350nm'] = copy.deepcopy(value['power_law_opacity_350nm'])
                spectral_parameters['power_law_opacity_coefficient'] = copy.deepcopy(
                    value['power_law_opacity_coefficient'])
            else:
                spectral_parameters[key] = copy.deepcopy(value)

        opacity_contributions['Total'] = self.__getattr__(spectral_function)(
            **spectral_parameters
        )

        # Contribution spectra
        _default_radtrans = None
        _default_mass_fractions = None

        for opacity_type, opacity_source in opacity_sources.items():
            if opacity_type == 'init':  # fixed (at instantiation) opacity sources
                for opacity, species_list in opacity_source.items():  # all fixed opacity sources are lists
                    opacity_contributions[opacity] = {}

                    # Init a temporary radtrans object with only the relevant opacity source
                    for species in species_list:
                        _user_cloud_species = None

                        # Use full names for CIA and clouds to have a clean legend when plotting the opacities
                        if opacity == 'gas_continuum_contributors':
                            species = cia.get_cia_aliases(species)
                        elif opacity == 'cloud_species':
                            _user_cloud_species = species
                            species = CloudOpacity.get_aliases(species)

                        fixed_opacity_sources = opacity_source_default[opacity_type]
                        fixed_opacity_sources[opacity] = [species]

                        print(f"Generating temporary Radtrans object with '{opacity}': '{species}'")

                        # Instantiate a default Radtrans object that will be used for runtime opacity sources
                        if _default_radtrans is None:
                            _default_radtrans = Radtrans(
                                pressures=self._pressures * 1e-6,  # cgs to bar
                                wavelength_boundaries=self.wavelength_boundaries,
                                line_opacity_mode=self.line_opacity_mode,
                                line_by_line_opacity_sampling=self.line_by_line_opacity_sampling,
                                scattering_in_emission=self.scattering_in_emission,
                                emission_angle_grid=self.emission_angle_grid,
                                anisotropic_cloud_scattering=self.anisotropic_cloud_scattering,
                                path_input_data=self.path_input_data,
                                **fixed_opacity_sources
                            )

                        if _default_mass_fractions is None:
                            _default_mass_fractions = {species: jnp.zeros(self._pressures.size, dtype=JAX_DTYPE)}

                        # Use the first line species in order to keep the resolving power consistent in all spectra
                        if opacity != 'line_species':
                            fixed_opacity_sources['line_species'] = _default_radtrans.line_species

                        _radtrans = Radtrans(
                            pressures=self._pressures * 1e-6,  # cgs to bar
                            wavelength_boundaries=self.wavelength_boundaries,
                            line_opacity_mode=self.line_opacity_mode,
                            line_by_line_opacity_sampling=self.line_by_line_opacity_sampling,
                            scattering_in_emission=self.scattering_in_emission,
                            emission_angle_grid=self.emission_angle_grid,
                            anisotropic_cloud_scattering=self.anisotropic_cloud_scattering,
                            path_input_data=self.path_input_data,
                            **fixed_opacity_sources
                        )

                        # Generate the spectrum containing only the relevant fixed opacity source
                        spectral_parameters = {}

                        # Fill the spectral parameters with default values for all runtime opacity sources
                        for _key, _value in kwargs.items():
                            if _key in opacity_source_default['runtime']:
                                spectral_parameters[_key] = copy.deepcopy(opacity_source_default['runtime'][_key])
                            elif _key == 'mass_fractions':
                                spectral_parameters[_key] = copy.deepcopy(_value)

                                # Fixes the first line species MMR to 0 so that it does not impact the spectrum
                                if species not in _default_mass_fractions:
                                    spectral_parameters[_key].update(_default_mass_fractions)
                            else:  # non-opacity spectral parameters
                                spectral_parameters[_key] = copy.deepcopy(_value)

                        # Update the user's cloud name to the full one everywhere it is needed to prevent crashes
                        if opacity == 'cloud_species':
                            if _user_cloud_species != species:
                                for _key, _value in spectral_parameters.items():
                                    if isinstance(_value, dict):
                                        if _user_cloud_species in _value:
                                            _value[species] = copy.deepcopy(_value[_user_cloud_species])
                                            del _value[_user_cloud_species]

                        opacity_contributions[opacity][species] = _radtrans.__getattr__(spectral_function)(
                            **spectral_parameters
                        )

                        del _radtrans  # free up some memory
            else:  # variable (at runtime) opacity sources
                for opacity, opacity_value in opacity_source.items():
                    if opacity_value is None:
                        opacity_contributions[opacity] = None
                        continue

                    spectral_parameters = {}

                    # Handle power law dual parameters special case
                    if opacity == 'power_law':
                        for key, value in opacity_value.items():
                            spectral_parameters[key] = copy.deepcopy(value)

                        if jnp.any(jnp.equal(list(spectral_parameters.values()), None)):
                            if not jnp.all(jnp.equal(list(spectral_parameters.values()), None)):
                                missing = []

                                for k, v in spectral_parameters.items():
                                    if v is None:
                                        missing.append(k)

                                missing = "'" + ", '".join(missing) + "'"

                                debug.print(f"opacity '{opacity}' requires {len(spectral_parameters)} parameters, "
                                              f"but lacks value for parameters {missing}")

                            opacity_contributions[opacity] = None
                            continue
                    else:
                        spectral_parameters[opacity] = copy.deepcopy(opacity_value)

                    # Fill the spectral parameters with default values for non-relevant opacity sources
                    for key, value in kwargs.items():
                        if key in opacity_source_default['runtime']:
                            spectral_parameters[key] = copy.deepcopy(opacity_source_default['runtime'][key])
                        elif key == 'mass_fractions':
                            spectral_parameters[key] = copy.deepcopy(_default_mass_fractions)
                        else:  # non-opacity spectral parameters
                            spectral_parameters[key] = copy.deepcopy(value)

                    opacity_contributions[opacity] = _default_radtrans.__getattr__(spectral_function)(
                        **spectral_parameters
                    )

        return opacity_contributions

    # TODO: JIT here, or leave to user?
    def calculate_flux(
        self,
        temperatures: ArrayLike,
        mass_fractions: dict[str, ArrayLike],
        mean_molar_masses: ArrayLike,
        reference_gravity: float,
        planet_radius: float = None,
        opaque_cloud_top_pressure: float = None,
        cloud_particles_mean_radii: dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution_std: float = None,
        cloud_particles_radius_distribution: str = 'lognormal',
        cloud_hansen_a: dict[str, ArrayLike] = None,
        cloud_hansen_b: dict[str, ArrayLike] = None,
        cloud_particle_number_density_grid: dict[str, ArrayLike] = None,
        clouds_particles_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: ArrayLike = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_photosphere_wavelength_boundaries: ArrayLike = None,
        cloud_photosphere_median_optical_depth: float = None,
        cloud_fraction: float = 1.0,
        complete_coverage_clouds: list[str] = None,
        irradiation_geometry: str = 'dayside_ave',
        emission_geometry: str = None,  # TODO deprecated, replace with irradiation_geometry everywhere in the code
        stellar_intensities: ArrayLike = None,
        star_effective_temperature: float = None,
        star_radius: float = None,
        orbit_semi_major_axis: float = None,
        star_irradiation_angle: float = 0.0,
        reflectances: ArrayLike = None,
        emissivities: ArrayLike = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        adaptive_feautrier_iterations: bool = False,
        chunk_feautrier: bool = False,
        feautrier_chunk_size: int = 1024,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_photosphere_radius: bool = False,
        return_rosseland_optical_depths: bool = False,
        return_cloud_contribution: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_particles_radii_bins: bool = False,
        return_particles_radii: bool = False,
        target_particle_radii: ArrayLike = None,
        target_pressure: ArrayLike = None
    ) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
        """Method to calculate the atmosphere's emitted flux (emission spectrum).

            Args:
                temperatures:
                    The atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    Dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names. Every mass fraction array has same length as pressure array.
                mean_molar_masses:
                    The atmospheric mean molecular weight in amu, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission spectra.
                planet_radius: planet radius at maximum pressure in cm. Only used to calculate the planet's changing
                    Photospheric radius as function of wavelength, if return_photosphere_radius is True.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the absorption opacity.
                cloud_particles_mean_radii (Optional):
                    Dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names. Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    Width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. Equivalent to cloud_particles_mean_radii.
                    If cloud_hansen_a is not included and dist is "hansen", then it will be computed using Kzz and fsed
                    (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. This is the width of the hansen distribution normalized by
                    the particle area (1/cloud_hansen_a^2)
                cloud_particle_number_density_grid (Optional[dict]):
                    EXPERIMENTAL
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    Cloud settling parameter.
                eddy_diffusion_coefficients (Optional[float]):
                    The atmospheric eddy diffusion coefficient in cgs (i.e. :math:`\\rm cm^2/s`), at each atmospheric
                    layer (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all pressures and wavelengths
                    (units :math:`\\rm cm^2/g`)
                cloud_photosphere_median_optical_depth (Optional[float]):
                    Median optical depth (across ``wavelength_boundaries``) of the clouds from the top of the
                    atmosphere down to the gas-only photosphere. This parameter can be used for enforcing the presence
                    of clouds in the photospheric region.
                cloud_photosphere_wavelength_boundaries (Optional[NDArray]):
                    Min and max wavelength boundaries when calculating the median of the photospheric cloud optical
                    depth. By default, the whole spectral range is used.
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). By default, assume a full cloud cover.
                complete_coverage_clouds:
                    List of clouds that have full coverage, regardless of the value of cloud_fraction.
                irradiation_geometry (Optional[string]):
                    if equal to ``'dayside_ave'``: use the dayside average geometry.
                    If equal to ``'planetary_ave'``: use the planetary average geometry.
                    If equal to ``'non-isotropic'``: use the non-isotropic geometry.
                emission_geometry (Optional[string]):
                    Deprecated, same as irradiation_geometry.
                stellar_intensities (Optional[array]):
                    The stellar intensity to use. If None, it will be calculated using a PHOENIX model.
                star_effective_temperature (Optional[float]):
                    The temperature of the host star in K, used only if the
                    scattering is considered. If not specified, the direct light contribution is not calculated.
                star_radius (Optional[float]):
                    The radius of the star in cm. If specified, used to scale the to scale the stellar flux,
                    otherwise it uses PHOENIX radius.
                orbit_semi_major_axis (Optional[float]):
                    The distance of the planet from the star. Used to scale the stellar flux when the scattering of the
                    direct light is considered.
                star_irradiation_angle (Optional[float]):
                    Inclination angle of the direct light with respect to the normal to the atmosphere. Used only in
                    the non-isotropic geometry scenario.
                reflectances (Optional):
                    Reflectances of the surface (layer with the highest pressure).
                emissivities (Optional):
                    Emissivities of the surface (layer with the highest pressure).
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                adaptive_feautrier_iterations (Optional[bool]):
                    If True, adapt the number of iterations of the Feautrier scattering-method
                    (scattering_in_emission=True) based on the photon destruction probability.
                frequencies_to_wavelengths (Optional[bool]):
                    if True, convert the frequencies (Hz) output to wavelengths (cm),
                    and the flux per frequency output (erg.s-1.cm-2/Hz) to flux per wavelength (erg.s-1.cm-2/cm)
                return_contribution (Optional[bool]):
                    If True, calculate the emission contribution function.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_photosphere_radius (Optional[bool]):
                    If True, calculate and return the photosphere radius.
                return_rosseland_optical_depths (Optional[bool]):
                    If True, calculate and return the Rosseland opacities and optical depths.
                return_cloud_contribution (Optional[bool]):
                    If True, calculate the cloud contribution.
                return_opacities (Optional[bool]):
                    If True, return the absorption opacities and scattering opacities for species and clouds, as well
                    as optical depths
                return_abundances (Optional[bool]):
                    If True, return the mass fractions.
                return_particles_radii_bins (Optional[bool]):
                    If True, return the particle radii bins.
                return_particles_radii (Optional[bool]):
                    If True, return the particle radii.
                target_particle_radii (Optional[ArrayLike]):
                    Optional particle radii grid used when evaluating cloud
                    opacities at specific particle sizes instead of the native
                    opacity table grid.
                target_pressure (Optional[ArrayLike]):
                    Optional pressure grid associated with
                    ``target_particle_radii`` when cloud properties are
                    requested on a custom grid.
        """
        (anisotropic_cloud_scattering, complete_coverage_clouds, flux_clear,
         sum_opacities) = (
            self._init_spectral_function(
                is_emission=True,
                reference_gravity=reference_gravity,
                complete_coverage_clouds=complete_coverage_clouds,
                return_clear_spectrum=return_clear_spectrum,
                mass_fractions=mass_fractions,
                cloud_fraction=cloud_fraction,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure
            )
        )

        # TODO JIT - replace with debug.print
        if opaque_cloud_top_pressure is not None and self._scattering_in_emission:
            debug.print("the use of opaque_cloud_top_pressure in conjunction with scattering for emission spectrum "
                        "is not recommended")

        emission_inputs = self._sanitize_emission_surface_inputs(
            star_irradiation_angle=star_irradiation_angle,
            stellar_intensities=stellar_intensities,
            star_effective_temperature=star_effective_temperature,
            star_radius=star_radius,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reflectances=reflectances,
            emissivities=emissivities,
        )
        sanitized_inputs = self._sanitize_flux_opacity_inputs(
            complete_coverage_clouds=complete_coverage_clouds,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )
        partial_coverage_cloud_species = self._get_partial_coverage_cloud_species(
            sanitized_inputs['complete_coverage_clouds']
        )
        should_compute_clear_opacities = (
            len(partial_coverage_cloud_species) > 0
            and (return_clear_spectrum or cloud_fraction < 1)
        )

        photosphere_radius = None
        flux_clear = None

        (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, cloud_particles_mean_radii, cloud_scattering_in_transmission,
         clear_opacities, clear_continuum_opacities_scattering,
         clear_cloud_anisotropic_scattering_opacities, clear_cloud_absorption_opacities) = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                opaque_cloud_top_pressure=sanitized_inputs['opaque_cloud_top_pressure'],
                cloud_particles_mean_radii=sanitized_inputs['cloud_particles_mean_radii'],
                cloud_particle_radius_distribution_std=sanitized_inputs['cloud_particle_radius_distribution_std'],
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=sanitized_inputs['cloud_hansen_a'],
                cloud_hansen_b=sanitized_inputs['cloud_hansen_b'],
                cloud_particle_number_density_grid=sanitized_inputs['cloud_particle_number_density_grid'],
                clouds_particles_porosity_factor=sanitized_inputs['clouds_particles_porosity_factor'],
                cloud_f_sed=sanitized_inputs['cloud_f_sed'],
                eddy_diffusion_coefficients=sanitized_inputs['eddy_diffusion_coefficients'],
                haze_factor=haze_factor,
                power_law_opacity_350nm=sanitized_inputs['power_law_opacity_350nm'],
                power_law_opacity_coefficient=sanitized_inputs['power_law_opacity_coefficient'],
                gray_opacity=sanitized_inputs['gray_opacity'],
                cloud_photosphere_median_optical_depth=sanitized_inputs['cloud_photosphere_median_optical_depth'],
                cloud_fraction=cloud_fraction,
                complete_coverage_clouds=sanitized_inputs['complete_coverage_clouds'],
                return_cloud_contribution=return_cloud_contribution,
                additional_absorption_opacities_function=sanitized_inputs['additional_absorption_opacities_function'],
                additional_scattering_opacities_function=sanitized_inputs['additional_scattering_opacities_function'],
                compute_clear_opacities=should_compute_clear_opacities,
                sum_opacities=sum_opacities,
                anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                target_particle_radii=sanitized_inputs['target_particle_radii'],
                target_pressure=sanitized_inputs['target_pressure']
            )
        )
        self.__scattering_in_transmission = cloud_scattering_in_transmission

        if return_rosseland_optical_depths:
            debug.print(
                "return_rosseland_optical_depths is no longer supported in calculate_flux; "
                "use calculate_rosseland_planck_opacities instead"
            )

        flux, emission_contribution, optical_depths, relative_cloud_scaling_factor = (
            self._calculate_flux(
                temperatures=temperatures, #ATMO
                reference_gravity=reference_gravity, # ATMO
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                emission_geometry=irradiation_geometry, # STAR
                star_irradiation_cos_angle=emission_inputs['star_irradiation_cos_angle'], # STAR
                stellar_intensity=emission_inputs['stellar_intensities'], # STAR
                reflectances=emission_inputs['reflectances'],
                emissivities=emission_inputs['emissivities'],
                return_contribution=return_contribution, # NOTE: Separate out
                cloud_f_sed=cloud_f_sed, # CLOUD
                photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth, # CLOUD
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries, # ??
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities, # CLOUD
                cloud_absorption_opacities=cloud_absorption_opacities, # CLOUD
                adaptive_feautrier_iterations=adaptive_feautrier_iterations, # default
                sum_opacities=sum_opacities,
                chunk_feautrier=chunk_feautrier,
                feautrier_chunk_size=feautrier_chunk_size,
            )
        )

        if len(partial_coverage_cloud_species) > 0 and (return_clear_spectrum or cloud_fraction < 1):
            flux_clear, _, _, _ = (
                self._calculate_flux(
                    temperatures=temperatures,
                    reference_gravity=reference_gravity,
                    opacities=clear_opacities,
                    continuum_opacities_scattering=clear_continuum_opacities_scattering,
                    emission_geometry=irradiation_geometry,
                    star_irradiation_cos_angle=emission_inputs['star_irradiation_cos_angle'],
                    stellar_intensity=emission_inputs['stellar_intensities'],
                    reflectances=emission_inputs['reflectances'],
                    emissivities=emission_inputs['emissivities'],
                    return_contribution=return_contribution,
                    cloud_f_sed=cloud_f_sed,
                    cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                    cloud_anisotropic_scattering_opacities=clear_cloud_anisotropic_scattering_opacities,
                    cloud_absorption_opacities=clear_cloud_absorption_opacities,
                    adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                    sum_opacities=sum_opacities,
                    chunk_feautrier=chunk_feautrier,
                    feautrier_chunk_size=feautrier_chunk_size,
                )
            )

            flux = (
                cloud_fraction * flux
                + (1 - cloud_fraction) * flux_clear
            )
        elif return_clear_spectrum:
            flux_clear = flux

        photosphere_radius = self._maybe_calculate_photosphere_radius(
            planet_radius=planet_radius,
            return_photosphere_radius=return_photosphere_radius,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            cloud_f_sed=cloud_f_sed,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            optical_depths=optical_depths,
        )

        # TODO separate out cloud contribution function
        if self._clouds_have_effect(mass_fractions, cloud_fraction) and return_cloud_contribution:
            cloud_contribution = clouds.compute_species_optical_depths(
                reference_gravity=reference_gravity,
                pressures=self._pressures,
                cloud_opacities=cloud_opacities
            )
        else:
            cloud_contribution = None

        # TODO wrap in helper function
        additional_outputs = {}

        if cloud_particles_mean_radii is not None:
            additional_outputs['cloud_particles_mean_radii'] = cloud_particles_mean_radii

        if return_particles_radii_bins:
            additional_outputs['particles_radii_bins'] = self._clouds_loaded_opacities['particles_radii_bins']

        if return_particles_radii:
            additional_outputs['particles_radii'] = self._clouds_loaded_opacities['particles_radii']

        if emission_inputs['stellar_intensities'] is not None:
            additional_outputs['stellar_intensities'] = emission_inputs['stellar_intensities']

        if return_contribution:
            additional_outputs['emission_contribution'] = emission_contribution

        if return_clear_spectrum:
            if frequencies_to_wavelengths:
                additional_outputs['clear_spectrum'] = flux_hz2flux_cm(
                    flux_hz=flux_clear,
                    frequency=self._frequencies
                )
            else:
                additional_outputs['clear_spectrum'] = flux_clear

        if return_photosphere_radius:
            additional_outputs['photosphere_radius'] = photosphere_radius

        if return_cloud_contribution:
            additional_outputs['cloud_contribution'] = cloud_contribution

        if return_opacities:
            additional_outputs['opacities'] = opacities
            additional_outputs['continuum_opacities_scattering'] = continuum_opacities_scattering
            additional_outputs['cloud_absorption_opacities'] = cloud_absorption_opacities
            additional_outputs['cloud_anisotropic_scattering_opacities'] = cloud_anisotropic_scattering_opacities
            additional_outputs['optical_depths'] = optical_depths
        if return_abundances:
            additional_outputs['abundances'] = mass_fractions
        if relative_cloud_scaling_factor is not None:
            additional_outputs['relative_cloud_scaling_factor'] = relative_cloud_scaling_factor

        if frequencies_to_wavelengths:
            return (
                frequency2wavelength(self._frequencies),
                flux_hz2flux_cm(
                    flux_hz=flux,
                    frequency=self._frequencies
                ),
                additional_outputs
            )
        else:
            return self._frequencies, flux, additional_outputs

    def simple_calculate_flux(
            self,
            temperatures: ArrayLike,
            mass_fractions: dict[str, ArrayLike],
            mean_molar_masses: ArrayLike,
            reference_gravity: float,
            planet_radius: float = None,
            opaque_cloud_top_pressure: float = None,
            cloud_particles_mean_radii: dict[str, ArrayLike] = None,
            cloud_particle_radius_distribution_std: float = None,
            cloud_particles_radius_distribution: str = 'lognormal',
            cloud_hansen_a: dict[str, ArrayLike] = None,
            cloud_hansen_b: dict[str, ArrayLike] = None,
            clouds_particles_porosity_factor: dict[str, float] = None,
            cloud_f_sed: float = None,
            eddy_diffusion_coefficients: ArrayLike = None,
            haze_factor: float = 1.0,
            power_law_opacity_350nm: float = None,
            power_law_opacity_coefficient: float = None,
            gray_opacity: float = None,
            cloud_photosphere_wavelength_boundaries: ArrayLike = None,
            cloud_photosphere_median_optical_depth: float = None,
            cloud_fraction: float = 1.0,
            complete_coverage_clouds: list[str] = None,
            adaptive_feautrier_iterations: bool = False,
            irradiation_geometry: str = 'dayside_ave',
            stellar_intensities: ArrayLike = None,
            star_effective_temperature: float = None,
            star_radius: float = None,
            orbit_semi_major_axis: float = None,
            star_irradiation_angle: float = 0.0,
            reflectances: ArrayLike = None,
            emissivities: ArrayLike = None,
            additional_absorption_opacities_function: callable = None,
            additional_scattering_opacities_function: callable = None,
            chunk_feautrier: bool = False,
            feautrier_chunk_size: int = 1024,
            frequencies_to_wavelengths: bool = True,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, any]]:
        """JAX-friendly emission-spectrum wrapper around the pure simple-flux kernel."""
        (anisotropic_cloud_scattering, complete_coverage_clouds, _,
         sum_opacities) = self._init_spectral_function(
            is_emission=True,
            reference_gravity=reference_gravity,
            complete_coverage_clouds=complete_coverage_clouds,
            return_clear_spectrum=False,
            mass_fractions=mass_fractions,
            cloud_fraction=cloud_fraction,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        )

        return self._simple_calculate_flux_kernel(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            planet_radius=planet_radius,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            haze_factor=haze_factor,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_fraction=cloud_fraction,
            complete_coverage_clouds=complete_coverage_clouds,
            adaptive_feautrier_iterations=adaptive_feautrier_iterations,
            irradiation_geometry=irradiation_geometry,
            stellar_intensities=stellar_intensities,
            star_effective_temperature=star_effective_temperature,
            star_radius=star_radius,
            orbit_semi_major_axis=orbit_semi_major_axis,
            star_irradiation_angle=star_irradiation_angle,
            reflectances=reflectances,
            emissivities=emissivities,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            chunk_feautrier=chunk_feautrier,
            feautrier_chunk_size=feautrier_chunk_size,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
        )

    def calculate_flux_autodiff(
            self,
            temperatures: ArrayLike,
            mass_fractions: dict[str, ArrayLike],
            mean_molar_masses: ArrayLike,
            reference_gravity: float,
            *,
            cloud_parameters: dict[str, Any] | None = None,
            irradiation_geometry: str = 'dayside_ave',
            stellar_intensities: ArrayLike = None,
            star_irradiation_angle: float = 0.0,
            reflectances: ArrayLike = None,
            emissivities: ArrayLike = None,
            additional_absorption_opacities_function: Callable = None,
            additional_scattering_opacities_function: Callable = None,
            chunk_feautrier: bool = False,
            feautrier_chunk_size: int = 1024,
            frequencies_to_wavelengths: bool = True,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, any]]:
        r"""Autodiff-oriented emission-spectrum API with grouped cloud controls.

        The method uses the same pure JAX kernel as simple_calculate_flux, but collapses the
        cloud-configuration surface into a single keyword argument and only accepts direct
        stellar intensities on the irradiation side.

        Example:
            >>> cloud_parameters = {
            ...     'cloud_f_sed': cloud_f_sed,
            ...     'eddy_diffusion_coefficients': eddy_diffusion_coefficients,
            ...     'cloud_particle_radius_distribution_std': 1.05,
            ... }
            >>> _, flux, _ = atmosphere.calculate_flux_autodiff(
            ...     temperatures=temperatures,
            ...     mass_fractions=mass_fractions,
            ...     mean_molar_masses=mean_molar_masses,
            ...     reference_gravity=reference_gravity,
            ...     cloud_parameters=cloud_parameters,
            ...     frequencies_to_wavelengths=False,
            ... )
            >>> temperature_gradient = jax.grad(
            ...     lambda t: jnp.sum(
            ...         atmosphere.calculate_flux_autodiff(
            ...             temperatures=t,
            ...             mass_fractions=mass_fractions,
            ...             mean_molar_masses=mean_molar_masses,
            ...             reference_gravity=reference_gravity,
            ...             cloud_parameters=cloud_parameters,
            ...             frequencies_to_wavelengths=False,
            ...         )[1]
            ...     )
            ... )(temperatures)
        """
        allowed_cloud_parameters = {
            'planet_radius',
            'opaque_cloud_top_pressure',
            'cloud_particles_mean_radii',
            'cloud_particle_radius_distribution_std',
            'cloud_particles_radius_distribution',
            'cloud_hansen_a',
            'cloud_hansen_b',
            'clouds_particles_porosity_factor',
            'cloud_f_sed',
            'eddy_diffusion_coefficients',
            'haze_factor',
            'power_law_opacity_350nm',
            'power_law_opacity_coefficient',
            'gray_opacity',
            'cloud_photosphere_wavelength_boundaries',
            'cloud_photosphere_median_optical_depth',
            'cloud_fraction',
            'complete_coverage_clouds',
            'adaptive_feautrier_iterations',
        }
        cloud_parameters = {} if cloud_parameters is None else dict(cloud_parameters)
        unsupported_cloud_parameters = sorted(set(cloud_parameters) - allowed_cloud_parameters)

        if unsupported_cloud_parameters:
            raise ValueError(
                "unsupported calculate_flux_autodiff cloud parameters: "
                f"{', '.join(unsupported_cloud_parameters)}"
            )

        return self.simple_calculate_flux(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            irradiation_geometry=irradiation_geometry,
            stellar_intensities=stellar_intensities,
            star_irradiation_angle=star_irradiation_angle,
            reflectances=reflectances,
            emissivities=emissivities,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            chunk_feautrier=chunk_feautrier,
            feautrier_chunk_size=feautrier_chunk_size,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            **cloud_parameters,
        )

    @jit(static_argnames=[
                'self',
                'cloud_particles_radius_distribution',
                'adaptive_feautrier_iterations',
                'irradiation_geometry',
                'chunk_feautrier',
                'feautrier_chunk_size',
                'frequencies_to_wavelengths',
                'complete_coverage_clouds',
                'sum_opacities',
                'anisotropic_cloud_scattering',
                ])
    def _simple_calculate_flux_kernel(
            self,
            temperatures: ArrayLike,
            mass_fractions: dict[str, ArrayLike],
            mean_molar_masses: ArrayLike,
            reference_gravity: float,
            planet_radius: float = None,
            opaque_cloud_top_pressure: float = None,
            cloud_particles_mean_radii: dict[str, ArrayLike] = None,
            cloud_particle_radius_distribution_std: float = None,
            cloud_particles_radius_distribution: str = 'lognormal',
            cloud_hansen_a: dict[str, ArrayLike] = None,
            cloud_hansen_b: dict[str, ArrayLike] = None,
            clouds_particles_porosity_factor: dict[str, float] = None,
            cloud_f_sed: float = None,
            eddy_diffusion_coefficients: ArrayLike = None,
            haze_factor: float = 1.0,
            power_law_opacity_350nm: float = None,
            power_law_opacity_coefficient: float = None,
            gray_opacity: float = None,
            cloud_photosphere_wavelength_boundaries: ArrayLike = None,
            cloud_photosphere_median_optical_depth: float = None,
            cloud_fraction: float = 1.0,
            complete_coverage_clouds: list[str] = None,
            adaptive_feautrier_iterations: bool = False,
            irradiation_geometry: str = 'dayside_ave',
            stellar_intensities: ArrayLike = None,
            star_effective_temperature: float = None,
            star_radius: float = None,
            orbit_semi_major_axis: float = None,
            star_irradiation_angle: float = 0.0,
            reflectances: ArrayLike = None,
            emissivities: ArrayLike = None,
            additional_absorption_opacities_function: callable = None,
            additional_scattering_opacities_function: callable = None,
            chunk_feautrier: bool = False,
            feautrier_chunk_size: int = 1024,
            sum_opacities: bool = None,
            anisotropic_cloud_scattering: bool = None,
            frequencies_to_wavelengths: bool = True,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, any]]:
        """ Method to calculate the atmosphere's emitted flux (emission spectrum).

            Args:
                temperatures:
                    The atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    Dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names. Every mass fraction array has same length as pressure array.
                mean_molar_masses:
                    The atmospheric mean molecular weight in amu, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission spectra.
                planet_radius: planet radius at maximum pressure in cm. Only used to calculate the planet's changing
                    Photospheric radius as function of wavelength, if return_photosphere_radius is True.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the absorption opacity.
                cloud_particles_mean_radii (Optional):
                    Dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names. Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    Width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. Equivalent to cloud_particles_mean_radii.
                    If cloud_hansen_a is not included and dist is "hansen", then it will be computed using Kzz and fsed
                    (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each included cloud species and for each atmospheric
                    layer, formatted as the kzz argument. This is the width of the hansen distribution normalized by
                    the particle area (1/cloud_hansen_a^2)
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    Cloud settling parameter.
                eddy_diffusion_coefficients (Optional[float]):
                    The atmospheric eddy diffusion coefficient in cgs (i.e. :math:`\\rm cm^2/s`), at each atmospheric
                    layer (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all pressures and wavelengths
                    (units :math:`\\rm cm^2/g`)
                cloud_photosphere_median_optical_depth (Optional[float]):
                    Median optical depth (across ``wavelength_boundaries``) of the clouds from the top of the
                    atmosphere down to the gas-only photosphere. This parameter can be used for enforcing the presence
                    of clouds in the photospheric region.
                cloud_photosphere_wavelength_boundaries (Optional[NDArray]):
                    Min and max wavelength boundaries when calculating the median of the photospheric cloud optical
                    depth. By default, the whole spectral range is used.
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). By default, assume a full cloud cover.
                complete_coverage_clouds:
                    List of clouds that have full coverage, regardless of the value of cloud_fraction.
                irradiation_geometry (Optional[string]):
                    if equal to ``'dayside_ave'``: use the dayside average geometry.
                    If equal to ``'planetary_ave'``: use the planetary average geometry.
                    If equal to ``'non-isotropic'``: use the non-isotropic geometry.
                emission_geometry (Optional[string]):
                    Deprecated, same as irradiation_geometry.
                stellar_intensities (Optional[array]):
                    The stellar intensity to use. If None, it will be calculated using a PHOENIX model.
                star_effective_temperature (Optional[float]):
                    The temperature of the host star in K, used only if the
                    scattering is considered. If not specified, the direct light contribution is not calculated.
                star_radius (Optional[float]):
                    The radius of the star in cm. If specified, used to scale the to scale the stellar flux,
                    otherwise it uses PHOENIX radius.
                orbit_semi_major_axis (Optional[float]):
                    The distance of the planet from the star. Used to scale the stellar flux when the scattering of the
                    direct light is considered.
                star_irradiation_angle (Optional[float]):
                    Inclination angle of the direct light with respect to the normal to the atmosphere. Used only in
                    the non-isotropic geometry scenario.
                reflectances (Optional):
                    Reflectances of the surface (layer with the highest pressure).
                emissivities (Optional):
                    Emissivities of the surface (layer with the highest pressure).
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                adaptive_feautrier_iterations (Optional[bool]):
                    If True, the number of iterations of the Feautrier scattering-method
                    (scattering_in_emission=True) is adapted based on the photon destruction probability.
                frequencies_to_wavelengths (Optional[bool]):
                    if True, convert the frequencies (Hz) output to wavelengths (cm),
                    and the flux per frequency output (erg.s-1.cm-2/Hz) to flux per wavelength (erg.s-1.cm-2/cm)
                return_contribution (Optional[bool]):
                    If True the emission contribution function is be calculated.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_photosphere_radius (Optional[bool]):
                    if True, the photosphere radius is calculated and returned
                return_rosseland_optical_depths (Optional[bool]):
                    if True, the Rosseland opacities and optical depths are calculated and returned
                return_cloud_contribution (Optional[bool]):
                    if True, the cloud contribution is calculated
                return_opacities (Optional[bool]):
                    if True, the absorption opacities and scattering opacities for species and clouds, as well as the
                    optical depths, are returned
        """
        if opaque_cloud_top_pressure is not None and self._scattering_in_emission:
            debug.print("the use of opaque_cloud_top_pressure in conjunction with scattering for emission spectrum "
                        "is not recommended")

        emission_inputs = self._sanitize_emission_surface_inputs(
            star_irradiation_angle=star_irradiation_angle,
            stellar_intensities=stellar_intensities,
            star_effective_temperature=star_effective_temperature,
            star_radius=star_radius,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reflectances=reflectances,
            emissivities=emissivities,
        )
        sanitized_inputs = self._sanitize_flux_opacity_inputs(
            complete_coverage_clouds=complete_coverage_clouds,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_particle_number_density_grid=None,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=None,
            target_pressure=None,
        )
        partial_coverage_cloud_species = self._get_partial_coverage_cloud_species(
            sanitized_inputs['complete_coverage_clouds']
        )
        should_compute_clear_opacities = jnp.logical_and(
            len(partial_coverage_cloud_species) > 0,
            jnp.less(cloud_fraction, 1),
        )

        (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, cloud_particles_mean_radii, cloud_scattering_in_transmission,
         clear_opacities, clear_continuum_opacities_scattering,
         clear_cloud_anisotropic_scattering_opacities, clear_cloud_absorption_opacities) = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                opaque_cloud_top_pressure=sanitized_inputs['opaque_cloud_top_pressure'],
                cloud_particles_mean_radii=sanitized_inputs['cloud_particles_mean_radii'],
                cloud_particle_radius_distribution_std=sanitized_inputs['cloud_particle_radius_distribution_std'],
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=sanitized_inputs['cloud_hansen_a'],
                cloud_hansen_b=sanitized_inputs['cloud_hansen_b'],
                cloud_particle_number_density_grid=sanitized_inputs['cloud_particle_number_density_grid'],
                clouds_particles_porosity_factor=sanitized_inputs['clouds_particles_porosity_factor'],
                cloud_f_sed=sanitized_inputs['cloud_f_sed'],
                eddy_diffusion_coefficients=sanitized_inputs['eddy_diffusion_coefficients'],
                haze_factor=haze_factor,
                power_law_opacity_350nm=sanitized_inputs['power_law_opacity_350nm'],
                power_law_opacity_coefficient=sanitized_inputs['power_law_opacity_coefficient'],
                gray_opacity=sanitized_inputs['gray_opacity'],
                cloud_photosphere_median_optical_depth=sanitized_inputs['cloud_photosphere_median_optical_depth'],
                cloud_fraction=cloud_fraction,
                complete_coverage_clouds=sanitized_inputs['complete_coverage_clouds'],
                return_cloud_contribution=False,
                additional_absorption_opacities_function=sanitized_inputs['additional_absorption_opacities_function'],
                additional_scattering_opacities_function=sanitized_inputs['additional_scattering_opacities_function'],
                compute_clear_opacities=should_compute_clear_opacities,
                sum_opacities=sum_opacities,
                anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                target_particle_radii=sanitized_inputs['target_particle_radii'],
                target_pressure=sanitized_inputs['target_pressure']
            ))

        flux, _, _, _ = (
            self._calculate_flux(
                temperatures=temperatures,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                emission_geometry=irradiation_geometry,
                star_irradiation_cos_angle=emission_inputs['star_irradiation_cos_angle'],
                stellar_intensity=emission_inputs['stellar_intensities'],
                reflectances=emission_inputs['reflectances'],
                emissivities=emission_inputs['emissivities'],
                return_contribution=False,
                cloud_f_sed=cloud_f_sed,
                photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=cloud_absorption_opacities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                sum_opacities=sum_opacities,
                chunk_feautrier=chunk_feautrier,
                feautrier_chunk_size=feautrier_chunk_size,
            )
        )

        if len(partial_coverage_cloud_species) > 0:
            def _blend_partial_cloud_flux(cloudy_flux):
                clear_flux, _, _, _ = self._calculate_flux(
                    temperatures=temperatures,
                    reference_gravity=reference_gravity,
                    opacities=clear_opacities,
                    continuum_opacities_scattering=clear_continuum_opacities_scattering,
                    emission_geometry=irradiation_geometry,
                    star_irradiation_cos_angle=emission_inputs['star_irradiation_cos_angle'],
                    stellar_intensity=emission_inputs['stellar_intensities'],
                    reflectances=emission_inputs['reflectances'],
                    emissivities=emission_inputs['emissivities'],
                    cloud_f_sed=cloud_f_sed,
                    cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                    cloud_anisotropic_scattering_opacities=clear_cloud_anisotropic_scattering_opacities,
                    cloud_absorption_opacities=clear_cloud_absorption_opacities,
                    adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                    sum_opacities=sum_opacities,
                    return_contribution=False,
                    chunk_feautrier=chunk_feautrier,
                    feautrier_chunk_size=feautrier_chunk_size,
                )

                return (
                    cloud_fraction * cloudy_flux
                    + (1 - cloud_fraction) * clear_flux
                )

            flux = lax.cond(
                jnp.less(cloud_fraction, 1),
                _blend_partial_cloud_flux,
                lambda cloudy_flux: cloudy_flux,
                flux
            )

        # TODO: properly deal with additional outputs
        output = lax.cond(
            frequencies_to_wavelengths,
            lambda f: (frequency2wavelength(f),
                       flux_hz2flux_cm(
                           flux_hz=flux,
                           frequency=f), {}),
            lambda f: (f, flux, {}),
            self._frequencies
            )
        return output

    def calculate_photosphere_radius(
            self,
            temperatures: ArrayLike,
            mean_molar_masses: ArrayLike,
            reference_gravity: float,
            planet_radius: float,
            opacities: ArrayLike,
            continuum_opacities_scattering: ArrayLike,
            cloud_f_sed: float,
            cloud_photosphere_median_optical_depth: float,
            cloud_anisotropic_scattering_opacities: ArrayLike,
            cloud_absorption_opacities: ArrayLike,
            optical_depths: ArrayLike = None,
            cloud_photosphere_wavelength_boundaries: ArrayLike = None
    ) -> ArrayLike:
        """Estimate the wavelength-dependent photosphere radius.

        The photosphere radius is obtained by computing or reusing optical
        depths, locating the layer where the angle-averaged optical depth
        reaches $2/3$, and interpolating the hydrostatic radius profile to that
        pressure.

        Args:
            temperatures:
                Atmospheric temperature profile in K.
            mean_molar_masses:
                Mean molecular mass profile in amu.
            reference_gravity:
                Reference gravity in cgs units.
            planet_radius:
                Radius in cm at the deepest atmospheric layer.
            opacities:
                Total opacity array used for the emission calculation.
            continuum_opacities_scattering:
                Continuum scattering opacity array.
            cloud_f_sed:
                Cloud settling parameter used when photospheric cloud scaling is
                active.
            cloud_photosphere_median_optical_depth:
                Optional target cloud optical depth near the photosphere.
            cloud_anisotropic_scattering_opacities:
                Cloud anisotropic scattering opacities.
            cloud_absorption_opacities:
                Cloud absorption opacities.
            optical_depths:
                Precomputed optical depths. If omitted, they are recomputed.
            cloud_photosphere_wavelength_boundaries:
                Optional wavelength bounds used for the photospheric cloud model.

        Returns:
            Photosphere radius on the internal frequency grid, in cm.
        """
        sum_opacities = self._set_sum_opacities(
            emission=True,
            line_opacity_mode=self._line_opacity_mode,
            scattering_in_emission=self._scattering_in_emission,
        )

        if planet_radius is not None:
            radius_hydrostatic_equilibrium = self._calculate_radius_hydrostatic_equilibrium(
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                planet_radius=planet_radius,
            )

            optical_depths = self._get_photosphere_optical_depths(
                optical_depths=optical_depths,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                cloud_f_sed=cloud_f_sed,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=cloud_absorption_opacities,
                sum_opacities=sum_opacities,
            )

            return self._interpolate_photosphere_radius_profile(
                optical_depths=optical_depths,
                radius_hydrostatic_equilibrium=radius_hydrostatic_equilibrium,
            )

        return jnp.zeros(self._frequencies.size, dtype=JAX_DTYPE)

    # TODO Not used
    def calculate_rosseland_planck_opacities(
            self,
            temperatures: npt.NDArray[float],
            mass_fractions: dict[str, npt.NDArray[float]],
            mean_molar_masses: npt.NDArray[float],
            reference_gravity: float,
            opaque_cloud_top_pressure: float = None,
            cloud_particles_mean_radii: dict[str, ArrayLike] = None,
            cloud_particle_radius_distribution_std: float = None,
            cloud_particles_radius_distribution: str = 'lognormal',
            cloud_hansen_a: float = None,
            cloud_hansen_b: float = None,
            clouds_particles_porosity_factor: dict[str, float] = None,
            cloud_f_sed: float = None,
            eddy_diffusion_coefficients: float = None,
            haze_factor: float = 1.0,
            power_law_opacity_350nm: float = None,
            power_law_opacity_coefficient: float = None,
            gray_opacity: float = None
    ) -> tuple[npt.NDArray[float], npt.NDArray[float], dict[str, any]]:
        """ Method to calculate the atmosphere's Rosseland and Planck mean opacities.

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                mean_molar_masses:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs. Vertically constant for emission
                    spectra.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                cloud_particles_mean_radii (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If cloud_hansen_a is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/cloud_hansen_a^2)
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary containing the particle porosity factor for each cloud species.
                cloud_f_sed (Optional[float]):
                    cloud settling parameter
                eddy_diffusion_coefficients (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
        """
        if not self._scattering_in_emission:
            raise ValueError(
                "pRT must run in scattering_in_emission = True mode to calculate kappa_Rosseland and kappa_Planck'"
            )

        auto_anisotropic_cloud_scattering = False

        if self._anisotropic_cloud_scattering == 'auto':
            self._anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self._anisotropic_cloud_scattering:
            debug.print(f"anisotropic cloud scattering is recommended for emission spectra, "
                          f"but 'anisotropic_cloud_scattering' was set to {self._anisotropic_cloud_scattering}; "
                          f"set it to 'auto' to disable this warning")

        sanitized_inputs = self._sanitize_flux_opacity_inputs(
            complete_coverage_clouds=(),
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=None,
            cloud_particle_number_density_grid=None,
            additional_absorption_opacities_function=None,
            additional_scattering_opacities_function=None,
            target_particle_radii=None,
            target_pressure=None,
        )

        # No photospheric cloud in Rosseland or Planck opacities
        (opacities, continuum_opacities_scattering, _, _, _, cloud_particles_mean_radii,
         cloud_scattering_in_transmission, _, _, _, _) = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                opaque_cloud_top_pressure=sanitized_inputs['opaque_cloud_top_pressure'],
                cloud_particles_mean_radii=sanitized_inputs['cloud_particles_mean_radii'],
                cloud_particle_radius_distribution_std=sanitized_inputs['cloud_particle_radius_distribution_std'],
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=sanitized_inputs['cloud_hansen_a'],
                cloud_hansen_b=sanitized_inputs['cloud_hansen_b'],
                cloud_particle_number_density_grid=sanitized_inputs['cloud_particle_number_density_grid'],
                clouds_particles_porosity_factor=sanitized_inputs['clouds_particles_porosity_factor'],
                cloud_f_sed=sanitized_inputs['cloud_f_sed'],
                eddy_diffusion_coefficients=sanitized_inputs['eddy_diffusion_coefficients'],
                haze_factor=haze_factor,
                power_law_opacity_350nm=sanitized_inputs['power_law_opacity_350nm'],
                power_law_opacity_coefficient=sanitized_inputs['power_law_opacity_coefficient'],
                gray_opacity=sanitized_inputs['gray_opacity'],
                cloud_photosphere_median_optical_depth=sanitized_inputs['cloud_photosphere_median_optical_depth'],
                cloud_fraction=1.0,
                complete_coverage_clouds=sanitized_inputs['complete_coverage_clouds'],
                return_cloud_contribution=False,
                additional_absorption_opacities_function=sanitized_inputs['additional_absorption_opacities_function'],
                additional_scattering_opacities_function=sanitized_inputs['additional_scattering_opacities_function'],
                compute_clear_opacities=False,
                target_particle_radii=sanitized_inputs['target_particle_radii'],
                target_pressure=sanitized_inputs['target_pressure']
            )
        )
        self.__scattering_in_transmission = cloud_scattering_in_transmission

        if auto_anisotropic_cloud_scattering:
            self._anisotropic_cloud_scattering = 'auto'

        opacities_rosseland = (
            rosseland.compute_rosseland_opacities(
                frequency_bins_edges=self._frequency_bins_edges,
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities['weights_gauss'],
                opacities=opacities[:, :, :1, :],
                continuum_opacities_scattering=continuum_opacities_scattering,
                scattering_in_emission=self._scattering_in_emission
            )
        )

        opacities_planck = (
            radiative_transfer_core.compute_planck_opacities(
                opacities[:, :, :1, :],
                temperatures,
                self._lines_loaded_opacities['weights_gauss'],
                self._frequency_bins_edges,
                self._scattering_in_emission,
                continuum_opacities_scattering
            )
        )

        return opacities_rosseland, opacities_planck, cloud_particles_mean_radii

    def calculate_transit_radii(
        self,
        temperatures: ArrayLike,
        mass_fractions: dict[str, ArrayLike],
        mean_molar_masses: ArrayLike,
        reference_gravity: float,
        reference_pressure: float,
        planet_radius: float,
        variable_gravity: bool = True,
        opaque_cloud_top_pressure: float = None,
        cloud_particles_mean_radii: dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution_std: float = None,
        cloud_particles_radius_distribution: str = 'lognormal',
        cloud_hansen_a: float = None,
        cloud_hansen_b: float = None,
        cloud_particle_number_density_grid: dict[str, ArrayLike] = None,
        clouds_particles_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: float = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_fraction: float = 1.0,
        complete_coverage_clouds: list[str] = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_cloud_contribution: bool = False,
        return_radius_hydrostatic_equilibrium: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_particles_radii_bins: bool = False,
        return_particles_radii: bool = False,
        return_particles_densities: bool = False,
        target_particle_radii: ArrayLike = None,
        target_pressure: ArrayLike = None,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
        """Method to calculate the atmosphere's transmission radius
        (for the transmission spectrum).

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                mean_molar_masses:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs at reference radius and pressure.
                reference_pressure (float):
                    Reference pressure P0 in bar where R(P=P0) = R_pl,
                    where R_pl is the reference radius (parameter of this
                    method), and g(P=P0) = gravity, where gravity is the
                    reference gravity (parameter of this method)
                planet_radius (float):
                    Reference radius R_pl, in cm.
                variable_gravity (Optional[bool]):
                    Standard is ``True``. If ``False`` the gravity will be
                    constant as a function of pressure, during the transmission
                    radius calculation.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                cloud_particles_mean_radii (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std (Optional[float]):
                    width of the log-normal cloud particle size distribution
                cloud_particles_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If cloud_hansen_a is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/cloud_hansen_a^2)
                cloud_particle_number_density_grid (Optional[dict]):
                    EXPERIMENTAL
                clouds_particles_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    cloud settling parameter
                eddy_diffusion_coefficients (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). By default, assume a full cloud cover.
                complete_coverage_clouds:
                    List of clouds that have full coverage, regardless of the value of cloud_fraction.
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                frequencies_to_wavelengths (Optional[bool]):
                    If True, convert the frequencies (Hz) output to wavelengths (cm).
                return_contribution (Optional[bool]):
                    If True the transmission and emission contribution function is calculated.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_cloud_contribution (Optional[bool]):
                    If True, calculate and return the cloud contribution.
                return_radius_hydrostatic_equilibrium (Optional[bool]):
                    If True, return the radius at hydrostatic equilibrium of the planet.
                return_opacities (Optional[bool]):
                    If True, return the absorption opacities and scattering opacities.
                return_abundances (Optional[bool]):
                    If True, return the mass fractions.
                return_particles_radii_bins (Optional[bool]):
                    If True, return the particles radii bins.
                return_particles_radii (Optional[bool]):
                    If True, return the particles radii.
                return_particles_densities (Optional[bool]):
                    If True, return the particles densities.
                target_particle_radii (Optional[ArrayLike]):
                    Optional particle radii grid used when evaluating cloud
                    opacity products on a custom particle-size sampling.
                target_pressure (Optional[ArrayLike]):
                    Optional pressure grid associated with
                    ``target_particle_radii``.
        """
        (anisotropic_cloud_scattering, complete_coverage_clouds, flux_clear,
         sum_opacities) = (
            self._init_spectral_function(
                is_emission=False,
                reference_gravity=reference_gravity,
                complete_coverage_clouds=complete_coverage_clouds,
                return_clear_spectrum=return_clear_spectrum,
                mass_fractions=mass_fractions,
                cloud_fraction=cloud_fraction,
                opaque_cloud_top_pressure=opaque_cloud_top_pressure
            )
        )
        copy_opacities = bool(return_contribution)

        sanitized_inputs = self._sanitize_flux_opacity_inputs(
            complete_coverage_clouds=complete_coverage_clouds,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=None,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )
        partial_coverage_cloud_species = self._get_partial_coverage_cloud_species(
            sanitized_inputs['complete_coverage_clouds']
        )
        should_compute_clear_opacities = (
            len(partial_coverage_cloud_species) > 0
            and (return_clear_spectrum or cloud_fraction < 1)
        )

        (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
         cloud_opacities, cloud_particles_mean_radii, cloud_scattering_in_transmission,
         clear_opacities, clear_continuum_opacities_scattering,
         clear_cloud_anisotropic_scattering_opacities, clear_cloud_absorption_opacities) = (
            self._calculate_opacities(
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                opaque_cloud_top_pressure=sanitized_inputs['opaque_cloud_top_pressure'],
                cloud_particles_mean_radii=sanitized_inputs['cloud_particles_mean_radii'],
                cloud_particle_radius_distribution_std=sanitized_inputs['cloud_particle_radius_distribution_std'],
                cloud_particles_radius_distribution=cloud_particles_radius_distribution,
                cloud_hansen_a=sanitized_inputs['cloud_hansen_a'],
                cloud_hansen_b=sanitized_inputs['cloud_hansen_b'],
                cloud_particle_number_density_grid=sanitized_inputs['cloud_particle_number_density_grid'],
                clouds_particles_porosity_factor=sanitized_inputs['clouds_particles_porosity_factor'],
                cloud_f_sed=sanitized_inputs['cloud_f_sed'],
                eddy_diffusion_coefficients=sanitized_inputs['eddy_diffusion_coefficients'],
                haze_factor=haze_factor,
                power_law_opacity_350nm=sanitized_inputs['power_law_opacity_350nm'],
                power_law_opacity_coefficient=sanitized_inputs['power_law_opacity_coefficient'],
                gray_opacity=sanitized_inputs['gray_opacity'],
                cloud_photosphere_median_optical_depth=sanitized_inputs['cloud_photosphere_median_optical_depth'],
                cloud_fraction=cloud_fraction,
                complete_coverage_clouds=sanitized_inputs['complete_coverage_clouds'],
                return_cloud_contribution=False,
                additional_absorption_opacities_function=sanitized_inputs['additional_absorption_opacities_function'],
                additional_scattering_opacities_function=sanitized_inputs['additional_scattering_opacities_function'],
                compute_clear_opacities=should_compute_clear_opacities,
                sum_opacities=sum_opacities,
                anisotropic_cloud_scattering=anisotropic_cloud_scattering,
                target_particle_radii=sanitized_inputs['target_particle_radii'],
                target_pressure=sanitized_inputs['target_pressure']
            ))
        self.__scattering_in_transmission = cloud_scattering_in_transmission

        # Start filling additional outputs here to get rid of unnecessary variables ASAP
        additional_outputs = {}

        #if return_opacities:
        #    additional_outputs['opacities'] = copy.deepcopy(opacities)
        #    additional_outputs['continuum_opacities_scattering'] = copy.deepcopy(continuum_opacities_scattering)

        #if cloud_particles_mean_radii is not None:
        #    additional_outputs['cloud_particles_mean_radii'] = cloud_particles_mean_radii

        if return_particles_radii_bins:
            additional_outputs['particles_radii_bins'] = self._clouds_loaded_opacities['particles_radii_bins']

        if return_particles_radii:
            additional_outputs['particles_radii'] = self._clouds_loaded_opacities['particles_radii']

        if return_particles_densities:
            additional_outputs['particles_densities'] = self._clouds_loaded_opacities['particles_densities']

        del cloud_particles_mean_radii

        #if auto_anisotropic_cloud_scattering:
        #    self._anisotropic_cloud_scattering = 'auto'

        # Calculate the transit radii
        transit_radii, radius_hydrostatic_equilibrium, transmission_contribution = self._calculate_transit_radii(
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            copy_opacities=copy_opacities,
            return_contributions=return_contribution
        )

        if len(partial_coverage_cloud_species) > 0 and (return_clear_spectrum or cloud_fraction < 1):
            transit_radii_clear, _, _ = self._calculate_transit_radii(
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
                opacities=clear_opacities,
                continuum_opacities_scattering=clear_continuum_opacities_scattering,
                copy_opacities=False,
                return_contributions=False
            )

            transit_radii = (
                cloud_fraction * transit_radii
                + (1 - cloud_fraction) * transit_radii_clear
            )
        elif return_clear_spectrum:
            transit_radii_clear = transit_radii

        del opacities

        # Fill last additional parameters
        #if return_contribution:
        #    additional_outputs['transmission_contribution'] = transmission_contribution

        """if return_clear_spectrum:
            if frequencies_to_wavelengths:
                additional_outputs['clear_spectrum'] = flux_hz2flux_cm(
                    flux_hz=transit_radii_clear,
                    frequency=self._frequencies
                )
            else:
                additional_outputs['clear_spectrum'] = transit_radii_clear"""

        #if return_radius_hydrostatic_equilibrium:
        #    additional_outputs['radius_hydrostatic_equilibrium'] = radius_hydrostatic_equilibrium

        #if return_abundances:
        #    additional_outputs['mass_fractions'] = mass_fractions

        #if frequencies_to_wavelengths:
        return (
            frequency2wavelength(self._frequencies),
            transit_radii,
            additional_outputs
        )
        #else:
        #    return self._frequencies, transit_radii, additional_outputs

    def calculate_transit_radii_autodiff(
            self,
            temperatures: ArrayLike,
            mass_fractions: dict[str, ArrayLike],
            mean_molar_masses: ArrayLike,
            reference_gravity: float,
            reference_pressure: float,
            planet_radius: float,
            *,
            variable_gravity: bool = True,
            cloud_parameters: dict[str, Any] | None = None,
            additional_absorption_opacities_function: Callable = None,
            additional_scattering_opacities_function: Callable = None,
            frequencies_to_wavelengths: bool = True,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
        r"""Autodiff-oriented transmission-spectrum API with grouped cloud controls.

        The method mirrors calculate_transit_radii while collapsing cloud controls into a
        single keyword argument and keeping the differentiated call surface small.

        Example:
            >>> cloud_parameters = {
            ...     'cloud_f_sed': cloud_f_sed,
            ...     'eddy_diffusion_coefficients': eddy_diffusion_coefficients,
            ...     'cloud_particle_radius_distribution_std': 1.05,
            ... }
            >>> _, transit_radii, _ = atmosphere.calculate_transit_radii_autodiff(
            ...     temperatures=temperatures,
            ...     mass_fractions=mass_fractions,
            ...     mean_molar_masses=mean_molar_masses,
            ...     reference_gravity=reference_gravity,
            ...     reference_pressure=reference_pressure,
            ...     planet_radius=planet_radius,
            ...     cloud_parameters=cloud_parameters,
            ...     frequencies_to_wavelengths=False,
            ... )
            >>> temperature_gradient = jax.grad(
            ...     lambda t: jnp.sum(
            ...         atmosphere.calculate_transit_radii_autodiff(
            ...             temperatures=t,
            ...             mass_fractions=mass_fractions,
            ...             mean_molar_masses=mean_molar_masses,
            ...             reference_gravity=reference_gravity,
            ...             reference_pressure=reference_pressure,
            ...             planet_radius=planet_radius,
            ...             cloud_parameters=cloud_parameters,
            ...             frequencies_to_wavelengths=False,
            ...         )[1]
            ...     )
            ... )(temperatures)
        """
        allowed_cloud_parameters = {
            'opaque_cloud_top_pressure',
            'cloud_particles_mean_radii',
            'cloud_particle_radius_distribution_std',
            'cloud_particles_radius_distribution',
            'cloud_hansen_a',
            'cloud_hansen_b',
            'cloud_particle_number_density_grid',
            'clouds_particles_porosity_factor',
            'cloud_f_sed',
            'eddy_diffusion_coefficients',
            'haze_factor',
            'power_law_opacity_350nm',
            'power_law_opacity_coefficient',
            'gray_opacity',
            'cloud_fraction',
            'complete_coverage_clouds',
        }
        cloud_parameters = {} if cloud_parameters is None else dict(cloud_parameters)
        unsupported_cloud_parameters = sorted(set(cloud_parameters) - allowed_cloud_parameters)

        if unsupported_cloud_parameters:
            raise ValueError(
                "unsupported calculate_transit_radii_autodiff cloud parameters: "
                f"{', '.join(unsupported_cloud_parameters)}"
            )

        return self.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            reference_pressure=reference_pressure,
            planet_radius=planet_radius,
            variable_gravity=variable_gravity,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            **cloud_parameters,
        )


    def get_wavelengths(self):
        """Convert the frequency grid into wavelengths in centimeter."""
        # Not a property because the child SpectralModel already have a 'wavelengths' attribute that behaves differently
        return frequency2wavelength(self._frequencies)

    def load_all_opacities(self):
        print("Loading Radtrans opacities...")

        # Load line opacities
        self.load_line_opacities(self._path_input_data)

        # Read continuum opacities
        # CIA
        if len(self._gas_continuum_contributors) > 0:
            self.load_cia_opacities(self._path_input_data)

        # Clouds
        if len(self._cloud_species) > 0:
            # Inherited from ReadOpacities in _read_opacities.py
            self.load_cloud_opacities(self._path_input_data)

        print("Successfully loaded all opacities")

    def load_cia_opacities(self, path_input_data):
        for collision in self._cias_loaded_opacities:
            if collision in hminus.get_non_cia_gas_continuum_contributions():
                continue

            hdf5_file = cia.get_cia_aliases(collision)
            hdf5_file = CIAOpacity.find(
                path_input_data=path_input_data,
                species=hdf5_file
            )

            print(f" Loading CIA opacities for {collision} from file '{hdf5_file}'...", end='')

            (
                self._cias_loaded_opacities[collision]['molecules'],
                self._cias_loaded_opacities[collision]['weight'],
                self._cias_loaded_opacities[collision]['lambda'],
                self._cias_loaded_opacities[collision]['temperature'],
                self._cias_loaded_opacities[collision]['alpha']
            ) = load.load_cia_opacity(hdf5_file)

            print(" Done.")

        print(" Successfully loaded all CIA opacities")

    def load_cloud_opacities(self, path_input_data):
        """ Load the cloud opacities.

        Args:
            path_input_data: path to petitRADTRANS input_data directory
        """
        hdf5_files = []
        internal_structures = []
        scattering_methods = []

        for i in range(len(self._cloud_species)):
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[i])

            hdf5_file = CloudOpacity.find(
                path_input_data=path_input_data,
                species=hdf5_file
            )

            internal_structures.append(hdf5_file.rsplit(')_', 1)[1].split('__', 1)[0])
            scattering_methods.append(hdf5_file.rsplit('__', 1)[1].split('.', 1)[0])

            hdf5_files.append(hdf5_file)

        clouds_particles_densities = jnp.zeros(len(hdf5_files), dtype=JAX_DTYPE)
        clouds_absorption_opacities = None
        clouds_scattering_opacities = None
        clouds_asymmetry_parameters = None
        cloud_wavelengths = None
        clouds_particles_radii_bins = None
        clouds_particles_radii = None

        for i, hdf5_file in enumerate(hdf5_files):
            print(f" Loading opacities of cloud species '{self._cloud_species[i]}' from file '{hdf5_file}' "
                  f"({internal_structures[i]}, using {scattering_methods[i]} scattering)...", end='')

            (
                _cloud_wavelengths,
                _cloud_particles_radii_bins,
                _cloud_particles_radii,
                _cloud_particles_densities,
                _cloud_absorption_opacities,
                _cloud_scattering_opacities,
                _cloud_asymmetry_parameters
            ) = load.load_cloud_opacity(hdf5_file)

            if i == 0:
                # Initialize cloud arrays
                cloud_wavelengths = _cloud_wavelengths
                clouds_particles_radii_bins = _cloud_particles_radii_bins
                clouds_particles_radii = _cloud_particles_radii

                clouds_absorption_opacities = jnp.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files)), dtype=JAX_DTYPE
                )
                clouds_scattering_opacities = jnp.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files)), dtype=JAX_DTYPE
                )
                clouds_asymmetry_parameters = jnp.zeros(
                    (clouds_particles_radii.size, cloud_wavelengths.size, len(hdf5_files)), dtype=JAX_DTYPE
                )

            clouds_particles_densities = clouds_particles_densities.at[i].set(_cloud_particles_densities)
            clouds_absorption_opacities = clouds_absorption_opacities.at[:, :, i].set(_cloud_absorption_opacities)
            clouds_scattering_opacities = clouds_scattering_opacities.at[:, :, i].set(_cloud_scattering_opacities)
            clouds_asymmetry_parameters = clouds_asymmetry_parameters.at[:, :, i].set(_cloud_asymmetry_parameters)

            print(" Done.")

        # Flip wavelengths/wavenumbers axis to match wavelengths ordering
        clouds_absorption_opacities = clouds_absorption_opacities[:, ::-1, :]
        clouds_scattering_opacities = clouds_scattering_opacities[:, ::-1, :]
        clouds_asymmetry_parameters = clouds_asymmetry_parameters[:, ::-1, :]

        clouds_absorption_opacities = clouds_absorption_opacities.at[clouds_absorption_opacities < 0.].set(0.)
        clouds_scattering_opacities = clouds_scattering_opacities.at[clouds_scattering_opacities < 0.].set(0.)

        self._clouds_loaded_opacities['particles_densities'] = (
            jnp.array(clouds_particles_densities, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['absorption_opacities'] = (
            jnp.array(clouds_absorption_opacities, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['scattering_opacities'] = (
            jnp.array(clouds_scattering_opacities, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['particles_asymmetry_parameters'] = (
            jnp.array(clouds_asymmetry_parameters, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['wavelengths'] = (
            jnp.array(cloud_wavelengths, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['particles_radii_bins'] = (
            jnp.array(clouds_particles_radii_bins, dtype=JAX_DTYPE)
        )
        self._clouds_loaded_opacities['particles_radii'] = (
            jnp.array(clouds_particles_radii, dtype=JAX_DTYPE)
        )

        print(" Successfully loaded all clouds opacities")

    @classmethod
    def load_hdf5_ktables(
        cls, file_path_hdf5, frequencies, g_size, temperature_pressure_grid_size, return_radtrans_opacities=True
    ):
        """Load k-coefficient tables in HDF5 format, based on the ExoMol setup."""
        with h5py.File(file_path_hdf5, 'r') as f:
            k_table = np.array(f['kcoeff'])

            if not return_radtrans_opacities:
                return (
                    f['DOI'][()][0].decode('utf-8'),
                    f['bin_centers'][:],
                    f['bin_edges'][:],
                    k_table,
                    f['mol_mass'][()][0],
                    f['mol_name'][()][0].decode('utf-8'),
                    f['p'][:],
                    f['t'][:],
                    f['samples'][:],
                    f['weights'][:]
                )

            n_wavelengths = len(f['bin_centers'][:])
            _frequencies = cst.c * f['bin_centers'][:][::-1]
            n_temperatures = len(f['t'][:])
            n_pressures = len(f['p'][:])

            # Reshape k-table to petitRADTRANS format
            k_table = jnp.swapaxes(k_table, 0, 1)
            k_table = k_table.reshape((n_pressures * n_temperatures, n_wavelengths, 16))
            k_table = jnp.swapaxes(k_table, 0, 2)
            k_table = k_table[:, ::-1, :]

            indices_opacities, indices_file = load.get_frequency_grids_intersection_indices(
                frequencies=frequencies,
                file_frequencies=_frequencies,
                tolerate_grid_misalignment=True
            )

            if np.nonzero(indices_file)[0].size != np.nonzero(indices_opacities)[0].size:
                grid_misalignment: bool = False

                if np.nonzero(indices_file)[0].size == frequencies.size - 1:
                    indices_file = _handle_grid_misalignment(
                        indices=indices_file,
                        frequencies_reference=frequencies,
                        frequencies_test=_frequencies
                    )
                    grid_misalignment = True
                elif np.nonzero(indices_opacities)[0].size == np.nonzero(indices_file)[0].size - 1:
                    indices_opacities = _handle_grid_misalignment(
                        indices=indices_opacities,
                        frequencies_reference=_frequencies,
                        frequencies_test=frequencies
                    )
                    grid_misalignment = True

                if grid_misalignment:
                    max_relative_difference = jnp.max(
                        jnp.abs(frequencies[indices_opacities] / _frequencies[indices_file] - 1)
                    )
                    spectral_info = Opacity.split_species_spectral_info(os.path.basename(file_path_hdf5))[1]
                    resolving_power = Opacity.split_spectral_info(spectral_info)[0]
                    resolving_power = Opacity.get_resolving_power_from_string(resolving_power)

                    if max_relative_difference > cls._frequency_grid_misalignment_tolerance / resolving_power:
                        debug.print(
                            f"maximum frequency grid misalignment from file ({max_relative_difference}) "
                            f"is larger than tolerated "
                            f"({cls._frequency_grid_misalignment_tolerance / resolving_power}); "
                            f"expect inaccuracies in spectral features location\n"
                            f"To remove this warning, ensure that your correlated-k opacity files "
                            f"have similar wavenumber grids"
                        )

            # Fill opacity array
            opacities = jnp.zeros((g_size, frequencies.size, 1, temperature_pressure_grid_size), dtype=JAX_DTYPE)
            opacities = opacities.at[:, indices_opacities, 0, :].set(k_table[:, indices_file, :])
            opacities = jnp.where(opacities < 0, 0., opacities)

            # Divide by mass to convert cross-sections to opacities
            mol_mass_inv = 1 / (f['mol_mass'][()] * cst.amu)

        line_opacities_grid = opacities * mol_mass_inv

        return line_opacities_grid


    def load_line_opacities(self, path_input_data):
        """Load the line opacities for spectral calculation.
        The default pressure-temperature grid is a log-uniform (10, 13) grid.

        Args:
            path_input_data: path to petitRADTRANS input_data directory
        """
        # TODO currently all the pressure-temperature grid is loaded, it could be more memory efficient to provide a T an p range at init and only load the relevant parts of the grid # noqa: E501
        if self._line_opacity_mode == 'c-k':
            category = 'correlated_k_opacities'
        elif self._line_opacity_mode == 'lbl':
            category = 'line_by_line_opacities'
        else:
            raise ValueError(f"invalid line opacity mode: '{self._line_opacity_mode}' (must be 'c-k'|'lbl')")

        # Read opacities grid
        if len(self._line_species) > 0:
            for i, species in enumerate(self._line_species):
                hdf5_file = load.get_line_opacity_file(
                    path_input_data=path_input_data,
                    species=species,
                    category=category
                )

                # Load g grid for correlated-k
                if self._line_opacity_mode == 'c-k' and i == 0:
                    with h5py.File(hdf5_file, 'r') as f:
                        self._lines_loaded_opacities['g_gauss'] = f['samples'][:]
                        self._lines_loaded_opacities['weights_gauss'] = f['weights'][:]

                    # Convert into F-ordered array for more efficient processing in the Fortran modules
                    self._lines_loaded_opacities['g_gauss'] = jnp.array(
                        self._lines_loaded_opacities['g_gauss'], dtype=JAX_DTYPE)
                    self._lines_loaded_opacities['weights_gauss'] = jnp.array(
                        self._lines_loaded_opacities['weights_gauss'], dtype=JAX_DTYPE)

                # Load temperature-pressure grid
                (
                    self._lines_loaded_opacities['temperature_pressure_grid'][species],
                    self._lines_loaded_opacities['temperature_grid_size'][species],
                    self._lines_loaded_opacities['pressure_grid_size'][species]
                ) = load.load_line_opacities_pressure_temperature_grid(
                    hdf5_file=hdf5_file
                )
                # Load the opacities
                print(f" Loading line opacities of species '{species}' from file '{hdf5_file}'...", end='')

                if self._line_opacity_mode == 'c-k':
                    self._lines_loaded_opacities['opacity_grid'][species] = self.load_hdf5_ktables(
                        file_path_hdf5=hdf5_file,
                        frequencies=self._frequencies,
                        g_size=self._lines_loaded_opacities['g_gauss'].size,
                        temperature_pressure_grid_size=self._lines_loaded_opacities['temperature_pressure_grid'][
                            species].shape[0]
                    )
                elif self._line_opacity_mode == 'lbl':
                    self._lines_loaded_opacities['opacity_grid'][species] = load.load_hdf5_line_opacity_table(
                        file_path_hdf5=hdf5_file,
                        frequencies=self._frequencies,
                        line_by_line_opacity_sampling=self._line_by_line_opacity_sampling
                    )

                print(" Done.")

                # Convert into F-ordered array for more efficient processing in the Fortran modules
                self._lines_loaded_opacities['opacity_grid'][species] = \
                    jnp.array(self._lines_loaded_opacities['opacity_grid'][species][:, :, 0, :], dtype=JAX_DTYPE)

            self._stack_loaded_line_opacities()

            print(" Successfully loaded all line opacities")


tree_util.register_pytree_node(
    Radtrans,
    Radtrans._tree_flatten,
    Radtrans._tree_unflatten
)
