import logging
import os
import warnings
from typing import Any, Callable

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike

from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.retrieval.data import Data, PhotometryData, TimeSeriesData
from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.utils import uniform_prior
from petitRADTRANS.synthetic_photometry import SyntheticPhotometry
# Import prior distributions
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

# MPI Multiprocessing
_UNSET = object()  # Sentinel for deprecated argument detection
rank = 0
comm = None

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None


class RetrievalConfig:
    """Contain all the data and model level information necessary to run a petitRADTRANS retrieval.

    The name of the class will be used to name outputs.
    This class is passed to the Retrieval, which runs the actual pymultinest retrieval and produces the outputs.

    The general usage of this class is to define it, add the parameters and their priors, add the opacity sources,
    the data together with a model for each dataset, and then configure a few plotting arguments.

    Args:
        retrieval_name : str
            Name of this retrieval. Make it informative so that you can keep track of the outputs!
        run_mode : str
            Can be either 'retrieval', which runs the retrieval normally using pymultinest,
            or 'evaluate', which produces plots from the best fit parameters stored in the
            output post_equal_weights file.
        adaptive_mesh_refinement : bool
            Use an adaptive high resolution pressure grid around the location of cloud condensation.
            This will increase the size of the pressure grid by a constant factor that can be adjusted
            in the setup_pres function.
        scattering_in_emission : bool
            If using emission spectra, turn scattering on or off.
        pressures : numpy.array
            A log-spaced array of pressures over which to retrieve. 100 points is standard, between
            10^-6 and 10^3.
    """

    def __init__(self,
                 retrieval_name: str = "retrieval_name",
                 run_mode: str = "retrieval",
                 sampler_type: str = 'pymultinest',
                 adaptive_mesh_refinement: str = False,
                 scattering_in_emission: bool = False,
                 pressures: ArrayLike | None = None,
                 # --- deprecated arguments ---
                 amr=_UNSET):

        if amr is not _UNSET:
            warnings.warn(
                "The 'amr' argument is deprecated; use 'adaptive_mesh_refinement' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            adaptive_mesh_refinement = amr

        self.retrieval_name = retrieval_name
        self.run_mode = run_mode
        self.sampler_type = sampler_type

        if self.run_mode != 'retrieval' and self.run_mode != 'evaluate':
            raise ValueError(f"run_mode must be 'retrieval' or 'evaluate', but was '{self.run_mode}'")

        self.adaptive_mesh_refinement = adaptive_mesh_refinement
        self.scaling = 1
        self.width = 1

        if pressures is not None:
            self.pressures = pressures
        else:
            self.pressures = jnp.logspace(-6, 3, 100)

        self.scattering_in_emission = scattering_in_emission
        self.parameters = {}  #: Dictionary of the parameters passed to the model generating function
        self.data = {}  #: Dictionary of the datasets used in the retrieval.
        self.instruments = ()
        self.line_species = ()
        self.cloud_species = ()
        self.rayleigh_species = ()
        self.continuum_opacities = ()

        self.add_parameter("pressure_scaling", False, value=1)
        self.add_parameter("pressure_width", False, value=1)
        self.add_parameter("pressure_simple", False, value=self.pressures.shape[0])

    def _setup_pres(self, scaling: int = 10, width: int = 3) -> ArrayLike:
        """
        This converts the standard pressure grid into the correct length
        for the AMR pressure grid. The scaling adjusts the resolution of the
        high resolution grid, while the width determines the size of the high
        pressure region. This function is automatically called in
        Retrieval.setupData().

        Args:
            scaling : int
                A multiplicative factor that determines the size of the full high resolution pressure grid,
                which will have length self.p_global.shape[0] * scaling.
            width : int
                The number of cells in the low pressure grid to replace with the high resolution grid.
        """

        print("Setting up AMR pressure grid.")
        self.scaling = scaling
        self.width = width
        nclouds = len(self.cloud_species)
        if nclouds == 0:
            print("WARNING: there are no clouds in the retrieval, please add cloud species before setting up AMR")
        new_len = self.pressures.shape[0] + nclouds * width * (scaling - 1)
        self.adaptive_mesh_refinement_pressure = jnp.logspace(jnp.log10(self.pressures[0]), jnp.log10(self.pressures[-1]), new_len)
        self.add_parameter("pressure_scaling", False, value=scaling)
        self.add_parameter("pressure_width", False, value=width)
        self.add_parameter("pressure_simple", False, value=self.pressures.shape[0])

        return self.adaptive_mesh_refinement_pressure

    def add_parameter(
            self,
            name: str,
            is_free_parameter: bool,
            value: float | None = None,
            distribution: Any | None = None,
            plot_in_corner: bool = False,
            corner_ranges: tuple[float, float] | None = None,
            corner_transform: Callable | None = None,
            transform_prior_cube_coordinate: Callable | None = None,
            corner_label: str | None = None,
            # --- deprecated arguments ---
            free=_UNSET) -> None:
        """
        This function adds a Parameter (see parameter.py) to the dictionary of parameters. A Parameter
        has a name and a boolean parameter to set whether it is a free or fixed parameter during the retrieval.
        In addition, a value can be set, or a prior function can be given that transforms a random variable in
        [0,1] to the physical dimensions of the Parameter.

        Args:
            name : str
                The name of the parameter. Must match the name used in the model function for the retrieval.
            free : bool
                True if the parameter is a free parameter in the retrieval, false if it is fixed.
            value : float
                The value of the parameter in the units used by the model function.
            transform_prior_cube_coordinate : method
                A function that transforms the unit interval to the physical units of the parameter.
                Typically given as a lambda function.
        """

        if free is not _UNSET:
            warnings.warn(
                "The 'free' argument is deprecated; use 'is_free_parameter' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            is_free_parameter = free

        self.parameters[name] = Parameter(
            name=name,
            is_free_parameter=is_free_parameter,
            value=value,
            distribution=distribution,
            plot_in_corner=plot_in_corner,
            corner_ranges=corner_ranges,
            corner_transform=corner_transform,
            transform_prior_cube_coordinate=transform_prior_cube_coordinate,
            corner_label=corner_label
            )

    def _add_uniform_free_parameter(self, name: str, lower: float, upper: float, corner_label: str = None) -> None:
        if name in self.parameters:
            return

        if self.sampler_type == 'pymultinest':
            self.parameters[name] = Parameter(
                name=name,
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x, lo=lower, hi=upper: uniform_prior(x, lo, hi),
                corner_label=corner_label,
            )
        else:
            self.parameters[name] = Parameter(
                name=name,
                is_free_parameter=True,
                distribution=tfpd.Uniform(low=lower, high=upper),
                corner_label=corner_label,
            )

    @staticmethod
    def _default_covariance_parameter_priors(data_object: Data) -> dict[str, tuple[float, float]]:
        amplitude_scale = None
        if data_object.uncertainties is not None:
            uncertainties = np.asarray(data_object.uncertainties, dtype=float).reshape(-1)
            uncertainties = uncertainties[np.isfinite(uncertainties) & (uncertainties > 0)]
            if uncertainties.size > 0:
                amplitude_scale = float(np.median(uncertainties))

        if amplitude_scale is None or amplitude_scale <= 0:
            if data_object.spectrum is not None:
                spectrum = np.asarray(data_object.spectrum, dtype=float).reshape(-1)
                spectrum = np.abs(spectrum[np.isfinite(spectrum)])
                if spectrum.size > 0:
                    amplitude_scale = float(np.median(spectrum)) * 1e-2

        if amplitude_scale is None or amplitude_scale <= 0:
            amplitude_scale = 1.0

        wavelengths = np.asarray(data_object.wavelengths, dtype=float).reshape(-1)
        wavelength_span = float(np.max(wavelengths) - np.min(wavelengths)) if wavelengths.size > 1 else 1.0
        wavelength_diffs = np.diff(wavelengths)
        wavelength_diffs = wavelength_diffs[np.isfinite(wavelength_diffs) & (wavelength_diffs > 0)]
        min_length = float(np.median(wavelength_diffs)) if wavelength_diffs.size > 0 else wavelength_span
        min_length = max(min_length, 1e-12)
        max_length = max(wavelength_span, min_length)

        return {
            'global_amp': (0.0, amplitude_scale * 10.0),
            'global_length': (min_length, max_length),
            'local_amp': (0.0, amplitude_scale * 10.0),
            'local_length': (min_length, max_length),
            'local_center': (float(np.min(wavelengths)), float(np.max(wavelengths))),
        }

    def _add_covariance_hyperparameters(
        self,
        data_object: Data,
        covariance_mode: str,
        n_local_covariance_kernels: int,
        covariance_parameter_priors: dict | None = None,
    ) -> None:
        prior_ranges = self._default_covariance_parameter_priors(data_object)
        if covariance_parameter_priors is not None:
            prior_ranges.update(covariance_parameter_priors)

        if covariance_mode in {'global', 'global_local'}:
            self._add_uniform_free_parameter(
                name=f'{data_object.name}_gp_global_amp',
                lower=prior_ranges['global_amp'][0],
                upper=prior_ranges['global_amp'][1],
                corner_label=f'{data_object.name} gp global amp',
            )
            self._add_uniform_free_parameter(
                name=f'{data_object.name}_gp_global_length',
                lower=prior_ranges['global_length'][0],
                upper=prior_ranges['global_length'][1],
                corner_label=f'{data_object.name} gp global length',
            )

        if covariance_mode in {'local', 'global_local'}:
            for kernel_index in range(1, n_local_covariance_kernels + 1):
                self._add_uniform_free_parameter(
                    name=f'{data_object.name}_gp_local_{kernel_index}_amp',
                    lower=prior_ranges['local_amp'][0],
                    upper=prior_ranges['local_amp'][1],
                    corner_label=f'{data_object.name} gp local {kernel_index} amp',
                )
                self._add_uniform_free_parameter(
                    name=f'{data_object.name}_gp_local_{kernel_index}_length',
                    lower=prior_ranges['local_length'][0],
                    upper=prior_ranges['local_length'][1],
                    corner_label=f'{data_object.name} gp local {kernel_index} length',
                )
                self._add_uniform_free_parameter(
                    name=f'{data_object.name}_gp_local_{kernel_index}_center',
                    lower=prior_ranges['local_center'][0],
                    upper=prior_ranges['local_center'][1],
                    corner_label=f'{data_object.name} gp local {kernel_index} center',
                )

    @staticmethod
    def list_available_line_species() -> set[str]:
        """
        List the currently installed opacity tables that are available for species that contribute to the line opacity.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/lines/correlated_k/")]
        files = set([f.split('.R')[0] for f in files])
        print("\ncorrelated-k opacities")

        for f in files:
            print(f)

        lbl = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/lines/line_by_line/")]
        lbl = set(lbl)
        print("\nline-by-line opacities")

        for f in lbl:
            print(f)

        return files.union(lbl)

    @staticmethod
    def list_available_cloud_species() -> set[str]:
        """
        List the currently installed opacity tables that are available for cloud species.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/continuum/clouds/")]
        files = set(files)

        for f in files:
            print(f)

        return files

    @staticmethod
    def list_available_cia_species() -> set[str]:
        """
        List the currently installed opacity tables that are available for CIA species.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/continuum/cia/")]
        files = set(files)

        for f in files:
            print(f)

        return files

    def set_line_species(
            self,
            line_species: list[str],
            use_equilibrium_chemistry: bool = False,
            free_mass_fraction_limits: tuple[float, float] = (-6.0, -0.5),
            plot_in_corner: bool = True,
            # --- deprecated arguments ---
            linelist=_UNSET,
            eq=_UNSET,
            abund_lim=_UNSET) -> None:
        """
        Set RadTrans.line_species

        This function adds a list of species to the pRT object that will define the line
        opacities of the model. The values in the list are strings, with the names matching
        the pRT opacity names, which vary between the c-k line opacities and the line-by-line opacities.

        NOTE: As of pRT version 2.4.9, the behaviour of this function has changed. In previous versions the
        abundance limits were set from free_mass_fraction_limits[0] to (free_mass_fraction_limits[0] + free_mass_fraction_limits[1]). This has been changed
        so that the limits of the prior range are from free_mass_fraction_limits[0] to free_mass_fraction_limits[1] (ie the actual boundaries).

        Args:
            line_species : List(str)
                The list of species to include in the retrieval
            use_equilibrium_chemistry : bool
                If false, the retrieval should use free chemistry, and Parameters for the abundance of each
                species in the line_species will be added to the retrieval. Otherwise, equilibrium chemistry will
                be used. If you need fine control species, use the add_line_species and set up each species
                individually.
            free_mass_fraction_limits : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied for
                each species in line_species. The range of the prior goes from free_mass_fraction_limits[0] to free_mass_fraction_limits[1].
                The abundance limits must be given in log10 units of the mass fraction.
        """
        if linelist is not _UNSET:
            warnings.warn(
                "The 'linelist' argument is deprecated; use 'line_species' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            line_species = linelist
        if eq is not _UNSET:
            warnings.warn(
                "The 'eq' argument is deprecated; use 'use_equilibrium_chemistry' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_equilibrium_chemistry = eq
        if abund_lim is not _UNSET:
            warnings.warn(
                "The 'abund_lim' argument is deprecated; use 'free_mass_fraction_limits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            free_mass_fraction_limits = abund_lim

        if free_mass_fraction_limits[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {free_mass_fraction_limits})! Please set abundance limits as (low, high)"
            )

        self.line_species = tuple(line_species)

        if not use_equilibrium_chemistry:
            for species in self.line_species:
                species_info = Opacity.split_species_all_info(species)
                species_name = species_info[0]  # remove possible previous spectral info

                if self.sampler_type == 'pymultinest':
                    self.parameters[species] = Parameter(
                        name=species,
                        is_free_parameter=True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            free_mass_fraction_limits[0],
                            free_mass_fraction_limits[1]
                            ),
                        plot_in_corner=plot_in_corner,
                        corner_label=Opacity.get_species_scientific_name(species_name),
                        corner_ranges=free_mass_fraction_limits
                    )
                else:
                    self.parameters[species] = Parameter(
                        name=species,
                        is_free_parameter=True,
                        distribution=tfpd.Uniform(low=free_mass_fraction_limits[0], high=free_mass_fraction_limits[1]),
                        plot_in_corner=plot_in_corner,
                        corner_label=Opacity.get_species_scientific_name(species_name),
                        corner_ranges=free_mass_fraction_limits
                    )

    def set_rayleigh_species(self, rayleigh_species: list[str] = _UNSET, linelist=_UNSET) -> None:
        """
        Set the list of species that contribute to the rayleigh scattering in the pRT object.

        Args:
            rayleigh_species : List(str)
                A list of species that contribute to the rayleigh opacity.
        """
        if linelist is not _UNSET:
            warnings.warn(
                "The 'linelist' argument is deprecated; use 'rayleigh_species' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            rayleigh_species = linelist
        if rayleigh_species is _UNSET:
            raise TypeError("set_rayleigh_species() missing required argument: 'rayleigh_species'")

        self.rayleigh_species = rayleigh_species

    def set_continuum_opacities(self, continuum_opacities: list[str] = _UNSET, linelist=_UNSET) -> None:
        """
        Set the list of species that contribute to the continuum opacity in the pRT object.

        Args:
            continuum_opacities : List(str)
                A list of species that contribute to the continuum opacity.
        """
        if linelist is not _UNSET:
            warnings.warn(
                "The 'linelist' argument is deprecated; use 'continuum_opacities' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            continuum_opacities = linelist
        if continuum_opacities is _UNSET:
            raise TypeError("set_continuum_opacities() missing required argument: 'continuum_opacities'")

        self.continuum_opacities = continuum_opacities

    def add_line_species(
            self,
            species: str,
            use_equilibrium_chemistry: bool = False,
            free_mass_fraction_limits: tuple[float, float] = (-6.0, -0.5),
            fixed_mass_fraction_value: float | None = None,
            plot_in_corner: bool = True,
            corner_label: str | None = None,
            corner_ranges: tuple[float, float] | None = None,
            # --- deprecated arguments ---
            eq=_UNSET,
            abund_lim=_UNSET,
            fixed_abund=_UNSET) -> None:
        """
        This function adds a single species to the pRT object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line
        opacities.

        NOTE: As of pRT version 2.4.9, the behaviour of this function has changed. In previous versions the
        abundance limits were set from free_mass_fraction_limits[0] to (free_mass_fraction_limits[0] + free_mass_fraction_limits[1]). This has been changed
        so that the limits of the prior range are from free_mass_fraction_limits[0] to free_mass_fraction_limits[1] (ie the actual boundaries).

        Args:
            species : str
                The species to include in the retrieval
            use_equilibrium_chemistry : bool
                If False, the retrieval should use free chemistry, and Parameters for the abundance of the
                species will be added to the retrieval. Otherwise, (dis)equilibrium chemistry will be used.
            free_mass_fraction_limits : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied the species given.
                The range of the prior goes from free_mass_fraction_limits[0] to free_mass_fraction_limits[1]
                The abundance limits must be given in log10 units of the mass fraction.
            fixed_mass_fraction_value : float
                The log-mass fraction abundance of the species. Currently only supports vertically constant
                abundances. If this is set, then the species will not be a free parameter in the retrieval.
        """
        if eq is not _UNSET:
            warnings.warn(
                "The 'eq' argument is deprecated; use 'use_equilibrium_chemistry' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_equilibrium_chemistry = eq
        if abund_lim is not _UNSET:
            warnings.warn(
                "The 'abund_lim' argument is deprecated; use 'free_mass_fraction_limits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            free_mass_fraction_limits = abund_lim
        if fixed_abund is not _UNSET:
            warnings.warn(
                "The 'fixed_abund' argument is deprecated; use 'fixed_mass_fraction_value' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fixed_mass_fraction_value = fixed_abund

        # parameter passed through loglike is log10 abundance
        if free_mass_fraction_limits[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {free_mass_fraction_limits})! Please set abundance limits as (low, high)"
            )

        Opacity.check_name(species)
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]

        self.line_species = self.line_species + (species_full_name,)
        if not use_equilibrium_chemistry:
            if corner_label is None:
                corner_label = Opacity.get_species_scientific_name(species)
            if corner_ranges is None:
                corner_ranges = free_mass_fraction_limits
            if fixed_mass_fraction_value is not None:
                self.parameters[species] = Parameter(
                    species,
                    False,
                    value=fixed_mass_fraction_value,
                    plot_in_corner=False)
            else:
                if self.sampler_type == 'pymultinest':
                    self.parameters[species] = Parameter(
                        name=species,
                        is_free_parameter=True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            free_mass_fraction_limits[0],
                            free_mass_fraction_limits[1]
                            ),
                        plot_in_corner=plot_in_corner,
                        corner_label=corner_label,
                        corner_ranges=corner_ranges
                    )
                else:
                    self.parameters[species] = Parameter(
                        name=species,
                        is_free_parameter=True,
                        distribution=tfpd.Uniform(low=free_mass_fraction_limits[0], high=free_mass_fraction_limits[1]),
                        plot_in_corner=plot_in_corner,
                        corner_label=corner_label,
                        corner_ranges=corner_ranges
                    )

    def add_pressure_varying_line_species(
            self,
            species: str,
            mode: str = 'linear',
            pressure_spacing: str = 'relative',
            n_nodes: int = 3,
            free_mass_fraction_limits: tuple[float, float] = (-7.0, 0.0),
            log_pressure_range_prior: tuple[float, float] = (0, 9),
            fixed_pressure_node_species: str | None = None,
            # --- deprecated arguments ---
            abund_lim=_UNSET
            ) -> None:
        """
        This function adds a single species to the Radtrans object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line
        opacities. This species will have and abundance that varies with pressure, defined by the retrieved
        abundance at each of the provided abundance nodes.

        This function adds a set of parameters to the retrieval to define the abundance profile of the species.
         - {species}_{mode}_abundance_profile defines which profile will be used (fixed parameter)
         - {species}_n_abundance_nodes sets the number of abundance nodes (fixed parameter)
         - {species}_n_pressure_nodes sets the number of pressure nodes (n_nodes - 2) (fixed parameter)
         - {species}_interpolation_mode sets the pressure spacing mode (fixed parameter)
         - {species}_pressure_node_{i} for i in 0 to n_nodes-2, the pressure at each node (free parameter)
         - {species}_abundance_node_{i} for i in 0 to n_nodes, the abundance at each node (free parameter)

        Args:
            species: str
                The species to include in the retrieval
            mode: str
                One of 'linear', 'cubic', or 'stepped' - determines the interpolation method between abundance nodes.
            pressure_spacing: str
                One of 'relative', 'absolute', or 'fixed' - determines whether the pressure nodes are spaced
                relative to the top of the atmosphere pressure, retrieved in absolute log pressue, or fixed
                to the pressure nodes set by fixed_pressure_node_species.
            n_nodes: int
                The number of abundance nodes to use in the retrieval, including top and bottom nodes.
            free_mass_fraction_limits : Tuple(float,float)
                If free is True, this sets the boundaries of the uniform prior that will be applied the species given.
                The range of the prior goes from free_mass_fraction_limits[0] to free_mass_fraction_limits[1]
                The abundance limits must be given in log10 units of the mass fraction.
            log_pressure_range_prior: Tuple(float,float)
                The prior range on the log pressure of the pressure nodes in log bar, only used if pressure_spacing is
                'absolute' or 'relative'.
            fixed_pressure_node_species: str
                If pressure_spacing is 'fixed', this species pressure nodes will be used as the pressure nodes.
                Note that this species must already have been added to the retrieval with
                add_pressure_varying_line_species.
        """
        if abund_lim is not _UNSET:
            warnings.warn(
                "The 'abund_lim' argument is deprecated; use 'free_mass_fraction_limits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            free_mass_fraction_limits = abund_lim

        Opacity.check_name(species)
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        # parameter passed through loglike is log10 abundance
        if free_mass_fraction_limits[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {free_mass_fraction_limits})! Please set abundance limits as (low, high)"
            )

        self.parameters[f"{species_full_name}_{mode}_abundance_profile"] = Parameter(
            name=f"{species_full_name}_linear_abundance_profile",
            is_free_parameter=False,
            value=True
        )
        self.parameters[f"{species_full_name}_n_abundance_nodes"] = Parameter(
            name=f"{species_full_name}_n_abundance_nodes",
            is_free_parameter=False,
            value=n_nodes
        )
        self.parameters[f"{species_full_name}_n_pressure_nodes"] = Parameter(
            name=f"{species_full_name}_n_pressure_nodes",
            is_free_parameter=False,
            value=n_nodes-2
        )
        self.parameters[f"{species_full_name}_interpolation_mode"] = Parameter(
            name=f"{species_full_name}_interpolation_mode",
            is_free_parameter=False,
            value=pressure_spacing
        )

        self.line_species = self.line_species + (species_full_name,)

        for i in range(n_nodes - 2):
            if pressure_spacing != 'fixed':
                self.parameters[f"{species_full_name}_pressure_node_{i}"] = Parameter(
                    name=f"{species_full_name}_pressure_node_{i}",
                    is_free_parameter=True,
                    transform_prior_cube_coordinate=lambda x: log_pressure_range_prior[0] +
                    log_pressure_range_prior[1] * x
                )
            else:
                self.parameters[f"{species_full_name}_pressure_node_{i}"] = Parameter(
                    name=f"{fixed_pressure_node_species}_pressure_node_{i}",
                    is_free_parameter=False,
                    value=self.parameters[f"{fixed_pressure_node_species}_pressure_node_{i}"].value
                )

        for i in range(n_nodes):
            self.parameters[f"{species_full_name}_abundance_node_{i}"] = Parameter(
                name=f"{species_full_name}_abundance_node_{i}",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: free_mass_fraction_limits[0] +
                (free_mass_fraction_limits[1] - free_mass_fraction_limits[0]) * x
            )

    def remove_species_lines(self, species: str, free: bool = False) -> None:
        """
        This function removes a species from the pRT line list, and if using a free chemistry retrieval,
        removes the associated Parameter of the species.

        Args:
            species : str
                The species to remove from the retrieval
            free : bool
                If true, the retrieval should use free chemistry, and Parameters for the abundance of the
                species will be removed to the retrieval
        """
        remove_opacity = Opacity([species])
        species_full_name = remove_opacity.get_full_name().split('.R')[0]
        if species in self.line_species:
            self.line_species = filter(species, self.line_species)
        elif species_full_name in self.line_species:
            self.line_species = filter(species_full_name, self.line_species)
        if free:
            self.parameters.pop(species, None)
            self.parameters.pop(species_full_name, None)

    def add_cloud_species(self,
                          species: str,
                          use_equilibrium_chemistry: bool = True,
                          free_mass_fraction_limits: tuple[float, float] = (-3.5, 1.5),
                          cloud_base_pressure_limits: tuple[float, float] | None = None,
                          equilbrium_mass_fraction_scaling_factor: tuple[float, float] | None = None,
                          fixed_mass_fraction_value: float | None = None,
                          fixed_base_pressure_value: float | None = None,
                          # --- deprecated arguments ---
                          eq=_UNSET,
                          abund_lim=_UNSET,
                          p_base_lim=_UNSET,
                          scaling_factor=_UNSET,
                          fixed_abund=_UNSET,
                          fixed_base=_UNSET) -> None:
        """
        This function adds a single cloud species to the list of species. Optionally,
        it will add parameters to allow for a retrieval using an ackermann-marley model.
        If an equilibrium condensation model is used in th retrieval model function (use_equilibrium_chemistry=True),
        then a parameter is added that scales the equilibrium cloud abundance, as in Molliere (2020).
        If eq is false, two parameters are added, the cloud abundnace and the cloud base pressure.
        The limits set the prior ranges, both on a log scale.

        NOTE: As of pRT version 2.4.9, the behaviour of this function has changed. In previous versions the
        abundance limits were set from free_mass_fraction_limits[0] to (free_mass_fraction_limits[0] + free_mass_fraction_limits[1]). This has been changed
        so that the limits of the prior range are from free_mass_fraction_limits[0] to free_mass_fraction_limits[1] (ie the actual boundaries).
        The same is true for PBase_lim.

        Args:
            species : str
                Name of the pRT cloud species, including the cloud shape tag.
            use_equilibrium_chemistry : bool
                Does the retrieval model use an equilibrium cloud model. This restricts the available species!
            free_mass_fraction_limits : tuple(float,float)
                If use_equilibrium_chemistry is True, this sets the prior range on the log mass fraction of the
                cloud species relative to the equilibrium abundance, with a typical range being (-3.5, 1.5).
                If use_equilibrium_chemistry is False, this sets the prior range on the actual log mass fraction
                abundance of the cloud species, with a typical range being (-5.0, 0.0).
                The abundance limits must be given in log10 units of the mass fraction.
            cloud_base_pressure_limits : tuple(float,float)
                Only used if not using an equilibrium model. Sets the prior range on the log10 of the cloud base
                pressure in bar, e.g. (-3, 3). If None, no cloud base pressure parameter is added.
            equilbrium_mass_fraction_scaling_factor : tuple(float,float)
                If provided, adds a free parameter ``eq_scaling_{cloud_name}`` that scales the equilibrium
                condensate mass fraction. The tuple defines the (lower, upper) bounds of the uniform prior on
                this scaling factor in log10 space, e.g. (-3.0, 1.0). Only used when
                use_equilibrium_chemistry is True. If None, no scaling parameter is added.
            fixed_mass_fraction_value : Optional(float)
                A vertically constant log mass fraction abundance for the cloud species. If set, this will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds.
            fixed_base_pressure_value : Optional(float)
                The log cloud base pressure. If set, fixes this parameter to a constant value, and it will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds. Not yet compatible
                with most built in pRT models.
        """
        if eq is not _UNSET:
            warnings.warn(
                "The 'eq' argument is deprecated; use 'use_equilibrium_chemistry' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            use_equilibrium_chemistry = eq
        if abund_lim is not _UNSET:
            warnings.warn(
                "The 'abund_lim' argument is deprecated; use 'free_mass_fraction_limits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            free_mass_fraction_limits = abund_lim
        if p_base_lim is not _UNSET:
            warnings.warn(
                "The 'p_base_lim' argument is deprecated; use 'cloud_base_pressure_limits' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cloud_base_pressure_limits = p_base_lim
        if scaling_factor is not _UNSET:
            warnings.warn(
                "The 'scaling_factor' argument is deprecated; use 'equilbrium_mass_fraction_scaling_factor' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            equilbrium_mass_fraction_scaling_factor = scaling_factor
        if fixed_abund is not _UNSET:
            warnings.warn(
                "The 'fixed_abund' argument is deprecated; use 'fixed_mass_fraction_value' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fixed_mass_fraction_value = fixed_abund
        if fixed_base is not _UNSET:
            warnings.warn(
                "The 'fixed_base' argument is deprecated; use 'fixed_base_pressure_value' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fixed_base_pressure_value = fixed_base

        if species.endswith("(c)"):
            logging.warning("Ensure you set the cloud particle shape!")
            logging.warning(species + " was not added to the list of cloud species")
            return
        cloud_opacity = CloudOpacity([species], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        self.cloud_species = self.cloud_species + (species_full_name,)
        if equilbrium_mass_fraction_scaling_factor is not None:
            if self.sampler_type == 'pymultinest':
                self.parameters['eq_scaling_' + cloud_name] = Parameter(
                    'eq_scaling_' + cloud_name,
                    True,
                    transform_prior_cube_coordinate=lambda x: uniform_prior(
                        x,
                        equilbrium_mass_fraction_scaling_factor[0],
                        equilbrium_mass_fraction_scaling_factor[1]
                        )
                )
            else:
                self.parameters['eq_scaling_' + cloud_name] = Parameter(
                    'eq_scaling_' + cloud_name,
                    True,
                    distribution=tfpd.Uniform(
                        low=equilbrium_mass_fraction_scaling_factor[0],
                        high=equilbrium_mass_fraction_scaling_factor[1]
                    )
                )
        if not use_equilibrium_chemistry:
            if free_mass_fraction_limits[1] > 0.0:
                raise ValueError(
                    f"upper limit must be <= 0.0 (was {free_mass_fraction_limits})! Please set abundance limits as (low, high)"
                )

            if fixed_mass_fraction_value is None:
                if self.sampler_type == 'pymultinest':
                    self.parameters['log_X_cb_' + cloud_name] = Parameter(
                        'log_X_cb_' + cloud_name,
                        True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            free_mass_fraction_limits[0],
                            free_mass_fraction_limits[1]
                        )
                    )
                else:
                    self.parameters['log_X_cb_' + cloud_name] = Parameter(
                        'log_X_cb_' + cloud_name,
                        True,
                        distribution=tfpd.Uniform(
                            low=free_mass_fraction_limits[0],
                            high=free_mass_fraction_limits[1]
                        )
                    )
            else:
                self.parameters['log_X_cb_' + cloud_name] = Parameter(
                    'log_X_cb_' + cloud_name,
                    False,
                    value=fixed_mass_fraction_value
                )

        if cloud_base_pressure_limits is not None or fixed_base_pressure_value is not None:
            if fixed_base_pressure_value is None:
                if self.sampler_type == 'pymultinest':
                    self.parameters['log_Pbase_' + cloud_name] = Parameter(
                        'log_Pbase_' + cloud_name,
                        True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            cloud_base_pressure_limits[0],
                            cloud_base_pressure_limits[1]
                        )
                    )
                else:
                    self.parameters['log_Pbase_' + cloud_name] = Parameter(
                        'log_Pbase_' + cloud_name,
                        True,
                        distribution=tfpd.Uniform(
                            low=cloud_base_pressure_limits[0],
                            high=cloud_base_pressure_limits[1]
                        )
                    )
            else:
                self.parameters['log_Pbase_' + cloud_name] = Parameter(
                    'log_Pbase_' + cloud_name,
                    False,
                    value=fixed_base_pressure_value
                )

    def add_data(self,
                 name: str,
                 path_to_observations: str,
                 model_generating_function: Callable,
                 data_resolution: float | None = None,
                 model_resolution: float | None = None,
                 system_distance: float | None = None,
                 scale_flux: bool = False,
                 scale_uncertainties: bool = False,
                 fit_flux_offset: bool = False,
                 fit_instrumental_resolution: bool = False,
                 subtract_continuum: bool = False,
                 photometry: bool = False,
                 photometric_transformation_function: Callable | None = None,
                 photometric_bin_edges: tuple[float, float] | None = None,
                 wavelength_boundaries: tuple[float, float] | None = None,
                 external_radtrans_reference: str | None = None,
                 line_opacity_mode: str = 'c-k',
                 wavelength_bin_widths: ArrayLike | None = None,
                 radtrans_grid: bool = False,
                 radtrans_object: Any | None = None,
                 wavelengths: ArrayLike | None = None,
                 spectrum: ArrayLike | None = None,
                 uncertainties: ArrayLike | None = None,
                 mask: ArrayLike | None = None,
                 covariance: ArrayLike | None = None,
                 fit_covariance: bool = False,
                 covariance_mode: str = 'none',
                 global_covariance_kernel: str = 'squared_exponential',
                 local_covariance_kernel: str = 'squared_exponential',
                 n_local_covariance_kernels: int = 0,
                 covariance_jitter: float = 0.0,
                 covariance_parameter_priors: dict | None = None,
                 # --- deprecated arguments ---
                 scale=_UNSET,
                 scale_err=_UNSET,
                 offset_bool=_UNSET,
                 resample=_UNSET,
                 concatenate_flux_epochs_variability=_UNSET,
                 variability_atmospheric_column_model_flux_return_mode=_UNSET,
                 atmospheric_column_flux_mixer=_UNSET) -> None:
        """
        Create a Data class object.

        Args:
            name : str
                Identifier for this data set.
            path_to_observations : str
                Path to observations file, including filename. This can be a txt or dat file containing the wavelength,
                flux, transit depth and error, or a fits file containing the wavelength, spectrum and covariance matrix.
            model_generating_function : fnc
                A function, typically defined in run_definition.py that returns the model wavelength and
                spectrum (emission or transmission). This is the function that contains the physics
                of the model, and calls pRT in order to compute the spectrum.
            data_resolution : float
                Spectral resolution of the instrument. Optional, allows convolution of model to instrumental line width.
            model_resolution : float
                Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
            system_distance : float
                The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must
                be scaled to the same distance before running the retrieval, which can be done using the
                scale_to_distance method in the Data class.
            scale_flux : bool
                Turn on or off scaling the data by a constant factor.
            wavelength_boundaries : Tuple
                A pair of wavelengths in units of micron that determine the lower and upper boundaries of the
                model computation.
            external_radtrans_reference : str
                The name of an existing Data object. This object's prt_object will be used to calculate the chi squared
                of the new Data object. This is useful when two datasets overlap, as only one model computation is
                required to compute the log likelihood of both datasets.
            line_opacity_mode : str
                Should the retrieval be run using correlated-k opacities (default, 'c-k'),
                or line by line ('lbl') opacities? If 'lbl' is selected, git push origin --delete it is HIGHLY
                recommended to set the model_resolution parameter.
            radtrans_grid: bool
                Set to true if data has been binned to a pRT opacity grid, exactly.
            fit_covariance : bool
                If True, fit a parametric covariance model for this dataset during the retrieval.
                This is currently supported for 1D spectroscopic datasets only.
            covariance_mode : str
                Covariance model to fit. Supported values are ``'none'``, ``'local'``, ``'global'``, and
                ``'global_local'``.
            global_covariance_kernel : str
                Name of the global covariance kernel. Supported values are ``'squared_exponential'`` and
                ``'matern32'``.
            local_covariance_kernel : str
                Name of the local covariance kernel. Currently only ``'squared_exponential'`` is supported.
            n_local_covariance_kernels : int
                Number of local covariance kernels to include when ``covariance_mode='global_local'``.
            covariance_jitter : float
                Small diagonal stabilization term added to the fitted covariance matrix.
            covariance_parameter_priors : dict
                Optional override dictionary for the automatically generated covariance hyperparameter priors.
                Supported keys are ``'global_amp'``, ``'global_length'``, ``'local_amp'``, ``'local_length'``,
                and ``'local_center'``.
        """
        if scale is not _UNSET:
            warnings.warn(
                "The 'scale' argument is deprecated; use 'scale_flux' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scale_flux = scale
        if scale_err is not _UNSET:
            warnings.warn(
                "The 'scale_err' argument is deprecated; use 'scale_uncertainties' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scale_uncertainties = scale_err
        if offset_bool is not _UNSET:
            warnings.warn(
                "The 'offset_bool' argument is deprecated; use 'fit_flux_offset' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fit_flux_offset = offset_bool
        if resample is not _UNSET:
            warnings.warn(
                "The 'resample' argument is deprecated; use 'fit_instrumental_resolution' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fit_instrumental_resolution = resample
        if concatenate_flux_epochs_variability is not _UNSET:
            warnings.warn(
                "The 'concatenate_flux_epochs_variability' argument is deprecated and has been removed from "
                "add_data(). Use add_time_series_data() for time-variable retrievals.",
                DeprecationWarning,
                stacklevel=2,
            )
        if variability_atmospheric_column_model_flux_return_mode is not _UNSET:
            warnings.warn(
                "The 'variability_atmospheric_column_model_flux_return_mode' argument is deprecated and has been "
                "removed from add_data(). Use add_time_series_data() for time-variable retrievals.",
                DeprecationWarning,
                stacklevel=2,
            )
        if atmospheric_column_flux_mixer is not _UNSET:
            warnings.warn(
                "The 'atmospheric_column_flux_mixer' argument is deprecated and has been removed from "
                "add_data(). Use add_time_series_data() for time-variable retrievals.",
                DeprecationWarning,
                stacklevel=2,
            )

        data_object = Data(
            name=name,
            path_to_observations=path_to_observations,
            model_generating_function=model_generating_function,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            system_distance=system_distance,
            scale_flux=scale_flux,
            scale_uncertainties=scale_uncertainties,
            fit_flux_offset=fit_flux_offset,
            fit_instrumental_resolution=fit_instrumental_resolution,
            subtract_continuum=subtract_continuum,
            photometry=photometry,
            photometric_transformation_function=photometric_transformation_function,
            photometric_bin_edges=photometric_bin_edges,
            wavelength_boundaries=wavelength_boundaries,
            external_radtrans_reference=external_radtrans_reference,
            line_opacity_mode=line_opacity_mode,
            wavelength_bin_widths=wavelength_bin_widths,
            radtrans_grid=radtrans_grid,
            radtrans_object=radtrans_object,
            wavelengths=wavelengths,
            spectrum=spectrum,
            uncertainties=uncertainties,
            covariance=covariance,
            fit_covariance=fit_covariance,
            covariance_mode=covariance_mode,
            global_covariance_kernel=global_covariance_kernel,
            local_covariance_kernel=local_covariance_kernel,
            n_local_covariance_kernels=n_local_covariance_kernels,
            covariance_jitter=covariance_jitter,
            mask=mask,
        )
        self.data[name] = data_object

        if fit_covariance:
            self._add_covariance_hyperparameters(
                data_object=data_object,
                covariance_mode=data_object.covariance_mode,
                n_local_covariance_kernels=data_object.n_local_covariance_kernels,
                covariance_parameter_priors=covariance_parameter_priors,
            )

    def add_time_series_data(
        self,
        name: str,
        observation_times: ArrayLike,
        model_generating_function: Callable,
        n_model_timesteps: int,
        time_varying_parameters: list[str] | None = None,
        sinusoidal_parameters: list[str] | None = None,
        *,
        path_to_observations: str | None = None,
        filename_list: list[str] | None = None,
        wavelengths: ArrayLike | None = None,
        spectrum: ArrayLike | None = None,
        uncertainties: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        wavelength_bin_widths: ArrayLike | None = None,
        data_resolution: float | None = None,
        model_resolution: float | None = None,
        system_distance: float | None = None,
        wavelength_boundaries: tuple | None = None,
        external_radtrans_reference: str | None = None,
        line_opacity_mode: str = 'c-k',
        scale_flux: bool = False,
        scale_uncertainties: bool = False,
        fit_flux_offset: bool = False,
        radtrans_object: object = None,
        covariance: ArrayLike | None = None,
    ):
        """Create a :class:`TimeSeriesData` object for time-variable retrievals.

        This method creates the data object **and** registers the fixed
        ``N_time`` parameter.  The caller is still responsible for adding the
        per-timestep or sinusoidal free parameters with appropriate priors via
        :meth:`add_parameter`.

        Args:
            name : str
                Identifier for this data set.
            observation_times : ArrayLike
                1-D array of observation times in seconds (absolute or relative
                to the first observation).
            model_generating_function : Callable
                A runtime-native model function (e.g.
                ``time_series_gradient_emission``).
            n_model_timesteps : int
                Number of model spectra computed at regular intervals spanning
                the observation window.  Spectra at the actual observation
                times are interpolated from these.
            time_varying_parameters : list[str], optional
                Parameter names that are freely varied at each model timestep.
                For each name ``P`` the retrieval must contain free parameters
                ``{P}_t_0`` through ``{P}_t_{n_model_timesteps - 1}``.
            sinusoidal_parameters : list[str], optional
                Parameter names whose time variation is described by a
                sinusoid.  For each name ``P`` the retrieval must contain free
                parameters ``{P}_amplitude``, ``{P}_period``, ``{P}_phase``,
                and ``{P}_offset``.
            path_to_observations : str, optional
                Path to a directory containing individual epoch files when
                using *filename_list*.
            filename_list : list[str], optional
                Per-epoch filenames inside *path_to_observations*.  If given,
                data will be loaded via
                :meth:`TimeSeriesData.load_single_spectrum_txt`.
            wavelengths : ArrayLike, optional
                1-D wavelength grid (micron) shared by all epochs.
            spectrum : ArrayLike, optional
                2-D flux array of shape ``(N_obs, N_wavelength)``.
            uncertainties : ArrayLike, optional
                2-D uncertainty array matching *spectrum*.
            mask : ArrayLike, optional
                2-D boolean mask matching *spectrum*.
            wavelength_bin_widths : ArrayLike, optional
                1-D or 2-D bin widths.
            data_resolution, model_resolution, system_distance,
            wavelength_boundaries, external_radtrans_reference,
            line_opacity_mode, scale_flux, scale_uncertainties,
            fit_flux_offset, radtrans_object, covariance
                Same semantics as :meth:`add_data`.
        """
        if time_varying_parameters is None:
            time_varying_parameters = []
        if sinusoidal_parameters is None:
            sinusoidal_parameters = []

        overlap = set(time_varying_parameters) & set(sinusoidal_parameters)
        if overlap:
            raise ValueError(
                f"Parameters cannot be both time-varying and sinusoidal: {overlap}"
            )

        data_object = TimeSeriesData(
            name=name,
            path_to_observations=path_to_observations,
            filename_list=filename_list,
            observation_times=observation_times,
            model_generating_function=model_generating_function,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            system_distance=system_distance,
            wavelength_boundaries=wavelength_boundaries,
            external_radtrans_reference=external_radtrans_reference,
            line_opacity_mode=line_opacity_mode,
            scale_flux=scale_flux,
            scale_uncertainties=scale_uncertainties,
            fit_flux_offset=fit_flux_offset,
            radtrans_object=radtrans_object,
            wavelengths=wavelengths,
            spectrum=spectrum,
            uncertainties=uncertainties,
            wavelength_bin_widths=wavelength_bin_widths,
            covariance=covariance,
            mask=mask,
        )

        # Attach time-series configuration so the model function can read it
        # via ModelContext.model_metadata['time_series_config'].
        data_object.time_series_config = {
            'n_model_timesteps': n_model_timesteps,
            'time_varying_parameters': list(time_varying_parameters),
            'sinusoidal_parameters': list(sinusoidal_parameters),
        }

        self.data[name] = data_object

        # Register the fixed N_time parameter (shared across all datasets)
        if 'N_time' not in self.parameters:
            self.add_parameter(
                'N_time', is_free_parameter=False, value=n_model_timesteps,
            )

    def add_photometry(
        self,
        path: str,
        model_generating_function: Callable,
        model_resolution: float = 10.0,
        distance: float | None = None,
        scale_flux: bool = False,
        wlen_range_micron: tuple[float, float] = None,
        photometric_transformation_function: Callable = None,
        external_prt_reference: object = None,
        opacity_mode: str = 'c-k'
    ) -> None:
        """
        Create a Data class object for each photometric point in a photometry file.
        The photometry file must be a csv file and have the following structure:
        name, lower wavelength bound [um], upper wavelength boundary[um], flux [W/m2/micron], flux error [W/m2/micron]

        Photometric data requires a transformation function to convert a spectrum into synthetic photometry.
        You must provide this function yourself, or have the species package installed.
        If using species, the name in the data file must be of the format instrument/filter.

        Args:
            model_generating_function : str
                Identifier for this data set.
            path : str
                Path to observations file, including filename.
            model_resolution : float
                Spectral resolution of the model, allowing for low resolution correlated k tables from exo-k.
            scale_flux : bool
                Turn on or off scaling the data by a constant factor. Currently only set up to scale all
                photometric data in a given file.
            distance : float
                The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must
                be scaled to the same distance before running the retrieval, which can be done using the
                scale_to_distance method in the Data class.
            wlen_range_micron : Tuple
                A pair of wavelengths in units of micron that determine the lower and upper boundaries of
                the model computation.
            external_prt_reference : str
                The name of an existing Data object. This object's prt_object will be used to calculate the
                chi squared of the new Data object. This is useful when two datasets overlap, as only
                one model computation is required to compute the log likelihood of both datasets.
            photometric_transformation_function : method
                A function that will transform a spectrum into an average synthetic photometric point,
                typically accounting for filter transmission.
            opacity_mode: str
                Opacity mode.
        """
        with open(path) as photometry:
            for line in photometry:
                # # must be the comment character
                if line[0] == '#':
                    continue

                columns = line.split(',')
                name = columns[0]
                wavelengths_low = float(columns[1])
                wavelengths_high = float(columns[2])
                flux = float(columns[3])
                uncertainties = float(columns[4])

                if photometric_transformation_function is None:
                    if comm is not None and comm.Get_size() > 1:
                        if rank == 0:
                            transform = SyntheticPhotometry(name).spectrum_to_flux
                        else:
                            transform = None  # transform still needs to exist in the other processes
                        transform = comm.bcast(transform, 0)
                    else:
                        transform = SyntheticPhotometry(name).spectrum_to_flux
                else:
                    transform = photometric_transformation_function

                if wlen_range_micron is None:
                    wbins = [0.95 * wavelengths_low, 1.05 * wavelengths_high]
                else:
                    wbins = wlen_range_micron

                if opacity_mode == 'lbl':
                    logging.warning("Are you sure you want a high resolution model for photometry?")

                data_resolution: float = (
                    np.array(
                        [wavelengths_low, wavelengths_high]
                    ).mean()
                    / (wavelengths_high - wavelengths_low)
                )

                self.data[name] = PhotometryData(
                    name=name,
                    path_to_observations=path,
                    model_generating_function=model_generating_function,
                    system_distance=distance,
                    wavelength_boundaries=wbins,
                    photometric_bin_edges=(wavelengths_low, wavelengths_high),
                    data_resolution=data_resolution,
                    model_resolution=model_resolution,
                    scale_flux=scale_flux,
                    photometric_transformation_function=transform,
                    external_radtrans_reference=external_prt_reference,
                    line_opacity_mode=opacity_mode
                )
                self.data[name].wavelengths = np.mean(np.vstack((wavelengths_low, wavelengths_high)), axis=0)
                self.data[name].spectrum = flux
                self.data[name].uncertainties = uncertainties
