import logging
import os

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike

from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.retrieval.data import Data, PhotometryData, TimeSeriesData
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.retrieval.jax_parameter import Parameter
from petitRADTRANS.retrieval.utils import uniform_prior

# Import prior distributions
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

# MPI Multiprocessing
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
                 sampler_type='pymultinest',
                 adaptive_mesh_refinement: str = False,
                 scattering_in_emission: bool = False,
                 pressures: ArrayLike | None = None):

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
        self.plot_kwargs = {}

        self._plot_defaults()

        self.add_parameter("pressure_scaling", False, value=1)
        self.add_parameter("pressure_width", False, value=1)
        self.add_parameter("pressure_simple", False, value=self.pressures.shape[0])

    def _plot_defaults(self):
        ##################################################################
        # Define axis properties of spectral plot if run_mode == 'evaluate'
        ##################################################################
        self.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
        self.plot_kwargs["spec_ylabel"] = "Flux [W/m2/micron]"
        self.plot_kwargs["y_axis_scaling"] = 1.0
        self.plot_kwargs["xscale"] = 'log'
        self.plot_kwargs["yscale"] = 'linear'
        self.plot_kwargs["resolution"] = 1500.
        self.plot_kwargs["nsample"] = 10.

        ##################################################################
        # Define from which observation object to take P-T
        # in evaluation mode (if run_mode == 'evaluate'),
        # add PT-envelope plotting options
        ##################################################################
        self.plot_kwargs["temp_limits"] = [150, 3000]
        self.plot_kwargs["press_limits"] = [1e2, 1e-5]

    def _setup_pres(self, scaling=10, width=3):
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
            name,
            is_free_parameter,
            value=None,
            distribution=None,
            plot_in_corner=False,
            corner_ranges=None,
            corner_transform=None,
            transform_prior_cube_coordinate=None,
            corner_label=None):
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

    @staticmethod
    def list_available_line_species():
        """
        List the currently installed opacity tables that are available for species that contribute to the line opacity.
        """

        prt_path = petitradtrans_config_parser.get_input_data_path()

        files = [f[0].split('/')[-1] for f in os.walk(prt_path + "/opacities/lines/correlated_k/")]
        files = set([f.split('_R_')[0] for f in files])
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
    def list_available_cloud_species():
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
    def list_available_cia_species():
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
            line_species,
            use_equilibrium_chemistry=False,
            free_mass_fraction_limits=(-6.0, -0.5),
            plot_in_corner=True):
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
                        corner_label=species_name,
                        corner_ranges=free_mass_fraction_limits
                    )
                else:
                    self.parameters[species] = Parameter(
                        name=species,
                        is_free_parameter=True,
                        distribution=tfpd.Uniform(low=free_mass_fraction_limits[0], high=free_mass_fraction_limits[1]),
                        plot_in_corner=plot_in_corner,
                        corner_label=species_name,
                        corner_ranges=free_mass_fraction_limits
                    )

    def set_rayleigh_species(self, rayleigh_species):
        """
        Set the list of species that contribute to the rayleigh scattering in the pRT object.

        Args:
            rayleigh_species : List(str)
                A list of species that contribute to the rayleigh opacity.
        """

        self.rayleigh_species = rayleigh_species

    def set_continuum_opacities(self, continuum_opacities):
        """
        Set the list of species that contribute to the continuum opacity in the pRT object.

        Args:
            continuum_opacities : List(str)
                A list of species that contribute to the continuum opacity.
        """

        self.continuum_opacities = continuum_opacities

    def add_line_species(
            self,
            species,
            use_equilibrium_chemistry=False,
            free_mass_fraction_limits=(-6.0, -0.5),
            fixed_mass_fraction_value=None,
            plot_in_corner=True,
            corner_label=None,
            corner_ranges=None):
        """
        This function adds a single species to the pRT object that will define the line opacities of the model.
        The name must match the pRT opacity name, which vary between the c-k line opacities and the line-by-line
        opacities.

        NOTE: As of pRT version 2.4.9, the behaviour of this function has changed. In previous versions the
        abundance limits were set from free_mass_fraction_limits[0] to (free_mass_fraction_limits[0] + free_mass_fraction_limits[1]). This has been changed
        so that the limits of the prior range are from free_mass_fraction_limits[0] to abund_lim[1] (ie the actual boundaries).

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

        # parameter passed through loglike is log10 abundance
        if free_mass_fraction_limits[1] > 0.0:
            raise ValueError(
                f"upper limit must be <= 0.0 (was {free_mass_fraction_limits})! Please set abundance limits as (low, high)"
            )

        self.line_species = self.line_species + (species,)
        if not use_equilibrium_chemistry:
            species_info = Opacity.split_species_all_info(species)
            species_name = species_info[0]  # remove possible previous spectral info
            if corner_label is None:
                corner_label = species_name
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

    def remove_species_lines(self, species, free=False):
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
        line_list = list(self.line_species)
        if species in line_list:
            line_list.remove(species)
        if free:
            self.parameters.pop(species, None)
        self.line_species = tuple(line_list)

    def add_cloud_species(self,
                          species,
                          use_equilibrium_chemistry=True,
                          free_mass_fraction_limits=(-3.5, 1.5),
                          cloud_base_pressure_limits=None,
                          equilbrium_mass_fraction_scaling_factor=None,
                          fixed_mass_fraction_value=None,
                          fixed_base_pressure_value=None):
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
                If eq is True, this sets the scaling factor for the equilibrium condensate abundance, typical
                range would be (-3,1). If eq is false, this sets the range on the actual cloud abundance,
                with a typical range being (-5,0).
            cloud_base_pressure_limits : tuple(float,float)
                Only used if not using an equilibrium model. Sets the limits on the log of the cloud base pressure.
                Obsolete.
            fixed_mass_fraction_value : Optional(float)
                A vertically constant log mass fraction abundance for the cloud species. If set, this will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds.
            fixed_base_pressure_value : Optional(float)
                The log cloud base pressure. If set, fixes this parameter to a constant value, and it will not be
                a free parameter in the retrieval. Only compatible with non-equilibrium clouds. Not yet compatible
                with most built in pRT models.
            scaling_factor :
                # TODO complete docstring
        """

        if species.endswith("(c)"):
            logging.warning("Ensure you set the cloud particle shape!")
            logging.warning(species + " was not added to the list of cloud species")
            return

        self.cloud_species = self.cloud_species + (species,)
        cname = species.split('_')[0]
        if equilbrium_mass_fraction_scaling_factor is not None:
            if self.sampler_type == 'pymultinest':
                self.parameters['eq_scaling_' + cname] = Parameter(
                    'eq_scaling_' + cname,
                    True,
                    transform_prior_cube_coordinate=lambda x: uniform_prior(
                        x,
                        equilbrium_mass_fraction_scaling_factor[0],
                        equilbrium_mass_fraction_scaling_factor[1]
                        )
                )
            else:
                self.parameters['eq_scaling_' + cname] = Parameter(
                    'eq_scaling_' + cname,
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
                    self.parameters['log_X_cb_' + cname] = Parameter(
                        'log_X_cb_' + cname,
                        True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            free_mass_fraction_limits[0],
                            free_mass_fraction_limits[1]
                        )
                    )
                else:
                    self.parameters['log_X_cb_' + cname] = Parameter(
                        'log_X_cb_' + cname,
                        True,
                        distribution=tfpd.Uniform(
                            low=free_mass_fraction_limits[0],
                            high=free_mass_fraction_limits[1]
                        )
                    )
            else:
                self.parameters['log_X_cb_' + cname] = Parameter(
                    'log_X_cb_' + cname,
                    False,
                    value=fixed_mass_fraction_value
                )

        if cloud_base_pressure_limits is not None or fixed_base_pressure_value is not None:
            if fixed_base_pressure_value is None:
                if self.sampler_type == 'pymultinest':
                    self.parameters['log_Pbase_' + cname] = Parameter(
                        'log_Pbase_' + cname,
                        True,
                        transform_prior_cube_coordinate=lambda x: uniform_prior(
                            x,
                            cloud_base_pressure_limits[0],
                            cloud_base_pressure_limits[1]
                        )
                    )
                else:
                    self.parameters['log_Pbase_' + cname] = Parameter(
                        'log_Pbase_' + cname,
                        True,
                        distribution=tfpd.Uniform(
                            low=cloud_base_pressure_limits[0],
                            high=cloud_base_pressure_limits[1]
                        )
                    )
            else:
                self.parameters['log_Pbase_' + cname] = Parameter(
                    'log_Pbase_' + cname,
                    False,
                    value=fixed_base_pressure_value
                )

    def add_data(self,
                 name,
                 path_to_observations,
                 model_generating_function,
                 data_resolution=None,
                 model_resolution=None,
                 system_distance=None,
                 scale=False,
                 scale_err=False,
                 offset_bool=False,
                 wavelength_boundaries=None,
                 external_radtrans_reference=None,
                 line_opacity_mode='c-k',
                 wavelength_bin_widths=None,
                 radtrans_grid=False,
                 radtrans_object=None,
                 wavelengths=None,
                 spectrum=None,
                 uncertainties=None,
                 mask=None,
                 covariance=None,
                 concatenate_flux_epochs_variability=False,
                 variability_atmospheric_column_model_flux_return_mode=False,
                 atmospheric_column_flux_mixer=None):
        """
        Create a Data class object.
        # TODO complete docstring
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
            scale : bool
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
                or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
                recommended to set the model_resolution parameter.
            radtrans_grid: bool
                Set to true if data has been binned to pRT R = 1,000 c-k grid.
        """
        self.data[name] = Data(
            name=name,
            path_to_observations=path_to_observations,
            model_generating_function=model_generating_function,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            system_distance=system_distance,
            scale=scale,
            scale_err=scale_err,
            offset_bool=offset_bool,
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
            mask=mask,
        )

    def add_photometry(self,
                       path,
                       model_generating_function,
                       model_resolution=10,
                       distance=None,
                       scale=False,
                       wlen_range_micron=None,
                       photometric_transformation_function=None,
                       external_prt_reference=None,
                       opacity_mode='c-k'):
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
            scale : bool
                Turn on or off scaling the data by a constant factor. Currently only set up to scale all
                photometric data in a given file.
            distance : float
                The distance to the object in cgs units. Defaults to a 10pc normalized distance. All data must
                be scaled to the same distance before running the retrieval, which can be done using the
                scale_to_distance method in the Data class.
            wlen_range_micron : Tuple
                A pair of wavelenths in units of micron that determine the lower and upper boundaries of
                the model computation.
            external_prt_reference : str
                The name of an existing Data object. This object's prt_object will be used to calculate the
                chi squared of the new Data object. This is useful when two datasets overlap, as only
                one model computation is required to compute the log likelihood of both datasets.
            photometric_transformation_function : method
                A function that will transform a spectrum into an average synthetic photometric point,
                typicall accounting for filter transmission.
            opacity_mode: str
                Opacity mode.
        """

        with open(path) as photometry:
            if photometric_transformation_function is None:
                try:
                    # TODO: switch to pyphot library? Lighter weight/less complicated than species.
                    import species as sp
                    from species.phot.syn_phot import SyntheticPhotometry

                    sp.SpeciesInit()
                except ModuleNotFoundError:  # TODO find what error is expected here
                    logging.error(
                        "Please provide a function to transform a spectrum into photometry, or pip install species"
                    )

            for line in photometry:
                # # must be the comment character
                if line[0] == '#':
                    continue

                vals = line.split(',')
                name = vals[0]
                wlow = float(vals[1])
                whigh = float(vals[2])
                flux = float(vals[3])
                err = float(vals[4])

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
                    wbins = [0.95 * wlow, 1.05 * whigh]
                else:
                    wbins = wlen_range_micron

                if opacity_mode == 'lbl':
                    logging.warning("Are you sure you want a high resolution model for photometry?")

                self.data[name] = PhotometryData(
                    name,
                    path,
                    model_generating_function=model_generating_function,
                    system_distance=distance,
                    wavelength_boundaries=wbins,
                    photometric_bin_edges=[wlow, whigh],
                    data_resolution=jnp.mean([wlow, whigh]) / (whigh - wlow),
                    model_resolution=model_resolution,
                    scale=scale,
                    photometric_transformation_function=transform,
                    external_radtrans_reference=external_prt_reference,
                    line_opacity_mode=opacity_mode
                )
                self.data[name].spectrum = flux
                self.data[name].uncertainties = err
