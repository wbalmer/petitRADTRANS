"""SpectralModel object and related."""

import copy
import os
import sys
import warnings

import h5py
import numpy as np
from petitRADTRANS.fort_rebin import fort_rebin as fr
from scipy.ndimage import gaussian_filter1d

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.ccf.pipeline import simple_pipeline
from petitRADTRANS.ccf.utils import dict2hdf5, hdf52dict, fill_object
from petitRADTRANS.containers.planet import Planet
from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import doppler_shift, guillot_metallic_temperature_profile
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, log_prior, uniform_prior, gaussian_prior, log_gaussian_prior, \
    delta_prior


class RetrievalParameter:
    available_priors = [
        'log',
        'uniform',
        'gaussian',
        'log_gaussian',
        'delta',
        'custom'
    ]

    def __init__(self, name, prior_parameters, prior_type='uniform', custom_prior=None):
        """Used to set up retrievals.
        Stores the prior function. Prior parameters depends on the type of prior. e.g., for uniform and log prior, these
        are the bounds of the prior. For gaussian priors and alike, these are the values of the mean and full width
        half maximum.

        Args:
            name: name of the parameter to retrieve, must match the corresponding model parameter of a SpectralModel
            prior_parameters: list of two values for the prior parameters, depends on the prior type
            prior_type: type of prior to use, the available types are stored into available_priors
            custom_prior: function with arguments (cube, *args), args being positional arguments in prior_parameters
        """
        # Check prior parameters validity
        if not hasattr(prior_parameters, '__iter__'):
            raise ValueError(
                f"'prior_parameters' must be an iterable of size 2, but is of type '{type(prior_parameters)}'"
            )
        elif np.size(prior_parameters) < 2:
            raise ValueError(
                f"'prior_parameters' must be of size 2, but is of size '{np.size(prior_parameters)}'"
            )
        elif prior_parameters[0] > prior_parameters[1] and (prior_type == 'log' or prior_type == 'uniform'):
            raise ValueError(
                f"lower prior boundaries ({prior_parameters[0]}) "
                f"must be lower than upper prior boundaries ({prior_parameters[1]})"
            )

        self.name = name
        self.prior_parameters = prior_parameters
        self.prior_type = prior_type

        # Set prior
        if self.prior_type == 'log':
            def prior(x):
                return log_prior(
                    cube=x,
                    lx1=self.prior_parameters[0],
                    lx2=self.prior_parameters[1]
                )
        elif self.prior_type == 'uniform':
            def prior(x):
                return uniform_prior(
                    cube=x,
                    x1=self.prior_parameters[0],
                    x2=self.prior_parameters[1]
                )
        elif self.prior_type == 'gaussian':
            def prior(x):
                return gaussian_prior(
                    cube=x,
                    mu=self.prior_parameters[0],
                    sigma=self.prior_parameters[1]
                )
        elif self.prior_type == 'log_gaussian':
            def prior(x):
                return log_gaussian_prior(
                    cube=x,
                    mu=self.prior_parameters[0],
                    sigma=self.prior_parameters[1]
                )
        elif self.prior_type == 'delta':
            def prior(x):
                return delta_prior(
                    cube=x,  # actually useless
                    x1=self.prior_parameters[0],
                    x2=self.prior_parameters[1]  # actually useless
                )
        elif self.prior_type == 'custom':
            def prior(x):
                return custom_prior(
                    cube=x,
                    *prior_parameters
                )
        else:
            raise ValueError(
                f"prior type '{prior_type}' not implemented "
                f"(available prior types: {'|'.join(RetrievalParameter.available_priors)})"
            )

        self.prior_function = prior

    @classmethod
    def from_dict(cls, dictionary):
        """Convert a dictionary into a list of RetrievalParameter.
        The keys of the dictionary are the names of the RetrievalParameter. The values of the dictionary must be
        dictionaries with keys 'prior_parameters' and 'prior_type'.

        Args:
            dictionary: a dictionary

        Returns:
            A list of RetrievalParameter.
        """
        new_retrieval_parameters = []

        for key, parameters in dictionary.items():
            new_retrieval_parameters.append(
                cls(
                    name=key,
                    prior_parameters=parameters['prior_parameters'],
                    prior_type=parameters['prior_type']
                )
            )

        return new_retrieval_parameters

    def put_into_dict(self, dictionary=None):
        """Convert a RetrievalParameter into a dictionary.

        Args:
            dictionary: a dictionary; if None, a new dictionary is created

        Returns:
            A dictionary.
        """
        if dictionary is None:
            dictionary = {}

        dictionary[self.name] = {
            'prior_boundaries': self.prior_parameters,
            'prior_type': self.prior_type
        }

        return dictionary


class BaseSpectralModel:
    # TODO ideally this should inherit from Radtrans, but it cannot be done right now because when Radtrans is init, it takes ages to load opacity data
    def __init__(self, pressures,
                 line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                 opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1,
                 temperatures=None, mass_mixing_ratios=None, mean_molar_masses=None,
                 wavelengths_boundaries=None, wavelengths=None, transit_radii=None, spectral_radiosities=None,
                 times=None,
                 **model_parameters):
        """Base for SpectralModel. Essentially a wrapper of Radtrans.
        Can be used to construct custom spectral models.

        Initialised as a Radtrans object. Additional parameters, that are used to generate the spectrum, are stored into
        the attribute model_parameters.
        The corresponding Radtrans object must be generated with the init_radtrans function.
        The spectrum is generated with the get_spectrum_model function.
        A retrieval object can be initialised with the init_retrieval function.
        The generated Retrieval can be run using the run_retrieval function.

        The model wavelength boundaries can be determined from the output_wavelengths model parameter, taking into
        account Doppler shift if necessary, using the following model parameters:
            - relative_velocities: (cm.s-1) array of relative velocities between the target and the observer
            - system_observer_radial_velocities: (cm.s-1) array of velocities between the system and the observer
            - is_orbiting: if True, planet_max_radial_orbital_velocity is calculated and relative velocities depends on
                orbital position parameters, listed below.
            - orbital_longitudes (deg) array of orbital longitudes
            - orbital_phases: array of orbital phases
            - planet_rest_frame_shift: (cm.s-1) array of offsets to the calculated relative_velocities
            - planet_max_radial_orbital_velocity: (cm.s-1) max radial orbital velocity of the planet (Kp), can be calc.
        The size of the arrays is used to generate multiple observations of the spectrum. For example, if n_phases
        orbital phases are given, the generated spectrum of size n_wavelengths is shifted according to the relative
        velocity at each orbital phase. This generates a time-dependent spectrum of shape (n_phases, n_wavelengths).

        The calculate_* functions can be rewritten in scripts if necessary. Functions get_* should not be rewritten.

        Args:
            pressures:
                (bar) array containing the pressures of the model, from lowest to highest.
            line_species:
                list of strings, denoting which line absorber species to include; must match the opacity file names.
            rayleigh_species:
                list of strings, denoting which Rayleigh scattering species to include; must match the opacity file
                names.
            continuum_opacities:
                list of strings, denoting which continuum absorber species to include; must match the opacity file
                names.
            cloud_species:
                list of strings, denoting which cloud opacity species to include; must match the opacity file names.
            opacity_mode:
                if equal to ``'c-k'``: use low-resolution mode, at
                :math:`\\lambda/\\Delta \\lambda = 1000`, with the correlated-k
                assumption. if equal to ``'lbl'``: use high-resolution mode, at
                :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line
                treatment.
            do_scat_emis:
                Will be ``False`` by default.
                If ``True`` scattering will be included in the emission spectral
                calculations. Note that this increases the runtime of pRT!
            lbl_opacity_sampling:
                Will be ``None`` by default. If integer positive value, and if
                ``mode == 'lbl'`` is ``True``, then this will only consider every
                lbl_opacity_sampling-nth point of the high-resolution opacities.
                This may be desired in the case where medium-resolution spectra are
                required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
                :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
                may make sense to carry out the calculations with lbl_opacity_sampling = 10,
                for example, and then re-binning to the final desired resolution:
                this may save time! The user should verify whether this leads to
                solutions which are identical to the re-binned results of the fiducial
                :math:`10^6` resolution. If not, this parameter must not be used.
            temperatures:
                array containing the temperatures of the model, at each pressure.
            mass_mixing_ratios:
                dictionary containing the mass mixing ratios of the model, at each pressure, for every species.
            mean_molar_masses:
                dictionary containing the mean_molar_masses of the model, at each pressure.
            wavelengths_boundaries:
                (um) list containing the min and max wavelength of the model. Can be automatically determined from
            wavelengths:
                (um) wavelengths of the model.
            transit_radii:
                transit radii of the model.
            spectral_radiosities:
                (erg.s-1.cm-2.sr-1/cm) spectral radiosities of the spectrum.
            times:
                (s) # TODO implement to calculate orbital longitudes, etc.
            **model_parameters:
                dictionary of parameters. The keys can match arguments of functions used to generate the model.
        """
        # Atmosphere/Radtrans parameters
        self.pressures = pressures
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.do_scat_emis = do_scat_emis
        self.opacity_mode = opacity_mode

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        if rayleigh_species is None:
            self.rayleigh_species = []
        else:
            self.rayleigh_species = rayleigh_species

        if continuum_opacities is None:
            self.continuum_opacities = []
        else:
            self.continuum_opacities = continuum_opacities

        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        # Spectrum generation base parameters
        self.temperatures = temperatures
        self.mass_mixing_ratios = mass_mixing_ratios
        self.mean_molar_masses = mean_molar_masses

        # Spectrum parameters
        self.wavelengths = wavelengths
        self.transit_radii = transit_radii
        self.spectral_radiosities = spectral_radiosities

        # Time-dependent parameters
        self.times = times

        # Other model parameters
        self.model_parameters = model_parameters

        # Wavelength boundaries
        if wavelengths_boundaries is None:  # calculate the optimal wavelength boundaries
            if self.wavelengths is not None:
                self.model_parameters['output_wavelengths'] = copy.deepcopy(self.wavelengths)
            elif 'output_wavelengths' not in self.model_parameters:
                raise TypeError(f"missing required argument "
                                f"'wavelengths_boundaries', add this argument to manually set the boundaries or "
                                f"add keyword argument 'output_wavelengths' to set the boundaries automatically")

            # Calculate relative velocity to take Doppler shift into account
            if 'relative_velocities' in self.model_parameters \
                    or 'orbital_longitudes' in self.model_parameters \
                    or 'orbital_phases' in self.model_parameters \
                    or 'system_observer_radial_velocities' in self.model_parameters \
                    or 'planet_rest_frame_shift' in self.model_parameters:
                self.model_parameters['relative_velocities'], \
                    self.model_parameters['planet_max_radial_orbital_velocity'], \
                    self.model_parameters['orbital_longitudes'] = \
                    self.__calculate_relative_velocities_wrap(
                        calculate_max_radial_orbital_velocity_function=self.calculate_max_radial_orbital_velocity,
                        calculate_relative_velocities_function=self.calculate_relative_velocities,
                        **self.model_parameters
                    )
            else:
                self.model_parameters['relative_velocities'] = np.zeros(1)

            wavelengths_boundaries = self.get_optimal_wavelength_boundaries()

        self.wavelengths_boundaries = wavelengths_boundaries

    @staticmethod
    def __calculate_relative_velocities_wrap(calculate_max_radial_orbital_velocity_function,
                                             calculate_relative_velocities_function, **kwargs):
        """Calculate the relative velocities using the provided functions."""
        if 'planet_max_radial_orbital_velocity' in kwargs:
            if kwargs['planet_max_radial_orbital_velocity'] is None:
                planet_max_radial_orbital_velocity = calculate_max_radial_orbital_velocity_function(
                    **kwargs
                )
            else:
                planet_max_radial_orbital_velocity = kwargs['planet_max_radial_orbital_velocity']
        else:
            if 'is_orbiting' in kwargs:
                if kwargs['is_orbiting']:
                    planet_max_radial_orbital_velocity = calculate_max_radial_orbital_velocity_function(
                        **kwargs
                    )
                else:
                    planet_max_radial_orbital_velocity = 0.0
            else:  # assuming that the planet is orbiting
                try:
                    planet_max_radial_orbital_velocity = calculate_max_radial_orbital_velocity_function(
                        **kwargs
                    )
                except TypeError as msg:
                    raise TypeError(
                        str(msg) + '\n'
                        + f"This error was raised because the modelled object was assumed to be orbiting "
                          f"and some related model parameters were missing.\n"
                          f"If the modelled object is not orbiting, add key 'is_orbiting' to the model parameters "
                          f"and set its value to False"
                    )

        kwargs['planet_max_radial_orbital_velocity'] = planet_max_radial_orbital_velocity

        if 'orbital_phases' in kwargs and 'orbital_longitudes' not in kwargs:
            if kwargs['orbital_phases'] is not None:
                orbital_longitudes = np.rad2deg(2 * np.pi * kwargs['orbital_phases'])
            else:
                orbital_longitudes = np.zeros(1)
        elif 'orbital_longitudes' in kwargs:
            orbital_longitudes = kwargs['orbital_longitudes']
        else:
            if 'is_orbiting' in kwargs:
                if kwargs['is_orbiting']:
                    warnings.warn(f"Modelled object is orbiting but no orbital position information were provided, "
                                  f"assumed an orbital longitude of 0; "
                                  f"add key 'orbital_longitudes' or 'orbital_phases' to model parameters "
                                  f"to dismiss this warning")

            orbital_longitudes = np.zeros(1)

        kwargs['orbital_longitudes'] = orbital_longitudes

        relative_velocities = calculate_relative_velocities_function(
            **kwargs
        )

        return relative_velocities, planet_max_radial_orbital_velocity, orbital_longitudes

    @staticmethod
    def calculate_mass_mixing_ratios(pressures, **kwargs):
        """Template for mass mixing ratio profile function.
        Here, generate iso-abundant mass mixing ratios profiles.

        Args:
            pressures: (bar) pressures of the temperature profile
            **kwargs: other parameters needed to generate the temperature profile

        Returns:
            A 1D-array containing the temperatures as a function of pressures
        """
        return {
            species: mass_mixing_ratio * np.ones(np.size(pressures))
            for species, mass_mixing_ratio in kwargs['imposed_mass_mixing_ratios'].items()
        }

    @staticmethod
    def calculate_max_radial_orbital_velocity(star_mass, semi_major_axis, **kwargs):
        """
        Calculate the max radial orbital velocity.

        Args:
            star_mass: (g) mass of the star
            semi_major_axis: (cm) orbit semi major axis
            **kwargs: used to store unnecessary parameters

        Returns:
            (cm.s-1) the planet max radial orbital velocity
        """
        return Planet.calculate_orbital_velocity(
            star_mass=star_mass,
            semi_major_axis=semi_major_axis
        )

    @staticmethod
    def calculate_mean_molar_masses(mass_mixing_ratios, **kwargs):
        """Calculate the mean molar masses.

        Args:
            mass_mixing_ratios: dictionary of the mass mixing ratios of the model
            **kwargs: used to store unnecessary parameters

        Returns:

        """
        return calc_MMW(mass_mixing_ratios)

    @staticmethod
    def calculate_model_parameters(**kwargs):
        """Function to update model parameters.
        This function can be expanded to include anything.

        Args:
            **kwargs: parameters to update

        Returns:
            Updated parameters
        """
        if 'star_spectral_radiosities' in kwargs:
            if kwargs['star_spectral_radiosities'] is None:
                kwargs['star_spectral_radiosities'] = BaseSpectralModel.calculate_star_spectral_radiosity(
                    **kwargs
                )

        if 'planet_max_radial_orbital_velocity' in kwargs \
                or 'relative_velocities' in kwargs \
                or 'orbital_longitudes' in kwargs \
                or 'orbital_phases' in kwargs \
                or 'system_observer_radial_velocities' in kwargs \
                or 'planet_rest_frame_shift' in kwargs:
            kwargs['relative_velocities'], \
                kwargs['planet_max_radial_orbital_velocity'], \
                kwargs['orbital_longitudes'] = \
                BaseSpectralModel.__calculate_relative_velocities_wrap(
                    calculate_max_radial_orbital_velocity_function=
                    BaseSpectralModel.calculate_max_radial_orbital_velocity,
                    calculate_relative_velocities_function=
                    BaseSpectralModel.calculate_relative_velocities,
                    **kwargs
                )

        if 'line_species' in kwargs:
            del kwargs['line_species']

        if 'pressures' in kwargs:
            del kwargs['pressures']

        if 'wavelengths' in kwargs:
            del kwargs['wavelengths']

        return kwargs

    @staticmethod
    def calculate_relative_velocities(orbital_longitudes, planet_orbital_inclination=90.0,
                                      planet_max_radial_orbital_velocity=None, system_observer_radial_velocities=0.0,
                                      planet_rest_frame_shift=0.0, **kwargs):
        if planet_max_radial_orbital_velocity is None:
            planet_max_radial_orbital_velocity = BaseSpectralModel.calculate_max_radial_orbital_velocity(
                **kwargs
            )

        if -sys.float_info.min < planet_max_radial_orbital_velocity < sys.float_info.min:
            relative_velocities = 0.0
        else:
            relative_velocities = Planet.calculate_planet_radial_velocity(
                planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
                planet_orbital_inclination=planet_orbital_inclination,
                orbital_longitude=orbital_longitudes
            )

        relative_velocities += system_observer_radial_velocities + planet_rest_frame_shift  # planet + system velocity

        return relative_velocities

    @staticmethod
    def calculate_spectral_parameters(temperature_profile_function, mass_mixing_ratios_function,
                                      mean_molar_masses_function, spectral_parameters_function, **kwargs):
        """Calculate the temperature profile, the mass mixing ratios, the mean molar masses and other parameters
        required for spectral calculation.

        This function define how these parameters are calculated and how they are combined.

        Args:
            temperature_profile_function:
            mass_mixing_ratios_function:
            mean_molar_masses_function:
            spectral_parameters_function:
            **kwargs:

        Returns:

        """
        temperatures = temperature_profile_function(
            **kwargs
        )

        mass_mixing_ratios = mass_mixing_ratios_function(
            **kwargs
        )

        mean_molar_masses = mean_molar_masses_function(
            mass_mixing_ratios=mass_mixing_ratios,
            **kwargs
        )

        model_parameters = spectral_parameters_function(
            **kwargs
        )

        return temperatures, mass_mixing_ratios, mean_molar_masses, model_parameters

    @staticmethod
    def calculate_spectral_radiosity_spectrum(radtrans: Radtrans, temperatures, mass_mixing_ratios,
                                              planet_surface_gravity, mean_molar_mass, star_spectral_radiosities=None,
                                              star_effective_temperature=None, cloud_pressure=None, cloud_sigma=None,
                                              cloud_particle_radii=None,
                                              **kwargs):
        """Wrapper of Radtrans.calc_flux that output wavelengths in um and spectral radiosity in erg.s-1.cm-2.sr-1/cm.
        # TODO move to Radtrans or outside of object
        Args:
            radtrans:
            temperatures:
            mass_mixing_ratios:
            planet_surface_gravity:
            mean_molar_mass:
            star_effective_temperature:
            star_spectral_radiosities:
            cloud_pressure:
            cloud_sigma:
            cloud_particle_radii:

        Returns:

        """
        # Calculate the spectrum
        # TODO units in native calc_flux units for more performances?
        if star_spectral_radiosities is not None:
            star_spectral_radiosities = BaseSpectralModel.radiosity_erg_cm2radiosity_erg_hz(
                star_spectral_radiosities, nc.c / radtrans.freq  # Hz to cm
            )

        radtrans.calc_flux(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=planet_surface_gravity,
            mmw=mean_molar_mass,
            Tstar=star_effective_temperature,
            Pcloud=cloud_pressure,
            stellar_intensity=star_spectral_radiosities,
            sigma_lnorm=cloud_sigma,
            radius=cloud_particle_radii
            # **kwargs  # TODO add kwargs once arguments names are made unambiguous
        )

        # Transform the outputs into the units of our data
        spectral_radiosity = BaseSpectralModel.radiosity_erg_hz2radiosity_erg_cm(radtrans.flux, radtrans.freq)
        wavelengths = BaseSpectralModel.hz2um(radtrans.freq)

        return wavelengths, spectral_radiosity

    @staticmethod
    def calculate_star_spectral_radiosity(wavelengths, star_effective_temperature, star_radius, semi_major_axis,
                                          **kwargs):
        star_data = get_PHOENIX_spec(star_effective_temperature)

        star_radiosities = star_data[:, 1]
        star_wavelengths = star_data[:, 0] * 1e4  # cm to um

        star_radiosities = fr.rebin_spectrum(star_wavelengths, star_radiosities, wavelengths)
        star_radiosities *= (star_radius / semi_major_axis) ** 2 / np.pi

        star_radiosities = BaseSpectralModel.radiosity_erg_hz2radiosity_erg_cm(
            star_radiosities, BaseSpectralModel.um2hz(wavelengths)
        )

        return star_radiosities

    @staticmethod
    def calculate_temperature_profile(pressures, **kwargs):
        """Template for temperature profile function.
        Here, generate an isothermal temperature profile.

        Args:
            pressures: (bar) pressures of the temperature profile
            **kwargs: other parameters needed to generate the temperature profile

        Returns:
            A 1D-array containing the temperatures as a function of pressures
        """
        return np.ones(np.size(pressures)) * kwargs['temperature']

    @staticmethod
    def calculate_transit_spectrum(radtrans: Radtrans, temperatures, mass_mixing_ratios, mean_molar_masses,
                                   planet_surface_gravity, reference_pressure, planet_radius, cloud_pressure=None,
                                   **kwargs):
        """Wrapper of Radtrans.calc_transm that output wavelengths in um and transit radius in cm.
        # TODO move to Radtrans or outside of object

        Args:
            radtrans:
            temperatures:
            mass_mixing_ratios:
            planet_surface_gravity:
            mean_molar_masses:
            reference_pressure:
            planet_radius:
            cloud_pressure:

        Returns:

        """
        # Calculate the spectrum
        radtrans.calc_transm(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=planet_surface_gravity,
            mmw=mean_molar_masses,
            P0_bar=reference_pressure,
            R_pl=planet_radius,
            Pcloud=cloud_pressure,
            # **kwargs  # TODO add kwargs once arguments names are made unambiguous
        )

        # Convert into more useful units
        planet_transit_radius = copy.copy(radtrans.transm_rad)
        wavelengths = BaseSpectralModel.hz2um(radtrans.freq)

        return wavelengths, planet_transit_radius

    @staticmethod
    def convolve(input_wavelengths, input_spectrum, new_resolving_power, **kwargs):
        """Convolve a spectrum to a new resolving power.
        The spectrum is convolved using a Gaussian filter with a standard deviation ~R_in/R_new input wavelengths bins.
        The spectrum must have a constant resolving power as a function of wavelength.
        The input resolving power is given by:
            lambda / Delta_lambda
        where lambda is the center of a wavelength bin and Delta_lambda the difference between the edges of the bin.

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        # Compute resolving power of the model
        # In petitRADTRANS, the wavelength grid is log-spaced, so the resolution is constant as a function of wavelength
        model_resolving_power = np.mean(
            (input_wavelengths[1:] + input_wavelengths[:-1]) / (2 * np.diff(input_wavelengths))
        )

        # Calculate the sigma to be used in the gauss filter in units of input wavelength bins
        # Delta lambda of resolution element is the FWHM of the instrument's LSF (here: a gaussian)
        sigma_lsf_gauss_filter = model_resolving_power / new_resolving_power / (2 * np.sqrt(2 * np.log(2)))

        convolved_spectrum = gaussian_filter1d(
            input=input_spectrum,
            sigma=sigma_lsf_gauss_filter,
            mode='reflect'
        )

        return convolved_spectrum

    @staticmethod
    def cnvl2(y, convolution_filter):
        # TODO [WIP] implement sliding convolution
        yc = np.zeros(y.size + convolution_filter.shape[-1] - 1)
        yp = copy.copy(yc)
        filterc = np.zeros((yc.size, convolution_filter.shape[-1]))
        yp[:y.size] = y
        filterc[int(convolution_filter.shape[-1] / 2):y.size + int(convolution_filter.shape[-1] / 2), :] = \
            convolution_filter
        filterc[y.size + int(convolution_filter.shape[-1] / 2):, :] = convolution_filter[-1]
        filterc[:int(convolution_filter.shape[-1] / 2), :] = convolution_filter[0]

        for i, yy in enumerate(yp):
            for j, g in enumerate(filterc[i, :np.min((i, filterc.shape[-1]))]):
                yc[i] += yp[i - j] * g

        return yc[int(convolution_filter.shape[-1] / 2):y.size + int(convolution_filter.shape[-1] / 2)]

    def get_instrument_model(self, wavelengths, spectrum,
                             relative_velocities=None, shift=False, convolve=False, rebin=False):
        if shift:
            if 'relative_velocities' not in self.model_parameters:
                raise TypeError(f"missing required parameter 'relative_velocities' for shifting")

            # Pop relative velocities outside of model parameters to prevent multiple arguments definition
            relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
            del self.model_parameters['relative_velocities']

            if relative_velocities is None:
                relative_velocities = copy.deepcopy(relative_velocities_tmp)

            wavelengths = self.shift_wavelengths(
                wavelengths_rest=wavelengths,
                relative_velocities=relative_velocities,
                **self.model_parameters
            )

            # Put relative velocities back into model parameters
            self.model_parameters['relative_velocities'] = relative_velocities_tmp

        if convolve:
            if np.ndim(wavelengths) <= 1:
                spectrum = self.convolve(
                    input_wavelengths=wavelengths,
                    input_spectrum=spectrum,
                    **self.model_parameters
                )
            else:
                spectrum = self.convolve(
                    input_wavelengths=wavelengths[0],  # assuming Doppler shifting doesn't change the resolving power
                    input_spectrum=spectrum,
                    **self.model_parameters
                )

        if rebin:
            if np.ndim(wavelengths) <= 1:
                wavelengths_tmp, spectrum = self.rebin_spectrum(
                    input_wavelengths=wavelengths,
                    input_spectrum=spectrum,
                    **self.model_parameters
                )

                if spectrum.dtype == 'O' and np.ndim(spectrum) >= 2:
                    spectrum = np.moveaxis(spectrum, 0, 1)

            elif np.ndim(wavelengths) == 2:
                spectrum_tmp = []
                wavelengths_tmp = None

                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = self.rebin_spectrum(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum,
                        **self.model_parameters
                    )

                spectrum = np.array(spectrum_tmp)

                if np.ndim(spectrum) == 3 or spectrum.dtype == 'O':
                    spectrum = np.moveaxis(spectrum, 0, 1)
                elif np.ndim(spectrum) > 3:
                    raise ValueError(f"spectrum must have at most 3 dimensions, but has {np.ndim(spectrum)}")
            else:
                raise ValueError(f"argument 'wavelength' must have at most 2 dimensions, "
                                 f"but has {np.ndim(wavelengths)}")

            wavelengths = wavelengths_tmp

        return wavelengths, spectrum

    def get_optimal_wavelength_boundaries(self, output_wavelengths=None, relative_velocities=None):
        """Return the optimal wavelength boundaries for rebin on output wavelengths.
        This minimises the number of wavelengths to load and over which to calculate the spectra.
        Doppler shifting is also taken into account.

        The SpectralModel must have in its model_parameters keys:
            -  'output_wavelengths': (um) the wavelengths to rebin to
        # TODO complete docstring

        The SpectralModel can have in its model_parameters keys:
            - 'relative_velocities' (cm.s-1) the velocities of the source relative to the observer, in that case the
                wavelength range is increased to take into account Doppler shifting

        Returns:
            rebin_required_interval: (um) the optimal wavelengths boundaries for the spectrum
        """
        if output_wavelengths is None:
            output_wavelengths = self.model_parameters['output_wavelengths']

        if relative_velocities is None and 'relative_velocities' in self.model_parameters:
            relative_velocities = self.model_parameters['relative_velocities']

        # Re-bin requirement is an interval half a bin larger than re-binning interval
        if hasattr(output_wavelengths, 'dtype'):
            if output_wavelengths.dtype != 'O':
                wavelengths_flat = output_wavelengths.flatten()
            else:
                wavelengths_flat = np.concatenate(output_wavelengths)
        else:
            wavelengths_flat = np.concatenate(output_wavelengths)

        if np.ndim(wavelengths_flat) > 1:
            wavelengths_flat = np.concatenate(wavelengths_flat)

        rebin_required_interval = [
            wavelengths_flat[0]
            - (wavelengths_flat[1] - wavelengths_flat[0]) / 2,
            wavelengths_flat[-1]
            + (wavelengths_flat[-1] - wavelengths_flat[-2]) / 2,
        ]

        # Take Doppler shifting into account
        rebin_required_interval_shifted = copy.copy(rebin_required_interval)

        if relative_velocities is not None:
            if 'relative_velocities' in self.model_parameters:
                relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
                del self.model_parameters['relative_velocities']  # tmp rm parameters to prevent multiple argument def
            else:
                relative_velocities_tmp = None

            rebin_required_interval_shifted[0] = self.shift_wavelengths(
                wavelengths_rest=np.array([rebin_required_interval[0]]),
                relative_velocities=np.array([
                    -np.max(relative_velocities)
                ]),
                **self.model_parameters
            )[0][0]

            rebin_required_interval_shifted[1] = self.shift_wavelengths(
                wavelengths_rest=np.array([rebin_required_interval[1]]),
                relative_velocities=np.array([
                    -np.min(relative_velocities)
                ]),
                **self.model_parameters
            )[0][0]

            if relative_velocities_tmp is not None:
                self.model_parameters['relative_velocities'] = relative_velocities_tmp

        # Ensure that non-shifted spectrum can still be re-binned
        rebin_required_interval[0] = np.min((rebin_required_interval_shifted[0], rebin_required_interval[0]))
        rebin_required_interval[1] = np.max((rebin_required_interval_shifted[1], rebin_required_interval[1]))

        # Satisfy re-bin requirement by increasing the range by the smallest possible significant value
        rebin_required_interval[0] -= 10 ** (np.floor(np.log10(rebin_required_interval[0])) - sys.float_info.dig)
        rebin_required_interval[1] += 10 ** (np.floor(np.log10(rebin_required_interval[1])) - sys.float_info.dig)

        return rebin_required_interval

    def get_radtrans(self):
        """Return the Radtrans object corresponding to this SpectrumModel."""
        return self.init_radtrans(
            wavelengths_boundaries=self.wavelengths_boundaries,
            pressures=self.pressures,
            line_species=self.line_species,
            rayleigh_species=self.rayleigh_species,
            continuum_opacities=self.continuum_opacities,
            cloud_species=self.cloud_species,
            opacity_mode=self.opacity_mode,
            do_scat_emis=self.do_scat_emis,
            lbl_opacity_sampling=self.lbl_opacity_sampling
        )

    def get_reduced_spectrum(self, spectrum, **kwargs):
        return self.pipeline(spectrum, **kwargs)

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None, **kwargs):
        if pressures is None:
            pressures = self.pressures

        if wavelengths is None:
            wavelengths = self.wavelengths

        return self.calculate_spectral_parameters(
            temperature_profile_function=self.calculate_temperature_profile,
            mass_mixing_ratios_function=self.calculate_mass_mixing_ratios,
            mean_molar_masses_function=self.calculate_mean_molar_masses,
            spectral_parameters_function=self.calculate_model_parameters,
            pressures=pressures,
            wavelengths=wavelengths,
            **kwargs
        )

    def get_spectral_radiosity_spectrum_model(self, radtrans: Radtrans, parameters):
        self.wavelengths, self.spectral_radiosities = self.calculate_spectral_radiosity_spectrum(
            radtrans=radtrans,
            temperatures=self.temperatures,
            mass_mixing_ratios=self.mass_mixing_ratios,
            mean_molar_mass=self.mean_molar_masses,
            **parameters
        )

        return self.wavelengths, self.spectral_radiosities

    def get_spectrum_model(self, radtrans: Radtrans, mode='emission', parameters=None, update_parameters=False,
                           deformation_matrix=None, noise_matrix=None,
                           scale=False, shift=False, convolve=False, rebin=False, reduce=False):
        if parameters is None:
            parameters = self.model_parameters

        if update_parameters:
            self.update_spectral_calculation_parameters(
                radtrans=radtrans,
                **parameters
            )

            self.model_parameters['mode'] = mode
            self.model_parameters['deformation_matrix'] = deformation_matrix
            self.model_parameters['noise_matrix'] = noise_matrix
            self.model_parameters['scale'] = scale
            self.model_parameters['shift'] = shift
            self.model_parameters['convolve'] = convolve
            self.model_parameters['rebin'] = rebin
            self.model_parameters['reduce'] = reduce

            parameters = copy.deepcopy(self.model_parameters)

        if mode == 'emission':
            self.wavelengths, self.spectral_radiosities = self.get_spectral_radiosity_spectrum_model(
                radtrans=radtrans,
                parameters=parameters
            )
            spectrum = copy.copy(self.spectral_radiosities)
        elif mode == 'transmission':
            self.wavelengths, self.transit_radii = self.get_transit_spectrum_model(
                radtrans=radtrans,
                parameters=parameters
            )
            spectrum = copy.copy(self.transit_radii)
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

        wavelengths = copy.copy(self.wavelengths)

        wavelengths, spectrum = self.get_instrument_model(
            wavelengths=wavelengths,
            spectrum=spectrum,
            shift=shift,
            convolve=convolve,
            rebin=rebin
        )

        if scale:
            if mode == 'emission':  # shift the star spectrum as well for scaling
                if 'star_observed_spectral_radiosities' not in parameters:
                    missing = []

                    if 'star_spectral_radiosities' not in parameters:
                        missing.append('star_spectral_radiosities')

                    if shift:
                        if 'system_observer_radial_velocities' not in parameters:
                            if 'relative_velocities' in parameters:
                                parameters['system_observer_radial_velocities'] = \
                                    np.zeros(parameters['relative_velocities'].shape)
                            else:
                                missing.append('system_observer_radial_velocities')

                    if len(missing) > 0:
                        joint = "', '".join(missing)

                        raise TypeError(f"missing {len(missing)} parameters for scaling: '{joint}'")

                    _, parameters['star_observed_spectral_radiosities'] = self.get_instrument_model(
                        wavelengths=copy.copy(self.wavelengths),
                        spectrum=parameters['star_spectral_radiosities'],
                        relative_velocities=parameters['system_observer_radial_velocities'],
                        shift=shift,
                        convolve=convolve,
                        rebin=rebin
                    )

                    if update_parameters:
                        self.model_parameters['star_observed_spectral_radiosities'] = \
                            copy.deepcopy(parameters['star_observed_spectral_radiosities'])

            spectrum = self.scale_spectrum(
                spectrum=spectrum,
                **parameters
            )

        if deformation_matrix is not None:
            spectrum *= deformation_matrix

        if noise_matrix is not None:
            spectrum += noise_matrix

        if reduce:
            spectrum, parameters['reduction_matrix'], parameters['reduced_uncertainties'] = \
                self.get_reduced_spectrum(
                    spectrum=spectrum,
                    wavelengths=wavelengths,
                    **parameters
                )
        else:
            parameters['reduction_matrix'] = np.ones(spectrum.shape)

            if 'data_uncertainties' in parameters:
                parameters['reduced_uncertainties'] = \
                    copy.deepcopy(parameters['data_uncertainties'])
            else:
                parameters['reduced_uncertainties'] = None

        if update_parameters:
            self.model_parameters['reduction_matrix'] = parameters['reduction_matrix']
            self.model_parameters['reduced_uncertainties'] = parameters['reduced_uncertainties']

        return wavelengths, spectrum

    def get_transit_spectrum_model(self, radtrans: Radtrans, parameters):
        self.wavelengths, self.transit_radii = self.calculate_transit_spectrum(
            radtrans=radtrans,
            temperatures=self.temperatures,
            mass_mixing_ratios=self.mass_mixing_ratios,
            mean_molar_masses=self.mean_molar_masses,
            **parameters
        )

        return self.wavelengths, self.transit_radii

    @staticmethod
    def hz2um(frequency):
        return nc.c / frequency * 1e4  # cm to um

    @staticmethod
    def init_radtrans(wavelengths_boundaries, pressures,
                      line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                      opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1):
        print('Generating atmosphere...')

        radtrans = Radtrans(
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            cloud_species=cloud_species,
            wlen_bords_micron=wavelengths_boundaries,
            mode=opacity_mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling,
            pressures=pressures
        )

        return radtrans

    def init_retrieval(self, radtrans: Radtrans, data, data_wavelengths, data_uncertainties, retrieval_directory,
                       retrieved_parameters, model_parameters=None, retrieval_name='retrieval',
                       mode='emission', update_parameters=False, deformation_matrix=None, noise_matrix=None,
                       scale=False, shift=False, convolve=False, rebin=False, reduce=False,
                       run_mode='retrieval', amr=False, scattering=False, distribution='lognormal', pressures=None,
                       write_out_spec_sample=False, dataset_name='data', **kwargs):
        if pressures is None:
            pressures = copy.copy(self.pressures)

        if model_parameters is None:
            model_parameters = copy.deepcopy(self.model_parameters)

        retrieval_configuration = RetrievalConfig(
            retrieval_name=retrieval_name,
            run_mode=run_mode,
            AMR=amr,
            scattering=scattering,  # scattering is automatically included for transmission spectra
            distribution=distribution,
            pressures=pressures,
            write_out_spec_sample=write_out_spec_sample  # TODO unused parameter?
        )

        # Retrieved parameters
        if isinstance(retrieved_parameters, dict):
            retrieved_parameters = RetrievalParameter.from_dict(retrieved_parameters)

        for parameter in retrieved_parameters:
            if not hasattr(parameter, 'prior_function'):
                raise AttributeError(
                    f"'{type(parameter)}' object has no attribute 'prior_function': "
                    f"usage of dictionary or a '{type(RetrievalParameter)}' instance is recommended"
                )

            retrieval_configuration.add_parameter(
                name=parameter.name,
                free=True,
                value=None,
                transform_prior_cube_coordinate=parameter.prior_function
            )

        # Fixed parameters
        retrieved_parameters_names = [retrieved_parameter.name for retrieved_parameter in retrieved_parameters]

        for parameter in model_parameters:
            if parameter not in retrieved_parameters_names:
                retrieval_configuration.add_parameter(
                    name=parameter,
                    free=False,
                    value=model_parameters[parameter],
                    transform_prior_cube_coordinate=None
                )

        # Remove masked values if necessary
        if hasattr(data, 'mask'):
            data, data_uncertainties, data_mask = BaseSpectralModel.remove_mask(
                data=data,
                data_uncertainties=data_uncertainties
            )
        else:
            data_mask = fill_object(copy.deepcopy(data), False)

        # Set model generating function
        def model_generating_function(prt_object, parameters, pt_plot_mode=None, AMR=False):
            # TODO AMR in lowercase
            # A special function is needed due to the specificity of the Retrieval object
            return self.retrieval_model_generating_function(
                prt_object=prt_object,
                parameters=parameters,
                pt_plot_mode=pt_plot_mode,
                AMR=AMR,
                spectrum_model=self,
                mode=mode,
                update_parameters=update_parameters,
                deformation_matrix=deformation_matrix,
                noise_matrix=noise_matrix,
                scale=scale,
                shift=shift,
                convolve=convolve,
                rebin=rebin,
                reduce=reduce
            )

        # Set Data object
        retrieval_configuration.add_data(
            name=dataset_name,
            path=None,
            model_generating_function=model_generating_function,
            pRT_object=radtrans,
            wlen=data_wavelengths,
            flux=data,
            flux_error=data_uncertainties,
            mask=data_mask
        )

        retrieval = Retrieval(
            run_definition=retrieval_configuration,
            output_dir=retrieval_directory,
            **kwargs
        )

        return retrieval

    @classmethod
    def load(cls, filename):
        # Generate an empty SpectralModel
        new_spectrum_model = cls(pressures=None, wavelengths_boundaries=[0.0, 0.0])

        # Update the SpectralModel attributes from the file
        with h5py.File(filename, 'r') as f:
            new_spectrum_model.__dict__ = hdf52dict(f)

        return new_spectrum_model

    @staticmethod
    def pipeline(spectrum, **kwargs):
        """Simplistic pipeline model. Do nothing.
        To be updated when initializing an instance of retrieval model.

        Args:
            spectrum: a spectrum

        Returns:
            spectrum: the spectrum reduced by the pipeline
        """
        return spectrum

    @staticmethod
    def radiosity_erg_cm2radiosity_erg_hz(radiosity_erg_cm, wavelength):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2.sr-1/Hz at a given wavelength.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_hz = radiosity_erg_cm * d[cm]/d[Hz]
            => radiosity_erg_hz = radiosity_erg_cm * wavelength**2 / c

        Args:
            radiosity_erg_cm: (erg.s-1.cm-2.sr-1/cm)
            wavelength: (cm)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_cm * wavelength ** 2 / nc.c

    @staticmethod
    def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
        """
        Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.
        Steps:
            [cm] = c[cm.s-1] / [Hz]
            => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
            => d[cm]/d[Hz] = c / [Hz]**2
            => d[Hz]/d[cm] = [Hz]**2 / c
            integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
            radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
            => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

        Args:
            radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
            frequency: (Hz)

        Returns:
            (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
        """
        return radiosity_erg_hz * frequency ** 2 / nc.c

    @staticmethod
    def rebin_spectrum(input_wavelengths, input_spectrum, output_wavelengths, **kwargs):
        if np.ndim(output_wavelengths) <= 1 and isinstance(output_wavelengths, np.ndarray):
            return output_wavelengths, fr.rebin_spectrum(input_wavelengths, input_spectrum, output_wavelengths)
        else:
            if (np.ndim(output_wavelengths) == 2 and isinstance(output_wavelengths, np.ndarray)) \
                    or hasattr(output_wavelengths, '__iter__'):
                spectra = []
                lengths = []

                for wavelengths in output_wavelengths:
                    spectra.append(fr.rebin_spectrum(input_wavelengths, input_spectrum, wavelengths))
                    lengths.append(spectra[-1].size)

                if np.all(np.array(lengths) == lengths[0]):
                    spectra = np.array(spectra)
                else:
                    spectra = np.array(spectra, dtype=object)

                return output_wavelengths, spectra
            else:
                raise ValueError(f"parameter 'output_wavelengths' must have at most 2 dimensions, "
                                 f"but has {np.ndim(output_wavelengths)}")

    @staticmethod
    def remove_mask(data, data_uncertainties):
        print('Taking care of mask...')
        data_ = []
        error_ = []
        mask_ = copy.copy(data.mask)
        lengths = []

        for i in range(data.shape[0]):
            data_.append([])
            error_.append([])

            for j in range(data.shape[1]):
                data_[i].append(np.array(
                    data[i, j, ~mask_[i, j, :]]
                ))
                error_[i].append(np.array(data_uncertainties[i, j, ~mask_[i, j, :]]))
                lengths.append(data_[i][j].size)

        # Handle jagged arrays
        if np.all(np.asarray(lengths) == lengths[0]):
            data_ = np.asarray(data_)
            error_ = np.asarray(error_)
        else:
            print("Array is jagged, generating object array...")
            data_ = np.asarray(data_, dtype=object)
            error_ = np.asarray(error_, dtype=object)

        return data_, error_, mask_

    @staticmethod
    def retrieval_model_generating_function(prt_object: Radtrans, parameters, pt_plot_mode=None, AMR=False,
                                            spectrum_model=None, mode='emission', update_parameters=False,
                                            deformation_matrix=None, noise_matrix=None,
                                            scale=False, shift=False, convolve=False, rebin=False, reduce=False):
        # TODO Change model generating function template to not include pt_plot_mode
        # Convert from Parameter object to dictionary
        p = copy.deepcopy(parameters)  # copy to avoid over-writing

        for key, value in p.items():
            if hasattr(value, 'value'):
                p[key] = p[key].value

        # Put retrieved species into imposed mass mixing ratio
        imposed_mass_mixing_ratios = {}

        for species in prt_object.line_species:
            if species in p:
                if species == 'CO_36':
                    imposed_mass_mixing_ratios[species] = 10 ** p[species] \
                                                          * np.ones(prt_object.press.shape)
                else:
                    spec = species.split('_R_')[
                        0]  # deal with the naming scheme for binned down opacities (see below)
                    # spec = spec.split('_', 1)[0]
                    imposed_mass_mixing_ratios[spec] = 10 ** p[species] \
                        * np.ones(prt_object.press.shape)

                del p[species]

        p['imposed_mass_mixing_ratios'] = imposed_mass_mixing_ratios

        return spectrum_model.get_spectrum_model(
            radtrans=prt_object,
            mode=mode,
            parameters=p,
            update_parameters=update_parameters,
            deformation_matrix=deformation_matrix,
            noise_matrix=noise_matrix,
            scale=scale,
            shift=shift,
            convolve=convolve,
            rebin=rebin,
            reduce=reduce
        )

    @staticmethod
    def run_retrieval(retrieval: Retrieval, n_live_points=100, resume=False, sampling_efficiency=0.8,
                      const_efficiency_mode=False, log_z_convergence=0.5, n_iter_before_update=50, max_iter=0,
                      save=True, filename='retrieval_parameters', rank=None, **kwargs):
        if save:
            parameter_dict = {}  # copy.deepcopy(retrieval.__dict__)  # TODO fix issues with objects not stored in HDF5

            for arg_name, arg_value in kwargs.items():
                parameter_dict[arg_name] = arg_value

            for argument, value in locals().items():
                if argument in parameter_dict and argument != 'self' and argument != 'kwargs':
                    del parameter_dict[argument]

            if rank is None or rank == 0:
                BaseSpectralModel.save_parameters(
                    file=os.path.join(retrieval.output_dir, filename + '.h5'),
                    n_live_points=n_live_points,
                    const_efficiency_mode=const_efficiency_mode,
                    log_z_convergence=log_z_convergence,
                    n_iter_before_update=n_iter_before_update,
                    max_iter=max_iter,
                    **parameter_dict
                )

        retrieval.run(
            sampling_efficiency=sampling_efficiency,
            const_efficiency_mode=const_efficiency_mode,
            n_live_points=n_live_points,
            log_z_convergence=log_z_convergence,
            n_iter_before_update=n_iter_before_update,
            max_iter=max_iter,
            resume=resume,
            **kwargs
        )

    def save(self, file):
        self.save_parameters(
            file=file,
            **self.__dict__
        )

    @staticmethod
    def save_parameters(file, **kwargs):
        with h5py.File(file, 'w') as f:
            dict2hdf5(
                dictionary=kwargs,  # TODO units as attributes of dataset
                hdf5_file=f
            )
            f.create_dataset(
                name='units',
                data='pressures are in bar, radiosities are in erg.s-1.cm-2.sr-1/cm, wavelengths are in um, '
                     'otherwise all other units are in CGS'
            )

    @staticmethod
    def scale_spectrum(spectrum, star_radius, planet_radius=None, star_observed_spectral_radiosities=None,
                       mode='emission', **kwargs):
        if mode == 'emission':
            if planet_radius is None or star_observed_spectral_radiosities is None:
                missing = []

                if planet_radius is None:
                    missing.append('planet_radius')

                if star_observed_spectral_radiosities is None:
                    missing.append('star_spectral_radiosities')

                joint = "', '".join(missing)

                raise TypeError(f"missing {len(missing)} positional arguments: '{joint}'")

            return 1 + spectrum / star_observed_spectral_radiosities * (planet_radius / star_radius) ** 2
        elif mode == 'transmission':
            return 1 - (spectrum / star_radius) ** 2
        else:
            raise ValueError(f"mode must be 'emission' or 'transmission', not '{mode}'")

    @staticmethod
    def shift_wavelengths(wavelengths_rest, relative_velocities, **kwargs):
        wavelengths_shift = np.zeros((relative_velocities.size, wavelengths_rest.size))

        for i, relative_velocity in enumerate(relative_velocities):
            wavelengths_shift[i] = doppler_shift(wavelengths_rest, relative_velocity)

        return wavelengths_shift

    @staticmethod
    def um2hz(wavelength):
        return nc.c / (wavelength * 1e-4)  # um to cm

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **kwargs):
        self.temperatures, self.mass_mixing_ratios, self.mean_molar_masses, self.model_parameters = \
            self.get_spectral_calculation_parameters(
                pressures=radtrans.press * 1e-6,  # cgs to bar
                wavelengths=BaseSpectralModel.hz2um(radtrans.freq),
                **kwargs
            )


class SpectralModel(BaseSpectralModel):
    default_line_species = [
        'CH4_main_iso',
        'CO_all_iso',
        'CO2_main_iso',
        'H2O_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'TiO_all_iso_exo',
        'VO'
    ]
    default_rayleigh_species = [
        'H2',
        'He'
    ]
    default_continuum_opacities = [
        'H2-H2',
        'H2-He'
    ]

    def __init__(self, pressures,
                 line_species=None, rayleigh_species=None, continuum_opacities=None, cloud_species=None,
                 opacity_mode='lbl', do_scat_emis=True, lbl_opacity_sampling=1,
                 temperatures=None, mass_mixing_ratios=None, mean_molar_masses=None,
                 wavelengths_boundaries=None, wavelengths=None, transit_radii=None, spectral_radiosities=None,
                 times=None, **model_parameters):
        super().__init__(
            wavelengths_boundaries=wavelengths_boundaries,
            pressures=pressures,
            temperatures=temperatures,
            mass_mixing_ratios=mass_mixing_ratios,
            mean_molar_masses=mean_molar_masses,
            line_species=line_species,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            cloud_species=cloud_species,
            opacity_mode=opacity_mode,
            do_scat_emis=do_scat_emis,
            lbl_opacity_sampling=lbl_opacity_sampling,
            wavelengths=wavelengths,
            transit_radii=transit_radii,
            spectral_radiosities=spectral_radiosities,
            times=times,
            **model_parameters
        )

    @staticmethod
    def __calculate_metallicity_wrap(metallicity=None, log10_metallicity=None,
                                     planet_mass=None, planet_radius=None, planet_surface_gravity=None,
                                     star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                     verbose=False, **kwargs):
        if log10_metallicity is None:
            if metallicity is None:
                if verbose:
                    print(f"log10 metallicity set to None, calculating it using scaled metallicity...")

                if planet_mass is None:
                    if planet_radius is None or planet_surface_gravity is None:
                        raise ValueError(f"both planet radius ({planet_radius}) "
                                         f"and surface gravity ({planet_surface_gravity}) "
                                         f"are required to calculate planet mass")
                    elif planet_radius <= 0:
                        raise ValueError(f"cannot calculate planet mass from surface gravity with a radius <= 0")

                    planet_mass = Planet.surface_gravity2mass(
                        surface_gravity=planet_surface_gravity,
                        radius=planet_radius
                    )[0]

                metallicity = SpectralModel.calculate_scaled_metallicity(
                    planet_mass=planet_mass,
                    star_metallicity=star_metallicity,
                    atmospheric_mixing=atmospheric_mixing,
                    alpha=alpha,
                    beta=beta
                )

            if metallicity <= 0:
                metallicity = sys.float_info.min

            log10_metallicity = np.log10(metallicity)

        return log10_metallicity, metallicity, planet_mass, star_metallicity, atmospheric_mixing, alpha, beta

    @staticmethod
    def _calculate_equilibrium_mass_mixing_ratios(pressures, temperatures, co_ratio, log10_metallicity,
                                                  line_species, included_line_species,
                                                  carbon_pressure_quench=None, imposed_mass_mixing_ratios=None):
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm  # import is here because it is long to load TODO add a load_data function to the module instead?
        if imposed_mass_mixing_ratios is None:
            imposed_mass_mixing_ratios = {}

        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(log10_metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * log10_metallicity
        else:
            log10_metallicities = log10_metallicity

        equilibrium_mass_mixing_ratios = pm.interpol_abundances(
            COs_goal_in=co_ratios,
            FEHs_goal_in=log10_metallicities,
            temps_goal_in=temperatures,
            pressures_goal_in=pressures,
            Pquench_carbon=carbon_pressure_quench
        )

        # Check imposed mass mixing ratios keys
        for key in imposed_mass_mixing_ratios:
            if key not in line_species and key not in equilibrium_mass_mixing_ratios:
                raise KeyError(f"key '{key}' not in retrieved species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_mixing_ratios = {}

        if included_line_species == 'all':
            included_line_species = []

            for line_species_name in line_species:
                included_line_species.append(line_species_name.split('_', 1)[0])

        for key in equilibrium_mass_mixing_ratios:
            found = False

            # Set line species mass mixing ratios into to their imposed one
            for line_species_name_ in line_species:
                if line_species_name_ == 'CO_36':  # CO_36 special case
                    if line_species_name_ in imposed_mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios[line_species_name_] = imposed_mass_mixing_ratios[line_species_name_]

                    continue

                # Correct for line species name to match pRT chemistry name
                line_species_name = line_species_name_.split('_', 1)[0]

                if line_species_name == 'C2H2':  # C2H2 special case
                    line_species_name += ',acetylene'

                if key == line_species_name:
                    if key not in included_line_species:
                        # Species not included, set mass mixing ratio to 0
                        mass_mixing_ratios[line_species_name] = np.zeros(np.shape(temperatures))
                    elif line_species_name_ in imposed_mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios[line_species_name_] = imposed_mass_mixing_ratios[line_species_name_]
                    else:
                        # Use calculated mass mixing ratio
                        mass_mixing_ratios[line_species_name_] = equilibrium_mass_mixing_ratios[line_species_name]

                    found = True

                    break

            # Set species mass mixing ratio to their imposed one
            if not found:
                if key in imposed_mass_mixing_ratios:
                    # Use imposed mass mixing ratio
                    mass_mixing_ratios[key] = imposed_mass_mixing_ratios[key]
                else:
                    # Use calculated mass mixing ratio
                    mass_mixing_ratios[key] = equilibrium_mass_mixing_ratios[key]

        return mass_mixing_ratios

    @staticmethod
    def calculate_mass_mixing_ratios(pressures, line_species=None,
                                     included_line_species='all', temperatures=None, co_ratio=0.55,
                                     log10_metallicity=None, carbon_pressure_quench=None,
                                     imposed_mass_mixing_ratios=None, heh2_ratio=0.324324, c13c12_ratio=0.01,
                                     planet_mass=None, planet_radius=None, planet_surface_gravity=None,
                                     star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                     use_equilibrium_chemistry=False, verbose=False, **kwargs):
        """Initialize a model mass mixing ratios.
        Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
        their imposed value as long as the sum of the imposed values is lower or equal to 1. H2 and He are used as
        filling gases.
        The different possible cases are dealt with as follows:
            - Sum of imposed mass mixing ratios > 1: the mass mixing ratios are scaled down, conserving the ratio
            between them. Non-imposed mass mixing ratios are set to 0.
            - Sum of imposed mass mixing ratio of all imposed species < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.
            - Sum of imposed and non-imposed mass mixing ratios > 1: the non-imposed mass mixing ratios are scaled down,
            conserving the ratios between them. Imposed mass mixing ratios are unchanged.
            - Sum of imposed and non-imposed mass mixing ratios < 1: if equilibrium chemistry is used or if H2 and
            He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
            Otherwise, the heh2_ratio parameter is used.

        When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
        imposed value regardless of chemical equilibrium consistency.

        Args:
            pressures: (bar) pressures of the mass mixing ratios
            line_species: list of line species, required to manage naming differences between opacities and chemistry
            included_line_species: which line species of the list to include, mass mixing ratio set to 0 otherwise
            temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
            co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
            log10_metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium chemistry
            imposed_mass_mixing_ratios: imposed mass mixing ratios
            heh2_ratio: H2 over He mass mixing ratio
            c13c12_ratio: 13C over 12C mass mixing ratio in equilibrium chemistry
            planet_mass: (g) mass of the planet; if None, planet mass is calculated from planet radius and surface gravity, used to calulate metallicity
            planet_radius: (cm) radius of the planet, used to calculate the mass
            planet_surface_gravity: (cm.s-2) surface gravity of the planet, used to calculate the mass
            star_metallicity: (solar metallicity) metallicity of the planet's star, used to calulate metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere, used to calulate metallicity
            alpha: power of the mass-metallicity relation
            beta: scaling factor of the mass-metallicity relation
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module
            verbose: if True, print additional information

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        # Initialization
        mass_mixing_ratios = {}
        m_sum_imposed_species = np.zeros(np.shape(pressures))
        m_sum_species = np.zeros(np.shape(pressures))

        if line_species is None:
            line_species = []

        # Initialize imposed mass mixing ratios
        if imposed_mass_mixing_ratios is not None:
            for species, mass_mixing_ratio in imposed_mass_mixing_ratios.items():
                if np.size(mass_mixing_ratio) == 1:
                    imposed_mass_mixing_ratios[species] = np.ones(np.shape(pressures)) * mass_mixing_ratio
                elif np.size(mass_mixing_ratio) != np.size(pressures):
                    raise ValueError(f"mass mixing ratio for species '{species}' must be a scalar or an array of the"
                                     f"size of the pressure array ({np.size(pressures)}), "
                                     f"but is of size ({np.size(mass_mixing_ratio)})")
        else:
            # Nothing is imposed
            imposed_mass_mixing_ratios = {}

        # Chemical equilibrium mass mixing ratios
        if use_equilibrium_chemistry:
            # Calculate metallicity
            if log10_metallicity is None:
                if 'metallicity' in kwargs:
                    metallicity = kwargs['metallicity']
                else:
                    metallicity = None

                log10_metallicity, _, _, _, _, _, _ = SpectralModel.__calculate_metallicity_wrap(
                    log10_metallicity=log10_metallicity,
                    metallicity=metallicity,
                    planet_mass=planet_mass,
                    planet_radius=planet_radius,
                    planet_surface_gravity=planet_surface_gravity,
                    star_metallicity=star_metallicity,
                    atmospheric_mixing=atmospheric_mixing,
                    alpha=alpha,
                    beta=beta,
                    verbose=verbose
                )

            # Interpolate chemical equilibrium
            mass_mixing_ratios_equilibrium = SpectralModel._calculate_equilibrium_mass_mixing_ratios(
                pressures=pressures,
                temperatures=temperatures,
                co_ratio=co_ratio,
                log10_metallicity=log10_metallicity,
                line_species=line_species,
                included_line_species=included_line_species,
                carbon_pressure_quench=carbon_pressure_quench,
                imposed_mass_mixing_ratios=imposed_mass_mixing_ratios
            )

            # TODO more general handling of isotopologues (use smarter species names)
            if 'CO_main_iso' in line_species and 'CO_all_iso' in line_species:
                raise ValueError(f"cannot add main isotopologue and all isotopologues of CO at the same time")

            if 'CO_main_iso' not in imposed_mass_mixing_ratios and 'CO_36' not in imposed_mass_mixing_ratios:
                if 'CO_all_iso' not in line_species:
                    if 'CO_main_iso' in mass_mixing_ratios_equilibrium:
                        co_mass_mixing_ratio = copy.copy(mass_mixing_ratios_equilibrium['CO_main_iso'])
                    else:
                        co_mass_mixing_ratio = copy.copy(mass_mixing_ratios_equilibrium['CO'])

                    if 'CO_main_iso' in line_species:
                        mass_mixing_ratios_equilibrium['CO_main_iso'] = co_mass_mixing_ratio / (1 + c13c12_ratio)
                        mass_mixing_ratios_equilibrium['CO_36'] = \
                            co_mass_mixing_ratio - mass_mixing_ratios_equilibrium['CO_main_iso']
                    elif 'CO_36' in line_species:
                        mass_mixing_ratios_equilibrium['CO_36'] = co_mass_mixing_ratio / (1 + 1 / c13c12_ratio)
                        mass_mixing_ratios_equilibrium['CO'] = \
                            co_mass_mixing_ratio - mass_mixing_ratios_equilibrium['CO_36']
        else:
            mass_mixing_ratios_equilibrium = None

        # Imposed mass mixing ratios
        # Ensure that the sum of mass mixing ratios of imposed species is <= 1
        for species in imposed_mass_mixing_ratios:
            mass_mixing_ratios[species] = imposed_mass_mixing_ratios[species]
            m_sum_imposed_species += imposed_mass_mixing_ratios[species]

        for i in range(np.size(m_sum_imposed_species)):
            if m_sum_imposed_species[i] > 1:
                # TODO changing retrieved mmr might come problematic in some retrievals (retrieved value not corresponding to actual value in model)
                if verbose:
                    warnings.warn(f"sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                                  f"correcting...")

                for species in imposed_mass_mixing_ratios:
                    mass_mixing_ratios[species][i] /= m_sum_imposed_species[i]

        m_sum_imposed_species = np.sum(list(mass_mixing_ratios.values()), axis=0)

        # Get the sum of mass mixing ratios of non-imposed species
        if mass_mixing_ratios_equilibrium is None:
            # TODO this is assuming an H2-He atmosphere with line species, this could be more general
            species_list = copy.copy(line_species)
        else:
            species_list = list(mass_mixing_ratios_equilibrium.keys())

        for species in species_list:
            # Ignore the non-MMR keys coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            # Search for imposed species
            found = False

            for key in imposed_mass_mixing_ratios:
                spec = key.split('_R_')[0]  # deal with the naming scheme for binned down opacities

                if species == spec:
                    found = True

                    break

            # Only take into account non-imposed species and ignore imposed species
            if not found:
                if mass_mixing_ratios_equilibrium is None:
                    if verbose:
                        warnings.warn(
                            f"line species '{species}' initialised to {sys.float_info.min} ; "
                            f"to remove this warning set use_equilibrium_chemistry to True "
                            f"or add '{species}' and the desired mass mixing ratio to imposed_mass_mixing_ratios"
                        )

                    mass_mixing_ratios[species] = sys.float_info.min
                else:
                    mass_mixing_ratios[species] = mass_mixing_ratios_equilibrium[species]
                    m_sum_species += mass_mixing_ratios_equilibrium[species]

        # Ensure that the sum of mass mixing ratios of all species is = 1
        m_sum_total = m_sum_species + m_sum_imposed_species

        if np.any(np.logical_or(m_sum_total > 1, m_sum_total < 1)):
            # Search for H2 and He in both imposed and non-imposed species
            h2_in_imposed_mass_mixing_ratios = False
            he_in_imposed_mass_mixing_ratios = False
            h2_in_mass_mixing_ratios = False
            he_in_mass_mixing_ratios = False

            if 'H2' in imposed_mass_mixing_ratios:
                h2_in_imposed_mass_mixing_ratios = True

            if 'He' in imposed_mass_mixing_ratios:
                he_in_imposed_mass_mixing_ratios = True

            if 'H2' in mass_mixing_ratios:
                h2_in_mass_mixing_ratios = True

            if 'He' in mass_mixing_ratios:
                he_in_mass_mixing_ratios = True

            if not h2_in_mass_mixing_ratios or not he_in_mass_mixing_ratios:
                if not h2_in_mass_mixing_ratios:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))

                if not he_in_mass_mixing_ratios:
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

            for i in range(np.size(m_sum_total)):
                if m_sum_total[i] > 1:
                    if verbose:
                        warnings.warn(f"sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                                      f"is > 1, correcting...")

                    for species in mass_mixing_ratios:
                        if species not in imposed_mass_mixing_ratios:
                            if m_sum_species[i] > 0:
                                mass_mixing_ratios[species][i] = \
                                    mass_mixing_ratios[species][i] * (1 - m_sum_imposed_species[i]) / m_sum_species[i]
                            else:
                                mass_mixing_ratios[species][i] = mass_mixing_ratios[species][i] / m_sum_total[i]
                elif m_sum_total[i] == 0:
                    if verbose:
                        print(f"sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                              f"is 0")
                elif m_sum_total[i] < 1:
                    # Fill atmosphere with H2 and He
                    # TODO there might be a better filling species, N2?
                    if h2_in_imposed_mass_mixing_ratios and he_in_imposed_mass_mixing_ratios:
                        if imposed_mass_mixing_ratios['H2'][i] > 0:
                            # Use imposed He/H2 ratio
                            heh2_ratio = imposed_mass_mixing_ratios['He'][i] / imposed_mass_mixing_ratios['H2'][i]
                        else:
                            heh2_ratio = None

                    if h2_in_mass_mixing_ratios and he_in_mass_mixing_ratios:
                        # Use calculated He/H2 ratio
                        heh2_ratio = mass_mixing_ratios['He'][i] / mass_mixing_ratios['H2'][i]

                        mass_mixing_ratios['H2'][i] += (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                    else:
                        # Remove H2 and He mass mixing ratios from total for correct mass mixing ratio calculation
                        if h2_in_mass_mixing_ratios:
                            m_sum_total[i] -= mass_mixing_ratios['H2'][i]
                        elif he_in_mass_mixing_ratios:
                            m_sum_total[i] -= mass_mixing_ratios['He'][i]

                        # Use He/H2 ratio in argument
                        mass_mixing_ratios['H2'][i] = (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                else:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))
        else:
            if 'H2' not in imposed_mass_mixing_ratios:
                mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))

            if 'He' not in imposed_mass_mixing_ratios:
                mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

        return mass_mixing_ratios

    @staticmethod
    def calculate_scaled_metallicity(planet_mass, star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2):
        """Calculate the scaled metallicity of a planet.
        The relation used is a power law. Default parameters come from the source.

        Source: Mordasini et al. 2014 (https://www.aanda.org/articles/aa/pdf/2014/06/aa21479-13.pdf)

        Args:
            planet_mass: (g) mass of the planet
            star_metallicity: metallicity of the planet in solar metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere
            alpha: power of the relation
            beta: scaling factor of the relation

        Returns:
            An estimation of the planet atmospheric metallicity in solar metallicity.
        """
        return beta * (planet_mass / nc.m_jup) ** alpha * star_metallicity * atmospheric_mixing

    @staticmethod
    def calculate_spectral_parameters(temperature_profile_function, mass_mixing_ratios_function,
                                      mean_molar_masses_function, spectral_parameters_function, **kwargs):
        temperatures = temperature_profile_function(
            **kwargs
        )

        mass_mixing_ratios = mass_mixing_ratios_function(
            temperatures=temperatures,  # use the newly calculated temperature profile to obtain the mass mixing ratios
            **kwargs
        )

        # Find the mean molar mass in each layer
        mean_molar_mass = mean_molar_masses_function(
            mass_mixing_ratios=mass_mixing_ratios,
            **kwargs
        )

        model_parameters = spectral_parameters_function(
            **kwargs
        )

        return temperatures, mass_mixing_ratios, mean_molar_mass, model_parameters

    @staticmethod
    def calculate_temperature_profile(pressures, temperature_profile_mode='isothermal', temperature=None,
                                      intrinsic_temperature=None, planet_surface_gravity=None, metallicity=None,
                                      guillot_temperature_profile_gamma=0.4,
                                      guillot_temperature_profile_kappa_ir_z0=0.01, **kwargs):
        if temperature is None:
            raise TypeError(f"missing required argument 'temperature'")

        if temperature_profile_mode == 'isothermal':
            if isinstance(temperature, (float, int)):
                temperatures = np.ones(np.shape(pressures)) * temperature
            elif np.size(temperature) == np.size(pressures):
                temperatures = np.asarray(temperature)
            else:
                raise ValueError(f"could not initialize isothermal temperature profile ; "
                                 f"possible inputs are float, int, "
                                 f"or a 1-D array of the same size of parameter 'pressures' ({np.size(pressures)})")
        elif temperature_profile_mode == 'guillot':
            temperatures = guillot_metallic_temperature_profile(
                pressures=pressures,
                gamma=guillot_temperature_profile_gamma,
                surface_gravity=planet_surface_gravity,
                intrinsic_temperature=intrinsic_temperature,
                equilibrium_temperature=temperature,
                kappa_ir_z0=guillot_temperature_profile_kappa_ir_z0,
                metallicity=metallicity
            )
        else:
            raise ValueError(f"mode must be 'isothermal' or 'guillot', but was '{temperature_profile_mode}'")

        return temperatures

    def get_orbital_phases(self, phase_start, orbital_period):
        orbital_phases = Planet.get_orbital_phases(
            phase_start=phase_start,
            orbital_period=orbital_period,
            times=self.times
        )

        return orbital_phases

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None,
                                            temperature_profile_mode='isothermal',
                                            temperature=None, line_species=None,
                                            included_line_species='all',
                                            imposed_mass_mixing_ratios=None,
                                            intrinsic_temperature=None, planet_surface_gravity=None, metallicity=None,
                                            guillot_temperature_profile_gamma=0.4,
                                            guillot_temperature_profile_kappa_ir_z0=0.01,
                                            co_ratio=0.55, carbon_pressure_quench=None, heh2_ratio=0.324324,
                                            use_equilibrium_chemistry=False, **kwargs
                                            ):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            pressures:
            wavelengths:
            temperature_profile_mode:
            temperature:
            line_species:
            included_line_species:
            imposed_mass_mixing_ratios:
            intrinsic_temperature:
            planet_surface_gravity:
            metallicity:
            guillot_temperature_profile_gamma:
            guillot_temperature_profile_kappa_ir_z0:
            co_ratio:
            carbon_pressure_quench:
            heh2_ratio:
            use_equilibrium_chemistry:

        Returns:

        """
        if pressures is None:
            pressures = self.pressures

        if metallicity is not None:
            log10_metallicity = np.log10(metallicity)
        elif use_equilibrium_chemistry:
            log10_metallicity, metallicity, planet_mass, star_metallicity, atmospheric_mixing, alpha, beta = \
                self.__calculate_metallicity_wrap(
                    log10_metallicity=None,
                    metallicity=metallicity,
                    planet_surface_gravity=planet_surface_gravity,
                    **kwargs
                )
        else:
            log10_metallicity = None

        # Put this function's arguments (except self and kwargs) into the model parameters dict
        for argument, value in locals().items():
            if argument not in kwargs and argument != 'self' and argument != 'kwargs':
                # self is not a model parameter, and adding kwargs to itself must be prevented
                kwargs[argument] = value

        return self.calculate_spectral_parameters(
            temperature_profile_function=self.calculate_temperature_profile,
            mass_mixing_ratios_function=self.calculate_mass_mixing_ratios,
            mean_molar_masses_function=self.calculate_mean_molar_masses,
            spectral_parameters_function=self.calculate_model_parameters,
            **kwargs
        )

    @staticmethod
    def pipeline(spectrum, **kwargs):
        """Interface with simple_pipeline.

        Args:
            spectrum: spectrum to reduce
            **kwargs: simple_pipeline arguments

        Returns:
            The reduced spectrum, matrix, and uncertainties
        """
        # simple_pipeline interface
        if not hasattr(spectrum, 'mask'):
            spectrum = np.ma.masked_array(spectrum)

        if 'uncertainties' in kwargs:  # ensure that spectrum and uncertainties share the same mask
            if hasattr(kwargs['uncertainties'], 'mask'):
                spectrum = np.ma.masked_where(kwargs['uncertainties'].mask, spectrum)

        return simple_pipeline(spectrum=spectrum, full=True, **kwargs)

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **parameters):
        pressures = radtrans.press * 1e-6  # cgs to bar

        kwargs = {'imposed_mass_mixing_ratios': {}}

        for parameter, value in parameters.items():
            if 'log10_' in parameter and value is not None:
                kwargs[parameter.split('log10_', 1)[-1]] = 10 ** value
            else:
                if parameter == 'imposed_mass_mixing_ratios':
                    if parameter is None:
                        parameter = {}

                    for species, mass_mixing_ratios in parameters[parameter].items():
                        if species not in kwargs[parameter]:
                            kwargs[parameter][species] = copy.copy(mass_mixing_ratios)
                else:
                    kwargs[parameter] = copy.copy(value)

        self.temperatures, self.mass_mixing_ratios, self.mean_molar_masses, self.model_parameters = \
            self.get_spectral_calculation_parameters(
                pressures=pressures,
                wavelengths=BaseSpectralModel.hz2um(radtrans.freq),
                line_species=radtrans.line_species,
                **kwargs
            )

        # Adapt chemical names to line species names, as required by Retrieval
        for species in radtrans.line_species:
            spec = species.split('_', 1)[0]

            if spec in self.mass_mixing_ratios:
                if species not in self.mass_mixing_ratios and species != 'K':
                    self.mass_mixing_ratios[species] = self.mass_mixing_ratios[spec]

                if species != 'K':  # TODO fix this K special case by choosing smarter opacities names
                    del self.mass_mixing_ratios[spec]
