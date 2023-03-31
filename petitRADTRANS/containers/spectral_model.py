"""SpectralModel object and related."""

import copy
import os
import inspect
import sys
import warnings

import h5py
import numpy as np
import scipy.ndimage

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval.preparing import preparing_pipeline
from petitRADTRANS.containers.planet import Planet
#from petitRADTRANS.phoenix import get_PHOENIX_spec
from petitRADTRANS.physics import doppler_shift, guillot_metallic_temperature_profile, hz2um, \
    radiosity_erg_hz2radiosity_erg_cm, radiosity2irradiance
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.util import calc_MMW, log_prior, getMM, \
    uniform_prior, gaussian_prior, log_gaussian_prior, delta_prior
from petitRADTRANS.utils import dict2hdf5, hdf52dict, fill_object, gaussian_weights_running, rebin_spectrum, remove_mask


# TODO c-k binned directly to user-provided wavelength grid

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
    # TODO warning when changing a Radtrans parameter
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
            - is_orbiting: if True, planet_radial_velocity_amplitude is calculated and relative velocities depends on
                orbital position parameters, listed below.
            - orbital_longitudes (deg) array of orbital longitudes
            - orbital_phases: array of orbital phases
            - planet_rest_frame_velocity_shift: (cm.s-1) array of offsets to the calculated relative_velocities
            - planet_radial_velocity_amplitude: (cm.s-1) radial orbital velocity semi-amplitude of the planet (Kp)
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

        # TODO if spectrum generation parameters are not None, change functions to get them so that they return the initialised value
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

        # Velocities
        self.model_parameters['relative_velocities'], \
            self.model_parameters['system_observer_radial_velocities'], \
            self.model_parameters['planet_radial_velocity_amplitude'], \
            self.model_parameters['planet_radial_velocities'], \
            self.model_parameters['orbital_longitudes'], \
            self.model_parameters['is_orbiting'] = \
            self.__init_velocities(
                radial_velocity_amplitude_function=self.calculate_radial_velocity_amplitude,
                planet_radial_velocities_function=self.calculate_planet_radial_velocities,
                relative_velocities_function=self.calculate_relative_velocities,
                **self.model_parameters
            )

        # Wavelength boundaries
        if wavelengths_boundaries is None:  # calculate the optimal wavelength boundaries
            if self.wavelengths is not None:
                self.model_parameters['output_wavelengths'] = copy.deepcopy(self.wavelengths)
            elif 'output_wavelengths' not in self.model_parameters:
                raise TypeError(f"missing required argument "
                                f"'wavelengths_boundaries', add this argument to manually set the boundaries or "
                                f"add keyword argument 'output_wavelengths' to set the boundaries automatically")

            wavelengths_boundaries = self.get_optimal_wavelength_boundaries()

        self.wavelengths_boundaries = wavelengths_boundaries

    @staticmethod
    def __init_velocities(radial_velocity_amplitude_function, planet_radial_velocities_function,
                          relative_velocities_function,
                          relative_velocities=None, system_observer_radial_velocities=None,
                          planet_radial_velocity_amplitude=None, planet_radial_velocities=None,
                          planet_rest_frame_velocity_shift=0.0,
                          orbital_longitudes=None, orbital_phases=None, is_orbiting=None, **kwargs):
        if system_observer_radial_velocities is None:
            system_observer_radial_velocities = np.zeros(1)

        # Determine if the planet is orbiting or not from the given information
        if orbital_longitudes is not None or orbital_phases is not None:
            if orbital_longitudes is None and orbital_phases is not None:
                orbital_longitudes = np.rad2deg(2 * np.pi * orbital_phases)

            if is_orbiting is None:
                is_orbiting = True
        elif is_orbiting is None:
            is_orbiting = False

        # Handle case where orbital longitudes are required but not given
        if relative_velocities is None \
                and is_orbiting \
                and orbital_longitudes is None:
            warnings.warn(f"Modelled object is orbiting but no orbital position information were provided, "
                          f"assumed an orbital longitude of 0; "
                          f"add key 'orbital_longitudes' or 'orbital_phases' to model parameters, "
                          f"or set model parameter 'is_orbiting' to False")

            orbital_longitudes = np.zeros(1)

        # Calculate relative velocities if needed
        if relative_velocities is None:
            relative_velocities, planet_radial_velocities, planet_radial_velocity_amplitude = \
                BaseSpectralModel._calculate_relative_velocities_wrap(
                    radial_velocity_amplitude_function=radial_velocity_amplitude_function,
                    planet_radial_velocities_function=planet_radial_velocities_function,
                    relative_velocities_function=relative_velocities_function,
                    system_observer_radial_velocities=system_observer_radial_velocities,
                    planet_rest_frame_velocity_shift=planet_rest_frame_velocity_shift,
                    is_orbiting=is_orbiting,
                    orbital_longitudes=orbital_longitudes,
                    planet_radial_velocity_amplitude=planet_radial_velocity_amplitude,
                    **kwargs
                )

        return relative_velocities, system_observer_radial_velocities, planet_radial_velocity_amplitude, \
            planet_radial_velocities, orbital_longitudes, is_orbiting

    @staticmethod
    def __convolve_wrap(wavelengths, convolve_function, spectrum, **kwargs):
        if np.ndim(wavelengths) <= 1:
            spectrum = convolve_function(
                input_wavelengths=wavelengths,
                input_spectrum=spectrum,
                **kwargs
            )
        else:
            spectrum = np.array([convolve_function(
                input_wavelengths=wavelength,
                input_spectrum=spectrum[i],
                **kwargs
            ) for i, wavelength in enumerate(wavelengths)])

        return spectrum

    @staticmethod
    def __rebin_wrap(wavelengths, spectrum, rebin_spectrum_function, **kwargs):
        if np.ndim(wavelengths) <= 1:
            wavelengths_tmp, spectrum = rebin_spectrum_function(
                input_wavelengths=wavelengths,
                input_spectrum=spectrum,
                **kwargs
            )

            if spectrum.dtype == 'O' and np.ndim(spectrum) >= 2:
                spectrum = np.moveaxis(spectrum, 0, 1)
        elif np.ndim(wavelengths) == 2:
            spectrum_tmp = []  # create a list to handle detectors with varying number of wavelengths
            wavelengths_tmp = None

            if np.ndim(spectrum) == 1:
                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = rebin_spectrum_function(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum,
                        **kwargs
                    )
            elif np.ndim(spectrum) == 2:
                for i, wavelength_shift in enumerate(wavelengths):
                    spectrum_tmp.append([])
                    wavelengths_tmp, spectrum_tmp[-1] = rebin_spectrum_function(
                        input_wavelengths=wavelength_shift,
                        input_spectrum=spectrum[i],
                        **kwargs
                    )
            else:
                raise ValueError(f"spectrum must have at most 2 dimensions, but has {np.ndim(spectrum)}")

            spectrum = np.array(spectrum_tmp)

            if np.ndim(spectrum) == 3 or spectrum.dtype == 'O':
                spectrum = np.moveaxis(spectrum, 0, 1)
            elif np.ndim(spectrum) > 3:
                raise ValueError(f"output spectrum must have at most 3 dimensions, but has {np.ndim(spectrum)}")
        else:
            raise ValueError(f"argument 'wavelength' must have at most 2 dimensions, "
                             f"but has {np.ndim(wavelengths)}")

        return wavelengths_tmp, spectrum

    @staticmethod
    def _calculate_relative_velocities_wrap(radial_velocity_amplitude_function, planet_radial_velocities_function,
                                            relative_velocities_function,
                                            system_observer_radial_velocities, planet_rest_frame_velocity_shift,
                                            is_orbiting=False, orbital_longitudes=None,
                                            planet_radial_velocity_amplitude=None, planet_radial_velocities=None,
                                            **kwargs):
        # Calculate planet radial velocities if needed
        if is_orbiting:
            if orbital_longitudes is None:
                raise TypeError(f"missing model parameter 'orbital_longitude' "
                                f"required to calculate planet radial velocities")

            if planet_radial_velocity_amplitude is None:
                planet_radial_velocity_amplitude = radial_velocity_amplitude_function(
                    **kwargs
                )

            planet_radial_velocities = planet_radial_velocities_function(
                orbital_longitudes=orbital_longitudes,
                planet_radial_velocity_amplitude=planet_radial_velocity_amplitude,
                **kwargs
            )
        else:
            planet_radial_velocity_amplitude = None

            if planet_radial_velocities is None:
                planet_radial_velocities = np.zeros(1)

        relative_velocities = relative_velocities_function(
            planet_radial_velocities=planet_radial_velocities,
            system_observer_radial_velocities=system_observer_radial_velocities,
            planet_rest_frame_velocity_shift=planet_rest_frame_velocity_shift,
            **kwargs
        )

        return relative_velocities, planet_radial_velocities, planet_radial_velocity_amplitude

    @staticmethod
    def _check_missing_model_parameters(model_parameters, explanation_message_=None, *args):
        missing = []

        for parameter_name in args:
            if parameter_name not in model_parameters:
                missing.append(parameter_name)

        if len(missing) >= 1:
            joint = "', '".join(missing)

            base_error_message = f"missing {len(missing)} required model parameters: '{joint}'"

            raise TypeError(BaseSpectralModel._explained_error(base_error_message, explanation_message_))

    @staticmethod
    def _check_none_model_parameters(explanation_message_=None, **kwargs):
        missing = []

        for parameter_name, value in kwargs.items():
            if value is None:
                missing.append(parameter_name)

        if len(missing) >= 1:
            joint = "', '".join(missing)

            base_error_message = f"missing {len(missing)} required model parameters: '{joint}'"

            raise TypeError(BaseSpectralModel._explained_error(base_error_message, explanation_message_))

    @staticmethod
    def _explained_error(base_error_message, explanation_message):
        # TODO deprecated in Python 3.11
        if explanation_message is None:
            explanation_message = ''
        else:
            explanation_message = '\n' + explanation_message

        return str(base_error_message) + explanation_message

    @staticmethod
    def calculate_bins_resolving_power(wavelengths):
        """Calculate the resolving power of wavelengths bins.
        The "resolving power" of the bins is defined here as:
            R = wavelengths / wavelength_steps
        This is different from the "true" (/spectral) resolving power:
            R = wavelengths / FWHM_LSF
        where FWHM_LSF is the full width half maximum of the line spread function (aka Delta lambda)

        Args:
            wavelengths: wavelengths at the center of the bins

        Returns:
            The resolving power for each bins
        """
        input_wavelengths_half_diff = np.diff(wavelengths) / 2

        # Take second to before-last element on the last axis of wavelengths
        wavelengths_shape = wavelengths.shape[-1]
        taken_wavelengths = np.take(wavelengths, np.arange(1, wavelengths_shape - 1, 1), axis=-1)
        taken_diffs_m = np.take(input_wavelengths_half_diff, np.arange(0, wavelengths_shape - 2, 1), axis=-1)
        taken_diffs_p = np.take(input_wavelengths_half_diff, np.arange(1, wavelengths_shape - 1, 1), axis=-1)

        resolving_power = taken_wavelengths / (taken_diffs_m + taken_diffs_p)
        resolving_power = np.concatenate((
            [resolving_power[0]],
            resolving_power,
            [resolving_power[-1]],
        ))

        return resolving_power

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
    def calculate_radial_velocity_amplitude(star_mass, semi_major_axis, **kwargs):
        """
        Calculate the planet orbital radial velocity semi-amplitude (aka K_p).

        Args:
            star_mass: (g) mass of the star
            semi_major_axis: (cm) orbit semi major axis
            **kwargs: used to store unnecessary parameters

        Returns:
            (cm.s-1) the planet orbital radial velocity semi-amplitude
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
    def calculate_optimal_wavelengths_boundaries(output_wavelengths, shift_wavelengths_function,
                                                 relative_velocities=None, rebin_range_margin_power=6, **kwargs):
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
            rebin_required_interval_shifted[0] = shift_wavelengths_function(
                wavelengths_rest=np.array([rebin_required_interval[0]]),
                relative_velocities=np.array([
                    -np.max(relative_velocities)
                ]),
                **kwargs
            )[0][0]

            rebin_required_interval_shifted[1] = shift_wavelengths_function(
                wavelengths_rest=np.array([rebin_required_interval[1]]),
                relative_velocities=np.array([
                    -np.min(relative_velocities)
                ]),
                **kwargs
            )[0][0]

        # Ensure that non-shifted spectrum can still be re-binned
        rebin_required_interval[0] = np.min((rebin_required_interval_shifted[0], rebin_required_interval[0]))
        rebin_required_interval[1] = np.max((rebin_required_interval_shifted[1], rebin_required_interval[1]))

        # Satisfy re-bin requirement by increasing the range by the smallest possible significant value
        rebin_required_interval[0] -= 10 ** (np.floor(np.log10(rebin_required_interval[0])) - rebin_range_margin_power)
        rebin_required_interval[1] += 10 ** (np.floor(np.log10(rebin_required_interval[1])) - rebin_range_margin_power)

        return rebin_required_interval

    @staticmethod
    def calculate_planet_radial_velocities(orbital_longitudes, planet_radial_velocity_amplitude,
                                           planet_orbital_inclination=90.0, **kwargs):
        return Planet.calculate_planet_radial_velocity(
                planet_radial_velocity_amplitude=planet_radial_velocity_amplitude,
                planet_orbital_inclination=planet_orbital_inclination,
                orbital_longitude=orbital_longitudes
            )

    @staticmethod
    def calculate_relative_velocities(planet_radial_velocities, system_observer_radial_velocities=0.0,
                                      planet_rest_frame_velocity_shift=0.0, **kwargs):
        return planet_radial_velocities + system_observer_radial_velocities + planet_rest_frame_velocity_shift

    @staticmethod
    def calculate_spectral_parameters(temperature_profile_function, mass_mixing_ratios_function,
                                      mean_molar_masses_function,
                                      star_spectral_radiosities_function, planet_star_spectral_radiances_function,
                                      radial_velocity_amplitude_function, planet_radial_velocities_function,
                                      relative_velocities_function, **kwargs):
        """Calculate the temperature profile, the mass mixing ratios, the mean molar masses and other parameters
        required for spectral calculation.

        This function define how these parameters are calculated and how they are combined.

        Args:
            temperature_profile_function:
            mass_mixing_ratios_function:
            mean_molar_masses_function:
            star_spectral_radiosities_function:
            planet_star_spectral_radiances_function:
            radial_velocity_amplitude_function:
            planet_radial_velocities_function:
            relative_velocities_function:
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

        if 'is_orbiting' in kwargs:
            # TODO probably need to have wavelengths in argument as well
            if kwargs['is_orbiting']:
                if 'star_spectral_radiosities' not in kwargs:
                    kwargs['star_spectrum_wavelengths'], kwargs['star_spectral_radiosities'] = \
                        star_spectral_radiosities_function(**kwargs)

                kwargs['planet_star_spectral_radiances'] = planet_star_spectral_radiances_function(**kwargs)

        if 'relative_velocities' in kwargs:
            kwargs['relative_velocities'], \
                kwargs['planet_radial_velocities'], \
                kwargs['planet_radial_velocity_amplitude'] = \
                BaseSpectralModel._calculate_relative_velocities_wrap(
                    radial_velocity_amplitude_function=radial_velocity_amplitude_function,
                    planet_radial_velocities_function=planet_radial_velocities_function,
                    relative_velocities_function=relative_velocities_function,
                    **kwargs
                )

        return temperatures, mass_mixing_ratios, mean_molar_masses, kwargs

    @staticmethod
    def calculate_spectral_radiosity_spectrum(radtrans: Radtrans, temperatures, mass_mixing_ratios,
                                              planet_surface_gravity, mean_molar_mass,
                                              planet_star_spectral_radiances=None,
                                              star_effective_temperature=None, star_radius=None, semi_major_axis=None,
                                              cloud_pressure=None, cloud_sigma=None, cloud_sedimentation_factor=None,
                                              cloud_particle_radii=None, cloud_particle_size_distribution='lognormal',
                                              cloud_hansen_a=None, cloud_hansen_b=None, eddy_diffusion_coefficient=None,
                                              scattering_opacity_350nm=None, scattering_opacity_coefficient=None,
                                              cloud_photospheric_optical_depth=None,
                                              cloud_photospheric_wavelengths_boundaries=None, uniform_gray_opacity=None,
                                              irradiation_geometry='dayside_ave', irradiation_inclination=0.0,
                                              planet_radius=None, system_distance=None, is_observed=False,
                                              calculate_contribution=False, add_cloud_scattering_as_absorption=False,
                                              absorption_opacity_function=None, scattering_opacity_function=None,
                                              **kwargs):
        """Wrapper of Radtrans.calc_flux that output wavelengths in um and spectral radiosity in W.m-2/um.
        Args:
            radtrans:
            temperatures:
            mass_mixing_ratios:
            planet_surface_gravity:
            mean_molar_mass:
            star_effective_temperature:
            star_radius:
            semi_major_axis:
            planet_star_spectral_radiances:
            cloud_pressure:
            cloud_sigma:
            cloud_sedimentation_factor:
            cloud_particle_radii:
            cloud_particle_size_distribution:
            cloud_hansen_a:
            cloud_hansen_b:
            eddy_diffusion_coefficient:
            scattering_opacity_350nm:
            scattering_opacity_coefficient:
            cloud_photospheric_optical_depth:
            cloud_photospheric_wavelengths_boundaries:
            uniform_gray_opacity:
            irradiation_geometry:
            irradiation_inclination:
            planet_radius:
            system_distance:
            is_observed:
            calculate_contribution:
            add_cloud_scattering_as_absorption:
            absorption_opacity_function:
            scattering_opacity_function:

        Returns:

        """
        if is_observed:
            BaseSpectralModel._check_none_model_parameters(
                explanation_message_=f"These model parameters are required to calculate "
                                     f"the observed irradiance of the planet ; "
                                     f"set model parameter 'is_observed' to False if the planet is not observed",
                planet_radius=planet_radius,
                system_distance=system_distance
            )

        # Calculate the spectrum
        radtrans.calc_flux(
            temp=temperatures,
            abunds=mass_mixing_ratios,
            gravity=planet_surface_gravity,
            mmw=mean_molar_mass,
            R_pl=planet_radius,
            sigma_lnorm=cloud_sigma,
            fsed=cloud_sedimentation_factor,
            Kzz=eddy_diffusion_coefficient,
            radius=cloud_particle_radii,
            contribution=calculate_contribution,
            gray_opacity=uniform_gray_opacity,
            Pcloud=cloud_pressure,
            kappa_zero=scattering_opacity_350nm,
            gamma_scat=scattering_opacity_coefficient,
            add_cloud_scat_as_abs=add_cloud_scattering_as_absorption,
            Tstar=star_effective_temperature,
            Rstar=star_radius,
            semimajoraxis=semi_major_axis,
            geometry=irradiation_geometry,
            theta_star=irradiation_inclination,
            hack_cloud_photospheric_tau=cloud_photospheric_optical_depth,
            dist=cloud_particle_size_distribution,
            a_hans=cloud_hansen_a,
            b_hans=cloud_hansen_b,
            stellar_intensity=planet_star_spectral_radiances,
            give_absorption_opacity=absorption_opacity_function,
            give_scattering_opacity=scattering_opacity_function,
            cloud_wlen=cloud_photospheric_wavelengths_boundaries,
            # **kwargs  # TODO add kwargs once arguments names are made unambiguous
            # TODO add the other arguments
        )

        # TODO unit change as an option?
        # Transform the outputs into the units of our data
        wavelengths = hz2um(radtrans.freq)
        spectral_radiosity = radiosity_erg_hz2radiosity_erg_cm(radtrans.flux, radtrans.freq) \
            * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

        return wavelengths, spectral_radiosity

    @staticmethod
    def calculate_star_spectral_radiosities(star_effective_temperature, **kwargs):
        # The PHOENIX data are loaded only when the module is imported
        star_data = nc.get_PHOENIX_spec(star_effective_temperature)

        star_spectral_radiosities = star_data[:, 1]
        star_spectrum_wavelengths = star_data[:, 0] * 1e4  # cm to um

        return star_spectrum_wavelengths, star_spectral_radiosities

    @staticmethod
    def calculate_planet_star_spectral_radiances(star_spectral_radiosities, star_radius, semi_major_axis,
                                                 star_spectrum_wavelengths=None, wavelengths=None, **kwargs):
        planet_star_spectral_irradiances = radiosity2irradiance(
            spectral_radiosity=star_spectral_radiosities,
            source_radius=star_radius,
            target_distance=semi_major_axis
        )  # ingoing radiosity of the star on the planet

        planet_star_spectral_radiances = planet_star_spectral_irradiances / np.pi  # W.m-2/um to W.m-2.sr-1/um

        if star_spectrum_wavelengths is not None:  # otherwise, assume that the star spectral radiosities are re-binned
            planet_star_spectral_radiances = rebin_spectrum(
                input_wavelengths=star_spectrum_wavelengths,
                input_spectrum=planet_star_spectral_radiances,
                rebinned_wavelengths=wavelengths
            )

            if planet_star_spectral_radiances < 0:
                raise ValueError(f"something went wrong")

        return planet_star_spectral_radiances

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
                                   planet_surface_gravity, reference_pressure, planet_radius,
                                   cloud_pressure=None, haze_factor=None, cloud_particle_size_distribution='lognormal',
                                   cloud_particle_radii=None, cloud_particle_log_normal_width=None,
                                   cloud_hansen_a=None, cloud_hansen_b=None,
                                   cloud_sedimentation_factor=None, eddy_diffusion_coefficient=None,
                                   scattering_opacity_350nm=None, scattering_opacity_coefficient=None,
                                   uniform_gray_opacity=None,
                                   absorption_opacity_function=None, scattering_opacity_function=None,
                                   gravity_is_variable=True, calculate_contribution=False,
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
            haze_factor:
            cloud_particle_size_distribution:
            cloud_particle_radii:
            cloud_particle_log_normal_width:
            cloud_hansen_a:
            cloud_hansen_b:
            cloud_sedimentation_factor:
            eddy_diffusion_coefficient:
            scattering_opacity_350nm:
            scattering_opacity_coefficient:
            uniform_gray_opacity:
            absorption_opacity_function:
            scattering_opacity_function:
            gravity_is_variable:
            calculate_contribution:

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
            sigma_lnorm=cloud_particle_log_normal_width,
            fsed=cloud_sedimentation_factor,
            Kzz=eddy_diffusion_coefficient,
            radius=cloud_particle_radii,
            Pcloud=cloud_pressure,
            kappa_zero=scattering_opacity_350nm,
            gamma_scat=scattering_opacity_coefficient,
            contribution=calculate_contribution,
            haze_factor=haze_factor,
            gray_opacity=uniform_gray_opacity,
            variable_gravity=gravity_is_variable,
            dist=cloud_particle_size_distribution,
            b_hans=cloud_hansen_b,
            a_hans=cloud_hansen_a,
            give_absorption_opacity=absorption_opacity_function,
            give_scattering_opacity=scattering_opacity_function
        )

        # Convert into more useful units
        planet_transit_radius = copy.copy(radtrans.transm_rad)
        wavelengths = hz2um(radtrans.freq)
        # TODO convert into radiosities by adding a star spectrum, so that it is consistent with emission?

        return wavelengths, planet_transit_radius

    @staticmethod
    def convolve(input_wavelengths, input_spectrum, new_resolving_power, **kwargs):
        """Convolve a spectrum to a new resolving power with a Gaussian filter.
        The original spectrum must have a resolving power very large compared to the target resolving power.
        The new resolving power is given in that case by:
            new_resolving_power = input_wavelengths / FWHM_LSF  (FWHM_LSF <=> "Delta_lambda" in Wikipedia)
        Therefore, the full width half maximum (FWHM) of the target line spread function (LSF) is given by:
            FWHM_LSF = input_wavelengths / new_resolving_power
        This FWHM is converted in terms of wavelength steps by:
            FWHM_LSF_Delta = FWHM_LSF / Delta_input_wavelengths
        where Delta_input_wavelengths is the difference between the edges of the bin.
        And converted into a Gaussian standard deviation by:
            sigma = FWHM_LSF_Delta / 2 * sqrt(2 * ln(2))

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        # Get input wavelengths over input wavelength steps
        input_bins_resolving_power = np.mean(BaseSpectralModel.calculate_bins_resolving_power(input_wavelengths))

        # Calculate the sigma to be used in the Gaussian filter in units of input wavelength bins
        # Conversion from FWHM to Gaussian sigma
        sigma_lsf_gauss_filter = input_bins_resolving_power / new_resolving_power / (2 * np.sqrt(2 * np.log(2)))

        convolved_spectrum = scipy.ndimage.gaussian_filter1d(
            input=input_spectrum,
            sigma=sigma_lsf_gauss_filter,
            mode='reflect'
        )

        return convolved_spectrum

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
            optimal_wavelengths_boundaries: (um) the optimal wavelengths boundaries for the spectrum
        """
        if output_wavelengths is None:
            output_wavelengths = self.model_parameters['output_wavelengths']

        if relative_velocities is None and 'relative_velocities' in self.model_parameters:
            relative_velocities = copy.deepcopy(self.model_parameters['relative_velocities'])

        # Prevent multiple definitions and give priority to the function argument
        if 'output_wavelengths' in self.model_parameters:
            output_wavelengths_tmp = copy.deepcopy(self.model_parameters['output_wavelengths'])
            del self.model_parameters['output_wavelengths']
            save_output_wavelengths = True
        else:
            output_wavelengths_tmp = None
            save_output_wavelengths = False

        if 'relative_velocities' in self.model_parameters:
            relative_velocities_tmp = copy.deepcopy(self.model_parameters['relative_velocities'])
            del self.model_parameters['relative_velocities']
            save_relative_velocities = True
        else:
            relative_velocities_tmp = None
            save_relative_velocities = False

        optimal_wavelengths_boundaries = self.calculate_optimal_wavelengths_boundaries(
            output_wavelengths=output_wavelengths,
            shift_wavelengths_function=self.shift_wavelengths,
            relative_velocities=relative_velocities,
            **self.model_parameters
        )

        if save_output_wavelengths:
            self.model_parameters['output_wavelengths'] = output_wavelengths_tmp

        if save_relative_velocities:
            self.model_parameters['relative_velocities'] = relative_velocities_tmp

        return optimal_wavelengths_boundaries

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

    def get_reprocessed_spectrum(self, spectrum, **kwargs):
        return self.pipeline(spectrum, **kwargs)

    def get_relative_velocities(self, system_observer_radial_velocities=None, planet_radial_velocity_amplitude=None,
                                planet_rest_frame_velocity_shift=None, orbital_longitudes=None, is_orbiting=None,
                                planet_orbital_inclination=None, planet_radial_velocities=None, full=False, **kwargs):
        if system_observer_radial_velocities is None:
            system_observer_radial_velocities = self.model_parameters['system_observer_radial_velocities']

        if planet_radial_velocity_amplitude is None:
            planet_radial_velocity_amplitude = self.model_parameters['planet_radial_velocity_amplitude']

        if planet_rest_frame_velocity_shift is None:
            planet_rest_frame_velocity_shift = self.model_parameters['planet_rest_frame_velocity_shift']

        if orbital_longitudes is None:
            orbital_longitudes = self.model_parameters['orbital_longitudes']

        if is_orbiting is None:
            is_orbiting = self.model_parameters['is_orbiting']

        if planet_orbital_inclination is None:
            planet_orbital_inclination = self.model_parameters['planet_orbital_inclination']

        relative_velocities, planet_radial_velocities, planet_radial_velocity_amplitude = \
            self._calculate_relative_velocities_wrap(
                radial_velocity_amplitude_function=self.calculate_radial_velocity_amplitude,
                planet_radial_velocities_function=self.calculate_planet_radial_velocities,
                relative_velocities_function=self.calculate_relative_velocities,
                system_observer_radial_velocities=system_observer_radial_velocities,
                planet_rest_frame_velocity_shift=planet_rest_frame_velocity_shift,
                is_orbiting=is_orbiting,
                orbital_longitudes=orbital_longitudes,
                planet_radial_velocity_amplitude=planet_radial_velocity_amplitude,
                planet_radial_velocities=planet_radial_velocities,
                planet_orbital_inclination=planet_orbital_inclination,
                **kwargs
            )

        if full:
            return relative_velocities, planet_radial_velocities, planet_radial_velocity_amplitude
        else:
            return relative_velocities

    def get_retrieval_velocities(self, planet_radial_velocity_amplitude_range=None,
                                 planet_rest_frame_velocity_shift_range=None,
                                 system_observer_radial_velocities=None, orbital_longitudes=None,
                                 planet_orbital_inclination=None, **kwargs):
        if system_observer_radial_velocities is None:
            system_observer_radial_velocities = self.model_parameters['system_observer_radial_velocities']

        if planet_radial_velocity_amplitude_range is None:
            planet_radial_velocity_amplitude_range = self.model_parameters['planet_radial_velocity_amplitude_range']

        if planet_rest_frame_velocity_shift_range is None:
            planet_rest_frame_velocity_shift_range = self.model_parameters['planet_rest_frame_velocity_shift_range']

        if orbital_longitudes is None:
            orbital_longitudes = self.model_parameters['orbital_longitudes']

        if planet_orbital_inclination is None:
            planet_orbital_inclination = self.model_parameters['planet_orbital_inclination']

        velocities_min = np.min(self.get_relative_velocities(
            system_observer_radial_velocities=system_observer_radial_velocities,
            planet_radial_velocity_amplitude=np.max(np.abs(planet_radial_velocity_amplitude_range)),
            planet_rest_frame_velocity_shift=np.min(planet_rest_frame_velocity_shift_range),
            orbital_longitudes=orbital_longitudes,
            is_orbiting=True,
            planet_orbital_inclination=planet_orbital_inclination,
            planet_radial_velocities=None,
            full=False,
            **kwargs
        ))

        velocities_max = np.max(self.get_relative_velocities(
            system_observer_radial_velocities=system_observer_radial_velocities,
            planet_radial_velocity_amplitude=np.max(np.abs(planet_radial_velocity_amplitude_range)),
            planet_rest_frame_velocity_shift=np.max(planet_rest_frame_velocity_shift_range),
            orbital_longitudes=orbital_longitudes,
            is_orbiting=True,
            planet_orbital_inclination=planet_orbital_inclination,
            planet_radial_velocities=None,
            full=False,
            **kwargs
        ))

        return np.array([velocities_min, velocities_max])

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None, **kwargs):
        if pressures is None:
            pressures = self.pressures  # coming from Radtrans

        if wavelengths is None:
            wavelengths = self.wavelengths  # coming from Radtrans

        functions_dict = {
            'temperature_profile_function': self.calculate_temperature_profile,
            'mass_mixing_ratios_function': self.calculate_mass_mixing_ratios,
            'mean_molar_masses_function': self.calculate_mean_molar_masses,
            'star_spectral_radiosities_function': self.calculate_star_spectral_radiosities,
            'planet_star_spectral_radiances_function': self.calculate_planet_star_spectral_radiances,
            'radial_velocity_amplitude_function': self.calculate_radial_velocity_amplitude,
            'planet_radial_velocities_function': self.calculate_planet_radial_velocities,
            'relative_velocities_function': self.calculate_relative_velocities
        }

        # Put all used functions arguments default value into the model parameters
        for function in functions_dict.values():
            signature = inspect.signature(function)

            for parameter, value in signature.parameters.items():
                if parameter not in self.__dict__.keys() \
                        and parameter not in functions_dict \
                        and parameter not in kwargs \
                        and value.default is not inspect.Parameter.empty:
                    kwargs[parameter] = value.default

        return self.calculate_spectral_parameters(
            **functions_dict,
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
                           telluric_transmittances_wavelengths=None, telluric_transmittances=None,
                           instrumental_deformations=None, noise_matrix=None,
                           scale=False, shift=False, convolve=False, rebin=False, reduce=False):
        if parameters is None:
            parameters = self.model_parameters

        if update_parameters:
            parameters['mode'] = mode
            parameters['telluric_transmittances_wavelengths'] = telluric_transmittances_wavelengths
            parameters['telluric_transmittances'] = telluric_transmittances
            parameters['instrumental_deformations'] = instrumental_deformations
            parameters['noise_matrix'] = noise_matrix
            parameters['scale'] = scale
            parameters['shift'] = shift
            parameters['convolve'] = convolve
            parameters['rebin'] = rebin
            parameters['reduce'] = reduce

            self.update_spectral_calculation_parameters(
                radtrans=radtrans,
                **parameters
            )

            parameters = copy.deepcopy(self.model_parameters)

        # Raw spectrum
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

        # Modified spectrum
        wavelengths, spectrum, star_observed_spectrum = self.modify_spectrum(
            wavelengths=wavelengths,
            spectrum=spectrum,
            shift_wavelengths_function=self.shift_wavelengths,
            convolve_function=self.convolve,
            rebin_spectrum_function=self.rebin_spectrum,
            **self.model_parameters
        )

        if star_observed_spectrum is not None:
            if 'star_observed_spectrum' not in parameters:
                parameters['star_observed_spectrum'] = star_observed_spectrum

            if update_parameters:
                self.model_parameters['star_observed_spectrum'] = star_observed_spectrum

        # if scale:
        #     spectrum = self.scale_spectrum(
        #         spectrum=spectrum,
        #         **parameters
        #     )
        #
        # if instrumental_deformations is not None:
        #     spectrum *= instrumental_deformations
        #
        # if noise_matrix is not None:
        #     spectrum += noise_matrix

        # Reduced spectrum
        if reduce:
            spectrum, parameters['reduction_matrix'], parameters['reduced_uncertainties'] = \
                self.get_reprocessed_spectrum(
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

    def get_telluric_transmittances(self, file, relative_velocities=None, rewrite=False, tellurics_resolving_power=1e6,
                                    **kwargs):
        from petitRADTRANS.cli.eso_skycalc_cli import get_tellurics_npz

        if relative_velocities is None:
            relative_velocities = self.model_parameters['relative_velocities']

        wavelengths_range = self.calculate_optimal_wavelengths_boundaries(
            output_wavelengths=self.wavelengths,
            shift_wavelengths_function=self.shift_wavelengths,
            relative_velocities=-relative_velocities,
            **kwargs
        )

        # Add one step at each end to guarantee that at least the required range is fetched for
        wavelengths_range[0] -= wavelengths_range[0] / tellurics_resolving_power
        wavelengths_range[1] += wavelengths_range[1] / tellurics_resolving_power

        return get_tellurics_npz(
            file=file,
            wavelength_range=wavelengths_range,
            rewrite=rewrite,
            **kwargs
        )

    def get_volume_mixing_ratios(self):
        volume_mixing_ratios = {}

        for species_, mass_mixing_ratio in self.mass_mixing_ratios.items():
            species = species_.split('(', 1)[0]
            volume_mixing_ratios[species_] = self.mean_molar_masses / getMM(species) * mass_mixing_ratio

        return volume_mixing_ratios

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
                       mode='emission', update_parameters=False, telluric_transmittances=None,
                       instrumental_deformations=None, noise_matrix=None,
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
            if parameter not in retrieved_parameters_names and 'log10_' + parameter not in retrieved_parameters_names:
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
                telluric_transmittances=telluric_transmittances,
                instrumental_deformations=instrumental_deformations,
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
    def modify_spectrum(wavelengths, spectrum, mode,
                        scale=False, shift=False, convolve=False, rebin=False,
                        telluric_transmittances_wavelengths=None, telluric_transmittances=None, airmass=None,
                        instrumental_deformations=None, noise_matrix=None,
                        output_wavelengths=None, relative_velocities=None, planet_radial_velocities=None,
                        star_spectrum_wavelengths=None, star_spectral_radiosities=None, star_observed_spectrum=None,
                        is_observed=False, star_radius=None, system_distance=None,
                        scale_function=None, shift_wavelengths_function=None, convolve_function=None,
                        rebin_spectrum_function=None,
                        **kwargs):
        # TODO check emission spectrum
        # TODO star observed spectrum will always be re-calculated in this configuration
        # Initialization
        if scale_function is None:
            scale_function = BaseSpectralModel.scale_spectrum

        if shift_wavelengths_function is None:
            shift_wavelengths_function = BaseSpectralModel.shift_wavelengths

        if convolve_function is None:
            convolve_function = BaseSpectralModel.convolve

        if rebin_spectrum_function is None:
            rebin_spectrum_function = BaseSpectralModel.rebin_spectrum

        if output_wavelengths is not None:
            if np.ndim(output_wavelengths) <= 1:
                output_wavelengths = np.array([output_wavelengths])

        star_spectrum = star_spectral_radiosities

        if rebin and telluric_transmittances is not None:  # TODO test if it works
            wavelengths_0 = copy.deepcopy(output_wavelengths)
        elif telluric_transmittances is not None:
            wavelengths_0 = copy.deepcopy(wavelengths)
        else:
            wavelengths_0 = None

        # Shift from the planet rest frame to the star system rest frame
        if shift and star_spectral_radiosities is not None and mode == 'emission':
            wavelengths_shift_system = shift_wavelengths_function(
                wavelengths_rest=wavelengths,
                relative_velocities=planet_radial_velocities,
                **kwargs
            )

            star_spectrum = np.zeros(wavelengths_shift_system.shape)

            # Get a star spectrum for each exposure
            for i, wavelength_shift in enumerate(wavelengths_shift_system):
                _, star_spectrum[i] = BaseSpectralModel.__rebin_wrap(
                    wavelengths=star_spectrum_wavelengths,
                    spectrum=star_spectral_radiosities,
                    output_wavelengths=wavelength_shift,
                    rebin_spectrum_function=rebin_spectrum_function,
                    **kwargs
                )

        # Calculate flux received by the observer
        if is_observed and mode == 'emission':
            # Calculate planet radiosity + star radiosity
            if star_spectrum is not None:
                spectrum = spectrum + star_spectrum

                star_observed_spectrum = radiosity2irradiance(
                    spectral_radiosity=star_spectrum,
                    source_radius=star_radius,
                    target_distance=system_distance
                )

            spectrum = radiosity2irradiance(
                spectral_radiosity=spectrum,
                source_radius=star_radius,
                target_distance=system_distance
            )
        else:
            star_observed_spectrum = star_spectrum

        # Scale the spectrum
        if scale:
            spectrum = scale_function(
                spectrum=spectrum,
                star_radius=star_radius,
                star_observed_spectrum=star_observed_spectrum,
                mode=mode,
                **kwargs
            )

        # Shift from the planet rest frame to the observer rest frame
        if shift:
            wavelengths = shift_wavelengths_function(
                wavelengths_rest=wavelengths,
                relative_velocities=relative_velocities,
                **kwargs
            )

            if np.ndim(spectrum) <= 1:  # generate 2D spectrum
                spectrum = np.tile(spectrum, (wavelengths.shape[0], 1))
        else:
            wavelengths = np.array([wavelengths])

            if np.ndim(spectrum) <= 1:  # generate 2D spectrum
                spectrum = np.array([spectrum])

        # Add telluric transmittance
        if telluric_transmittances is not None:
            telluric_transmittances_rebin = np.zeros(wavelengths.shape)

            if telluric_transmittances_wavelengths is None:
                telluric_transmittances_wavelengths = wavelengths_0

            if np.ndim(wavelengths) == 1:
                wavelengths_rebin = np.array([wavelengths])
            else:
                wavelengths_rebin = wavelengths

            # Get a telluric transmittance for each exposure
            if np.ndim(telluric_transmittances) == 1:
                for i, wavelength_shift in enumerate(wavelengths_rebin):
                    _, telluric_transmittances_rebin[i] = BaseSpectralModel.__rebin_wrap(
                        wavelengths=telluric_transmittances_wavelengths,
                        spectrum=telluric_transmittances,
                        output_wavelengths=wavelength_shift,
                        rebin_spectrum_function=rebin_spectrum_function,
                        **kwargs
                    )

                # Add airmass effect
                if airmass is not None:
                    telluric_transmittances_rebin = np.moveaxis(
                        np.ma.exp(np.ma.log(np.moveaxis(telluric_transmittances_rebin, 0, 1)) * airmass),
                        1, 0
                    )
            elif np.ndim(telluric_transmittances) == 2:
                warnings.warn(f"using 2D telluric transmittances is not recommended for precise modelling, "
                              f"consider using 1D telluric transmittances and providing for airmass at each exposure")

                for i, wavelength_shift in enumerate(wavelengths_rebin):
                    _, telluric_transmittances_rebin[i] = BaseSpectralModel.__rebin_wrap(
                        wavelengths=telluric_transmittances_wavelengths[i],
                        spectrum=telluric_transmittances[i],
                        output_wavelengths=wavelength_shift,
                        rebin_spectrum_function=rebin_spectrum_function,
                        **kwargs
                    )
            else:
                raise ValueError(f"telluric transmittances must have 1 or 2 dimensions, "
                                 f"but have {np.ndim(telluric_transmittances)}")

            spectrum = spectrum * telluric_transmittances_rebin

        # Convolve the spectrum
        if convolve:
            spectrum = BaseSpectralModel.__convolve_wrap(
                wavelengths=wavelengths,
                convolve_function=convolve_function,
                spectrum=spectrum,
                **kwargs
            )

        # Rebin the spectrum
        if rebin:
            wavelengths, spectrum = BaseSpectralModel.__rebin_wrap(
                wavelengths=wavelengths,
                spectrum=spectrum,
                output_wavelengths=output_wavelengths,
                rebin_spectrum_function=rebin_spectrum_function,
                **kwargs
            )

        # Add instrumental deformations
        if instrumental_deformations is not None:
            spectrum = spectrum * instrumental_deformations

        # Add noise
        if noise_matrix is not None:
            spectrum = spectrum + noise_matrix

        return wavelengths, spectrum, star_observed_spectrum

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
    def rebin_spectrum(input_wavelengths, input_spectrum, output_wavelengths, **kwargs):
        if np.ndim(output_wavelengths) <= 1 and isinstance(output_wavelengths, np.ndarray):
            rebinned_spectrum = rebin_spectrum(
                input_wavelengths=input_wavelengths,
                input_spectrum=input_spectrum,
                rebinned_wavelengths=output_wavelengths
            )

            return output_wavelengths, rebinned_spectrum
        else:
            if np.ndim(output_wavelengths) == 2 and isinstance(output_wavelengths, np.ndarray):
                spectra = []
                lengths = []

                for wavelengths in output_wavelengths:
                    rebinned_spectrum = rebin_spectrum(
                        input_wavelengths=input_wavelengths,
                        input_spectrum=input_spectrum,
                        rebinned_wavelengths=wavelengths
                    )

                    spectra.append(rebinned_spectrum)
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

        return remove_mask(
            data=data,
            data_uncertainties=data_uncertainties
        )

    @staticmethod
    def retrieval_model_generating_function(prt_object: Radtrans, parameters, pt_plot_mode=None, AMR=False,
                                            spectrum_model=None, mode='emission', update_parameters=False,
                                            telluric_transmittances_wavelengths=None, telluric_transmittances=None,
                                            instrumental_deformations=None, noise_matrix=None,
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
                    imposed_mass_mixing_ratios[spec] = 10 ** p[species] \
                        * np.ones(prt_object.press.shape)

                del p[species]

        # TODO add cloud MMR model(s)

        for species in p['imposed_mass_mixing_ratios']:
            if species in p and species not in prt_object.line_species:
                if species == 'CO_36':
                    imposed_mass_mixing_ratios[species] = 10 ** p[species] \
                                                          * np.ones(prt_object.press.shape)
                else:
                    spec = species.split('_R_')[
                        0]  # deal with the naming scheme for binned down opacities (see below)
                    imposed_mass_mixing_ratios[spec] = 10 ** p[species] \
                        * np.ones(prt_object.press.shape)

                del p[species]

        for key, value in imposed_mass_mixing_ratios.items():
            p['imposed_mass_mixing_ratios'][key] = value

        return spectrum_model.get_spectrum_model(
            radtrans=prt_object,
            mode=mode,
            parameters=p,
            update_parameters=update_parameters,
            telluric_transmittances_wavelengths=telluric_transmittances_wavelengths,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=instrumental_deformations,
            noise_matrix=noise_matrix,
            scale=scale,
            shift=shift,
            convolve=convolve,
            rebin=rebin,
            reduce=reduce
        )

    @staticmethod
    def run_retrieval(retrieval: Retrieval, n_live_points=100, resume=False, sampling_efficiency=0.8,
                      const_efficiency_mode=False, log_z_convergence=0.5, n_iter_before_update=50, max_iterations=0,
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
                    max_iters=max_iterations,
                    **parameter_dict
                )

        retrieval.run(
            sampling_efficiency=sampling_efficiency,
            const_efficiency_mode=const_efficiency_mode,
            n_live_points=n_live_points,
            log_z_convergence=log_z_convergence,
            n_iter_before_update=n_iter_before_update,
            max_iters=max_iterations,
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
    def scale_spectrum(spectrum, star_radius, star_observed_spectrum=None,
                       mode='emission', **kwargs):
        if mode == 'emission':
            if star_observed_spectrum is None:
                missing = []

                if star_observed_spectrum is None:
                    missing.append('star_observed_spectrum')

                joint = "', '".join(missing)

                raise TypeError(f"missing {len(missing)} positional arguments: '{joint}'")

            return 1 + spectrum / star_observed_spectrum
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
                wavelengths=hz2um(radtrans.freq),
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
    def _calculate_metallicity_wrap(planet_mass=None,
                                    star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                    verbose=False, **kwargs):
        if verbose:
            print(f"metallicity set to None, calculating it using scaled metallicity...")

        metallicity = SpectralModel.calculate_scaled_metallicity(
            planet_mass=planet_mass,
            star_metallicity=star_metallicity,
            atmospheric_mixing=atmospheric_mixing,
            alpha=alpha,
            beta=beta
        )

        if metallicity <= 0:
            warnings.warn(f"non-physical metallicity ({metallicity}), setting its value to near 0")
            metallicity = sys.float_info.min

        return metallicity

    @staticmethod
    def _calculate_equilibrium_mass_mixing_ratios(pressures, temperatures, co_ratio, metallicity,
                                                  line_species, included_line_species,
                                                  carbon_pressure_quench=None, imposed_mass_mixing_ratios=None):
        # Import is here because it is long to load TODO add a load_data function to the module instead?
        from petitRADTRANS.poor_mans_nonequ_chem import poor_mans_nonequ_chem as pm

        if imposed_mass_mixing_ratios is None:
            imposed_mass_mixing_ratios = {}

        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * metallicity
        else:
            log10_metallicities = metallicity

        log10_metallicities = np.log10(log10_metallicities)

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
    def _convolve_constant(input_wavelengths, input_spectrum, new_resolving_power, input_resolving_power=None,
                           **kwargs):
        """Convolve a spectrum to a new resolving power with a Gaussian filter.
        The original spectrum must have a resolving power very large compared to the target resolving power.
        The new resolving power is given in that case by:
            new_resolving_power = input_wavelengths / FWHM_LSF  (FWHM_LSF <=> "Delta_lambda" in Wikipedia)
        Therefore, the full width half maximum (FWHM) of the target line spread function (LSF) is given by:
            FWHM_LSF = input_wavelengths / new_resolving_power
        This FWHM is converted in terms of wavelength steps by:
            FWHM_LSF_Delta = FWHM_LSF / Delta_input_wavelengths
        where Delta_input_wavelengths is the difference between the edges of the bin.
        And converted into a Gaussian standard deviation by:
            sigma = FWHM_LSF_Delta / 2 * sqrt(2 * ln(2))

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        # Get input wavelengths over input wavelength steps
        if input_resolving_power is None:
            input_resolving_power = np.mean(SpectralModel.calculate_bins_resolving_power(input_wavelengths))

        # Calculate the sigma to be used in the Gaussian filter in units of input wavelength bins
        # Conversion from FWHM to Gaussian sigma
        sigma_lsf_gauss_filter = input_resolving_power / new_resolving_power / (2 * np.sqrt(2 * np.log(2)))

        convolved_spectrum = scipy.ndimage.gaussian_filter1d(
            input=input_spectrum,
            sigma=sigma_lsf_gauss_filter,
            mode='reflect'
        )

        return convolved_spectrum

    @staticmethod
    def _convolve_running(input_wavelengths, input_spectrum, new_resolving_power, input_resolving_power=None, **kwargs):
        """Convolve a spectrum to a new resolving power.
        The spectrum is convolved using Gaussian filters with a standard deviation
            std_dev = R_in(lambda) / R_new(lambda) * input_wavelengths_bins.
        Both the input resolving power and output resolving power can vary with wavelength.
        The input resolving power is given by:
            lambda / Delta_lambda
        where lambda is the center of a wavelength bin and Delta_lambda is the difference between the edges of the bin.

        The weights of the convolution are stored in a (N, M) matrix, with N being the size of the input, and M the size
        of the convolution kernels.
        To speed-up calculations, a matrix A of shape (N, M) is built from the inputs such as:
            A[i, :] = s[i - M/2], s[i - M/2 + 1], ..., s[i - M/2 + M],
        with s the input spectrum.
        The definition of the convolution C of s by constant weights with wavelength is:
            C[i] = sum_{j=0}^{j=M-1} s[i - M/2 + j] * weights[j].
        Thus, the convolution of s by weights at index i is:
            C[i] = sum_{j=0}^{j=M-1} A[i, j] * weights[i, j].

        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum
            input_resolving_power: if not None, skip its calculation using input_wavelengths

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        if input_resolving_power is None:
            input_resolving_power = SpectralModel.calculate_bins_resolving_power(input_wavelengths)

        sigma_lsf_gauss_filter = input_resolving_power / new_resolving_power / (2 * np.sqrt(2 * np.log(2)))
        weights = gaussian_weights_running(sigma_lsf_gauss_filter)

        input_length = weights.shape[1]
        central_index = int(input_length / 2)

        # Create a matrix
        input_matrix = np.moveaxis(
            np.array([np.roll(input_spectrum, i - central_index, axis=-1) for i in range(input_length)]),
            0,
            -1
        )

        convolved_spectrum = np.sum(input_matrix * weights, axis=-1)
        n_dims_non_wavelength = len(input_spectrum.shape[:-1])

        # Replace non-valid convolved values by non-convolved values (inaccurate but better than 'reflect' or 0 padding)
        for i in range(input_length):
            if i - central_index < 0:
                ind = np.arange(0, central_index - i, dtype=int)

                for j in range(n_dims_non_wavelength):
                    ind = np.expand_dims(ind, axis=0)

                np.put_along_axis(
                    convolved_spectrum,
                    ind,
                    np.take_along_axis(
                        input_spectrum,
                        ind,
                        axis=-1
                    ),
                    axis=-1
                )
            elif i - central_index > 0:
                ind = -np.arange(1, i - central_index + 1, dtype=int)

                for j in range(n_dims_non_wavelength):
                    ind = np.expand_dims(ind, axis=0)

                np.put_along_axis(
                    convolved_spectrum,
                    ind,
                    np.take_along_axis(
                        input_spectrum,
                        ind,
                        axis=-1
                    ),
                    axis=-1
                )

        return convolved_spectrum

    @staticmethod
    def calculate_mass_mixing_ratios(pressures, line_species=None,
                                     included_line_species='all', temperatures=None, co_ratio=0.55,
                                     metallicity=None, carbon_pressure_quench=None,
                                     imposed_mass_mixing_ratios=None, heh2_ratio=12/37, c13c12_ratio=0.01,
                                     planet_mass=None, planet_radius=None, planet_surface_gravity=None,
                                     star_metallicity=1.0, atmospheric_mixing=1.0, alpha=-0.68, beta=7.2,
                                     use_equilibrium_chemistry=False, fill_atmosphere=False, verbose=False, **kwargs):
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
            metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium
                chemistry
            imposed_mass_mixing_ratios: imposed mass mixing ratios
            heh2_ratio: H2 over He mass mixing ratio
            c13c12_ratio: 13C over 12C mass mixing ratio in equilibrium chemistry
            planet_mass: (g) mass of the planet; if None, planet mass is calculated from planet radius and surface
                gravity, used to calulate metallicity
            planet_radius: (cm) radius of the planet, used to calculate the mass
            planet_surface_gravity: (cm.s-2) surface gravity of the planet, used to calculate the mass
            star_metallicity: (solar metallicity) metallicity of the planet's star, used to calulate metallicity
            atmospheric_mixing: scaling factor [0, 1] representing how well metals are mixed in the atmosphere, used to
                calulate metallicity
            alpha: power of the mass-metallicity relation
            beta: scaling factor of the mass-metallicity relation
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module
            fill_atmosphere: if True, the atmosphere will be filled with H2 and He (using h2h2_ratio)
                if the sum of MMR is < 1 TODO use None and a dict of species instead of a flag
            verbose: if True, print additional information

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        # TODO should fill_atmosphere be True by default? Currently it means more work for the casual user.
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
        # TODO fix bug when all line species are imposed species
        if use_equilibrium_chemistry:
            # Calculate metallicity
            if metallicity is None:
                metallicity = SpectralModel._calculate_metallicity_wrap(
                    metallicity=metallicity,
                    planet_mass=planet_mass,
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
                metallicity=metallicity,
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
                    raise ValueError(f"total mass mixing ratio at pressure level {i} is 0; "
                                     f"add at least one species with non-zero imposed mass mixing ratio "
                                     f"or set equilibrium chemistry to True")
                elif m_sum_total[i] < 1:
                    # Fill atmosphere with H2 and He
                    # TODO there might be a better filling species, N2?
                    if not fill_atmosphere:
                        warnings.warn(f"the sum of mass mixing ratios at level {i} is lower than 1 ({m_sum_total[i]}). "
                                      f"Set fill_atmosphere to True to automatically fill the atmosphere "
                                      f"with H2 and He (with He MMR / H2 MMR = heh2_ratio), "
                                      f"or manually adjust the imposed mass mixing ratios")

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
                                      mean_molar_masses_function,
                                      star_spectral_radiosities_function, planet_star_spectral_radiances_function,
                                      radial_velocity_amplitude_function, planet_radial_velocities_function,
                                      relative_velocities_function,
                                      wavelengths=None, pressures=None, line_species=None,
                                      metallicity_function=None,
                                      mass2surface_gravity_function=None, surface_gravity2mass_function=None,
                                      **kwargs):
        if kwargs['planet_mass'] is None:
            kwargs['planet_mass'] = surface_gravity2mass_function(**kwargs)
        elif kwargs['planet_surface_gravity'] is None:
            kwargs['planet_surface_gravity'] = mass2surface_gravity_function(**kwargs)

        if kwargs['metallicity'] is None:
            kwargs['metallicity'] = metallicity_function(**kwargs)

        temperatures = temperature_profile_function(
            pressures=pressures,
            **kwargs
        )

        mass_mixing_ratios = mass_mixing_ratios_function(
            pressures=pressures,
            line_species=line_species,
            temperatures=temperatures,  # use the newly calculated temperature profile to obtain the mass mixing ratios
            **kwargs
        )

        # Find the mean molar mass in each layer
        mean_molar_mass = mean_molar_masses_function(
            pressures=pressures,
            mass_mixing_ratios=mass_mixing_ratios,
            **kwargs
        )

        # Calculate star radiosities
        if 'mode' in kwargs:
            if kwargs['mode'] == 'emission' and 'is_orbiting' in kwargs:
                if kwargs['is_orbiting']:
                    if 'star_spectral_radiosities' not in kwargs:
                        kwargs['star_spectrum_wavelengths'], kwargs['star_spectral_radiosities'] = \
                            star_spectral_radiosities_function(
                                **kwargs
                            )

                    kwargs['planet_star_spectral_radiances'] = planet_star_spectral_radiances_function(
                        wavelengths=wavelengths,
                        **kwargs
                    )

        if 'relative_velocities' in kwargs:
            kwargs['relative_velocities'], \
                kwargs['planet_radial_velocities'], \
                kwargs['planet_radial_velocity_amplitude'] = \
                SpectralModel._calculate_relative_velocities_wrap(
                    radial_velocity_amplitude_function=radial_velocity_amplitude_function,
                    planet_radial_velocities_function=planet_radial_velocities_function,
                    relative_velocities_function=relative_velocities_function,
                    **kwargs
                )

        return temperatures, mass_mixing_ratios, mean_molar_mass, kwargs

    @staticmethod
    def calculate_temperature_profile(pressures, temperature_profile_mode='isothermal', temperature=None,
                                      intrinsic_temperature=None, planet_surface_gravity=None, metallicity=None,
                                      guillot_temperature_profile_gamma=0.4,
                                      guillot_temperature_profile_kappa_ir_z0=0.01, **kwargs):
        SpectralModel._check_none_model_parameters(
            explanation_message_=f"Required for calculating the temperature profile",
            temperature=temperature
        )

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

    @staticmethod
    def convolve(input_wavelengths, input_spectrum, new_resolving_power, constance_tolerance=1e-6, **kwargs):
        """
        Args:
            input_wavelengths: (cm) wavelengths of the input spectrum
            input_spectrum: input spectrum
            new_resolving_power: resolving power of output spectrum
            constance_tolerance: relative tolerance on input resolving power to apply constant or running convolutions

        Returns:
            convolved_spectrum: the convolved spectrum at the new resolving power
        """
        input_resolving_powers = SpectralModel.calculate_bins_resolving_power(input_wavelengths)

        if np.allclose(input_resolving_powers, np.mean(input_resolving_powers), atol=0.0, rtol=constance_tolerance) \
                and np.size(new_resolving_power) <= 1:
            convolved_spectrum = SpectralModel._convolve_constant(
                input_wavelengths=input_wavelengths,
                input_spectrum=input_spectrum,
                new_resolving_power=new_resolving_power,
                input_resolving_power=input_resolving_powers[0],
                **kwargs
            )
        else:
            convolved_spectrum = SpectralModel._convolve_running(
                input_wavelengths=input_wavelengths,
                input_spectrum=input_spectrum,
                new_resolving_power=new_resolving_power,
                input_resolving_power=input_resolving_powers,
                **kwargs
            )

        return convolved_spectrum

    def get_orbital_phases(self, phase_start, orbital_period):
        orbital_phases = Planet.get_orbital_phases(
            phase_start=phase_start,
            orbital_period=orbital_period,
            times=self.times
        )

        return orbital_phases

    def get_spectral_calculation_parameters(self, pressures=None, wavelengths=None, line_species=None,
                                            **kwargs
                                            ):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            pressures:
            wavelengths:
            line_species:

        Returns:

        """
        if pressures is None:
            pressures = self.pressures  # coming from Radtrans

        if wavelengths is None:
            wavelengths = self.wavelengths  # coming from Radtrans

        functions_dict = {
            'temperature_profile_function': self.calculate_temperature_profile,
            'mass_mixing_ratios_function': self.calculate_mass_mixing_ratios,
            'mean_molar_masses_function': self.calculate_mean_molar_masses,
            'star_spectral_radiosities_function': self.calculate_star_spectral_radiosities,
            'planet_star_spectral_radiances_function': self.calculate_planet_star_spectral_radiances,
            'radial_velocity_amplitude_function': self.calculate_radial_velocity_amplitude,
            'planet_radial_velocities_function': self.calculate_planet_radial_velocities,
            'relative_velocities_function': self.calculate_relative_velocities,
            'metallicity_function': self._calculate_metallicity_wrap,
            'mass2surface_gravity_function': self.mass2surface_gravity,
            'surface_gravity2mass_function': self.surface_gravity2mass,
        }

        # Put all used functions arguments default value into the model parameters
        for function in functions_dict.values():
            signature = inspect.signature(function)

            for parameter, value in signature.parameters.items():
                if parameter not in self.__dict__.keys() \
                        and parameter not in functions_dict \
                        and parameter not in kwargs \
                        and value.default is not inspect.Parameter.empty:
                    kwargs[parameter] = value.default

        return self.calculate_spectral_parameters(
            pressures=pressures,
            wavelengths=wavelengths,
            line_species=line_species,
            **functions_dict,
            **kwargs
        )

    @staticmethod
    def mass2surface_gravity(planet_mass, planet_radius, verbose=False, **kwargs):
        if verbose:
            print(f"planet_surface_gravity set to None, calculating it using surface gravity and radius...")

        if planet_radius is None or planet_mass is None:
            raise ValueError(f"both planet radius ({planet_radius}) "
                             f"and planet mass ({planet_mass}) "
                             f"are required to calculate planet surface gravity")
        elif planet_radius <= 0:
            raise ValueError(f"cannot calculate surface gravity from planet mass with a radius <= 0")

        return Planet.mass2surface_gravity(
            mass=planet_mass,
            radius=planet_radius
        )[0]

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

        return preparing_pipeline(spectrum=spectrum, full=True, **kwargs)

    @staticmethod
    def surface_gravity2mass(planet_surface_gravity, planet_radius, verbose=False, **kwargs):
        if verbose:
            print(f"planet_mass set to None, calculating it using surface gravity and radius...")

        if planet_radius is None or planet_surface_gravity is None:
            raise ValueError(f"both planet radius ({planet_radius}) "
                             f"and surface gravity ({planet_surface_gravity}) "
                             f"are required to calculate planet mass")
        elif planet_radius <= 0:
            raise ValueError(f"cannot calculate planet mass from surface gravity with a radius <= 0")

        return Planet.surface_gravity2mass(
            surface_gravity=planet_surface_gravity,
            radius=planet_radius
        )[0]

    def update_spectral_calculation_parameters(self, radtrans: Radtrans, **parameters):
        pressures = radtrans.press * 1e-6  # cgs to bar

        kwargs = {'imposed_mass_mixing_ratios': {}}

        for parameter, value in parameters.items():
            if 'log10_' in parameter and value is not None:
                parameter_name = parameter.split('log10_', 1)[-1]

                if parameter_name in kwargs:
                    raise TypeError(f"got multiple values for parameter '{parameter_name}'; "
                                    f"this may be caused by "
                                    f"giving both '{parameter_name}' and e.g. 'log10_{parameter_name}'")

                kwargs[parameter.split('log10_', 1)[-1]] = 10 ** value
            else:
                if parameter == 'imposed_mass_mixing_ratios':
                    if parameter is None:
                        parameter = {}

                    for species, mass_mixing_ratios in parameters[parameter].items():
                        if species not in kwargs[parameter]:
                            kwargs[parameter][species] = copy.copy(mass_mixing_ratios)
                elif parameter in kwargs:
                    raise TypeError(f"got multiple values for parameter '{parameter}'; "
                                    f"this may be caused by "
                                    f"giving both '{parameter}' and e.g. 'log10_{parameter}'")
                else:
                    kwargs[parameter] = copy.copy(value)

        self.temperatures, self.mass_mixing_ratios, self.mean_molar_masses, self.model_parameters = \
            self.get_spectral_calculation_parameters(
                pressures=pressures,
                wavelengths=hz2um(radtrans.freq),
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
