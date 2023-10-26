import copy
import os
import pickle

import numpy as np
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.stellar_spectra.phoenix import compute_phoenix_spectrum
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.utils import calc_mmw, log_prior, uniform_prior, gaussian_prior, log_gaussian_prior, \
    delta_prior

# from petitRADTRANS.config import petitradtrans_config

# planet_models_directory = os.path_input_data.abspath(Path.home()) + os.path_input_data.sep + 'Downloads' + os.path_input_data.sep + 'tmp' #os.path_input_data.abspath(os.path_input_data.dirname(__file__) + os.path_input_data.sep + 'planet_models')
planet_models_directory = os.path.abspath(os.path.dirname(__file__) + os.path.sep + 'planet_models')  # TODO change that


# planet_models_directory = petitradtrans_config['Paths']['pRT_outputs_path']


class Param:
    """Object used only to satisfy the requirements of the retrieval module."""
    def __init__(self, value):
        self.value = value


class ParametersDict(dict):
    def __init__(self, t_int, metallicity, co_ratio, p_cloud):
        super().__init__()

        self['intrinsic_temperature'] = t_int
        self['metallicity'] = metallicity
        self['co_ratio'] = co_ratio
        self['p_cloud'] = p_cloud

    def to_str(self):
        return f"T_int = {self['intrinsic_temperature']}, [Fe/H] = {self['metallicity']}, C/O = {self['co_ratio']}, " \
               f"P_cloud = {self['p_cloud']}"


class RetrievalParameter:
    available_priors = [
        'log',
        'uniform',
        'gaussian',
        'log_gaussian',
        'delta'
    ]

    def __init__(self, name, prior_parameters, prior_type='uniform'):
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
        else:
            raise ValueError(
                f"prior type '{prior_type}' not implemented "
                f"(available prior types: {'|'.join(RetrievalParameter.available_priors)})"
            )

        self.prior_function = prior

    @classmethod
    def from_dict(cls, dictionary):
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

    def put_into_dict(self, dictionary):
        dictionary[self.name] = {
            'prior_boundaries': self.prior_parameters,
            'prior_type': self.prior_type
        }

        return dictionary


class SimplePlanet(Planet):
    def __init__(self, name, radius, surface_gravity, star_effective_temperature, star_radius, orbit_semi_major_axis,
                 reference_pressure=0.01, bond_albedo=0, equilibrium_temperature=None, mass=None):
        """

        Args:
            name: name of the planet
            radius: (cm) radius of the planet
            surface_gravity: (cm.s-2) gravity of the planet
            star_effective_temperature: (K) surface effective temperature of the star
            star_radius: (cm) mean radius of the star
            orbit_semi_major_axis: (cm) distance between the planet and the star
            reference_pressure: (bar) reference pressure for the radius and the gravity of the planet
            bond_albedo: bond albedo of the planet
        """
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            reference_gravity=surface_gravity,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reference_pressure=reference_pressure,
            equilibrium_temperature=equilibrium_temperature,
            bond_albedo=bond_albedo,
            star_radius=star_radius,
            star_effective_temperature=star_effective_temperature
        )

        if equilibrium_temperature is None:
            self.equilibrium_temperature = self.calculate_planetary_equilibrium_temperature()[0]
        else:
            self.equilibrium_temperature = equilibrium_temperature

        if mass is None:
            self.mass = self.reference_gravity2mass(self.reference_gravity, self.radius)
        else:
            self.mass = mass

    @staticmethod
    def reference_gravity2mass(surface_gravity, radius, **kwargs):
        return surface_gravity * radius ** 2 / cst.G


class SpectralModelLegacy:
    default_line_species = [
        'CH4_main_iso',
        'CO-NatAbund',
        'CO2_main_iso',
        'H2O_main_iso',
        'HCN_main_iso',
        'K',
        'Na_allard',
        'NH3_main_iso',
        'TiO-NatAbund_exo',
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

    module_dir = os.path.abspath('./')

    def __init__(self, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                 t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                 atmosphere_file=None, wavelengths=None, transit_radius=None, eclipse_depth=None,
                 spectral_radiosity=None, star_spectral_radiosity=None, opacity_mode='lbl',
                 heh2_ratio=0.324, use_equilibrium_chemistry=False,
                 temperature=None, mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        self.planet_name = planet_name
        self.wavelength_boundaries = wavelength_boundaries
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.do_scat_emis = do_scat_emis
        self.opacity_mode = opacity_mode
        self.t_int = t_int
        self.metallicity = metallicity
        self.co_ratio = co_ratio
        self.p_cloud = p_cloud

        self.kappa_ir_z0 = kappa_ir_z0
        self.gamma = gamma
        self.p_quench_c = p_quench_c
        self.haze_factor = haze_factor

        self.atmosphere_file = atmosphere_file

        self.temperature = temperature
        self.mass_fractions = mass_fractions

        self.wavelengths = wavelengths
        self.transit_radius = transit_radius
        self.eclipse_depth = eclipse_depth
        self.spectral_radiosity = spectral_radiosity
        self.star_spectral_radiosity = star_spectral_radiosity

        self.heh2_ratio = heh2_ratio
        self.use_equilibrium_chemistry = use_equilibrium_chemistry

        self.name_suffix = model_suffix

        if planet_model_file is None:
            self.planet_model_file = Planet(planet_name).get_filename()
        else:
            self.planet_model_file = planet_model_file

        if filename is None:
            self.filename = self.get_filename()

    @staticmethod
    def _init_equilibrium_chemistry(pressures, temperatures, co_ratio, log10_metallicity,
                                    line_species, included_line_species,
                                    carbon_pressure_quench=None, mass_mixing_ratios=None):
        from petitRADTRANS.chemistry import pre_calculated_chemistry as pm  # import is here because it is long to load

        if np.size(co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * co_ratio
        else:
            co_ratios = co_ratio

        if np.size(log10_metallicity) == 1:
            log10_metallicities = np.ones_like(pressures) * log10_metallicity
        else:
            log10_metallicities = log10_metallicity

        abundances = pm.interpolate_mass_fractions_chemical_table(
            co_ratios=co_ratios,
            log10_metallicities=log10_metallicities,
            temperatures=temperatures,
            pressures=pressures,
            carbon_pressure_quench=carbon_pressure_quench
        )

        # Check mass_mixing_ratios keys
        for key in mass_mixing_ratios:
            if key not in line_species and key not in abundances:
                raise KeyError(f"key '{key}' not in retrieved species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_mixing_ratios_dict = {}

        if included_line_species == 'all':
            included_line_species = copy.copy(line_species)

        for key in abundances:
            found = False

            # Set line species mass mixing ratios into to their imposed one
            for line_species_name in line_species:
                # Correct for line species name to match pRT chemistry name
                line_species_name = line_species_name.split('_', 1)[0]

                if line_species_name == 'C2H2':  # C2H2 special case
                    line_species_name += ',acetylene'

                if key == line_species_name:
                    if key not in included_line_species:
                        # Species not included, set mass mixing ratio to 0
                        mass_mixing_ratios_dict[line_species_name] = np.zeros(np.shape(temperatures))
                    elif line_species_name in mass_mixing_ratios:
                        # Use imposed mass mixing ratio
                        mass_mixing_ratios_dict[line_species_name] = 10 ** mass_mixing_ratios[line_species_name]
                    else:
                        # Use calculated mass mixing ratio
                        mass_mixing_ratios_dict[line_species_name] = abundances[line_species_name]

                    found = True

                    break

            # Set species mass mixing ratio to their imposed one
            if not found:
                if key in mass_mixing_ratios:
                    # Use imposed mass mixing ratio
                    mass_mixing_ratios_dict[key] = mass_mixing_ratios[key]
                else:
                    # Use calculated mass mixing ratio
                    mass_mixing_ratios_dict[key] = abundances[key]

        return mass_mixing_ratios_dict

    @staticmethod
    def _init_mass_mixing_ratios(pressures, line_species,
                                 included_line_species='all', temperatures=None, co_ratio=0.55, log10_metallicity=0,
                                 carbon_pressure_quench=None,
                                 imposed_mass_fractions=None, heh2_ratio=0.324324, use_equilibrium_chemistry=False):
        """Initialize a model mass mixing ratios.
        Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
        their value as much as possible.
        If the sum of mass mixing ratios of all imposed species is greater than 1, the mass mixing ratios will be scaled
        down, conserving the ratio between them. In that case, non-imposed mass mixing ratios are set to 0.
        If the sum of mass mixing ratio of all imposed species is less than 1, then if equilibrium chemistry is used or
        if H2 and He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He
        ratio. Otherwise, the heh2_ratio parameter is used.
        When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
        required value regardless of chemical equilibrium consistency.

        Args:
            pressures: (bar) pressures of the mass mixing ratios
            line_species: list of line species, required to manage naming differences between opacities and chemistry
            included_line_species: which line species of the list to include, mass mixing ratio set to 0 otherwise
            temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
            co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
            log10_metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
            carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium chemistry
            imposed_mass_fractions: imposed mass mixing ratios
            heh2_ratio: H2 over He mass mixing ratio
            use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module

        Returns:
            A dictionary containing the mass mixing ratios.
        """
        # Initialization
        mass_mixing_ratios = {}
        m_sum_imposed_species = np.zeros(np.shape(pressures))
        m_sum_species = np.zeros(np.shape(pressures))

        # Initialize imposed mass mixing ratios
        if imposed_mass_fractions is not None:
            for species, mass_mixing_ratio in imposed_mass_fractions.items():
                if np.size(mass_mixing_ratio) == 1:
                    imposed_mass_fractions[species] = np.ones(np.shape(pressures)) * mass_mixing_ratio
                elif np.size(mass_mixing_ratio) != np.size(pressures):
                    raise ValueError(f"mass mixing ratio for species '{species}' must be a scalar or an array of the"
                                     f"size of the pressure array ({np.size(pressures)}), "
                                     f"but is of size ({np.size(mass_mixing_ratio)})")
        else:
            # Nothing is imposed
            imposed_mass_fractions = {}

        # Chemical equilibrium
        if use_equilibrium_chemistry:
            mass_mixing_ratios_equilibrium = SpectralModelLegacy._init_equilibrium_chemistry(
                pressures=pressures,
                temperatures=temperatures,
                co_ratio=co_ratio,
                log10_metallicity=log10_metallicity,
                line_species=line_species,
                included_line_species=included_line_species,
                carbon_pressure_quench=carbon_pressure_quench,
                mass_mixing_ratios=imposed_mass_fractions
            )

            if imposed_mass_fractions == {}:
                imposed_mass_fractions = copy.copy(mass_mixing_ratios_equilibrium)
        else:
            mass_mixing_ratios_equilibrium = None

        # Ensure that the sum of mass mixing ratios of imposed species is <= 1
        for species in imposed_mass_fractions:
            # Ignore the non-abundances coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities
            mass_mixing_ratios[species] = imposed_mass_fractions[spec]
            m_sum_imposed_species += imposed_mass_fractions[spec]

        for i in range(np.size(m_sum_imposed_species)):
            if m_sum_imposed_species[i] > 1:
                # TODO changing retrieved mmr might come problematic in some retrievals (retrieved value not corresponding to actual value in model)
                print(f"Warning: sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                      f"correcting...")

                for species in imposed_mass_fractions:
                    mass_mixing_ratios[species][i] /= m_sum_imposed_species[i]

        m_sum_imposed_species = np.sum(list(mass_mixing_ratios.values()), axis=0)

        # Get the sum of mass mixing ratios of non-imposed species
        if mass_mixing_ratios_equilibrium is None:
            # TODO this is assuming an H2-He atmosphere with line species, this could be more general
            species_list = copy.copy(line_species)
        else:
            species_list = list(mass_mixing_ratios_equilibrium.keys())

        for species in species_list:
            # Ignore the non-abundances coming from the chemistry module
            if species == 'nabla_ad' or species == 'MMW':
                continue

            # Search for imposed species
            found = False

            for key in imposed_mass_fractions:
                spec = key.split('_R_')[0]  # deal with the naming scheme for binned down opacities

                if species == spec:
                    found = True

                    break

            # Only take into account non-imposed species and ignore imposed species
            if not found:
                mass_mixing_ratios[species] = mass_mixing_ratios_equilibrium[species]
                m_sum_species += mass_mixing_ratios_equilibrium[species]

        # Ensure that the sum of mass mixing ratios of all species is = 1
        m_sum_total = m_sum_species + m_sum_imposed_species

        if np.any(np.logical_or(m_sum_total > 1, m_sum_total < 1)):
            # Search for H2 and He in both imposed and non-imposed species
            h2_found_in_mass_mixing_ratios = False
            he_found_in_mass_mixing_ratios = False
            h2_found_in_abundances = False
            he_found_in_abundances = False

            for key in imposed_mass_fractions:
                if key == 'H2':
                    h2_found_in_mass_mixing_ratios = True
                elif key == 'He':
                    he_found_in_mass_mixing_ratios = True

            for key in mass_mixing_ratios:
                if key == 'H2':
                    h2_found_in_abundances = True
                elif key == 'He':
                    he_found_in_abundances = True

            if not h2_found_in_abundances or not he_found_in_abundances:
                if not h2_found_in_abundances:
                    mass_mixing_ratios['H2'] = np.zeros(np.shape(pressures))

                if not he_found_in_abundances:
                    mass_mixing_ratios['He'] = np.zeros(np.shape(pressures))

            for i in range(np.size(m_sum_total)):
                if m_sum_total[i] > 1:
                    print(f"Warning: sum of species mass fraction ({m_sum_species[i]} + {m_sum_imposed_species[i]}) "
                          f"is > 1, correcting...")

                    for species in mass_mixing_ratios:
                        found = False

                        for key in imposed_mass_fractions:
                            if species == key:
                                found = True

                                break

                        if not found:
                            mass_mixing_ratios[species][i] = \
                                mass_mixing_ratios[species][i] * (1 - m_sum_imposed_species[i]) / m_sum_species[i]
                elif m_sum_total[i] < 1:
                    # Fill atmosphere with H2 and He
                    # TODO there might be a better filling species, N2?
                    if h2_found_in_mass_mixing_ratios and he_found_in_mass_mixing_ratios:
                        # Use imposed He/H2 ratio
                        heh2_ratio = 10 ** imposed_mass_fractions['He'][i] / 10 ** imposed_mass_fractions['H2'][i]

                    if h2_found_in_abundances and he_found_in_abundances:
                        # Use calculated He/H2 ratio
                        heh2_ratio = mass_mixing_ratios['He'][i] / mass_mixing_ratios['H2'][i]

                        mass_mixing_ratios['H2'][i] += (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio
                    else:
                        # Remove H2 and He mass mixing ratios from total for correct mass mixing ratio calculation
                        if h2_found_in_abundances:
                            m_sum_total[i] -= mass_mixing_ratios['H2'][i]
                        elif he_found_in_abundances:
                            m_sum_total[i] -= mass_mixing_ratios['He'][i]

                        # Use He/H2 ratio in argument
                        mass_mixing_ratios['H2'][i] = (1 - m_sum_total[i]) / (1 + heh2_ratio)
                        mass_mixing_ratios['He'][i] = mass_mixing_ratios['H2'][i] * heh2_ratio

        return mass_mixing_ratios

    @staticmethod
    def _init_model(atmosphere: Radtrans, parameters: dict):
        """Initialize the temperature profile, mass mixing ratios and mean molar mass of a model.

        Args:
            atmosphere: an instance of Radtrans object
            parameters: dictionary of parameters

        Returns:
            The temperature, mass mixing ratio and mean molar mass at each pressure as 1D-arrays
        """
        pressures = atmosphere.pressures * 1e-6  # bar to cgs

        if parameters['intrinsic_temperature'].value is not None:
            temperatures = SpectralModelLegacy._init_temperature_profile_guillot(
                pressures=pressures,
                gamma=parameters['guillot_temperature_profile_gamma'].value,
                surface_gravity=10 ** parameters['log10_surface_gravity'].value,
                intrinsic_temperature=parameters['intrinsic_temperature'].value,
                equilibrium_temperature=parameters['temperature'].value,
                kappa_ir_z0=parameters['guillot_temperature_profile_kappa_ir_z0'].value,
                metallicity=10 ** parameters['log10_metallicity'].value
            )
        elif isinstance(parameters['temperature'].value, (float, int)):
            temperatures = np.ones(np.shape(atmosphere.pressures)) * parameters['temperature'].value
        elif np.size(parameters['temperature'].value) == np.size(pressures):
            temperatures = np.asarray(parameters['temperature'].value)
        else:
            raise ValueError(f"could not initialize temperature profile; "
                             f"possible inputs are float, int, "
                             f"or a 1-D array of the same size of parameter 'pressures' ({np.size(atmosphere.pressures)})")

        imposed_mass_fractions = {}

        for species in atmosphere.line_species:
            # TODO mass mixing ratio dict initialization more general
            spec = species.split('_R_')[0]  # deal with the naming scheme for binned down opacities
            # Convert from log-abundance
            imposed_mass_fractions[species] = 10 ** parameters[spec].value * np.ones_like(pressures)

        mass_mixing_ratios = SpectralModelLegacy._init_mass_mixing_ratios(
            pressures=pressures,
            line_species=atmosphere.line_species,
            included_line_species=parameters['included_line_species'].value,
            temperatures=temperatures,
            co_ratio=parameters['co_ratio'].value,
            log10_metallicity=parameters['log10_metallicity'].value,
            carbon_pressure_quench=parameters['carbon_pressure_quench'].value,
            imposed_mass_fractions=imposed_mass_fractions,
            heh2_ratio=parameters['heh2_ratio'].value,
            use_equilibrium_chemistry=parameters['use_equilibrium_chemistry'].value
        )

        # Find the mean molar mass in each layer
        mean_molar_mass = calc_mmw(mass_mixing_ratios)

        return temperatures, mass_mixing_ratios, mean_molar_mass

    @staticmethod
    def _get_parameters_dict(surface_gravity, planet_radius=None, reference_pressure=1e-2,
                             temperature=None, mass_mixing_ratios=None, cloud_pressure=None,
                             guillot_temperature_profile_gamma=0.4, guillot_temperature_profile_kappa_ir_z0=0.01,
                             included_line_species=None, intrinsic_temperature=None, heh2_ratio=0.324,
                             use_equilibrium_chemistry=False,
                             co_ratio=0.55, metallicity=1.0, carbon_pressure_quench=None,
                             star_effective_temperature=None, star_radius=None, star_spectral_radiosity=None,
                             radial_velocity_semi_amplitude=None, planet_orbital_inclination=None,
                             semi_major_axis=None,
                             rest_frame_velocity_shift=0.0, orbital_phases=None, system_observer_radial_velocities=None,
                             wavelengths_instrument=None, instrument_resolving_power=None,
                             data=None, data_uncertainties=None,
                             reduced_data=None, reduced_data_uncertainties=None, reduction_matrix=None,
                             airmass=None, telluric_transmittance=None, variable_throughput=None
                             ):
        # Conversions to log-space
        if cloud_pressure is not None:
            cloud_pressure = np.log10(cloud_pressure)

        if metallicity is not None:
            metallicity = np.log10(metallicity)

        if surface_gravity is not None:
            surface_gravity = np.log10(surface_gravity)

        # TODO expand to include all possible parameters of transm and calc_flux
        parameters = {
            'airmass': Param(airmass),
            'carbon_pressure_quench': Param(carbon_pressure_quench),
            'co_ratio': Param(co_ratio),
            'data': Param(data),
            'data_uncertainties': Param(data_uncertainties),
            'guillot_temperature_profile_gamma': Param(guillot_temperature_profile_gamma),
            'guillot_temperature_profile_kappa_ir_z0': Param(guillot_temperature_profile_kappa_ir_z0),
            'heh2_ratio': Param(heh2_ratio),
            'included_line_species': Param(included_line_species),
            'instrument_resolving_power': Param(instrument_resolving_power),
            'intrinsic_temperature': Param(intrinsic_temperature),
            'log10_cloud_pressure': Param(cloud_pressure),
            'log10_metallicity': Param(metallicity),
            'log10_surface_gravity': Param(surface_gravity),
            'orbital_phases': Param(orbital_phases),
            'radial_velocity_semi_amplitude': Param(radial_velocity_semi_amplitude),
            'planet_radius': Param(planet_radius),
            'rest_frame_velocity_shift': Param(rest_frame_velocity_shift),
            'planet_orbital_inclination': Param(planet_orbital_inclination),
            'reduced_data': Param(reduced_data),
            'reduction_matrix': Param(reduction_matrix),
            'reduced_data_uncertainties': Param(reduced_data_uncertainties),
            'reference_pressure': Param(reference_pressure),
            'semi_major_axis': Param(semi_major_axis),
            'star_effective_temperature': Param(star_effective_temperature),
            'star_radius': Param(star_radius),
            'star_spectral_radiosity': Param(star_spectral_radiosity),
            'system_observer_radial_velocities': Param(system_observer_radial_velocities),
            'telluric_transmittance': Param(telluric_transmittance),
            'temperature': Param(temperature),
            'use_equilibrium_chemistry': Param(use_equilibrium_chemistry),
            'variable_throughput': Param(variable_throughput),
            'wavelengths_instrument': Param(wavelengths_instrument),
        }

        if mass_mixing_ratios is None:
            mass_mixing_ratios = {}

        for species, mass_mixing_ratio in mass_mixing_ratios.items():
            parameters[species] = Param(np.log10(mass_mixing_ratio))

        return parameters

    @staticmethod
    def _init_temperature_profile_guillot(pressures, gamma, surface_gravity,
                                          intrinsic_temperature, equilibrium_temperature,
                                          kappa_ir_z0=None, metallicity=None):
        if metallicity is not None:
            kappa_ir = kappa_ir_z0 * metallicity
        else:
            kappa_ir = kappa_ir_z0

        temperatures = temperature_profile_function_guillot_global(
            pressures=pressures,
            infrared_mean_opacity=kappa_ir,
            gamma=gamma,
            gravities=surface_gravity,
            intrinsic_temperature=intrinsic_temperature,
            equilibrium_temperature=equilibrium_temperature
        )

        return temperatures

    @staticmethod
    def _spectral_radiosity_model(atmosphere: Radtrans, parameters: dict):
        temperatures, mass_mixing_ratios, mean_molar_mass = SpectralModelLegacy._init_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        # Calculate the spectrum
        atmosphere.calculate_flux(
            temperatures=temperatures,
            mass_fractions=mass_mixing_ratios,
            reference_gravity=10 ** parameters['log10_surface_gravity'].value,
            mean_molar_masses=mean_molar_mass,
            star_effective_temperature=parameters['star_effective_temperature'].value,
            star_radius=parameters['star_radius'].value,
            orbit_semi_major_axis=parameters['semi_major_axis'].value,
            opaque_cloud_top_pressure=10 ** parameters['log10_cloud_pressure'].value,
            # stellar_intensity=parameters['star_spectral_radiosity'].value
        )

        # Transform the outputs into the units of our data.
        planet_radiosity = SpectralModelLegacy.radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere._frequencies)
        wlen_model = cst.c / atmosphere._frequencies * 1e4  # cm to um

        return wlen_model, planet_radiosity

    @staticmethod
    def _transit_radius_model(atmosphere: Radtrans, parameters: dict):
        temperatures, mass_mixing_ratios, mean_molar_mass = SpectralModelLegacy._init_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        # Calculate the spectrum
        atmosphere.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=mass_mixing_ratios,
            reference_gravity=10 ** parameters['log10_surface_gravity'].value,
            mean_molar_masses=mean_molar_mass,
            reference_pressure=parameters['reference_pressure'].value,
            planet_radius=parameters['planet_radius'].value
        )

        # Transform the outputs into the units of our data.
        planet_transit_radius = atmosphere.transit_radii
        wavelengths = cst.c / atmosphere._frequencies * 1e4  # cm to um

        return wavelengths, planet_transit_radius

    def calculate_transit_radius(self, planet: Planet, atmosphere: Radtrans = None, pressures=None,
                                 line_species=None, rayleigh_species=None, continuum_opacities=None):
        if line_species is None:
            line_species = self.default_line_species

        if rayleigh_species is None:
            rayleigh_species = self.default_rayleigh_species

        if continuum_opacities is None:
            continuum_opacities = self.default_continuum_opacities

        if atmosphere is None:
            atmosphere = self.init_atmosphere(
                pressures=pressures,
                wlen_bords_micron=self.wavelength_boundaries,
                line_species_list=line_species,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                do_scat_emis=self.do_scat_emis,
                mode=self.opacity_mode
            )

        parameters = self.get_parameters_dict(planet)

        wavelengths, transit_radius = self._transit_radius_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        self.wavelengths = wavelengths
        self.transit_radius = transit_radius

        # Initialized afterward because we need wavelengths first!
        # TODO find a way to prevent that
        parameters['star_spectral_radiosity'] = Param(self.get_phoenix_star_spectral_radiosity(planet))

        return wavelengths, transit_radius

    def calculate_spectral_radiosity(self, planet: Planet, atmosphere: Radtrans = None, pressures=None,
                                     line_species=None, rayleigh_species=None, continuum_opacities=None):
        if line_species is None:
            line_species = self.default_line_species

        if rayleigh_species is None:
            rayleigh_species = self.default_rayleigh_species

        if continuum_opacities is None:
            continuum_opacities = self.default_continuum_opacities

        if atmosphere is None:
            atmosphere = self.init_atmosphere(
                pressures=pressures,
                wlen_bords_micron=self.wavelength_boundaries,
                line_species_list=line_species,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                do_scat_emis=self.do_scat_emis,
                mode=self.opacity_mode
            )

        parameters = self.get_parameters_dict(planet)

        wavelengths, spectral_radiosity = self._spectral_radiosity_model(
            atmosphere=atmosphere,
            parameters=parameters
        )

        self.wavelengths = wavelengths
        self.spectral_radiosity = spectral_radiosity

        # Initialized afterward because we need wavelengths first!
        # TODO find a way to prevent that
        parameters['star_spectral_radiosity'] = Param(self.get_phoenix_star_spectral_radiosity(planet))

        return wavelengths, spectral_radiosity

    def calculate_eclipse_depth(self, atmosphere: Radtrans, planet: Planet, star_radiosity_filename=None):
        if star_radiosity_filename is None:
            star_radiosity_filename = self.get_star_radiosity_filename(
                planet.star_effective_temperature, path=SpectralModelLegacy.module_dir
            )

        if not os.path.isfile(star_radiosity_filename):
            self.generate_phoenix_star_spectrum_file(star_radiosity_filename, planet.star_effective_temperature)

        data = np.loadtxt(star_radiosity_filename)
        star_wavelength = data[:, 0] * 1e6  # m to um
        star_radiosities = data[:, 1] * 1e8 * np.pi  # erg.s-1.cm-2.sr-1/A to erg.s-1.cm-2/cm

        print('Calculating eclipse depth...')
        # TODO fix stellar flux calculated multiple time if do_scat_emis is True
        wavelengths, planet_radiosity = self.calculate_emission_spectrum(atmosphere, planet)
        star_radiosities = frebin.rebin_spectrum(star_wavelength, star_radiosities, wavelengths)

        eclipse_depth = (planet_radiosity * planet.radius ** 2) / (star_radiosities * planet.star_radius ** 2)

        return wavelengths, eclipse_depth, planet_radiosity

    def calculate_emission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating emission spectrum...')

        atmosphere.calculate_flux(
            self.temperature,
            self.mass_fractions,
            planet.reference_gravity,
            self.mass_fractions['MMW'],
            star_effective_temperature=planet.star_effective_temperature,
            star_radius=planet.star_radius,
            orbit_semi_major_axis=planet.orbit_semi_major_axis,
            opaque_cloud_top_pressure=self.p_cloud
        )

        flux = self.radiosity_erg_hz2radiosity_erg_cm(atmosphere.flux, atmosphere._frequencies)
        wavelengths = cst.c / atmosphere._frequencies * 1e4  # cm to um

        return wavelengths, flux

    def calculate_transmission_spectrum(self, atmosphere: Radtrans, planet: Planet):
        print('Calculating transmission spectrum...')
        # TODO better transmission spectrum with Doppler shift, RM effect, limb-darkening effect (?)
        # Doppler shift should be low, RM effect and limb-darkening might be removed by the pipeline
        atmosphere.calculate_transit_radii(
            self.temperature,
            self.mass_fractions,
            planet.reference_gravity,
            self.mass_fractions['MMW'],
            planet_radius=planet.radius,
            reference_pressure=planet.reference_pressure,
            opaque_cloud_top_pressure=self.p_cloud,
            haze_factor=self.haze_factor,
        )

        transit_radius = (atmosphere.transit_radii / planet.star_radius) ** 2
        wavelengths = cst.c / atmosphere._frequencies * 1e4  # m to um

        return wavelengths, transit_radius

    @staticmethod
    def generate_phoenix_star_spectrum_file(star_spectrum_file, star_effective_temperature):
        stellar_spectral_radiance = compute_phoenix_spectrum(star_effective_temperature)

        # Convert the spectrum to units accepted by the ETC website
        # Don't take the first wavelength to avoid spike in convolution
        wavelength_stellar = \
            stellar_spectral_radiance[1:, 0]  # in cm
        stellar_spectral_radiance = SpectralModelLegacy.radiosity_erg_hz2radiosity_erg_cm(
            stellar_spectral_radiance[1:, 1],
            cst.c / wavelength_stellar  # cm to Hz
        )

        wavelength_stellar *= 1e-2  # cm to m
        stellar_spectral_radiance *= 1e-8 / np.pi  # erg.s-1.cm-2/cm to erg.s-1.cm-2.sr-1/A

        np.savetxt(star_spectrum_file, np.transpose((wavelength_stellar, stellar_spectral_radiance)))

    def get_filename(self):
        name = self.get_name()

        return planet_models_directory + os.path.sep + name + '.pkl'

    def get_parameters_dict(self, planet: Planet, included_line_species='all'):
        # star_spectral_radiosity = self.get_phoenix_star_spectral_radiosity(planet)
        radial_velocity_semi_amplitude = planet.calculate_orbital_velocity(
            planet.star_mass, planet.orbit_semi_major_axis
        )

        return self._get_parameters_dict(
            surface_gravity=planet.reference_gravity,
            planet_radius=planet.radius,
            reference_pressure=planet.reference_pressure,
            temperature=self.temperature,
            mass_mixing_ratios=self.mass_fractions,
            cloud_pressure=self.p_cloud,
            guillot_temperature_profile_gamma=self.gamma,
            guillot_temperature_profile_kappa_ir_z0=self.kappa_ir_z0,
            included_line_species=included_line_species,
            intrinsic_temperature=self.t_int,
            heh2_ratio=self.heh2_ratio,
            use_equilibrium_chemistry=self.use_equilibrium_chemistry,
            co_ratio=self.co_ratio,
            metallicity=10 ** self.metallicity,
            carbon_pressure_quench=self.p_quench_c,
            star_effective_temperature=planet.star_effective_temperature,
            star_radius=planet.star_radius,
            # star_spectral_radiosity=star_spectral_radiosity,
            radial_velocity_semi_amplitude=radial_velocity_semi_amplitude,
            planet_orbital_inclination=planet.orbital_inclination,
            semi_major_axis=planet.orbit_semi_major_axis,
            rest_frame_velocity_shift=0.0,
            orbital_phases=None,
            system_observer_radial_velocities=None,
            wavelengths_instrument=None,
            instrument_resolving_power=None,
            data=None,
            data_uncertainties=None,
            reduced_data=None,
            reduced_data_uncertainties=None,
            reduction_matrix=None,
            airmass=None,
            telluric_transmittance=None,
            variable_throughput=None
        )

    @staticmethod
    def _get_phoenix_star_spectral_radiosity(star_effective_temperature, wavelengths):
        star_data = compute_phoenix_spectrum(star_effective_temperature)
        star_data[:, 1] = SpectralModelLegacy.radiosity_erg_hz2radiosity_erg_cm(
            star_data[:, 1], cst.c / star_data[:, 0]  # cm to Hz
        )

        star_data[:, 0] *= 1e4  # cm to um

        star_radiosities = frebin.rebin_spectrum(
            star_data[:, 0],
            star_data[:, 1],
            wavelengths
        )

        return star_radiosities

    def get_phoenix_star_spectral_radiosity(self, planet: Planet):
        return self._get_phoenix_star_spectral_radiosity(planet.star_effective_temperature, self.wavelengths)

    def get_name(self):
        name = 'spectral_model_'
        name += f"{self.planet_name.replace(' ', '_')}_" \
                f"Tint{self.t_int}K_Z{self.metallicity}_co{self.co_ratio}_pc{self.p_cloud}bar_" \
                f"{self.wavelength_boundaries[0]}-{self.wavelength_boundaries[1]}um_ds{self.lbl_opacity_sampling}"

        if self.do_scat_emis:
            name += '_scat'
        else:
            name += '_noscat'

        if self.name_suffix != '':
            name += f'_{self.name_suffix}'

        return name

    @staticmethod
    def get_star_radiosity_filename(star_effective_temperature, path='.'):
        return f'{path}/crires/star_spectrum_{star_effective_temperature}K.dat'

    def init_mass_fractions(self, atmosphere, temperature, include_species, mass_fractions=None):
        from petitRADTRANS.chemistry import pre_calculated_chemistry as pm  # import is here because it's long to load

        if mass_fractions is None:
            mass_fractions = {}
        elif not isinstance(mass_fractions, dict):
            raise ValueError(
                f"mass fractions must be in a dict, but the input was of type '{type(mass_fractions)}'")

        pressures = atmosphere.table_pressures * 1e-6  # cgs to bar

        if np.size(self.co_ratio) == 1:
            co_ratios = np.ones_like(pressures) * self.co_ratio
        else:
            co_ratios = self.co_ratio

        if np.size(self.metallicity) == 1:
            metallicity = np.ones_like(pressures) * self.metallicity
        else:
            metallicity = self.metallicity

        abundances = pm.interpolate_mass_fractions_chemical_table(
            co_ratios=co_ratios,
            log10_metallicities=metallicity,
            temperatures=temperature,
            pressures=pressures,
            carbon_pressure_quench=self.p_quench_c
        )

        # Check mass_mixing_ratios keys
        for key in mass_fractions:
            if key not in atmosphere.line_species and key not in abundances:
                raise KeyError(f"key '{key}' not in line species list or "
                               f"standard petitRADTRANS mass fractions dict")

        # Get the right keys for the mass fractions dictionary
        mass_fractions_dict = {}

        for key in abundances:
            found = False

            for line_species_name in atmosphere.line_species:
                line_species = line_species_name.split('_', 1)[0]

                if line_species == 'C2H2':   # C2H2 special case
                    line_species += ',acetylene'

                if key == line_species:
                    if key not in include_species:
                        mass_fractions_dict[line_species_name] = np.zeros_like(temperature)
                    elif line_species_name in mass_fractions:
                        mass_fractions_dict[line_species_name] = mass_fractions[line_species_name]
                    else:
                        mass_fractions_dict[line_species_name] = abundances[line_species]

                    found = True

                    break

            if not found:
                if key in mass_fractions:
                    mass_fractions_dict[key] = mass_fractions[key]
                else:
                    mass_fractions_dict[key] = abundances[key]

        for key in mass_fractions:
            if key not in mass_fractions_dict:
                if key not in include_species:
                    mass_fractions_dict[key] = np.zeros_like(temperature)
                else:
                    mass_fractions_dict[key] = mass_fractions[key]

        return mass_fractions_dict

    def init_temperature_guillot(self, planet: Planet, atmosphere: Radtrans):
        pressures = atmosphere.pressures * 1e-6  # cgs to bar
        temperatures = self._init_temperature_profile_guillot(
            pressures=pressures,
            gamma=self.gamma,
            surface_gravity=planet.reference_gravity,
            intrinsic_temperature=self.t_int,
            equilibrium_temperature=planet.equilibrium_temperature,
            kappa_ir_z0=self.kappa_ir_z0,
            metallicity=10 ** self.metallicity
        )

        return temperatures

    def save(self):
        with open(self.get_filename(), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get(cls, planet_name, wavelength_boundaries, lbl_opacity_sampling, pressures, do_scat_emis, t_int,
            metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
            line_species_list='default', rayleigh_species='default', continuum_opacities='default',
            include_species='all', model_suffix='', atmosphere=None, calculate_transmission_spectrum=False,
            calculate_emission_spectrum=False, calculate_eclipse_depth=False,
            rewrite=True):
        # Initialize model
        model = cls.species_init(
            include_species=include_species,
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            model_suffix=model_suffix
        )

        # Generate or load model
        return cls.generate_from(
            model=model,
            pressures=pressures,
            line_species_list=line_species_list,
            rayleigh_species=rayleigh_species,
            continuum_opacities=continuum_opacities,
            include_species=include_species,
            model_suffix=model_suffix,
            atmosphere=atmosphere,
            calculate_transmission_spectrum=calculate_transmission_spectrum,
            calculate_emission_spectrum=calculate_emission_spectrum,
            calculate_eclipse_depth=calculate_eclipse_depth,
            rewrite=rewrite
        )

    @classmethod
    def generate_from(cls, model, pressures,
                      line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                      include_species=None, model_suffix='',
                      atmosphere=None, temperature_profile=None, mass_fractions=None,
                      calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                      calculate_eclipse_depth=False,
                      rewrite=False):
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]
        elif include_species is None:
            include_species = ['all']

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")

        # Check if model already exists
        if os.path.isfile(model.filename) and not rewrite:
            print(f"Model '{model.filename}' already exists, loading from file...")
            return model.power_load(model.filename)
        else:
            if os.path.isfile(model.filename) and rewrite:
                print(f"Rewriting already existing model '{model.filename}'...")

            print(f"Generating model '{model.filename}'...")

            # Initialize species
            if line_species_list == 'default':
                line_species_list = cls.default_line_species

            if rayleigh_species == 'default':
                rayleigh_species = cls.default_rayleigh_species

            if continuum_opacities == 'default':
                continuum_opacities = cls.default_continuum_opacities

            if include_species == ['all']:
                include_species = []

                for species_name in line_species_list:
                    if species_name == 'CO_36':
                        include_species.append(species_name)
                    else:
                        include_species.append(species_name.split('_', 1)[0])

            # Generate the model
            return cls._generate(
                model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                model_suffix, atmosphere, temperature_profile, mass_fractions, calculate_transmission_spectrum,
                calculate_emission_spectrum, calculate_eclipse_depth
            )

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
        return radiosity_erg_cm * wavelength ** 2 / cst.c

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
        return radiosity_erg_hz * frequency ** 2 / cst.c

    @classmethod
    def species_init(cls, include_species, planet_name, wavelength_boundaries, lbl_opacity_sampling, do_scat_emis,
                     t_int, metallicity, co_ratio, p_cloud, kappa_ir_z0=0.01, gamma=0.4, p_quench_c=None, haze_factor=1,
                     atmosphere_file=None, wavelengths=None, transit_radius=None, temperature=None,
                     mass_fractions=None, planet_model_file=None, model_suffix='', filename=None):
        # Initialize include_species
        if not hasattr(include_species, '__iter__') or isinstance(include_species, str):
            include_species = [include_species]

        if len(include_species) > 1:
            raise ValueError("Please include either only one species or all of them using keyword 'all'")
        else:
            if model_suffix == '':
                species_suffix = f'{include_species[0]}'
            else:
                species_suffix = f'_{include_species[0]}'

        # Initialize model
        return cls(
            planet_name=planet_name,
            wavelength_boundaries=wavelength_boundaries,
            lbl_opacity_sampling=lbl_opacity_sampling,
            do_scat_emis=do_scat_emis,
            t_int=t_int,
            metallicity=metallicity,
            co_ratio=co_ratio,
            p_cloud=p_cloud,
            kappa_ir_z0=kappa_ir_z0,
            gamma=gamma,
            p_quench_c=p_quench_c,
            haze_factor=haze_factor,
            atmosphere_file=atmosphere_file,
            wavelengths=wavelengths,
            transit_radius=transit_radius,
            temperature=temperature,
            mass_fractions=mass_fractions,
            planet_model_file=planet_model_file,
            model_suffix=model_suffix + species_suffix,
            filename=filename
        )

    @staticmethod
    def _generate(model, pressures, line_species_list, rayleigh_species, continuum_opacities, include_species,
                  model_suffix, atmosphere=None, temperature_profile=None, mass_fractions=None,
                  calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                  calculate_eclipse_depth=False):
        if atmosphere is None:
            atmosphere, model.atmosphere_file = model.get_atmosphere_model(
                model.wavelengths_boundaries, pressures, line_species_list, rayleigh_species, continuum_opacities,
                model_suffix
            )
        else:
            model.atmosphere_file = SpectralModelLegacy._get_hires_atmosphere_filename(
                pressures, model.wavelengths_boundaries, model.line_by_line_opacity_sampling, model_suffix
            )

        # A Planet needs to be generated and saved first
        model.planet_model_file = Planet.generate_filename(model.planet_name)
        planet = Planet.load(model.planet_name, model.planet_model_file)

        if temperature_profile is None:
            model.temperature = model.init_temperature_guillot(
                planet=planet,
                atmosphere=atmosphere
            )
        elif isinstance(temperature_profile, (float, int)):
            model.temperature = np.ones_like(atmosphere.press) * temperature_profile
        elif np.size(temperature_profile) == np.size(atmosphere.press):
            model.temperature = np.asarray(temperature_profile)
        else:
            raise ValueError(f"could not initialize temperature profile using input {temperature_profile}; "
                             f"possible inputs are None, float, int, "
                             f"or a 1-D array of the same size of argument 'pressures' ({np.size(atmosphere.press)})")

        # Generate mass fractions from equilibrium chemistry first to have all the keys
        # TODO generate the mass fractions dict without calling equilibrium chemistry
        model.mass_fractions = model.init_mass_fractions(
            atmosphere=atmosphere,
            temperature=model.temperature,
            include_species=include_species,
            mass_fractions=mass_fractions
        )

        if not calculate_transmission_spectrum and not calculate_emission_spectrum and not calculate_eclipse_depth:
            print(f"No spectrum will be calculated")

            return model

        if calculate_transmission_spectrum:
            model.wavelengths, model.transit_radius = model.calculate_transmission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )

        if calculate_emission_spectrum and not calculate_eclipse_depth:
            model.wavelengths, model.flux = model.calculate_emission_spectrum(
                atmosphere=atmosphere,
                planet=planet
            )
        elif calculate_eclipse_depth:
            model.wavelengths, model.eclipse_depth, model.flux = model.calculate_eclipse_depth(
                atmosphere=atmosphere,
                planet=planet
            )

        return model

    @staticmethod
    def _get_hires_atmosphere_filename(pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis,
                                       model_suffix=''):
        filename = planet_models_directory + os.path.sep \
                   + f"atmosphere_{np.max(pressures)}-{np.min(pressures)}bar_" \
                     f"{wlen_bords_micron[0]}-{wlen_bords_micron[1]}um_ds{lbl_opacity_sampling}"

        if do_scat_emis:
            filename += '_scat'

        if model_suffix != '':
            filename += f"_{model_suffix}"

        filename += '.pkl'

        return filename

    @staticmethod
    def get_atmosphere_model(wlen_bords_micron, pressures,
                             line_species_list=None, rayleigh_species=None, continuum_opacities=None,
                             lbl_opacity_sampling=1, do_scat_emis=False, save=False,
                             model_suffix=''):
        atmosphere_filename = SpectralModelLegacy._get_hires_atmosphere_filename(
            pressures, wlen_bords_micron, lbl_opacity_sampling, do_scat_emis, model_suffix
        )

        if os.path.isfile(atmosphere_filename):
            print('Loading atmosphere model...')
            with open(atmosphere_filename, 'rb') as f:
                atmosphere = pickle.load(f)
        else:
            atmosphere = SpectralModelLegacy.init_atmosphere(
                pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                lbl_opacity_sampling, do_scat_emis
            )

            if save:
                print('Saving atmosphere model...')
                with open(atmosphere_filename, 'wb') as f:
                    pickle.dump(atmosphere, f)

        return atmosphere, atmosphere_filename

    @staticmethod
    def init_atmosphere(pressures, wlen_bords_micron, line_species_list, rayleigh_species, continuum_opacities,
                        lbl_opacity_sampling, do_scat_emis, mode='lbl'):
        print('Generating atmosphere...')

        atmosphere = Radtrans(
            line_species=line_species_list,
            rayleigh_species=rayleigh_species,
            gas_continuum_contributors=continuum_opacities,
            wavelengths_boundaries=wlen_bords_micron,
            line_opacity_mode=mode,
            scattering_in_emission=do_scat_emis,
            line_by_line_opacity_sampling=lbl_opacity_sampling
        )

        atmosphere.setup_opa_structure(pressures)

        return atmosphere


def get_orbital_phases(phase_start, orbital_period, dit, ndit, return_times=False):
    """Calculate orbital phases assuming low eccentricity.

    Args:
        phase_start: planet phase at the start of observations
        orbital_period: (s) orbital period of the planet
        dit: (s) integration duration
        ndit: number of integrations
        return_times: if true, also returns the time used to calculate the orbital phases

    Returns:
        ndit phases from start_phase at t=0 to the phase at t=dit * ndit
    """
    times = np.linspace(0, dit * ndit, ndit)
    phases = np.mod(phase_start + times / orbital_period, 1.0)

    if return_times:
        return phases, times  # the 2 * pi factors cancel out
    else:
        return phases


def _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature):
    return f"generic_{radius / cst.r_jup:.2f}Rjup_logg{np.log10(surface_gravity):.2f}_teq{equilibrium_temperature:.2f}K"


def generate_model_grid(models, pressures,
                        line_species_list='default', rayleigh_species='default', continuum_opacities='default',
                        model_suffix='', atmosphere=None, temperature_profile=None, mass_fractions=None,
                        calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                        calculate_eclipse_depth=False,
                        rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        models: dictionary of models
        pressures: (bar) 1D-array containing the pressure grid of the models
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        temperature_profile: if None, a Guillot temperature profile is generated, if int or float, an isothermal
            temperature profile is generated, if 1-D array of the same size of pressures, the temperature profile is
            directly used
        mass_fractions: if None, equilibrium chemistry is used, if dict, the values from the dict are used
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Model {i}/{len(models) * len(models[model])}...")

            models[model][species] = SpectralModelLegacy.generate_from(
                model=models[model][species],
                pressures=pressures,
                include_species=species,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                temperature_profile=temperature_profile,
                mass_fractions=mass_fractions,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[model][species].save()

    return models


def get_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list, pressures,
                   wavelength_boundaries, line_species_list='default', rayleigh_species='default',
                   continuum_opacities='default', model_suffix='', atmosphere=None,
                   calculate_transmission_spectrum=False, calculate_emission_spectrum=False,
                   calculate_eclipse_depth=False,
                   rewrite=False, save=False):
    """
    Get a grid of models, generate it if needed.
    Models are generated using petitRADTRANS in its line-by-line mode. Clouds are modelled as a gray deck.
    Output will be organized hierarchically as follows: model string > included species

    Args:
        planet_name: name of the planet modelled
        lbl_opacity_sampling: downsampling coefficient of
        do_scat_emis: if True, include the scattering for emission spectra
        parameter_dicts: list of dictionaries containing the models parameters
        species_list: list of lists of species to include in the models (e.g. ['all', 'H2O', 'CH4'])
        pressures: (bar) 1D-array containing the pressure grid of the models
        wavelength_boundaries: (um) size-2 array containing the min and max wavelengths
        line_species_list: list containing all the line species to include in the models
        rayleigh_species: list containing all the rayleigh species to include in the models
        continuum_opacities: list containing all the continua to include in the models
        model_suffix: suffix of the model
        atmosphere: pre-loaded Radtrans object
        calculate_transmission_spectrum: if True, calculate the transmission spectrum of the model
        calculate_emission_spectrum: if True, calculate the emission spectrum of the model
        calculate_eclipse_depth: if True, calculate the eclipse depth, and the emission spectrum, of the model
        rewrite: if True, rewrite all the models, even if they already exists
        save: if True, save the models once generated

    Returns:
        models: a dictionary containing all the requested models
    """
    models = {}
    i = 0

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            i += 1
            print(f"Model {i}/{len(parameter_dicts) * len(species_list)}...")

            models[parameter_dict.to_str()][species] = SpectralModelLegacy.get(
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                pressures=pressures,
                include_species=species,
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                line_species_list=line_species_list,
                rayleigh_species=rayleigh_species,
                continuum_opacities=continuum_opacities,
                model_suffix=model_suffix,
                atmosphere=atmosphere,
                calculate_transmission_spectrum=calculate_transmission_spectrum,
                calculate_emission_spectrum=calculate_emission_spectrum,
                calculate_eclipse_depth=calculate_eclipse_depth,
                rewrite=rewrite
            )

            if save:
                models[parameter_dict.to_str()][species].save()

    return models


def get_parameter_dicts(t_int: list, metallicity: list, co_ratio: list, p_cloud: list):
    """
    Generate a parameter dictionary from parameters.
    To be used in get_model_grid()

    Args:
        t_int: (K) intrinsic temperature of the planet
        metallicity: metallicity of the planet
        co_ratio: C/O ratio of the planet
        p_cloud: (bar) cloud top pressure of the planet

    Returns:
        parameter_dict: a ParameterDict
    """

    parameter_dicts = []

    for t in t_int:
        for z in metallicity:
            for co in co_ratio:
                for pc in p_cloud:
                    parameter_dicts.append(
                        ParametersDict(
                            t_int=t,
                            metallicity=z,
                            co_ratio=co,
                            p_cloud=pc
                        )
                    )

    return parameter_dicts


def init_model_grid(planet_name, lbl_opacity_sampling, do_scat_emis, parameter_dicts, species_list,
                    wavelength_boundaries, model_suffix=''):
    # Initialize models
    models = {}
    all_models_exist = True

    for parameter_dict in parameter_dicts:
        models[parameter_dict.to_str()] = {}

        for species in species_list:
            models[parameter_dict.to_str()][species] = SpectralModelLegacy.species_init(
                include_species=species,
                planet_name=planet_name,
                wavelength_boundaries=wavelength_boundaries,
                lbl_opacity_sampling=lbl_opacity_sampling,
                do_scat_emis=do_scat_emis,
                t_int=parameter_dict['intrinsic_temperature'],
                metallicity=parameter_dict['metallicity'],
                co_ratio=parameter_dict['co_ratio'],
                p_cloud=parameter_dict['p_cloud'],
                kappa_ir_z0=0.01,
                gamma=0.4,
                p_quench_c=None,
                haze_factor=1,
                model_suffix=model_suffix
            )

            if not os.path.isfile(models[parameter_dict.to_str()][species].filename) and all_models_exist:
                all_models_exist = False

    return models, all_models_exist


def load_model_grid(models):
    i = 0

    for model in models:
        for species in models[model]:
            i += 1
            print(f"Loading model {i}/{len(models) * len(models[model])} from '{models[model][species].filename}'...")

            models[model][species] = models[model][species].power_load(models[model][species].filename)

    return models


def make_generic_planet(radius, surface_gravity, equilibrium_temperature,
                        star_effective_temperature=5500, star_radius=cst.r_sun, orbit_semi_major_axis=cst.au):
    name = _get_generic_planet_name(radius, surface_gravity, equilibrium_temperature)

    return SimplePlanet(
        name,
        radius,
        surface_gravity,
        star_effective_temperature,
        star_radius,
        orbit_semi_major_axis,
        equilibrium_temperature=equilibrium_temperature
    )
