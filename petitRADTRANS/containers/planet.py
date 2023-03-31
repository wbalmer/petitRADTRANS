"""Planet object."""
import os
import warnings

import h5py
import numpy as np
import pyvo
from astropy.table.table import Table
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import astropy.units as u

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.utils import calculate_uncertainty


class Planet:
    default_planet_models_directory = os.path.abspath(
        os.path.join(os.environ.get("pRT_input_data_path"), 'planet_data')
    )

    def __init__(
            self,
            name,
            mass=0.,
            mass_error_upper=0.,
            mass_error_lower=0.,
            radius=0.,
            radius_error_upper=0.,
            radius_error_lower=0.,
            orbit_semi_major_axis=0.,
            orbit_semi_major_axis_error_upper=0.,
            orbit_semi_major_axis_error_lower=0.,
            orbital_eccentricity=0.,
            orbital_eccentricity_error_upper=0.,
            orbital_eccentricity_error_lower=0.,
            orbital_inclination=0.,
            orbital_inclination_error_upper=0.,
            orbital_inclination_error_lower=0.,
            orbital_period=0.,
            orbital_period_error_upper=0.,
            orbital_period_error_lower=0.,
            argument_of_periastron=0.,
            argument_of_periastron_error_upper=0.,
            argument_of_periastron_error_lower=0.,
            epoch_of_periastron=0.,
            epoch_of_periastron_error_upper=0.,
            epoch_of_periastron_error_lower=0.,
            ra=0.,
            dec=0.,
            x=0.,
            y=0.,
            z=0.,
            reference_pressure=0.01,
            density=0.,
            density_error_upper=0.,
            density_error_lower=0.,
            surface_gravity=0.,
            surface_gravity_error_upper=0.,
            surface_gravity_error_lower=0.,
            equilibrium_temperature=0.,
            equilibrium_temperature_error_upper=0.,
            equilibrium_temperature_error_lower=0.,
            insolation_flux=0.,
            insolation_flux_error_upper=0.,
            insolation_flux_error_lower=0.,
            bond_albedo=0.,
            bond_albedo_error_upper=0.,
            bond_albedo_error_lower=0.,
            transit_depth=0.,
            transit_depth_error_upper=0.,
            transit_depth_error_lower=0.,
            transit_midpoint_time=0.,
            transit_midpoint_time_error_upper=0.,
            transit_midpoint_time_error_lower=0.,
            transit_duration=0.,
            transit_duration_error_upper=0.,
            transit_duration_error_lower=0.,
            projected_obliquity=0.,
            projected_obliquity_error_upper=0.,
            projected_obliquity_error_lower=0.,
            true_obliquity=0.,
            true_obliquity_error_upper=0.,
            true_obliquity_error_lower=0.,
            radial_velocity_amplitude=0.,
            radial_velocity_amplitude_error_upper=0.,
            radial_velocity_amplitude_error_lower=0.,
            planet_stellar_radius_ratio=0.,
            planet_stellar_radius_ratio_error_upper=0.,
            planet_stellar_radius_ratio_error_lower=0.,
            semi_major_axis_stellar_radius_ratio=0.,
            semi_major_axis_stellar_radius_ratio_error_upper=0.,
            semi_major_axis_stellar_radius_ratio_error_lower=0.,
            reference='',
            discovery_year=0,
            discovery_method='',
            discovery_reference='',
            confirmation_status='',
            host_name='',
            star_spectral_type='',
            star_mass=0.,
            star_mass_error_upper=0.,
            star_mass_error_lower=0.,
            star_radius=0.,
            star_radius_error_upper=0.,
            star_radius_error_lower=0.,
            star_age=0.,
            star_age_error_upper=0.,
            star_age_error_lower=0.,
            star_metallicity=0.,
            star_metallicity_error_upper=0.,
            star_metallicity_error_lower=0.,
            star_effective_temperature=0.,
            star_effective_temperature_error_upper=0.,
            star_effective_temperature_error_lower=0.,
            star_luminosity=0.,
            star_luminosity_error_upper=0.,
            star_luminosity_error_lower=0.,
            star_rotational_period=0.,
            star_rotational_period_error_upper=0.,
            star_rotational_period_error_lower=0.,
            star_radial_velocity=0.,
            star_radial_velocity_error_upper=0.,
            star_radial_velocity_error_lower=0.,
            star_rotational_velocity=0.,
            star_rotational_velocity_error_upper=0.,
            star_rotational_velocity_error_lower=0.,
            star_density=0.,
            star_density_error_upper=0.,
            star_density_error_lower=0.,
            star_surface_gravity=0.,
            star_surface_gravity_error_upper=0.,
            star_surface_gravity_error_lower=0.,
            star_reference='',
            system_star_number=0,
            system_planet_number=0,
            system_moon_number=0,
            system_distance=0.,
            system_distance_error_upper=0.,
            system_distance_error_lower=0.,
            system_apparent_magnitude_v=0.,
            system_apparent_magnitude_v_error_upper=0.,
            system_apparent_magnitude_v_error_lower=0.,
            system_apparent_magnitude_j=0.,
            system_apparent_magnitude_j_error_upper=0.,
            system_apparent_magnitude_j_error_lower=0.,
            system_apparent_magnitude_k=0.,
            system_apparent_magnitude_k_error_upper=0.,
            system_apparent_magnitude_k_error_lower=0.,
            system_proper_motion=0.,
            system_proper_motion_error_upper=0.,
            system_proper_motion_error_lower=0.,
            system_proper_motion_ra=0.,
            system_proper_motion_ra_error_upper=0.,
            system_proper_motion_ra_error_lower=0.,
            system_proper_motion_dec=0.,
            system_proper_motion_dec_error_upper=0.,
            system_proper_motion_dec_error_lower=0.,
            units=None
    ):
        self.name = name
        self.mass = mass
        self.mass_error_upper = mass_error_upper
        self.mass_error_lower = mass_error_lower
        self.radius = radius
        self.radius_error_upper = radius_error_upper
        self.radius_error_lower = radius_error_lower
        self.orbit_semi_major_axis = orbit_semi_major_axis
        self.orbit_semi_major_axis_error_upper = orbit_semi_major_axis_error_upper
        self.orbit_semi_major_axis_error_lower = orbit_semi_major_axis_error_lower
        self.orbital_eccentricity = orbital_eccentricity
        self.orbital_eccentricity_error_upper = orbital_eccentricity_error_upper
        self.orbital_eccentricity_error_lower = orbital_eccentricity_error_lower
        self.orbital_inclination = orbital_inclination
        self.orbital_inclination_error_upper = orbital_inclination_error_upper
        self.orbital_inclination_error_lower = orbital_inclination_error_lower
        self.orbital_period = orbital_period
        self.orbital_period_error_upper = orbital_period_error_upper
        self.orbital_period_error_lower = orbital_period_error_lower
        self.argument_of_periastron = argument_of_periastron
        self.argument_of_periastron_error_upper = argument_of_periastron_error_upper
        self.argument_of_periastron_error_lower = argument_of_periastron_error_lower
        self.epoch_of_periastron = epoch_of_periastron
        self.epoch_of_periastron_error_upper = epoch_of_periastron_error_upper
        self.epoch_of_periastron_error_lower = epoch_of_periastron_error_lower
        self.ra = ra
        self.dec = dec
        self.x = x
        self.y = y
        self.z = z
        self.reference_pressure = reference_pressure
        self.density = density
        self.density_error_upper = density_error_upper
        self.density_error_lower = density_error_lower
        self.surface_gravity = surface_gravity
        self.surface_gravity_error_upper = surface_gravity_error_upper
        self.surface_gravity_error_lower = surface_gravity_error_lower
        self.equilibrium_temperature = equilibrium_temperature
        self.equilibrium_temperature_error_upper = equilibrium_temperature_error_upper
        self.equilibrium_temperature_error_lower = equilibrium_temperature_error_lower
        self.insolation_flux = insolation_flux
        self.insolation_flux_error_upper = insolation_flux_error_upper
        self.insolation_flux_error_lower = insolation_flux_error_lower
        self.bond_albedo = bond_albedo
        self.bond_albedo_error_upper = bond_albedo_error_upper
        self.bond_albedo_error_lower = bond_albedo_error_lower
        self.transit_depth = transit_depth
        self.transit_depth_error_upper = transit_depth_error_upper
        self.transit_depth_error_lower = transit_depth_error_lower
        self.transit_midpoint_time = transit_midpoint_time
        self.transit_midpoint_time_error_upper = transit_midpoint_time_error_upper
        self.transit_midpoint_time_error_lower = transit_midpoint_time_error_lower
        self.transit_duration = transit_duration
        self.transit_duration_error_upper = transit_duration_error_upper
        self.transit_duration_error_lower = transit_duration_error_lower
        self.projected_obliquity = projected_obliquity
        self.projected_obliquity_error_upper = projected_obliquity_error_upper
        self.projected_obliquity_error_lower = projected_obliquity_error_lower
        self.true_obliquity = true_obliquity
        self.true_obliquity_error_upper = true_obliquity_error_upper
        self.true_obliquity_error_lower = true_obliquity_error_lower
        self.radial_velocity_amplitude = radial_velocity_amplitude
        self.radial_velocity_amplitude_error_upper = radial_velocity_amplitude_error_upper
        self.radial_velocity_amplitude_error_lower = radial_velocity_amplitude_error_lower
        self.planet_stellar_radius_ratio = planet_stellar_radius_ratio
        self.planet_stellar_radius_ratio_error_upper = planet_stellar_radius_ratio_error_upper
        self.planet_stellar_radius_ratio_error_lower = planet_stellar_radius_ratio_error_lower
        self.semi_major_axis_stellar_radius_ratio = semi_major_axis_stellar_radius_ratio
        self.semi_major_axis_stellar_radius_ratio_error_upper = semi_major_axis_stellar_radius_ratio_error_upper
        self.semi_major_axis_stellar_radius_ratio_error_lower = semi_major_axis_stellar_radius_ratio_error_lower
        self.reference = reference
        self.discovery_year = discovery_year
        self.discovery_method = discovery_method
        self.discovery_reference = discovery_reference
        self.confirmation_status = confirmation_status
        self.host_name = host_name
        self.star_spectral_type = star_spectral_type
        self.star_mass = star_mass
        self.star_mass_error_upper = star_mass_error_upper
        self.star_mass_error_lower = star_mass_error_lower
        self.star_radius = star_radius
        self.star_radius_error_upper = star_radius_error_upper
        self.star_radius_error_lower = star_radius_error_lower
        self.star_age = star_age
        self.star_age_error_upper = star_age_error_upper
        self.star_age_error_lower = star_age_error_lower
        self.star_metallicity = star_metallicity
        self.star_metallicity_error_upper = star_metallicity_error_upper
        self.star_metallicity_error_lower = star_metallicity_error_lower
        self.star_effective_temperature = star_effective_temperature
        self.star_effective_temperature_error_upper = star_effective_temperature_error_upper
        self.star_effective_temperature_error_lower = star_effective_temperature_error_lower
        self.star_luminosity = star_luminosity
        self.star_luminosity_error_upper = star_luminosity_error_upper
        self.star_luminosity_error_lower = star_luminosity_error_lower
        self.star_rotational_period = star_rotational_period
        self.star_rotational_period_error_upper = star_rotational_period_error_upper
        self.star_rotational_period_error_lower = star_rotational_period_error_lower
        self.star_radial_velocity = star_radial_velocity
        self.star_radial_velocity_error_upper = star_radial_velocity_error_upper
        self.star_radial_velocity_error_lower = star_radial_velocity_error_lower
        self.star_rotational_velocity = star_rotational_velocity
        self.star_rotational_velocity_error_upper = star_rotational_velocity_error_upper
        self.star_rotational_velocity_error_lower = star_rotational_velocity_error_lower
        self.star_density = star_density
        self.star_density_error_upper = star_density_error_upper
        self.star_density_error_lower = star_density_error_lower
        self.star_surface_gravity = star_surface_gravity
        self.star_surface_gravity_error_upper = star_surface_gravity_error_upper
        self.star_surface_gravity_error_lower = star_surface_gravity_error_lower
        self.star_reference = star_reference
        self.system_star_number = system_star_number
        self.system_planet_number = system_planet_number
        self.system_moon_number = system_moon_number
        self.system_distance = system_distance
        self.system_distance_error_upper = system_distance_error_upper
        self.system_distance_error_lower = system_distance_error_lower
        self.system_apparent_magnitude_v = system_apparent_magnitude_v
        self.system_apparent_magnitude_v_error_upper = system_apparent_magnitude_v_error_upper
        self.system_apparent_magnitude_v_error_lower = system_apparent_magnitude_v_error_lower
        self.system_apparent_magnitude_j = system_apparent_magnitude_j
        self.system_apparent_magnitude_j_error_upper = system_apparent_magnitude_j_error_upper
        self.system_apparent_magnitude_j_error_lower = system_apparent_magnitude_j_error_lower
        self.system_apparent_magnitude_k = system_apparent_magnitude_k
        self.system_apparent_magnitude_k_error_upper = system_apparent_magnitude_k_error_upper
        self.system_apparent_magnitude_k_error_lower = system_apparent_magnitude_k_error_lower
        self.system_proper_motion = system_proper_motion
        self.system_proper_motion_error_upper = system_proper_motion_error_upper
        self.system_proper_motion_error_lower = system_proper_motion_error_lower
        self.system_proper_motion_ra = system_proper_motion_ra
        self.system_proper_motion_ra_error_upper = system_proper_motion_ra_error_upper
        self.system_proper_motion_ra_error_lower = system_proper_motion_ra_error_lower
        self.system_proper_motion_dec = system_proper_motion_dec
        self.system_proper_motion_dec_error_upper = system_proper_motion_dec_error_upper
        self.system_proper_motion_dec_error_lower = system_proper_motion_dec_error_lower

        if units is None:
            self.units = {
                'name': 'N/A',
                'mass': 'g',
                'mass_error_upper': 'g',
                'mass_error_lower': 'g',
                'radius': 'cm',
                'radius_error_upper': 'cm',
                'radius_error_lower': 'cm',
                'orbit_semi_major_axis': 'cm',
                'orbit_semi_major_axis_error_upper': 'cm',
                'orbit_semi_major_axis_error_lower': 'cm',
                'orbital_eccentricity': 'None',
                'orbital_eccentricity_error_upper': 'None',
                'orbital_eccentricity_error_lower': 'None',
                'orbital_inclination': 'deg',
                'orbital_inclination_error_upper': 'deg',
                'orbital_inclination_error_lower': 'deg',
                'orbital_period': 's',
                'orbital_period_error_upper': 's',
                'orbital_period_error_lower': 's',
                'argument_of_periastron': 'deg',
                'argument_of_periastron_error_upper': 'deg',
                'argument_of_periastron_error_lower': 'deg',
                'epoch_of_periastron': 's',
                'epoch_of_periastron_error_upper': 's',
                'epoch_of_periastron_error_lower': 's',
                'ra': 'deg',
                'dec': 'deg',
                'x': 'cm',
                'y': 'cm',
                'z': 'cm',
                'reference_pressure': 'bar',
                'density': 'g/cm^3',
                'density_error_upper': 'g/cm^3',
                'density_error_lower': 'g/cm^3',
                'surface_gravity': 'cm/s^2',
                'surface_gravity_error_upper': 'cm/s^2',
                'surface_gravity_error_lower': 'cm/s^2',
                'equilibrium_temperature': 'K',
                'equilibrium_temperature_error_upper': 'K',
                'equilibrium_temperature_error_lower': 'K',
                'insolation_flux': 'erg/s/cm^2',
                'insolation_flux_error_upper': 'erg/s/cm^2',
                'insolation_flux_error_lower': 'erg/s/cm^2',
                'bond_albedo': 'None',
                'bond_albedo_error_upper': 'None',
                'bond_albedo_error_lower': 'None',
                'transit_depth': 'None',
                'transit_depth_error_upper': 'None',
                'transit_depth_error_lower': 'None',
                'transit_midpoint_time': 's',
                'transit_midpoint_time_error_upper': 's',
                'transit_midpoint_time_error_lower': 's',
                'transit_duration': 's',
                'transit_duration_error_upper': 's',
                'transit_duration_error_lower': 's',
                'projected_obliquity': 'deg',
                'projected_obliquity_error_upper': 'deg',
                'projected_obliquity_error_lower': 'deg',
                'true_obliquity': 'deg',
                'true_obliquity_error_upper': 'deg',
                'true_obliquity_error_lower': 'deg',
                'radial_velocity_amplitude': 'cm/s',
                'radial_velocity_amplitude_error_upper': 'cm/s',
                'radial_velocity_amplitude_error_lower': 'cm/s',
                'planet_stellar_radius_ratio': 'None',
                'planet_stellar_radius_ratio_error_upper': 'None',
                'planet_stellar_radius_ratio_error_lower': 'None',
                'semi_major_axis_stellar_radius_ratio': 'None',
                'semi_major_axis_stellar_radius_ratio_error_upper': 'None',
                'semi_major_axis_stellar_radius_ratio_error_lower': 'None',
                'reference': 'N/A',
                'discovery_year': 'year',
                'discovery_method': 'N/A',
                'discovery_reference': 'N/A',
                'confirmation_status': 'N/A',
                'host_name': 'N/A',
                'star_spectral_type': 'N/A',
                'star_mass': 'g',
                'star_mass_error_upper': 'g',
                'star_mass_error_lower': 'g',
                'star_radius': 'cm',
                'star_radius_error_upper': 'cm',
                'star_radius_error_lower': 'cm',
                'star_age': 's',
                'star_age_error_upper': 's',
                'star_age_error_lower': 's',
                'star_metallicity': 'dex',
                'star_metallicity_error_upper': 'dex',
                'star_metallicity_error_lower': 'dex',
                'star_effective_temperature': 'K',
                'star_effective_temperature_error_upper': 'K',
                'star_effective_temperature_error_lower': 'K',
                'star_luminosity': 'erg/s',
                'star_luminosity_error_upper': 'erg/s',
                'star_luminosity_error_lower': 'erg/s',
                'star_rotational_period': 's',
                'star_rotational_period_error_upper': 's',
                'star_rotational_period_error_lower': 's',
                'star_radial_velocity': 'cm/s',
                'star_radial_velocity_error_upper': 'cm/s',
                'star_radial_velocity_error_lower': 'cm/s',
                'star_rotational_velocity': 'cm/s',
                'star_rotational_velocity_error_upper': 'cm/s',
                'star_rotational_velocity_error_lower': 'cm/s',
                'star_density': 'g/cm^3',
                'star_density_error_upper': 'g/cm^3',
                'star_density_error_lower': 'g/cm^3',
                'star_surface_gravity': 'cm/s^2',
                'star_surface_gravity_error_upper': 'cm/s^2',
                'star_surface_gravity_error_lower': 'cm/s^2',
                'star_reference': 'N/A',
                'system_star_number': 'None',
                'system_planet_number': 'None',
                'system_moon_number': 'None',
                'system_distance': 'cm',
                'system_distance_error_upper': 'cm',
                'system_distance_error_lower': 'cm',
                'system_apparent_magnitude_v': 'None',
                'system_apparent_magnitude_v_error_upper': 'None',
                'system_apparent_magnitude_v_error_lower': 'None',
                'system_apparent_magnitude_j': 'None',
                'system_apparent_magnitude_j_error_upper': 'None',
                'system_apparent_magnitude_j_error_lower': 'None',
                'system_apparent_magnitude_k': 'None',
                'system_apparent_magnitude_k_error_upper': 'None',
                'system_apparent_magnitude_k_error_lower': 'None',
                'system_proper_motion': 'deg/s',
                'system_proper_motion_error_upper': 'deg/s',
                'system_proper_motion_error_lower': 'deg/s',
                'system_proper_motion_ra': 'deg/s',
                'system_proper_motion_ra_error_upper': 'deg/s',
                'system_proper_motion_ra_error_lower': 'deg/s',
                'system_proper_motion_dec': 'deg/s',
                'system_proper_motion_dec_error_upper': 'deg/s',
                'system_proper_motion_dec_error_lower': 'deg/s',
                'units': 'N/A'
            }
        else:
            self.units = units

    def calculate_planetary_equilibrium_temperature(self):
        """
        Calculate the equilibrium temperature of a planet.
        """
        equilibrium_temperature = \
            self.star_effective_temperature * np.sqrt(self.star_radius / (2 * self.orbit_semi_major_axis)) \
            * (1 - self.bond_albedo) ** 0.25

        partial_derivatives = np.array([
            equilibrium_temperature / self.star_effective_temperature,  # dt_eq/dt_eff
            0.5 * equilibrium_temperature / self.star_radius,  # dt_eq/dr*
            - 0.5 * equilibrium_temperature / self.orbit_semi_major_axis  # dt_eq/dd
        ])
        uncertainties = np.abs(np.array([
            [self.star_effective_temperature_error_lower, self.star_effective_temperature_error_upper],
            [self.star_radius_error_lower, self.star_radius_error_upper],
            [self.orbit_semi_major_axis_error_lower, self.orbit_semi_major_axis_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return equilibrium_temperature, errors[1], -errors[0]

    def get_filename(self):
        return self.generate_filename(self.name)

    def save(self, filename=None):
        if filename is None:
            filename = self.get_filename()

        with h5py.File(filename, 'w') as f:
            for key in self.__dict__:
                if key == 'units':
                    continue

                data_set = f.create_dataset(
                    name=key,
                    data=self.__dict__[key]
                )

                if self.units[key] != 'N/A':
                    data_set.attrs['units'] = self.units[key]

    @classmethod
    def from_tab_file(cls, filename, use_best_mass=True):
        """Read from a NASA Exoplanet Archive Database .tab file.
        Args:
            filename: file to read
            use_best_mass: if True, use NASA Exoplanet Archive Database 'bmass' argument instead of 'mass'.

        Returns:
            planets: a list of Planet objects
        """
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()

            # Skip header
            while line[0] == '#':
                line = f.readline()
                line = line.strip()

            # Read column names
            columns_name = line.split('\t')

            planet_name_index = columns_name.index('pl_name')

            planets = {}

            # Read data
            for line in f:
                line = line.strip()
                columns = line.split('\t')

                new_planet = cls(columns[planet_name_index])
                keys = []

                for i, value in enumerate(columns):
                    # Clearer keynames
                    keys.append(columns_name[i])

                    if value != '':
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                        value, keys[i] = Planet.__convert_nasa_exoplanet_archive(
                            value, keys[i], use_best_mass=use_best_mass
                        )
                    else:
                        value = None

                    if keys[i] in new_planet.__dict__:
                        new_planet.__dict__[keys[i]] = value

                # Try to calculate the planet mass and radius if missing
                if new_planet.radius == 0 and new_planet.mass > 0 and new_planet.density > 0:
                    new_planet.radius = (3 * new_planet.mass / (4 * np.pi * new_planet.density)) ** (1 / 3)

                    partial_derivatives = np.array([
                        new_planet.radius / (3 * new_planet.mass),  # dr/dm
                        - new_planet.radius / (3 * new_planet.density)  # dr/drho
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.mass_error_lower, new_planet.mass_error_upper],
                        [new_planet.density_error_lower, new_planet.density_error_upper]
                    ]))

                    new_planet.radius_error_lower, new_planet.radius_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors
                elif new_planet.mass == 0 and new_planet.radius > 0 and new_planet.density > 0:
                    new_planet.mass = new_planet.density * 4 / 3 * np.pi * new_planet.radius ** 3

                    partial_derivatives = np.array([
                        new_planet.mass / new_planet.density,  # dm/drho
                        3 * new_planet.radius / new_planet.radius  # dm/dr
                    ])
                    uncertainties = np.abs(np.array([
                        [new_planet.density_error_lower, new_planet.density_error_upper],
                        [new_planet.radius_error_lower, new_planet.radius_error_upper]
                    ]))

                    new_planet.mass_error_lower, new_planet.mass_error_upper = \
                        calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                # Try to calculate the star radius if missing
                if new_planet.star_radius == 0 and new_planet.star_mass > 0:
                    if new_planet.star_surface_gravity > 0:
                        new_planet.star_radius, \
                            new_planet.star_radius_error_upper, new_planet.star_radius_error_lower = \
                            new_planet.surface_gravity2radius(
                                new_planet.star_surface_gravity,
                                new_planet.star_mass,
                                surface_gravity_error_upper=new_planet.star_surface_gravity_error_upper,
                                surface_gravity_error_lower=new_planet.star_surface_gravity_error_lower,
                                mass_error_upper=new_planet.star_mass_error_upper,
                                mass_error_lower=new_planet.star_mass_error_lower
                            )
                    elif new_planet.star_density > 0:
                        new_planet.star_radius = \
                            (3 * new_planet.star_mass / (4 * np.pi * new_planet.star_density)) ** (1 / 3)

                        partial_derivatives = np.array([
                            new_planet.star_radius / (3 * new_planet.star_mass),  # dr/dm
                            - new_planet.star_radius / (3 * new_planet.star_density)  # dr/drho
                        ])
                        uncertainties = np.abs(np.array([
                            [new_planet.star_mass_error_lower, new_planet.star_mass_error_upper],
                            [new_planet.star_density_error_lower, new_planet.star_density_error_upper]
                        ]))

                        new_planet.star_radius_error_lower, new_planet.star_radius_error_upper = \
                            calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

                if 'surface_gravity' not in keys and new_planet.radius > 0 and new_planet.mass > 0:
                    new_planet.surface_gravity, \
                        new_planet.surface_gravity_error_upper, new_planet.surface_gravity_error_lower = \
                        new_planet.mass2surface_gravity(
                            new_planet.mass,
                            new_planet.radius,
                            mass_error_upper=new_planet.mass_error_upper,
                            mass_error_lower=new_planet.mass_error_lower,
                            radius_error_upper=new_planet.radius_error_upper,
                            radius_error_lower=new_planet.radius_error_lower
                        )

                if 'equilibrium_temperature' not in keys \
                        and new_planet.orbit_semi_major_axis > 0 \
                        and new_planet.star_effective_temperature > 0 \
                        and new_planet.star_radius > 0:
                    new_planet.equilibrium_temperature, \
                        new_planet.equilibrium_temperature_error_upper, \
                        new_planet.equilibrium_temperature_error_lower = \
                        new_planet.calculate_planetary_equilibrium_temperature()

                planets[new_planet.name] = new_planet

        return planets

    @classmethod
    def from_votable(cls, votable):
        new_planet = cls('new_planet')
        parameter_dict = {}

        for key in votable.keys():
            # Clearer keynames
            value, key = Planet.__convert_nasa_exoplanet_archive(votable[key], key)
            parameter_dict[key] = value

        parameter_dict = new_planet.select_best_in_column(parameter_dict)

        for key in parameter_dict:
            if key in new_planet.__dict__:
                new_planet.__dict__[key] = parameter_dict[key]

        if 'surface_gravity' not in parameter_dict:
            new_planet.surface_gravity, \
                new_planet.surface_gravity_error_upper, new_planet.surface_gravity_error_lower = \
                new_planet.mass2surface_gravity(
                    new_planet.mass,
                    new_planet.radius,
                    mass_error_upper=new_planet.mass_error_upper,
                    mass_error_lower=new_planet.mass_error_lower,
                    radius_error_upper=new_planet.radius_error_upper,
                    radius_error_lower=new_planet.radius_error_lower
                )

        if 'equilibrium_temperature' not in parameter_dict:
            new_planet.equilibrium_temperature, \
                new_planet.equilibrium_temperature_error_upper, new_planet.equilibrium_temperature_error_lower = \
                new_planet.calculate_planetary_equilibrium_temperature()

        return new_planet

    @classmethod
    def from_votable_file(cls, filename):
        astro_table = Table.read(filename)

        return cls.from_votable(astro_table)

    @classmethod
    def get(cls, name):
        filename = cls.generate_filename(name)

        return cls.get_from(name, filename)

    @classmethod
    def get_from(cls, name, filename):
        if not os.path.exists(filename):
            filename_vot = filename.rsplit('.', 1)[0] + '.vot'  # search for votable

            if not os.path.exists(filename_vot):
                print(f"file '{filename_vot}' not found, downloading...")

                directory = os.path.dirname(filename_vot)

                if not os.path.isdir(directory):
                    os.mkdir(directory)

                cls.download_from_nasa_exoplanet_archive(name)

            # Save into HDF5 and remove the VO table
            new_planet = cls.from_votable_file(filename_vot)
            new_planet.save()
            os.remove(filename_vot)

            return new_planet
        else:
            return cls.load(name, filename)

    @classmethod
    def load(cls, name, filename=None):
        new_planet = cls(name)

        if filename is None:
            filename = new_planet.get_filename()

        with h5py.File(filename, 'r') as f:
            for key in f:
                if isinstance(f[key][()], bytes):
                    value = str(f[key][()], 'utf-8')
                else:
                    value = f[key][()]

                new_planet.__dict__[key] = value

                if 'units' in f[key].attrs:
                    if key in new_planet.units:
                        if f[key].attrs['units'] != new_planet.units[key]:
                            raise ValueError(f"units of key '{key}' must be '{new_planet.units[key]}', "
                                             f"but is '{f[key].attrs['units']}'")
                    else:
                        new_planet.units[key] = f[key].attrs['units'][()]
                else:
                    new_planet.units[key] = 'N/A'

        return new_planet

    @staticmethod
    def __convert_nasa_exoplanet_archive(value, key, verbose=False, use_best_mass=False):
        skip_unit_conversion = False

        # Heads
        if key[:3] == 'sy_':
            key = 'system_' + key[3:]
        elif key[:3] == 'st_':
            key = 'star_' + key[3:]
        elif key[:5] == 'disc_':
            key = 'discovery_' + key[5:]

        # Tails
        if key[-4:] == 'err1':
            key = key[:-4] + '_error_upper'
        elif key[-4:] == 'err2':
            key = key[:-4] + '_error_lower'
        elif key[-3:] == 'lim':
            key = key[:-3] + '_limit_flag'
            skip_unit_conversion = True
        elif key[-3:] == 'str':
            key = key[:-3] + '_str'
            skip_unit_conversion = True

        # Parameters of interest
        if '_orbper' in key:
            key = key.replace('_orbper', '_orbital_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orblper' in key:
            key = key.replace('_orblper', '_argument_of_periastron')
        elif '_orbsmax' in key:
            key = key.replace('_orbsmax', '_orbit_semi_major_axis')

            if not skip_unit_conversion:
                value *= nc.AU
        elif '_orbincl' in key:
            key = key.replace('_orbincl', '_orbital_inclination')
        elif '_orbtper' in key:
            key = key.replace('_orbtper', '_epoch_of_periastron')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_orbeccen' in key:
            key = key.replace('_orbeccen', '_orbital_eccentricity')
        elif '_eqt' in key:
            key = key.replace('_eqt', '_equilibrium_temperature')
        elif '_occdep' in key:
            key = key.replace('_occdep', '_occultation_depth')
        elif '_insol' in key:
            key = key.replace('_insol', '_insolation_flux')

            if not skip_unit_conversion:
                value *= nc.s_earth
        elif '_dens' in key:
            key = key.replace('_dens', '_density')
        elif '_trandep' in key:
            key = key.replace('_trandep', '_transit_depth')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_tranmid' in key:
            key = key.replace('_tranmid', '_transit_midpoint_time')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_trandur' in key:
            key = key.replace('_trandur', '_transit_duration')

            if not skip_unit_conversion:
                value *= nc.snc.hour
        elif '_spectype' in key:
            key = key.replace('_spectype', '_spectral_type')
        elif '_rotp' in key:
            key = key.replace('_rotp', '_rotational_period')

            if not skip_unit_conversion:
                value *= nc.snc.day
        elif '_projobliq' in key:
            key = key.replace('_projobliq', '_projected_obliquity')
        elif '_rvamp' in key:
            key = key.replace('_rvamp', '_radial_velocity_amplitude')

            if not skip_unit_conversion:
                value *= 1e2
        elif '_radj' in key:
            key = key.replace('_radj', '_radius')

            if not skip_unit_conversion:
                value *= nc.r_jup
        elif '_ratror' in key:
            key = key.replace('_ratror', '_planet_stellar_radius_ratio')
        elif '_trueobliq' in key:
            key = key.replace('_trueobliq', '_true_obliquity')
        elif '_ratdor' in key:
            key = key.replace('_ratdor', '_semi_major_axis_stellar_radius_ratio')
        elif '_imppar' in key:
            key = key.replace('_imppar', '_impact_parameter')
        elif '_msinij' in key:
            key = key.replace('_msinij', '_mass_sini')

            if not skip_unit_conversion:
                value *= nc.m_jup
        elif '_massj' in key:
            if not use_best_mass:
                key = key.replace('_massj', '_mass')

                if not skip_unit_conversion:
                    value *= nc.m_jup
        elif '_bmassj' in key:
            if use_best_mass:
                key = key.replace('_bmassj', '_mass')

                if not skip_unit_conversion:
                    value *= nc.m_jup
        elif '_teff' in key:
            key = key.replace('_teff', '_effective_temperature')
        elif '_met' in key:
            key = key.replace('_met', '_metallicity')
        elif '_radv' in key:
            key = key.replace('_radv', '_radial_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_vsin' in key:
            key = key.replace('_vsin', '_rotational_velocity')

            if not skip_unit_conversion:
                value *= 1e5
        elif '_lum' in key:
            key = key.replace('_lum', '_luminosity')

            if not skip_unit_conversion:
                value = 10 ** value * nc.l_sun
        elif '_logg' in key:
            key = key.replace('_logg', '_surface_gravity')

            if not skip_unit_conversion:
                value = 10 ** value
        elif '_age' in key:
            if not skip_unit_conversion:
                value *= 1e9 * nc.snc.year
        elif 'star_mass' in key:
            if not skip_unit_conversion:
                value *= nc.m_sun
        elif 'star_rad' in key:
            key = key.replace('star_rad', 'star_radius')

            if not skip_unit_conversion:
                value *= nc.r_sun
        elif '_dist' in key:
            key = key.replace('_dist', '_distance')

            if not skip_unit_conversion:
                value *= nc.pc
        elif '_plx' in key:
            key = key.replace('_plx', '_parallax')

            if not skip_unit_conversion:
                value *= 3.6e-6
        elif '_pm' in key:
            if key[-3:] == '_pm':
                key = key.replace('_pm', '_proper_motion')
            else:
                i = key.find('_pm')
                key = key[:i] + '_proper_motion_' + key[i + len('_pm'):]

            if not skip_unit_conversion:
                value *= np.deg2rad(1e-3 / 3600 / nc.snc.year)

        elif key == 'hostname':
            key = 'host_name'
        elif key == 'discoverymethod':
            key = 'discovery_method'
        elif key == 'discovery_refname':
            key = 'discovery_reference'
        elif 'controv_flag' in key:
            key = 'controversy_flag'
        elif key == 'star_refname':
            key = 'star_reference'
        elif key == 'soltype':
            key = 'confirmation_status'
        elif key == 'system_snum':
            key = 'system_star_number'
        elif key == 'system_pnum':
            key = 'system_planet_number'
        elif key == 'system_mnum':
            key = 'system_moon_number'
        elif 'mag' in key:
            i = key.find('mag')

            if i + len('mag') == len(key):
                tail = ''
            else:
                tail = key[i + 3:]

                if tail[0] != '_':  # should not be necessary
                    tail = '_' + tail

            # Move magnitude band to the end
            if key[i - 2] == '_':  # one-character band
                letter = key[i - 1]
                key = key[:i - 1] + 'apparent_magnitude_' + letter + tail
            elif key[i - 3] == '_':  # two-characters band
                letters = key[i - 2:i]
                key = key[:i - 2] + 'apparent_magnitude_' + letters + tail
            elif 'kepmag' in key:
                key = key[:i - 3] + 'apparent_magnitude_' + 'kepler' + tail
            elif 'gaiamag' in key:
                key = key[:i - 4] + 'apparent_magnitude_' + 'gaia' + tail
            else:
                raise ValueError(f"unidentified apparent magnitude key '{key}'")
        elif verbose:
            print(f"unchanged key '{key}' with value {value}")

        if key[:3] == 'pl_':
            key = key[3:]

        return value, key

    @staticmethod
    def calculate_full_transit_duration(total_transit_duration, planet_radius, star_radius, impact_parameter):
        k = planet_radius / star_radius

        return total_transit_duration * np.sqrt(
            ((1 - k) ** 2 - impact_parameter ** 2)
            / ((1 + k) ** 2 - impact_parameter ** 2)
        )

    @staticmethod
    def calculate_impact_parameter(planet_orbit_semi_major_axis, planet_orbital_inclination, star_radius):
        return planet_orbit_semi_major_axis * np.cos(np.deg2rad(planet_orbital_inclination)) / star_radius

    @staticmethod
    def calculate_planet_radial_velocity(planet_radial_velocity_amplitude, planet_orbital_inclination,
                                         orbital_longitude, **kwargs):
        """Calculate the planet radial velocity as seen by an observer.

        Args:
            planet_radial_velocity_amplitude: maximum radial velocity for an inclination angle of 90 degree
            planet_orbital_inclination: (degree) angle between the normal of the planet orbital plane and the axis of
                observation, i.e. 90 degree: edge view, 0 degree: top view
            orbital_longitude: (degree) angle between the closest point from the observer on the planet orbit and the
                planet position, i.e. if the planet orbital inclination is 0 degree, 0 degree: mid-primary transit
                point, 180 degree: mid-secondary eclipse point

        Returns:

        """
        kp = planet_radial_velocity_amplitude * np.sin(np.deg2rad(planet_orbital_inclination))  # (cm.s-1)

        return kp * np.sin(np.deg2rad(orbital_longitude))

    @staticmethod
    def calculate_orbital_velocity(star_mass, semi_major_axis):
        """Calculate an approximation of the orbital velocity.
        This equation is valid if the mass of the object is negligible compared to the mass of the star, and if the
        eccentricity of the object is close to 0.

        Args:
            star_mass: (g) mass of the star
            semi_major_axis: (cm) semi-major axis of the orbit of the object

        Returns: (cm.s-1) the mean orbital velocity, assuming 0 eccentricity and mass_object << mass_star
        """
        return np.sqrt(nc.G * star_mass / semi_major_axis)

    @staticmethod
    def generate_filename(name, directory=default_planet_models_directory):
        return f"{directory}{os.path.sep}planet_{name.replace(' ', '_')}.h5"

    @staticmethod
    def get_astropy_coordinates(ra, dec,  site_name=None, latitude=None, longitude=None, height=None):
        if site_name is not None:
            observer_location = EarthLocation.of_site(site_name)
        else:
            observer_location = EarthLocation.from_geodetic(
                lat=latitude * u.deg,
                lon=longitude * u.deg,
                height=height * u.m
            )

        target_coordinates = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg
        )

        return observer_location, target_coordinates

    @staticmethod
    def get_airmass(ra, dec, time, site_name=None, latitude=None, longitude=None, height=None,
                    time_format='mjd'):
        observer_location, target_coordinates = Planet.get_astropy_coordinates(
            ra=ra,
            dec=dec,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height
        )

        frame = AltAz(
            obstime=Time(time, format=time_format),
            location=observer_location
        )

        taget_altaz = target_coordinates.transform_to(frame)

        return taget_altaz.secz

    @staticmethod
    def get_barycentric_velocities(ra, dec, time, site_name=None, latitude=None, longitude=None, height=None,
                                   time_format='mjd'):
        observer_location, target_coordinates = Planet.get_astropy_coordinates(
            ra=ra,
            dec=dec,
            site_name=site_name,
            latitude=latitude,
            longitude=longitude,
            height=height
        )

        return target_coordinates.radial_velocity_correction(
            obstime=Time(time, format=time_format),
            location=observer_location
        ).value

    @staticmethod
    def get_simple_transit_curve(time_from_mid_transit, planet_radius, star_radius,
                                 planet_orbital_velocity=None, star_mass=None, orbit_semi_major_axis=None):
        """
        Assume no inclination, circular orbit, observer infinitely far away, spherical objects, perfectly sharp and
        black planet and perfectly sharp and uniformly luminous star.

        Args:
            time_from_mid_transit: (s) time from mid-transit, 0 is the mid-transit time, < 0 before and > 0 after
            planet_radius: (cm) radius of the planet
            star_radius: (cm) radius of the star
            planet_orbital_velocity: (cm.s-1) planet velocity along its orbit.
            star_mass: (g) mass of the star
            orbit_semi_major_axis: (cm) planet orbit semi major axis

        Returns:

        """
        if planet_orbital_velocity is None:
            planet_orbital_velocity = Planet.calculate_orbital_velocity(star_mass, orbit_semi_major_axis)

        planet_center_to_star_center = planet_orbital_velocity * time_from_mid_transit

        if np.abs(planet_center_to_star_center) >= star_radius + planet_radius:
            return 1.0  # planet is not transiting yet
        elif np.abs(planet_center_to_star_center) <= star_radius - planet_radius:
            return 1 - (planet_radius / star_radius) ** 2  # planet is fully transiting
        else:
            # Get the vertical coordinate intersection between the two discs
            x_intersection = (star_radius ** 2 - planet_radius ** 2 + planet_center_to_star_center ** 2) \
                             / (2 * planet_center_to_star_center)
            y_intersection = np.sqrt(star_radius ** 2 - np.abs(x_intersection) ** 2)

            # Get the half angle between the two intersection points and the center of each disc
            theta_half_intersection_planet = np.arcsin(y_intersection / planet_radius)
            theta_half_intersection_star = np.arcsin(y_intersection / star_radius)

            if np.abs(planet_center_to_star_center) < star_radius:
                theta_half_intersection_planet = np.pi - theta_half_intersection_planet

            # Calculate the area of the sector between the 2 intersection point for the 2 discs
            planet_sector_area = planet_radius ** 2 * theta_half_intersection_planet
            star_sector_area = star_radius ** 2 * theta_half_intersection_star

            # Calculate the area of the triangles formed by the 2 intersection points and the center of each disc
            planet_triangle_area = 0.5 * planet_radius ** 2 * np.sin(2 * theta_half_intersection_planet)
            star_triangle_area = 0.5 * star_radius ** 2 * np.sin(2 * theta_half_intersection_star)

            return 1 - (planet_sector_area - planet_triangle_area + star_sector_area - star_triangle_area) \
                / (np.pi * star_radius ** 2)

    @staticmethod
    def get_orbital_phases(phase_start, orbital_period, times):
        """Calculate orbital phases assuming low eccentricity.

        Args:
            phase_start: planet phase at the start of observations
            orbital_period: (s) orbital period of the planet
            times: (s) time array

        Returns:
            The orbital phases for the given time
        """
        phases = phase_start + times / orbital_period
        add = np.zeros(times.size)
        add[np.less(phases, 0)] = - 1

        return add + np.mod(phase_start + times / orbital_period, 1.0)

    @staticmethod
    def download_from_nasa_exoplanet_archive(name):
        service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        result_set = service.search(f"select * from ps where pl_name = '{name}'")

        astro_table = result_set.to_table()
        filename = Planet.generate_filename(name).rsplit('.', 1)[0] + '.vot'

        astro_table.write(filename, format='votable')

        return astro_table

    @staticmethod
    def select_best_in_column(dictionary):
        parameter_dict = {}
        tails = ['_error_upper', '_error_lower', '_limit_flag', '_str']

        for key in dictionary.keys():
            if tails[0] in key or tails[1] in key or tails[2] in key or tails[3] in key:
                continue  # skip every tailed parameters
            elif dictionary[key].dtype == object or not (key + tails[0] in dictionary and key + tails[1] in dictionary):
                # if object or no error tailed parameters, get the first value that is not masked
                if not hasattr(dictionary[key], '__iter__'):
                    raise ValueError(f"No value found for parameter '{key}'; "
                                     f"this error is most often caused by a misspelling of a planet name")

                parameter_dict[key] = dictionary[key][0]

                for value in dictionary[key][1:]:
                    if not hasattr(value, 'mask'):
                        parameter_dict[key] = value

                        break
            else:
                value_error_upper = dictionary[key + tails[0]]
                value_error_lower = dictionary[key + tails[1]]
                error_interval = np.abs(value_error_upper) + np.abs(value_error_lower)

                wh = np.where(error_interval == np.min(error_interval))[0]

                parameter_dict[key] = dictionary[key][wh][0]

                for tail in tails:
                    if key + tail in dictionary:
                        parameter_dict[key + tail] = dictionary[key + tail][wh][0]

        return parameter_dict

    @staticmethod
    def mass2surface_gravity(mass, radius,
                             mass_error_upper=0., mass_error_lower=0., radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            mass: (g) mass of the planet
            radius: (cm) radius of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        if radius <= 0:
            warnings.warn(f"unknown or invalid radius ({radius}), surface gravity not calculated")

            return None, None, None

        surface_gravity = nc.G * mass / radius ** 2

        partial_derivatives = np.array([
            surface_gravity / mass,  # dg/dm
            - 2 * surface_gravity / radius  # dg/dr
        ])
        uncertainties = np.abs(np.array([
            [mass_error_lower, mass_error_upper],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return surface_gravity, errors[1], -errors[0]

    @staticmethod
    def surface_gravity2radius(surface_gravity, mass,
                               surface_gravity_error_upper=0., surface_gravity_error_lower=0.,
                               mass_error_upper=0., mass_error_lower=0.):
        """
        Convert the mass of a planet to its surface gravity.
        Args:
            surface_gravity: (cm.s-2) surface_gravity of the planet
            mass: (g) mass of the planet
            mass_error_upper: (g) upper error on the planet mass
            mass_error_lower: (g) lower error on the planet mass
            surface_gravity_error_upper: (cm.s-2) upper error on the planet radius
            surface_gravity_error_lower: (cm.s-2) lower error on the planet radius

        Returns:
            (cm.s-2) the surface gravity of the planet, and its upper and lower error
        """
        if surface_gravity <= 0:
            warnings.warn(f"unknown or invalid surface gravity ({surface_gravity}), radius not calculated")

            return None, None, None

        radius = (nc.G * mass / surface_gravity) ** 0.5

        partial_derivatives = np.array([
            radius / (2 * mass),  # dr/dm
            - radius / (2 * surface_gravity)  # dr/dg
        ])
        uncertainties = np.abs(np.array([
            [mass_error_lower, mass_error_upper],
            [surface_gravity_error_lower, surface_gravity_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return radius, errors[1], -errors[0]

    @staticmethod
    def surface_gravity2mass(surface_gravity, radius,
                             surface_gravity_error_upper=0., surface_gravity_error_lower=0.,
                             radius_error_upper=0., radius_error_lower=0.):
        """
        Convert the surface gravity of a planet to its mass.
        Args:
            surface_gravity: (cm.s-2) surface gravity of the planet
            radius: (cm) radius of the planet
            surface_gravity_error_upper: (cm.s-2) upper error on the planet surface gravity
            surface_gravity_error_lower: (cm.s-2) lower error on the planet surface gravity
            radius_error_upper: (cm) upper error on the planet radius
            radius_error_lower: (cm) lower error on the planet radius

        Returns:
            (g) the mass of the planet, and its upper and lower error
        """
        mass = surface_gravity / nc.G * radius ** 2

        partial_derivatives = np.array([
            mass / surface_gravity,  # dm/dg
            2 * mass / radius  # dm/dr
        ])
        uncertainties = np.abs(np.array([
            [surface_gravity_error_lower, surface_gravity_error_upper],
            [radius_error_lower, radius_error_upper]
        ]))

        errors = calculate_uncertainty(partial_derivatives, uncertainties)  # lower and upper errors

        return mass, errors[1], -errors[0]
