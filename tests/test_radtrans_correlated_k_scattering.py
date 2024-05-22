"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010

relative_tolerance = 1e-6  # relative tolerance when comparing with older spectra


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(test_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        line_opacity_mode='c-k',
        scattering_in_emission=True
    )

    return atmosphere


atmosphere_ck_scattering = init_radtrans_correlated_k()


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = \
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering_with_variable_fsed():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = \
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    fsed_min = test_parameters['cloud_parameters']['cloud_species'][
        'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed_variable_setup'][0]
    fsed_max = test_parameters['cloud_parameters']['cloud_species'][
        'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed_variable_setup'][1]
    fseds = np.linspace(fsed_min, fsed_max, len(atmosphere_ck_scattering._pressures))

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=fseds,
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_photospheric_radius_calculation():
    from .benchmark import ReferenceFile

    original_parameters_relative_tolerance = copy.copy(ReferenceFile.parameters_relative_tolerance)
    ReferenceFile.parameters_relative_tolerance = 1e-14  # to make test work on Paul's laptop...

    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = \
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_photosphere_radius,
        relative_tolerance=relative_tolerance
    )

    cloud_particles_mean_radii = {
        'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu': (
            np.ones_like(temperature_guillot_2010)
            * test_parameters['cloud_parameters'][
                'cloud_species']['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['radius']
        )
    }

    atmosphere_ck_scattering._Radtrans__set_sum_opacities(emission=True)

    (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
     cloud_opacities, cloud_particles_mean_radii) = (
        atmosphere_ck_scattering._calculate_opacities(
            temperatures=temperature_guillot_2010,
            mass_fractions=mass_fractions,
            mean_molar_masses=test_parameters['mean_molar_mass'],
            reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=test_parameters['cloud_parameters'][
                'cloud_species']['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        )
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mean_molar_masses=test_parameters['mean_molar_mass'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup,
        opacities=opacities,
        continuum_opacities_scattering=continuum_opacities_scattering,
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_photosphere_median_optical_depth=None,
        cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities=cloud_absorption_opacities
    )

    ReferenceFile.parameters_relative_tolerance = original_parameters_relative_tolerance


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_planetary_average():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = \
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']

    geometry = 'planetary_ave'

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_dayside():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = (
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']
    )

    geometry = 'dayside_ave'

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = (
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']
    )

    geometry = 'non-isotropic'

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=test_parameters['stellar_parameters']['incidence_angle'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = (
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']
    )

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )
