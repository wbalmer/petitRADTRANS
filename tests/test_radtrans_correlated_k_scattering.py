"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import (
    check_cloud_complete_coverage_consistency, check_partial_cloud_coverage_full_consistency, get_cloud_parameters,
    test_parameters, temperature_guillot_2010
)

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
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

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
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering_with_variable_fsed():
    mass_fractions, _, _, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

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
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_photospheric_radius_calculation():
    mass_fractions, cloud_particles_mean_radii, _, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_photosphere_radius,
        relative_tolerance=relative_tolerance
    )

    atmosphere_ck_scattering._Radtrans__set_sum_opacities(emission=True)

    (opacities, continuum_opacities_scattering, cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
     cloud_opacities, cloud_particles_mean_radii) = (
        atmosphere_ck_scattering._calculate_opacities(
            temperatures=temperature_guillot_2010,
            mass_fractions=mass_fractions,
            mean_molar_masses=test_parameters['mean_molar_mass'],
            reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std
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


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_planetary_average():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

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
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        irradiation_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_dayside():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
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
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        irradiation_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
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
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        irradiation_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=test_parameters['stellar_parameters']['incidence_angle'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_complete_coverage_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    mass_fractions_clear = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])

    for complete_coverage_cloud in test_parameters['cloud_parameters']['complete_coverage_clouds']:
        mass_fractions_clear[complete_coverage_cloud] = mass_fractions[complete_coverage_cloud]

    geometry = 'non-isotropic'

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    check_cloud_complete_coverage_consistency(
        spectrum_function=atmosphere_ck_scattering.calculate_flux,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        complete_coverage_clouds=test_parameters['cloud_parameters']['complete_coverage_clouds'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        opaque_cloud_top_pressure=None,
        temperatures=temperature_guillot_2010,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        irradiation_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=test_parameters['stellar_parameters']['incidence_angle'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_spectrum_partial_cloud_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    mass_fractions_clear = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])

    geometry = 'non-isotropic'

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck_scattering.calculate_flux,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        opaque_cloud_top_pressure=None,
        temperatures=temperature_guillot_2010,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        irradiation_geometry=geometry,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=test_parameters['stellar_parameters']['incidence_angle'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
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
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_complete_coverage_calculated_radius_scattering():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    mass_fractions_clear = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])

    for complete_coverage_cloud in test_parameters['cloud_parameters']['complete_coverage_clouds']:
        mass_fractions_clear[complete_coverage_cloud] = mass_fractions[complete_coverage_cloud]

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    check_cloud_complete_coverage_consistency(
        spectrum_function=atmosphere_ck_scattering.calculate_transit_radii,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        complete_coverage_clouds=test_parameters['cloud_parameters']['complete_coverage_clouds'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        opaque_cloud_top_pressure=None,
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_partial_cloud_calculated_radius_scattering():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    mass_fractions_clear = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck_scattering.calculate_transit_radii,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        opaque_cloud_top_pressure=None,
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )
