"""Test the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import (check_partial_cloud_coverage_full_consistency, get_cloud_parameters,
                    test_parameters, temperature_guillot_2010, temperature_isothermal)

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(test_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        line_opacity_mode='c-k'
    )

    return atmosphere


atmosphere_ck = init_radtrans_correlated_k()


# Tests
def test_correlated_k_emission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_contribution_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
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
        return_contribution=True,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
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


def test_correlated_k_emission_spectrum_cloud_hansen_radius():
    mass_fractions, _, cloud_f_sed, _, cloud_hansen_b = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particles_radius_distribution='hansen',
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_partial_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    mass_fractions_clear = copy.deepcopy(test_parameters['mass_fractions_correlated_k'])

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck.calculate_flux,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        temperatures=temperature_guillot_2010,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_cloud_power_law():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        power_law_opacity_350nm=test_parameters['cloud_parameters']['kappa_zero'],
        power_law_opacity_coefficient=test_parameters['cloud_parameters']['gamma_scattering'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_gray_cloud():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        opaque_cloud_top_pressure=test_parameters['cloud_parameters']['cloud_pressure'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_partial_gray_cloud():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck.calculate_transit_radii,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters['cloud_parameters']['cloud_fraction'],
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        mass_fractions_clear=None,
        opaque_cloud_top_pressure=test_parameters['cloud_parameters']['cloud_pressure'],
        temperatures=temperature_isothermal,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_rayleigh():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        haze_factor=test_parameters['cloud_parameters']['haze_factor'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_spectrum_cloud_fixed_radius():
    mass_fractions, cloud_particles_mean_radii, _, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )


def test_correlated_k_transmission_contribution_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        return_contribution=True,
        frequencies_to_wavelengths=False
    )

    # the python transit radius calculation function outputs are slightly
    # different than the fortran version, creating a slight error, especially in deeper
    # atmospheric levels
    # absolute_tolerance = 8e-7  # there is a max absolute error of 3.28989835374216e-10 between windows and WSL
    # generated files TODO investigate why


def test_correlated_k_transmission_spectrum_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )
