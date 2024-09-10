"""Test the radtrans module in line-by-line mode.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .benchmark import Benchmark
from .context import petitRADTRANS
from .utils import check_partial_cloud_coverage_full_consistency, test_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(test_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        line_opacity_mode='lbl'
    )

    return atmosphere


atmosphere_lbl = init_radtrans_line_by_line()


def test_line_by_line_emission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters['mass_fractions_line_by_line'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        frequencies_to_wavelengths=False
    )


def test_line_by_line_emission_partial_cloud_calculated_radius():
    mass_fractions = copy.deepcopy(test_parameters['mass_fractions_line_by_line'])
    mass_fractions_clear =  copy.deepcopy(mass_fractions)
    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = \
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']

    benchmark = Benchmark(
        function=atmosphere_lbl.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_lbl.calculate_flux,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_coverage_fraction=test_parameters['cloud_parameters']['cloud_coverage_fraction'],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        opaque_cloud_top_pressure=None,
        temperatures=temperature_guillot_2010,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )


def test_line_by_line_transmission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_line_by_line'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        frequencies_to_wavelengths=False
    )


def test_line_by_line_transmission_spectrum_partial_gray_cloud():
    benchmark = Benchmark(
        function=atmosphere_lbl.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_lbl.calculate_transit_radii,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_coverage_fraction=test_parameters['cloud_parameters']['cloud_coverage_fraction'],
        mass_fractions=test_parameters['mass_fractions_line_by_line'],
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
