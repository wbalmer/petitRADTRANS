import cProfile
import os
import tracemalloc

import numpy as np

from .context import petitRADTRANS
import petitRADTRANS.__debug
from .utils import test_parameters, tests_results_directory, temperature_guillot_2010


def init_radtrans_correlated_k():
    radtrans = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures_performance'],
        line_species=test_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(test_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_correlated_k_performance'],
        line_opacity_mode='c-k',
        scattering_in_emission=True
    )

    return radtrans


def init_parameters():
    mass_fractions = {}

    for key in test_parameters['mass_fractions_correlated_k']:
        mass_fractions[key] = np.ones_like(test_parameters['pressures_performance'])

    mass_fractions['Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu'] = (
        test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['mass_fraction']
    )

    mean_molar_masses = np.ones_like(test_parameters['pressures_performance'])
    eddy_diffusion_coefficients = np.ones_like(test_parameters['pressures_performance'])

    temperatures = np.interp(
        test_parameters['pressures_performance'],
        test_parameters['pressures'],
        temperature_guillot_2010
    )

    return temperatures, mass_fractions, mean_molar_masses, eddy_diffusion_coefficients


def performance_test_correlated_k_emission(radtrans, temperatures,
                                           mass_fractions, mean_molar_masses, eddy_diffusion_coefficients):
    geometry = 'dayside_ave'

    radtrans.calculate_flux(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=mean_molar_masses,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
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


def performance_test_correlated_k_transmission(radtrans, temperatures,
                                               mass_fractions, mean_molar_masses, eddy_diffusion_coefficients):

    radtrans.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=mean_molar_masses,
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_f_sed=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['f_sed'],
        cloud_particle_radius_distribution_std=test_parameters['cloud_parameters']['cloud_species'][
            'Mg2-Si-O4-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )


def main(n_runs, trace_memory):
    print(f"Initialisation...")
    radtrans = None

    for i in range(n_runs):
        print(f"Opacity loading, run {i + 1}/{n_runs}")
        radtrans = init_radtrans_correlated_k()

    temperatures, mass_fractions, mean_molar_masses, eddy_diffusion_coefficients = init_parameters()

    for i in range(n_runs):
        print(f"Emission, run {i + 1}/{n_runs}")
        performance_test_correlated_k_emission(
            radtrans=radtrans,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients
        )

    if trace_memory:
        petitRADTRANS.__debug.malloc_peak_snapshot(' Memory usage')

    for i in range(n_runs):
        print(f"Transmission, run {i + 1}/{n_runs}")
        performance_test_correlated_k_transmission(
            radtrans=radtrans,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients
        )

    if trace_memory:
        petitRADTRANS.__debug.malloc_peak_snapshot(' Memory usage')
        petitRADTRANS.__debug.malloc_top_lines_snapshot('Correlated-k memory usage')

    print(f"Done.")


def run(n_runs=7, trace_memory=True):
    if trace_memory:
        trace_memory_str = 'tm_on'
    else:
        trace_memory_str = 'tm_off'

    output_file = os.path.join(
        tests_results_directory,
        f'prof_ck_prt{petitRADTRANS.__version__}_{n_runs}runs_{trace_memory_str}.out'
    )

    if os.path.isfile(output_file):
        raise FileExistsError(f"file '{output_file}' already exists")

    pr = cProfile.Profile()
    pr.enable()

    if trace_memory:
        tracemalloc.start()

    main(n_runs, trace_memory)

    if trace_memory:
        tracemalloc.stop()

    pr.disable()
    pr.dump_stats(output_file)
