import cProfile
import os
import tracemalloc

import numpy as np

from .context import petitRADTRANS
import petitRADTRANS.__debug
from .utils import test_parameters, tests_results_directory, temperature_isothermal


def init_spectral_model_line_by_line():
    mass_fractions = {
        species: mass_fraction[0]
        for species, mass_fraction in test_parameters['mass_fractions_line_by_line'].items()
        if species not in test_parameters['filling_species']
    }

    spectral_model = petitRADTRANS.spectral_model.SpectralModel(
        # Radtrans parameters
        pressures=test_parameters['pressures_performance'],
        line_species=test_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        line_opacity_mode='lbl',
        # SpectralModel parameters
        # Planet parameters
        planet_radius=test_parameters[
                'planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,  # cm
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,  # cm
        transit_duration=test_parameters['planetary_parameters']['transit_duration'],
        orbital_period=test_parameters['planetary_parameters']['orbital_period'],
        # Velocity parameters
        star_mass=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.m_sun,  # g
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],  # cm
        orbital_inclination=test_parameters['planetary_parameters']['orbital_inclination'],
        rest_frame_velocity_shift=test_parameters['mock_observation_parameters'][
                "rest_frame_velocity_shift"],  # cm.s-1
        system_observer_radial_velocities=np.linspace(
                test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][0],
                test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][1],
                test_parameters['mock_observation_parameters']['n_exposures']
            ),  # cm.s-1
        # Temperature profile parameters
        temperature_profile_mode='isothermal',
        temperature=temperature_isothermal[0],
        # Mass fractions
        use_equilibrium_chemistry=False,
        imposed_mass_fractions=mass_fractions,
        filling_species=test_parameters['filling_species'],
        # Observation parameters
        is_around_star=True,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],  # K
        rebinned_wavelengths=petitRADTRANS.math.resolving_space(
            test_parameters['spectrum_parameters']['wavelength_range_correlated_k'][0] * 1e-4,
            test_parameters['spectrum_parameters']['wavelength_range_correlated_k'][1] * 1e-4,
            test_parameters['mock_observation_parameters']['high_resolution_resolving_power']
        ),
        # (cm) used for the rebinning, and also to set the wavelengths boundaries
        convolve_resolving_power=test_parameters['mock_observation_parameters']['high_resolution_resolving_power'],
        mid_transit_time=0,
        times=2 * test_parameters['planetary_parameters']['transit_duration'] * (
                np.linspace(0, 1, test_parameters['mock_observation_parameters']['n_exposures']) - 0.5),
        # Preparation parameters
        tellurics_mask_threshold=test_parameters['preparing_parameters']['tellurics_mask_threshold'],
        polynomial_fit_degree=test_parameters['preparing_parameters']['polynomial_fit_degree']
    )

    return spectral_model


def performance_test_line_by_line_spectral_model_emission_full_all(spectral_model):
    spectral_model.calculate_spectrum(
        mode='emission',
        update_parameters=True,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=True
    )


def performance_test_line_by_line_spectral_model_transmission_full_all(spectral_model):
    spectral_model.calculate_spectrum(
        mode='transmission',
        update_parameters=True,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=True
    )


def main(n_runs, trace_memory):
    print(f"Initialisation...")
    spectral_model = init_spectral_model_line_by_line()

    for i in range(n_runs):
        print(f"Emission, run {i + 1}/{n_runs}")
        performance_test_line_by_line_spectral_model_emission_full_all(
            spectral_model=spectral_model
        )

    if trace_memory:
        petitRADTRANS.__debug.malloc_peak_snapshot(' Memory usage')

    for i in range(n_runs):
        print(f"Transmission, run {i + 1}/{n_runs}")
        performance_test_line_by_line_spectral_model_transmission_full_all(
            spectral_model=spectral_model
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
        f'prof_spectral_model_lbl_prt{petitRADTRANS.__version__}_{n_runs}runs_{trace_memory_str}.out'
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
