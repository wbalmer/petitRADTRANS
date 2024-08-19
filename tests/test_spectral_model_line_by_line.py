import copy
from functools import partial

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import get_main_model_parameters, temperature_isothermal, test_parameters

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def init_spectral_model_line_by_line():
    mass_fractions = {
        species: mass_fraction
        for species, mass_fraction in test_parameters['mass_fractions_line_by_line'].items()
        if species not in test_parameters['filling_species']
    }

    spectral_model = petitRADTRANS.spectral_model.SpectralModel(
        # Radtrans parameters
        pressures=test_parameters['pressures'],
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
        # Velocity paramters
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
        temperature=temperature_isothermal,
        # Mass fractions
        use_equilibrium_chemistry=False,
        imposed_mass_fractions=mass_fractions,
        filling_species=test_parameters['filling_species'],
        # Observation parameters
        is_around_star=True,
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],  # K
        rebinned_wavelengths=petitRADTRANS.math.resolving_space(
            test_parameters['spectrum_parameters']['wavelength_range_line_by_line'][0] * 1e-4,
            test_parameters['spectrum_parameters']['wavelength_range_line_by_line'][1] * 1e-4,
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


spectral_model_lbl_full = init_spectral_model_line_by_line()


def test_line_by_line_spectral_model_emission_full_all():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        update_parameters=True,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=True
    )


def test_line_by_line_spectral_model_transmission_full():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True
    )


def test_line_by_line_spectral_model_transmission_full_all():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=True
    )


def test_line_by_line_spectral_model_transmission_full_convolved():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        convolve=True
    )


def test_line_by_line_spectral_model_transmission_full_light_loss():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        scale=True,  # scale must be True for transit light loss to work
        shift=True,  # shift must be True for transit light loss to work
        use_transit_light_loss=True
    )


def test_line_by_line_spectral_model_transmission_full_scaled():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        scale=True
    )


def test_line_by_line_spectral_model_transmission_full_shifted():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        shift=True
    )


def test_line_by_line_spectral_model_transmission_full_rebinned():
    spectral_model = copy.deepcopy(spectral_model_lbl_full)

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True,
        rebin=True
    )
