import copy
from functools import partial
import tempfile

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import get_main_model_parameters, test_parameters

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def init_spectral_model_correlated_k():
    mass_fractions = {
        species: mass_fraction
        for species, mass_fraction in test_parameters['mass_fractions_correlated_k'].items()
        if species not in test_parameters['filling_species']
    }

    spectral_model = petitRADTRANS.spectral_model.SpectralModel(
        # Radtrans parameters
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        scattering_in_emission=True,
        line_opacity_mode='c-k',
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        # SpectralModel parameters
        # Planet parameters
        planet_radius=test_parameters[
                'planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,  # cm
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        # Star, system, orbit
        is_observed=True,  # return the flux observed at system_distance
        is_around_star=True,  # if True, calculate a PHOENIX stellar spectrum and add it to the emission spectrum
        system_distance=test_parameters['stellar_parameters'][
            'system_distance'] * petitRADTRANS.physical_constants.s_cst.light_year * 1e2,  # m to cm
        star_effective_temperature=test_parameters['stellar_parameters']['effective_temperature'],  # K
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,  # cm
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],  # cm
        # Temperature profile parameters
        temperature_profile_mode='guillot',
        temperature=test_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature'],
        intrinsic_temperature=test_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
        guillot_temperature_profile_gamma=test_parameters['temperature_guillot_2010_parameters']['gamma'],
        guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=test_parameters['temperature_guillot_2010_parameters'][
                'infrared_mean_opacity'],
        # Mass fractions
        use_equilibrium_chemistry=True,
        metallicity=10 ** test_parameters['chemical_parameters']['metallicities'][1],
        co_ratio=test_parameters['chemical_parameters']['c_o_ratios'][0],
        imposed_mass_fractions=mass_fractions,
        filling_species=test_parameters['filling_species']
    )

    return spectral_model


spectral_model_ck = init_spectral_model_correlated_k()


def test_correlated_k_spectral_model_emission():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = False

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_emission_equilibrium_chemistry():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = True
    spectral_model.model_parameters['imposed_mass_fractions'] = None

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_emission_equilibrium_chemistry_imposed():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = True
    spectral_model.model_parameters['imposed_mass_fractions'] = {
        species: np.logspace(
            test_parameters['mass_fractions_test'][0],
            test_parameters['mass_fractions_test'][1],
            spectral_model.pressures.size,
        )
        for species in spectral_model.line_species
    }

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_emission_observed():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = True
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = False

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_emission_with_star():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = True
    spectral_model.model_parameters['use_equilibrium_chemistry'] = False

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )
    benchmark.run(
        mode='emission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_transmission():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = False

    calculate_spectrum = partial(get_main_model_parameters, spectral_model=spectral_model)

    benchmark = Benchmark(
        function=calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        update_parameters=True
    )


def test_correlated_k_spectral_model_transmission_save_load():
    spectral_model = copy.deepcopy(spectral_model_ck)
    spectral_model.model_parameters['is_observed'] = False
    spectral_model.model_parameters['is_around_star'] = False
    spectral_model.model_parameters['star_effective_temperature'] = None
    spectral_model.model_parameters['star_radius'] = None
    spectral_model.model_parameters['use_equilibrium_chemistry'] = False

    wavelengths, spectrum = spectral_model.calculate_spectrum(
        mode='transmission',
        update_parameters=True
    )

    with tempfile.TemporaryFile() as f:
        spectral_model.save(f)
        loaded_spectral_model = petitRADTRANS.spectral_model.SpectralModel.load(f)

    loaded_wavelengths, loaded_spectrum = loaded_spectral_model.calculate_spectrum(
        **loaded_spectral_model.model_parameters['modification_parameters']
    )

    assert np.allclose(
        wavelengths,
        loaded_wavelengths,
        atol=0,
        rtol=relative_tolerance
    )

    assert np.allclose(
        spectrum,
        loaded_spectrum,
        atol=0,
        rtol=relative_tolerance
    )
