"""Test petitRADTRANS utility functions.
"""
import jax
import numpy as np

from .context import petitRADTRANS
from .utils import reference_filenames, test_parameters, temperature_guillot_2010

relative_tolerance = 1e-6


def test_guillot_2010_temperature_profile():
    # Load expected results
    reference_data = np.load(reference_filenames['guillot_2010'])
    print(f"Comparing generated spectrum to result from petitRADTRANS-{reference_data['prt_version']}...")
    temperature_ref = reference_data['temperature']
    pressure_ref = reference_data['pressure']

    # Check if temperature is as expected
    assert np.allclose(
        test_parameters['pressures'],
        pressure_ref,
        rtol=relative_tolerance,
        atol=0
    )

    assert np.allclose(
        temperature_guillot_2010,
        temperature_ref,
        rtol=relative_tolerance,
        atol=0
    )


def test_planck_function():
    frequencies = petitRADTRANS.physical_constants.c / np.linspace(
        test_parameters['spectrum_parameters']['wavelength_range_correlated_k'][0] * 1e-4,
        test_parameters['spectrum_parameters']['wavelength_range_correlated_k'][1] * 1e-4,
        10
    )

    planck = petitRADTRANS.physics.planck_function_hz(
        test_parameters['stellar_parameters']['effective_temperature'],
        frequencies
    )

    reference_spectral_radiance = np.array([
        3.65580443e-05, 3.64376693e-05, 3.62173874e-05, 3.59133996e-05, 3.55399032e-05,
        3.51092680e-05, 3.46322194e-05, 3.41180180e-05, 3.35746299e-05, 3.30088840e-05
    ])  # (erg.s-1.sr-1.cm-2.Hz-1)

    assert np.allclose(
        planck,
        reference_spectral_radiance,
        rtol=relative_tolerance,
        atol=0
    )


def test_stellar_model():
    phoenix_star_table = petitRADTRANS.stellar_spectra.phoenix.PhoenixStarTable()
    stellar_spectrum, _ = phoenix_star_table.compute_spectrum(test_parameters['stellar_parameters']['effective_temperature'])

    reference_spectral_radiosity = np.array([
        7.66275355e-114, 1.93595034e-058, 9.90294639e-023, 2.70611283e-008, 8.19891143e-005,
        1.05038448e-004, 2.47953373e-005, 3.87505010e-006, 5.57445915e-007, 7.72638833e-008
    ])  # (erg.s-1.cm-2.Hz-1)

    reference_wavenumber = np.array([
        1.0002808e-06, 2.6753539e-06, 7.1555055e-06, 1.9138131e-05, 5.1186878e-05,
        1.3690452e-04, 3.6616507e-04, 9.7934564e-04, 2.6193606e-03, 7.0057423e-03
    ])  # (cm)

    assert np.allclose(
        stellar_spectrum[1::3500, 0],
        reference_wavenumber,
        rtol=relative_tolerance,
        atol=0
    )

    assert np.allclose(
        stellar_spectrum[1::3500, 1],
        reference_spectral_radiosity,
        rtol=relative_tolerance,
        atol=0
    )


def test_stellar_model_jit():
    phoenix_star_table_eager = petitRADTRANS.stellar_spectra.phoenix.PhoenixStarTable()
    phoenix_star_table_jit = petitRADTRANS.stellar_spectra.phoenix.PhoenixStarTable()

    @jax.jit
    def compute_stellar_model(temperature):
        return phoenix_star_table_jit.compute_spectrum(temperature)

    temperature = test_parameters['stellar_parameters']['effective_temperature']
    stellar_spectrum_eager, radius_eager = phoenix_star_table_eager.compute_spectrum(temperature)
    stellar_spectrum_jit, radius_jit = compute_stellar_model(temperature)

    assert np.allclose(stellar_spectrum_jit, stellar_spectrum_eager, rtol=relative_tolerance, atol=0)
    assert np.allclose(radius_jit, radius_eager, rtol=relative_tolerance, atol=0)
