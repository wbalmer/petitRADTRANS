"""Test the spectrum_utility modules.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import reference_filenames
from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore
from petitRADTRANS.radtrans_core import radiative_transfer_core as rtc

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def test_rebin_spectrum():
    reference_data = np.load(reference_filenames['simple_spectrum'])

    benchmark = Benchmark(
        function=petitRADTRANS.radtrans_core.spectrum_utils.rebin_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        input_wavelengths=reference_data['wavelengths'],
        input_spectrum=reference_data['spectrum'],
        rebinned_wavelengths=reference_data['rebin_wavelengths']
    )


def test_rebin_spectrum_reverse_mode_differentiable():
    input_wavelengths = jnp.linspace(0.95, 2.05, 16)
    input_spectrum = jnp.sin(input_wavelengths) + 0.2 * input_wavelengths
    rebinned_wavelengths = jnp.linspace(1.05, 1.95, 8)

    def loss_fn(spectrum):
        rebinned = petitRADTRANS.radtrans_core.spectrum_utils.rebin_spectrum(
            input_wavelengths,
            spectrum,
            rebinned_wavelengths,
        )
        return jnp.sum(rebinned ** 2)

    gradient = jax.grad(loss_fn)(input_spectrum)

    assert gradient.shape == input_spectrum.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_rebin_spectrum_bin_preserves_float64_scan_dtype():
    input_wavelengths = jnp.linspace(0.95, 2.05, 16, dtype=jnp.float64)
    input_spectrum = (jnp.sin(input_wavelengths) + 0.2 * input_wavelengths).astype(jnp.float64)
    rebinned_wavelengths = jnp.linspace(1.05, 1.95, 8, dtype=jnp.float64)
    bin_widths = jnp.full(rebinned_wavelengths.shape, 0.1, dtype=jnp.float64)

    rebinned = petitRADTRANS.radtrans_core.spectrum_utils.rebin_spectrum_bin(
        input_wavelengths,
        input_spectrum,
        rebinned_wavelengths,
        bin_widths,
    )

    assert rebinned.dtype == jnp.float64
    assert rebinned.shape == rebinned_wavelengths.shape
    assert jnp.all(jnp.isfinite(rebinned))


def test_compute_radius_hydrostatic_equilibrium_reference_pressure_below_grid():
    pressures = np.logspace(-6, 2, 100, dtype=np.float64) * 1e6
    temperatures = np.linspace(500.0, 2000.0, pressures.size, dtype=np.float64)
    mean_molar_masses = np.full(pressures.size, 2.3, dtype=np.float64)
    atmosphere_densities = (
        pressures * mean_molar_masses * petitRADTRANS.physical_constants.amu
        / (petitRADTRANS.physical_constants.kB * temperatures)
    )

    reference_gravity = 1000.0
    reference_pressure = 1e-2
    planet_radius = 7e9

    jax_result = np.asarray(
        petitRADTRANS.radtrans_core.radiative_transfer_core.compute_radius_hydrostatic_equilibrium(
            jnp.array(pressures),
            reference_gravity,
            jnp.array(atmosphere_densities),
            jnp.array(reference_pressure),
            jnp.array(planet_radius),
            True,
        )
    )
    fortran_result = fcore.compute_radius_hydrostatic_equilibrium(
        pressures,
        reference_gravity,
        atmosphere_densities,
        reference_pressure,
        planet_radius,
        True,
        pressures.size,
    )

    assert np.allclose(jax_result, fortran_result, rtol=1e-8, atol=1e-10)


def test_combine_ck_opacities_jax_matches_fortran_with_continuum_threshold():
    g_gauss = jnp.array([0.05, 0.3, 0.7, 0.95], dtype=jnp.float64)
    weights_gauss = jnp.full((4,), 0.25, dtype=jnp.float64)

    opacities = jnp.array(
        [
            [[[1.0e-3], [2.0e-1]]],
            [[[2.0e-3], [2.5e-1]]],
            [[[3.0e-3], [3.0e-1]]],
            [[[4.0e-3], [3.5e-1]]],
        ],
        dtype=jnp.float64,
    )
    line_species_mass_fractions = jnp.array([[1.0, 1.0]], dtype=jnp.float64)
    continuum_opacities = jnp.array([[100.0]], dtype=jnp.float64)

    g_combination_weights, g_combination_weights_sum = rtc.build_ck_combination_state(
        weights_gauss
    )
    jax_result = np.asarray(
        rtc._combine_ck_opacities_jax(
            opacities,
            g_gauss,
            weights_gauss=weights_gauss,
            line_species_mass_fractions=line_species_mass_fractions,
            continuum_opacities=continuum_opacities,
            g_combination_weights=g_combination_weights,
            g_combination_weights_sum=g_combination_weights_sum,
            input_layout="ng_f_s_l",
        )
    )

    expected_if_species_skipped = np.asarray(opacities[:, 0, 0, 0] + continuum_opacities[0, 0])

    np.testing.assert_allclose(
        jax_result[:, 0, 0],
        expected_if_species_skipped,
        rtol=1e-12,
        atol=1e-12,
    )


def test_combine_ck_opacities_cpu_float32_callback_matches_fortran_reference():
    if jax.default_backend() != "cpu":
        pytest.skip("callback-path regression is CPU-only")

    g_gauss = jnp.array([0.05, 0.3, 0.7, 0.95], dtype=jnp.float32)
    weights_gauss = jnp.full((4,), 0.25, dtype=jnp.float32)
    opacities = jnp.array(
        [
            [[[1.0e-3], [2.0e-1]]],
            [[[2.0e-3], [2.5e-1]]],
            [[[3.0e-3], [3.0e-1]]],
            [[[4.0e-3], [3.5e-1]]],
        ],
        dtype=jnp.float32,
    )
    line_species_mass_fractions = jnp.array([[1.0, 1.0]], dtype=jnp.float32)
    continuum_opacities = jnp.array([[100.0]], dtype=jnp.float32)

    callback_result = np.array(
        rtc.combine_ck_opacities(
            opacities,
            g_gauss,
            weights_gauss=weights_gauss,
            line_species_mass_fractions=line_species_mass_fractions,
            continuum_opacities=continuum_opacities,
            input_layout="ng_f_s_l",
        ),
        copy=True,
    )

    expected_if_species_skipped = np.asarray(
        opacities[:, 0, 0, 0] + continuum_opacities[0, 0],
        dtype=np.float32,
    )[:, None, None]

    assert callback_result.dtype == np.float32
    np.testing.assert_allclose(
        callback_result,
        expected_if_species_skipped,
        rtol=5e-6,
        atol=5e-6,
    )
