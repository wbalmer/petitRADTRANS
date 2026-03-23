"""Test the spectrum_utility modules.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import reference_filenames
from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore

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
