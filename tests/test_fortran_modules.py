"""Test the spectrum_utility modules.
"""
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
