"""Test the JAX/native replacements of the former Fortran modules.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import reference_filenames
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
    # Cross-validation against the legacy Fortran backend; only runs when the
    # (no longer built by default) f2py extension is available.
    fcore_module = pytest.importorskip(
        "petitRADTRANS.fortran_radtrans_core",
        reason="legacy Fortran extension is not built",
    )
    fcore = fcore_module.fortran_radtrans_core

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


@pytest.mark.parametrize("scattering_in_transmission", [False, True])
def test_compute_transmission_spectrum_contribution_matches_previous_leave_one_out_algebra(
    scattering_in_transmission,
):
    def _safe_sqrt_positive(values):
        positive_values = values > 0.0
        safe_values = jnp.where(
            positive_values,
            values,
            jax.lax.stop_gradient(jnp.ones_like(values)),
        )
        return jnp.where(positive_values, jnp.sqrt(safe_values), 0.0)

    key = jax.random.key(123)
    subkeys = jax.random.split(key, 5)

    opacities = jax.random.uniform(
        subkeys[0],
        (3, 4, 2, 6),
        minval=1e-8,
        maxval=1e2,
        dtype=jnp.float32,
    )
    temperatures = jax.random.uniform(
        subkeys[1],
        (6,),
        minval=500.0,
        maxval=1500.0,
        dtype=jnp.float32,
    )
    pressures = jnp.logspace(-6, 2, 6, dtype=jnp.float32)
    reference_gravity = jnp.array(1e3, dtype=jnp.float32)
    mean_molar_masses = jax.random.uniform(
        subkeys[2],
        (6,),
        minval=2.0,
        maxval=3.0,
        dtype=jnp.float32,
    )
    reference_pressure = jnp.array(1.0, dtype=jnp.float32)
    planet_radius = jnp.array(7e9, dtype=jnp.float32)
    weights_gauss = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    transit_radii_squared = jax.random.uniform(
        subkeys[3],
        (4,),
        minval=1e18,
        maxval=1e20,
        dtype=jnp.float32,
    )
    continuum_opacities_scattering = jax.random.uniform(
        subkeys[4],
        (4, 6),
        minval=1e-10,
        maxval=1e-2,
        dtype=jnp.float32,
    )

    contribution, radius_hydrostatic_equilibrium = (
        rtc.compute_transmission_spectrum_contribution(
            opacities,
            temperatures,
            pressures,
            reference_gravity,
            mean_molar_masses,
            reference_pressure,
            planet_radius,
            weights_gauss,
            transit_radii_squared,
            continuum_opacities_scattering,
            scattering_in_transmission,
            True,
        )
    )

    pressures_cgs = pressures * 1e6
    reference_pressure_cgs = reference_pressure * 1e6
    rho = (mean_molar_masses * petitRADTRANS.physical_constants.amu * pressures_cgs) / (
        petitRADTRANS.physical_constants.kB * temperatures
    )
    reference_radius = rtc.compute_radius_hydrostatic_equilibrium(
        pressures_cgs,
        reference_gravity,
        rho,
        reference_pressure_cgs,
        planet_radius,
        True,
    )
    radius_squared = reference_radius**2

    rho_work = jnp.asarray(rho, dtype=rtc.WORKING_JAX_DTYPE)
    opacities_work = jnp.asarray(opacities, dtype=rtc.WORKING_JAX_DTYPE)
    continuum_work = jnp.asarray(
        continuum_opacities_scattering,
        dtype=rtc.WORKING_JAX_DTYPE,
    )
    weights_work = jnp.asarray(weights_gauss, dtype=rtc.WORKING_JAX_DTYPE)

    opacity_times_rho = opacities_work * rho_work[None, None, None, :]
    pair_opacity_times_rho = opacity_times_rho[..., :-1] + opacity_times_rho[..., 1:]

    impact_radius_squared = radius_squared[1:, None]
    upper_shell_boundaries = radius_squared[:-1][None, :]
    lower_shell_boundaries = radius_squared[1:][None, :]
    diff_s_active = _safe_sqrt_positive(
        upper_shell_boundaries - impact_radius_squared
    ) - _safe_sqrt_positive(lower_shell_boundaries - impact_radius_squared)
    diff_s_active = jnp.asarray(diff_s_active, dtype=rtc.WORKING_JAX_DTYPE)

    gas_optical_depths_active = jnp.einsum(
        "gfsl,ml->gfsm", pair_opacity_times_rho, diff_s_active, optimize=True
    )

    if scattering_in_transmission:
        continuum_times_rho = continuum_work * rho_work[None, :]
        pair_continuum_times_rho = (
            continuum_times_rho[:, :-1] + continuum_times_rho[:, 1:]
        )
        scattering_optical_depths_active = jnp.einsum(
            "fl,ml->fm", pair_continuum_times_rho, diff_s_active, optimize=True
        )
        continuum_times_rho_by_layer = jnp.swapaxes(continuum_times_rho, 0, 1)
    else:
        scattering_optical_depths_active = jnp.zeros(
            (opacities.shape[1], opacities.shape[3] - 1), dtype=opacities.dtype
        )
        continuum_times_rho_by_layer = jnp.zeros(
            (opacities.shape[3], opacities.shape[1]), dtype=opacities.dtype
        )

    layer_path_weights = jnp.pad(diff_s_active, ((0, 0), (1, 0))) + jnp.pad(
        diff_s_active, ((0, 0), (0, 1))
    )
    layer_path_weights_by_layer = jnp.swapaxes(layer_path_weights, 0, 1)
    opacity_times_rho_by_layer = jnp.moveaxis(opacity_times_rho, -1, 0)
    weights_gauss_expanded = weights_work[:, None, None, None]
    annulus_radius_increments = reference_radius[:-1] - reference_radius[1:]
    top_of_atmosphere_transmission = jnp.ones(
        (opacities.shape[1], 1), dtype=rtc.WORKING_JAX_DTYPE
    )

    def _reference_leave_one_out(i_leave_str_idx):
        path_by_layer = layer_path_weights_by_layer[i_leave_str_idx]
        gas_optical_depths_leave_one_out = gas_optical_depths_active - (
            opacity_times_rho_by_layer[i_leave_str_idx][..., None] * path_by_layer
        )

        gas_transmission_active = jnp.exp(-gas_optical_depths_leave_one_out)
        gas_transmission_by_species = jnp.sum(
            gas_transmission_active * weights_gauss_expanded,
            axis=0,
        )
        total_transmission_active = jnp.prod(gas_transmission_by_species, axis=1)

        if scattering_in_transmission:
            scattering_optical_depths_leave_one_out = (
                scattering_optical_depths_active
                - continuum_times_rho_by_layer[i_leave_str_idx][:, None] * path_by_layer
            )
            total_transmission_active = total_transmission_active * jnp.exp(
                -scattering_optical_depths_leave_one_out
            )

        total_transmission = jnp.concatenate(
            (top_of_atmosphere_transmission, total_transmission_active),
            axis=1,
        )
        effective_area_fraction = 1.0 - total_transmission
        annulus_terms = (
            effective_area_fraction[:, 1:] * reference_radius[None, 1:]
            + effective_area_fraction[:, :-1] * reference_radius[None, :-1]
        )
        transit_radii_leave_one_out_squared = (
            jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)
            + radius_squared[-1]
        )
        return transit_radii_squared - transit_radii_leave_one_out_squared

    reference_contribution_raw = jax.lax.map(
        _reference_leave_one_out,
        jnp.arange(opacities.shape[3], dtype=jnp.int32),
    )
    reference_sum = jnp.sum(reference_contribution_raw, axis=0, keepdims=True)
    reference_contribution = jnp.where(
        reference_sum == 0,
        0.0,
        reference_contribution_raw / reference_sum,
    )

    np.testing.assert_allclose(
        np.asarray(contribution),
        np.asarray(reference_contribution),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(radius_hydrostatic_equilibrium),
        np.asarray(reference_radius),
        rtol=1e-6,
        atol=1e-7,
    )


def test_combine_ck_opacities_cpu_float32_matches_threshold_reference():
    # Exercises the native FFI forward path on CPU, including the
    # weak-species threshold skip inherited from the Fortran kernel.
    if jax.default_backend() != "cpu":
        pytest.skip("native FFI forward path is CPU-only")

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
