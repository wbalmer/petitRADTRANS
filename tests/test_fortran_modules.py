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


def test_spline_continuum_subtraction_accepts_flat_knot_array():
    # A flat knot array has 0-d elements, which are still jnp.ndarray instances;
    # they must not be mistaken for a list of knot sets.
    wavelengths = jnp.linspace(1.0, 2.0, 300, dtype=jnp.float64)
    spectrum = 1.0 + 0.3 * wavelengths + 0.05 * jnp.sin(40.0 * wavelengths)

    filtered = petitRADTRANS.radtrans_core.spectrum_utils.continuum_subtract_spectrum(
        wavelengths, spectrum, {"continuum_subtract_nodes": 8}, method="spline"
    )

    assert filtered.shape == spectrum.shape
    assert jnp.all(jnp.isfinite(filtered))
    # The smooth continuum is removed and only the 0.05-amplitude sinusoid
    # remains. The two end samples coincide with the outermost nodes, which the
    # spline basis excludes (strict inequality), so they keep their raw value.
    assert float(jnp.max(jnp.abs(filtered[1:-1]))) < 0.1


@pytest.mark.parametrize("convolve_resolving_power", [1e5, 1e4, 1e3])
def test_convolve_constant_resolution_matches_scipy_gaussian_filter(convolve_resolving_power):
    # pRT 3 convolved with scipy.ndimage.gaussian_filter1d(mode="reflect"); the kernel must not be truncated before
    # 4 standard deviations, whatever their number of bins (~4 to ~400 here)
    from scipy.ndimage import gaussian_filter1d

    input_resolving_power = 1e6
    wavelengths = 2.3e-4 * (1 + 1 / input_resolving_power) ** np.arange(4000)
    spectrum = 1.0 + 0.3 * np.sin(np.arange(wavelengths.size) / 7.0) + 1e-3 * np.arange(wavelengths.size)

    convolved = petitRADTRANS.radtrans_core.spectrum_utils.convolve_constant_resolution(
        wavelengths, spectrum, convolve_resolving_power
    )

    sigma = np.mean(wavelengths[:-1] / np.diff(wavelengths) + 0.5) / convolve_resolving_power / (
        2 * np.sqrt(2 * np.log(2))
    )
    np.testing.assert_allclose(convolved, gaussian_filter1d(spectrum, sigma, mode="reflect"), rtol=1e-10)


def test_convolve_constant_resolution_under_jit_refuses_a_too_small_kernel():
    wavelengths = 2.3e-4 * (1 + 1e-6) ** np.arange(500)
    spectrum = 1.0 + 0.3 * np.sin(np.arange(wavelengths.size) / 7.0)
    convolve = jax.jit(petitRADTRANS.radtrans_core.spectrum_utils.convolve_constant_resolution)

    # sigma ~4 bins fits in the default traced kernel half-width
    np.testing.assert_allclose(
        convolve(wavelengths, spectrum, 1e5),
        petitRADTRANS.radtrans_core.spectrum_utils.convolve_constant_resolution(wavelengths, spectrum, 1e5),
        rtol=1e-12,
    )
    # sigma ~42 bins does not: the result is NaN instead of a silently truncated convolution
    assert np.all(np.isnan(convolve(wavelengths, spectrum, 1e4)))


def test_linear_spline_interpolation_knot_set_forms():
    x_samples = jnp.linspace(1.0, 2.0, 50, dtype=jnp.float64)
    spline_interpolation = petitRADTRANS.radtrans_core.spectrum_utils.linear_spline_interpolation

    single_set = spline_interpolation(jnp.linspace(1.0, 2.0, 6), x_samples)
    two_sets = spline_interpolation(
        [jnp.linspace(1.0, 1.4, 4), jnp.linspace(1.6, 2.0, 5)], x_samples
    )

    assert single_set.shape == (50, 6)
    assert two_sets.shape == (50, 9)


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

    def _reference_transit_radii_squared(gas_optical_depths, scattering_optical_depths):
        gas_transmission = jnp.exp(-gas_optical_depths)
        gas_transmission_by_species = jnp.sum(
            gas_transmission * weights_gauss_expanded,
            axis=0,
        )
        total_transmission_active = jnp.prod(gas_transmission_by_species, axis=1)

        if scattering_in_transmission:
            total_transmission_active = total_transmission_active * jnp.exp(
                -scattering_optical_depths
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
        return (
            jnp.sum(annulus_terms * annulus_radius_increments[None, :], axis=1)
            + radius_squared[-1]
        )

    # The baseline is computed with the same numerics as the leave-one-out
    # evaluations, matching the function's internal-baseline contract.
    transit_radii_baseline_squared = _reference_transit_radii_squared(
        gas_optical_depths_active, scattering_optical_depths_active
    )

    def _reference_leave_one_out(i_leave_str_idx):
        path_by_layer = layer_path_weights_by_layer[i_leave_str_idx]
        gas_optical_depths_leave_one_out = gas_optical_depths_active - (
            opacity_times_rho_by_layer[i_leave_str_idx][..., None] * path_by_layer
        )

        if scattering_in_transmission:
            scattering_optical_depths_leave_one_out = (
                scattering_optical_depths_active
                - continuum_times_rho_by_layer[i_leave_str_idx][:, None] * path_by_layer
            )
        else:
            scattering_optical_depths_leave_one_out = scattering_optical_depths_active

        transit_radii_leave_one_out_squared = _reference_transit_radii_squared(
            gas_optical_depths_leave_one_out,
            scattering_optical_depths_leave_one_out,
        )
        return transit_radii_baseline_squared - transit_radii_leave_one_out_squared

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


def _random_ck_mixing_inputs():
    # 16-point Gauss-Legendre g grid on [0, 1], opacities ascending along g as in the k-tables
    nodes, weights = np.polynomial.legendre.leggauss(16)
    rng = np.random.default_rng(0)
    n_g, n_frequencies, n_species, n_layers = 16, 7, 4, 5

    return {
        "opacities_fortran_order": jnp.asarray(
            np.sort(10 ** rng.uniform(-6, 2, (n_g, n_frequencies, n_species, n_layers)), axis=0)
        ),
        "g_gauss": jnp.asarray((nodes + 1) / 2),
        "weights_gauss": jnp.asarray(weights / 2),
        "line_species_mass_fractions": jnp.asarray(10 ** rng.uniform(-5, -1, (n_layers, n_species))),
        "continuum_opacities": jnp.asarray(10 ** rng.uniform(-6, -2, (n_frequencies, n_layers))),
    }


def test_combine_ck_opacities_cpu_kernel_matches_jax_implementation():
    # The native CPU kernel and the JAX implementation (GPU, and higher derivatives on CPU) must agree
    if jax.default_backend() != "cpu":
        pytest.skip("native FFI path is CPU-only")

    inputs = _random_ck_mixing_inputs()
    g_combination_weights, g_combination_weights_sum = rtc.build_ck_combination_state(inputs["weights_gauss"])

    native = rtc.combine_ck_opacities(**inputs)
    reference = rtc._combine_ck_opacities_jax(
        **inputs,
        g_combination_weights=g_combination_weights,
        g_combination_weights_sum=g_combination_weights_sum,
    )

    np.testing.assert_allclose(np.asarray(native), np.asarray(reference), rtol=1e-12, atol=0.0)


def test_combine_ck_opacities_cpu_kernel_derivatives_agree_across_modes():
    # Forward- and reverse-mode derivatives with respect to the abundances, first and second order
    # (forward over reverse is what the Laplace covariance and the Hessian-based NUTS mass matrix use)
    if jax.default_backend() != "cpu":
        pytest.skip("native FFI path is CPU-only")

    inputs = _random_ck_mixing_inputs()
    mass_fractions = inputs.pop("line_species_mass_fractions")

    def mixed_opacities(mass_fractions):
        return rtc.combine_ck_opacities(line_species_mass_fractions=mass_fractions, **inputs)

    def total_log_opacity(mass_fractions):
        return jnp.sum(jnp.log(mixed_opacities(mass_fractions)))

    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(mixed_opacities)(mass_fractions)),
        np.asarray(jax.jacrev(mixed_opacities)(mass_fractions)),
        rtol=1e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(jax.jacrev(total_log_opacity))(mass_fractions)),
        np.asarray(jax.jacrev(jax.jacrev(total_log_opacity))(mass_fractions)),
        rtol=1e-10,
        atol=0.0,
    )


def _ck_table_like_mixing_inputs():
    # pRT's split g grid (8 Gauss-Legendre points on [0, 0.9], 8 on [0.9, 1]), whose quadrature weights differ
    # by up to 30x, and opacities rising steeply toward g = 1 as in the k-tables
    nodes, weights = np.polynomial.legendre.leggauss(8)
    g_gauss = np.concatenate([0.45 * (nodes + 1), 0.9 + 0.05 * (nodes + 1)])
    weights_gauss = np.concatenate([0.45 * weights, 0.05 * weights])
    rng = np.random.default_rng(0)
    n_frequencies, n_species, n_layers = 7, 4, 5
    shape = (n_frequencies, n_species, n_layers)
    opacities = 10 ** rng.uniform(-4, 1, shape) * 10 ** (
        rng.uniform(1, 4, shape) * g_gauss[:, None, None, None] ** rng.uniform(1, 4, shape)
    )

    return {
        "opacities_fortran_order": jnp.asarray(opacities),
        "g_gauss": jnp.asarray(g_gauss),
        "weights_gauss": jnp.asarray(weights_gauss),
        "line_species_mass_fractions": jnp.asarray(10 ** rng.uniform(-4, -1, (n_layers, n_species))),
        "continuum_opacities": jnp.asarray(10 ** rng.uniform(-5, -2, (n_frequencies, n_layers))),
    }


def _integrated_derivative_and_change(function, start, stop, n_points=401):
    # Trapezoid integral of the derivative of a scalar-argument function over [start, stop], the change of its
    # value, and its value at start
    value_and_tangent = jax.jit(lambda t: jax.jvp(function, (t,), (jnp.ones_like(t),)))
    steps = np.linspace(start, stop, n_points)
    values, tangents = (np.array(a) for a in zip(*(value_and_tangent(jnp.asarray(t)) for t in steps)))

    return np.trapezoid(tangents, steps, axis=0), values[-1] - values[0], values[0]


def test_combine_ck_opacities_derivative_integrates_to_value_change():
    # The mixed opacities are continuous in the abundances, so their derivative along a path integrates to the
    # change of their value, up to the trapezoid error at the kinks. Interpolating the sorted pairwise sums at the
    # midpoints of their cumulative weights made the value jump whenever two sums of unequal weight swapped
    # places, and the derivative missed the jumps
    inputs = _ck_table_like_mixing_inputs()
    mass_fractions = inputs.pop("line_species_mass_fractions")
    direction = jnp.asarray(np.random.default_rng(1).choice([-1.0, 1.0], mass_fractions.shape))

    def mixed_opacities(log_scale):
        return rtc.combine_ck_opacities(
            line_species_mass_fractions=mass_fractions * 10 ** (log_scale * direction), **inputs
        )

    integral, change, start_value = _integrated_derivative_and_change(mixed_opacities, -0.05, 0.05)

    assert np.max(np.abs(integral - change) / np.abs(start_value)) < 2e-3


def test_combine_ck_opacities_switches_species_on_continuously():
    # A species is skipped below half the mixing threshold and switched on smoothly up to the threshold, so the
    # mixed opacities stay continuous as its abundance crosses it
    nodes, weights = np.polynomial.legendre.leggauss(16)
    g_gauss = jnp.asarray((nodes + 1) / 2)
    weights_gauss = jnp.asarray(weights / 2)
    # Species 0 plus the continuum is 1 at g = 0, so the threshold is 0.01, reached by species 1 at a mass
    # fraction of 0.01
    opacities = jnp.asarray(
        np.stack([np.geomspace(0.5, 20.0, 16), np.geomspace(1e-4, 1.0, 16)], axis=-1)[:, None, :, None]
    )
    continuum_opacities = jnp.asarray([[0.5]])

    def mixed_opacities(log_ratio):
        return rtc.combine_ck_opacities(
            opacities,
            g_gauss,
            weights_gauss=weights_gauss,
            line_species_mass_fractions=jnp.stack([jnp.ones_like(log_ratio), 1e-2 * 10 ** log_ratio])[None, :],
            continuum_opacities=continuum_opacities,
        )[:, 0, 0]

    integral, change, _ = _integrated_derivative_and_change(mixed_opacities, np.log10(0.3), np.log10(2.0))

    np.testing.assert_allclose(integral, change, rtol=0.0, atol=1e-3 * np.max(np.abs(change)))
    np.testing.assert_array_equal(
        np.asarray(mixed_opacities(jnp.log10(0.45))),
        np.asarray(opacities[:, 0, 0, 0] + continuum_opacities[0, 0]),
    )


def test_combine_ck_opacities_close_to_exact_random_overlap():
    # Under random overlap, the transmission of the mixture through any column is exactly the product of the
    # transmissions of the species and the continuum. Window means of the quantile function reproduce it more
    # than twice as accurately on average as interpolating the sorted pairwise sums did (mean error 6.8e-3, max
    # 4.0e-2 here)
    inputs = _ck_table_like_mixing_inputs()
    mixed = np.asarray(rtc.combine_ck_opacities(**inputs))  # (n_g, n_frequencies, n_layers)
    weights_gauss = np.asarray(inputs["weights_gauss"])
    weighted_opacities = (
        np.asarray(inputs["opacities_fortran_order"])
        * np.asarray(inputs["line_species_mass_fractions"]).T[None, None, :, :]
    )
    errors = []

    for column_scale in (0.1, 1.0, 10.0):
        columns = column_scale / np.median(mixed, axis=0)  # (n_frequencies, n_layers)
        transmission = np.einsum("g,gfl->fl", weights_gauss, np.exp(-columns * mixed))
        exact_transmission = np.exp(-columns * np.asarray(inputs["continuum_opacities"])) * np.prod(
            np.einsum("g,gfsl->fsl", weights_gauss, np.exp(-columns[None, :, None, :] * weighted_opacities)),
            axis=1,
        )
        errors.append(np.abs(transmission - exact_transmission))

    assert np.mean(errors) < 4e-3
    assert np.max(errors) < 3.5e-2


def test_combine_ck_opacities_cpu_kernel_derivatives_match_jax_implementation():
    # The native forward- and reverse-mode derivatives of the CPU kernel, also under vmap, against those
    # of the JAX implementation. The derivative through a species in the threshold ramp multiplies the
    # small change that species makes to the mixture, which both compute to ~1e-8 at worst
    if jax.default_backend() != "cpu":
        pytest.skip("native FFI path is CPU-only")

    inputs = _ck_table_like_mixing_inputs()
    mass_fractions = inputs.pop("line_species_mass_fractions")
    g_combination_weights, g_combination_weights_sum = rtc.build_ck_combination_state(inputs["weights_gauss"])
    rng = np.random.default_rng(2)

    def native(mass_fractions):
        return rtc.combine_ck_opacities(line_species_mass_fractions=mass_fractions, **inputs)

    def reference(mass_fractions):
        return rtc._combine_ck_opacities_jax(
            line_species_mass_fractions=mass_fractions,
            g_combination_weights=g_combination_weights,
            g_combination_weights_sum=g_combination_weights_sum,
            **inputs,
        )

    tangent = mass_fractions * jnp.asarray(rng.standard_normal(mass_fractions.shape))
    cotangent = jnp.asarray(rng.standard_normal(native(mass_fractions).shape))
    batch = jnp.stack([mass_fractions, 3 * mass_fractions])
    derivatives = {
        "forward": lambda f: jax.jvp(f, (mass_fractions,), (tangent,))[1],
        "reverse": lambda f: jax.vjp(f, mass_fractions)[1](cotangent)[0],
        "reverse, vmap": lambda f: jax.vmap(lambda m: jax.vjp(f, m)[1](cotangent)[0])(batch),
    }

    for mode, derivative in derivatives.items():
        expected = np.asarray(derivative(reference))
        np.testing.assert_allclose(
            np.asarray(derivative(native)), expected, rtol=0.0, atol=1e-10 * np.max(np.abs(expected)), err_msg=mode
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
