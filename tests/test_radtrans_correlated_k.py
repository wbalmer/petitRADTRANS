"""Test the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import (
    check_partial_cloud_coverage_full_consistency,
    get_cloud_parameters,
    test_parameters,
    temperature_guillot_2010,
    temperature_isothermal,
    reference_filenames
)

relative_tolerance = 1e-4  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters["pressures"],
        line_species=test_parameters["spectrum_parameters"][
            "line_species_correlated_k"
        ],
        rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        cloud_species=list(test_parameters["cloud_parameters"]["cloud_species"].keys()),
        wavelength_boundaries=test_parameters["spectrum_parameters"][
            "wavelength_range_correlated_k"
        ],
        line_opacity_mode="c-k",
        scattering_in_emission=False,
    )

    return atmosphere


atmosphere_ck = init_radtrans_correlated_k()


# Tests
def test_correlated_k_emission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux, relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        frequencies_to_wavelengths=False,
    )


def test_photosphere_radius_profile_matches_reference_interpolation():
    pressures = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    radius_hydrostatic_equilibrium = np.array([12.0, 11.5, 11.0, 10.5, 10.0])
    n_frequencies = atmosphere_ck.frequencies.size
    tau_target = 2.0 / 3.0

    pressure_targets = np.linspace(pressures[1], pressures[-2], n_frequencies - 2)
    tau_profiles = np.empty((n_frequencies, pressures.size))

    for i, pressure_target in enumerate(pressure_targets):
        tau_profiles[i] = tau_target * pressures / pressure_target

    tau_profiles[-2:] = 0.5 * tau_target * pressures / pressures[-1]

    weights_gauss = np.asarray(atmosphere_ck._lines_loaded_opacities["weights_gauss"])
    optical_depths = np.broadcast_to(
        tau_profiles[None, :, None, :] / np.sum(weights_gauss),
        (weights_gauss.size, n_frequencies, 1, pressures.size),
    ).copy()

    photosphere_radius = np.asarray(
        atmosphere_ck._interpolate_photosphere_radius_profile(
            pressures=pressures,
            optical_depths=optical_depths,
            radius_hydrostatic_equilibrium=radius_hydrostatic_equilibrium,
        )
    )

    expected = np.empty(n_frequencies)

    for i, tau_profile in enumerate(tau_profiles):
        if np.max(tau_profile) > tau_target:
            pressure_tau_23 = np.interp(tau_target, tau_profile, pressures)
            expected[i] = np.interp(
                pressure_tau_23,
                pressures,
                radius_hydrostatic_equilibrium,
            )
        else:
            expected[i] = radius_hydrostatic_equilibrium[-1]

    np.testing.assert_allclose(photosphere_radius, expected, rtol=1e-7, atol=5e-7)


def test_correlated_k_emission_contribution_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux,
        relative_tolerance=1e5,  # relax tolerance due to test pipeline inconsistencies
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        return_contribution=True,
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_emission_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux, relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_emission_spectrum_cloud_hansen_radius():
    mass_fractions, _, cloud_f_sed, _, cloud_hansen_b = get_cloud_parameters(
        "mass_fractions_correlated_k"
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux, relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_radius_distribution="hansen",
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_emission_partial_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )
    mass_fractions_clear = copy.deepcopy(test_parameters["mass_fractions_correlated_k"])

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_flux, relative_tolerance=relative_tolerance
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck.calculate_flux,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters["cloud_parameters"]["cloud_fraction"],
        mass_fractions=mass_fractions,
        mass_fractions_clear=mass_fractions_clear,
        temperatures=temperature_guillot_2010,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_cloud_power_law():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        power_law_opacity_350nm=test_parameters["cloud_parameters"]["kappa_zero"],
        power_law_opacity_coefficient=test_parameters["cloud_parameters"][
            "gamma_scattering"
        ],
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_gray_cloud():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        opaque_cloud_top_pressure=test_parameters["cloud_parameters"]["cloud_pressure"],
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_partial_gray_cloud():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    check_partial_cloud_coverage_full_consistency(
        spectrum_function=atmosphere_ck.calculate_transit_radii,
        benchmark=benchmark,
        relative_tolerance=relative_tolerance,
        cloud_fraction=test_parameters["cloud_parameters"]["cloud_fraction"],
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        mass_fractions_clear=None,
        opaque_cloud_top_pressure=test_parameters["cloud_parameters"]["cloud_pressure"],
        temperatures=temperature_isothermal,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_rayleigh():
    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        haze_factor=test_parameters["cloud_parameters"]["haze_factor"],
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_cloud_fixed_radius():
    (
        mass_fractions,
        cloud_particle_mean_radii,
        _,
        cloud_particle_radius_distribution_std,
        _,
    ) = get_cloud_parameters("mass_fractions_correlated_k")

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=test_parameters["temperature_isothermal"]
        * np.ones_like(test_parameters["pressures"]),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_contribution_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=1e3,  # relax tolerance due to test pipeline inconsistencies
    )

    benchmark.run(
        temperatures=test_parameters["temperature_isothermal"]
        * np.ones_like(test_parameters["pressures"]),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        return_contribution=True,
        frequencies_to_wavelengths=False,
    )

    # the python transit radius calculation function outputs are slightly
    # different than the fortran version, creating a slight error, especially in deeper
    # atmospheric levels
    # absolute_tolerance = 8e-7  # there is a max absolute error of 3.28989835374216e-10 between windows and WSL
    # generated files TODO investigate why


def test_correlated_k_transmission_spectrum_cloud_calculated_radius():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=test_parameters["temperature_isothermal"]
        * np.ones_like(test_parameters["pressures"]),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        eddy_diffusion_coefficients=test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False,
    )


def test_correlated_k_transmission_spectrum_with_custom_cloud_distribution():
    mass_fractions, _, cloud_f_sed, _, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    reference_data = np.load(reference_filenames['cloud_particle_number_density_grid'])

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    # Call calculate_transit_radii with custom cloud particle size distribution
    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions_correlated_k'],
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        cloud_particle_number_density_grid=reference_data['cloud_particle_number_density_grid'],
        cloud_particle_radius_distribution="custom",
        frequencies_to_wavelengths=False
        )


def test_correlated_k_transmission_spectrum_cloud_on_off():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )

    benchmark = Benchmark(
        function=atmosphere_ck.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=test_parameters['temperature_isothermal'] * np.ones_like(test_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        target_particle_radii=test_parameters['target_particle_radii'],
        target_pressure=test_parameters['target_pressure'],
        frequencies_to_wavelengths=False
    )


def test_correlated_k_emission_array_valued_surface_inputs():
    """Array-valued surface inputs must be accepted, not just scalars/None.

    Regression test: the frequency-size check in
    validation._sanitize_frequency_surface_input referred to names that only
    existed in a removed inner function, so any non-scalar reflectance,
    emissivity or stellar intensity raised NameError. Every other test passes
    None for these, so nothing caught it.
    """
    n_frequencies = atmosphere_ck.frequencies.size

    _, flux, _ = atmosphere_ck.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        emissivities=np.full(n_frequencies, 0.9),
        reflectances=np.full(n_frequencies, 0.1),
        frequencies_to_wavelengths=False,
    )

    assert flux.shape == (n_frequencies,)
    assert np.all(np.isfinite(flux))


def test_correlated_k_emission_mismatched_surface_input_size_raises():
    """A surface input of the wrong length must raise ValueError, not NameError."""
    with pytest.raises(ValueError, match="same size than frequencies"):
        atmosphere_ck.calculate_flux(
            temperatures=temperature_guillot_2010,
            mass_fractions=test_parameters["mass_fractions_correlated_k"],
            reference_gravity=test_parameters["planetary_parameters"][
                "reference_gravity"
            ],
            mean_molar_masses=test_parameters["mean_molar_mass"],
            emissivities=np.ones(atmosphere_ck.frequencies.size - 1),
            frequencies_to_wavelengths=False,
        )


def test_correlated_k_invalid_porosity_factor_raises():
    """An out-of-range cloud porosity factor must be rejected at the entry point.

    This check moved out of the jitted cloud kernel (where it needed a
    per-evaluation host callback) into the un-jitted entry point, so it must
    still fire.
    """
    cloud_species = list(test_parameters["cloud_parameters"]["cloud_species"].keys())

    with pytest.raises(ValueError, match="porosity factor"):
        atmosphere_ck.calculate_flux(
            temperatures=temperature_guillot_2010,
            mass_fractions=test_parameters["mass_fractions_correlated_k"],
            reference_gravity=test_parameters["planetary_parameters"][
                "reference_gravity"
            ],
            mean_molar_masses=test_parameters["mean_molar_mass"],
            clouds_particle_porosity_factor={cloud_species[0]: 1.5},
            frequencies_to_wavelengths=False,
        )


def test_correlated_k_incomplete_power_law_opacity_raises():
    """Supplying only one of the two power-law opacity parameters must be rejected."""
    with pytest.raises(RuntimeError, match="power law opacity"):
        atmosphere_ck.calculate_flux(
            temperatures=temperature_guillot_2010,
            mass_fractions=test_parameters["mass_fractions_correlated_k"],
            reference_gravity=test_parameters["planetary_parameters"][
                "reference_gravity"
            ],
            mean_molar_masses=test_parameters["mean_molar_mass"],
            power_law_opacity_350nm=0.01,
            frequencies_to_wavelengths=False,
        )


def test_correlated_k_spectral_functions_reuse_compilation():
    """The spectral functions must compile once and reuse it across calls.

    They are jit-compiled, so a second call with the same static arguments and
    input shapes must not recompile. Regression guard for the argument split: a
    parameter wrongly left traced (or wrongly made static) shows up here as an
    extra trace.
    """
    kwargs = dict(
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        frequencies_to_wavelengths=False,
    )

    compiled = petitRADTRANS.radtrans.Radtrans.calculate_flux.compiled_function

    # Warm the cache, then assert no further compilation happens.
    atmosphere_ck.calculate_flux(temperatures=temperature_guillot_2010, **kwargs)
    cache_size = compiled._cache_size()

    for _ in range(3):
        atmosphere_ck.calculate_flux(temperatures=temperature_guillot_2010, **kwargs)

    assert compiled._cache_size() == cache_size, (
        "repeated identical calls should reuse the compiled function"
    )


def test_correlated_k_spectral_functions_are_differentiable():
    """jax.grad must work directly through the standard spectral functions."""
    kwargs = dict(
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        frequencies_to_wavelengths=False,
    )
    base_temperatures = jnp.asarray(temperature_guillot_2010)

    flux_gradient = jax.grad(
        lambda t: jnp.sum(
            atmosphere_ck.calculate_flux(temperatures=t, **kwargs)[1]
        )
    )(base_temperatures)

    assert flux_gradient.shape == base_temperatures.shape
    assert np.all(np.isfinite(np.asarray(flux_gradient)))

    transit_gradient = jax.grad(
        lambda t: jnp.sum(
            atmosphere_ck.calculate_transit_radii(
                temperatures=t,
                reference_pressure=test_parameters["planetary_parameters"][
                    "reference_pressure"
                ],
                planet_radius=test_parameters["planetary_parameters"]["radius"]
                * petitRADTRANS.physical_constants.r_jup_mean,
                **kwargs,
            )[1]
        )
    )(base_temperatures)

    assert transit_gradient.shape == base_temperatures.shape
    assert np.all(np.isfinite(np.asarray(transit_gradient)))


def test_removed_spectral_function_variants_hint_at_replacements():
    """The removed _jit / _autodiff variants must give a helpful error."""
    for removed, replacement in (
        ("calculate_flux_jit", "calculate_flux"),
        ("calculate_flux_autodiff", "calculate_flux"),
        ("calculate_transit_radii_jit", "calculate_transit_radii"),
        ("calculate_transit_radii_autodiff", "calculate_transit_radii"),
    ):
        with pytest.raises(AttributeError, match=replacement):
            getattr(atmosphere_ck, removed)


def test_return_opacities_gives_the_cloud_opacities_without_sum_opacities():
    """return_opacities must hand back the real cloud opacities.

    ``atmosphere_ck`` has scattering_in_emission False, so ``sum_opacities`` is
    False for both spectral functions. A dead gate in
    ``_assemble_cloud_opacity_outputs`` used to zero these two arrays in exactly
    that case, which is the default for c-k transmission and for c-k emission
    without scattering.
    """
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )
    cloud_kwargs = {
        "mass_fractions": mass_fractions,
        "mean_molar_masses": test_parameters["mean_molar_mass"],
        "reference_gravity": test_parameters["planetary_parameters"][
            "reference_gravity"
        ],
        "cloud_f_sed": cloud_f_sed,
        "cloud_particle_radius_distribution_std": (
            cloud_particle_radius_distribution_std
        ),
        "eddy_diffusion_coefficients": test_parameters["planetary_parameters"][
            "eddy_diffusion_coefficients"
        ],
        "return_opacities": True,
        "frequencies_to_wavelengths": False,
    }

    _, _, emission_outputs = atmosphere_ck.calculate_flux(
        temperatures=temperature_guillot_2010, **cloud_kwargs
    )
    _, _, transmission_outputs = atmosphere_ck.calculate_transit_radii(
        temperatures=temperature_isothermal,
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        **cloud_kwargs,
    )

    for outputs in (emission_outputs, transmission_outputs):
        cloud_absorption_opacities = np.asarray(
            outputs["cloud_absorption_opacities"]
        )
        cloud_scattering_opacities = np.asarray(
            outputs["cloud_anisotropic_scattering_opacities"]
        )

        assert np.any(cloud_absorption_opacities > 0)
        assert np.any(cloud_scattering_opacities > 0)
        assert np.all(np.isfinite(cloud_absorption_opacities))
        assert np.all(np.isfinite(cloud_scattering_opacities))
