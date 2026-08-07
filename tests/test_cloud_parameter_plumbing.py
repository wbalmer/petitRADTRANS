import numpy as np
import jax.numpy as jnp

from .context import petitRADTRANS


def test_sanitize_flux_opacity_inputs_accepts_radius_std_mapping():
    sanitized_inputs = petitRADTRANS.validation._sanitize_flux_opacity_inputs(
        patchy_clouds=("MgSiO3(c)",),
        cloud_particle_mean_radii=None,
        cloud_hansen_a={
            "MgSiO3(c)": jnp.array([1.0e-4, 2.0e-4]),
            "Fe(c)": jnp.array([3.0e-4, 4.0e-4]),
        },
        cloud_hansen_b={
            "Fe(c)": 1.5,
            "MgSiO3(c)": jnp.array([1.1, 1.2]),
        },
        clouds_particle_porosity_factor=None,
        cloud_f_sed=2.0,
        eddy_diffusion_coefficients=None,
        opaque_cloud_top_pressure=None,
        cloud_particle_radius_distribution_std={
            "Fe(c)": 1.4,
            "MgSiO3(c)": jnp.array([1.05, 1.1]),
        },
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        cloud_particle_number_density_grid=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        target_particle_radii=None,
        target_pressure=None,
        cloud_species=("MgSiO3(c)", "Fe(c)"),
        missing_float=petitRADTRANS.radtrans._MISSING_FLOAT,
        missing_neg_inf=petitRADTRANS.radtrans._MISSING_NEG_INF,
        additional_opacities_placeholder_fn=petitRADTRANS.radtrans._additional_opacities_placeholder_fn,
    )

    assert isinstance(
        sanitized_inputs["cloud_particle_radius_distribution_std"],
        dict,
    )
    np.testing.assert_allclose(
        np.asarray(
            sanitized_inputs["cloud_particle_radius_distribution_std"]["Fe(c)"]
        ),
        np.asarray(1.4),
    )
    np.testing.assert_allclose(
        np.asarray(
            sanitized_inputs["cloud_particle_radius_distribution_std"]["MgSiO3(c)"]
        ),
        np.asarray([1.05, 1.1]),
    )


def test_initialize_hansen_b_respects_species_order():
    pressures = jnp.array([1.0e6, 1.0e7])
    cloud_hansen_b = {
        "MgSiO3(c)": jnp.array([1.1, 1.2]),
        "Fe(c)": 1.5,
    }

    standardized = petitRADTRANS.opacities.clouds._initialize_hansen_b(
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_radius_distribution="hansen",
        cloud_species=("Fe(c)", "MgSiO3(c)"),
        pressures=pressures,
    )

    np.testing.assert_allclose(
        np.asarray(standardized),
        np.asarray(
            [
                [1.5, 1.1],
                [1.5, 1.2],
            ]
        ),
    )


def test_initialize_cloud_species_inputs_respects_hansen_a_mapping():
    pressures = jnp.array([1.0e6, 1.0e7])
    mass_fractions = {
        "Fe(c)": jnp.array([1.0e-4, 1.0e-4]),
        "MgSiO3(c)": jnp.array([2.0e-4, 2.0e-4]),
    }
    clouds_loaded_opacities = {
        "particle_densities": jnp.array([3.0, 4.0]),
    }

    _, _, mean_radii = petitRADTRANS.opacities.clouds._initialize_cloud_species_inputs(
        cloud_species=("Fe(c)", "MgSiO3(c)"),
        cloud_species_mass_fractions=mass_fractions,
        pressures=pressures,
        clouds_loaded_opacities=clouds_loaded_opacities,
        clouds_particle_porosity_factor=None,
        cloud_particle_mean_radii=None,
        cloud_hansen_a={
            "MgSiO3(c)": jnp.array([2.1e-4, 2.2e-4]),
            "Fe(c)": 1.5e-4,
        },
    )

    np.testing.assert_allclose(
        np.asarray(mean_radii),
        np.asarray(
            [
                [1.5e-4, 2.1e-4],
                [1.5e-4, 2.2e-4],
            ]
        ),
    )


def test_compute_cloud_particle_mean_radius_accepts_per_species_width_grid():
    mean_radii = (
        petitRADTRANS.radtrans_core.cloud_core.compute_cloud_particle_mean_radius(
            reference_gravity=1.0e3,
            atmospheric_densities=jnp.array([1.0e-9, 2.0e-9]),
            clouds_particle_densities=jnp.array([3.0, 3.0]),
            temperatures=jnp.array([900.0, 1100.0]),
            mean_molar_masses=jnp.array([2.3, 2.3]),
            f_seds=jnp.ones((2, 2)),
            cloud_particle_radius_distribution_std=jnp.array(
                [
                    [1.05, 1.5],
                    [1.05, 1.5],
                ]
            ),
            eddy_diffusion_coefficients=jnp.array([1.0e7, 1.0e7]),
        )
    )

    assert mean_radii.shape == (2, 2)
    assert np.isfinite(np.asarray(mean_radii)).all()
    assert not np.allclose(
        np.asarray(mean_radii[:, 0]),
        np.asarray(mean_radii[:, 1]),
    )