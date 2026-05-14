import jax
import jax.numpy as jnp
import numpy as np

import petitRADTRANS.physical_constants as cst

from .context import petitRADTRANS
from .utils import get_cloud_parameters, test_parameters, temperature_guillot_2010

# Narrowed wavelength range and single line species to keep gradient memory requirements low.
_AUTODIFF_WAVELENGTH_BOUNDARIES = [1.0, 1.1]
_AUTODIFF_LINE_SPECIES = test_parameters["spectrum_parameters"][
    "line_species_correlated_k"
][:1]


def _init_atmosphere_ck_autodiff(scattering_in_emission=False):
    return petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters["pressures"],
        line_species=_AUTODIFF_LINE_SPECIES,
        rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        cloud_species=list(test_parameters["cloud_parameters"]["cloud_species"].keys()),
        wavelength_boundaries=_AUTODIFF_WAVELENGTH_BOUNDARIES,
        line_opacity_mode="c-k",
        scattering_in_emission=scattering_in_emission,
    )


atmosphere_ck = _init_atmosphere_ck_autodiff(scattering_in_emission=False)
atmosphere_ck_scattering = _init_atmosphere_ck_autodiff(scattering_in_emission=True)


def _build_flux_inputs():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = (
        get_cloud_parameters("mass_fractions_correlated_k")
    )
    return {
        "mass_fractions": mass_fractions,
        "mean_molar_masses": jnp.array(test_parameters["mean_molar_mass"]),
        "reference_gravity": test_parameters["planetary_parameters"][
            "reference_gravity"
        ],
        "cloud_parameters": {
            "cloud_f_sed": cloud_f_sed,
            "eddy_diffusion_coefficients": jnp.array(
                test_parameters["planetary_parameters"]["eddy_diffusion_coefficients"]
            ),
            "cloud_particle_radius_distribution_std": cloud_particle_radius_distribution_std,
        },
    }


def _build_transit_inputs():
    flux_inputs = _build_flux_inputs()
    return {
        **flux_inputs,
        "reference_pressure": test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        "planet_radius": test_parameters["planetary_parameters"]["radius"]
        * cst.r_jup_mean,
    }


def _build_power_law_inputs():
    return {
        "mass_fractions": {
            species: jnp.array(values)
            for species, values in test_parameters["mass_fractions_correlated_k"].items()
        },
        "mean_molar_masses": jnp.array(test_parameters["mean_molar_mass"]),
        "reference_gravity": test_parameters["planetary_parameters"][
            "reference_gravity"
        ],
        "power_law_opacity_350nm": test_parameters["cloud_parameters"][
            "kappa_zero"
        ],
        "power_law_opacity_coefficient": test_parameters["cloud_parameters"][
            "gamma_scattering"
        ],
    }


def _build_simple_flux_objective(atmosphere):
    flux_inputs = _build_flux_inputs()

    def objective(temperatures):
        _, flux, _ = atmosphere.calculate_flux(
            temperatures=temperatures,
            mass_fractions=flux_inputs["mass_fractions"],
            mean_molar_masses=flux_inputs["mean_molar_masses"],
            reference_gravity=flux_inputs["reference_gravity"],
            **flux_inputs["cloud_parameters"],
            frequencies_to_wavelengths=False,
        )
        return jnp.sum(flux)

    return objective, jnp.array(temperature_guillot_2010)


def _calculate_non_scattering_cloud_flux(cloud_fraction):
    flux_inputs = _build_flux_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    _, flux, _ = atmosphere_ck.calculate_flux(
        temperatures=base_temperatures,
        mass_fractions=flux_inputs["mass_fractions"],
        mean_molar_masses=flux_inputs["mean_molar_masses"],
        reference_gravity=flux_inputs["reference_gravity"],
        cloud_fraction=cloud_fraction,
        patchy_clouds=None,
        **flux_inputs["cloud_parameters"],
        frequencies_to_wavelengths=False,
    )

    return flux


def _calculate_power_law_flux(cloud_fraction):
    flux_inputs = _build_power_law_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    _, flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=base_temperatures,
        mass_fractions=flux_inputs["mass_fractions"],
        mean_molar_masses=flux_inputs["mean_molar_masses"],
        reference_gravity=flux_inputs["reference_gravity"],
        power_law_opacity_350nm=flux_inputs["power_law_opacity_350nm"],
        power_law_opacity_coefficient=flux_inputs[
            "power_law_opacity_coefficient"
        ],
        cloud_fraction=cloud_fraction,
        patchy_clouds=[],
        frequencies_to_wavelengths=False,
    )

    return flux


def _calculate_power_law_transit_radii(cloud_fraction):
    transit_inputs = _build_transit_inputs()
    power_law_inputs = _build_power_law_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    _, transit_radii, _ = atmosphere_ck.calculate_transit_radii(
        temperatures=base_temperatures,
        mass_fractions=power_law_inputs["mass_fractions"],
        mean_molar_masses=power_law_inputs["mean_molar_masses"],
        reference_gravity=power_law_inputs["reference_gravity"],
        reference_pressure=transit_inputs["reference_pressure"],
        planet_radius=transit_inputs["planet_radius"],
        power_law_opacity_350nm=power_law_inputs["power_law_opacity_350nm"],
        power_law_opacity_coefficient=power_law_inputs[
            "power_law_opacity_coefficient"
        ],
        cloud_fraction=cloud_fraction,
        patchy_clouds=[],
        frequencies_to_wavelengths=False,
    )

    return transit_radii


def test_calculate_flux_grad_correlated_k():
    objective, base_temperatures = _build_simple_flux_objective(atmosphere_ck)

    gradients = jax.grad(objective)(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_flux_grad_correlated_k_scattering():
    objective, base_temperatures = _build_simple_flux_objective(
        atmosphere_ck_scattering
    )

    gradients = jax.grad(objective)(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_flux_clouds_affect_correlated_k_non_scattering_emission_full_cover():
    cloudy_flux = _calculate_non_scattering_cloud_flux(1.0)
    clear_flux = _calculate_non_scattering_cloud_flux(0.0)

    relative_difference = jnp.abs(
        (cloudy_flux - clear_flux) / jnp.maximum(jnp.abs(clear_flux), 1e-30)
    )

    assert float(jnp.max(relative_difference)) > 1e-2


def test_calculate_flux_patchy_clouds_blend_correlated_k_non_scattering_emission():
    cloud_fraction = 0.37
    cloudy_flux = _calculate_non_scattering_cloud_flux(1.0)
    clear_flux = _calculate_non_scattering_cloud_flux(0.0)
    patchy_flux = _calculate_non_scattering_cloud_flux(cloud_fraction)

    expected_flux = cloud_fraction * cloudy_flux + (1 - cloud_fraction) * clear_flux

    np.testing.assert_allclose(
        np.asarray(patchy_flux),
        np.asarray(expected_flux),
        atol=0.0,
        rtol=1e-6,
    )
    assert not np.allclose(
        np.asarray(patchy_flux), np.asarray(cloudy_flux), atol=0.0, rtol=1e-6
    )
    assert not np.allclose(
        np.asarray(patchy_flux), np.asarray(clear_flux), atol=0.0, rtol=1e-6
    )


def test_calculate_flux_power_law_clouds_respect_cloud_fraction():
    cloud_fraction = 0.37
    cloudy_flux = _calculate_power_law_flux(1.0)
    clear_flux = _calculate_power_law_flux(0.0)
    patchy_flux = _calculate_power_law_flux(cloud_fraction)

    expected_flux = cloud_fraction * cloudy_flux + (1 - cloud_fraction) * clear_flux

    np.testing.assert_allclose(
        np.asarray(patchy_flux),
        np.asarray(expected_flux),
        atol=0.0,
        rtol=1e-6,
    )
    assert not np.allclose(
        np.asarray(patchy_flux), np.asarray(cloudy_flux), atol=0.0, rtol=1e-6
    )
    assert not np.allclose(
        np.asarray(patchy_flux), np.asarray(clear_flux), atol=0.0, rtol=1e-6
    )


def test_calculate_flux_jit_correlated_k_scattering():
    flux_inputs = _build_flux_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    compiled_flux = jax.jit(
        lambda temperatures: atmosphere_ck_scattering.calculate_flux(
            temperatures=temperatures,
            mass_fractions=flux_inputs["mass_fractions"],
            mean_molar_masses=flux_inputs["mean_molar_masses"],
            reference_gravity=flux_inputs["reference_gravity"],
            **flux_inputs["cloud_parameters"],
            frequencies_to_wavelengths=False,
        )[1]
    )

    flux = compiled_flux(base_temperatures)

    assert flux.shape[0] > 0
    assert np.isfinite(np.asarray(flux)).all()


def test_calculate_flux_autodiff_matches_calculate_flux_scattering():
    flux_inputs = _build_flux_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    simple_frequencies, simple_flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=base_temperatures,
        mass_fractions=flux_inputs["mass_fractions"],
        mean_molar_masses=flux_inputs["mean_molar_masses"],
        reference_gravity=flux_inputs["reference_gravity"],
        **flux_inputs["cloud_parameters"],
        frequencies_to_wavelengths=False,
    )
    autodiff_frequencies, autodiff_flux, _ = (
        atmosphere_ck_scattering.calculate_flux_autodiff(
            temperatures=base_temperatures,
            mass_fractions=flux_inputs["mass_fractions"],
            mean_molar_masses=flux_inputs["mean_molar_masses"],
            reference_gravity=flux_inputs["reference_gravity"],
            cloud_parameters=flux_inputs["cloud_parameters"],
            frequencies_to_wavelengths=False,
        )
    )

    assert np.allclose(np.asarray(autodiff_frequencies), np.asarray(simple_frequencies))
    assert np.allclose(np.asarray(autodiff_flux), np.asarray(simple_flux))

    gradients = jax.grad(
        lambda temperatures: jnp.sum(
            atmosphere_ck_scattering.calculate_flux_autodiff(
                temperatures=temperatures,
                mass_fractions=flux_inputs["mass_fractions"],
                mean_molar_masses=flux_inputs["mean_molar_masses"],
                reference_gravity=flux_inputs["reference_gravity"],
                cloud_parameters=flux_inputs["cloud_parameters"],
                frequencies_to_wavelengths=False,
            )[1]
        )
    )(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_transit_radii_grad_correlated_k_scattering():
    transit_inputs = _build_transit_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    gradients = jax.grad(
        lambda temperatures: jnp.sum(
            atmosphere_ck_scattering.calculate_transit_radii(
                temperatures=temperatures,
                mass_fractions=transit_inputs["mass_fractions"],
                mean_molar_masses=transit_inputs["mean_molar_masses"],
                reference_gravity=transit_inputs["reference_gravity"],
                reference_pressure=transit_inputs["reference_pressure"],
                planet_radius=transit_inputs["planet_radius"],
                **transit_inputs["cloud_parameters"],
                frequencies_to_wavelengths=False,
            )[1]
        )
    )(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_transit_radii_autodiff_matches_calculate_transit_radii_scattering():
    transit_inputs = _build_transit_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    reference_frequencies, reference_transit_radii, _ = (
        atmosphere_ck_scattering.calculate_transit_radii(
            temperatures=base_temperatures,
            mass_fractions=transit_inputs["mass_fractions"],
            mean_molar_masses=transit_inputs["mean_molar_masses"],
            reference_gravity=transit_inputs["reference_gravity"],
            reference_pressure=transit_inputs["reference_pressure"],
            planet_radius=transit_inputs["planet_radius"],
            **transit_inputs["cloud_parameters"],
            frequencies_to_wavelengths=False,
        )
    )
    autodiff_frequencies, autodiff_transit_radii, _ = (
        atmosphere_ck_scattering.calculate_transit_radii_autodiff(
            temperatures=base_temperatures,
            mass_fractions=transit_inputs["mass_fractions"],
            mean_molar_masses=transit_inputs["mean_molar_masses"],
            reference_gravity=transit_inputs["reference_gravity"],
            reference_pressure=transit_inputs["reference_pressure"],
            planet_radius=transit_inputs["planet_radius"],
            cloud_parameters=transit_inputs["cloud_parameters"],
            frequencies_to_wavelengths=False,
        )
    )

    assert np.allclose(
        np.asarray(autodiff_frequencies), np.asarray(reference_frequencies)
    )
    assert np.allclose(
        np.asarray(autodiff_transit_radii), np.asarray(reference_transit_radii)
    )

    gradients = jax.grad(
        lambda temperatures: jnp.sum(
            atmosphere_ck_scattering.calculate_transit_radii_autodiff(
                temperatures=temperatures,
                mass_fractions=transit_inputs["mass_fractions"],
                mean_molar_masses=transit_inputs["mean_molar_masses"],
                reference_gravity=transit_inputs["reference_gravity"],
                reference_pressure=transit_inputs["reference_pressure"],
                planet_radius=transit_inputs["planet_radius"],
                cloud_parameters=transit_inputs["cloud_parameters"],
                frequencies_to_wavelengths=False,
            )[1]
        )
    )(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_transit_radii_power_law_clouds_respect_cloud_fraction():
    cloud_fraction = 0.37
    cloudy_radii = _calculate_power_law_transit_radii(1.0)
    clear_radii = _calculate_power_law_transit_radii(0.0)
    patchy_radii = _calculate_power_law_transit_radii(cloud_fraction)

    expected_radii = cloud_fraction * cloudy_radii + (1 - cloud_fraction) * clear_radii

    np.testing.assert_allclose(
        np.asarray(patchy_radii),
        np.asarray(expected_radii),
        atol=0.0,
        rtol=1e-6,
    )
    assert not np.allclose(
        np.asarray(patchy_radii), np.asarray(cloudy_radii), atol=0.0, rtol=1e-6
    )
    assert not np.allclose(
        np.asarray(patchy_radii), np.asarray(clear_radii), atol=0.0, rtol=1e-6
    )
