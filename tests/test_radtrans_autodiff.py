import jax
import jax.numpy as jnp
import numpy as np

import petitRADTRANS.physical_constants as cst

from .test_radtrans_correlated_k import atmosphere_ck
from .test_radtrans_correlated_k_scattering import atmosphere_ck_scattering
from .utils import get_cloud_parameters, test_parameters, temperature_guillot_2010


def _build_flux_inputs():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    return {
        'mass_fractions': mass_fractions,
        'mean_molar_masses': jnp.array(test_parameters['mean_molar_mass']),
        'reference_gravity': test_parameters['planetary_parameters']['reference_gravity'],
        'cloud_parameters': {
            'cloud_f_sed': cloud_f_sed,
            'eddy_diffusion_coefficients': jnp.array(
                test_parameters['planetary_parameters']['eddy_diffusion_coefficients']
            ),
            'cloud_particle_radius_distribution_std': cloud_particle_radius_distribution_std,
        },
    }


def _build_transit_inputs():
    flux_inputs = _build_flux_inputs()
    return {
        **flux_inputs,
        'reference_pressure': test_parameters['planetary_parameters']['reference_pressure'],
        'planet_radius': test_parameters['planetary_parameters']['radius'] * cst.r_jup_mean,
    }


def _build_simple_flux_objective(atmosphere):
    flux_inputs = _build_flux_inputs()

    def objective(temperatures):
        _, flux, _ = atmosphere.simple_calculate_flux(
            temperatures=temperatures,
            mass_fractions=flux_inputs['mass_fractions'],
            mean_molar_masses=flux_inputs['mean_molar_masses'],
            reference_gravity=flux_inputs['reference_gravity'],
            **flux_inputs['cloud_parameters'],
            frequencies_to_wavelengths=False,
        )
        return jnp.sum(flux)

    return objective, jnp.array(temperature_guillot_2010)


def test_simple_calculate_flux_grad_correlated_k():
    objective, base_temperatures = _build_simple_flux_objective(atmosphere_ck)

    gradients = jax.grad(objective)(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_simple_calculate_flux_grad_correlated_k_scattering():
    objective, base_temperatures = _build_simple_flux_objective(atmosphere_ck_scattering)

    gradients = jax.grad(objective)(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_flux_autodiff_matches_simple_calculate_flux_scattering():
    flux_inputs = _build_flux_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    simple_frequencies, simple_flux, _ = atmosphere_ck_scattering.simple_calculate_flux(
        temperatures=base_temperatures,
        mass_fractions=flux_inputs['mass_fractions'],
        mean_molar_masses=flux_inputs['mean_molar_masses'],
        reference_gravity=flux_inputs['reference_gravity'],
        **flux_inputs['cloud_parameters'],
        frequencies_to_wavelengths=False,
    )
    autodiff_frequencies, autodiff_flux, _ = atmosphere_ck_scattering.calculate_flux_autodiff(
        temperatures=base_temperatures,
        mass_fractions=flux_inputs['mass_fractions'],
        mean_molar_masses=flux_inputs['mean_molar_masses'],
        reference_gravity=flux_inputs['reference_gravity'],
        cloud_parameters=flux_inputs['cloud_parameters'],
        frequencies_to_wavelengths=False,
    )

    assert np.allclose(np.asarray(autodiff_frequencies), np.asarray(simple_frequencies))
    assert np.allclose(np.asarray(autodiff_flux), np.asarray(simple_flux))

    gradients = jax.grad(
        lambda temperatures: jnp.sum(
            atmosphere_ck_scattering.calculate_flux_autodiff(
                temperatures=temperatures,
                mass_fractions=flux_inputs['mass_fractions'],
                mean_molar_masses=flux_inputs['mean_molar_masses'],
                reference_gravity=flux_inputs['reference_gravity'],
                cloud_parameters=flux_inputs['cloud_parameters'],
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
                mass_fractions=transit_inputs['mass_fractions'],
                mean_molar_masses=transit_inputs['mean_molar_masses'],
                reference_gravity=transit_inputs['reference_gravity'],
                reference_pressure=transit_inputs['reference_pressure'],
                planet_radius=transit_inputs['planet_radius'],
                **transit_inputs['cloud_parameters'],
                frequencies_to_wavelengths=False,
            )[1]
        )
    )(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()


def test_calculate_transit_radii_autodiff_matches_calculate_transit_radii_scattering():
    transit_inputs = _build_transit_inputs()
    base_temperatures = jnp.array(temperature_guillot_2010)

    reference_frequencies, reference_transit_radii, _ = atmosphere_ck_scattering.calculate_transit_radii(
        temperatures=base_temperatures,
        mass_fractions=transit_inputs['mass_fractions'],
        mean_molar_masses=transit_inputs['mean_molar_masses'],
        reference_gravity=transit_inputs['reference_gravity'],
        reference_pressure=transit_inputs['reference_pressure'],
        planet_radius=transit_inputs['planet_radius'],
        **transit_inputs['cloud_parameters'],
        frequencies_to_wavelengths=False,
    )
    autodiff_frequencies, autodiff_transit_radii, _ = atmosphere_ck_scattering.calculate_transit_radii_autodiff(
        temperatures=base_temperatures,
        mass_fractions=transit_inputs['mass_fractions'],
        mean_molar_masses=transit_inputs['mean_molar_masses'],
        reference_gravity=transit_inputs['reference_gravity'],
        reference_pressure=transit_inputs['reference_pressure'],
        planet_radius=transit_inputs['planet_radius'],
        cloud_parameters=transit_inputs['cloud_parameters'],
        frequencies_to_wavelengths=False,
    )

    assert np.allclose(np.asarray(autodiff_frequencies), np.asarray(reference_frequencies))
    assert np.allclose(np.asarray(autodiff_transit_radii), np.asarray(reference_transit_radii))

    gradients = jax.grad(
        lambda temperatures: jnp.sum(
            atmosphere_ck_scattering.calculate_transit_radii_autodiff(
                temperatures=temperatures,
                mass_fractions=transit_inputs['mass_fractions'],
                mean_molar_masses=transit_inputs['mean_molar_masses'],
                reference_gravity=transit_inputs['reference_gravity'],
                reference_pressure=transit_inputs['reference_pressure'],
                planet_radius=transit_inputs['planet_radius'],
                cloud_parameters=transit_inputs['cloud_parameters'],
                frequencies_to_wavelengths=False,
            )[1]
        )
    )(base_temperatures)

    assert gradients.shape == base_temperatures.shape
    assert np.isfinite(np.asarray(gradients)).all()
