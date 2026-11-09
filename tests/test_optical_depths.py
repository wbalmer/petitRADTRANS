"""Test the scaling of the cloud opacity to a given optical depth at the cloud-free photosphere.
"""
from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from .context import petitRADTRANS  # noqa: E402

optical_depths_module = petitRADTRANS.opacities.optical_depths
_DTYPE = petitRADTRANS.jax_dtype.WORKING_JAX_DTYPE  # optical depths are calculated in this precision

_N_FREQUENCIES = 5
_N_LAYERS = 60
_PRESSURES = jnp.logspace(-6.0, 2.0, _N_LAYERS) * 1e6  # (cgs)
_REFERENCE_GRAVITY = 1e3
_FREQUENCIES = jnp.linspace(1.0e14, 3.0e14, _N_FREQUENCIES)
# Frequency-dependent gas opacities, so that the median over frequencies selects one of them
_GAS_OPACITIES = jnp.broadcast_to(
    jnp.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=_DTYPE)[None, :, None, None] * 1e-3,
    (1, _N_FREQUENCIES, 1, _N_LAYERS),
)
_CLOUD_ABSORPTION_OPACITIES = jnp.full((_N_FREQUENCIES, _N_LAYERS), 1e-3, dtype=_DTYPE)
_CLOUD_SCATTERING_OPACITIES = jnp.full((_N_FREQUENCIES, _N_LAYERS), 2e-3, dtype=_DTYPE)


def _compute_scaled_optical_depths(photospheric_cloud_optical_depths, cloud_f_sed):
    return optical_depths_module.compute_optical_depths_and_pdp(
        pressures=_PRESSURES,
        reference_gravity=_REFERENCE_GRAVITY,
        # The cloud opacities are already included in the total opacities, as in Radtrans
        opacities=_GAS_OPACITIES + _CLOUD_ABSORPTION_OPACITIES[None, :, None, :],
        continuum_opacities_scattering=_CLOUD_SCATTERING_OPACITIES,
        scattering_in_emission=True,
        sum_opacities=True,
        photospheric_cloud_optical_depths=photospheric_cloud_optical_depths,
        frequencies=_FREQUENCIES,
        weights_gauss=jnp.ones(1),
        cloud_wavelengths=None,
        cloud_f_sed=cloud_f_sed,
        cloud_anisotropic_scattering_opacities=_CLOUD_SCATTERING_OPACITIES,
        cloud_absorption_opacities=_CLOUD_ABSORPTION_OPACITIES,
    )


def test_photospheric_cloud_scaling_sets_the_cloud_optical_depth_at_the_clear_photosphere():
    target_cloud_optical_depth = 0.5
    cloud_f_sed = 2.0

    optical_depths, photon_destruction_probabilities, relative_cloud_scaling_factor = (
        _compute_scaled_optical_depths(
            jnp.asarray(target_cloud_optical_depth), {"cloud": jnp.asarray(cloud_f_sed)}
        )
    )

    clear_optical_depths, _ = optical_depths_module._compute_total_optical_depths_and_pdp(
        pressures=_PRESSURES,
        reference_gravity=_REFERENCE_GRAVITY,
        opacities=_GAS_OPACITIES,
        continuum_opacities_scattering=jnp.zeros((_N_FREQUENCIES, _N_LAYERS), dtype=_DTYPE),
        scattering_in_emission=True,
    )

    # Optical depths are linear in the opacities: the difference is the scaled cloud optical depth
    clear_optical_depth = np.median(np.asarray(clear_optical_depths[0, :, 0, :]), axis=0)
    cloud_optical_depth = np.median(np.asarray(optical_depths - clear_optical_depths)[0, :, 0, :], axis=0)

    unscaled_cloud_optical_depths, _ = optical_depths_module._compute_total_optical_depths_and_pdp(
        pressures=_PRESSURES,
        reference_gravity=_REFERENCE_GRAVITY,
        opacities=jnp.broadcast_to(_CLOUD_ABSORPTION_OPACITIES[None, :, None, :], _GAS_OPACITIES.shape),
        continuum_opacities_scattering=_CLOUD_SCATTERING_OPACITIES,
        scattering_in_emission=True,
    )

    clear_photosphere_pressure = np.interp(1.0, clear_optical_depth, np.asarray(_PRESSURES))
    cloud_scaling_factor = target_cloud_optical_depth / np.interp(
        clear_photosphere_pressure,
        np.asarray(_PRESSURES),
        np.median(np.asarray(unscaled_cloud_optical_depths[0, :, 0, :]), axis=0),
    )

    assert np.all(np.isfinite(optical_depths))
    assert np.all(np.isfinite(photon_destruction_probabilities))
    np.testing.assert_allclose(
        np.interp(clear_photosphere_pressure, np.asarray(_PRESSURES), cloud_optical_depth),
        target_cloud_optical_depth,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        relative_cloud_scaling_factor, cloud_scaling_factor / (2.0 * (cloud_f_sed + 1.0)), rtol=1e-5
    )


def test_photospheric_cloud_scaling_is_jit_compatible():
    eager = _compute_scaled_optical_depths(jnp.asarray(0.5), {})
    jitted = jax.jit(_compute_scaled_optical_depths)(jnp.asarray(0.5), {})

    assert eager[2] is None
    np.testing.assert_allclose(jitted[0], eager[0], rtol=1e-6)
    np.testing.assert_allclose(jitted[1], eager[1], rtol=1e-6)


def test_photospheric_cloud_scaling_uses_the_smallest_f_sed_under_jit():
    cloud_f_sed = {"a": jnp.asarray(3.0), "b": jnp.asarray(1.0)}

    eager = _compute_scaled_optical_depths(jnp.asarray(0.5), cloud_f_sed)
    jitted = jax.jit(_compute_scaled_optical_depths)(jnp.asarray(0.5), cloud_f_sed)
    without_f_sed = _compute_scaled_optical_depths(jnp.asarray(0.5), None)

    assert without_f_sed[2] is None
    np.testing.assert_allclose(jitted[2], eager[2], rtol=1e-6)
    # The maximum rescaling is set by the smallest f_sed
    np.testing.assert_allclose(
        eager[2],
        _compute_scaled_optical_depths(jnp.asarray(0.5), {"b": jnp.asarray(1.0)})[2],
        rtol=1e-12,
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_photospheric_cloud_scaling_rejects_non_scalar_f_sed(use_jit):
    compute = jax.jit(_compute_scaled_optical_depths) if use_jit else _compute_scaled_optical_depths

    for cloud_f_sed in (
        {"a": jnp.linspace(1.0, 3.0, _N_LAYERS)},  # per-layer f_sed
        {"a": jnp.asarray(2.0), "b": jnp.linspace(1.0, 3.0, _N_LAYERS)},
    ):
        with pytest.raises(ValueError, match="Only scalar cloud_f_sed values"):
            compute(jnp.asarray(0.5), cloud_f_sed)
