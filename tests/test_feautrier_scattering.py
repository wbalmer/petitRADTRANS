"""Test the convergence and the gradient of the Feautrier scattering solver on a grey atmosphere.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .context import petitRADTRANS

radiative_transfer_core = petitRADTRANS.radtrans_core.radiative_transfer_core

_N_LAYERS = 30
_N_FREQUENCIES = 4
_FREQUENCIES_BIN_EDGES = jnp.asarray(
    petitRADTRANS.physical_constants.c / np.linspace(1.0e-4, 2.0e-4, _N_FREQUENCIES + 1)  # wavelengths in cm
)
_TEMPERATURES = jnp.asarray(np.linspace(800.0, 2000.0, _N_LAYERS))

# Settings giving a converged reference solution, far beyond the default tolerance
_REFERENCE_CONVERGENCE_THRESHOLD = 1e-13
_REFERENCE_ITER_SCAT_MAX = 1000


def _compute_flux(
    temperatures,
    photon_destruction_probability,
    maximum_optical_depth,
    return_solver_diagnostics=False,
    **kwargs,
):
    optical_depths = jnp.broadcast_to(
        maximum_optical_depth * jnp.logspace(-6.0, 0.0, _N_LAYERS), (1, _N_FREQUENCIES, _N_LAYERS)
    )

    return radiative_transfer_core.compute_feautrier_radiative_transfer(
        frequencies_bin_edges=_FREQUENCIES_BIN_EDGES,
        temperatures=temperatures,
        weights_gauss=jnp.ones(1),
        emission_cos_angles=jnp.asarray([0.1127016654, 0.5, 0.8872983346]),
        emission_cos_angles_weights=jnp.asarray([0.2777777778, 0.4444444444, 0.2777777778]),
        optical_depths_in=optical_depths,
        photon_destruction_probabilities_in=jnp.full(
            (1, _N_FREQUENCIES, _N_LAYERS), photon_destruction_probability
        ),
        stellar_intensity=jnp.zeros(_N_FREQUENCIES),
        star_irradiation_cos_angle=1.0,
        reflectances=jnp.zeros(_N_FREQUENCIES),
        emissivities=jnp.ones(_N_FREQUENCIES),
        return_solver_diagnostics=return_solver_diagnostics,
        **kwargs,
    )


def test_feautrier_scattering_optically_thick_lower_layer_converges():
    # Strongly scattering atmosphere with an optically thick lowest layer
    flux, _, _, diagnostics = _compute_flux(_TEMPERATURES, 1e-4, 1e3, return_solver_diagnostics=True)

    assert bool(diagnostics["converged_before_cap"])
    assert np.all(np.isfinite(flux))

    # The iteration must remain stable beyond convergence
    flux_long, _, _, _ = _compute_flux(
        _TEMPERATURES,
        1e-4,
        1e3,
        return_solver_diagnostics=True,
        convergence_threshold=0.0,
        iter_scat_max=_REFERENCE_ITER_SCAT_MAX,
    )

    assert np.all(np.isfinite(flux_long))
    np.testing.assert_allclose(flux, flux_long, rtol=1e-6)


def test_feautrier_scattering_gradient_matches_finite_differences():
    photon_destruction_probability = 1e-3
    maximum_optical_depth = 1e3
    direction = jnp.ones(_N_LAYERS)
    step = 1e-2  # temperature step (K)

    def total_flux(temperatures, **kwargs):
        return jnp.sum(
            _compute_flux(temperatures, photon_destruction_probability, maximum_optical_depth, **kwargs)[0]
        )

    gradient = jnp.dot(jax.grad(total_flux)(_TEMPERATURES), direction)

    reference_settings = {
        "convergence_threshold": _REFERENCE_CONVERGENCE_THRESHOLD,
        "iter_scat_max": _REFERENCE_ITER_SCAT_MAX,
    }
    finite_difference = (
        total_flux(_TEMPERATURES + step * direction, **reference_settings)
        - total_flux(_TEMPERATURES - step * direction, **reference_settings)
    ) / (2.0 * step)

    # Differentiating through the Ng-accelerated iteration is off by ~1e-4 in this atmosphere
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6)


def test_feautrier_scattering_hessian_vector_product_matches_finite_differences():
    # Forward-over-reverse second derivatives, as used by the Laplace covariance and the
    # Hessian-based NUTS mass matrix
    photon_destruction_probability = 1e-3
    maximum_optical_depth = 1e3
    direction = jnp.ones(_N_LAYERS)
    step = 1e-2  # temperature step (K)

    def total_flux(temperatures):
        return jnp.sum(_compute_flux(temperatures, photon_destruction_probability, maximum_optical_depth)[0])

    gradient = jax.grad(total_flux)
    hessian_vector_product = jax.jvp(gradient, (_TEMPERATURES,), (direction,))[1]
    finite_difference = (
        gradient(_TEMPERATURES + step * direction) - gradient(_TEMPERATURES - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        hessian_vector_product, finite_difference, rtol=1e-6, atol=1e-6 * np.max(np.abs(finite_difference))
    )
