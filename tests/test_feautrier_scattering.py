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


def _compute_flux_per_column(temperatures, optical_depths, photon_destruction_probabilities, chunk_size=None):
    """Flux of an atmosphere whose (g, frequency) columns differ, optionally in frequency chunks."""
    n_g = optical_depths.shape[0]

    return radiative_transfer_core.compute_feautrier_radiative_transfer(
        frequencies_bin_edges=_FREQUENCIES_BIN_EDGES,
        temperatures=temperatures,
        weights_gauss=jnp.full(n_g, 1.0 / n_g),
        emission_cos_angles=jnp.asarray([0.1127016654, 0.5, 0.8872983346]),
        emission_cos_angles_weights=jnp.asarray([0.2777777778, 0.4444444444, 0.2777777778]),
        optical_depths_in=optical_depths,
        photon_destruction_probabilities_in=photon_destruction_probabilities,
        stellar_intensity=jnp.zeros(_N_FREQUENCIES),
        star_irradiation_cos_angle=1.0,
        reflectances=jnp.zeros(_N_FREQUENCIES),
        emissivities=jnp.ones(_N_FREQUENCIES),
        chunk_feautrier=chunk_size is not None,
        feautrier_chunk_size=chunk_size if chunk_size is not None else _N_FREQUENCIES,
    )[0]


def _column_dependent_atmosphere(n_g=2, seed=0):
    rng = np.random.default_rng(seed)
    optical_depths = jnp.asarray(
        10.0 ** rng.uniform(0.0, 3.0, (n_g, _N_FREQUENCIES, 1)) * np.logspace(-6.0, 0.0, _N_LAYERS)
    )
    photon_destruction_probabilities = jnp.asarray(10.0 ** rng.uniform(-3.0, 0.0, (n_g, _N_FREQUENCIES, _N_LAYERS)))

    return optical_depths, photon_destruction_probabilities


def test_feautrier_scattering_gradient_with_respect_to_photon_destruction_probabilities_only():
    # The surface intensities of the scattering fixed point do not depend on the photon destruction
    # probabilities: the implicit gradient must still be computable with respect to them alone
    optical_depths, photon_destruction_probabilities = _column_dependent_atmosphere()

    def total_flux(probabilities):
        return jnp.sum(_compute_flux_per_column(_TEMPERATURES, optical_depths, probabilities))

    gradient = jax.grad(total_flux)(photon_destruction_probabilities)

    direction = jnp.asarray(np.random.default_rng(1).normal(size=photon_destruction_probabilities.shape))
    direction = direction * photon_destruction_probabilities * 1e-4
    finite_difference = (
        total_flux(photon_destruction_probabilities + direction)
        - total_flux(photon_destruction_probabilities - direction)
    ) / 2.0

    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(jnp.sum(gradient * direction), finite_difference, rtol=1e-5)


def test_feautrier_scattering_gradient_does_not_depend_on_the_frequency_chunks():
    # In reverse mode, the frequency chunks are recomputed during the backward pass (rematerialisation),
    # keeping only the converged scattering state of each chunk
    optical_depths, photon_destruction_probabilities = _column_dependent_atmosphere(seed=2)

    def total_flux(temperatures, optical_depths_, probabilities, chunk_size):
        return jnp.sum(_compute_flux_per_column(temperatures, optical_depths_, probabilities, chunk_size))

    gradients = {
        chunk_size: jax.grad(total_flux, argnums=(0, 1, 2))(
            _TEMPERATURES, optical_depths, photon_destruction_probabilities, chunk_size
        )
        for chunk_size in (None, 3)  # 3 does not divide the 4 frequencies: the last chunk overlaps
    }

    for chunked, unchunked in zip(gradients[3], gradients[None]):
        # The tolerance of the implicit linear solve is relative to the norm of the whole chunk, so the
        # gradient of each element is only exact relative to the largest gradient of its chunk
        np.testing.assert_allclose(chunked, unchunked, rtol=0, atol=1e-4 * np.max(np.abs(unchunked)))
