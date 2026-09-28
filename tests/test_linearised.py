"""Tests for the closed-form modulation solve of linearised time-series retrievals.

Every result in :mod:`petitRADTRANS.retrieval.linearised` is an exact identity
rather than an approximation, so these tests check the identities directly
against dense reference implementations built with NumPy. That is deliberate:
the fast path never forms the ``(n_time, n_time)`` noise covariance, the
``(n_wavelength * n_time, n_column)`` design, or the marginal covariance of the
data, and a reference that does form them shares no code with the thing under
test. Where an identity holds to machine precision it is asserted as such,
because a tolerance loose enough to hide a sign error is not a test.

The reference for the evidence is the strongest of these: it evaluates the
marginal likelihood as a single multivariate normal density with covariance
``Sigma + A Sigma_beta A^T``, reaching the same number by the Woodbury identity
from the opposite direction.
"""

import types

import jax
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import tensorflow_probability.substrates.jax.distributions as tfpd  # noqa: E402

from petitRADTRANS.retrieval import linearised as lin  # noqa: E402
from petitRADTRANS.retrieval.plotting import RetrievalPlotter  # noqa: E402
from petitRADTRANS.retrieval.retrieval import Retrieval  # noqa: E402
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig  # noqa: E402
from petitRADTRANS.retrieval.runtime import compute_linearised_system  # noqa: E402

# A small but non-degenerate geometry: enough wavelengths that the sum over them
# is not trivial, enough epochs to resolve two harmonics, and a response count
# above one so the Kronecker structure is exercised.
N_WAVELENGTH = 7
N_TIME = 13
N_RESPONSE = 3
N_HARMONICS = 2
N_MODE = 2 * N_HARMONICS
N_COLUMN = N_RESPONSE * N_MODE
PERIOD = 9.0
# Seconds; only used by the configuration tests, which bin by rotational phase.
NOMINAL_PERIOD = 8000.0
OFFSET_VARIANCE = 0.05**2
CORRELATION_RHO = 0.4
# Identities that involve only sums and one Cholesky hold to near machine
# precision in float64; the looser value covers the dense matrix inversions in
# the references, which are the least accurate step on either side.
EXACT_RTOL = 1e-10
DENSE_RTOL = 1e-8


def _dense_design(log_jacobian, basis):
    """The full ``(n_wavelength * n_time, n_column)`` design, built explicitly.

    Rows run wavelength-major and columns response-major, matching the
    flattening :func:`petitRADTRANS.retrieval.linearised.build_linear_system`
    assumes.
    """
    n_wavelength, n_response = log_jacobian.shape
    n_time, n_mode = basis.shape

    return np.einsum("ip,jm->ijpm", log_jacobian, basis).reshape(
        n_wavelength * n_time, n_response * n_mode
    )


def _dense_noise_covariance(var_point, correlation_rho):
    """Per-wavelength measurement-noise covariance ``S_i``, without the offset."""
    n_wavelength, n_time = var_point.shape
    lag = np.abs(np.subtract.outer(np.arange(n_time), np.arange(n_time)))
    correlation = correlation_rho**lag

    return np.stack([
        correlation * np.outer(np.sqrt(var_point[i]), np.sqrt(var_point[i]))
        for i in range(n_wavelength)
    ])


def _dense_precision(var_point, offset_variance, correlation_rho):
    """Block-diagonal ``Sigma^-1`` for the whole dataset.

    An infinite offset width has no covariance to invert, so its precision is
    built from the limit of the Sherman-Morrison correction, which annihilates
    anything constant in time.
    """
    blocks = _dense_noise_covariance(var_point, correlation_rho)
    n_wavelength, n_time, _ = blocks.shape
    ones = np.ones(n_time)
    precisions = []

    for i in range(n_wavelength):
        noise_precision = np.linalg.inv(blocks[i])

        if np.isinf(offset_variance):
            weights = noise_precision @ ones
            precisions.append(
                noise_precision - np.outer(weights, weights) / (ones @ weights)
            )
        else:
            precisions.append(
                np.linalg.inv(blocks[i] + offset_variance * np.outer(ones, ones))
            )

    from scipy.linalg import block_diag

    return block_diag(*precisions)


def _random_problem(seed=0, correlation_rho=0.0, offset_variance=OFFSET_VARIANCE):
    """A complete, self-consistent set of inputs plus its dense counterparts."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 2.3 * PERIOD, N_TIME)
    log_jacobian = rng.normal(size=(N_WAVELENGTH, N_RESPONSE))
    residual = rng.normal(size=(N_WAVELENGTH, N_TIME)) * 1e-3
    var_point = (rng.uniform(0.5, 1.5, size=(N_WAVELENGTH, N_TIME)) * 1e-6)
    prior_variance = rng.uniform(0.5, 2.0, size=N_COLUMN)
    reference_indices = np.array([0, 1, 2])
    reference_weight_vector = np.full(reference_indices.size, 1.0 / reference_indices.size)
    basis = np.asarray(lin.reference_anchored_basis(
        jnp.asarray(times),
        PERIOD,
        jnp.asarray(reference_indices),
        N_HARMONICS,
        jnp.asarray(reference_weight_vector),
    ))

    return {
        "times": times,
        "basis": basis,
        "log_jacobian": log_jacobian,
        "residual": residual,
        "var_point": var_point,
        "prior_variance": prior_variance,
        "offset_variance": offset_variance,
        "correlation_rho": correlation_rho,
        "reference_indices": reference_indices,
        "reference_weight_vector": reference_weight_vector,
    }


def _solve_linear_system(problem):
    """Run the module's assembly on a problem from :func:`_random_problem`."""
    return lin.build_linear_system(
        jnp.asarray(problem["log_jacobian"]),
        jnp.asarray(problem["basis"]),
        jnp.asarray(problem["residual"]),
        jnp.asarray(problem["var_point"]),
        problem["offset_variance"],
        problem["correlation_rho"],
    )


def test_anchored_basis_vanishes_on_the_reference_set():
    problem = _random_problem()
    weighted_mean = (
        problem["reference_weight_vector"]
        @ problem["basis"][problem["reference_indices"]]
    )

    np.testing.assert_allclose(weighted_mean, 0.0, atol=1e-14)


def test_bias_projection_matches_a_solve_against_the_error_field():
    problem = _random_problem(seed=5)
    error_field = np.asarray(
        np.random.default_rng(11).normal(size=(N_WAVELENGTH, N_TIME)) * 1e-4
    )
    shift = np.asarray(lin.amplitude_bias_projection(
        jnp.asarray(problem["log_jacobian"]),
        jnp.asarray(problem["basis"]),
        jnp.asarray(error_field),
        jnp.asarray(problem["var_point"]),
        problem["offset_variance"],
        jnp.asarray(problem["prior_variance"]),
        problem["correlation_rho"],
        units="absolute",
    ))

    # The same quantity by definition: solve the normal equations built from the
    # error field in place of the data.
    normal_matrix, normal_vector, _ = lin.build_linear_system(
        jnp.asarray(problem["log_jacobian"]),
        jnp.asarray(problem["basis"]),
        jnp.asarray(error_field),
        jnp.asarray(problem["var_point"]),
        problem["offset_variance"],
        problem["correlation_rho"],
    )
    mean, _ = lin.solve_modulation(
        normal_matrix,
        normal_vector,
        jnp.asarray(problem["prior_variance"]),
        N_RESPONSE,
        N_HARMONICS,
    )

    np.testing.assert_allclose(shift, np.asarray(mean).reshape(-1), rtol=EXACT_RTOL)


def test_infinite_offset_annihilates_a_time_constant_residual():
    """A per-wavelength constant is exactly what the free offset absorbs."""
    problem = _random_problem(offset_variance=jnp.inf)
    _, normal_vector, _ = _solve_linear_system(problem)

    shifted = dict(problem)
    shifted["residual"] = problem["residual"] + np.arange(N_WAVELENGTH)[:, None] * 0.3
    _, shifted_vector, _ = _solve_linear_system(shifted)

    np.testing.assert_allclose(
        np.asarray(shifted_vector), np.asarray(normal_vector), rtol=EXACT_RTOL
    )


@pytest.mark.parametrize("correlation_rho", [0.0, CORRELATION_RHO])
@pytest.mark.parametrize("offset_variance", [0.0, OFFSET_VARIANCE, jnp.inf])
def test_linear_system_matches_a_dense_reference(offset_variance, correlation_rho):
    """The Kronecker assembly equals the explicit ``A^T Sigma^-1 A``."""
    problem = _random_problem(
        correlation_rho=correlation_rho, offset_variance=offset_variance
    )
    normal_matrix, normal_vector, chi2_data = _solve_linear_system(problem)

    design = _dense_design(problem["log_jacobian"], problem["basis"])
    precision = _dense_precision(
        problem["var_point"], offset_variance, correlation_rho
    )
    residual = problem["residual"].reshape(-1)

    np.testing.assert_allclose(
        np.asarray(normal_matrix), design.T @ precision @ design, rtol=DENSE_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(normal_vector), design.T @ precision @ residual, rtol=DENSE_RTOL
    )
    np.testing.assert_allclose(
        float(chi2_data), residual @ precision @ residual, rtol=DENSE_RTOL
    )


@pytest.mark.parametrize("correlation_rho", [0.0, CORRELATION_RHO])
def test_log_evidence_matches_the_marginal_multivariate_normal(correlation_rho):
    """``ln Z`` equals ``N(r; 0, Sigma + A Sigma_beta A^T)`` by Woodbury."""
    problem = _random_problem(correlation_rho=correlation_rho)
    normal_matrix, normal_vector, chi2_data = _solve_linear_system(problem)

    log_determinant, n_constrained = lin.noise_log_determinant(
        jnp.asarray(problem["var_point"]),
        problem["offset_variance"],
        correlation_rho,
    )
    noise_normalisation = float(
        -0.5 * log_determinant - 0.5 * n_constrained * np.log(2.0 * np.pi)
    )
    log_evidence, _ = lin.modulation_log_evidence(
        normal_matrix,
        normal_vector,
        chi2_data,
        jnp.asarray(problem["prior_variance"]),
        noise_normalisation,
    )

    design = _dense_design(problem["log_jacobian"], problem["basis"])
    blocks = _dense_noise_covariance(problem["var_point"], correlation_rho)
    from scipy.linalg import block_diag

    # The marginalised offset is per wavelength, so its rank-one term belongs
    # inside each block and not across the whole matrix.
    covariance = block_diag(*[
        block + problem["offset_variance"] * np.ones_like(block) for block in blocks
    ])
    marginal = covariance + design @ np.diag(problem["prior_variance"]) @ design.T
    residual = problem["residual"].reshape(-1)
    sign, log_determinant_marginal = np.linalg.slogdet(marginal)
    reference = -0.5 * (
        residual @ np.linalg.solve(marginal, residual)
        + log_determinant_marginal
        + residual.size * np.log(2.0 * np.pi)
    )

    assert sign > 0
    np.testing.assert_allclose(float(log_evidence), reference, rtol=DENSE_RTOL)


@pytest.mark.parametrize("correlation_rho", [0.0, CORRELATION_RHO])
def test_noise_log_determinant_matches_a_dense_slogdet(correlation_rho):
    problem = _random_problem(correlation_rho=correlation_rho)
    log_determinant, n_constrained = lin.noise_log_determinant(
        jnp.asarray(problem["var_point"]),
        OFFSET_VARIANCE,
        correlation_rho,
    )
    blocks = _dense_noise_covariance(problem["var_point"], correlation_rho)
    reference = sum(
        np.linalg.slogdet(block + OFFSET_VARIANCE * np.ones_like(block))[1]
        for block in blocks
    )

    assert n_constrained == N_WAVELENGTH * N_TIME
    np.testing.assert_allclose(float(log_determinant), reference, rtol=DENSE_RTOL)


def test_smooth_wavelength_modes_are_orthonormal():
    wavelengths = jnp.linspace(0.9, 5.3, 40)
    modes = np.asarray(lin.smooth_wavelength_modes(wavelengths, 4))

    np.testing.assert_allclose(modes.T @ modes, np.eye(4), atol=1e-12)
    # Mode 0 is the grey component, so it must be constant in wavelength.
    np.testing.assert_allclose(modes[:, 0], modes[0, 0], rtol=EXACT_RTOL)


def test_solve_recovers_injected_amplitudes():
    """With small noise the posterior mean returns what was injected."""
    problem = _random_problem(seed=3)
    rng = np.random.default_rng(17)
    truth = rng.normal(size=(N_RESPONSE, N_MODE)) * 0.01
    clean = np.einsum(
        "ip,jm,pm->ij", problem["log_jacobian"], problem["basis"], truth
    )
    problem["residual"] = clean + rng.normal(size=clean.shape) * 1e-9

    normal_matrix, normal_vector, _ = _solve_linear_system(problem)
    mean, covariance = lin.solve_modulation(
        normal_matrix,
        normal_vector,
        jnp.asarray(problem["prior_variance"]),
        N_RESPONSE,
        N_HARMONICS,
    )

    np.testing.assert_allclose(np.asarray(mean), truth, rtol=1e-4, atol=1e-9)
    # The covariance is the inverse of the precision it was built from.
    precision = np.asarray(normal_matrix) + np.diag(1.0 / problem["prior_variance"])
    np.testing.assert_allclose(
        np.asarray(covariance) @ precision, np.eye(N_COLUMN), atol=1e-8
    )


def test_trigonometric_resampling_is_exact_for_a_band_limited_signal():
    n_harmonics = 3
    node_times = jnp.linspace(0.0, PERIOD, 2 * n_harmonics + 2)[:-1]
    target_times = jnp.linspace(0.0, 2.0 * PERIOD, 29)
    matrix = lin.trigonometric_resampling_matrix(
        node_times, target_times, PERIOD, n_harmonics
    )

    angle = 2.0 * np.pi / PERIOD
    signal = lambda t: (  # noqa: E731 - a one-line probe, not an API
        0.7 + 0.3 * np.cos(angle * t) - 0.2 * np.sin(2 * angle * t)
        + 0.05 * np.cos(3 * angle * t)
    )

    np.testing.assert_allclose(
        np.asarray(matrix) @ signal(np.asarray(node_times)),
        signal(np.asarray(target_times)),
        atol=1e-10,
    )


def test_trigonometric_resampling_rejects_duplicated_phases():
    """A grid spanning exactly one period repeats its endpoint phase."""
    n_harmonics = 3
    node_times = jnp.linspace(0.0, PERIOD, 2 * n_harmonics + 1)

    with pytest.raises(ValueError, match="rotational phases"):
        lin.trigonometric_resampling_matrix(
            node_times, node_times, PERIOD, n_harmonics
        )


def test_evidence_gradient_matches_central_differences():
    """``ln Z`` is differentiable in the period, which the outer sampler needs."""
    problem = _random_problem(seed=7)

    def log_evidence(period):
        basis = lin.reference_anchored_basis(
            jnp.asarray(problem["times"]),
            period,
            jnp.asarray(problem["reference_indices"]),
            N_HARMONICS,
            jnp.asarray(problem["reference_weight_vector"]),
        )
        normal_matrix, normal_vector, chi2_data = lin.build_linear_system(
            jnp.asarray(problem["log_jacobian"]),
            basis,
            jnp.asarray(problem["residual"]),
            jnp.asarray(problem["var_point"]),
            problem["offset_variance"],
            problem["correlation_rho"],
        )
        value, _ = lin.modulation_log_evidence(
            normal_matrix,
            normal_vector,
            chi2_data,
            jnp.asarray(problem["prior_variance"]),
        )
        return value

    step = 1e-5
    analytic = float(jax.grad(log_evidence)(PERIOD))
    numeric = float(
        (log_evidence(PERIOD + step) - log_evidence(PERIOD - step)) / (2.0 * step)
    )

    assert np.isfinite(analytic)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5)


def test_evidence_prior_scan_matches_direct_evaluation():
    problem = _random_problem(seed=9)
    normal_matrix, normal_vector, chi2_data = _solve_linear_system(problem)
    scales = jnp.asarray([0.5, 1.0, 2.0, 10.0])
    scanned = np.asarray(lin.evidence_prior_scan(
        normal_matrix,
        normal_vector,
        chi2_data,
        jnp.asarray(problem["prior_variance"]),
        scales,
    ))
    direct = [
        float(lin.modulation_log_evidence(
            normal_matrix,
            normal_vector,
            chi2_data,
            jnp.asarray(problem["prior_variance"]) * float(scale) ** 2,
        )[0])
        for scale in scales
    ]

    np.testing.assert_allclose(scanned, direct, rtol=EXACT_RTOL)


def test_mixture_moments_match_the_law_of_total_variance():
    rng = np.random.default_rng(23)
    n_sample = 6
    means = rng.normal(size=(n_sample, N_COLUMN))
    factors = rng.normal(size=(n_sample, N_COLUMN, N_COLUMN)) * 0.1
    covariances = np.einsum("sij,skj->sik", factors, factors)

    mean, total, systematic = lin.mixture_moments(
        jnp.asarray(means), jnp.asarray(covariances)
    )
    centred = means - means.mean(axis=0)

    np.testing.assert_allclose(np.asarray(mean), means.mean(axis=0), rtol=EXACT_RTOL)
    np.testing.assert_allclose(
        np.asarray(systematic), centred.T @ centred / n_sample, rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(total),
        covariances.mean(axis=0) + centred.T @ centred / n_sample,
        rtol=EXACT_RTOL,
    )


def test_modulation_time_series_matches_an_explicit_sum():
    problem = _random_problem(seed=13)
    normal_matrix, normal_vector, _ = _solve_linear_system(problem)
    mean, covariance = lin.solve_modulation(
        normal_matrix,
        normal_vector,
        jnp.asarray(problem["prior_variance"]),
        N_RESPONSE,
        N_HARMONICS,
    )
    excursion, excursion_sd = lin.modulation_time_series(
        mean, covariance, jnp.asarray(problem["basis"])
    )

    np.testing.assert_allclose(
        np.asarray(excursion),
        problem["basis"] @ np.asarray(mean).T,
        rtol=EXACT_RTOL,
    )
    # The band is the diagonal of Phi C_pp Phi^T for each response block.
    blocks = np.asarray(covariance).reshape(N_RESPONSE, N_MODE, N_RESPONSE, N_MODE)
    reference = np.stack([
        np.sqrt(np.einsum(
            "tm,mn,tn->t", problem["basis"], blocks[p, :, p, :], problem["basis"]
        ))
        for p in range(N_RESPONSE)
    ], axis=1)

    np.testing.assert_allclose(np.asarray(excursion_sd), reference, rtol=DENSE_RTOL)


def test_predictive_log_density_matches_a_dense_joint():
    """The held-out density equals ``N(r; 0, Sigma + A C A^T)`` formed densely."""
    problem = _random_problem(seed=19)
    normal_matrix, _, _ = _solve_linear_system(problem)

    # A second, independent block standing in for the held-out epochs.
    test = _random_problem(seed=29)
    test_matrix, test_vector, test_chi2 = _solve_linear_system(test)
    test_determinant, n_test = lin.noise_log_determinant(
        jnp.asarray(test["var_point"]),
        test["offset_variance"],
        test["correlation_rho"],
    )
    density = lin.predictive_log_density(
        normal_matrix,
        jnp.asarray(problem["prior_variance"]),
        test_matrix,
        test_vector,
        test_chi2,
        test_determinant,
        n_test,
    )

    # Reference: the training data fix the amplitude covariance, and the test
    # block is then Gaussian with that covariance propagated through its design.
    posterior = np.linalg.inv(
        np.asarray(normal_matrix) + np.diag(1.0 / problem["prior_variance"])
    )
    design = _dense_design(test["log_jacobian"], test["basis"])
    blocks = _dense_noise_covariance(test["var_point"], test["correlation_rho"])
    from scipy.linalg import block_diag

    covariance = block_diag(*[
        block + test["offset_variance"] * np.ones_like(block) for block in blocks
    ])
    marginal = covariance + design @ posterior @ design.T
    residual = test["residual"].reshape(-1)
    _, log_determinant = np.linalg.slogdet(marginal)
    reference = -0.5 * (
        residual @ np.linalg.solve(marginal, residual)
        + log_determinant
        + residual.size * np.log(2.0 * np.pi)
    )

    np.testing.assert_allclose(float(density), reference, rtol=DENSE_RTOL)


def _linearised_configuration(**overrides):
    """A minimal registered dataset, built without touching any opacity file."""
    n_time, n_wavelength = 24, 9
    times = np.linspace(0.0, 2.0 * NOMINAL_PERIOD, n_time)
    wavelengths = np.linspace(1.0, 5.0, n_wavelength)
    spectrum = np.tile(1e-15 * (1.0 + 0.01 * np.sin(wavelengths)), (n_time, 1))
    configuration = RetrievalConfig(
        retrieval_name="linearised_test",
        run_mode="retrieval",
        sampler_type="jaxns",
        pressures=jnp.logspace(-5, 2, 20),
        scattering_in_emission=False,
    )
    configuration.add_parameter(
        name="period",
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.5 * NOMINAL_PERIOD, high=1.5 * NOMINAL_PERIOD),
    )
    configuration.parameters["period"].value = NOMINAL_PERIOD
    arguments = {
        "name": "linearised_test",
        "observation_times": times,
        "model_generating_function": lambda *args, **kwargs: None,
        "modulated_parameters": {"T_bottom": 100.0, "H2O": 1.0},
        "n_harmonics": 2,
        "reference_bin": 0,
        "n_phase_bins": 4,
        "wavelengths": wavelengths,
        "spectrum": spectrum,
        "uncertainties": 0.01 * spectrum,
    }
    arguments.update(overrides)
    configuration.add_linearised_time_series_data(**arguments)

    return configuration


def test_configuration_orders_prior_variance_to_match_the_column_names():
    """The flattened prior must line up with the amplitude labels, response-major."""
    configuration = _linearised_configuration()
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    names = lin.amplitude_column_names(("T_bottom", "H2O"), 2)
    prior_variance = np.asarray(state["prior_variance"])

    assert len(names) == prior_variance.size
    # T_bottom has a width of 100 and H2O of 1, so the first block is 1e4.
    np.testing.assert_allclose(prior_variance[:4], 100.0**2, rtol=EXACT_RTOL)
    np.testing.assert_allclose(prior_variance[4:], 1.0**2, rtol=EXACT_RTOL)
    assert all(name.startswith("T_bottom") for name in names[:4])
    assert all(name.startswith("H2O") for name in names[4:])


def test_configuration_registers_a_consistent_linearised_state():
    configuration = _linearised_configuration()
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    n_wavelength, n_time = np.asarray(state["var_point"]).shape

    assert np.asarray(state["log_spectrum"]).shape == (n_wavelength, n_time)
    # The reference set is one bin of equal counts, so it holds n_time / n_bins.
    assert np.asarray(state["reference_indices"]).size == n_time // 4
    np.testing.assert_allclose(
        float(np.sum(np.asarray(state["reference_weights"]))), 1.0, rtol=EXACT_RTOL
    )
    assert state["n_constrained"] == n_wavelength * n_time
    assert np.isfinite(state["noise_normalisation"])

    # The stored normalisation must be the one the evidence expects.
    log_determinant, n_constrained = lin.noise_log_determinant(
        jnp.asarray(state["var_point"]),
        state["offset_variance"],
        state["correlation_rho"],
    )
    np.testing.assert_allclose(
        state["noise_normalisation"],
        float(-0.5 * log_determinant - 0.5 * n_constrained * np.log(2.0 * np.pi)),
        rtol=DENSE_RTOL,
    )


def test_configuration_rejects_a_non_positive_prior_width():
    with pytest.raises(ValueError, match="finite and positive"):
        _linearised_configuration(modulated_parameters={"T_bottom": 0.0})


def test_configuration_rejects_an_empty_modulated_set():
    with pytest.raises(ValueError, match="at least one parameter"):
        _linearised_configuration(modulated_parameters={})


def test_infinite_offset_scale_drops_one_degree_of_freedom_per_wavelength():
    configuration = _linearised_configuration(static_offset_scale=None)
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    n_wavelength, n_time = np.asarray(state["var_point"]).shape

    assert np.isinf(state["offset_variance"])
    assert state["n_constrained"] == n_wavelength * (n_time - 1)


def test_nuisance_modes_extend_the_design_and_the_prior():
    """Extra response columns must line up with extra prior entries."""
    n_modes = 3
    configuration = _linearised_configuration(
        nuisance_response_modes=n_modes, nuisance_prior_width=0.02
    )
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    modes = np.asarray(state["nuisance_modes"])
    n_wavelength, n_time = np.asarray(state["var_point"]).shape
    n_mode = 2 * state["n_harmonics"]

    assert modes.shape == (n_wavelength, n_modes)
    # Two physical parameters plus three nuisance modes, each with 2K amplitudes.
    assert np.asarray(state["prior_variance"]).size == (2 + n_modes) * n_mode
    np.testing.assert_allclose(
        np.asarray(state["prior_variance"])[-n_modes * n_mode:],
        0.02**2,
        rtol=EXACT_RTOL,
    )

    # The concatenated Jacobian solves without special-casing the extra columns.
    rng = np.random.default_rng(31)
    log_jacobian = np.concatenate(
        [rng.normal(size=(n_wavelength, 2)), modes], axis=1
    )
    basis = np.asarray(lin.reference_anchored_basis(
        jnp.asarray(state["times"]),
        NOMINAL_PERIOD,
        jnp.asarray(state["reference_indices"]),
        state["n_harmonics"],
        jnp.asarray(state["reference_weights"]),
    ))
    normal_matrix, normal_vector, _ = lin.build_linear_system(
        jnp.asarray(log_jacobian),
        jnp.asarray(basis),
        jnp.asarray(rng.normal(size=(n_wavelength, n_time)) * 1e-3),
        jnp.asarray(state["var_point"]),
        state["offset_variance"],
        state["correlation_rho"],
    )
    mean, _ = lin.solve_modulation(
        normal_matrix,
        normal_vector,
        jnp.asarray(state["prior_variance"]),
        2 + n_modes,
        state["n_harmonics"],
    )

    assert np.asarray(mean).shape == (2 + n_modes, n_mode)
    assert np.all(np.isfinite(np.asarray(mean)))


def test_nuisance_modes_require_a_width_without_a_static_offset_scale():
    with pytest.raises(ValueError, match="no static scale"):
        _linearised_configuration(
            nuisance_response_modes=2, static_offset_scale=None
        )


def _stub_retrieval(configuration):
    """A Retrieval carrying only a configuration, for the name-resolution helpers."""
    retrieval = object.__new__(Retrieval)
    retrieval.configuration = configuration

    return retrieval


def test_resolving_the_dataset_name_reports_absence_and_ambiguity():
    configuration = _linearised_configuration()
    retrieval = _stub_retrieval(configuration)

    assert retrieval._resolve_linearised_data_name(None) == "linearised_test"
    assert (
        retrieval._resolve_linearised_data_name("linearised_test") == "linearised_test"
    )

    with pytest.raises(ValueError, match="not a registered linearised dataset"):
        retrieval._resolve_linearised_data_name("absent")

    empty = RetrievalConfig(
        retrieval_name="empty",
        run_mode="retrieval",
        sampler_type="jaxns",
        pressures=jnp.logspace(-5, 2, 20),
        scattering_in_emission=False,
    )

    with pytest.raises(ValueError, match="no linearised time series"):
        _stub_retrieval(empty)._resolve_linearised_data_name(None)


def test_response_names_label_physical_parameters_then_nuisances():
    configuration = _linearised_configuration(
        nuisance_response_modes=2, nuisance_prior_width=0.02
    )
    retrieval = _stub_retrieval(configuration)

    assert retrieval._linearised_response_names("linearised_test", 4) == (
        "T_bottom", "H2O", "nuisance_mode_0", "nuisance_mode_1",
    )

    with pytest.raises(ValueError, match="response columns"):
        retrieval._linearised_response_names("linearised_test", 1)


def test_runtime_assembles_the_same_system_as_the_module():
    """The runtime path and a direct call to the algebra must agree exactly."""
    configuration = _linearised_configuration()
    observation = configuration.data["linearised_test"].to_observation_state()
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    n_wavelength, n_time = np.asarray(state["var_point"]).shape

    rng = np.random.default_rng(37)
    wavelengths = jnp.asarray(np.linspace(1.0, 5.0, n_wavelength))
    reference = jnp.asarray(1e-15 * (1.0 + 0.01 * np.sin(np.asarray(wavelengths))))
    flux_jacobian = jnp.asarray(rng.normal(size=(n_wavelength, 2)) * 1e-16)

    normal_matrix, normal_vector, chi2_data, basis, log_jacobian, returned = (
        compute_linearised_system(
            wavelengths,
            reference,
            flux_jacobian,
            observation,
            configuration.parameters,
            "linearised_test",
        )
    )

    assert returned is state
    assert log_jacobian.shape == (n_wavelength, 2)
    assert basis.shape == (n_time, 2 * state["n_harmonics"])

    # The same numbers, assembled by hand from the module's own pieces.
    expected_basis = lin.reference_anchored_basis(
        jnp.asarray(state["times"]),
        configuration.parameters["period"].value,
        jnp.asarray(state["reference_indices"]),
        state["n_harmonics"],
        jnp.asarray(state["reference_weights"]),
    )
    residual = jnp.asarray(state["log_spectrum"]) - jnp.log(reference)[:, None]
    expected = lin.build_linear_system(
        flux_jacobian / reference[:, None],
        expected_basis,
        residual,
        jnp.asarray(state["var_point"]),
        state["offset_variance"],
        state["correlation_rho"],
    )

    np.testing.assert_allclose(
        np.asarray(normal_matrix), np.asarray(expected[0]), rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(normal_vector), np.asarray(expected[1]), rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(float(chi2_data), float(expected[2]), rtol=EXACT_RTOL)


def test_runtime_appends_nuisance_modes_to_the_design():
    configuration = _linearised_configuration(
        nuisance_response_modes=3, nuisance_prior_width=0.02
    )
    observation = configuration.data["linearised_test"].to_observation_state()
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    n_wavelength = np.asarray(state["var_point"]).shape[0]

    rng = np.random.default_rng(41)
    wavelengths = jnp.asarray(np.linspace(1.0, 5.0, n_wavelength))
    reference = jnp.asarray(1e-15 * (1.0 + 0.01 * np.sin(np.asarray(wavelengths))))
    _, _, _, _, log_jacobian, _ = compute_linearised_system(
        wavelengths,
        reference,
        jnp.asarray(rng.normal(size=(n_wavelength, 2)) * 1e-16),
        observation,
        configuration.parameters,
        "linearised_test",
    )

    # Two physical responses plus three nuisance modes.
    assert log_jacobian.shape == (n_wavelength, 5)
    np.testing.assert_allclose(
        np.asarray(log_jacobian)[:, 2:],
        np.asarray(state["nuisance_modes"]),
        rtol=EXACT_RTOL,
    )


def _stub_plotter(configuration, base_figure_name):
    """A RetrievalPlotter carrying only what the modulation plot reads."""
    retrieval = _stub_retrieval(configuration)
    retrieval.comm = None
    retrieval.rank = 0
    retrieval.get_base_figure_name = lambda: str(base_figure_name)

    return RetrievalPlotter(retrieval)


def test_modulation_reference_states_fall_back_to_the_best_fit():
    configuration = _linearised_configuration()
    plotter = _stub_plotter(configuration, "unused")

    assert plotter._modulation_reference_states(None, None, 4) == ["best fit"]

    names = ["alpha", "beta"]
    samples = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-7.0, -8.0, -9.0]])
    states = plotter._modulation_reference_states(
        {configuration.retrieval_name: samples}, names, 5
    )

    # One state per requested draw, each holding the sampled parameters only:
    # the trailing log-likelihood row of the sample array is not a parameter.
    assert len(states) == 5

    for state in states:
        assert list(state) == names
        assert state["alpha"] in set(samples[0])
        assert state["beta"] in set(samples[1])


def test_plot_modulation_draws_the_cached_mixture(tmp_path):
    configuration = _linearised_configuration()
    base_figure_name = tmp_path / "figure"
    plotter = _stub_plotter(configuration, base_figure_name)

    rng = np.random.default_rng(97)
    times = np.linspace(0.0, 3.0 * PERIOD, N_TIME)
    basis = np.asarray(lin.reference_anchored_basis(
        jnp.asarray(times),
        PERIOD,
        jnp.asarray([0]),
        N_HARMONICS,
        jnp.asarray([1.0]),
    ))
    n_column = N_RESPONSE * N_MODE
    means = rng.normal(size=(4, n_column))
    factors = rng.normal(size=(4, n_column, n_column))
    covariances = np.einsum("sij,skj->sik", factors, factors) / n_column

    np.savez(
        f"{base_figure_name}_modulation_linearised_test.npz",
        basis=basis,
        covariances=covariances,
        means=means,
        response_names=np.asarray(["first", "second", "third"]),
        times=times,
    )

    figure, axes = plotter.plot_modulation(data_name="linearised_test")

    assert (tmp_path / "figure_modulation_linearised_test.pdf").exists()
    assert len(axes) == N_RESPONSE

    # The drawn curve is the mixture mean pushed through the harmonic basis, and
    # the two bands must bracket each other: the total variance is the
    # statistical variance plus a non-negative reference-state term.
    expected = np.einsum(
        "pm,tm->tp", means.mean(axis=0).reshape(N_RESPONSE, N_MODE), basis
    )

    for i_response, ax in enumerate(axes):
        np.testing.assert_allclose(
            ax.lines[-1].get_xdata(), times / 3600.0, rtol=DENSE_RTOL
        )
        np.testing.assert_allclose(
            ax.lines[-1].get_ydata(), expected[:, i_response], rtol=DENSE_RTOL
        )

    total_band = axes[0].collections[0].get_paths()[0].vertices[:, 1]
    statistical_band = axes[0].collections[1].get_paths()[0].vertices[:, 1]
    assert total_band.max() >= statistical_band.max()
    assert total_band.min() <= statistical_band.min()

    plt.close(figure)


def _modulation_posterior_retrieval(configuration, flux_jacobian):
    """A Retrieval whose model returns a fixed reference spectrum and Jacobian.

    The glue under test is everything between the model call and the solve, so
    the radiative transfer is replaced by a constant: a model that returns the
    same spectrum and Jacobian at every state isolates the assembly from the
    physics.
    """
    data_object = configuration.data["linearised_test"]
    n_wavelength = np.asarray(data_object.wavelengths).shape[-1]
    wavelengths = jnp.asarray(np.linspace(1.0, 5.0, n_wavelength))
    reference = jnp.asarray(1e-15 * (1.0 + 0.01 * np.sin(np.asarray(wavelengths))))

    retrieval = _stub_retrieval(configuration)
    retrieval.runtime = types.SimpleNamespace(
        observations={"linearised_test": data_object.to_observation_state()}
    )
    retrieval._call_model_generating_function = (
        lambda *args, **kwargs: (wavelengths, reference, {"flux_jacobian": flux_jacobian})
    )

    return retrieval, wavelengths, reference


def test_modulation_posterior_matches_a_direct_solve():
    configuration = _linearised_configuration()
    state = configuration.data["linearised_test"].time_series_config["linearised"]
    n_wavelength = np.asarray(state["var_point"]).shape[0]

    rng = np.random.default_rng(23)
    flux_jacobian = jnp.asarray(rng.normal(size=(n_wavelength, 2)) * 1e-16)
    retrieval, wavelengths, reference = _modulation_posterior_retrieval(
        configuration, flux_jacobian
    )

    posterior = retrieval.calculate_modulation_posterior()

    normal_matrix, normal_vector, chi2_data, basis, _, resolved = (
        compute_linearised_system(
            wavelengths,
            reference,
            flux_jacobian,
            retrieval.runtime.observations["linearised_test"],
            configuration.parameters,
            "linearised_test",
        )
    )
    amplitudes, covariance = lin.solve_modulation(
        normal_matrix,
        normal_vector,
        jnp.asarray(resolved["prior_variance"]),
        2,
        resolved["n_harmonics"],
    )

    np.testing.assert_allclose(
        np.asarray(posterior["amplitudes"]), np.asarray(amplitudes), rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(posterior["covariance"]), np.asarray(covariance), rtol=EXACT_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(posterior["basis"]), np.asarray(basis), rtol=EXACT_RTOL
    )
    assert posterior["response_names"] == ("T_bottom", "H2O")
    assert posterior["column_names"] == (
        "T_bottom:c1",
        "T_bottom:s1",
        "T_bottom:c2",
        "T_bottom:s2",
        "H2O:c1",
        "H2O:s1",
        "H2O:c2",
        "H2O:s2",
    )

    # The excursion is the amplitudes pushed through the basis, so the two
    # reported time series must agree with that product exactly.
    np.testing.assert_allclose(
        np.asarray(posterior["excursion"]),
        np.einsum("pm,tm->tp", np.asarray(amplitudes), np.asarray(basis)),
        rtol=EXACT_RTOL,
    )


def test_modulation_posterior_requires_a_flux_jacobian():
    configuration = _linearised_configuration()
    retrieval, wavelengths, reference = _modulation_posterior_retrieval(
        configuration, jnp.zeros((9, 2))
    )

    retrieval._call_model_generating_function = (
        lambda *args, **kwargs: (wavelengths, reference)
    )

    with pytest.raises(ValueError, match="no auxiliary outputs"):
        retrieval.calculate_modulation_posterior()

    retrieval._call_model_generating_function = (
        lambda *args, **kwargs: (wavelengths, reference, {})
    )

    with pytest.raises(ValueError, match="without a 'flux_jacobian'"):
        retrieval.calculate_modulation_posterior()
