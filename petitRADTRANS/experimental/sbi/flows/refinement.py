"""Test-time, per-observation posterior refinement.

Free functions that sharpen the amortized posterior for one specific target
spectrum using the differentiable forward model: a flow-ELBO refinement
(:func:`refine_posterior_on_observation`) and a MAP + Laplace / Gaussian-VI
refinement (:func:`map_laplace_posterior_on_observation`). Each takes the trained
:class:`~petitRADTRANS.experimental.sbi.flows.posterior.ConditionalFlowPosterior` as its first
argument (``posterior``); the estimator exposes them as thin methods. Extracted
from ``flows.posterior`` to keep the estimator focused on amortized inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import copy

import jax
from jax import Array, random, vmap, nn
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from .cube import _clip_cube_support
from .objectives import batch_embeddings

if TYPE_CHECKING:
    from .posterior import ConditionalFlowPosterior


def _tile_prestacked_observation(observation: Any, n_tiles: int) -> Any:
    """Repeat a single (batch-1) prestacked observation ``n_tiles`` times."""
    from ..observation import PreStackedObservations

    def _tile(arr: Any) -> jnp.ndarray:
        arr = jnp.asarray(arr)
        return jnp.repeat(arr, n_tiles, axis=0)

    return PreStackedObservations(
        stacked_blocks=tuple(
            tuple(_tile(arr) for arr in block)
            for block in observation.stacked_blocks
        ),
        modalities=observation.modalities,
    )


def refine_posterior_on_observation(
    posterior,
    target_observation: Any,
    elbo_log_likelihood_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
    *,
    num_steps: int = 400,
    learning_rate: float = 1e-4,
    num_samples: int = 16,
    beta_start: float = 1e-3,
    beta_anneal_fraction: float = 0.5,
    gradient_clip_norm: float = 1.0,
    gradient_clip_value: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> "ConditionalFlowPosterior":
    """Semi-amortized test-time refinement of the posterior for one
    observation (flow-based ELBO variant).

    Starting from the trained amortized model, optimize the
    single-observation tempered ELBO

        ELBO_beta(x_obs) = E_{q(theta|c)}[ beta * log p(x_obs|theta)
                                           - log q(theta|c) ]

    adapting the flow while holding the encoder (hence the conditioning
    embedding ``c = encoder(x_obs)``) fixed. ``beta`` is annealed soft -> 1
    across the steps so the very small observational noise does not blow up
    the single-observation gradient early (which previously made this step
    diverge); a low learning rate and gradient clipping (both by global norm
    and by value) further stabilize it. The **best beta=1 ELBO** state seen
    during optimization is returned (not the last), so a late instability
    cannot degrade the result.

    Returns a shallow copy of this posterior whose flow is specialized to
    ``target_observation``; use it only for inference on that observation.

    See also :meth:`map_laplace_posterior_on_observation`, the more robust
    MAP + Laplace variant that sidesteps reverse-KL under-dispersion.
    """
    if posterior.parameter_space != "cube":
        raise ValueError(
            "Semi-amortized refinement requires parameter_space='cube'."
        )
    n_draws = max(int(num_samples), 1)
    beta_start = float(beta_start)
    anneal_fraction = float(beta_anneal_fraction)

    embedding = batch_embeddings(
        posterior.model, target_observation
    )
    embedding = jax.lax.stop_gradient(
        jnp.asarray(embedding, dtype=jnp.float32)
    )
    param_dim = posterior.model.flow.parameter_dim
    target_tiled = _tile_prestacked_observation(target_observation, n_draws)
    embedding_tiled = jnp.broadcast_to(
        embedding, (n_draws, embedding.shape[-1])
    )

    def _refine_loss(flow: Any, key: Array, beta: Any) -> tuple:
        latent = flow._base_sample(key, (n_draws, param_dim), jnp.float32)
        theta_unc, log_det_inv = vmap(flow.inverse)(latent, embedding_tiled)
        base_log_prob = vmap(flow._base_log_prob)(latent)
        log_q_unc = base_log_prob - log_det_inv
        theta_cube = nn.sigmoid(theta_unc)
        clipped = jnp.clip(theta_cube, 1e-7, 1.0 - 1e-7)
        log_jacobian = jnp.sum(
            jnp.log(clipped) + jnp.log1p(-clipped), axis=-1
        )
        log_q_cube = log_q_unc - log_jacobian
        log_likelihood = jnp.reshape(
            elbo_log_likelihood_fn(theta_cube, target_tiled), (n_draws,)
        )
        tempered = jnp.where(
            jnp.isfinite(beta * log_likelihood - log_q_cube),
            beta * log_likelihood - log_q_cube,
            -1e6,
        )
        # beta=1 ELBO, used only for best-state tracking.
        elbo_full = jnp.where(
            jnp.isfinite(log_likelihood - log_q_cube),
            log_likelihood - log_q_cube,
            -1e6,
        )
        return -jnp.mean(tempered), (
            jnp.mean(log_likelihood),
            -jnp.mean(elbo_full),
        )

    optimizer = optax.chain(
        optax.clip(float(gradient_clip_value)),
        optax.clip_by_global_norm(float(gradient_clip_norm)),
        optax.adam(float(learning_rate)),
    )
    flow = posterior.model.flow
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def _step(_flow: Any, _opt_state: Any, _key: Array, _beta: Any) -> tuple:
        (loss_val, (mean_ll, neg_elbo_full)), grads = (
            eqx.filter_value_and_grad(_refine_loss, has_aux=True)(
                _flow, _key, _beta
            )
        )
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g))
            if g is not None
            else None,
            grads,
            is_leaf=lambda x: x is None,
        )
        updates, _opt_state = optimizer.update(
            grads, _opt_state, eqx.filter(_flow, eqx.is_inexact_array)
        )
        return (
            eqx.apply_updates(_flow, updates),
            _opt_state,
            loss_val,
            mean_ll,
            neg_elbo_full,
        )

    def _beta_for_step(index: int) -> float:
        if anneal_fraction <= 0.0 or beta_start >= 1.0:
            return 1.0
        window = max(1.0, anneal_fraction * float(num_steps))
        progress = min(1.0, float(index) / window)
        return float(beta_start ** (1.0 - progress))

    best_flow = flow
    best_obj = float("inf")
    key = random.PRNGKey(int(seed))
    for step_index in range(int(num_steps)):
        key, step_key = random.split(key)
        beta = jnp.asarray(_beta_for_step(step_index), dtype=jnp.float32)
        flow, opt_state, loss_val, mean_ll, neg_elbo_full = _step(
            flow, opt_state, step_key, beta
        )
        obj = float(neg_elbo_full)
        if np.isfinite(obj) and obj < best_obj:
            best_obj = obj
            best_flow = flow
        if verbose and (
            step_index % 50 == 0 or step_index == int(num_steps) - 1
        ):
            print(
                f"  refine step {step_index:4d} | beta {float(beta):.3g} "
                f"| -ELBO(beta) {float(loss_val):.4f} "
                f"| -ELBO(beta=1) {obj:.4f} | best {best_obj:.4f} "
                f"| mean log p(x|theta) {float(mean_ll):.4f}"
            )

    refined_model = eqx.tree_at(lambda m: m.flow, posterior.model, best_flow)
    refined = copy.copy(posterior)
    refined.model = refined_model
    return refined


def _chunked_forward_chi2(
    cube_samples: jnp.ndarray,
    standardized_residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    forward_batch_size: int,
    verbose: bool = False,
    label: str = "forward model",
) -> tuple[np.ndarray, int]:
    """Chi-squared for a batch of cube vectors, chunked with OOM backoff.

    Shared by importance sampling and by MAP start selection so the memory
    handling lives in one place. Each chunk blocks and copies to the host: a
    lazily dispatched loop keeps every chunk's device buffers live and
    surfaces allocator failures at the first host sync rather than at the
    forward-model call that caused them. On out-of-memory the batch is halved
    and retried, because per-sample forward-model memory is task-dependent
    (scattering emission holds far larger intermediates than transmission).

    Returns the host-side chi-squared per sample and the batch size that
    actually worked.
    """
    n_samples = int(cube_samples.shape[0])
    residual_batch = jax.jit(vmap(standardized_residual_fn))

    def _chi2_for_chunk(chunk: jnp.ndarray) -> np.ndarray:
        residuals = residual_batch(chunk)
        chunk_chi2 = jnp.sum(residuals * residuals, axis=-1)
        chunk_chi2 = jnp.reshape(chunk_chi2, (-1,))
        chunk_chi2 = jax.block_until_ready(chunk_chi2)
        host_chi2 = np.asarray(chunk_chi2, dtype=np.float64)
        del residuals, chunk_chi2
        return host_chi2

    def _is_out_of_memory(error: Exception) -> bool:
        message = str(error)
        return "RESOURCE_EXHAUSTED" in message or "Out of memory" in message

    chi2_parts: list[np.ndarray] = []
    active_batch_size = forward_batch_size
    start = 0
    while start < n_samples:
        chunk = cube_samples[start : start + active_batch_size]
        try:
            chi2_parts.append(_chi2_for_chunk(chunk))
        except Exception as error:  # noqa: BLE001 - re-raised unless OOM
            if not _is_out_of_memory(error) or active_batch_size <= 1:
                raise
            active_batch_size = max(1, active_batch_size // 2)
            if verbose:
                print(
                    f"  {label}: forward-model chunk hit an out-of-memory "
                    f"error; retrying with forward batch {active_batch_size}"
                )
            continue
        start += chunk.shape[0]

    return np.concatenate(chi2_parts), active_batch_size


def importance_sampling_posterior_on_observation(
    posterior,
    target_observation: Any,
    standardized_residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    n_proposal_samples: int = 16384,
    forward_batch_size: int = 256,
    n_posterior_samples: int = 4096,
    seed: int = 0,
    verbose: bool = True,
) -> "ConditionalFlowPosterior":
    """Likelihood importance sampling on top of the amortized posterior.

    Draws proposal samples from the amortized flow conditioned on the target
    observation, evaluates the exact Gaussian likelihood of each through the
    differentiable forward model, and reweights by
    ``log w = log L(theta) + log pi(theta) - log q(theta | x)``. This is the
    field-standard correction for amortized posteriors (Gebhard et al. 2025):
    unlike MAP+Laplace it cannot collapse to a degenerate covariance, and its
    sampling efficiency ``eps = ESS / N`` is a built-in trust diagnostic —
    ``eps`` near 1 means the amortized posterior already matches the true one;
    ``eps`` near 0 means the amortized posterior misses the true posterior
    mass and the importance estimate itself should not be trusted either.

    Weights are computed in the flow's unconstrained (logit-of-cube) space,
    where the uniform-cube prior is the standard logistic density and the
    proposal density is the flow's own ``log_prob`` — both exact.

    Parameters
    ----------
    target_observation:
        Batch-1 prestacked target observation (conditions the proposal).
    standardized_residual_fn:
        Differentiable ``theta_cube[param_dim] -> (n_wavelengths,)`` returning
        noise-standardized residuals ``(model(theta) - x_obs)/sigma`` for the
        target; masked channels contribute 0.
    n_proposal_samples:
        Number of amortized proposal draws to weight.
    forward_batch_size:
        Forward-model chunk size (vmapped); bounds device memory exactly like
        the ELBO per-batch forward passes did.
    n_posterior_samples:
        Number of importance-resampled posterior draws stored on the returned
        estimator.

    Returns
    -------
    ConditionalFlowPosterior
        A shallow copy of ``posterior`` whose ``_fixed_cube_samples`` hold the
        importance-resampled posterior, with the weighting diagnostics attached
        as ``_importance_sampling_diagnostics``.
    """
    if posterior.parameter_space != "cube":
        raise ValueError("Importance sampling requires parameter_space='cube'.")

    n_proposal_samples = int(n_proposal_samples)
    forward_batch_size = max(int(forward_batch_size), 1)

    embedding = jnp.asarray(
        batch_embeddings(posterior.model, target_observation), dtype=jnp.float32
    )
    key = random.PRNGKey(int(seed))
    unconstrained = posterior.model.flow.sample(
        context=embedding[0], n_samples=n_proposal_samples, key=key
    )
    log_q = vmap(lambda w: posterior.model.flow.log_prob(w, embedding[0]))(
        unconstrained
    )
    # Uniform-cube prior expressed in logit space: standard logistic per dim.
    log_prior = -jnp.sum(
        nn.softplus(unconstrained) + nn.softplus(-unconstrained), axis=1
    )
    cube_samples = _clip_cube_support(nn.sigmoid(unconstrained))

    chi2, active_batch_size = _chunked_forward_chi2(
        cube_samples,
        standardized_residual_fn,
        forward_batch_size,
        verbose=verbose,
        label="importance sampling",
    )

    # Weights are computed on the host in float64: the arrays are one scalar
    # per proposal (a few hundred kB), and this keeps the normalisation exact
    # and free of device-memory effects.
    log_likelihood = -0.5 * chi2
    log_prior_np = np.asarray(log_prior, dtype=np.float64).reshape(-1)
    log_q_np = np.asarray(log_q, dtype=np.float64).reshape(-1)
    if not (log_likelihood.shape == log_prior_np.shape == log_q_np.shape):
        raise RuntimeError(
            "Importance-sampling weight terms must all be 1-D and equal length; "
            f"got chi2 {log_likelihood.shape}, log_prior {log_prior_np.shape}, "
            f"log_q {log_q_np.shape}. The residual function must return one "
            "vector of standardized residuals per parameter vector."
        )

    log_weights = log_likelihood + log_prior_np - log_q_np
    log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)

    log_weight_max = float(np.max(log_weights))
    raw = np.exp(log_weights - log_weight_max)
    weight_sum = float(np.sum(raw))
    weights = raw / weight_sum
    effective_sample_size = float(1.0 / np.sum(weights**2))
    efficiency = effective_sample_size / float(n_proposal_samples)
    # Evidence up to the (theta-independent) Gaussian normalizer of the
    # likelihood, which the standardized residuals do not carry.
    log_evidence_minus_const = float(
        log_weight_max + np.log(weight_sum) - np.log(n_proposal_samples)
    )

    # Systematic resampling by the normalized weights.
    rng = np.random.default_rng(int(seed) + 1)
    positions = (rng.uniform() + np.arange(int(n_posterior_samples))) / float(
        n_posterior_samples
    )
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    resampled_indices = np.searchsorted(cumulative, positions)
    cube_np = np.asarray(cube_samples, dtype=np.float64)

    best_index = int(np.argmax(log_likelihood + log_prior_np))
    diagnostics = {
        "n_proposal_samples": n_proposal_samples,
        "effective_sample_size": effective_sample_size,
        "sampling_efficiency": efficiency,
        "max_weight_fraction": float(np.max(weights)),
        "chi2_best": float(chi2[best_index]),
        "chi2_weighted_mean": float(np.sum(weights * chi2)),
        "forward_batch_size_used": int(active_batch_size),
        "log_evidence_minus_const": log_evidence_minus_const,
    }

    if verbose:
        print(
            f"  importance sampling done | efficiency {efficiency:.3%} "
            f"(ESS {effective_sample_size:.1f} / {n_proposal_samples}) "
            f"| chi2(best) {diagnostics['chi2_best']:.2f} "
            f"| max weight {diagnostics['max_weight_fraction']:.3f}"
        )
        n_data = int(np.size(np.asarray(standardized_residual_fn(cube_samples[0]))))
        chi2_per_point = diagnostics["chi2_best"] / max(n_data, 1)
        if chi2_per_point > 100.0:
            print(
                f"  WARNING: chi2/N at the best proposal is {chi2_per_point:.3g}, "
                "far above the ~1 expected for a well-fitting model. That points "
                "at the LIKELIHOOD rather than the posterior -- e.g. the "
                "observation reaching the residual function is not in the same "
                "units/normalization as the forward model output. Check the "
                "residual construction before interpreting these weights."
            )
        if efficiency < 1e-3:
            print(
                "  WARNING: sampling efficiency < 0.1% — the amortized posterior "
                "misses the true posterior mass for this observation; the "
                "importance-weighted result is itself unreliable."
            )

    refined = copy.copy(posterior)
    refined._fixed_cube_samples = cube_np[resampled_indices]
    refined._fixed_point_estimate_cube = cube_np[best_index]
    refined._importance_sampling_diagnostics = diagnostics
    return refined


def map_laplace_posterior_on_observation(
    posterior,
    target_observation: Any,
    standardized_residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    num_steps: int = 600,
    learning_rate: float = 5e-2,
    beta_start: float = 1e-3,
    beta_anneal_fraction: float = 0.5,
    n_posterior_samples: int = 4096,
    gradient_clip_value: float = 10.0,
    eigenvalue_floor: float = 1e-6,
    covariance_method: str = "laplace",
    covariance_chi2_inflation: bool = False,
    vi_num_steps: int = 300,
    vi_learning_rate: float = 1e-2,
    vi_num_samples: int = 32,
    vi_freeze_mean: bool = True,
    vi_beta: float = 1.0,
    num_starts: int = 1,
    start_selection: str = "amortized_mean",
    num_scored_starts: int = 512,
    start_forward_batch_size: int = 256,
    convergence_chi2_per_n: float | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> "ConditionalFlowPosterior":
    """MAP + Laplace test-time inference for one observation (robust path).

    Optimizes the (beta-annealed -> 1) log-posterior in the unconstrained
    logit space ``u`` (``theta_cube = sigmoid(u)``, uniform-cube prior ->
    logistic prior on ``u``) to the MAP, initialized from the amortized
    posterior mean for the target, then forms a Gauss-Newton Laplace
    covariance from the forward-model Jacobian and the prior curvature:

        precision Lambda = J^T J + diag(2 c (1 - c)),  J = d r / d u

    where ``r`` are the noise-standardized residuals. Posterior samples are
    drawn as ``u ~ N(u*, Lambda^-1)`` and mapped through the sigmoid. This
    gives the correct location (MAP) and correctly-scaled, per-direction
    local widths (tight where well-constrained, e.g. planet radius; moderate
    on degenerate directions), sidestepping the reverse-KL under-dispersion
    of the flow ELBO.

    Returns a shallow copy of this posterior carrying the Laplace cube
    samples; its :meth:`sample_posterior` returns those samples, so the
    existing inference/predictive pipeline can consume it unchanged. Use it
    only for inference on ``target_observation``.

    Parameters
    ----------
    target_observation:
        Batch-1 prestacked target observation (used only for the amortized
        MAP initialization via the encoder).
    standardized_residual_fn:
        Differentiable ``(theta_cube[param_dim]) -> (n_wavelengths,)``
        returning noise-standardized residuals ``(model(theta) - x_obs)/sigma``
        for the target. Masked channels should contribute 0.
    covariance_method:
        ``"hessian"`` (the exact Hessian of the negative log posterior at the
        MAP -- the Laplace approximation as defined), ``"laplace"`` (its
        Gauss-Newton approximation, which drops the residual-curvature term and
        is therefore too narrow whenever the fit is imperfect), or
        ``"gaussian_vi"`` (a short mean-field Gaussian VI; note that being
        mean-field and reverse-KL it estimates *conditional* rather than
        marginal widths, so it is expected to be narrower still).
    covariance_chi2_inflation:
        Scale the covariance by ``chi2/dof`` when the fit is worse than its
        degrees of freedom. The factor is reported either way; enabling this
        applies it.
    """
    if posterior.parameter_space != "cube":
        raise ValueError("MAP+Laplace requires parameter_space='cube'.")
    param_dim = posterior.model.flow.parameter_dim
    beta_start = float(beta_start)
    anneal_fraction = float(beta_anneal_fraction)

    # Amortized initialization: posterior mean (cube) for the target.
    embedding = batch_embeddings(
        posterior.model, target_observation
    )
    embedding = jnp.asarray(embedding, dtype=jnp.float32)
    init_unconstrained = posterior.model.flow.sample(
        context=embedding[0],
        n_samples=max(int(num_scored_starts), 1),
        key=random.PRNGKey(int(seed)),
    )
    init_cube = jnp.clip(nn.sigmoid(init_unconstrained).mean(axis=0), 1e-4, 1.0 - 1e-4)
    u0 = jnp.log(init_cube) - jnp.log1p(-init_cube)
    start_chi2: np.ndarray | None = None
    if start_selection == "likelihood_ranked":
        # Score the amortized draws with one batched forward-model pass and
        # start from the best. The mean of a posterior that is near-prior in
        # most dimensions is a low-likelihood point between the modes, which
        # is why 6 of 16 held-out cases were landing at chi2/N of 11-131 and
        # dragging interval coverage to 0.611 against a nominal 0.90 --
        # 1 - 6/16 accounts for almost exactly that shortfall. One forward
        # batch is cheap beside several hundred optimizer steps per start.
        scored_cube = _clip_cube_support(nn.sigmoid(init_unconstrained))
        start_chi2, _ = _chunked_forward_chi2(
            scored_cube,
            standardized_residual_fn,
            int(start_forward_batch_size),
            verbose=verbose,
            label="MAP start selection",
        )
        ranking = np.argsort(start_chi2)
        init_unconstrained = init_unconstrained[jnp.asarray(ranking)]
        best_cube = jnp.clip(nn.sigmoid(init_unconstrained[0]), 1e-4, 1.0 - 1e-4)
        u0 = jnp.log(best_cube) - jnp.log1p(-best_cube)
    # Multi-start: the amortized init can be biased, and a single gradient
    # descent from a bad start can run a coordinate into the cube boundary
    # where the sigmoid parametrization has vanishing gradients (observed:
    # chi2(MAP) ~ 1e9 with the radius pinned at the prior edge). Additional
    # starts from individual amortized draws let at least one path avoid the
    # trap; the best final full-objective start wins.
    start_points = [u0]
    for start_index in range(max(int(num_starts) - 1, 0)):
        draw_cube = jnp.clip(
            nn.sigmoid(init_unconstrained[start_index]), 1e-4, 1.0 - 1e-4
        )
        start_points.append(jnp.log(draw_cube) - jnp.log1p(-draw_cube))

    def _residual_of_u(u: jnp.ndarray) -> jnp.ndarray:
        return standardized_residual_fn(nn.sigmoid(u))

    def _neg_log_posterior(u: jnp.ndarray, beta: Any) -> jnp.ndarray:
        residual = _residual_of_u(u)
        log_likelihood = -0.5 * jnp.sum(residual * residual)
        log_prior = -jnp.sum(nn.softplus(u) + nn.softplus(-u))
        return -(beta * log_likelihood + log_prior)

    optimizer = optax.chain(
        optax.clip(float(gradient_clip_value)),
        optax.adam(float(learning_rate)),
    )
    opt_state = optimizer.init(u0)

    @jax.jit
    def _step(u: jnp.ndarray, state: Any, beta: Any) -> tuple:
        obj_beta, grad = jax.value_and_grad(_neg_log_posterior)(u, beta)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        updates, state = optimizer.update(grad, state)
        u = optax.apply_updates(u, updates)
        # Keep every coordinate away from the sigmoid-saturation region
        # (|u| = 12 -> cube within ~6e-6 of the boundary): beyond it the
        # gradient through sigmoid(u) vanishes and the coordinate freezes.
        u = jnp.clip(u, -12.0, 12.0)
        obj_full = _neg_log_posterior(u, jnp.asarray(1.0, jnp.float32))
        return u, state, obj_full

    def _beta_for_step(index: int) -> float:
        if anneal_fraction <= 0.0 or beta_start >= 1.0:
            return 1.0
        window = max(1.0, anneal_fraction * float(num_steps))
        progress = min(1.0, float(index) / window)
        return float(beta_start ** (1.0 - progress))

    best_u = u0
    best_obj = float(_neg_log_posterior(u0, jnp.asarray(1.0, jnp.float32)))
    for start_number, u_start in enumerate(start_points):
        u = u_start
        opt_state = optimizer.init(u)
        for step_index in range(int(num_steps)):
            beta = jnp.asarray(_beta_for_step(step_index), dtype=jnp.float32)
            u, opt_state, obj_full = _step(u, opt_state, beta)
            obj = float(obj_full)
            if np.isfinite(obj) and obj < best_obj:
                best_obj = obj
                best_u = u
            if verbose and (
                step_index % 200 == 0 or step_index == int(num_steps) - 1
            ):
                chi2 = float(jnp.sum(_residual_of_u(best_u) ** 2))
                print(
                    f"  map start {start_number + 1}/{len(start_points)} "
                    f"step {step_index:4d} | beta {float(beta):.3g} "
                    f"| -logpost(beta=1) {obj:.4f} | best {best_obj:.4f} "
                    f"| chi2 {chi2:.2f}"
                )

    u_map = best_u
    cube_map = nn.sigmoid(u_map)
    chi2_map = float(jnp.sum(_residual_of_u(u_map) ** 2))
    n_residuals = max(int(np.size(np.asarray(_residual_of_u(u_map)))), 1)
    chi2_map_per_n = chi2_map / n_residuals
    map_converged = (
        True
        if convergence_chi2_per_n is None
        else bool(chi2_map_per_n <= float(convergence_chi2_per_n))
    )
    if verbose and not map_converged:
        print(
            f"  WARNING: MAP did not converge -- chi2/N {chi2_map_per_n:.2f} "
            f"exceeds {float(convergence_chi2_per_n):.2f}. A Laplace covariance "
            "around a non-converged optimum is not meaningful; count this case "
            "as a failure rather than averaging it in."
        )

    # A posterior marginal must never be broader than the prior. In the
    # logit space the logistic prior has variance pi^2/3, so we floor the
    # precision eigenvalues at the corresponding value. This caps every
    # direction's width at the prior and prevents weakly-identified /
    # near-edge directions from blowing up (the old 1e-6 floor let them
    # reach ~1e6 covariance -> >20x the noise in spectrum space, which
    # Jensen-biased the posterior-mean prediction by ~6 sigma).
    prior_variance_u = float(np.pi ** 2 / 3.0)
    precision_floor = max(float(eigenvalue_floor), 1.0 / prior_variance_u)
    jacobian = jax.jacrev(_residual_of_u)(u_map)  # (n_wavelengths, param_dim)

    chi2_inflation_factor = 1.0
    covariance_width_ratio = 1.0
    covariance_width_ratio_median = 1.0
    sample_key = random.PRNGKey(int(seed) + 1)
    standard_normal = random.normal(
        sample_key, (int(n_posterior_samples), param_dim), dtype=jnp.float32
    )

    if covariance_method == "gaussian_vi":
        # Short mean-field Gaussian VI at beta=1 around the MAP:
        # q(u) = N(m, diag(std^2)). Maximize E_q[log p(x|u) + log prior(u)]
        # + H[q]. The likelihood tightens constrained directions, the
        # entropy keeps q from collapsing, and the prior term bounds it near
        # the prior width on unconstrained directions -- a properly
        # conditioned, calibrated covariance, unlike the raw Gauss-Newton
        # inverse. std is clamped to <= prior std so no marginal exceeds the
        # prior.
        diag_precision = jnp.sum(jacobian ** 2, axis=0) + 2.0 * cube_map * (
            1.0 - cube_map
        )
        diag_precision = jnp.maximum(diag_precision, precision_floor)
        init_log_std = 0.5 * jnp.log(1.0 / diag_precision)
        log_std_min = float(np.log(1e-3))
        log_std_max = float(0.5 * np.log(prior_variance_u))  # prior std
        vi_beta_value = jnp.asarray(float(vi_beta), dtype=jnp.float32)
        # By default freeze the mean at the MAP: optimizing it at beta=1
        # under the tiny observational noise destabilizes it (it drifts off
        # the MAP, ~doubling chi2). Only the widths are calibrated by the VI;
        # use vi_beta < 1 to keep the width optimization out of the
        # sigma-dominated regime.
        freeze_mean = bool(vi_freeze_mean)
        if freeze_mean:
            vi_params = {"log_std": init_log_std}
        else:
            vi_params = {"mean": u_map, "log_std": init_log_std}

        def _vi_neg_elbo(params: Any, key: Array) -> jnp.ndarray:
            log_std = jnp.clip(params["log_std"], log_std_min, log_std_max)
            std = jnp.exp(log_std)
            eps = random.normal(
                key, (int(vi_num_samples), param_dim), dtype=jnp.float32
            )
            mean = u_map if freeze_mean else params["mean"]
            u = mean[None, :] + eps * std[None, :]
            residual = vmap(_residual_of_u)(u)
            log_likelihood = -0.5 * jnp.sum(residual * residual, axis=1)
            log_prior = -jnp.sum(
                nn.softplus(u) + nn.softplus(-u), axis=1
            )
            entropy = jnp.sum(log_std)  # H[N(m,diag std^2)] up to a constant
            return -(jnp.mean(vi_beta_value * log_likelihood + log_prior) + entropy)

        vi_optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(float(vi_learning_rate)),
        )
        vi_state = vi_optimizer.init(vi_params)

        @jax.jit
        def _vi_step(params: Any, state: Any, key: Array) -> tuple:
            loss, grads = jax.value_and_grad(_vi_neg_elbo)(params, key)
            grads = jax.tree_util.tree_map(
                lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
            )
            updates, state = vi_optimizer.update(grads, state)
            return optax.apply_updates(params, updates), state, loss

        vi_key = random.PRNGKey(int(seed) + 2)
        for vi_index in range(int(vi_num_steps)):
            vi_key, step_key = random.split(vi_key)
            vi_params, vi_state, vi_loss = _vi_step(
                vi_params, vi_state, step_key
            )
            if verbose and (
                vi_index % 100 == 0 or vi_index == int(vi_num_steps) - 1
            ):
                print(f"  vi step {vi_index:4d} | -ELBO {float(vi_loss):.4f}")

        u_center = u_map if freeze_mean else vi_params["mean"]
        posterior_std = jnp.exp(
            jnp.clip(vi_params["log_std"], log_std_min, log_std_max)
        )
        u_samples = u_center[None, :] + standard_normal * posterior_std[None, :]
    else:
        # Gauss-Newton drops the residual-curvature term sum_i r_i d2r_i/du2,
        # negligible only for a near-perfect fit. At chi2/N ~ 1.3 it is
        # comparable to the term kept, and dropping it narrows the posterior --
        # the observed failure (coverage 0.52 vs nominal 0.90). "hessian" keeps
        # it: the actual Hessian of the negative log posterior.
        prior_precision = 2.0 * cube_map * (1.0 - cube_map)
        gauss_newton_precision = jacobian.T @ jacobian + jnp.diag(prior_precision)
        if covariance_method == "hessian":
            precision = jax.hessian(_neg_log_posterior)(
                u_map, jnp.asarray(1.0, jnp.float32)
            )
        elif covariance_method == "laplace":
            precision = gauss_newton_precision
        else:
            raise ValueError(
                "covariance_method must be 'hessian', 'laplace' or 'gaussian_vi'; "
                f"got {covariance_method!r}"
            )

        # A least-squares fit whose chi2 exceeds its degrees of freedom is
        # telling you the reported error bars are optimistic; the textbook
        # correction scales the covariance by chi2/dof. Reported always, applied
        # only on request -- applying it silently would hide whatever residual
        # miscalibration the curvature term does not explain.
        degrees_of_freedom = max(n_residuals - param_dim, 1)
        chi2_inflation_factor = max(chi2_map / degrees_of_freedom, 1.0)
        if covariance_chi2_inflation:
            precision = precision / chi2_inflation_factor

        precision = 0.5 * (precision + precision.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(precision)
        eigenvalues = jnp.maximum(eigenvalues, precision_floor)
        cov_sqrt = eigenvectors * jnp.sqrt(1.0 / eigenvalues)
        u_center = u_map
        u_samples = u_map[None, :] + standard_normal @ cov_sqrt.T

        # How much of the width came from keeping the residual-curvature term?
        # Reported as the largest per-marginal ratio, not the median: the term
        # typically bites on the one or two coordinates the model is most
        # nonlinear in, and a median over 19 parameters reports 1.0 while a
        # marginal has moved by 50%.
        width_ratios = np.asarray(
            jnp.sqrt(
                jnp.diag(eigenvectors @ jnp.diag(1.0 / eigenvalues) @ eigenvectors.T)
            )
            / jnp.sqrt(
                jnp.clip(
                    jnp.diag(jnp.linalg.inv(gauss_newton_precision)), 1e-12, None
                )
            )
        )
        covariance_width_ratio = float(np.max(width_ratios))
        covariance_width_ratio_median = float(np.median(width_ratios))

    cube_samples = _clip_cube_support(nn.sigmoid(u_samples))

    if verbose:
        chi2_center = float(jnp.sum(_residual_of_u(u_center) ** 2))
        print(
            f"  {covariance_method} done | chi2(MAP) {chi2_map:.2f} "
            f"| chi2(posterior centre) {chi2_center:.2f} "
            f"| cube std (first dims) "
            f"{np.round(np.asarray(jnp.std(cube_samples, axis=0))[:4], 4)}"
        )

    refined = copy.copy(posterior)
    refined._fixed_cube_samples = np.asarray(cube_samples, dtype=np.float64)
    # MAP point estimate (cube): the un-Jensen-biased point prediction; the
    # forward model evaluated here gives chi2(MAP) ~= the achievable floor.
    refined._fixed_point_estimate_cube = np.asarray(cube_map, dtype=np.float64)
    refined._map_diagnostics = {
        "chi2_map": chi2_map,
        "chi2_map_per_n": chi2_map_per_n,
        "converged": map_converged,
        "start_selection": start_selection,
        "num_starts": int(num_starts),
        "best_start_chi2_per_n": (
            None if start_chi2 is None else float(np.min(start_chi2) / n_residuals)
        ),
        "covariance_method": covariance_method,
        # Median ratio of the delivered marginal width to the Gauss-Newton one.
        # >1 means the residual-curvature term widened the posterior; ~1 means
        # the linearization was fine and a coverage shortfall lies elsewhere.
        "covariance_width_ratio_vs_gauss_newton": covariance_width_ratio,
        "covariance_width_ratio_vs_gauss_newton_median": covariance_width_ratio_median,
        # chi2/dof at the MAP: the factor the covariance would be scaled by if
        # the misfit were attributed to optimistic error bars.
        "chi2_inflation_factor": chi2_inflation_factor,
        "chi2_inflation_applied": bool(covariance_chi2_inflation),
    }
    return refined
