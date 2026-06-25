"""Conditional encoder+flow model container and the training objective.

Holds :class:`_PosteriorModel` (the encoder + flow + auxiliary scale head bundled
as one Equinox module) and the loss functions optimised by
:class:`~petitRADTRANS.sbi.flows.posterior.ConditionalFlowPosterior`: the
conditional negative-log-likelihood (NPE) objective with its optional spline and
invertibility regularization, and the amortized-variational (ELBO) objective.
Extracted from ``flows.posterior`` so the objective maths lives apart from the
estimator's plumbing.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, cast

import equinox as eqx
import jax
from jax import Array, nn, random, vmap
import jax.numpy as jnp
import numpy as np

from ..encoders import ObservationEncoder
from ..observation import ObservationBlock
from .base import PosteriorBatch
from .cube import _flow_space_parameters, _parameter_space_log_prob
from .regularization import (
    _DIAGNOSTIC_EDGE_THRESHOLD,
    _slice_batch_for_diagnostics,
    _spline_parameter_statistics,
    _spline_regularization_terms,
)


class _PosteriorModel(eqx.Module):
    """Trainable encoder + conditional flow bundled as one Equinox module.

    Holds the observation ``encoder``, the conditional ``flow``, and an auxiliary
    ``scale_head`` (a small linear readout used by the optional amplitude-scale
    loss), together with the static regularization weights/targets read by the
    loss functions in this module. Bundling them keeps the whole trainable model a
    single pytree so Optax updates and Equinox (de)serialization act on it as a
    unit.
    """

    encoder: ObservationEncoder
    flow: Any
    scale_head: eqx.nn.Linear
    aux_scale_loss_weight: float = eqx.field(static=True)
    spline_small_bin_regularization_weight: float = eqx.field(static=True)
    spline_derivative_regularization_weight: float = eqx.field(static=True)
    spline_min_bin_ratio_target: float = eqx.field(static=True)
    spline_min_derivative_target: float = eqx.field(static=True)
    spline_max_derivative_target: float = eqx.field(static=True)
    invertibility_loss_weight: float = eqx.field(static=True)
    invertibility_loss_n_samples: int = eqx.field(static=True)


def _mean_spectral_log_scale(blocks: list[ObservationBlock]) -> float | None:
    """Mean log-amplitude across the spectrum blocks of one observation.

    Reads each spectrum block's ``log_spectrum_scale`` (or ``log_median_flux``)
    metadata and averages the finite values, giving a single scalar that
    summarises the observation's overall flux scale. Used as the regression target
    of the optional auxiliary scale loss. Returns ``None`` when no spectrum block
    carries a finite scale.
    """
    scales = []
    for block in blocks:
        modality = getattr(block.modality, "value", block.modality)
        if modality != "spectrum":
            continue
        metadata = block.metadata if isinstance(block.metadata, Mapping) else {}
        scale = metadata.get("log_spectrum_scale", metadata.get("log_median_flux"))
        if scale is None:
            continue
        scale_value = float(scale)
        if np.isfinite(scale_value):
            scales.append(scale_value)

    if not scales:
        return None
    return float(np.mean(scales))


def _scale_targets_from_observations(observations: Any) -> jnp.ndarray | None:
    """Per-sample log-amplitude targets for the auxiliary scale loss.

    Extracts one scalar log-scale per observation in the batch -- from the
    pre-stacked ``log_scale`` channel (field index 4) of each spectrum block, or
    via :func:`_mean_spectral_log_scale` for block-list observations. The result
    is the regression target the ``scale_head`` is trained against when
    ``aux_scale_loss_weight > 0``, which keeps an explicit amplitude signal in the
    embedding. Returns ``None`` for observation representations that carry no
    spectral scale.
    """
    from ..observation import PreStackedObservations

    if isinstance(observations, PreStackedObservations):
        targets = []
        for block_idx, block_tuple in enumerate(observations.stacked_blocks):
            modality = (
                observations.modalities[block_idx]
                if block_idx < len(observations.modalities)
                else None
            )
            if modality != "spectrum" or len(block_tuple) < 5:
                continue
            targets.append(jnp.asarray(block_tuple[4], dtype=jnp.float32))

        if not targets:
            return None
        return jnp.mean(jnp.stack(targets, axis=0), axis=0)

    if (
        isinstance(observations, list)
        and observations
        and isinstance(observations[0], list)
    ):
        target_values = []
        for blocks in observations:
            mean_scale = _mean_spectral_log_scale(blocks)
            target_values.append(np.nan if mean_scale is None else mean_scale)
        targets = jnp.asarray(target_values, dtype=jnp.float32)
        return targets

    return None


def batch_embeddings(model: _PosteriorModel, observations: Any) -> jnp.ndarray:
    """Encode a batch of observations into conditioning vectors.

    Dispatches over the supported observation representations and returns a
    ``(batch, embedding_dim)`` float32 array: already-encoded arrays pass through;
    :class:`~petitRADTRANS.sbi.observation.PreStackedObservations` use the
    JIT-friendly ``encode_from_prestacked`` path; a list of equally-structured
    block lists uses the vmapped ``encode_stacked_batch``; otherwise each
    observation is encoded individually. This is the single entry point the loss
    and refinement paths use to turn observations into the flow's conditioning
    context.
    """
    from ..observation import PreStackedObservations

    # Pre-stacked arrays produced by the training loop (JIT-friendly path).
    if isinstance(observations, PreStackedObservations):
        return jnp.asarray(
            model.encoder.encode_from_prestacked(observations), dtype=jnp.float32
        )

    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        return jnp.asarray(observations, dtype=jnp.float32)

    # Use vmap-based batch encoding when all observations share the same
    # block structure (always the case for SBI tasks).
    if (
        isinstance(observations, list)
        and observations
        and isinstance(observations[0], list)
    ):
        return jnp.asarray(
            model.encoder.encode_stacked_batch(observations), dtype=jnp.float32
        )

    embeddings = [model.encoder.encode(blocks).embedding for blocks in observations]
    return jnp.stack(embeddings, axis=0)


def compute_loss(model: _PosteriorModel, batch: PosteriorBatch) -> jnp.ndarray:
    """Evaluation NPE loss for one batch (no regularization terms).

    Plain conditional negative log-likelihood, used for validation / model
    selection where the optimisation-only regularization terms should not perturb
    the reported value. See :func:`loss_from_observations`.
    """
    parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
    return loss_from_observations(
        model,
        parameters,
        batch.observations,
        str(batch.metadata.get("parameter_space", "physical")),
        include_spline_regularization=False,
    )


def training_loss(model: _PosteriorModel, batch: PosteriorBatch) -> jnp.ndarray:
    """Training NPE loss for one batch (with spline / invertibility regularization).

    Identical to :func:`compute_loss` but enables the regularization terms that
    only shape the optimisation (spline bin/derivative penalties and the
    invertibility penalty). This is the function minimised by the trainer.
    """
    parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
    return loss_from_observations(
        model,
        parameters,
        batch.observations,
        str(batch.metadata.get("parameter_space", "physical")),
        include_spline_regularization=True,
    )


def loss_from_observations(
    model: _PosteriorModel,
    parameters: jnp.ndarray,
    observations: Any,
    parameter_space: str,
    *,
    include_spline_regularization: bool,
) -> jnp.ndarray:
    """Conditional negative-log-likelihood (NPE) objective for one batch.

    Encodes ``observations`` into conditioning vectors, evaluates the flow's
    log-density of ``parameters`` in the requested ``parameter_space`` (applying
    the unit-cube change of variables when needed), and returns the mean negative
    log-likelihood over the finite entries. Non-finite log-probabilities are
    masked out rather than allowed to poison the average, and the log-density is
    clipped to a wide finite range for numerical stability.

    When ``include_spline_regularization`` is set, two optimisation-only terms are
    added (and only if their weights are positive): the spline bin-ratio /
    derivative penalties from :func:`_spline_regularization_terms`, which keep a
    rational-quadratic spline well-conditioned, and the invertibility penalty
    (:func:`invertibility_penalty`), which keeps an iterative-inverse flow
    invertible as it sharpens. When ``model.aux_scale_loss_weight > 0`` an
    auxiliary mean-squared error between the ``scale_head`` prediction and the
    per-observation log-amplitude target (:func:`_scale_targets_from_observations`)
    is also added, preserving an explicit amplitude signal in the embedding.

    Parameters
    ----------
    model:
        The encoder + flow bundle (:class:`_PosteriorModel`).
    parameters:
        Parameter batch of shape ``(batch, n_parameters)``, in ``parameter_space``
        coordinates.
    observations:
        Batch of observations in any representation accepted by
        :func:`batch_embeddings`.
    parameter_space:
        ``'physical'``, ``'cube'``, or ``'unconstrained'`` -- selects whether the
        cube change-of-variables correction is applied.
    include_spline_regularization:
        Whether to add the optimisation-only regularization terms (used for
        training, disabled for evaluation).

    Returns
    -------
    jnp.ndarray
        Scalar loss (mean negative log-likelihood plus any enabled penalties).
    """
    embeddings = batch_embeddings(model, observations)
    log_probabilities = _parameter_space_log_prob(
        model.flow,
        parameters,
        embeddings,
        parameter_space,
    )
    log_probabilities = jnp.clip(log_probabilities, -50000.0, 15000.0)
    finite_mask = jnp.isfinite(log_probabilities)
    safe_log_probs = jnp.where(finite_mask, log_probabilities, 0.0)
    n_valid = jnp.maximum(finite_mask.sum(), 1)
    total_loss = -jnp.sum(safe_log_probs) / n_valid

    if include_spline_regularization and (
        model.spline_small_bin_regularization_weight > 0.0
        or model.spline_derivative_regularization_weight > 0.0
    ):
        flow_inputs, _ = _flow_space_parameters(parameters, parameter_space)
        (
            small_bin_penalty,
            _,
            _,
            derivative_penalty,
            _,
            _,
            _,
            _,
        ) = _spline_regularization_terms(
            model.flow,
            flow_inputs,
            embeddings,
            min_bin_ratio_target=model.spline_min_bin_ratio_target,
            min_derivative_target=model.spline_min_derivative_target,
            max_derivative_target=model.spline_max_derivative_target,
        )
        total_loss = (
            total_loss
            + model.spline_small_bin_regularization_weight * small_bin_penalty
            + model.spline_derivative_regularization_weight * derivative_penalty
        )

    if include_spline_regularization and model.invertibility_loss_weight > 0.0:
        total_loss = total_loss + model.invertibility_loss_weight * (
            invertibility_penalty(
                model, embeddings, model.invertibility_loss_n_samples
            )
        )

    if model.aux_scale_loss_weight <= 0.0:
        return total_loss

    scale_targets = _scale_targets_from_observations(observations)
    if scale_targets is None:
        return total_loss

    scale_predictions = vmap(model.scale_head)(embeddings).reshape(-1)
    finite_scale_mask = jnp.isfinite(scale_targets)
    safe_scale_targets = jnp.where(finite_scale_mask, scale_targets, 0.0)
    scale_residual = scale_predictions - safe_scale_targets
    n_scale = jnp.maximum(finite_scale_mask.sum(), 1)
    scale_loss = jnp.sum(jnp.square(scale_residual) * finite_scale_mask) / n_scale
    return total_loss + model.aux_scale_loss_weight * scale_loss

def invertibility_penalty(
    model: _PosteriorModel,
    embeddings: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    """Differentiable inverse round-trip + log-det-closure penalty.

    Draws base samples ``z ~ N(0, I)``, maps them to flow space with the
    (iterative) inverse, then back to latent space with the analytic
    forward. The inverse output is detached with ``stop_gradient`` so no
    gradient flows through the bisection solver; the gradient instead trains
    the *forward* map to be consistent with its own inverse. This penalises
    exactly the ``inverse_forward`` round-trip error and the inverse/forward
    log-det closure that the validation diagnostics report -- the quantities
    that diverge as a neural-autoregressive flow sharpens (val_loss keeps
    falling while the sampler that draws posterior samples becomes
    non-invertible, so the deployed posterior predicts nonsense).

    The penalty is bounded in scale: the reconstruction term is a mean
    squared error in latent space and the log-det term is normalised by the
    parameter dimension, so a single ``invertibility_loss_weight`` controls
    both. For flows with an exact analytic inverse (affine/spline) the
    penalty is ~0 and merely costs a little compute.
    """
    n_context = int(embeddings.shape[0])
    n = max(1, min(int(n_samples), n_context))
    context = embeddings[:n]
    # Fixed key: the base points are held constant, but the conditioning
    # embeddings vary per batch, so the regularised (z, context) pairs differ
    # across steps. This mirrors how ``_validation_diagnostics`` samples.
    latent = random.normal(
        random.PRNGKey(0),
        shape=(n, model.flow.parameter_dim),
        dtype=embeddings.dtype,
    )
    flow_inputs, inverse_logdet = vmap(
        lambda z, ctx: model.flow.inverse(z, ctx)
    )(latent, context)
    flow_inputs = jax.lax.stop_gradient(flow_inputs)
    inverse_logdet = jax.lax.stop_gradient(inverse_logdet)
    recovered_latent, forward_logdet = vmap(
        lambda sample, ctx: model.flow.forward(sample, ctx)
    )(flow_inputs, context)
    reconstruction = jnp.mean(jnp.square(recovered_latent - latent))
    logdet_closure = jnp.mean(
        jnp.square(
            (forward_logdet + inverse_logdet)
            / jnp.asarray(model.flow.parameter_dim, dtype=embeddings.dtype)
        )
    )
    return reconstruction + logdet_closure


def make_elbo_loss(
    log_likelihood_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
    parameter_space: str,
    *,
    num_samples: int = 1,
) -> Callable[[_PosteriorModel, PosteriorBatch, Any], jnp.ndarray]:
    """Return an amortized-variational (ELBO) training loss.

    Maximizes, for observations ``x`` drawn from the (prior-predictive)
    dataset, the (tempered) ELBO

        ELBO_beta(x) = E_{q(theta|x)}[ beta * log p(x|theta) + log p(theta)
                                       - log q(theta|x) ]

    with a single- (or few-) sample reparameterized estimator. ``q(theta|x)``
    is the conditional flow: latents ``z`` are drawn from the flow base and
    pushed through ``flow.inverse`` to reparameterized samples, so the
    gradient flows through both the flow and the encoder.

    The returned loss takes a third argument ``beta`` (a scalar in (0, 1]):
    the inverse likelihood temperature. ``beta`` scales the log-likelihood,
    which for a Gaussian is exactly equivalent to inflating the noise to
    ``sigma_eff = sigma / sqrt(beta)`` (a power posterior ``p_beta`` whose
    target is the true posterior at ``beta = 1``). The entropy ``-log q`` is
    kept at full weight. Annealing ``beta`` from small -> 1 tames the
    gradient stiffness from the very small observational noise (which would
    otherwise swamp the entropy term ~10^6 : 1 and drive an over-confident,
    poorly-located posterior), lets the entropy shape the width during
    annealing, and recovers the untempered posterior once ``beta = 1``.

    The ``-log q`` (entropy) term diverges to ``-inf`` as ``q`` collapses to
    a point mass, so a delta posterior is *structurally* impossible.
    ``log p(x|theta)`` is supplied by the injected differentiable
    forward-model likelihood evaluated at the physical parameters
    reconstructed from the sampled cube coordinates. With
    ``parameter_space='cube'`` the prior is uniform on the unit hypercube,
    so ``log p(theta_cube) = 0`` and is dropped.

    Parameters
    ----------
    log_likelihood_fn:
        ``(theta_cube, observations) -> (batch,)`` returning the Gaussian
        observational log-likelihood ``log p(x | theta)`` for each batch
        element. Receives unit-cube coordinates (shape ``(batch, dim)``)
        and the batch's prestacked observations; it owns the cube->physical
        transform, the differentiable forward model, and the noise model.
        Must be JAX-differentiable w.r.t. ``theta_cube``.
    parameter_space:
        Must be ``"cube"``.
    num_samples:
        Number of reparameterized posterior draws per observation used to
        estimate the ELBO expectation. ``1`` is standard for amortized VI;
        larger values reduce gradient variance at a proportional
        forward-model cost.
    """
    if parameter_space != "cube":
        raise ValueError(
            "ELBO objective requires parameter_space='cube' (uniform prior)."
        )
    n_draws = max(int(num_samples), 1)

    def _elbo_loss(
        model: _PosteriorModel, batch: PosteriorBatch, beta: Any = 1.0
    ) -> jnp.ndarray:
        observations = batch.observations
        embeddings = batch_embeddings(
            model, observations
        )
        embeddings = jnp.asarray(embeddings, dtype=jnp.float32)
        batch_size = embeddings.shape[0]
        param_dim = model.flow.parameter_dim
        beta = jnp.asarray(beta, dtype=jnp.float32)

        # Derive a per-step key from the batch contents so that the
        # reparameterized draws differ across steps/epochs.
        seed = jnp.bitwise_xor(
            jax.lax.bitcast_convert_type(embeddings[0, 0], jnp.int32),
            jax.lax.bitcast_convert_type(embeddings[-1, -1], jnp.int32),
        )
        key = random.fold_in(random.PRNGKey(0x5E1B0), seed)

        def _single_draw(draw_key: Array) -> jnp.ndarray:
            latent = model.flow._base_sample(
                draw_key, (batch_size, param_dim), embeddings.dtype
            )
            # Reparameterized sample in the flow's unconstrained space and
            # the log-density of that sample under q.
            theta_unc, log_det_inv = vmap(model.flow.inverse)(
                latent, embeddings
            )
            base_log_prob = vmap(model.flow._base_log_prob)(latent)
            # log q(theta_unc) = log base(z) - log|det d theta_unc / d z|.
            log_q_unc = base_log_prob - log_det_inv  # (batch,)

            # Map to the unit cube and change variables:
            # q_cube(c) = q_unc(u) / |d c / d u|, c = sigmoid(u),
            # |d c / d u| = c (1 - c).
            theta_cube = nn.sigmoid(theta_unc)
            clipped = jnp.clip(theta_cube, 1e-7, 1.0 - 1e-7)
            log_jacobian = jnp.sum(
                jnp.log(clipped) + jnp.log1p(-clipped), axis=-1
            )  # (batch,)
            log_q_cube = log_q_unc - log_jacobian  # (batch,)

            log_likelihood = log_likelihood_fn(theta_cube, observations)
            log_likelihood = jnp.reshape(log_likelihood, (batch_size,))
            # Tempered likelihood (sigma_eff = sigma / sqrt(beta)); uniform-
            # cube prior contributes log p(theta_cube) = 0.
            return beta * log_likelihood - log_q_cube  # (batch,)

        draw_keys = random.split(key, n_draws)
        elbo_total = jnp.zeros((batch_size,), dtype=jnp.float32)
        for draw_key in draw_keys:
            elbo_total = elbo_total + _single_draw(draw_key)
        elbo = elbo_total / float(n_draws)
        elbo = jnp.where(jnp.isfinite(elbo), elbo, -1e6)
        return -jnp.mean(elbo)

    return _elbo_loss


def compute_validation_diagnostics(
    model: _PosteriorModel,
    batch: PosteriorBatch,
    *,
    verbose_diagnostics: bool = False,
) -> dict[str, float]:
    """Per-checkpoint flow invertibility / collapse diagnostics.

    Probes the trained flow on a small slice of a validation batch and returns a
    dictionary of scalar metrics that quantify how trustworthy the flow's sampler
    and density are -- the inputs to the checkpoint-selection gate
    (:func:`_flow_validation_stability_metrics` /
    :func:`_flow_validation_selection_metric`). The metrics fall into a few
    groups:

    * **Round-trip errors** -- ``forward_inverse_*`` and ``inverse_forward_*``
      max/mean absolute errors and the log-det closure of the
      forward/inverse pair. A flow whose analytic forward and (possibly iterative)
      inverse disagree produces samples that do not match its own density; these
      errors grow as a neural-autoregressive flow sharpens, which is the dominant
      way a low-noise posterior silently degrades.
    * **Iterative-solver health** -- residuals, bracket widths, expansion steps,
      and converged/failure fractions for flows inverted by bisection.
    * **Support occupancy** -- ``cube_edge_hit_rate``: the fraction of sampled
      coordinates pinned at the unit-cube edges, a signature of density collapse.
    * **Spline conditioning** -- the statistics from
      :func:`_spline_parameter_statistics` (bin widths/heights, derivative
      extremes, floor-hit fractions).

    Parameters
    ----------
    model:
        The encoder + flow bundle to probe.
    batch:
        Validation batch; only the leading
        :data:`~petitRADTRANS.sbi.flows.regularization._VALIDATION_DIAGNOSTIC_SAMPLES`
        rows are used.
    verbose_diagnostics:
        When ``True``, include the extra per-dimension cube-edge breakdown used by
        the verbose training diagnostics.

    Returns
    -------
    dict[str, float]
        Flat mapping of metric name to scalar value (``nan`` for metrics that do
        not apply to the current flow family or an empty batch).
    """
    diagnostic_batch = _slice_batch_for_diagnostics(batch)
    parameters = jnp.asarray(diagnostic_batch.parameters, dtype=jnp.float32)
    if parameters.shape[0] == 0:
        return {
            "forward_inverse_max_abs_error": float("nan"),
            "forward_inverse_mean_abs_error": float("nan"),
            "forward_inverse_logdet_closure_max_abs_error": float("nan"),
            "inverse_forward_max_abs_error": float("nan"),
            "inverse_forward_mean_abs_error": float("nan"),
            "inverse_forward_logdet_closure_max_abs_error": float("nan"),
            "inverse_solver_max_abs_residual": float("nan"),
            "inverse_solver_mean_abs_residual": float("nan"),
            "inverse_solver_max_bracket_width": float("nan"),
            "inverse_solver_mean_bracket_width": float("nan"),
            "inverse_solver_max_expansion_steps": float("nan"),
            "inverse_solver_mean_expansion_steps": float("nan"),
            "inverse_solver_converged_fraction": float("nan"),
            "inverse_solver_failure_rate": float("nan"),
            "cube_edge_hit_rate": float("nan"),
            "cube_low_edge_hit_rate": float("nan"),
            "cube_high_edge_hit_rate": float("nan"),
        }

    embeddings = batch_embeddings(
        model, diagnostic_batch.observations
    )
    parameter_space = str(diagnostic_batch.metadata.get("parameter_space", "physical"))
    flow_inputs, _ = _flow_space_parameters(parameters, parameter_space)

    latent, forward_logdet = vmap(
        lambda sample, context: model.flow.forward(sample, context)
    )(flow_inputs, embeddings)
    recovered_inputs, inverse_logdet = vmap(
        lambda latent_sample, context: model.flow.inverse(latent_sample, context)
    )(latent, embeddings)
    forward_inverse_error = jnp.abs(recovered_inputs - flow_inputs)

    latent_samples = random.normal(
        random.PRNGKey(0),
        shape=(flow_inputs.shape[0], model.flow.parameter_dim),
        dtype=flow_inputs.dtype,
    )
    sampled_inverse_diagnostics = None
    inverse_with_diagnostics = getattr(model.flow, "inverse_with_diagnostics", None)
    if callable(inverse_with_diagnostics):
        inverse_with_diagnostics_fn = cast(
            Callable[[Any, Any], tuple[Any, Any, Any]],
            inverse_with_diagnostics,
        )
        (
            sampled_inputs,
            sampled_inverse_logdet,
            sampled_inverse_diagnostics,
        ) = vmap(
            lambda latent_sample, context: inverse_with_diagnostics_fn(
                latent_sample,
                context,
            )
        )(latent_samples, embeddings)
    else:
        sampled_inputs, sampled_inverse_logdet = vmap(
            lambda latent_sample, context: model.flow.inverse(latent_sample, context)
        )(latent_samples, embeddings)
    recovered_latents, sampled_forward_logdet = vmap(
        lambda sample, context: model.flow.forward(sample, context)
    )(sampled_inputs, embeddings)
    inverse_forward_error = jnp.abs(recovered_latents - latent_samples)

    diagnostics = {
        "forward_inverse_max_abs_error": float(jnp.max(forward_inverse_error)),
        "forward_inverse_mean_abs_error": float(jnp.mean(forward_inverse_error)),
        "forward_inverse_logdet_closure_max_abs_error": float(
            jnp.max(jnp.abs(forward_logdet + inverse_logdet))
        ),
        "inverse_forward_max_abs_error": float(jnp.max(inverse_forward_error)),
        "inverse_forward_mean_abs_error": float(jnp.mean(inverse_forward_error)),
        "inverse_forward_logdet_closure_max_abs_error": float(
            jnp.max(jnp.abs(sampled_inverse_logdet + sampled_forward_logdet))
        ),
        "inverse_solver_max_abs_residual": float("nan"),
        "inverse_solver_mean_abs_residual": float("nan"),
        "inverse_solver_max_bracket_width": float("nan"),
        "inverse_solver_mean_bracket_width": float("nan"),
        "inverse_solver_max_expansion_steps": float("nan"),
        "inverse_solver_mean_expansion_steps": float("nan"),
        "inverse_solver_converged_fraction": float("nan"),
        "inverse_solver_failure_rate": float("nan"),
        "cube_edge_hit_rate": float("nan"),
        "cube_low_edge_hit_rate": float("nan"),
        "cube_high_edge_hit_rate": float("nan"),
        "spline_small_bin_regularization_term": float("nan"),
        "spline_small_bin_width_regularization_term": float("nan"),
        "spline_small_bin_height_regularization_term": float("nan"),
        "spline_derivative_regularization_term": float("nan"),
        "spline_mean_min_width_ratio": float("nan"),
        "spline_mean_min_height_ratio": float("nan"),
        "spline_global_min_width_ratio": float("nan"),
        "spline_global_min_height_ratio": float("nan"),
    }

    if sampled_inverse_diagnostics is not None:
        diagnostics["inverse_solver_max_abs_residual"] = float(
            jnp.max(sampled_inverse_diagnostics.max_abs_residual)
        )
        diagnostics["inverse_solver_mean_abs_residual"] = float(
            jnp.mean(sampled_inverse_diagnostics.mean_abs_residual)
        )
        diagnostics["inverse_solver_max_bracket_width"] = float(
            jnp.max(sampled_inverse_diagnostics.max_bracket_width)
        )
        diagnostics["inverse_solver_mean_bracket_width"] = float(
            jnp.mean(sampled_inverse_diagnostics.mean_bracket_width)
        )
        diagnostics["inverse_solver_max_expansion_steps"] = float(
            jnp.max(sampled_inverse_diagnostics.max_expansion_steps)
        )
        diagnostics["inverse_solver_mean_expansion_steps"] = float(
            jnp.mean(sampled_inverse_diagnostics.mean_expansion_steps)
        )
        diagnostics["inverse_solver_converged_fraction"] = float(
            jnp.mean(sampled_inverse_diagnostics.converged_fraction)
        )
        diagnostics["inverse_solver_failure_rate"] = float(
            1.0 - jnp.mean(sampled_inverse_diagnostics.converged_fraction)
        )

    if parameter_space == "cube":
        cube_samples = np.asarray(nn.sigmoid(sampled_inputs))
        low_edge_hits = cube_samples <= _DIAGNOSTIC_EDGE_THRESHOLD
        high_edge_hits = cube_samples >= 1.0 - _DIAGNOSTIC_EDGE_THRESHOLD
        edge_hits = np.logical_or(low_edge_hits, high_edge_hits)
        diagnostics["cube_edge_hit_rate"] = float(edge_hits.mean())
        diagnostics["cube_low_edge_hit_rate"] = float(low_edge_hits.mean())
        diagnostics["cube_high_edge_hit_rate"] = float(high_edge_hits.mean())
        if verbose_diagnostics and cube_samples.ndim == 2:
            overall_by_dimension = edge_hits.mean(axis=0)
            low_by_dimension = low_edge_hits.mean(axis=0)
            high_by_dimension = high_edge_hits.mean(axis=0)
            for dimension, rate in enumerate(overall_by_dimension.tolist()):
                diagnostics[f"cube_edge_hit_rate_dim_{dimension}"] = float(rate)
            for dimension, rate in enumerate(low_by_dimension.tolist()):
                diagnostics[f"cube_low_edge_hit_rate_dim_{dimension}"] = float(rate)
            for dimension, rate in enumerate(high_by_dimension.tolist()):
                diagnostics[f"cube_high_edge_hit_rate_dim_{dimension}"] = float(rate)

    diagnostics.update(
        _spline_parameter_statistics(model.flow, flow_inputs, embeddings)
    )
    (
        small_bin_penalty,
        small_bin_width_penalty,
        small_bin_height_penalty,
        derivative_penalty,
        mean_min_width_ratio,
        mean_min_height_ratio,
        global_min_width_ratio,
        global_min_height_ratio,
    ) = _spline_regularization_terms(
        model.flow,
        flow_inputs,
        embeddings,
        min_bin_ratio_target=model.spline_min_bin_ratio_target,
        min_derivative_target=model.spline_min_derivative_target,
        max_derivative_target=model.spline_max_derivative_target,
    )
    diagnostics["spline_small_bin_regularization_term"] = float(small_bin_penalty)
    diagnostics["spline_small_bin_width_regularization_term"] = float(
        small_bin_width_penalty
    )
    diagnostics["spline_small_bin_height_regularization_term"] = float(
        small_bin_height_penalty
    )
    diagnostics["spline_derivative_regularization_term"] = float(derivative_penalty)
    diagnostics["spline_mean_min_width_ratio"] = float(mean_min_width_ratio)
    diagnostics["spline_mean_min_height_ratio"] = float(mean_min_height_ratio)
    diagnostics["spline_global_min_width_ratio"] = float(global_min_width_ratio)
    diagnostics["spline_global_min_height_ratio"] = float(global_min_height_ratio)
    return diagnostics
