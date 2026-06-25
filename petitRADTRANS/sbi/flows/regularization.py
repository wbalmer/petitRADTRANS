"""Spline regularization and checkpoint-selection stability diagnostics.

Training-time helpers for the conditional spline flow: the rational-quadratic
spline regularization terms (bin-ratio / derivative penalties) added to the NPE
loss, the per-checkpoint spline parameter statistics, and the
flow-invertibility / collapse metrics used to gate and score checkpoint
selection. Also holds the small batch-slicing helpers shared by the validation
diagnostics. Extracted from ``flows.posterior`` to keep that module focused on
the estimator itself.
"""

from __future__ import annotations

from typing import Any, Mapping, cast

import jax.numpy as jnp
import numpy as np
from jax import nn, vmap

from .base import PosteriorBatch
from .transforms import (
    ConditionalRationalQuadraticSplineCoupling,
    ConditionalScalarSplineTransform,
    ConditionalSplineFlow,
    _normalize_bin_sizes,
    _spline_boundary_derivatives,
)


_VALIDATION_DIAGNOSTIC_SAMPLES = 64
_DIAGNOSTIC_EDGE_THRESHOLD = 1e-6
_REGULARIZATION_TAIL_FRACTION = 0.25
_DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE = 1.05

# Loose invertibility/collapse thresholds for the ELBO checkpoint-stability gate.
# Much looser than the NPE defaults (~0.03) because a reverse-KL flow does not
# optimise the round trip, but tight enough to reject the collapsed flows
# (round-trip error O(10), log-det closure O(100)) that produce overconfident,
# miscalibrated posteriors.
_ELBO_STABILITY_INVERSE_FORWARD_MAX_ABS_ERROR = 5.0
_ELBO_STABILITY_LOGDET_CLOSURE_MAX_ABS_ERROR = 10.0
_ELBO_STABILITY_CUBE_EDGE_HIT_RATE = 0.5

# Hard-pathology gate for the iterative-inverse round-trip error of the NPE
# (forward-KL) flows. A neural autoregressive flow inverts by bisection, so a
# small, non-zero worst-case round-trip error is expected -- and it *grows as
# the posterior sharpens*. It is a sampler-accuracy property, not a density
# collapse. The original hardcoded 0.1 cutoff flagged well-trained, usable
# epochs as "pathological", which (a) poisoned their selection metric to the 1e6
# sentinel so an undertrained early epoch was deployed, and (b) barred them from
# resetting early-stop patience so the run halted ~halfway through the budget
# with val_loss still descending. A first relaxation to 1.0 (one unit-cube
# width) was still too strict for the emission NAF, whose inv_rt_err reaches
# ~1.2-1.7 at exactly its best-loss epochs. Set the *hard* sentinel only for
# genuinely collapsed / non-invertible maps (O(several), well above a unit-cube
# width) and let the bounded soft penalty below grade everything in between, so
# a much-better-loss epoch with a moderate inverse error can still be selected.
_PATHOLOGICAL_INVERSE_FORWARD_MAX_ABS_ERROR = 2.0

# Bounded soft penalty applied to the selection metric of an unstable-but-not-
# pathological checkpoint (val_loss + penalty). Lowering the per-term weight
# (2.0 -> 1.0) and the cap (25 -> 12) keeps the penalty discriminating between
# better- and worse-conditioned inverses while no longer dominating val_loss:
# with the previous weight/cap the large log-det-closure error (~2-5) saturated
# the penalty for every well-trained epoch, pinning deployment to an
# undertrained early epoch. These are paired with the emission run's conditioning
# tuning (lower LR, tighter grad clip, larger NAF min-slope/min-residual), which
# is what actually keeps the inverse error small.
_SELECTION_PENALTY_WEIGHT = 1.0
_SELECTION_PENALTY_CAP = 12.0


def _slice_observations_for_diagnostics(observations: Any, n_samples: int) -> Any:
    """Return the first ``n_samples`` observations, preserving the container type."""
    from ..observation import PreStackedObservations

    if isinstance(observations, PreStackedObservations):
        return PreStackedObservations(
            stacked_blocks=tuple(
                tuple(arr[:n_samples] for arr in block_tuple)
                for block_tuple in observations.stacked_blocks
            ),
            modalities=observations.modalities,
        )

    if isinstance(observations, list):
        return observations[:n_samples]

    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        return observations[:n_samples]

    return observations


def _slice_batch_for_diagnostics(
    batch: PosteriorBatch,
    max_samples: int = _VALIDATION_DIAGNOSTIC_SAMPLES,
) -> PosteriorBatch:
    """Sub-sample a batch to at most ``max_samples`` rows for cheap diagnostics."""
    parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
    n_samples = min(int(parameters.shape[0]), int(max_samples))
    return PosteriorBatch(
        parameters=parameters[:n_samples],
        observations=_slice_observations_for_diagnostics(batch.observations, n_samples),
        metadata=batch.metadata,
    )


def _normalized_categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Entropy of ``softmax(logits)`` normalised to ``[0, 1]`` by ``log(n_categories)``.

    A spline-bin uniformity measure: ``1`` when the predicted bin widths/heights
    are uniform, decaying toward ``0`` as probability mass collapses onto a few
    bins.
    """
    probabilities = nn.softmax(logits, axis=-1)
    safe_probabilities = jnp.clip(probabilities, 1e-8, 1.0)
    entropy = -jnp.sum(safe_probabilities * jnp.log(safe_probabilities), axis=-1)
    n_categories = logits.shape[-1]
    if n_categories <= 1:
        return jnp.ones_like(entropy)
    normalizer = jnp.log(jnp.asarray(float(n_categories), dtype=entropy.dtype))
    return entropy / jnp.maximum(normalizer, jnp.asarray(1.0, dtype=entropy.dtype))


def _weighted_average(
    values: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Weighted mean of ``values`` (plain mean when ``weights`` is ``None``).

    Weights are broadcast to the value shape and the result is normalised by their
    sum (floored at 1 to avoid division by zero).
    """
    values = jnp.asarray(values)
    if weights is None:
        return jnp.mean(values)

    expanded_weights = jnp.asarray(weights, dtype=values.dtype)
    while expanded_weights.ndim < values.ndim:
        expanded_weights = jnp.expand_dims(expanded_weights, axis=-1)
    expanded_weights = jnp.broadcast_to(expanded_weights, values.shape)
    normalizer = jnp.maximum(
        jnp.sum(expanded_weights),
        jnp.asarray(1.0, dtype=values.dtype),
    )
    return jnp.sum(values * expanded_weights) / normalizer


def _upper_tail_average(
    values: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    *,
    tail_fraction: float = _REGULARIZATION_TAIL_FRACTION,
) -> jnp.ndarray:
    """Mean of the largest ``tail_fraction`` of ``values`` (optionally weighted).

    Concentrates a penalty on the worst-offending elements -- e.g. the few spline
    bins closest to collapse -- instead of letting a global average dilute them.
    With ``weights`` only positive-weight elements are eligible; at least one
    element is always included in the tail.
    """
    values_array = jnp.asarray(values)
    flat_values = jnp.reshape(values_array, (-1,))

    if weights is None:
        masked_values = flat_values
        active_count = jnp.asarray(flat_values.shape[0], dtype=jnp.int32)
    else:
        expanded_weights = jnp.asarray(weights, dtype=values_array.dtype)
        while expanded_weights.ndim < values_array.ndim:
            expanded_weights = jnp.expand_dims(expanded_weights, axis=-1)
        expanded_weights = jnp.broadcast_to(expanded_weights, values_array.shape)
        active_mask = jnp.reshape(expanded_weights > 0.0, (-1,))
        masked_values = jnp.where(
            active_mask,
            flat_values,
            jnp.asarray(0.0, dtype=values_array.dtype),
        )
        active_count = jnp.sum(active_mask.astype(jnp.int32))

    safe_tail_fraction = jnp.clip(
        jnp.asarray(tail_fraction, dtype=values_array.dtype),
        0.0,
        1.0,
    )
    requested_count = jnp.ceil(
        safe_tail_fraction * active_count.astype(values_array.dtype)
    ).astype(jnp.int32)
    tail_count = jnp.maximum(requested_count, jnp.asarray(1, dtype=jnp.int32))

    sorted_values = jnp.sort(masked_values)
    indices = jnp.arange(sorted_values.shape[0], dtype=jnp.int32)
    tail_mask = indices >= (sorted_values.shape[0] - tail_count)
    tail_sum = jnp.sum(
        jnp.where(
            tail_mask,
            sorted_values,
            jnp.asarray(0.0, dtype=values_array.dtype),
        )
    )
    normalizer = jnp.maximum(
        tail_count.astype(values_array.dtype),
        jnp.asarray(1.0, dtype=values_array.dtype),
    )
    return tail_sum / normalizer


def _log_barrier_below_target(values: jnp.ndarray, target: float) -> jnp.ndarray:
    """Squared log-barrier penalising ``values`` that fall below ``target``.

    Returns ``relu(log(target) - log(value))**2``: zero once a value reaches the
    target, growing quadratically in log-space as it drops below it.
    """
    safe_values = jnp.clip(jnp.asarray(values), 1e-8, None)
    safe_target = jnp.clip(
        jnp.asarray(target, dtype=safe_values.dtype),
        1e-8,
        None,
    )
    return jnp.square(nn.relu(jnp.log(safe_target) - jnp.log(safe_values)))


def _minimum_bin_ratio_statistics(
    bin_sizes: jnp.ndarray,
    uniform_bin_size: Any,
    active_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-sample and global minimum bin-size ratios relative to a uniform bin.

    Each ratio is a spline bin size divided by the uniform bin size; small values
    flag bins collapsing toward their floor. Returns ``(minimum_ratios,
    weighted_mean_minimum_ratio, global_minimum_ratio)``; ``active_weights``
    restricts the mean and global minimum to the unmasked dimensions.
    """
    bin_sizes = jnp.asarray(bin_sizes)
    ratios = bin_sizes / jnp.asarray(uniform_bin_size, dtype=bin_sizes.dtype)
    minimum_ratios = jnp.min(ratios, axis=-1)
    mean_minimum_ratio = _weighted_average(minimum_ratios, active_weights)

    if active_weights is None:
        global_minimum_ratio = jnp.min(minimum_ratios)
    else:
        active_mask = jnp.asarray(active_weights > 0.0)
        global_minimum_ratio = jnp.min(
            jnp.where(active_mask, minimum_ratios, jnp.inf)
        )

    return minimum_ratios, mean_minimum_ratio, global_minimum_ratio


def _spline_layer_regularization_terms(
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    *,
    min_derivative: float,
    active_weights: jnp.ndarray | None,
    min_bin_ratio_target: float,
    min_derivative_target: float,
    max_derivative_target: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Regularization penalties and conditioning statistics for one spline layer.

    From a layer's normalised bin ``widths`` / ``heights`` and raw
    ``unnormalized_derivatives``, builds two penalties: a **small-bin penalty**
    (a log-barrier on the minimum width/height ratios that fall below
    ``min_bin_ratio_target``, averaged over the worst tail) and a **derivative
    penalty** pushing the softplus-mapped slopes into
    ``[min_derivative_target, max_derivative_target]`` in log-space. Together they
    stop the spline from developing vanishing bins or extreme slopes as the
    posterior sharpens. ``active_weights`` masks the transformed (non-identity)
    dimensions of a coupling layer.

    Returns
    -------
    tuple
        ``(small_bin_penalty, width_penalty, height_penalty, derivative_penalty,
        mean_min_width_ratio, mean_min_height_ratio, global_min_width_ratio,
        global_min_height_ratio)`` -- the first four are penalties, the last four
        are diagnostic ratios.
    """
    width_minimum_ratios, mean_minimum_width_ratio, global_minimum_width_ratio = (
        _minimum_bin_ratio_statistics(
            widths,
            jnp.mean(widths, axis=-1, keepdims=True),
            active_weights,
        )
    )
    height_minimum_ratios, mean_minimum_height_ratio, global_minimum_height_ratio = (
        _minimum_bin_ratio_statistics(
            heights,
            jnp.mean(heights, axis=-1, keepdims=True),
            active_weights,
        )
    )

    width_penalty = _upper_tail_average(
        _log_barrier_below_target(width_minimum_ratios, min_bin_ratio_target),
        active_weights,
    )
    height_penalty = _upper_tail_average(
        _log_barrier_below_target(height_minimum_ratios, min_bin_ratio_target),
        active_weights,
    )
    small_bin_penalty = jnp.maximum(width_penalty + height_penalty, 0.0)

    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)
    safe_derivatives = jnp.clip(derivatives, 1e-8, None)
    log_derivatives = jnp.log(safe_derivatives)
    min_log_derivative = jnp.log(
        jnp.asarray(min_derivative_target, dtype=log_derivatives.dtype)
    )
    max_log_derivative = jnp.log(
        jnp.asarray(max_derivative_target, dtype=log_derivatives.dtype)
    )
    derivative_penalty = _weighted_average(
        jnp.square(nn.relu(min_log_derivative - log_derivatives))
        + jnp.square(nn.relu(log_derivatives - max_log_derivative)),
        active_weights,
    )
    derivative_penalty = jnp.maximum(derivative_penalty, 0.0)

    return (
        small_bin_penalty,
        jnp.maximum(width_penalty, 0.0),
        jnp.maximum(height_penalty, 0.0),
        derivative_penalty,
        mean_minimum_width_ratio,
        mean_minimum_height_ratio,
        global_minimum_width_ratio,
        global_minimum_height_ratio,
    )


def _spline_regularization_terms(
    flow: Any,
    inputs: jnp.ndarray,
    embeddings: jnp.ndarray,
    *,
    min_bin_ratio_target: float,
    min_derivative_target: float,
    max_derivative_target: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Batch spline regularization terms added to the NPE training loss.

    Walks the flow's layers and, for every rational-quadratic spline layer
    (coupling or scalar), accumulates the per-layer penalties from
    :func:`_spline_layer_regularization_terms`, evaluated at each layer's own
    inputs as the batch is pushed forward through the flow. The penalties keep the
    spline well-conditioned -- no vanishing bins, slopes in a sane range -- so a
    sharpening posterior does not develop pathological transforms. Returns zeros
    (and ``nan`` diagnostics) for non-spline flows.

    Parameters
    ----------
    flow:
        Conditional flow; penalties apply only to its spline layers.
    inputs:
        Flow-space parameter batch of shape ``(batch, n_parameters)``.
    embeddings:
        Conditioning vectors of shape ``(batch, embedding_dim)``.
    min_bin_ratio_target, min_derivative_target, max_derivative_target:
        Targets defining where each penalty begins to bite.

    Returns
    -------
    tuple
        ``(small_bin_penalty, width_penalty, height_penalty, derivative_penalty,
        mean_min_width_ratio, mean_min_height_ratio, global_min_width_ratio,
        global_min_height_ratio)`` aggregated over the batch.
    """
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
    if not isinstance(flow, ConditionalSplineFlow):
        return zero, zero, zero, zero, nan, nan, nan, nan

    def _sample_regularization(
        sample: jnp.ndarray,
        context: jnp.ndarray,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        current = sample
        mean_min_width_ratio = jnp.asarray(0.0, dtype=sample.dtype)
        mean_min_height_ratio = jnp.asarray(0.0, dtype=sample.dtype)
        global_min_width_ratio = jnp.asarray(jnp.inf, dtype=sample.dtype)
        global_min_height_ratio = jnp.asarray(jnp.inf, dtype=sample.dtype)
        layer_small_bin_penalties: list[jnp.ndarray] = []
        layer_width_penalties: list[jnp.ndarray] = []
        layer_height_penalties: list[jnp.ndarray] = []
        layer_derivative_penalties: list[jnp.ndarray] = []
        spline_layers = 0

        for layer_index, layer in enumerate(flow.layers):
            if isinstance(layer, ConditionalRationalQuadraticSplineCoupling):
                spline_layer = cast(ConditionalRationalQuadraticSplineCoupling, layer)
                mask = spline_layer.mask.astype(current.dtype)
                active_weights = 1.0 - mask
                masked_current = current * mask
                widths_raw, heights_raw, derivatives_raw = spline_layer._parameters(
                    masked_current,
                    context,
                )
                widths = vmap(
                    lambda values: _normalize_bin_sizes(
                        values,
                        spline_layer.min_bin_width,
                        2.0 * spline_layer.bound,
                    )
                )(widths_raw)
                heights = vmap(
                    lambda values: _normalize_bin_sizes(
                        values,
                        spline_layer.min_bin_height,
                        2.0 * spline_layer.bound,
                    )
                )(heights_raw)
                (
                    layer_small_bin_penalty,
                    layer_width_penalty,
                    layer_height_penalty,
                    layer_derivative_penalty,
                    layer_mean_min_width_ratio,
                    layer_mean_min_height_ratio,
                    layer_global_min_width_ratio,
                    layer_global_min_height_ratio,
                ) = (
                    _spline_layer_regularization_terms(
                        widths,
                        heights,
                        derivatives_raw,
                        min_derivative=spline_layer.min_derivative,
                        active_weights=active_weights,
                        min_bin_ratio_target=min_bin_ratio_target,
                        min_derivative_target=min_derivative_target,
                        max_derivative_target=max_derivative_target,
                    )
                )
                layer_small_bin_penalties.append(layer_small_bin_penalty)
                layer_width_penalties.append(layer_width_penalty)
                layer_height_penalties.append(layer_height_penalty)
                layer_derivative_penalties.append(layer_derivative_penalty)
                mean_min_width_ratio = mean_min_width_ratio + layer_mean_min_width_ratio
                mean_min_height_ratio = mean_min_height_ratio + layer_mean_min_height_ratio
                global_min_width_ratio = jnp.minimum(
                    global_min_width_ratio,
                    layer_global_min_width_ratio,
                )
                global_min_height_ratio = jnp.minimum(
                    global_min_height_ratio,
                    layer_global_min_height_ratio,
                )
                current, _ = spline_layer.forward(current, context)
                spline_layers += 1
            elif isinstance(layer, ConditionalScalarSplineTransform):
                widths_raw, heights_raw, derivatives_raw = layer._parameters(context)
                widths = _normalize_bin_sizes(
                    widths_raw,
                    layer.min_bin_width,
                    2.0 * layer.bound,
                )
                heights = _normalize_bin_sizes(
                    heights_raw,
                    layer.min_bin_height,
                    2.0 * layer.bound,
                )
                (
                    layer_small_bin_penalty,
                    layer_width_penalty,
                    layer_height_penalty,
                    layer_derivative_penalty,
                    layer_mean_min_width_ratio,
                    layer_mean_min_height_ratio,
                    layer_global_min_width_ratio,
                    layer_global_min_height_ratio,
                ) = (
                    _spline_layer_regularization_terms(
                        widths,
                        heights,
                        derivatives_raw,
                        min_derivative=layer.min_derivative,
                        active_weights=None,
                        min_bin_ratio_target=min_bin_ratio_target,
                        min_derivative_target=min_derivative_target,
                        max_derivative_target=max_derivative_target,
                    )
                )
                layer_small_bin_penalties.append(layer_small_bin_penalty)
                layer_width_penalties.append(layer_width_penalty)
                layer_height_penalties.append(layer_height_penalty)
                layer_derivative_penalties.append(layer_derivative_penalty)
                mean_min_width_ratio = mean_min_width_ratio + layer_mean_min_width_ratio
                mean_min_height_ratio = mean_min_height_ratio + layer_mean_min_height_ratio
                global_min_width_ratio = jnp.minimum(
                    global_min_width_ratio,
                    layer_global_min_width_ratio,
                )
                global_min_height_ratio = jnp.minimum(
                    global_min_height_ratio,
                    layer_global_min_height_ratio,
                )
                current, _ = layer.forward(current, context)
                spline_layers += 1
            else:
                current, _ = cast(Any, layer).forward(current, context)

            if layer_index < len(flow.layers) - 1:
                current = flow._reverse_vector(current)

        if spline_layers > 0:
            small_bin_penalty = _upper_tail_average(jnp.stack(layer_small_bin_penalties))
            width_penalty = _upper_tail_average(jnp.stack(layer_width_penalties))
            height_penalty = _upper_tail_average(jnp.stack(layer_height_penalties))
            derivative_penalty = _upper_tail_average(
                jnp.stack(layer_derivative_penalties)
            )
        else:
            small_bin_penalty = jnp.asarray(0.0, dtype=sample.dtype)
            width_penalty = jnp.asarray(0.0, dtype=sample.dtype)
            height_penalty = jnp.asarray(0.0, dtype=sample.dtype)
            derivative_penalty = jnp.asarray(0.0, dtype=sample.dtype)

        normalizer = jnp.asarray(max(spline_layers, 1), dtype=sample.dtype)
        return (
            small_bin_penalty,
            width_penalty,
            height_penalty,
            derivative_penalty,
            mean_min_width_ratio / normalizer,
            mean_min_height_ratio / normalizer,
            global_min_width_ratio,
            global_min_height_ratio,
        )

    (
        small_bin_penalty,
        width_penalty,
        height_penalty,
        derivative_penalty,
        mean_min_width_ratio,
        mean_min_height_ratio,
        global_min_width_ratio,
        global_min_height_ratio,
    ) = vmap(_sample_regularization)(inputs, embeddings)
    return (
        jnp.maximum(jnp.mean(small_bin_penalty), 0.0),
        jnp.maximum(jnp.mean(width_penalty), 0.0),
        jnp.maximum(jnp.mean(height_penalty), 0.0),
        jnp.maximum(jnp.mean(derivative_penalty), 0.0),
        jnp.mean(mean_min_width_ratio),
        jnp.mean(mean_min_height_ratio),
        jnp.min(global_min_width_ratio),
        jnp.min(global_min_height_ratio),
    )


def _spline_parameter_statistics(
    flow: Any,
    inputs: jnp.ndarray,
    embeddings: jnp.ndarray,
) -> dict[str, float]:
    """Diagnostic statistics of a spline flow's bin and derivative parameters.

    Pushes the batch through the flow and, for each spline layer, records the
    extremes and floor-hit fractions of the normalised bin widths/heights and the
    boundary derivatives, plus the mean normalised width/height entropy and mean
    absolute log-derivative. These feed the checkpoint diagnostics -- e.g.
    ``spline_max_derivative`` participates in the pathology gate. Returns a dict of
    ``nan`` for non-spline flows.
    """
    stats = {
        "spline_min_bin_width": float("nan"),
        "spline_max_bin_width": float("nan"),
        "spline_min_bin_height": float("nan"),
        "spline_max_bin_height": float("nan"),
        "spline_min_derivative": float("nan"),
        "spline_max_derivative": float("nan"),
        "spline_bin_width_floor_fraction": float("nan"),
        "spline_bin_height_floor_fraction": float("nan"),
        "spline_derivative_floor_fraction": float("nan"),
        "spline_mean_width_entropy": float("nan"),
        "spline_mean_height_entropy": float("nan"),
        "spline_mean_abs_log_derivative": float("nan"),
    }
    if not isinstance(flow, ConditionalSplineFlow):
        return stats

    current = jnp.asarray(inputs, dtype=jnp.float32)
    min_width = float("inf")
    max_width = float("-inf")
    min_height = float("inf")
    max_height = float("-inf")
    min_derivative = float("inf")
    max_derivative = float("-inf")
    width_floor_hits = 0.0
    width_total = 0.0
    height_floor_hits = 0.0
    height_total = 0.0
    derivative_floor_hits = 0.0
    derivative_total = 0.0
    width_entropy_sum = 0.0
    height_entropy_sum = 0.0
    abs_log_derivative_sum = 0.0
    entropy_layers = 0.0
    saw_spline_layer = False

    for layer_index, layer in enumerate(flow.layers):
        if isinstance(layer, ConditionalRationalQuadraticSplineCoupling):
            spline_layer = cast(ConditionalRationalQuadraticSplineCoupling, layer)
            layer_mask = spline_layer.mask.astype(current.dtype)
            active_weights = 1.0 - layer_mask
            active_dimensions = active_weights > 0.0
            masked_current = current * layer_mask[None, :]
            widths_raw, heights_raw, derivatives_raw = vmap(
                lambda sample, context: spline_layer._parameters(sample, context)
            )(masked_current, embeddings)
            widths = vmap(
                vmap(
                    lambda values: _normalize_bin_sizes(
                        values,
                        spline_layer.min_bin_width,
                        2.0 * spline_layer.bound,
                    )
                )
            )(widths_raw)
            heights = vmap(
                vmap(
                    lambda values: _normalize_bin_sizes(
                        values,
                        spline_layer.min_bin_height,
                        2.0 * spline_layer.bound,
                    )
                )
            )(heights_raw)
            derivatives = jnp.asarray(
                _spline_boundary_derivatives(
                    derivatives_raw,
                    spline_layer.min_derivative,
                    spline_layer.max_derivative,
                )
            )
            width_entropy_sum += float(
                _weighted_average(
                    _normalized_categorical_entropy(widths_raw),
                    active_weights[None, :],
                )
            )
            height_entropy_sum += float(
                _weighted_average(
                    _normalized_categorical_entropy(heights_raw),
                    active_weights[None, :],
                )
            )
            abs_log_derivative_sum += float(
                _weighted_average(
                    jnp.mean(jnp.abs(jnp.log(jnp.clip(derivatives, 1e-8, None))), axis=-1),
                    active_weights[None, :],
                )
            )
            entropy_layers += 1.0
            width_floor_hits += float(
                jnp.sum(
                    jnp.where(active_dimensions[None, :, None], widths, jnp.inf)
                    <= spline_layer.min_bin_width * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            width_total += float(jnp.sum(active_weights) * widths.shape[0] * widths.shape[-1])
            height_floor_hits += float(
                jnp.sum(
                    jnp.where(active_dimensions[None, :, None], heights, jnp.inf)
                    <= spline_layer.min_bin_height * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            height_total += float(jnp.sum(active_weights) * heights.shape[0] * heights.shape[-1])
            derivative_floor_hits += float(
                jnp.sum(
                    jnp.where(active_dimensions[None, :, None], derivatives, jnp.inf)
                    <= spline_layer.min_derivative * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            derivative_total += float(
                jnp.sum(active_weights) * derivatives.shape[0] * derivatives.shape[-1]
            )
            saw_spline_layer = True
        elif isinstance(layer, ConditionalScalarSplineTransform):
            scalar_layer = cast(ConditionalScalarSplineTransform, layer)
            widths_raw, heights_raw, derivatives_raw = vmap(scalar_layer._parameters)(embeddings)
            widths = vmap(
                lambda values: _normalize_bin_sizes(
                    values,
                    scalar_layer.min_bin_width,
                    2.0 * scalar_layer.bound,
                )
            )(widths_raw)
            heights = vmap(
                lambda values: _normalize_bin_sizes(
                    values,
                    scalar_layer.min_bin_height,
                    2.0 * scalar_layer.bound,
                )
            )(heights_raw)
            derivatives = _spline_boundary_derivatives(
                derivatives_raw,
                scalar_layer.min_derivative,
                scalar_layer.max_derivative,
            )
            width_entropy_sum += float(_weighted_average(_normalized_categorical_entropy(widths_raw)))
            height_entropy_sum += float(_weighted_average(_normalized_categorical_entropy(heights_raw)))
            abs_log_derivative_sum += float(
                _weighted_average(
                    jnp.mean(jnp.abs(jnp.log(jnp.clip(derivatives, 1e-8, None))), axis=-1)
                )
            )
            entropy_layers += 1.0
            width_floor_hits += float(
                jnp.sum(
                    widths
                    <= scalar_layer.min_bin_width * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            width_total += float(widths.size)
            height_floor_hits += float(
                jnp.sum(
                    heights
                    <= scalar_layer.min_bin_height * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            height_total += float(heights.size)
            derivative_floor_hits += float(
                jnp.sum(
                    derivatives
                    <= scalar_layer.min_derivative * _DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE
                )
            )
            derivative_total += float(derivatives.size)
            saw_spline_layer = True
        else:
            widths = heights = derivatives = None

        if saw_spline_layer and widths is not None and heights is not None and derivatives is not None:
            min_width = min(min_width, float(jnp.min(widths)))
            max_width = max(max_width, float(jnp.max(widths)))
            min_height = min(min_height, float(jnp.min(heights)))
            max_height = max(max_height, float(jnp.max(heights)))
            min_derivative = min(min_derivative, float(jnp.min(derivatives)))
            max_derivative = max(max_derivative, float(jnp.max(derivatives)))

        layer_module = cast(Any, layer)
        current, _ = vmap(lambda sample, context: layer_module.forward(sample, context))(current, embeddings)
        if layer_index < len(flow.layers) - 1 and current.shape[-1] > 1:
            current = current[:, ::-1]

    if saw_spline_layer:
        stats.update(
            {
                "spline_min_bin_width": min_width,
                "spline_max_bin_width": max_width,
                "spline_min_bin_height": min_height,
                "spline_max_bin_height": max_height,
                "spline_min_derivative": min_derivative,
                "spline_max_derivative": max_derivative,
                "spline_bin_width_floor_fraction": (
                    width_floor_hits / width_total if width_total > 0.0 else float("nan")
                ),
                "spline_bin_height_floor_fraction": (
                    height_floor_hits / height_total if height_total > 0.0 else float("nan")
                ),
                "spline_derivative_floor_fraction": (
                    derivative_floor_hits / derivative_total
                    if derivative_total > 0.0
                    else float("nan")
                ),
                "spline_mean_width_entropy": (
                    width_entropy_sum / entropy_layers
                    if entropy_layers > 0.0
                    else float("nan")
                ),
                "spline_mean_height_entropy": (
                    height_entropy_sum / entropy_layers
                    if entropy_layers > 0.0
                    else float("nan")
                ),
                "spline_mean_abs_log_derivative": (
                    abs_log_derivative_sum / entropy_layers
                    if entropy_layers > 0.0
                    else float("nan")
                ),
            }
        )

    return stats


def _flow_validation_selection_metric(
    validation_loss: float,
    diagnostics: Mapping[str, Any] | None,
) -> float:
    """Checkpoint-selection metric: validation loss plus a bounded instability penalty.

    Lower is better. Returns the raw validation loss for a stable checkpoint; a
    large constant (``1e6``) for a pathological/collapsed one so it can never be
    selected; and, for an unstable-but-usable checkpoint, the loss plus a smooth,
    capped penalty that grows with how far each round-trip / log-det-closure /
    cube-edge-hit diagnostic exceeds its threshold. The cap keeps a
    much-better-loss checkpoint selectable rather than letting a
    marginally-over-threshold diagnostic veto it.
    """
    selection_metric = float(validation_loss)
    if not np.isfinite(selection_metric) or diagnostics is None:
        return selection_metric

    checkpoint_is_stable = float(diagnostics.get("checkpoint_is_stable", float("nan")))
    if np.isfinite(checkpoint_is_stable) and checkpoint_is_stable >= 0.5:
        return selection_metric

    # A collapsed (pathological) checkpoint must never win selection, regardless
    # of how negative its loss is. Return a large constant so any usable
    # checkpoint -- and the early epochs before collapse -- rank strictly better.
    if float(diagnostics.get("checkpoint_is_pathological", 0.0)) >= 0.5:
        return 1.0e6

    def _threshold_excess(metric_name: str) -> tuple[float, float]:
        metric_value = float(diagnostics.get(metric_name, float("nan")))
        margin = float(
            diagnostics.get(
                f"checkpoint_stability_margin_{metric_name}",
                float("nan"),
            )
        )
        if np.isfinite(metric_value) and np.isfinite(margin):
            threshold = max(metric_value + margin, 1e-8)
            return max(-margin, 0.0), threshold
        if np.isfinite(metric_value) and metric_value > 0.0:
            return metric_value, max(metric_value, 1e-8)
        return 0.0, 1e-8

    # Soft, bounded penalty. Each violation contributes a smooth, saturating
    # term (in nats) rather than a steep linear/constant cliff. The previous
    # +50/+250/+1e3 scheme catapulted an unstable epoch's selection metric to
    # several hundred, so a checkpoint with dramatically better validation loss
    # could never be chosen once any earlier epoch was flagged stable. Here the
    # total penalty is capped, so selection still prefers a stable checkpoint
    # over an otherwise-equal unstable one, but a much-better-loss checkpoint is
    # not discarded purely because an iterative-inverse diagnostic is marginally
    # over threshold.
    penalty = 0.0

    for key in (
        "forward_inverse_max_abs_error",
        "inverse_forward_max_abs_error",
        "forward_inverse_logdet_closure_max_abs_error",
        "inverse_forward_logdet_closure_max_abs_error",
    ):
        excess, threshold = _threshold_excess(key)
        if excess > 0.0:
            penalty += _SELECTION_PENALTY_WEIGHT * float(np.log1p(excess / threshold))

    edge_hit_excess, edge_hit_threshold = _threshold_excess("cube_edge_hit_rate")
    if edge_hit_excess > 0.0:
        penalty += _SELECTION_PENALTY_WEIGHT * float(
            np.log1p(edge_hit_excess / edge_hit_threshold)
        )

    max_derivative = float(diagnostics.get("spline_max_derivative", float("nan")))
    if np.isfinite(max_derivative) and max_derivative > 0.0:
        penalty += _SELECTION_PENALTY_WEIGHT * max(
            0.0, float(np.log10(max_derivative)) - 6.0
        )

    penalty = min(penalty, _SELECTION_PENALTY_CAP)
    if penalty <= 0.0:
        penalty = 1.0

    return selection_metric + penalty


def _flow_validation_stability_metrics(
    validation_loss: float,
    diagnostics: Mapping[str, Any] | None,
    *,
    inverse_forward_max_abs_error_threshold: float | None,
    inverse_forward_logdet_closure_max_abs_error_threshold: float | None,
    cube_edge_hit_rate_threshold: float | None,
) -> dict[str, float]:
    """Classify a checkpoint as stable and/or pathological from its diagnostics.

    Compares the per-checkpoint diagnostics against the supplied thresholds and
    returns a flat metric dict containing, per checked quantity, a pass flag and
    signed margin, plus two summary gates: ``checkpoint_is_stable`` (no soft
    threshold violated) and ``checkpoint_is_pathological`` (density collapse --
    a non-finite loss, most samples pinned at the cube edges, exploded spline
    derivatives, or an inverse round-trip error so large the map is effectively
    non-invertible). The selection metric uses the soft gate; the hard pathology
    gate additionally disqualifies a checkpoint and bars it from resetting
    early-stopping patience, so a collapsing run is stopped rather than allowed to
    masquerade its diverging loss as improvement.

    Parameters
    ----------
    validation_loss:
        The checkpoint's validation loss (a non-finite value is itself
        pathological).
    diagnostics:
        Per-checkpoint diagnostics (see
        :func:`~petitRADTRANS.sbi.flows.objectives.compute_validation_diagnostics`).
    inverse_forward_max_abs_error_threshold:
    inverse_forward_logdet_closure_max_abs_error_threshold:
    cube_edge_hit_rate_threshold:
        Soft-gate thresholds; ``None`` disables the corresponding check.
    """
    diagnostics = diagnostics or {}
    stable = np.isfinite(float(validation_loss))
    violation_count = 0
    metrics: dict[str, float] = {
        "checkpoint_is_stable": 0.0,
        "checkpoint_stability_violation_count": 0.0,
    }

    def _apply_threshold(
        metric_name: str,
        threshold: float | None,
        *,
        ignore_non_finite: bool = False,
    ) -> None:
        nonlocal stable, violation_count
        if threshold is None:
            return

        metric_value = float(diagnostics.get(metric_name, float("nan")))
        margin_key = f"checkpoint_stability_margin_{metric_name}"
        pass_key = f"checkpoint_stability_pass_{metric_name}"

        if not np.isfinite(metric_value):
            metrics[margin_key] = float("nan")
            metrics[pass_key] = 1.0 if ignore_non_finite else 0.0
            if not ignore_non_finite:
                stable = False
                violation_count += 1
            return

        passed = metric_value <= threshold
        metrics[margin_key] = float(threshold - metric_value)
        metrics[pass_key] = 1.0 if passed else 0.0
        if not passed:
            stable = False
            violation_count += 1

    _apply_threshold(
        "inverse_forward_max_abs_error",
        inverse_forward_max_abs_error_threshold,
    )
    _apply_threshold(
        "inverse_forward_logdet_closure_max_abs_error",
        inverse_forward_logdet_closure_max_abs_error_threshold,
    )
    _apply_threshold(
        "cube_edge_hit_rate",
        cube_edge_hit_rate_threshold,
        ignore_non_finite=True,
    )

    metrics["checkpoint_is_stable"] = 1.0 if stable else 0.0
    metrics["checkpoint_stability_violation_count"] = float(violation_count)

    # Hard-pathology flag (distinct from the soft "stable" gate). This marks
    # density collapse -- the dominant failure of an unbounded flow on a
    # low-noise task: most samples piled at the cube edges, exploded spline
    # derivatives, or an inverse error so large the map is non-invertible in
    # practice. These are categorically different from a marginally-over-
    # threshold-but-usable checkpoint, so they are handled separately:
    # disqualified from selection and barred from resetting early-stop patience,
    # which stops a collapsing run instead of letting its diverging loss
    # masquerade as improvement.
    edge_hit_rate = float(diagnostics.get("cube_edge_hit_rate", float("nan")))
    max_derivative = float(diagnostics.get("spline_max_derivative", float("nan")))
    inverse_error = float(diagnostics.get("inverse_forward_max_abs_error", float("nan")))
    is_pathological = (
        not np.isfinite(float(validation_loss))
        or (np.isfinite(edge_hit_rate) and edge_hit_rate > 0.5)
        or (np.isfinite(max_derivative) and max_derivative > 1.0e4)
        or (
            np.isfinite(inverse_error)
            and inverse_error > _PATHOLOGICAL_INVERSE_FORWARD_MAX_ABS_ERROR
        )
    )
    metrics["checkpoint_is_pathological"] = 1.0 if is_pathological else 0.0
    return metrics
