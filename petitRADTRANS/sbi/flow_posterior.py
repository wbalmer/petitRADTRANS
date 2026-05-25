"""Conditional normalizing-flow posterior estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, cast

import equinox as eqx
from jax import Array, random, vmap
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from .encoders import HierarchicalObservationEncoder
from .flows import (
    ConditionalAutoregressiveFlow,
    ConditionalAffineFlow,
    ConditionalNeuralAutoregressiveFlow,
    ConditionalRationalQuadraticSplineCoupling,
    ConditionalScalarSplineTransform,
    ConditionalSplineFlow,
    _normalize_bin_sizes,
)
from .observation import EncodedObservation, ObservationBlock
from .posterior_base import (
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorSamples,
    TrainingArtifacts,
)
from .training import EarlyStoppingConfig, SBITrainer, TrainingConfig


class _PosteriorModel(eqx.Module):
    encoder: HierarchicalObservationEncoder
    flow: Any
    scale_head: eqx.nn.Linear
    aux_scale_loss_weight: float = eqx.field(static=True)
    spline_small_bin_regularization_weight: float = eqx.field(static=True)
    spline_derivative_regularization_weight: float = eqx.field(static=True)
    spline_min_bin_ratio_target: float = eqx.field(static=True)
    spline_min_derivative_target: float = eqx.field(static=True)
    spline_max_derivative_target: float = eqx.field(static=True)


_VALIDATION_DIAGNOSTIC_SAMPLES = 64
_DIAGNOSTIC_EDGE_THRESHOLD = 1e-6
_REGULARIZATION_TAIL_FRACTION = 0.25
_DIAGNOSTIC_FLOOR_RELATIVE_TOLERANCE = 1.05


def _cube_epsilon(dtype: Any) -> float:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return 1e-7
    return 1e-12


def _clip_cube_support(values: Any) -> jnp.ndarray:
    array_values = jnp.asarray(values)
    epsilon = _cube_epsilon(array_values.dtype)
    return jnp.clip(array_values, epsilon, 1.0 - epsilon)


def _cube_to_unconstrained(values: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    cube_values = _clip_cube_support(values)
    unconstrained_values = jnp.log(cube_values) - jnp.log1p(-cube_values)
    return unconstrained_values, cube_values


def _cube_log_abs_det_jacobian(cube_values: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(jnp.log(cube_values) + jnp.log1p(-cube_values), axis=-1)


def _minimum_cube_spline_bound(dtype: Any = jnp.float32, margin: float = 1.0) -> float:
    epsilon = _cube_epsilon(dtype)
    interior_edge = 1.0 - epsilon
    required = float(np.log(interior_edge) - np.log(epsilon))
    return required + float(margin)


def _parameter_space_log_prob(
    flow: Any,
    parameters: jnp.ndarray,
    embeddings: jnp.ndarray,
    parameter_space: str,
) -> jnp.ndarray:
    if parameter_space == "cube":
        unconstrained_parameters, cube_parameters = _cube_to_unconstrained(parameters)
        log_probabilities = vmap(flow.log_prob)(unconstrained_parameters, embeddings)
        return log_probabilities + _cube_log_abs_det_jacobian(cube_parameters)

    return vmap(flow.log_prob)(parameters, embeddings)


def _mean_spectral_log_scale(blocks: list[ObservationBlock]) -> float | None:
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
    from .observation import PreStackedObservations

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


def _slice_observations_for_diagnostics(observations: Any, n_samples: int) -> Any:
    from .observation import PreStackedObservations

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
    parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
    n_samples = min(int(parameters.shape[0]), int(max_samples))
    return PosteriorBatch(
        parameters=parameters[:n_samples],
        observations=_slice_observations_for_diagnostics(batch.observations, n_samples),
        metadata=batch.metadata,
    )


def _flow_space_parameters(
    parameters: jnp.ndarray,
    parameter_space: str,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    if parameter_space == "cube":
        unconstrained_parameters, cube_parameters = _cube_to_unconstrained(parameters)
        return unconstrained_parameters, cube_parameters
    return parameters, None


def _normalized_categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
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
                spline_layer.min_derivative + nn.softplus(derivatives_raw)
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
            derivatives = scalar_layer.min_derivative + nn.softplus(derivatives_raw)
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
    selection_metric = float(validation_loss)
    if not np.isfinite(selection_metric) or diagnostics is None:
        return selection_metric

    checkpoint_is_stable = float(diagnostics.get("checkpoint_is_stable", float("nan")))
    if np.isfinite(checkpoint_is_stable) and checkpoint_is_stable >= 0.5:
        return selection_metric

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

    penalty = 0.0

    for key in (
        "forward_inverse_max_abs_error",
        "inverse_forward_max_abs_error",
        "forward_inverse_logdet_closure_max_abs_error",
        "inverse_forward_logdet_closure_max_abs_error",
    ):
        excess, threshold = _threshold_excess(key)
        if excess > 0.0:
            penalty += 50.0 * float(np.log1p(excess / threshold))

    edge_hit_excess, edge_hit_threshold = _threshold_excess("cube_edge_hit_rate")
    if edge_hit_excess > 0.0:
        penalty += 250.0 * (edge_hit_excess / edge_hit_threshold)

    max_derivative = float(diagnostics.get("spline_max_derivative", float("nan")))
    if np.isfinite(max_derivative) and max_derivative > 0.0:
        penalty += 100.0 * max(0.0, float(np.log10(max_derivative)) - 6.0)

    if penalty <= 0.0:
        penalty = 1e3

    return selection_metric + penalty


def _flow_validation_stability_metrics(
    validation_loss: float,
    diagnostics: Mapping[str, Any] | None,
    *,
    inverse_forward_max_abs_error_threshold: float | None,
    inverse_forward_logdet_closure_max_abs_error_threshold: float | None,
    cube_edge_hit_rate_threshold: float | None,
) -> dict[str, float]:
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
    return metrics


class ConditionalFlowPosterior(PersistentPosteriorEstimator):
    """Concrete amortized posterior using a conditional flow backend.

    Parameters
    ----------
    parameter_dim:
        Number of inferred free parameters represented by the posterior.
    embedding_dim:
        Size of the learned observation embedding consumed by the flow.
    num_coupling_layers:
        Number of conditional coupling or spline-transform layers in the flow.
    hidden_dim:
        Hidden width used by encoder-side and flow-side MLP conditioners.
    learning_rate:
        Optimizer learning rate used by :class:`SBITrainer`.
    batch_size:
        Number of simulations processed per optimization step.
    num_epochs:
        Maximum number of passes through the training split.
    parameter_space:
        Parameter coordinates learned by the posterior. Supported values are
        ``'physical'``, ``'cube'``, and ``'unconstrained'``.
    flow_family:
        Conditional density-transform family. Supported values are
        ``'spline'``, ``'affine'``, ``'autoregressive'``, and
        ``'neural_autoregressive'``.
    num_spline_bins:
        Number of rational-quadratic spline bins when ``flow_family='spline'``.
    spline_bound:
        Finite support bound of each spline transform in latent space.
    early_stopping_patience:
        Optional number of non-improving epochs tolerated before stopping.
    early_stopping_min_delta:
        Minimum improvement required for early-stopping comparisons.
    checkpoint_directory:
        Optional directory used to persist resumable trainer checkpoints.
    checkpoint_backend:
        Checkpoint persistence backend name. ``'auto'`` selects Orbax when
        available and otherwise falls back to Equinox serialization.
    resume_from_checkpoint:
        Whether ``fit`` should attempt to resume from the latest checkpoint.
    seed:
        Base random seed used for flow initialization and posterior sampling.
    task_metadata:
        Optional user-supplied metadata persisted alongside the trained model.

    Notes
    -----
    The posterior stores task fingerprinting, observation schema, and
    preprocessing payload information when those are available from the training
    dataset reader. That metadata is later reused by inference and artifact
    registration paths.
    """

    estimator_family = "conditional_flow"

    def __init__(
        self,
        parameter_dim: int,
        embedding_dim: int = 128,
        num_coupling_layers: int = 4,
        hidden_dim: int = 128,
        conditioner_depth: int = 2,
        autoregressive_transform_units: int = 16,
        neural_autoregressive_min_slope: float = 1e-3,
        neural_autoregressive_min_residual: float = 0.05,
        neural_autoregressive_inverse_bisection_steps: int = 48,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 5,
        parameter_space: str = "unconstrained",
        flow_family: str = "spline",
        num_spline_bins: int = 8,
        spline_bound: float = 10.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        checkpoint_directory: str | None = None,
        checkpoint_backend: str = "auto",
        resume_from_checkpoint: bool = False,
        gradient_clip_norm: float | None = 1.0,
        embedding_noise_std: float = 0.0,
        embedding_noise_min_scale: float = 0.0,
        aux_scale_loss_weight: float = 0.0,
        spline_small_bin_regularization_weight: float | None = None,
        spline_min_bin_ratio_target: float = 0.2,
        spline_entropy_regularization_weight: float = 0.0,
        spline_derivative_regularization_weight: float = 0.0,
        spline_entropy_floor: float = 0.6,
        spline_min_derivative_target: float = 0.25,
        spline_max_derivative_target: float = 5.0,
        weight_decay: float = 0.0,
        use_cosine_schedule: bool = False,
        warmup_fraction: float = 0.02,
        warmup_epochs: float | None = None,
        min_learning_rate: float = 1e-6,
        lr_schedule_total_epochs: float | None = None,
        stable_inverse_forward_max_abs_error_threshold: float | None = 1e-4,
        stable_inverse_forward_logdet_closure_max_abs_error_threshold: float | None = 1e-4,
        stable_cube_edge_hit_rate_threshold: float | None = 0.0,
        spectrum_encoder_type: str = "conv1d",
        n_wavelengths: int = 233,
        spectrum_embedding_dim: int = 64,
        photometry_embedding_dim: int = 64,
        encoder_hidden_dim: int | None = None,
        encoder_patch_size: int = 32,
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
        verbose_diagnostics: bool = False,
        diagnostics_output_directory: str | None = None,
        diagnostics_plot_interval: int = 1,
    ):
        super().__init__(
            parameter_dim=parameter_dim,
            parameter_space=parameter_space,
            seed=seed,
            task_metadata=task_metadata,
        )
        self.embedding_dim = int(embedding_dim)
        self.num_coupling_layers = int(num_coupling_layers)
        self.hidden_dim = int(hidden_dim)
        self.conditioner_depth = int(conditioner_depth)
        self.autoregressive_transform_units = int(autoregressive_transform_units)
        self.neural_autoregressive_min_slope = float(
            neural_autoregressive_min_slope
        )
        self.neural_autoregressive_min_residual = float(
            neural_autoregressive_min_residual
        )
        self.neural_autoregressive_inverse_bisection_steps = int(
            neural_autoregressive_inverse_bisection_steps
        )
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.flow_family = str(flow_family)
        self.effective_flow_family = self.flow_family
        self.num_spline_bins = int(num_spline_bins)
        requested_spline_bound = float(spline_bound)
        if self.parameter_space == "cube" and self.flow_family == "spline":
            self.spline_bound = max(
                requested_spline_bound,
                _minimum_cube_spline_bound(jnp.float32),
            )
        else:
            self.spline_bound = requested_spline_bound
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.gradient_clip_norm = gradient_clip_norm
        self.embedding_noise_std = float(embedding_noise_std)
        self.embedding_noise_min_scale = float(embedding_noise_min_scale)
        self.aux_scale_loss_weight = float(aux_scale_loss_weight)
        self.spline_entropy_regularization_weight = float(
            spline_entropy_regularization_weight
        )
        if spline_small_bin_regularization_weight is None:
            spline_small_bin_regularization_weight = self.spline_entropy_regularization_weight
        self.spline_small_bin_regularization_weight = float(
            spline_small_bin_regularization_weight
        )
        self.spline_derivative_regularization_weight = float(
            spline_derivative_regularization_weight
        )
        self.spline_min_bin_ratio_target = float(spline_min_bin_ratio_target)
        self.spline_entropy_floor = float(np.clip(spline_entropy_floor, 0.0, 1.0))
        self.spline_min_derivative_target = float(spline_min_derivative_target)
        self.spline_max_derivative_target = float(spline_max_derivative_target)
        self.weight_decay = float(weight_decay)
        self.use_cosine_schedule = bool(use_cosine_schedule)
        self.warmup_fraction = float(warmup_fraction)
        self.warmup_epochs = (
            None if warmup_epochs is None else float(warmup_epochs)
        )
        self.min_learning_rate = float(min_learning_rate)
        self.lr_schedule_total_epochs = (
            None
            if lr_schedule_total_epochs is None
            else float(lr_schedule_total_epochs)
        )
        self.stable_inverse_forward_max_abs_error_threshold = (
            None
            if stable_inverse_forward_max_abs_error_threshold is None
            else float(stable_inverse_forward_max_abs_error_threshold)
        )
        self.stable_inverse_forward_logdet_closure_max_abs_error_threshold = (
            None
            if stable_inverse_forward_logdet_closure_max_abs_error_threshold is None
            else float(stable_inverse_forward_logdet_closure_max_abs_error_threshold)
        )
        self.stable_cube_edge_hit_rate_threshold = (
            None
            if stable_cube_edge_hit_rate_threshold is None
            else float(stable_cube_edge_hit_rate_threshold)
        )
        self.spectrum_encoder_type = str(spectrum_encoder_type)
        self.n_wavelengths = int(n_wavelengths)
        self.spectrum_embedding_dim = int(spectrum_embedding_dim)
        self.photometry_embedding_dim = int(photometry_embedding_dim)
        self.encoder_hidden_dim = (
            int(encoder_hidden_dim) if encoder_hidden_dim is not None else self.hidden_dim
        )
        self.encoder_patch_size = int(encoder_patch_size)
        self.verbose_diagnostics = bool(verbose_diagnostics)
        self.diagnostics_output_directory = diagnostics_output_directory
        self.diagnostics_plot_interval = int(diagnostics_plot_interval)

        if self.spline_small_bin_regularization_weight < 0.0:
            raise ValueError("spline_small_bin_regularization_weight must be non-negative")
        if self.spline_derivative_regularization_weight < 0.0:
            raise ValueError("spline_derivative_regularization_weight must be non-negative")
        if self.spline_min_bin_ratio_target <= 0.0:
            raise ValueError("spline_min_bin_ratio_target must be positive")
        if self.spline_min_derivative_target <= 0.0:
            raise ValueError("spline_min_derivative_target must be positive")
        if self.spline_max_derivative_target <= 0.0:
            raise ValueError("spline_max_derivative_target must be positive")
        if self.spline_max_derivative_target <= self.spline_min_derivative_target:
            raise ValueError(
                "spline_max_derivative_target must exceed spline_min_derivative_target"
            )
        if self.autoregressive_transform_units <= 0:
            raise ValueError("autoregressive_transform_units must be positive")
        if self.neural_autoregressive_min_slope <= 0.0:
            raise ValueError("neural_autoregressive_min_slope must be positive")
        if self.embedding_noise_min_scale < 0.0:
            raise ValueError("embedding_noise_min_scale must be non-negative")
        if self.warmup_epochs is not None and self.warmup_epochs < 0.0:
            raise ValueError("warmup_epochs must be non-negative when provided")
        if (
            self.lr_schedule_total_epochs is not None
            and self.lr_schedule_total_epochs <= 0.0
        ):
            raise ValueError(
                "lr_schedule_total_epochs must be positive when provided"
            )
        if not 0.0 < self.neural_autoregressive_min_residual < 1.0:
            raise ValueError(
                "neural_autoregressive_min_residual must lie in (0, 1)"
            )
        if self.neural_autoregressive_inverse_bisection_steps <= 0:
            raise ValueError(
                "neural_autoregressive_inverse_bisection_steps must be positive"
            )
        if (
            self.stable_inverse_forward_max_abs_error_threshold is not None
            and self.stable_inverse_forward_max_abs_error_threshold < 0.0
        ):
            raise ValueError(
                "stable_inverse_forward_max_abs_error_threshold must be non-negative"
            )
        if (
            self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
            is not None
            and self.stable_inverse_forward_logdet_closure_max_abs_error_threshold < 0.0
        ):
            raise ValueError(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold must be non-negative"
            )
        if (
            self.stable_cube_edge_hit_rate_threshold is not None
            and self.stable_cube_edge_hit_rate_threshold < 0.0
        ):
            raise ValueError("stable_cube_edge_hit_rate_threshold must be non-negative")

        encoder_key, flow_key, scale_key = random.split(random.PRNGKey(self.seed), 3)
        self.model = _PosteriorModel(
            encoder=HierarchicalObservationEncoder(
                embedding_dim=self.embedding_dim,
                spectrum_embedding_dim=self.spectrum_embedding_dim,
                photometry_embedding_dim=self.photometry_embedding_dim,
                patch_size=self.encoder_patch_size,
                hidden_dim=self.encoder_hidden_dim,
                spectrum_encoder_type=self.spectrum_encoder_type,
                n_wavelengths=self.n_wavelengths,
                key=encoder_key,
            ),
            flow=self._build_flow(flow_key),
            scale_head=eqx.nn.Linear(
                self.embedding_dim,
                1,
                key=scale_key,
                dtype=jnp.float32,
            ),
            aux_scale_loss_weight=self.aux_scale_loss_weight,
            spline_small_bin_regularization_weight=self.spline_small_bin_regularization_weight,
            spline_derivative_regularization_weight=self.spline_derivative_regularization_weight,
            spline_min_bin_ratio_target=self.spline_min_bin_ratio_target,
            spline_min_derivative_target=self.spline_min_derivative_target,
            spline_max_derivative_target=self.spline_max_derivative_target,
        )

    def _build_flow(self, key: Array) -> Any:
        if self.flow_family == "affine":
            self.effective_flow_family = "affine"
            return ConditionalAffineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                key=key,
            )
        if self.flow_family == "autoregressive":
            self.effective_flow_family = "autoregressive"
            return ConditionalAutoregressiveFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                key=key,
            )
        if self.flow_family in {"neural_autoregressive", "naf"}:
            self.effective_flow_family = "neural_autoregressive"
            return ConditionalNeuralAutoregressiveFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                transform_units=self.autoregressive_transform_units,
                min_slope=self.neural_autoregressive_min_slope,
                min_residual=self.neural_autoregressive_min_residual,
                inverse_bisection_steps=self.neural_autoregressive_inverse_bisection_steps,
                key=key,
            )
        if self.flow_family == "spline":
            self.effective_flow_family = "spline"
            return ConditionalSplineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                num_bins=self.num_spline_bins,
                bound=self.spline_bound,
                key=key,
            )
        raise ValueError(f"Unsupported flow_family: {self.flow_family!r}")

    @staticmethod
    def _batch_embeddings(model: _PosteriorModel, observations: Any) -> jnp.ndarray:
        from .observation import PreStackedObservations

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

    @staticmethod
    def _loss(model: _PosteriorModel, batch: PosteriorBatch) -> jnp.ndarray:
        parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
        return ConditionalFlowPosterior._loss_from_observations(
            model,
            parameters,
            batch.observations,
            str(batch.metadata.get("parameter_space", "physical")),
            include_spline_regularization=False,
        )

    @staticmethod
    def _training_loss(model: _PosteriorModel, batch: PosteriorBatch) -> jnp.ndarray:
        parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
        return ConditionalFlowPosterior._loss_from_observations(
            model,
            parameters,
            batch.observations,
            str(batch.metadata.get("parameter_space", "physical")),
            include_spline_regularization=True,
        )

    @staticmethod
    def _loss_from_observations(
        model: _PosteriorModel,
        parameters: jnp.ndarray,
        observations: Any,
        parameter_space: str,
        *,
        include_spline_regularization: bool,
    ) -> jnp.ndarray:
        embeddings = ConditionalFlowPosterior._batch_embeddings(model, observations)
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

    def _validation_diagnostics(
        self,
        model: _PosteriorModel,
        batch: PosteriorBatch,
    ) -> dict[str, float]:
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

        embeddings = ConditionalFlowPosterior._batch_embeddings(
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
            if self.verbose_diagnostics and cube_samples.ndim == 2:
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

    @staticmethod
    def _make_noisy_loss(
        noise_std: float,
        preprocessing_metadata: Any = None,
        *,
        noise_min_scale: float = 0.0,
        include_spline_regularization: bool = True,
    ) -> Callable[[_PosteriorModel, PosteriorBatch], jnp.ndarray]:
        """Return a loss function that injects on-the-fly Gaussian noise into
        the observation spectra before encoding.

        For spectra preprocessed into a per-sample transformed value space,
        the uncertainty channel is assumed to already be expressed in the same
        transformed units as the value channel, so noise can be injected
        directly from that channel. When preprocessing metadata is provided,
        non-spectral blocks still reconstruct raw measurement uncertainties so
        noise is injected at the correct physical scale in normalized-value
        space. ``noise_std`` acts as a multiplier on the observational noise
        level (1.0 = exact observational noise, >1 = regularising over-noise).

        When preprocessing metadata is unavailable, a flat Gaussian noise with
        scale ``noise_std`` is applied as a fallback.
        """

        # Pre-compute per-block factors for reconstructing raw uncertainties
        # from normalised values.  These are broadcast-safe scalars captured as
        # compile-time constants by JAX.
        _block_noise_params: list[tuple[Any, float]] = []
        if preprocessing_metadata is not None:
            for _bname, _bstats in preprocessing_metadata.blocks.items():
                if _bstats.uncertainty_median_per_wl is not None:
                    _unc_scale = np.asarray(
                        _bstats.uncertainty_median_per_wl,
                        dtype=np.float32,
                    )
                    _unc_scale = np.where(
                        np.isfinite(_unc_scale) & (_unc_scale > 0.0),
                        _unc_scale,
                        float(
                            _bstats.uncertainty_median
                            if _bstats.uncertainty_median is not None
                            and _bstats.uncertainty_median > 0
                            else (
                                _bstats.uncertainty_std
                                if _bstats.uncertainty_std is not None
                                and _bstats.uncertainty_std > 0
                                else 1.0
                            )
                        ),
                    )
                else:
                    _unc_scale = float(
                        _bstats.uncertainty_median
                        if _bstats.uncertainty_median is not None
                        and _bstats.uncertainty_median > 0
                        else (
                            _bstats.uncertainty_std
                            if _bstats.uncertainty_std is not None
                            and _bstats.uncertainty_std > 0
                            else 1.0
                        )
                    )

                _value_scale = float(
                    _bstats.value_iqr
                    if _bstats.value_iqr is not None and _bstats.value_iqr > 0
                    else (
                        _bstats.value_std
                        if _bstats.value_std is not None and _bstats.value_std > 0
                        else (
                            abs(_bstats.value_median)
                            if _bstats.value_median is not None and abs(_bstats.value_median) > 0
                            else 1.0
                        )
                    )
                )
                _block_noise_params.append((_unc_scale, _value_scale))

        def _noisy_loss(
            model: _PosteriorModel, batch: PosteriorBatch
        ) -> jnp.ndarray:
            parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
            observations = batch.observations

            # Derive a noise key from the batch content so each step gets
            # unique but reproducible noise without needing external state.
            key = random.fold_in(
                random.PRNGKey(0),
                jnp.bitwise_xor(
                    jax.lax.bitcast_convert_type(
                        parameters[0, 0], jnp.int32
                    ),
                    jax.lax.bitcast_convert_type(
                        parameters[-1, -1], jnp.int32
                    ),
                ),
            )

            # Inject noise into spectral values before encoding.
            from .observation import PreStackedObservations

            if isinstance(observations, PreStackedObservations):
                noisy_blocks = []
                for block_idx, block_tuple in enumerate(observations.stacked_blocks):
                    v, u, c, m = block_tuple[0], block_tuple[1], block_tuple[2], block_tuple[3]
                    lmf = block_tuple[4] if len(block_tuple) > 4 else jnp.zeros(v.shape[0], dtype=v.dtype)
                    sub_key = random.fold_in(key, block_idx)
                    noise = random.normal(sub_key, shape=v.shape, dtype=v.dtype)
                    modality = (
                        observations.modalities[block_idx]
                        if block_idx < len(observations.modalities)
                        else None
                    )
                    updated_block = list(block_tuple)

                    if modality == 'spectrum':
                        safe_log_scale = jnp.where(jnp.isfinite(lmf), lmf, 0.0)
                        sample_scale = jnp.maximum(
                            jnp.power(jnp.asarray(10.0, dtype=v.dtype), safe_log_scale),
                            jnp.asarray(1e-30, dtype=v.dtype),
                        )
                        if len(block_tuple) > 6:
                            global_scale = jnp.asarray(block_tuple[6], dtype=v.dtype)
                        else:
                            global_scale = sample_scale
                        global_scale = jnp.maximum(
                            jnp.where(jnp.isfinite(global_scale), global_scale, sample_scale),
                            jnp.asarray(1e-30, dtype=v.dtype),
                        )
                        noise_scale = jnp.maximum(
                            noise_std * jnp.abs(u),
                            jnp.asarray(noise_min_scale, dtype=v.dtype),
                        )
                        v_noisy = v + noise_scale * noise
                        updated_block[0] = v_noisy

                        if len(block_tuple) > 5:
                            absolute_values = jnp.asarray(block_tuple[5], dtype=v.dtype)
                            raw_uncertainty = (
                                jnp.abs(u)
                                * global_scale[:, None]
                                * jnp.cosh(v)
                            )
                            absolute_local_scale = sample_scale[:, None] * jnp.cosh(absolute_values)
                            absolute_noise_scale = raw_uncertainty / jnp.maximum(
                                absolute_local_scale,
                                jnp.asarray(1e-30, dtype=v.dtype),
                            )
                            absolute_noise = jnp.maximum(
                                noise_std * absolute_noise_scale,
                                jnp.asarray(noise_min_scale, dtype=v.dtype),
                            )
                            updated_block[5] = absolute_values + absolute_noise * noise
                    elif block_idx < len(_block_noise_params):
                        # Reconstruct raw uncertainty from the normalized
                        # uncertainty channel and convert it into the scale of
                        # the normalized value channel.
                        _unc_scale, _value_scale = _block_noise_params[block_idx]
                        u_raw = u * jnp.asarray(_unc_scale, dtype=u.dtype)
                        noise_scale = jnp.abs(u_raw) / jnp.asarray(_value_scale, dtype=u.dtype)
                        effective_noise_scale = jnp.maximum(
                            noise_std * noise_scale,
                            jnp.asarray(noise_min_scale, dtype=v.dtype),
                        )
                        v_noisy = v + effective_noise_scale * noise
                        updated_block[0] = v_noisy
                    else:
                        # Fallback: no preprocessing stats for this block.
                        effective_noise_scale = jnp.asarray(
                            max(float(noise_std), float(noise_min_scale)),
                            dtype=v.dtype,
                        )
                        v_noisy = v + effective_noise_scale * noise
                        updated_block[0] = v_noisy

                    noisy_blocks.append(tuple(updated_block))
                observations = PreStackedObservations(
                    stacked_blocks=tuple(noisy_blocks),
                    modalities=observations.modalities,
                )

            return ConditionalFlowPosterior._loss_from_observations(
                model,
                parameters,
                observations,
                str(batch.metadata.get("parameter_space", "physical")),
                include_spline_regularization=include_spline_regularization,
            )

        return _noisy_loss

    def fit(self, dataset: Any) -> TrainingArtifacts:
        """Train the posterior on one normalized simulation dataset reader.

        Parameters
        ----------
        dataset:
            Reader-like object that yields :class:`PosteriorBatch` instances via
            ``iter_batches`` and exposes dataset manifest metadata when
            available.

        Returns
        -------
        TrainingArtifacts
            Training history, validation metrics, and trainer metadata for the
            completed optimization run.

        Notes
        -----
        If the reader exposes preprocessing metadata or a manifest fingerprint,
        those are cached on the posterior so they can be saved and reused by the
        inference and artifact layers.
        """
        manifest = getattr(getattr(dataset, "dataset", None), "manifest", None)
        if manifest is not None:
            self.task_name = str(
                getattr(manifest, "task_name", self.task_name or "unknown_task")
            )
            self.task_fingerprint = getattr(manifest, "task_fingerprint", None)
            self.observation_schema = dict(getattr(manifest, "observation_schema", {}))
        preprocessing_metadata = getattr(dataset, "preprocessing_metadata", None)
        if preprocessing_metadata is not None:
            self.preprocessing_metadata_payload = preprocessing_metadata.to_payload()

        early_stopping = None
        if self.early_stopping_patience is not None:
            early_stopping = EarlyStoppingConfig(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
            )

        trainer = SBITrainer(
            TrainingConfig(
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                parameter_space=self.parameter_space,
                seed=self.seed,
                shuffle_train=True,
                early_stopping=early_stopping,
                checkpoint_directory=self.checkpoint_directory,
                checkpoint_backend=self.checkpoint_backend,
                resume_from_checkpoint=self.resume_from_checkpoint,
                gradient_clip_norm=self.gradient_clip_norm,
                data_parallel=len(jax.devices()) > 1,
                weight_decay=self.weight_decay,
                use_cosine_schedule=self.use_cosine_schedule,
                warmup_fraction=self.warmup_fraction,
                warmup_epochs=self.warmup_epochs,
                min_learning_rate=self.min_learning_rate,
                lr_schedule_total_epochs=self.lr_schedule_total_epochs,
                verbose_diagnostics=self.verbose_diagnostics,
                diagnostics_output_directory=self.diagnostics_output_directory,
                diagnostics_plot_interval=self.diagnostics_plot_interval,
            )
        )
        train_loss_fn = (
            self._make_noisy_loss(
                self.embedding_noise_std,
                preprocessing_metadata,
                noise_min_scale=self.embedding_noise_min_scale,
                include_spline_regularization=True,
            )
            if self.embedding_noise_std > 0 or self.embedding_noise_min_scale > 0
            else self._training_loss
        )
        self.model, history, validation_metrics, training_metadata = trainer.fit(
            self.model,
            dataset,
            train_loss_fn,
            eval_loss_fn=self._loss,
            eval_diagnostic_fn=self._validation_diagnostics,
            selection_metric_fn=_flow_validation_selection_metric,
            selection_metric_name="validation_selection_metric",
            stability_metric_fn=lambda validation_loss, diagnostics: (
                _flow_validation_stability_metrics(
                    validation_loss,
                    diagnostics,
                    inverse_forward_max_abs_error_threshold=(
                        self.stable_inverse_forward_max_abs_error_threshold
                    ),
                    inverse_forward_logdet_closure_max_abs_error_threshold=(
                        self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
                    ),
                    cube_edge_hit_rate_threshold=self.stable_cube_edge_hit_rate_threshold,
                )
            ),
            stability_flag_key="checkpoint_is_stable",
        )
        self.training_artifacts = TrainingArtifacts(
            history=history,
            validation_metrics=validation_metrics,
            metadata={
                "parameter_dim": self.parameter_dim,
                "parameter_space": self.parameter_space,
                "task_metadata": dict(self.task_metadata),
                "task_name": self.task_name,
                "task_fingerprint": self.task_fingerprint,
                "flow_family": self.flow_family,
                "effective_flow_family": self.effective_flow_family,
                "num_spline_bins": self.num_spline_bins,
                "spline_bound": self.spline_bound,
                **training_metadata,
            },
        )
        return self.training_artifacts

    def _build_estimator_config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "num_coupling_layers": self.num_coupling_layers,
            "hidden_dim": self.hidden_dim,
            "conditioner_depth": self.conditioner_depth,
            "autoregressive_transform_units": self.autoregressive_transform_units,
            "neural_autoregressive_min_slope": self.neural_autoregressive_min_slope,
            "neural_autoregressive_min_residual": self.neural_autoregressive_min_residual,
            "neural_autoregressive_inverse_bisection_steps": self.neural_autoregressive_inverse_bisection_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "flow_family": self.flow_family,
            "effective_flow_family": self.effective_flow_family,
            "num_spline_bins": self.num_spline_bins,
            "spline_bound": self.spline_bound,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "checkpoint_directory": self.checkpoint_directory,
            "checkpoint_backend": self.checkpoint_backend,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "aux_scale_loss_weight": self.aux_scale_loss_weight,
            "spline_small_bin_regularization_weight": self.spline_small_bin_regularization_weight,
            "spline_min_bin_ratio_target": self.spline_min_bin_ratio_target,
            "spline_entropy_regularization_weight": self.spline_entropy_regularization_weight,
            "spline_derivative_regularization_weight": self.spline_derivative_regularization_weight,
            "spline_entropy_floor": self.spline_entropy_floor,
            "spline_min_derivative_target": self.spline_min_derivative_target,
            "spline_max_derivative_target": self.spline_max_derivative_target,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "use_cosine_schedule": self.use_cosine_schedule,
            "embedding_noise_std": self.embedding_noise_std,
            "embedding_noise_min_scale": self.embedding_noise_min_scale,
            "warmup_fraction": self.warmup_fraction,
            "warmup_epochs": self.warmup_epochs,
            "min_learning_rate": self.min_learning_rate,
            "lr_schedule_total_epochs": self.lr_schedule_total_epochs,
            "stable_inverse_forward_max_abs_error_threshold": (
                self.stable_inverse_forward_max_abs_error_threshold
            ),
            "stable_inverse_forward_logdet_closure_max_abs_error_threshold": (
                self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
            ),
            "stable_cube_edge_hit_rate_threshold": self.stable_cube_edge_hit_rate_threshold,
            "spectrum_encoder_type": self.spectrum_encoder_type,
            "n_wavelengths": self.n_wavelengths,
            "spectrum_embedding_dim": self.spectrum_embedding_dim,
            "photometry_embedding_dim": self.photometry_embedding_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "encoder_patch_size": self.encoder_patch_size,
            "verbose_diagnostics": self.verbose_diagnostics,
            "diagnostics_output_directory": self.diagnostics_output_directory,
            "diagnostics_plot_interval": self.diagnostics_plot_interval,
        }

    def _build_serialized_metadata(self, artifact_metadata) -> dict[str, Any]:
        metadata = super()._build_serialized_metadata(artifact_metadata)
        metadata.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_coupling_layers": self.num_coupling_layers,
                "hidden_dim": self.hidden_dim,
                "conditioner_depth": self.conditioner_depth,
                "autoregressive_transform_units": self.autoregressive_transform_units,
                "neural_autoregressive_min_slope": self.neural_autoregressive_min_slope,
                "neural_autoregressive_min_residual": self.neural_autoregressive_min_residual,
                "neural_autoregressive_inverse_bisection_steps": self.neural_autoregressive_inverse_bisection_steps,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "flow_family": self.flow_family,
                "effective_flow_family": self.effective_flow_family,
                "num_spline_bins": self.num_spline_bins,
                "spline_bound": self.spline_bound,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "checkpoint_directory": self.checkpoint_directory,
                "checkpoint_backend": self.checkpoint_backend,
                "resume_from_checkpoint": self.resume_from_checkpoint,
                "aux_scale_loss_weight": self.aux_scale_loss_weight,
                "spline_small_bin_regularization_weight": self.spline_small_bin_regularization_weight,
                "spline_min_bin_ratio_target": self.spline_min_bin_ratio_target,
                "spline_entropy_regularization_weight": self.spline_entropy_regularization_weight,
                "spline_derivative_regularization_weight": self.spline_derivative_regularization_weight,
                "spline_entropy_floor": self.spline_entropy_floor,
                "spline_min_derivative_target": self.spline_min_derivative_target,
                "spline_max_derivative_target": self.spline_max_derivative_target,
                "weight_decay": self.weight_decay,
                "gradient_clip_norm": self.gradient_clip_norm,
                "use_cosine_schedule": self.use_cosine_schedule,
                "embedding_noise_std": self.embedding_noise_std,
                "embedding_noise_min_scale": self.embedding_noise_min_scale,
                "warmup_fraction": self.warmup_fraction,
                "warmup_epochs": self.warmup_epochs,
                "min_learning_rate": self.min_learning_rate,
                "lr_schedule_total_epochs": self.lr_schedule_total_epochs,
                "stable_inverse_forward_max_abs_error_threshold": (
                    self.stable_inverse_forward_max_abs_error_threshold
                ),
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold": (
                    self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
                ),
                "stable_cube_edge_hit_rate_threshold": self.stable_cube_edge_hit_rate_threshold,
                "spectrum_encoder_type": self.spectrum_encoder_type,
                "n_wavelengths": self.n_wavelengths,
                "spectrum_embedding_dim": self.spectrum_embedding_dim,
                "photometry_embedding_dim": self.photometry_embedding_dim,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encoder_patch_size": self.encoder_patch_size,
                "verbose_diagnostics": self.verbose_diagnostics,
                "diagnostics_output_directory": self.diagnostics_output_directory,
                "diagnostics_plot_interval": self.diagnostics_plot_interval,
            }
        )
        return metadata

    @staticmethod
    def _resolve_estimator_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
        estimator_config = metadata.get("estimator_config")
        if isinstance(estimator_config, Mapping):
            return dict(estimator_config)

        small_bin_weight = metadata.get("spline_small_bin_regularization_weight")
        if small_bin_weight is None:
            small_bin_weight = metadata.get("spline_entropy_regularization_weight", 0.0)

        return {
            "embedding_dim": metadata["embedding_dim"],
            "num_coupling_layers": metadata["num_coupling_layers"],
            "hidden_dim": metadata["hidden_dim"],
            "learning_rate": metadata["learning_rate"],
            "batch_size": metadata["batch_size"],
            "num_epochs": metadata["num_epochs"],
            "autoregressive_transform_units": metadata.get(
                "autoregressive_transform_units",
                16,
            ),
            "neural_autoregressive_min_slope": metadata.get(
                "neural_autoregressive_min_slope",
                1e-3,
            ),
            "neural_autoregressive_min_residual": metadata.get(
                "neural_autoregressive_min_residual",
                0.05,
            ),
            "neural_autoregressive_inverse_bisection_steps": metadata.get(
                "neural_autoregressive_inverse_bisection_steps",
                48,
            ),
            "flow_family": metadata.get("flow_family", "affine"),
            "effective_flow_family": metadata.get(
                "effective_flow_family", metadata.get("flow_family", "affine")
            ),
            "num_spline_bins": metadata.get("num_spline_bins", 8),
            "spline_bound": metadata.get("spline_bound", 10.0),
            "early_stopping_patience": metadata.get("early_stopping_patience"),
            "early_stopping_min_delta": metadata.get("early_stopping_min_delta", 0.0),
            "checkpoint_directory": metadata.get("checkpoint_directory"),
            "checkpoint_backend": metadata.get("checkpoint_backend", "auto"),
            "resume_from_checkpoint": metadata.get("resume_from_checkpoint", False),
            "conditioner_depth": metadata.get("conditioner_depth", 2),
            "aux_scale_loss_weight": metadata.get("aux_scale_loss_weight", 0.0),
            "spline_small_bin_regularization_weight": small_bin_weight,
            "spline_min_bin_ratio_target": metadata.get("spline_min_bin_ratio_target", 0.2),
            "spline_entropy_regularization_weight": metadata.get(
                "spline_entropy_regularization_weight",
                0.0,
            ),
            "spline_derivative_regularization_weight": metadata.get(
                "spline_derivative_regularization_weight",
                0.0,
            ),
            "spline_entropy_floor": metadata.get("spline_entropy_floor", 0.6),
            "spline_min_derivative_target": metadata.get(
                "spline_min_derivative_target",
                0.25,
            ),
            "spline_max_derivative_target": metadata.get(
                "spline_max_derivative_target",
                5.0,
            ),
            "weight_decay": metadata.get("weight_decay", 0.0),
            "gradient_clip_norm": metadata.get("gradient_clip_norm", 1.0),
            "use_cosine_schedule": metadata.get("use_cosine_schedule", False),
            "embedding_noise_std": metadata.get("embedding_noise_std", 0.0),
            "embedding_noise_min_scale": metadata.get(
                "embedding_noise_min_scale",
                0.0,
            ),
            "warmup_fraction": metadata.get("warmup_fraction", 0.02),
            "warmup_epochs": metadata.get("warmup_epochs"),
            "min_learning_rate": metadata.get("min_learning_rate", 1e-6),
            "lr_schedule_total_epochs": metadata.get(
                "lr_schedule_total_epochs"
            ),
            "stable_inverse_forward_max_abs_error_threshold": metadata.get(
                "stable_inverse_forward_max_abs_error_threshold",
                1e-4,
            ),
            "stable_inverse_forward_logdet_closure_max_abs_error_threshold": metadata.get(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold",
                1e-4,
            ),
            "stable_cube_edge_hit_rate_threshold": metadata.get(
                "stable_cube_edge_hit_rate_threshold",
                0.0,
            ),
            "spectrum_encoder_type": metadata.get("spectrum_encoder_type", "conv1d"),
            "n_wavelengths": metadata.get("n_wavelengths", 233),
            "spectrum_embedding_dim": metadata.get("spectrum_embedding_dim", 64),
            "photometry_embedding_dim": metadata.get("photometry_embedding_dim", 64),
            "encoder_hidden_dim": metadata.get("encoder_hidden_dim"),
            "encoder_patch_size": metadata.get("encoder_patch_size", 32),
            "verbose_diagnostics": metadata.get("verbose_diagnostics", False),
            "diagnostics_output_directory": metadata.get("diagnostics_output_directory"),
            "diagnostics_plot_interval": metadata.get("diagnostics_plot_interval", 1),
        }

    def hydrate_loaded_metadata(self, metadata: Mapping[str, Any]) -> None:
        super().hydrate_loaded_metadata(metadata)
        self.effective_flow_family = str(
            self._resolve_estimator_config(metadata).get(
                "effective_flow_family", self.effective_flow_family
            )
        )

    @classmethod
    def from_serialized_metadata(
        cls, metadata: Mapping[str, Any]
    ) -> "ConditionalFlowPosterior":
        estimator_config = cls._resolve_estimator_config(metadata)
        return cls(
            parameter_dim=int(metadata["parameter_dim"]),
            embedding_dim=int(estimator_config["embedding_dim"]),
            num_coupling_layers=int(estimator_config["num_coupling_layers"]),
            hidden_dim=int(estimator_config["hidden_dim"]),
            conditioner_depth=int(estimator_config.get("conditioner_depth", 2)),
            autoregressive_transform_units=int(
                estimator_config.get("autoregressive_transform_units", 16)
            ),
            neural_autoregressive_min_slope=float(
                estimator_config.get("neural_autoregressive_min_slope", 1e-3)
            ),
            neural_autoregressive_min_residual=float(
                estimator_config.get("neural_autoregressive_min_residual", 0.05)
            ),
            neural_autoregressive_inverse_bisection_steps=int(
                estimator_config.get(
                    "neural_autoregressive_inverse_bisection_steps",
                    48,
                )
            ),
            learning_rate=float(estimator_config["learning_rate"]),
            batch_size=int(estimator_config["batch_size"]),
            num_epochs=int(estimator_config["num_epochs"]),
            parameter_space=str(metadata["parameter_space"]),
            flow_family=str(estimator_config.get("flow_family", "affine")),
            num_spline_bins=int(estimator_config.get("num_spline_bins", 8)),
            spline_bound=float(estimator_config.get("spline_bound", 10.0)),
            early_stopping_patience=estimator_config.get("early_stopping_patience"),
            early_stopping_min_delta=float(
                estimator_config.get("early_stopping_min_delta", 0.0)
            ),
            checkpoint_directory=estimator_config.get("checkpoint_directory"),
            checkpoint_backend=str(estimator_config.get("checkpoint_backend", "auto")),
            resume_from_checkpoint=bool(
                estimator_config.get("resume_from_checkpoint", False)
            ),
            aux_scale_loss_weight=float(
                estimator_config.get("aux_scale_loss_weight", 0.0)
            ),
            spline_small_bin_regularization_weight=float(
                estimator_config.get(
                    "spline_small_bin_regularization_weight",
                    estimator_config.get("spline_entropy_regularization_weight", 0.0),
                )
            ),
            spline_min_bin_ratio_target=float(
                estimator_config.get("spline_min_bin_ratio_target", 0.2)
            ),
            spline_entropy_regularization_weight=float(
                estimator_config.get("spline_entropy_regularization_weight", 0.0)
            ),
            spline_derivative_regularization_weight=float(
                estimator_config.get("spline_derivative_regularization_weight", 0.0)
            ),
            spline_entropy_floor=float(estimator_config.get("spline_entropy_floor", 0.6)),
            spline_min_derivative_target=float(
                estimator_config.get("spline_min_derivative_target", 0.25)
            ),
            spline_max_derivative_target=float(
                estimator_config.get("spline_max_derivative_target", 5.0)
            ),
            weight_decay=float(estimator_config.get("weight_decay", 0.0)),
            gradient_clip_norm=estimator_config.get("gradient_clip_norm", 1.0),
            embedding_noise_std=float(
                estimator_config.get("embedding_noise_std", 0.0)
            ),
            embedding_noise_min_scale=float(
                estimator_config.get("embedding_noise_min_scale", 0.0)
            ),
            use_cosine_schedule=bool(estimator_config.get("use_cosine_schedule", False)),
            warmup_fraction=float(estimator_config.get("warmup_fraction", 0.02)),
            warmup_epochs=(
                None
                if estimator_config.get("warmup_epochs") is None
                else float(estimator_config.get("warmup_epochs"))
            ),
            min_learning_rate=float(estimator_config.get("min_learning_rate", 1e-6)),
            lr_schedule_total_epochs=(
                None
                if estimator_config.get("lr_schedule_total_epochs") is None
                else float(estimator_config.get("lr_schedule_total_epochs"))
            ),
            stable_inverse_forward_max_abs_error_threshold=estimator_config.get(
                "stable_inverse_forward_max_abs_error_threshold",
                1e-4,
            ),
            stable_inverse_forward_logdet_closure_max_abs_error_threshold=estimator_config.get(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold",
                1e-4,
            ),
            stable_cube_edge_hit_rate_threshold=estimator_config.get(
                "stable_cube_edge_hit_rate_threshold",
                0.0,
            ),
            spectrum_encoder_type=str(estimator_config.get("spectrum_encoder_type", "conv1d")),
            n_wavelengths=int(estimator_config.get("n_wavelengths", 233)),
            spectrum_embedding_dim=int(estimator_config.get("spectrum_embedding_dim", 64)),
            photometry_embedding_dim=int(estimator_config.get("photometry_embedding_dim", 64)),
            encoder_hidden_dim=estimator_config.get("encoder_hidden_dim"),
            encoder_patch_size=int(estimator_config.get("encoder_patch_size", 32)),
            verbose_diagnostics=bool(estimator_config.get("verbose_diagnostics", False)),
            diagnostics_output_directory=estimator_config.get("diagnostics_output_directory"),
            diagnostics_plot_interval=int(estimator_config.get("diagnostics_plot_interval", 1)),
            seed=int(metadata["seed"]),
            task_metadata=dict(metadata.get("task_metadata", {})),
        )

    def save_backend_state(self, output_path: Path) -> None:
        eqx.tree_serialise_leaves(output_path / "model.eqx", self.model)

    def load_backend_state(self, input_path: Path) -> None:
        self.model = eqx.tree_deserialise_leaves(input_path / "model.eqx", self.model)

    def encode_observation(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        """Encode one structured observation into the posterior context space.

        Parameters
        ----------
        blocks:
            Observation blocks describing one spectral/photometric observation.

        Returns
        -------
        EncodedObservation
            Aggregated embedding and lightweight metadata describing the input
            observation family.
        """
        return self.model.encoder.encode(blocks)

    def batch_encode_observation(
        self, blocks_list: list[list[ObservationBlock]]
    ) -> list[EncodedObservation]:
        """Encode a batch of observations using a single vmapped forward pass.

        Parameters
        ----------
        blocks_list:
            List of per-sample observation block lists.

        Returns
        -------
        list[EncodedObservation]
            One encoded observation per input sample.
        """
        embeddings = self.model.encoder.encode_stacked_batch(blocks_list)
        return [
            EncodedObservation(embedding=embeddings[i]) for i in range(len(blocks_list))
        ]

    def sample_posterior(
        self,
        observation: EncodedObservation,
        n_samples: int,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw posterior samples conditioned on one encoded observation.

        Parameters
        ----------
        observation:
            Encoded observation produced by :meth:`encode_observation`.
        n_samples:
            Number of posterior draws to generate.
        seed:
            Optional random seed overriding the model-level default seed.

        Returns
        -------
        PosteriorSamples
            Samples in the posterior's configured parameter space. Non-finite
            outputs are clamped to large finite values for downstream stability.
        """
        sample_seed = self.seed if seed is None else int(seed)
        samples = self.model.flow.sample(
            context=jnp.asarray(observation.embedding, dtype=jnp.float32),
            n_samples=n_samples,
            key=random.PRNGKey(sample_seed),
        )
        if self.parameter_space == "cube":
            samples = _clip_cube_support(nn.sigmoid(samples))
        return PosteriorSamples(
            samples=np.nan_to_num(np.asarray(samples), nan=0.0, posinf=1e6, neginf=-1e6)
        )

    def batch_sample_posterior(
        self,
        embeddings: Any,
        n_samples: int,
        base_seed: int = 0,
    ) -> np.ndarray:
        """Draw posterior samples for a batch of encoded observations.

        Runs a single JIT-compiled vmapped call over all contexts rather than
        looping ``sample_posterior`` once per observation, eliminating the
        Python overhead of per-observation dispatch.

        Parameters
        ----------
        embeddings:
            Float32 array of shape ``(batch_size, embedding_dim)`` produced by
            stacking :attr:`EncodedObservation.embedding` vectors.
        n_samples:
            Number of posterior draws per observation.
        base_seed:
            Base random seed. Each observation in the batch receives a unique
            sub-key derived from this seed.

        Returns
        -------
        np.ndarray
            Array of shape ``(batch_size, n_samples, parameter_dim)`` with
            non-finite values clamped to large finite values.
        """
        contexts = jnp.asarray(embeddings, dtype=jnp.float32)
        n = contexts.shape[0]
        keys = random.split(random.PRNGKey(base_seed), n)
        samples = _jit_batch_sample(self.model.flow, contexts, n_samples, keys)
        if self.parameter_space == "cube":
            samples = _clip_cube_support(nn.sigmoid(samples))
        return np.nan_to_num(np.asarray(samples), nan=0.0, posinf=1e6, neginf=-1e6)

    def log_prob(self, observation: EncodedObservation, parameters: Any) -> Any:
        """Evaluate posterior log-density for one or many parameter vectors.

        Parameters
        ----------
        observation:
            Encoded observation that defines the posterior context.
        parameters:
            One parameter vector or a batch of parameter vectors in the
            posterior's configured parameter space.

        Returns
        -------
        Any
            Scalar log-density or a vector of log-densities matching the input
            batch structure.
        """
        parameter_values = jnp.asarray(parameters, dtype=jnp.float32)
        context = jnp.asarray(observation.embedding, dtype=jnp.float32)

        if parameter_values.ndim == 1:
            if self.parameter_space == "cube":
                unconstrained_values, cube_values = _cube_to_unconstrained(parameter_values)
                return jnp.nan_to_num(
                    self.model.flow.log_prob(unconstrained_values, context)
                    + _cube_log_abs_det_jacobian(cube_values),
                    nan=-1e6,
                    neginf=-1e6,
                )
            return jnp.nan_to_num(
                self.model.flow.log_prob(parameter_values, context),
                nan=-1e6,
                neginf=-1e6,
            )

        if self.parameter_space == "cube":
            unconstrained_values, cube_values = _cube_to_unconstrained(parameter_values)
            return vmap(
                lambda value, cube_value: jnp.nan_to_num(
                    self.model.flow.log_prob(value, context)
                    + _cube_log_abs_det_jacobian(cube_value),
                    nan=-1e6,
                    neginf=-1e6,
                )
            )(unconstrained_values, cube_values)

        return vmap(
            lambda value: jnp.nan_to_num(
                self.model.flow.log_prob(value, context), nan=-1e6, neginf=-1e6
            )
        )(parameter_values)


@eqx.filter_jit
def _jit_batch_sample(
    flow: Any,
    contexts: Any,
    n_samples: int,
    keys: Any,
) -> Any:
    """JIT-compiled vmapped sampling over a batch of observation contexts.

    Parameters
    ----------
    flow:
        Trained conditional flow (e.g. :class:`ConditionalSplineFlow`).
    contexts:
        Float32 array of shape ``(batch_size, embedding_dim)``.
    n_samples:
        Number of posterior draws per observation. Treated as a compile-time
        constant; retracing only occurs if this value changes.
    keys:
        Array of JAX PRNGKeys with shape ``(batch_size, 2)``.

    Returns
    -------
    Any
        Float32 array of shape ``(batch_size, n_samples, parameter_dim)``.
    """
    return vmap(lambda ctx, k: flow.sample(ctx, n_samples, k))(contexts, keys)


class ConditionalSplineFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the spline flow backend.

    Notes
    -----
    This convenience subclass forces ``flow_family='spline'`` while keeping the
    rest of the :class:`ConditionalFlowPosterior` API unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "spline")
        super().__init__(*args, **kwargs)


class ConditionalAutoregressiveFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the autoregressive flow backend."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "autoregressive")
        super().__init__(*args, **kwargs)


class ConditionalNeuralAutoregressiveFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the neural autoregressive backend."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "neural_autoregressive")
        super().__init__(*args, **kwargs)


__all__ = [
    "ConditionalAutoregressiveFlowPosterior",
    "ConditionalFlowPosterior",
    "ConditionalNeuralAutoregressiveFlowPosterior",
    "ConditionalSplineFlowPosterior",
]
