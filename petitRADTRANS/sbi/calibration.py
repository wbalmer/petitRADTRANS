"""Calibration and posterior-predictive reporting for amortized SBI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import DatasetSplit, NormalizedObservationDatasetReader
from .observation import build_observation_block_batch
from .preprocessing import normalize_observation_blocks
from .simulator import RuntimeSimulator
from .task import SBITask


@dataclass(frozen=True)
class CoverageLevelReport:
    """Coverage summary for one nominal interval level.

    Attributes
    ----------
    level:
        Nominal central credible-interval level.
    empirical_coverage:
        Measured coverage fraction over the evaluated cases.
    absolute_error:
        Absolute difference between empirical and nominal coverage.
    """

    level: float
    empirical_coverage: float
    absolute_error: float


@dataclass(frozen=True)
class SimulationBasedCalibrationReport:
    """Rank-based SBC summary over a held-out set of observations.

    Attributes
    ----------
    ranks:
        Integer rank of the ground-truth parameter within posterior samples for
        each held-out case and parameter dimension.
    rank_histogram_counts:
        Per-parameter SBC histogram counts.
    posterior_means:
        Posterior mean for each held-out case.
    truths:
        Ground-truth parameters paired with each posterior sample set.
    coverages:
        Coverage summaries for the requested nominal interval levels.
    mean_rank:
        Mean empirical rank for each parameter dimension.
    normalized_mean_rank_error:
        Absolute mean-rank error normalized by the expected average rank.
    metadata:
        Auxiliary run metadata such as split name and number of posterior draws.
    """

    ranks: np.ndarray
    rank_histogram_counts: np.ndarray
    posterior_means: np.ndarray
    truths: np.ndarray
    coverages: tuple[CoverageLevelReport, ...]
    mean_rank: np.ndarray
    normalized_mean_rank_error: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorPredictiveReport:
    """Aggregate posterior-predictive summaries for held-out observations.

    Attributes
    ----------
    observed_values:
        Observed values kept on the original observation scale.
    predictive_mean:
        Posterior-predictive mean curves or vectors.
    predictive_std:
        Posterior-predictive standard deviations.
    interval_lower, interval_upper:
        Central predictive interval bounds for the requested level.
    interval_coverage:
        Per-dataset fraction of observed points covered by the predictive
        interval.
    mean_absolute_error:
        Mean absolute deviation between predictive mean and observed values.
    metadata:
        Auxiliary metadata such as split name, number of cases, and parameter
        space used during predictive generation.
    """

    observed_values: Mapping[str, np.ndarray]
    predictive_mean: Mapping[str, np.ndarray]
    predictive_std: Mapping[str, np.ndarray]
    interval_lower: Mapping[str, np.ndarray]
    interval_upper: Mapping[str, np.ndarray]
    interval_coverage: Mapping[str, float]
    mean_absolute_error: Mapping[str, float]
    mean_absolute_error_sigma: Mapping[str, float] = field(default_factory=dict)
    median_interval_width_over_uncertainty: Mapping[str, float] = field(default_factory=dict)
    crps: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PredictiveResamplingExhaustedError(RuntimeError):
    def __init__(self, metadata: Mapping[str, Any]):
        self.metadata = dict(metadata)
        super().__init__(
            'Posterior-predictive simulation could not collect the requested '
            'number of valid forward draws: '
            f"requested={self.metadata['requested_valid_forward_model_samples']}, "
            f"collected={self.metadata['n_valid_forward_model_samples']}, "
            f"simulated={self.metadata['n_forward_model_samples']}."
        )


@dataclass(frozen=True)
class LocalSensitivityPointReport:
    """Local linear-identifiability summary around one representative point.

    Attributes
    ----------
    label:
        Short human-readable label for the representative point.
    parameters:
        Physical parameter vector at which the Jacobian was evaluated.
    finite_difference_steps:
        Per-parameter finite-difference step sizes used during Jacobian
        construction.
    finite_difference_schemes:
        Per-parameter scheme labels such as ``'central'`` or ``'forward'``.
    whitened_jacobian:
        Observation Jacobian divided by observational uncertainty, with shape
        ``(n_observation_values, n_parameters)``.
    singular_values:
        Singular values of the whitened Jacobian.
    effective_rank:
        Number of singular values larger than the configured relative cutoff.
    condition_number:
        Condition number inferred from the singular spectrum.
    fisher_matrix:
        Approximate Fisher information matrix ``J^T J``.
    fisher_covariance:
        Damped pseudo-inverse of the Fisher matrix.
    fisher_correlation:
        Correlation matrix derived from the approximate Fisher covariance.
    parameter_sensitivity_norm:
        Per-parameter square-root Fisher diagonal.
    local_sigma:
        Approximate local standard deviation from the Fisher covariance.
    metadata:
        Auxiliary diagnostics such as the ridge term and failed columns.
    """

    label: str
    parameters: np.ndarray
    finite_difference_steps: np.ndarray
    finite_difference_schemes: tuple[str, ...]
    whitened_jacobian: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    condition_number: float
    fisher_matrix: np.ndarray
    fisher_covariance: np.ndarray
    fisher_correlation: np.ndarray
    parameter_sensitivity_norm: np.ndarray
    local_sigma: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalSensitivityReport:
    """Aggregate local information-content diagnostics for one observation.

    Attributes
    ----------
    parameter_names:
        Parameter ordering used throughout the report.
    posterior_mean:
        Posterior mean in physical parameter space.
    posterior_std:
        Posterior standard deviation in physical parameter space.
    posterior_median:
        Posterior median in physical parameter space.
    posterior_iqr:
        Posterior interquartile range in physical parameter space.
    representative_points:
        Local sensitivity summaries evaluated at representative posterior
        points such as the posterior mean and highest-density sample.
    aggregate_local_sigma:
        Median local Fisher sigma across representative points.
    aggregate_parameter_sensitivity_norm:
        Median parameter sensitivity norm across representative points.
    posterior_to_local_sigma_ratio:
        Ratio between posterior standard deviation and local Fisher sigma.
    parameter_diagnostics:
        Per-parameter heuristic summary separating weak data constraints from
        broader-than-local posterior structure.
    metadata:
        Auxiliary metadata such as quantile levels and observation slices.
    """

    parameter_names: tuple[str, ...]
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    posterior_median: np.ndarray
    posterior_iqr: np.ndarray
    representative_points: tuple[LocalSensitivityPointReport, ...]
    aggregate_local_sigma: np.ndarray
    aggregate_parameter_sensitivity_norm: np.ndarray
    posterior_to_local_sigma_ratio: np.ndarray
    parameter_diagnostics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _coerce_sample_array(samples: Any) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)

    if sample_array.ndim == 1:
        return sample_array[:, None]
    if sample_array.ndim == 2:
        return sample_array
    if sample_array.ndim == 3:
        return sample_array

    raise ValueError(
        f"Expected posterior samples with 1, 2, or 3 dimensions, got shape {sample_array.shape}."
    )


def _coerce_truth_array(truths: Any) -> np.ndarray:
    truth_array = np.asarray(truths, dtype=float)

    if truth_array.ndim == 0:
        return truth_array.reshape(1, 1)
    if truth_array.ndim == 1:
        return truth_array[:, None]
    return truth_array


def _selected_case_indices(
    n_cases: int,
    max_cases: int | None,
    seed: int | None = None,
) -> list[int]:
    if max_cases is None:
        return list(range(n_cases))
    n_selected = min(int(max_cases), n_cases)
    if n_selected <= 0:
        return []
    rng = np.random.default_rng(0 if seed is None else seed)
    return sorted(rng.choice(n_cases, size=n_selected, replace=False).tolist())


def _predictive_valid_sample_mask(predictive_batch: Any) -> np.ndarray:
    n_samples = int(getattr(predictive_batch, "n_samples", 0))
    if n_samples <= 0:
        return np.zeros((0,), dtype=bool)

    valid_mask = np.ones((n_samples,), dtype=bool)
    diagnostics = getattr(predictive_batch, "diagnostics", {}) or {}
    invalid_mask = diagnostics.get("invalid_sample_mask")
    if invalid_mask is not None:
        invalid_array = np.asarray(invalid_mask, dtype=bool).reshape(-1)
        if invalid_array.shape[0] == n_samples:
            valid_mask &= ~invalid_array

    for payload in getattr(predictive_batch, "observations", {}).values():
        predictive_values = np.asarray(payload.get("values"), dtype=float)
        if predictive_values.ndim == 0:
            predictive_values = predictive_values.reshape(1, 1)
        if predictive_values.ndim >= 1 and predictive_values.shape[0] == n_samples:
            finite_mask = np.all(
                np.isfinite(predictive_values.reshape(n_samples, -1)),
                axis=1,
            )
            valid_mask &= finite_mask

    return valid_mask


def _coerce_optional_observation_array(
    values: Any,
    target_shape: tuple[int, ...],
) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=float)
    if array.shape == target_shape:
        return array

    target_size = int(np.prod(target_shape))
    flat = array.reshape(-1)
    if flat.size == 0:
        return np.full(target_shape, np.nan, dtype=float)
    if flat.size < target_size:
        flat = np.concatenate(
            [
                flat,
                np.full(target_size - flat.size, np.nan, dtype=float),
            ]
        )
    elif flat.size > target_size:
        flat = flat[:target_size]

    return flat.reshape(target_shape)


def _empirical_crps(
    predictive_values: np.ndarray,
    observed_value: np.ndarray,
) -> np.ndarray:
    sample_array = np.asarray(predictive_values, dtype=float)
    observed_array = np.asarray(observed_value, dtype=float)

    if observed_array.ndim == 0:
        sample_matrix = sample_array.reshape(sample_array.shape[0], 1)
        observed_flat = observed_array.reshape(1)
        observed_shape: tuple[int, ...] = ()
    else:
        sample_matrix = sample_array.reshape(sample_array.shape[0], -1)
        observed_flat = observed_array.reshape(-1)
        observed_shape = observed_array.shape

    n_samples = int(sample_matrix.shape[0])
    if n_samples <= 0:
        return np.full(observed_shape, np.nan, dtype=float)

    first_term = np.mean(np.abs(sample_matrix - observed_flat[None, :]), axis=0)
    sorted_samples = np.sort(sample_matrix, axis=0)
    weights = (
        (2.0 * np.arange(1, n_samples + 1, dtype=float)) - n_samples - 1.0
    )[:, None]
    second_term = np.sum(weights * sorted_samples, axis=0) / float(n_samples * n_samples)
    crps = first_term - second_term
    return crps.reshape(observed_shape)


def _summarize_predictive_observation(
    predictive_values: np.ndarray,
    observed_value: np.ndarray,
    observed_uncertainty: np.ndarray | None,
    *,
    interval_level: float,
) -> dict[str, Any]:
    lower, upper = central_credible_interval(predictive_values, level=interval_level)
    predictive_mean = np.mean(predictive_values, axis=0)
    predictive_std = np.std(predictive_values, axis=0)

    absolute_error = np.abs(np.asarray(predictive_mean, dtype=float) - observed_value)
    interval_width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)

    mean_absolute_error_sigma = float('nan')
    median_interval_width_over_uncertainty = float('nan')
    if observed_uncertainty is not None:
        uncertainty_flat = np.asarray(observed_uncertainty, dtype=float).reshape(-1)
        safe_uncertainty = np.isfinite(uncertainty_flat) & (uncertainty_flat > 0.0)
        if np.any(safe_uncertainty):
            absolute_error_flat = np.asarray(absolute_error, dtype=float).reshape(-1)
            interval_width_flat = np.asarray(interval_width, dtype=float).reshape(-1)
            mean_absolute_error_sigma = float(
                np.mean(absolute_error_flat[safe_uncertainty] / uncertainty_flat[safe_uncertainty])
            )
            median_interval_width_over_uncertainty = float(
                np.median(interval_width_flat[safe_uncertainty] / uncertainty_flat[safe_uncertainty])
            )

    crps_value = float(
        np.nanmean(np.asarray(_empirical_crps(predictive_values, observed_value), dtype=float))
    )

    return {
        'predictive_mean': predictive_mean,
        'predictive_std': predictive_std,
        'interval_lower': lower,
        'interval_upper': upper,
        'interval_coverage': float(
            np.mean((observed_value >= lower) & (observed_value <= upper))
        ),
        'mean_absolute_error': float(np.mean(absolute_error)),
        'mean_absolute_error_sigma': mean_absolute_error_sigma,
        'median_interval_width_over_uncertainty': median_interval_width_over_uncertainty,
        'crps': crps_value,
    }


def _collect_valid_predictive_draws(
    simulator: RuntimeSimulator,
    physical_parameters: np.ndarray,
    *,
    requested_valid_samples: int | None,
    seed: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parameter_array = np.asarray(physical_parameters, dtype=float)
    if parameter_array.ndim != 2 or parameter_array.shape[0] <= 0:
        raise ValueError(
            'Posterior-predictive simulation requires at least one parameter draw.'
        )

    desired_valid_samples = (
        int(parameter_array.shape[0])
        if requested_valid_samples is None
        else max(int(requested_valid_samples), 1)
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(parameter_array.shape[0])
    cursor = 0
    total_forward_draws = 0
    total_invalid_forward_draws = 0
    total_valid_forward_draws = 0
    discarded_valid_forward_draws = 0
    collected_valid_draws = 0
    rounds_used = 0
    max_attempted_forward_draws = max(
        desired_valid_samples * 16,
        desired_valid_samples + max(parameter_array.shape[0], 64),
    )
    minimum_accepted_valid_samples = max(
        desired_valid_samples - (1 if desired_valid_samples >= 128 else 0),
        1,
    )
    collected_values: dict[str, list[np.ndarray]] = {}
    tolerated_shortfall_rounds = 0

    while (
        collected_valid_draws < desired_valid_samples
        and total_forward_draws < max_attempted_forward_draws
    ):
        remaining = desired_valid_samples - collected_valid_draws
        if total_forward_draws > 0:
            estimated_valid_fraction = np.clip(
                total_valid_forward_draws / max(total_forward_draws, 1),
                0.05,
                1.0,
            )
            batch_size = int(np.ceil(remaining / estimated_valid_fraction))
        else:
            batch_size = remaining
        batch_size = max(batch_size, remaining, 1)
        batch_size = min(
            batch_size,
            max_attempted_forward_draws - total_forward_draws,
        )
        if batch_size <= 0:
            break
        if cursor < parameter_array.shape[0]:
            take = min(batch_size, parameter_array.shape[0] - cursor)
            batch_parameters = parameter_array[permutation[cursor: cursor + take]]
            cursor += take
        else:
            batch_parameters = parameter_array[
                rng.choice(parameter_array.shape[0], size=batch_size, replace=True)
            ]

        predictive_batch = simulator.simulate_from_parameters(batch_parameters)
        valid_mask = _predictive_valid_sample_mask(predictive_batch)
        n_total_predictive = int(valid_mask.shape[0])
        valid_indices = np.flatnonzero(valid_mask)
        total_forward_draws += n_total_predictive
        total_invalid_forward_draws += max(
            n_total_predictive - int(valid_indices.size),
            0,
        )
        total_valid_forward_draws += int(valid_indices.size)

        if valid_indices.size > remaining:
            discarded_valid_forward_draws += int(valid_indices.size - remaining)
            valid_indices = valid_indices[:remaining]

        if valid_indices.size > 0:
            for observation_name, payload in predictive_batch.observations.items():
                predictive_values = np.asarray(payload.get('values'), dtype=float)
                if predictive_values.ndim == 0:
                    predictive_values = predictive_values.reshape(1, 1)
                if (
                    predictive_values.ndim >= 1
                    and predictive_values.shape[0] == n_total_predictive
                ):
                    collected_values.setdefault(observation_name, []).append(
                        predictive_values[valid_indices]
                    )
            collected_valid_draws += int(valid_indices.size)

        rounds_used += 1
        if collected_valid_draws >= desired_valid_samples:
            break
        if collected_valid_draws >= minimum_accepted_valid_samples:
            tolerated_shortfall_rounds += 1
            if tolerated_shortfall_rounds >= 2:
                break
        else:
            tolerated_shortfall_rounds = 0

    predictive_metadata = {
        'requested_valid_forward_model_samples': desired_valid_samples,
        'n_forward_model_samples': total_forward_draws,
        'n_valid_forward_model_samples': collected_valid_draws,
        'invalid_forward_model_sample_count': total_invalid_forward_draws,
        'invalid_forward_model_sample_fraction': float(
            total_invalid_forward_draws / max(total_forward_draws, 1)
        ),
        'discarded_valid_forward_model_sample_count': discarded_valid_forward_draws,
        'simulated_valid_forward_model_sample_count': total_valid_forward_draws,
        'valid_forward_model_resampling_rounds': rounds_used,
        'max_attempted_forward_model_samples': int(max_attempted_forward_draws),
        'accepted_valid_forward_model_shortfall_count': int(
            max(desired_valid_samples - collected_valid_draws, 0)
        ),
    }

    if collected_valid_draws < minimum_accepted_valid_samples:
        raise PredictiveResamplingExhaustedError(predictive_metadata)

    if not collected_values:
        raise ValueError(
            'Posterior-predictive simulation produced no valid observation payloads.'
        )

    return (
        {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in collected_values.items()
        },
        predictive_metadata,
    )


def _posterior_space_to_physical(
    task: SBITask, samples: Any, parameter_space: str
) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=float)

    if parameter_space == "physical":
        return sample_array
    if parameter_space == "cube":
        return np.asarray(
            task.parameter_layout.transform_cube_to_physical_batch(sample_array),
            dtype=float,
        )
    if parameter_space == "unconstrained":
        return np.asarray(
            task.parameter_layout.transform_unconstrained_batch(sample_array),
            dtype=float,
        )

    raise ValueError(f"Unsupported parameter_space: {parameter_space!r}")


def _flatten_observation_blocks(
    observation_blocks: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...], float]:
    observed_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []
    observation_slices: list[dict[str, Any]] = []
    offset = 0

    for block in observation_blocks:
        block_values = np.asarray(block.values, dtype=float).reshape(-1)
        if block.uncertainties is None:
            block_uncertainties = np.full(block_values.shape, np.nan, dtype=float)
        else:
            block_uncertainties = np.asarray(block.uncertainties, dtype=float)
            if block_uncertainties.ndim == 0:
                block_uncertainties = np.full(
                    block_values.shape,
                    float(block_uncertainties),
                    dtype=float,
                )
            else:
                block_uncertainties = np.asarray(
                    np.broadcast_to(block_uncertainties.reshape(-1), block_values.shape),
                    dtype=float,
                )

        observed_parts.append(block_values)
        uncertainty_parts.append(np.abs(block_uncertainties))
        observation_slices.append(
            {
                "name": str(block.name),
                "start": offset,
                "stop": offset + int(block_values.size),
                "size": int(block_values.size),
            }
        )
        offset += int(block_values.size)

    if not observed_parts:
        raise ValueError("At least one observation block is required for local sensitivity diagnostics.")

    observed_vector = np.concatenate(observed_parts).astype(float, copy=False)
    uncertainty_vector = np.concatenate(uncertainty_parts).astype(float, copy=False)
    positive_uncertainties = uncertainty_vector[
        np.isfinite(uncertainty_vector) & (uncertainty_vector > 0.0)
    ]
    uncertainty_floor = float(np.nanmedian(positive_uncertainties) * 1e-3) if positive_uncertainties.size else 1e-8
    uncertainty_floor = max(uncertainty_floor, 1e-8)
    uncertainty_vector = np.where(
        np.isfinite(uncertainty_vector) & (uncertainty_vector > uncertainty_floor),
        uncertainty_vector,
        uncertainty_floor,
    )
    return observed_vector, uncertainty_vector, tuple(observation_slices), uncertainty_floor


def _flatten_simulated_observations(
    observation_blocks: Sequence[Any],
    simulated_observations: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    simulated_parts: list[np.ndarray] = []
    for block in observation_blocks:
        if block.name not in simulated_observations:
            raise KeyError(
                f"Simulated observations do not contain required dataset {block.name!r}."
            )
        simulated_parts.append(
            np.asarray(simulated_observations[block.name]["values"], dtype=float).reshape(-1)
        )

    simulated_vector = np.concatenate(simulated_parts).astype(float, copy=False)
    if not np.all(np.isfinite(simulated_vector)):
        raise ValueError("Simulated observation vector contains non-finite values.")
    return simulated_vector


def _simulate_deterministic_observation_vector(
    simulator: RuntimeSimulator,
    parameters: np.ndarray,
    observation_blocks: Sequence[Any],
) -> np.ndarray:
    physical_params = simulator.runtime.parameter_layout.physical_params_from_vector(parameters)
    simulated_observations = simulator.runtime.simulate_observations_scalar(
        physical_params=physical_params
    )
    return _flatten_simulated_observations(observation_blocks, simulated_observations)


def _select_representative_posterior_points(
    samples: np.ndarray,
    posterior_log_probabilities: Any = None,
    quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, np.ndarray], ...]]:
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 2:
        raise ValueError(
            f"Expected posterior samples with shape (n_samples, n_dim), got {sample_array.shape}."
        )

    posterior_mean = np.mean(sample_array, axis=0)
    posterior_std = np.std(sample_array, axis=0)
    posterior_median = np.median(sample_array, axis=0)
    posterior_iqr = np.subtract(
        np.quantile(sample_array, 0.75, axis=0),
        np.quantile(sample_array, 0.25, axis=0),
    )
    representatives: list[tuple[str, np.ndarray]] = [("posterior_mean", posterior_mean)]

    used_sample_indices: set[int] = set()
    if posterior_log_probabilities is not None:
        log_probabilities = np.asarray(posterior_log_probabilities, dtype=float).reshape(-1)
        finite_mask = np.isfinite(log_probabilities)
        if finite_mask.any():
            best_index = int(np.argmax(np.where(finite_mask, log_probabilities, -np.inf)))
            representatives.append(
                ("highest_log_probability_sample", sample_array[best_index].copy())
            )
            used_sample_indices.add(best_index)

    distance_scale = np.maximum(posterior_std, 1e-6)
    for quantile_level in quantile_levels:
        target = np.quantile(sample_array, float(quantile_level), axis=0)
        scaled_distances = np.linalg.norm((sample_array - target) / distance_scale, axis=1)
        for candidate_index in np.argsort(scaled_distances):
            if int(candidate_index) in used_sample_indices:
                continue
            representatives.append(
                (
                    f"quantile_{int(round(float(quantile_level) * 100.0)):02d}",
                    sample_array[int(candidate_index)].copy(),
                )
            )
            used_sample_indices.add(int(candidate_index))
            break

    return (
        posterior_mean,
        posterior_std,
        posterior_median,
        posterior_iqr,
        tuple(representatives),
    )


def _robust_finite_difference_column(
    simulator: RuntimeSimulator,
    center_parameters: np.ndarray,
    center_observation_vector: np.ndarray,
    observation_blocks: Sequence[Any],
    parameter_index: int,
    base_step: float,
    max_step_reduction_attempts: int,
) -> tuple[np.ndarray, float, str]:
    for reduction_attempt in range(max_step_reduction_attempts + 1):
        step = float(base_step) * (0.5 ** reduction_attempt)
        if not np.isfinite(step) or step <= 0.0:
            continue

        plus_parameters = np.array(center_parameters, dtype=float, copy=True)
        minus_parameters = np.array(center_parameters, dtype=float, copy=True)
        plus_parameters[parameter_index] += step
        minus_parameters[parameter_index] -= step

        plus_vector: np.ndarray | None = None
        minus_vector: np.ndarray | None = None

        try:
            plus_vector = _simulate_deterministic_observation_vector(
                simulator, plus_parameters, observation_blocks
            )
        except Exception:
            plus_vector = None

        try:
            minus_vector = _simulate_deterministic_observation_vector(
                simulator, minus_parameters, observation_blocks
            )
        except Exception:
            minus_vector = None

        if plus_vector is not None and minus_vector is not None:
            return (plus_vector - minus_vector) / (2.0 * step), step, "central"
        if plus_vector is not None:
            return (plus_vector - center_observation_vector) / step, step, "forward"
        if minus_vector is not None:
            return (center_observation_vector - minus_vector) / step, step, "backward"

    return np.zeros_like(center_observation_vector), float("nan"), "failed"


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance_array = np.asarray(covariance, dtype=float)
    standard_deviations = np.sqrt(np.clip(np.diag(covariance_array), 0.0, None))
    outer = np.outer(standard_deviations, standard_deviations)
    correlation = np.full_like(covariance_array, np.nan, dtype=float)
    valid = outer > 0.0
    correlation[valid] = covariance_array[valid] / outer[valid]
    np.fill_diagonal(correlation, 1.0)
    return correlation


def generate_local_sensitivity_report(
    task: SBITask,
    posterior_samples: Any,
    observation_blocks: Sequence[Any],
    posterior_log_probabilities: Any = None,
    parameter_space: str = "physical",
    simulator: RuntimeSimulator | None = None,
    quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
    finite_difference_relative_step: float = 1e-3,
    finite_difference_std_fraction: float = 0.1,
    finite_difference_absolute_floor: float = 1e-5,
    max_step_reduction_attempts: int = 6,
    svd_relative_tolerance: float = 1e-3,
    posterior_underexploited_ratio_threshold: float = 1.5,
    weak_sensitivity_fraction_threshold: float = 0.15,
    seed: int | None = None,
) -> LocalSensitivityReport:
    """Diagnose local physical identifiability around representative posterior points.

    The report evaluates a deterministic simulator Jacobian at representative
    posterior points, whitens it by observational uncertainty, and derives a
    Fisher-style local covariance approximation for each point.
    """

    sample_array = _coerce_sample_array(posterior_samples)
    if sample_array.ndim != 2:
        raise ValueError(
            "generate_local_sensitivity_report expects samples for one posterior, not a batch."
        )

    physical_samples = _posterior_space_to_physical(task, sample_array, parameter_space)
    (
        posterior_mean,
        posterior_std,
        posterior_median,
        posterior_iqr,
        representative_points,
    ) = _select_representative_posterior_points(
        physical_samples,
        posterior_log_probabilities=posterior_log_probabilities,
        quantile_levels=quantile_levels,
    )

    _, uncertainty_vector, observation_slices, uncertainty_floor = _flatten_observation_blocks(
        observation_blocks
    )
    resolved_simulator = simulator or RuntimeSimulator(
        task=task,
        seed=seed,
        data_parallel=False,
    )

    point_reports: list[LocalSensitivityPointReport] = []
    for label, representative_parameters in representative_points:
        center_parameters = np.asarray(representative_parameters, dtype=float)
        center_observation_vector = _simulate_deterministic_observation_vector(
            resolved_simulator,
            center_parameters,
            observation_blocks,
        )

        jacobian_columns: list[np.ndarray] = []
        finite_difference_steps: list[float] = []
        finite_difference_schemes: list[str] = []
        failed_columns = 0

        for parameter_index in range(center_parameters.shape[0]):
            base_step = max(
                float(finite_difference_absolute_floor),
                abs(float(center_parameters[parameter_index])) * float(finite_difference_relative_step),
                float(posterior_std[parameter_index]) * float(finite_difference_std_fraction),
            )
            column, actual_step, scheme = _robust_finite_difference_column(
                resolved_simulator,
                center_parameters,
                center_observation_vector,
                observation_blocks,
                parameter_index,
                base_step,
                max_step_reduction_attempts,
            )
            jacobian_columns.append(column)
            finite_difference_steps.append(actual_step)
            finite_difference_schemes.append(scheme)
            if scheme == "failed":
                failed_columns += 1

        jacobian = np.column_stack(jacobian_columns)
        whitened_jacobian = jacobian / uncertainty_vector[:, None]
        singular_values = np.linalg.svd(whitened_jacobian, compute_uv=False)
        if singular_values.size and np.isfinite(singular_values[0]) and singular_values[0] > 0.0:
            effective_rank = int(
                np.sum(singular_values > (float(singular_values[0]) * float(svd_relative_tolerance)))
            )
            smallest_singular_value = float(singular_values[-1])
            if smallest_singular_value > 0.0:
                condition_number = float(singular_values[0] / smallest_singular_value)
            else:
                condition_number = float("inf")
        else:
            effective_rank = 0
            condition_number = float("inf")

        fisher_matrix = whitened_jacobian.T @ whitened_jacobian
        fisher_scale = float(np.nanmax(np.diag(fisher_matrix))) if fisher_matrix.size else 0.0
        ridge = max(fisher_scale * 1e-8, 1e-10)
        fisher_covariance = np.linalg.pinv(
            fisher_matrix + (ridge * np.eye(fisher_matrix.shape[0], dtype=float))
        )
        fisher_correlation = _covariance_to_correlation(fisher_covariance)
        parameter_sensitivity_norm = np.sqrt(np.clip(np.diag(fisher_matrix), 0.0, None))
        local_sigma = np.sqrt(np.clip(np.diag(fisher_covariance), 0.0, None))

        point_reports.append(
            LocalSensitivityPointReport(
                label=label,
                parameters=center_parameters,
                finite_difference_steps=np.asarray(finite_difference_steps, dtype=float),
                finite_difference_schemes=tuple(finite_difference_schemes),
                whitened_jacobian=np.asarray(whitened_jacobian, dtype=float),
                singular_values=np.asarray(singular_values, dtype=float),
                effective_rank=effective_rank,
                condition_number=condition_number,
                fisher_matrix=np.asarray(fisher_matrix, dtype=float),
                fisher_covariance=np.asarray(fisher_covariance, dtype=float),
                fisher_correlation=np.asarray(fisher_correlation, dtype=float),
                parameter_sensitivity_norm=np.asarray(parameter_sensitivity_norm, dtype=float),
                local_sigma=np.asarray(local_sigma, dtype=float),
                metadata={
                    "ridge": ridge,
                    "failed_columns": failed_columns,
                    "whitened_observation_size": int(whitened_jacobian.shape[0]),
                },
            )
        )

    local_sigma_stack = np.stack([point.local_sigma for point in point_reports], axis=0)
    parameter_sensitivity_stack = np.stack(
        [point.parameter_sensitivity_norm for point in point_reports],
        axis=0,
    )
    aggregate_local_sigma = np.nanmedian(local_sigma_stack, axis=0)
    aggregate_parameter_sensitivity_norm = np.nanmedian(
        parameter_sensitivity_stack,
        axis=0,
    )
    posterior_to_local_sigma_ratio = np.divide(
        posterior_std,
        np.maximum(aggregate_local_sigma, 1e-12),
    )
    max_parameter_sensitivity = float(
        np.nanmax(aggregate_parameter_sensitivity_norm)
    ) if np.any(np.isfinite(aggregate_parameter_sensitivity_norm)) else 1.0
    normalized_parameter_sensitivity = aggregate_parameter_sensitivity_norm / max(
        max_parameter_sensitivity,
        1e-12,
    )

    parameter_diagnostics: dict[str, dict[str, Any]] = {}
    for parameter_index, parameter_name in enumerate(task.parameter_names):
        normalized_sensitivity = float(normalized_parameter_sensitivity[parameter_index])
        ratio = float(posterior_to_local_sigma_ratio[parameter_index])
        local_sigma_value = float(aggregate_local_sigma[parameter_index])
        posterior_std_value = float(posterior_std[parameter_index])

        if (not np.isfinite(normalized_sensitivity)) or (
            normalized_sensitivity <= float(weak_sensitivity_fraction_threshold)
        ):
            status = "weakly_identified_by_data"
        elif np.isfinite(ratio) and ratio >= float(
            posterior_underexploited_ratio_threshold
        ):
            status = "posterior_broader_than_local_information"
        else:
            status = "consistent_with_local_information"

        parameter_diagnostics[str(parameter_name)] = {
            "status": status,
            "posterior_std": posterior_std_value,
            "aggregate_local_sigma": local_sigma_value,
            "posterior_to_local_sigma_ratio": ratio,
            "normalized_sensitivity": normalized_sensitivity,
        }

    return LocalSensitivityReport(
        parameter_names=tuple(task.parameter_names),
        posterior_mean=np.asarray(posterior_mean, dtype=float),
        posterior_std=np.asarray(posterior_std, dtype=float),
        posterior_median=np.asarray(posterior_median, dtype=float),
        posterior_iqr=np.asarray(posterior_iqr, dtype=float),
        representative_points=tuple(point_reports),
        aggregate_local_sigma=np.asarray(aggregate_local_sigma, dtype=float),
        aggregate_parameter_sensitivity_norm=np.asarray(
            aggregate_parameter_sensitivity_norm,
            dtype=float,
        ),
        posterior_to_local_sigma_ratio=np.asarray(
            posterior_to_local_sigma_ratio,
            dtype=float,
        ),
        parameter_diagnostics=parameter_diagnostics,
        metadata={
            "task_name": task.name,
            "parameter_space": parameter_space,
            "quantile_levels": [float(level) for level in quantile_levels],
            "observation_slices": list(observation_slices),
            "uncertainty_floor": uncertainty_floor,
            "finite_difference_relative_step": float(finite_difference_relative_step),
            "finite_difference_std_fraction": float(finite_difference_std_fraction),
            "finite_difference_absolute_floor": float(finite_difference_absolute_floor),
            "max_step_reduction_attempts": int(max_step_reduction_attempts),
            "svd_relative_tolerance": float(svd_relative_tolerance),
            "posterior_underexploited_ratio_threshold": float(
                posterior_underexploited_ratio_threshold
            ),
            "weak_sensitivity_fraction_threshold": float(
                weak_sensitivity_fraction_threshold
            ),
        },
    )


def local_sensitivity_report_to_payload(
    report: LocalSensitivityReport,
) -> dict[str, Any]:
    """Convert a local sensitivity report into a JSON-serializable payload."""

    return {
        "parameter_names": list(report.parameter_names),
        "posterior_mean": np.asarray(report.posterior_mean, dtype=float).tolist(),
        "posterior_std": np.asarray(report.posterior_std, dtype=float).tolist(),
        "posterior_median": np.asarray(report.posterior_median, dtype=float).tolist(),
        "posterior_iqr": np.asarray(report.posterior_iqr, dtype=float).tolist(),
        "aggregate_local_sigma": np.asarray(
            report.aggregate_local_sigma,
            dtype=float,
        ).tolist(),
        "aggregate_parameter_sensitivity_norm": np.asarray(
            report.aggregate_parameter_sensitivity_norm,
            dtype=float,
        ).tolist(),
        "posterior_to_local_sigma_ratio": np.asarray(
            report.posterior_to_local_sigma_ratio,
            dtype=float,
        ).tolist(),
        "parameter_diagnostics": {
            str(name): dict(values)
            for name, values in report.parameter_diagnostics.items()
        },
        "representative_points": [
            {
                "label": point.label,
                "parameters": np.asarray(point.parameters, dtype=float).tolist(),
                "finite_difference_steps": np.asarray(
                    point.finite_difference_steps,
                    dtype=float,
                ).tolist(),
                "finite_difference_schemes": list(point.finite_difference_schemes),
                "whitened_jacobian": np.asarray(
                    point.whitened_jacobian,
                    dtype=float,
                ).tolist(),
                "singular_values": np.asarray(
                    point.singular_values,
                    dtype=float,
                ).tolist(),
                "effective_rank": int(point.effective_rank),
                "condition_number": float(point.condition_number),
                "fisher_matrix": np.asarray(point.fisher_matrix, dtype=float).tolist(),
                "fisher_covariance": np.asarray(
                    point.fisher_covariance,
                    dtype=float,
                ).tolist(),
                "fisher_correlation": np.asarray(
                    point.fisher_correlation,
                    dtype=float,
                ).tolist(),
                "parameter_sensitivity_norm": np.asarray(
                    point.parameter_sensitivity_norm,
                    dtype=float,
                ).tolist(),
                "local_sigma": np.asarray(point.local_sigma, dtype=float).tolist(),
                "metadata": dict(point.metadata),
            }
            for point in report.representative_points
        ],
        "metadata": dict(report.metadata),
    }


def central_credible_interval(
    samples: Any, level: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a central credible interval from posterior samples.

    Parameters
    ----------
    samples:
        Posterior samples for one observation, with shape ``(n_samples,)`` or
        ``(n_samples, n_dim)``.
    level:
        Nominal central interval level between 0 and 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper quantile bounds for each inferred parameter dimension.
        The returned arrays are one-dimensional even for scalar posteriors.
    """

    sample_array = _coerce_sample_array(samples)
    if sample_array.ndim == 3:
        raise ValueError(
            "central_credible_interval expects samples for one posterior, not a batch of posteriors."
        )

    alpha = max(0.0, min(1.0, 1.0 - float(level))) / 2.0
    lower = np.quantile(sample_array, alpha, axis=0)
    upper = np.quantile(sample_array, 1.0 - alpha, axis=0)
    return lower, upper


def empirical_interval_coverage(samples: Any, truths: Any, level: float = 0.8) -> float:
    """Measure empirical central-interval coverage over a batch of posteriors.

    Parameters
    ----------
    samples:
        Posterior draws for one or more held-out cases. Expected shape is
        ``(n_cases, n_draws, n_dim)`` or a shape that can be promoted to it.
    truths:
        Ground-truth parameters matched to each held-out case.
    level:
        Nominal central interval level between 0 and 1.

    Returns
    -------
    float
        Fraction of truth values falling inside the requested central interval.
    """

    sample_array = np.asarray(samples, dtype=float)
    truth_array = _coerce_truth_array(truths)

    if sample_array.ndim == 2:
        sample_array = sample_array[:, :, None]
    if sample_array.ndim != 3:
        raise ValueError(
            f"empirical_interval_coverage expects a 2D or 3D sample array, got {sample_array.shape}."
        )

    alpha = max(0.0, min(1.0, 1.0 - float(level))) / 2.0
    lower = np.quantile(sample_array, alpha, axis=1)
    upper = np.quantile(sample_array, 1.0 - alpha, axis=1)
    covered = (truth_array >= lower) & (truth_array <= upper)
    return float(np.mean(covered))


def simulation_based_calibration_report(
    samples: Any,
    truths: Any,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    metadata: Mapping[str, Any] | None = None,
) -> SimulationBasedCalibrationReport:
    """Build an SBC report from posterior draws and matched ground truths.

    Parameters
    ----------
    samples:
        Posterior draws for held-out cases with shape ``(n_cases, n_draws,
        n_dim)`` or a promotable equivalent.
    truths:
        Ground-truth parameters paired with each held-out case.
    levels:
        Coverage levels to summarize alongside the rank statistics.
    metadata:
        Optional metadata attached to the returned report.

    Returns
    -------
    SimulationBasedCalibrationReport
        SBC summary containing ranks, histograms, posterior means, and coverage
        summaries.
    """

    sample_array = np.asarray(samples, dtype=float)
    truth_array = _coerce_truth_array(truths)

    if sample_array.ndim == 2:
        sample_array = sample_array[:, :, None]
    if sample_array.ndim != 3:
        raise ValueError(
            f"simulation_based_calibration_report expects a 3D sample array, got {sample_array.shape}."
        )
    if sample_array.shape[0] != truth_array.shape[0]:
        raise ValueError(
            "Samples and truths must agree on the number of held-out observations."
        )

    n_draws = int(sample_array.shape[1])
    ranks = np.sum(sample_array < truth_array[:, None, :], axis=1, dtype=int)
    rank_histogram_counts = np.stack(
        [
            np.bincount(ranks[:, dimension], minlength=n_draws + 1)
            for dimension in range(ranks.shape[1])
        ],
        axis=0,
    )
    posterior_means = np.mean(sample_array, axis=1)
    mean_rank = np.mean(ranks, axis=0)
    expected_mean_rank = n_draws / 2.0
    normalized_mean_rank_error = np.abs(mean_rank - expected_mean_rank) / max(
        expected_mean_rank, 1.0
    )
    coverages = tuple(
        CoverageLevelReport(
            level=float(level),
            empirical_coverage=empirical_interval_coverage(
                sample_array, truth_array, level=level
            ),
            absolute_error=abs(
                empirical_interval_coverage(sample_array, truth_array, level=level)
                - float(level)
            ),
        )
        for level in levels
    )

    return SimulationBasedCalibrationReport(
        ranks=ranks,
        rank_histogram_counts=rank_histogram_counts,
        posterior_means=posterior_means,
        truths=truth_array,
        coverages=coverages,
        mean_rank=mean_rank,
        normalized_mean_rank_error=normalized_mean_rank_error,
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class PosteriorInformativenessReport:
    """Joint contraction + calibration verdict per parameter.

    ``contraction`` is the ratio of the posterior 68% width to the prior 68%
    width. On its own it is *misleading*: a posterior that has collapsed to a
    narrow but wrong region has a tiny contraction yet is useless. Combining it
    with the SBC normalized mean-rank error separates the three regimes.
    """

    parameter_names: list[str]
    contraction: np.ndarray
    sbc_normalized_rank_error: np.ndarray | None
    verdict: list[str]


def classify_posterior_informativeness(
    contraction: Any,
    sbc_normalized_rank_error: Any | None = None,
    parameter_names: Sequence[str] | None = None,
    *,
    contraction_informative: float = 0.9,
    contraction_overconfident: float = 0.1,
    sbc_miscalibrated: float = 0.3,
) -> PosteriorInformativenessReport:
    """Classify each parameter by contraction *and* SBC calibration.

    Verdicts:

    * ``'uninformative'`` -- ``contraction >= contraction_informative``: the
      marginal is effectively the prior.
    * ``'overconfident'`` -- ``contraction <= contraction_overconfident`` **and**
      the SBC normalized mean-rank error ``>= sbc_miscalibrated``: the posterior
      collapsed to a narrow, miscalibrated region (confidently wrong). This is
      exactly the failure a bare contraction metric mislabels as "constrained".
    * ``'constrained'`` -- informative and (no SBC supplied, or SBC acceptable).
    * ``'unknown'`` -- contraction not finite.

    ``sbc_normalized_rank_error`` is optional; without it the function can only
    distinguish informative from uninformative (an informative marginal is
    reported as ``'constrained'`` but cannot be confirmed calibrated).
    """
    contraction_array = np.asarray(contraction, dtype=float)
    sbc_array = (
        None
        if sbc_normalized_rank_error is None
        else np.asarray(sbc_normalized_rank_error, dtype=float)
    )
    n = int(contraction_array.shape[0])
    if parameter_names is None:
        parameter_names = [f"param_{index}" for index in range(n)]
    parameter_names = list(parameter_names)

    verdict: list[str] = []
    for index in range(n):
        value = float(contraction_array[index])
        sbc_error = (
            None
            if sbc_array is None or index >= sbc_array.shape[0]
            else float(sbc_array[index])
        )
        if not np.isfinite(value):
            verdict.append("unknown")
        elif value >= contraction_informative:
            verdict.append("uninformative")
        elif (
            value <= contraction_overconfident
            and sbc_error is not None
            and np.isfinite(sbc_error)
            and sbc_error >= sbc_miscalibrated
        ):
            verdict.append("overconfident")
        else:
            verdict.append("constrained")

    return PosteriorInformativenessReport(
        parameter_names=parameter_names,
        contraction=contraction_array,
        sbc_normalized_rank_error=sbc_array,
        verdict=verdict,
    )


def generate_sbc_report(
    posterior: Any,
    dataset_reader: NormalizedObservationDatasetReader,
    split: DatasetSplit = DatasetSplit.TEST,
    n_posterior_samples: int = 256,
    batch_size: int = 32,
    parameter_space: str | None = None,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
    max_cases: int | None = None,
    seed: int | None = None,
    data_parallel: bool | None = None,
) -> SimulationBasedCalibrationReport:
    """Run SBC over a dataset reader using normalized observation batches.

    Parameters
    ----------
    posterior:
        Trained posterior estimator exposing ``encode_observation`` and
        ``sample_posterior``.
    dataset_reader:
        Reader yielding normalized held-out observations and matched parameter
        values.
    split:
        Dataset split used for the SBC evaluation.
    n_posterior_samples:
        Number of posterior draws generated per held-out case.
    batch_size:
        Reader batch size used during report generation.
    parameter_space:
        Optional parameter space override. Defaults to the posterior's own
        configured parameter space.
    levels:
        Coverage levels summarized in the returned report.
    max_cases:
        Optional cap on the number of held-out observations evaluated.
    seed:
        Optional base seed used to generate reproducible posterior draws.

    Returns
    -------
    SimulationBasedCalibrationReport
        SBC summary computed from the requested dataset split.
    """

    resolved_parameter_space = parameter_space or getattr(
        posterior, "parameter_space", "physical"
    )
    _data_parallel = len(jax.devices()) > 1 if data_parallel is None else data_parallel
    _devices = jax.devices()
    _n_devices = len(_devices)
    _mesh = (
        jax.sharding.Mesh(np.array(_devices), axis_names=('batch',))
        if _data_parallel and _n_devices > 1
        else None
    )
    posterior_draws = []
    truths = []
    base_seed = 0 if seed is None else int(seed)
    case_index = 0
    has_batch_encode = hasattr(posterior, "batch_encode_observation")
    has_batch_sample = hasattr(posterior, "batch_sample_posterior")

    for batch in dataset_reader.iter_batches(
        split=split,
        batch_size=batch_size,
        shuffle=False,
        parameter_space=resolved_parameter_space,
    ):
        if max_cases is not None and case_index >= int(max_cases):
            break

        # Encode all observations in the batch with one vmapped forward pass
        # when the posterior supports it; otherwise fall back to serial encoding.
        if has_batch_encode:
            encoded_batch = posterior.batch_encode_observation(batch.observations)
        else:
            encoded_batch = [
                posterior.encode_observation(obs) for obs in batch.observations
            ]

        n_in_batch = len(encoded_batch)
        if max_cases is not None:
            n_in_batch = min(n_in_batch, int(max_cases) - case_index)

        if has_batch_sample and n_in_batch > 0:
            # One JIT dispatch for the entire batch instead of one per observation.
            stacked_embeddings = jnp.stack(
                [encoded_batch[i].embedding for i in range(n_in_batch)]
            )
            # Optionally shard embeddings across devices for parallel sampling
            if _mesh is not None:
                n_emb = stacked_embeddings.shape[0]
                remainder = n_emb % _n_devices
                if remainder != 0:
                    pad_count = _n_devices - remainder
                    padding = jnp.repeat(stacked_embeddings[-1:], pad_count, axis=0)
                    stacked_embeddings = jnp.concatenate([stacked_embeddings, padding], axis=0)
                from jax.sharding import NamedSharding, PartitionSpec as P
                stacked_embeddings = jax.device_put(
                    stacked_embeddings, NamedSharding(_mesh, P('batch'))
                )
            batch_draws = posterior.batch_sample_posterior(
                stacked_embeddings,
                n_samples=n_posterior_samples,
                base_seed=base_seed + case_index,
            )
            for i in range(n_in_batch):
                posterior_draws.append(np.asarray(batch_draws[i], dtype=float))
                truths.append(np.asarray(batch.parameters[i], dtype=float))
            case_index += n_in_batch
        else:
            for sample_offset, encoded in enumerate(encoded_batch[:n_in_batch]):
                draws = posterior.sample_posterior(
                    encoded,
                    n_samples=n_posterior_samples,
                    seed=base_seed + case_index,
                ).samples
                posterior_draws.append(np.asarray(draws, dtype=float))
                truths.append(np.asarray(batch.parameters[sample_offset], dtype=float))
                case_index += 1

    if not posterior_draws:
        raise ValueError(
            f"Split '{split.value}' did not yield any observations for SBC evaluation."
        )

    return simulation_based_calibration_report(
        np.asarray(posterior_draws, dtype=float),
        np.asarray(truths, dtype=float),
        levels=levels,
        metadata={
            "split": split.value,
            "n_cases": len(posterior_draws),
            "n_posterior_samples": int(n_posterior_samples),
            "parameter_space": resolved_parameter_space,
            "task_name": dataset_reader.dataset.manifest.task_name,
        },
    )


def posterior_predictive_report_from_samples(
    task: SBITask,
    posterior_samples: Any,
    observation_blocks: Sequence[Any],
    parameter_space: str = "physical",
    interval_level: float = 0.9,
    simulator: RuntimeSimulator | None = None,
    seed: int | None = None,
    n_predictive_forward_model_samples: int | None = None,
) -> PosteriorPredictiveReport:
    """Summarize posterior-predictive draws for one observed system.

    Parameters
    ----------
    task:
        SBI task used to map posterior samples back into physical parameter
        space and construct the simulator.
    posterior_samples:
        Posterior draws in the specified parameter space.
    observation_blocks:
        Original user-facing observation blocks. Reported observations remain on
        this raw scale.
    parameter_space:
        Coordinate system in which ``posterior_samples`` are expressed.
    interval_level:
        Central predictive interval level used for reporting.
    simulator:
        Optional simulator override. When omitted a new runtime-backed
        simulator is constructed from ``task``.
    seed:
        Optional seed used when instantiating the simulator.
    n_predictive_forward_model_samples:
        Number of posterior draws passed through the forward model to generate
        the predictive distribution.  When ``None`` all ``posterior_samples``
        are used.  Setting this to a small value (e.g. 50–200) dramatically
        reduces the number of expensive petitRADTRANS calls without materially
        affecting the predictive summary.

    Returns
    -------
    PosteriorPredictiveReport
        Posterior-predictive summary for the supplied observation.

    Notes
    -----
    The observations in the returned report remain on the original observation
    scale even when the posterior itself was trained on normalized inputs.
    """

    sample_array = _coerce_sample_array(posterior_samples)
    physical_parameters = _posterior_space_to_physical(
        task, sample_array, parameter_space
    )
    simulator = simulator or RuntimeSimulator(task=task, seed=seed)
    predictive_observations, predictive_metadata = _collect_valid_predictive_draws(
        simulator,
        physical_parameters,
        requested_valid_samples=n_predictive_forward_model_samples,
        seed=seed,
    )

    observed_values: dict[str, np.ndarray] = {}
    predictive_mean: dict[str, np.ndarray] = {}
    predictive_std: dict[str, np.ndarray] = {}
    interval_lower: dict[str, np.ndarray] = {}
    interval_upper: dict[str, np.ndarray] = {}
    interval_coverage: dict[str, float] = {}
    mean_absolute_error: dict[str, float] = {}
    mean_absolute_error_sigma: dict[str, float] = {}
    median_interval_width_over_uncertainty: dict[str, float] = {}
    crps: dict[str, float] = {}

    blocks_by_name = {block.name: block for block in observation_blocks}

    for observation_name, predictive_values in predictive_observations.items():
        if observation_name not in blocks_by_name:
            continue

        observed_value = np.asarray(
            blocks_by_name[observation_name].values, dtype=float
        )
        observed_uncertainty = _coerce_optional_observation_array(
            blocks_by_name[observation_name].uncertainties,
            observed_value.shape,
        )
        summary = _summarize_predictive_observation(
            np.asarray(predictive_values, dtype=float),
            observed_value,
            observed_uncertainty,
            interval_level=interval_level,
        )

        observed_values[observation_name] = observed_value
        predictive_mean[observation_name] = np.asarray(
            summary['predictive_mean'],
            dtype=float,
        )
        predictive_std[observation_name] = np.asarray(
            summary['predictive_std'],
            dtype=float,
        )
        interval_lower[observation_name] = np.asarray(
            summary['interval_lower'],
            dtype=float,
        )
        interval_upper[observation_name] = np.asarray(
            summary['interval_upper'],
            dtype=float,
        )
        interval_coverage[observation_name] = float(summary['interval_coverage'])
        mean_absolute_error[observation_name] = float(summary['mean_absolute_error'])
        mean_absolute_error_sigma[observation_name] = float(
            summary['mean_absolute_error_sigma']
        )
        median_interval_width_over_uncertainty[observation_name] = float(
            summary['median_interval_width_over_uncertainty']
        )
        crps[observation_name] = float(summary['crps'])

    return PosteriorPredictiveReport(
        observed_values=observed_values,
        predictive_mean=predictive_mean,
        predictive_std=predictive_std,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        interval_coverage=interval_coverage,
        mean_absolute_error=mean_absolute_error,
        mean_absolute_error_sigma=mean_absolute_error_sigma,
        median_interval_width_over_uncertainty=median_interval_width_over_uncertainty,
        crps=crps,
        metadata={
            "n_cases": 1,
            "n_posterior_samples": int(sample_array.shape[0]),
            **predictive_metadata,
            "interval_level": float(interval_level),
            "parameter_space": parameter_space,
            "task_name": task.name,
        },
    )


def generate_posterior_predictive_report(
    task: SBITask,
    posterior: Any,
    dataset_reader: NormalizedObservationDatasetReader,
    split: DatasetSplit = DatasetSplit.TEST,
    n_posterior_samples: int = 256,
    interval_level: float = 0.9,
    max_cases: int | None = None,
    seed: int | None = None,
    simulator: RuntimeSimulator | None = None,
    n_predictive_forward_model_samples: int | None = None,
    checkpoint_directory: str | Path | None = None,
    data_parallel: bool | None = None,
) -> PosteriorPredictiveReport:
    """Generate posterior-predictive summaries for a held-out dataset split.

    Parameters
    ----------
    task:
        SBI task defining parameter transforms and the simulator configuration.
    posterior:
        Trained posterior estimator used to sample held-out predictive draws.
    dataset_reader:
        Reader providing normalized observations and preprocessing metadata.
    split:
        Held-out split used for the predictive report.
    n_posterior_samples:
        Number of posterior draws generated per held-out observation.
    interval_level:
        Central predictive interval level reported for each dataset.
    max_cases:
        Optional cap on the number of held-out observations evaluated.
    seed:
        Optional base seed used to make predictive sampling reproducible.
    simulator:
        Optional runtime simulator override.
    n_predictive_forward_model_samples:
        Number of posterior draws passed through the forward model per
        held-out case.  When ``None`` all ``n_posterior_samples`` draws are
        forwarded.  Subsampling here is the primary lever for keeping the
        total number of petitRADTRANS calls to a manageable level when
        evaluating many held-out cases.
    checkpoint_directory:
        Optional directory for per-case checkpoints.  When provided, each
        completed case is written to disk as a compressed ``.npz`` file and
        skipped on resume.  This makes the expensive forward-model loop
        restartable after interruption.

    Returns
    -------
    PosteriorPredictiveReport
        Aggregate posterior-predictive summary over the requested split.

    Notes
    -----
    Observations are normalized internally for posterior encoding but compared
    on the original observation scale in the returned report.
    """

    split_size = int(dataset_reader.dataset.manifest.splits.get(split.value, 0))
    selected_indices = _selected_case_indices(split_size, max_cases, seed=seed)

    if not selected_indices:
        raise ValueError(
            f"Split '{split.value}' did not yield any observations for posterior-predictive evaluation."
        )

    resolved_parameter_space = getattr(posterior, "parameter_space", "physical")
    _dp = len(jax.devices()) > 1 if data_parallel is None else data_parallel
    simulator = simulator or RuntimeSimulator(task=task, seed=seed, data_parallel=_dp)
    observed_values: dict[str, list[np.ndarray]] = {}
    predictive_mean: dict[str, list[np.ndarray]] = {}
    predictive_std: dict[str, list[np.ndarray]] = {}
    interval_lower: dict[str, list[np.ndarray]] = {}
    interval_upper: dict[str, list[np.ndarray]] = {}
    interval_hits: dict[str, list[np.ndarray]] = {}
    mean_absolute_error_sigma_values: dict[str, list[float]] = {}
    median_interval_width_over_uncertainty_values: dict[str, list[float]] = {}
    crps_values: dict[str, list[float]] = {}

    base_seed = 0 if seed is None else int(seed)
    n_cases = len(selected_indices)
    skipped_invalid_cases = 0
    skipped_predictive_collection_failures = 0
    included_cases = 0
    total_forward_draws = 0
    total_invalid_forward_draws = 0
    total_valid_forward_draws = 0
    total_discarded_valid_forward_draws = 0
    total_resampling_rounds = 0
    total_accepted_valid_shortfall = 0
    has_batch_encode = hasattr(posterior, "batch_encode_observation")
    observation_fields = frozenset({"values", "uncertainties", "coordinates", "mask"})

    # Checkpoint directory setup
    ckpt_dir: Path | None = None
    if checkpoint_directory is not None:
        ckpt_dir = Path(checkpoint_directory)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    pending_cases: list[tuple[int, int, Path | None]] = []
    for case_offset, sample_index in enumerate(selected_indices):
        ckpt_file: Path | None = None
        if ckpt_dir is not None:
            ckpt_file = ckpt_dir / f"case_{case_offset:06d}.npz"
            if ckpt_file.exists():
                print(
                    f"Posterior-predictive case {case_offset + 1}/{n_cases} "
                    f"[restored from checkpoint]",
                    flush=True,
                )
                restored = np.load(ckpt_file, allow_pickle=False)
                restored_obs_names = [
                    name
                    for name in str(
                    restored["_observation_names_joined"].item()
                    ).split(",")
                    if name
                ]
                restored_total_forward_draws = int(
                    np.asarray(
                        restored.get(
                            "_total_forward_draw_count",
                            np.array([0], dtype=np.int32),
                        )
                    ).reshape(-1)[0]
                )
                restored_valid_forward_draws = int(
                    np.asarray(
                        restored.get(
                            "_valid_forward_draw_count",
                            np.array([0], dtype=np.int32),
                        )
                    ).reshape(-1)[0]
                )
                restored_invalid_forward_draws = int(
                    np.asarray(
                        restored.get(
                            "_invalid_forward_draw_count",
                            np.array(
                                [
                                    max(
                                        restored_total_forward_draws
                                        - restored_valid_forward_draws,
                                        0,
                                    )
                                ],
                                dtype=np.int32,
                            ),
                        )
                    ).reshape(-1)[0]
                )
                restored_discarded_valid_forward_draws = int(
                    np.asarray(
                        restored.get(
                            "_discarded_valid_forward_draw_count",
                            np.array([0], dtype=np.int32),
                        )
                    ).reshape(-1)[0]
                )
                restored_resampling_rounds = int(
                    np.asarray(
                        restored.get(
                            "_valid_forward_model_resampling_rounds",
                            np.array([0], dtype=np.int32),
                        )
                    ).reshape(-1)[0]
                )
                restored_accepted_shortfall = int(
                    np.asarray(
                        restored.get(
                            "_accepted_valid_forward_draw_shortfall_count",
                            np.array([0], dtype=np.int32),
                        )
                    ).reshape(-1)[0]
                )
                restored_collection_error = str(
                    np.asarray(
                        restored.get(
                            "_predictive_collection_error",
                            np.array([""], dtype=np.str_),
                        )
                    ).reshape(-1)[0]
                )
                total_forward_draws += restored_total_forward_draws
                total_valid_forward_draws += restored_valid_forward_draws
                total_invalid_forward_draws += restored_invalid_forward_draws
                total_discarded_valid_forward_draws += (
                    restored_discarded_valid_forward_draws
                )
                total_resampling_rounds += restored_resampling_rounds
                total_accepted_valid_shortfall += restored_accepted_shortfall
                if not restored_obs_names:
                    if restored_collection_error:
                        skipped_predictive_collection_failures += 1
                    else:
                        skipped_invalid_cases += 1
                    continue
                included_cases += 1
                for obs_name in restored_obs_names:
                    observed_values.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_observed"]
                    )
                    predictive_mean.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_pred_mean"]
                    )
                    predictive_std.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_pred_std"]
                    )
                    interval_lower.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_lower"]
                    )
                    interval_upper.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_upper"]
                    )
                    interval_hits.setdefault(obs_name, []).append(
                        restored[f"{obs_name}_hits"]
                    )
                    mean_absolute_error_sigma_values.setdefault(obs_name, []).append(
                        float(
                            np.asarray(
                                restored.get(
                                    f"{obs_name}_mean_absolute_error_sigma",
                                    np.array(np.nan, dtype=float),
                                ),
                                dtype=float,
                            )
                        )
                    )
                    median_interval_width_over_uncertainty_values.setdefault(
                        obs_name,
                        [],
                    ).append(
                        float(
                            np.asarray(
                                restored.get(
                                    f"{obs_name}_median_interval_width_over_uncertainty",
                                    np.array(np.nan, dtype=float),
                                ),
                                dtype=float,
                            )
                        )
                    )
                    crps_values.setdefault(obs_name, []).append(
                        float(
                            np.asarray(
                                restored.get(
                                    f"{obs_name}_crps",
                                    np.array(np.nan, dtype=float),
                                ),
                                dtype=float,
                            )
                        )
                    )
                continue

        pending_cases.append((case_offset, sample_index, ckpt_file))

    if pending_cases:
        pending_indices = np.asarray(
            [sample_index for _, sample_index, _ in pending_cases],
            dtype=int,
        )
        pending_cursor = 0
        stream_chunk_size = max(1, min(32, len(pending_cases)))

        for chunk in dataset_reader.dataset.iter_split_chunks(
            split=split,
            chunk_size=stream_chunk_size,
            indices=pending_indices,
            observation_fields=observation_fields,
        ):
            raw_blocks_batch = build_observation_block_batch(
                chunk["observations"],
                task.observation_schema.modalities,
            )
            normalized_blocks_batch = [
                normalize_observation_blocks(blocks, dataset_reader.preprocessing_metadata)
                for blocks in raw_blocks_batch
            ]
            encoded_batch = (
                posterior.batch_encode_observation(normalized_blocks_batch)
                if has_batch_encode
                else None
            )

            for row_offset, raw_blocks in enumerate(raw_blocks_batch):
                case_offset, _sample_index, ckpt_file = pending_cases[
                    pending_cursor + row_offset
                ]
                print(
                    f"Posterior-predictive case {case_offset + 1}/{n_cases} ...",
                    flush=True,
                )
                if encoded_batch is not None:
                    encoded = encoded_batch[row_offset]
                else:
                    encoded = posterior.encode_observation(
                        normalized_blocks_batch[row_offset]
                    )
                posterior_samples = posterior.sample_posterior(
                    encoded,
                    n_samples=n_posterior_samples,
                    seed=base_seed + case_offset,
                ).samples
                physical_parameters = _posterior_space_to_physical(
                    task, posterior_samples, resolved_parameter_space
                )
                try:
                    predictive_observations, predictive_metadata = _collect_valid_predictive_draws(
                        simulator,
                        physical_parameters,
                        requested_valid_samples=n_predictive_forward_model_samples,
                        seed=base_seed + case_offset,
                    )
                except PredictiveResamplingExhaustedError as error:
                    predictive_metadata = dict(error.metadata)
                    total_forward_draws += int(
                        predictive_metadata["n_forward_model_samples"]
                    )
                    total_valid_forward_draws += int(
                        predictive_metadata["n_valid_forward_model_samples"]
                    )
                    total_invalid_forward_draws += int(
                        predictive_metadata["invalid_forward_model_sample_count"]
                    )
                    total_discarded_valid_forward_draws += int(
                        predictive_metadata[
                            "discarded_valid_forward_model_sample_count"
                        ]
                    )
                    total_resampling_rounds += int(
                        predictive_metadata["valid_forward_model_resampling_rounds"]
                    )
                    total_accepted_valid_shortfall += int(
                        predictive_metadata[
                            "accepted_valid_forward_model_shortfall_count"
                        ]
                    )
                    skipped_predictive_collection_failures += 1
                    if ckpt_file is not None:
                        np.savez_compressed(
                            ckpt_file,
                            _observation_names_joined=np.array([""], dtype=np.str_),
                            _predictive_collection_error=np.array([str(error)], dtype=np.str_),
                            _total_forward_draw_count=np.array(
                                [int(predictive_metadata["n_forward_model_samples"])],
                                dtype=np.int32,
                            ),
                            _valid_forward_draw_count=np.array(
                                [int(predictive_metadata["n_valid_forward_model_samples"])],
                                dtype=np.int32,
                            ),
                            _invalid_forward_draw_count=np.array(
                                [
                                    int(
                                        predictive_metadata[
                                            "invalid_forward_model_sample_count"
                                        ]
                                    )
                                ],
                                dtype=np.int32,
                            ),
                            _discarded_valid_forward_draw_count=np.array(
                                [
                                    int(
                                        predictive_metadata[
                                            "discarded_valid_forward_model_sample_count"
                                        ]
                                    )
                                ],
                                dtype=np.int32,
                            ),
                            _valid_forward_model_resampling_rounds=np.array(
                                [
                                    int(
                                        predictive_metadata[
                                            "valid_forward_model_resampling_rounds"
                                        ]
                                    )
                                ],
                                dtype=np.int32,
                            ),
                            _requested_valid_forward_draw_count=np.array(
                                [
                                    int(
                                        predictive_metadata[
                                            "requested_valid_forward_model_samples"
                                        ]
                                    )
                                ],
                                dtype=np.int32,
                            ),
                            _accepted_valid_forward_draw_shortfall_count=np.array(
                                [
                                    int(
                                        predictive_metadata[
                                            "accepted_valid_forward_model_shortfall_count"
                                        ]
                                    )
                                ],
                                dtype=np.int32,
                            ),
                        )
                    continue
                total_forward_draws += int(
                    predictive_metadata["n_forward_model_samples"]
                )
                total_valid_forward_draws += int(
                    predictive_metadata["n_valid_forward_model_samples"]
                )
                total_invalid_forward_draws += int(
                    predictive_metadata["invalid_forward_model_sample_count"]
                )
                total_discarded_valid_forward_draws += int(
                    predictive_metadata[
                        "discarded_valid_forward_model_sample_count"
                    ]
                )
                total_resampling_rounds += int(
                    predictive_metadata["valid_forward_model_resampling_rounds"]
                )
                total_accepted_valid_shortfall += int(
                    predictive_metadata[
                        "accepted_valid_forward_model_shortfall_count"
                    ]
                )
                raw_blocks_by_name = {block.name: block for block in raw_blocks}

                case_save_arrays: dict[str, np.ndarray] = {}
                case_observation_names: list[str] = []
                case_save_arrays["_total_forward_draw_count"] = np.array(
                    [int(predictive_metadata["n_forward_model_samples"])],
                    dtype=np.int32,
                )
                case_save_arrays["_valid_forward_draw_count"] = np.array(
                    [int(predictive_metadata["n_valid_forward_model_samples"])],
                    dtype=np.int32,
                )
                case_save_arrays["_invalid_forward_draw_count"] = np.array(
                    [int(predictive_metadata["invalid_forward_model_sample_count"])],
                    dtype=np.int32,
                )
                case_save_arrays["_invalid_forward_draw_fraction"] = np.array(
                    [float(predictive_metadata["invalid_forward_model_sample_fraction"])],
                    dtype=float,
                )
                case_save_arrays["_discarded_valid_forward_draw_count"] = np.array(
                    [
                        int(
                            predictive_metadata[
                                "discarded_valid_forward_model_sample_count"
                            ]
                        )
                    ],
                    dtype=np.int32,
                )
                case_save_arrays[
                    "_accepted_valid_forward_draw_shortfall_count"
                ] = np.array(
                    [
                        int(
                            predictive_metadata[
                                "accepted_valid_forward_model_shortfall_count"
                            ]
                        )
                    ],
                    dtype=np.int32,
                )
                case_save_arrays["_valid_forward_model_resampling_rounds"] = np.array(
                    [int(predictive_metadata["valid_forward_model_resampling_rounds"])],
                    dtype=np.int32,
                )
                case_save_arrays["_requested_valid_forward_draw_count"] = np.array(
                    [
                        int(
                            predictive_metadata[
                                "requested_valid_forward_model_samples"
                            ]
                        )
                    ],
                    dtype=np.int32,
                )

                included_cases += 1

                for observation_name, predictive_values in predictive_observations.items():
                    if observation_name not in raw_blocks_by_name:
                        continue
                    observed_value = np.asarray(
                        raw_blocks_by_name[observation_name].values,
                        dtype=float,
                    )
                    observed_uncertainty = _coerce_optional_observation_array(
                        raw_blocks_by_name[observation_name].uncertainties,
                        observed_value.shape,
                    )
                    summary = _summarize_predictive_observation(
                        np.asarray(predictive_values, dtype=float),
                        observed_value,
                        observed_uncertainty,
                        interval_level=interval_level,
                    )
                    pred_mean = np.asarray(summary["predictive_mean"], dtype=float)
                    pred_std = np.asarray(summary["predictive_std"], dtype=float)
                    lower = np.asarray(summary["interval_lower"], dtype=float)
                    upper = np.asarray(summary["interval_upper"], dtype=float)
                    hits = (observed_value >= lower) & (observed_value <= upper)

                    observed_values.setdefault(observation_name, []).append(observed_value)
                    predictive_mean.setdefault(observation_name, []).append(pred_mean)
                    predictive_std.setdefault(observation_name, []).append(pred_std)
                    interval_lower.setdefault(observation_name, []).append(lower)
                    interval_upper.setdefault(observation_name, []).append(upper)
                    interval_hits.setdefault(observation_name, []).append(hits)
                    mean_absolute_error_sigma_values.setdefault(
                        observation_name,
                        [],
                    ).append(float(summary["mean_absolute_error_sigma"]))
                    median_interval_width_over_uncertainty_values.setdefault(
                        observation_name,
                        [],
                    ).append(
                        float(summary["median_interval_width_over_uncertainty"])
                    )
                    crps_values.setdefault(observation_name, []).append(
                        float(summary["crps"])
                    )

                    case_observation_names.append(observation_name)
                    case_save_arrays[f"{observation_name}_observed"] = observed_value
                    case_save_arrays[f"{observation_name}_pred_mean"] = pred_mean
                    case_save_arrays[f"{observation_name}_pred_std"] = pred_std
                    case_save_arrays[f"{observation_name}_lower"] = lower
                    case_save_arrays[f"{observation_name}_upper"] = upper
                    case_save_arrays[f"{observation_name}_hits"] = hits
                    case_save_arrays[
                        f"{observation_name}_mean_absolute_error_sigma"
                    ] = np.asarray(
                        summary["mean_absolute_error_sigma"],
                        dtype=float,
                    )
                    case_save_arrays[
                        f"{observation_name}_median_interval_width_over_uncertainty"
                    ] = np.asarray(
                        summary["median_interval_width_over_uncertainty"],
                        dtype=float,
                    )
                    case_save_arrays[f"{observation_name}_crps"] = np.asarray(
                        summary["crps"],
                        dtype=float,
                    )

                if ckpt_file is not None:
                    case_save_arrays["_observation_names_joined"] = np.array(
                        [",".join(case_observation_names)]
                    )
                    np.savez_compressed(ckpt_file, **case_save_arrays)

            pending_cursor += len(raw_blocks_batch)

    observed_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in observed_values.items()
    }
    predictive_mean_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in predictive_mean.items()
    }
    predictive_std_arrays = {
        name: np.asarray(values, dtype=float) for name, values in predictive_std.items()
    }
    interval_lower_arrays = {
        name: np.asarray(values, dtype=float) for name, values in interval_lower.items()
    }
    interval_upper_arrays = {
        name: np.asarray(values, dtype=float) for name, values in interval_upper.items()
    }
    interval_coverage = {
        name: float(np.mean(np.asarray(values, dtype=bool)))
        for name, values in interval_hits.items()
    }
    mean_absolute_error = {
        name: float(
            np.mean(np.abs(predictive_mean_arrays[name] - observed_arrays[name]))
        )
        for name in observed_arrays
    }
    mean_absolute_error_sigma = {
        name: float(np.nanmean(np.asarray(values, dtype=float)))
        for name, values in mean_absolute_error_sigma_values.items()
    }
    median_interval_width_over_uncertainty = {
        name: float(np.nanmedian(np.asarray(values, dtype=float)))
        for name, values in median_interval_width_over_uncertainty_values.items()
    }
    crps = {
        name: float(np.nanmean(np.asarray(values, dtype=float)))
        for name, values in crps_values.items()
    }

    if not observed_arrays:
        raise ValueError(
            "Posterior-predictive evaluation produced no cases with valid forward draws."
        )

    return PosteriorPredictiveReport(
        observed_values=observed_arrays,
        predictive_mean=predictive_mean_arrays,
        predictive_std=predictive_std_arrays,
        interval_lower=interval_lower_arrays,
        interval_upper=interval_upper_arrays,
        interval_coverage=interval_coverage,
        mean_absolute_error=mean_absolute_error,
        mean_absolute_error_sigma=mean_absolute_error_sigma,
        median_interval_width_over_uncertainty=median_interval_width_over_uncertainty,
        crps=crps,
        metadata={
            "split": split.value,
            "n_cases": included_cases,
            "requested_cases": len(selected_indices),
            "skipped_invalid_cases": skipped_invalid_cases,
            "skipped_predictive_collection_failures": skipped_predictive_collection_failures,
            "n_posterior_samples": int(n_posterior_samples),
            "n_forward_model_samples": int(total_forward_draws),
            "n_valid_forward_model_samples": int(total_valid_forward_draws),
            "requested_valid_forward_model_samples_per_case": (
                int(n_posterior_samples)
                if n_predictive_forward_model_samples is None
                else int(n_predictive_forward_model_samples)
            ),
            "invalid_forward_model_sample_count": int(total_invalid_forward_draws),
            "invalid_forward_model_sample_fraction": float(
                total_invalid_forward_draws / max(total_forward_draws, 1)
            ),
            "discarded_valid_forward_model_sample_count": int(
                total_discarded_valid_forward_draws
            ),
            "accepted_valid_forward_model_shortfall_count": int(
                total_accepted_valid_shortfall
            ),
            "valid_forward_model_resampling_rounds": int(total_resampling_rounds),
            "interval_level": float(interval_level),
            "parameter_space": resolved_parameter_space,
            "task_name": task.name,
        },
    )


__all__ = [
    "CoverageLevelReport",
    "LocalSensitivityPointReport",
    "LocalSensitivityReport",
    "PosteriorPredictiveReport",
    "SimulationBasedCalibrationReport",
    "central_credible_interval",
    "empirical_interval_coverage",
    "generate_local_sensitivity_report",
    "generate_posterior_predictive_report",
    "generate_sbc_report",
    "local_sensitivity_report_to_payload",
    "posterior_predictive_report_from_samples",
    "simulation_based_calibration_report",
]
