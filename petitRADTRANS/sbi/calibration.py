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


def _selected_case_indices(n_cases: int, max_cases: int | None) -> list[int]:
    if max_cases is None:
        return list(range(n_cases))
    return list(range(min(int(max_cases), n_cases)))


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
    if n_predictive_forward_model_samples is not None:
        n_use = min(
            int(n_predictive_forward_model_samples), physical_parameters.shape[0]
        )
        rng = np.random.default_rng(seed)
        indices = rng.choice(physical_parameters.shape[0], size=n_use, replace=False)
        physical_parameters = physical_parameters[indices]
    simulator = simulator or RuntimeSimulator(task=task, seed=seed)
    predictive_batch = simulator.simulate_from_parameters(physical_parameters)

    observed_values: dict[str, np.ndarray] = {}
    predictive_mean: dict[str, np.ndarray] = {}
    predictive_std: dict[str, np.ndarray] = {}
    interval_lower: dict[str, np.ndarray] = {}
    interval_upper: dict[str, np.ndarray] = {}
    interval_coverage: dict[str, float] = {}
    mean_absolute_error: dict[str, float] = {}

    blocks_by_name = {block.name: block for block in observation_blocks}

    for observation_name, payload in predictive_batch.observations.items():
        if observation_name not in blocks_by_name:
            continue

        predictive_values = np.asarray(payload["values"], dtype=float)
        lower, upper = central_credible_interval(
            predictive_values, level=interval_level
        )
        observed_value = np.asarray(
            blocks_by_name[observation_name].values, dtype=float
        )

        observed_values[observation_name] = observed_value
        predictive_mean[observation_name] = np.mean(predictive_values, axis=0)
        predictive_std[observation_name] = np.std(predictive_values, axis=0)
        interval_lower[observation_name] = lower
        interval_upper[observation_name] = upper
        interval_coverage[observation_name] = float(
            np.mean((observed_value >= lower) & (observed_value <= upper))
        )
        mean_absolute_error[observation_name] = float(
            np.mean(np.abs(predictive_mean[observation_name] - observed_value))
        )

    return PosteriorPredictiveReport(
        observed_values=observed_values,
        predictive_mean=predictive_mean,
        predictive_std=predictive_std,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        interval_coverage=interval_coverage,
        mean_absolute_error=mean_absolute_error,
        metadata={
            "n_cases": 1,
            "n_posterior_samples": int(sample_array.shape[0]),
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

    raw_split = dataset_reader.dataset.read_split(split)
    raw_blocks_batch = build_observation_block_batch(
        raw_split["observations"], task.observation_schema.modalities
    )
    normalized_blocks_batch = [
        normalize_observation_blocks(blocks, dataset_reader.preprocessing_metadata)
        for blocks in raw_blocks_batch
    ]
    selected_indices = _selected_case_indices(len(normalized_blocks_batch), max_cases)

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

    base_seed = 0 if seed is None else int(seed)
    n_cases = len(selected_indices)
    rng = np.random.default_rng(seed)
    has_batch_encode = hasattr(posterior, "batch_encode_observation")

    # Checkpoint directory setup
    ckpt_dir: Path | None = None
    if checkpoint_directory is not None:
        ckpt_dir = Path(checkpoint_directory)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Batch-encode all selected observations in one vmapped forward pass when
    # the posterior supports it; otherwise encode case-by-case.
    if has_batch_encode:
        selected_blocks = [normalized_blocks_batch[i] for i in selected_indices]
        encoded_all = posterior.batch_encode_observation(selected_blocks)
    else:
        encoded_all = None

    for case_offset, sample_index in enumerate(selected_indices):
        # --- Checkpoint resume: skip cases already completed ---
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
                restored_obs_names = str(
                    restored["_observation_names_joined"].item()
                ).split(",")
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
                continue

        print(f"Posterior-predictive case {case_offset + 1}/{n_cases} ...", flush=True)
        if encoded_all is not None:
            encoded = encoded_all[case_offset]
        else:
            encoded = posterior.encode_observation(
                normalized_blocks_batch[sample_index]
            )
        posterior_samples = posterior.sample_posterior(
            encoded,
            n_samples=n_posterior_samples,
            seed=base_seed + case_offset,
        ).samples
        physical_parameters = _posterior_space_to_physical(
            task, posterior_samples, resolved_parameter_space
        )
        if n_predictive_forward_model_samples is not None:
            n_use = min(
                int(n_predictive_forward_model_samples), physical_parameters.shape[0]
            )
            indices = rng.choice(
                physical_parameters.shape[0], size=n_use, replace=False
            )
            physical_parameters = physical_parameters[indices]
        predictive_batch = simulator.simulate_from_parameters(physical_parameters)

        # Collect per-observation results for this case
        case_save_arrays: dict[str, np.ndarray] = {}
        case_observation_names: list[str] = []

        for observation_name, payload in predictive_batch.observations.items():
            predictive_values = np.asarray(payload["values"], dtype=float)
            lower, upper = central_credible_interval(
                predictive_values, level=interval_level
            )
            observed_value = np.asarray(
                raw_split["observations"][observation_name]["values"][sample_index],
                dtype=float,
            )
            pred_mean = np.mean(predictive_values, axis=0)
            pred_std = np.std(predictive_values, axis=0)
            hits = (observed_value >= lower) & (observed_value <= upper)

            observed_values.setdefault(observation_name, []).append(observed_value)
            predictive_mean.setdefault(observation_name, []).append(pred_mean)
            predictive_std.setdefault(observation_name, []).append(pred_std)
            interval_lower.setdefault(observation_name, []).append(lower)
            interval_upper.setdefault(observation_name, []).append(upper)
            interval_hits.setdefault(observation_name, []).append(hits)

            case_observation_names.append(observation_name)
            case_save_arrays[f"{observation_name}_observed"] = observed_value
            case_save_arrays[f"{observation_name}_pred_mean"] = pred_mean
            case_save_arrays[f"{observation_name}_pred_std"] = pred_std
            case_save_arrays[f"{observation_name}_lower"] = lower
            case_save_arrays[f"{observation_name}_upper"] = upper
            case_save_arrays[f"{observation_name}_hits"] = hits

        # --- Checkpoint save: persist this case ---
        if ckpt_file is not None:
            case_save_arrays["_observation_names_joined"] = np.array(
                [",".join(case_observation_names)]
            )
            np.savez_compressed(ckpt_file, **case_save_arrays)

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

    return PosteriorPredictiveReport(
        observed_values=observed_arrays,
        predictive_mean=predictive_mean_arrays,
        predictive_std=predictive_std_arrays,
        interval_lower=interval_lower_arrays,
        interval_upper=interval_upper_arrays,
        interval_coverage=interval_coverage,
        mean_absolute_error=mean_absolute_error,
        metadata={
            "split": split.value,
            "n_cases": len(selected_indices),
            "n_posterior_samples": int(n_posterior_samples),
            "interval_level": float(interval_level),
            "parameter_space": resolved_parameter_space,
            "task_name": task.name,
        },
    )


__all__ = [
    "CoverageLevelReport",
    "PosteriorPredictiveReport",
    "SimulationBasedCalibrationReport",
    "central_credible_interval",
    "empirical_interval_coverage",
    "generate_posterior_predictive_report",
    "generate_sbc_report",
    "posterior_predictive_report_from_samples",
    "simulation_based_calibration_report",
]
