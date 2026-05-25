"""User-facing amortized inference services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .calibration import (
    PosteriorPredictiveReport,
    _posterior_space_to_physical,
    posterior_predictive_report_from_samples,
)
from .observation import ObservationBlock
from .posterior import PosteriorEstimator, PosteriorSamples
from .preprocessing import TaskPreprocessingMetadata, normalize_observation_blocks
from .simulator import RuntimeSimulator
from .task import SBITask


def _coerce_block_array(values: Any, target_size: int) -> np.ndarray | None:
    if values is None:
        return None

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == target_size:
        return array
    if array.size == 0:
        return np.full(target_size, np.nan, dtype=float)
    if array.size < target_size:
        return np.concatenate(
            [array, np.full(target_size - array.size, np.nan, dtype=float)]
        )
    return array[:target_size]


def _robust_score_summary(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    safe_scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, np.nan)
    z_scores = np.abs(values - center) / safe_scale
    finite_valid = valid_mask & np.isfinite(z_scores)
    if not np.any(finite_valid):
        return 0.0, 0.0
    return (
        float(np.nanquantile(z_scores[finite_valid], 0.95)),
        float(np.nanmax(z_scores[finite_valid])),
    )


@dataclass(frozen=True)
class OODDiagnostic:
    """Describe whether an observation is inside the training support.

    Attributes
    ----------
    score:
        Scalar out-of-distribution score.
    threshold:
        Optional threshold used to convert the score into a pass/fail decision.
    passed:
        Optional boolean indicating whether the observation passed the OOD test.
    metadata:
        Auxiliary diagnostic metadata.
    """

    score: float
    threshold: float | None = None
    passed: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AmortizedRetrievalResult:
    """Return type for amortized inference queries.

    Attributes
    ----------
    posterior:
        Posterior samples and any attached diagnostics.
    posterior_predictive:
        Optional posterior-predictive report for the same observation.
    ood_diagnostic:
        Optional in/out-of-distribution assessment.
    metadata:
        Additional metadata such as preprocessing usage during inference.
    """

    posterior: PosteriorSamples
    posterior_predictive: Any = None
    ood_diagnostic: OODDiagnostic | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AmortizedRetrieval:
    """Serve trained SBI models through a retrieval-like interface.

    Parameters
    ----------
    task:
        SBI task compatible with the trained posterior.
    posterior_estimator:
        Trained posterior estimator used for encoding and sampling.
    simulator:
        Optional simulator override used for posterior-predictive generation.
    preprocessing_metadata:
        Optional preprocessing metadata or payload. When omitted the inference
        service attempts to recover the preprocessing payload saved inside the
        posterior artifact.

    Notes
    -----
    Raw user-facing observation blocks are normalized internally before
    encoding when preprocessing metadata is available, but posterior-predictive
    comparisons remain on the original observation scale.
    """

    def __init__(
        self,
        task: SBITask,
        posterior_estimator: PosteriorEstimator,
        simulator: RuntimeSimulator | None = None,
        preprocessing_metadata: (
            TaskPreprocessingMetadata | Mapping[str, Any] | None
        ) = None,
    ):
        self.task = task
        self.posterior_estimator = posterior_estimator
        self.simulator = simulator
        self.preprocessing_metadata = self._resolve_preprocessing_metadata(
            preprocessing_metadata
        )

    def _resolve_preprocessing_metadata(
        self,
        preprocessing_metadata: TaskPreprocessingMetadata | Mapping[str, Any] | None,
    ) -> TaskPreprocessingMetadata | None:
        if isinstance(preprocessing_metadata, TaskPreprocessingMetadata):
            return preprocessing_metadata

        if preprocessing_metadata is not None:
            return TaskPreprocessingMetadata.from_payload(preprocessing_metadata)

        payload = getattr(
            self.posterior_estimator, "preprocessing_metadata_payload", None
        )
        if payload:
            return TaskPreprocessingMetadata.from_payload(payload)

        return None

    def prepare_observation_blocks(
        self, observation_blocks: list[ObservationBlock]
    ) -> list[ObservationBlock]:
        """Normalize user-facing observation blocks when preprocessing metadata is available.

        Parameters
        ----------
        observation_blocks:
            Raw observation blocks provided by the caller.

        Returns
        -------
        list[ObservationBlock]
            Either the original blocks or normalized copies, depending on
            whether preprocessing metadata is available.
        """
        if self.preprocessing_metadata is None:
            return observation_blocks

        return normalize_observation_blocks(
            observation_blocks, self.preprocessing_metadata
        )

    def infer(
        self,
        observation_blocks: list[ObservationBlock],
        n_posterior_samples: int = 1000,
        include_posterior_predictive: bool = False,
        posterior_predictive_interval_level: float = 0.9,
        n_predictive_forward_model_samples: int | None = None,
        seed: int | None = None,
    ) -> AmortizedRetrievalResult:
        """Infer a posterior for one structured observation.

        Parameters
        ----------
        observation_blocks:
            Raw observation blocks for one target system.
        n_posterior_samples:
            Number of posterior draws to generate.
        include_posterior_predictive:
            Whether to also generate a posterior-predictive summary.
        posterior_predictive_interval_level:
            Predictive interval level used when generating the optional
            posterior-predictive report.
        n_predictive_forward_model_samples:
            Number of posterior draws passed through the forward model when
            generating the posterior-predictive summary.  When ``None`` all
            ``n_posterior_samples`` draws are forwarded.  Set this to a small
            value (e.g. 50–200) to avoid a multi-hour hang when
            ``n_posterior_samples`` is large.
        seed:
            Optional seed for posterior and posterior-predictive sampling.

        Returns
        -------
        AmortizedRetrievalResult
            Posterior samples and any optional predictive metadata.
        """
        ood_diagnostic = self.diagnose_domain(observation_blocks)
        prepared_observation_blocks = self.prepare_observation_blocks(
            observation_blocks
        )
        encoded_observation = self.posterior_estimator.encode_observation(
            prepared_observation_blocks
        )
        posterior = self.posterior_estimator.sample_posterior(
            observation=encoded_observation,
            n_samples=n_posterior_samples,
            seed=seed,
        )
        posterior_predictive = None

        if include_posterior_predictive:
            # posterior_predictive_report_from_samples reads parameter_space from the
            # estimator and performs the transform internally, so pass the original
            # (possibly unconstrained) posterior here.
            posterior_predictive = self.posterior_predictive(
                posterior=posterior,
                observation_blocks=observation_blocks,
                interval_level=posterior_predictive_interval_level,
                n_predictive_forward_model_samples=n_predictive_forward_model_samples,
                seed=seed,
            )

        # Transform samples into physical parameter space so that callers always
        # receive meaningful, directly interpretable parameter values.
        parameter_space = getattr(
            self.posterior_estimator, "parameter_space", "physical"
        )
        if parameter_space != "physical":
            physical_samples = np.asarray(
                _posterior_space_to_physical(
                    self.task, posterior.samples, parameter_space
                ),
                dtype=float,
            )
            posterior = PosteriorSamples(
                samples=physical_samples,
                log_probabilities=posterior.log_probabilities,
                weights=posterior.weights,
                metadata=posterior.metadata,
            )

        return AmortizedRetrievalResult(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            ood_diagnostic=ood_diagnostic,
            metadata={
                "preprocessing_applied": self.preprocessing_metadata is not None,
                "preprocessing_version": (
                    None
                    if self.preprocessing_metadata is None
                    else self.preprocessing_metadata.version
                ),
                "ood_score": (
                    None if ood_diagnostic is None else float(ood_diagnostic.score)
                ),
                "ood_threshold": (
                    None
                    if ood_diagnostic is None or ood_diagnostic.threshold is None
                    else float(ood_diagnostic.threshold)
                ),
                "ood_passed": (
                    None if ood_diagnostic is None else ood_diagnostic.passed
                ),
                "parameter_space": "physical",
            },
        )

    def posterior_predictive(
        self,
        posterior: PosteriorSamples,
        observation_blocks: list[ObservationBlock],
        interval_level: float = 0.9,
        n_predictive_forward_model_samples: int | None = None,
        seed: int | None = None,
    ) -> PosteriorPredictiveReport:
        """Generate posterior-predictive draws for a fitted observation.

        Parameters
        ----------
        posterior:
            Posterior samples associated with one observed system.
        observation_blocks:
            Original observation blocks used for user-facing comparison.
        interval_level:
            Central predictive interval level to report.
        n_predictive_forward_model_samples:
            Number of posterior draws passed through the forward model.  When
            ``None`` all samples in ``posterior`` are used.  Set this to a
            small value (e.g. 50–200) to keep the number of expensive
            petitRADTRANS calls manageable.
        seed:
            Optional seed for predictive simulation.

        Returns
        -------
        PosteriorPredictiveReport
            Predictive summary for the supplied posterior samples.
        """
        return posterior_predictive_report_from_samples(
            task=self.task,
            posterior_samples=posterior.samples,
            observation_blocks=observation_blocks,
            parameter_space=getattr(
                self.posterior_estimator, "parameter_space", "physical"
            ),
            interval_level=interval_level,
            simulator=self.simulator,
            seed=seed,
            n_predictive_forward_model_samples=n_predictive_forward_model_samples,
        )

    def diagnose_domain(
        self, observation_blocks: list[ObservationBlock]
    ) -> OODDiagnostic | None:
        """Estimate whether the observation is in-distribution for the model.

        Parameters
        ----------
        observation_blocks:
            Observation blocks to diagnose.

        Returns
        -------
        OODDiagnostic | None
            Robust support-distance diagnostic derived from preprocessing
            statistics, or ``None`` when preprocessing metadata is unavailable.
        """
        if self.preprocessing_metadata is None:
            return None

        per_block_scores: dict[str, float] = {}
        per_block_details: dict[str, Mapping[str, float]] = {}
        global_score = 0.0
        threshold = 8.0

        for block in observation_blocks:
            block_name = str(block.name)
            stats = self.preprocessing_metadata.blocks.get(block_name)
            if stats is None:
                continue

            block_values = np.asarray(block.values, dtype=float).reshape(-1)
            if block_values.size == 0:
                continue

            valid_mask = np.isfinite(block_values)
            block_mask = _coerce_block_array(block.mask, block_values.size)
            if block_mask is not None:
                valid_mask &= ~np.asarray(block_mask, dtype=bool)

            value_center = _coerce_block_array(
                stats.value_median_per_wl,
                block_values.size,
            )
            if value_center is None:
                value_center = np.full(
                    block_values.size,
                    float(
                        stats.value_median
                        if stats.value_median is not None
                        else stats.value_mean
                    ),
                    dtype=float,
                )
            value_scale = _coerce_block_array(
                stats.value_iqr_per_wl,
                block_values.size,
            )
            if value_scale is None:
                value_scale = np.full(
                    block_values.size,
                    float(
                        stats.value_iqr
                        if stats.value_iqr is not None and stats.value_iqr > 0.0
                        else max(float(stats.value_std), 1e-8)
                    ),
                    dtype=float,
                )

            value_score_p95, value_score_max = _robust_score_summary(
                block_values,
                value_center,
                value_scale,
                valid_mask,
            )
            block_score = value_score_p95
            block_details: dict[str, float] = {
                "value_robust_z_p95": value_score_p95,
                "value_robust_z_max": value_score_max,
            }

            block_uncertainties = _coerce_block_array(
                block.uncertainties,
                block_values.size,
            )
            if block_uncertainties is not None:
                uncertainty_valid_mask = (
                    valid_mask
                    & np.isfinite(block_uncertainties)
                    & (block_uncertainties > 0.0)
                )
                uncertainty_center = _coerce_block_array(
                    stats.uncertainty_median_per_wl,
                    block_values.size,
                )
                if uncertainty_center is None:
                    default_uncertainty_center = (
                        stats.uncertainty_median
                        if stats.uncertainty_median is not None
                        else stats.uncertainty_mean
                    )
                    if default_uncertainty_center is not None:
                        uncertainty_center = np.full(
                            block_values.size,
                            float(default_uncertainty_center),
                            dtype=float,
                        )
                uncertainty_scale = _coerce_block_array(
                    stats.uncertainty_iqr_per_wl,
                    block_values.size,
                )
                if uncertainty_scale is None:
                    default_uncertainty_scale = (
                        stats.uncertainty_iqr
                        if stats.uncertainty_iqr is not None
                        and stats.uncertainty_iqr > 0.0
                        else stats.uncertainty_std
                    )
                    if default_uncertainty_scale is not None:
                        uncertainty_scale = np.full(
                            block_values.size,
                            max(float(default_uncertainty_scale), 1e-8),
                            dtype=float,
                        )

                if uncertainty_center is not None and uncertainty_scale is not None:
                    uncertainty_score_p95, uncertainty_score_max = (
                        _robust_score_summary(
                            block_uncertainties,
                            uncertainty_center,
                            uncertainty_scale,
                            uncertainty_valid_mask,
                        )
                    )
                    block_score = max(block_score, uncertainty_score_p95)
                    block_details["uncertainty_robust_z_p95"] = (
                        uncertainty_score_p95
                    )
                    block_details["uncertainty_robust_z_max"] = (
                        uncertainty_score_max
                    )

            per_block_scores[block_name] = block_score
            per_block_details[block_name] = block_details
            global_score = max(global_score, block_score)

        return OODDiagnostic(
            score=float(global_score),
            threshold=threshold,
            passed=bool(global_score <= threshold),
            metadata={
                "method": "robust_blockwise_z_p95",
                "preprocessing_version": self.preprocessing_metadata.version,
                "per_block_scores": per_block_scores,
                "per_block_details": per_block_details,
            },
        )
