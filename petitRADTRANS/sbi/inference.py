"""User-facing amortized inference services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .calibration import (
    PosteriorPredictiveReport,
    posterior_predictive_report_from_samples,
)
from .observation import ObservationBlock
from .posterior import PosteriorEstimator, PosteriorSamples
from .preprocessing import TaskPreprocessingMetadata, normalize_observation_blocks
from .simulator import RuntimeSimulator
from .task import SBITask


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
            posterior_predictive = self.posterior_predictive(
                posterior=posterior,
                observation_blocks=observation_blocks,
                interval_level=posterior_predictive_interval_level,
                n_predictive_forward_model_samples=n_predictive_forward_model_samples,
                seed=seed,
            )

        return AmortizedRetrievalResult(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            metadata={
                "preprocessing_applied": self.preprocessing_metadata is not None,
                "preprocessing_version": (
                    None
                    if self.preprocessing_metadata is None
                    else self.preprocessing_metadata.version
                ),
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
            Currently ``None`` in the base implementation. Subclasses may return
            concrete domain-diagnostic results.
        """
        del observation_blocks
        return None
