"""Posterior estimator interfaces and shared persistence base classes."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import ArtifactMetadata, ArtifactRegistry, ArtifactRole
from ..observation import EncodedObservation, ObservationBlock


POSTERIOR_METADATA_SCHEMA_VERSION = "0.2.0"


@dataclass(frozen=True)
class TrainingArtifacts:
    """Structured outputs of a posterior training run."""

    history: Mapping[str, Any] = field(default_factory=dict)
    validation_metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorSamples:
    """Posterior samples and optional per-sample diagnostics."""

    samples: Any
    log_probabilities: Any = None
    weights: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorBatch:
    """Training batch passed to amortized posterior estimators."""

    parameters: Any
    observations: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Optional per-sample loss weights (x-space focal reweighting). ``None``
    # means uniform. Reweighting in x does not bias the learned conditional
    # p(theta|x); it only reallocates capacity across observation space.
    sample_weights: Any = None


class PosteriorEstimator(abc.ABC):
    """Backend-agnostic interface for amortized posterior models."""

    @abc.abstractmethod
    def fit(self, dataset: Any) -> TrainingArtifacts:
        """Train the posterior estimator on a simulation dataset."""

    @abc.abstractmethod
    def encode_observation(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        """Encode a structured observation into the estimator input space."""

    @abc.abstractmethod
    def sample_posterior(
        self,
        observation: EncodedObservation,
        n_samples: int,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Sample the amortized posterior for one encoded observation."""

    def log_prob(self, observation: EncodedObservation, parameters: Any) -> Any:
        """Evaluate posterior log-density when supported by the backend."""
        raise NotImplementedError(
            "This posterior estimator does not expose normalized posterior log-probabilities."
        )

    @abc.abstractmethod
    def save(self, output_directory: str) -> None:
        """Persist trained model weights and metadata."""

    @classmethod
    @abc.abstractmethod
    def load(cls, input_directory: str) -> "PosteriorEstimator":
        """Restore a saved estimator from disk."""


class PersistentPosteriorEstimator(PosteriorEstimator):
    """Shared persistence helper for estimator backends with on-disk artifacts."""

    estimator_family = "persistent_estimator"
    metadata_schema_version = POSTERIOR_METADATA_SCHEMA_VERSION

    def __init__(
        self,
        parameter_dim: int,
        parameter_space: str = "unconstrained",
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
    ):
        self.parameter_dim = int(parameter_dim)
        self.parameter_space = parameter_space
        self.seed = int(seed)
        self.task_metadata = dict(task_metadata or {})
        self.training_artifacts: TrainingArtifacts | None = None
        self.task_name: str | None = self.task_metadata.get("task_name")
        self.task_version: str | None = None
        self.task_fingerprint: str | None = None
        self.observation_schema: Mapping[str, Any] = {}
        self.preprocessing_metadata_payload: Mapping[str, Any] = {}
        self.artifact_metadata: ArtifactMetadata | None = None

    @abc.abstractmethod
    def _build_estimator_config(self) -> dict[str, Any]:
        """Return backend-specific configuration for metadata persistence."""

    @classmethod
    @abc.abstractmethod
    def from_serialized_metadata(
        cls, metadata: Mapping[str, Any]
    ) -> "PersistentPosteriorEstimator":
        """Rebuild an estimator instance from persisted metadata only."""

    @abc.abstractmethod
    def save_backend_state(self, output_path: Path) -> None:
        """Persist backend-specific model state into the output directory."""

    @abc.abstractmethod
    def load_backend_state(self, input_path: Path) -> None:
        """Restore backend-specific model state from the input directory."""

    @staticmethod
    def _load_training_artifacts(
        metadata: Mapping[str, Any],
    ) -> TrainingArtifacts | None:
        training_artifacts = metadata.get("training_artifacts")
        if training_artifacts is None:
            return None

        return TrainingArtifacts(
            history=dict(training_artifacts.get("history", {})),
            validation_metrics=dict(training_artifacts.get("validation_metrics", {})),
            metadata=dict(training_artifacts.get("metadata", {})),
        )

    def hydrate_loaded_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.task_name = metadata.get("task_name")
        self.task_version = metadata.get("task_version")
        self.task_fingerprint = metadata.get("task_fingerprint")
        self.observation_schema = dict(metadata.get("observation_schema", {}))
        self.preprocessing_metadata_payload = dict(
            metadata.get("preprocessing_metadata_payload", {})
        )
        artifact_metadata = metadata.get("artifact_metadata")
        if artifact_metadata is not None:
            self.artifact_metadata = ArtifactMetadata.from_payload(artifact_metadata)
        self.training_artifacts = self._load_training_artifacts(metadata)

    def _build_artifact_metadata_payload(self) -> dict[str, Any]:
        training_metadata = (
            {}
            if self.training_artifacts is None
            else {
                "history": dict(self.training_artifacts.history),
                "validation_metrics": dict(self.training_artifacts.validation_metrics),
                "training_metadata": dict(self.training_artifacts.metadata),
            }
        )
        return {
            "parameter_dim": self.parameter_dim,
            "parameter_space": self.parameter_space,
            "estimator_config_summary": self._build_estimator_config(),
            "task_metadata": dict(self.task_metadata),
            **training_metadata,
        }

    def build_artifact_metadata(self, version: str) -> ArtifactMetadata:
        """Assemble registry metadata for the currently trained estimator."""
        artifact_metadata = ArtifactMetadata(
            role=ArtifactRole.POSTERIOR_MODEL,
            task_name=self.task_name
            or self.task_metadata.get("task_name", "unknown_task"),
            version=version,
            estimator_family=self.estimator_family,
            task_version=self.task_version,
            task_fingerprint=self.task_fingerprint,
            observation_schema=dict(self.observation_schema),
            preprocessing_metadata=dict(self.preprocessing_metadata_payload),
            metadata=self._build_artifact_metadata_payload(),
        )
        self.artifact_metadata = artifact_metadata
        return artifact_metadata

    def _build_serialized_metadata(
        self, artifact_metadata: ArtifactMetadata
    ) -> dict[str, Any]:
        return {
            "metadata_schema_version": self.metadata_schema_version,
            "estimator_family": self.estimator_family,
            "parameter_dim": self.parameter_dim,
            "parameter_space": self.parameter_space,
            "seed": self.seed,
            "task_metadata": dict(self.task_metadata),
            "task_name": self.task_name,
            "task_version": self.task_version,
            "task_fingerprint": self.task_fingerprint,
            "observation_schema": dict(self.observation_schema),
            "preprocessing_metadata_payload": dict(self.preprocessing_metadata_payload),
            "estimator_config": self._build_estimator_config(),
            "artifact_metadata": artifact_metadata.to_payload(),
            "training_artifacts": (
                None
                if self.training_artifacts is None
                else {
                    "history": dict(self.training_artifacts.history),
                    "validation_metrics": dict(
                        self.training_artifacts.validation_metrics
                    ),
                    "metadata": dict(self.training_artifacts.metadata),
                }
            ),
        }

    def save(
        self,
        output_directory: str,
        artifact_registry: ArtifactRegistry | None = None,
        artifact_version: str = "0.1.0",
    ) -> None:
        """Persist model weights, metadata, and optional artifact registration."""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_metadata = self.build_artifact_metadata(artifact_version)
        metadata = self._build_serialized_metadata(artifact_metadata)
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2)
        )
        self.save_backend_state(output_path)
        if artifact_registry is not None:
            artifact_registry.register(artifact_metadata, output_path.as_posix())

    @classmethod
    def load(cls, input_directory: str) -> "PersistentPosteriorEstimator":
        """Restore a saved persistent estimator from disk."""
        input_path = Path(input_directory)
        metadata = json.loads((input_path / "metadata.json").read_text())
        estimator = cls.from_serialized_metadata(metadata)
        estimator.load_backend_state(input_path)
        estimator.hydrate_loaded_metadata(metadata)
        return estimator


# ---------------------------------------------------------------------------
# Estimator-family registry + generic loader
# ---------------------------------------------------------------------------
#
# Concrete estimator modules (``flows.posterior``, ``flows.matching``) register
# their family here at import time, so a saved model can be reloaded without the
# caller naming its concrete class. The registry lives in this module -- next to
# the ``PosteriorEstimator`` interface it maps onto -- and holds no imports of the
# concrete estimators, so registration introduces no import cycle.

_ESTIMATOR_REGISTRY: dict[str, type[PosteriorEstimator]] = {}


def register_posterior_estimator(
    estimator_family: str, estimator_cls: type[PosteriorEstimator]
) -> None:
    """Register one estimator family for generic load dispatch."""
    _ESTIMATOR_REGISTRY[str(estimator_family)] = estimator_cls


def get_posterior_estimator_class(estimator_family: str) -> type[PosteriorEstimator]:
    """Return the registered estimator class for one estimator family."""
    try:
        return _ESTIMATOR_REGISTRY[str(estimator_family)]
    except KeyError as exc:
        known_families = ", ".join(sorted(_ESTIMATOR_REGISTRY)) or "<none>"
        raise KeyError(
            f"No posterior estimator is registered for estimator_family={estimator_family!r}. "
            f"Known families: {known_families}."
        ) from exc


def read_posterior_metadata(input_directory: str) -> dict[str, Any]:
    """Read the persisted metadata envelope for one saved estimator."""
    input_path = Path(input_directory)
    return dict(json.loads((input_path / "metadata.json").read_text()))


def resolve_estimator_family_from_metadata(metadata: dict[str, Any]) -> str:
    """Infer the estimator family from current or legacy saved metadata."""
    estimator_family = metadata.get("estimator_family")
    if estimator_family:
        return str(estimator_family)

    estimator_config = metadata.get("estimator_config")
    if (
        isinstance(estimator_config, dict)
        and estimator_config.get("flow_family") is not None
    ):
        return "conditional_flow"

    if metadata.get("flow_family") is not None:
        return "conditional_flow"

    raise KeyError(
        "Saved posterior metadata does not define 'estimator_family' and no legacy flow metadata was found."
    )


def load_posterior_estimator(input_directory: str) -> PosteriorEstimator:
    """Load a saved posterior estimator without naming its concrete class."""
    metadata = read_posterior_metadata(input_directory)
    estimator_family = resolve_estimator_family_from_metadata(metadata)
    estimator_cls = get_posterior_estimator_class(estimator_family)
    return estimator_cls.load(input_directory)


__all__ = [
    "POSTERIOR_METADATA_SCHEMA_VERSION",
    "PersistentPosteriorEstimator",
    "PosteriorBatch",
    "PosteriorEstimator",
    "PosteriorSamples",
    "TrainingArtifacts",
    "get_posterior_estimator_class",
    "load_posterior_estimator",
    "read_posterior_metadata",
    "register_posterior_estimator",
    "resolve_estimator_family_from_metadata",
]
