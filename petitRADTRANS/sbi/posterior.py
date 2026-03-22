"""Posterior estimator interfaces and concrete flow backends."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import ArtifactMetadata, ArtifactRegistry, ArtifactRole
from .encoders import HierarchicalObservationEncoder
from .estimator_registry import register_posterior_estimator
from .flows import ConditionalAffineFlow, ConditionalSplineFlow
from .observation import EncodedObservation, ObservationBlock
from .training import EarlyStoppingConfig, SBITrainer, TrainingConfig


POSTERIOR_METADATA_SCHEMA_VERSION = '0.2.0'


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
        raise NotImplementedError('This posterior estimator does not expose normalized posterior log-probabilities.')

    @abc.abstractmethod
    def save(self, output_directory: str) -> None:
        """Persist trained model weights and metadata."""

    @classmethod
    @abc.abstractmethod
    def load(cls, input_directory: str) -> 'PosteriorEstimator':
        """Restore a saved estimator from disk."""


class PersistentPosteriorEstimator(PosteriorEstimator):
    """Shared persistence helper for estimator backends with on-disk artifacts."""

    estimator_family = 'persistent_estimator'
    metadata_schema_version = POSTERIOR_METADATA_SCHEMA_VERSION

    def __init__(
        self,
        parameter_dim: int,
        parameter_space: str = 'unconstrained',
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
    ):
        self.parameter_dim = int(parameter_dim)
        self.parameter_space = parameter_space
        self.seed = int(seed)
        self.task_metadata = dict(task_metadata or {})
        self.training_artifacts: TrainingArtifacts | None = None
        self.task_name: str | None = self.task_metadata.get('task_name')
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
    def from_serialized_metadata(cls, metadata: Mapping[str, Any]) -> 'PersistentPosteriorEstimator':
        """Rebuild an estimator instance from persisted metadata only."""

    @abc.abstractmethod
    def save_backend_state(self, output_path: Path) -> None:
        """Persist backend-specific model state into the output directory."""

    @abc.abstractmethod
    def load_backend_state(self, input_path: Path) -> None:
        """Restore backend-specific model state from the input directory."""

    @staticmethod
    def _load_training_artifacts(metadata: Mapping[str, Any]) -> TrainingArtifacts | None:
        training_artifacts = metadata.get('training_artifacts')
        if training_artifacts is None:
            return None

        return TrainingArtifacts(
            history=dict(training_artifacts.get('history', {})),
            validation_metrics=dict(training_artifacts.get('validation_metrics', {})),
            metadata=dict(training_artifacts.get('metadata', {})),
        )

    def hydrate_loaded_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.task_name = metadata.get('task_name')
        self.task_version = metadata.get('task_version')
        self.task_fingerprint = metadata.get('task_fingerprint')
        self.observation_schema = dict(metadata.get('observation_schema', {}))
        self.preprocessing_metadata_payload = dict(metadata.get('preprocessing_metadata_payload', {}))
        artifact_metadata = metadata.get('artifact_metadata')
        if artifact_metadata is not None:
            self.artifact_metadata = ArtifactMetadata.from_payload(artifact_metadata)
        self.training_artifacts = self._load_training_artifacts(metadata)

    def _build_artifact_metadata_payload(self) -> dict[str, Any]:
        training_metadata = {} if self.training_artifacts is None else {
            'history': dict(self.training_artifacts.history),
            'validation_metrics': dict(self.training_artifacts.validation_metrics),
            'training_metadata': dict(self.training_artifacts.metadata),
        }
        return {
            'parameter_dim': self.parameter_dim,
            'parameter_space': self.parameter_space,
            'estimator_config_summary': self._build_estimator_config(),
            'task_metadata': dict(self.task_metadata),
            **training_metadata,
        }

    def build_artifact_metadata(self, version: str) -> ArtifactMetadata:
        """Assemble registry metadata for the currently trained estimator."""
        artifact_metadata = ArtifactMetadata(
            role=ArtifactRole.POSTERIOR_MODEL,
            task_name=self.task_name or self.task_metadata.get('task_name', 'unknown_task'),
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

    def _build_serialized_metadata(self, artifact_metadata: ArtifactMetadata) -> dict[str, Any]:
        return {
            'metadata_schema_version': self.metadata_schema_version,
            'estimator_family': self.estimator_family,
            'parameter_dim': self.parameter_dim,
            'parameter_space': self.parameter_space,
            'seed': self.seed,
            'task_metadata': dict(self.task_metadata),
            'task_name': self.task_name,
            'task_version': self.task_version,
            'task_fingerprint': self.task_fingerprint,
            'observation_schema': dict(self.observation_schema),
            'preprocessing_metadata_payload': dict(self.preprocessing_metadata_payload),
            'estimator_config': self._build_estimator_config(),
            'artifact_metadata': artifact_metadata.to_payload(),
            'training_artifacts': None if self.training_artifacts is None else {
                'history': dict(self.training_artifacts.history),
                'validation_metrics': dict(self.training_artifacts.validation_metrics),
                'metadata': dict(self.training_artifacts.metadata),
            },
        }

    def save(
        self,
        output_directory: str,
        artifact_registry: ArtifactRegistry | None = None,
        artifact_version: str = '0.1.0',
    ) -> None:
        """Persist model weights, metadata, and optional artifact registration."""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_metadata = self.build_artifact_metadata(artifact_version)
        metadata = self._build_serialized_metadata(artifact_metadata)
        (output_path / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2))
        self.save_backend_state(output_path)
        if artifact_registry is not None:
            artifact_registry.register(artifact_metadata, output_path.as_posix())

    @classmethod
    def load(cls, input_directory: str) -> 'PersistentPosteriorEstimator':
        """Restore a saved persistent estimator from disk."""
        input_path = Path(input_directory)
        metadata = json.loads((input_path / 'metadata.json').read_text())
        estimator = cls.from_serialized_metadata(metadata)
        estimator.load_backend_state(input_path)
        estimator.hydrate_loaded_metadata(metadata)
        return estimator


class _PosteriorModel(eqx.Module):
    encoder: HierarchicalObservationEncoder
    flow: Any


class _FlowMatchingVelocityField(eqx.Module):
    """Small conditional vector field used by the flow-matching skeleton."""

    network: eqx.nn.MLP

    def __init__(self, parameter_dim: int, context_dim: int, hidden_dim: int, depth: int, key: jax.Array):
        self.network = eqx.nn.MLP(
            in_size=int(parameter_dim) + int(context_dim) + 1,
            out_size=int(parameter_dim),
            width_size=int(hidden_dim),
            depth=int(depth),
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, state: jnp.ndarray, time: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        time_input = jnp.reshape(jnp.asarray(time, dtype=jnp.float32), (1,))
        inputs = jnp.concatenate([
            jnp.asarray(state, dtype=jnp.float32),
            jnp.asarray(context, dtype=jnp.float32),
            time_input,
        ])
        return self.network(inputs)


class _FlowMatchingModel(eqx.Module):
    encoder: HierarchicalObservationEncoder
    velocity_field: _FlowMatchingVelocityField


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
        Conditional density-transform family. ``'spline'`` is the default and
        ``'affine'`` is kept as a lighter-weight alternative.
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

    estimator_family = 'conditional_flow'

    def __init__(
        self,
        parameter_dim: int,
        embedding_dim: int = 128,
        num_coupling_layers: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 5,
        parameter_space: str = 'unconstrained',
        flow_family: str = 'spline',
        num_spline_bins: int = 8,
        spline_bound: float = 5.0,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        checkpoint_directory: str | None = None,
        checkpoint_backend: str = 'auto',
        resume_from_checkpoint: bool = False,
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
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
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.flow_family = str(flow_family)
        self.effective_flow_family = self.flow_family
        self.num_spline_bins = int(num_spline_bins)
        self.spline_bound = float(spline_bound)
        self.early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)

        encoder_key, flow_key = jax.random.split(jax.random.PRNGKey(self.seed))
        self.model = _PosteriorModel(
            encoder=HierarchicalObservationEncoder(
                embedding_dim=self.embedding_dim,
                key=encoder_key,
            ),
            flow=self._build_flow(flow_key),
        )

    def _build_flow(self, key: jax.Array) -> Any:
        if self.flow_family == 'affine':
            self.effective_flow_family = 'affine'
            return ConditionalAffineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                key=key,
            )
        if self.flow_family == 'spline':
            self.effective_flow_family = 'spline'
            return ConditionalSplineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                num_bins=self.num_spline_bins,
                bound=self.spline_bound,
                key=key,
            )
        raise ValueError(f"Unsupported flow_family: {self.flow_family!r}")

    @staticmethod
    def _batch_embeddings(model: _PosteriorModel, observations: Any) -> jnp.ndarray:
        if isinstance(observations, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(observations, dtype=jnp.float32)

        embeddings = [model.encoder.encode(blocks).embedding for blocks in observations]
        return jnp.stack(embeddings, axis=0)

    @staticmethod
    def _loss(model: _PosteriorModel, batch: PosteriorBatch) -> jnp.ndarray:
        parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
        embeddings = ConditionalFlowPosterior._batch_embeddings(model, batch.observations)
        log_probabilities = jax.vmap(model.flow.log_prob)(parameters, embeddings)
        log_probabilities = jnp.nan_to_num(log_probabilities, nan=-1e6, neginf=-1e6, posinf=1e6)
        return -jnp.mean(log_probabilities)

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
        manifest = getattr(getattr(dataset, 'dataset', None), 'manifest', None)
        if manifest is not None:
            self.task_name = str(getattr(manifest, 'task_name', self.task_name or 'unknown_task'))
            self.task_fingerprint = getattr(manifest, 'task_fingerprint', None)
            self.observation_schema = dict(getattr(manifest, 'observation_schema', {}))
        preprocessing_metadata = getattr(dataset, 'preprocessing_metadata', None)
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
            )
        )
        self.model, history, validation_metrics, training_metadata = trainer.fit(
            self.model,
            dataset,
            self._loss,
        )
        self.training_artifacts = TrainingArtifacts(
            history=history,
            validation_metrics=validation_metrics,
            metadata={
                'parameter_dim': self.parameter_dim,
                'parameter_space': self.parameter_space,
                'task_metadata': dict(self.task_metadata),
                'task_name': self.task_name,
                'task_fingerprint': self.task_fingerprint,
                'flow_family': self.flow_family,
                'effective_flow_family': self.effective_flow_family,
                'num_spline_bins': self.num_spline_bins,
                'spline_bound': self.spline_bound,
                **training_metadata,
            },
        )
        return self.training_artifacts

    def _build_estimator_config(self) -> dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'num_coupling_layers': self.num_coupling_layers,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'flow_family': self.flow_family,
            'effective_flow_family': self.effective_flow_family,
            'num_spline_bins': self.num_spline_bins,
            'spline_bound': self.spline_bound,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'checkpoint_directory': self.checkpoint_directory,
            'checkpoint_backend': self.checkpoint_backend,
            'resume_from_checkpoint': self.resume_from_checkpoint,
        }

    def _build_serialized_metadata(self, artifact_metadata: ArtifactMetadata) -> dict[str, Any]:
        metadata = super()._build_serialized_metadata(artifact_metadata)
        metadata.update({
            'embedding_dim': self.embedding_dim,
            'num_coupling_layers': self.num_coupling_layers,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'flow_family': self.flow_family,
            'effective_flow_family': self.effective_flow_family,
            'num_spline_bins': self.num_spline_bins,
            'spline_bound': self.spline_bound,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'checkpoint_directory': self.checkpoint_directory,
            'checkpoint_backend': self.checkpoint_backend,
            'resume_from_checkpoint': self.resume_from_checkpoint,
        })
        return metadata

    @staticmethod
    def _resolve_estimator_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
        estimator_config = metadata.get('estimator_config')
        if isinstance(estimator_config, Mapping):
            return dict(estimator_config)

        return {
            'embedding_dim': metadata['embedding_dim'],
            'num_coupling_layers': metadata['num_coupling_layers'],
            'hidden_dim': metadata['hidden_dim'],
            'learning_rate': metadata['learning_rate'],
            'batch_size': metadata['batch_size'],
            'num_epochs': metadata['num_epochs'],
            'flow_family': metadata.get('flow_family', 'affine'),
            'effective_flow_family': metadata.get('effective_flow_family', metadata.get('flow_family', 'affine')),
            'num_spline_bins': metadata.get('num_spline_bins', 8),
            'spline_bound': metadata.get('spline_bound', 5.0),
            'early_stopping_patience': metadata.get('early_stopping_patience'),
            'early_stopping_min_delta': metadata.get('early_stopping_min_delta', 0.0),
            'checkpoint_directory': metadata.get('checkpoint_directory'),
            'checkpoint_backend': metadata.get('checkpoint_backend', 'auto'),
            'resume_from_checkpoint': metadata.get('resume_from_checkpoint', False),
        }

    def hydrate_loaded_metadata(self, metadata: Mapping[str, Any]) -> None:
        super().hydrate_loaded_metadata(metadata)
        self.effective_flow_family = str(
            self._resolve_estimator_config(metadata).get('effective_flow_family', self.effective_flow_family)
        )

    @classmethod
    def from_serialized_metadata(cls, metadata: Mapping[str, Any]) -> 'ConditionalFlowPosterior':
        estimator_config = cls._resolve_estimator_config(metadata)
        return cls(
            parameter_dim=int(metadata['parameter_dim']),
            embedding_dim=int(estimator_config['embedding_dim']),
            num_coupling_layers=int(estimator_config['num_coupling_layers']),
            hidden_dim=int(estimator_config['hidden_dim']),
            learning_rate=float(estimator_config['learning_rate']),
            batch_size=int(estimator_config['batch_size']),
            num_epochs=int(estimator_config['num_epochs']),
            parameter_space=str(metadata['parameter_space']),
            flow_family=str(estimator_config.get('flow_family', 'affine')),
            num_spline_bins=int(estimator_config.get('num_spline_bins', 8)),
            spline_bound=float(estimator_config.get('spline_bound', 5.0)),
            early_stopping_patience=estimator_config.get('early_stopping_patience'),
            early_stopping_min_delta=float(estimator_config.get('early_stopping_min_delta', 0.0)),
            checkpoint_directory=estimator_config.get('checkpoint_directory'),
            checkpoint_backend=str(estimator_config.get('checkpoint_backend', 'auto')),
            resume_from_checkpoint=bool(estimator_config.get('resume_from_checkpoint', False)),
            seed=int(metadata['seed']),
            task_metadata=dict(metadata.get('task_metadata', {})),
        )

    def save_backend_state(self, output_path: Path) -> None:
        eqx.tree_serialise_leaves(output_path / 'model.eqx', self.model)

    def load_backend_state(self, input_path: Path) -> None:
        self.model = eqx.tree_deserialise_leaves(input_path / 'model.eqx', self.model)

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
            key=jax.random.PRNGKey(sample_seed),
        )
        return PosteriorSamples(samples=np.nan_to_num(np.asarray(samples), nan=0.0, posinf=1e6, neginf=-1e6))

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
            return jnp.nan_to_num(self.model.flow.log_prob(parameter_values, context), nan=-1e6, neginf=-1e6)

        return jax.vmap(
            lambda value: jnp.nan_to_num(self.model.flow.log_prob(value, context), nan=-1e6, neginf=-1e6)
        )(parameter_values)


class ConditionalSplineFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the spline flow backend.

    Notes
    -----
    This convenience subclass forces ``flow_family='spline'`` while keeping the
    rest of the :class:`ConditionalFlowPosterior` API unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault('flow_family', 'spline')
        super().__init__(*args, **kwargs)


class FlowMatchingPosterior(PersistentPosteriorEstimator):
    """Conditional flow-matching posterior skeleton.

    This estimator family trains a conditional vector field on straight-line
    interpolation paths between Gaussian noise and target parameters, then
    generates posterior samples by integrating the learned field from noise to
    the terminal parameter state.

    The implementation is intentionally lightweight: it provides a concrete,
    saveable, loadable estimator family that exercises the registry and shared
    persistence path without changing the public inference or calibration
    interfaces.

    TODO:  harden FlowMatchingPosterior into a more serious FMPE implementation
    with better time sampling, better integration, and improved numerical
    stabilization.
    """

    estimator_family = 'flow_matching'

    def __init__(
        self,
        parameter_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_velocity_layers: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 5,
        parameter_space: str = 'unconstrained',
        integration_steps: int = 32,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        checkpoint_directory: str | None = None,
        checkpoint_backend: str = 'auto',
        resume_from_checkpoint: bool = False,
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            parameter_dim=parameter_dim,
            parameter_space=parameter_space,
            seed=seed,
            task_metadata=task_metadata,
        )
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_velocity_layers = int(num_velocity_layers)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.integration_steps = int(integration_steps)
        self.early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)

        encoder_key, vector_field_key = jax.random.split(jax.random.PRNGKey(self.seed))
        self.model = _FlowMatchingModel(
            encoder=HierarchicalObservationEncoder(
                embedding_dim=self.embedding_dim,
                key=encoder_key,
            ),
            velocity_field=_FlowMatchingVelocityField(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                depth=self.num_velocity_layers,
                key=vector_field_key,
            ),
        )

    @staticmethod
    def _batch_embeddings(model: _FlowMatchingModel, observations: Any) -> jnp.ndarray:
        if isinstance(observations, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(observations, dtype=jnp.float32)

        embeddings = [model.encoder.encode(blocks).embedding for blocks in observations]
        return jnp.stack(embeddings, axis=0)

    @staticmethod
    def _loss(model: _FlowMatchingModel, batch: PosteriorBatch) -> jnp.ndarray:
        parameters = jnp.asarray(batch.parameters, dtype=jnp.float32)
        embeddings = jnp.nan_to_num(
            FlowMatchingPosterior._batch_embeddings(model, batch.observations),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        base_key = jax.random.PRNGKey(0)
        noise_key, time_key = jax.random.split(base_key)
        base_noise = jax.random.normal(noise_key, shape=parameters.shape, dtype=jnp.float32)
        interpolation_time = jax.random.uniform(
            time_key,
            shape=(parameters.shape[0], 1),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        interpolated_state = ((1.0 - interpolation_time) * base_noise) + (interpolation_time * parameters)
        target_velocity = parameters - base_noise
        predicted_velocity = jax.vmap(model.velocity_field)(
            interpolated_state,
            interpolation_time[:, 0],
            embeddings,
        )
        squared_error = jnp.square(predicted_velocity - target_velocity)
        return jnp.mean(jnp.nan_to_num(squared_error, nan=1e6, posinf=1e6, neginf=1e6))

    def fit(self, dataset: Any) -> TrainingArtifacts:
        manifest = getattr(getattr(dataset, 'dataset', None), 'manifest', None)
        if manifest is not None:
            self.task_name = str(getattr(manifest, 'task_name', self.task_name or 'unknown_task'))
            self.task_fingerprint = getattr(manifest, 'task_fingerprint', None)
            self.observation_schema = dict(getattr(manifest, 'observation_schema', {}))
        preprocessing_metadata = getattr(dataset, 'preprocessing_metadata', None)
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
            )
        )
        self.model, history, validation_metrics, training_metadata = trainer.fit(
            self.model,
            dataset,
            self._loss,
        )
        self.training_artifacts = TrainingArtifacts(
            history=history,
            validation_metrics=validation_metrics,
            metadata={
                'parameter_dim': self.parameter_dim,
                'parameter_space': self.parameter_space,
                'task_metadata': dict(self.task_metadata),
                'task_name': self.task_name,
                'task_fingerprint': self.task_fingerprint,
                'integration_steps': self.integration_steps,
                'num_velocity_layers': self.num_velocity_layers,
                **training_metadata,
            },
        )
        return self.training_artifacts

    def _build_estimator_config(self) -> dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_velocity_layers': self.num_velocity_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'integration_steps': self.integration_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'checkpoint_directory': self.checkpoint_directory,
            'checkpoint_backend': self.checkpoint_backend,
            'resume_from_checkpoint': self.resume_from_checkpoint,
        }

    @staticmethod
    def _resolve_estimator_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
        estimator_config = metadata.get('estimator_config')
        if isinstance(estimator_config, Mapping):
            return dict(estimator_config)

        return {
            'embedding_dim': metadata['embedding_dim'],
            'hidden_dim': metadata['hidden_dim'],
            'num_velocity_layers': metadata.get('num_velocity_layers', 3),
            'learning_rate': metadata['learning_rate'],
            'batch_size': metadata['batch_size'],
            'num_epochs': metadata['num_epochs'],
            'integration_steps': metadata.get('integration_steps', 32),
            'early_stopping_patience': metadata.get('early_stopping_patience'),
            'early_stopping_min_delta': metadata.get('early_stopping_min_delta', 0.0),
            'checkpoint_directory': metadata.get('checkpoint_directory'),
            'checkpoint_backend': metadata.get('checkpoint_backend', 'auto'),
            'resume_from_checkpoint': metadata.get('resume_from_checkpoint', False),
        }

    def _build_serialized_metadata(self, artifact_metadata: ArtifactMetadata) -> dict[str, Any]:
        metadata = super()._build_serialized_metadata(artifact_metadata)
        metadata.update({
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_velocity_layers': self.num_velocity_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'integration_steps': self.integration_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'checkpoint_directory': self.checkpoint_directory,
            'checkpoint_backend': self.checkpoint_backend,
            'resume_from_checkpoint': self.resume_from_checkpoint,
        })
        return metadata

    @classmethod
    def from_serialized_metadata(cls, metadata: Mapping[str, Any]) -> 'FlowMatchingPosterior':
        estimator_config = cls._resolve_estimator_config(metadata)
        return cls(
            parameter_dim=int(metadata['parameter_dim']),
            embedding_dim=int(estimator_config['embedding_dim']),
            hidden_dim=int(estimator_config['hidden_dim']),
            num_velocity_layers=int(estimator_config.get('num_velocity_layers', 3)),
            learning_rate=float(estimator_config['learning_rate']),
            batch_size=int(estimator_config['batch_size']),
            num_epochs=int(estimator_config['num_epochs']),
            parameter_space=str(metadata['parameter_space']),
            integration_steps=int(estimator_config.get('integration_steps', 32)),
            early_stopping_patience=estimator_config.get('early_stopping_patience'),
            early_stopping_min_delta=float(estimator_config.get('early_stopping_min_delta', 0.0)),
            checkpoint_directory=estimator_config.get('checkpoint_directory'),
            checkpoint_backend=str(estimator_config.get('checkpoint_backend', 'auto')),
            resume_from_checkpoint=bool(estimator_config.get('resume_from_checkpoint', False)),
            seed=int(metadata['seed']),
            task_metadata=dict(metadata.get('task_metadata', {})),
        )

    def save_backend_state(self, output_path: Path) -> None:
        eqx.tree_serialise_leaves(output_path / 'model.eqx', self.model)

    def load_backend_state(self, input_path: Path) -> None:
        self.model = eqx.tree_deserialise_leaves(input_path / 'model.eqx', self.model)

    def encode_observation(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        return self.model.encoder.encode(blocks)

    def sample_posterior(
        self,
        observation: EncodedObservation,
        n_samples: int,
        seed: int | None = None,
    ) -> PosteriorSamples:
        sample_seed = self.seed if seed is None else int(seed)
        state = jax.random.normal(
            jax.random.PRNGKey(sample_seed),
            shape=(int(n_samples), self.parameter_dim),
            dtype=jnp.float32,
        )
        context = jnp.nan_to_num(
            jnp.asarray(observation.embedding, dtype=jnp.float32),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        step_size = jnp.asarray(1.0 / float(self.integration_steps), dtype=jnp.float32)

        def integration_step(current_state: jnp.ndarray, step_index: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            time = (jnp.asarray(step_index, dtype=jnp.float32) + jnp.asarray(0.5, dtype=jnp.float32)) * step_size
            velocity = jax.vmap(lambda value: self.model.velocity_field(value, time, context))(current_state)
            velocity = jnp.asarray(velocity, dtype=jnp.float32)
            next_state = current_state + (step_size * velocity)
            return next_state, None

        state, _ = jax.lax.scan(
            integration_step,
            state,
            jnp.arange(self.integration_steps, dtype=jnp.int32),
        )
        return PosteriorSamples(samples=np.nan_to_num(np.asarray(state), nan=0.0, posinf=1e6, neginf=-1e6))

    def log_prob(self, observation: EncodedObservation, parameters: Any) -> Any:
        del observation, parameters
        raise NotImplementedError(
            'FlowMatchingPosterior currently exposes sampling only; normalized log-probabilities are not implemented.'
        )


__all__ = [
    'ConditionalFlowPosterior',
    'ConditionalSplineFlowPosterior',
    'FlowMatchingPosterior',
    'PersistentPosteriorEstimator',
    'POSTERIOR_METADATA_SCHEMA_VERSION',
    'PosteriorBatch',
    'PosteriorEstimator',
    'PosteriorSamples',
    'TrainingArtifacts',
]


register_posterior_estimator(ConditionalFlowPosterior.estimator_family, ConditionalFlowPosterior)
register_posterior_estimator(FlowMatchingPosterior.estimator_family, FlowMatchingPosterior)
