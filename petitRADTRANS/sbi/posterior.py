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
from .flows import ConditionalAffineFlow, ConditionalSplineFlow
from .observation import EncodedObservation, ObservationBlock
from .training import EarlyStoppingConfig, SBITrainer, TrainingConfig


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


class _PosteriorModel(eqx.Module):
    encoder: HierarchicalObservationEncoder
    flow: Any


class ConditionalFlowPosterior(PosteriorEstimator):
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
        self.parameter_dim = int(parameter_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_coupling_layers = int(num_coupling_layers)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.parameter_space = parameter_space
        self.flow_family = str(flow_family)
        self.effective_flow_family = self.flow_family
        self.num_spline_bins = int(num_spline_bins)
        self.spline_bound = float(spline_bound)
        self.early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.seed = int(seed)
        self.task_metadata = dict(task_metadata or {})
        self.training_artifacts: TrainingArtifacts | None = None
        self.task_name: str | None = self.task_metadata.get('task_name')
        self.task_version: str | None = None
        self.task_fingerprint: str | None = None
        self.observation_schema: Mapping[str, Any] = {}
        self.preprocessing_metadata_payload: Mapping[str, Any] = {}
        self.artifact_metadata: ArtifactMetadata | None = None

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

    def build_artifact_metadata(self, version: str) -> ArtifactMetadata:
        """Assemble registry metadata for the currently trained posterior.

        Parameters
        ----------
        version:
            Semantic or user-defined version string to assign to the persisted
            posterior artifact.

        Returns
        -------
        ArtifactMetadata
            Metadata bundle containing task, preprocessing, and training
            provenance for artifact registration.
        """
        training_metadata = {} if self.training_artifacts is None else {
            'history': dict(self.training_artifacts.history),
            'validation_metrics': dict(self.training_artifacts.validation_metrics),
            'training_metadata': dict(self.training_artifacts.metadata),
        }
        artifact_metadata = ArtifactMetadata(
            role=ArtifactRole.POSTERIOR_MODEL,
            task_name=self.task_name or self.task_metadata.get('task_name', 'unknown_task'),
            version=version,
            task_version=self.task_version,
            task_fingerprint=self.task_fingerprint,
            observation_schema=dict(self.observation_schema),
            preprocessing_metadata=dict(self.preprocessing_metadata_payload),
            metadata={
                'parameter_dim': self.parameter_dim,
                'parameter_space': self.parameter_space,
                'flow_family': self.flow_family,
                'effective_flow_family': self.effective_flow_family,
                'num_coupling_layers': self.num_coupling_layers,
                'num_spline_bins': self.num_spline_bins,
                'spline_bound': self.spline_bound,
                'task_metadata': dict(self.task_metadata),
                **training_metadata,
            },
        )
        self.artifact_metadata = artifact_metadata
        return artifact_metadata

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

    def save(
        self,
        output_directory: str,
        artifact_registry: ArtifactRegistry | None = None,
        artifact_version: str = '0.1.0',
    ) -> None:
        """Persist model weights, metadata, and optional artifact registration.

        Parameters
        ----------
        output_directory:
            Directory where ``metadata.json`` and serialized model weights are
            written.
        artifact_registry:
            Optional artifact registry that should receive a registration entry
            for the saved posterior.
        artifact_version:
            Version string used for the generated artifact metadata.

        Returns
        -------
        None

        Notes
        -----
        The saved metadata includes task fingerprinting, preprocessing payloads,
        effective flow-family details, and any available training artifacts.
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        artifact_metadata = self.build_artifact_metadata(artifact_version)
        metadata = {
            'parameter_dim': self.parameter_dim,
            'embedding_dim': self.embedding_dim,
            'num_coupling_layers': self.num_coupling_layers,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'parameter_space': self.parameter_space,
            'flow_family': self.flow_family,
            'effective_flow_family': self.effective_flow_family,
            'num_spline_bins': self.num_spline_bins,
            'spline_bound': self.spline_bound,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'checkpoint_directory': self.checkpoint_directory,
            'checkpoint_backend': self.checkpoint_backend,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'seed': self.seed,
            'task_metadata': dict(self.task_metadata),
            'task_name': self.task_name,
            'task_version': self.task_version,
            'task_fingerprint': self.task_fingerprint,
            'observation_schema': dict(self.observation_schema),
            'preprocessing_metadata_payload': dict(self.preprocessing_metadata_payload),
            'artifact_metadata': artifact_metadata.to_payload(),
            'training_artifacts': None if self.training_artifacts is None else {
                'history': dict(self.training_artifacts.history),
                'validation_metrics': dict(self.training_artifacts.validation_metrics),
                'metadata': dict(self.training_artifacts.metadata),
            },
        }
        (output_path / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2))
        eqx.tree_serialise_leaves(output_path / 'model.eqx', self.model)
        if artifact_registry is not None:
            artifact_registry.register(artifact_metadata, output_path.as_posix())

    @classmethod
    def load(cls, input_directory: str) -> 'ConditionalFlowPosterior':
        """Restore a saved conditional-flow posterior from disk.

        Parameters
        ----------
        input_directory:
            Directory containing ``metadata.json`` and ``model.eqx``.

        Returns
        -------
        ConditionalFlowPosterior
            Reconstructed posterior instance with saved weights and metadata.
        """
        input_path = Path(input_directory)
        metadata = json.loads((input_path / 'metadata.json').read_text())
        posterior = cls(
            parameter_dim=int(metadata['parameter_dim']),
            embedding_dim=int(metadata['embedding_dim']),
            num_coupling_layers=int(metadata['num_coupling_layers']),
            hidden_dim=int(metadata['hidden_dim']),
            learning_rate=float(metadata['learning_rate']),
            batch_size=int(metadata['batch_size']),
            num_epochs=int(metadata['num_epochs']),
            parameter_space=str(metadata['parameter_space']),
            flow_family=str(metadata.get('flow_family', 'affine')),
            num_spline_bins=int(metadata.get('num_spline_bins', 8)),
            spline_bound=float(metadata.get('spline_bound', 5.0)),
            early_stopping_patience=metadata.get('early_stopping_patience'),
            early_stopping_min_delta=float(metadata.get('early_stopping_min_delta', 0.0)),
            checkpoint_directory=metadata.get('checkpoint_directory'),
            checkpoint_backend=str(metadata.get('checkpoint_backend', 'auto')),
            resume_from_checkpoint=bool(metadata.get('resume_from_checkpoint', False)),
            seed=int(metadata['seed']),
            task_metadata=dict(metadata.get('task_metadata', {})),
        )
        posterior.model = eqx.tree_deserialise_leaves(input_path / 'model.eqx', posterior.model)
        posterior.effective_flow_family = str(metadata.get('effective_flow_family', posterior.effective_flow_family))
        posterior.task_name = metadata.get('task_name')
        posterior.task_version = metadata.get('task_version')
        posterior.task_fingerprint = metadata.get('task_fingerprint')
        posterior.observation_schema = dict(metadata.get('observation_schema', {}))
        posterior.preprocessing_metadata_payload = dict(metadata.get('preprocessing_metadata_payload', {}))
        artifact_metadata = metadata.get('artifact_metadata')
        if artifact_metadata is not None:
            posterior.artifact_metadata = ArtifactMetadata.from_payload(artifact_metadata)
        training_artifacts = metadata.get('training_artifacts')
        if training_artifacts is not None:
            posterior.training_artifacts = TrainingArtifacts(
                history=dict(training_artifacts.get('history', {})),
                validation_metrics=dict(training_artifacts.get('validation_metrics', {})),
                metadata=dict(training_artifacts.get('metadata', {})),
            )
        return posterior


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

    @classmethod
    def load(cls, input_directory: str) -> 'ConditionalSplineFlowPosterior':
        """Restore a saved spline posterior from disk.

        Parameters
        ----------
        input_directory:
            Directory containing a previously saved spline posterior.

        Returns
        -------
        ConditionalSplineFlowPosterior
            Restored spline posterior with serialized metadata and weights.
        """
        input_path = Path(input_directory)
        metadata = json.loads((input_path / 'metadata.json').read_text())
        posterior = cls(
            parameter_dim=int(metadata['parameter_dim']),
            embedding_dim=int(metadata['embedding_dim']),
            num_coupling_layers=int(metadata['num_coupling_layers']),
            hidden_dim=int(metadata['hidden_dim']),
            learning_rate=float(metadata['learning_rate']),
            batch_size=int(metadata['batch_size']),
            num_epochs=int(metadata['num_epochs']),
            parameter_space=str(metadata['parameter_space']),
            flow_family='spline',
            num_spline_bins=int(metadata.get('num_spline_bins', 8)),
            spline_bound=float(metadata.get('spline_bound', 5.0)),
            early_stopping_patience=metadata.get('early_stopping_patience'),
            early_stopping_min_delta=float(metadata.get('early_stopping_min_delta', 0.0)),
            checkpoint_directory=metadata.get('checkpoint_directory'),
            checkpoint_backend=str(metadata.get('checkpoint_backend', 'auto')),
            resume_from_checkpoint=bool(metadata.get('resume_from_checkpoint', False)),
            seed=int(metadata['seed']),
            task_metadata=dict(metadata.get('task_metadata', {})),
        )
        posterior.model = eqx.tree_deserialise_leaves(input_path / 'model.eqx', posterior.model)
        posterior.effective_flow_family = str(metadata.get('effective_flow_family', posterior.effective_flow_family))
        posterior.task_name = metadata.get('task_name')
        posterior.task_version = metadata.get('task_version')
        posterior.task_fingerprint = metadata.get('task_fingerprint')
        posterior.observation_schema = dict(metadata.get('observation_schema', {}))
        posterior.preprocessing_metadata_payload = dict(metadata.get('preprocessing_metadata_payload', {}))
        artifact_metadata = metadata.get('artifact_metadata')
        if artifact_metadata is not None:
            posterior.artifact_metadata = ArtifactMetadata.from_payload(artifact_metadata)
        training_artifacts = metadata.get('training_artifacts')
        if training_artifacts is not None:
            posterior.training_artifacts = TrainingArtifacts(
                history=dict(training_artifacts.get('history', {})),
                validation_metrics=dict(training_artifacts.get('validation_metrics', {})),
                metadata=dict(training_artifacts.get('metadata', {})),
            )
        return posterior


__all__ = [
    'ConditionalFlowPosterior',
    'ConditionalSplineFlowPosterior',
    'PosteriorBatch',
    'PosteriorEstimator',
    'PosteriorSamples',
    'TrainingArtifacts',
]
