"""Flow-matching posterior estimator (experimental)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .encoders import HierarchicalObservationEncoder
from .observation import EncodedObservation, ObservationBlock
from .posterior_base import (
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorSamples,
    TrainingArtifacts,
)
from .training import EarlyStoppingConfig, SBITrainer, TrainingConfig


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


class FlowMatchingPosterior(PersistentPosteriorEstimator):
    """Conditional flow-matching posterior skeleton.

    .. warning::

        This estimator is **experimental**. It does not expose ``log_prob``
        and the ODE integration scheme is a simple midpoint rule.  Expect
        the API and numerical behaviour to change in future releases.

    This estimator family trains a conditional vector field on straight-line
    interpolation paths between Gaussian noise and target parameters, then
    generates posterior samples by integrating the learned field from noise to
    the terminal parameter state.
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
        warnings.warn(
            'FlowMatchingPosterior is experimental. Its API and numerical '
            'behaviour may change in future releases.',
            stacklevel=2,
        )
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
        batch_indices = batch.metadata.get('batch_indices', [0])
        batch_seed = int(sum(batch_indices)) % (2**31) if batch_indices else 0
        base_key = jax.random.PRNGKey(batch_seed)
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

    def _build_serialized_metadata(self, artifact_metadata) -> dict[str, Any]:
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
    'FlowMatchingPosterior',
]
