"""Conditional normalizing-flow posterior estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
from jax import Array, random, vmap
import jax
import jax.numpy as jnp
import numpy as np

from .encoders import HierarchicalObservationEncoder
from .flows import ConditionalAffineFlow, ConditionalSplineFlow
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

    estimator_family = "conditional_flow"

    def __init__(
        self,
        parameter_dim: int,
        embedding_dim: int = 128,
        num_coupling_layers: int = 4,
        hidden_dim: int = 128,
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
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)

        encoder_key, flow_key = random.split(random.PRNGKey(self.seed))
        self.model = _PosteriorModel(
            encoder=HierarchicalObservationEncoder(
                embedding_dim=self.embedding_dim,
                key=encoder_key,
            ),
            flow=self._build_flow(flow_key),
        )

    def _build_flow(self, key: Array) -> Any:
        if self.flow_family == "affine":
            self.effective_flow_family = "affine"
            return ConditionalAffineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                key=key,
            )
        if self.flow_family == "spline":
            self.effective_flow_family = "spline"
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
        embeddings = ConditionalFlowPosterior._batch_embeddings(
            model, batch.observations
        )
        log_probabilities = vmap(model.flow.log_prob)(parameters, embeddings)
        log_probabilities = jnp.nan_to_num(
            log_probabilities, nan=-1e6, neginf=-1e6, posinf=1e6
        )
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
                data_parallel=len(jax.devices()) > 1,
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
        }

    def _build_serialized_metadata(self, artifact_metadata) -> dict[str, Any]:
        metadata = super()._build_serialized_metadata(artifact_metadata)
        metadata.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_coupling_layers": self.num_coupling_layers,
                "hidden_dim": self.hidden_dim,
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
            }
        )
        return metadata

    @staticmethod
    def _resolve_estimator_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
        estimator_config = metadata.get("estimator_config")
        if isinstance(estimator_config, Mapping):
            return dict(estimator_config)

        return {
            "embedding_dim": metadata["embedding_dim"],
            "num_coupling_layers": metadata["num_coupling_layers"],
            "hidden_dim": metadata["hidden_dim"],
            "learning_rate": metadata["learning_rate"],
            "batch_size": metadata["batch_size"],
            "num_epochs": metadata["num_epochs"],
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
            return jnp.nan_to_num(
                self.model.flow.log_prob(parameter_values, context),
                nan=-1e6,
                neginf=-1e6,
            )

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


__all__ = [
    "ConditionalFlowPosterior",
    "ConditionalSplineFlowPosterior",
]
