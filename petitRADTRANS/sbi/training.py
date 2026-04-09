"""Training and checkpoint orchestration for SBI posterior models."""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dill
import equinox as eqx  # pyright: ignore[reportMissingImports]
import numpy as np
import optax  # pyright: ignore[reportMissingImports]


def _prestack_observation_blocks(blocks_batch: Any) -> Any:
    """Convert a list-of-ObservationBlock-lists to a JIT-compatible PreStackedObservations.

    Extracts numpy arrays from Python ``ObservationBlock`` objects *outside*
    the JAX JIT boundary so that the compiled training step sees only dense
    numeric arrays whose shapes (not values) determine the XLA cache key.

    Returns ``None`` when ``blocks_batch`` is not in list-of-lists format,
    leaving the caller to pass observations through unchanged.
    """
    if (
        not isinstance(blocks_batch, list)
        or not blocks_batch
        or not isinstance(blocks_batch[0], list)
    ):
        return None

    from .observation import PreStackedObservations

    def _resize(arr: Any, target: int, dtype: Any) -> np.ndarray:
        """Flatten and resize an array to exactly `target` elements."""
        flat = np.ravel(np.asarray(arr, dtype=dtype))
        if flat.shape[0] == target:
            return flat
        if flat.shape[0] > target:
            return flat[:target]
        return np.concatenate([flat, np.zeros(target - flat.shape[0], dtype=dtype)])

    first_blocks = blocks_batch[0]
    modalities = tuple(block.modality.value for block in first_blocks)

    stacked = []
    for j in range(len(first_blocks)):
        v_list, u_list, c_list, m_list = [], [], [], []
        for sample_blocks in blocks_batch:
            block = sample_blocks[j]
            v = np.ravel(np.asarray(block.values, dtype=np.float32))
            n_wl = v.shape[0]
            v_list.append(v)
            u_list.append(
                _resize(block.uncertainties, n_wl, np.float32)
                if block.uncertainties is not None
                else np.zeros(n_wl, dtype=np.float32)
            )
            c_list.append(
                _resize(block.coordinates, n_wl, np.float32)
                if block.coordinates is not None
                else np.zeros(n_wl, dtype=np.float32)
            )
            raw_m = (
                np.ravel(np.asarray(block.mask, dtype=np.bool_))
                if block.mask is not None
                else np.array([], dtype=np.bool_)
            )
            # Resize to n_wl — mirrors _safe_mask in encoders.py.
            # observation.mask from pRT may be a scalar or have a different
            # length than the values array (e.g. stored as a per-spectrum flag
            # rather than a per-wavelength boolean vector).
            if raw_m.shape[0] != n_wl:
                if raw_m.shape[0] > n_wl:
                    raw_m = raw_m[:n_wl]
                else:
                    raw_m = np.concatenate(
                        [raw_m, np.zeros(n_wl - raw_m.shape[0], dtype=np.bool_)]
                    )
            m_list.append(raw_m)
        stacked.append((
            np.stack(v_list),
            np.stack(u_list),
            np.stack(c_list),
            np.stack(m_list),
        ))

    return PreStackedObservations(stacked_blocks=tuple(stacked), modalities=modalities)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping policy for SBI training."""

    patience: int
    min_delta: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for posterior optimization."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    parameter_space: str = 'unconstrained'
    seed: int = 0
    shuffle_train: bool = True
    early_stopping: EarlyStoppingConfig | None = None
    checkpoint_directory: str | None = None
    checkpoint_backend: str = 'auto'
    resume_from_checkpoint: bool = False


class CheckpointBackend(abc.ABC):
    """Persistence backend for trainer checkpoints."""

    name: str

    @abc.abstractmethod
    def save(self, state: Any, output_directory: Path, metadata: dict[str, Any]) -> None:
        """Persist checkpoint state and metadata.

        Parameters
        ----------
        state:
            Serializable optimizer/model state payload.
        output_directory:
            Directory that should receive the checkpoint files.
        metadata:
            Small JSON-serializable metadata dictionary written next to the
            checkpoint payload.

        Returns
        -------
        None
        """

    @abc.abstractmethod
    def restore(self, template_state: Any, output_directory: Path) -> tuple[Any, dict[str, Any]]:
        """Restore persisted checkpoint state into the supplied template.

        Parameters
        ----------
        template_state:
            Structure used by some backends to describe the expected state
            layout during restoration.
        output_directory:
            Directory containing the serialized checkpoint payload.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Restored state object together with the deserialized checkpoint
            metadata dictionary.
        """


class EquinoxCheckpointBackend(CheckpointBackend):
    """Checkpoint backend backed by Equinox tree serialization."""

    name = 'equinox'

    def save(self, state: Any, output_directory: Path, metadata: dict[str, Any]) -> None:
        """Serialize checkpoint state with dill and save JSON metadata."""
        output_directory.mkdir(parents=True, exist_ok=True)
        with (output_directory / 'state.pkl').open('wb') as stream:
            dill.dump(state, stream)
        (output_directory / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2))

    def restore(self, template_state: Any, output_directory: Path) -> tuple[Any, dict[str, Any]]:
        """Load a dill-serialized checkpoint and its metadata."""
        del template_state
        metadata = json.loads((output_directory / 'metadata.json').read_text())
        with (output_directory / 'state.pkl').open('rb') as stream:
            state = dill.load(stream)
        return state, metadata


class OrbaxCheckpointBackend(CheckpointBackend):
    """Checkpoint backend that uses Orbax when available."""

    name = 'orbax'

    def __init__(self):
        try:
            import orbax.checkpoint as ocp  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - exercised when Orbax is installed
            raise RuntimeError('Orbax checkpointing requested, but orbax-checkpoint is not installed.') from error

        self._ocp = ocp
        self._checkpointer = ocp.PyTreeCheckpointer()

    def save(self, state: Any, output_directory: Path, metadata: dict[str, Any]) -> None:
        """Persist checkpoint state with Orbax and write JSON metadata."""
        output_directory.mkdir(parents=True, exist_ok=True)
        self._checkpointer.save(str(output_directory / 'state.orbax'), state, force=True)
        (output_directory / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True, indent=2))

    def restore(self, template_state: Any, output_directory: Path) -> tuple[Any, dict[str, Any]]:
        """Restore Orbax checkpoint state and metadata."""
        del template_state
        metadata = json.loads((output_directory / 'metadata.json').read_text())
        restored_state = self._checkpointer.restore(str(output_directory / 'state.orbax'))
        return restored_state, metadata


def instantiate_checkpoint_backend(name: str) -> CheckpointBackend:
    """Instantiate one concrete checkpoint backend by explicit name."""
    normalized = name.lower()
    if normalized == 'equinox':
        return EquinoxCheckpointBackend()
    if normalized == 'orbax':
        return OrbaxCheckpointBackend()
    raise ValueError(f"Unknown checkpoint backend '{name}'.")


def _is_orbax_unsupported_type_error(error: Exception) -> bool:
    message = str(error)
    return 'TypeHandler lookup failed' in message or 'Must register a TypeHandler' in message


def resolve_checkpoint_backend(preferred: str = 'auto') -> CheckpointBackend:
    """Resolve the requested checkpoint backend with graceful fallback.

    Parameters
    ----------
    preferred:
        Backend name. Supported values are ``'auto'``, ``'equinox'``, and
        ``'orbax'``.

    Returns
    -------
    CheckpointBackend
        Concrete checkpoint backend implementation.

    Notes
    -----
    ``'auto'`` prefers Orbax when installed and otherwise falls back to the
    dill-based Equinox backend.
    """

    normalized = preferred.lower()
    if normalized == 'equinox':
        return instantiate_checkpoint_backend('equinox')
    if normalized == 'orbax':
        return instantiate_checkpoint_backend('orbax')
    if normalized == 'auto':
        try:
            return instantiate_checkpoint_backend('orbax')
        except RuntimeError:
            return instantiate_checkpoint_backend('equinox')
    raise ValueError(f"Unknown checkpoint backend '{preferred}'.")


class SBITrainer:
    """Reusable optimization loop for amortized SBI posteriors."""

    def __init__(self, config: TrainingConfig):
        """Create one trainer for repeated posterior-fitting runs.

        Parameters
        ----------
        config:
            Optimization, checkpointing, and early-stopping configuration.

        Returns
        -------
        None
        """
        self.config = config
        self.checkpoint_directory = None if config.checkpoint_directory is None else Path(config.checkpoint_directory)
        self.checkpoint_backend = (
            resolve_checkpoint_backend(config.checkpoint_backend)
            if config.checkpoint_directory is not None
            else None
        )
        self._checkpoint_backend_fallback_reason: str | None = None

    @property
    def _latest_checkpoint_directory(self) -> Path | None:
        if self.checkpoint_directory is None:
            return None
        return self.checkpoint_directory / 'latest'

    @property
    def _best_checkpoint_directory(self) -> Path | None:
        if self.checkpoint_directory is None:
            return None
        return self.checkpoint_directory / 'best'

    def _save_checkpoint(self, checkpoint_kind: str, state: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.checkpoint_backend is None or self.checkpoint_directory is None:
            return

        checkpoint_path = (
            self._latest_checkpoint_directory
            if checkpoint_kind == 'latest'
            else self._best_checkpoint_directory
        )
        assert checkpoint_path is not None
        payload = {
            'checkpoint_kind': checkpoint_kind,
            **metadata,
        }

        try:
            payload['checkpoint_backend'] = self.checkpoint_backend.name
            if self._checkpoint_backend_fallback_reason is not None:
                payload['checkpoint_backend_fallback_reason'] = self._checkpoint_backend_fallback_reason
            self.checkpoint_backend.save(state, checkpoint_path, payload)
        except ValueError as error:
            if self.checkpoint_backend.name != 'orbax' or not _is_orbax_unsupported_type_error(error):
                raise

            self.checkpoint_backend = instantiate_checkpoint_backend('equinox')
            self._checkpoint_backend_fallback_reason = 'orbax_unsupported_type'
            payload['checkpoint_backend'] = self.checkpoint_backend.name
            payload['checkpoint_backend_fallback_reason'] = self._checkpoint_backend_fallback_reason
            self.checkpoint_backend.save(state, checkpoint_path, payload)

    def _restore_checkpoint(
        self,
        template_state: dict[str, Any],
        checkpoint_kind: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self.checkpoint_backend is None:
            return None

        checkpoint_path = (
            self._latest_checkpoint_directory
            if checkpoint_kind == 'latest'
            else self._best_checkpoint_directory
        )
        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        metadata_path = checkpoint_path / 'metadata.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            backend_name = str(metadata.get('checkpoint_backend', self.checkpoint_backend.name))
            checkpoint_backend = instantiate_checkpoint_backend(backend_name)
            self.checkpoint_backend = checkpoint_backend
            return checkpoint_backend.restore(template_state, checkpoint_path)

        return self.checkpoint_backend.restore(template_state, checkpoint_path)

    def fit(
        self,
        model: Any,
        dataset: Any,
        loss_fn: Callable[[Any, Any], Any],
    ) -> tuple[Any, dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
        """Optimize one posterior model against a dataset reader.

        Parameters
        ----------
        model:
            Trainable Equinox-style model to optimize.
        dataset:
            Reader object exposing ``iter_batches`` for train and optional
            validation splits.
        loss_fn:
            Callable receiving ``(model, batch)`` and returning the scalar loss
            to minimize.

        Returns
        -------
        tuple[Any, dict[str, list[float]], dict[str, list[float]], dict[str, Any]]
            Best model, train-loss history, validation-loss history, and a
            metadata dictionary describing checkpointing and stopping behavior.

        Notes
        -----
        The trainer monitors validation loss when a validation split exists and
        otherwise falls back to training loss for best-model selection.
        Checkpoints may be written after each epoch when enabled.
        """
        from .dataset import DatasetSplit

        optimizer = optax.adam(self.config.learning_rate)
        optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        train_loss_history: list[float] = []
        validation_loss_history: list[float] = []
        best_model = model
        best_metric = float('inf')
        best_epoch = -1
        patience_counter = 0
        start_epoch = 0
        resumed_from_checkpoint = False
        has_validation = dataset.dataset.manifest.splits.get('validation', 0) > 0

        template_state = {
            'model': model,
            'optimizer_state': optimizer_state,
            'history': {
                'train_loss': [],
                'validation_loss': [],
            },
            'start_epoch': 0,
            'best_epoch': -1,
            'best_metric': float('inf'),
            'patience_counter': 0,
        }

        if self.config.resume_from_checkpoint:
            restored = self._restore_checkpoint(template_state, 'latest')
            if restored is not None:
                restored_state, restored_metadata = restored
                model = restored_state['model']
                optimizer_state = restored_state['optimizer_state']
                train_loss_history = list(restored_state['history'].get('train_loss', []))
                validation_loss_history = list(restored_state['history'].get('validation_loss', []))
                start_epoch = int(restored_state.get('start_epoch', 0))
                best_epoch = int(restored_state.get('best_epoch', -1))
                best_metric = float(restored_state.get('best_metric', float('inf')))
                patience_counter = int(restored_state.get('patience_counter', 0))
                resumed_from_checkpoint = True

                restored_best = self._restore_checkpoint(template_state, 'best')
                if restored_best is not None:
                    best_model = restored_best[0]['model']
                else:
                    best_model = model

                if best_epoch < 0 and start_epoch > 0:
                    best_epoch = start_epoch - 1
                best_metric = float(restored_metadata.get('best_metric', best_metric))

        from .posterior_base import PosteriorBatch

        # Build the JIT-compiled training step once.  ``loss_fn`` and
        # ``optimizer`` are captured as constants in the closure; only the
        # model weights, optimizer state, and batch arrays are traced.
        @eqx.filter_jit
        def _jit_step(
            _model: Any,
            _opt_state: Any,
            _parameters: Any,
            _observations: Any,
        ) -> tuple:
            _batch = PosteriorBatch(parameters=_parameters, observations=_observations)
            _loss_val, _grads = eqx.filter_value_and_grad(loss_fn)(_model, _batch)
            _updates, _new_opt_state = optimizer.update(
                _grads,
                _opt_state,
                eqx.filter(_model, eqx.is_inexact_array),
            )
            return eqx.apply_updates(_model, _updates), _new_opt_state, _loss_val

        # Separate JIT-compiled evaluation step: same compilation path as
        # _jit_step but without gradients (cheaper and produces the same
        # compiled kernel for loss-only evaluation).
        @eqx.filter_jit
        def _jit_eval_loss(
            _model: Any,
            _parameters: Any,
            _observations: Any,
        ) -> Any:
            _batch = PosteriorBatch(parameters=_parameters, observations=_observations)
            return loss_fn(_model, _batch)

        for epoch in range(start_epoch, self.config.num_epochs):
            _epoch_start = time.monotonic()
            epoch_losses = []
            for batch in dataset.iter_batches(
                split=DatasetSplit.TRAIN,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_train,
                seed=self.config.seed + epoch,
                parameter_space=self.config.parameter_space,
            ):
                prestacked = _prestack_observation_blocks(batch.observations)
                _obs = prestacked if prestacked is not None else batch.observations
                model, optimizer_state, loss_value = _jit_step(
                    model,
                    optimizer_state,
                    batch.parameters,
                    _obs,
                )
                epoch_losses.append(float(loss_value))

            train_metric = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            if epoch < len(train_loss_history):
                train_loss_history[epoch] = train_metric
            else:
                train_loss_history.append(train_metric)

            validation_metric = float('nan')
            if has_validation:
                validation_losses = []
                for batch in dataset.iter_batches(
                    split=DatasetSplit.VALIDATION,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    parameter_space=self.config.parameter_space,
                ):
                    prestacked_val = _prestack_observation_blocks(batch.observations)
                    _val_obs = prestacked_val if prestacked_val is not None else batch.observations
                    validation_losses.append(float(_jit_eval_loss(model, batch.parameters, _val_obs)))
                validation_metric = float(np.mean(validation_losses)) if validation_losses else float('nan')
                if epoch < len(validation_loss_history):
                    validation_loss_history[epoch] = validation_metric
                else:
                    validation_loss_history.append(validation_metric)

            _elapsed = time.monotonic() - _epoch_start
            _val_str = (
                f" | val_loss: {validation_metric:.6f}" if has_validation else ""
            )
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs}"
                f" | train_loss: {train_metric:.6f}{_val_str}"
                f" | {_elapsed:.1f}s",
                flush=True,
            )

            monitor_metric = validation_metric if has_validation else train_metric
            improved = best_epoch < 0 or (
                np.isfinite(monitor_metric)
                and (
                    monitor_metric < best_metric - self._min_delta
                    or not np.isfinite(best_metric)
                )
            )
            if improved:
                best_metric = monitor_metric
                best_epoch = epoch
                best_model = model
                patience_counter = 0
                self._save_checkpoint(
                    'best',
                    {
                        'model': best_model,
                        'optimizer_state': optimizer_state,
                        'history': {
                            'train_loss': list(train_loss_history),
                            'validation_loss': list(validation_loss_history),
                        },
                        'start_epoch': epoch + 1,
                        'best_epoch': best_epoch,
                        'best_metric': best_metric,
                        'patience_counter': patience_counter,
                    },
                    {
                        'epoch': epoch + 1,
                        'best_metric': best_metric,
                        'checkpoint_backend': None if self.checkpoint_backend is None else self.checkpoint_backend.name,
                    },
                )
            else:
                patience_counter += 1

            self._save_checkpoint(
                'latest',
                {
                    'model': model,
                    'optimizer_state': optimizer_state,
                    'history': {
                        'train_loss': list(train_loss_history),
                        'validation_loss': list(validation_loss_history),
                    },
                    'start_epoch': epoch + 1,
                    'best_epoch': best_epoch,
                    'best_metric': best_metric,
                    'patience_counter': patience_counter,
                },
                {
                    'epoch': epoch + 1,
                    'best_metric': best_metric,
                    'checkpoint_backend': None if self.checkpoint_backend is None else self.checkpoint_backend.name,
                },
            )

            if self._should_stop(patience_counter):
                break

        return (
            best_model,
            {'train_loss': train_loss_history},
            {'validation_loss': validation_loss_history},
            {
                'best_epoch': best_epoch + 1 if best_epoch >= 0 else None,
                'best_metric': best_metric,
                'stopped_early': self._should_stop(patience_counter),
                'resumed_from_checkpoint': resumed_from_checkpoint,
                'completed_epochs': len(train_loss_history),
                'checkpoint_backend': None if self.checkpoint_backend is None else self.checkpoint_backend.name,
            },
        )

    @property
    def _min_delta(self) -> float:
        if self.config.early_stopping is None:
            return 0.0
        return float(self.config.early_stopping.min_delta)

    def _should_stop(self, patience_counter: int) -> bool:
        if self.config.early_stopping is None:
            return False
        return patience_counter >= self.config.early_stopping.patience


__all__ = [
    'CheckpointBackend',
    'EarlyStoppingConfig',
    'EquinoxCheckpointBackend',
    'OrbaxCheckpointBackend',
    'SBITrainer',
    'TrainingConfig',
    'resolve_checkpoint_backend',
]
