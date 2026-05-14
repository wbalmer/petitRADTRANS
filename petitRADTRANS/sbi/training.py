"""Training and checkpoint orchestration for SBI posterior models."""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import dill
import equinox as eqx  # pyright: ignore[reportMissingImports]
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingImports]


def _get_device_mesh() -> Any:
    """Return a JAX device mesh for data-parallel training, or None if single-device."""
    devices = jax.devices()
    if len(devices) <= 1:
        return None
    return jax.sharding.Mesh(np.array(devices), axis_names=('batch',))


def _shard_batch(array: Any, mesh: Any) -> Any:
    """Shard an array along its first axis across the device mesh."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(array, NamedSharding(mesh, P('batch')))


def _pad_to_multiple(array: Any, multiple: int) -> Any:
    """Pad array along axis 0 to a multiple of ``multiple`` by repeating the last row."""
    n = array.shape[0]
    remainder = n % multiple
    if remainder == 0:
        return array
    pad_count = multiple - remainder
    padding = np.repeat(array[-1:], pad_count, axis=0)
    return np.concatenate([array, padding], axis=0)


def _write_epoch_metrics(
    metric_history: dict[str, list[float]],
    epoch: int,
    epoch_metrics: Mapping[str, Any],
) -> None:
    keys = set(metric_history) | set(epoch_metrics)
    for key in keys:
        values = metric_history.setdefault(key, [])
        while len(values) < epoch:
            values.append(float('nan'))
        metric_value = float(epoch_metrics.get(key, float('nan')))
        if epoch < len(values):
            values[epoch] = metric_value
        else:
            values.append(metric_value)


def _log_scale_from_metadata(metadata: object) -> float:
    if not isinstance(metadata, dict):
        return 0.0
    if "log_spectrum_scale" in metadata:
        return float(metadata.get("log_spectrum_scale", 0.0))
    return float(metadata.get("log_median_flux", 0.0))


def _absolute_values_from_metadata(metadata: object, target: int) -> np.ndarray:
    if not isinstance(metadata, dict) or "absolute_values" not in metadata:
        return np.zeros(target, dtype=np.float32)

    absolute_values = np.ravel(np.asarray(metadata.get("absolute_values"), dtype=np.float32))
    if absolute_values.shape[0] == target:
        return absolute_values
    if absolute_values.shape[0] > target:
        return absolute_values[:target]
    return np.concatenate(
        [absolute_values, np.zeros(target - absolute_values.shape[0], dtype=np.float32)]
    )


def _global_scale_from_metadata(metadata: object) -> float:
    if not isinstance(metadata, dict):
        return 1.0

    if "global_spectrum_scale" in metadata:
        global_scale = float(metadata.get("global_spectrum_scale", 1.0))
        if np.isfinite(global_scale) and global_scale > 0.0:
            return global_scale

    if "spectrum_scale" in metadata:
        sample_scale = float(metadata.get("spectrum_scale", 1.0))
        if np.isfinite(sample_scale) and sample_scale > 0.0:
            return sample_scale

    return 1.0


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
        v_list, u_list, c_list, m_list, lmf_list, av_list, gsf_list = [], [], [], [], [], [], []
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
            lmf = _log_scale_from_metadata(block.metadata)
            lmf_list.append(np.float32(lmf))
            av_list.append(_absolute_values_from_metadata(block.metadata, n_wl))
            gsf_list.append(np.float32(_global_scale_from_metadata(block.metadata)))
        stacked.append((
            np.stack(v_list),
            np.stack(u_list),
            np.stack(c_list),
            np.stack(m_list),
            np.array(lmf_list, dtype=np.float32),
            np.stack(av_list),
            np.array(gsf_list, dtype=np.float32),
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
    data_parallel: bool = False
    gradient_clip_norm: float | None = 1.0
    weight_decay: float = 0.0
    use_cosine_schedule: bool = False
    warmup_fraction: float = 0.02
    min_learning_rate: float = 1e-6
    verbose_diagnostics: bool = False
    diagnostics_output_directory: str | None = None
    diagnostics_plot_interval: int = 1


def _resolve_diagnostics_output_directory(config: TrainingConfig) -> Path | None:
    if config.diagnostics_output_directory:
        return Path(config.diagnostics_output_directory)
    if config.checkpoint_directory:
        return Path(config.checkpoint_directory).parent
    return None


def _top_edge_hit_dimensions(
    epoch_metrics: Mapping[str, Any],
    limit: int = 3,
) -> list[tuple[int, float]]:
    ranked_dimensions: list[tuple[int, float]] = []
    for key, value in epoch_metrics.items():
        if not key.startswith('cube_edge_hit_rate_dim_'):
            continue
        suffix = key.rsplit('_', 1)[-1]
        if not suffix.isdigit():
            continue
        rate = float(value)
        if np.isfinite(rate) and rate > 0.0:
            ranked_dimensions.append((int(suffix), rate))
    ranked_dimensions.sort(key=lambda item: item[1], reverse=True)
    return ranked_dimensions[:limit]


def _plot_metric_series(
    axis: Any,
    epochs: np.ndarray,
    metric_history: Mapping[str, list[float]],
    keys: tuple[str, ...],
    title: str,
    *,
    log_scale: bool = False,
) -> None:
    plotted = False
    for key in keys:
        values = np.asarray(metric_history.get(key, []), dtype=float)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        axis.plot(epochs[: values.size], values, label=key)
        plotted = True

    if not plotted:
        axis.set_visible(False)
        return

    axis.set_title(title)
    axis.set_xlabel('Epoch')
    axis.grid(alpha=0.3)
    if log_scale:
        axis.set_yscale('log')
    if len(keys) > 1:
        axis.legend(fontsize=8)


def _write_verbose_diagnostic_artifacts(
    metric_history: Mapping[str, list[float]],
    output_directory: Path,
    completed_epochs: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        key: [float(value) for value in values]
        for key, values in sorted(metric_history.items())
    }
    (output_directory / 'training_verbose_metrics.json').write_text(
        json.dumps(
            {
                'completed_epochs': int(completed_epochs),
                'validation_metrics': metrics_payload,
            },
            sort_keys=True,
            indent=2,
        )
    )

    try:
        import matplotlib

        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    validation_loss = np.asarray(metric_history.get('validation_loss', []), dtype=float)
    if validation_loss.size == 0:
        return

    epochs = np.arange(1, validation_loss.size + 1, dtype=int)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_metric_series(
        axes[0, 0],
        epochs,
        metric_history,
        ('validation_loss', 'validation_selection_metric'),
        'Validation Loss And Selection',
    )
    _plot_metric_series(
        axes[0, 1],
        epochs,
        metric_history,
        (
            'forward_inverse_max_abs_error',
            'inverse_forward_max_abs_error',
            'forward_inverse_logdet_closure_max_abs_error',
            'inverse_forward_logdet_closure_max_abs_error',
        ),
        'Round-Trip Errors',
        log_scale=True,
    )
    neural_inverse_keys = (
        'inverse_solver_max_abs_residual',
        'inverse_solver_max_bracket_width',
        'inverse_solver_failure_rate',
        'inverse_solver_max_expansion_steps',
    )
    if any(
        np.isfinite(np.asarray(metric_history.get(key, []), dtype=float)).any()
        for key in neural_inverse_keys
    ):
        _plot_metric_series(
            axes[1, 0],
            epochs,
            metric_history,
            neural_inverse_keys,
            'Neural Inverse Diagnostics',
            log_scale=True,
        )
    else:
        _plot_metric_series(
            axes[1, 0],
            epochs,
            metric_history,
            ('spline_max_derivative', 'spline_min_bin_width', 'spline_min_bin_height'),
            'Spline Extremes',
            log_scale=True,
        )
    _plot_metric_series(
        axes[1, 1],
        epochs,
        metric_history,
        (
            'cube_edge_hit_rate',
            'checkpoint_is_stable',
            'checkpoint_stability_violation_count',
            'spline_bin_width_floor_fraction',
            'spline_bin_height_floor_fraction',
            'spline_mean_min_width_ratio',
            'spline_mean_min_height_ratio',
            'spline_small_bin_regularization_term',
            'spline_derivative_regularization_term',
        ),
        'Edge Hits, Bin Ratios, And Penalties',
    )
    figure.suptitle(f'Validation Diagnostics Through Epoch {completed_epochs}', fontsize=14)
    figure.savefig(output_directory / 'training_verbose_diagnostics.png', dpi=160)
    plt.close(figure)

    dimension_keys = sorted(
        (
            key for key in metric_history
            if key.startswith('cube_edge_hit_rate_dim_')
            and key.rsplit('_', 1)[-1].isdigit()
        ),
        key=lambda item: int(item.rsplit('_', 1)[-1]),
    )
    if not dimension_keys:
        return

    dimensions = np.asarray([int(key.rsplit('_', 1)[-1]) for key in dimension_keys], dtype=int)
    overall = np.asarray([metric_history[key][-1] for key in dimension_keys], dtype=float)
    low = np.asarray(
        [metric_history.get(f'cube_low_edge_hit_rate_dim_{dim}', [float('nan')])[-1] for dim in dimensions],
        dtype=float,
    )
    high = np.asarray(
        [metric_history.get(f'cube_high_edge_hit_rate_dim_{dim}', [float('nan')])[-1] for dim in dimensions],
        dtype=float,
    )

    edge_figure, edge_axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    edge_axis.bar(dimensions - 0.25, overall, width=0.25, label='overall')
    edge_axis.bar(dimensions, low, width=0.25, label='low')
    edge_axis.bar(dimensions + 0.25, high, width=0.25, label='high')
    edge_axis.set_title(f'Per-Dimension Edge Hit Rates At Epoch {completed_epochs}')
    edge_axis.set_xlabel('Parameter Dimension')
    edge_axis.set_ylabel('Rate')
    edge_axis.set_ylim(0.0, 1.0)
    edge_axis.grid(alpha=0.3, axis='y')
    edge_axis.legend(fontsize=8)
    edge_figure.savefig(output_directory / 'training_verbose_edge_hits.png', dpi=160)
    plt.close(edge_figure)


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


def load_trainer_checkpoint_state(
    checkpoint_directory: str | Path,
    checkpoint_kind: str = 'best',
) -> tuple[Any, dict[str, Any]] | None:
    """Load one persisted trainer checkpoint state and its metadata.

    Parameters
    ----------
    checkpoint_directory:
        Root directory containing trainer checkpoint subdirectories such as
        ``best_selection`` and ``best_loss``.
    checkpoint_kind:
        Checkpoint subdirectory to restore.

    Returns
    -------
    tuple[Any, dict[str, Any]] | None
        Restored state payload and metadata, or ``None`` when the requested
        checkpoint does not exist.
    """
    checkpoint_path = Path(checkpoint_directory) / checkpoint_kind
    metadata_path = checkpoint_path / 'metadata.json'
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    backend_name = str(metadata.get('checkpoint_backend', 'equinox'))
    checkpoint_backend = instantiate_checkpoint_backend(backend_name)
    return checkpoint_backend.restore(None, checkpoint_path)


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
        return self._checkpoint_directory_for_kind('latest')

    @property
    def _best_checkpoint_directory(self) -> Path | None:
        return self._checkpoint_directory_for_kind('best')

    def _checkpoint_directory_for_kind(self, checkpoint_kind: str) -> Path | None:
        if self.checkpoint_directory is None:
            return None
        return self.checkpoint_directory / checkpoint_kind

    def _save_checkpoint(self, checkpoint_kind: str, state: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.checkpoint_backend is None or self.checkpoint_directory is None:
            return

        checkpoint_path = self._checkpoint_directory_for_kind(checkpoint_kind)
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

        checkpoint_path = self._checkpoint_directory_for_kind(checkpoint_kind)
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
        eval_loss_fn: Callable[[Any, Any], Any] | None = None,
        eval_diagnostic_fn: Callable[[Any, Any], Mapping[str, float]] | None = None,
        selection_metric_fn: Callable[[float, Mapping[str, float] | None], float] | None = None,
        selection_metric_name: str | None = None,
        stability_metric_fn: Callable[[float, Mapping[str, float] | None], Mapping[str, float]] | None = None,
        stability_flag_key: str = 'checkpoint_is_stable',
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
        eval_loss_fn:
            Optional loss used for validation. Defaults to ``loss_fn`` when not
            provided.

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

        if eval_loss_fn is None:
            eval_loss_fn = loss_fn

        # --- Compute total training steps for LR scheduling ---------------
        n_train_samples = int(dataset.dataset.manifest.splits.get('train', 0))
        steps_per_epoch = max(1, n_train_samples // self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs

        # --- Build optimizer with optional weight decay & cosine schedule --
        if self.config.use_cosine_schedule and total_steps > 0:
            warmup_steps = max(1, int(self.config.warmup_fraction * total_steps))
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.config.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=self.config.min_learning_rate,
            )
        else:
            lr_schedule = self.config.learning_rate

        if self.config.weight_decay > 0:
            base_optimizer = optax.adamw(lr_schedule, weight_decay=self.config.weight_decay)
        else:
            base_optimizer = optax.adam(lr_schedule)

        if self.config.gradient_clip_norm is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip_norm),
                base_optimizer,
            )
        else:
            optimizer = base_optimizer
        optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        has_validation = dataset.dataset.manifest.splits.get('validation', 0) > 0
        effective_selection_metric_name = selection_metric_name or (
            'validation_selection_metric' if has_validation else 'train_loss'
        )
        primary_metric_name = 'validation_loss' if has_validation else 'train_loss'
        train_loss_history: list[float] = []
        validation_loss_history: list[float] = []
        validation_metrics_history: dict[str, list[float]] = {'validation_loss': validation_loss_history}
        best_model = model
        best_metric = float('inf')
        best_metric_name = primary_metric_name
        best_validation_loss = float('inf')
        best_validation_metrics: dict[str, float] = {}
        best_epoch = -1
        best_checkpoint_kind = 'best_selection'
        best_selection_model = model
        best_selection_metric = float('inf')
        best_selection_validation_loss = float('inf')
        best_selection_validation_metrics: dict[str, float] = {}
        best_selection_epoch = -1
        best_loss_model = model
        best_loss_validation_loss = float('inf')
        best_loss_validation_metrics: dict[str, float] = {}
        best_loss_epoch = -1
        best_stable_model = model
        best_stable_validation_loss = float('inf')
        best_stable_validation_metrics: dict[str, float] = {}
        best_stable_epoch = -1
        patience_counter = 0
        start_epoch = 0
        resumed_from_checkpoint = False

        template_state = {
            'model': model,
            'optimizer_state': optimizer_state,
            'history': {
                'train_loss': [],
                'validation_loss': [],
            },
            'validation_metrics': {
                'validation_loss': [],
            },
            'start_epoch': 0,
            'best_epoch': -1,
            'best_metric': float('inf'),
            'best_metric_name': primary_metric_name,
            'best_validation_loss': float('inf'),
            'best_validation_metrics': {},
            'best_checkpoint_kind': best_checkpoint_kind,
            'best_selection_epoch': -1,
            'best_selection_metric': float('inf'),
            'best_selection_validation_loss': float('inf'),
            'best_selection_validation_metrics': {},
            'best_loss_epoch': -1,
            'best_loss_validation_loss': float('inf'),
            'best_loss_validation_metrics': {},
            'best_stable_epoch': -1,
            'best_stable_validation_loss': float('inf'),
            'best_stable_validation_metrics': {},
            'patience_counter': 0,
        }

        if self.config.resume_from_checkpoint:
            restored = self._restore_checkpoint(template_state, 'latest')
            if restored is not None:
                restored_state, restored_metadata = restored
                model = restored_state['model']
                optimizer_state = restored_state['optimizer_state']
                train_loss_history = list(restored_state['history'].get('train_loss', []))
                restored_validation_metrics = {
                    str(key): list(values)
                    for key, values in restored_state.get('validation_metrics', {}).items()
                }
                if not restored_validation_metrics:
                    restored_validation_metrics = {
                        'validation_loss': list(
                            restored_state['history'].get('validation_loss', [])
                        )
                    }
                elif 'validation_loss' not in restored_validation_metrics:
                    restored_validation_metrics['validation_loss'] = list(
                        restored_state['history'].get('validation_loss', [])
                    )
                validation_metrics_history = restored_validation_metrics
                validation_loss_history = validation_metrics_history['validation_loss']
                start_epoch = int(restored_state.get('start_epoch', 0))
                best_epoch = int(restored_state.get('best_epoch', -1))
                best_metric = float(restored_state.get('best_metric', float('inf')))
                best_metric_name = str(
                    restored_state.get(
                        'best_metric_name',
                        primary_metric_name,
                    )
                )
                best_validation_loss = float(
                    restored_state.get('best_validation_loss', float('inf'))
                )
                best_validation_metrics = {
                    str(key): float(value)
                    for key, value in restored_state.get('best_validation_metrics', {}).items()
                }
                best_checkpoint_kind = str(
                    restored_state.get(
                        'best_checkpoint_kind',
                        'best_selection',
                    )
                )
                best_selection_epoch = int(
                    restored_state.get('best_selection_epoch', best_epoch)
                )
                best_selection_metric = float(
                    restored_state.get('best_selection_metric', best_metric)
                )
                best_selection_validation_loss = float(
                    restored_state.get(
                        'best_selection_validation_loss',
                        best_validation_loss,
                    )
                )
                best_selection_validation_metrics = {
                    str(key): float(value)
                    for key, value in restored_state.get(
                        'best_selection_validation_metrics',
                        best_validation_metrics,
                    ).items()
                }
                best_loss_epoch = int(restored_state.get('best_loss_epoch', best_epoch))
                best_loss_validation_loss = float(
                    restored_state.get('best_loss_validation_loss', best_validation_loss)
                )
                best_loss_validation_metrics = {
                    str(key): float(value)
                    for key, value in restored_state.get(
                        'best_loss_validation_metrics',
                        best_validation_metrics,
                    ).items()
                }
                best_stable_epoch = int(restored_state.get('best_stable_epoch', -1))
                best_stable_validation_loss = float(
                    restored_state.get('best_stable_validation_loss', float('inf'))
                )
                best_stable_validation_metrics = {
                    str(key): float(value)
                    for key, value in restored_state.get(
                        'best_stable_validation_metrics',
                        {},
                    ).items()
                }
                patience_counter = int(restored_state.get('patience_counter', 0))
                resumed_from_checkpoint = True

                restored_best = self._restore_checkpoint(template_state, 'best')
                if restored_best is not None:
                    best_model = restored_best[0]['model']
                else:
                    best_model = model

                restored_best_selection = self._restore_checkpoint(
                    template_state,
                    'best_selection',
                )
                if restored_best_selection is not None:
                    best_selection_model = restored_best_selection[0]['model']
                else:
                    best_selection_model = best_model

                restored_best_loss = self._restore_checkpoint(template_state, 'best_loss')
                if restored_best_loss is not None:
                    best_loss_model = restored_best_loss[0]['model']
                else:
                    best_loss_model = best_model

                restored_best_stable = self._restore_checkpoint(
                    template_state,
                    'best_stable',
                )
                if restored_best_stable is not None:
                    best_stable_model = restored_best_stable[0]['model']
                else:
                    best_stable_model = best_model

                if best_epoch < 0 and start_epoch > 0:
                    best_epoch = start_epoch - 1
                best_metric = float(restored_metadata.get('best_metric', best_metric))
                best_metric_name = str(
                    restored_metadata.get('best_metric_name', best_metric_name)
                )
                best_validation_loss = float(
                    restored_metadata.get('best_validation_loss', best_validation_loss)
                )

        from .posterior_base import PosteriorBatch

        # Multi-device setup: detect available devices and create a mesh
        # when data-parallel training is enabled.
        mesh = _get_device_mesh() if self.config.data_parallel else None
        n_devices = len(jax.devices()) if mesh is not None else 1
        batch_metadata = {'parameter_space': self.config.parameter_space}

        # When data-parallel is active, ensure batch_size is a multiple of
        # n_devices to avoid uneven splits and extra XLA recompilations.
        effective_batch_size = self.config.batch_size
        if mesh is not None and effective_batch_size % n_devices != 0:
            effective_batch_size = ((effective_batch_size + n_devices - 1) // n_devices) * n_devices

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
            _batch = PosteriorBatch(
                parameters=_parameters,
                observations=_observations,
                metadata=batch_metadata,
            )
            _loss_val, _grads = eqx.filter_value_and_grad(loss_fn)(_model, _batch)
            # Zero out any NaN/Inf gradient values so bad batches cannot
            # corrupt model weights or optimizer state.  The None leaves
            # produced by filter_value_and_grad for non-differentiable
            # parameters are preserved by the is_leaf guard.
            _grads = jax.tree_util.tree_map(
                lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g))
                if g is not None else None,
                _grads,
                is_leaf=lambda x: x is None,
            )
            # When data-parallel is active the inputs are sharded via
            # NamedSharding and JAX's GSPMD compiler automatically inserts
            # the required all-reduce collectives; explicit pmean is not
            # needed (and would fail because filter_jit does not provide
            # a named axis context).
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
            _batch = PosteriorBatch(
                parameters=_parameters,
                observations=_observations,
                metadata=batch_metadata,
            )
            return eval_loss_fn(_model, _batch)

        def _maybe_pad_and_shard(params: Any, obs: Any) -> tuple:
            """Pad batch to ``effective_batch_size`` and shard across devices.

            Padding to the full batch size (instead of just a device multiple)
            ensures that every training step sees exactly the same array shapes,
            preventing costly XLA recompilations for trailing partial batches.
            """
            if mesh is None:
                return params, obs
            params = np.asarray(params)
            params = _pad_to_multiple(params, effective_batch_size)
            if hasattr(obs, 'stacked_blocks'):
                from .observation import PreStackedObservations
                padded_blocks = tuple(
                    tuple(_pad_to_multiple(np.asarray(arr), effective_batch_size) for arr in block)
                    for block in obs.stacked_blocks
                )
                obs = PreStackedObservations(
                    stacked_blocks=padded_blocks, modalities=obs.modalities
                )
            params = _shard_batch(params, mesh)
            if hasattr(obs, 'stacked_blocks'):
                from .observation import PreStackedObservations
                sharded_blocks = tuple(
                    tuple(_shard_batch(arr, mesh) for arr in block)
                    for block in obs.stacked_blocks
                )
                obs = PreStackedObservations(
                    stacked_blocks=sharded_blocks, modalities=obs.modalities
                )
            else:
                obs = _shard_batch(obs, mesh)
            return params, obs

        def _checkpoint_state(checkpoint_model: Any, next_epoch: int) -> dict[str, Any]:
            return {
                'model': checkpoint_model,
                'optimizer_state': optimizer_state,
                'history': {
                    'train_loss': list(train_loss_history),
                    'validation_loss': list(validation_loss_history),
                },
                'validation_metrics': {
                    key: list(values)
                    for key, values in validation_metrics_history.items()
                },
                'start_epoch': next_epoch,
                'best_epoch': best_epoch,
                'best_metric': best_metric,
                'best_metric_name': best_metric_name,
                'best_validation_loss': best_validation_loss,
                'best_validation_metrics': dict(best_validation_metrics),
                'best_checkpoint_kind': best_checkpoint_kind,
                'best_selection_epoch': best_selection_epoch,
                'best_selection_metric': best_selection_metric,
                'best_selection_validation_loss': best_selection_validation_loss,
                'best_selection_validation_metrics': dict(best_selection_validation_metrics),
                'best_loss_epoch': best_loss_epoch,
                'best_loss_validation_loss': best_loss_validation_loss,
                'best_loss_validation_metrics': dict(best_loss_validation_metrics),
                'best_stable_epoch': best_stable_epoch,
                'best_stable_validation_loss': best_stable_validation_loss,
                'best_stable_validation_metrics': dict(best_stable_validation_metrics),
                'patience_counter': patience_counter,
            }

        def _checkpoint_metadata(
            checkpoint_kind: str,
            epoch_index: int,
            checkpoint_metric: float,
            checkpoint_metric_name: str,
            checkpoint_validation_loss: float,
            checkpoint_validation_metrics: Mapping[str, float],
        ) -> dict[str, Any]:
            return {
                'epoch': epoch_index,
                'checkpoint_kind': checkpoint_kind,
                'checkpoint_metric': checkpoint_metric,
                'checkpoint_metric_name': checkpoint_metric_name,
                'checkpoint_validation_loss': checkpoint_validation_loss,
                'checkpoint_validation_metrics': dict(checkpoint_validation_metrics),
                'best_epoch': best_epoch + 1 if best_epoch >= 0 else None,
                'best_metric': best_metric,
                'best_metric_name': best_metric_name,
                'best_validation_loss': best_validation_loss,
                'best_validation_metrics': dict(best_validation_metrics),
                'best_checkpoint_kind': best_checkpoint_kind,
                'best_selection_epoch': best_selection_epoch + 1 if best_selection_epoch >= 0 else None,
                'best_selection_metric': best_selection_metric,
                'best_selection_validation_loss': best_selection_validation_loss,
                'best_selection_validation_metrics': dict(best_selection_validation_metrics),
                'best_loss_epoch': best_loss_epoch + 1 if best_loss_epoch >= 0 else None,
                'best_loss_validation_loss': best_loss_validation_loss,
                'best_loss_validation_metrics': dict(best_loss_validation_metrics),
                'best_stable_epoch': best_stable_epoch + 1 if best_stable_epoch >= 0 else None,
                'best_stable_validation_loss': best_stable_validation_loss,
                'best_stable_validation_metrics': dict(best_stable_validation_metrics),
                'selection_metric_name': effective_selection_metric_name,
                'checkpoint_backend': (
                    None if self.checkpoint_backend is None else self.checkpoint_backend.name
                ),
            }

        # --- Pre-load and pre-stack training and validation data --------
        # Reading from HDF5 with shuffled (scattered) indices each epoch is
        # extremely slow due to non-sequential I/O.  Instead, load every split
        # once, normalize, and prestack into dense arrays.  Per-epoch shuffling
        # is achieved by permuting a NumPy index array — no H5 re-reads.
        def _preload_split(split: DatasetSplit) -> tuple[np.ndarray, Any] | None:
            """Load and prestack one full split into RAM."""
            from .observation import PreStackedObservations

            n_split = int(dataset.dataset.manifest.splits.get(split.value, 0))
            if n_split == 0:
                return None
            print(f"  Pre-loading {split.value} split ({n_split} samples)...", flush=True)
            all_params = []
            block_accumulators: list[list[list]] | None = None  # [block_j][field_k] -> list of arrays
            modalities: tuple | None = None
            for batch in dataset.iter_batches(
                split=split,
                batch_size=min(n_split, max(effective_batch_size, 4096)),
                shuffle=False,
                parameter_space=self.config.parameter_space,
            ):
                prestacked = _prestack_observation_blocks(batch.observations)
                if prestacked is None:
                    # Fallback: cannot prestack; return None to use streaming
                    return None
                all_params.append(np.asarray(batch.parameters))
                if block_accumulators is None:
                    block_accumulators = [
                        [[] for _ in range(len(block_tuple))]
                        for block_tuple in prestacked.stacked_blocks
                    ]
                    modalities = prestacked.modalities
                for j, block_tuple in enumerate(prestacked.stacked_blocks):
                    for field_index, field_array in enumerate(block_tuple):
                        block_accumulators[j][field_index].append(np.asarray(field_array))
            if block_accumulators is None or modalities is None:
                return None
            params = np.concatenate(all_params, axis=0)
            stacked_blocks = tuple(
                tuple(np.concatenate(field_values, axis=0) for field_values in acc)
                for acc in block_accumulators
            )
            obs = PreStackedObservations(stacked_blocks=stacked_blocks, modalities=modalities)
            total_bytes = params.nbytes + sum(a.nbytes for blk in stacked_blocks for a in blk)
            print(
                f"  Pre-loaded {split.value}: {params.shape[0]} samples, "
                f"{total_bytes / 1e6:.0f} MB",
                flush=True,
            )
            return params, obs

        preloaded_train = _preload_split(DatasetSplit.TRAIN)
        preloaded_val = _preload_split(DatasetSplit.VALIDATION) if has_validation else None

        for epoch in range(start_epoch, self.config.num_epochs):
            _epoch_start = time.monotonic()
            epoch_losses = []

            if preloaded_train is not None:
                # Fast path: slice pre-loaded arrays by shuffled index
                _all_params, _all_obs = preloaded_train
                n_train = _all_params.shape[0]
                _perm = np.random.default_rng(self.config.seed + epoch).permutation(n_train) if self.config.shuffle_train else np.arange(n_train)
                for _start in range(0, n_train, effective_batch_size):
                    _idx = _perm[_start : _start + effective_batch_size]
                    if len(_idx) < effective_batch_size:
                        if mesh is not None:
                            continue  # drop last incomplete batch
                    _bp = _all_params[_idx]
                    from .observation import PreStackedObservations
                    _bo = PreStackedObservations(
                        stacked_blocks=tuple(
                            tuple(arr[_idx] for arr in block_tuple)
                            for block_tuple in _all_obs.stacked_blocks
                        ),
                        modalities=_all_obs.modalities,
                    )
                    if mesh is not None:
                        _bp, _bo = _maybe_pad_and_shard(_bp, _bo)
                    model, optimizer_state, loss_value = _jit_step(model, optimizer_state, _bp, _bo)
                    epoch_losses.append(float(loss_value))
            else:
                # Fallback: stream from disk (non-prestackable observations)
                for batch in dataset.iter_batches(
                    split=DatasetSplit.TRAIN,
                    batch_size=effective_batch_size,
                    shuffle=self.config.shuffle_train,
                    seed=self.config.seed + epoch,
                    parameter_space=self.config.parameter_space,
                ):
                    prestacked = _prestack_observation_blocks(batch.observations)
                    _obs = prestacked if prestacked is not None else batch.observations
                    if mesh is not None:
                        batch_params = np.asarray(batch.parameters)
                        if batch_params.shape[0] < effective_batch_size:
                            continue
                        batch_params, _obs = _maybe_pad_and_shard(batch_params, _obs)
                    else:
                        batch_params = batch.parameters
                    model, optimizer_state, loss_value = _jit_step(model, optimizer_state, batch_params, _obs)
                    epoch_losses.append(float(loss_value))

            train_metric = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            if epoch < len(train_loss_history):
                train_loss_history[epoch] = train_metric
            else:
                train_loss_history.append(train_metric)

            validation_metric = float('nan')
            epoch_validation_metrics: dict[str, float] = {'validation_loss': float('nan')}
            selection_metric = float('nan')
            if has_validation:
                validation_losses = []
                diagnostic_batch: PosteriorBatch | None = None
                if preloaded_val is not None:
                    _val_params, _val_all_obs = preloaded_val
                    n_val = _val_params.shape[0]
                    for _start in range(0, n_val, effective_batch_size):
                        _idx = np.arange(_start, min(_start + effective_batch_size, n_val))
                        _vp = _val_params[_idx]
                        from .observation import PreStackedObservations
                        _vo = PreStackedObservations(
                            stacked_blocks=tuple(
                                tuple(arr[_idx] for arr in block_tuple)
                                for block_tuple in _val_all_obs.stacked_blocks
                            ),
                            modalities=_val_all_obs.modalities,
                        )
                        if diagnostic_batch is None:
                            diagnostic_batch = PosteriorBatch(
                                parameters=_vp,
                                observations=_vo,
                                metadata=batch_metadata,
                            )
                        if mesh is not None:
                            _vp, _vo = _maybe_pad_and_shard(_vp, _vo)
                        validation_losses.append(float(_jit_eval_loss(model, _vp, _vo)))
                else:
                    for batch in dataset.iter_batches(
                        split=DatasetSplit.VALIDATION,
                        batch_size=effective_batch_size,
                        shuffle=False,
                        parameter_space=self.config.parameter_space,
                    ):
                        prestacked_val = _prestack_observation_blocks(batch.observations)
                        _val_obs = prestacked_val if prestacked_val is not None else batch.observations
                        val_params = batch.parameters
                        if diagnostic_batch is None:
                            diagnostic_batch = PosteriorBatch(
                                parameters=val_params,
                                observations=_val_obs,
                                metadata=batch_metadata,
                            )
                        if mesh is not None:
                            val_params, _val_obs = _maybe_pad_and_shard(val_params, _val_obs)
                        validation_losses.append(float(_jit_eval_loss(model, val_params, _val_obs)))
                validation_metric = float(np.mean(validation_losses)) if validation_losses else float('nan')
                epoch_validation_metrics = {'validation_loss': validation_metric}
                if eval_diagnostic_fn is not None and diagnostic_batch is not None:
                    diagnostic_metrics = {
                        str(key): float(value)
                        for key, value in eval_diagnostic_fn(model, diagnostic_batch).items()
                    }
                    epoch_validation_metrics.update(diagnostic_metrics)
                selection_metric = (
                    float(selection_metric_fn(validation_metric, epoch_validation_metrics))
                    if selection_metric_fn is not None
                    else validation_metric
                )
                epoch_validation_metrics[effective_selection_metric_name] = selection_metric
                if stability_metric_fn is not None:
                    stability_metrics = {
                        str(key): float(value)
                        for key, value in stability_metric_fn(
                            validation_metric,
                            epoch_validation_metrics,
                        ).items()
                    }
                    epoch_validation_metrics.update(stability_metrics)
                _write_epoch_metrics(
                    validation_metrics_history,
                    epoch,
                    epoch_validation_metrics,
                )
                validation_loss_history = validation_metrics_history['validation_loss']

                if self.config.verbose_diagnostics:
                    diagnostics_output_directory = _resolve_diagnostics_output_directory(self.config)
                    should_write_verbose_artifacts = (
                        diagnostics_output_directory is not None
                        and max(int(self.config.diagnostics_plot_interval), 1) > 0
                        and ((epoch + 1) % max(int(self.config.diagnostics_plot_interval), 1) == 0)
                    )
                    if should_write_verbose_artifacts and diagnostics_output_directory is not None:
                        _write_verbose_diagnostic_artifacts(
                            validation_metrics_history,
                            diagnostics_output_directory,
                            epoch + 1,
                        )

            _elapsed = time.monotonic() - _epoch_start
            is_stable_checkpoint = True
            if has_validation:
                if stability_metric_fn is not None:
                    is_stable_checkpoint = bool(
                        epoch_validation_metrics.get(stability_flag_key, 0.0) >= 0.5
                    )
                else:
                    is_stable_checkpoint = np.isfinite(validation_metric)
            _val_str = (
                f" | val_loss: {validation_metric:.6f}" if has_validation else ""
            )
            _selection_str = ""
            if has_validation and effective_selection_metric_name != 'validation_loss':
                _selection_str += (
                    f" | {effective_selection_metric_name}: {selection_metric:.6f}"
                )
            if has_validation:
                edge_hit_rate = epoch_validation_metrics.get('cube_edge_hit_rate', float('nan'))
                if np.isfinite(edge_hit_rate):
                    _selection_str += f" | edge_hit: {edge_hit_rate:.4f}"
                roundtrip_error = epoch_validation_metrics.get(
                    'forward_inverse_max_abs_error',
                    float('nan'),
                )
                if np.isfinite(roundtrip_error):
                    _selection_str += f" | rt_err: {roundtrip_error:.3e}"
                inverse_roundtrip_error = epoch_validation_metrics.get(
                    'inverse_forward_max_abs_error',
                    float('nan'),
                )
                if np.isfinite(inverse_roundtrip_error):
                    _selection_str += f" | inv_rt_err: {inverse_roundtrip_error:.3e}"
                inverse_logdet_error = epoch_validation_metrics.get(
                    'inverse_forward_logdet_closure_max_abs_error',
                    float('nan'),
                )
                if np.isfinite(inverse_logdet_error):
                    _selection_str += f" | inv_logdet_err: {inverse_logdet_error:.3e}"
                _selection_str += f" | stable: {'yes' if is_stable_checkpoint else 'no'}"
                if self.config.verbose_diagnostics:
                    max_derivative = epoch_validation_metrics.get('spline_max_derivative', float('nan'))
                    min_bin_width = epoch_validation_metrics.get('spline_min_bin_width', float('nan'))
                    min_bin_height = epoch_validation_metrics.get('spline_min_bin_height', float('nan'))
                    width_floor_fraction = epoch_validation_metrics.get(
                        'spline_bin_width_floor_fraction',
                        float('nan'),
                    )
                    height_floor_fraction = epoch_validation_metrics.get(
                        'spline_bin_height_floor_fraction',
                        float('nan'),
                    )
                    derivative_floor_fraction = epoch_validation_metrics.get(
                        'spline_derivative_floor_fraction',
                        float('nan'),
                    )
                    mean_min_width_ratio = epoch_validation_metrics.get(
                        'spline_mean_min_width_ratio',
                        float('nan'),
                    )
                    mean_min_height_ratio = epoch_validation_metrics.get(
                        'spline_mean_min_height_ratio',
                        float('nan'),
                    )
                    small_bin_penalty = epoch_validation_metrics.get(
                        'spline_small_bin_regularization_term',
                        float('nan'),
                    )
                    derivative_penalty = epoch_validation_metrics.get(
                        'spline_derivative_regularization_term',
                        float('nan'),
                    )
                    if np.isfinite(max_derivative):
                        _selection_str += f" | max_d: {max_derivative:.3e}"
                    if np.isfinite(min_bin_width):
                        _selection_str += f" | min_w: {min_bin_width:.3e}"
                    if np.isfinite(min_bin_height):
                        _selection_str += f" | min_h: {min_bin_height:.3e}"
                    if np.isfinite(mean_min_width_ratio):
                        _selection_str += f" | w_ratio: {mean_min_width_ratio:.3f}"
                    if np.isfinite(mean_min_height_ratio):
                        _selection_str += f" | h_ratio: {mean_min_height_ratio:.3f}"
                    if np.isfinite(width_floor_fraction):
                        _selection_str += f" | w_floor: {width_floor_fraction:.3f}"
                    if np.isfinite(height_floor_fraction):
                        _selection_str += f" | h_floor: {height_floor_fraction:.3f}"
                    if np.isfinite(derivative_floor_fraction):
                        _selection_str += f" | d_floor: {derivative_floor_fraction:.3f}"
                    if np.isfinite(small_bin_penalty):
                        _selection_str += f" | bin_pen: {small_bin_penalty:.3e}"
                    if np.isfinite(derivative_penalty):
                        _selection_str += f" | reg_d: {derivative_penalty:.3e}"
                    top_dimensions = _top_edge_hit_dimensions(epoch_validation_metrics)
                    if top_dimensions:
                        top_dimensions_str = ','.join(
                            f"d{dimension}:{rate:.2f}"
                            for dimension, rate in top_dimensions
                        )
                        _selection_str += f" | edge_dims: {top_dimensions_str}"
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs}"
                f" | train_loss: {train_metric:.6f}{_val_str}{_selection_str}"
                f" | {_elapsed:.1f}s",
                flush=True,
            )

            selection_monitor_metric = selection_metric if has_validation else train_metric
            selection_improved = best_selection_epoch < 0 or (
                np.isfinite(selection_monitor_metric)
                and (
                    selection_monitor_metric < best_selection_metric - self._min_delta
                    or not np.isfinite(best_selection_metric)
                )
            )
            if selection_improved:
                best_selection_metric = selection_monitor_metric
                if has_validation and np.isfinite(validation_metric):
                    best_selection_validation_loss = validation_metric
                best_selection_validation_metrics = dict(epoch_validation_metrics)
                best_selection_epoch = epoch
                best_selection_model = model
                patience_counter = 0
                self._save_checkpoint(
                    'best_selection',
                    _checkpoint_state(best_selection_model, epoch + 1),
                    _checkpoint_metadata(
                        'best_selection',
                        epoch + 1,
                        best_selection_metric,
                        effective_selection_metric_name,
                        best_selection_validation_loss,
                        best_selection_validation_metrics,
                    ),
                )
            else:
                patience_counter += 1

            if has_validation and np.isfinite(validation_metric):
                loss_improved = best_loss_epoch < 0 or (
                    validation_metric < best_loss_validation_loss - self._min_delta
                    or not np.isfinite(best_loss_validation_loss)
                )
                if loss_improved:
                    best_loss_epoch = epoch
                    best_loss_validation_loss = validation_metric
                    best_loss_validation_metrics = dict(epoch_validation_metrics)
                    best_loss_model = model
                    self._save_checkpoint(
                        'best_loss',
                        _checkpoint_state(best_loss_model, epoch + 1),
                        _checkpoint_metadata(
                            'best_loss',
                            epoch + 1,
                            best_loss_validation_loss,
                            'validation_loss',
                            best_loss_validation_loss,
                            best_loss_validation_metrics,
                        ),
                    )

                stable_improved = is_stable_checkpoint and (
                    best_stable_epoch < 0
                    or validation_metric < best_stable_validation_loss - self._min_delta
                    or not np.isfinite(best_stable_validation_loss)
                )
                if stable_improved:
                    best_stable_epoch = epoch
                    best_stable_validation_loss = validation_metric
                    best_stable_validation_metrics = dict(epoch_validation_metrics)
                    best_stable_model = model
                    self._save_checkpoint(
                        'best_stable',
                        _checkpoint_state(best_stable_model, epoch + 1),
                        _checkpoint_metadata(
                            'best_stable',
                            epoch + 1,
                            best_stable_validation_loss,
                            'validation_loss',
                            best_stable_validation_loss,
                            best_stable_validation_metrics,
                        ),
                    )

            previous_best_signature = (
                best_checkpoint_kind,
                best_epoch,
                best_metric_name,
                best_metric,
            )
            if has_validation:
                # Keep the strictly stable checkpoint exported separately, but do
                # not auto-promote it over the soft-penalized selection model.
                if best_selection_epoch >= 0:
                    best_checkpoint_kind = 'best_selection'
                    best_epoch = best_selection_epoch
                    best_metric = best_selection_metric
                    best_metric_name = effective_selection_metric_name
                    best_validation_loss = best_selection_validation_loss
                    best_validation_metrics = dict(best_selection_validation_metrics)
                    best_model = best_selection_model
                elif best_loss_epoch >= 0:
                    best_checkpoint_kind = 'best_loss'
                    best_epoch = best_loss_epoch
                    best_metric = best_loss_validation_loss
                    best_metric_name = 'validation_loss'
                    best_validation_loss = best_loss_validation_loss
                    best_validation_metrics = dict(best_loss_validation_metrics)
                    best_model = best_loss_model
                elif best_stable_epoch >= 0:
                    best_checkpoint_kind = 'best_stable'
                    best_epoch = best_stable_epoch
                    best_metric = best_stable_validation_loss
                    best_metric_name = 'validation_loss'
                    best_validation_loss = best_stable_validation_loss
                    best_validation_metrics = dict(best_stable_validation_metrics)
                    best_model = best_stable_model
                else:
                    best_checkpoint_kind = 'best_selection'
                    best_epoch = best_selection_epoch
                    best_metric = best_selection_metric
                    best_metric_name = effective_selection_metric_name
                    best_validation_loss = best_selection_validation_loss
                    best_validation_metrics = dict(best_selection_validation_metrics)
                    best_model = best_selection_model
            else:
                best_checkpoint_kind = 'best_selection'
                best_epoch = best_selection_epoch
                best_metric = best_selection_metric
                best_metric_name = effective_selection_metric_name
                best_validation_loss = best_selection_validation_loss
                best_validation_metrics = dict(best_selection_validation_metrics)
                best_model = best_selection_model

            if previous_best_signature != (
                best_checkpoint_kind,
                best_epoch,
                best_metric_name,
                best_metric,
            ):
                self._save_checkpoint(
                    'best',
                    _checkpoint_state(best_model, epoch + 1),
                    _checkpoint_metadata(
                        'best',
                        epoch + 1,
                        best_metric,
                        best_metric_name,
                        best_validation_loss,
                        best_validation_metrics,
                    ),
                )

            self._save_checkpoint(
                'latest',
                _checkpoint_state(model, epoch + 1),
                _checkpoint_metadata(
                    'latest',
                    epoch + 1,
                    selection_monitor_metric,
                    effective_selection_metric_name if has_validation else 'train_loss',
                    validation_metric,
                    epoch_validation_metrics,
                ),
            )

            if self._should_stop(patience_counter):
                break

        return (
            best_model,
            {'train_loss': train_loss_history},
            validation_metrics_history,
            {
                'best_epoch': best_epoch + 1 if best_epoch >= 0 else None,
                'best_metric': best_metric,
                'best_metric_name': best_metric_name,
                'best_validation_loss': best_validation_loss,
                'best_validation_metrics': dict(best_validation_metrics),
                'best_checkpoint_kind': best_checkpoint_kind,
                'best_selection_epoch': best_selection_epoch + 1 if best_selection_epoch >= 0 else None,
                'best_selection_metric': best_selection_metric,
                'best_selection_validation_loss': best_selection_validation_loss,
                'best_selection_validation_metrics': dict(best_selection_validation_metrics),
                'best_loss_epoch': best_loss_epoch + 1 if best_loss_epoch >= 0 else None,
                'best_loss_validation_loss': best_loss_validation_loss,
                'best_loss_validation_metrics': dict(best_loss_validation_metrics),
                'best_stable_epoch': best_stable_epoch + 1 if best_stable_epoch >= 0 else None,
                'best_stable_validation_loss': best_stable_validation_loss,
                'best_stable_validation_metrics': dict(best_stable_validation_metrics),
                'selection_metric_name': effective_selection_metric_name,
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
    'load_trainer_checkpoint_state',
    'OrbaxCheckpointBackend',
    'SBITrainer',
    'TrainingConfig',
    'resolve_checkpoint_backend',
]
