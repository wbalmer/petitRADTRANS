"""Preprocessing helpers and normalization metadata for SBI observation blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .observation import ObservationBlock


_NORMALIZATION_SCALE_FLOOR = 1e-30


def _positive_scale(*candidates: Any, default: float = 1.0) -> float:
    for candidate in candidates:
        if candidate is None:
            continue
        value = float(candidate)
        if np.isfinite(value) and abs(value) > _NORMALIZATION_SCALE_FLOOR:
            return abs(value)
    return default


def _positive_scale_vector(
    values: Any,
    *,
    fallback: float,
) -> np.ndarray:
    array_values = np.asarray(values, dtype=float)
    return np.where(
        np.isfinite(array_values) & (np.abs(array_values) > _NORMALIZATION_SCALE_FLOOR),
        np.abs(array_values),
        fallback,
    )


def _valid_sample_values(values: Any, mask: Any) -> np.ndarray:
    raw_values = np.asarray(values, dtype=float).reshape(-1)
    if raw_values.size == 0:
        return np.asarray([], dtype=float)

    if mask is None:
        valid_mask = np.zeros(raw_values.shape[0], dtype=bool)
    else:
        flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if flat_mask.shape[0] < raw_values.shape[0]:
            valid_mask = np.concatenate(
                [flat_mask, np.zeros(raw_values.shape[0] - flat_mask.shape[0], dtype=bool)]
            )
        else:
            valid_mask = flat_mask[: raw_values.shape[0]]

    finite_valid = np.isfinite(raw_values) & (~valid_mask)
    if not np.any(finite_valid):
        return np.asarray([], dtype=float)

    return raw_values[finite_valid]


def _sample_robust_scale(values: Any, mask: Any) -> float | None:
    valid_values = _valid_sample_values(values, mask)
    if valid_values.size == 0:
        return None

    sample_scale = np.nanmedian(np.abs(valid_values))
    return _positive_scale(sample_scale, default=1.0)


def _sample_log_median_flux(values: Any, mask: Any) -> float | None:
    sample_scale = _sample_robust_scale(values, mask)
    if sample_scale is None:
        return None

    return float(np.log10(max(float(sample_scale), _NORMALIZATION_SCALE_FLOOR)))


def _global_spectral_scale(stats: Any) -> float:
    log_scale = getattr(stats, "log_median_flux", None)
    if log_scale is not None:
        return _positive_scale(10.0 ** float(log_scale), default=1.0)

    return _positive_scale(
        getattr(stats, "value_iqr", None),
        getattr(stats, "value_std", None),
        getattr(stats, "value_median", None),
        default=1.0,
    )


def _apply_spectral_value_transform(values: Any, sample_scale: float) -> np.ndarray:
    scale = max(float(sample_scale), _NORMALIZATION_SCALE_FLOOR)
    raw_values = np.asarray(values, dtype=float)
    return np.arcsinh(raw_values / scale)


def _apply_absolute_spectral_value_transform(
    values: Any,
    global_scale: float,
) -> np.ndarray:
    scale = max(float(global_scale), _NORMALIZATION_SCALE_FLOOR)
    raw_values = np.asarray(values, dtype=float)
    return np.arcsinh(raw_values / scale)


def _transform_spectral_uncertainties(
    uncertainties: Any,
    values: Any,
    sample_scale: float,
) -> np.ndarray:
    scale = max(float(sample_scale), _NORMALIZATION_SCALE_FLOOR)
    raw_uncertainties = np.asarray(uncertainties, dtype=float)
    raw_values = np.asarray(values, dtype=float)
    safe_ratio = np.where(np.isfinite(raw_values), raw_values / scale, 0.0)
    local_scale = scale * np.sqrt(1.0 + safe_ratio**2)
    return raw_uncertainties / np.maximum(local_scale, _NORMALIZATION_SCALE_FLOOR)


@dataclass(frozen=True)
class BlockNormalizationStats:
    """Normalization statistics for one observation block family."""

    name: str
    modality: str
    value_mean: float
    value_std: float
    uncertainty_mean: float | None = None
    uncertainty_std: float | None = None
    coordinate_mean: float | None = None
    coordinate_std: float | None = None
    n_samples: int = 0
    n_values: int = 0
    value_median: float | None = None
    value_iqr: float | None = None
    uncertainty_median: float | None = None
    uncertainty_iqr: float | None = None
    # Per-wavelength robust normalization vectors (lists for JSON serialization)
    value_median_per_wl: list[float] | None = None
    value_iqr_per_wl: list[float] | None = None
    uncertainty_median_per_wl: list[float] | None = None
    uncertainty_iqr_per_wl: list[float] | None = None
    # Log-median flux (log10 of abs(median flux)), a scalar absolute-scale feature
    log_median_flux: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPreprocessingMetadata:
    """Serializable preprocessing metadata for an SBI task family."""

    version: str
    blocks: Mapping[str, BlockNormalizationStats] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "blocks": {name: stats.to_payload() for name, stats in self.blocks.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TaskPreprocessingMetadata":
        payload_dict = dict(payload)
        blocks = {
            name: BlockNormalizationStats(**stats)
            for name, stats in payload_dict.get("blocks", {}).items()
        }
        return cls(
            version=str(payload_dict.get("version", "0.1.0")),
            blocks=blocks,
            metadata=dict(payload_dict.get("metadata", {})),
        )


def _finite_mean_std(values: Any) -> tuple[float | None, float | None, int]:
    if values is None:
        return None, None, 0

    array_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = array_values[np.isfinite(array_values)]
    if finite_values.size == 0:
        return None, None, 0

    return (
        float(np.mean(finite_values)),
        float(np.std(finite_values)),
        int(finite_values.size),
    )


def _finite_median_iqr(values: Any) -> tuple[float | None, float | None, int]:
    """Compute median and interquartile range from finite values."""
    if values is None:
        return None, None, 0

    array_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = array_values[np.isfinite(array_values)]
    if finite_values.size == 0:
        return None, None, 0

    median = float(np.median(finite_values))
    q75, q25 = np.percentile(finite_values, [75, 25])
    iqr = float(q75 - q25)
    return median, iqr, int(finite_values.size)


def compute_block_normalization_stats(
    blocks: list[ObservationBlock],
) -> dict[str, BlockNormalizationStats]:
    """Compute per-block normalization statistics from training observations."""
    grouped_blocks: dict[str, list[ObservationBlock]] = {}

    for block in blocks:
        grouped_blocks.setdefault(block.name, []).append(block)

    stats_by_name: dict[str, BlockNormalizationStats] = {}

    for block_name, block_group in grouped_blocks.items():
        concatenated_values = np.concatenate(
            [np.asarray(block.values, dtype=float).reshape(-1) for block in block_group]
        )
        concatenated_uncertainties = []
        concatenated_coordinates = []

        for block in block_group:
            if block.uncertainties is not None:
                concatenated_uncertainties.append(
                    np.asarray(block.uncertainties, dtype=float).reshape(-1)
                )
            if block.coordinates is not None:
                concatenated_coordinates.append(
                    np.asarray(block.coordinates, dtype=float).reshape(-1)
                )

        value_mean, value_std, n_values = _finite_mean_std(concatenated_values)
        uncertainty_mean, uncertainty_std, _ = _finite_mean_std(
            None
            if not concatenated_uncertainties
            else np.concatenate(concatenated_uncertainties)
        )
        coordinate_mean, coordinate_std, _ = _finite_mean_std(
            None
            if not concatenated_coordinates
            else np.concatenate(concatenated_coordinates)
        )

        value_median, value_iqr, _ = _finite_median_iqr(concatenated_values)
        uncertainty_median, uncertainty_iqr, _ = _finite_median_iqr(
            None
            if not concatenated_uncertainties
            else np.concatenate(concatenated_uncertainties)
        )

        # Per-wavelength robust statistics for spectral blocks
        value_median_per_wl = None
        value_iqr_per_wl = None
        uncertainty_median_per_wl = None
        uncertainty_iqr_per_wl = None
        log_median_flux = None

        # Detect if all blocks have the same length (i.e. a spectral block)
        block_lengths = [np.asarray(b.values, dtype=float).reshape(-1).shape[0]
                         for b in block_group]
        if len(set(block_lengths)) == 1 and block_lengths[0] > 1:
            n_wl = block_lengths[0]
            # Build (n_samples, n_wl) matrix
            value_matrix = np.stack(
                [np.asarray(b.values, dtype=float).reshape(-1) for b in block_group],
                axis=0,
            )
            finite_matrix = np.where(np.isfinite(value_matrix), value_matrix, np.nan)
            with np.errstate(all='ignore'):
                _med_per_wl = np.nanmedian(finite_matrix, axis=0)
                _q75 = np.nanpercentile(finite_matrix, 75, axis=0)
                _q25 = np.nanpercentile(finite_matrix, 25, axis=0)
                _iqr_per_wl = _q75 - _q25
            # Replace zero IQR with global IQR fallback
            _iqr_per_wl = np.where(
                _iqr_per_wl > 0,
                _iqr_per_wl,
                _positive_scale(value_iqr, value_std, value_median),
            )
            value_median_per_wl = _med_per_wl.tolist()
            value_iqr_per_wl = _iqr_per_wl.tolist()

            sample_scales = [
                _sample_robust_scale(block.values, block.mask)
                for block in block_group
            ]
            finite_sample_scales = np.asarray(
                [scale for scale in sample_scales if scale is not None],
                dtype=float,
            )
            if finite_sample_scales.size > 0:
                median_scale = np.nanmedian(finite_sample_scales)
                log_median_flux = float(
                    np.log10(max(float(median_scale), _NORMALIZATION_SCALE_FLOOR))
                )

            # Per-wavelength uncertainty stats
            if concatenated_uncertainties:
                unc_matrix = np.stack(
                    [np.asarray(b.uncertainties, dtype=float).reshape(-1)
                     for b in block_group if b.uncertainties is not None],
                    axis=0,
                )
                finite_unc = np.where(np.isfinite(unc_matrix), unc_matrix, np.nan)
                with np.errstate(all='ignore'):
                    _umed = np.nanmedian(finite_unc, axis=0)
                    _uq75 = np.nanpercentile(finite_unc, 75, axis=0)
                    _uq25 = np.nanpercentile(finite_unc, 25, axis=0)
                    _uiqr = _uq75 - _uq25
                _uiqr = np.where(
                    _uiqr > 0,
                    _uiqr,
                    _positive_scale(
                        uncertainty_iqr,
                        uncertainty_std,
                        uncertainty_median,
                    ),
                )
                uncertainty_median_per_wl = _umed.tolist()
                uncertainty_iqr_per_wl = _uiqr.tolist()

        stats_by_name[block_name] = BlockNormalizationStats(
            name=block_name,
            modality=block_group[0].modality.value,
            value_mean=0.0 if value_mean is None else value_mean,
            value_std=1.0 if value_std in (None, 0.0) else value_std,
            uncertainty_mean=uncertainty_mean,
            uncertainty_std=uncertainty_std,
            coordinate_mean=coordinate_mean,
            coordinate_std=coordinate_std,
            n_samples=len(block_group),
            n_values=n_values,
            value_median=value_median,
            value_iqr=value_iqr,
            uncertainty_median=uncertainty_median,
            uncertainty_iqr=uncertainty_iqr,
            value_median_per_wl=value_median_per_wl,
            value_iqr_per_wl=value_iqr_per_wl,
            uncertainty_median_per_wl=uncertainty_median_per_wl,
            uncertainty_iqr_per_wl=uncertainty_iqr_per_wl,
            log_median_flux=log_median_flux,
        )

    return stats_by_name


def fit_task_preprocessing(
    training_observations: list[list[ObservationBlock]],
    version: str = "0.5.0",
    metadata: Mapping[str, Any] | None = None,
) -> TaskPreprocessingMetadata:
    """Fit preprocessing statistics from training observation blocks."""
    flattened_blocks = [
        block for sample_blocks in training_observations for block in sample_blocks
    ]
    return TaskPreprocessingMetadata(
        version=version,
        blocks=compute_block_normalization_stats(flattened_blocks),
        metadata=dict(metadata or {}),
    )


def normalize_observation_block(
    block: ObservationBlock,
    preprocessing_metadata: TaskPreprocessingMetadata,
) -> ObservationBlock:
    """Normalize one observation block with fitted preprocessing statistics.

    Uses robust (median/IQR) normalization for values and uncertainties when
    available, falling back to mean/std for backward compatibility with older
    preprocessing metadata.  Coordinates always use mean/std normalization.
    """
    stats = preprocessing_metadata.blocks.get(block.name)
    if stats is None:
        return block

    raw_values = np.asarray(block.values, dtype=float)
    is_spectral_block = (
        stats.modality == "spectrum"
        and raw_values.reshape(-1).shape[0] > 1
    )
    sample_robust_scale = None
    global_spectral_scale = None

    if is_spectral_block:
        sample_robust_scale = _sample_robust_scale(raw_values, block.mask)
        global_spectral_scale = _global_spectral_scale(stats)
        if sample_robust_scale is None and stats.log_median_flux is not None:
            sample_robust_scale = float(10.0 ** stats.log_median_flux)
        if sample_robust_scale is None:
            sample_robust_scale = _positive_scale(
                stats.value_median,
                stats.value_mean,
                default=1.0,
            )
        normalized_values = _apply_spectral_value_transform(
            raw_values,
            sample_robust_scale,
        )
    else:
        if stats.value_median is not None and stats.value_iqr is not None and stats.value_iqr > 0:
            normalized_values = (raw_values - stats.value_median) / _positive_scale(
                stats.value_iqr,
                stats.value_std,
                stats.value_median,
            )
        else:
            normalized_values = (raw_values - stats.value_mean) / _positive_scale(
                stats.value_std,
                stats.value_median,
            )

    normalized_uncertainties = block.uncertainties
    if block.uncertainties is not None:
        raw_unc = np.asarray(block.uncertainties, dtype=float)
        if is_spectral_block and sample_robust_scale is not None:
            normalized_uncertainties = _transform_spectral_uncertainties(
                raw_unc,
                raw_values,
                sample_robust_scale,
            )
        else:
            if (
                stats.uncertainty_median_per_wl is not None
                and len(stats.uncertainty_median_per_wl) == raw_unc.reshape(-1).shape[0]
            ):
                unc_scale = _positive_scale_vector(
                    stats.uncertainty_median_per_wl,
                    fallback=_positive_scale(
                        stats.uncertainty_median,
                        stats.uncertainty_std,
                        np.nanmedian(np.abs(raw_unc)),
                    ),
                )
                normalized_uncertainties = raw_unc / unc_scale
            else:
                normalized_uncertainties = raw_unc / _positive_scale(
                    stats.uncertainty_median,
                    stats.uncertainty_std,
                    np.nanmedian(np.abs(raw_unc)),
                )

    # --- Coordinates: always use mean/std (linearly spaced, no outliers) ---
    normalized_coordinates = block.coordinates
    if (
        block.coordinates is not None
        and stats.coordinate_mean is not None
        and stats.coordinate_std not in (None, 0.0)
    ):
        normalized_coordinates = (
            np.asarray(block.coordinates, dtype=float) - stats.coordinate_mean
        ) / max(stats.coordinate_std, 1e-8)

    metadata = dict(block.metadata)
    sample_log_median_flux = _sample_log_median_flux(raw_values, block.mask)
    if is_spectral_block and sample_robust_scale is not None:
        sample_log_scale = float(
            np.log10(max(sample_robust_scale, _NORMALIZATION_SCALE_FLOOR))
        )
        metadata["absolute_values"] = _apply_absolute_spectral_value_transform(
            raw_values,
            global_spectral_scale if global_spectral_scale is not None else sample_robust_scale,
        )
        metadata["global_spectrum_scale"] = float(
            global_spectral_scale if global_spectral_scale is not None else sample_robust_scale
        )
        metadata["spectrum_scale"] = float(sample_robust_scale)
        metadata["log_spectrum_scale"] = sample_log_scale
        metadata["log_median_flux"] = sample_log_scale
        metadata["value_transform"] = "asinh_sample_scale"
        metadata["absolute_value_transform"] = "asinh_global_scale"
    elif sample_log_median_flux is not None:
        metadata["log_median_flux"] = sample_log_median_flux
    elif stats.log_median_flux is not None:
        metadata.setdefault("log_median_flux", stats.log_median_flux)
        metadata.setdefault("log_spectrum_scale", stats.log_median_flux)
        metadata.setdefault(
            "spectrum_scale",
            float(10.0 ** stats.log_median_flux),
        )

    return ObservationBlock(
        name=block.name,
        modality=block.modality,
        values=normalized_values,
        uncertainties=normalized_uncertainties,
        coordinates=normalized_coordinates,
        mask=block.mask,
        metadata=metadata,
    )


def normalize_observation_blocks(
    blocks: list[ObservationBlock],
    preprocessing_metadata: TaskPreprocessingMetadata,
) -> list[ObservationBlock]:
    """Normalize a list of observation blocks."""
    return [
        normalize_observation_block(block, preprocessing_metadata) for block in blocks
    ]


__all__ = [
    "BlockNormalizationStats",
    "TaskPreprocessingMetadata",
    "compute_block_normalization_stats",
    "fit_task_preprocessing",
    "normalize_observation_block",
    "normalize_observation_blocks",
]
