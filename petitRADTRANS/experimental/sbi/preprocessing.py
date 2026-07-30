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


def _apply_spectral_value_transform(values: Any, scale_value: float) -> np.ndarray:
    scale = max(float(scale_value), _NORMALIZATION_SCALE_FLOOR)
    raw_values = np.asarray(values, dtype=float)
    return np.arcsinh(raw_values / scale)


def _per_wavelength_stats_arrays(
    stats: "BlockNormalizationStats",
    n_wavelengths: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-wavelength robust center/scale vectors, or ``None`` if unavailable.

    Returns the fitted per-wavelength median and (floored) IQR when both are
    present and match the block length; the caller then normalizes each
    wavelength as ``arcsinh((x - median_wl) / iqr_wl)``. Per-wavelength
    standardization removes the prior-median continuum and rescales the
    prior-induced spread to order unity at EVERY wavelength, so the overall
    amplitude is carried by all channels at O(1) instead of by one or two
    scalar summary features — the failure mode where the flow could not pin
    the observation's overall scale despite the data constraining it to
    ~0.02%. The arcsinh keeps the transform near-linear over the bulk while
    compressing heavy tails (emission fluxes span decades).
    """
    per_wl_median = stats.value_median_per_wl
    per_wl_iqr = stats.value_iqr_per_wl
    if (
        per_wl_median is None
        or per_wl_iqr is None
        or len(per_wl_median) != n_wavelengths
        or len(per_wl_iqr) != n_wavelengths
    ):
        return None
    center = np.asarray(per_wl_median, dtype=float)
    scale = _positive_scale_vector(
        per_wl_iqr,
        fallback=_positive_scale(
            stats.value_iqr,
            stats.value_std,
            stats.value_median,
            default=1.0,
        ),
    )
    return center, scale


def _apply_per_wavelength_value_transform(
    values: np.ndarray,
    uncertainties: np.ndarray | None,
    center_wl: np.ndarray,
    scale_wl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``arcsinh`` of the per-wavelength robust z-score, with matched errors.

    Uncertainties are mapped through the local slope of the transform so error
    bars stay consistent with the transformed values.
    """
    z_raw = (values - center_wl) / scale_wl
    normalized_values = np.arcsinh(z_raw)
    if uncertainties is None:
        return normalized_values, None
    safe_z = np.where(np.isfinite(z_raw), z_raw, 0.0)
    local_scale = scale_wl * np.sqrt(1.0 + safe_z**2)
    normalized_uncertainties = np.asarray(uncertainties, dtype=float) / np.maximum(
        local_scale, _NORMALIZATION_SCALE_FLOOR
    )
    return normalized_values, normalized_uncertainties


def _apply_relative_spectral_value_transform(
    values: Any,
    sample_scale: float,
    sample_center: float,
) -> np.ndarray:
    scale = max(float(sample_scale), _NORMALIZATION_SCALE_FLOOR)
    raw_values = np.asarray(values, dtype=float)
    return np.arcsinh((raw_values - float(sample_center)) / scale)


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
    """Compute per-block normalization statistics from training observations.
    Calculates the robust per-wavelength median/IQR, global scale, and `log_median_flux`
    from the simulated training set.
    """
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
    spectral_value_scale = None
    per_wavelength_stats = None
    per_wavelength_uncertainties: np.ndarray | None = None

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
        spectral_value_scale = (
            global_spectral_scale
            if global_spectral_scale is not None
            else sample_robust_scale
        )
        per_wavelength_stats = _per_wavelength_stats_arrays(
            stats, raw_values.reshape(-1).shape[0]
        )
        if per_wavelength_stats is not None:
            center_wl, scale_wl = per_wavelength_stats
            normalized_values, per_wavelength_uncertainties = (
                _apply_per_wavelength_value_transform(
                    raw_values.reshape(-1),
                    None
                    if block.uncertainties is None
                    else np.asarray(block.uncertainties, dtype=float).reshape(-1),
                    center_wl,
                    scale_wl,
                )
            )
            normalized_values = normalized_values.reshape(raw_values.shape)
            if per_wavelength_uncertainties is not None:
                per_wavelength_uncertainties = per_wavelength_uncertainties.reshape(
                    np.asarray(block.uncertainties, dtype=float).shape
                )
        else:
            normalized_values = _apply_spectral_value_transform(
                raw_values,
                spectral_value_scale,
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
        if is_spectral_block and per_wavelength_uncertainties is not None:
            normalized_uncertainties = per_wavelength_uncertainties
        elif is_spectral_block and spectral_value_scale is not None:
            normalized_uncertainties = _transform_spectral_uncertainties(
                raw_unc,
                raw_values,
                spectral_value_scale,
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
        valid_sample_values = _valid_sample_values(raw_values, block.mask)
        sample_center = (
            float(np.nanmedian(valid_sample_values))
            if valid_sample_values.size > 0
            else 0.0
        )
        sample_log_scale = float(
            np.log10(max(sample_robust_scale, _NORMALIZATION_SCALE_FLOOR))
        )
        absolute_values = _apply_spectral_value_transform(
            raw_values,
            spectral_value_scale if spectral_value_scale is not None else sample_robust_scale,
        )
        metadata["absolute_values"] = absolute_values
        metadata["relative_values"] = _apply_relative_spectral_value_transform(
            raw_values,
            sample_robust_scale,
            sample_center,
        )
        metadata["relative_spectrum_center"] = sample_center
        metadata["global_spectrum_scale"] = float(
            spectral_value_scale if spectral_value_scale is not None else sample_robust_scale
        )
        metadata["spectrum_scale"] = float(sample_robust_scale)
        metadata["log_spectrum_scale"] = sample_log_scale
        metadata["log_median_flux"] = sample_log_scale
        metadata["value_transform"] = (
            "asinh_per_wavelength_robust_z"
            if per_wavelength_stats is not None
            else "asinh_global_scale"
        )
        metadata["relative_value_transform"] = "asinh_centered_sample_scale"
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


def denormalize_spectral_values(
    normalized_values: Any,
    normalized_uncertainties: Any,
    stats: "BlockNormalizationStats",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Exact inverse of the spectral value transform, in raw physical units.

    MUST remain the exact inverse of the spectral branch of
    :func:`normalize_observation_block` / :func:`normalize_spectral_block_matrix`
    (there is a round-trip unit test). Anything that needs the observation in
    raw units -- notably the likelihood used by test-time refinement -- has to
    invert whichever transform was actually applied. A hardcoded inverse of the
    *global-scale* transform silently corrupted the target spectrum by ~100x
    (and flipped its sign) once per-wavelength normalization became the default,
    which is why the inverse now lives here beside the forward transform rather
    than being restated by each consumer.

    Returns ``(raw_values, raw_uncertainties)``; the uncertainty is ``None``
    when ``normalized_uncertainties`` is ``None``.
    """
    normalized = np.asarray(normalized_values, dtype=float)
    n_wavelengths = normalized.shape[-1]
    per_wavelength = _per_wavelength_stats_arrays(stats, n_wavelengths)

    if per_wavelength is not None:
        center, scale = per_wavelength
    else:
        center = 0.0
        scale = max(float(_global_spectral_scale(stats)), _NORMALIZATION_SCALE_FLOOR)

    raw_values = center + scale * np.sinh(normalized)
    if normalized_uncertainties is None:
        return raw_values, None

    # Forward: sigma_norm = sigma_raw / (scale * sqrt(1 + z^2)) with
    # z = sinh(normalized), so the local slope is scale * cosh(normalized).
    local_scale = scale * np.cosh(normalized)
    raw_uncertainties = np.asarray(normalized_uncertainties, dtype=float) * local_scale
    return raw_values, raw_uncertainties


def normalize_spectral_block_matrix(
    values: Any,
    uncertainties: Any,
    mask: Any,
    stats: BlockNormalizationStats,
) -> dict[str, np.ndarray]:
    """Vectorized spectral normalization for a ``(batch, n_wl)`` value matrix.

    Mirrors the spectral branch of :func:`normalize_observation_block` (same
    per-wavelength robust ``arcsinh`` z-transform — falling back to the
    global-scale ``arcsinh`` when per-wavelength statistics are absent —
    uncertainty rescaling, per-sample relative shape channel, and
    log-amplitude scalar) but operates on a dense batch at once. Used by the training-time noise-resampling path, which
    redraws observational noise per batch and therefore must re-normalize on
    the fly; a batch normalized here is numerically identical to normalizing
    each sample individually.

    Parameters
    ----------
    values:
        ``(batch, n_wl)`` raw (physical-units) spectral values.
    uncertainties:
        ``(batch, n_wl)`` raw uncertainties, or ``None``.
    mask:
        ``(batch, n_wl)`` boolean invalid-point mask (``True`` = masked), or
        ``None``.
    stats:
        Fitted :class:`BlockNormalizationStats` for this block family.

    Returns
    -------
    dict[str, np.ndarray]
        ``values`` / ``uncertainties`` / ``relative_values`` of shape
        ``(batch, n_wl)`` and ``log_spectrum_scale`` / ``global_spectrum_scale``
        of shape ``(batch,)`` — the fields consumed by the pre-stacked
        training representation.
    """
    raw_values = np.asarray(values, dtype=float)
    n_batch, n_wavelengths = raw_values.shape

    global_scale = max(float(_global_spectral_scale(stats)), _NORMALIZATION_SCALE_FLOOR)
    per_wavelength_stats = _per_wavelength_stats_arrays(stats, n_wavelengths)

    if mask is None:
        invalid = np.zeros(raw_values.shape, dtype=bool)
    else:
        invalid = np.broadcast_to(np.asarray(mask, dtype=bool), raw_values.shape)
    valid = np.isfinite(raw_values) & ~invalid
    masked_values = np.where(valid, raw_values, np.nan)
    with np.errstate(all="ignore"):
        sample_scale = np.nanmedian(np.abs(masked_values), axis=1)
        sample_center = np.nanmedian(masked_values, axis=1)

    # Row fallbacks, matching the per-sample path: rows with valid values but a
    # degenerate median fall back to 1.0 (_positive_scale default); rows with
    # no valid values at all fall back to the fitted global amplitude.
    has_valid = np.any(valid, axis=1)
    degenerate = ~np.isfinite(sample_scale) | (np.abs(sample_scale) <= _NORMALIZATION_SCALE_FLOOR)
    if stats.log_median_flux is not None:
        no_valid_fallback = float(10.0 ** stats.log_median_flux)
    else:
        no_valid_fallback = _positive_scale(stats.value_median, stats.value_mean, default=1.0)
    sample_scale = np.where(degenerate, 1.0, np.abs(sample_scale))
    sample_scale = np.where(has_valid, sample_scale, no_valid_fallback)
    sample_scale = np.maximum(sample_scale, _NORMALIZATION_SCALE_FLOOR)
    sample_center = np.where(np.isfinite(sample_center), sample_center, 0.0)

    if per_wavelength_stats is not None:
        center_wl, scale_wl = per_wavelength_stats
        normalized_values, normalized_uncertainties = (
            _apply_per_wavelength_value_transform(
                raw_values,
                None if uncertainties is None else np.asarray(uncertainties, dtype=float),
                center_wl,
                scale_wl,
            )
        )
        if normalized_uncertainties is None:
            normalized_uncertainties = np.zeros_like(raw_values)
    else:
        normalized_values = np.arcsinh(raw_values / global_scale)
        if uncertainties is not None:
            raw_unc = np.asarray(uncertainties, dtype=float)
            safe_ratio = np.where(np.isfinite(raw_values), raw_values / global_scale, 0.0)
            local_scale = global_scale * np.sqrt(1.0 + safe_ratio**2)
            normalized_uncertainties = raw_unc / np.maximum(local_scale, _NORMALIZATION_SCALE_FLOOR)
        else:
            normalized_uncertainties = np.zeros_like(raw_values)

    relative_values = np.arcsinh(
        (raw_values - sample_center[:, None]) / sample_scale[:, None]
    )

    return {
        "values": normalized_values,
        "uncertainties": normalized_uncertainties,
        "relative_values": relative_values,
        "log_spectrum_scale": np.log10(sample_scale),
        "global_spectrum_scale": np.full(n_batch, global_scale, dtype=float),
    }


__all__ = [
    "BlockNormalizationStats",
    "TaskPreprocessingMetadata",
    "compute_block_normalization_stats",
    "fit_task_preprocessing",
    "normalize_observation_block",
    "normalize_observation_blocks",
    "normalize_spectral_block_matrix",
    "denormalize_spectral_values",
]
