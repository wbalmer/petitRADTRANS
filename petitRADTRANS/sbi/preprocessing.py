"""Preprocessing helpers and normalization metadata for SBI observation blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from .observation import ObservationBlock


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
            'version': self.version,
            'blocks': {
                name: stats.to_payload()
                for name, stats in self.blocks.items()
            },
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'TaskPreprocessingMetadata':
        payload_dict = dict(payload)
        blocks = {
            name: BlockNormalizationStats(**stats)
            for name, stats in payload_dict.get('blocks', {}).items()
        }
        return cls(
            version=str(payload_dict.get('version', '0.1.0')),
            blocks=blocks,
            metadata=dict(payload_dict.get('metadata', {})),
        )


def _finite_mean_std(values: Any) -> tuple[float | None, float | None, int]:
    if values is None:
        return None, None, 0

    array_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = array_values[np.isfinite(array_values)]
    if finite_values.size == 0:
        return None, None, 0

    return float(np.mean(finite_values)), float(np.std(finite_values)), int(finite_values.size)


def compute_block_normalization_stats(blocks: list[ObservationBlock]) -> dict[str, BlockNormalizationStats]:
    """Compute per-block normalization statistics from training observations."""
    grouped_blocks: dict[str, list[ObservationBlock]] = {}

    for block in blocks:
        grouped_blocks.setdefault(block.name, []).append(block)

    stats_by_name: dict[str, BlockNormalizationStats] = {}

    for block_name, block_group in grouped_blocks.items():
        concatenated_values = np.concatenate([
            np.asarray(block.values, dtype=float).reshape(-1)
            for block in block_group
        ])
        concatenated_uncertainties = []
        concatenated_coordinates = []

        for block in block_group:
            if block.uncertainties is not None:
                concatenated_uncertainties.append(np.asarray(block.uncertainties, dtype=float).reshape(-1))
            if block.coordinates is not None:
                concatenated_coordinates.append(np.asarray(block.coordinates, dtype=float).reshape(-1))

        value_mean, value_std, n_values = _finite_mean_std(concatenated_values)
        uncertainty_mean, uncertainty_std, _ = _finite_mean_std(
            None if not concatenated_uncertainties else np.concatenate(concatenated_uncertainties)
        )
        coordinate_mean, coordinate_std, _ = _finite_mean_std(
            None if not concatenated_coordinates else np.concatenate(concatenated_coordinates)
        )

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
        )

    return stats_by_name


def fit_task_preprocessing(
    training_observations: list[list[ObservationBlock]],
    version: str = '0.1.0',
    metadata: Mapping[str, Any] | None = None,
) -> TaskPreprocessingMetadata:
    """Fit preprocessing statistics from training observation blocks."""
    flattened_blocks = [block for sample_blocks in training_observations for block in sample_blocks]
    return TaskPreprocessingMetadata(
        version=version,
        blocks=compute_block_normalization_stats(flattened_blocks),
        metadata=dict(metadata or {}),
    )


def normalize_observation_block(
    block: ObservationBlock,
    preprocessing_metadata: TaskPreprocessingMetadata,
) -> ObservationBlock:
    """Normalize one observation block with fitted preprocessing statistics."""
    stats = preprocessing_metadata.blocks.get(block.name)
    if stats is None:
        return block

    normalized_values = (np.asarray(block.values, dtype=float) - stats.value_mean) / max(stats.value_std, 1e-8)
    normalized_uncertainties = block.uncertainties
    normalized_coordinates = block.coordinates

    if (
        block.uncertainties is not None
        and stats.uncertainty_mean is not None
        and stats.uncertainty_std not in (None, 0.0)
    ):
        normalized_uncertainties = (
            np.asarray(block.uncertainties, dtype=float) - stats.uncertainty_mean
        ) / max(stats.uncertainty_std, 1e-8)

    if (
        block.coordinates is not None
        and stats.coordinate_mean is not None
        and stats.coordinate_std not in (None, 0.0)
    ):
        normalized_coordinates = (
            np.asarray(block.coordinates, dtype=float) - stats.coordinate_mean
        ) / max(stats.coordinate_std, 1e-8)

    return ObservationBlock(
        name=block.name,
        modality=block.modality,
        values=normalized_values,
        uncertainties=normalized_uncertainties,
        coordinates=normalized_coordinates,
        mask=block.mask,
        metadata=dict(block.metadata),
    )


def normalize_observation_blocks(
    blocks: list[ObservationBlock],
    preprocessing_metadata: TaskPreprocessingMetadata,
) -> list[ObservationBlock]:
    """Normalize a list of observation blocks."""
    return [normalize_observation_block(block, preprocessing_metadata) for block in blocks]


__all__ = [
    'BlockNormalizationStats',
    'TaskPreprocessingMetadata',
    'compute_block_normalization_stats',
    'fit_task_preprocessing',
    'normalize_observation_block',
    'normalize_observation_blocks',
]
