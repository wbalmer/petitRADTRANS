"""Observation containers and block builders for amortized inference."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ObservationModality(str, Enum):
    """Supported observation block types for SBI conditioning."""

    SPECTRUM = "spectrum"
    PHOTOMETRY = "photometry"
    TIME_SERIES = "time_series"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class ObservationBlock:
    """Represent one modality-specific observation block.

    Attributes:
        name:
            Stable identifier of the block within a task.
        modality:
            Semantic block type used to dispatch encoding logic.
        values:
            Observed values after any task-level preprocessing.
        uncertainties:
            Optional per-element uncertainty representation.
        coordinates:
            Optional coordinate arrays such as wavelengths or timestamps.
        mask:
            Optional mask applied to the values.
        metadata:
            Additional instrument and preprocessing metadata.
    """

    name: str
    modality: ObservationModality
    values: Any
    uncertainties: Any = None
    coordinates: Any = None
    mask: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedObservation:
    """Encoded representation consumed by a posterior estimator."""

    embedding: Any
    mask: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationEncoder(abc.ABC):
    """Transform structured observation blocks into model-ready embeddings."""

    @abc.abstractmethod
    def encode(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        """Encode a list of observation blocks into a shared representation."""

    def batch_encode(
        self, observations: list[list[ObservationBlock]]
    ) -> list[EncodedObservation]:
        """Encode multiple observations using repeated single-item encoding."""
        return [self.encode(blocks) for blocks in observations]


def build_observation_block(
    name: str,
    modality: ObservationModality | str,
    values: Any,
    uncertainties: Any = None,
    coordinates: Any = None,
    mask: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObservationBlock:
    """Build one observation block with modality normalization."""
    resolved_modality = modality
    if isinstance(modality, str):
        resolved_modality = ObservationModality(modality)

    return ObservationBlock(
        name=name,
        modality=resolved_modality,
        values=values,
        uncertainties=uncertainties,
        coordinates=coordinates,
        mask=mask,
        metadata=dict(metadata or {}),
    )


def build_observation_blocks_from_sample(
    observation_payloads: Mapping[str, Mapping[str, Any]],
    modalities: Mapping[str, str],
    sample_index: int,
) -> list[ObservationBlock]:
    """Build modality-aware observation blocks for one simulated sample."""
    blocks: list[ObservationBlock] = []

    for observation_name in modalities.keys():
        if observation_name not in observation_payloads:
            continue

        payload = observation_payloads[observation_name]
        metadata = payload.get("metadata")
        if isinstance(metadata, list):
            metadata = metadata[sample_index]

        def _select(
            field_name: str, observation_payload: Mapping[str, Any] = payload
        ) -> Any:
            if field_name not in observation_payload:
                return None

            field_value = observation_payload[field_name]
            if field_value is None:
                return None

            array_value = np.asarray(field_value)
            if array_value.ndim == 0:
                return array_value.item()
            if array_value.shape[0] <= sample_index:
                return field_value
            return field_value[sample_index]

        blocks.append(
            build_observation_block(
                name=observation_name,
                modality=modalities[observation_name],
                values=_select("values"),
                uncertainties=_select("uncertainties"),
                coordinates=_select("coordinates"),
                mask=_select("mask"),
                metadata=metadata,
            )
        )

    return blocks


def build_observation_block_batch(
    observation_payloads: Mapping[str, Mapping[str, Any]],
    modalities: Mapping[str, str],
) -> list[list[ObservationBlock]]:
    """Build observation blocks for each sample in a batched payload."""
    if not observation_payloads:
        return []

    first_payload = next(iter(observation_payloads.values()))
    sample_values = np.asarray(first_payload["values"])
    n_samples = int(sample_values.shape[0]) if sample_values.ndim > 0 else 1

    return [
        build_observation_blocks_from_sample(
            observation_payloads, modalities, sample_index
        )
        for sample_index in range(n_samples)
    ]


__all__ = [
    "EncodedObservation",
    "ObservationBlock",
    "ObservationEncoder",
    "ObservationModality",
    "build_observation_block",
    "build_observation_block_batch",
    "build_observation_blocks_from_sample",
]
