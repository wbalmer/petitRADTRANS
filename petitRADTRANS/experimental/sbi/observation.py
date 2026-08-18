"""
Observation containers and block builders for amortized inference.

This module implements classes to store real or simulated observations
in structured, frozen data classes, enabling them to be passed through
JAX transformations and JIT compilation. It also includes functions
to build batches of observations, enabling parallelised processing of
multiple samples, while also limiting peak memory usage.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import jax
import numpy as np


class ObservationModality(str, Enum):
    """Supported observation block types for SBI conditioning."""

    SPECTRUM = "spectrum"
    PHOTOMETRY = "photometry"
    TIME_SERIES = "time_series"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class ObservationBlock:
    """Representation of one (simulated or real) observation.

    The modality can be 'spectrum', 'photometry', 'time_series', or 'auxiliary'.
    However, for SBI usage we're generally only concerned with spectrum.

    Attributes:
        name:
            Stable identifier of the block within a task.
        modality:
            Description of the type of observation, can be one of
            'spectrum', 'photometry', 'time_series', or 'auxiliary'.
        values:
            Observed values after any task-level preprocessing. This
            stores either transit depths or fluxes, modulo some processing.
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


class AbstractObservationEncoder(abc.ABC):
    """Transform structured observation blocks into model-ready embeddings.
    This allows an observation to be represented in a shared latent space,
    with the encoders implemented in encorders.py.
    """

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


@dataclass(frozen=True)
class PreStackedObservations:
    """Pre-stacked observation arrays ready to cross the JAX JIT boundary.

    All per-sample ObservationBlock data has been extracted into dense numpy
    arrays before this object is constructed, so JAX can trace through the
    encoder without re-compiling per batch. Registered as a pytree node
    to enable JIT compilation.

    Attributes
    ----------
    stacked_blocks:
        One ``(values, uncertainties, coordinates, mask, log_median_flux,
        spectral_auxiliary_values, global_spectrum_scale, exact_summary)``
        tuple per observation block. ``values``, ``uncertainties``,
        ``coordinates``, ``mask``, and ``spectral_auxiliary_values`` have shape
        ``(batch_size, n_wl)``; ``log_median_flux`` and
        ``global_spectrum_scale`` have shape ``(batch_size,)``; and
        ``exact_summary`` has shape ``(batch_size, EXACT_SUMMARY_DIM)`` (see
        :func:`~petitRADTRANS.experimental.sbi.preprocessing.exact_summary_feature_matrix`).
        Trailing fields are optional: consumers read them by index and fall
        back to zeros, so tuples written before a field existed still load.
    modalities:
        Modality value string for each block, e.g. ``'spectrum'``.
        Treated as static metadata by JAX (does not trigger retrace between
        batches since it is constant for a given SBI task).
    """

    stacked_blocks: tuple  # tuple[(v, u, c, m, lmf, aux, gsf), ...], one entry per block
    modalities: tuple  # tuple[str, ...] – static pytree aux data


def _prestacked_flatten(
    obs: PreStackedObservations,
) -> tuple[list, tuple]:
    leaves = [arr for block in obs.stacked_blocks for arr in block]
    n_fields = len(obs.stacked_blocks[0]) if obs.stacked_blocks else 6
    return leaves, (obs.modalities, len(obs.stacked_blocks), n_fields)


def _prestacked_unflatten(
    aux_data: tuple,
    leaves: list,
) -> PreStackedObservations:
    modalities, n_blocks, n_fields = aux_data
    stacked_blocks = tuple(
        tuple(leaves[i * n_fields + j] for j in range(n_fields))
        for i in range(n_blocks)
    )
    return PreStackedObservations(stacked_blocks=stacked_blocks, modalities=modalities)


jax.tree_util.register_pytree_node(
    PreStackedObservations,
    _prestacked_flatten,
    _prestacked_unflatten,
)


def build_raw_observation_blocks(retrieval_config: Any) -> list:
    """Build raw (un-normalized) spectrum ObservationBlocks for every dataset in
    a retrieval configuration -- the target observation(s) for amortized
    inference and the startup scale guard.
    """
    return [
        build_observation_block(
            name=name,
            modality=ObservationModality.SPECTRUM,
            values=data.spectrum,
            uncertainties=data.uncertainties,
            coordinates=data.wavelengths,
            mask=data.mask,
            metadata={
                'line_opacity_mode': data.line_opacity_mode,
                'source_name': data.name,
            },
        )
        for name, data in retrieval_config.data.items()
    ]


__all__ = [
    "EncodedObservation",
    "ObservationBlock",
    "AbstractObservationEncoder",
    "ObservationModality",
    "PreStackedObservations",
    "build_observation_block",
    "build_observation_block_batch",
    "build_observation_blocks_from_sample",
    "build_raw_observation_blocks",
]
