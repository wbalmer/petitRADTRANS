"""Dataset storage interfaces for SBI simulation corpora."""

from __future__ import annotations

import abc
from dataclasses import asdict
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .artifacts import ArtifactMetadata, ArtifactRole
from .observation import build_observation_block_batch
from .posterior import PosteriorBatch
from .preprocessing import (
    TaskPreprocessingMetadata,
    fit_task_preprocessing,
    normalize_observation_blocks,
)
from .simulator import SimulationBatch
from .task import SBITask

try:
    import zarr
except ImportError:  # pragma: no cover - exercised through runtime guards.
    zarr = None


class DatasetSplit(str, Enum):
    """Named dataset partitions used during training and evaluation."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"


class SplitSamplingPolicy(str, Enum):
    """Deterministic policies for assigning generated samples to dataset splits."""

    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class SimulationDatasetManifest:
    """Describe a stored SBI simulation corpus."""

    task_name: str
    parameter_names: tuple[str, ...]
    n_simulations: int
    splits: Mapping[str, int] = field(default_factory=dict)
    storage_uri: str | None = None
    task_fingerprint: str | None = None
    observation_schema: Mapping[str, Any] = field(default_factory=dict)
    available_fields: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: SBITask,
        storage_uri: str,
        n_simulations: int = 0,
        splits: Mapping[str, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SimulationDatasetManifest":
        return cls(
            task_name=task.name,
            parameter_names=task.parameter_names,
            n_simulations=n_simulations,
            splits=dict(splits or {}),
            storage_uri=storage_uri,
            task_fingerprint=task.task_fingerprint.digest,
            observation_schema={
                "dataset_names": task.observation_schema.dataset_names,
                "modalities": dict(task.observation_schema.modalities),
            },
            metadata=dict(metadata or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_names"] = list(self.parameter_names)
        payload["available_fields"] = list(self.available_fields)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SimulationDatasetManifest":
        payload_dict = dict(payload)
        payload_dict["parameter_names"] = tuple(payload_dict.get("parameter_names", ()))
        payload_dict["available_fields"] = tuple(
            payload_dict.get("available_fields", ())
        )
        return cls(**payload_dict)


class SimulationDatasetWriter(abc.ABC):
    """Append-only writer for chunked simulation datasets."""

    @abc.abstractmethod
    def append(
        self, batch: SimulationBatch, split: DatasetSplit = DatasetSplit.TRAIN
    ) -> None:
        """Persist one simulation batch to the selected split."""

    @abc.abstractmethod
    def finalize(self) -> SimulationDatasetManifest:
        """Seal the dataset and return its manifest."""


class SimulationDatasetStore(abc.ABC):
    """Backend-independent interface for reading and writing simulation data."""

    @abc.abstractmethod
    def create_writer(
        self, manifest: SimulationDatasetManifest
    ) -> SimulationDatasetWriter:
        """Create a writer for a new simulation dataset."""

    @abc.abstractmethod
    def open(self, manifest_or_uri: SimulationDatasetManifest | str) -> Any:
        """Open a stored dataset for training or evaluation."""


@dataclass(frozen=True)
class StoredSimulationDataset:
    """Read-only handle for a stored chunked simulation dataset."""

    manifest: SimulationDatasetManifest
    storage_uri: str
    root: Any

    def read_split(self, split: DatasetSplit = DatasetSplit.TRAIN) -> dict[str, Any]:
        split_name = split.value

        if split_name not in self.root:
            raise KeyError(
                f"Split '{split_name}' is not present in dataset '{self.storage_uri}'."
            )

        split_group = self.root[split_name]
        loaded = {
            "parameters": split_group["parameters"][:],
            "cube_parameters": (
                split_group["cube_parameters"][:]
                if "cube_parameters" in split_group
                else None
            ),
            "unconstrained_parameters": (
                split_group["unconstrained_parameters"][:]
                if "unconstrained_parameters" in split_group
                else None
            ),
            "log_likelihood": (
                split_group["log_likelihood"][:]
                if "log_likelihood" in split_group
                else None
            ),
            "observations": {},
        }

        if "observations" not in split_group:
            return loaded

        for observation_name in split_group["observations"].keys():
            observation_group = split_group["observations"][observation_name]
            observation_payload = {}

            for field_name in observation_group.array_keys():
                observation_payload[field_name] = observation_group[field_name][:]

            if "metadata_json" in observation_group.attrs:
                observation_payload["metadata"] = json.loads(
                    observation_group.attrs["metadata_json"]
                )

            loaded["observations"][observation_name] = observation_payload

        return loaded


@dataclass(frozen=True)
class GeneratedSimulationDataset:
    """Return value for one-call simulation corpus generation."""

    manifest: SimulationDatasetManifest
    artifact_metadata: ArtifactMetadata
    preprocessing_metadata: TaskPreprocessingMetadata | None = None


def _validate_finite_split_observations(dataset: StoredSimulationDataset) -> None:
    """Abort when generated observations contain non-finite values."""
    for split_name, split_count in dataset.manifest.splits.items():
        if int(split_count) == 0:
            continue

        split = dataset.read_split(DatasetSplit(split_name))
        observations = split.get("observations", {})

        for observation_name, payload in observations.items():
            values = payload.get("values")
            if values is None:
                continue

            finite_mask = np.isfinite(np.asarray(values, dtype=float))
            if finite_mask.all():
                continue

            non_finite_count = int(finite_mask.size - finite_mask.sum())
            raise ValueError(
                "Generated simulation dataset contains non-finite observation values: "
                f"split='{split_name}', observation='{observation_name}', "
                f"non_finite_count={non_finite_count}, total_values={int(finite_mask.size)}."
            )


def _validate_preprocessing_metadata(
    preprocessing_metadata: TaskPreprocessingMetadata | None,
) -> None:
    """Abort when preprocessing stats indicate no finite training values."""
    if preprocessing_metadata is None:
        return

    for block_name, stats in preprocessing_metadata.blocks.items():
        if int(stats.n_values) > 0:
            continue

        raise ValueError(
            "Generated simulation dataset preprocessing is invalid: "
            f"block '{block_name}' has n_values == 0, indicating no finite training observations."
        )


@dataclass(frozen=True)
class NormalizedObservationDatasetReader:
    """Lightweight reader yielding normalized ObservationBlock batches for training."""

    dataset: StoredSimulationDataset
    preprocessing_metadata: TaskPreprocessingMetadata

    def iter_batches(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        parameter_space: str = "physical",
        encoder: Any = None,
    ) -> Iterator[PosteriorBatch]:
        """Yield mini-batches of normalized observations and matched parameters.

        Parameters
        ----------
        split:
            Dataset split to iterate over.
        batch_size:
            Number of samples yielded in each batch.
        shuffle:
            Whether to shuffle sample order within the requested split.
        seed:
            Optional random seed used when ``shuffle=True``.
        parameter_space:
            Parameter representation returned in each batch. Supported values
            are ``'physical'``, ``'cube'``, and ``'unconstrained'``.
        encoder:
            Optional encoder used to convert block lists into dense embedding
            arrays before batches are yielded.

        Returns
        -------
        Iterator[PosteriorBatch]
            Iterator over batches containing parameters, observations, and small
            metadata dictionaries describing the batch provenance.

        Notes
        -----
        Observation blocks are always normalized with the stored preprocessing
        metadata before they are exposed to training code.
        """
        split_data = self.dataset.read_split(split)
        raw_blocks = build_observation_block_batch(
            split_data["observations"],
            self.dataset.manifest.observation_schema["modalities"],
        )
        normalized_blocks = [
            normalize_observation_blocks(blocks, self.preprocessing_metadata)
            for blocks in raw_blocks
        ]

        if parameter_space == "physical":
            parameters = split_data["parameters"]
        elif parameter_space == "cube":
            parameters = split_data["cube_parameters"]
        elif parameter_space == "unconstrained":
            parameters = split_data["unconstrained_parameters"]
        else:
            raise ValueError(f"Unsupported parameter_space: {parameter_space!r}")

        indices = np.arange(len(normalized_blocks))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_observations = [normalized_blocks[index] for index in batch_indices]

            if encoder is not None:
                batch_observations = np.asarray(
                    [encoder.encode(blocks).embedding for blocks in batch_observations],
                    dtype=float,
                )

            yield PosteriorBatch(
                parameters=(
                    None
                    if parameters is None
                    else np.asarray(parameters)[batch_indices]
                ),
                observations=batch_observations,
                metadata={
                    "split": split.value,
                    "parameter_space": parameter_space,
                    "batch_indices": batch_indices.tolist(),
                    "encoded": encoder is not None,
                },
            )


class ZarrSimulationDatasetWriter(SimulationDatasetWriter):
    """Append-only Zarr-backed writer for simulation corpora."""

    def __init__(
        self, manifest: SimulationDatasetManifest, root: Any, chunk_size: int = 256
    ):
        self._manifest = manifest
        self._root = root
        self._chunk_size = int(chunk_size)
        self._split_counts = {
            str(name): int(count) for name, count in manifest.splits.items()
        }
        self._available_fields = set(manifest.available_fields)
        self._write_manifest()

    @staticmethod
    def _get_or_create_group(parent: Any, name: str) -> Any:
        if name in parent:
            return parent[name]
        return parent.create_group(name)

    def _write_manifest(self) -> None:
        manifest_payload = self._manifest.to_payload()
        manifest_payload["splits"] = dict(self._split_counts)
        manifest_payload["available_fields"] = sorted(self._available_fields)
        manifest_payload["n_simulations"] = int(sum(self._split_counts.values()))
        self._root.attrs["manifest_json"] = json.dumps(manifest_payload, sort_keys=True)

    def _append_array(
        self, group: Any, name: str, values: Any, leading_chunk: int | None = None
    ) -> None:
        array_values = np.asarray(values)

        if array_values.ndim == 0:
            array_values = array_values.reshape(1)

        chunk_leading = self._chunk_size if leading_chunk is None else leading_chunk
        target_chunks = (
            min(chunk_leading, max(array_values.shape[0], 1)),
        ) + array_values.shape[1:]

        if name not in group:
            group.create_array(
                name,
                shape=(0,) + array_values.shape[1:],
                chunks=target_chunks,
                dtype=array_values.dtype,
            )

        array = group[name]
        start = array.shape[0]
        stop = start + array_values.shape[0]
        array.resize((stop,) + array.shape[1:])
        array[start:stop] = array_values

    def _append_observations(
        self, split_group: Any, observations: Mapping[str, Any]
    ) -> None:
        observations_group = self._get_or_create_group(split_group, "observations")

        for observation_name, payload in observations.items():
            observation_group = self._get_or_create_group(
                observations_group, observation_name
            )

            metadata_payload = payload.get("metadata")
            if metadata_payload is not None:
                if isinstance(metadata_payload, list):
                    if len(metadata_payload) == 0:
                        metadata_payload = None
                    else:
                        first_metadata = metadata_payload[0]
                        if any(item != first_metadata for item in metadata_payload[1:]):
                            raise ValueError(
                                f"Observation metadata for dataset '{observation_name}' varies within one batch."
                            )
                        metadata_payload = first_metadata

            if metadata_payload is not None:
                metadata_json = json.dumps(metadata_payload, sort_keys=True)
                existing_metadata_json = observation_group.attrs.get("metadata_json")

                if (
                    existing_metadata_json is not None
                    and existing_metadata_json != metadata_json
                ):
                    raise ValueError(
                        f"Observation metadata mismatch for dataset '{observation_name}' across appended batches."
                    )

                observation_group.attrs["metadata_json"] = metadata_json

            for field_name, values in payload.items():
                if field_name == "metadata" or values is None:
                    continue
                self._append_array(observation_group, field_name, values)

    def append(
        self, batch: SimulationBatch, split: DatasetSplit = DatasetSplit.TRAIN
    ) -> None:
        split_name = split.value
        split_group = self._get_or_create_group(self._root, split_name)
        self._append_array(split_group, "parameters", batch.parameters)
        self._available_fields.add("parameters")

        if batch.cube_parameters is not None:
            self._append_array(split_group, "cube_parameters", batch.cube_parameters)
            self._available_fields.add("cube_parameters")

        if batch.unconstrained_parameters is not None:
            self._append_array(
                split_group, "unconstrained_parameters", batch.unconstrained_parameters
            )
            self._available_fields.add("unconstrained_parameters")

        if batch.log_likelihood is not None:
            self._append_array(split_group, "log_likelihood", batch.log_likelihood)
            self._available_fields.add("log_likelihood")

        self._append_observations(split_group, batch.observations)
        self._available_fields.add("observations")
        self._split_counts[split_name] = (
            self._split_counts.get(split_name, 0) + batch.n_samples
        )
        self._write_manifest()

    def finalize(self) -> SimulationDatasetManifest:
        finalized_manifest = SimulationDatasetManifest(
            task_name=self._manifest.task_name,
            parameter_names=self._manifest.parameter_names,
            n_simulations=int(sum(self._split_counts.values())),
            splits=dict(self._split_counts),
            storage_uri=self._manifest.storage_uri,
            task_fingerprint=self._manifest.task_fingerprint,
            observation_schema=self._manifest.observation_schema,
            available_fields=tuple(sorted(self._available_fields)),
            metadata=dict(self._manifest.metadata),
        )
        self._manifest = finalized_manifest
        self._write_manifest()
        return finalized_manifest


class ZarrSimulationDatasetStore(SimulationDatasetStore):
    """Concrete Zarr-backed store for chunked simulation corpora."""

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _require_zarr() -> None:
        if zarr is None:
            raise ImportError(
                "Zarr is required for SimulationDatasetStore operations. Install petitRADTRANS with the 'sbi' extra."
            )

    def create_writer(
        self,
        manifest: SimulationDatasetManifest,
        mode: str = "w",
    ) -> SimulationDatasetWriter:
        self._require_zarr()

        if not manifest.storage_uri:
            raise ValueError(
                "SimulationDatasetManifest.storage_uri must be set before creating a writer."
            )

        storage_path = Path(manifest.storage_uri)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(storage_path.as_posix(), mode=mode)

        if mode == "a" and "manifest_json" in root.attrs:
            existing_manifest = SimulationDatasetManifest.from_payload(
                json.loads(root.attrs["manifest_json"])
            )
            if (
                manifest.task_fingerprint is not None
                and existing_manifest.task_fingerprint is not None
                and manifest.task_fingerprint != existing_manifest.task_fingerprint
            ):
                raise ValueError(
                    "Existing dataset fingerprint does not match the requested task family."
                )
            manifest = existing_manifest

        return ZarrSimulationDatasetWriter(
            manifest=manifest, root=root, chunk_size=self.chunk_size
        )

    def open(
        self, manifest_or_uri: SimulationDatasetManifest | str
    ) -> StoredSimulationDataset:
        self._require_zarr()

        if isinstance(manifest_or_uri, SimulationDatasetManifest):
            storage_uri = manifest_or_uri.storage_uri
            manifest = manifest_or_uri
        else:
            storage_uri = manifest_or_uri
            manifest = None

        if not storage_uri:
            raise ValueError(
                "A dataset manifest or URI with a valid storage location is required."
            )

        root = zarr.open_group(Path(storage_uri).as_posix(), mode="r")

        if manifest is None:
            manifest_json = root.attrs.get("manifest_json")
            if manifest_json is None:
                raise ValueError(
                    f"Dataset at '{storage_uri}' does not contain a serialized manifest."
                )
            manifest = SimulationDatasetManifest.from_payload(json.loads(manifest_json))

        return StoredSimulationDataset(
            manifest=manifest, storage_uri=storage_uri, root=root
        )


def _resolve_split_counts(
    n_simulations: int,
    split_counts: Mapping[DatasetSplit | str, int] | None = None,
    split_fractions: Mapping[DatasetSplit | str, float] | None = None,
) -> dict[DatasetSplit, int]:
    if split_counts is not None and split_fractions is not None:
        raise ValueError("Specify either split_counts or split_fractions, not both.")

    if split_counts is None and split_fractions is None:
        return {DatasetSplit.TRAIN: n_simulations}

    if split_fractions is not None:
        normalized_fractions = {
            (key if isinstance(key, DatasetSplit) else DatasetSplit(key)): float(value)
            for key, value in split_fractions.items()
        }
        total_fraction = float(sum(normalized_fractions.values()))
        if not np.isclose(total_fraction, 1.0):
            raise ValueError("Split fractions must sum to 1.0.")

        split_names = list(normalized_fractions.keys())
        raw_counts = [
            normalized_fractions[split] * n_simulations for split in split_names
        ]
        floored_counts = [int(np.floor(count)) for count in raw_counts]
        remainder = int(n_simulations - sum(floored_counts))
        fractional_parts = np.argsort(
            [count - np.floor(count) for count in raw_counts]
        )[::-1]

        for index in fractional_parts[:remainder]:
            floored_counts[int(index)] += 1

        return {
            split_names[index]: floored_counts[index]
            for index in range(len(split_names))
        }

    normalized = {
        (key if isinstance(key, DatasetSplit) else DatasetSplit(key)): int(value)
        for key, value in split_counts.items()
    }
    if sum(normalized.values()) != int(n_simulations):
        raise ValueError("Split counts must sum to n_simulations.")
    return normalized


def _build_split_assignments(
    split_counts: Mapping[DatasetSplit, int],
    split_policy: SplitSamplingPolicy,
    split_seed: int | None,
) -> list[DatasetSplit]:
    if split_policy is SplitSamplingPolicy.SEQUENTIAL:
        assignments: list[DatasetSplit] = []
        for split, count in split_counts.items():
            assignments.extend([split] * int(count))
        return assignments

    if split_policy is SplitSamplingPolicy.ROUND_ROBIN:
        remaining = {split: int(count) for split, count in split_counts.items()}
        ordered_splits = list(split_counts.keys())
        assignments = []
        while any(count > 0 for count in remaining.values()):
            for split in ordered_splits:
                if remaining[split] > 0:
                    assignments.append(split)
                    remaining[split] -= 1
        return assignments

    sequential_assignments = _build_split_assignments(
        split_counts=split_counts,
        split_policy=SplitSamplingPolicy.SEQUENTIAL,
        split_seed=split_seed,
    )
    rng = np.random.default_rng(split_seed)
    shuffled_indices = np.arange(len(sequential_assignments))
    rng.shuffle(shuffled_indices)
    return [sequential_assignments[index] for index in shuffled_indices]


def _iter_assignment_runs(
    assignments: list[DatasetSplit],
) -> Iterator[tuple[DatasetSplit, int]]:
    if not assignments:
        return

    current_split = assignments[0]
    current_count = 1
    for split in assignments[1:]:
        if split == current_split:
            current_count += 1
            continue
        yield current_split, current_count
        current_split = split
        current_count = 1

    yield current_split, current_count


def _advance_simulator(
    simulator: Any,
    skipped_assignments: list[DatasetSplit],
    batch_size: int,
) -> None:
    for _split, run_count in _iter_assignment_runs(skipped_assignments):
        remaining = run_count
        while remaining > 0:
            consume = min(batch_size, remaining)
            simulator.simulate_batch(consume)
            remaining -= consume


def generate_simulation_dataset(
    task: SBITask,
    storage_uri: str,
    n_simulations: int,
    chunk_size: int = 256,
    split_counts: Mapping[DatasetSplit | str, int] | None = None,
    split_fractions: Mapping[DatasetSplit | str, float] | None = None,
    split_policy: SplitSamplingPolicy | str = SplitSamplingPolicy.SEQUENTIAL,
    split_seed: int | None = None,
    simulator: Any = None,
    include_preprocessing_metadata: bool = True,
    dataset_version: str = "0.1.0",
    resume: bool = False,
) -> GeneratedSimulationDataset:
    """Generate and persist a simulation corpus in one call."""
    resolved_split_policy = split_policy
    if isinstance(split_policy, str):
        resolved_split_policy = SplitSamplingPolicy(split_policy)

    split_plan = _resolve_split_counts(
        n_simulations=n_simulations,
        split_counts=split_counts,
        split_fractions=split_fractions,
    )
    split_assignments = _build_split_assignments(
        split_plan, resolved_split_policy, split_seed
    )
    dataset_store = ZarrSimulationDatasetStore(chunk_size=chunk_size)
    requested_manifest = SimulationDatasetManifest.from_task(
        task,
        storage_uri=storage_uri,
        metadata={
            "dataset_version": dataset_version,
            "split_policy": resolved_split_policy.value,
            "split_seed": split_seed,
        },
    )

    existing_count = 0
    if resume and Path(storage_uri).exists():
        existing_dataset = dataset_store.open(storage_uri)
        existing_manifest = existing_dataset.manifest
        if existing_manifest.task_fingerprint != requested_manifest.task_fingerprint:
            raise ValueError(
                "Cannot resume dataset generation for a different task family."
            )
        if (
            existing_manifest.metadata.get("split_policy")
            != resolved_split_policy.value
        ):
            raise ValueError(
                "Cannot resume dataset generation with a different split policy."
            )
        if existing_manifest.metadata.get("split_seed") != split_seed:
            raise ValueError(
                "Cannot resume dataset generation with a different split seed."
            )
        existing_count = existing_manifest.n_simulations
        if existing_count > n_simulations:
            raise ValueError(
                "Existing dataset already has more samples than requested n_simulations."
            )
        writer = dataset_store.create_writer(requested_manifest, mode="a")
    else:
        writer = dataset_store.create_writer(requested_manifest, mode="w")

    simulation_backend = simulator
    if simulation_backend is None:
        from .simulator import RuntimeSimulator

        simulation_backend = RuntimeSimulator(task=task)

    if resume and existing_count > 0:
        _advance_simulator(
            simulator=simulation_backend,
            skipped_assignments=split_assignments[:existing_count],
            batch_size=task.simulation_config.batch_size,
        )

    for split, split_size in _iter_assignment_runs(split_assignments[existing_count:]):
        remaining = int(split_size)
        while remaining > 0:
            batch_size = min(task.simulation_config.batch_size, remaining)
            batch = simulation_backend.simulate_batch(batch_size)
            writer.append(batch, split=split)

            remaining -= batch_size

    finalized_manifest = writer.finalize()
    finalized_dataset = dataset_store.open(finalized_manifest)
    _validate_finite_split_observations(finalized_dataset)

    preprocessing_metadata = None
    if include_preprocessing_metadata:
        training_split = finalized_dataset.read_split(DatasetSplit.TRAIN)
        training_blocks = build_observation_block_batch(
            training_split["observations"],
            task.observation_schema.modalities,
        )
        preprocessing_metadata = fit_task_preprocessing(
            training_blocks,
            version=task.preprocessing_version,
            metadata={
                "task_name": task.name,
                "task_fingerprint": task.task_fingerprint.digest,
                "split_policy": resolved_split_policy.value,
            },
        )
        _validate_preprocessing_metadata(preprocessing_metadata)

    artifact_metadata = ArtifactMetadata.from_task(
        task=task,
        role=ArtifactRole.SIMULATION_DATASET,
        version=dataset_version,
        metadata={
            "storage_uri": storage_uri,
            "n_simulations": finalized_manifest.n_simulations,
            "splits": dict(finalized_manifest.splits),
        },
        preprocessing_metadata=(
            {}
            if preprocessing_metadata is None
            else preprocessing_metadata.to_payload()
        ),
    )

    return GeneratedSimulationDataset(
        manifest=finalized_manifest,
        artifact_metadata=artifact_metadata,
        preprocessing_metadata=preprocessing_metadata,
    )


__all__ = [
    "DatasetSplit",
    "GeneratedSimulationDataset",
    "NormalizedObservationDatasetReader",
    "SimulationDatasetManifest",
    "SimulationDatasetStore",
    "SimulationDatasetWriter",
    "SplitSamplingPolicy",
    "StoredSimulationDataset",
    "ZarrSimulationDatasetStore",
    "ZarrSimulationDatasetWriter",
    "generate_simulation_dataset",
]
