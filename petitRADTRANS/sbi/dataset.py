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
    import h5py
except ImportError:  # pragma: no cover - exercised through runtime guards.
    h5py = None


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
class GeneratedSimulationDataset:
    """Return value for one-call simulation corpus generation."""

    manifest: SimulationDatasetManifest
    artifact_metadata: ArtifactMetadata
    preprocessing_metadata: TaskPreprocessingMetadata | None = None


def _validate_finite_split_observations(dataset: HDF5StoredSimulationDataset) -> None:
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


# Splits smaller than this threshold (in number of simulations) are cached in
# RAM after the first read.  Larger splits are streamed directly from disk on
# every epoch to avoid exhausting available memory.
_STREAMING_THRESHOLD = 10_000

# Observation fields consumed by build_observation_block_batch / the encoder.
_OBSERVATION_TRAINING_FIELDS: frozenset[str] = frozenset(
    {"values", "uncertainties", "coordinates", "mask"}
)


def _prepare_batch_for_storage(
    batch: SimulationBatch,
    *,
    store_covariance: bool,
) -> SimulationBatch:
    """Return a storage-friendly copy of one simulated batch.

    When ``store_covariance`` is ``False`` the full covariance matrices are
    reduced to their diagonals before they are written to disk.  This keeps the
    stored field name as ``covariance`` while collapsing the shape from
    ``(n_samples, n_wavelengths, n_wavelengths)`` to
    ``(n_samples, n_wavelengths)``.
    """
    if store_covariance:
        return batch

    observations = batch.observations
    if not isinstance(observations, Mapping):
        return batch

    updated_observations: dict[str, Any] = {}
    changed = False

    for observation_name, payload in observations.items():
        if not isinstance(payload, Mapping):
            updated_observations[observation_name] = payload
            continue

        payload_dict = dict(payload)
        covariance = payload_dict.get("covariance")
        if covariance is not None:
            covariance_array = np.asarray(covariance)
            if (
                covariance_array.ndim >= 2
                and covariance_array.shape[-1] == covariance_array.shape[-2]
            ):
                payload_dict["covariance"] = np.diagonal(
                    covariance_array,
                    axis1=-2,
                    axis2=-1,
                )
                changed = True

        updated_observations[observation_name] = payload_dict

    if not changed:
        return batch

    return SimulationBatch(
        parameters=batch.parameters,
        observations=updated_observations,
        cube_parameters=batch.cube_parameters,
        unconstrained_parameters=batch.unconstrained_parameters,
        log_likelihood=batch.log_likelihood,
        diagnostics=batch.diagnostics,
    )


@dataclass
class NormalizedObservationDatasetReader:
    """Lightweight reader yielding normalized ObservationBlock batches for training."""

    dataset: HDF5StoredSimulationDataset
    preprocessing_metadata: TaskPreprocessingMetadata
    _split_cache: dict = field(default_factory=dict, repr=False, compare=False)

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
        Splits with fewer than ``_STREAMING_THRESHOLD`` rows are cached in RAM
        after the first read so that repeated epoch calls (e.g. validation) pay
        only slicing cost.  Larger splits are streamed directly from disk one
        batch at a time to avoid loading the full dataset into memory.
        """
        n_split = int(self.dataset.manifest.splits.get(split.value, 0))
        use_cache = n_split <= _STREAMING_THRESHOLD

        if use_cache:
            yield from self._iter_batches_cached(
                split, batch_size, shuffle, seed, parameter_space, encoder
            )
        else:
            yield from self._iter_batches_streaming(
                split, batch_size, shuffle, seed, parameter_space, encoder
            )

    def _iter_batches_cached(
        self,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool,
        seed: int | None,
        parameter_space: str,
        encoder: Any,
    ) -> Iterator[PosteriorBatch]:
        """Iterate using the full-split RAM cache (small splits)."""
        cache_key = (split, parameter_space)
        if cache_key not in self._split_cache:
            # read_split loads all fields; filter to training fields after load
            # to avoid keeping covariance (42 GB) in the RAM cache.
            split_data = self.dataset.read_split(split)
            for obs_payload in split_data.get("observations", {}).values():
                for field_name in list(obs_payload):
                    if field_name not in _OBSERVATION_TRAINING_FIELDS and field_name != "metadata":
                        del obs_payload[field_name]
            raw_blocks = build_observation_block_batch(
                split_data["observations"],
                self.dataset.manifest.observation_schema["modalities"],
            )
            normalized_blocks = [
                normalize_observation_blocks(blocks, self.preprocessing_metadata)
                for blocks in raw_blocks
            ]
            parameters = self._select_parameters(split_data, parameter_space)
            self._split_cache[cache_key] = (normalized_blocks, parameters)

        normalized_blocks, parameters = self._split_cache[cache_key]
        indices = np.arange(len(normalized_blocks))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_obs = [normalized_blocks[i] for i in batch_idx]
            if encoder is not None:
                batch_obs = np.asarray(encoder.encode_stacked_batch(batch_obs))
            yield PosteriorBatch(
                parameters=None if parameters is None else np.asarray(parameters)[batch_idx],
                observations=batch_obs,
                metadata={
                    "split": split.value,
                    "parameter_space": parameter_space,
                    "batch_indices": batch_idx.tolist(),
                    "encoded": encoder is not None,
                },
            )

    def _iter_batches_streaming(
        self,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool,
        seed: int | None,
        parameter_space: str,
        encoder: Any,
    ) -> Iterator[PosteriorBatch]:
        """Iterate by streaming rows directly from disk (large splits).

        When ``shuffle=True`` a global index permutation is computed in RAM
        (one integer per simulation row — negligible memory) and used to fetch
        HDF5 rows in sorted sub-windows of ``batch_size``, satisfying h5py's
        monotonic-index requirement while still presenting shuffled order to the
        training loop.
        """
        n_split = int(self.dataset.manifest.splits.get(split.value, 0))
        indices = np.arange(n_split)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)

        modalities = self.dataset.manifest.observation_schema["modalities"]
        batch_num = 0
        for chunk in self.dataset.iter_split_chunks(
            split, chunk_size=batch_size, indices=indices,
            observation_fields=_OBSERVATION_TRAINING_FIELDS,
        ):
            raw_blocks = build_observation_block_batch(chunk["observations"], modalities)
            normalized_blocks = [
                normalize_observation_blocks(blocks, self.preprocessing_metadata)
                for blocks in raw_blocks
            ]
            parameters = self._select_parameters(chunk, parameter_space)
            batch_obs: Any = normalized_blocks
            if encoder is not None:
                batch_obs = np.asarray(encoder.encode_stacked_batch(normalized_blocks))
            yield PosteriorBatch(
                parameters=parameters,
                observations=batch_obs,
                metadata={
                    "split": split.value,
                    "parameter_space": parameter_space,
                    "batch_indices": list(range(batch_num * batch_size, batch_num * batch_size + len(raw_blocks))),
                    "encoded": encoder is not None,
                },
            )
            batch_num += 1

    @staticmethod
    def _select_parameters(split_data: dict, parameter_space: str) -> Any:
        """Return the parameter array for the requested coordinate space."""
        if parameter_space == "physical":
            return split_data["parameters"]
        if parameter_space == "cube":
            return split_data["cube_parameters"]
        if parameter_space == "unconstrained":
            return split_data["unconstrained_parameters"]
        raise ValueError(f"Unsupported parameter_space: {parameter_space!r}")


# ---------------------------------------------------------------------------
# HDF5-backed implementation (single file, no file-count explosion)
# ---------------------------------------------------------------------------


class HDF5SimulationDatasetWriter(SimulationDatasetWriter):
    """Append-only HDF5-backed writer for simulation corpora.

    All simulation data is stored in a single ``.h5`` file, keeping the file
    count at 1 regardless of the number of simulations or splits.
    Datasets are created with an unlimited first dimension (``maxshape=(None,
    ...)``) so that batches can be appended incrementally without pre-allocation.
    """

    def __init__(
        self, manifest: SimulationDatasetManifest, file: Any, chunk_size: int = 256
    ):
        self._manifest = manifest
        self._file = file
        self._chunk_size = int(chunk_size)
        self._split_counts = {
            str(name): int(count) for name, count in manifest.splits.items()
        }
        self._available_fields = set(manifest.available_fields)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest_payload = self._manifest.to_payload()
        manifest_payload["splits"] = dict(self._split_counts)
        manifest_payload["available_fields"] = sorted(self._available_fields)
        manifest_payload["n_simulations"] = int(sum(self._split_counts.values()))
        self._file.attrs["manifest_json"] = json.dumps(manifest_payload, sort_keys=True)

    @staticmethod
    def _get_or_create_group(parent: Any, name: str) -> Any:
        if name in parent:
            return parent[name]
        return parent.require_group(name)

    def _append_array(self, group: Any, name: str, values: Any) -> None:
        array_values = np.asarray(values)
        if array_values.ndim == 0:
            array_values = array_values.reshape(1)

        n_new = array_values.shape[0]
        rest_shape = array_values.shape[1:]
        chunk_leading = min(self._chunk_size, max(n_new, 1))
        chunks = (chunk_leading,) + rest_shape if rest_shape else (chunk_leading,)

        if name not in group:
            max_shape = (None,) + rest_shape
            group.create_dataset(
                name,
                shape=(0,) + rest_shape,
                maxshape=max_shape,
                dtype=array_values.dtype,
                chunks=chunks,
            )

        dataset = group[name]
        start = dataset.shape[0]
        dataset.resize((start + n_new,) + rest_shape)
        dataset[start : start + n_new] = array_values

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
        split_group = self._get_or_create_group(self._file, split_name)
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
        self._file.flush()

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
        self._file.flush()
        self._file.close()
        return finalized_manifest


@dataclass(frozen=True)
class HDF5StoredSimulationDataset:
    """Read-only handle for an HDF5-backed simulation dataset."""

    manifest: SimulationDatasetManifest
    storage_uri: str
    _file_path: str

    def iter_split_chunks(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        chunk_size: int = 256,
        indices: Any = None,
        observation_fields: frozenset[str] | None = None,
    ) -> "Iterator[dict[str, Any]]":
        """Yield row-slices of one split without loading it all into RAM.

        Parameters
        ----------
        split:
            Dataset split to iterate.
        chunk_size:
            Number of rows per yielded chunk.
        indices:
            Optional integer array of row indices to read.  When supplied only
            those rows are fetched in ascending order.  When ``None`` every row
            in the split is visited sequentially.
        observation_fields:
            Optional set of observation field names to load.  When ``None``
            every dataset field is loaded.  Pass
            ``_OBSERVATION_TRAINING_FIELDS`` to skip large unused fields such
            as ``covariance`` during training.

        Yields
        ------
        dict[str, Any]
            Same structure as :meth:`read_split` but covering only
            ``chunk_size`` rows at a time.  The HDF5 file is opened once and
            kept open for the duration of the iteration.
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5SimulationDatasetStore operations."
            )
        split_name = split.value
        with h5py.File(self._file_path, "r") as root:
            if split_name not in root:
                raise KeyError(
                    f"Split '{split_name}' is not present in dataset '{self.storage_uri}'."
                )
            split_group = root[split_name]
            n_total = int(split_group["parameters"].shape[0])
            row_indices = np.arange(n_total) if indices is None else np.asarray(indices)
            obs_names = list(split_group["observations"].keys()) if "observations" in split_group else []
            obs_metadata: dict[str, Any] = {}
            for obs_name in obs_names:
                obs_group = split_group["observations"][obs_name]
                metadata_json = obs_group.attrs.get("metadata_json")
                if metadata_json is not None:
                    obs_metadata[obs_name] = json.loads(metadata_json)

            for start in range(0, len(row_indices), chunk_size):
                chunk_idx = np.sort(row_indices[start : start + chunk_size])
                chunk: dict[str, Any] = {
                    "parameters": split_group["parameters"][chunk_idx],
                    "cube_parameters": (
                        split_group["cube_parameters"][chunk_idx]
                        if "cube_parameters" in split_group
                        else None
                    ),
                    "unconstrained_parameters": (
                        split_group["unconstrained_parameters"][chunk_idx]
                        if "unconstrained_parameters" in split_group
                        else None
                    ),
                    "observations": {},
                }
                for obs_name in obs_names:
                    obs_group = split_group["observations"][obs_name]
                    payload: dict[str, Any] = {
                        field: obs_group[field][chunk_idx]
                        for field in obs_group.keys()
                        if isinstance(obs_group[field], h5py.Dataset)
                        and (observation_fields is None or field in observation_fields)
                    }
                    if obs_name in obs_metadata:
                        payload["metadata"] = obs_metadata[obs_name]
                    chunk["observations"][obs_name] = payload
                yield chunk

    def read_split(self, split: DatasetSplit = DatasetSplit.TRAIN) -> dict[str, Any]:
        split_name = split.value

        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5SimulationDatasetStore operations."
            )

        with h5py.File(self._file_path, "r") as root:
            if split_name not in root:
                raise KeyError(
                    f"Split '{split_name}' is not present in dataset '{self.storage_uri}'."
                )

            split_group = root[split_name]
            loaded: dict[str, Any] = {
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

            obs_group = split_group["observations"]
            for observation_name in obs_group.keys():
                observation_group = obs_group[observation_name]
                observation_payload: dict[str, Any] = {}

                for field_name, item in observation_group.items():
                    if isinstance(item, h5py.Dataset):
                        observation_payload[field_name] = item[:]

                metadata_json = observation_group.attrs.get("metadata_json")
                if metadata_json is not None:
                    observation_payload["metadata"] = json.loads(metadata_json)

                loaded["observations"][observation_name] = observation_payload

        return loaded


class HDF5SimulationDatasetStore(SimulationDatasetStore):
    """HDF5-backed store for simulation corpora.

    Stores all simulation data for a corpus in a single ``.h5`` file,
    keeping file counts at 1 regardless of the number of simulations or splits.
    Requires ``h5py`` (already present in the ``jaxprt`` environment).
    """

    def __init__(self, chunk_size: int = 256):
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _require_h5py() -> None:
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5SimulationDatasetStore operations. "
                "Install petitRADTRANS with the 'sbi' extra or run: pip install h5py"
            )

    def create_writer(
        self,
        manifest: SimulationDatasetManifest,
        mode: str = "w",
    ) -> HDF5SimulationDatasetWriter:
        self._require_h5py()

        if not manifest.storage_uri:
            raise ValueError(
                "SimulationDatasetManifest.storage_uri must be set before creating a writer."
            )

        storage_path = Path(manifest.storage_uri)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "a" and storage_path.exists():
            f = h5py.File(storage_path.as_posix(), "a")
            existing_manifest_json = f.attrs.get("manifest_json")
            if existing_manifest_json is not None:
                existing_manifest = SimulationDatasetManifest.from_payload(
                    json.loads(existing_manifest_json)
                )
                if (
                    manifest.task_fingerprint is not None
                    and existing_manifest.task_fingerprint is not None
                    and manifest.task_fingerprint != existing_manifest.task_fingerprint
                ):
                    f.close()
                    raise ValueError(
                        "Existing dataset fingerprint does not match the requested task family."
                    )
                manifest = existing_manifest
        else:
            f = h5py.File(storage_path.as_posix(), "w")

        return HDF5SimulationDatasetWriter(
            manifest=manifest, file=f, chunk_size=self.chunk_size
        )

    def open(
        self, manifest_or_uri: SimulationDatasetManifest | str
    ) -> HDF5StoredSimulationDataset:
        self._require_h5py()

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

        storage_path = Path(storage_uri)
        with h5py.File(storage_path.as_posix(), "r") as f:
            if manifest is None:
                manifest_json = f.attrs.get("manifest_json")
                if manifest_json is None:
                    raise ValueError(
                        f"Dataset at '{storage_uri}' does not contain a serialized manifest."
                    )
                manifest = SimulationDatasetManifest.from_payload(json.loads(manifest_json))

        return HDF5StoredSimulationDataset(
            manifest=manifest,
            storage_uri=storage_uri,
            _file_path=storage_path.as_posix(),
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
    # Prefer the cheap PRNG-only skip when the simulator supports it.
    use_cheap_skip = hasattr(simulator, '_advance_rng_for_batch')

    for _split, run_count in _iter_assignment_runs(skipped_assignments):
        remaining = run_count
        while remaining > 0:
            consume = min(batch_size, remaining)
            if use_cheap_skip:
                simulator._advance_rng_for_batch(consume)
            else:
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
    data_parallel: bool | None = None,
    store_covariance: bool = False,
) -> GeneratedSimulationDataset:
    """Generate and persist a simulation corpus in one call.

    All simulation data is written to a single ``.h5`` file via
    :class:`HDF5SimulationDatasetStore`, keeping the file count at 1 regardless
    of the number of simulations.

    Parameters
    ----------
    data_parallel:
        When ``True`` (or ``None`` with multiple JAX devices), the vmapped RT
        kernel is distributed across devices using ``jax.pmap``.  The
        effective per-iteration batch size is automatically scaled by the
        device count so each device processes the configured
        ``simulation_config.batch_size`` samples.
    store_covariance:
        When ``True`` the full covariance matrix for each simulated spectrum is
        written to disk.  When ``False`` (default) only the covariance diagonal
        is stored under the ``covariance`` field to reduce storage pressure.
    """
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
    dataset_store: SimulationDatasetStore = HDF5SimulationDatasetStore(
        chunk_size=chunk_size
    )
    requested_manifest = SimulationDatasetManifest.from_task(
        task,
        storage_uri=storage_uri,
        metadata={
            "dataset_version": dataset_version,
            "split_policy": resolved_split_policy.value,
            "split_seed": split_seed,
            "store_covariance": bool(store_covariance),
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
        existing_store_covariance = bool(
            existing_manifest.metadata.get("store_covariance", True)
        )
        if existing_store_covariance != bool(store_covariance):
            raise ValueError(
                "Cannot resume dataset generation with a different store_covariance setting."
            )
        existing_count = existing_manifest.n_simulations
        if existing_count > n_simulations:
            raise ValueError(
                "Existing dataset already has more samples than requested n_simulations."
            )
        writer = dataset_store.create_writer(requested_manifest, mode="a")
    else:
        writer = dataset_store.create_writer(requested_manifest, mode="w")

    n_new_simulations = n_simulations - existing_count

    simulation_backend = simulator
    if simulation_backend is None:
        from .simulator import RuntimeSimulator

        simulation_backend = RuntimeSimulator(task=task, data_parallel=data_parallel)

    # Resolve the effective per-iteration batch size.  When data_parallel is
    # active (multiple devices), scale the configured batch_size by the device
    # count so that each device processes its full share.
    import jax as _jax

    _n_devices = len(_jax.devices())
    _dp_active = (
        simulation_backend._data_parallel
        if hasattr(simulation_backend, '_data_parallel')
        else False
    )
    _effective_batch_size = (
        task.simulation_config.batch_size * _n_devices
        if _dp_active and _n_devices > 1
        else task.simulation_config.batch_size
    )

    # Only advance the simulator RNG when we will actually generate new samples.
    # Skipping this when the dataset is already complete avoids re-running the
    # full forward model n_existing times just to discard the output.
    if n_new_simulations > 0 and resume and existing_count > 0:
        _advance_simulator(
            simulator=simulation_backend,
            skipped_assignments=split_assignments[:existing_count],
            batch_size=_effective_batch_size,
        )

    for split, split_size in _iter_assignment_runs(split_assignments[existing_count:]):
        remaining = int(split_size)
        while remaining > 0:
            batch_size = min(_effective_batch_size, remaining)
            batch = simulation_backend.simulate_batch(batch_size)
            batch = _prepare_batch_for_storage(
                batch,
                store_covariance=store_covariance,
            )
            writer.append(batch, split=split)

            remaining -= batch_size

    finalized_manifest = writer.finalize()
    finalized_dataset = dataset_store.open(finalized_manifest)

    # Only validate the dataset when new simulations were actually written.
    # Revalidating a large existing dataset on every resume wastes hours of I/O.
    if n_new_simulations > 0:
        _validate_finite_split_observations(finalized_dataset)

    # Preprocessing metadata: use a sidecar JSON cache so that re-runs with
    # resume=True on a complete dataset avoid re-reading the full training split
    # and rebuilding ~100k Python ObservationBlock objects.
    preprocessing_sidecar = Path(storage_uri).with_suffix('.preprocessing.json')
    preprocessing_metadata = None
    if include_preprocessing_metadata:
        if n_new_simulations == 0 and preprocessing_sidecar.exists():
            cached_preprocessing = TaskPreprocessingMetadata.from_payload(
                json.loads(preprocessing_sidecar.read_text())
            )
            if cached_preprocessing.version == task.preprocessing_version:
                preprocessing_metadata = cached_preprocessing

        if preprocessing_metadata is None:
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
            preprocessing_sidecar.write_text(
                json.dumps(preprocessing_metadata.to_payload(), sort_keys=True, indent=2)
            )

    artifact_metadata = ArtifactMetadata.from_task(
        task=task,
        role=ArtifactRole.SIMULATION_DATASET,
        version=dataset_version,
        metadata={
            "storage_uri": storage_uri,
            "n_simulations": finalized_manifest.n_simulations,
            "splits": dict(finalized_manifest.splits),
            "store_covariance": bool(store_covariance),
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
    "HDF5SimulationDatasetStore",
    "HDF5SimulationDatasetWriter",
    "HDF5StoredSimulationDataset",
    "NormalizedObservationDatasetReader",
    "SimulationDatasetManifest",
    "SimulationDatasetStore",
    "SimulationDatasetWriter",
    "SplitSamplingPolicy",
    "generate_simulation_dataset",
]
