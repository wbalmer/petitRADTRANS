"""Artifact metadata and registry interfaces for SBI assets.

This module defines the metadata and interfaces for saving and loading
the datasets, trained posterior models, and validation reports produced by the
SBI workflow. The metadata is used to ensure that the current SBI Task and the
generated dataset are compatible.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from petitRADTRANS.sbi.task import SBITask


class ArtifactRole(str, Enum):
    """Artifact categories produced by the SBI workflow."""

    SIMULATION_DATASET = "simulation_dataset"
    POSTERIOR_MODEL = "posterior_model"
    BENCHMARK_REPORT = "benchmark_report"
    CALIBRATION_REPORT = "calibration_report"


@dataclass(frozen=True)
class ArtifactMetadata:
    """Traceability metadata shared by all persisted SBI artifacts.

    This is used to ensure that the current SBI Task and the generated dataset
    are compatible.
    """

    role: ArtifactRole
    task_name: str
    version: str
    estimator_family: str | None = None
    task_version: str | None = None
    task_fingerprint: str | None = None
    preprocessing_version: str | None = None
    forward_model_family: str | None = None
    petitradtrans_version: str | None = None
    retrieval_config_digest: str | None = None
    opacity_digest: str | None = None
    observation_schema: Mapping[str, Any] = field(default_factory=dict)
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "task_name": self.task_name,
            "version": self.version,
            "estimator_family": self.estimator_family,
            "task_version": self.task_version,
            "task_fingerprint": self.task_fingerprint,
            "preprocessing_version": self.preprocessing_version,
            "forward_model_family": self.forward_model_family,
            "petitradtrans_version": self.petitradtrans_version,
            "retrieval_config_digest": self.retrieval_config_digest,
            "opacity_digest": self.opacity_digest,
            "observation_schema": dict(self.observation_schema),
            "preprocessing_metadata": dict(self.preprocessing_metadata),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ArtifactMetadata":
        payload_dict = dict(payload)
        payload_dict["role"] = ArtifactRole(str(payload_dict["role"]))
        payload_dict["observation_schema"] = dict(
            payload_dict.get("observation_schema", {})
        )
        payload_dict["preprocessing_metadata"] = dict(
            payload_dict.get("preprocessing_metadata", {})
        )
        payload_dict["metadata"] = dict(payload_dict.get("metadata", {}))
        return cls(**payload_dict)

    @classmethod
    def from_task(
        cls,
        task: "SBITask",
        role: ArtifactRole,
        version: str,
        estimator_family: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        preprocessing_metadata: Mapping[str, Any] | None = None,
        retrieval_config_digest: str | None = None,
        opacity_digest: str | None = None,
    ) -> "ArtifactMetadata":
        """Build artifact metadata from an SBI task family."""
        return cls(
            role=role,
            task_name=task.name,
            version=version,
            estimator_family=estimator_family,
            task_version=task.task_version,
            task_fingerprint=task.task_fingerprint.digest,
            preprocessing_version=task.preprocessing_version,
            forward_model_family=task.forward_model_family,
            petitradtrans_version=task.petitradtrans_version,
            retrieval_config_digest=retrieval_config_digest,
            opacity_digest=opacity_digest,
            observation_schema={
                "dataset_names": task.observation_schema.dataset_names,
                "modalities": dict(task.observation_schema.modalities),
            },
            preprocessing_metadata=dict(preprocessing_metadata or {}),
            metadata=dict(metadata or {}),
        )


class ArtifactRegistry(abc.ABC):
    """Resolve and persist versioned SBI artifacts."""

    @abc.abstractmethod
    def register(self, metadata: ArtifactMetadata, location: str) -> None:
        """Register an artifact location under the supplied metadata."""

    @abc.abstractmethod
    def resolve(
        self, role: ArtifactRole, task_name: str, version: str | None = None
    ) -> str:
        """Resolve an artifact path or URI for the requested task and role."""


@dataclass(frozen=True)
class RegisteredArtifact:
    """Stored registry entry for one persisted artifact."""

    metadata: ArtifactMetadata
    location: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_payload(),
            "location": self.location,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RegisteredArtifact":
        payload_dict = dict(payload)
        return cls(
            metadata=ArtifactMetadata.from_payload(payload_dict["metadata"]),
            location=str(payload_dict["location"]),
        )


class FileArtifactRegistry(ArtifactRegistry):
    """Simple JSON-backed registry for local SBI artifacts."""

    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)
        self.root_directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_directory / "registry.json"

    def _load_entries(self) -> list[RegisteredArtifact]:
        if not self.index_path.exists():
            return []
        payload = json.loads(self.index_path.read_text())
        return [
            RegisteredArtifact.from_payload(entry)
            for entry in payload.get("artifacts", [])
        ]

    def _write_entries(self, entries: list[RegisteredArtifact]) -> None:
        self.index_path.write_text(
            json.dumps(
                {"artifacts": [entry.to_payload() for entry in entries]},
                sort_keys=True,
                indent=2,
            )
        )

    def register(self, metadata: ArtifactMetadata, location: str) -> None:
        entries = [
            entry
            for entry in self._load_entries()
            if not (
                entry.metadata.role == metadata.role
                and entry.metadata.task_name == metadata.task_name
                and entry.metadata.version == metadata.version
            )
        ]
        entries.append(RegisteredArtifact(metadata=metadata, location=location))
        self._write_entries(entries)

    def resolve(
        self, role: ArtifactRole, task_name: str, version: str | None = None
    ) -> str:
        entries = self._load_entries()
        matching_entries = [
            entry
            for entry in entries
            if entry.metadata.role == role and entry.metadata.task_name == task_name
        ]
        if version is not None:
            matching_entries = [
                entry for entry in matching_entries if entry.metadata.version == version
            ]

        if not matching_entries:
            raise KeyError(
                f"No artifact registered for role='{role.value}', task_name='{task_name}', version='{version}'."
            )

        return matching_entries[-1].location


def save_predictive_spectrum(
    output_path: Path,
    retrieval_config: Any,
    predictive_report: Any,
) -> Path:
    """Write a posterior-predictive spectrum (wavelength, observed, uncertainty,
    predictive mean and interval) as a plain-text table.

    The dataset is taken from the predictive report itself, so the saver is
    generic across observations.
    """
    name = next(iter(predictive_report.observed_values))
    data = retrieval_config.data[name]
    np.savetxt(
        output_path,
        np.column_stack(
            [
                np.asarray(data.wavelengths, dtype=float),
                np.asarray(predictive_report.observed_values[name], dtype=float),
                np.asarray(data.uncertainties, dtype=float),
                np.asarray(predictive_report.predictive_mean[name], dtype=float),
                np.asarray(predictive_report.interval_lower[name], dtype=float),
                np.asarray(predictive_report.interval_upper[name], dtype=float),
            ]
        ),
        header='wavelength observed uncertainty predictive_mean interval_lower interval_upper',
    )
    return output_path


# --- Run-directory / manifest / JSON IO -------------------------------------
# Generic run-lifecycle helpers: deterministic run identity, JSON-safe writing,
# and resolving/persisting run manifests and the "latest run" pointer.


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def compute_run_config_hash(run_config: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(run_config), sort_keys=True).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:12]


def build_run_label(run_config: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{timestamp}-{compute_run_config_hash(run_config)}'


def latest_run_pointer_path(output_root: Path) -> Path:
    return output_root / 'latest_run.json'


def resolve_run_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / 'run_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Run manifest not found: {manifest_path}')
    return read_json(manifest_path)


def resolve_posterior_dir(run_root: Path, manifest: dict[str, Any]) -> Path:
    manifest_posterior_dir = manifest.get('posterior_dir')
    if manifest_posterior_dir is not None:
        posterior_dir = Path(str(manifest_posterior_dir)).resolve()
    else:
        posterior_dir = (run_root / 'posterior_model').resolve()

    if not (posterior_dir / 'metadata.json').exists():
        raise FileNotFoundError(
            f'Persisted posterior metadata was not found in {posterior_dir}'
        )

    return posterior_dir


def resolve_latest_posterior_run(output_root: Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = latest_run_pointer_path(output_root)
    if pointer_path.exists():
        pointer_payload = read_json(pointer_path)
        run_root = Path(str(pointer_payload['run_root'])).resolve()
        try:
            manifest = resolve_run_manifest(run_root)
            resolve_posterior_dir(run_root, manifest)
            return run_root, manifest
        except FileNotFoundError:
            pass

    runs_dir = output_root / 'runs'
    for manifest_path in sorted(runs_dir.glob('*/run_manifest.json'), reverse=True):
        manifest = read_json(manifest_path)
        run_root = manifest_path.parent.resolve()
        try:
            resolve_posterior_dir(run_root, manifest)
        except FileNotFoundError:
            continue
        return run_root, manifest

    raise FileNotFoundError(
        f'No persisted SBI posterior was found under {output_root}'
    )


def resolve_latest_completed_run(output_root: Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = latest_run_pointer_path(output_root)
    if pointer_path.exists():
        pointer_payload = read_json(pointer_path)
        run_root = Path(str(pointer_payload['run_root'])).resolve()
        manifest_path = run_root / 'run_manifest.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get('status') == 'completed':
                return run_root, manifest

    runs_dir = output_root / 'runs'
    for manifest_path in sorted(runs_dir.glob('*/run_manifest.json'), reverse=True):
        manifest = read_json(manifest_path)
        if manifest.get('status') == 'completed':
            return manifest_path.parent.resolve(), manifest

    raise FileNotFoundError(
        f'No completed SBI validation run was found under {output_root}'
    )


def count_predictive_case_checkpoints(checkpoint_directory: Path) -> int:
    if not checkpoint_directory.exists():
        return 0
    return sum(1 for _ in checkpoint_directory.glob('case_*.npz'))


def write_run_manifest(
    run_root: Path,
    run_label: str,
    run_config: dict[str, Any],
    task: Any,
    dataset_uri: str,
    posterior_dir: Path,
    summary_path: Path | None,
    status: str,
) -> None:
    manifest = {
        'run_label': run_label,
        'status': status,
        'config_hash': compute_run_config_hash(run_config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_name': task.name,
        'task_fingerprint': task.task_fingerprint.digest,
        'dataset_uri': dataset_uri,
        'posterior_dir': posterior_dir.as_posix(),
        'summary_path': None if summary_path is None else summary_path.as_posix(),
        'run_config': run_config,
    }

    metadata_path = posterior_dir / 'metadata.json'
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text())
        manifest['posterior_metadata_excerpt'] = {
            'parameter_space': metadata_payload.get('parameter_space'),
            'effective_flow_family': metadata_payload.get('effective_flow_family'),
            'training_metadata': metadata_payload.get('training_metadata'),
        }

    write_json(run_root / 'run_manifest.json', manifest)


def update_latest_run_pointer(
    output_root: Path, run_root: Path, run_label: str
) -> None:
    write_json(
        latest_run_pointer_path(output_root),
        {
            'run_label': run_label,
            'run_root': run_root.as_posix(),
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        },
    )
