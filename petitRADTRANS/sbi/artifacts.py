"""Artifact metadata and registry interfaces for SBI assets."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from petitRADTRANS.sbi.task import SBITask


class ArtifactRole(str, Enum):
    """Artifact categories produced by the SBI workflow."""

    SIMULATION_DATASET = 'simulation_dataset'
    POSTERIOR_MODEL = 'posterior_model'
    BENCHMARK_REPORT = 'benchmark_report'
    CALIBRATION_REPORT = 'calibration_report'


@dataclass(frozen=True)
class ArtifactMetadata:
    """Traceability metadata shared by all persisted SBI artifacts."""

    role: ArtifactRole
    task_name: str
    version: str
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
            'role': self.role.value,
            'task_name': self.task_name,
            'version': self.version,
            'task_version': self.task_version,
            'task_fingerprint': self.task_fingerprint,
            'preprocessing_version': self.preprocessing_version,
            'forward_model_family': self.forward_model_family,
            'petitradtrans_version': self.petitradtrans_version,
            'retrieval_config_digest': self.retrieval_config_digest,
            'opacity_digest': self.opacity_digest,
            'observation_schema': dict(self.observation_schema),
            'preprocessing_metadata': dict(self.preprocessing_metadata),
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'ArtifactMetadata':
        payload_dict = dict(payload)
        payload_dict['role'] = ArtifactRole(str(payload_dict['role']))
        payload_dict['observation_schema'] = dict(payload_dict.get('observation_schema', {}))
        payload_dict['preprocessing_metadata'] = dict(payload_dict.get('preprocessing_metadata', {}))
        payload_dict['metadata'] = dict(payload_dict.get('metadata', {}))
        return cls(**payload_dict)

    @classmethod
    def from_task(
        cls,
        task: 'SBITask',
        role: ArtifactRole,
        version: str,
        metadata: Mapping[str, Any] | None = None,
        preprocessing_metadata: Mapping[str, Any] | None = None,
        retrieval_config_digest: str | None = None,
        opacity_digest: str | None = None,
    ) -> 'ArtifactMetadata':
        """Build artifact metadata from an SBI task family."""
        return cls(
            role=role,
            task_name=task.name,
            version=version,
            task_version=task.task_version,
            task_fingerprint=task.task_fingerprint.digest,
            preprocessing_version=task.preprocessing_version,
            forward_model_family=task.forward_model_family,
            petitradtrans_version=task.petitradtrans_version,
            retrieval_config_digest=retrieval_config_digest,
            opacity_digest=opacity_digest,
            observation_schema={
                'dataset_names': task.observation_schema.dataset_names,
                'modalities': dict(task.observation_schema.modalities),
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
    def resolve(self, role: ArtifactRole, task_name: str, version: str | None = None) -> str:
        """Resolve an artifact path or URI for the requested task and role."""


@dataclass(frozen=True)
class RegisteredArtifact:
    """Stored registry entry for one persisted artifact."""

    metadata: ArtifactMetadata
    location: str

    def to_payload(self) -> dict[str, Any]:
        return {
            'metadata': self.metadata.to_payload(),
            'location': self.location,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'RegisteredArtifact':
        payload_dict = dict(payload)
        return cls(
            metadata=ArtifactMetadata.from_payload(payload_dict['metadata']),
            location=str(payload_dict['location']),
        )


class FileArtifactRegistry(ArtifactRegistry):
    """Simple JSON-backed registry for local SBI artifacts."""

    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)
        self.root_directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_directory / 'registry.json'

    def _load_entries(self) -> list[RegisteredArtifact]:
        if not self.index_path.exists():
            return []
        payload = json.loads(self.index_path.read_text())
        return [RegisteredArtifact.from_payload(entry) for entry in payload.get('artifacts', [])]

    def _write_entries(self, entries: list[RegisteredArtifact]) -> None:
        self.index_path.write_text(
            json.dumps(
                {'artifacts': [entry.to_payload() for entry in entries]},
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

    def resolve(self, role: ArtifactRole, task_name: str, version: str | None = None) -> str:
        entries = self._load_entries()
        matching_entries = [
            entry
            for entry in entries
            if entry.metadata.role == role and entry.metadata.task_name == task_name
        ]
        if version is not None:
            matching_entries = [entry for entry in matching_entries if entry.metadata.version == version]

        if not matching_entries:
            raise KeyError(
                f"No artifact registered for role='{role.value}', task_name='{task_name}', version='{version}'."
            )

        return matching_entries[-1].location
