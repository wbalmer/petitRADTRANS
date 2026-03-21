"""Compatibility metadata and schema checks for SBI task families."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _freeze_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_freeze_json_value(item) for item in value]
    if isinstance(value, list):
        return [_freeze_json_value(item) for item in value]
    if isinstance(value, set):
        return [_freeze_json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        module = getattr(value, '__module__', '__callable__')
        qualname = getattr(value, '__qualname__', repr(value))
        return f'{module}.{qualname}'
    return repr(value)


def make_digest(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 digest for a JSON-serializable payload."""
    normalized_payload = _freeze_json_value(payload)
    encoded_payload = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded_payload).hexdigest()


@dataclass(frozen=True)
class TaskFingerprint:
    """Deterministic fingerprint of an SBI task family."""

    digest: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'TaskFingerprint':
        frozen_payload = _freeze_json_value(payload)
        return cls(digest=make_digest(frozen_payload), payload=frozen_payload)


@dataclass(frozen=True)
class CompatibilityReport:
    """Describe whether two task-family descriptors are compatible."""

    is_compatible: bool
    mismatches: tuple[str, ...] = ()
    expected: Mapping[str, Any] = field(default_factory=dict)
    observed: Mapping[str, Any] = field(default_factory=dict)

    def raise_for_incompatibility(self) -> None:
        """Raise a typed error when the report contains mismatches."""
        if self.is_compatible:
            return
        raise TaskCompatibilityError('; '.join(self.mismatches))


class TaskCompatibilityError(ValueError):
    """Raised when an observation or artifact is incompatible with a task."""


def compare_named_mappings(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    prefix: str,
) -> CompatibilityReport:
    """Compare two mapping payloads and report incompatible keys or values."""
    mismatches: list[str] = []
    expected_keys = tuple(sorted(expected.keys()))
    observed_keys = tuple(sorted(observed.keys()))

    if expected_keys != observed_keys:
        missing_keys = sorted(set(expected_keys) - set(observed_keys))
        extra_keys = sorted(set(observed_keys) - set(expected_keys))
        if missing_keys:
            mismatches.append(f'{prefix}: missing keys {missing_keys}')
        if extra_keys:
            mismatches.append(f'{prefix}: unexpected keys {extra_keys}')

    for key in sorted(set(expected.keys()).intersection(observed.keys())):
        if _freeze_json_value(expected[key]) != _freeze_json_value(observed[key]):
            mismatches.append(
                f'{prefix}: value mismatch for {key!r} '
                f'(expected {_freeze_json_value(expected[key])!r}, '
                f'observed {_freeze_json_value(observed[key])!r})'
            )

    return CompatibilityReport(
        is_compatible=not mismatches,
        mismatches=tuple(mismatches),
        expected=_freeze_json_value(expected),
        observed=_freeze_json_value(observed),
    )
