"""Registry and generic loader utilities for persisted posterior estimators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .posterior import PosteriorEstimator


_ESTIMATOR_REGISTRY: dict[str, type['PosteriorEstimator']] = {}


def register_posterior_estimator(estimator_family: str, estimator_cls: type['PosteriorEstimator']) -> None:
    """Register one estimator family for generic load dispatch."""
    _ESTIMATOR_REGISTRY[str(estimator_family)] = estimator_cls


def get_posterior_estimator_class(estimator_family: str) -> type['PosteriorEstimator']:
    """Return the registered estimator class for one estimator family."""
    try:
        return _ESTIMATOR_REGISTRY[str(estimator_family)]
    except KeyError as exc:
        known_families = ', '.join(sorted(_ESTIMATOR_REGISTRY)) or '<none>'
        raise KeyError(
            f"No posterior estimator is registered for estimator_family={estimator_family!r}. "
            f"Known families: {known_families}."
        ) from exc


def read_posterior_metadata(input_directory: str) -> dict[str, Any]:
    """Read the persisted metadata envelope for one saved estimator."""
    input_path = Path(input_directory)
    return dict(json.loads((input_path / 'metadata.json').read_text()))


def resolve_estimator_family_from_metadata(metadata: dict[str, Any]) -> str:
    """Infer the estimator family from current or legacy saved metadata."""
    estimator_family = metadata.get('estimator_family')
    if estimator_family:
        return str(estimator_family)

    estimator_config = metadata.get('estimator_config')
    if isinstance(estimator_config, dict) and estimator_config.get('flow_family') is not None:
        return 'conditional_flow'

    if metadata.get('flow_family') is not None:
        return 'conditional_flow'

    raise KeyError(
        "Saved posterior metadata does not define 'estimator_family' and no legacy flow metadata was found."
    )


def load_posterior_estimator(input_directory: str) -> 'PosteriorEstimator':
    """Load a saved posterior estimator without naming its concrete class."""
    metadata = read_posterior_metadata(input_directory)
    estimator_family = resolve_estimator_family_from_metadata(metadata)
    estimator_cls = get_posterior_estimator_class(estimator_family)
    return estimator_cls.load(input_directory)


__all__ = [
    'get_posterior_estimator_class',
    'load_posterior_estimator',
    'read_posterior_metadata',
    'register_posterior_estimator',
    'resolve_estimator_family_from_metadata',
]
