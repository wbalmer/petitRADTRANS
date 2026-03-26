"""Posterior estimator interfaces and concrete flow backends.

This module re-exports all public symbols from the split posterior
sub-modules for backward compatibility.  New code should import from
:mod:`posterior_base`, :mod:`flow_posterior`, or
:mod:`flow_matching_posterior` directly.
"""

from .estimator_registry import register_posterior_estimator
from .flow_matching_posterior import FlowMatchingPosterior
from .flow_posterior import ConditionalFlowPosterior, ConditionalSplineFlowPosterior
from .posterior_base import (
    POSTERIOR_METADATA_SCHEMA_VERSION,
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorEstimator,
    PosteriorSamples,
    TrainingArtifacts,
)

__all__ = [
    'ConditionalFlowPosterior',
    'ConditionalSplineFlowPosterior',
    'FlowMatchingPosterior',
    'PersistentPosteriorEstimator',
    'POSTERIOR_METADATA_SCHEMA_VERSION',
    'PosteriorBatch',
    'PosteriorEstimator',
    'PosteriorSamples',
    'TrainingArtifacts',
]


register_posterior_estimator(ConditionalFlowPosterior.estimator_family, ConditionalFlowPosterior)
register_posterior_estimator(FlowMatchingPosterior.estimator_family, FlowMatchingPosterior)
