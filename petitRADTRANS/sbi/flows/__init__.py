"""Normalizing-flow transforms and flow-based posterior estimators.

This subpackage groups everything related to the conditional normalizing-flow
machinery and the posterior estimators built on top of it:

* :mod:`~petitRADTRANS.sbi.flows.transforms` -- low-level conditional flow
  transforms and flow stacks (affine / spline / autoregressive / NAF).
* :mod:`~petitRADTRANS.sbi.flows.base` -- backend-agnostic posterior interfaces,
  shared dataclasses, and the persistence base class.
* :mod:`~petitRADTRANS.sbi.flows.cube` -- parameter-space / unit-cube transforms.
* :mod:`~petitRADTRANS.sbi.flows.regularization` -- spline regularization and
  checkpoint-selection stability diagnostics.
* :mod:`~petitRADTRANS.sbi.flows.objectives` -- the training objective and the
  conditional encoder+flow model container.
* :mod:`~petitRADTRANS.sbi.flows.refinement` -- test-time per-observation
  posterior refinement (MAP+Laplace / flow-ELBO).
* :mod:`~petitRADTRANS.sbi.flows.posterior` -- :class:`ConditionalFlowPosterior`.
* :mod:`~petitRADTRANS.sbi.flows.matching` -- :class:`FlowMatchingPosterior`.

The estimator-family registry and the generic :func:`load_posterior_estimator`
loader live in :mod:`.base`. Importing this package eagerly imports
:mod:`.posterior` and :mod:`.matching`, which register their families, so a bare
``import petitRADTRANS.sbi`` is enough for :func:`load_posterior_estimator` to
resolve either family.
"""

from .transforms import *  # noqa: F401,F403  (re-export the flow transforms)
from .transforms import __all__ as _transforms_all
from .base import (
    POSTERIOR_METADATA_SCHEMA_VERSION,
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorEstimator,
    PosteriorSamples,
    TrainingArtifacts,
    get_posterior_estimator_class,
    load_posterior_estimator,
    read_posterior_metadata,
    register_posterior_estimator,
    resolve_estimator_family_from_metadata,
)
from .posterior import (
    ConditionalAutoregressiveFlowPosterior,
    ConditionalFlowPosterior,
    ConditionalNeuralAutoregressiveFlowPosterior,
    ConditionalSplineFlowPosterior,
)
from .matching import FlowMatchingPosterior

__all__ = [
    *_transforms_all,
    # base interfaces + dataclasses
    "POSTERIOR_METADATA_SCHEMA_VERSION",
    "PersistentPosteriorEstimator",
    "PosteriorBatch",
    "PosteriorEstimator",
    "PosteriorSamples",
    "TrainingArtifacts",
    # estimator registry + generic loader
    "get_posterior_estimator_class",
    "load_posterior_estimator",
    "read_posterior_metadata",
    "register_posterior_estimator",
    "resolve_estimator_family_from_metadata",
    # flow-based estimators
    "ConditionalAutoregressiveFlowPosterior",
    "ConditionalFlowPosterior",
    "ConditionalNeuralAutoregressiveFlowPosterior",
    "ConditionalSplineFlowPosterior",
    "FlowMatchingPosterior",
]
