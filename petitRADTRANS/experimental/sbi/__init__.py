"""Simulation-based inference interfaces for petitRADTRANS (EXPERIMENTAL).

The modules in :mod:`petitRADTRANS.experimental.sbi` implement amortized
simulation-based inference (neural posterior estimation) on top of the existing
retrieval runtime: task definition, prior-predictive simulation, dataset
generation/streaming, preprocessing, observation encoders, conditional
normalizing flows, training, calibration diagnostics, and an end-to-end
:func:`run_sbi` workflow.

.. warning::

   This package is **under active development and subject to change**. It lives
   under :mod:`petitRADTRANS.experimental` precisely because it is not yet
   covered by the usual deprecation policy: public names, call signatures,
   default hyperparameters, on-disk artifacts (simulation datasets, saved
   posteriors, run manifests), and numerical results may change in any release.
   Cached datasets and trained posteriors are keyed by a task fingerprint and
   may need regenerating after an upgrade. Pin an exact petitRADTRANS version
   if you depend on this package.

See ``ARCHITECTURE.md`` in this directory for a module-by-module overview and a
separation of the scientific core from the supporting infrastructure.
"""

# Calibration / diagnostics
from .calibration import (
    LocalSensitivityPointReport,
    LocalSensitivityReport,
    PosteriorInformativenessReport,
    PosteriorPredictiveReport,
    SimulationBasedCalibrationReport,
    classify_posterior_informativeness,
    generate_local_sensitivity_report,
    generate_posterior_predictive_report,
    generate_sbc_report,
    local_sensitivity_report_to_payload,
    serialize_ood_diagnostic,
    summarize_parameter_units,
    summarize_posterior_width_ratios,
    summarize_prior_edge_occupancy,
    summarize_posterior_samples,
    summarize_predictive_report,
    summarize_training_diagnostics,
)

# Artifact saving
from .artifacts import save_predictive_spectrum

# Error types users may need to catch
from .compatibility import TaskCompatibilityError

# Dataset generation and access
from .dataset import (
    HDF5SimulationDatasetStore,
    SimulationDatasetStore,
    generate_simulation_dataset,
    DatasetSplit,
    NormalizedObservationDatasetReader,
    estimate_preloaded_split_memory_gib,
    infer_split_counts,
)

# Observation encoders
from .encoders import (
    ObservationEncoder,
    PhotometryPointEncoder,
)

# Posterior loading
from .flows import load_posterior_estimator

# Grouped configuration dataclasses + flow-posterior factory
from .config import (
    EncoderConfig,
    FlowConfig,
    RefinementConfig,
    SBIConfig,
    StabilityConfig,
    TrainingConfig,
    TruncationConfig,
    build_conditional_flow_posterior,
)
from .truncation import (
    TruncatedParameterLayout,
    TruncationBox,
    build_truncation_box,
    truncation_box_report,
)

# End-to-end validation workflow
from .workflow import run_sbi

# High-level inference entry point
from .inference import AmortizedRetrieval, AmortizedRetrievalResult, OODDiagnostic

# Observation types and utilities
from .observation import (
    ObservationBlock,
    AbstractObservationEncoder,
    ObservationModality,
    build_observation_block,
    build_observation_block_batch,
    build_observation_blocks_from_sample,
    build_raw_observation_blocks,
)

# Posterior estimators
from .flows import (
    ConditionalAutoregressiveFlowPosterior,
    ConditionalFlowPosterior,
    ConditionalNeuralAutoregressiveFlowPosterior,
    ConditionalSplineFlowPosterior,
    FlowMatchingPosterior,
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorEstimator,
    PosteriorSamples,
)

# Plotting
from .plotting import (
    plot_local_sensitivity_fisher_correlations,
    plot_local_sensitivity_jacobians,
    plot_local_sensitivity_singular_values,
    plot_posterior_corner,
    plot_posterior_marginals,
    plot_posterior_predictive_report,
    plot_sbc_rank_histograms,
    save_validation_plots,
)

# Preprocessing
from .preprocessing import (
    TaskPreprocessingMetadata,
    fit_task_preprocessing,
    normalize_observation_block,
    normalize_observation_blocks,
)

# Simulator
from .simulator import ProposalSampler, RuntimeSimulator, SimulationBatch, Simulator

# Task configuration
from .task import (
    NoiseModelConfig,
    ObservationSchema,
    ObservationValueConstraint,
    SBITask,
    SimulationConfig,
)

from .training import EarlyStoppingConfig, SBITrainer

# Pre-training data-sanity guards.
from .validation import ObservationScaleReport, check_observation_in_training_range

__all__ = [
    "ObservationScaleReport",
    "check_observation_in_training_range",
    # Calibration / diagnostics
    "LocalSensitivityPointReport",
    "LocalSensitivityReport",
    "OODDiagnostic",
    "PosteriorPredictiveReport",
    "PosteriorInformativenessReport",
    "SimulationBasedCalibrationReport",
    "classify_posterior_informativeness",
    "generate_local_sensitivity_report",
    "generate_posterior_predictive_report",
    "generate_sbc_report",
    "local_sensitivity_report_to_payload",
    "save_predictive_spectrum",
    "serialize_ood_diagnostic",
    "summarize_parameter_units",
    "summarize_posterior_width_ratios",
    "summarize_prior_edge_occupancy",
    "summarize_posterior_samples",
    "summarize_predictive_report",
    "summarize_training_diagnostics",
    # Errors
    "TaskCompatibilityError",
    # Datasets
    "HDF5SimulationDatasetStore",
    "SimulationDatasetStore",
    "generate_simulation_dataset",
    "DatasetSplit",
    "NormalizedObservationDatasetReader",
    "estimate_preloaded_split_memory_gib",
    "infer_split_counts",
    # Encoders
    "ObservationEncoder",
    "PhotometryPointEncoder",
    # Posterior loading
    "load_posterior_estimator",
    # Inference
    "AmortizedRetrieval",
    "AmortizedRetrievalResult",
    "EncoderConfig",
    "FlowConfig",
    "RefinementConfig",
    "StabilityConfig",
    "TrainingConfig",
    "TruncationConfig",
    "TruncatedParameterLayout",
    "TruncationBox",
    "build_truncation_box",
    "truncation_box_report",
    "SBIConfig",
    "build_conditional_flow_posterior",
    "run_sbi",
    # Observations
    "ObservationBlock",
    "AbstractObservationEncoder",
    "ObservationModality",
    "build_observation_block",
    "build_observation_block_batch",
    "build_observation_blocks_from_sample",
    "build_raw_observation_blocks",
    # Posteriors
    "ConditionalAutoregressiveFlowPosterior",
    "ConditionalFlowPosterior",
    "ConditionalNeuralAutoregressiveFlowPosterior",
    "ConditionalSplineFlowPosterior",
    "FlowMatchingPosterior",
    "PersistentPosteriorEstimator",
    "PosteriorBatch",
    "PosteriorEstimator",
    "PosteriorSamples",
    # Plotting
    "plot_local_sensitivity_fisher_correlations",
    "plot_local_sensitivity_jacobians",
    "plot_local_sensitivity_singular_values",
    "plot_posterior_corner",
    "plot_posterior_marginals",
    "plot_posterior_predictive_report",
    "plot_sbc_rank_histograms",
    "save_validation_plots",
    # Preprocessing
    "TaskPreprocessingMetadata",
    "fit_task_preprocessing",
    "normalize_observation_block",
    "normalize_observation_blocks",
    # Simulator
    "ProposalSampler",
    "RuntimeSimulator",
    "SimulationBatch",
    "Simulator",
    # Task configuration
    "NoiseModelConfig",
    "ObservationSchema",
    "ObservationValueConstraint",
    "SBITask",
    "SimulationConfig",
    # Training
    "EarlyStoppingConfig",
    "SBITrainer",
]
