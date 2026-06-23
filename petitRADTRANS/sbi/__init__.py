"""Simulation-based inference interfaces for petitRADTRANS.

The modules in :mod:`petitRADTRANS.sbi` provide a production-oriented
architecture for amortized inference workflows built on top of the existing
retrieval runtime. The package is intentionally lightweight at this stage: it
defines the core task, simulation, dataset, and posterior interfaces without
committing to one training backend or storage engine.
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
    summarize_posterior_contraction,
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
    HierarchicalObservationEncoder,
    PhotometryPointEncoder,
    SpectralConv1DEncoder,
    SpectralPatchEncoder,
)

# Posterior loading
from .estimator_registry import load_posterior_estimator

# Grouped configuration dataclasses + flow-posterior factory
from .config import (
    EncoderConfig,
    FlowConfig,
    RefinementConfig,
    SBIConfig,
    StabilityConfig,
    TrainingConfig,
    build_conditional_flow_posterior,
)

# End-to-end validation workflow
from .workflow import run_sbi

# High-level inference entry point
from .inference import AmortizedRetrieval, AmortizedRetrievalResult, OODDiagnostic

# Observation types and utilities
from .observation import (
    ObservationBlock,
    ObservationEncoder,
    ObservationModality,
    build_observation_block,
    build_observation_block_batch,
    build_observation_blocks_from_sample,
    build_raw_observation_blocks,
)

# Posterior estimators
from .posterior import (
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

# Training. NOTE: the trainer's internal ``TrainingConfig`` is intentionally not
# re-exported here -- the public ``TrainingConfig`` is the grouped config
# dataclass from ``.config``. The trainer one remains at
# ``petitRADTRANS.sbi.training.TrainingConfig``.
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
    "summarize_posterior_contraction",
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
    "HierarchicalObservationEncoder",
    "PhotometryPointEncoder",
    "SpectralConv1DEncoder",
    "SpectralPatchEncoder",
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
    "SBIConfig",
    "build_conditional_flow_posterior",
    "run_sbi",
    # Observations
    "ObservationBlock",
    "ObservationEncoder",
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
