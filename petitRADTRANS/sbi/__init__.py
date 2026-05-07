"""Simulation-based inference interfaces for petitRADTRANS.

The modules in :mod:`petitRADTRANS.sbi` provide a production-oriented
architecture for amortized inference workflows built on top of the existing
retrieval runtime. The package is intentionally lightweight at this stage: it
defines the core task, simulation, dataset, posterior, and benchmarking
interfaces without committing to one training backend or storage engine.
"""

# Benchmarking
from .benchmark import (
    BenchmarkComparison,
    BenchmarkMetrics,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkSuite,
)

# Calibration / diagnostics
from .calibration import (
    LocalSensitivityPointReport,
    LocalSensitivityReport,
    PosteriorPredictiveReport,
    SimulationBasedCalibrationReport,
    generate_local_sensitivity_report,
    generate_posterior_predictive_report,
    generate_sbc_report,
    local_sensitivity_report_to_payload,
)

# Error types users may need to catch
from .compatibility import TaskCompatibilityError

# Dataset generation and access
from .dataset import (
    HDF5SimulationDatasetStore,
    SimulationDatasetStore,
    StoredSimulationDataset,
    ZarrSimulationDatasetStore,
    generate_simulation_dataset,
    DatasetSplit,
    NormalizedObservationDatasetReader
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
from .task import NoiseModelConfig, ObservationSchema, SBITask, SimulationConfig

# Training
from .training import EarlyStoppingConfig, SBITrainer, TrainingConfig

__all__ = [
    # Benchmarking
    "BenchmarkComparison",
    "BenchmarkMetrics",
    "RetrievalBenchmarkCase",
    "RetrievalBenchmarkSuite",
    # Calibration / diagnostics
    "LocalSensitivityPointReport",
    "LocalSensitivityReport",
    "OODDiagnostic",
    "PosteriorPredictiveReport",
    "SimulationBasedCalibrationReport",
    "generate_local_sensitivity_report",
    "generate_posterior_predictive_report",
    "generate_sbc_report",
    "local_sensitivity_report_to_payload",
    # Errors
    "TaskCompatibilityError",
    # Datasets
    "HDF5SimulationDatasetStore",
    "SimulationDatasetStore",
    "StoredSimulationDataset",
    "ZarrSimulationDatasetStore",
    "generate_simulation_dataset",
    "DatasetSplit",
    "NormalizedObservationDatasetReader",
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
    # Observations
    "ObservationBlock",
    "ObservationEncoder",
    "ObservationModality",
    "build_observation_block",
    "build_observation_block_batch",
    "build_observation_blocks_from_sample",
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
    "SBITask",
    "SimulationConfig",
    # Training
    "EarlyStoppingConfig",
    "SBITrainer",
    "TrainingConfig",
]
