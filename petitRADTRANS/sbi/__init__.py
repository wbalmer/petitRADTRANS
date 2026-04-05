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
    PosteriorPredictiveReport,
    SimulationBasedCalibrationReport,
    generate_posterior_predictive_report,
    generate_sbc_report,
)

# Error types users may need to catch
from .compatibility import TaskCompatibilityError

# Dataset generation and access
from .dataset import (
    SimulationDatasetStore,
    StoredSimulationDataset,
    ZarrSimulationDatasetStore,
    generate_simulation_dataset,
)

# Observation encoders
from .encoders import (
    HierarchicalObservationEncoder,
    PhotometryPointEncoder,
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
    ConditionalFlowPosterior,
    ConditionalSplineFlowPosterior,
    FlowMatchingPosterior,
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorEstimator,
    PosteriorSamples,
)

# Plotting
from .plotting import (
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
    "OODDiagnostic",
    "PosteriorPredictiveReport",
    "SimulationBasedCalibrationReport",
    "generate_posterior_predictive_report",
    "generate_sbc_report",
    # Errors
    "TaskCompatibilityError",
    # Datasets
    "SimulationDatasetStore",
    "StoredSimulationDataset",
    "ZarrSimulationDatasetStore",
    "generate_simulation_dataset",
    # Encoders
    "HierarchicalObservationEncoder",
    "PhotometryPointEncoder",
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
    "ConditionalFlowPosterior",
    "ConditionalSplineFlowPosterior",
    "FlowMatchingPosterior",
    "PersistentPosteriorEstimator",
    "PosteriorBatch",
    "PosteriorEstimator",
    "PosteriorSamples",
    # Plotting
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
