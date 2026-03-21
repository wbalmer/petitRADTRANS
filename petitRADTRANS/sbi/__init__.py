"""Simulation-based inference interfaces for petitRADTRANS.

The modules in :mod:`petitRADTRANS.sbi` provide a production-oriented
architecture for amortized inference workflows built on top of the existing
retrieval runtime. The package is intentionally lightweight at this stage: it
defines the core task, simulation, dataset, posterior, and benchmarking
interfaces without committing to one training backend or storage engine.
"""

from .artifacts import ArtifactMetadata, ArtifactRegistry, ArtifactRole, FileArtifactRegistry, RegisteredArtifact
from .benchmark import BenchmarkComparison, BenchmarkMetrics, RetrievalBenchmarkCase, RetrievalBenchmarkSuite
from .calibration import (
    CoverageLevelReport,
    PosteriorPredictiveReport,
    SimulationBasedCalibrationReport,
    central_credible_interval,
    empirical_interval_coverage,
    generate_posterior_predictive_report,
    generate_sbc_report,
    posterior_predictive_report_from_samples,
    simulation_based_calibration_report,
)
from .compatibility import CompatibilityReport, TaskCompatibilityError, TaskFingerprint
from .dataset import (
    DatasetSplit,
    GeneratedSimulationDataset,
    NormalizedObservationDatasetReader,
    SimulationDatasetManifest,
    SimulationDatasetStore,
    SimulationDatasetWriter,
    SplitSamplingPolicy,
    StoredSimulationDataset,
    ZarrSimulationDatasetStore,
    ZarrSimulationDatasetWriter,
    generate_simulation_dataset,
)
from .encoders import DatasetSetAggregator, HierarchicalObservationEncoder, PhotometryPointEncoder, SpectralPatchEncoder
from .flows import (
    ConditionalAffineCoupling,
    ConditionalAffineFlow,
    ConditionalRationalQuadraticSplineCoupling,
    ConditionalSplineFlow,
    CouplingConditioner,
)
from .inference import AmortizedRetrieval, AmortizedRetrievalResult, OODDiagnostic
from .observation import (
    EncodedObservation,
    ObservationBlock,
    ObservationEncoder,
    ObservationModality,
    build_observation_block,
    build_observation_block_batch,
    build_observation_blocks_from_sample,
)
from .posterior import (
    ConditionalFlowPosterior,
    ConditionalSplineFlowPosterior,
    PosteriorBatch,
    PosteriorEstimator,
    PosteriorSamples,
    TrainingArtifacts,
)
from .plotting import (
    plot_posterior_marginals,
    plot_posterior_predictive_report,
    plot_sbc_rank_histograms,
)
from .preprocessing import (
    BlockNormalizationStats,
    TaskPreprocessingMetadata,
    compute_block_normalization_stats,
    fit_task_preprocessing,
    normalize_observation_block,
    normalize_observation_blocks,
)
from .simulator import BatchedSimulator, ProposalSampler, RuntimeSimulator, SimulationBatch, Simulator
from .task import NoiseModelConfig, ObservationSchema, SBITask, SimulationConfig
from .training import (
    CheckpointBackend,
    EarlyStoppingConfig,
    EquinoxCheckpointBackend,
    OrbaxCheckpointBackend,
    SBITrainer,
    TrainingConfig,
    resolve_checkpoint_backend,
)

__all__ = [
    'AmortizedRetrieval',
    'AmortizedRetrievalResult',
    'ArtifactMetadata',
    'ArtifactRegistry',
    'ArtifactRole',
    'FileArtifactRegistry',
    'BatchedSimulator',
    'BenchmarkComparison',
    'BenchmarkMetrics',
    'central_credible_interval',
    'CoverageLevelReport',
    'CompatibilityReport',
    'ConditionalAffineCoupling',
    'ConditionalAffineFlow',
    'ConditionalRationalQuadraticSplineCoupling',
    'ConditionalFlowPosterior',
    'ConditionalSplineFlow',
    'ConditionalSplineFlowPosterior',
    'CouplingConditioner',
    'DatasetSplit',
    'EncodedObservation',
    'GeneratedSimulationDataset',
    'HierarchicalObservationEncoder',
    'NoiseModelConfig',
    'NormalizedObservationDatasetReader',
    'ObservationBlock',
    'ObservationEncoder',
    'ObservationModality',
    'ObservationSchema',
    'OODDiagnostic',
    'PosteriorBatch',
    'PosteriorEstimator',
    'PosteriorSamples',
    'plot_posterior_marginals',
    'plot_posterior_predictive_report',
    'plot_sbc_rank_histograms',
    'ProposalSampler',
    'RetrievalBenchmarkCase',
    'RetrievalBenchmarkSuite',
    'RuntimeSimulator',
    'SBITask',
    'SimulationBatch',
    'SimulationConfig',
    'SimulationDatasetManifest',
    'SimulationDatasetStore',
    'SimulationDatasetWriter',
    'Simulator',
    'SplitSamplingPolicy',
    'SpectralPatchEncoder',
    'StoredSimulationDataset',
    'TaskPreprocessingMetadata',
    'TaskCompatibilityError',
    'TaskFingerprint',
    'TrainingArtifacts',
    'ZarrSimulationDatasetStore',
    'ZarrSimulationDatasetWriter',
    'BlockNormalizationStats',
    'build_observation_block',
    'build_observation_block_batch',
    'build_observation_blocks_from_sample',
    'compute_block_normalization_stats',
    'DatasetSetAggregator',
    'empirical_interval_coverage',
    'CheckpointBackend',
    'EarlyStoppingConfig',
    'EquinoxCheckpointBackend',
    'fit_task_preprocessing',
    'generate_simulation_dataset',
    'normalize_observation_block',
    'normalize_observation_blocks',
    'OrbaxCheckpointBackend',
    'PhotometryPointEncoder',
    'PosteriorPredictiveReport',
    'RegisteredArtifact',
    'resolve_checkpoint_backend',
    'SBITrainer',
    'SimulationBasedCalibrationReport',
    'simulation_based_calibration_report',
    'TrainingConfig',
    'generate_posterior_predictive_report',
    'generate_sbc_report',
    'posterior_predictive_report_from_samples',
]
