import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    import pymultinest as _real_pymultinest  # noqa: F401 – ensure the real module wins if installed
except Exception:  # noqa: BLE001 - deliberately broad, see below
    # pymultinest initialises MPI at import, and outside a SLURM allocation
    # mpi4py raises RuntimeError ("cannot load MPI library") rather than
    # ImportError. Nothing in this suite uses pymultinest -- the import exists
    # only so the real module wins where it is usable -- so any failure must
    # fall through to the stub rather than make the whole SBI suite
    # uncollectable on a login node.
    pass
sys.modules.setdefault("pymultinest", SimpleNamespace())

from petitRADTRANS.retrieval.runtime import ParameterLayout  # noqa: E402
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig  # noqa: E402
from petitRADTRANS.experimental.sbi import (  # noqa: E402
    AmortizedRetrieval,
    build_observation_block_batch,
    ConditionalSplineFlowPosterior,
    ConditionalNeuralAutoregressiveFlowPosterior,
    DatasetSplit,
    fit_task_preprocessing,
    FlowMatchingPosterior,
    generate_local_sensitivity_report,
    generate_posterior_predictive_report,
    generate_sbc_report,
    generate_simulation_dataset,
    ObservationEncoder,
    load_posterior_estimator,
    NoiseModelConfig,
    NormalizedObservationDatasetReader,
    ObservationBlock,
    ObservationModality,
    ObservationSchema,
    PosteriorEstimator,
    PosteriorSamples,
    RuntimeSimulator,
    SBITask,
    SimulationBatch,
    SimulationConfig,
    TaskCompatibilityError,
    TaskPreprocessingMetadata,
    HDF5SimulationDatasetStore,
    local_sensitivity_report_to_payload,
    normalize_observation_blocks,
    plot_posterior_corner,
)
from petitRADTRANS.experimental.sbi.artifacts import (
    ArtifactMetadata,
    ArtifactRole,
    FileArtifactRegistry,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.calibration import (
    central_credible_interval,
    empirical_interval_coverage,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.compatibility import CompatibilityReport  # noqa: E402
from petitRADTRANS.experimental.sbi.dataset import (
    SimulationDatasetManifest,
    SplitSamplingPolicy,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.flows import (
    ConditionalNeuralAutoregressiveFlow,
    ConditionalSplineFlow,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.preprocessing import (
    EXACT_SUMMARY_DIM,
    EXACT_SUMMARY_FEATURE_NAMES,
    compute_block_normalization_stats,
    exact_summary_feature_matrix,
    standardize_exact_summary_features,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.config import (
    build_conditional_flow_posterior,
)  # noqa: E402
from petitRADTRANS.experimental.sbi.training import (  # noqa: E402
    SBITrainer,
    TrainingSchemeConfig,
    _prestack_observation_blocks,
)


def _dummy_model(*args, **kwargs):
    del args, kwargs
    return None


def _dummy_runtime_model(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True
):
    del prt_object, pt_plot_mode, adaptive_mesh_refinement
    amplitude = parameters["amplitude"].value
    return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])


def _nan_runtime_model(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True
):
    del prt_object, pt_plot_mode, adaptive_mesh_refinement
    amplitude = parameters["amplitude"].value
    return jnp.array([1.0, 2.0]), jnp.array([jnp.nan, amplitude + jnp.nan])


def _partial_nan_runtime_model(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True
):
    """Return a spectrum where only some values are NaN, passing the validity check."""
    del prt_object, pt_plot_mode, adaptive_mesh_refinement
    amplitude = parameters["amplitude"].value
    return jnp.array([1.0, 2.0]), jnp.array([jnp.nan, amplitude + 1.0])


def _dummy_photometric_transform(wavelengths, spectrum):
    del wavelengths
    return jnp.asarray([jnp.mean(jnp.asarray(spectrum))])


def _build_runtime_task(
    retrieval_name="simulator_demo",
    noise_mode="no_noise",
    seed=0,
    include_nuisance=False,
    store_log_likelihood=False,
    uncertainties=None,
    model_generating_function=_dummy_runtime_model,
):
    if uncertainties is None:
        uncertainties = jnp.array([0.01, 0.01])

    configuration = RetrievalConfig(retrieval_name=retrieval_name)
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "amplitude", True, transform_prior_cube_coordinate=lambda x: x
    )

    if include_nuisance:
        configuration.add_parameter(
            "synthetic_spectrum_scale_factor",
            True,
            transform_prior_cube_coordinate=lambda x: 0.5 + (2.5 * x),
        )
        configuration.add_parameter(
            "synthetic_spectrum_offset",
            True,
            transform_prior_cube_coordinate=lambda x: 2.0 * x,
        )
        configuration.add_parameter(
            "synthetic_spectrum_b",
            True,
            transform_prior_cube_coordinate=lambda x: -4.0 + (3.0 * x),
        )

    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=model_generating_function,
        radtrans_object=object(),
        scale_flux=include_nuisance,
        scale_uncertainties=include_nuisance,
        fit_flux_offset=include_nuisance,
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.0, 0.0]),
        uncertainties=uncertainties,
    )

    return SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=4,
            noise_model=NoiseModelConfig(mode=noise_mode, seed=seed),
            store_per_datapoint_log_likelihood=store_log_likelihood,
        ),
    )


def _build_mixed_modality_task(
    retrieval_name="mixed_modality_demo",
    noise_mode="no_noise",
    seed=0,
):
    configuration = RetrievalConfig(retrieval_name=retrieval_name)
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "amplitude", True, transform_prior_cube_coordinate=lambda x: x
    )
    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=_dummy_runtime_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.0, 0.0]),
        uncertainties=jnp.array([0.05, 0.05]),
    )
    configuration.add_data(
        name="synthetic_photometry",
        path_to_observations=None,
        model_generating_function=_dummy_runtime_model,
        radtrans_object=object(),
        photometric_transformation_function=_dummy_photometric_transform,
        photometric_bin_edges=(1.0, 2.0),
        wavelengths=jnp.array([1.5]),
        spectrum=jnp.array([0.0]),
        uncertainties=jnp.array([0.1]),
    )

    return SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=2,
            noise_model=NoiseModelConfig(mode=noise_mode, seed=seed),
            store_per_datapoint_log_likelihood=True,
        ),
    )


class _DummyPosteriorEstimator(PosteriorEstimator):
    def fit(self, dataset):
        del dataset
        raise NotImplementedError

    def encode_observation(self, blocks):
        return blocks

    def sample_posterior(self, observation, n_samples, seed=None):
        del observation, n_samples, seed
        return PosteriorSamples(samples=[[1.0]])

    def save(self, output_directory: str) -> None:
        del output_directory

    @classmethod
    def load(cls, input_directory: str) -> "_DummyPosteriorEstimator":
        del input_directory
        return cls()


def test_sbi_task_from_retrieval_config_materializes_runtime_metadata():
    configuration = RetrievalConfig(retrieval_name="demo_task")
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "temperature", True, transform_prior_cube_coordinate=lambda x: 500.0 + 500.0 * x
    )
    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=_dummy_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
    )

    task = SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=16, noise_model=NoiseModelConfig(mode="observational")
        ),
    )

    assert task.name == "demo_task"
    assert task.parameter_names == ("temperature",)
    assert task.observation_schema.dataset_names == ("synthetic_spectrum",)
    assert task.observation_schema.modalities["synthetic_spectrum"] == "spectrum"
    assert task.task_version == "0.1.0"
    assert task.preprocessing_version == "0.5.0"
    assert "_dummy_model" in task.forward_model_family
    assert task.task_fingerprint.digest


def test_amortized_retrieval_returns_posterior_samples():
    task = SBITask(
        name="demo",
        retrieval_config=RetrievalConfig(retrieval_name="demo"),
        parameter_layout=type("Layout", (), {"free_parameter_names": ()})(),
        observation_schema=ObservationSchema(dataset_names=(), modalities={}),
    )

    retrieval = AmortizedRetrieval(
        task=task, posterior_estimator=_DummyPosteriorEstimator()
    )
    result = retrieval.infer(
        [
            ObservationBlock(
                name="obs", modality=ObservationModality.SPECTRUM, values=[1.0, 2.0]
            )
        ]
    )

    assert result.posterior.samples == [[1.0]]


def test_amortized_retrieval_can_emit_posterior_predictive_report(tmp_path):
    task = _build_runtime_task(
        retrieval_name="amortized_predictive_demo",
        noise_mode="no_noise",
        seed=101,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "amortized_predictive_dataset").as_posix(),
        n_simulations=18,
        chunk_size=3,
        split_counts={
            DatasetSplit.TRAIN: 12,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.6",
    )
    store = HDF5SimulationDatasetStore(chunk_size=3)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=16,
        parameter_space="unconstrained",
        seed=37,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    raw_test_split = dataset.read_split(DatasetSplit.TEST)
    raw_blocks = build_observation_block_batch(
        raw_test_split["observations"], task.observation_schema.modalities
    )
    retrieval = AmortizedRetrieval(task=task, posterior_estimator=posterior)
    result = retrieval.infer(
        raw_blocks[0],
        n_posterior_samples=48,
        include_posterior_predictive=True,
        posterior_predictive_interval_level=0.9,
        seed=41,
    )

    assert np.asarray(result.posterior.samples).shape == (48, 1)
    assert result.posterior_predictive is not None
    assert result.posterior_predictive.metadata["n_cases"] == 1
    assert result.posterior_predictive.metadata["parameter_space"] == "unconstrained"
    assert result.posterior_predictive.observed_values["synthetic_spectrum"].shape == (
        2,
    )
    assert result.posterior_predictive.predictive_mean["synthetic_spectrum"].shape == (
        2,
    )
    assert result.posterior_predictive.predictive_std["synthetic_spectrum"].shape == (
        2,
    )
    assert (
        0.0
        <= result.posterior_predictive.interval_coverage["synthetic_spectrum"]
        <= 1.0
    )


def test_amortized_retrieval_normalizes_raw_observations_but_reports_raw_scale(
    tmp_path,
):
    task = _build_runtime_task(
        retrieval_name="amortized_preprocessing_demo",
        noise_mode="no_noise",
        seed=111,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "amortized_preprocessing_dataset").as_posix(),
        n_simulations=18,
        chunk_size=3,
        split_counts={
            DatasetSplit.TRAIN: 12,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.7",
    )
    store = HDF5SimulationDatasetStore(chunk_size=3)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=16,
        parameter_space="unconstrained",
        seed=43,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    raw_test_split = dataset.read_split(DatasetSplit.TEST)
    raw_blocks = build_observation_block_batch(
        raw_test_split["observations"], task.observation_schema.modalities
    )
    normalized_blocks = normalize_observation_blocks(
        raw_blocks[0], generated.preprocessing_metadata
    )
    retrieval = AmortizedRetrieval(
        task=task,
        posterior_estimator=posterior,
        preprocessing_metadata=generated.preprocessing_metadata,
    )

    result = retrieval.infer(
        raw_blocks[0],
        n_posterior_samples=48,
        include_posterior_predictive=True,
        posterior_predictive_interval_level=0.9,
        seed=47,
    )
    manual_encoded = posterior.encode_observation(normalized_blocks)
    manual_posterior = posterior.sample_posterior(manual_encoded, n_samples=48, seed=47)
    manual_physical_posterior = task.parameter_layout.transform_unconstrained_batch(
        manual_posterior.samples
    )

    np.testing.assert_allclose(
        np.asarray(result.posterior.samples),
        np.asarray(manual_physical_posterior),
    )
    np.testing.assert_allclose(
        result.posterior_predictive.observed_values["synthetic_spectrum"],
        np.asarray(raw_blocks[0][0].values),
    )
    assert not np.allclose(
        result.posterior_predictive.observed_values["synthetic_spectrum"],
        np.asarray(normalized_blocks[0].values),
    )
    assert result.metadata["preprocessing_applied"] is True
    assert (
        result.metadata["preprocessing_version"]
        == generated.preprocessing_metadata.version
    )
    assert result.metadata["parameter_space"] == "physical"


def test_amortized_retrieval_uses_posterior_preprocessing_metadata_by_default(tmp_path):
    task = _build_runtime_task(
        retrieval_name="amortized_preprocessing_payload_demo",
        noise_mode="no_noise",
        seed=121,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(
            tmp_path / "amortized_preprocessing_payload_dataset"
        ).as_posix(),
        n_simulations=18,
        chunk_size=3,
        split_counts={
            DatasetSplit.TRAIN: 12,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.8",
    )
    store = HDF5SimulationDatasetStore(chunk_size=3)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=16,
        parameter_space="unconstrained",
        seed=53,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    raw_test_split = dataset.read_split(DatasetSplit.TEST)
    raw_blocks = build_observation_block_batch(
        raw_test_split["observations"], task.observation_schema.modalities
    )
    normalized_blocks = normalize_observation_blocks(
        raw_blocks[0], generated.preprocessing_metadata
    )
    retrieval = AmortizedRetrieval(task=task, posterior_estimator=posterior)

    prepared_blocks = retrieval.prepare_observation_blocks(raw_blocks[0])
    direct_encoded = posterior.encode_observation(normalized_blocks)
    prepared_encoded = posterior.encode_observation(prepared_blocks)

    np.testing.assert_allclose(
        np.asarray(prepared_encoded.embedding), np.asarray(direct_encoded.embedding)
    )
    assert retrieval.preprocessing_metadata is not None
    assert (
        retrieval.preprocessing_metadata.version
        == generated.preprocessing_metadata.version
    )


def test_sbi_support_types_are_importable():
    batch = SimulationBatch(parameters=[[0.1]], observations={"obs": [1.0]})
    artifact = ArtifactMetadata(
        role=ArtifactRole.POSTERIOR_MODEL, task_name="demo", version="0.1.0"
    )

    assert batch.parameters == [[0.1]]
    assert artifact.role is ArtifactRole.POSTERIOR_MODEL
    assert DatasetSplit.TRAIN.value == "train"


def test_task_artifact_metadata_round_trip_and_schema_validation():
    configuration = RetrievalConfig(retrieval_name="artifact_demo")
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "temperature", True, transform_prior_cube_coordinate=lambda x: 500.0 + 100.0 * x
    )
    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=_dummy_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
    )

    task = SBITask.from_retrieval_config(configuration)
    artifact = ArtifactMetadata.from_task(
        task=task,
        role=ArtifactRole.POSTERIOR_MODEL,
        version="0.2.0",
    )

    report = task.validate_artifact_metadata(artifact)

    assert report.is_compatible is True
    assert artifact.task_fingerprint == task.task_fingerprint.digest
    assert artifact.observation_schema["modalities"]["synthetic_spectrum"] == "spectrum"


def test_task_validation_reports_schema_and_artifact_mismatches():
    configuration = RetrievalConfig(retrieval_name="mismatch_demo")
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "temperature", True, transform_prior_cube_coordinate=lambda x: 700.0 + 100.0 * x
    )
    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=_dummy_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
    )

    task = SBITask.from_retrieval_config(configuration)

    schema_report = task.validate_observation_schema(
        ObservationSchema(
            dataset_names=("synthetic_spectrum",),
            modalities={"synthetic_spectrum": "photometry"},
        )
    )
    assert isinstance(schema_report, CompatibilityReport)
    assert schema_report.is_compatible is False
    assert "value mismatch" in schema_report.mismatches[0]

    artifact = ArtifactMetadata.from_task(
        task=task,
        role=ArtifactRole.POSTERIOR_MODEL,
        version="0.2.0",
    )
    mismatched_artifact = replace(artifact, task_version="99.0.0")
    artifact_report = task.validate_artifact_metadata(mismatched_artifact)

    assert artifact_report.is_compatible is False
    assert "task_version" in artifact_report.mismatches[0]

    try:
        artifact_report.raise_for_incompatibility()
    except TaskCompatibilityError:
        pass
    else:
        raise AssertionError(
            "Expected TaskCompatibilityError for incompatible artifact metadata"
        )


def test_parameter_layout_batch_transforms_and_metadata_serialization():
    parameters = {"amplitude": type("ParameterProxy", (), {})()}
    del parameters

    from petitRADTRANS.retrieval.parameter import Parameter

    layout = ParameterLayout.from_parameters(
        {
            "amplitude": Parameter(
                name="amplitude",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 2.0 * x - 1.0,
            ),
            "offset": Parameter(
                name="offset",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 10.0 + x,
            ),
        }
    )

    cube_positions = jnp.array([[0.25, 0.5], [0.75, 0.1]], dtype=jnp.float64)
    unconstrained_positions = jnp.array(
        [[0.0, jnp.log(3.0)], [jnp.log(3.0), 0.0]], dtype=jnp.float64
    )

    transformed_cube = layout.transform_cube_to_physical_batch(cube_positions)
    transformed_unconstrained = layout.transform_unconstrained_batch(
        unconstrained_positions
    )
    physical_params_batch = layout.physical_params_from_matrix(
        jnp.array([[1.0, 2.0], [3.0, 4.0]])
    )
    metadata = layout.serialize_metadata()

    assert transformed_cube.shape == (2, 2)
    assert transformed_unconstrained.shape == (2, 2)
    assert len(physical_params_batch) == 2
    assert metadata["free_parameter_names"] == ["amplitude", "offset"]
    assert metadata["parameters"][0]["transform_prior_cube_coordinate"].endswith(
        "<lambda>"
    )


def test_runtime_simulator_generates_deterministic_observations_without_noise():
    configuration = RetrievalConfig(retrieval_name="simulator_demo")
    configuration.add_parameter("planet_radius", False, value=1.0)
    configuration.add_parameter(
        "amplitude", True, transform_prior_cube_coordinate=lambda x: x
    )
    configuration.add_data(
        name="synthetic_spectrum",
        path_to_observations=None,
        model_generating_function=_dummy_runtime_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.0, 0.0]),
        uncertainties=jnp.array([0.01, 0.01]),
    )

    task = SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=2,
            noise_model=NoiseModelConfig(mode="no_noise", seed=7),
        ),
    )
    simulator = RuntimeSimulator(task=task, seed=7)

    batch = simulator.simulate_from_parameters(
        jnp.array([[2.0], [4.0]], dtype=jnp.float64)
    )

    values = batch.observations["synthetic_spectrum"]["values"]
    deterministic_values = batch.observations["synthetic_spectrum"][
        "deterministic_values"
    ]

    assert batch.parameters.shape == (2, 1)
    assert values.shape == (2, 2)
    assert jnp.allclose(values, deterministic_values)
    assert jnp.allclose(values[0], jnp.array([2.0, 3.0]))
    assert jnp.allclose(values[1], jnp.array([4.0, 5.0]))


def test_runtime_simulator_prior_sampling_and_gaussian_noise_are_seeded():
    task = _build_runtime_task(
        retrieval_name="simulator_noise_demo",
        noise_mode="gaussian",
        seed=11,
        uncertainties=jnp.array([0.05, 0.05]),
    )

    simulator_a = RuntimeSimulator(task=task, seed=11)
    simulator_b = RuntimeSimulator(task=task, seed=11)

    sampled_a = simulator_a.sample_parameters(3)
    sampled_b = simulator_b.sample_parameters(3)

    assert jnp.allclose(sampled_a, sampled_b)

    noisy_batch_a = simulator_a.simulate_from_parameters(
        jnp.array([[1.5]], dtype=jnp.float64)
    )
    noisy_batch_b = simulator_b.simulate_from_parameters(
        jnp.array([[1.5]], dtype=jnp.float64)
    )

    assert jnp.allclose(
        noisy_batch_a.observations["synthetic_spectrum"]["values"],
        noisy_batch_b.observations["synthetic_spectrum"]["values"],
    )


def test_runtime_simulator_supports_nuisance_parameters():
    task = _build_runtime_task(
        retrieval_name="simulator_nuisance_demo",
        noise_mode="no_noise",
        seed=13,
        include_nuisance=True,
        uncertainties=jnp.array([0.5, 0.5]),
    )
    simulator = RuntimeSimulator(task=task, seed=13)

    batch = simulator.simulate_from_parameters(
        jnp.array([[2.0, 3.0, 1.0, -2.0]], dtype=jnp.float64)
    )
    observation = batch.observations["synthetic_spectrum"]
    expected_uncertainties = jnp.sqrt((jnp.array([0.5, 0.5]) * 3.0) ** 2 + (10.0**-2.0))

    assert jnp.allclose(observation["deterministic_values"][0], jnp.array([5.0, 8.0]))
    assert jnp.allclose(observation["values"], observation["deterministic_values"])
    assert jnp.allclose(observation["uncertainties"][0], expected_uncertainties)
    assert observation["metadata"][0]["scale_flux"] is True
    assert observation["metadata"][0]["fit_flux_offset"] is True


def test_row_invalid_value_mask_accepts_padded_per_sample_mask_vectors():
    values = np.ones((3, 2), dtype=float)
    mask = np.array([False, True, False, True], dtype=bool)

    row_invalid = RuntimeSimulator._row_invalid_value_mask(
        values,
        mask,
        constraint=None,
    )

    np.testing.assert_array_equal(row_invalid, np.array([False, False, False]))


def test_slice_batch_rows_preserves_static_masks():
    batch = SimulationBatch(
        parameters=jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float64),
        observations={
            "synthetic_spectrum": {
                "values": jnp.array(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64
                ),
                "deterministic_values": jnp.array(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64
                ),
                "uncertainties": jnp.array(
                    [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], dtype=jnp.float64
                ),
                "mask": np.array([False, True]),
                "coordinates": jnp.array(
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=jnp.float64
                ),
                "metadata": [
                    {"source_name": "synthetic_spectrum"},
                    {"source_name": "synthetic_spectrum"},
                    {"source_name": "synthetic_spectrum"},
                ],
            }
        },
    )

    sliced = RuntimeSimulator._slice_batch_rows(batch, np.array([0, 2]))

    np.testing.assert_array_equal(
        sliced.observations["synthetic_spectrum"]["mask"],
        np.array([False, True]),
    )


def test_concatenate_simulation_batches_preserves_static_masks():
    batch_a = SimulationBatch(
        parameters=jnp.array([[1.0]], dtype=jnp.float64),
        observations={
            "synthetic_spectrum": {
                "values": jnp.array([[1.0, 2.0]], dtype=jnp.float64),
                "deterministic_values": jnp.array([[1.0, 2.0]], dtype=jnp.float64),
                "uncertainties": jnp.array([[0.1, 0.1]], dtype=jnp.float64),
                "mask": np.array([False, True]),
                "coordinates": jnp.array([[1.0, 2.0]], dtype=jnp.float64),
                "metadata": [{"source_name": "synthetic_spectrum"}],
            }
        },
    )
    batch_b = SimulationBatch(
        parameters=jnp.array([[3.0]], dtype=jnp.float64),
        observations={
            "synthetic_spectrum": {
                "values": jnp.array([[3.0, 4.0]], dtype=jnp.float64),
                "deterministic_values": jnp.array([[3.0, 4.0]], dtype=jnp.float64),
                "uncertainties": jnp.array([[0.1, 0.1]], dtype=jnp.float64),
                "mask": np.array([False, True]),
                "coordinates": jnp.array([[1.0, 2.0]], dtype=jnp.float64),
                "metadata": [{"source_name": "synthetic_spectrum"}],
            }
        },
    )

    combined = RuntimeSimulator._concatenate_simulation_batches([batch_a, batch_b])

    np.testing.assert_array_equal(
        combined.observations["synthetic_spectrum"]["mask"],
        np.array([False, True]),
    )
    assert combined.parameters.shape == (2, 1)


def test_hdf5_simulation_dataset_store_round_trip(tmp_path):
    task = _build_runtime_task(
        retrieval_name="dataset_round_trip_demo", noise_mode="no_noise", seed=21
    )
    manifest = SimulationDatasetManifest.from_task(
        task,
        storage_uri=(tmp_path / "simulation_dataset").as_posix(),
        metadata={"test_case": "round_trip"},
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    writer = store.create_writer(manifest)
    observation_metadata = {
        "photometry": False,
        "source_name": "synthetic_spectrum",
        "line_opacity_mode": "c-k",
        "scale_flux": False,
        "scale_uncertainties": False,
        "fit_flux_offset": False,
    }

    writer.append(
        SimulationBatch(
            parameters=jnp.array([[1.0], [2.0]], dtype=jnp.float64),
            cube_parameters=jnp.array([[0.2], [0.4]], dtype=jnp.float64),
            unconstrained_parameters=jnp.array([[-1.0], [0.0]], dtype=jnp.float64),
            log_likelihood=jnp.array([-1.0, -2.0], dtype=jnp.float64),
            observations={
                "synthetic_spectrum": {
                    "values": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),
                    "deterministic_values": jnp.array(
                        [[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64
                    ),
                    "uncertainties": jnp.array(
                        [[0.1, 0.1], [0.1, 0.1]], dtype=jnp.float64
                    ),
                    "mask": jnp.array([[False, False], [False, False]]),
                    "coordinates": jnp.array(
                        [[1.0, 2.0], [1.0, 2.0]], dtype=jnp.float64
                    ),
                    "metadata": [observation_metadata, observation_metadata],
                }
            },
        ),
        split=DatasetSplit.TRAIN,
    )
    writer.append(
        SimulationBatch(
            parameters=jnp.array([[4.0]], dtype=jnp.float64),
            cube_parameters=jnp.array([[0.6]], dtype=jnp.float64),
            unconstrained_parameters=jnp.array([[1.0]], dtype=jnp.float64),
            log_likelihood=jnp.array([-3.0], dtype=jnp.float64),
            observations={
                "synthetic_spectrum": {
                    "values": jnp.array([[5.0, 6.0]], dtype=jnp.float64),
                    "deterministic_values": jnp.array([[5.0, 6.0]], dtype=jnp.float64),
                    "uncertainties": jnp.array([[0.2, 0.2]], dtype=jnp.float64),
                    "mask": jnp.array([[False, False]]),
                    "coordinates": jnp.array([[1.0, 2.0]], dtype=jnp.float64),
                    "metadata": [observation_metadata],
                }
            },
        ),
        split=DatasetSplit.VALIDATION,
    )

    finalized_manifest = writer.finalize()
    opened_dataset = store.open(finalized_manifest)
    train_split = opened_dataset.read_split(DatasetSplit.TRAIN)
    validation_split = opened_dataset.read_split(DatasetSplit.VALIDATION)

    assert finalized_manifest.n_simulations == 3
    assert finalized_manifest.splits == {"train": 2, "validation": 1}
    assert "cube_parameters" in finalized_manifest.available_fields
    assert "observations" in finalized_manifest.available_fields
    assert train_split["parameters"].shape == (2, 1)
    assert validation_split["parameters"].shape == (1, 1)
    np.testing.assert_allclose(train_split["cube_parameters"], np.array([[0.2], [0.4]]))
    np.testing.assert_allclose(validation_split["log_likelihood"], np.array([-3.0]))
    assert (
        train_split["observations"]["synthetic_spectrum"]["metadata"]["source_name"]
        == "synthetic_spectrum"
    )


def test_runtime_simulator_and_hdf5_store_integration(tmp_path):
    task = _build_runtime_task(
        retrieval_name="integration_dataset_demo",
        noise_mode="gaussian",
        seed=5,
        include_nuisance=True,
        store_log_likelihood=True,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    simulator = RuntimeSimulator(task=task, seed=5)
    batch = simulator.simulate_batch(3)

    manifest = SimulationDatasetManifest.from_task(
        task,
        storage_uri=(tmp_path / "integration_dataset").as_posix(),
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    writer = store.create_writer(manifest)
    writer.append(batch, split=DatasetSplit.TRAIN)
    finalized_manifest = writer.finalize()
    opened_dataset = store.open(finalized_manifest)
    train_split = opened_dataset.read_split(DatasetSplit.TRAIN)

    assert batch.cube_parameters is not None
    assert batch.unconstrained_parameters is not None
    assert batch.log_likelihood is not None
    assert finalized_manifest.n_simulations == 3
    assert finalized_manifest.task_fingerprint == task.task_fingerprint.digest
    assert train_split["parameters"].shape == (3, 4)
    assert train_split["cube_parameters"].shape == (3, 4)
    assert train_split["unconstrained_parameters"].shape == (3, 4)
    assert train_split["log_likelihood"].shape == (3,)
    assert train_split["observations"]["synthetic_spectrum"]["values"].shape == (3, 2)


def test_build_observation_blocks_and_fit_preprocessing():
    observations = {
        "synthetic_spectrum": {
            "values": jnp.array([[1.0, 3.0], [2.0, 4.0]], dtype=jnp.float64),
            "uncertainties": jnp.array([[0.1, 0.2], [0.2, 0.3]], dtype=jnp.float64),
            "coordinates": jnp.array([[1.0, 2.0], [1.0, 2.0]], dtype=jnp.float64),
            "mask": jnp.array([[False, False], [False, True]]),
            "metadata": [
                {"photometry": False, "source_name": "synthetic_spectrum"},
                {"photometry": False, "source_name": "synthetic_spectrum"},
            ],
        },
        "synthetic_photometry": {
            "values": jnp.array([[2.0], [4.0]], dtype=jnp.float64),
            "uncertainties": jnp.array([[0.5], [0.5]], dtype=jnp.float64),
            "coordinates": jnp.array([[1.5], [1.5]], dtype=jnp.float64),
            "mask": jnp.array([[False], [False]]),
            "metadata": [
                {"photometry": True, "source_name": "synthetic_photometry"},
                {"photometry": True, "source_name": "synthetic_photometry"},
            ],
        },
    }
    modalities = {
        "synthetic_spectrum": "spectrum",
        "synthetic_photometry": "photometry",
    }

    block_batch = build_observation_block_batch(observations, modalities)
    stats = compute_block_normalization_stats(block_batch[0] + block_batch[1])
    preprocessing = fit_task_preprocessing(
        block_batch, version="0.2.0", metadata={"source": "unit_test"}
    )

    assert len(block_batch) == 2
    assert block_batch[0][0].modality is ObservationModality.SPECTRUM
    assert block_batch[0][1].modality is ObservationModality.PHOTOMETRY
    assert stats["synthetic_spectrum"].n_samples == 2
    assert np.isclose(stats["synthetic_spectrum"].value_mean, 2.5)
    assert np.isclose(stats["synthetic_photometry"].value_mean, 3.0)
    assert isinstance(preprocessing, TaskPreprocessingMetadata)
    assert preprocessing.version == "0.2.0"
    assert preprocessing.metadata["source"] == "unit_test"
    assert preprocessing.blocks["synthetic_photometry"].modality == "photometry"


def test_generate_simulation_dataset_returns_preprocessing_artifact(tmp_path):
    task = _build_runtime_task(
        retrieval_name="corpus_generation_demo",
        noise_mode="gaussian",
        seed=17,
        include_nuisance=True,
        store_log_likelihood=True,
        uncertainties=jnp.array([0.05, 0.05]),
    )

    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "generated_dataset").as_posix(),
        n_simulations=5,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 3, DatasetSplit.VALIDATION: 2},
        dataset_version="0.3.0",
    )

    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    train_split = dataset.read_split(DatasetSplit.TRAIN)

    assert generated.manifest.n_simulations == 5
    assert generated.manifest.splits == {"train": 3, "validation": 2}
    assert generated.artifact_metadata.role is ArtifactRole.SIMULATION_DATASET
    assert generated.artifact_metadata.version == "0.3.0"
    assert generated.preprocessing_metadata is not None
    assert generated.preprocessing_metadata.blocks["synthetic_spectrum"].n_samples == 3
    assert (
        generated.artifact_metadata.preprocessing_metadata["version"]
        == task.preprocessing_version
    )
    assert generated.artifact_metadata.metadata["n_simulations"] == 5
    assert train_split["parameters"].shape == (3, 4)


def test_generate_simulation_dataset_aborts_on_non_finite_observations(tmp_path):
    task = _build_runtime_task(
        retrieval_name="non_finite_dataset_demo",
        noise_mode="no_noise",
        seed=41,
        uncertainties=jnp.array([0.05, 0.05]),
        model_generating_function=_nan_runtime_model,
    )

    with pytest.raises(
        ValueError,
        match="Model evaluation for source 'synthetic_spectrum' returned an invalid result",
    ):
        generate_simulation_dataset(
            task=task,
            storage_uri=(tmp_path / "invalid_dataset").as_posix(),
            n_simulations=2,
            chunk_size=2,
            split_counts={DatasetSplit.TRAIN: 2},
            dataset_version="0.3.1",
        )


def test_runtime_simulator_logs_non_finite_projection(caplog):
    task = _build_runtime_task(
        retrieval_name="non_finite_projection_demo",
        noise_mode="no_noise",
        seed=43,
        uncertainties=jnp.array([0.05, 0.05]),
        model_generating_function=_partial_nan_runtime_model,
    )
    simulator = RuntimeSimulator(task=task, seed=43)

    with caplog.at_level(
        "ERROR", logger="petitRADTRANS.retrieval.runtime"
    ), pytest.raises(
        ValueError, match="Projected observation values contain non-finite entries"
    ):
        simulator.simulate_from_parameters(jnp.array([[0.5]], dtype=jnp.float64))

    assert "source 'synthetic_spectrum'" in caplog.text
    assert "dataset 'synthetic_spectrum'" in caplog.text
    assert "parameters={'amplitude': 0.5}" in caplog.text


def test_generate_simulation_dataset_supports_mixed_spectrum_and_photometry(tmp_path):
    task = _build_mixed_modality_task(seed=9, noise_mode="gaussian")

    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "mixed_dataset").as_posix(),
        n_simulations=4,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 2, DatasetSplit.TEST: 2},
        dataset_version="0.4.0",
    )

    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    train_split = dataset.read_split(DatasetSplit.TRAIN)
    test_split = dataset.read_split(DatasetSplit.TEST)

    assert generated.manifest.splits == {"train": 2, "test": 2}
    assert generated.preprocessing_metadata is not None
    assert "synthetic_spectrum" in generated.preprocessing_metadata.blocks
    assert "synthetic_photometry" in generated.preprocessing_metadata.blocks
    assert (
        generated.preprocessing_metadata.blocks["synthetic_spectrum"].modality
        == "spectrum"
    )
    assert (
        generated.preprocessing_metadata.blocks["synthetic_photometry"].modality
        == "photometry"
    )
    assert train_split["observations"]["synthetic_spectrum"]["values"].shape == (2, 2)
    assert train_split["observations"]["synthetic_photometry"]["values"].shape == (2, 1)
    assert test_split["observations"]["synthetic_photometry"]["values"].shape == (2, 1)
    assert (
        generated.artifact_metadata.observation_schema["modalities"][
            "synthetic_photometry"
        ]
        == "photometry"
    )


def test_hierarchical_observation_encoder_produces_fixed_width_embeddings():
    observations = {
        "synthetic_spectrum": {
            "values": jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float64),
            "uncertainties": jnp.array([[0.1, 0.2, 0.3]], dtype=jnp.float64),
            "coordinates": jnp.array([[1.0, 1.5, 2.0]], dtype=jnp.float64),
            "mask": jnp.array([[False, False, True]]),
            "metadata": [{"photometry": False}],
        },
        "synthetic_photometry": {
            "values": jnp.array([[4.0]], dtype=jnp.float64),
            "uncertainties": jnp.array([[0.5]], dtype=jnp.float64),
            "coordinates": jnp.array([[1.5]], dtype=jnp.float64),
            "mask": jnp.array([[False]]),
            "metadata": [{"photometry": True}],
        },
    }
    blocks = build_observation_block_batch(
        observations,
        {"synthetic_spectrum": "spectrum", "synthetic_photometry": "photometry"},
    )[0]
    encoder = ObservationEncoder(embedding_dim=96)

    encoded = encoder.encode(blocks)

    assert encoded.embedding.shape == (96,)
    assert encoded.metadata["n_blocks"] == 2
    assert encoded.metadata["modalities"] == ["spectrum", "photometry"]


def test_normalized_dataset_reader_yields_training_batches(tmp_path):
    task = _build_mixed_modality_task(seed=12, noise_mode="gaussian")
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "reader_dataset").as_posix(),
        n_simulations=4,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 4},
        dataset_version="0.5.0",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )

    batches = list(
        reader.iter_batches(
            split=DatasetSplit.TRAIN, batch_size=2, parameter_space="cube"
        )
    )

    assert len(batches) == 2
    assert batches[0].parameters.shape == (2, 1)
    assert len(batches[0].observations) == 2
    assert len(batches[0].observations[0]) == 2
    assert batches[0].observations[0][0].modality is ObservationModality.SPECTRUM
    assert batches[0].observations[0][1].modality is ObservationModality.PHOTOMETRY
    assert abs(float(np.mean(batches[0].observations[0][0].values))) < 5.0
    assert batches[0].metadata["parameter_space"] == "cube"


def test_normalized_dataset_reader_supports_encoded_mode(tmp_path):
    task = _build_mixed_modality_task(seed=14, noise_mode="gaussian")
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "encoded_reader_dataset").as_posix(),
        n_simulations=4,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 4},
        dataset_version="0.5.1",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    encoder = ObservationEncoder(embedding_dim=48)

    batches = list(
        reader.iter_batches(
            split=DatasetSplit.TRAIN,
            batch_size=2,
            parameter_space="unconstrained",
            encoder=encoder,
        )
    )

    assert len(batches) == 2
    assert batches[0].parameters.shape == (2, 1)
    assert batches[0].observations.shape == (2, 48)
    assert batches[0].metadata["encoded"] is True


def test_conditional_spline_flow_roundtrip():
    context = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float32)
    parameters = jnp.asarray([0.15, -0.35, 0.55], dtype=jnp.float32)
    flow = ConditionalSplineFlow(
        parameter_dim=3,
        context_dim=4,
        num_layers=4,
        hidden_dim=32,
        num_bins=8,
        bound=3.0,
        key=jax.random.PRNGKey(5),
    )

    latent, forward_log_det = flow.forward(parameters, context)
    recovered, inverse_log_det = flow.inverse(latent, context)
    samples = flow.sample(context=context, n_samples=6, key=jax.random.PRNGKey(6))

    assert recovered.shape == parameters.shape
    assert np.allclose(np.asarray(recovered), np.asarray(parameters), atol=1e-5)
    assert np.allclose(float(forward_log_det + inverse_log_det), 0.0, atol=1e-5)
    assert samples.shape == (6, 3)


def test_conditional_scalar_spline_flow_roundtrip():
    context = jnp.asarray([0.25, -0.15, 0.4], dtype=jnp.float32)
    parameter = jnp.asarray([0.35], dtype=jnp.float32)
    flow = ConditionalSplineFlow(
        parameter_dim=1,
        context_dim=3,
        num_layers=3,
        hidden_dim=32,
        num_bins=8,
        bound=3.0,
        key=jax.random.PRNGKey(15),
    )

    latent, forward_log_det = flow.forward(parameter, context)
    recovered, inverse_log_det = flow.inverse(latent, context)
    samples = flow.sample(context=context, n_samples=5, key=jax.random.PRNGKey(16))

    assert latent.shape == (1,)
    assert recovered.shape == (1,)
    assert np.allclose(np.asarray(recovered), np.asarray(parameter), atol=1e-5)
    assert np.allclose(float(forward_log_det + inverse_log_det), 0.0, atol=1e-5)
    assert samples.shape == (5, 1)


def test_conditional_neural_autoregressive_flow_roundtrip():
    context = jnp.asarray([0.2, -0.3, 0.4, 0.1], dtype=jnp.float32)
    parameters = jnp.asarray([0.15, -0.35, 0.55], dtype=jnp.float32)
    flow = ConditionalNeuralAutoregressiveFlow(
        parameter_dim=3,
        context_dim=4,
        num_layers=2,
        hidden_dim=32,
        conditioner_depth=3,
        transform_units=8,
        key=jax.random.PRNGKey(25),
    )

    latent, forward_log_det = flow.forward(parameters, context)
    recovered, inverse_log_det = flow.inverse(latent, context)
    samples = flow.sample(context=context, n_samples=5, key=jax.random.PRNGKey(26))

    assert latent.shape == parameters.shape
    assert recovered.shape == parameters.shape
    assert np.allclose(np.asarray(recovered), np.asarray(parameters), atol=2e-4)
    assert np.allclose(float(forward_log_det + inverse_log_det), 0.0, atol=2e-4)
    assert samples.shape == (5, 3)


def test_conditional_neural_autoregressive_flow_inverse_diagnostics():
    context = jnp.asarray([0.2, -0.3, 0.4, 0.1], dtype=jnp.float32)
    parameters = jnp.asarray([0.15, -0.35, 0.55], dtype=jnp.float32)
    flow = ConditionalNeuralAutoregressiveFlow(
        parameter_dim=3,
        context_dim=4,
        num_layers=2,
        hidden_dim=32,
        conditioner_depth=3,
        transform_units=8,
        key=jax.random.PRNGKey(125),
    )

    latent, forward_log_det = flow.forward(parameters, context)
    recovered, inverse_log_det, diagnostics = flow.inverse_with_diagnostics(
        latent, context
    )

    assert np.allclose(np.asarray(recovered), np.asarray(parameters), atol=2e-4)
    assert np.allclose(float(forward_log_det + inverse_log_det), 0.0, atol=2e-4)
    assert float(diagnostics.max_abs_residual) < 1e-4
    assert float(diagnostics.mean_abs_residual) < 1e-4
    assert float(diagnostics.max_bracket_width) >= 0.0
    assert float(diagnostics.mean_bracket_width) >= 0.0
    assert float(diagnostics.max_expansion_steps) >= 0.0
    assert float(diagnostics.mean_expansion_steps) >= 0.0
    assert float(diagnostics.converged_fraction) >= 0.99


def test_conditional_flow_posterior_fit_sample_and_save_load(tmp_path):
    task = _build_runtime_task(
        retrieval_name="posterior_demo",
        noise_mode="gaussian",
        seed=41,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "posterior_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.0",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    checkpoint_directory = tmp_path / "posterior_checkpoints"
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=2,
        parameter_space="unconstrained",
        checkpoint_directory=checkpoint_directory.as_posix(),
        seed=7,
        task_metadata={"task_name": task.name},
    )

    artifacts = posterior.fit(reader)
    train_batch = next(
        reader.iter_batches(
            split=DatasetSplit.TRAIN, batch_size=1, parameter_space="unconstrained"
        )
    )
    encoded_observation = posterior.encode_observation(train_batch.observations[0])
    samples = posterior.sample_posterior(encoded_observation, n_samples=5, seed=13)
    log_probabilities = posterior.log_prob(
        encoded_observation, jnp.asarray(samples.samples, dtype=jnp.float32)
    )

    save_directory = tmp_path / "saved_posterior"
    posterior.save(save_directory.as_posix())
    loaded = ConditionalSplineFlowPosterior.load(save_directory.as_posix())
    loaded_generic = load_posterior_estimator(save_directory.as_posix())
    loaded_samples = loaded.sample_posterior(encoded_observation, n_samples=5, seed=13)
    loaded_generic_samples = loaded_generic.sample_posterior(
        encoded_observation, n_samples=5, seed=13
    )

    saved_metadata = json.loads((save_directory / "metadata.json").read_text())

    assert len(artifacts.history["train_loss"]) == 2
    assert len(artifacts.validation_metrics["validation_loss"]) == 2
    assert artifacts.metadata["flow_family"] == "spline"
    assert artifacts.metadata["effective_flow_family"] == "spline"
    assert np.asarray(samples.samples).shape == (5, 1)
    assert np.asarray(log_probabilities).shape == (5,)
    assert np.asarray(loaded_samples.samples).shape == (5, 1)
    assert np.asarray(loaded_generic_samples.samples).shape == (5, 1)
    assert saved_metadata["estimator_family"] == "conditional_flow"
    assert saved_metadata["metadata_schema_version"] == "0.2.0"
    assert saved_metadata["estimator_config"]["flow_family"] == "spline"
    assert (checkpoint_directory / "latest" / "metadata.json").exists()


def test_neural_autoregressive_posterior_fit_sample_and_generic_load(tmp_path):
    task = _build_runtime_task(
        retrieval_name="neural_autoregressive_posterior_demo",
        noise_mode="gaussian",
        seed=43,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "neural_autoregressive_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.0",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    checkpoint_directory = tmp_path / "neural_autoregressive_checkpoints"
    posterior = ConditionalNeuralAutoregressiveFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=2,
        hidden_dim=32,
        conditioner_depth=3,
        autoregressive_transform_units=8,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=1,
        parameter_space="unconstrained",
        checkpoint_directory=checkpoint_directory.as_posix(),
        stable_inverse_forward_max_abs_error_threshold=1e-2,
        stable_inverse_forward_logdet_closure_max_abs_error_threshold=1e-2,
        seed=27,
        task_metadata={"task_name": task.name},
    )

    artifacts = posterior.fit(reader)
    train_batch = next(
        reader.iter_batches(
            split=DatasetSplit.TRAIN, batch_size=1, parameter_space="unconstrained"
        )
    )
    encoded_observation = posterior.encode_observation(train_batch.observations[0])
    samples = posterior.sample_posterior(encoded_observation, n_samples=5, seed=29)

    save_directory = tmp_path / "saved_neural_autoregressive_posterior"
    posterior.save(save_directory.as_posix())
    loaded = load_posterior_estimator(save_directory.as_posix())
    loaded_samples = loaded.sample_posterior(encoded_observation, n_samples=5, seed=29)
    saved_metadata = json.loads((save_directory / "metadata.json").read_text())

    assert len(artifacts.history["train_loss"]) == 1
    assert artifacts.metadata["flow_family"] == "neural_autoregressive"
    assert artifacts.metadata["effective_flow_family"] == "neural_autoregressive"
    assert artifacts.metadata["best_checkpoint_kind"] == "best_selection"
    assert artifacts.metadata["best_loss_epoch"] == 1
    assert artifacts.metadata["best_stable_epoch"] == 1
    assert np.asarray(samples.samples).shape == (5, 1)
    assert np.asarray(loaded_samples.samples).shape == (5, 1)
    assert "inverse_solver_max_abs_residual" in artifacts.validation_metrics
    assert len(artifacts.validation_metrics["inverse_solver_max_abs_residual"]) == 1
    assert saved_metadata["estimator_family"] == "conditional_flow"
    assert saved_metadata["estimator_config"]["flow_family"] == "neural_autoregressive"
    assert saved_metadata["estimator_config"]["autoregressive_transform_units"] == 8
    assert (
        saved_metadata["estimator_config"][
            "stable_inverse_forward_max_abs_error_threshold"
        ]
        == 1e-2
    )
    assert (checkpoint_directory / "best" / "metadata.json").exists()
    assert (checkpoint_directory / "best_loss" / "metadata.json").exists()
    assert (checkpoint_directory / "best_selection" / "metadata.json").exists()
    assert (checkpoint_directory / "best_stable" / "metadata.json").exists()
    assert loaded.flow_family == "neural_autoregressive"


def test_trainer_tracks_best_loss_and_best_stable_checkpoints(tmp_path):
    class ScalarModel(eqx.Module):
        weight: jnp.ndarray

    class ToyDataset:
        def __init__(self):
            self.dataset = SimpleNamespace(
                manifest=SimpleNamespace(
                    splits={"train": 1, "validation": 1},
                )
            )

        def iter_batches(
            self,
            split,
            batch_size,
            shuffle=False,
            seed=0,
            parameter_space="unconstrained",
        ):
            del split, batch_size, shuffle, seed, parameter_space
            yield SimpleNamespace(
                parameters=np.zeros((1, 1), dtype=np.float32),
                observations=np.zeros((1, 1), dtype=np.float32),
            )

    trainer = SBITrainer(
        TrainingSchemeConfig(
            learning_rate=0.5,
            batch_size=1,
            num_epochs=3,
            checkpoint_directory=(tmp_path / "toy_trainer_checkpoints").as_posix(),
            checkpoint_backend="equinox",
        )
    )
    model = ScalarModel(weight=jnp.asarray(0.0, dtype=jnp.float32))
    dataset = ToyDataset()

    def loss_fn(model, batch):
        del batch
        return -model.weight

    def eval_diagnostic_fn(model, batch):
        del batch
        unstable = float(model.weight >= 0.75)
        diagnostic_value = 1e-2 if unstable else 1e-6
        return {
            "inverse_forward_max_abs_error": diagnostic_value,
            "inverse_forward_logdet_closure_max_abs_error": diagnostic_value,
            "cube_edge_hit_rate": 0.0,
        }

    def stability_metric_fn(validation_loss, diagnostics):
        del validation_loss
        stable = (
            diagnostics["inverse_forward_max_abs_error"] <= 1e-4
            and diagnostics["inverse_forward_logdet_closure_max_abs_error"] <= 1e-4
        )
        return {"checkpoint_is_stable": 1.0 if stable else 0.0}

    _, history, validation_metrics, metadata = trainer.fit(
        model,
        dataset,
        loss_fn,
        eval_loss_fn=loss_fn,
        eval_diagnostic_fn=eval_diagnostic_fn,
        selection_metric_fn=lambda validation_loss, diagnostics: validation_loss,
        selection_metric_name="validation_selection_metric",
        stability_metric_fn=stability_metric_fn,
        stability_flag_key="checkpoint_is_stable",
    )

    checkpoint_directory = tmp_path / "toy_trainer_checkpoints"
    best_metadata = json.loads(
        (checkpoint_directory / "best" / "metadata.json").read_text()
    )
    best_loss_metadata = json.loads(
        (checkpoint_directory / "best_loss" / "metadata.json").read_text()
    )
    best_stable_metadata = json.loads(
        (checkpoint_directory / "best_stable" / "metadata.json").read_text()
    )

    assert len(history["train_loss"]) == 3
    assert len(validation_metrics["validation_loss"]) == 3
    assert metadata["best_checkpoint_kind"] == "best_selection"
    assert metadata["best_stable_epoch"] == 1
    assert metadata["best_loss_epoch"] == 3
    assert metadata["best_selection_epoch"] == 3
    assert best_metadata["best_checkpoint_kind"] == "best_selection"
    assert best_loss_metadata["checkpoint_kind"] == "best_loss"
    assert best_stable_metadata["checkpoint_kind"] == "best_stable"


def test_spline_posterior_training_resumes_from_latest_checkpoint(tmp_path):
    task = _build_runtime_task(
        retrieval_name="resume_posterior_demo",
        noise_mode="gaussian",
        seed=51,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "resume_posterior_dataset").as_posix(),
        n_simulations=10,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 8, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.1",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    checkpoint_directory = tmp_path / "resume_checkpoints"

    initial = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=1,
        parameter_space="unconstrained",
        checkpoint_directory=checkpoint_directory.as_posix(),
        seed=9,
        task_metadata={"task_name": task.name},
    )
    initial_artifacts = initial.fit(reader)

    resumed = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=3,
        parameter_space="unconstrained",
        checkpoint_directory=checkpoint_directory.as_posix(),
        resume_from_checkpoint=True,
        seed=9,
        task_metadata={"task_name": task.name},
    )
    resumed_artifacts = resumed.fit(reader)

    latest_metadata = json.loads(
        (checkpoint_directory / "latest" / "metadata.json").read_text()
    )

    assert len(initial_artifacts.history["train_loss"]) == 1
    assert len(resumed_artifacts.history["train_loss"]) == 3
    assert np.allclose(
        resumed_artifacts.history["train_loss"][0],
        initial_artifacts.history["train_loss"][0],
        equal_nan=True,
    )
    assert resumed_artifacts.metadata["resumed_from_checkpoint"] is True
    assert resumed_artifacts.metadata["completed_epochs"] == 3
    assert latest_metadata["epoch"] == 3


def test_posterior_model_save_registers_artifact_metadata(tmp_path):
    task = _build_runtime_task(
        retrieval_name="registry_demo",
        noise_mode="gaussian",
        seed=61,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "registry_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.2",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=2,
        parameter_space="unconstrained",
        seed=11,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    registry = FileArtifactRegistry((tmp_path / "artifact_registry").as_posix())
    save_directory = tmp_path / "registered_posterior"
    posterior.save(
        save_directory.as_posix(),
        artifact_registry=registry,
        artifact_version="1.2.3",
    )

    registry_payload = json.loads(
        (tmp_path / "artifact_registry" / "registry.json").read_text()
    )
    registered_entry = registry_payload["artifacts"][0]


def test_posterior_checkpoint_falls_back_from_orbax_for_callable_leaves(
    tmp_path, monkeypatch
):
    task = _build_runtime_task(
        retrieval_name="orbax_fallback_demo",
        noise_mode="gaussian",
        seed=71,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "orbax_fallback_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.3",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )

    class _FailingOrbaxBackend:
        name = "orbax"

        def save(self, state, output_directory, metadata):
            del state, output_directory, metadata
            raise ValueError(
                'TypeHandler lookup failed for: type=<class "jax._src.custom_derivatives.custom_jvp">'
            )

        def restore(self, template_state, output_directory):
            del template_state, output_directory
            raise AssertionError("restore should not be called in this test")

    monkeypatch.setattr(
        "petitRADTRANS.experimental.sbi.training.resolve_checkpoint_backend",
        lambda preferred="auto": _FailingOrbaxBackend(),
    )

    checkpoint_directory = tmp_path / "orbax_fallback_checkpoints"
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=1,
        parameter_space="unconstrained",
        checkpoint_directory=checkpoint_directory.as_posix(),
        checkpoint_backend="auto",
        seed=13,
        task_metadata={"task_name": task.name},
    )
    artifacts = posterior.fit(reader)

    latest_metadata = json.loads(
        (checkpoint_directory / "latest" / "metadata.json").read_text()
    )
    best_metadata = json.loads(
        (checkpoint_directory / "best" / "metadata.json").read_text()
    )

    assert artifacts.metadata["checkpoint_backend"] == "equinox"
    assert latest_metadata["checkpoint_backend"] == "equinox"
    assert best_metadata["checkpoint_backend"] == "equinox"
    assert (
        latest_metadata["checkpoint_backend_fallback_reason"]
        == "orbax_unsupported_type"
    )
    assert (checkpoint_directory / "latest" / "state.pkl").exists()
    assert (checkpoint_directory / "best" / "state.pkl").exists()


def test_generic_loader_supports_legacy_flow_metadata(tmp_path):
    task = _build_runtime_task(
        retrieval_name="legacy_loader_demo",
        noise_mode="gaussian",
        seed=62,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "legacy_loader_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.2",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=5e-3,
        batch_size=2,
        num_epochs=2,
        parameter_space="unconstrained",
        seed=17,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    save_directory = tmp_path / "legacy_saved_posterior"
    posterior.save(save_directory.as_posix())

    metadata_path = save_directory / "metadata.json"
    legacy_metadata = json.loads(metadata_path.read_text())
    del legacy_metadata["estimator_family"]
    del legacy_metadata["metadata_schema_version"]
    del legacy_metadata["estimator_config"]
    metadata_path.write_text(json.dumps(legacy_metadata, sort_keys=True, indent=2))

    loaded = load_posterior_estimator(save_directory.as_posix())

    assert loaded.estimator_family == "conditional_flow"
    assert loaded.flow_family == "spline"


def test_flow_matching_posterior_fit_sample_and_generic_load(tmp_path):
    task = _build_runtime_task(
        retrieval_name="flow_matching_demo",
        noise_mode="gaussian",
        seed=67,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "flow_matching_dataset").as_posix(),
        n_simulations=8,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 6, DatasetSplit.VALIDATION: 2},
        dataset_version="0.8.9",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = FlowMatchingPosterior(
        parameter_dim=1,
        embedding_dim=24,
        hidden_dim=32,
        num_velocity_layers=2,
        learning_rate=1e-3,
        batch_size=2,
        num_epochs=2,
        parameter_space="unconstrained",
        integration_steps=12,
        seed=73,
        task_metadata={"task_name": task.name},
    )

    artifacts = posterior.fit(reader)
    train_batch = next(
        reader.iter_batches(
            split=DatasetSplit.TRAIN, batch_size=1, parameter_space="unconstrained"
        )
    )
    encoded_observation = posterior.encode_observation(train_batch.observations[0])
    samples = posterior.sample_posterior(encoded_observation, n_samples=6, seed=79)

    save_directory = tmp_path / "saved_flow_matching_posterior"
    posterior.save(save_directory.as_posix())
    loaded = load_posterior_estimator(save_directory.as_posix())
    loaded_samples = loaded.sample_posterior(encoded_observation, n_samples=6, seed=79)
    saved_metadata = json.loads((save_directory / "metadata.json").read_text())

    assert len(artifacts.history["train_loss"]) == 2
    assert artifacts.metadata["integration_steps"] == 12
    assert np.asarray(samples.samples).shape == (6, 1)
    assert np.asarray(loaded_samples.samples).shape == (6, 1)
    assert saved_metadata["estimator_family"] == "flow_matching"
    assert saved_metadata["estimator_config"]["num_velocity_layers"] == 2
    assert loaded.estimator_family == "flow_matching"


def test_spline_posterior_holdout_interval_coverage(tmp_path):
    task = _build_runtime_task(
        retrieval_name="calibration_demo",
        noise_mode="no_noise",
        seed=71,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "calibration_dataset").as_posix(),
        n_simulations=18,
        chunk_size=3,
        split_counts={
            DatasetSplit.TRAIN: 12,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.3",
    )
    store = HDF5SimulationDatasetStore(chunk_size=3)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=20,
        parameter_space="unconstrained",
        seed=13,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    test_batch = next(
        reader.iter_batches(
            split=DatasetSplit.TEST, batch_size=4, parameter_space="unconstrained"
        )
    )
    posterior_draws = []
    for blocks in test_batch.observations:
        encoded = posterior.encode_observation(blocks)
        posterior_draws.append(
            posterior.sample_posterior(encoded, n_samples=128, seed=17).samples
        )

    posterior_draws_array = np.asarray(posterior_draws, dtype=float)
    truths = np.asarray(test_batch.parameters, dtype=float)
    coverage_50 = empirical_interval_coverage(posterior_draws_array, truths, level=0.5)
    coverage_95 = empirical_interval_coverage(posterior_draws_array, truths, level=0.95)
    lower_50, upper_50 = central_credible_interval(posterior_draws_array[0], level=0.5)
    lower_95, upper_95 = central_credible_interval(posterior_draws_array[0], level=0.95)

    assert posterior_draws_array.shape == (4, 128, 1)
    assert lower_50.shape == (1,)
    assert upper_50.shape == (1,)
    assert lower_95.shape == (1,)
    assert upper_95.shape == (1,)
    assert 0.0 <= coverage_50 <= coverage_95 <= 1.0
    assert np.all((upper_95 - lower_95) >= (upper_50 - lower_50))


def test_generate_sbc_report_returns_rank_summaries(tmp_path):
    task = _build_runtime_task(
        retrieval_name="sbc_report_demo",
        noise_mode="no_noise",
        seed=81,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "sbc_dataset").as_posix(),
        n_simulations=16,
        chunk_size=2,
        split_counts={
            DatasetSplit.TRAIN: 10,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.4",
    )
    store = HDF5SimulationDatasetStore(chunk_size=2)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=12,
        parameter_space="unconstrained",
        seed=19,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    report = generate_sbc_report(
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=64,
        batch_size=2,
        seed=23,
    )

    assert report.ranks.shape == (4, 1)
    assert report.rank_histogram_counts.shape == (1, 65)
    assert report.posterior_means.shape == (4, 1)
    assert report.truths.shape == (4, 1)
    assert len(report.coverages) == 3
    assert report.metadata["split"] == "test"
    assert report.metadata["parameter_space"] == "unconstrained"
    assert np.all(report.ranks >= 0)
    assert np.all(report.ranks <= 64)


def test_plot_posterior_corner_returns_square_axes_grid():
    matplotlib = pytest.importorskip("matplotlib")
    del matplotlib

    samples = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 1.9],
            [0.2, 0.9, 2.1],
            [0.3, 1.2, 1.8],
        ],
        dtype=float,
    )

    figure, axes = plot_posterior_corner(
        samples,
        parameter_names=["alpha", "beta", "gamma"],
        bins=8,
    )

    assert axes.shape == (3, 3)
    assert len(figure.axes) == 9


def test_generate_local_sensitivity_report_tracks_linear_information_content():
    task = _build_runtime_task(
        retrieval_name="local_sensitivity_demo",
        noise_mode="no_noise",
        seed=121,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.1, 0.2]),
    )
    observation_blocks = [
        ObservationBlock(
            name="synthetic_spectrum",
            modality=ObservationModality.SPECTRUM,
            values=np.array([0.5, 1.5], dtype=float),
            uncertainties=np.array([0.1, 0.2], dtype=float),
            coordinates=np.array([1.0, 2.0], dtype=float),
        )
    ]
    posterior_samples = np.asarray([[0.2], [0.4], [0.6], [0.8]], dtype=float)
    posterior_log_probabilities = np.asarray([-2.0, -0.5, -0.2, -1.0], dtype=float)

    report = generate_local_sensitivity_report(
        task=task,
        posterior_samples=posterior_samples,
        observation_blocks=observation_blocks,
        posterior_log_probabilities=posterior_log_probabilities,
        quantile_levels=(0.1, 0.9),
        finite_difference_std_fraction=0.25,
        finite_difference_relative_step=1e-3,
        seed=7,
    )
    payload = local_sensitivity_report_to_payload(report)

    assert report.parameter_names == ("amplitude",)
    assert len(report.representative_points) == 4
    assert report.representative_points[0].label == "posterior_mean"
    assert np.allclose(
        report.representative_points[0].whitened_jacobian[:, 0],
        np.array([10.0, 5.0]),
        atol=0.2,
    )
    assert report.representative_points[0].effective_rank == 1
    assert payload["parameter_names"] == ["amplitude"]
    assert payload["representative_points"][0]["label"] == "posterior_mean"


def test_generate_posterior_predictive_report_summarizes_heldout_split(tmp_path):
    task = _build_runtime_task(
        retrieval_name="predictive_report_demo",
        noise_mode="no_noise",
        seed=91,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "predictive_dataset").as_posix(),
        n_simulations=18,
        chunk_size=3,
        split_counts={
            DatasetSplit.TRAIN: 12,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TEST: 4,
        },
        dataset_version="0.8.5",
    )
    store = HDF5SimulationDatasetStore(chunk_size=3)
    dataset = store.open(generated.manifest)
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=1,
        embedding_dim=32,
        num_coupling_layers=3,
        hidden_dim=32,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=16,
        parameter_space="unconstrained",
        seed=29,
        task_metadata={"task_name": task.name},
    )
    posterior.fit(reader)

    report = generate_posterior_predictive_report(
        task,
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=48,
        interval_level=0.9,
        seed=31,
    )

    assert report.metadata["split"] == "test"
    assert report.metadata["n_cases"] == 4
    assert report.metadata["parameter_space"] == "unconstrained"
    assert "synthetic_spectrum" in report.observed_values
    assert report.observed_values["synthetic_spectrum"].shape == (4, 2)
    assert report.predictive_mean["synthetic_spectrum"].shape == (4, 2)
    assert report.predictive_std["synthetic_spectrum"].shape == (4, 2)
    assert report.interval_lower["synthetic_spectrum"].shape == (4, 2)
    assert report.interval_upper["synthetic_spectrum"].shape == (4, 2)
    assert 0.0 <= report.interval_coverage["synthetic_spectrum"] <= 1.0
    assert report.mean_absolute_error["synthetic_spectrum"] >= 0.0


def test_generate_simulation_dataset_supports_resume_and_deterministic_split_policy(
    tmp_path,
):
    storage_uri = (tmp_path / "resume_dataset").as_posix()
    task = _build_runtime_task(
        retrieval_name="resume_demo",
        noise_mode="no_noise",
        seed=23,
        include_nuisance=False,
        store_log_likelihood=False,
        uncertainties=jnp.array([0.05, 0.05]),
    )

    initial = generate_simulation_dataset(
        task=task,
        storage_uri=storage_uri,
        n_simulations=4,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 2, DatasetSplit.VALIDATION: 2},
        split_policy=SplitSamplingPolicy.ROUND_ROBIN,
        dataset_version="0.6.0",
    )
    resumed = generate_simulation_dataset(
        task=task,
        storage_uri=storage_uri,
        n_simulations=6,
        chunk_size=2,
        split_counts={DatasetSplit.TRAIN: 3, DatasetSplit.VALIDATION: 3},
        split_policy=SplitSamplingPolicy.ROUND_ROBIN,
        dataset_version="0.6.0",
        resume=True,
    )

    store = HDF5SimulationDatasetStore(chunk_size=2)
    resumed_dataset = store.open(resumed.manifest)
    train_split = resumed_dataset.read_split(DatasetSplit.TRAIN)
    validation_split = resumed_dataset.read_split(DatasetSplit.VALIDATION)

    assert initial.manifest.n_simulations == 4
    assert resumed.manifest.n_simulations == 6
    assert resumed.manifest.splits == {"train": 3, "validation": 3}
    assert resumed.preprocessing_metadata is not None
    assert resumed.artifact_metadata.metadata["n_simulations"] == 6
    assert train_split["parameters"].shape == (3, 1)
    assert validation_split["parameters"].shape == (3, 1)


def test_shuffled_split_policy_is_deterministic_for_same_seed(tmp_path):
    task = _build_runtime_task(
        retrieval_name="shuffle_demo",
        noise_mode="no_noise",
        seed=31,
        uncertainties=jnp.array([0.05, 0.05]),
    )

    first = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "shuffle_a").as_posix(),
        n_simulations=5,
        chunk_size=2,
        split_fractions={DatasetSplit.TRAIN: 0.6, DatasetSplit.TEST: 0.4},
        split_policy=SplitSamplingPolicy.SHUFFLED,
        split_seed=99,
        dataset_version="0.7.0",
    )
    second = generate_simulation_dataset(
        task=task,
        storage_uri=(tmp_path / "shuffle_b").as_posix(),
        n_simulations=5,
        chunk_size=2,
        split_fractions={DatasetSplit.TRAIN: 0.6, DatasetSplit.TEST: 0.4},
        split_policy=SplitSamplingPolicy.SHUFFLED,
        split_seed=99,
        dataset_version="0.7.0",
    )

    store = HDF5SimulationDatasetStore(chunk_size=2)
    first_dataset = store.open(first.manifest)
    second_dataset = store.open(second.manifest)
    first_train = first_dataset.read_split(DatasetSplit.TRAIN)
    second_train = second_dataset.read_split(DatasetSplit.TRAIN)
    first_test = first_dataset.read_split(DatasetSplit.TEST)
    second_test = second_dataset.read_split(DatasetSplit.TEST)

    assert first.manifest.splits == {"train": 3, "test": 2}
    assert second.manifest.splits == {"train": 3, "test": 2}
    np.testing.assert_allclose(first_train["parameters"], second_train["parameters"])
    np.testing.assert_allclose(first_test["parameters"], second_test["parameters"])


@pytest.mark.parametrize("spectrum_encoder_type", ["convolution", "resmlp"])
@pytest.mark.parametrize("exact_summary_dim", [0, EXACT_SUMMARY_DIM])
def test_encode_paths_agree(spectrum_encoder_type, exact_summary_dim):
    """``encode`` must return exactly what the training path produces.

    The flow is conditioned on embeddings built by ``encode_from_prestacked``
    (training) and ``encode_stacked_batch`` (batched validation). The
    single-observation ``encode`` path serves target inference, so any
    discrepancy silently feeds the trained flow an out-of-distribution context.
    A unit-RMS rescaling here previously destroyed the overall embedding
    scale -- the amplitude information that pins the planet radius -- leaving
    validation metrics healthy while the target posterior collapsed.
    """
    from petitRADTRANS.experimental.sbi.flows.objectives import batch_embeddings

    n_wavelengths = 32
    rng = np.random.default_rng(0)
    coordinates = np.linspace(1.0, 3.0, n_wavelengths)
    encoder = ObservationEncoder(
        embedding_dim=16,
        spectrum_embedding_dim=16,
        hidden_dim=16,
        spectrum_encoder_type=spectrum_encoder_type,
        n_wavelengths=n_wavelengths,
        exact_summary_dim=exact_summary_dim,
        key=jax.random.PRNGKey(0),
    )

    # Two observations differing only by an overall amplitude factor: the
    # embeddings must differ, and every path must agree on each of them.
    blocks_batch = [
        [
            ObservationBlock(
                name="obs",
                modality=ObservationModality.SPECTRUM,
                values=scale * (1.0 + 0.05 * rng.standard_normal(n_wavelengths)),
                uncertainties=np.full(n_wavelengths, 0.01 * scale),
                coordinates=coordinates,
                # The exact-summary channel reaches the encoder through block
                # metadata on the ``encode``/``encode_stacked_batch`` paths and
                # through the pre-stacked field on the training path; all three
                # must pick up the same vector.
                metadata={
                    "exact_summary": np.arange(EXACT_SUMMARY_DIM, dtype=float) + scale
                },
            )
        ]
        for scale in (1.0, 4.0)
    ]

    single = np.stack(
        [np.asarray(encoder.encode(blocks).embedding) for blocks in blocks_batch]
    )
    stacked = np.asarray(encoder.encode_stacked_batch(blocks_batch))
    prestacked = np.asarray(
        batch_embeddings(
            SimpleNamespace(encoder=encoder), _prestack_observation_blocks(blocks_batch)
        )
    )

    width = 16 + exact_summary_dim
    assert single.shape == stacked.shape == prestacked.shape == (2, width)
    assert encoder.context_dim == width
    np.testing.assert_allclose(single, stacked, rtol=0, atol=1e-5)
    np.testing.assert_allclose(single, prestacked, rtol=0, atol=1e-5)
    # Guard the property the bug removed: the embedding must respond to the
    # observation's absolute scale, not just its shape.
    assert np.max(np.abs(single[0] - single[1])) > 1e-3
    if exact_summary_dim:
        # The exact channel must arrive bit-for-bit, not merely close: its whole
        # purpose is to deliver a statistic at a precision no learned layer can.
        np.testing.assert_array_equal(
            single[:, 16:],
            np.stack(
                [
                    np.arange(EXACT_SUMMARY_DIM, dtype=np.float32) + scale
                    for scale in (1.0, 4.0)
                ]
            ),
        )


def test_checkpoint_restore_reads_pre_move_module_paths(tmp_path):
    """Checkpoints written before the package moved must still restore.

    Trainer checkpoints pickle classes by dotted module path. Runs that
    predate the ``petitRADTRANS.sbi`` -> ``petitRADTRANS.experimental.sbi``
    move therefore reference the old path, which no longer imports; restoring
    them is required to re-run inference on an already-trained posterior.
    """
    import importlib
    import io

    import dill

    from petitRADTRANS.experimental.sbi.training import (
        EquinoxCheckpointBackend,
        _LegacyPathUnpickler,
        TrainingSchemeConfig as _TrainingSchemeConfig,
    )

    # The old path must genuinely be gone -- no import alias was added.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("petitRADTRANS.sbi.training")

    resolved = _LegacyPathUnpickler(io.BytesIO(b"")).find_class(
        "petitRADTRANS.sbi.training", "TrainingSchemeConfig"
    )
    assert resolved is _TrainingSchemeConfig

    # ...and the restore path must actually route through the remapping
    # unpickler, not plain ``dill.load``.
    backend = EquinoxCheckpointBackend()
    checkpoint_dir = tmp_path / "best"
    backend.save({"scheme": _TrainingSchemeConfig()}, checkpoint_dir, {"epoch": 7})

    looked_up: list[str] = []
    original_find_class = _LegacyPathUnpickler.find_class

    def _spy(self, module, name):
        looked_up.append(module)
        return original_find_class(self, module, name)

    _LegacyPathUnpickler.find_class = _spy
    try:
        state, metadata = backend.restore(None, checkpoint_dir)
    finally:
        _LegacyPathUnpickler.find_class = original_find_class

    assert metadata["epoch"] == 7
    assert isinstance(state["scheme"], _TrainingSchemeConfig)
    assert any(
        module.startswith("petitRADTRANS.experimental.sbi") for module in looked_up
    ), f"restore did not resolve classes through the remapping unpickler: {looked_up}"
    assert dill is not None


def test_refinement_verdict_rejects_the_failures_seen_in_production():
    """The delivery gate must reject a collapsed or uncalibrated refinement.

    Both benchmarks have shipped a repeated point estimate as "the posterior"
    while the diagnostics naming the failure sat unread on disk. The numbers
    below are the ones those runs actually produced.
    """
    from petitRADTRANS.experimental.sbi.config import RefinementConfig
    from petitRADTRANS.experimental.sbi.workflow import (
        _count_degenerate_marginals,
        _refinement_verdict,
    )

    importance = RefinementConfig(method="importance_sampling")

    # Emission run 20260729T121612Z: ESS 1.0 out of 16384 proposals.
    collapsed = _refinement_verdict(
        "importance_sampling",
        importance,
        {"effective_sample_size": 1.0, "sampling_efficiency": 6.103515625e-05},
        None,
    )
    assert collapsed["accepted"] is False
    assert collapsed["status"] == "rejected"
    assert "effective sample size" in collapsed["reasons"][0]

    healthy = _refinement_verdict(
        "importance_sampling",
        importance,
        {"effective_sample_size": 4000.0, "sampling_efficiency": 0.25},
        None,
    )
    assert healthy["accepted"] is True

    # Transmission run 20260729T185922Z: held-out coverage 0.476 (nominal 0.90)
    # and median chi2/N 11.55.
    laplace = RefinementConfig(method="map_laplace", test_time_validation=True)
    uncalibrated = _refinement_verdict(
        "map_laplace",
        laplace,
        None,
        {"overall_coverage_90": 0.4756944444444444, "median_chi2_per_n": 11.546550664885721},
    )
    assert uncalibrated["accepted"] is False
    assert len(uncalibrated["reasons"]) == 2

    calibrated = _refinement_verdict(
        "map_laplace", laplace, None, {"overall_coverage_90": 0.88, "median_chi2_per_n": 1.2}
    )
    assert calibrated["accepted"] is True

    # No refinement requested is not a rejection.
    assert _refinement_verdict("none", importance, None, None)["status"] == "not_requested"

    # A refinement with no diagnostics at all is delivered, but flagged.
    assert (
        _refinement_verdict("map_laplace", laplace, None, None)["status"]
        == "accepted_without_validation"
    )

    # Degenerate-marginal counting: 4096 copies of one point vs real draws.
    assert _count_degenerate_marginals(np.tile([1.0, 2.0, 3.0], (4096, 1))) == 3
    assert _count_degenerate_marginals(np.random.default_rng(0).normal(size=(4096, 3))) == 0


def test_truncation_box_construction_guards_and_composition():
    """Truncation must be generous, verifiable, and chainable.

    A box that drops posterior mass silently bakes round one's mistakes into
    round two, so construction is deliberately conservative and the report
    is what authorises the truncation, not the construction itself.
    """
    from petitRADTRANS.experimental.sbi.truncation import (
        TruncationBox,
        build_truncation_box,
        truncation_box_report,
    )

    rng = np.random.default_rng(0)
    names = ("sharp", "broad", "skewed")
    # One tightly-constrained parameter, one at the prior, one skewed.
    samples = np.column_stack(
        [
            np.clip(rng.normal(0.5, 0.01, 20000), 0, 1),
            rng.uniform(0.0, 1.0, 20000),
            np.clip(rng.beta(5.0, 2.0, 20000), 0, 1),
        ]
    )

    box = build_truncation_box(samples, names, coverage=0.999, safety_factor=1.5)
    assert box.parameter_names == names
    # The sharp direction truncates hard; the uninformative one cannot.
    assert box.widths[0] < 0.15
    assert box.widths[1] > 0.9
    assert 0.0 < box.volume_fraction < 1.0

    report = truncation_box_report(box, samples)
    assert report["safe"] is True
    assert report["captured_fraction"] > 0.99

    # A box that excludes the truth is refused, and says which parameter.
    outside = truncation_box_report(
        box, samples, truth_cube=np.array([0.99, 0.5, 0.5])
    )
    assert outside["safe"] is False
    assert outside["truth_inside_box"] is False
    assert "sharp" in outside["reasons"][0]

    # A box that keeps too little round-one mass is refused too.
    stingy = build_truncation_box(
        samples, names, coverage=0.30, safety_factor=1.0, min_box_fraction=1e-3
    )
    assert truncation_box_report(stingy, samples)["safe"] is False

    # min_box_fraction floors a marginal that round one over-sharpened.
    floored = build_truncation_box(
        samples, names, coverage=0.5, safety_factor=1.0, min_box_fraction=0.25
    )
    assert np.all(floored.widths >= 0.25 - 1e-12)

    # Bounds always stay inside the unit cube, even pushed against an edge.
    edge = build_truncation_box(
        np.column_stack([np.full(500, 0.999), np.full(500, 0.001), np.full(500, 0.5)]),
        names,
        min_box_fraction=0.1,
    )
    assert np.all(edge.lower >= 0.0) and np.all(edge.upper <= 1.0)

    # Round-two coordinates are relative to round one's box; composing yields
    # the absolute cube box, and round counting follows.
    inner = build_truncation_box(box.from_cube(samples[box.contains(samples)]), names)
    composed = box.compose(inner)
    assert np.all(composed.lower >= box.lower - 1e-12)
    assert np.all(composed.upper <= box.upper + 1e-12)
    assert composed.metadata["round"] == 2
    with pytest.raises(ValueError):
        box.compose(
            TruncationBox(
                lower=np.zeros(3), upper=np.ones(3),
                parameter_names=("a", "b", "c"), metadata={},
            )
        )

    # Round-trip through the payload and through the coordinate maps.
    restored = TruncationBox.from_payload(box.to_payload())
    np.testing.assert_allclose(restored.lower, box.lower)
    np.testing.assert_allclose(box.to_cube(box.from_cube(samples)), samples, atol=1e-12)

    # Degenerate boxes are rejected at construction, not at use.
    with pytest.raises(ValueError):
        TruncationBox(
            lower=np.array([0.5]), upper=np.array([0.5]),
            parameter_names=("x",), metadata={},
        )


def test_truncated_parameter_layout_restricts_simulation_and_refingerprints():
    """Wrapping the layout is the entire truncation mechanism.

    The simulator keeps drawing the unit cube exactly as before; the box turns
    those draws into physical values inside the retained region. Because the
    prior quantiles recorded in the task fingerprint go through the same
    transform, a truncated round fingerprints differently and so cannot reuse
    its parent round's simulation dataset.
    """
    from dataclasses import replace

    from petitRADTRANS.experimental.sbi.truncation import (
        TruncatedParameterLayout,
        TruncationBox,
    )

    task = _build_runtime_task(retrieval_name="truncation_demo")
    names = tuple(task.parameter_layout.free_parameter_names)
    n_free = len(names)
    index = names.index("amplitude")

    lower = np.zeros(n_free)
    upper = np.ones(n_free)
    lower[index], upper[index] = 0.25, 0.35
    box = TruncationBox(
        lower=lower, upper=upper, parameter_names=names, metadata={"round": 1}
    )
    truncated = replace(
        task,
        parameter_layout=TruncatedParameterLayout(
            base=task.parameter_layout, box=box
        ),
        metadata={**dict(task.metadata), "truncation_box": box.to_payload()},
    )

    assert task.task_fingerprint.digest != truncated.task_fingerprint.digest

    cube = np.random.default_rng(0).uniform(0.0, 1.0, size=(512, n_free))
    parent = np.asarray(task.parameter_layout.transform_cube_to_physical_batch(cube))
    child = np.asarray(
        truncated.parameter_layout.transform_cube_to_physical_batch(cube)
    )

    # The prior is the identity on the cube here, so the retained interval is
    # exactly the box.
    assert child[:, index].min() >= 0.25 - 1e-9
    assert child[:, index].max() <= 0.35 + 1e-9
    assert parent[:, index].min() < 0.25 and parent[:, index].max() > 0.35

    untouched = [i for i in range(n_free) if i != index]
    if untouched:
        np.testing.assert_allclose(parent[:, untouched], child[:, untouched])

    # Attributes unrelated to cube coordinates delegate to the base layout.
    assert truncated.parameter_layout.n_free_parameters == n_free
    assert tuple(truncated.parameter_layout.free_parameter_names) == names

    # The box survives into the run manifest, which is how the next round
    # composes back to absolute cube coordinates.
    assert truncated.metadata["truncation_box"]["lower"][index] == 0.25

    # Test-time refinement differentiates the forward model with respect to the
    # cube position, so the layout is called with JAX tracers, not concrete
    # arrays. Exercising it only on numpy arrays let a np.asarray on this path
    # reach a GPU run, which died with TracerArrayConversionError after the
    # round had already finished training.
    layout = truncated.parameter_layout

    def _mean_physical(position):
        return jnp.sum(layout.transform_cube_to_physical_batch(position))

    traced_input = jnp.full((1, n_free), 0.5)
    jitted = jax.jit(_mean_physical)(traced_input)
    assert np.isfinite(float(jitted))
    gradient = jax.grad(_mean_physical)(traced_input)
    assert np.all(np.isfinite(np.asarray(gradient)))
    # d(physical)/d(box coordinate) is the box width for the identity prior.
    np.testing.assert_allclose(
        float(np.asarray(gradient)[0, index]), box.widths[index], rtol=1e-5
    )

    unconstrained = jax.jit(
        lambda u: jnp.sum(layout.transform_unconstrained_batch(u))
    )(jnp.zeros((1, n_free)))
    assert np.isfinite(float(unconstrained))


def test_sbi_config_carries_truncation_settings():
    """A truncated round is configured through SBIConfig, not a stand-in.

    Exercising the resolver with a hand-rolled object that merely quacks like a
    config let a missing ``SBIConfig.truncation`` field reach the queue, where
    it failed at construction after the job had already been scheduled.
    """
    from pathlib import Path

    from petitRADTRANS.experimental.sbi import SBIConfig, TruncationConfig

    configuration = SBIConfig(
        retrieval_config=RetrievalConfig(retrieval_name="truncation_config_demo"),
        output_root=Path("/tmp/prt_sbi_truncation_config_demo"),
        truncation=TruncationConfig(enabled=True, source_run_label="round-one"),
    )
    assert configuration.truncation.enabled is True
    assert configuration.truncation.source_run_label == "round-one"
    # The setting is part of the run's provenance record.
    assert configuration.to_run_config()["truncation"]["source_run_label"] == "round-one"

    # Truncation stays off unless asked for.
    default = SBIConfig(
        retrieval_config=RetrievalConfig(retrieval_name="truncation_default_demo"),
        output_root=Path("/tmp/prt_sbi_truncation_default_demo"),
    )
    assert default.truncation.enabled is False


def test_embedding_information_report_separates_carried_from_discarded():
    """The probe must distinguish 'the flow failed' from 'the encoder discarded'.

    A broad posterior marginal is ambiguous on its own. This report regresses
    the truth on the embedding, so a near-zero R^2 with agreeing linear and
    nonlinear fits means the information is absent rather than merely hard to
    decode -- the difference between an encoder problem and a flow problem, and
    the measurement that was missing while a dozen runs chased the wrong one.
    """
    from petitRADTRANS.experimental.sbi.calibration import (
        generate_embedding_information_report,
    )

    rng = np.random.default_rng(0)
    n_cases, n_parameters, embedding_dim = 2048, 3, 8
    cube = rng.uniform(0.0, 1.0, size=(n_cases, n_parameters))
    # The embedding encodes parameter 0 exactly, parameter 1 only through a
    # nonlinearity, and parameter 2 not at all.
    sign = rng.choice([-1.0, 1.0], size=n_cases)
    embeddings = np.zeros((n_cases, embedding_dim))
    embeddings[:, 0] = cube[:, 0]
    # Parameter 1 survives only as a product with a random sign, so recovering
    # it needs the two embedding components multiplied together -- invisible to
    # a linear fit, available to the random-feature one.
    embeddings[:, 1] = cube[:, 1] * sign
    embeddings[:, 2] = sign
    embeddings[:, 3:] = rng.normal(size=(n_cases, embedding_dim - 3))

    class _Batch:
        def __init__(self, start, stop):
            self.observations = list(range(start, stop))
            self.parameters = cube[start:stop]

    class _Reader:
        def iter_batches(self, split=None, batch_size=256, shuffle=False, **kwargs):
            for start in range(0, n_cases, batch_size):
                yield _Batch(start, min(start + batch_size, n_cases))

    class _Posterior:
        def encode_observation(self, index):
            return SimpleNamespace(embedding=embeddings[index])

    report = generate_embedding_information_report(
        _Posterior(), _Reader(), ("carried", "nonlinear_only", "discarded"),
        max_cases=n_cases, batch_size=256,
    )

    assert report["n_cases"] == n_cases
    # Carried linearly: both probes see it.
    assert report["r2_linear"]["carried"] > 0.99
    assert report["r2_nonlinear"]["carried"] > 0.99
    # Carried nonlinearly: ridge cannot see it, the random-feature probe can.
    # This is why the nonlinear probe exists -- without it, a parameter the
    # encoder does carry would be misread as discarded.
    assert report["r2_linear"]["nonlinear_only"] < 0.2
    assert (
        report["r2_nonlinear"]["nonlinear_only"]
        > report["r2_linear"]["nonlinear_only"] + 0.1
    )
    # Discarded: both probes agree it is gone, and the implied scatter is the
    # prior's own (a uniform prior on the cube has sd 1/sqrt(12) ~ 0.289).
    assert report["r2_nonlinear"]["discarded"] < 0.05
    assert abs(report["implied_residual_sd_fraction"]["discarded"] - 0.289) < 0.02
    # A parameter the embedding pins has a much smaller implied scatter.
    assert report["implied_residual_sd_fraction"]["carried"] < 0.03


def test_run_sbi_end_to_end_on_an_analytic_task(tmp_path):
    """Drive the whole ``run_sbi`` pipeline once, on a task that runs in seconds.

    ``run_sbi`` is a long linear function whose stages share local state, and
    the unit tests around it all exercise individual pieces. Three separate
    changes have reached a GPU queue and failed there -- a config field that was
    never added, a numpy call on a JAX tracer, and a local read before its
    assignment -- none of which any unit test could see. This walks the real
    entry point end to end so that ordering and wiring break here instead.

    Deliberately tiny: an analytic forward model, a handful of wavelengths, and
    the smallest budgets the workflow accepts. It asserts the pipeline's
    contract, not inference quality.
    """
    from dataclasses import replace

    import tensorflow_probability.substrates.jax as tfp

    from petitRADTRANS.retrieval.runtime import ModelResult
    from petitRADTRANS.experimental.sbi import (
        EncoderConfig,
        FlowConfig,
        ObservationValueConstraint,
        RefinementConfig,
        SBIConfig,
        TrainingConfig,
        run_sbi,
    )

    distributions = tfp.distributions
    wavelengths = np.linspace(1.0, 2.5, 6)
    truth = {"amplitude": 1.3, "tilt": 0.2}

    def analytic_model(model_context, physical_params, pt_plot_mode=False):
        del model_context, pt_plot_mode
        grid = jnp.asarray(wavelengths)
        spectrum = physical_params["amplitude"] + physical_params["tilt"] * (grid - 1.75)
        return ModelResult(kind="spectrum", wavelengths=grid, spectrum=spectrum)

    setattr(analytic_model, "_prt_model_contract", "differentiable")

    noiseless = np.asarray(
        analytic_model(None, truth).spectrum, dtype=float
    )
    observed = noiseless + np.random.default_rng(0).normal(0.0, 0.05, noiseless.shape)

    configuration = RetrievalConfig(retrieval_name="run_sbi_smoke")
    configuration.add_parameter(
        "amplitude", is_free_parameter=True, distribution=distributions.Uniform(0.5, 2.0)
    )
    configuration.add_parameter(
        "tilt", is_free_parameter=True, distribution=distributions.Uniform(-0.5, 0.5)
    )
    configuration.add_data(
        name="smoke",
        path_to_observations=None,
        model_generating_function=analytic_model,
        radtrans_object=object(),
        wavelengths=jnp.asarray(wavelengths),
        spectrum=jnp.asarray(observed),
        uncertainties=jnp.full(wavelengths.shape, 0.05),
    )

    configuration_obj = SBIConfig(
            retrieval_config=configuration,
            output_root=tmp_path,
            n_simulations=512,
            simulation_batch_size=128,
            observation_value_constraint=ObservationValueConstraint(
                min_value=0.0, min_inclusive=False
            ),
            flow=FlowConfig(family="spline", num_transforms=2, hidden_dim=16, num_spline_bins=4),
            encoder=EncoderConfig(embedding_dim=8, spectrum_embedding_dim=8, hidden_dim=8),
            training=TrainingConfig(
                num_epochs=2, batch_size=32, learning_rate=1e-3, lr_schedule_total_epochs=2.0
            ),
            refinement=RefinementConfig(method="none"),
            n_posterior_samples=64,
            n_predictive_forward_model_samples=8,
            validation_max_cases=4,
            verbose_training_diagnostics=False,
            save_checkpoint_guardrails=False,
            truth_parameter_values=truth,
    )
    run_root = run_sbi(configuration_obj)
    configuration = configuration_obj

    summary = json.loads((run_root / "amortized_summary.json").read_text())
    # The stages whose wiring has broken before, each asserted explicitly.
    assert summary["delivered_posterior"] == "amortized"
    assert summary["refinement_verdict"]["status"] == "not_requested"
    assert summary["truncation_box"] is None
    assert set(summary["posterior_summary"]) == {"amplitude", "tilt"}
    assert (run_root / "target_posterior_samples.npy").exists()
    assert (run_root / "refinement_verdict.json").exists()
    # The embedding probe runs and reports both parameters, including the
    # target-neighbourhood scatter that needs the target observation threaded
    # through from much earlier in the pipeline.
    probe = summary["embedding_information"]
    assert probe is not None
    assert set(probe["r2_nonlinear"]) == {"amplitude", "tilt"}
    assert set(probe["target_neighbourhood_residual_sd_fraction"]) == {"amplitude", "tilt"}

    # Resuming interrupted training reuses the run directory and its trainer
    # checkpoints, and must still TRAIN rather than load a finished posterior.
    # Walltime-limited jobs depend on this, and conflating it with the
    # completed-model path sent a resubmitted 8M run straight to an assertion.
    resumed_root = run_sbi(
        replace(
            configuration,
            training=TrainingConfig(
                num_epochs=3, batch_size=32, learning_rate=1e-3, lr_schedule_total_epochs=3.0
            ),
            resume_training=True,
            run_label=run_root.name,
        )
    )
    assert resumed_root == run_root
    resumed = json.loads((resumed_root / "amortized_summary.json").read_text())
    assert resumed["delivered_posterior"] == "amortized"

    # The conditioning must carry the observation's amplitude exactly. This is
    # the whole point of the exact-summary channel, and it is asserted here --
    # on the real workflow -- rather than only at the encoder, because the
    # constants travel from the preprocessing sidecar through the pre-stacked
    # training arrays to the flow's context, and every one of those hops has to
    # hold for the number to come out right.
    statistic = probe["observation_statistic"]["log10_median_spectrum_scale"]
    assert statistic["r2_linear"] > 0.999, statistic
    stage_report = summary["encoder_stage_information"]
    assert "error" not in stage_report
    assert stage_report["stages"]["amplitude_features"]["r2_best"] > 0.99

    # Reusing an existing dataset whose sidecar predates the channel must
    # BACKFILL the constants, not silently emit zeros -- a disabled channel is
    # indistinguishable from a working one from the outside. Strip them and
    # rerun against the same dataset.
    sidecar = next((tmp_path / "datasets").glob("*.preprocessing.json"))
    payload = json.loads(sidecar.read_text())
    for block in payload["blocks"].values():
        block["exact_summary_center"] = None
        block["exact_summary_scale"] = None
    sidecar.write_text(json.dumps(payload))

    # ... and in the same run, exercise the observation-conditioned parameter
    # basis end to end: the fit reads the backfilled constants, the flow trains
    # in the reparameterised coordinate, and every delivered sample is mapped
    # back to prior cube coordinates before anything downstream sees it.
    reparameterised_root = run_sbi(
        replace(
            configuration,
            run_label=None,
            amplitude_conditioned_parameter="amplitude",
            seed=configuration.seed + 1,
        )
    )
    refitted = TaskPreprocessingMetadata.from_payload(json.loads(sidecar.read_text()))
    for stats in refitted.blocks.values():
        assert stats.exact_summary_center is not None
        assert len(stats.exact_summary_center) == EXACT_SUMMARY_DIM

    # The comparative delivery gate must actually execute in the real workflow.
    # It lives in a branch that only runs when a refinement survives the
    # absolute thresholds, and a long linear function with an untested branch is
    # how three defects reached the compute queue in this project already.
    refined_root = run_sbi(
        replace(
            configuration,
            run_label=None,
            refinement=RefinementConfig(
                method="map_laplace",
                map_num_steps=20,
                map_num_starts=1,
                map_num_scored_starts=8,
                test_time_validation=False,
            ),
            seed=configuration.seed + 2,
        )
    )
    refined_summary = json.loads((refined_root / "amortized_summary.json").read_text())
    verdict = refined_summary["refinement_verdict"]
    assert verdict["method"] == "map_laplace"
    # Whatever it decided, it must have made the comparison and recorded it.
    comparison = verdict["predictive_comparison"]
    assert comparison["refined_crps"] is not None
    assert comparison["amortized_crps"] is not None
    assert isinstance(comparison["refined_is_better"], bool)
    # And the delivered artifact must match the decision.
    expected = "refined_map_laplace" if verdict["accepted"] else "amortized"
    assert refined_summary["delivered_posterior"] == expected, (
        refined_summary["delivered_posterior"], verdict
    )

    reparameterised = json.loads(
        (reparameterised_root / "amortized_summary.json").read_text()
    )
    fit_report = reparameterised["amplitude_reparameterization"]
    assert fit_report["parameter"] == "amplitude"
    assert fit_report["captured_fraction"] >= 0.995
    # Delivered draws must lie in the prior cube, not the reparameterised one.
    samples = np.load(reparameterised_root / "target_posterior_samples.npy")
    assert samples.shape[1] == 2
    assert np.all(np.isfinite(samples))
    assert np.all(samples[:, 0] >= 0.5) and np.all(samples[:, 0] <= 2.0)


@pytest.mark.parametrize("space", ["raw", "transit_radius"])
def test_spectral_value_space_round_trips_and_paths_agree(space):
    """The forward transform, its inverse, and the batched path must agree.

    Every consumer needing raw physical units -- above all the likelihood used
    by test-time refinement -- goes through ``denormalize_spectral_values``. A
    second, restated inverse silently corrupted the target spectrum by ~100x
    once before, so a new value space is only safe if the inverse and the
    batched training path are pinned to the per-sample path here.
    """
    from petitRADTRANS.experimental.sbi.preprocessing import (
        compute_block_normalization_stats,
        denormalize_spectral_values,
        from_spectral_value_space,
        normalize_observation_block,
        normalize_spectral_block_matrix,
        to_spectral_value_space,
    )

    rng = np.random.default_rng(0)
    n_wavelengths = 32
    coordinates = np.linspace(1.0, 3.0, n_wavelengths)
    # Transmission physics: the atmosphere adds an annulus to the radius,
    # R_p(lambda) = R0 + h(lambda), with h a few scale heights. The scale
    # height depends on temperature and gravity, not on R0 -- log g is a free
    # parameter here, independent of the radius -- so h does not scale with R0.
    # Depth then squares that sum, which is exactly how the radius becomes
    # entangled with the feature amplitude.
    radii = rng.uniform(0.6, 2.0, 64)
    scale_heights = rng.uniform(0.004, 0.010, 64)
    annulus = scale_heights[:, None] * rng.standard_normal((64, n_wavelengths))
    depths = ((radii[:, None] + annulus) * 0.1027 / 1.0) ** 2
    uncertainties = 5.3e-6 * np.ones_like(depths)

    # 1. The transform itself round-trips, values and uncertainties alike.
    forward_values, forward_uncertainties = to_spectral_value_space(
        depths, uncertainties, space
    )
    back_values, back_uncertainties = from_spectral_value_space(
        forward_values, forward_uncertainties, space
    )
    np.testing.assert_allclose(back_values, depths, rtol=1e-12)
    np.testing.assert_allclose(back_uncertainties, uncertainties, rtol=1e-12)

    blocks = [
        ObservationBlock(
            name="spec",
            modality=ObservationModality.SPECTRUM,
            values=depths[i],
            uncertainties=uncertainties[i],
            coordinates=coordinates,
        )
        for i in range(depths.shape[0])
    ]
    stats = compute_block_normalization_stats(blocks, spectral_value_space=space)["spec"]
    assert stats.spectral_value_space == space

    # 2. The batched path equals the per-sample path bit-for-bit; training uses
    # the batched one whenever noise is resampled.
    batched = normalize_spectral_block_matrix(depths, uncertainties, None, stats)
    metadata = TaskPreprocessingMetadata(version="0.6.0", blocks={"spec": stats})
    per_sample = np.stack(
        [np.asarray(normalize_observation_block(block, metadata).values) for block in blocks]
    )
    np.testing.assert_allclose(batched["values"], per_sample, rtol=0, atol=1e-12)

    # 3. Normalization inverts back to raw physical units.
    recovered, recovered_uncertainties = denormalize_spectral_values(
        batched["values"], batched["uncertainties"], stats
    )
    np.testing.assert_allclose(recovered, depths, rtol=1e-6)
    np.testing.assert_allclose(recovered_uncertainties, uncertainties, rtol=1e-5)

    # 4. The point of the transit-radius space: after per-spectrum
    # standardization the feature amplitude must no longer scale with radius.
    centered = per_sample - np.median(per_sample, axis=1, keepdims=True)
    spread = np.std(centered, axis=1)
    correlation = float(np.corrcoef(radii, spread)[0, 1])
    if space == "raw":
        # Depth space: the same atmosphere looks different at every radius.
        assert abs(correlation) > 0.5, (
            "depth space is expected to entangle radius with feature amplitude, "
            f"got correlation {correlation:.3f}"
        )
    else:
        assert abs(correlation) < 0.35, (
            "transit-radius space should decouple feature amplitude from radius, "
            f"got correlation {correlation:.3f}"
        )


def test_noise_level_conditioning_randomises_error_bars():
    """Training-time error-bar scaling, and the fixed-noise path unchanged.

    The encoder already ingests sigma per wavelength and as an amplitude
    feature, so noise-level conditioning needs no architecture change -- only
    that training sees a range of error-bar scales, with the reported
    uncertainties matching the noise that was actually added.
    """
    from petitRADTRANS.experimental.sbi.preprocessing import (
        compute_block_normalization_stats,
    )
    from petitRADTRANS.experimental.sbi.training import _noisy_prestacked_observations

    rng_data = np.random.default_rng(0)
    n_samples, n_wavelengths = 256, 16
    deterministic = 0.0106 * (1.0 + 0.01 * rng_data.standard_normal((n_samples, n_wavelengths)))
    uncertainties = np.full((n_samples, n_wavelengths), 5.3e-6)
    coordinates = np.linspace(1.0, 3.0, n_wavelengths)

    stats = compute_block_normalization_stats(
        [
            ObservationBlock(
                name="spec",
                modality=ObservationModality.SPECTRUM,
                values=deterministic[i],
                uncertainties=uncertainties[i],
                coordinates=coordinates,
            )
            for i in range(n_samples)
        ]
    )
    metadata = TaskPreprocessingMetadata(version="0.6.0", blocks=stats)
    raw_blocks = [
        {
            "name": "spec",
            "modality": "spectrum",
            "deterministic": deterministic.astype(np.float32),
            "uncertainties": uncertainties.astype(np.float32),
            "coordinates": np.tile(coordinates, (n_samples, 1)).astype(np.float32),
            "mask": np.ones((n_samples, n_wavelengths), dtype=bool),
        }
    ]
    indices = np.arange(n_samples)

    # Disabled: the reported uncertainties are exactly the stored ones.
    fixed = _noisy_prestacked_observations(
        raw_blocks, indices, np.random.default_rng(1), metadata
    )
    # Enabled: scales span the requested decade and differ between samples.
    conditioned = _noisy_prestacked_observations(
        raw_blocks,
        indices,
        np.random.default_rng(1),
        metadata,
        noise_level_log10_range=(-0.5, 0.5),
    )
    fixed_unc = np.asarray(fixed.stacked_blocks[0][1])
    conditioned_unc = np.asarray(conditioned.stacked_blocks[0][1])
    assert fixed_unc.shape == conditioned_unc.shape

    # The fixed path gives every sample the same normalized uncertainty level;
    # the conditioned path spreads it over roughly a decade.
    fixed_levels = np.median(fixed_unc, axis=1)
    conditioned_levels = np.median(conditioned_unc, axis=1)
    fixed_spread = fixed_levels.max() / max(fixed_levels.min(), 1e-30)
    conditioned_spread = conditioned_levels.max() / max(conditioned_levels.min(), 1e-30)
    assert fixed_spread < 1.5, f"fixed-noise levels should not vary, got {fixed_spread:.2f}"
    assert conditioned_spread > 3.0, (
        f"conditioned levels should span ~10x, got {conditioned_spread:.2f}"
    )


def test_map_likelihood_ranked_starts_beat_the_posterior_mean():
    """Starting from the best-scoring draw must beat starting from the mean.

    Held-out MAP refinement failed on 6 of 16 cases (chi2/N of 11 to 131),
    dragging interval coverage to 0.611 against a nominal 0.90 -- and
    1 - 6/16 accounts for almost exactly that shortfall. The cause is the
    initialisation: the mean of a posterior that is near-prior in most
    dimensions is a low-likelihood point between the modes.
    """
    from petitRADTRANS.experimental.sbi.flows.refinement import _chunked_forward_chi2

    # A bimodal amortized posterior: the mean sits in the empty middle.
    draws = jnp.asarray(
        np.concatenate(
            [
                np.random.default_rng(0).normal(0.2, 0.02, (256, 1)),
                np.random.default_rng(1).normal(0.8, 0.02, (256, 1)),
            ]
        )
    )
    truth = 0.8

    def residual(cube):
        return (cube - truth) / 0.01

    chi2, batch_used = _chunked_forward_chi2(draws, residual, forward_batch_size=64)
    assert chi2.shape == (512,)
    assert batch_used == 64

    mean_start = float(jnp.mean(draws))
    best_start = float(draws[int(np.argmin(chi2))][0])
    mean_chi2 = float(((mean_start - truth) / 0.01) ** 2)
    best_chi2 = float(np.min(chi2))

    # The mean lands in the gap; the best draw lands on the correct mode.
    assert abs(mean_start - truth) > 0.2
    assert abs(best_start - truth) < 0.1
    assert best_chi2 < mean_chi2 / 100.0

    # Ranking is what the MAP path consumes: the ordering must put the draw
    # nearest the truth first, so the optimiser starts on the right mode.
    ranking = np.argsort(chi2)
    assert abs(float(draws[int(ranking[0])][0]) - truth) < 0.1
    assert abs(float(draws[int(ranking[-1])][0]) - truth) > 0.4


def _amplitude_spanning_blocks(n_cases=240, n_wavelengths=48, seed=0):
    """Transmission-like spectra spanning the benchmark's amplitude range."""
    wavelengths = np.linspace(1.0, 5.0, n_wavelengths)
    blocks = []
    log_depths = []
    for index in range(n_cases):
        generator = np.random.default_rng(seed + index)
        log_depth = -2.17 + 0.32 * generator.standard_normal()
        depth = 10.0**log_depth
        feature = 1.6e-3 * np.exp(-0.5 * ((wavelengths - 2.7) / 0.25) ** 2)
        values = depth * (1.0 + feature * generator.uniform(0.5, 1.5))
        uncertainties = np.full(n_wavelengths, depth * 2.6e-4)
        values = values + uncertainties * generator.standard_normal(n_wavelengths)
        blocks.append(
            [
                ObservationBlock(
                    name="spectrum",
                    modality=ObservationModality.SPECTRUM,
                    values=values,
                    uncertainties=uncertainties,
                    coordinates=wavelengths,
                    mask=np.zeros(n_wavelengths, dtype=bool),
                )
            ]
        )
        log_depths.append(log_depth)
    return blocks, np.asarray(log_depths)


def test_exact_summary_channel_delivers_log_amplitude_at_data_precision():
    """The conditioning vector must carry log median depth, not approximate it.

    This is the property the whole channel exists for. A transmission spectrum
    determines log10(median depth) to a few parts in 10^5; the trained
    convolution encoder was measured to deliver it with a residual of 0.085 dex
    against a prior spread of 0.32. The exact channel is a fixed affine map of
    the statistic itself, so a linear readout of the context must recover it to
    numerical precision -- and must do so far better than the learned half of
    the same vector.
    """
    from petitRADTRANS.experimental.sbi.preprocessing import (
        fit_task_preprocessing,
        normalize_observation_blocks,
    )

    raw_blocks, truth = _amplitude_spanning_blocks()
    n_wavelengths = len(raw_blocks[0][0].values)
    preprocessing = fit_task_preprocessing(raw_blocks[:180])
    stats = preprocessing.blocks["spectrum"]

    # The fitted constants are the only thing standing between the raw scalar
    # and the flow, and they must come from the training split, not the batch.
    assert stats.exact_summary_center is not None
    assert len(stats.exact_summary_center) == EXACT_SUMMARY_DIM
    assert all(value > 0.0 for value in stats.exact_summary_scale)

    normalized = [
        normalize_observation_blocks(blocks, preprocessing) for blocks in raw_blocks
    ]
    encoder = ObservationEncoder(
        embedding_dim=32,
        spectrum_embedding_dim=16,
        hidden_dim=16,
        n_wavelengths=n_wavelengths,
        exact_summary_dim=EXACT_SUMMARY_DIM,
        key=jax.random.PRNGKey(0),
    )
    context = np.stack(
        [np.asarray(encoder.encode(blocks).embedding) for blocks in normalized]
    )

    def _held_out_residual(features):
        n_train = 180
        centre = features[:n_train].mean(axis=0)
        spread = features[:n_train].std(axis=0) + 1e-8
        design = np.c_[(features - centre) / spread, np.ones(features.shape[0])]
        weights = np.linalg.solve(
            design[:n_train].T @ design[:n_train]
            + 1e-8 * np.eye(design.shape[1]),
            design[:n_train].T @ truth[:n_train],
        )
        return float(np.std(design[n_train:] @ weights - truth[n_train:]))

    exact_residual = _held_out_residual(context[:, 32:])
    learned_residual = _held_out_residual(context[:, :32])

    assert exact_residual < 1e-3, exact_residual
    assert learned_residual > 10.0 * exact_residual, (learned_residual, exact_residual)


def test_exact_summary_features_are_standardised_by_fixed_constants():
    """A per-batch rescaling here would remove the amplitude it must carry.

    The deleted ``_normalize_embedding`` failed exactly this way. Splitting one
    batch in two must not change any observation's exact-summary vector.
    """
    from petitRADTRANS.experimental.sbi.preprocessing import fit_task_preprocessing

    raw_blocks, _ = _amplitude_spanning_blocks(n_cases=64, n_wavelengths=32, seed=7)
    preprocessing = fit_task_preprocessing(raw_blocks)
    stats = preprocessing.blocks["spectrum"]

    values = np.stack([np.asarray(blocks[0].values) for blocks in raw_blocks])
    uncertainties = np.stack(
        [np.asarray(blocks[0].uncertainties) for blocks in raw_blocks]
    )
    log_scale = np.log10(np.median(np.abs(values), axis=1))

    whole = standardize_exact_summary_features(
        exact_summary_feature_matrix(values, uncertainties, None, log_scale), stats
    )
    halves = np.concatenate(
        [
            standardize_exact_summary_features(
                exact_summary_feature_matrix(
                    values[start:stop],
                    uncertainties[start:stop],
                    None,
                    log_scale[start:stop],
                ),
                stats,
            )
            for start, stop in ((0, 17), (17, 64))
        ]
    )
    np.testing.assert_allclose(whole, halves, rtol=0, atol=0)

    # The first feature is the raw log amplitude, so it must be an exactly
    # affine function of it -- recoverable, not merely correlated.
    assert EXACT_SUMMARY_FEATURE_NAMES[0] == "log_spectrum_scale"
    recovered = (
        whole[:, 0] * stats.exact_summary_scale[0] + stats.exact_summary_center[0]
    )
    np.testing.assert_allclose(recovered, log_scale, rtol=0, atol=1e-9)


def test_missing_exact_summary_constants_disable_the_channel():
    """Preprocessing sidecars written before the channel existed must still load."""
    from petitRADTRANS.experimental.sbi.preprocessing import (
        BlockNormalizationStats,
        TaskPreprocessingMetadata,
    )

    legacy_payload = {
        "version": "0.5.0",
        "blocks": {
            "spectrum": {
                "name": "spectrum",
                "modality": "spectrum",
                "value_mean": 0.0,
                "value_std": 1.0,
            }
        },
        "metadata": {},
    }
    metadata = TaskPreprocessingMetadata.from_payload(legacy_payload)
    stats = metadata.blocks["spectrum"]
    assert isinstance(stats, BlockNormalizationStats)
    assert stats.exact_summary_center is None
    zeros = standardize_exact_summary_features(np.ones((3, EXACT_SUMMARY_DIM)), stats)
    np.testing.assert_array_equal(zeros, np.zeros((3, EXACT_SUMMARY_DIM)))


def test_encoder_stage_audit_runs_through_the_real_forward_pass():
    """The stage audit must instrument the encoder, not a re-implementation.

    A previous audit rebuilt the forward pass to inspect it and reported R^2 of
    exactly 0.00000 for three consecutive stages, because the reconstruction fed
    the amplitude in at the wrong layer. Here the stages come from the same
    ``_forward_with_stages`` call that produces the embedding, so the audited
    tensors are the ones the encoder actually computes.
    """
    from petitRADTRANS.experimental.sbi.calibration import (
        generate_encoder_stage_information_report,
    )
    from petitRADTRANS.experimental.sbi.preprocessing import (
        fit_task_preprocessing,
        normalize_observation_blocks,
    )

    raw_blocks, _ = _amplitude_spanning_blocks(n_cases=160, n_wavelengths=32, seed=3)
    n_wavelengths = 32
    preprocessing = fit_task_preprocessing(raw_blocks)
    normalized = [
        normalize_observation_blocks(blocks, preprocessing) for blocks in raw_blocks
    ]
    encoder = ObservationEncoder(
        embedding_dim=16,
        spectrum_embedding_dim=16,
        hidden_dim=16,
        n_wavelengths=n_wavelengths,
        exact_summary_dim=EXACT_SUMMARY_DIM,
        key=jax.random.PRNGKey(1),
    )

    # The instrumented pass must agree with the plain one exactly.
    prestacked = _prestack_observation_blocks(normalized[:8])
    stages = encoder.stage_features_from_prestacked(prestacked)
    np.testing.assert_allclose(
        np.asarray(stages["context"]),
        np.asarray(encoder.encode_from_prestacked(prestacked)),
        rtol=0,
        atol=0,
    )

    class _Batch:
        def __init__(self, observations):
            self.observations = observations
            self.parameters = np.zeros((len(observations), 1))

    class _Reader:
        preprocessing_metadata = preprocessing

        def iter_batches(self, split=None, batch_size=64, shuffle=False, **kwargs):
            for start in range(0, len(normalized), batch_size):
                yield _Batch(normalized[start:start + batch_size])

    report = generate_encoder_stage_information_report(
        SimpleNamespace(model=SimpleNamespace(encoder=encoder)),
        _Reader(),
        max_cases=160,
        batch_size=64,
    )
    assert "error" not in report
    assert report["target"] == "log10_median_spectrum_scale"
    # The raw amplitude features hold the statistic exactly; the audit is only
    # meaningful if it can see that.
    assert report["stages"]["amplitude_features"]["r2_best"] > 0.999
    for stage in report["stages"].values():
        assert stage["r2_best"] >= min(stage["r2_linear"], stage["r2_nonlinear"])


def test_map_covariance_keeps_the_residual_curvature_term():
    """``hessian`` must widen a curved fit relative to Gauss-Newton.

    Gauss-Newton drops ``sum_i r_i d^2 r_i / du^2``. With a converged but
    imperfect fit that term is not negligible, and dropping it makes the
    posterior too narrow -- the measured failure (coverage 0.52 against a
    nominal 0.90 at chi2/N 1.28). This builds a residual with real curvature and
    a non-zero optimum and checks the two methods disagree in that direction.
    """
    from petitRADTRANS.experimental.sbi.flows.refinement import (
        map_laplace_posterior_on_observation,
    )

    param_dim = 3

    def residual(cube):
        # The first coordinate carries a large, irreducible, concave residual:
        # a linear term pins it near zero, where ``1 - 10 c^2`` stays close to 1
        # with second derivative -20. The dropped Gauss-Newton term is then
        # r * d^2r/dc^2 ~ -20, comparable to and opposing J^T J -- the regime
        # where linearising makes the posterior too narrow. The other
        # coordinates are linear, so they must be untouched.
        centred = cube - 0.5
        return jnp.stack(
            [
                1.0 - 10.0 * centred[0] ** 2,
                6.0 * centred[0],
                4.0 * centred[1] - 0.4,
                3.0 * centred[2] - 0.3,
            ]
        )

    posterior = build_conditional_flow_posterior(
        parameter_dim=param_dim, n_wavelengths=8, seed=0
    )
    observation = _prestack_observation_blocks(
        [
            [
                ObservationBlock(
                    name="obs",
                    modality=ObservationModality.SPECTRUM,
                    values=np.ones(8),
                    uncertainties=np.full(8, 0.1),
                    coordinates=np.linspace(1.0, 2.0, 8),
                )
            ]
        ]
    )

    widths = {}
    diagnostics = {}
    for method in ("laplace", "hessian"):
        refined = map_laplace_posterior_on_observation(
            posterior,
            observation,
            residual,
            num_steps=250,
            n_posterior_samples=512,
            covariance_method=method,
            seed=0,
            verbose=False,
        )
        widths[method] = np.std(np.asarray(refined._fixed_cube_samples), axis=0)
        diagnostics[method] = refined._map_diagnostics

    assert diagnostics["hessian"]["covariance_method"] == "hessian"
    assert diagnostics["laplace"]["covariance_width_ratio_vs_gauss_newton"] == pytest.approx(
        1.0, abs=1e-4
    )
    # The curvature term must actually change the answer, and in the widening
    # direction on the curved coordinate.
    assert diagnostics["hessian"]["covariance_width_ratio_vs_gauss_newton"] > 1.05
    assert widths["hessian"][0] > 1.05 * widths["laplace"][0]
    # ... and only there: the linear coordinates have no curvature term.
    np.testing.assert_allclose(
        widths["hessian"][1:], widths["laplace"][1:], rtol=0.02
    )
    # Both must still report the misfit factor, applied or not.
    assert diagnostics["hessian"]["chi2_inflation_applied"] is False
    assert diagnostics["hessian"]["chi2_inflation_factor"] >= 1.0


def test_map_covariance_rejects_an_unknown_method():
    """A typo in the covariance method must fail loudly, not fall back."""
    from petitRADTRANS.experimental.sbi.flows.refinement import (
        map_laplace_posterior_on_observation,
    )

    posterior = build_conditional_flow_posterior(
        parameter_dim=2, n_wavelengths=4, seed=0
    )
    observation = _prestack_observation_blocks(
        [
            [
                ObservationBlock(
                    name="obs",
                    modality=ObservationModality.SPECTRUM,
                    values=np.ones(4),
                    uncertainties=np.full(4, 0.1),
                    coordinates=np.linspace(1.0, 2.0, 4),
                )
            ]
        ]
    )
    with pytest.raises(ValueError, match="covariance_method"):
        map_laplace_posterior_on_observation(
            posterior,
            observation,
            lambda cube: jnp.asarray([cube[0] - 0.5, cube[1] - 0.5]),
            num_steps=5,
            n_posterior_samples=8,
            covariance_method="gauss_newton",
            verbose=False,
        )


def _fitted_amplitude_reparameterization(n=2000, seed=0):
    """A radius/annulus setup with the benchmark's amplitude spread."""
    from petitRADTRANS.experimental.sbi.flows import (
        fit_amplitude_conditioned_parameter,
    )

    rng = np.random.default_rng(seed)
    log_amplitude = -2.17 + 0.32 * rng.standard_normal(n)
    # depth = (R_p + h)^2 / R_s^2, so the total radius goes as sqrt(amplitude);
    # the planet radius is that less a positive atmospheric annulus.
    total_radius = 100.0 * 10.0 ** (0.5 * log_amplitude)
    radius = total_radius * rng.uniform(0.6, 0.999, n)
    reparameterization, report = fit_amplitude_conditioned_parameter(
        parameter_name="planet_radius",
        parameter_index=2,
        physical_values=radius,
        log_amplitudes=log_amplitude,
        prior_lower=float(radius.min() * 0.8),
        prior_upper=float(radius.max() * 1.2),
        context_index=8,
        amplitude_center=-2.17,
        amplitude_scale=0.32,
    )
    return reparameterization, report, radius, log_amplitude


def test_amplitude_conditioned_basis_round_trips_and_touches_one_coordinate():
    """The coordinate change must be exactly invertible and strictly local.

    Every delivered posterior passes back through ``to_prior_cube``; an
    inaccurate inverse would corrupt the radius marginal of every run while
    leaving training metrics healthy -- the shape of the amplitude bug that
    started this campaign.
    """
    reparameterization, report, radius, log_amplitude = (
        _fitted_amplitude_reparameterization()
    )
    assert report["captured_fraction"] == 1.0
    # The point of the basis: the coordinate the flow must resolve is far
    # narrower than the raw prior range in the same units.
    assert report["compression_vs_prior"] > 3.0

    n = radius.shape[0]
    cube = np.zeros((n, 4))
    cube[:, 2] = (radius - reparameterization.prior_lower) / (
        reparameterization.prior_upper - reparameterization.prior_lower
    )
    context = np.zeros((n, 12))
    context[:, 8] = (log_amplitude + 2.17) / 0.32

    flow_cube = np.asarray(reparameterization.to_flow_cube(cube, context))
    recovered = np.asarray(reparameterization.to_prior_cube(flow_cube, context))

    np.testing.assert_allclose(recovered[:, 2], cube[:, 2], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(flow_cube[:, [0, 1, 3]], cube[:, [0, 1, 3]])
    # The log amplitude must come back out of the context exactly -- it is a
    # fixed affine map of it, not an estimate.
    np.testing.assert_allclose(
        np.asarray(reparameterization.log_amplitude(context)),
        log_amplitude,
        rtol=0,
        atol=1e-5,
    )


def test_amplitude_conditioned_basis_moves_the_radius_with_the_observed_depth():
    """The sharp direction must be handled by the formula, not by the network.

    Two observations whose only difference is the measured amplitude must yield
    posterior radii scaled by exactly ``10^(exponent * delta log amplitude)``,
    for an untrained flow. That is the property no amount of training can supply
    at the required four-and-a-half significant digits, and it is what the whole
    reparameterisation buys.
    """
    reparameterization, _, _, _ = _fitted_amplitude_reparameterization()
    n_samples = 512
    flow_cube = np.tile(
        np.linspace(0.05, 0.95, n_samples)[:, None], (1, 4)
    )  # identical draws under both observations

    delta = 1.0  # one standard-deviation step in the standardised coordinate
    contexts = []
    for offset in (0.0, delta):
        context = np.zeros((n_samples, 12))
        context[:, 8] = offset
        contexts.append(context)

    radii = []
    for context in contexts:
        prior_cube = np.asarray(reparameterization.to_prior_cube(flow_cube, context))
        radii.append(
            reparameterization.prior_lower
            + prior_cube[:, 2]
            * (reparameterization.prior_upper - reparameterization.prior_lower)
        )

    expected_ratio = 10.0 ** (
        reparameterization.amplitude_exponent * reparameterization.amplitude_scale * delta
    )
    np.testing.assert_allclose(radii[1] / radii[0], expected_ratio, rtol=1e-4)


def test_amplitude_conditioned_basis_serializes_with_the_posterior(tmp_path):
    """A reparameterised model must reload with the identical coordinate.

    The constants are not recoverable from the weights, so losing them on reload
    would silently reinterpret every sample.
    """
    from petitRADTRANS.experimental.sbi.flows import (
        AmplitudeConditionedParameter,
        load_posterior_estimator,
    )

    reparameterization, _, _, _ = _fitted_amplitude_reparameterization()
    posterior = build_conditional_flow_posterior(
        parameter_dim=4,
        n_wavelengths=8,
        seed=0,
        reparameterization=reparameterization,
    )
    assert posterior.model.reparameterization is reparameterization

    payload = reparameterization.to_payload()
    restored = AmplitudeConditionedParameter.from_payload(payload)
    assert restored == reparameterization
    assert AmplitudeConditionedParameter.from_payload(None) is None

    config = posterior._build_estimator_config()
    assert config["reparameterization"] == payload
    rebuilt = AmplitudeConditionedParameter.from_payload(config["reparameterization"])
    assert rebuilt == reparameterization
    del tmp_path, load_posterior_estimator


def test_amplitude_conditioned_basis_rejects_an_ill_posed_configuration():
    """Bad bounds must fail at construction, not produce silent NaNs."""
    from petitRADTRANS.experimental.sbi.flows import (
        AmplitudeConditionedParameter,
        fit_amplitude_conditioned_parameter,
    )

    kwargs = dict(
        parameter_name="planet_radius",
        parameter_index=0,
        context_index=1,
        amplitude_center=0.0,
        amplitude_scale=1.0,
        ratio_lower=-1.0,
        ratio_upper=1.0,
    )
    # A log ratio needs a strictly positive lower bound.
    with pytest.raises(ValueError, match="positive prior lower bound"):
        AmplitudeConditionedParameter(prior_lower=0.0, prior_upper=2.0, **kwargs)
    with pytest.raises(ValueError, match="prior_upper must exceed"):
        AmplitudeConditionedParameter(prior_lower=2.0, prior_upper=1.0, **kwargs)
    with pytest.raises(ValueError, match="ratio_upper must exceed"):
        AmplitudeConditionedParameter(
            **{**kwargs, "ratio_lower": 1.0, "ratio_upper": -1.0},
            prior_lower=1.0,
            prior_upper=2.0,
        )
    with pytest.raises(ValueError, match="at least 16"):
        fit_amplitude_conditioned_parameter(
            parameter_name="r",
            parameter_index=0,
            physical_values=np.ones(4),
            log_amplitudes=np.zeros(4),
            prior_lower=0.5,
            prior_upper=2.0,
            context_index=1,
            amplitude_center=0.0,
            amplitude_scale=1.0,
        )


def test_reparameterised_log_prob_stays_a_density_in_the_original_cube():
    """The change of variables must be accounted for, not dropped.

    A validation loss that omitted the Jacobian would not be comparable with a
    run that did not reparameterise, and checkpoint selection reads that loss.
    """
    from petitRADTRANS.experimental.sbi.flows.cube import _parameter_space_log_prob

    reparameterization, _, _, _ = _fitted_amplitude_reparameterization()
    posterior = build_conditional_flow_posterior(
        parameter_dim=4, n_wavelengths=8, seed=0
    )
    flow = posterior.model.flow
    context_dim = posterior.context_dim

    rng = np.random.default_rng(1)
    cube = jnp.asarray(rng.uniform(0.2, 0.8, size=(16, 4)), dtype=jnp.float32)
    embeddings = jnp.zeros((16, context_dim), dtype=jnp.float32)

    plain = np.asarray(_parameter_space_log_prob(flow, cube, embeddings, "cube"))
    reparameterised = np.asarray(
        _parameter_space_log_prob(
            flow, cube, embeddings, "cube", reparameterization=reparameterization
        )
    )
    assert np.all(np.isfinite(reparameterised))
    # The Jacobian term is a real contribution, not a no-op.
    assert not np.allclose(plain, reparameterised)

    # Numerically: the reparameterised density in cube coordinates must equal
    # the flow density in its own coordinates times |d(flow cube)/d(cube)|.
    flow_cube = reparameterization.to_flow_cube(cube, embeddings)
    without_jacobian = np.asarray(
        _parameter_space_log_prob(flow, flow_cube, embeddings, "cube")
    )
    jacobian = np.asarray(
        reparameterization.log_abs_det_jacobian(cube, embeddings)
    )
    np.testing.assert_allclose(
        reparameterised, without_jacobian + jacobian, rtol=1e-4, atol=1e-3
    )


def test_broadcast_spectral_mask_matches_the_reader_convention():
    """A per-spectrum flag must mark one wavelength, not the whole spectrum.

    Masks arrive in four shapes and ``np.broadcast_to`` handles only one: an
    HDF5 dataset stores one flag per spectrum, ``(batch,)``, which numpy aligns
    against the trailing wavelength axis and rejects. That killed a submitted
    24M-dataset run at the sidecar-backfill step.

    The convention is set by ``NormalizedObservationDatasetReader`` (dataset.py:
    ``mask[:, 0] = stored_mask`` for a 1-D stored mask) and mirrors the
    per-sample path, which pads a short mask with ``False``. Statistics fitted
    on a different valid set than the encoder sees would be quietly wrong, so
    this pins the two together.
    """
    from petitRADTRANS.experimental.sbi.preprocessing import broadcast_spectral_mask

    n_samples, n_wavelengths = 4, 6

    # None and scalars.
    np.testing.assert_array_equal(
        broadcast_spectral_mask(None, (n_samples, n_wavelengths)),
        np.zeros((n_samples, n_wavelengths), dtype=bool),
    )
    scalar = broadcast_spectral_mask(True, (n_samples, n_wavelengths))
    assert scalar[:, 0].all() and not scalar[:, 1:].any()

    # One flag per spectrum -> first wavelength only. This is the shape that
    # crashed, and marking the whole spectrum invalid would throw away 220
    # points per flagged row instead of 1.
    per_spectrum = np.array([True, False, True, False])
    expanded = broadcast_spectral_mask(per_spectrum, (n_samples, n_wavelengths))
    np.testing.assert_array_equal(expanded[:, 0], per_spectrum)
    assert not expanded[:, 1:].any()

    # Exactly what the reader produces for the same input, byte for byte.
    reader_equivalent = np.zeros((n_samples, n_wavelengths), dtype=bool)
    reader_equivalent[:, 0] = per_spectrum
    np.testing.assert_array_equal(expanded, reader_equivalent)

    # Per wavelength, shared across the batch.
    per_wavelength = np.array([False, True, False, False, True, False])
    shared = broadcast_spectral_mask(per_wavelength, (n_samples, n_wavelengths))
    for row in range(n_samples):
        np.testing.assert_array_equal(shared[row], per_wavelength)

    # A single observation: a length-n_wavelengths mask is per wavelength, not
    # per spectrum. This is the target-spectrum path.
    single = broadcast_spectral_mask(per_wavelength, (1, n_wavelengths))
    np.testing.assert_array_equal(single[0], per_wavelength)

    # Already per wavelength per spectrum: returned unchanged.
    full = np.zeros((n_samples, n_wavelengths), dtype=bool)
    full[2, 3] = True
    np.testing.assert_array_equal(
        broadcast_spectral_mask(full, (n_samples, n_wavelengths)), full
    )


def test_exact_summary_features_accept_a_per_spectrum_mask():
    """The statistic must be computable from a stored dataset mask directly.

    The failing job read masks straight from HDF5 and handed them to
    ``exact_summary_feature_matrix``; a shape it cannot take is a crash at the
    only point in the pipeline that touches an 8M dataset.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(32, 20))
    uncertainties = np.full((32, 20), 0.1)
    log_scale = rng.normal(size=32)

    per_spectrum = np.zeros(32, dtype=bool)
    per_spectrum[::7] = True
    features = exact_summary_feature_matrix(
        values, uncertainties, per_spectrum, log_scale
    )
    assert features.shape == (32, EXACT_SUMMARY_DIM)
    assert np.all(np.isfinite(features))
    # The first feature is the supplied log amplitude, untouched by masking.
    np.testing.assert_allclose(features[:, 0], log_scale, rtol=0, atol=0)


def test_checkpoint_selection_is_not_dominated_by_the_worst_single_parameter():
    """The calibration penalty must rank on the mean, not the max.

    Measured over all 384 epochs of the 8M transmission run:
    ``calibration_coverage_error_max`` has a floor of 0.312 and never falls
    below its 0.25 threshold in any epoch of any run, while the mean sits at
    0.183-0.240 and always passes. Ranking on the max therefore penalised every
    checkpoint by ~1.8 nats that jittered by 0.35 nats epoch to epoch -- more
    than the 0.1-0.3 nats separating good late epochs -- and selection was
    decided by that noise. The chosen checkpoint delivered a target predictive
    1.8x wider and 2.5x more biased than the same run's best-loss checkpoint.
    """
    from petitRADTRANS.experimental.sbi.flows.regularization import (
        _flow_validation_selection_metric,
    )

    def diagnostics(mean_error, max_error, threshold=0.25):
        return {
            "checkpoint_is_stable": 0.0,
            "checkpoint_is_pathological": 0.0,
            "calibration_coverage_error_mean": mean_error,
            "calibration_coverage_error_max": max_error,
            "checkpoint_stability_margin_calibration_coverage_error_mean": (
                threshold - mean_error
            ),
            "checkpoint_stability_margin_calibration_coverage_error_max": (
                threshold - max_error
            ),
        }

    # Two epochs from the real run: the better-loss one has a slightly worse
    # MAX and a slightly better MEAN. Ranking on the max inverts the order.
    better_loss = _flow_validation_selection_metric(-12.44, diagnostics(0.213, 0.379))
    worse_loss = _flow_validation_selection_metric(-12.42, diagnostics(0.221, 0.331))
    assert better_loss < worse_loss, (better_loss, worse_loss)

    # A calibrated-on-average checkpoint must carry no calibration penalty, so
    # the term cannot jitter the ranking when it is not diagnosing anything.
    calibrated = _flow_validation_selection_metric(-10.0, diagnostics(0.20, 0.40))
    assert calibrated == pytest.approx(-10.0 + 1.0)  # only the unstable floor

    # Genuine collapse raises the mean across parameters and must still be
    # penalised -- that is what the term exists for.
    collapsed = _flow_validation_selection_metric(-10.0, diagnostics(0.60, 0.85))
    assert collapsed > calibrated + 1.0

    # And a pathological checkpoint is still barred outright.
    pathological = dict(diagnostics(0.60, 0.85), checkpoint_is_pathological=1.0)
    assert _flow_validation_selection_metric(-99.0, pathological) == 1.0e6


def test_delivery_gate_rejects_a_refinement_worse_than_the_amortized_posterior():
    """The gate must compare, not just threshold.

    On run 20260821T162601Z the MAP+Laplace refinement passed every absolute
    threshold -- held-out coverage error 0.164 against a limit of 0.25, median
    chi2/N 1.12 against a limit of 10 -- and was delivered, while being
    measurably worse than the amortized posterior it displaced: CRPS 8.65e-06
    against 3.65e-06, MAE 1.74 sigma against 0.83, and a predictive interval
    28.5 sigma wide against 6.8. The run shipped the worse of the two artifacts
    it had already computed, because nothing ever asked which was better.

    CRPS is the criterion because it is a proper scoring rule: a posterior
    cannot improve it by collapsing onto a biased point (which flatters MAE) or
    by inflating until it contains everything (which flatters coverage).
    """
    from types import SimpleNamespace
    from petitRADTRANS.experimental.sbi.workflow import (
        _predictive_crps,
        _refinement_predictive_comparison,
    )

    def result(crps):
        return SimpleNamespace(
            posterior_predictive=SimpleNamespace(crps={"TransitSynthetic05pct": crps})
        )

    # The real numbers from that run.
    refined, amortized = result(8.650727e-06), result(3.652579e-06)
    assert _predictive_crps(refined) == pytest.approx(8.650727e-06)

    verdict = _refinement_predictive_comparison(refined, amortized, max_crps_ratio=1.0)
    assert verdict["refined_is_better"] is False
    assert verdict["crps_ratio"] == pytest.approx(8.650727 / 3.652579, rel=1e-6)

    # A refinement that genuinely improves the predictive is still delivered.
    better = _refinement_predictive_comparison(
        result(1.5e-06), amortized, max_crps_ratio=1.0
    )
    assert better["refined_is_better"] is True

    # The knob can restore the old threshold-only behaviour.
    tolerant = _refinement_predictive_comparison(
        refined, amortized, max_crps_ratio=float("inf")
    )
    assert tolerant["refined_is_better"] is True

    # A missing predictive must not silently reject a refinement the thresholds
    # accepted -- it defers rather than inventing a decision.
    absent = _refinement_predictive_comparison(
        SimpleNamespace(posterior_predictive=None), amortized, max_crps_ratio=1.0
    )
    assert absent["refined_is_better"] is True
    assert "skipped" in absent["note"]

    # Non-finite scores must not poison the mean.
    mixed = SimpleNamespace(
        posterior_predictive=SimpleNamespace(
            crps={"a": float("nan"), "b": 2.0e-06}
        )
    )
    assert _predictive_crps(mixed) == pytest.approx(2.0e-06)


def test_refinement_config_carries_the_comparative_threshold():
    """The new knob must round-trip through the config and its provenance dict."""
    from petitRADTRANS.experimental.sbi.config import RefinementConfig

    assert RefinementConfig().max_refinement_crps_ratio == 1.0
    assert RefinementConfig(max_refinement_crps_ratio=1.25).max_refinement_crps_ratio == 1.25
