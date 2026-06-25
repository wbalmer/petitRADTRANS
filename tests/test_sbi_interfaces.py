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
except (ImportError, SystemExit):
    pass
sys.modules.setdefault("pymultinest", SimpleNamespace())

from petitRADTRANS.retrieval.runtime import ParameterLayout  # noqa: E402
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig  # noqa: E402
from petitRADTRANS.sbi import (  # noqa: E402
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
from petitRADTRANS.sbi.artifacts import (
    ArtifactMetadata,
    ArtifactRole,
    FileArtifactRegistry,
)  # noqa: E402
from petitRADTRANS.sbi.calibration import (
    central_credible_interval,
    empirical_interval_coverage,
)  # noqa: E402
from petitRADTRANS.sbi.compatibility import CompatibilityReport  # noqa: E402
from petitRADTRANS.sbi.dataset import (
    SimulationDatasetManifest,
    SplitSamplingPolicy,
)  # noqa: E402
from petitRADTRANS.sbi.flows import (
    ConditionalNeuralAutoregressiveFlow,
    ConditionalSplineFlow,
)  # noqa: E402
from petitRADTRANS.sbi.preprocessing import (
    compute_block_normalization_stats,
)  # noqa: E402
from petitRADTRANS.sbi.training import SBITrainer, TrainingConfig  # noqa: E402


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
        TrainingConfig(
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
        "petitRADTRANS.sbi.training.resolve_checkpoint_backend",
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
