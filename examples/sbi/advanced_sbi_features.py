"""Feature-oriented SBI walkthrough for petitRADTRANS.

This script goes beyond the bare minimum and highlights several pieces of the
current SBI stack:

1. defining a task with both spectral and photometric observation blocks;
2. generating a reproducible simulation corpus with preprocessing metadata;
3. training a checkpointable spline posterior;
4. saving and registering the trained posterior as an artifact;
5. loading the saved posterior back from disk;
6. generating SBC and posterior-predictive calibration reports;
7. serving the trained posterior through the preprocessing-aware
   ``AmortizedRetrieval`` interface.

The script remains synthetic so that it can run as a self-contained example.
It is intended to show how the different module pieces fit together.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.sbi import (
    AmortizedRetrieval,
    ArtifactRole,
    ConditionalSplineFlowPosterior,
    DatasetSplit,
    FileArtifactRegistry,
    NoiseModelConfig,
    NormalizedObservationDatasetReader,
    ObservationBlock,
    SBITask,
    SimulationConfig,
    SplitSamplingPolicy,
    ZarrSimulationDatasetStore,
    build_observation_block_batch,
    generate_posterior_predictive_report,
    generate_sbc_report,
    generate_simulation_dataset,
)


def synthetic_spectrum_model(
    prt_object,
    parameters,
    pt_plot_mode: bool = False,
    adaptive_mesh_refinement: bool = True,
):
    """Small forward model reused for both spectral and photometric data."""

    del prt_object, pt_plot_mode, adaptive_mesh_refinement
    amplitude = parameters['amplitude'].value
    tilt = parameters['tilt'].value
    wavelengths = jnp.array([1.0, 1.5, 2.0], dtype=jnp.float64)
    spectrum = amplitude + tilt * (wavelengths - 1.5)
    return wavelengths, spectrum


def synthetic_photometric_transform(wavelengths, spectrum):
    """Compress a spectrum into a one-point synthetic photometric datum."""

    del wavelengths
    return jnp.mean(jnp.asarray(spectrum, dtype=jnp.float64))


def build_feature_demo_task() -> SBITask:
    """Create a mixed-modality task used throughout the walkthrough."""

    configuration = RetrievalConfig(retrieval_name='advanced_sbi_demo')
    configuration.add_parameter('planet_radius', False, value=1.0)
    configuration.add_parameter('amplitude', True, transform_prior_cube_coordinate=lambda x: 2.0 * x - 0.5)
    configuration.add_parameter('tilt', True, transform_prior_cube_coordinate=lambda x: x - 0.5)

    configuration.add_data(
        name='synthetic_spectrum',
        path_to_observations=None,
        model_generating_function=synthetic_spectrum_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 1.5, 2.0], dtype=jnp.float64),
        spectrum=jnp.zeros(3, dtype=jnp.float64),
        uncertainties=jnp.array([0.05, 0.05, 0.05], dtype=jnp.float64),
    )
    configuration.add_data(
        name='synthetic_photometry',
        path_to_observations=None,
        model_generating_function=synthetic_spectrum_model,
        radtrans_object=object(),
        photometry=True,
        photometric_transformation_function=synthetic_photometric_transform,
        photometric_bin_edges=(1.0, 2.0),
        wavelengths=jnp.array([1.5], dtype=jnp.float64),
        spectrum=jnp.array([0.0], dtype=jnp.float64),
        uncertainties=jnp.array([0.08], dtype=jnp.float64),
    )

    return SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=6,
            noise_model=NoiseModelConfig(mode='gaussian', seed=5),
            store_per_datapoint_log_likelihood=True,
        ),
    )


def describe_observation_blocks(blocks: list[ObservationBlock]) -> None:
    """Pretty-print the raw observation blocks used for one inference call."""

    for block in blocks:
        print(f'  - {block.name} [{block.modality.value}] values={np.asarray(block.values)}')


def main() -> None:
    task = build_feature_demo_task()

    with tempfile.TemporaryDirectory(prefix='prt_sbi_advanced_') as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_uri = (tmp_path / 'advanced_dataset.zarr').as_posix()
        checkpoint_dir = tmp_path / 'checkpoints'
        registry_dir = tmp_path / 'artifact_registry'
        saved_posterior_dir = tmp_path / 'saved_posterior'

        # Step 1: generate a corpus with deterministic split assignment so the
        # train/validation/test breakdown is reproducible.
        generated = generate_simulation_dataset(
            task=task,
            storage_uri=dataset_uri,
            n_simulations=24,
            chunk_size=6,
            split_counts={
                DatasetSplit.TRAIN: 16,
                DatasetSplit.VALIDATION: 4,
                DatasetSplit.TEST: 4,
            },
            split_policy=SplitSamplingPolicy.ROUND_ROBIN,
            dataset_version='0.2.0',
        )
        print('Dataset manifest splits:', generated.manifest.splits)
        print('Preprocessing version:', generated.preprocessing_metadata.version)

        store = ZarrSimulationDatasetStore(chunk_size=6)
        dataset = store.open(generated.manifest)
        reader = NormalizedObservationDatasetReader(
            dataset=dataset,
            preprocessing_metadata=generated.preprocessing_metadata,
        )

        # Step 2: train a posterior with checkpointing enabled. This mirrors how
        # a longer-running experiment would preserve resumable state.
        posterior = ConditionalSplineFlowPosterior(
            parameter_dim=2,
            embedding_dim=48,
            num_coupling_layers=4,
            hidden_dim=64,
            learning_rate=1e-3,
            batch_size=4,
            num_epochs=18,
            parameter_space='unconstrained',
            checkpoint_directory=checkpoint_dir.as_posix(),
            seed=7,
            task_metadata={'task_name': task.name},
        )
        training_artifacts = posterior.fit(reader)
        print('Completed epochs:', training_artifacts.metadata['completed_epochs'])
        print('Best validation loss:', np.min(training_artifacts.validation_metrics['validation_loss']))

        # Step 3: save and register the posterior artifact. The registry is local
        # JSON storage for now, but the saved metadata already includes task and
        # preprocessing provenance.
        registry = FileArtifactRegistry(registry_dir.as_posix())
        posterior.save(
            saved_posterior_dir.as_posix(),
            artifact_registry=registry,
            artifact_version='1.0.0',
        )
        resolved_location = registry.resolve(ArtifactRole.POSTERIOR_MODEL, task.name, version='1.0.0')
        print('Registered posterior location:', resolved_location)

        registry_payload = json.loads((registry_dir / 'registry.json').read_text())
        print('Registered artifact count:', len(registry_payload['artifacts']))

        # Step 4: restore the saved model to demonstrate the persistence path.
        loaded_posterior = ConditionalSplineFlowPosterior.load(saved_posterior_dir.as_posix())
        print('Loaded posterior parameter space:', loaded_posterior.parameter_space)
        print('Loaded preprocessing payload keys:', sorted(loaded_posterior.preprocessing_metadata_payload.keys()))

        # Step 5: compute calibration-style summaries on the held-out split.
        sbc_report = generate_sbc_report(
            loaded_posterior,
            reader,
            split=DatasetSplit.TEST,
            n_posterior_samples=96,
            batch_size=2,
            seed=11,
        )
        predictive_report = generate_posterior_predictive_report(
            task,
            loaded_posterior,
            reader,
            split=DatasetSplit.TEST,
            n_posterior_samples=96,
            interval_level=0.9,
            seed=13,
        )
        print('SBC mean rank:', sbc_report.mean_rank)
        print('SBC normalized mean-rank error:', sbc_report.normalized_mean_rank_error)
        print('Predictive report datasets:', sorted(predictive_report.observed_values.keys()))

        # Step 6: use the public inference API with raw observations. Because the
        # loaded posterior carries preprocessing metadata, the inference service
        # can normalize raw blocks internally before encoding, while the returned
        # predictive report still compares against the original raw values.
        raw_test_split = dataset.read_split(DatasetSplit.TEST)
        raw_blocks_batch = build_observation_block_batch(
            raw_test_split['observations'],
            task.observation_schema.modalities,
        )
        raw_observation = raw_blocks_batch[0]
        print('Raw observation blocks used for amortized inference:')
        describe_observation_blocks(raw_observation)

        retrieval = AmortizedRetrieval(task=task, posterior_estimator=loaded_posterior)
        retrieval_result = retrieval.infer(
            raw_observation,
            n_posterior_samples=96,
            include_posterior_predictive=True,
            posterior_predictive_interval_level=0.9,
            seed=17,
        )

        posterior_samples = np.asarray(retrieval_result.posterior.samples, dtype=float)
        print('Posterior sample matrix shape:', posterior_samples.shape)
        print('First posterior draw:', posterior_samples[0])
        print('Inference preprocessing applied:', retrieval_result.metadata['preprocessing_applied'])
        print(
            'Predictive mean for synthetic_spectrum:',
            retrieval_result.posterior_predictive.predictive_mean['synthetic_spectrum'],
        )
        print(
            'Observed raw synthetic_spectrum:',
            retrieval_result.posterior_predictive.observed_values['synthetic_spectrum'],
        )
        print(
            'Predictive coverage for synthetic_photometry:',
            retrieval_result.posterior_predictive.interval_coverage['synthetic_photometry'],
        )


if __name__ == '__main__':
    main()
