"""Minimal end-to-end SBI workflow for petitRADTRANS.

This script shows the smallest useful path through the current
``petitRADTRANS.sbi`` module:

1. define a synthetic retrieval task;
2. generate a small simulation dataset;
3. train a conditional spline posterior;
4. run amortized inference on one raw observation;
5. inspect posterior samples and a posterior-predictive report.

The example uses a deliberately simple synthetic forward model so that the
workflow is easy to read and can run quickly on a laptop. It is intended as a
reference for the control flow and APIs rather than as a scientifically useful
retrieval setup.

If you persist a trained SBI estimator from this workflow and reload it later,
prefer ``load_posterior_estimator(...)`` over a concrete class loader so the
artifact remains compatible with future estimator families.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.sbi import (
    AmortizedRetrieval,
    ConditionalSplineFlowPosterior,
    DatasetSplit,
    NoiseModelConfig,
    NormalizedObservationDatasetReader,
    RuntimeSimulator,
    SBITask,
    SimulationConfig,
    ZarrSimulationDatasetStore,
    build_observation_block_batch,
    generate_simulation_dataset,
)


def synthetic_spectrum_model(
    prt_object,
    parameters,
    pt_plot_mode: bool = False,
    adaptive_mesh_refinement: bool = True,
):
    """Return a tiny two-point synthetic spectrum.

    The retrieval runtime expects a model-generating callable that returns
    wavelengths and spectrum values. Here the free parameter ``amplitude``
    controls a simple linear toy spectrum.
    """

    del prt_object, pt_plot_mode, adaptive_mesh_refinement
    amplitude = parameters['amplitude'].value
    wavelengths = jnp.array([1.0, 2.0], dtype=jnp.float64)
    spectrum = jnp.array([amplitude, amplitude + 1.0], dtype=jnp.float64)
    return wavelengths, spectrum


def build_demo_task() -> SBITask:
    """Create a minimal SBI task backed by the retrieval runtime."""

    configuration = RetrievalConfig(retrieval_name='basic_sbi_demo')
    configuration.add_parameter('planet_radius', False, value=1.0)
    configuration.add_parameter('amplitude', True, transform_prior_cube_coordinate=lambda x: x)
    configuration.add_data(
        name='synthetic_spectrum',
        path_to_observations=None,
        model_generating_function=synthetic_spectrum_model,
        radtrans_object=object(),
        wavelengths=jnp.array([1.0, 2.0], dtype=jnp.float64),
        spectrum=jnp.array([0.0, 0.0], dtype=jnp.float64),
        uncertainties=jnp.array([0.05, 0.05], dtype=jnp.float64),
    )

    return SBITask.from_retrieval_config(
        configuration,
        simulation_config=SimulationConfig(
            batch_size=4,
            noise_model=NoiseModelConfig(mode='no_noise', seed=0),
        ),
    )


def main() -> None:
    task = build_demo_task()

    # Use a temporary workspace so the example is self-contained and leaves no
    # files behind once the process exits.
    with tempfile.TemporaryDirectory(prefix='prt_sbi_basic_') as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_uri = (tmp_path / 'basic_dataset.zarr').as_posix()

        # Step 1: simulate a small prior-predictive dataset. The helper returns
        # both a dataset manifest and fitted preprocessing metadata.
        generated = generate_simulation_dataset(
            task=task,
            storage_uri=dataset_uri,
            n_simulations=16,
            chunk_size=4,
            split_counts={
                DatasetSplit.TRAIN: 12,
                DatasetSplit.VALIDATION: 2,
                DatasetSplit.TEST: 2,
            },
            dataset_version='0.1.0',
        )

        store = ZarrSimulationDatasetStore(chunk_size=4)
        dataset = store.open(generated.manifest)
        reader = NormalizedObservationDatasetReader(
            dataset=dataset,
            preprocessing_metadata=generated.preprocessing_metadata,
        )

        # Step 2: train a small spline posterior in unconstrained parameter
        # space. This is the default space used by the current SBI design.
        posterior = ConditionalSplineFlowPosterior(
            parameter_dim=1,
            embedding_dim=32,
            num_coupling_layers=3,
            hidden_dim=32,
            learning_rate=1e-3,
            batch_size=4,
            num_epochs=12,
            parameter_space='unconstrained',
            seed=1,
            task_metadata={'task_name': task.name},
        )
        training_artifacts = posterior.fit(reader)

        # Step 3: take one raw observation from the held-out split. Notice that
        # we pass the raw observation blocks to AmortizedRetrieval rather than a
        # normalized version. The inference service will apply preprocessing
        # internally before encoding.
        raw_test_split = dataset.read_split(DatasetSplit.TEST)
        raw_blocks = build_observation_block_batch(
            raw_test_split['observations'],
            task.observation_schema.modalities,
        )
        retrieval = AmortizedRetrieval(task=task, posterior_estimator=posterior)
        result = retrieval.infer(
            raw_blocks[0],
            n_posterior_samples=64,
            include_posterior_predictive=True,
            posterior_predictive_interval_level=0.9,
            seed=2,
        )

        posterior_samples = np.asarray(result.posterior.samples, dtype=float)
        predictive_report = result.posterior_predictive

        print('Training epochs:', len(training_artifacts.history['train_loss']))
        print('Posterior sample shape:', posterior_samples.shape)
        print('First five posterior samples:', posterior_samples[:5, 0])
        print('Preprocessing applied internally:', result.metadata['preprocessing_applied'])
        print(
            'Observed spectrum on original scale:',
            predictive_report.observed_values['synthetic_spectrum'],
        )
        print(
            'Posterior-predictive mean spectrum:',
            predictive_report.predictive_mean['synthetic_spectrum'],
        )
        print(
            'Posterior-predictive interval coverage:',
            predictive_report.interval_coverage['synthetic_spectrum'],
        )

        # Optional sanity check: the simulator can still be used directly if you
        # want to inspect prior or posterior predictive draws yourself.
        simulator = RuntimeSimulator(task=task, seed=3)
        direct_predictive = simulator.simulate_from_parameters(jnp.array([[0.25]], dtype=jnp.float64))
        print(
            'Direct simulator output for amplitude=0.25:',
            np.asarray(direct_predictive.observations['synthetic_spectrum']['values'])[0],
        )


if __name__ == '__main__':
    main()
