"""Real-data SBI example based on the JAX runtime emission tutorial.

This script converts the notebook workflow in
``docs/content/notebooks/retrieval_jax_runtime_emission_tutorial.ipynb`` into a
plain Python program and extends it with the current ``petitRADTRANS.sbi``
module.

The script does three things:

1. load the real HR8799e GRAVITY spectrum used in the notebook;
2. build the same runtime-native emission retrieval configuration;
3. train a small amortized posterior on prior-predictive simulations and apply
   it to the real observed spectrum.

Optionally, the script can also run the exact JAXNS retrieval from the notebook
for side-by-side comparison. That path is disabled by default because it is much
more expensive than the demonstration SBI workflow.

Configuration is controlled through editable constants near the top of the
script rather than command-line arguments.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('JAX_CHECK_TRACER_LEAKS', 'False')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

multinest_lib_dir = '/Users/nasedkin/software/MultiNest/lib/'
existing_dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
if multinest_lib_dir not in existing_dyld_path.split(':'):
    os.environ['DYLD_LIBRARY_PATH'] = (
        f'{multinest_lib_dir}:{existing_dyld_path}'
        if existing_dyld_path
        else multinest_lib_dir
    )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

jax.config.update('jax_enable_x64', True)

from petitRADTRANS import physical_constants as cst  # noqa: E402
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig  # noqa: E402
from petitRADTRANS.retrieval.models import runtime_native_guillot_emission  # noqa: E402
from petitRADTRANS.sbi import (  # noqa: E402
    AmortizedRetrieval,
    ConditionalSplineFlowPosterior,
    DatasetSplit,
    NoiseModelConfig,
    NormalizedObservationDatasetReader,
    ObservationModality,
    PosteriorPredictiveReport,
    SBITask,
    SimulationConfig,
    SimulationBasedCalibrationReport,
    ZarrSimulationDatasetStore,
    build_observation_block,
    generate_posterior_predictive_report,
    generate_sbc_report,
    generate_simulation_dataset,
    plot_posterior_marginals,
    plot_posterior_predictive_report,
    plot_sbc_rank_histograms,
)


tfpd = tfp.distributions


# User-editable configuration
# Currently set to small values for demonstration.
# Commented values are production recommendations.
# N_SIMULATIONS is the most important parameter for scaling to more complex problems
N_SIMULATIONS = 48  # 100_000 (minimum), probably closer to 1_000_000 for a serious retrieval.
NUM_EPOCHS = 20  # 160
BATCH_SIZE = 8   # 256, or 512 if hardware allows
N_POSTERIOR_SAMPLES = 256  # 16_384
RUN_EXACT_RETRIEVAL = False
OUTPUT_ROOT: Path | None = None
GENERATE_PLOTS = True
PLOT_DPI = 160


def locate_input_files() -> tuple[Path, Path | None]:
    """Locate the HR8799e spectrum and optional photometry file inside the repo."""

    spectrum_candidates = [
        Path('/u/nnas/packages/petitRADTRANS/docs/content/notebooks/retrievals/emission/observations/HR8799e_Spectra.fits'),
        Path('docs/content/notebooks/retrievals/emission/observations/HR8799e_Spectra.fits'),
    ]
    photometry_candidates = [
        Path('/u/nnas/packages/petitRADTRANS/docs/content/notebooks/retrievals/emission/observations/hr8799e_photometry.txt'),
        Path('docs/content/notebooks/retrievals/emission/observations/hr8799e_photometry.txt'),
    ]

    for candidate in spectrum_candidates:
        if candidate.exists():
            spectrum_file = candidate.resolve()
            break
    else:
        raise FileNotFoundError(
            'Could not locate HR8799e_Spectra.fits. Run the script from the repo root or from docs/content/notebooks.'
        )

    photometry_file = next((path.resolve() for path in photometry_candidates if path.exists()), None)
    return spectrum_file, photometry_file


def build_retrieval_config(example_data_file: Path) -> tuple[RetrievalConfig, jnp.ndarray]:
    """Create the runtime-native HR8799e emission configuration from the notebook."""

    pressures = jnp.logspace(-6.0, 2.0, 100)
    retrieval_config = RetrievalConfig(
        retrieval_name='hr8799e_runtime_native_emission_jaxns',
        run_mode='retrieval',
        sampler_type='jaxns',
        adaptive_mesh_refinement=False,  # type: ignore[arg-type]
        scattering_in_emission=False,
        pressures=pressures,
    )

    retrieval_config.add_parameter(
        name='distance_to_system',
        is_free_parameter=False,
        value=41.2925 * cst.pc,
        plot_in_corner=False,
    )
    retrieval_config.add_parameter(
        name='log_g',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=3.0, high=5.0),
        plot_in_corner=True,
        corner_ranges=(3.0, 5.0),
        corner_label=r'$\log(g)$',
    )
    retrieval_config.add_parameter(
        name='planet_radius',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.8 * cst.r_jup, high=1.6 * cst.r_jup),
        plot_in_corner=True,
        corner_ranges=(0.8, 1.6),
        corner_label=r'R$_{pl}$',
        corner_transform=lambda radius: radius / cst.r_jup_mean,
    )
    retrieval_config.add_parameter(
        name='T_int',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=200.0, high=2000.0),
        plot_in_corner=True,
        corner_ranges=(200.0, 2000.0),
        corner_label=r'$T_{int}$',
    )
    retrieval_config.add_parameter(
        name='T_equ',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=600.0, high=2200.0),
        plot_in_corner=True,
        corner_ranges=(600.0, 2200.0),
        corner_label=r'$T_{eq}$',
    )
    retrieval_config.add_parameter(
        name='gamma',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.1, high=2.0),
        plot_in_corner=True,
        corner_ranges=(0.1, 2.0),
        corner_label=r'$\gamma$',
    )
    retrieval_config.add_parameter(
        name='log_kappa_IR',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-5.0, high=0.0),
        plot_in_corner=True,
        corner_ranges=(-5.0, 0.0),
        corner_label=r'$\log \kappa_{IR}$',
    )
    retrieval_config.add_parameter(
        name='Fe/H',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-1.0, high=1.5),
        plot_in_corner=True,
        corner_ranges=(-1.0, 1.5),
        corner_label=r'[Fe/H]',
    )
    retrieval_config.add_parameter(
        name='C/O',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.2, high=1.2),
        plot_in_corner=True,
        corner_ranges=(0.2, 1.2),
        corner_label=r'C/O',
    )

    retrieval_config.set_rayleigh_species(['H2', 'He'])
    retrieval_config.set_continuum_opacities(['H2--H2', 'H2--He'])
    retrieval_config.set_line_species(
        [
            '1H2-16O__POKAZATEL',
            'C-O-NatAbund__HITEMP',
        ],
        use_equilibrium_chemistry=True,
    )
    retrieval_config.add_data(
        name='GRAVITY',
        path_to_observations=str(example_data_file),
        model_generating_function=runtime_native_guillot_emission,
        data_resolution=500,
        model_resolution=300,
        wavelength_boundaries=(1.0, 2.5),
        line_opacity_mode='c-k',
    )
    return retrieval_config, pressures


def configure_retrieval_plotting(retrieval: Retrieval) -> None:
    """Apply plotting settings through the retrieval plotter companion."""

    retrieval.plotter.spec_xlabel = 'Wavelength [micron]'
    retrieval.plotter.spec_ylabel = 'Flux [W/m2/micron]'
    retrieval.plotter.y_axis_scaling = 1.0
    retrieval.plotter.xscale = 'linear'
    retrieval.plotter.yscale = 'linear'
    retrieval.plotter.nsample = 10
    retrieval.plotter.reference_data_name = 'GRAVITY'
    retrieval.plotter.temp_limits = [150, 3500]
    retrieval.plotter.press_limits = [1e2, 1e-6]


def build_sbi_task(retrieval_config: RetrievalConfig) -> SBITask:
    """Wrap the retrieval configuration in the current SBI task abstraction."""

    return SBITask.from_retrieval_config(
        retrieval_config,
        simulation_config=SimulationConfig(
            batch_size=8,
            noise_model=NoiseModelConfig(mode='observational', seed=123),
        ),
    )


def build_raw_observation_blocks(retrieval_config: RetrievalConfig) -> list:
    """Build raw user-facing observation blocks from the loaded real dataset."""

    gravity_data = retrieval_config.data['GRAVITY']
    return [
        build_observation_block(
            name='GRAVITY',
            modality=ObservationModality.SPECTRUM,
            values=gravity_data.spectrum,
            uncertainties=gravity_data.uncertainties,
            coordinates=gravity_data.wavelengths,
            mask=gravity_data.mask,
            metadata={
                'line_opacity_mode': gravity_data.line_opacity_mode,
                'photometry': gravity_data.photometry,
                'source_name': gravity_data.name,
            },
        )
    ]


def maybe_run_exact_retrieval(
    retrieval_config: RetrievalConfig,
    output_root: Path,
    run_exact: bool,
) -> None:
    """Optionally launch the exact JAXNS retrieval from the original notebook."""

    runtime = SBITask.from_retrieval_config(retrieval_config).build_runtime()
    print('Runtime model groups:')
    for group in runtime.model_groups:
        print(
            {
                'source_name': group.source_name,
                'model_contract': group.model_contract,
                'observation_names': group.observation_names,
                'model_function': group.model_generating_function.__name__,
            }
        )

    if not run_exact:
        print('Skipping exact JAXNS retrieval. Use --run-exact to enable it.')
        return

    retrieval = Retrieval(
        configuration=retrieval_config,
        output_directory=str(output_root),
        use_mpi=False,
        evaluate_sample_spectra=False,
        use_prt_plot_style=True,
    )
    configure_retrieval_plotting(retrieval)

    retrieval.run(
        num_live_points=80,
        max_samples=60_000,
        s=4,
        k=0,
        c=200,
        shell_fraction=0.5,
        gradient_guided=False,
        init_efficiency_threshold=0.05,
        verbose=True,
        jit=True,
        seed=12345,
    )

    free_parameter_names = [
        name
        for name, parameter in retrieval.configuration.parameters.items()
        if parameter.is_free_parameter
    ]
    assert retrieval.sampler is not None
    summary = retrieval.sampler.get_summary(free_parameter_names=free_parameter_names)
    print('Exact retrieval evidence:', summary.get('evidence', {}))
    print('Exact retrieval diagnostics:', summary.get('diagnostics', {}))


def maybe_save_plots(
    output_directory: Path,
    posterior_samples: np.ndarray,
    parameter_names: list[str],
    sbc_report: SimulationBasedCalibrationReport,
    posterior_predictive_report: PosteriorPredictiveReport,
) -> None:
    """Save a standard set of SBI diagnostic plots when matplotlib is available."""

    if not GENERATE_PLOTS:
        print('Plot generation disabled by configuration.')
        return

    try:
        posterior_figure, _ = plot_posterior_marginals(
            posterior_samples,
            parameter_names=parameter_names,
        )
        posterior_figure.savefig(output_directory / 'posterior_marginals.png', dpi=PLOT_DPI, bbox_inches='tight')

        sbc_figure, _ = plot_sbc_rank_histograms(
            sbc_report,
            parameter_names=parameter_names,
        )
        sbc_figure.savefig(output_directory / 'sbc_rank_histograms.png', dpi=PLOT_DPI, bbox_inches='tight')

        predictive_figure, _ = plot_posterior_predictive_report(
            posterior_predictive_report,
            dataset_names=['GRAVITY'],
        )
        predictive_figure.savefig(
            output_directory / 'gravity_posterior_predictive.png',
            dpi=PLOT_DPI,
            bbox_inches='tight',
        )
        print('Saved SBI plots to:', output_directory)
    except ImportError as error:
        print(f'Skipping plot generation because matplotlib is unavailable: {error}')


def main() -> None:
    print(f'JAX devices: {jax.devices()}')
    example_data_file, photometry_data_file = locate_input_files()
    print(f'Using emission spectrum file: {example_data_file}')
    print(f'Photometry file available: {photometry_data_file is not None}')

    if OUTPUT_ROOT is None:
        output_root = example_data_file.parents[4] / 'runs' / 'retrieval_jax_runtime_emission'
    else:
        output_root = OUTPUT_ROOT.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_root}')

    retrieval_config, pressures = build_retrieval_config(example_data_file)
    del pressures
    gravity_data = retrieval_config.data['GRAVITY']
    print(f'Model function: {gravity_data.model_generating_function.__name__}')
    print(f'Loaded points: {gravity_data.wavelengths.shape[0]}')
    print(
        f'Wavelength range: {float(gravity_data.wavelengths.min()):.3f}--'
        f'{float(gravity_data.wavelengths.max()):.3f} micron'
    )

    maybe_run_exact_retrieval(retrieval_config, output_root, run_exact=RUN_EXACT_RETRIEVAL)

    # The SBI path below uses the exact same real observed spectrum for the
    # inference target, while training on simulated spectra generated from the
    # runtime-native forward model and prior ranges above.
    task = build_sbi_task(retrieval_config)
    sbi_output_root = output_root / 'sbi'
    sbi_output_root.mkdir(parents=True, exist_ok=True)
    dataset_uri = (sbi_output_root / 'simulation_dataset.zarr').as_posix()
    checkpoint_dir = sbi_output_root / 'checkpoints'
    posterior_dir = sbi_output_root / 'posterior_model'

    split_counts: dict[str, int] = {
        DatasetSplit.TRAIN.value: max(N_SIMULATIONS - 8, 8),
        DatasetSplit.VALIDATION.value: 4,
        DatasetSplit.TEST.value: 4,
    }
    total_requested = sum(split_counts.values())
    if total_requested != N_SIMULATIONS:
        split_counts[DatasetSplit.TRAIN.value] += N_SIMULATIONS - total_requested

    print('Generating simulation dataset...')
    generated = generate_simulation_dataset(
        task=task,
        storage_uri=dataset_uri,
        n_simulations=N_SIMULATIONS,
        chunk_size=BATCH_SIZE,
        split_counts=split_counts,
        dataset_version='0.1.0',
    )
    print('Simulation dataset splits:', generated.manifest.splits)

    store = ZarrSimulationDatasetStore(chunk_size=BATCH_SIZE)
    dataset = store.open(generated.manifest)
    assert generated.preprocessing_metadata is not None
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=generated.preprocessing_metadata,
    )

    print('Training conditional spline posterior...')
    posterior = ConditionalSplineFlowPosterior(
        parameter_dim=len(task.parameter_names),
        embedding_dim=64,  # 256
        num_coupling_layers=4,  # 12
        hidden_dim=64,  # 512
        learning_rate=1e-3,  # 2e-4
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        parameter_space='unconstrained',
        checkpoint_directory=checkpoint_dir.as_posix(),
        checkpoint_backend='equinox',
        seed=123,
        task_metadata={'task_name': task.name},
    )
    training_artifacts = posterior.fit(reader)
    posterior.save(posterior_dir.as_posix())
    print('Finished training epochs:', len(training_artifacts.history['train_loss']))
    print('Posterior saved to:', posterior_dir)

    print('Computing calibration reports on held-out simulations...')
    sbc_report = generate_sbc_report(
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(N_POSTERIOR_SAMPLES, 2_048),
        batch_size=BATCH_SIZE,
        seed=321,
    )
    predictive_report = generate_posterior_predictive_report(
        task,
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(N_POSTERIOR_SAMPLES, 4_096),
        interval_level=0.9,
        seed=654,
    )
    print('SBC mean rank:', sbc_report.mean_rank)
    print('SBC normalized mean-rank error:', sbc_report.normalized_mean_rank_error)
    print(
        'Held-out predictive datasets:',
        sorted(predictive_report.observed_values.keys()),
    )

    print('Running amortized inference on the real GRAVITY observation...')
    raw_observation_blocks = build_raw_observation_blocks(retrieval_config)
    amortized_retrieval = AmortizedRetrieval(task=task, posterior_estimator=posterior)
    result = amortized_retrieval.infer(
        raw_observation_blocks,
        n_posterior_samples=N_POSTERIOR_SAMPLES,
        include_posterior_predictive=True,
        posterior_predictive_interval_level=0.9,
        seed=777,
    )

    posterior_samples = np.asarray(result.posterior.samples, dtype=float)
    print('Posterior sample matrix shape:', posterior_samples.shape)
    print('First posterior draw:', posterior_samples[0])
    print('Inference preprocessing applied:', result.metadata['preprocessing_applied'])
    print(
        'Observed GRAVITY flux on original scale:',
        result.posterior_predictive.observed_values['GRAVITY'][:5],
        '...',
    )
    print(
        'Posterior-predictive mean flux:',
        result.posterior_predictive.predictive_mean['GRAVITY'][:5],
        '...',
    )

    maybe_save_plots(
        output_directory=sbi_output_root,
        posterior_samples=posterior_samples,
        parameter_names=list(task.parameter_names),
        sbc_report=sbc_report,
        posterior_predictive_report=result.posterior_predictive,
    )

    summary_path = sbi_output_root / 'amortized_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'task_name': task.name,
                'parameter_names': list(task.parameter_names),
                'n_simulations': N_SIMULATIONS,
                'num_epochs': NUM_EPOCHS,
                'n_posterior_samples': N_POSTERIOR_SAMPLES,
                'run_exact_retrieval': RUN_EXACT_RETRIEVAL,
                'generate_plots': GENERATE_PLOTS,
                'preprocessing_applied': result.metadata['preprocessing_applied'],
                'preprocessing_version': result.metadata['preprocessing_version'],
                'sbc_mean_rank': np.asarray(sbc_report.mean_rank, dtype=float).tolist(),
                'sbc_normalized_mean_rank_error': (
                    np.asarray(sbc_report.normalized_mean_rank_error, dtype=float).tolist()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print('Wrote amortized summary to:', summary_path)


if __name__ == '__main__':
    main()
