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

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
from petitRADTRANS.retrieval.models import guillot_emission  # noqa: E402
from petitRADTRANS.sbi import (  # noqa: E402
    AmortizedRetrieval,
    ConditionalFlowPosterior,
    ConditionalSplineFlowPosterior,
    DatasetSplit,
    build_observation_block_batch,
    fit_task_preprocessing,
    NoiseModelConfig,
    NormalizedObservationDatasetReader,
    ObservationModality,
    PosteriorPredictiveReport,
    SBITask,
    SimulationConfig,
    SimulationBasedCalibrationReport,
    TaskPreprocessingMetadata,
    HDF5SimulationDatasetStore,
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
N_SIMULATIONS = 2_000_000  # 100_000 (minimum), probably closer to 1_000_000 for a serious retrieval.
NUM_EPOCHS = 1024
BATCH_SIZE = 512   # 256, or 512 if hardware allows
N_POSTERIOR_SAMPLES = 512  # 16_384
EARLY_STOPPING_PATIENCE = 64
# Maximum number of posterior draws forwarded through the pRT forward model per
# posterior-predictive check.
N_PREDICTIVE_FORWARD_MODEL_SAMPLES = 256  # None to use all N_POSTERIOR_SAMPLES
# When True, skip simulation and training and load the previously saved posterior
RESUME_FROM_SAVED_POSTERIOR = False
RUN_EXACT_RETRIEVAL = False
OUTPUT_ROOT: Path | None = None
GENERATE_PLOTS = True
PLOT_DPI = 160
VERBOSE_TRAINING_DIAGNOSTICS = True
TRAINING_DIAGNOSTICS_PLOT_INTERVAL = 1
SPLINE_BOUND = 18.0
LEARNING_RATE = 3e-5
GRADIENT_CLIP_NORM = 0.5
EMBEDDING_NOISE_STD = 0.05
AUX_SCALE_LOSS_WEIGHT = 0.02
SPLINE_SMALL_BIN_REGULARIZATION_WEIGHT = 0.05
SPLINE_MIN_BIN_RATIO_TARGET = 0.2
SPLINE_DERIVATIVE_REGULARIZATION_WEIGHT = 0.05
SPLINE_MIN_DERIVATIVE_TARGET = 0.25
SPLINE_MAX_DERIVATIVE_TARGET = 5.0
RUN_LABEL: str | None = None
RESUME_RUN_LABEL: str | None = None


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


def refit_preprocessing_metadata(
    task: SBITask,
    dataset: Any,
    sidecar_path: Path,
) -> TaskPreprocessingMetadata:
    """Rebuild preprocessing metadata from an existing simulation dataset."""

    training_split = dataset.read_split(DatasetSplit.TRAIN)
    training_blocks = build_observation_block_batch(
        training_split['observations'],
        task.observation_schema.modalities,
    )
    preprocessing_metadata = fit_task_preprocessing(
        training_blocks,
        version=task.preprocessing_version,
        metadata={
            'task_name': task.name,
            'task_fingerprint': task.task_fingerprint.digest,
            'split_policy': dataset.manifest.metadata.get('split_policy', 'shuffled'),
        },
    )
    sidecar_path.write_text(
        json.dumps(preprocessing_metadata.to_payload(), sort_keys=True, indent=2)
    )
    return preprocessing_metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def build_run_config(task: SBITask, n_wavelength_points: int) -> dict[str, Any]:
    return {
        'task_name': task.name,
        'task_fingerprint': task.task_fingerprint.digest,
        'preprocessing_version': task.preprocessing_version,
        'n_simulations': N_SIMULATIONS,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'n_posterior_samples': N_POSTERIOR_SAMPLES,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'n_predictive_forward_model_samples': N_PREDICTIVE_FORWARD_MODEL_SAMPLES,
        'parameter_space': 'cube',
        'flow_family': 'spline',
        'num_coupling_layers': 6,
        'num_spline_bins': 8,
        'spline_bound': SPLINE_BOUND,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'conditioner_depth': 2,
        'learning_rate': LEARNING_RATE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'embedding_noise_std': EMBEDDING_NOISE_STD,
        'aux_scale_loss_weight': AUX_SCALE_LOSS_WEIGHT,
        'spline_small_bin_regularization_weight': SPLINE_SMALL_BIN_REGULARIZATION_WEIGHT,
        'spline_min_bin_ratio_target': SPLINE_MIN_BIN_RATIO_TARGET,
        'spline_derivative_regularization_weight': SPLINE_DERIVATIVE_REGULARIZATION_WEIGHT,
        'spline_min_derivative_target': SPLINE_MIN_DERIVATIVE_TARGET,
        'spline_max_derivative_target': SPLINE_MAX_DERIVATIVE_TARGET,
        'weight_decay': 1e-3,
        'use_cosine_schedule': True,
        'warmup_fraction': 0.1,
        'min_learning_rate': 1e-6,
        'spectrum_encoder_type': 'conv1d',
        'n_wavelengths': n_wavelength_points,
        'spectrum_embedding_dim': 64,
        'photometry_embedding_dim': 64,
        'encoder_hidden_dim': 128,
        'encoder_patch_size': 32,
        'seed': 123,
        'run_exact_retrieval': RUN_EXACT_RETRIEVAL,
        'generate_plots': GENERATE_PLOTS,
        'verbose_training_diagnostics': VERBOSE_TRAINING_DIAGNOSTICS,
        'training_diagnostics_plot_interval': TRAINING_DIAGNOSTICS_PLOT_INTERVAL,
    }


def build_posterior_estimator(
    task: SBITask,
    run_config: dict[str, Any],
    *,
    checkpoint_dir: Path,
    run_root: Path,
) -> ConditionalFlowPosterior:
    return ConditionalFlowPosterior(
        parameter_dim=len(task.parameter_names),
        embedding_dim=int(run_config.get('embedding_dim', 128)),
        num_coupling_layers=int(run_config.get('num_coupling_layers', 6)),
        hidden_dim=int(run_config.get('hidden_dim', 256)),
        conditioner_depth=int(run_config.get('conditioner_depth', 2)),
        learning_rate=float(run_config.get('learning_rate', LEARNING_RATE)),
        batch_size=int(run_config.get('batch_size', BATCH_SIZE)),
        num_epochs=int(run_config.get('num_epochs', NUM_EPOCHS)),
        parameter_space=str(run_config.get('parameter_space', 'cube')),
        flow_family=str(run_config.get('flow_family', 'spline')),
        num_spline_bins=int(run_config.get('num_spline_bins', 8)),
        spline_bound=float(run_config.get('spline_bound', SPLINE_BOUND)),
        early_stopping_patience=run_config.get(
            'early_stopping_patience',
            EARLY_STOPPING_PATIENCE,
        ),
        checkpoint_directory=checkpoint_dir.as_posix(),
        checkpoint_backend='equinox',
        resume_from_checkpoint=False,
        gradient_clip_norm=run_config.get('gradient_clip_norm', GRADIENT_CLIP_NORM),
        embedding_noise_std=float(
            run_config.get('embedding_noise_std', EMBEDDING_NOISE_STD)
        ),
        aux_scale_loss_weight=float(
            run_config.get('aux_scale_loss_weight', AUX_SCALE_LOSS_WEIGHT)
        ),
        spline_small_bin_regularization_weight=float(
            run_config.get(
                'spline_small_bin_regularization_weight',
                SPLINE_SMALL_BIN_REGULARIZATION_WEIGHT,
            )
        ),
        spline_min_bin_ratio_target=float(
            run_config.get('spline_min_bin_ratio_target', SPLINE_MIN_BIN_RATIO_TARGET)
        ),
        spline_derivative_regularization_weight=float(
            run_config.get(
                'spline_derivative_regularization_weight',
                SPLINE_DERIVATIVE_REGULARIZATION_WEIGHT,
            )
        ),
        spline_min_derivative_target=float(
            run_config.get(
                'spline_min_derivative_target',
                SPLINE_MIN_DERIVATIVE_TARGET,
            )
        ),
        spline_max_derivative_target=float(
            run_config.get(
                'spline_max_derivative_target',
                SPLINE_MAX_DERIVATIVE_TARGET,
            )
        ),
        weight_decay=float(run_config.get('weight_decay', 1e-3)),
        use_cosine_schedule=bool(run_config.get('use_cosine_schedule', True)),
        warmup_fraction=float(run_config.get('warmup_fraction', 0.1)),
        min_learning_rate=float(run_config.get('min_learning_rate', 1e-6)),
        spectrum_encoder_type=str(
            run_config.get('spectrum_encoder_type', 'conv1d')
        ),
        n_wavelengths=int(run_config.get('n_wavelengths', 233)),
        spectrum_embedding_dim=int(run_config.get('spectrum_embedding_dim', 64)),
        photometry_embedding_dim=int(run_config.get('photometry_embedding_dim', 64)),
        encoder_hidden_dim=int(run_config.get('encoder_hidden_dim', 128)),
        seed=int(run_config.get('seed', 123)),
        task_metadata={'task_name': task.name},
        verbose_diagnostics=bool(
            run_config.get(
                'verbose_training_diagnostics',
                VERBOSE_TRAINING_DIAGNOSTICS,
            )
        ),
        diagnostics_output_directory=run_root.as_posix(),
        diagnostics_plot_interval=int(
            run_config.get(
                'training_diagnostics_plot_interval',
                TRAINING_DIAGNOSTICS_PLOT_INTERVAL,
            )
        ),
    )


def compute_run_config_hash(run_config: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(run_config), sort_keys=True).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:12]


def build_run_label(run_config: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{timestamp}-{compute_run_config_hash(run_config)}'


def latest_run_pointer_path(sbi_output_root: Path) -> Path:
    return sbi_output_root / 'latest_run.json'


def resolve_saved_run_root(sbi_output_root: Path, run_label: str | None) -> Path:
    if run_label is not None:
        run_root = sbi_output_root / 'runs' / run_label
    else:
        pointer_path = latest_run_pointer_path(sbi_output_root)
        if not pointer_path.exists():
            raise FileNotFoundError(
                'No latest_run.json found; set RESUME_RUN_LABEL to load a specific saved posterior run.'
            )
        pointer_payload = json.loads(pointer_path.read_text())
        latest_label = str(pointer_payload.get('run_label', ''))
        if not latest_label:
            raise ValueError(f'Invalid latest-run pointer in {pointer_path}')
        run_root = sbi_output_root / 'runs' / latest_label

    if not run_root.exists():
        raise FileNotFoundError(f'Saved posterior run directory not found: {run_root}')

    return run_root


def load_run_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / 'run_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing run manifest: {manifest_path}')

    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid run manifest payload in {manifest_path}')
    return payload


def validate_saved_posterior_run(
    run_root: Path,
    expected_config_hash: str,
    task: SBITask,
) -> dict[str, Any]:
    manifest = load_run_manifest(run_root)
    saved_hash = str(manifest.get('config_hash', ''))
    if saved_hash != expected_config_hash:
        raise ValueError(
            'Saved posterior run configuration does not match the current script configuration. '
            f'Expected config hash {expected_config_hash}, found {saved_hash}. '
            f'Run root: {run_root}'
        )

    saved_task_name = str(manifest.get('task_name', ''))
    if saved_task_name != task.name:
        raise ValueError(
            f'Saved posterior task name {saved_task_name!r} does not match current task {task.name!r}'
        )

    saved_fingerprint = str(manifest.get('task_fingerprint', ''))
    if saved_fingerprint != task.task_fingerprint.digest:
        raise ValueError(
            'Saved posterior task fingerprint does not match the current task. '
            f'Expected {task.task_fingerprint.digest}, found {saved_fingerprint}. '
            f'Run root: {run_root}'
        )

    return manifest


def write_run_manifest(
    run_root: Path,
    run_label: str,
    run_config: dict[str, Any],
    task: SBITask,
    dataset_uri: str,
    posterior_dir: Path,
    summary_path: Path | None,
    status: str,
) -> None:
    manifest = {
        'run_label': run_label,
        'status': status,
        'config_hash': compute_run_config_hash(run_config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_name': task.name,
        'task_fingerprint': task.task_fingerprint.digest,
        'dataset_uri': dataset_uri,
        'posterior_dir': posterior_dir.as_posix(),
        'summary_path': None if summary_path is None else summary_path.as_posix(),
        'run_config': run_config,
    }

    metadata_path = posterior_dir / 'metadata.json'
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text())
        manifest['posterior_metadata_excerpt'] = {
            'parameter_space': metadata_payload.get('parameter_space'),
            'spline_bound': metadata_payload.get('spline_bound'),
            'effective_flow_family': metadata_payload.get('effective_flow_family'),
            'training_metadata': metadata_payload.get('training_metadata'),
        }

    write_json(run_root / 'run_manifest.json', manifest)


def update_latest_run_pointer(sbi_output_root: Path, run_root: Path, run_label: str) -> None:
    write_json(
        latest_run_pointer_path(sbi_output_root),
        {
            'run_label': run_label,
            'run_root': run_root.as_posix(),
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        },
    )


def build_retrieval_config(example_data_file: Path) -> tuple[RetrievalConfig, jnp.ndarray]:
    """Create the runtime-native HR8799e emission configuration from the notebook."""

    pressures = jnp.logspace(-6.0, 2.0, 100)
    retrieval_config = RetrievalConfig(
        retrieval_name='hr8799e_gravity_sbi_v3',
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
        model_generating_function=guillot_emission,
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
            batch_size=64,
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


def posterior_samples_to_physical(
    task: SBITask,
    posterior_samples: np.ndarray,
    parameter_space: str,
) -> np.ndarray:
    """Convert posterior samples into physical parameter values for reporting."""

    sample_array = np.asarray(posterior_samples, dtype=float)

    if parameter_space == 'cube':
        return np.asarray(
            task.parameter_layout.transform_cube_to_physical_batch(sample_array),
            dtype=float,
        )
    if parameter_space == 'unconstrained':
        return np.asarray(
            task.parameter_layout.transform_unconstrained_batch(sample_array),
            dtype=float,
        )

    return sample_array


def maybe_save_plots(
    output_directory: Path,
    task: SBITask,
    posterior_samples: np.ndarray,
    parameter_space: str,
    parameter_names: list[str],
    sbc_report: SimulationBasedCalibrationReport,
    posterior_predictive_report: PosteriorPredictiveReport,
) -> None:
    """Save a standard set of SBI diagnostic plots when matplotlib is available."""

    if not GENERATE_PLOTS:
        print('Plot generation disabled by configuration.')
        return

    try:
        physical_samples = posterior_samples_to_physical(
            task,
            posterior_samples,
            parameter_space=parameter_space,
        )
        posterior_figure, _ = plot_posterior_marginals(
            physical_samples,
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
        output_root = example_data_file.parents[4] / 'runs' / 'hr8799e_gravity_sbi_v3'
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
    dataset_uri = (sbi_output_root / 'simulation_dataset.h5').as_posix()
    n_wavelength_points = int(gravity_data.wavelengths.shape[0])
    run_config = build_run_config(task, n_wavelength_points)
    run_label = RUN_LABEL or build_run_label(run_config)
    run_root = sbi_output_root / 'runs' / run_label
    posterior_dir = run_root / 'posterior_model'
    checkpoint_dir = run_root / 'checkpoints'
    predictive_checkpoint_dir = run_root / 'predictive_checkpoints'
    summary_path = run_root / 'amortized_summary.json'
    reader: NormalizedObservationDatasetReader | None = None

    if RESUME_FROM_SAVED_POSTERIOR:
        run_root = resolve_saved_run_root(sbi_output_root, RESUME_RUN_LABEL)
        run_label = run_root.name
        posterior_dir = run_root / 'posterior_model'
        checkpoint_dir = run_root / 'checkpoints'
        predictive_checkpoint_dir = run_root / 'predictive_checkpoints'
        summary_path = run_root / 'amortized_summary.json'
        validate_saved_posterior_run(
            run_root,
            expected_config_hash=compute_run_config_hash(run_config),
            task=task,
        )
    else:
        if run_root.exists():
            raise FileExistsError(
                f'Run artifact directory already exists: {run_root}. '
                'Set RUN_LABEL explicitly or remove the existing directory before rerunning.'
            )
        run_root.mkdir(parents=True, exist_ok=False)

    print(f'Run artifacts directory: {run_root}')

    if RESUME_FROM_SAVED_POSTERIOR:
        print('Resuming: loading saved posterior and reopening existing dataset...')
        store = HDF5SimulationDatasetStore(chunk_size=BATCH_SIZE)
        dataset = store.open(dataset_uri)
        posterior = ConditionalFlowPosterior.load(posterior_dir.as_posix())
        preprocessing_metadata = TaskPreprocessingMetadata.from_payload(
            posterior.preprocessing_metadata_payload
        )
        reader = NormalizedObservationDatasetReader(
            dataset=dataset,
            preprocessing_metadata=preprocessing_metadata,
        )
        print('Loaded posterior parameter_dim:', posterior.parameter_dim)
        print('Loaded posterior parameter_space:', posterior.parameter_space)
    else:
        write_run_manifest(
            run_root=run_root,
            run_label=run_label,
            run_config=run_config,
            task=task,
            dataset_uri=dataset_uri,
            posterior_dir=posterior_dir,
            summary_path=None,
            status='training',
        )
        dataset_path = Path(dataset_uri)
        preprocessing_sidecar = dataset_path.with_suffix('.preprocessing.json')
        preprocessing_payload: dict[str, object] | None = None
        reuse_existing_dataset = False

        if dataset_path.exists():
            store = HDF5SimulationDatasetStore(chunk_size=BATCH_SIZE)
            dataset = store.open(dataset_uri)
            expected_task_fingerprint = task.task_fingerprint.digest
            cached_dataset_fingerprint = str(
                getattr(dataset.manifest, 'task_fingerprint', '') or ''
            )
            if (
                cached_dataset_fingerprint
                and cached_dataset_fingerprint != expected_task_fingerprint
            ):
                raise RuntimeError(
                    'Existing simulation dataset fingerprint does not match the current task. '
                    f'{cached_dataset_fingerprint} in {dataset_path}. '
                    f'Expected {expected_task_fingerprint}, found '
                    'Remove the stale dataset and rerun to regenerate simulations.'
                )

            if preprocessing_sidecar.exists():
                payload = json.loads(preprocessing_sidecar.read_text())
                if not isinstance(payload, dict):
                    raise ValueError(
                        f'Invalid preprocessing sidecar payload in {preprocessing_sidecar}'
                    )
                preprocessing_payload = payload
                cached_version = str(preprocessing_payload.get('version', '0.1.0'))
                sidecar_metadata = preprocessing_payload.get('metadata', {})
                if not isinstance(sidecar_metadata, dict):
                    sidecar_metadata = {}
                cached_sidecar_fingerprint = str(
                    sidecar_metadata.get('task_fingerprint', '') or ''
                )
                reuse_existing_dataset = (
                    cached_version == task.preprocessing_version
                    and (
                        not cached_sidecar_fingerprint
                        or cached_sidecar_fingerprint == expected_task_fingerprint
                    )
                )
            else:
                cached_version = None
                cached_sidecar_fingerprint = ''

            if reuse_existing_dataset and preprocessing_payload is not None:
                print('Found existing dataset and compatible preprocessing sidecar, skipping generation.')
                preprocessing_metadata = TaskPreprocessingMetadata.from_payload(preprocessing_payload)
                reader = NormalizedObservationDatasetReader(
                    dataset=dataset,
                    preprocessing_metadata=preprocessing_metadata,
                )
            else:
                print(
                    'Found existing dataset but missing or stale preprocessing sidecar '
                    f"(cached_version={cached_version}, "
                    f"expected_version={task.preprocessing_version}, "
                    f"cached_task_fingerprint={cached_sidecar_fingerprint or 'missing'}, "
                    f"expected_task_fingerprint={expected_task_fingerprint}); "
                    'recomputing preprocessing metadata without regenerating simulations.'
                )
                preprocessing_metadata = refit_preprocessing_metadata(
                    task,
                    dataset,
                    preprocessing_sidecar,
                )
                reader = NormalizedObservationDatasetReader(
                    dataset=dataset,
                    preprocessing_metadata=preprocessing_metadata,
                )
                reuse_existing_dataset = True

        if not reuse_existing_dataset:
            n_test = max(4, min(80_000, N_SIMULATIONS // 50))   # ~2 %, capped at 2 000
            n_val  = max(4, min(40_000, N_SIMULATIONS // 100))  # ~1 %, capped at 2 000
            split_counts: dict[str, int] = {
                DatasetSplit.TRAIN.value: N_SIMULATIONS - n_test - n_val,
                DatasetSplit.VALIDATION.value: n_val,
                DatasetSplit.TEST.value: n_test,
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
                resume=True,
                backend="hdf5",
                store_covariance=False,
            )
            print('Simulation dataset splits:', generated.manifest.splits)

            store = HDF5SimulationDatasetStore(chunk_size=BATCH_SIZE)
            dataset = store.open(generated.manifest)
            assert generated.preprocessing_metadata is not None
            reader = NormalizedObservationDatasetReader(
                dataset=dataset,
                preprocessing_metadata=generated.preprocessing_metadata,
            )

        flow_family = str(run_config.get('flow_family', 'spline'))
        print(f'Training conditional {flow_family} posterior...')
        posterior = build_posterior_estimator(
            task,
            run_config,
            checkpoint_dir=checkpoint_dir,
            run_root=run_root,
        )
        training_artifacts = posterior.fit(reader)
        posterior.save(posterior_dir.as_posix())
        print('Finished training epochs:', len(training_artifacts.history['train_loss']))
        print('Posterior saved to:', posterior_dir)
        write_run_manifest(
            run_root=run_root,
            run_label=run_label,
            run_config=run_config,
            task=task,
            dataset_uri=dataset_uri,
            posterior_dir=posterior_dir,
            summary_path=None,
            status='trained',
        )

    assert reader is not None

    print('Computing calibration reports on held-out simulations...')
    sbc_report = generate_sbc_report(
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(N_POSTERIOR_SAMPLES, 2048),
        batch_size=BATCH_SIZE,
        max_cases=256,
        seed=321,
    )
    predictive_report = generate_posterior_predictive_report(
        task,
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(N_POSTERIOR_SAMPLES, 2048),
        interval_level=0.9,
        max_cases=256,
        seed=654,
        n_predictive_forward_model_samples=N_PREDICTIVE_FORWARD_MODEL_SAMPLES,
        checkpoint_directory=predictive_checkpoint_dir.as_posix(),
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
        n_predictive_forward_model_samples=N_PREDICTIVE_FORWARD_MODEL_SAMPLES,
        seed=777,
    )

    if result.metadata.get('parameter_space') != 'physical':
        raise RuntimeError(
            'AmortizedRetrieval is expected to return posterior samples in physical space. '
            f"Received parameter_space={result.metadata.get('parameter_space')!r}."
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
        output_directory=run_root,
        task=task,
        posterior_samples=posterior_samples,
        parameter_space='physical',
        parameter_names=list(task.parameter_names),
        sbc_report=sbc_report,
        posterior_predictive_report=result.posterior_predictive,
    )

    summary_path.write_text(
        json.dumps(
            {
                'run_label': run_label,
                'run_root': run_root.as_posix(),
                'config_hash': compute_run_config_hash(run_config),
                'task_name': task.name,
                'task_fingerprint': task.task_fingerprint.digest,
                'parameter_names': list(task.parameter_names),
                'n_simulations': N_SIMULATIONS,
                'num_epochs': NUM_EPOCHS,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
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
    write_run_manifest(
        run_root=run_root,
        run_label=run_label,
        run_config=run_config,
        task=task,
        dataset_uri=dataset_uri,
        posterior_dir=posterior_dir,
        summary_path=summary_path,
        status='completed',
    )
    update_latest_run_pointer(sbi_output_root, run_root, run_label)
    print('Wrote amortized summary to:', summary_path)


if __name__ == '__main__':
    main()
