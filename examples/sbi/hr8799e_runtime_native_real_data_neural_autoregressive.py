"""Neural-autoregressive variant of the HR8799e runtime-native SBI example.

This variant reuses the runtime-native workflow from
``hr8799e_runtime_native_real_data.py`` but swaps the spline posterior for a
more expressive conditional neural autoregressive flow. The backend uses a
monotonic deep-sigmoidal-style scalar transform per parameter dimension, which
is more flexible than the affine autoregressive flow while remaining exactly
invertible for sampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hr8799e_runtime_native_real_data as base


# User-editable configuration for the neural-autoregressive variant.
N_SIMULATIONS = 2_000_000
BATCH_SIZE = 1024
N_POSTERIOR_SAMPLES = 512
EARLY_STOPPING_PATIENCE = 64
N_PREDICTIVE_FORWARD_MODEL_SAMPLES = 256
RESUME_FROM_SAVED_POSTERIOR = False
RUN_EXACT_RETRIEVAL = False
OUTPUT_ROOT: Path | None = None
GENERATE_PLOTS = True
PLOT_DPI = 160
VERBOSE_TRAINING_DIAGNOSTICS = True
TRAINING_DIAGNOSTICS_PLOT_INTERVAL = 1
RESUME_FROM_CHECKPOINT = True

ACTIVE_OPTIMIZER_PILOT = 'pilot_1'
OPTIMIZER_PILOTS: dict[str, dict[str, float | int | str]] = {
    'pilot_1': {
        'description': 'First optimizer-only pilot; compare best stable checkpoint only.',
        'num_epochs': 96,
        'learning_rate': 3e-4,
        'warmup_fraction': 0.05,
        'weight_decay': 3e-3,
        'gradient_clip_norm': 0.5,
    },
    'pilot_2': {
        'description': (
            'Second optimizer-only pilot to use only if pilot_1 still shows '
            'inverse instability at the best stable checkpoint.'
        ),
        'num_epochs': 96,
        'learning_rate': 1e-4,
        'warmup_fraction': 0.1,
        'weight_decay': 1e-3,
        'gradient_clip_norm': 0.25,
    },
}

if ACTIVE_OPTIMIZER_PILOT not in OPTIMIZER_PILOTS:
    raise ValueError(
        f'Unknown ACTIVE_OPTIMIZER_PILOT {ACTIVE_OPTIMIZER_PILOT!r}; '
        f'expected one of {sorted(OPTIMIZER_PILOTS)}.'
    )

_ACTIVE_PILOT = OPTIMIZER_PILOTS[ACTIVE_OPTIMIZER_PILOT]

NUM_EPOCHS = int(_ACTIVE_PILOT['num_epochs'])
LEARNING_RATE = float(_ACTIVE_PILOT['learning_rate'])
MIN_LEARNING_RATE = 1e-6
WARMUP_FRACTION = float(_ACTIVE_PILOT['warmup_fraction'])
USE_COSINE_SCHEDULE = True
WEIGHT_DECAY = float(_ACTIVE_PILOT['weight_decay'])
GRADIENT_CLIP_NORM = float(_ACTIVE_PILOT['gradient_clip_norm'])
EMBEDDING_NOISE_STD = 0.05
AUX_SCALE_LOSS_WEIGHT = 0.02
STABLE_INVERSE_FORWARD_MAX_ABS_ERROR_THRESHOLD = 1e-4
STABLE_INVERSE_FORWARD_LOGDET_CLOSURE_MAX_ABS_ERROR_THRESHOLD = 1e-4
STABLE_CUBE_EDGE_HIT_RATE_THRESHOLD = 0.0

NEURAL_AUTOREGRESSIVE_EMBEDDING_DIM = 64
NEURAL_AUTOREGRESSIVE_NUM_TRANSFORMS = 3
NEURAL_AUTOREGRESSIVE_HIDDEN_DIM = 512
NEURAL_AUTOREGRESSIVE_CONDITIONER_DEPTH = 5
NEURAL_AUTOREGRESSIVE_TRANSFORM_UNITS = 16
NEURAL_AUTOREGRESSIVE_MIN_SLOPE = 1e-3
NEURAL_AUTOREGRESSIVE_MIN_RESIDUAL = 0.05
NEURAL_AUTOREGRESSIVE_INVERSE_BISECTION_STEPS = 48
SPECTRUM_ENCODER_TYPE = 'conv1d'
SPECTRUM_EMBEDDING_DIM = 64
PHOTOMETRY_EMBEDDING_DIM = 64
ENCODER_HIDDEN_DIM = 128

RUN_LABEL: str | None = None
RESUME_RUN_LABEL: str | None = None


_ORIGINAL_BUILD_RUN_CONFIG = base.build_run_config


def _resolve_output_root() -> Path:
    if OUTPUT_ROOT is not None:
        return OUTPUT_ROOT.resolve()

    example_data_file, _ = base.locate_input_files()
    return example_data_file.parents[4] / 'runs' / 'hr8799e_gravity_sbi_v3'


def _resolve_latest_compatible_resume_run_label() -> str | None:
    if RESUME_RUN_LABEL is not None:
        return RESUME_RUN_LABEL

    resume_from_checkpoint = (
        RESUME_FROM_CHECKPOINT and not RESUME_FROM_SAVED_POSTERIOR
    )

    if not (RESUME_FROM_SAVED_POSTERIOR or resume_from_checkpoint):
        return None

    example_data_file, _ = base.locate_input_files()
    retrieval_config, pressures = base.build_retrieval_config(example_data_file)
    del pressures

    gravity_data = retrieval_config.data['GRAVITY']
    task = base.build_sbi_task(retrieval_config)
    run_config = build_run_config(task, int(gravity_data.wavelengths.shape[0]))
    expected_config_hash = base.compute_run_config_hash(run_config)
    runs_root = _resolve_output_root() / 'sbi' / 'runs'

    if not runs_root.exists():
        raise FileNotFoundError(f'No saved runs directory found for resume: {runs_root}')

    compatible_runs: list[Path] = []
    for run_root in runs_root.iterdir():
        if not run_root.is_dir() or not (run_root / 'run_manifest.json').exists():
            continue

        manifest = base.load_run_manifest(run_root)
        if str(manifest.get('config_hash', '')) != expected_config_hash:
            continue
        if RESUME_FROM_SAVED_POSTERIOR and not (run_root / 'posterior_model').exists():
            continue
        if resume_from_checkpoint and not (run_root / 'checkpoints').exists():
            continue

        compatible_runs.append(run_root)

    if not compatible_runs:
        raise FileNotFoundError(
            'Could not find a saved neural-autoregressive run matching the current '
            f'configuration hash {expected_config_hash} in {runs_root}. '
            'Set RESUME_RUN_LABEL explicitly to override the auto-selected run.'
        )

    return max(compatible_runs, key=lambda path: path.name).name


def _apply_base_overrides() -> None:
    resume_from_checkpoint = (
        RESUME_FROM_CHECKPOINT and not RESUME_FROM_SAVED_POSTERIOR
    )
    resume_run_label = _resolve_latest_compatible_resume_run_label()

    base.N_SIMULATIONS = N_SIMULATIONS
    base.NUM_EPOCHS = NUM_EPOCHS
    base.BATCH_SIZE = BATCH_SIZE
    base.N_POSTERIOR_SAMPLES = N_POSTERIOR_SAMPLES
    base.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    base.N_PREDICTIVE_FORWARD_MODEL_SAMPLES = N_PREDICTIVE_FORWARD_MODEL_SAMPLES
    base.RESUME_FROM_SAVED_POSTERIOR = RESUME_FROM_SAVED_POSTERIOR
    base.RESUME_FROM_CHECKPOINT = resume_from_checkpoint
    base.RUN_EXACT_RETRIEVAL = RUN_EXACT_RETRIEVAL
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.GENERATE_PLOTS = GENERATE_PLOTS
    base.PLOT_DPI = PLOT_DPI
    base.VERBOSE_TRAINING_DIAGNOSTICS = VERBOSE_TRAINING_DIAGNOSTICS
    base.TRAINING_DIAGNOSTICS_PLOT_INTERVAL = TRAINING_DIAGNOSTICS_PLOT_INTERVAL
    base.LEARNING_RATE = LEARNING_RATE
    base.GRADIENT_CLIP_NORM = GRADIENT_CLIP_NORM
    base.EMBEDDING_NOISE_STD = EMBEDDING_NOISE_STD
    base.AUX_SCALE_LOSS_WEIGHT = AUX_SCALE_LOSS_WEIGHT
    base.SPLINE_SMALL_BIN_REGULARIZATION_WEIGHT = 0.0
    base.SPLINE_DERIVATIVE_REGULARIZATION_WEIGHT = 0.0
    base.STABLE_INVERSE_FORWARD_MAX_ABS_ERROR_THRESHOLD = (
        STABLE_INVERSE_FORWARD_MAX_ABS_ERROR_THRESHOLD
    )
    base.STABLE_INVERSE_FORWARD_LOGDET_CLOSURE_MAX_ABS_ERROR_THRESHOLD = (
        STABLE_INVERSE_FORWARD_LOGDET_CLOSURE_MAX_ABS_ERROR_THRESHOLD
    )
    base.STABLE_CUBE_EDGE_HIT_RATE_THRESHOLD = STABLE_CUBE_EDGE_HIT_RATE_THRESHOLD
    base.RUN_LABEL = RUN_LABEL
    base.RESUME_RUN_LABEL = resume_run_label
    base.build_run_config = build_run_config


def build_run_config(task: Any, n_wavelength_points: int) -> dict[str, Any]:
    run_config = _ORIGINAL_BUILD_RUN_CONFIG(task, n_wavelength_points)
    run_config.update(
        {
            'flow_family': 'neural_autoregressive',
            'num_coupling_layers': NEURAL_AUTOREGRESSIVE_NUM_TRANSFORMS,
            'embedding_dim': NEURAL_AUTOREGRESSIVE_EMBEDDING_DIM,
            'hidden_dim': NEURAL_AUTOREGRESSIVE_HIDDEN_DIM,
            'conditioner_depth': NEURAL_AUTOREGRESSIVE_CONDITIONER_DEPTH,
            'autoregressive_transform_units': NEURAL_AUTOREGRESSIVE_TRANSFORM_UNITS,
            'neural_autoregressive_min_slope': NEURAL_AUTOREGRESSIVE_MIN_SLOPE,
            'neural_autoregressive_min_residual': NEURAL_AUTOREGRESSIVE_MIN_RESIDUAL,
            'neural_autoregressive_inverse_bisection_steps': NEURAL_AUTOREGRESSIVE_INVERSE_BISECTION_STEPS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'gradient_clip_norm': GRADIENT_CLIP_NORM,
            'embedding_noise_std': EMBEDDING_NOISE_STD,
            'aux_scale_loss_weight': AUX_SCALE_LOSS_WEIGHT,
            'spline_small_bin_regularization_weight': 0.0,
            'spline_derivative_regularization_weight': 0.0,
            'weight_decay': WEIGHT_DECAY,
            'use_cosine_schedule': USE_COSINE_SCHEDULE,
            'warmup_fraction': WARMUP_FRACTION,
            'min_learning_rate': MIN_LEARNING_RATE,
            'optimizer_pilot_name': ACTIVE_OPTIMIZER_PILOT,
            'optimizer_pilot_description': str(_ACTIVE_PILOT['description']),
            'stable_inverse_forward_max_abs_error_threshold': (
                STABLE_INVERSE_FORWARD_MAX_ABS_ERROR_THRESHOLD
            ),
            'stable_inverse_forward_logdet_closure_max_abs_error_threshold': (
                STABLE_INVERSE_FORWARD_LOGDET_CLOSURE_MAX_ABS_ERROR_THRESHOLD
            ),
            'stable_cube_edge_hit_rate_threshold': STABLE_CUBE_EDGE_HIT_RATE_THRESHOLD,
            'spectrum_encoder_type': SPECTRUM_ENCODER_TYPE,
            'n_wavelengths': n_wavelength_points,
            'spectrum_embedding_dim': SPECTRUM_EMBEDDING_DIM,
            'photometry_embedding_dim': PHOTOMETRY_EMBEDDING_DIM,
            'encoder_hidden_dim': ENCODER_HIDDEN_DIM,
        }
    )
    return run_config


def main() -> None:
    _apply_base_overrides()
    base.main()


if __name__ == '__main__':
    main()