"""Autoregressive-flow variant of the HR8799e runtime-native SBI example.

This variant reuses the runtime-native workflow from
``hr8799e_runtime_native_real_data.py`` but swaps the spline posterior for a
conditional autoregressive affine flow and updates the optimization settings to
match that simpler flow family more closely.

The trainer currently exposes constant and cosine schedules, but not
ReduceLROnPlateau. This script therefore keeps the higher initial learning rate
and stronger weight decay from the autoregressive-flow recipe while using a
cosine decay as the closest supported approximation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hr8799e_runtime_native_real_data as base


# User-editable configuration for the autoregressive-flow variant.
N_SIMULATIONS = 2_000_000
NUM_EPOCHS = 1024
BATCH_SIZE = 1024
N_POSTERIOR_SAMPLES = 512
EARLY_STOPPING_PATIENCE = 64
N_PREDICTIVE_FORWARD_MODEL_SAMPLES = 256
RESUME_FROM_SAVED_POSTERIOR = False
RESUME_FROM_CHECKPOINT = True
RUN_EXACT_RETRIEVAL = False
OUTPUT_ROOT: Path | None = None
GENERATE_PLOTS = True
PLOT_DPI = 160
VERBOSE_TRAINING_DIAGNOSTICS = True
TRAINING_DIAGNOSTICS_PLOT_INTERVAL = 1

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-6
WARMUP_FRACTION = 0.02
USE_COSINE_SCHEDULE = True
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP_NORM = 1.0
EMBEDDING_NOISE_STD = 0.05
AUX_SCALE_LOSS_WEIGHT = 0.02

AUTOREGRESSIVE_EMBEDDING_DIM = 64
AUTOREGRESSIVE_NUM_TRANSFORMS = 3
AUTOREGRESSIVE_HIDDEN_DIM = 512
AUTOREGRESSIVE_CONDITIONER_DEPTH = 5
SPECTRUM_ENCODER_TYPE = 'conv1d'
SPECTRUM_EMBEDDING_DIM = 64
PHOTOMETRY_EMBEDDING_DIM = 64
ENCODER_HIDDEN_DIM = 128

RUN_LABEL: str | None = None
RESUME_RUN_LABEL: str | None = None


_ORIGINAL_BUILD_RUN_CONFIG = base.build_run_config


def _apply_base_overrides() -> None:
    base.N_SIMULATIONS = N_SIMULATIONS
    base.NUM_EPOCHS = NUM_EPOCHS
    base.BATCH_SIZE = BATCH_SIZE
    base.N_POSTERIOR_SAMPLES = N_POSTERIOR_SAMPLES
    base.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    base.N_PREDICTIVE_FORWARD_MODEL_SAMPLES = N_PREDICTIVE_FORWARD_MODEL_SAMPLES
    base.RESUME_FROM_SAVED_POSTERIOR = RESUME_FROM_SAVED_POSTERIOR
    base.RESUME_FROM_CHECKPOINT = RESUME_FROM_CHECKPOINT
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
    base.RUN_LABEL = RUN_LABEL
    base.RESUME_RUN_LABEL = RESUME_RUN_LABEL
    base.build_run_config = build_run_config


def build_run_config(task: Any, n_wavelength_points: int) -> dict[str, Any]:
    run_config = _ORIGINAL_BUILD_RUN_CONFIG(task, n_wavelength_points)
    run_config.update(
        {
            'flow_family': 'autoregressive',
            'num_coupling_layers': AUTOREGRESSIVE_NUM_TRANSFORMS,
            'embedding_dim': AUTOREGRESSIVE_EMBEDDING_DIM,
            'hidden_dim': AUTOREGRESSIVE_HIDDEN_DIM,
            'conditioner_depth': AUTOREGRESSIVE_CONDITIONER_DEPTH,
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