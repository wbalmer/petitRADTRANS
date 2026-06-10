"""Grouped configuration dataclasses for the amortized SBI posterior.

``ConditionalFlowPosterior`` accepts ~40 flat keyword arguments. Driving it
directly forces callers into a long ``build_run_config`` / ``build_posterior_estimator``
ceremony. These dataclasses group the knobs by concern -- flow geometry, the
observation encoder, the training schedule, and the checkpoint-stability gate --
each with sensible defaults, and :func:`build_conditional_flow_posterior` expands
them back into the flat constructor. The common path becomes a few lines:

    posterior = build_conditional_flow_posterior(
        parameter_dim=len(task.parameter_names),
        n_wavelengths=n_wavelengths,
    )

or, end to end, via :meth:`AmortizedRetrieval.from_retrieval_config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .flow_posterior import ConditionalFlowPosterior


@dataclass
class FlowConfig:
    """Normalizing-flow family and geometry."""

    family: str = "neural_autoregressive"
    num_transforms: int = 4
    hidden_dim: int = 256
    conditioner_depth: int = 3
    num_spline_bins: int = 8
    spline_bound: float = 10.0
    base_distribution: str = "gaussian"
    use_base_affine: bool = False
    # Neural-autoregressive (NAF) specific knobs.
    autoregressive_transform_units: int = 16
    neural_autoregressive_min_slope: float = 5e-3
    neural_autoregressive_min_residual: float = 0.1
    neural_autoregressive_inverse_bisection_steps: int = 64
    # Rational-quadratic spline regularization (only used by spline flows).
    spline_small_bin_regularization_weight: float | None = None
    spline_derivative_regularization_weight: float = 0.0
    spline_entropy_regularization_weight: float = 0.0
    spline_entropy_floor: float = 0.6
    spline_min_bin_ratio_target: float = 0.2
    spline_min_derivative_target: float = 0.25
    spline_max_derivative_target: float = 5.0


@dataclass
class EncoderConfig:
    """Observation-embedding encoder. Defaults to the convolution encoder that
    explicitly factors amplitude from shape (see ``SpectralGenericEncoder``)."""

    spectrum_encoder_type: str = "convolution"
    embedding_dim: int = 80
    spectrum_embedding_dim: int = 64
    photometry_embedding_dim: int = 32
    hidden_dim: int = 128


@dataclass
class TrainingConfig:
    """Optimisation schedule and objective."""

    num_epochs: int = 96
    batch_size: int = 1024
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-6
    weight_decay: float = 1e-3
    gradient_clip_norm: float | None = 2.0
    early_stopping_patience: int | None = 20
    use_cosine_schedule: bool = True
    warmup_fraction: float = 0.02
    warmup_epochs: float | None = 1.0
    lr_schedule_total_epochs: float | None = None
    # 'npe' (conditional NLL, default and robust) or 'elbo' (amortized VI).
    training_objective: str = "npe"
    aux_scale_loss_weight: float = 0.0
    # 'cube' (uniform prior) or 'unconstrained'. ELBO requires 'cube'.
    parameter_space: str = "cube"
    # ELBO-objective knobs (ignored when training_objective='npe').
    elbo_num_samples: int = 1
    elbo_beta_start: float = 1e-3
    elbo_anneal_fraction: float = 0.6


@dataclass
class StabilityConfig:
    """Checkpoint-selection stability thresholds (flow invertibility / collapse)."""

    inverse_forward_max_abs_error: float | None = 0.03
    inverse_forward_logdet_closure_max_abs_error: float | None = 0.03
    cube_edge_hit_rate: float | None = 0.02


def build_conditional_flow_posterior(
    *,
    parameter_dim: int,
    n_wavelengths: int,
    flow: FlowConfig | None = None,
    encoder: EncoderConfig | None = None,
    training: TrainingConfig | None = None,
    stability: StabilityConfig | None = None,
    seed: int = 0,
    checkpoint_directory: str | None = None,
    checkpoint_backend: str = "equinox",
    resume_from_checkpoint: bool = False,
    verbose_diagnostics: bool = False,
    diagnostics_output_directory: str | None = None,
    diagnostics_plot_interval: int = 1,
    task_metadata: Mapping[str, Any] | None = None,
) -> ConditionalFlowPosterior:
    """Build a :class:`ConditionalFlowPosterior` from grouped config dataclasses."""
    flow = flow or FlowConfig()
    encoder = encoder or EncoderConfig()
    training = training or TrainingConfig()
    stability = stability or StabilityConfig()

    return ConditionalFlowPosterior(
        parameter_dim=int(parameter_dim),
        # encoder
        embedding_dim=int(encoder.embedding_dim),
        spectrum_encoder_type=str(encoder.spectrum_encoder_type),
        n_wavelengths=int(n_wavelengths),
        spectrum_embedding_dim=int(encoder.spectrum_embedding_dim),
        photometry_embedding_dim=int(encoder.photometry_embedding_dim),
        encoder_hidden_dim=int(encoder.hidden_dim),
        # flow
        flow_family=str(flow.family),
        num_coupling_layers=int(flow.num_transforms),
        hidden_dim=int(flow.hidden_dim),
        conditioner_depth=int(flow.conditioner_depth),
        num_spline_bins=int(flow.num_spline_bins),
        spline_bound=float(flow.spline_bound),
        base_distribution=str(flow.base_distribution),
        use_base_affine=bool(flow.use_base_affine),
        autoregressive_transform_units=int(flow.autoregressive_transform_units),
        neural_autoregressive_min_slope=float(flow.neural_autoregressive_min_slope),
        neural_autoregressive_min_residual=float(
            flow.neural_autoregressive_min_residual
        ),
        neural_autoregressive_inverse_bisection_steps=int(
            flow.neural_autoregressive_inverse_bisection_steps
        ),
        spline_small_bin_regularization_weight=(
            flow.spline_small_bin_regularization_weight
        ),
        spline_derivative_regularization_weight=float(
            flow.spline_derivative_regularization_weight
        ),
        spline_entropy_regularization_weight=float(
            flow.spline_entropy_regularization_weight
        ),
        spline_entropy_floor=float(flow.spline_entropy_floor),
        spline_min_bin_ratio_target=float(flow.spline_min_bin_ratio_target),
        spline_min_derivative_target=float(flow.spline_min_derivative_target),
        spline_max_derivative_target=float(flow.spline_max_derivative_target),
        # training
        learning_rate=float(training.learning_rate),
        batch_size=int(training.batch_size),
        num_epochs=int(training.num_epochs),
        parameter_space=str(training.parameter_space),
        training_objective=str(training.training_objective),
        elbo_num_samples=int(training.elbo_num_samples),
        elbo_beta_start=float(training.elbo_beta_start),
        elbo_anneal_fraction=float(training.elbo_anneal_fraction),
        aux_scale_loss_weight=float(training.aux_scale_loss_weight),
        weight_decay=float(training.weight_decay),
        gradient_clip_norm=training.gradient_clip_norm,
        early_stopping_patience=training.early_stopping_patience,
        use_cosine_schedule=bool(training.use_cosine_schedule),
        warmup_fraction=float(training.warmup_fraction),
        warmup_epochs=training.warmup_epochs,
        min_learning_rate=float(training.min_learning_rate),
        lr_schedule_total_epochs=training.lr_schedule_total_epochs,
        # stability gate
        stable_inverse_forward_max_abs_error_threshold=(
            stability.inverse_forward_max_abs_error
        ),
        stable_inverse_forward_logdet_closure_max_abs_error_threshold=(
            stability.inverse_forward_logdet_closure_max_abs_error
        ),
        stable_cube_edge_hit_rate_threshold=stability.cube_edge_hit_rate,
        # plumbing
        seed=int(seed),
        checkpoint_directory=checkpoint_directory,
        checkpoint_backend=checkpoint_backend,
        resume_from_checkpoint=bool(resume_from_checkpoint),
        verbose_diagnostics=bool(verbose_diagnostics),
        diagnostics_output_directory=diagnostics_output_directory,
        diagnostics_plot_interval=int(diagnostics_plot_interval),
        task_metadata=task_metadata,
    )


__all__ = [
    "FlowConfig",
    "EncoderConfig",
    "TrainingConfig",
    "StabilityConfig",
    "build_conditional_flow_posterior",
]
,
]
