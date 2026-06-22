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

from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    # Weight of the differentiable inverse round-trip / log-det-closure penalty
    # added to the NPE training loss. Keeps an iterative-inverse flow (NAF)
    # invertible as it sharpens; 0.0 disables it (default, unchanged behaviour).
    invertibility_loss_weight: float = 0.0
    invertibility_loss_n_samples: int = 64
    # 'cube' (uniform prior) or 'unconstrained'. ELBO requires 'cube'.
    parameter_space: str = "cube"
    # ELBO-objective knobs (ignored when training_objective='npe').
    elbo_num_samples: int = 1
    elbo_beta_start: float = 1e-3
    elbo_anneal_fraction: float = 0.6
    # Anneal beta log-linearly from elbo_beta_start to elbo_beta_end over the
    # first elbo_anneal_fraction of epochs, then hold. Ending at beta_end < 1
    # keeps the amortized phase out of the sigma-dominated (over-collapsed)
    # regime; the test-time refinement then supplies the sharp final fit.
    elbo_beta_end: float = 1.0


@dataclass
class StabilityConfig:
    """Checkpoint-selection stability thresholds (flow invertibility / collapse)."""

    inverse_forward_max_abs_error: float | None = 0.03
    inverse_forward_logdet_closure_max_abs_error: float | None = 0.03
    cube_edge_hit_rate: float | None = 0.02


@dataclass
class RefinementConfig:
    """Test-time, per-observation posterior refinement (ELBO runs only).

    ``method="none"`` (default) uses the amortized posterior directly for the
    target inference -- the behaviour of a plain NPE run. ``"map_laplace"`` or
    ``"elbo"`` refine the posterior for the specific target spectrum with the
    differentiable forward model, initialized from the amortized fit; only the
    target inference uses the refined posterior (validation/SBC stay amortized).
    Refinement is applied only when the training objective is ``"elbo"``.
    """

    method: str = "none"  # "none" | "map_laplace" | "elbo"
    # MAP + Laplace controls.
    map_num_steps: int = 600
    map_learning_rate: float = 5e-2
    map_beta_start: float = 1e-3
    map_beta_anneal_fraction: float = 0.5
    map_covariance_method: str = "laplace"  # "laplace" | "gaussian_vi"
    map_vi_num_steps: int = 300
    map_vi_learning_rate: float = 1e-2
    map_vi_num_samples: int = 32
    # Validate the actual test-time method (MAP+Laplace) on held-out cases.
    test_time_validation: bool = False
    test_time_validation_cases: int = 16
    test_time_validation_samples: int = 1024
    # Flow-ELBO refinement controls (used when method="elbo"); beta_start is
    # taken from TrainingConfig.elbo_beta_start.
    elbo_num_steps: int = 400
    elbo_learning_rate: float = 1e-4
    elbo_num_samples: int = 32
    elbo_beta_anneal_fraction: float = 0.5


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
        elbo_beta_end=float(training.elbo_beta_end),
        aux_scale_loss_weight=float(training.aux_scale_loss_weight),
        invertibility_loss_weight=float(training.invertibility_loss_weight),
        invertibility_loss_n_samples=int(training.invertibility_loss_n_samples),
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


@dataclass
class SBIConfig:
    """End-to-end configuration for :func:`run_validation_workflow`.

    Bundles the experiment definition (a built ``RetrievalConfig`` carrying the
    priors and model-generating function) with the grouped estimator configs and
    the workflow knobs, so a benchmark script declares only what differs from the
    library defaults and calls one function -- no module-global / monkeypatch
    plumbing.
    """

    retrieval_config: Any
    output_root: Path
    flow: FlowConfig = field(default_factory=FlowConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    # Simulation + workflow.
    n_simulations: int = 1_000_000
    simulation_batch_size: int = 256
    seed: int = 12345
    n_posterior_samples: int = 4096
    n_predictive_forward_model_samples: int = 256
    validation_max_cases: int = 128
    run_label: str | None = None
    load_completed_model: bool = False
    resume_predictive_checkpoints: bool = False
    generate_plots: bool = True
    plot_dpi: int = 160
    save_checkpoint_guardrails: bool = True
    guardrail_posterior_samples: int = 512
    guardrail_predictive_forward_model_samples: int = 96
    startup_scale_check: str = "raise"
    verbose_training_diagnostics: bool = True
    training_diagnostics_plot_interval: int = 10
    preprocessing_version: str = "0.6.0"
    # Per-dataset value constraint for prior-predictive rejection sampling. When
    # None the workflow applies "value > 0" (emission flux / transit depth).
    observation_value_constraint: Any = None

    # Presentation (predictive-spectrum plot axis labels) + optional known truths.
    # spectrum_y_label=None auto-derives from the retrieval config: emission flux
    # when scattering_in_emission is set, otherwise transit depth.
    spectrum_x_label: str = r"Wavelength [$\mu$m]"
    spectrum_y_label: str | None = None
    truth_parameter_values: Mapping[str, float] = field(default_factory=dict)

    def to_run_config(self) -> dict[str, Any]:
        """Flat, JSON-serializable provenance dict (also drives the run hash)."""
        return {
            "n_simulations": self.n_simulations,
            "simulation_batch_size": self.simulation_batch_size,
            "seed": self.seed,
            "n_posterior_samples": self.n_posterior_samples,
            "n_predictive_forward_model_samples": (
                self.n_predictive_forward_model_samples
            ),
            "validation_max_cases": self.validation_max_cases,
            "load_completed_model": self.load_completed_model,
            "resume_predictive_checkpoints": self.resume_predictive_checkpoints,
            "save_checkpoint_guardrails": self.save_checkpoint_guardrails,
            "generate_plots": self.generate_plots,
            "flow": asdict(self.flow),
            "encoder": asdict(self.encoder),
            "training": asdict(self.training),
            "stability": asdict(self.stability),
            "refinement": asdict(self.refinement),
        }


__all__ = [
    "FlowConfig",
    "EncoderConfig",
    "TrainingConfig",
    "StabilityConfig",
    "RefinementConfig",
    "SBIConfig",
    "build_conditional_flow_posterior",
]
