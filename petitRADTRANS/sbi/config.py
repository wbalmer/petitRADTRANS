"""User interface for configuration dataclasses for amortized SBI.

:class:`~petitRADTRANS.sbi.flows.posterior.ConditionalFlowPosterior` accepts ~50
flat keyword arguments. Driving it directly forces callers into a long, error-prone
keyword ceremony. This module provides user-facing access to those knobs by grouping
them into small dataclasses, each with sensible defaults, so a caller declares only
what differs from the library defaults.

Configuration groups
--------------------
* :class:`FlowConfig` -- the normalizing-flow family and geometry (the
  density-estimator hypothesis class).
* :class:`EncoderConfig` -- the observation-embedding encoder.
* :class:`TrainingConfig` -- the optimisation schedule and the training objective
  (NPE vs ELBO).
* :class:`StabilityConfig` -- the checkpoint-selection stability thresholds.
* :class:`RefinementConfig` -- optional test-time, per-observation posterior
  refinement (ELBO runs only).

Two entry points consume these groups:

* :func:`build_conditional_flow_posterior` expands the *estimator* groups
  (``flow`` / ``encoder`` / ``training`` / ``stability``) back into the flat
  ``ConditionalFlowPosterior`` constructor, returning an untrained estimator::

      posterior = build_conditional_flow_posterior(
          parameter_dim=len(task.parameter_names),
          n_wavelengths=n_wavelengths,
      )

  (or, end to end, via :meth:`AmortizedRetrieval.from_retrieval_config`).

* :class:`SBIConfig` bundles all five groups *plus* the simulation/workflow knobs
  and the built ``RetrievalConfig`` into the single object consumed by
  :func:`petitRADTRANS.sbi.run_sbi`, so a benchmark script configures an entire run
  declaratively and calls one function. ``SBIConfig.to_run_config`` serializes the
  estimator-relevant subset to a flat, JSON-able provenance dict that also seeds the
  run hash.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .flows import ConditionalFlowPosterior


@dataclass
class FlowConfig:
    """Normalizing-flow family and geometry.

    Selects the conditional density-estimator hypothesis class and its shape. See
    :mod:`petitRADTRANS.sbi.flows.transforms` for the trade-offs between families.

    Attributes
    ----------
    family:
        Flow family: ``'neural_autoregressive'`` (default; most expressive,
        iterative inverse), ``'spline'`` (rational-quadratic splines, analytic
        inverse), ``'affine'`` (cheapest), or ``'autoregressive'`` (affine
        autoregressive).
    num_transforms:
        Number of flow transform layers. More layers add capacity at higher cost.
    hidden_dim:
        Hidden width of the per-layer conditioner MLPs.
    conditioner_depth:
        Number of hidden layers in each conditioner MLP.
    num_spline_bins:
        Number of rational-quadratic bins per transformed dimension
        (``family='spline'``); more bins give finer per-dimension shape.
    spline_bound:
        Half-width of the finite spline support ``[-bound, bound]`` in latent
        coordinates (identity outside). Automatically raised to cover the
        logit-of-cube range for a uniform-prior cube flow.
    base_distribution:
        Flow base density, ``'gaussian'`` (default) or ``'logistic'``. A logistic
        base matches logit-of-uniform coordinates so an untrained cube flow already
        equals the prior.
    use_base_affine:
        If ``True`` (spline family), prepend a diagonal-affine layer so the flow can
        set each parameter's location/scale directly -- helps directions constrained
        far more tightly than the prior.
    autoregressive_transform_units:
        Number of monotonic mixture units per scalar map
        (``family='neural_autoregressive'``); higher = more flexible scalar
        transforms.
    neural_autoregressive_min_slope:
        Positive floor on the neural-autoregressive monotonic slopes (keeps the map
        strictly increasing and well-conditioned).
    neural_autoregressive_min_residual:
        Identity-residual fraction in ``(0, 1)`` blended into each
        neural-autoregressive scalar map; keeps it invertible as the posterior
        sharpens.
    neural_autoregressive_inverse_bisection_steps:
        Bisection steps used to numerically invert each neural-autoregressive scalar
        map when sampling.
    spline_small_bin_regularization_weight:
        Weight of the penalty discouraging collapsed spline bins (spline flows only).
        ``None`` falls back to ``spline_entropy_regularization_weight``.
    spline_derivative_regularization_weight:
        Weight of the penalty keeping spline slopes within the target derivative
        range.
    spline_entropy_regularization_weight:
        Legacy alias supplying the default for
        ``spline_small_bin_regularization_weight``.
    spline_entropy_floor:
        Target floor (in ``[0, 1]``) for the normalised spline-bin entropy.
    spline_min_bin_ratio_target:
        Minimum bin-size ratio (relative to a uniform bin) below which the small-bin
        penalty activates.
    spline_min_derivative_target, spline_max_derivative_target:
        Target range for spline boundary slopes. ``spline_max_derivative_target`` is
        also the hard slope cap that prevents unbounded-density collapse; it must
        exceed ``spline_min_derivative_target``.
    """

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
    """Observation-embedding encoder.

    Configures the network that turns observation blocks into the fixed-length
    embedding the flow conditions on. Defaults to the convolution encoder that
    explicitly factors a spectrum's amplitude from its shape (see
    :class:`petitRADTRANS.sbi.encoders.SpectralConvolutionEncoder`), so the
    posterior cannot collapse to the prior on shape-determined parameters.

    Attributes
    ----------
    spectrum_encoder_type:
        Spectral encoder architecture: ``'convolution'``.
    embedding_dim:
        Size of the final joint observation embedding consumed by the flow as its
        conditioning context.
    spectrum_embedding_dim:
        Intermediate embedding size of the spectral sub-encoder.
    photometry_embedding_dim:
        Intermediate embedding size of the photometric sub-encoder.
    hidden_dim:
        Hidden width of the encoder networks.
    """

    spectrum_encoder_type: str = "convolution"
    embedding_dim: int = 80
    spectrum_embedding_dim: int = 64
    photometry_embedding_dim: int = 32
    hidden_dim: int = 128


@dataclass
class TrainingConfig:
    """Optimisation schedule and training objective.

    Defines *what* is optimised and *how* the optimizer is scheduled. Note this is
    the public, grouped training config; the trainer's internal ``TrainingSchemeConfig``
    is separate.

    Attributes
    ----------
    num_epochs:
        Maximum number of passes over the training split.
    batch_size:
        Number of simulations per optimization step.
    learning_rate:
        Peak AdamW learning rate.
    min_learning_rate:
        Floor of the learning-rate schedule.
    weight_decay:
        AdamW weight-decay coefficient.
    gradient_clip_norm:
        Global gradient-norm clip value; ``None`` disables clipping.
    early_stopping_patience:
        Non-improving epochs tolerated before stopping; ``None`` disables early
        stopping.
    use_cosine_schedule:
        Enable cosine learning-rate decay (otherwise constant after warmup).
    warmup_fraction:
        Fraction of the schedule spent linearly warming up the learning rate (used
        when ``warmup_epochs`` is not set).
    warmup_epochs:
        Absolute number of warmup epochs; overrides ``warmup_fraction`` when set.
    lr_schedule_total_epochs:
        Horizon (in epochs) over which the schedule decays; defaults to
        ``num_epochs`` when ``None``.
    training_objective:
        ``'npe'`` (default; conditional negative log-likelihood, robust) or
        ``'elbo'`` (amortized variational inference, which requires
        ``parameter_space='cube'`` and a differentiable forward-model likelihood at
        fit time).
    aux_scale_loss_weight:
        Weight of the auxiliary loss training the scale head to predict each
        observation's log-amplitude; ``0`` disables it.
    invertibility_loss_weight:
        Weight of the differentiable inverse round-trip / log-det-closure penalty
        added to the NPE loss; keeps an iterative-inverse flow (NAF) invertible as
        it sharpens. ``0`` (default) disables it.
    invertibility_loss_n_samples:
        Number of base samples used to estimate the invertibility penalty.
    parameter_space:
        Coordinates the flow is trained in: ``'cube'`` (default; uniform-prior unit
        hypercube) or ``'unconstrained'``. ELBO requires ``'cube'``.
    elbo_num_samples:
        Reparameterized posterior draws per observation for the ELBO estimator
        (``training_objective='elbo'``).
    elbo_beta_start, elbo_beta_end:
        Start/end of the likelihood-temperature (``beta``) anneal, each in
        ``(0, 1]``. Ending below 1 keeps the amortized fit out of the
        noise-dominated, over-confident regime; test-time refinement then supplies
        the sharp final fit.
    elbo_anneal_fraction:
        Fraction of epochs over which ``beta`` anneals log-linearly from
        ``elbo_beta_start`` to ``elbo_beta_end`` before holding.
    """

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
    """Checkpoint-selection stability thresholds (flow invertibility / collapse).

    During training, candidate checkpoints are scored against these thresholds; a
    checkpoint that violates them is treated as unstable/collapsed and penalised (or
    disqualified) when selecting the best model. Setting a field to ``None`` disables
    that check.

    Attributes
    ----------
    inverse_forward_max_abs_error:
        Maximum tolerated inverse->forward round-trip error. A flow whose inverse
        disagrees with its forward map produces samples that do not match its own
        density.
    inverse_forward_logdet_closure_max_abs_error:
        Maximum tolerated inverse/forward log-det closure error.
    cube_edge_hit_rate:
        Maximum tolerated fraction of sampled coordinates pinned at the unit-cube
        edges (a signature of density collapse).
    """

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

    Attributes
    ----------
    method:
        Refinement strategy: ``"none"`` (default; use the amortized posterior
        directly), ``"map_laplace"`` (MAP point estimate + Laplace/Gaussian-VI
        covariance), or ``"elbo"`` (flow-ELBO refinement).
    map_num_steps:
        Number of optimization steps for the MAP point estimate
        (``method="map_laplace"``).
    map_learning_rate:
        Learning rate for the MAP optimization.
    map_beta_start:
        Initial likelihood temperature for the MAP anneal.
    map_beta_anneal_fraction:
        Fraction of MAP steps over which the temperature anneals to 1.
    map_covariance_method:
        How the posterior covariance is estimated around the MAP point:
        ``"laplace"`` (Hessian/Gauss-Newton) or ``"gaussian_vi"`` (Gaussian
        variational fit).
    map_vi_num_steps, map_vi_learning_rate, map_vi_num_samples:
        Controls for the ``"gaussian_vi"`` covariance estimate (steps, learning
        rate, reparameterized samples per step).
    test_time_validation:
        If ``True``, also run the actual test-time method (MAP+Laplace) on held-out
        simulations to measure its accuracy and calibration.
    test_time_validation_cases:
        Number of held-out cases used for that validation.
    test_time_validation_samples:
        Posterior samples drawn per validation case.
    elbo_num_steps:
        Number of optimization steps for flow-ELBO refinement (``method="elbo"``).
    elbo_learning_rate:
        Learning rate for the flow-ELBO refinement.
    elbo_num_samples:
        Reparameterized samples per refinement step.
    elbo_beta_anneal_fraction:
        Fraction of refinement steps over which the likelihood temperature anneals
        (``beta_start`` is taken from :attr:`TrainingConfig.elbo_beta_start`).
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
    """Build a :class:`ConditionalFlowPosterior` from grouped config dataclasses.

    Expands the themed config groups back into the estimator's ~50 flat constructor
    arguments, so the common path is a couple of lines instead of a long keyword
    ceremony. Any config left as ``None`` uses that group's defaults.

    Parameters
    ----------
    parameter_dim:
        Number of inferred free parameters (e.g. ``len(task.parameter_names)``).
    n_wavelengths:
        Number of spectral points per observation block (sizes the encoder); must
        match the task's observation schema.
    flow, encoder, training, stability:
        Grouped configs; ``None`` substitutes the corresponding default dataclass.
    seed:
        Base random seed for flow/encoder initialization and sampling.
    checkpoint_directory:
        Directory for resumable trainer checkpoints; ``None`` disables checkpointing.
    checkpoint_backend:
        Checkpoint backend (``'equinox'`` here by default, vs the estimator's own
        ``'auto'`` default).
    resume_from_checkpoint:
        Whether ``fit`` should resume from the latest checkpoint.
    verbose_diagnostics:
        Emit per-epoch diagnostic artifacts/plots during training.
    diagnostics_output_directory:
        Directory for the verbose diagnostic artifacts.
    diagnostics_plot_interval:
        Epoch interval between verbose diagnostic plots.
    task_metadata:
        Optional metadata persisted alongside the trained model (e.g. the task
        name).

    Returns
    -------
    ConditionalFlowPosterior
        An *untrained* estimator ready for :meth:`ConditionalFlowPosterior.fit`.
    """
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
    """End-to-end configuration for :func:`petitRADTRANS.sbi.run_sbi`.

    Bundles the experiment definition (a built ``RetrievalConfig`` carrying the
    priors and model-generating function) with the grouped estimator configs and
    the workflow knobs, so a benchmark script declares only what differs from the
    library defaults and calls one function -- no module-global / monkeypatch
    plumbing.

    Attributes
    ----------
    retrieval_config:
        Built ``RetrievalConfig`` defining the free parameters/priors, observation
        datasets, and the differentiable model-generating function. The single
        largest scientific input.
    output_root:
        Root directory under which the shared ``datasets/`` cache and per-run
        ``runs/<run_label>/`` directories are created.
    flow, encoder, training, stability, refinement:
        The grouped estimator configs (:class:`FlowConfig`, :class:`EncoderConfig`,
        :class:`TrainingConfig`, :class:`StabilityConfig`, :class:`RefinementConfig`).
    n_simulations:
        Total number of prior-predictive simulations to generate (split into
        train/validation/test).
    simulation_batch_size:
        Per-device simulator batch size during dataset generation.
    seed:
        Master random seed; per-stage seeds are derived from it for reproducibility.
    n_posterior_samples:
        Number of posterior draws for the delivered target inference.
    n_predictive_forward_model_samples:
        Number of posterior draws pushed through the forward model for
        posterior-predictive summaries (bounds the expensive pRT calls).
    validation_max_cases:
        Maximum held-out test cases used for the SBC and posterior-predictive
        validation reports.
    run_label:
        Explicit run directory name; when ``None`` a label is derived from the run
        config hash.
    load_completed_model:
        If ``True``, skip training and load a previously trained, fingerprint-
        compatible posterior.
    resume_predictive_checkpoints:
        If ``True`` (with ``load_completed_model``), reuse the source run directory
        and resume per-case predictive checkpoints.
    generate_plots:
        Whether to render the validation plot set.
    plot_dpi:
        Resolution of saved figures.
    save_checkpoint_guardrails:
        Whether to run inference for each saved checkpoint kind and persist a
        guardrail spectrum/plot/summary (makes checkpoint-selection regressions
        visible without retraining).
    guardrail_posterior_samples, guardrail_predictive_forward_model_samples:
        Smaller sample budgets used for the (cheaper) guardrail inference.
    startup_scale_check:
        Behaviour of the pre-training observation/training flux-scale guard:
        ``"raise"``, ``"warn"``, or ``"off"``.
    verbose_training_diagnostics:
        Whether the trainer emits per-epoch diagnostic artifacts/plots.
    training_diagnostics_plot_interval:
        Epoch interval between verbose training-diagnostic plots.
    preprocessing_version:
        Version tag stamped onto the fitted preprocessing metadata (participates in
        cache invalidation).
    observation_value_constraint:
        Per-dataset prior-predictive value constraint for rejection sampling. When
        ``None`` the workflow applies "value > 0" (positive emission flux / transit
        depth).
    spectrum_x_label:
        X-axis label for predictive-spectrum plots.
    spectrum_y_label:
        Y-axis label for predictive-spectrum plots; when ``None`` it is auto-derived
        (emission flux when ``scattering_in_emission`` is set, otherwise transit
        depth).
    truth_parameter_values:
        Optional known ground-truth parameter values, recorded in the summary and
        overlaid on plots when available.
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
