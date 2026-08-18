"""User interface for configuration dataclasses for amortized SBI.

:class:`~petitRADTRANS.experimental.sbi.flows.posterior.ConditionalFlowPosterior` accepts ~50
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
  :func:`petitRADTRANS.experimental.sbi.run_sbi`, so a benchmark script configures an entire run
  declaratively and calls one function. ``SBIConfig.to_run_config`` serializes the
  estimator-relevant subset to a flat, JSON-able provenance dict that also seeds the
  run hash.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .flows import ConditionalFlowPosterior
from .preprocessing import EXACT_SUMMARY_DIM


@dataclass
class FlowConfig:
    """Normalizing-flow family and geometry.

    Selects the conditional density-estimator hypothesis class and its shape. See
    :mod:`petitRADTRANS.experimental.sbi.flows.transforms` for the trade-offs between families.

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
        ``None`` defaults to ``1e-2`` for ``family='spline'`` (falling back to
        ``spline_entropy_regularization_weight`` when that legacy weight is set),
        and ``0.0`` otherwise. Sharpening the conditional density is otherwise
        unpenalized by the NPE objective, so spline flows keep a mild
        anti-collapse penalty on by default.
    spline_derivative_regularization_weight:
        Weight of the penalty keeping spline slopes within the target derivative
        range. ``None`` defaults to ``1e-2`` for ``family='spline'`` and ``0.0``
        otherwise.
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
    spline_derivative_regularization_weight: float | None = None
    spline_entropy_regularization_weight: float = 0.0
    spline_entropy_floor: float = 0.6
    spline_min_bin_ratio_target: float = 0.2
    spline_min_derivative_target: float = 0.25
    spline_max_derivative_target: float = 5.0

    def __post_init__(self) -> None:
        # Spline flows keep mild anti-collapse regularization on by default:
        # the NPE objective otherwise rewards arbitrarily sharp conditional
        # densities, which is the collapse failure mode seen in production.
        if self.spline_small_bin_regularization_weight is None:
            if self.family == "spline" and self.spline_entropy_regularization_weight == 0.0:
                self.spline_small_bin_regularization_weight = 1e-2
        if self.spline_derivative_regularization_weight is None:
            self.spline_derivative_regularization_weight = (
                1e-2 if self.family == "spline" else 0.0
            )


@dataclass
class EncoderConfig:
    """Observation-embedding encoder.

    Configures the network that turns observation blocks into the fixed-length
    embedding the flow conditions on. Defaults to the convolution encoder that
    explicitly factors a spectrum's amplitude from its shape (see
    :class:`petitRADTRANS.experimental.sbi.encoders.SpectralConvolutionEncoder`), so the
    posterior cannot collapse to the prior on shape-determined parameters.

    Attributes
    ----------
    spectrum_encoder_type:
        Spectral encoder architecture: ``'convolution'`` (dilated-conv +
        attention with amplitude/shape factorization) or ``'resmlp'`` (plain
        residual MLP over the whole normalized spectrum, the published-recipe
        embedding of Vasist et al. 2023; ``hidden_dim`` sets its first block
        width, 512 reproducing their 2x512/3x256/5x128 ladder).
    embedding_dim:
        Size of the final joint observation embedding consumed by the flow as its
        conditioning context.
    spectrum_embedding_dim:
        Intermediate embedding size of the spectral sub-encoder.
    photometry_embedding_dim:
        Intermediate embedding size of the photometric sub-encoder.
    hidden_dim:
        Hidden width of the encoder networks.
    exact_summary_channel:
        Append the analytically computed exact-summary scalars (log median
        transit depth / flux, the normalized median, the within-spectrum scatter
        and the noise level) directly to the flow's conditioning vector,
        bypassing the encoder network and every per-observation normalization.

        On the transmission benchmark the trained 256-dimensional embedding
        recovered ``log(median depth)`` at R^2 0.67 (residual 0.436) where a
        plain median of the encoder's own input reaches R^2 0.999992 (residual
        0.0022) -- so the network was destroying, not merely failing to sharpen,
        the statistic that pins ``R_p + h``. Enabled by default; the estimator's
        own default is off so that models saved before the channel existed
        reload with their original context width.
    """

    spectrum_encoder_type: str = "convolution"
    embedding_dim: int = 80
    spectrum_embedding_dim: int = 64
    photometry_embedding_dim: int = 32
    hidden_dim: int = 128
    exact_summary_channel: bool = True


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
    bnpe_balance_loss_weight:
        Weight of the optional BNPE balance regularizer that pushes the amortized
        posterior toward conservative (non-overconfident) coverage. ``0`` (default)
        disables it; only active for ``parameter_space='cube'``.
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
    bnpe_balance_loss_weight: float = 0.0
    # x-space focal reweighting of the NPE loss: per-sample weights peaked at
    # the target observation's log-median value. Conditional-MLE invariant
    # (does not bias p(theta|x)); reallocates capacity toward the target's
    # neighbourhood in observation space. Requires noise resampling (the raw
    # data path). Checkpoint selection then tracks the focal-weighted
    # validation loss; the unweighted loss remains logged for comparability.
    focal_reweighting: bool = False
    focal_weight_peak: float = 50.0
    focal_bandwidth_dex: float = 0.15
    # Noise-level conditioning: per-sample error-bar scale drawn as
    # 10**U(low, high) during training, with the scaled uncertainties reported
    # to the encoder alongside the matching noise, so the posterior is learned
    # as a function of the assumed noise instead of memorising one error-bar
    # scale. None keeps a single fixed level. Not in the task fingerprint, so
    # switching it reuses the simulation dataset.
    noise_level_log10_range: tuple[float, float] | None = None
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
    calibration_coverage_error:
        Maximum tolerated deviation of the per-epoch empirical calibration probe
        (central-80%-interval coverage over held-out validation simulations) from
        its nominal 0.8. Selecting checkpoints on validation NLL alone deploys the
        sharpest epoch even when it has become overconfident; this gate makes
        "best" mean calibrated *and* low-loss.
    """

    inverse_forward_max_abs_error: float | None = 0.03
    inverse_forward_logdet_closure_max_abs_error: float | None = 0.03
    cube_edge_hit_rate: float | None = 0.02
    calibration_coverage_error: float | None = 0.25


@dataclass
class RefinementConfig:
    """Test-time, per-observation posterior refinement (any objective).

    ``method="none"`` (default) uses the amortized posterior directly for the
    target inference -- the behaviour of a plain NPE run. ``"map_laplace"``,
    ``"importance_sampling"``, or ``"elbo"`` refine the posterior for the
    specific target spectrum with the differentiable forward model, starting
    from the amortized fit; only the target inference uses the refined
    posterior (validation/SBC stay amortized).

    Attributes
    ----------
    method:
        Refinement strategy: ``"none"`` (default; use the amortized posterior
        directly), ``"map_laplace"`` (MAP point estimate + Laplace/Gaussian-VI
        covariance), ``"importance_sampling"`` (exact-likelihood reweighting of
        amortized proposal draws -- cannot collapse, and its sampling
        efficiency is a built-in trust diagnostic), or ``"elbo"`` (flow-ELBO
        refinement).
    is_num_proposal_samples:
        Amortized proposal draws weighted by the exact likelihood
        (``method="importance_sampling"``).
    is_forward_batch_size:
        Forward-model chunk size for the importance weights. ``None`` (default)
        inherits ``SBIConfig.simulation_batch_size``, i.e. the per-device batch
        already validated for this task's forward model during dataset
        generation -- the scattering emission RT holds far larger per-sample
        intermediates than transmission, and a hardcoded chunk size OOMed a
        completed emission run at the refinement step.
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
        How the posterior covariance is estimated around the MAP point.
        ``"hessian"`` (default) uses the exact Hessian of the negative log
        posterior -- the Laplace approximation as defined. ``"laplace"`` keeps
        the historical Gauss-Newton approximation ``J^T J``, which drops the
        residual-curvature term ``sum_i r_i d^2 r_i / du^2``; that term only
        vanishes for a perfect fit, and dropping it makes the posterior too
        narrow, which is the observed failure (held-out coverage 0.52 against a
        nominal 0.90 with the optimizer already converged at chi2/N 1.28).
        ``"gaussian_vi"`` runs a short mean-field Gaussian variational fit;
        being mean-field and reverse-KL it targets conditional rather than
        marginal widths, so expect it to be narrower, not wider.
    map_covariance_chi2_inflation:
        Scale the MAP covariance by ``chi2/dof`` when the fit is worse than its
        degrees of freedom -- the textbook correction for optimistic error bars.
        Off by default: the factor is reported in the run diagnostics either
        way, and applying it before the curvature term is understood would
        paper over whatever else is wrong.
    map_vi_num_steps, map_vi_learning_rate, map_vi_num_samples:
        Controls for the ``"gaussian_vi"`` covariance estimate (steps, learning
        rate, reparameterized samples per step).
    map_num_starts:
        Independent MAP optimization starts (the amortized mean plus
        individual amortized draws); the best final objective wins. Defends
        against boundary-saturation divergence from a biased warm start.
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
    min_effective_sample_size:
        Smallest importance-sampling effective sample size accepted for
        delivery. Below this the refined "posterior" is a handful of duplicated
        particles rather than a distribution, so the workflow delivers the
        amortized posterior instead and records the rejection. Both benchmarks
        have returned an effective sample size of exactly 1 from 16384
        proposals: a proposal much broader than the true posterior has
        essentially no overlap with it in 18 dimensions, and one particle takes
        all the weight.
    max_test_time_coverage_error:
        Largest |measured - nominal| central-interval coverage accepted from the
        held-out test-time validation of ``map_laplace``. A MAP point with a
        too-narrow Laplace covariance fits the target while being badly
        overconfident; measured coverage of 0.48 against a nominal 0.90 is a
        rejection, not a detail.
    max_test_time_median_chi2_per_n:
        Largest median chi2/N accepted across the held-out test-time cases.
        Values far above 1 mean the refinement optimizer is not converging
        case-to-case, whatever it managed on the target.
    """

    method: str = "none"  # "none" | "map_laplace" | "importance_sampling" | "elbo"
    is_num_proposal_samples: int = 16384
    is_forward_batch_size: int | None = None
    # MAP + Laplace controls.
    map_num_steps: int = 600
    map_learning_rate: float = 5e-2
    map_beta_start: float = 1e-3
    map_beta_anneal_fraction: float = 0.5
    map_covariance_method: str = "hessian"  # "hessian" | "laplace" | "gaussian_vi"
    map_covariance_chi2_inflation: bool = False
    # Independent MAP starts (1 = amortized mean only; >1 adds individual
    # amortized draws). Guards against the logit-boundary saturation trap when
    # the amortized init is biased; best final objective wins.
    map_num_starts: int = 8
    # Where MAP starts come from. "likelihood_ranked" scores amortized draws
    # with one batched forward-model pass and starts from the best; the
    # default "amortized_mean" starts from the posterior mean, which for a
    # near-prior posterior is a low-likelihood point and left 6 of 16 held-out
    # cases at chi2/N of 11-131.
    map_start_selection: str = "likelihood_ranked"
    map_num_scored_starts: int = 512
    # chi2/N above which a MAP solve is recorded as non-converged. The Laplace
    # covariance around a non-converged optimum is meaningless, so those cases
    # are counted rather than averaged in.
    map_convergence_chi2_per_n: float = 5.0
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
    # Acceptance gate on the delivered target posterior. A refinement that
    # fails these is written as a secondary artifact and the amortized
    # posterior is delivered instead; see _refinement_verdict in workflow.py.
    min_effective_sample_size: float = 50.0
    max_test_time_coverage_error: float = 0.25
    max_test_time_median_chi2_per_n: float = 10.0


@dataclass(frozen=True)
class TruncationConfig:
    """Truncated second round (TSNPE) around a first-round posterior.

    Round one trains on the full prior, which for a sharp posterior means
    almost every simulation lands somewhere the posterior has no mass, and the
    flow must resolve a density feature finer than a single spline layer can
    represent. Round two simulates only inside a box around the round-one
    posterior and trains on box-normalized coordinates, so the same
    architecture spends its whole support where the mass is.

    The box comes from the posterior at the target observation, so a round-two
    posterior is amortized only over observations near that target. That is the
    price of the sharpness.

    Attributes
    ----------
    enabled:
        Run the truncated second round.
    coverage:
        Central round-one posterior mass each per-parameter interval spans
        before the safety margin.
    safety_factor:
        Multiplier on each interval's half-width. Round one is measurably
        off-centre on the transmission benchmark, so the box is deliberately
        generous.
    min_box_fraction:
        Floor on any interval's width as a fraction of the prior, so an
        over-sharpened round-one marginal cannot truncate a parameter to a
        sliver.
    min_captured_fraction:
        Round-one posterior mass the box must retain, else the run stops. Mass
        cut away here cannot be recovered by round two.
    require_truth_inside_box:
        When truth values are configured, abort if the box excludes any of
        them. Only meaningful for synthetic benchmarks, where it is the
        sharpest available check that truncation has not gone wrong.
    source_run_label:
        Run whose posterior defines the box. ``None`` uses the most recent
        completed run under the same output root. When that run was itself
        truncated, its box is composed with the new one so the result is always
        in absolute cube coordinates.
    n_simulations:
        Round simulation budget. ``None`` keeps :attr:`SBIConfig.n_simulations`.
    num_epochs:
        Round training epochs. ``None`` keeps :attr:`TrainingConfig.num_epochs`.

    Notes
    -----
    Each round trains a fresh flow rather than continuing the previous one.
    Rescaling the cube to the box changes the parameterisation the flow is
    conditioned on, so the previous weights describe a different coordinate
    system and continuing from them would not mean what it appears to mean.

    One round is limited by how informative the previous posterior is. On the
    transmission benchmark the first box retains 0.43 of the prior volume,
    because 17 of 18 marginals are still near-prior -- so this is used as a
    sequence of modest, individually verified rounds, not a single cut.
    """

    enabled: bool = False
    coverage: float = 0.9999
    safety_factor: float = 1.5
    min_box_fraction: float = 0.02
    min_captured_fraction: float = 0.99
    require_truth_inside_box: bool = True
    source_run_label: str | None = None
    # Reuse a completed round's stored box verbatim instead of deriving one.
    # The box determines the task fingerprint, so a resume that re-derives it
    # risks a different box, a different fingerprint, and a silently
    # regenerated dataset.
    box_from_run_label: str | None = None
    n_simulations: int | None = None
    num_epochs: int | None = None


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
    reparameterization: Any = None,
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
    reparameterization:
        Optional
        :class:`~petitRADTRANS.experimental.sbi.flows.reparameterization.AmplitudeConditionedParameter`
        replacing one parameter's cube coordinate by its observation-conditioned
        ratio. Built by the workflow from
        :attr:`SBIConfig.amplitude_conditioned_parameter`.

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
        exact_summary_dim=(
            EXACT_SUMMARY_DIM if bool(encoder.exact_summary_channel) else 0
        ),
        reparameterization=reparameterization,
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
        bnpe_balance_loss_weight=float(training.bnpe_balance_loss_weight),
        focal_reweighting=bool(training.focal_reweighting),
        focal_weight_peak=float(training.focal_weight_peak),
        focal_bandwidth_dex=float(training.focal_bandwidth_dex),
        noise_level_log10_range=training.noise_level_log10_range,
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
        stable_calibration_coverage_error_threshold=stability.calibration_coverage_error,
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
    """End-to-end configuration for :func:`petitRADTRANS.experimental.sbi.run_sbi`.

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
    truncation: TruncationConfig = field(default_factory=TruncationConfig)

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
    # Continue training from the trainer checkpoints in an existing run
    # directory (requires ``run_label``). For walltime-limited jobs: the
    # trainer checkpoints every epoch, so a timed-out run resumes instead of
    # repeating the whole schedule.
    resume_training: bool = False
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
    # Redraw observational noise per training batch from the stored noise-free
    # spectra (default) instead of replaying the single noise realization frozen
    # into the dataset. Frozen noise lets the network overfit that realization
    # and collapse the posterior; set False only to reproduce legacy runs.
    noise_resample_during_training: bool = True
    # Space the spectral value transform works in: "raw" (the observable) or
    # "transit_radius" (sqrt of transit depth). Not part of the task
    # fingerprint -- the stored simulations are unchanged, only their
    # read-time representation -- so switching it reuses the dataset.
    spectral_value_space: str = "raw"
    # Name of a free parameter to infer in an observation-conditioned basis: the
    # flow is given log10(theta) - exponent * log10(observed amplitude) instead
    # of theta's cube coordinate, so the sharp, data-determined direction is
    # handled by an exact formula rather than learned. For a transmission
    # retrieval this turns "infer the planet radius" into "infer the atmospheric
    # annulus", which is the part the spectrum leaves genuinely uncertain.
    # Requires encoder.exact_summary_channel. None disables it.
    amplitude_conditioned_parameter: str | None = None
    # d log10(theta) / d log10(amplitude): 0.5 for a radius against a transit
    # depth or a flux, both of which go as the square of the radius.
    amplitude_conditioned_exponent: float = 0.5

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
            "noise_resample_during_training": self.noise_resample_during_training,
            "load_completed_model": self.load_completed_model,
            "resume_predictive_checkpoints": self.resume_predictive_checkpoints,
            "save_checkpoint_guardrails": self.save_checkpoint_guardrails,
            "generate_plots": self.generate_plots,
            "flow": asdict(self.flow),
            "encoder": asdict(self.encoder),
            "training": asdict(self.training),
            "stability": asdict(self.stability),
            "refinement": asdict(self.refinement),
            "truncation": asdict(self.truncation),
            "spectral_value_space": self.spectral_value_space,
            "amplitude_conditioned_parameter": self.amplitude_conditioned_parameter,
            "amplitude_conditioned_exponent": self.amplitude_conditioned_exponent,
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
