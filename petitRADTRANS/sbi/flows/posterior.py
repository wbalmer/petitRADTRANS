"""Conditional normalizing-flow posterior estimator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping

import equinox as eqx
from jax import Array, random, vmap
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from ..encoders import ObservationEncoder
from .transforms import (
    ConditionalAutoregressiveFlow,
    ConditionalAffineFlow,
    ConditionalNeuralAutoregressiveFlow,
    ConditionalSplineFlow,
)
from ..observation import EncodedObservation, ObservationBlock
from .base import (
    PersistentPosteriorEstimator,
    PosteriorBatch,
    PosteriorSamples,
    TrainingArtifacts,
)
from ..training import EarlyStoppingConfig, SBITrainer, TrainingSchemeConfig
from . import refinement
from .cube import (
    _clip_cube_support,
    _cube_log_abs_det_jacobian,
    _cube_to_unconstrained,
    _minimum_cube_spline_bound,
)
from .objectives import (
    _PosteriorModel,
    compute_loss,
    compute_validation_diagnostics,
    make_elbo_loss,
    training_loss,
)
from .regularization import (
    _ELBO_STABILITY_CUBE_EDGE_HIT_RATE,
    _flow_validation_selection_metric,
    _flow_validation_stability_metrics,
)


class ConditionalFlowPosterior(PersistentPosteriorEstimator):
    """Concrete amortized posterior using a conditional flow backend.

    Parameters are grouped below by concern. Most callers should construct this
    object through :func:`petitRADTRANS.sbi.config.build_conditional_flow_posterior`
    (which expands the grouped config dataclasses), but every knob is also a direct
    constructor argument.

    Parameters
    ----------
    parameter_dim:
        Number of inferred free parameters represented by the posterior. The only
        required argument.

    flow_family:
        Conditional density-transform family (see
        :mod:`petitRADTRANS.sbi.flows.transforms` for the trade-offs). One of
        ``'spline'`` (default; rational-quadratic splines, analytic inverse),
        ``'affine'`` (cheapest, smooth posteriors), ``'autoregressive'`` (affine
        autoregressive), or ``'neural_autoregressive'`` (most expressive, iterative
        inverse).
    num_coupling_layers:
        Number of flow transform layers. More layers increase capacity at higher
        cost.
    hidden_dim:
        Hidden width of the flow conditioner MLPs (and the default for the
        encoder's hidden width).
    conditioner_depth:
        Number of hidden layers in each flow conditioner MLP.
    num_spline_bins:
        Number of rational-quadratic bins per transformed dimension
        (``flow_family='spline'``). More bins = finer per-dimension shape.
    spline_bound:
        Half-width of the finite spline support ``[-bound, bound]`` in latent
        coordinates; outside it the map is the identity. For a uniform-prior cube
        flow this is automatically raised to cover the logit-of-cube range.
    base_distribution:
        Flow base density, ``'gaussian'`` (default) or ``'logistic'``. ``'logistic'``
        matches logit-of-uniform coordinates so an untrained cube flow already
        equals the prior.
    use_base_affine:
        If ``True`` (spline family), prepend a :class:`ConditionalDiagonalAffine`
        layer so the flow can set each parameter's location/scale directly -- helps
        directions constrained far more tightly than the prior.
    autoregressive_transform_units:
        Number of monotonic mixture units per scalar map
        (``flow_family='neural_autoregressive'``); higher = more flexible scalar
        transforms.
    neural_autoregressive_min_slope:
        Positive floor on the neural-autoregressive monotonic slopes (keeps the map
        strictly increasing and well-conditioned).
    neural_autoregressive_min_residual:
        Identity-residual fraction in ``(0, 1)`` blended into each neural
        autoregressive scalar map; keeps it invertible as the posterior sharpens.
    neural_autoregressive_inverse_bisection_steps:
        Bisection steps used to numerically invert each neural-autoregressive scalar
        map when sampling.

    parameter_space:
        Coordinates the flow is trained in: ``'physical'``, ``'cube'`` (unit
        hypercube / uniform-prior coordinates, with the change-of-variables
        correction applied), or ``'unconstrained'`` (default). ``'elbo'`` training
        requires ``'cube'``.
    training_objective:
        ``'npe'`` (default) trains the conditional negative log-likelihood;
        ``'elbo'`` trains an amortized variational objective and requires
        ``parameter_space='cube'`` plus an ``elbo_log_likelihood_fn`` at
        :meth:`fit`.
    learning_rate:
        Peak optimizer (AdamW) learning rate.
    batch_size:
        Number of simulations per optimization step.
    num_epochs:
        Maximum number of passes over the training split.
    weight_decay:
        AdamW weight-decay coefficient.
    gradient_clip_norm:
        Global gradient-norm clip value; ``None`` disables clipping.
    use_cosine_schedule:
        Enable cosine learning-rate decay (otherwise the rate is constant after
        warmup).
    warmup_fraction:
        Fraction of the schedule spent linearly warming up the learning rate (used
        when ``warmup_epochs`` is not given).
    warmup_epochs:
        Absolute number of warmup epochs; overrides ``warmup_fraction`` when set.
    min_learning_rate:
        Floor of the learning-rate schedule.
    lr_schedule_total_epochs:
        Horizon (in epochs) over which the schedule decays; defaults to
        ``num_epochs``.
    early_stopping_patience:
        Number of non-improving epochs tolerated before stopping; ``None`` disables
        early stopping.
    early_stopping_min_delta:
        Minimum validation improvement that counts as progress for early stopping.

    elbo_num_samples:
        Reparameterized posterior draws per observation used to estimate the ELBO
        (``training_objective='elbo'``).
    elbo_beta_start, elbo_beta_end:
        Start/end of the likelihood-temperature (``beta``) anneal, each in
        ``(0, 1]``. Ending below 1 keeps the amortized fit out of the
        noise-dominated, over-confident regime.
    elbo_anneal_fraction:
        Fraction of epochs over which ``beta`` anneals log-linearly from
        ``elbo_beta_start`` to ``elbo_beta_end`` before holding.

    aux_scale_loss_weight:
        Weight of the auxiliary loss training the ``scale_head`` to predict each
        observation's log-amplitude; ``0`` disables it.
    spline_small_bin_regularization_weight:
        Weight of the penalty discouraging collapsed spline bins. Defaults to
        ``spline_entropy_regularization_weight`` when ``None``.
    spline_min_bin_ratio_target:
        Minimum bin-size ratio (relative to a uniform bin) below which the small-bin
        penalty activates.
    spline_entropy_regularization_weight:
        Legacy alias; supplies the default for
        ``spline_small_bin_regularization_weight``.
    spline_derivative_regularization_weight:
        Weight of the penalty keeping spline slopes within the target derivative
        range.
    spline_entropy_floor:
        Target floor (in ``[0, 1]``) for the normalised spline-bin entropy.
    spline_min_derivative_target, spline_max_derivative_target:
        Target range for spline boundary slopes. ``spline_max_derivative_target`` is
        also used as the hard slope cap that prevents unbounded-density collapse;
        ``max`` must exceed ``min``.
    invertibility_loss_weight:
        Weight of the differentiable inverse round-trip / log-det-closure penalty
        that keeps an iterative-inverse flow invertible as it sharpens; ``0``
        disables it.
    invertibility_loss_n_samples:
        Number of base samples used to estimate the invertibility penalty.

    stable_inverse_forward_max_abs_error_threshold:
        Checkpoint-selection gate: maximum tolerated inverse->forward round-trip
        error; ``None`` disables this check.
    stable_inverse_forward_logdet_closure_max_abs_error_threshold:
        Checkpoint-selection gate on the inverse/forward log-det closure error;
        ``None`` disables it.
    stable_cube_edge_hit_rate_threshold:
        Checkpoint-selection gate on the fraction of samples pinned at the cube
        edges (collapse indicator); ``None`` disables it.

    spectrum_encoder_type:
        Spectral encoder architecture: ``'convolution'`` (amplitude/shape
        factorizing).
    n_wavelengths:
        Number of spectral points per observation block; must match the task and
        sizes the encoder.
    spectrum_embedding_dim, photometry_embedding_dim:
        Intermediate embedding sizes of the spectral and photometric sub-encoders.
    embedding_dim:
        Size of the final joint observation embedding consumed by the flow as its
        conditioning context.
    encoder_hidden_dim:
        Hidden width of the encoder networks; defaults to ``hidden_dim`` when
        ``None``.

    checkpoint_directory:
        Directory for resumable trainer checkpoints; ``None`` disables
        checkpointing.
    checkpoint_backend:
        Checkpoint backend. ``'auto'`` prefers Orbax and falls back to Equinox
        serialization.
    resume_from_checkpoint:
        Whether :meth:`fit` should resume from the latest checkpoint in
        ``checkpoint_directory``.
    seed:
        Base random seed for flow/encoder initialization and posterior sampling.
    task_metadata:
        Optional user metadata persisted alongside the trained model.
    verbose_diagnostics:
        Emit per-epoch diagnostic artifacts and plots during training.
    diagnostics_output_directory:
        Directory for the verbose diagnostic artifacts; ``None`` uses the trainer
        default.
    diagnostics_plot_interval:
        Epoch interval between verbose diagnostic plots.

    Notes
    -----
    The posterior stores task fingerprinting, observation schema, and
    preprocessing payload information when those are available from the training
    dataset reader. That metadata is later reused by inference and artifact
    registration paths.
    """

    estimator_family = "conditional_flow"

    def __init__(
        self,
        parameter_dim: int,
        embedding_dim: int = 128,
        num_coupling_layers: int = 4,
        hidden_dim: int = 128,
        conditioner_depth: int = 2,
        autoregressive_transform_units: int = 16,
        neural_autoregressive_min_slope: float = 1e-3,
        neural_autoregressive_min_residual: float = 0.05,
        neural_autoregressive_inverse_bisection_steps: int = 48,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 5,
        parameter_space: str = "unconstrained",
        flow_family: str = "spline",
        num_spline_bins: int = 8,
        spline_bound: float = 10.0,
        base_distribution: str = "gaussian",
        use_base_affine: bool = False,
        training_objective: str = "npe",
        elbo_num_samples: int = 1,
        elbo_beta_start: float = 1e-3,
        elbo_beta_end: float = 1.0,
        elbo_anneal_fraction: float = 0.6,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        checkpoint_directory: str | None = None,
        checkpoint_backend: str = "auto",
        resume_from_checkpoint: bool = False,
        gradient_clip_norm: float | None = 1.0,
        aux_scale_loss_weight: float = 0.0,
        spline_small_bin_regularization_weight: float | None = None,
        spline_min_bin_ratio_target: float = 0.2,
        spline_entropy_regularization_weight: float = 0.0,
        spline_derivative_regularization_weight: float = 0.0,
        spline_entropy_floor: float = 0.6,
        spline_min_derivative_target: float = 0.25,
        spline_max_derivative_target: float = 5.0,
        invertibility_loss_weight: float = 0.0,
        invertibility_loss_n_samples: int = 64,
        weight_decay: float = 0.0,
        use_cosine_schedule: bool = False,
        warmup_fraction: float = 0.02,
        warmup_epochs: float | None = None,
        min_learning_rate: float = 1e-6,
        lr_schedule_total_epochs: float | None = None,
        stable_inverse_forward_max_abs_error_threshold: float | None = 1e-4,
        stable_inverse_forward_logdet_closure_max_abs_error_threshold: float | None = 1e-4,
        stable_cube_edge_hit_rate_threshold: float | None = 0.0,
        spectrum_encoder_type: str = "convolution",
        n_wavelengths: int = 233,
        spectrum_embedding_dim: int = 64,
        photometry_embedding_dim: int = 64,
        encoder_hidden_dim: int | None = None,
        seed: int = 0,
        task_metadata: Mapping[str, Any] | None = None,
        verbose_diagnostics: bool = False,
        diagnostics_output_directory: str | None = None,
        diagnostics_plot_interval: int = 1,
    ):
        super().__init__(
            parameter_dim=parameter_dim,
            parameter_space=parameter_space,
            seed=seed,
            task_metadata=task_metadata,
        )
        self.embedding_dim = int(embedding_dim)
        self.num_coupling_layers = int(num_coupling_layers)
        self.hidden_dim = int(hidden_dim)
        self.conditioner_depth = int(conditioner_depth)
        self.autoregressive_transform_units = int(autoregressive_transform_units)
        self.neural_autoregressive_min_slope = float(
            neural_autoregressive_min_slope
        )
        self.neural_autoregressive_min_residual = float(
            neural_autoregressive_min_residual
        )
        self.neural_autoregressive_inverse_bisection_steps = int(
            neural_autoregressive_inverse_bisection_steps
        )
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.flow_family = str(flow_family)
        self.effective_flow_family = self.flow_family
        self.base_distribution = str(base_distribution)
        self.use_base_affine = bool(use_base_affine)
        self.training_objective = str(training_objective).lower()
        if self.training_objective not in {"npe", "elbo"}:
            raise ValueError(
                "training_objective must be 'npe' (default, conditional NLL) "
                f"or 'elbo' (amortized variational); got {training_objective!r}"
            )
        if self.training_objective == "elbo" and self.parameter_space != "cube":
            raise ValueError(
                "training_objective='elbo' requires parameter_space='cube' so the "
                "prior is uniform on the unit hypercube and the entropy/likelihood "
                "terms are well defined."
            )
        self.elbo_num_samples = max(int(elbo_num_samples), 1)
        self.elbo_beta_start = float(elbo_beta_start)
        self.elbo_beta_end = float(elbo_beta_end)
        self.elbo_anneal_fraction = float(elbo_anneal_fraction)
        if not (0.0 < self.elbo_beta_start <= 1.0):
            raise ValueError("elbo_beta_start must be in (0, 1]")
        if not (0.0 < self.elbo_beta_end <= 1.0):
            raise ValueError("elbo_beta_end must be in (0, 1]")
        if not (0.0 <= self.elbo_anneal_fraction <= 1.0):
            raise ValueError("elbo_anneal_fraction must be in [0, 1]")
        self.num_spline_bins = int(num_spline_bins)
        requested_spline_bound = float(spline_bound)
        if self.parameter_space == "cube" and self.flow_family == "spline":
            self.spline_bound = max(
                requested_spline_bound,
                _minimum_cube_spline_bound(jnp.float32),
            )
        else:
            self.spline_bound = requested_spline_bound
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_backend = checkpoint_backend
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.gradient_clip_norm = gradient_clip_norm
        self.aux_scale_loss_weight = float(aux_scale_loss_weight)
        self.spline_entropy_regularization_weight = float(
            spline_entropy_regularization_weight
        )
        if spline_small_bin_regularization_weight is None:
            spline_small_bin_regularization_weight = self.spline_entropy_regularization_weight
        self.spline_small_bin_regularization_weight = float(
            spline_small_bin_regularization_weight
        )
        self.spline_derivative_regularization_weight = float(
            spline_derivative_regularization_weight
        )
        self.spline_min_bin_ratio_target = float(spline_min_bin_ratio_target)
        self.spline_entropy_floor = float(np.clip(spline_entropy_floor, 0.0, 1.0))
        self.spline_min_derivative_target = float(spline_min_derivative_target)
        self.spline_max_derivative_target = float(spline_max_derivative_target)
        self.invertibility_loss_weight = float(invertibility_loss_weight)
        self.invertibility_loss_n_samples = int(invertibility_loss_n_samples)
        self.weight_decay = float(weight_decay)
        self.use_cosine_schedule = bool(use_cosine_schedule)
        self.warmup_fraction = float(warmup_fraction)
        self.warmup_epochs = (
            None if warmup_epochs is None else float(warmup_epochs)
        )
        self.min_learning_rate = float(min_learning_rate)
        self.lr_schedule_total_epochs = (
            None
            if lr_schedule_total_epochs is None
            else float(lr_schedule_total_epochs)
        )
        self.stable_inverse_forward_max_abs_error_threshold = (
            None
            if stable_inverse_forward_max_abs_error_threshold is None
            else float(stable_inverse_forward_max_abs_error_threshold)
        )
        self.stable_inverse_forward_logdet_closure_max_abs_error_threshold = (
            None
            if stable_inverse_forward_logdet_closure_max_abs_error_threshold is None
            else float(stable_inverse_forward_logdet_closure_max_abs_error_threshold)
        )
        self.stable_cube_edge_hit_rate_threshold = (
            None
            if stable_cube_edge_hit_rate_threshold is None
            else float(stable_cube_edge_hit_rate_threshold)
        )
        self.spectrum_encoder_type = str(spectrum_encoder_type)
        self.n_wavelengths = int(n_wavelengths)
        self.spectrum_embedding_dim = int(spectrum_embedding_dim)
        self.photometry_embedding_dim = int(photometry_embedding_dim)
        self.encoder_hidden_dim = (
            int(encoder_hidden_dim) if encoder_hidden_dim is not None else self.hidden_dim
        )
        self.verbose_diagnostics = bool(verbose_diagnostics)
        self.diagnostics_output_directory = diagnostics_output_directory
        self.diagnostics_plot_interval = int(diagnostics_plot_interval)

        if self.spline_small_bin_regularization_weight < 0.0:
            raise ValueError("spline_small_bin_regularization_weight must be non-negative")
        if self.spline_derivative_regularization_weight < 0.0:
            raise ValueError("spline_derivative_regularization_weight must be non-negative")
        if self.invertibility_loss_weight < 0.0:
            raise ValueError("invertibility_loss_weight must be non-negative")
        if self.invertibility_loss_n_samples <= 0:
            raise ValueError("invertibility_loss_n_samples must be positive")
        if self.spline_min_bin_ratio_target <= 0.0:
            raise ValueError("spline_min_bin_ratio_target must be positive")
        if self.spline_min_derivative_target <= 0.0:
            raise ValueError("spline_min_derivative_target must be positive")
        if self.spline_max_derivative_target <= 0.0:
            raise ValueError("spline_max_derivative_target must be positive")
        if self.spline_max_derivative_target <= self.spline_min_derivative_target:
            raise ValueError(
                "spline_max_derivative_target must exceed spline_min_derivative_target"
            )
        if self.autoregressive_transform_units <= 0:
            raise ValueError("autoregressive_transform_units must be positive")
        if self.neural_autoregressive_min_slope <= 0.0:
            raise ValueError("neural_autoregressive_min_slope must be positive")
        if self.warmup_epochs is not None and self.warmup_epochs < 0.0:
            raise ValueError("warmup_epochs must be non-negative when provided")
        if (
            self.lr_schedule_total_epochs is not None
            and self.lr_schedule_total_epochs <= 0.0
        ):
            raise ValueError(
                "lr_schedule_total_epochs must be positive when provided"
            )
        if not 0.0 < self.neural_autoregressive_min_residual < 1.0:
            raise ValueError(
                "neural_autoregressive_min_residual must lie in (0, 1)"
            )
        if self.neural_autoregressive_inverse_bisection_steps <= 0:
            raise ValueError(
                "neural_autoregressive_inverse_bisection_steps must be positive"
            )
        if (
            self.stable_inverse_forward_max_abs_error_threshold is not None
            and self.stable_inverse_forward_max_abs_error_threshold < 0.0
        ):
            raise ValueError(
                "stable_inverse_forward_max_abs_error_threshold must be non-negative"
            )
        if (
            self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
            is not None
            and self.stable_inverse_forward_logdet_closure_max_abs_error_threshold < 0.0
        ):
            raise ValueError(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold must be non-negative"
            )
        if (
            self.stable_cube_edge_hit_rate_threshold is not None
            and self.stable_cube_edge_hit_rate_threshold < 0.0
        ):
            raise ValueError("stable_cube_edge_hit_rate_threshold must be non-negative")

        encoder_key, flow_key, scale_key = random.split(
            random.PRNGKey(self.seed), 3
        )
        self.model = _PosteriorModel(
            encoder=ObservationEncoder(
                embedding_dim=self.embedding_dim,
                spectrum_embedding_dim=self.spectrum_embedding_dim,
                photometry_embedding_dim=self.photometry_embedding_dim,
                hidden_dim=self.encoder_hidden_dim,
                spectrum_encoder_type=self.spectrum_encoder_type,
                n_wavelengths=self.n_wavelengths,
                key=encoder_key,
            ),
            flow=self._build_flow(flow_key),
            scale_head=eqx.nn.Linear(
                self.embedding_dim,
                1,
                key=scale_key,
                dtype=jnp.float32,
            ),
            aux_scale_loss_weight=self.aux_scale_loss_weight,
            spline_small_bin_regularization_weight=self.spline_small_bin_regularization_weight,
            spline_derivative_regularization_weight=self.spline_derivative_regularization_weight,
            spline_min_bin_ratio_target=self.spline_min_bin_ratio_target,
            spline_min_derivative_target=self.spline_min_derivative_target,
            spline_max_derivative_target=self.spline_max_derivative_target,
            invertibility_loss_weight=self.invertibility_loss_weight,
            invertibility_loss_n_samples=self.invertibility_loss_n_samples,
        )

    def _build_flow(self, key: Array) -> Any:
        if self.flow_family == "affine":
            self.effective_flow_family = "affine"
            return ConditionalAffineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                key=key,
            )
        if self.flow_family == "autoregressive":
            self.effective_flow_family = "autoregressive"
            return ConditionalAutoregressiveFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                key=key,
            )
        if self.flow_family in {"neural_autoregressive", "naf"}:
            self.effective_flow_family = "neural_autoregressive"
            return ConditionalNeuralAutoregressiveFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                transform_units=self.autoregressive_transform_units,
                min_slope=self.neural_autoregressive_min_slope,
                min_residual=self.neural_autoregressive_min_residual,
                inverse_bisection_steps=self.neural_autoregressive_inverse_bisection_steps,
                key=key,
            )
        if self.flow_family == "spline":
            self.effective_flow_family = "spline"
            return ConditionalSplineFlow(
                parameter_dim=self.parameter_dim,
                context_dim=self.embedding_dim,
                num_layers=self.num_coupling_layers,
                hidden_dim=self.hidden_dim,
                conditioner_depth=self.conditioner_depth,
                num_bins=self.num_spline_bins,
                bound=self.spline_bound,
                # Hard cap on the spline boundary derivatives (reusing the
                # derivative-target knob): structurally bounds the spline slope
                # and hence the density, preventing the unbounded-density
                # collapse, and keeping the analytic inverse numerically valid.
                max_derivative=self.spline_max_derivative_target,
                # rec. 3: a logistic base matches logit-of-uniform so the
                # untrained flow equals the prior (no cube-edge piling).
                base_distribution=self.base_distribution,
                # rec. 2: a leading conditional diagonal affine lets the flow
                # scale sharp directions (fixed-Rs amplitude) without narrow bins.
                use_base_affine=self.use_base_affine,
                key=key,
            )
        raise ValueError(f"Unsupported flow_family: {self.flow_family!r}")

    def _validation_diagnostics(
        self, model: _PosteriorModel, batch: PosteriorBatch
    ) -> dict[str, float]:
        """Per-checkpoint flow invertibility / collapse diagnostics.

        Thin wrapper around
        :func:`petitRADTRANS.sbi.flows.objectives.compute_validation_diagnostics`,
        forwarding this estimator's ``verbose_diagnostics`` flag.
        """
        return compute_validation_diagnostics(
            model, batch, verbose_diagnostics=self.verbose_diagnostics
        )

    def refine_posterior_on_observation(
        self, *args: Any, **kwargs: Any
    ) -> PosteriorSamples:
        """Test-time flow-ELBO posterior refinement for one observation.

        Thin wrapper around
        :func:`petitRADTRANS.sbi.flows.refinement.refine_posterior_on_observation`,
        which holds the implementation.
        """
        return refinement.refine_posterior_on_observation(self, *args, **kwargs)

    def map_laplace_posterior_on_observation(
        self, *args: Any, **kwargs: Any
    ) -> PosteriorSamples:
        """Test-time MAP + Laplace / Gaussian-VI posterior refinement.

        Thin wrapper around
        :func:`petitRADTRANS.sbi.flows.refinement.map_laplace_posterior_on_observation`,
        which holds the implementation.
        """
        return refinement.map_laplace_posterior_on_observation(self, *args, **kwargs)

    def fit(
        self,
        dataset: Any,
        *,
        elbo_log_likelihood_fn: Callable[[jnp.ndarray, Any], jnp.ndarray]
        | None = None,
        elbo_num_samples: int | None = None,
    ) -> TrainingArtifacts:
        """Train the posterior on one normalized simulation dataset reader.

        Parameters
        ----------
        dataset:
            Reader-like object that yields :class:`PosteriorBatch` instances via
            ``iter_batches`` and exposes dataset manifest metadata when
            available.
        elbo_log_likelihood_fn:
            Required when ``training_objective='elbo'``. A differentiable
            ``(theta_cube, observations) -> (batch,)`` callable returning the
            Gaussian observational log-likelihood through the forward model. See
            :meth:`_make_elbo_loss`. Ignored for the default NPE objective.
        elbo_num_samples:
            Number of reparameterized posterior draws per observation for the
            ELBO estimator (default 1).

        Returns
        -------
        TrainingArtifacts
            Training history, validation metrics, and trainer metadata for the
            completed optimization run.

        Notes
        -----
        If the reader exposes preprocessing metadata or a manifest fingerprint,
        those are cached on the posterior so they can be saved and reused by the
        inference and artifact layers.
        """
        manifest = getattr(getattr(dataset, "dataset", None), "manifest", None)
        if manifest is not None:
            self.task_name = str(
                getattr(manifest, "task_name", self.task_name or "unknown_task")
            )
            self.task_fingerprint = getattr(manifest, "task_fingerprint", None)
            self.observation_schema = dict(getattr(manifest, "observation_schema", {}))
        preprocessing_metadata = getattr(dataset, "preprocessing_metadata", None)
        if preprocessing_metadata is not None:
            self.preprocessing_metadata_payload = preprocessing_metadata.to_payload()

        early_stopping = None
        if self.early_stopping_patience is not None:
            early_stopping = EarlyStoppingConfig(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
            )

        trainer = SBITrainer(
            TrainingSchemeConfig(
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                parameter_space=self.parameter_space,
                seed=self.seed,
                shuffle_train=True,
                early_stopping=early_stopping,
                checkpoint_directory=self.checkpoint_directory,
                checkpoint_backend=self.checkpoint_backend,
                resume_from_checkpoint=self.resume_from_checkpoint,
                gradient_clip_norm=self.gradient_clip_norm,
                data_parallel=len(jax.devices()) > 1,
                weight_decay=self.weight_decay,
                use_cosine_schedule=self.use_cosine_schedule,
                warmup_fraction=self.warmup_fraction,
                warmup_epochs=self.warmup_epochs,
                min_learning_rate=self.min_learning_rate,
                lr_schedule_total_epochs=self.lr_schedule_total_epochs,
                verbose_diagnostics=self.verbose_diagnostics,
                diagnostics_output_directory=self.diagnostics_output_directory,
                diagnostics_plot_interval=self.diagnostics_plot_interval,
            )
        )
        if self.training_objective == "elbo":
            if elbo_log_likelihood_fn is None:
                raise ValueError(
                    "training_objective='elbo' requires an elbo_log_likelihood_fn "
                    "(a differentiable forward-model likelihood). Build it from the "
                    "task/runtime and pass it to fit(..., elbo_log_likelihood_fn=...)."
                )
            # The ELBO objective sets posterior width via the likelihood/entropy
            # balance, so observation-noise augmentation and the parameter-noise
            # floor (NPE-only regularizers) are not applied; train and eval use
            # the same ELBO so selection tracks the variational objective.
            resolved_num_samples = (
                self.elbo_num_samples
                if elbo_num_samples is None
                else int(elbo_num_samples)
            )
            elbo_loss_fn = make_elbo_loss(
                elbo_log_likelihood_fn,
                self.parameter_space,
                num_samples=resolved_num_samples,
            )
            train_loss_fn = elbo_loss_fn
            eval_loss_fn = elbo_loss_fn

            # Likelihood-temperature (beta-VI) annealing schedule. beta rises
            # log-linearly from elbo_beta_start to elbo_beta_end over the first
            # elbo_anneal_fraction of epochs, then holds at elbo_beta_end.
            # Ending at beta_end < 1 keeps the amortized phase out of the
            # sigma-dominated regime (sigma_eff = sigma/sqrt(beta_end)), which
            # stops the over-confidence and the run-to-run basin flipping in the
            # razor-sharp depth direction -- the amortized model is then a stable
            # coarse warm-start and the test-time MAP+Laplace supplies the sharp
            # final fit. Evaluation always uses beta = 1 (handled by the
            # trainer), so selection still tracks the true objective.
            _beta_start = self.elbo_beta_start
            _beta_end = self.elbo_beta_end
            _anneal_fraction = self.elbo_anneal_fraction

            def _elbo_beta_schedule(epoch: int, total_epochs: int) -> float:
                if _anneal_fraction <= 0.0:
                    return _beta_end
                anneal_epochs = max(1.0, _anneal_fraction * float(total_epochs))
                progress = min(1.0, float(epoch) / anneal_epochs)
                # log-linear interpolation beta_start -> beta_end
                log_beta = (1.0 - progress) * math.log(_beta_start) + (
                    progress
                ) * math.log(_beta_end)
                return float(math.exp(log_beta))

            loss_conditioning_schedule = _elbo_beta_schedule
            # ELBO is reverse-KL trained, so the *tight* NPE invertibility/edge
            # thresholds (which assume a forward-KL-optimised map) are too strict
            # and would mark every epoch unstable, pinning selection to epoch 1.
            # But a *catastrophically* broken flow -- a forward<->inverse round
            # trip error of O(10) or a log-det closure error of O(100), which
            # happens when the spline bins collapse -- still yields a degenerate,
            # overconfident posterior whose samples are meaningless (observed as
            # uniformly miscalibrated SBC when such a checkpoint was selected).
            # The previous gate only checked loss finiteness and so selected those
            # collapsed flows. ELBO therefore uses a LOOSE invertibility/collapse
            # gate: it passes usable flows but disqualifies the collapsed ones from
            # checkpoint selection (and the shared pathology gate, inverse_error >
            # 0.1 etc., still bars them from resetting early-stop patience).

            def _elbo_selection_metric(
                validation_loss: float, diagnostics: Mapping[str, float] | None
            ) -> float:
                return (
                    float(validation_loss)
                    if np.isfinite(validation_loss)
                    else 1e30
                )

            def _elbo_stability_metrics(
                validation_loss: float, diagnostics: Mapping[str, float] | None
            ) -> Mapping[str, float]:
                # The forward<->inverse round-trip (rt_err / logdet closure) is
                # irrelevant for a reverse-KL, sampling-only flow: that direction
                # is never optimized, and the entropy log-det used in the ELBO is
                # exact in the analytic spline inverse. Gating "stable" on it
                # (thresholds 5 / 10) marked every ELBO epoch unstable, so
                # best_stable never formed and the three checkpoints diverged.
                # For ELBO, stability/pathology depend only on a finite loss and
                # gross edge collapse (the one genuine flow failure here).
                diagnostics = diagnostics or {}
                finite = bool(np.isfinite(validation_loss))
                edge_hit_rate = float(
                    diagnostics.get("cube_edge_hit_rate", 0.0)
                )
                if not np.isfinite(edge_hit_rate):
                    edge_hit_rate = 0.0
                stable = finite and (
                    edge_hit_rate <= _ELBO_STABILITY_CUBE_EDGE_HIT_RATE
                )
                pathological = (not finite) or (edge_hit_rate > 0.9)
                return {
                    "checkpoint_is_stable": 1.0 if stable else 0.0,
                    "checkpoint_is_pathological": 1.0 if pathological else 0.0,
                    "checkpoint_stability_violation_count": (
                        0.0 if stable else 1.0
                    ),
                }

            selection_metric_fn = _elbo_selection_metric
            selection_metric_name = "validation_selection_metric"
            stability_metric_fn = _elbo_stability_metrics
        else:
            # Observational noise is injected once, at simulation time
            # (RuntimeSimulator). Re-injecting it here at training time would
            # double the effective noise the encoder sees relative to the target
            # observation, desensitising it to the true spectral signal -- so the
            # training loss is the plain conditional NLL with no noise/floor
            # augmentation.
            train_loss_fn = training_loss
            eval_loss_fn = compute_loss
            selection_metric_fn = _flow_validation_selection_metric
            selection_metric_name = "validation_selection_metric"
            loss_conditioning_schedule = None

            def stability_metric_fn(
                validation_loss: float, diagnostics: Mapping[str, float] | None
            ) -> Mapping[str, float]:
                return _flow_validation_stability_metrics(
                    validation_loss,
                    diagnostics,
                    inverse_forward_max_abs_error_threshold=(
                        self.stable_inverse_forward_max_abs_error_threshold
                    ),
                    inverse_forward_logdet_closure_max_abs_error_threshold=(
                        self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
                    ),
                    cube_edge_hit_rate_threshold=(
                        self.stable_cube_edge_hit_rate_threshold
                    ),
                )

        self.model, history, validation_metrics, training_metadata = trainer.fit(
            self.model,
            dataset,
            train_loss_fn,
            eval_loss_fn=eval_loss_fn,
            eval_diagnostic_fn=self._validation_diagnostics,
            selection_metric_fn=selection_metric_fn,
            selection_metric_name=selection_metric_name,
            stability_metric_fn=stability_metric_fn,
            stability_flag_key="checkpoint_is_stable",
            loss_conditioning_schedule=loss_conditioning_schedule,
        )
        self.training_artifacts = TrainingArtifacts(
            history=history,
            validation_metrics=validation_metrics,
            metadata={
                "parameter_dim": self.parameter_dim,
                "parameter_space": self.parameter_space,
                "task_metadata": dict(self.task_metadata),
                "task_name": self.task_name,
                "task_fingerprint": self.task_fingerprint,
                "flow_family": self.flow_family,
                "effective_flow_family": self.effective_flow_family,
                "num_spline_bins": self.num_spline_bins,
                "spline_bound": self.spline_bound,
                **training_metadata,
            },
        )
        return self.training_artifacts

    def _build_estimator_config(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "num_coupling_layers": self.num_coupling_layers,
            "hidden_dim": self.hidden_dim,
            "conditioner_depth": self.conditioner_depth,
            "autoregressive_transform_units": self.autoregressive_transform_units,
            "neural_autoregressive_min_slope": self.neural_autoregressive_min_slope,
            "neural_autoregressive_min_residual": self.neural_autoregressive_min_residual,
            "neural_autoregressive_inverse_bisection_steps": self.neural_autoregressive_inverse_bisection_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "flow_family": self.flow_family,
            "effective_flow_family": self.effective_flow_family,
            "num_spline_bins": self.num_spline_bins,
            "spline_bound": self.spline_bound,
            "base_distribution": self.base_distribution,
            "use_base_affine": self.use_base_affine,
            "training_objective": self.training_objective,
            "elbo_num_samples": self.elbo_num_samples,
            "elbo_beta_start": self.elbo_beta_start,
            "elbo_beta_end": self.elbo_beta_end,
            "elbo_anneal_fraction": self.elbo_anneal_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "checkpoint_directory": self.checkpoint_directory,
            "checkpoint_backend": self.checkpoint_backend,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "aux_scale_loss_weight": self.aux_scale_loss_weight,
            "invertibility_loss_weight": self.invertibility_loss_weight,
            "invertibility_loss_n_samples": self.invertibility_loss_n_samples,
            "spline_small_bin_regularization_weight": self.spline_small_bin_regularization_weight,
            "spline_min_bin_ratio_target": self.spline_min_bin_ratio_target,
            "spline_entropy_regularization_weight": self.spline_entropy_regularization_weight,
            "spline_derivative_regularization_weight": self.spline_derivative_regularization_weight,
            "spline_entropy_floor": self.spline_entropy_floor,
            "spline_min_derivative_target": self.spline_min_derivative_target,
            "spline_max_derivative_target": self.spline_max_derivative_target,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "use_cosine_schedule": self.use_cosine_schedule,
            "warmup_fraction": self.warmup_fraction,
            "warmup_epochs": self.warmup_epochs,
            "min_learning_rate": self.min_learning_rate,
            "lr_schedule_total_epochs": self.lr_schedule_total_epochs,
            "stable_inverse_forward_max_abs_error_threshold": (
                self.stable_inverse_forward_max_abs_error_threshold
            ),
            "stable_inverse_forward_logdet_closure_max_abs_error_threshold": (
                self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
            ),
            "stable_cube_edge_hit_rate_threshold": self.stable_cube_edge_hit_rate_threshold,
            "spectrum_encoder_type": self.spectrum_encoder_type,
            "n_wavelengths": self.n_wavelengths,
            "spectrum_embedding_dim": self.spectrum_embedding_dim,
            "photometry_embedding_dim": self.photometry_embedding_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "verbose_diagnostics": self.verbose_diagnostics,
            "diagnostics_output_directory": self.diagnostics_output_directory,
            "diagnostics_plot_interval": self.diagnostics_plot_interval,
        }

    def _build_serialized_metadata(self, artifact_metadata) -> dict[str, Any]:
        metadata = super()._build_serialized_metadata(artifact_metadata)
        metadata.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_coupling_layers": self.num_coupling_layers,
                "hidden_dim": self.hidden_dim,
                "conditioner_depth": self.conditioner_depth,
                "autoregressive_transform_units": self.autoregressive_transform_units,
                "neural_autoregressive_min_slope": self.neural_autoregressive_min_slope,
                "neural_autoregressive_min_residual": self.neural_autoregressive_min_residual,
                "neural_autoregressive_inverse_bisection_steps": self.neural_autoregressive_inverse_bisection_steps,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "flow_family": self.flow_family,
                "effective_flow_family": self.effective_flow_family,
                "num_spline_bins": self.num_spline_bins,
                "spline_bound": self.spline_bound,
                "base_distribution": self.base_distribution,
                "use_base_affine": self.use_base_affine,
                "training_objective": self.training_objective,
                "elbo_num_samples": self.elbo_num_samples,
                "elbo_beta_start": self.elbo_beta_start,
                "elbo_beta_end": self.elbo_beta_end,
                "elbo_anneal_fraction": self.elbo_anneal_fraction,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "checkpoint_directory": self.checkpoint_directory,
                "checkpoint_backend": self.checkpoint_backend,
                "resume_from_checkpoint": self.resume_from_checkpoint,
                "aux_scale_loss_weight": self.aux_scale_loss_weight,
                "invertibility_loss_weight": self.invertibility_loss_weight,
                "invertibility_loss_n_samples": self.invertibility_loss_n_samples,
                "spline_small_bin_regularization_weight": self.spline_small_bin_regularization_weight,
                "spline_min_bin_ratio_target": self.spline_min_bin_ratio_target,
                "spline_entropy_regularization_weight": self.spline_entropy_regularization_weight,
                "spline_derivative_regularization_weight": self.spline_derivative_regularization_weight,
                "spline_entropy_floor": self.spline_entropy_floor,
                "spline_min_derivative_target": self.spline_min_derivative_target,
                "spline_max_derivative_target": self.spline_max_derivative_target,
                "weight_decay": self.weight_decay,
                "gradient_clip_norm": self.gradient_clip_norm,
                "use_cosine_schedule": self.use_cosine_schedule,
                "warmup_fraction": self.warmup_fraction,
                "warmup_epochs": self.warmup_epochs,
                "min_learning_rate": self.min_learning_rate,
                "lr_schedule_total_epochs": self.lr_schedule_total_epochs,
                "stable_inverse_forward_max_abs_error_threshold": (
                    self.stable_inverse_forward_max_abs_error_threshold
                ),
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold": (
                    self.stable_inverse_forward_logdet_closure_max_abs_error_threshold
                ),
                "stable_cube_edge_hit_rate_threshold": self.stable_cube_edge_hit_rate_threshold,
                "spectrum_encoder_type": self.spectrum_encoder_type,
                "n_wavelengths": self.n_wavelengths,
                "spectrum_embedding_dim": self.spectrum_embedding_dim,
                "photometry_embedding_dim": self.photometry_embedding_dim,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "verbose_diagnostics": self.verbose_diagnostics,
                "diagnostics_output_directory": self.diagnostics_output_directory,
                "diagnostics_plot_interval": self.diagnostics_plot_interval,
            }
        )
        return metadata

    @staticmethod
    def _resolve_estimator_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
        estimator_config = metadata.get("estimator_config")
        if isinstance(estimator_config, Mapping):
            return dict(estimator_config)

        small_bin_weight = metadata.get("spline_small_bin_regularization_weight")
        if small_bin_weight is None:
            small_bin_weight = metadata.get("spline_entropy_regularization_weight", 0.0)

        return {
            "embedding_dim": metadata["embedding_dim"],
            "num_coupling_layers": metadata["num_coupling_layers"],
            "hidden_dim": metadata["hidden_dim"],
            "learning_rate": metadata["learning_rate"],
            "batch_size": metadata["batch_size"],
            "num_epochs": metadata["num_epochs"],
            "autoregressive_transform_units": metadata.get(
                "autoregressive_transform_units",
                16,
            ),
            "neural_autoregressive_min_slope": metadata.get(
                "neural_autoregressive_min_slope",
                1e-3,
            ),
            "neural_autoregressive_min_residual": metadata.get(
                "neural_autoregressive_min_residual",
                0.05,
            ),
            "neural_autoregressive_inverse_bisection_steps": metadata.get(
                "neural_autoregressive_inverse_bisection_steps",
                48,
            ),
            "flow_family": metadata.get("flow_family", "affine"),
            "effective_flow_family": metadata.get(
                "effective_flow_family", metadata.get("flow_family", "affine")
            ),
            "num_spline_bins": metadata.get("num_spline_bins", 8),
            "spline_bound": metadata.get("spline_bound", 10.0),
            "base_distribution": metadata.get("base_distribution", "gaussian"),
            "use_base_affine": metadata.get("use_base_affine", False),
            "training_objective": metadata.get("training_objective", "npe"),
            "elbo_num_samples": metadata.get("elbo_num_samples", 1),
            "elbo_beta_start": metadata.get("elbo_beta_start", 1e-3),
            "elbo_beta_end": metadata.get("elbo_beta_end", 1.0),
            "elbo_anneal_fraction": metadata.get("elbo_anneal_fraction", 0.6),
            "early_stopping_patience": metadata.get("early_stopping_patience"),
            "early_stopping_min_delta": metadata.get("early_stopping_min_delta", 0.0),
            "checkpoint_directory": metadata.get("checkpoint_directory"),
            "checkpoint_backend": metadata.get("checkpoint_backend", "auto"),
            "resume_from_checkpoint": metadata.get("resume_from_checkpoint", False),
            "conditioner_depth": metadata.get("conditioner_depth", 2),
            "aux_scale_loss_weight": metadata.get("aux_scale_loss_weight", 0.0),
            "invertibility_loss_weight": metadata.get("invertibility_loss_weight", 0.0),
            "invertibility_loss_n_samples": metadata.get(
                "invertibility_loss_n_samples", 64
            ),
            "spline_small_bin_regularization_weight": small_bin_weight,
            "spline_min_bin_ratio_target": metadata.get("spline_min_bin_ratio_target", 0.2),
            "spline_entropy_regularization_weight": metadata.get(
                "spline_entropy_regularization_weight",
                0.0,
            ),
            "spline_derivative_regularization_weight": metadata.get(
                "spline_derivative_regularization_weight",
                0.0,
            ),
            "spline_entropy_floor": metadata.get("spline_entropy_floor", 0.6),
            "spline_min_derivative_target": metadata.get(
                "spline_min_derivative_target",
                0.25,
            ),
            "spline_max_derivative_target": metadata.get(
                "spline_max_derivative_target",
                5.0,
            ),
            "weight_decay": metadata.get("weight_decay", 0.0),
            "gradient_clip_norm": metadata.get("gradient_clip_norm", 1.0),
            "use_cosine_schedule": metadata.get("use_cosine_schedule", False),
            "warmup_fraction": metadata.get("warmup_fraction", 0.02),
            "warmup_epochs": metadata.get("warmup_epochs"),
            "min_learning_rate": metadata.get("min_learning_rate", 1e-6),
            "lr_schedule_total_epochs": metadata.get(
                "lr_schedule_total_epochs"
            ),
            "stable_inverse_forward_max_abs_error_threshold": metadata.get(
                "stable_inverse_forward_max_abs_error_threshold",
                1e-4,
            ),
            "stable_inverse_forward_logdet_closure_max_abs_error_threshold": metadata.get(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold",
                1e-4,
            ),
            "stable_cube_edge_hit_rate_threshold": metadata.get(
                "stable_cube_edge_hit_rate_threshold",
                0.0,
            ),
            "spectrum_encoder_type": metadata.get("spectrum_encoder_type", "convolution"),
            "n_wavelengths": metadata.get("n_wavelengths", 233),
            "spectrum_embedding_dim": metadata.get("spectrum_embedding_dim", 64),
            "photometry_embedding_dim": metadata.get("photometry_embedding_dim", 64),
            "encoder_hidden_dim": metadata.get("encoder_hidden_dim"),
            "verbose_diagnostics": metadata.get("verbose_diagnostics", False),
            "diagnostics_output_directory": metadata.get("diagnostics_output_directory"),
            "diagnostics_plot_interval": metadata.get("diagnostics_plot_interval", 1),
        }

    def hydrate_loaded_metadata(self, metadata: Mapping[str, Any]) -> None:
        super().hydrate_loaded_metadata(metadata)
        self.effective_flow_family = str(
            self._resolve_estimator_config(metadata).get(
                "effective_flow_family", self.effective_flow_family
            )
        )

    @classmethod
    def from_serialized_metadata(
        cls, metadata: Mapping[str, Any]
    ) -> "ConditionalFlowPosterior":
        estimator_config = cls._resolve_estimator_config(metadata)
        return cls(
            parameter_dim=int(metadata["parameter_dim"]),
            embedding_dim=int(estimator_config["embedding_dim"]),
            num_coupling_layers=int(estimator_config["num_coupling_layers"]),
            hidden_dim=int(estimator_config["hidden_dim"]),
            conditioner_depth=int(estimator_config.get("conditioner_depth", 2)),
            autoregressive_transform_units=int(
                estimator_config.get("autoregressive_transform_units", 16)
            ),
            neural_autoregressive_min_slope=float(
                estimator_config.get("neural_autoregressive_min_slope", 1e-3)
            ),
            neural_autoregressive_min_residual=float(
                estimator_config.get("neural_autoregressive_min_residual", 0.05)
            ),
            neural_autoregressive_inverse_bisection_steps=int(
                estimator_config.get(
                    "neural_autoregressive_inverse_bisection_steps",
                    48,
                )
            ),
            learning_rate=float(estimator_config["learning_rate"]),
            batch_size=int(estimator_config["batch_size"]),
            num_epochs=int(estimator_config["num_epochs"]),
            parameter_space=str(metadata["parameter_space"]),
            flow_family=str(estimator_config.get("flow_family", "affine")),
            num_spline_bins=int(estimator_config.get("num_spline_bins", 8)),
            spline_bound=float(estimator_config.get("spline_bound", 10.0)),
            base_distribution=str(estimator_config.get("base_distribution", "gaussian")),
            use_base_affine=bool(estimator_config.get("use_base_affine", False)),
            training_objective=str(estimator_config.get("training_objective", "npe")),
            elbo_num_samples=int(estimator_config.get("elbo_num_samples", 1)),
            elbo_beta_start=float(estimator_config.get("elbo_beta_start", 1e-3)),
            elbo_beta_end=float(estimator_config.get("elbo_beta_end", 1.0)),
            elbo_anneal_fraction=float(estimator_config.get("elbo_anneal_fraction", 0.6)),
            early_stopping_patience=estimator_config.get("early_stopping_patience"),
            early_stopping_min_delta=float(
                estimator_config.get("early_stopping_min_delta", 0.0)
            ),
            checkpoint_directory=estimator_config.get("checkpoint_directory"),
            checkpoint_backend=str(estimator_config.get("checkpoint_backend", "auto")),
            resume_from_checkpoint=bool(
                estimator_config.get("resume_from_checkpoint", False)
            ),
            aux_scale_loss_weight=float(
                estimator_config.get("aux_scale_loss_weight", 0.0)
            ),
            invertibility_loss_weight=float(
                estimator_config.get("invertibility_loss_weight", 0.0)
            ),
            invertibility_loss_n_samples=int(
                estimator_config.get("invertibility_loss_n_samples", 64)
            ),
            spline_small_bin_regularization_weight=float(
                estimator_config.get(
                    "spline_small_bin_regularization_weight",
                    estimator_config.get("spline_entropy_regularization_weight", 0.0),
                )
            ),
            spline_min_bin_ratio_target=float(
                estimator_config.get("spline_min_bin_ratio_target", 0.2)
            ),
            spline_entropy_regularization_weight=float(
                estimator_config.get("spline_entropy_regularization_weight", 0.0)
            ),
            spline_derivative_regularization_weight=float(
                estimator_config.get("spline_derivative_regularization_weight", 0.0)
            ),
            spline_entropy_floor=float(estimator_config.get("spline_entropy_floor", 0.6)),
            spline_min_derivative_target=float(
                estimator_config.get("spline_min_derivative_target", 0.25)
            ),
            spline_max_derivative_target=float(
                estimator_config.get("spline_max_derivative_target", 5.0)
            ),
            weight_decay=float(estimator_config.get("weight_decay", 0.0)),
            gradient_clip_norm=estimator_config.get("gradient_clip_norm", 1.0),
            use_cosine_schedule=bool(estimator_config.get("use_cosine_schedule", False)),
            warmup_fraction=float(estimator_config.get("warmup_fraction", 0.02)),
            warmup_epochs=(
                None
                if estimator_config.get("warmup_epochs") is None
                else float(estimator_config.get("warmup_epochs"))
            ),
            min_learning_rate=float(estimator_config.get("min_learning_rate", 1e-6)),
            lr_schedule_total_epochs=(
                None
                if estimator_config.get("lr_schedule_total_epochs") is None
                else float(estimator_config.get("lr_schedule_total_epochs"))
            ),
            stable_inverse_forward_max_abs_error_threshold=estimator_config.get(
                "stable_inverse_forward_max_abs_error_threshold",
                1e-4,
            ),
            stable_inverse_forward_logdet_closure_max_abs_error_threshold=estimator_config.get(
                "stable_inverse_forward_logdet_closure_max_abs_error_threshold",
                1e-4,
            ),
            stable_cube_edge_hit_rate_threshold=estimator_config.get(
                "stable_cube_edge_hit_rate_threshold",
                0.0,
            ),
            spectrum_encoder_type=str(estimator_config.get("spectrum_encoder_type", "convolution")),
            n_wavelengths=int(estimator_config.get("n_wavelengths", 233)),
            spectrum_embedding_dim=int(estimator_config.get("spectrum_embedding_dim", 64)),
            photometry_embedding_dim=int(estimator_config.get("photometry_embedding_dim", 64)),
            encoder_hidden_dim=estimator_config.get("encoder_hidden_dim"),
            verbose_diagnostics=bool(estimator_config.get("verbose_diagnostics", False)),
            diagnostics_output_directory=estimator_config.get("diagnostics_output_directory"),
            diagnostics_plot_interval=int(estimator_config.get("diagnostics_plot_interval", 1)),
            seed=int(metadata["seed"]),
            task_metadata=dict(metadata.get("task_metadata", {})),
        )

    def save_backend_state(self, output_path: Path) -> None:
        eqx.tree_serialise_leaves(output_path / "model.eqx", self.model)

    def load_backend_state(self, input_path: Path) -> None:
        self.model = eqx.tree_deserialise_leaves(input_path / "model.eqx", self.model)

    def encode_observation(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        """Encode one structured observation into the posterior context space.

        Parameters
        ----------
        blocks:
            Observation blocks describing one spectral/photometric observation.

        Returns
        -------
        EncodedObservation
            Aggregated embedding and lightweight metadata describing the input
            observation family.
        """
        return self.model.encoder.encode(blocks)

    def batch_encode_observation(
        self, blocks_list: list[list[ObservationBlock]]
    ) -> list[EncodedObservation]:
        """Encode a batch of observations using a single vmapped forward pass.

        Parameters
        ----------
        blocks_list:
            List of per-sample observation block lists.

        Returns
        -------
        list[EncodedObservation]
            One encoded observation per input sample.
        """
        embeddings = self.model.encoder.encode_stacked_batch(blocks_list)
        return [
            EncodedObservation(embedding=embeddings[i]) for i in range(len(blocks_list))
        ]

    def sample_posterior(
        self,
        observation: EncodedObservation,
        n_samples: int,
        seed: int | None = None,
    ) -> PosteriorSamples:
        """Draw posterior samples conditioned on one encoded observation.

        Parameters
        ----------
        observation:
            Encoded observation produced by :meth:`encode_observation`.
        n_samples:
            Number of posterior draws to generate.
        seed:
            Optional random seed overriding the model-level default seed.

        Returns
        -------
        PosteriorSamples
            Samples in the posterior's configured parameter space. Non-finite
            outputs are clamped to large finite values for downstream stability.
        """
        sample_seed = self.seed if seed is None else int(seed)
        # Test-time refinement override: if MAP+Laplace (or another per-target
        # method) attached fixed cube samples, return draws from those instead of
        # the amortized flow. The flow is not specialized to the target in that
        # case; the fixed samples already encode the target-specific posterior.
        fixed_cube_samples = getattr(self, "_fixed_cube_samples", None)
        if fixed_cube_samples is not None:
            fixed_cube_samples = np.asarray(fixed_cube_samples)
            rng = np.random.default_rng(sample_seed)
            pool_size = fixed_cube_samples.shape[0]
            if int(n_samples) <= pool_size:
                indices = rng.choice(pool_size, size=int(n_samples), replace=False)
            else:
                indices = rng.integers(0, pool_size, size=int(n_samples))
            return PosteriorSamples(
                samples=np.nan_to_num(
                    fixed_cube_samples[indices], nan=0.0, posinf=1e6, neginf=-1e6
                )
            )
        samples = self.model.flow.sample(
            context=jnp.asarray(observation.embedding, dtype=jnp.float32),
            n_samples=n_samples,
            key=random.PRNGKey(sample_seed),
        )
        if self.parameter_space == "cube":
            samples = _clip_cube_support(nn.sigmoid(samples))
        return PosteriorSamples(
            samples=np.nan_to_num(np.asarray(samples), nan=0.0, posinf=1e6, neginf=-1e6)
        )

    def batch_sample_posterior(
        self,
        embeddings: Any,
        n_samples: int,
        base_seed: int = 0,
    ) -> np.ndarray:
        """Draw posterior samples for a batch of encoded observations.

        Runs a single JIT-compiled vmapped call over all contexts rather than
        looping ``sample_posterior`` once per observation, eliminating the
        Python overhead of per-observation dispatch.

        Parameters
        ----------
        embeddings:
            Float32 array of shape ``(batch_size, embedding_dim)`` produced by
            stacking :attr:`EncodedObservation.embedding` vectors.
        n_samples:
            Number of posterior draws per observation.
        base_seed:
            Base random seed. Each observation in the batch receives a unique
            sub-key derived from this seed.

        Returns
        -------
        np.ndarray
            Array of shape ``(batch_size, n_samples, parameter_dim)`` with
            non-finite values clamped to large finite values.
        """
        contexts = jnp.asarray(embeddings, dtype=jnp.float32)
        n = contexts.shape[0]
        keys = random.split(random.PRNGKey(base_seed), n)
        samples = _jit_batch_sample(self.model.flow, contexts, n_samples, keys)
        if self.parameter_space == "cube":
            samples = _clip_cube_support(nn.sigmoid(samples))
        return np.nan_to_num(np.asarray(samples), nan=0.0, posinf=1e6, neginf=-1e6)

    def log_prob(self, observation: EncodedObservation, parameters: Any) -> Any:
        """Evaluate posterior log-density for one or many parameter vectors.

        Parameters
        ----------
        observation:
            Encoded observation that defines the posterior context.
        parameters:
            One parameter vector or a batch of parameter vectors in the
            posterior's configured parameter space.

        Returns
        -------
        Any
            Scalar log-density or a vector of log-densities matching the input
            batch structure.
        """
        parameter_values = jnp.asarray(parameters, dtype=jnp.float32)
        context = jnp.asarray(observation.embedding, dtype=jnp.float32)

        if parameter_values.ndim == 1:
            if self.parameter_space == "cube":
                unconstrained_values, cube_values = _cube_to_unconstrained(parameter_values)
                return jnp.nan_to_num(
                    self.model.flow.log_prob(unconstrained_values, context)
                    + _cube_log_abs_det_jacobian(cube_values),
                    nan=-1e6,
                    neginf=-1e6,
                )
            return jnp.nan_to_num(
                self.model.flow.log_prob(parameter_values, context),
                nan=-1e6,
                neginf=-1e6,
            )

        if self.parameter_space == "cube":
            unconstrained_values, cube_values = _cube_to_unconstrained(parameter_values)
            return vmap(
                lambda value, cube_value: jnp.nan_to_num(
                    self.model.flow.log_prob(value, context)
                    + _cube_log_abs_det_jacobian(cube_value),
                    nan=-1e6,
                    neginf=-1e6,
                )
            )(unconstrained_values, cube_values)

        return vmap(
            lambda value: jnp.nan_to_num(
                self.model.flow.log_prob(value, context), nan=-1e6, neginf=-1e6
            )
        )(parameter_values)


@eqx.filter_jit
def _jit_batch_sample(
    flow: Any,
    contexts: Any,
    n_samples: int,
    keys: Any,
) -> Any:
    """JIT-compiled vmapped sampling over a batch of observation contexts.

    Parameters
    ----------
    flow:
        Trained conditional flow (e.g. :class:`ConditionalSplineFlow`).
    contexts:
        Float32 array of shape ``(batch_size, embedding_dim)``.
    n_samples:
        Number of posterior draws per observation. Treated as a compile-time
        constant; retracing only occurs if this value changes.
    keys:
        Array of JAX PRNGKeys with shape ``(batch_size, 2)``.

    Returns
    -------
    Any
        Float32 array of shape ``(batch_size, n_samples, parameter_dim)``.
    """
    return vmap(lambda ctx, k: flow.sample(ctx, n_samples, k))(contexts, keys)


class ConditionalSplineFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the spline flow backend.

    Notes
    -----
    This convenience subclass forces ``flow_family='spline'`` while keeping the
    rest of the :class:`ConditionalFlowPosterior` API unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "spline")
        super().__init__(*args, **kwargs)


class ConditionalAutoregressiveFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the autoregressive flow backend."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "autoregressive")
        super().__init__(*args, **kwargs)


class ConditionalNeuralAutoregressiveFlowPosterior(ConditionalFlowPosterior):
    """Posterior estimator specialized to the neural autoregressive backend."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("flow_family", "neural_autoregressive")
        super().__init__(*args, **kwargs)


__all__ = [
    "ConditionalAutoregressiveFlowPosterior",
    "ConditionalFlowPosterior",
    "ConditionalNeuralAutoregressiveFlowPosterior",
    "ConditionalSplineFlowPosterior",
]


# Self-register this estimator family so ``load_posterior_estimator`` can resolve
# it after a bare ``import petitRADTRANS.sbi`` (the package __init__ imports this
# module). The registry lives in ``flows.base`` (already imported above), so this
# introduces no new import.
from .base import register_posterior_estimator as _register  # noqa: E402

_register(ConditionalFlowPosterior.estimator_family, ConditionalFlowPosterior)
