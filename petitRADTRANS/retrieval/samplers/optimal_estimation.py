"""Gradient-based optimisation ("optimal estimation") for petitRADTRANS retrievals.

This module provides an optax-powered maximum-a-posteriori (MAP) optimiser that
operates on the unconstrained-space log-posterior built by
:func:`petitRADTRANS.retrieval.samplers.interfaces.build_unconstrained_mcmc_interface`.
It is used in two ways:

* As a stand-alone "sampler" selectable with ``sampler_type="optimal_estimation"``
  (alias ``"optax"`` / ``"map"``): :class:`OptimalEstimationSampler` runs the
  optimiser and, by default, draws a Laplace (Rodgers optimal estimation)
  posterior from the inverse Hessian at the optimum so it emits a genuine sample
  distribution and 1-sigma error bars rather than a bare point estimate.

* As an optional warm-up step for the NumPyro and BlackJAX samplers
  (``optimize_init=True``): the same optimiser finds a MAP point that replaces the
  prior-median ``initial_position``, so the gradient sampler starts inside the
  high-probability region instead of in a stiff, divergence-prone tail.

The optimiser defaults to Adam with global-norm gradient clipping, but
any optax optimiser can be supplied.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_unconstrained_mcmc_interface,
)

# Lazily-populated JAX / optax handles (kept module-global so the heavy imports
# only happen when a gradient-based facility is actually used).
jax = None
jnp = None
optax = None


def _require_jax_handles():
    """Populate this module's own ``jax`` / ``jnp`` handles."""
    global jax, jnp
    if jnp is not None:
        return
    # Validates the JAX install and raises a helpful error if it is missing.
    _opt.require_jax()
    import jax as _jax
    import jax.numpy as _jnp

    jax = _jax
    jnp = _jnp


def _require_optax():
    """Populate the module-level ``optax`` handle."""
    global optax
    if optax is not None:
        return
    try:
        import optax as _optax
    except ImportError as error:
        raise ImportError(
            "optax is required for gradient-based optimisation (the "
            "'optimal_estimation' sampler and the optimize_init warm-up) but is "
            "not installed. Install it with `pip install optax` or via the "
            "petitRADTRANS retrieval extras."
        ) from error
    optax = _optax


# --------------------------------------------------------------------------- #
# Optimiser construction
# --------------------------------------------------------------------------- #
def build_optimizer(
    optimizer: Any = None,
    learning_rate: Any = 1e-2,
    clip_norm: Optional[float] = 1.0,
):
    """Return an ``optax.GradientTransformation``.

    Args:
        optimizer: selects the optax optimiser. May be
            * ``None``  -> ``optax.adam`` (the default);
            * a ``str`` -> ``getattr(optax, name)(learning_rate)``, e.g. ``"adamw"``,
              ``"adabelief"``, ``"sgd"``;
            * an already-built ``optax.GradientTransformation`` (used verbatim);
            * a callable factory -> ``optimizer(learning_rate)``.
        learning_rate: scalar or optax schedule, passed to the factory. Ignored
            when ``optimizer`` is a pre-built ``GradientTransformation``.
        clip_norm: if not ``None``, prepend ``optax.clip_by_global_norm(clip_norm)``
            so the huge gradients near the prior median produce bounded steps.

    Returns:
        An ``optax.GradientTransformation``.
    """
    _require_optax()

    if optimizer is None:
        base = optax.adam(learning_rate)
    elif isinstance(optimizer, str):
        factory = getattr(optax, optimizer, None)
        if factory is None or not callable(factory):
            raise ValueError(
                f"Unknown optax optimizer '{optimizer}'. Pass a name exposed by "
                "optax (e.g. 'adam', 'adamw', 'adabelief', 'sgd'), a built "
                "GradientTransformation, or a factory callable."
            )
        base = factory(learning_rate)
    elif hasattr(optimizer, "init") and hasattr(optimizer, "update"):
        # Already a GradientTransformation (NamedTuple); use as-is.
        base = optimizer
    elif callable(optimizer):
        base = optimizer(learning_rate)
    else:
        raise TypeError(
            "optimizer must be None, a string, an optax.GradientTransformation, "
            f"or a callable factory; got {type(optimizer)!r}."
        )

    if clip_norm is not None:
        return optax.chain(optax.clip_by_global_norm(clip_norm), base)

    return base


# --------------------------------------------------------------------------- #
# Core optimiser
# --------------------------------------------------------------------------- #
@dataclass
class OptimizeResult:
    """Outcome of :func:`optimize_unconstrained_position`."""

    best_position: Any  # unconstrained MAP estimate, shape (n_params,)
    best_logdensity: float
    trace: np.ndarray  # log-posterior per recorded step, for the winning start
    n_steps_run: int
    # Whether the gradient-infinity-norm fell below ``tol``; ``None`` if no ``tol``
    # was given, i.e. the test never ran. Not scale-free: prefer the decrement.
    converged: Optional[bool]
    n_starts: int = 1
    start_logdensities: Optional[np.ndarray] = None  # final logp of each start
    # See :func:`newton_decrement`; ``None`` unless ``report_newton_decrement``.
    newton_decrement_sq: Optional[float] = None
    n_nonpositive_curvature: Optional[int] = None


def optimize_unconstrained_position(
    logdensity: Callable,
    initial_position: Any,
    *,
    optimizer: Any = None,
    learning_rate: Any = 1e-2,
    clip_norm: Optional[float] = 1.0,
    n_steps: int = 300,
    tol: Optional[float] = None,
    patience: Optional[int] = 60,
    n_starts: int = 1,
    start_jitter: float = 1.0,
    seed: int = 0,
    lbfgs_polish_steps: int = 0,
    report_newton_decrement: bool = False,
    verbose: bool = True,
) -> OptimizeResult:
    """Maximise an unconstrained-space log-posterior with optax.

    The optimisation minimises ``-logdensity`` in the unconstrained space (so all
    parameter bounds are automatically respected via the change-of-variables that
    ``logdensity`` already includes). Non-finite gradient components are zeroed so
    a single bad coordinate cannot poison a step, and the best finite point seen
    is returned.

    Args:
        logdensity: callable mapping an unconstrained position (shape
            ``(n_params,)``) to a scalar log-posterior. Must be JAX-differentiable.
        initial_position: starting unconstrained position, shape ``(n_params,)``.
        optimizer / learning_rate / clip_norm: see :func:`build_optimizer`.
        n_steps: maximum optimisation steps per start.
        tol: optional gradient-infinity-norm tolerance for early stopping.
        patience: stop a start after this many steps without improvement
            (``None`` disables).
        n_starts: number of random restarts (start 0 is ``initial_position``;
            the rest are ``initial_position`` plus Gaussian jitter). The best
            optimum across starts is returned.
        start_jitter: standard deviation of the Gaussian jitter for extra starts.
        seed: RNG seed for the restart jitter.
        lbfgs_polish_steps: if > 0, refine the best optax optimum with this many
            ``optax.lbfgs`` (zoom line-search) iterations.
        report_newton_decrement: evaluate :func:`newton_decrement` at the returned
            position (one Hessian: ``n_params`` Hessian-vector products).
        verbose: print progress.

    Returns:
        An :class:`OptimizeResult`.
    """
    _require_jax_handles()
    _require_optax()

    initial_position = jnp.asarray(initial_position)
    n_params = int(initial_position.shape[-1])

    value_and_grad = jax.jit(jax.value_and_grad(lambda z: -logdensity(z)))
    optimiser = build_optimizer(optimizer, learning_rate, clip_norm)

    @jax.jit
    def _update(position, opt_state):
        neg_value, grad = value_and_grad(position)
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        updates, opt_state = optimiser.update(grad, opt_state, position)
        position = optax.apply_updates(position, updates)
        grad_inf_norm = jnp.max(jnp.abs(grad))
        return position, opt_state, -neg_value, grad_inf_norm

    def _run_single(start_position, label):
        position = jnp.asarray(start_position)
        opt_state = optimiser.init(position)

        best_value = -np.inf
        best_position = position
        steps_since_improvement = 0
        trace = []
        report_every = max(1, n_steps // 10)
        steps_run = 0
        converged = False

        for step in range(n_steps):
            steps_run = step + 1
            # Value/grad are evaluated at the CURRENT position, then a step is
            # taken -- so ``logd`` belongs to ``position`` BEFORE the update.
            # Keep that pre-update point as the candidate best; pairing the
            # value with the post-update position (as this loop used to) stores
            # a point whose true log-posterior differs from ``best_value`` and
            # silently hands the polish/sampler a worse start than reported.
            previous_position = position
            position, opt_state, logd, grad_inf_norm = _update(position, opt_state)
            logd_value = float(logd)
            grad_inf_norm = float(grad_inf_norm)
            trace.append(logd_value)

            if np.isfinite(logd_value) and logd_value > best_value:
                best_value = logd_value
                best_position = previous_position
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if verbose and (step % report_every == 0 or step == n_steps - 1):
                print(
                    f"  [optimal_estimation{label}] step {step:5d}: "
                    f"log-posterior = {logd_value:.6e}  |grad|_inf = {grad_inf_norm:.3e}"
                )

            if tol is not None and grad_inf_norm < tol:
                converged = True
                if verbose:
                    print(
                        f"  [optimal_estimation{label}] converged at step {step} "
                        f"(|grad|_inf {grad_inf_norm:.3e} < tol {tol:.3e})."
                    )
                break

            if patience and steps_since_improvement >= patience:
                if verbose:
                    print(
                        f"  [optimal_estimation{label}] early stop at step {step} "
                        f"(no improvement for {patience} steps)."
                    )
                break

        # No tolerance means the convergence test never ran.
        return (
            best_position,
            best_value,
            np.asarray(trace),
            steps_run,
            converged if tol is not None else None,
        )

    # Build the set of starting points.
    starts = [initial_position]
    if n_starts > 1:
        rng = np.random.default_rng(seed)
        for _ in range(n_starts - 1):
            jitter = rng.standard_normal(n_params) * start_jitter
            starts.append(initial_position + jnp.asarray(jitter))

    init_value = float(logdensity(initial_position))
    if verbose:
        print(
            f"optimal_estimation: optimising {n_params} parameters from "
            f"{n_starts} start(s); initial log-posterior = {init_value:.6e}"
        )

    best_overall_value = -np.inf
    best_overall_position = initial_position
    best_trace = np.asarray([init_value])
    best_steps = 0
    best_converged = None
    start_logdensities = []

    for index, start in enumerate(starts):
        label = "" if n_starts == 1 else f" start {index}"
        position, value, trace, steps_run, converged = _run_single(start, label)
        start_logdensities.append(value)
        if value > best_overall_value:
            best_overall_value = value
            best_overall_position = position
            best_trace = trace
            best_steps = steps_run
            best_converged = converged

    # Optional L-BFGS polish of the best optax optimum.
    if lbfgs_polish_steps and lbfgs_polish_steps > 0:
        try:
            best_overall_position, polished_value = _lbfgs_polish(
                logdensity, best_overall_position, lbfgs_polish_steps, verbose=verbose
            )
            if np.isfinite(polished_value) and polished_value > best_overall_value:
                best_overall_value = polished_value
        except Exception as error:  # pragma: no cover - polish is best-effort
            warnings.warn(
                f"L-BFGS polish failed ({error}); keeping the optax optimum.",
                RuntimeWarning,
            )

    if verbose:
        print(
            f"optimal_estimation: best log-posterior = {best_overall_value:.6e} "
            f"(improved by {best_overall_value - init_value:.3e})"
        )

    decrement = {}
    if report_newton_decrement:
        try:
            decrement = newton_decrement(logdensity, best_overall_position)
        except Exception as error:  # pragma: no cover - a diagnostic, best-effort
            warnings.warn(
                f"Newton decrement could not be evaluated ({error}).", RuntimeWarning
            )
        if verbose and decrement.get("newton_decrement_sq") is not None:
            print(
                "optimal_estimation: Newton decrement^2 = "
                f"{decrement['newton_decrement_sq']:.3e} (a Newton step would gain "
                f"{decrement['predicted_gain']:.3e} nats); "
                f"{decrement['n_nonpositive_curvature']} direction(s) of non-positive "
                "curvature."
            )

    return OptimizeResult(
        best_position=best_overall_position,
        best_logdensity=best_overall_value,
        trace=best_trace,
        n_steps_run=best_steps,
        converged=best_converged,
        n_starts=n_starts,
        start_logdensities=np.asarray(start_logdensities),
        newton_decrement_sq=decrement.get("newton_decrement_sq"),
        n_nonpositive_curvature=decrement.get("n_nonpositive_curvature"),
    )


def _negative_logdensity_hessian(logdensity, position):
    """Symmetrised Hessian of ``-logdensity`` at ``position``.

    Built column by column with Hessian-vector products rather than
    ``jax.hessian``, so only one reverse-mode tape is resident at a time.
    """
    _require_jax_handles()
    position = jnp.asarray(position)
    negative_logdensity = lambda z: -logdensity(z)  # noqa: E731

    @jax.jit
    def hessian_vector_product(z, vector):
        return jax.jvp(jax.grad(negative_logdensity), (z,), (vector,))[1]

    n_parameters = int(position.shape[-1])
    identity = np.eye(n_parameters)
    columns = [
        np.asarray(
            hessian_vector_product(position, jnp.asarray(identity[index])), dtype=float
        )
        for index in range(n_parameters)
    ]
    hessian = np.array(columns).T
    return 0.5 * (hessian + hessian.T)  # symmetrise against round-off


def newton_decrement(logdensity, position):
    """How far ``position`` is from a stationary point of the log-posterior, in nats.

    A gradient norm cannot tell on an ill-conditioned posterior: along a stiff
    direction a gradient of 1e3 can sit 1e-2 nats from the optimum, while along a
    soft one a gradient of 1e-2 can sit many nats away. The Newton decrement
    ``lambda^2 = g^T H^-1 g`` weights the gradient ``g`` by the local curvature
    ``H`` (the Hessian of ``-logdensity``): ``lambda^2 / 2`` is the gain in
    log-posterior that a Newton step predicts, so ``lambda^2 << 1`` means
    converged whatever the parameter scales are.

    Directions of non-positive curvature have no Newton step. They are left out
    of the sum and counted in ``n_nonpositive_curvature``; a non-zero count means
    ``position`` lies on a ridge or a saddle rather than at a maximum, however
    small the decrement.

    Returns a dict with ``newton_decrement_sq``, ``predicted_gain``,
    ``n_nonpositive_curvature`` and ``grad_inf_norm``. The first two are ``None``
    if the gradient or the Hessian is not finite.
    """
    _require_jax_handles()
    position = jnp.asarray(position)
    gradient = np.asarray(jax.grad(lambda z: -logdensity(z))(position), dtype=float)
    hessian = _negative_logdensity_hessian(logdensity, position)
    grad_inf_norm = float(np.max(np.abs(gradient)))

    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
        return {
            "newton_decrement_sq": None,
            "predicted_gain": None,
            "n_nonpositive_curvature": None,
            "grad_inf_norm": grad_inf_norm,
        }

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    # Curvature indistinguishable from zero at round-off counts as flat.
    positive = eigenvalues > 1e-12 * np.max(np.abs(eigenvalues))
    projected_gradient = eigenvectors.T @ gradient
    decrement_sq = float(
        np.sum(projected_gradient[positive] ** 2 / eigenvalues[positive])
    )

    return {
        "newton_decrement_sq": decrement_sq,
        "predicted_gain": 0.5 * decrement_sq,
        "n_nonpositive_curvature": int(np.sum(~positive)),
        "grad_inf_norm": grad_inf_norm,
    }


def _lbfgs_polish(logdensity, position, n_steps, verbose=True):
    """Refine ``position`` with optax.lbfgs (zoom line-search) minimising -logdensity.
    https://en.wikipedia.org/wiki/Limited-memory_BFGS

    Tracks and returns the BEST point seen (not the last iterate), and returns
    that best point even if a step raises mid-run (e.g. a physical-validity
    callback such as the reference-gravity check aborting an over-long line
    search step), so a late crash cannot discard the polish gains.
    """
    _require_jax_handles()
    _require_optax()

    def loss(z):
        return -logdensity(z)

    solver = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(loss)

    position = jnp.asarray(position)
    opt_state = solver.init(position)

    @jax.jit
    def _step(position, opt_state):
        value, grad = value_and_grad(position, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, position, value=value, grad=grad, value_fn=loss
        )
        new_position = optax.apply_updates(position, updates)
        return new_position, opt_state, value

    # ``value`` returned by step k is the loss at the PRE-update position, so
    # pair it with that position when tracking the best point.
    best_value = np.inf
    best_position = position
    for step in range(n_steps):
        try:
            new_position, opt_state, value = _step(position, opt_state)
        except Exception as error:  # noqa: BLE001 - keep the best point on any failure
            warnings.warn(
                f"L-BFGS polish aborted at step {step} ({error}); keeping the "
                "best point found so far.",
                RuntimeWarning,
            )
            break
        value = float(value)
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_position = position
        position = new_position
        if verbose and (step % max(1, n_steps // 5) == 0 or step == n_steps - 1):
            print(f"  [optimal_estimation L-BFGS] step {step}: -log-posterior = {value:.6e}")

    # The final iterate has not been evaluated inside the loop; score it too.
    try:
        final_value = float(loss(position))
        if np.isfinite(final_value) and final_value < best_value:
            best_value = final_value
            best_position = position
    except Exception:  # noqa: BLE001 - final point may be invalid; best is kept
        pass

    return best_position, -best_value


# --------------------------------------------------------------------------- #
# Walker initialisation (warm-up helper)
# --------------------------------------------------------------------------- #
def make_walker_inits(position, num_chains, *, jitter=0.0, seed=0):
    """Broadcast a single MAP point into per-chain initial positions.

    Returns a ``(n_params,)`` array for a single chain, or ``(num_chains, n_params)``
    for several chains (the first chain stays exactly at the optimum; the rest get
    Gaussian jitter of width ``jitter`` so multi-chain convergence diagnostics
    such as R-hat remain meaningful).
    """
    _require_jax_handles()
    position = jnp.asarray(position)

    if num_chains is None or num_chains <= 1:
        return position

    base = jnp.broadcast_to(position, (num_chains,) + position.shape)

    if jitter and jitter > 0:
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, base.shape) * jitter
        noise = noise.at[0].set(jnp.zeros_like(position))  # keep chain 0 at the MAP
        base = base + noise

    return base


def report_and_save_map_warmup(
    optimize_result: "OptimizeResult",
    transform_positions: Callable,
    parameter_names,
    output_directory: str,
    retrieval_name: str,
    *,
    output_subdirectory: str,
    label: str = "optimal_estimation",
    verbose: bool = True,
):
    """Print and persist the MAP point found by an ``optimize_init`` warm-up.

    Used by the NumPyro and BlackJAX samplers when ``optimize_init=True``: once the
    optax optimiser has located the MAP, this transforms the best unconstrained
    position into physical (constrained) parameter space, prints each parameter's
    MAP value, and writes the MAP sample to
    ``<output_directory>/<output_subdirectory>/<retrieval_name>_map_warmup.{npz,json}``.

    Args:
        optimize_result: the :class:`OptimizeResult` returned by
            :func:`optimize_unconstrained_position`.
        transform_positions: maps an unconstrained position to physical space.
        parameter_names: ordered free-parameter names matching the position.
        output_directory: retrieval output directory.
        retrieval_name: retrieval name used as the file-name stem.
        output_subdirectory: sampler-specific sub-directory (e.g. ``"out_BlackJAX"``
            or ``"out_NumPyro"``).
        label: short tag used in the printed lines.
        verbose: print the MAP values and the saved-file path.

    Returns:
        The path to the saved ``.npz`` file.
    """
    _require_jax_handles()

    parameter_names = list(parameter_names)
    best_unconstrained = jnp.asarray(optimize_result.best_position)
    best_physical = np.asarray(
        transform_positions(best_unconstrained), dtype=float
    ).reshape(-1)
    best_logdensity = float(optimize_result.best_logdensity)

    if verbose:
        print(
            f"  [{label}] MAP sample (best log-posterior = {best_logdensity:.6e}):"
        )
        for name, value in zip(parameter_names, best_physical):
            print(f"    {name:40s} = {value:.6g}")

    output_dir = os.path.join(output_directory, output_subdirectory)
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"{retrieval_name}_map_warmup")

    np.savez(
        f"{prefix}.npz",
        best_fit=best_physical,
        best_fit_unconstrained=np.asarray(best_unconstrained, dtype=float).reshape(-1),
        parameter_names=np.asarray(parameter_names),
        best_logdensity=np.asarray(best_logdensity),
    )
    with open(f"{prefix}.json", "w") as handle:
        json.dump(
            {
                "best_logdensity": best_logdensity,
                # null: no gradient tolerance was set, so the test never ran.
                "converged": optimize_result.converged,
                # null: not requested (report_newton_decrement) or not finite.
                "newton_decrement_sq": optimize_result.newton_decrement_sq,
                "n_nonpositive_curvature": optimize_result.n_nonpositive_curvature,
                "best_fit": {
                    name: float(value)
                    for name, value in zip(parameter_names, best_physical)
                },
            },
            handle,
            indent=4,
        )

    if verbose:
        print(f"  [{label}] MAP sample saved to {prefix}.npz")

    return f"{prefix}.npz"


# --------------------------------------------------------------------------- #
# Laplace (Rodgers optimal estimation) posterior
# --------------------------------------------------------------------------- #
def laplace_covariance(logdensity, position, *, eigenvalue_floor=1e-8):
    """Posterior covariance from the inverse Hessian of ``-logdensity`` at ``position``.

    This is the Laplace approximation underpinning Rodgers optimal estimation: near
    the MAP the posterior is approximately Gaussian with covariance equal to the
    inverse of the curvature (Fisher-information-like) matrix.

    Returns ``(covariance, hessian, is_positive_definite)``. If the raw Hessian is
    not positive definite (a poor or saddle-like optimum), its eigenvalues are
    floored at ``eigenvalue_floor`` before inversion and ``is_positive_definite``
    is ``False``.
    """
    hessian = _negative_logdensity_hessian(logdensity, position)

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    is_positive_definite = bool(np.all(eigenvalues > 0))

    if is_positive_definite:
        inverse_eigenvalues = 1.0 / eigenvalues
    else:
        inverse_eigenvalues = 1.0 / np.clip(eigenvalues, eigenvalue_floor, None)

    covariance = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)

    return covariance, hessian, is_positive_definite


def _multivariate_normal_logpdf(samples, mean, covariance):
    """Gaussian log-density of each row of ``samples`` (unconstrained space)."""
    samples = np.asarray(samples, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    precision = np.linalg.inv(covariance)
    deviation = samples - mean
    quadratic = np.einsum("ni,ij,nj->n", deviation, precision, deviation)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        log_determinant = np.log(np.clip(np.linalg.det(covariance), 1e-300, None))

    return -0.5 * (quadratic + log_determinant + mean.size * np.log(2.0 * np.pi))


def _sample_gaussian(mean, covariance, n_samples, seed):
    """Draw ``n_samples`` from N(mean, covariance); row 0 is the mean (the MAP)."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=int(n_samples))
    if draws.shape[0] > 0:
        draws[0] = mean
    return draws


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass
class OptimalEstimationResults:
    """Results emitted by :class:`OptimalEstimationSampler`.

    ``samples`` (physical/constrained space, shape ``(n_draw, n_params)``) and
    ``log_likelihood`` (shape ``(n_draw,)``) are the fields consumed by
    ``Retrieval._build_samples_from_sampler_results``. For a Laplace run the rows
    are draws from the Gaussian posterior (row 0 is the MAP) and ``log_likelihood``
    is the Gaussian (Laplace) log-density; for a point estimate there is a single
    row at the MAP. The true model log-posterior at the MAP is ``best_logdensity``.
    """

    samples: np.ndarray
    log_likelihood: np.ndarray
    best_fit: np.ndarray
    best_logdensity: float
    parameter_names: list
    covariance: Optional[np.ndarray] = None
    covariance_is_positive_definite: Optional[bool] = None
    unconstrained_samples: Optional[np.ndarray] = None
    trace: Optional[np.ndarray] = None
    converged: Optional[bool] = None  # None: no gradient tolerance was set
    output_file: Optional[str] = None
    metadata_file: Optional[str] = None


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
class OptimalEstimationSampler(Sampler):
    """Run an optax MAP optimisation (plus optional Laplace posterior) as a sampler.

    Selectable via ``RetrievalConfig(sampler_type="optimal_estimation")`` (aliases
    ``"optax"``, ``"map"``). With ``compute_laplace=True`` (default) it additionally
    draws a Gaussian posterior from the inverse Hessian at the optimum, so corner
    plots and 1-sigma intervals are available; with ``compute_laplace=False`` it
    returns the single MAP point.
    """

    sampler_type = "optimal_estimation"
    aliases = ("optimalestimation", "optax", "map")
    output_subdirectory = "out_OptimalEstimation"
    uses_distribution_priors = True

    def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        _require_jax_handles()
        _require_optax()
        self.logdensity = None
        self.transform_positions = None
        self.parameter_names = None

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        (
            free_parameter_names,
            initial_position,
            logdensity,
            transform_positions,
            log_jacobian,
        ) = build_unconstrained_mcmc_interface(context)

        run_kwargs = {
            "initial_position": sampler_kwargs.pop("initial_position", initial_position),
            "parameter_names": sampler_kwargs.pop("parameter_names", free_parameter_names),
            "optimizer": sampler_kwargs.pop("optimizer", None),
            "learning_rate": sampler_kwargs.pop("learning_rate", 1e-2),
            "clip_norm": sampler_kwargs.pop("clip_norm", 1.0),
            "n_steps": sampler_kwargs.pop("n_steps", sampler_kwargs.pop("num_steps", 300)),
            "tol": sampler_kwargs.pop("tol", None),
            "patience": sampler_kwargs.pop("patience", 60),
            "n_starts": sampler_kwargs.pop("n_starts", 1),
            "start_jitter": sampler_kwargs.pop("start_jitter", 1.0),
            "lbfgs_polish_steps": sampler_kwargs.pop("lbfgs_polish_steps", 0),
            "compute_laplace": sampler_kwargs.pop("compute_laplace", True),
            "n_posterior_samples": sampler_kwargs.pop("n_posterior_samples", 2000),
            "seed": sampler_kwargs.pop("seed", 0),
            "verbose": sampler_kwargs.pop("verbose", True),
        }
        run_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=logdensity,
            prior_func=transform_positions,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        sampler.logdensity = logdensity
        sampler.transform_positions = transform_positions

        optimizer = run_kwargs["optimizer"]
        if optimizer is None:
            optimizer_label = "adam"
        elif isinstance(optimizer, str):
            optimizer_label = optimizer
        else:
            optimizer_label = type(optimizer).__name__

        summary_parameters = {
            "sampler_type": "optimal_estimation",
            "optimizer": optimizer_label,
            "learning_rate": run_kwargs["learning_rate"]
            if np.isscalar(run_kwargs["learning_rate"])
            else "schedule",
            "clip_norm": run_kwargs["clip_norm"],
            "n_steps": run_kwargs["n_steps"],
            "n_starts": run_kwargs["n_starts"],
            "lbfgs_polish_steps": run_kwargs["lbfgs_polish_steps"],
            "compute_laplace": run_kwargs["compute_laplace"],
            "n_posterior_samples": run_kwargs["n_posterior_samples"],
            "seed": run_kwargs["seed"],
        }

        return sampler, run_kwargs, summary_parameters

    def run_sampler(
        self,
        initial_position,
        parameter_names,
        *,
        optimizer=None,
        learning_rate=1e-2,
        clip_norm=1.0,
        n_steps=300,
        tol=None,
        patience=60,
        n_starts=1,
        start_jitter=1.0,
        lbfgs_polish_steps=0,
        compute_laplace=True,
        n_posterior_samples=2000,
        seed=0,
        verbose=True,
        **_ignored,
    ):
        self.parameter_names = list(parameter_names)

        optimize_result = optimize_unconstrained_position(
            self.logdensity,
            initial_position,
            optimizer=optimizer,
            learning_rate=learning_rate,
            clip_norm=clip_norm,
            n_steps=n_steps,
            tol=tol,
            patience=patience,
            n_starts=n_starts,
            start_jitter=start_jitter,
            lbfgs_polish_steps=lbfgs_polish_steps,
            seed=seed,
            verbose=verbose,
        )

        best_unconstrained = jnp.asarray(optimize_result.best_position)
        best_physical = np.asarray(
            self.transform_positions(best_unconstrained), dtype=float
        ).reshape(-1)
        best_logdensity = float(self.logdensity(best_unconstrained))

        covariance = None
        covariance_is_positive_definite = None
        unconstrained_samples = np.asarray(best_unconstrained, dtype=float)[None, :]

        if compute_laplace:
            try:
                covariance, _hessian, covariance_is_positive_definite = (
                    laplace_covariance(self.logdensity, best_unconstrained)
                )
                if not covariance_is_positive_definite:
                    warnings.warn(
                        "Laplace Hessian at the optimum was not positive definite; "
                        "eigenvalues were floored before inversion. Treat the "
                        "covariance / error bars as approximate.",
                        RuntimeWarning,
                    )
                unconstrained_samples = _sample_gaussian(
                    np.asarray(best_unconstrained, dtype=float),
                    covariance,
                    n_posterior_samples,
                    seed,
                )
            except Exception as error:  # pragma: no cover - Laplace is best-effort
                warnings.warn(
                    f"Laplace covariance failed ({error}); returning the MAP point only.",
                    RuntimeWarning,
                )
                unconstrained_samples = np.asarray(best_unconstrained, dtype=float)[None, :]

        physical_samples = np.asarray(
            self.transform_positions(jnp.asarray(unconstrained_samples)), dtype=float
        )
        if physical_samples.ndim == 1:
            physical_samples = physical_samples[None, :]

        if covariance is not None and unconstrained_samples.shape[0] > 1:
            # Cheap Gaussian (Laplace) log-density; avoids one forward model per draw.
            log_likelihood = _multivariate_normal_logpdf(
                unconstrained_samples,
                np.asarray(best_unconstrained, dtype=float),
                covariance,
            )
        else:
            log_likelihood = np.asarray([best_logdensity], dtype=float)

        results = OptimalEstimationResults(
            samples=physical_samples,
            log_likelihood=np.asarray(log_likelihood, dtype=float).reshape(-1),
            best_fit=best_physical,
            best_logdensity=best_logdensity,
            parameter_names=self.parameter_names,
            covariance=covariance,
            covariance_is_positive_definite=covariance_is_positive_definite,
            unconstrained_samples=unconstrained_samples,
            trace=optimize_result.trace,
            converged=optimize_result.converged,
        )

        self.results = results
        self._save_results(results)
        return results

    # ------------------------------------------------------------------ #
    # Persistence / reporting
    # ------------------------------------------------------------------ #
    def _save_results(self, results: "OptimalEstimationResults"):
        prefix = self._get_output_prefix()
        arrays = {
            "samples": results.samples,
            "log_likelihood": results.log_likelihood,
            "best_fit": results.best_fit,
        }
        if results.unconstrained_samples is not None:
            arrays["unconstrained_samples"] = results.unconstrained_samples
        if results.covariance is not None:
            arrays["covariance"] = results.covariance
        if results.trace is not None:
            arrays["trace"] = results.trace
        np.savez(f"{prefix}_samples.npz", **arrays)

        with open(f"{prefix}_params.json", "w") as handle:
            json.dump(list(results.parameter_names), handle, indent=4)

        metadata = {
            "sampler": self.__class__.__name__,
            "best_logdensity": results.best_logdensity,
            "converged": results.converged,
            "covariance_is_positive_definite": results.covariance_is_positive_definite,
            "n_samples": int(results.samples.shape[0]),
            "best_fit": {
                name: float(value)
                for name, value in zip(results.parameter_names, results.best_fit)
            },
        }
        results.metadata_file = f"{prefix}_metadata.json"
        results.output_file = f"{prefix}_samples.npz"
        with open(results.metadata_file, "w") as handle:
            json.dump(metadata, handle, indent=4)

        return results.output_file, results.metadata_file

    def pretty_print_results(self):
        results = self.results
        if results is None:
            return
        print(f"{self.__class__.__name__} results:")
        print(f"  best log-posterior : {results.best_logdensity:.6e}")
        converged = (
            "not tested (no tol)" if results.converged is None else results.converged
        )
        print(f"  converged          : {converged}")
        if results.covariance is not None:
            print(
                f"  Laplace posterior  : {results.samples.shape[0]} samples "
                f"(PD Hessian: {results.covariance_is_positive_definite})"
            )
            sigma = np.sqrt(np.clip(np.diag(results.covariance), 0.0, None))
            for name, value, error in zip(
                results.parameter_names, results.best_fit, sigma
            ):
                print(f"    {name:40s} = {value:.6g}  (unconstrained 1-sigma {error:.3g})")
        else:
            for name, value in zip(results.parameter_names, results.best_fit):
                print(f"    {name:40s} = {value:.6g}")

    def get_summary(self, free_parameter_names=None):
        results = self.results
        if results is None:
            return super().get_summary(free_parameter_names)
        return {
            "sampler": self.__class__.__name__,
            "parameters": {
                "best_logdensity": results.best_logdensity,
                "converged": results.converged,
                "covariance_is_positive_definite": results.covariance_is_positive_definite,
            },
            "diagnostics": {
                "n_samples": int(results.samples.shape[0]),
                "n_optimisation_steps": int(np.size(results.trace))
                if results.trace is not None
                else 0,
            },
            "evidence": {},
            "posterior": self._posterior_summary_from_matrix(
                results.samples, results.parameter_names
            ),
        }
