"""NumPyro backend: HMC and NUTS driven by NumPyro's own MCMC engine.

NumPyro owns its warmup and chain loop, so this module is mostly translation:
pRT's unconstrained log-posterior becomes a ``potential_fn``, and pRT's kwargs
become kernel and ``MCMC`` kwargs. On top of that it adds three things NumPyro
does not provide:

* **Mass-matrix seeding from the MAP Hessian.**  See
  :func:`diagonal_inverse_mass_from_hessian`.
* **A multi-device path.** ``shard_map`` spreads chains across devices, with
  NumPyro's own ``chain_method="vectorized"`` inside each one.
* **Block checkpointing** for single-chain NUTS, so a SLURM wall-clock timeout
  becomes a resume rather than a total loss.

Everything shared with the BlackJAX backend lives in
:mod:`~petitRADTRANS.retrieval.samplers.gradient`.
"""

import inspect
import os
import pickle
import warnings

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import SamplerContext
from petitRADTRANS.retrieval.samplers.gradient import (
    GradientSampler,
    MCMCSamplingResults,
)
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_unconstrained_mcmc_interface,
)


class NumPyroSamplingResults(MCMCSamplingResults):
    """Results of a NumPyro run. See :class:`MCMCSamplingResults`."""


def diagonal_inverse_mass_from_hessian(logdensity, map_position):
    """Diagonal inverse mass matrix from the MAP Hessian of the log-density.

    The optimal HMC metric is the posterior precision; the *inverse* mass matrix
    is therefore an estimate of the posterior covariance. Here we use the cheap
    diagonal approximation 1 / precision_ii at the MAP. ``logdensity`` has a
    maximum at the MAP, so ``-diag(Hessian)`` is the (positive) precision
    diagonal. Any non-finite or non-positive entry falls back to 1.0, leaving
    NUTS' own warmup adaptation to refine it. Returns a 1-D NumPy array (the
    diagonal metric expected when ``is_mass_matrix_diagonal=True``).

    The diagonal is computed matrix-free, ONE coordinate at a time, via
    Hessian-vector products: ``diag(H)_i = (H e_i)_i`` where ``H e_i`` is the
    forward-over-reverse product ``jvp(grad(logdensity), e_i)``. This avoids ever
    materialising the dense Hessian. ``jax.hessian`` would vectorise all ``n``
    directions simultaneously, holding ``n`` reverse-mode tapes of the (very
    large, full-resolution radiative-transfer) forward model at once -- which for
    the lbl line-by-line models needs hundreds of GB and OOM-kills a whole node.
    Each HVP here keeps only a SINGLE such tape resident (~the cost of one
    gradient, which the MAP optimiser already evaluates comfortably), so peak
    memory is ~1x a gradient instead of ~n x.
    """
    _opt.require_jax()
    from jax import grad as _grad, jvp as _jvp

    jnp, jit = _opt.jnp, _opt.jit

    map_position = jnp.asarray(map_position)
    n = int(map_position.shape[0])
    grad_fn = _grad(logdensity)

    @jit
    def _hessian_vector_product(tangent):
        # H @ tangent via forward-over-reverse AD (jvp of the gradient). jit so the
        # forward model is compiled once and reused across the n coordinate calls.
        return _jvp(grad_fn, (map_position,), (tangent,))[1]

    eye = jnp.eye(n, dtype=map_position.dtype)
    # Plain Python loop (not vmap/hessian) so only one HVP's tape is live at a time;
    # each call returns just the length-n product vector and its transient is freed
    # before the next iteration. n is tiny (number of free parameters).
    precision_diag = -jnp.stack(
        [_hessian_vector_product(eye[i])[i] for i in range(n)]
    )
    valid = jnp.isfinite(precision_diag) & (precision_diag > 0.0)
    inv_mass = jnp.where(valid, 1.0 / jnp.where(valid, precision_diag, 1.0), 1.0)
    return np.asarray(inv_mass, dtype=float)


def dense_inverse_mass_from_hessian(
    logdensity, map_position, eigenvalue_floor=1.0
):
    """Dense inverse mass matrix from the MAP Hessian of the log-density.

    The dense analogue of :func:`_diagonal_inverse_mass_from_hessian`: the inverse
    mass matrix is the full posterior covariance, estimated as the inverse of the
    precision ``-Hessian`` at the MAP. The (symmetrised) precision is eigen-floored
    to a small positive value so the returned covariance is symmetric
    positive-definite even where the curvature is flat or slightly indefinite;
    if the Hessian is non-finite the whole matrix falls back to the identity,
    leaving NUTS' warmup to adapt it. Returns an ``(n, n)`` NumPy array (the dense
    metric used when ``is_mass_matrix_diagonal=False``).
    """
    _opt.require_jax()
    from jax import grad as _grad, jvp as _jvp

    jnp, jit = _opt.jnp, _opt.jit

    map_position = jnp.asarray(map_position)
    n = int(map_position.shape[0])

    # Build the Hessian one column at a time via Hessian-vector products.
    grad_fn = _grad(logdensity)

    @jit
    def _hessian_vector_product(tangent):
        return _jvp(grad_fn, (map_position,), (tangent,))[1]

    eye = jnp.eye(n, dtype=map_position.dtype)
    hess = jnp.stack([_hessian_vector_product(eye[i]) for i in range(n)]).T
    precision = -0.5 * (hess + hess.T)  # symmetrised precision (= inverse covariance)
    eigvals, eigvecs = jnp.linalg.eigh(precision)
    # SADDLE-FREE preconditioning: use |eigenvalue|, not the signed value.
    # At a saddle the negative-curvature directions are hyperbolic -- leapfrog
    # trajectories grow like exp(sqrt(|lambda|) * t) there instead of
    # oscillating -- so the metric must DAMP them in proportion to their own
    # |curvature|.
    eigvals = jnp.abs(eigvals)

    # Floor the PRECISION at roughly the prior precision (1.0), not at ~0.
    # The unconstrained priors have unit-ish scale (N(0,1) for affine-mapped
    # parameters, logistic with std 1.81 for cube-mapped ones), so a precision
    # floor of 1 caps the metric's widest direction at about the prior width.
    floor = float(eigenvalue_floor)
    safe_eigvals = jnp.where(jnp.isfinite(eigvals) & (eigvals > floor), eigvals, floor)
    covariance = (eigvecs * (1.0 / safe_eigvals)) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance = jnp.where(
        jnp.all(jnp.isfinite(covariance)), covariance, jnp.eye(n, dtype=covariance.dtype)
    )
    return np.asarray(covariance, dtype=float)


def _seed_inverse_mass_matrix(sampler_kwargs, logdensity, position, source_description):
    """Set ``sampler_kwargs["inverse_mass_matrix"]`` from the Hessian at ``position``.

    The optimal HMC metric is the posterior covariance, i.e. the inverse of the
    curvature. Seeding it from the MAP Hessian matters enormously here: with
    NumPyro's default identity metric the posterior's ~1e7 curvature range is
    unmodelled and the adapted step size collapses.

    The shape follows ``is_mass_matrix_diagonal``: a diagonal vector when True,
    the full dense covariance otherwise.
    """
    if sampler_kwargs.get("is_mass_matrix_diagonal", True):
        inverse_mass_matrix = diagonal_inverse_mass_from_hessian(logdensity, position)
    else:
        inverse_mass_matrix = dense_inverse_mass_from_hessian(logdensity, position)

    sampler_kwargs["inverse_mass_matrix"] = inverse_mass_matrix
    diagonal = (
        np.diag(inverse_mass_matrix)
        if np.ndim(inverse_mass_matrix) == 2
        else inverse_mass_matrix
    )
    print(
        "Initialised NUTS "
        f"{'dense' if np.ndim(inverse_mass_matrix) == 2 else 'diagonal'} inverse mass "
        f"matrix from {source_description} (diagonal range "
        f"{float(np.min(diagonal)):.3e} – {float(np.max(diagonal)):.3e})."
    )


class NumPyroSampler(GradientSampler):
    """Hamiltonian sampling with NumPyro."""

    output_subdirectory = "out_NumPyro"
    backend_label = "NumPyro"
    results_class = NumPyroSamplingResults

    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_numpyro()
        self.mcmc = None

    @property
    def kernel_class(self):
        """The NumPyro kernel class, e.g. ``numpyro.infer.NUTS``."""
        raise NotImplementedError

    @classmethod
    def _prepare_numpyro(
        cls, context: SamplerContext, algorithm_name: str, **sampler_kwargs
    ):
        """Shared preparation logic for all NumPyro sampler variants."""
        (
            free_parameter_names,
            initial_position,
            logdensity,
            transform_positions,
            log_jacobian,
        ) = build_unconstrained_mcmc_interface(context)

        optimize_init = sampler_kwargs.pop("optimize_init", False)
        warmup_optimizer_kwargs = dict(sampler_kwargs.pop("optimizer_kwargs", {}))
        # When set (with optimize_init), seed NUTS' metric from the MAP Hessian
        # so the step size is not collapsed by the posterior's anisotropy.
        init_imm_from_hessian = sampler_kwargs.pop(
            "init_inverse_mass_matrix_from_hessian", False
        )
        warmup_best_logdensity = None

        if optimize_init and "initial_position" not in sampler_kwargs:
            def seed_metric_at_optimum(optimize_result):
                # Must run before the warm-up state is released, and only if the
                # caller did not supply a metric of their own.
                if init_imm_from_hessian and "inverse_mass_matrix" not in sampler_kwargs:
                    _seed_inverse_mass_matrix(
                        sampler_kwargs,
                        logdensity,
                        optimize_result.best_position,
                        "the MAP Hessian",
                    )

            initial_position, warmup_best_logdensity = cls._run_optax_warmup(
                context=context,
                sampler_kwargs=sampler_kwargs,
                warmup_optimizer_kwargs=warmup_optimizer_kwargs,
                logdensity=logdensity,
                transform_positions=transform_positions,
                free_parameter_names=free_parameter_names,
                initial_position=initial_position,
                on_optimum=seed_metric_at_optimum,
            )

        # Hessian metric seeding for the case where the caller supplied the
        # initial position directly (e.g. a shared, pre-converged MAP file)
        # instead of asking for an in-process optax warm-start.
        if init_imm_from_hessian and "inverse_mass_matrix" not in sampler_kwargs:
            seed_position = _opt.jnp.asarray(
                sampler_kwargs.get("initial_position", initial_position)
            )
            if seed_position.ndim > 1:
                seed_position = seed_position[0]  # per-chain inits: expand about chain 0
            _seed_inverse_mass_matrix(
                sampler_kwargs,
                logdensity,
                seed_position,
                "the Hessian at the supplied initial position",
            )
            _opt.clear_caches()

        num_chains = sampler_kwargs.pop("num_chains", 1)
        # Default chain execution depends on the chain count. "vectorized" vmaps
        # every chain into one JIT call, multiplying the per-step memory by
        # num_chains -- which OOMs heavy forward models. A single chain therefore
        # defaults to "sequential" (one resident model copy); only fan out to
        # "vectorized" when the user asks for more than one chain or sets
        # chain_method explicitly.
        default_chain_method = "vectorized" if num_chains > 1 else "sequential"
        chain_method = sampler_kwargs.pop("chain_method", default_chain_method)
        thinning = sampler_kwargs.pop("thinning", 1)

        run_kwargs = cls._common_run_kwargs(
            sampler_kwargs,
            initial_position=initial_position,
            free_parameter_names=free_parameter_names,
            log_jacobian=log_jacobian,
            num_chains=num_chains,
        )
        run_kwargs["thinning"] = thinning
        run_kwargs["chain_method"] = chain_method

        if algorithm_name == "hmc":
            numpyro_num_steps = sampler_kwargs.pop("num_integration_steps", None)
            if numpyro_num_steps is None:
                numpyro_num_steps = sampler_kwargs.pop("num_steps", None)
            if numpyro_num_steps is None and "trajectory_length" not in sampler_kwargs:
                numpyro_num_steps = 20
            if numpyro_num_steps is not None:
                run_kwargs["num_steps"] = numpyro_num_steps
        run_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=logdensity,
            prior_func=transform_positions,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )

        summary_parameters = cls._common_summary_parameters(
            algorithm_name,
            run_kwargs,
            extra={
                "thinning": run_kwargs["thinning"],
                "chain_method": run_kwargs["chain_method"],
            },
        )
        if "num_steps" in run_kwargs:
            summary_parameters["num_integration_steps"] = run_kwargs["num_steps"]
        if "trajectory_length" in run_kwargs:
            summary_parameters["trajectory_length"] = run_kwargs["trajectory_length"]
        if optimize_init:
            summary_parameters["optimize_init"] = True
            summary_parameters["warmup_best_logdensity"] = warmup_best_logdensity

        return sampler, run_kwargs, summary_parameters

    @staticmethod
    def _prepare_kernel_kwargs(
        algorithm_name,
        log_likelihood_func,
        initial_step_size,
        is_mass_matrix_diagonal,
        target_acceptance_rate,
        numpyro_kwargs,
    ):
        numpyro_kwargs = dict(numpyro_kwargs)
        numpyro_kwargs.pop("use_jit", None)
        kernel_kwargs = {
            "potential_fn": lambda position: -log_likelihood_func(position),
            "step_size": initial_step_size,
            "adapt_step_size": numpyro_kwargs.pop("adapt_step_size", True),
            "adapt_mass_matrix": numpyro_kwargs.pop("adapt_mass_matrix", True),
            "dense_mass": numpyro_kwargs.pop("dense_mass", not is_mass_matrix_diagonal),
            "target_accept_prob": target_acceptance_rate,
            "find_heuristic_step_size": numpyro_kwargs.pop(
                "find_heuristic_step_size", False
            ),
            "forward_mode_differentiation": numpyro_kwargs.pop(
                "forward_mode_differentiation", False
            ),
            "regularize_mass_matrix": numpyro_kwargs.pop(
                "regularize_mass_matrix", True
            ),
        }
        kernel_kwargs.update(numpyro_kwargs)

        if (
            algorithm_name == "hmc"
            and "trajectory_length" in kernel_kwargs
            and "num_steps" in kernel_kwargs
        ):
            # Prefer trajectory_length so NumPyro can still adapt the step size.
            kernel_kwargs.pop("num_steps")

        return kernel_kwargs
    # --- block-wise checkpoint / resume (timeout-resilient single-chain NUTS) ---
    def _checkpoint_path(self):
        return self._get_output_prefix() + "_checkpoint.pkl"

    def _save_checkpoint(self, payload):
        """Atomically persist a checkpoint (write to .tmp then os.replace)."""
        path = self._checkpoint_path()
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as error:
            warnings.warn(
                f"Could not read NUTS checkpoint {path} ({error}); restarting the chain.",
                RuntimeWarning,
            )
            return None

    def _remove_checkpoint(self):
        for path in (self._checkpoint_path(), self._checkpoint_path() + ".tmp"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _concat_or_none(chunks):
        return np.concatenate(chunks, axis=0) if chunks else None

    def _run_checkpointed_nuts(
        self,
        *,
        kernel,
        initial_position,
        rng_key,
        num_warmup,
        num_samples,
        thinning,
        progress_bar,
        extra_fields,
        checkpoint_interval,
    ):
        """Run a single NUTS chain in blocks, checkpointing after each block.

        Collects ``num_samples`` draws in chunks of ``checkpoint_interval``. After
        every block the post-warmup MCMC state (``mcmc.last_state``, as NumPy) and
        all draws so far are saved to ``<prefix>_checkpoint.pkl``. If that file is
        present on entry (a resubmitted job after a timeout), sampling resumes from
        it -- warmup is NOT repeated, because NumPyro's ``post_warmup_state`` carries
        the frozen step size / mass matrix and the chain position forward.

        Returns the same tuple the non-checkpointed single-device path produces:
        ``(unconstrained_samples, potential_energy, acceptance_rate, is_divergent,
        num_integration_steps, warmup_parameters)``.
        """
        block = max(1, int(checkpoint_interval))

        unc_chunks, pe_chunks, accept_chunks, diverge_chunks, steps_chunks = (
            [],
            [],
            [],
            [],
            [],
        )
        collected = 0
        post_warmup_state = None
        warmup_parameters = {}

        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            stale = (
                checkpoint.get("num_samples_target") != num_samples
                or checkpoint.get("num_warmup") != num_warmup
            )
            if stale:
                warnings.warn(
                    f"Ignoring stale checkpoint for '{self.retrieval_name}' "
                    "(num_samples/num_warmup changed since it was written); "
                    "restarting the chain from warmup.",
                    RuntimeWarning,
                )
                self._remove_checkpoint()
                checkpoint = None

        if checkpoint is not None:
            if checkpoint.get("unconstrained_samples") is not None:
                unc_chunks.append(np.asarray(checkpoint["unconstrained_samples"]))
            for key, chunks in (
                ("potential_energy", pe_chunks),
                ("accept_prob", accept_chunks),
                ("diverging", diverge_chunks),
                ("num_steps", steps_chunks),
            ):
                if checkpoint.get(key) is not None:
                    chunks.append(np.asarray(checkpoint[key]))
            collected = int(checkpoint["collected"])
            # Rehydrate the saved (NumPy) pytree back to JAX arrays for NumPyro.
            post_warmup_state = _opt.tree.map(_opt.jnp.asarray, checkpoint["last_state"])
            warmup_parameters = dict(checkpoint.get("warmup_parameters", {}))
            print(
                f"[checkpoint] resuming '{self.retrieval_name}' at "
                f"{collected}/{num_samples} samples (warmup already complete)."
            )

        mcmc = None
        mcmc_block = None

        # --- adaptation phase (fresh chain only) ------------------------------
        # Running warmup through NumPyro's dedicated `warmup()`
        # lets us save the adapted state the moment adaptation ends, so a
        # resubmission resumes at 0 draws instead of restarting from scratch.
        if post_warmup_state is None:
            first_block = min(block, num_samples)
            mcmc = _opt.NumPyroMCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=first_block,
                num_chains=1,
                thinning=thinning,
                progress_bar=progress_bar,
                chain_method="sequential",
                jit_model_args=False,
            )
            mcmc_block = first_block
            mcmc.warmup(
                rng_key, init_params=initial_position, extra_fields=extra_fields
            )
            post_warmup_state = mcmc.post_warmup_state
            adapt_state = post_warmup_state.adapt_state
            warmup_parameters = {
                "step_size": np.asarray(adapt_state.step_size),
                "inverse_mass_matrix": np.asarray(adapt_state.inverse_mass_matrix),
            }
            self._save_checkpoint(
                {
                    "version": 1,
                    "retrieval_name": self.retrieval_name,
                    "num_warmup": num_warmup,
                    "num_samples_target": num_samples,
                    "collected": 0,
                    "last_state": _opt.tree.map(np.asarray, post_warmup_state),
                    "warmup_parameters": warmup_parameters,
                    "unconstrained_samples": None,
                    "potential_energy": None,
                    "accept_prob": None,
                    "diverging": None,
                    "num_steps": None,
                }
            )
            print(
                f"[checkpoint] '{self.retrieval_name}': warmup complete "
                f"({num_warmup} iterations), 0/{num_samples} samples saved "
                f"(step_size={float(np.asarray(adapt_state.step_size)):.3e})."
            )

        while collected < num_samples:
            this_block = min(block, num_samples - collected)
            # NumPyro returns exactly mcmc.num_samples draws per run
            if mcmc is None or mcmc_block != this_block:
                mcmc = _opt.NumPyroMCMC(
                    kernel,
                    num_warmup=num_warmup,
                    num_samples=this_block,
                    num_chains=1,
                    thinning=thinning,
                    progress_bar=progress_bar,
                    chain_method="sequential",
                    jit_model_args=False,
                )
                mcmc_block = this_block

            # Always a resume: adaptation ran above (fresh chain) or was restored
            # from the checkpoint, so warmup is never repeated here. Use the
            # state's own rng_key so the stream advances as in one long run.
            mcmc.post_warmup_state = post_warmup_state
            mcmc.run(
                post_warmup_state.rng_key,
                init_params=initial_position,
                extra_fields=extra_fields,
            )

            block_unc = np.asarray(mcmc.get_samples(group_by_chain=False))
            extra = mcmc.get_extra_fields(group_by_chain=False)
            unc_chunks.append(block_unc)
            for key, chunks in (
                ("potential_energy", pe_chunks),
                ("accept_prob", accept_chunks),
                ("diverging", diverge_chunks),
                ("num_steps", steps_chunks),
            ):
                if key in extra:
                    chunks.append(np.asarray(extra[key]))
            collected += int(block_unc.shape[0])
            post_warmup_state = mcmc.last_state

            adapt_state = mcmc.last_state.adapt_state
            warmup_parameters = {
                "step_size": np.asarray(adapt_state.step_size),
                "inverse_mass_matrix": np.asarray(adapt_state.inverse_mass_matrix),
            }
            self._save_checkpoint(
                {
                    "version": 1,
                    "retrieval_name": self.retrieval_name,
                    "num_warmup": num_warmup,
                    "num_samples_target": num_samples,
                    "collected": collected,
                    "last_state": _opt.tree.map(np.asarray, mcmc.last_state),
                    "warmup_parameters": warmup_parameters,
                    "unconstrained_samples": self._concat_or_none(unc_chunks),
                    "potential_energy": self._concat_or_none(pe_chunks),
                    "accept_prob": self._concat_or_none(accept_chunks),
                    "diverging": self._concat_or_none(diverge_chunks),
                    "num_steps": self._concat_or_none(steps_chunks),
                }
            )
            print(
                f"[checkpoint] '{self.retrieval_name}': {collected}/{num_samples} "
                f"samples saved (step_size="
                f"{float(np.asarray(adapt_state.step_size)):.3e})."
            )

        # The retained MCMC object holds only the final block; force the
        # samples-array diagnostics path (not az.from_numpyro on a partial object).
        self.mcmc = None

        return (
            self._concat_or_none(unc_chunks),
            self._concat_or_none(pe_chunks),
            self._concat_or_none(accept_chunks),
            self._concat_or_none(diverge_chunks),
            self._concat_or_none(steps_chunks),
            warmup_parameters,
        )
    def run_sampler(
        self,
        initial_position,
        parameter_names,
        num_samples=1000,
        num_warmup=1000,
        initial_step_size=1.0,
        target_acceptance_rate=0.8,
        is_mass_matrix_diagonal=True,
        progress_bar=False,
        seed=42654,
        use_jit=True,
        log_jacobian_fn=None,
        num_chains=1,
        thinning=1,
        chain_method="vectorized",
        checkpoint_interval=0,
        **numpyro_kwargs,
    ):
        self.parameter_names = list(parameter_names)
        kernel_kwargs = self._prepare_kernel_kwargs(
            algorithm_name=self.algorithm_name,
            log_likelihood_func=self.log_likelihood_func,
            initial_step_size=initial_step_size,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            target_acceptance_rate=target_acceptance_rate,
            numpyro_kwargs=numpyro_kwargs,
        )

        if not use_jit:
            warnings.warn(
                "NumPyro MCMC does not expose a simple non-JIT execution mode through this wrapper; "
                "'jit=False' is ignored.",
                RuntimeWarning,
            )

        extra_fields = ("potential_energy", "accept_prob", "diverging", "num_steps")
        rng_key = _opt.random.PRNGKey(seed)
        n_devices = _opt.local_device_count()
        use_multi_device = (
            n_devices > 1 and num_chains > 1 and chain_method != "sequential"
        )

        if use_multi_device:
            # Multi-device path: distribute chains across devices via shard_map.
            effective_devices = min(n_devices, num_chains)
            chains_per_device = max(1, num_chains // effective_devices)
            _kernel_class = self.kernel_class
            _kernel_kwargs = kernel_kwargs
            _initial_position = initial_position
            mesh = _opt.make_mesh((effective_devices,), ("devices",))

            def run_on_device(base_key, init_position):
                device_idx = _opt.lax.axis_index("devices")
                device_key = _opt.random.fold_in(base_key, device_idx)
                device_kernel = _kernel_class(**_kernel_kwargs)
                device_mcmc = _opt.NumPyroMCMC(
                    device_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=chains_per_device,
                    thinning=thinning,
                    progress_bar=False,  # must be False inside shard_map
                    chain_method="vectorized",
                    jit_model_args=False,
                )
                device_mcmc.run(
                    device_key, init_params=init_position, extra_fields=extra_fields
                )
                unc = device_mcmc.get_samples(group_by_chain=False)
                ex = device_mcmc.get_extra_fields(group_by_chain=False)
                adapt = device_mcmc.last_state.adapt_state
                # Return a flat tuple so out_specs can be an explicit matching tuple.
                return (
                    unc,
                    ex["potential_energy"],
                    ex["accept_prob"],
                    ex["diverging"],
                    ex["num_steps"],
                    adapt.step_size,
                    adapt.inverse_mass_matrix,
                )

            run_sharded = _opt.shard_map(
                run_on_device,
                mesh=mesh,
                # base_key and init_position both replicated to every device.
                in_specs=(_opt.P(), _opt.P()),
                out_specs=(_opt.P("devices"),)
                * 7,  # all 7 outputs concatenated along axis 0
            )
            with _opt.set_mesh(mesh):
                (
                    all_unc,
                    all_pe,
                    all_accept,
                    all_diverge,
                    all_steps,
                    all_step_sizes,
                    all_imm,
                ) = run_sharded(rng_key, _initial_position)

            unconstrained_samples = np.asarray(all_unc)
            potential_energy = np.asarray(all_pe)
            logdensity = -potential_energy
            acceptance_rate = np.asarray(all_accept)
            is_divergent = np.asarray(all_diverge)
            num_integration_steps = np.asarray(all_steps)
            warmup_parameters = {
                "step_size": float(np.mean(np.asarray(all_step_sizes))),
                "inverse_mass_matrix": np.mean(np.asarray(all_imm), axis=0),
            }
            self.mcmc = None
        elif checkpoint_interval and int(checkpoint_interval) > 0 and num_chains == 1:
            # Checkpointed single-chain path: sample in blocks of checkpoint_interval,
            # persisting the post-warmup MCMC state + collected draws after each block
            kernel = self.kernel_class(**kernel_kwargs)
            (
                unconstrained_samples,
                potential_energy,
                acceptance_rate,
                is_divergent,
                num_integration_steps,
                warmup_parameters,
            ) = self._run_checkpointed_nuts(
                kernel=kernel,
                initial_position=initial_position,
                rng_key=rng_key,
                num_warmup=num_warmup,
                num_samples=num_samples,
                thinning=thinning,
                progress_bar=progress_bar,
                extra_fields=extra_fields,
                checkpoint_interval=int(checkpoint_interval),
            )
            logdensity = -potential_energy if potential_energy is not None else None
        else:
            # Single-device path (also handles num_chains=1 on any device count).
            kernel = self.kernel_class(**kernel_kwargs)
            self.mcmc = _opt.NumPyroMCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                thinning=thinning,
                progress_bar=progress_bar,
                chain_method=chain_method,
                jit_model_args=False,
            )
            self.mcmc.run(
                rng_key, init_params=initial_position, extra_fields=extra_fields
            )

            unconstrained_samples = np.asarray(
                self.mcmc.get_samples(group_by_chain=False)
            )
            extra = self.mcmc.get_extra_fields(group_by_chain=False)
            potential_energy = (
                np.asarray(extra.get("potential_energy"))
                if "potential_energy" in extra
                else None
            )
            logdensity = -potential_energy if potential_energy is not None else None
            acceptance_rate = (
                np.asarray(extra.get("accept_prob")) if "accept_prob" in extra else None
            )
            is_divergent = (
                np.asarray(extra.get("diverging")) if "diverging" in extra else None
            )
            num_integration_steps = (
                np.asarray(extra.get("num_steps")) if "num_steps" in extra else None
            )
            adapt_state = self.mcmc.last_state.adapt_state
            warmup_parameters = {
                "step_size": np.asarray(adapt_state.step_size),
                "inverse_mass_matrix": np.asarray(adapt_state.inverse_mass_matrix),
            }

        transformed_samples = (
            unconstrained_samples
            if self.prior_func is None
            else np.asarray(self.prior_func(_opt.jnp.asarray(unconstrained_samples)))
        )

        model_log_likelihood = None
        if logdensity is not None and log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(
                logdensity
                - np.asarray(log_jacobian_fn(_opt.jnp.asarray(unconstrained_samples)))
            )

        metadata = {
            "algorithm": self.algorithm_name,
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            "num_chains": num_chains,
            "thinning": thinning,
            "chain_method": chain_method,
            "seed": seed,
            "parameter_names": parameter_names,
        }
        samples_file, metadata_file = self._save_results(
            parameter_names=parameter_names,
            samples=transformed_samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            warmup_parameters=warmup_parameters,
            metadata=metadata,
        )

        # The full posterior is now on disk as ``_samples.npz``; the partial
        # checkpoint (if any) is obsolete and would otherwise shadow a clean rerun.
        self._remove_checkpoint()

        self.results = NumPyroSamplingResults(
            algorithm=self.algorithm_name,
            samples=transformed_samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            warmup_parameters=self._to_serializable(warmup_parameters),
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            output_file=samples_file,
            metadata_file=metadata_file,
        )

        self.plot_diagnostics(self.results)

        return self.results
    def plot_diagnostics(self, results=None):
        if results is None:
            results = self.results

        if results is None:
            return {}

        try:
            _opt.require_arviz()
            _opt.require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        inference_data = None
        if self.mcmc is not None:
            try:
                inference_data = _opt.az.from_numpyro(self.mcmc)
            except Exception:
                inference_data = None

        parameter_names = list(self.parameter_names or [])
        samples = np.asarray(results.samples) if results.samples is not None else None

        if inference_data is None:
            if samples is None or samples.ndim != 2:
                warnings.warn(
                    "Unable to build NumPyro diagnostics plot because posterior samples are unavailable.",
                    RuntimeWarning,
                )
                return {}

            if not parameter_names:
                parameter_names = [
                    f"parameter_{index}" for index in range(samples.shape[1])
                ]

            if samples.shape[1] != len(parameter_names):
                warnings.warn(
                    "Unable to build NumPyro diagnostics plot because parameter names do not match the sample matrix.",
                    RuntimeWarning,
                )
                return {}

            posterior_groups = {
                name: samples[:, index][np.newaxis, :]
                for index, name in enumerate(parameter_names)
            }
            # ArviZ <1.0 takes the posterior as a keyword; ArviZ >=1.0 takes a
            # single mapping of group name -> variables.
            if "posterior" in inspect.signature(_opt.az.from_dict).parameters:
                inference_data = _opt.az.from_dict(posterior=posterior_groups)
            else:
                inference_data = _opt.az.from_dict({"posterior": posterior_groups})

        trace_output = self._get_output_prefix() + "_trace.pdf"

        try:
            try:
                plot_output = _opt.az.plot_trace(
                    inference_data,
                    compact=True,
                    figsize=(15, max(5, 2 * max(1, len(parameter_names)))),
                )
            except (TypeError, ValueError):
                # ArviZ >=1.0 dropped the `compact`/`figsize` kwargs and rejects
                # unmapped **kwargs outright; fall back to the bare call.
                plot_output = _opt.az.plot_trace(inference_data)
            return {"trace": self._save_plot_figure(plot_output, trace_output)}
        except Exception as error:
            warnings.warn(
                f"NumPyro diagnostics plotting failed: {error}", RuntimeWarning
            )
            return {}


class NumPyroHMCSampler(NumPyroSampler):
    """NumPyro Hamiltonian Monte Carlo with a fixed number of integration steps."""

    sampler_type = "numpyro_hmc"
    aliases = ("numpyrohmc",)

    @property
    def algorithm_name(self):
        return "hmc"

    @property
    def kernel_class(self):
        return _opt.NumPyroHMC

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_numpyro(context, algorithm_name="hmc", **sampler_kwargs)


class NumPyroNUTSSampler(NumPyroSampler):
    """NumPyro No-U-Turn Sampler, which chooses its trajectory length itself."""

    sampler_type = "numpyro_nuts"
    aliases = ("numpyronuts",)

    @property
    def algorithm_name(self):
        return "nuts"

    @property
    def kernel_class(self):
        return _opt.NumPyroNUTS

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_numpyro(context, algorithm_name="nuts", **sampler_kwargs)
