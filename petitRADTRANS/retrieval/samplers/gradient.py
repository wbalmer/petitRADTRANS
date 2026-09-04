"""Shared machinery for the gradient-based MCMC backends (BlackJAX and NumPyro).

Both backends sample the same object: the log-posterior over the
*unconstrained* space built by
:func:`~petitRADTRANS.retrieval.samplers.interfaces.build_unconstrained_mcmc_interface`.
They therefore need the same preparation (an optional optax MAP warm-up, the
same window/warmup knobs), produce the same diagnostics (acceptance rate,
divergences, integration steps), and write the same files. All of that lives
here; each backend module keeps only what is genuinely its own -- BlackJAX's
``shard_map`` multi-chain fan-out, NumPyro's mass-matrix seeding, chain
methods and block checkpointing.
"""

import json
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext


@dataclass
class MCMCSamplingResults:
    """Posterior draws and per-draw diagnostics from a gradient-based sampler.

    ``samples`` are in physical (constrained) space and ``unconstrained_samples``
    in the space the sampler actually explored. ``logdensity`` is the
    unconstrained log-posterior, which includes the change-of-variables
    Jacobian; ``model_log_likelihood`` is that same quantity with the Jacobian
    removed, i.e. the model's own log likelihood.
    """

    algorithm: str
    samples: np.ndarray
    unconstrained_samples: np.ndarray
    logdensity: np.ndarray
    model_log_likelihood: np.ndarray | None
    warmup_parameters: dict
    acceptance_rate: np.ndarray | None = None
    is_divergent: np.ndarray | None = None
    num_integration_steps: np.ndarray | None = None
    output_file: str | None = None
    metadata_file: str | None = None


class GradientSampler(Sampler):
    """Base class for the BlackJAX and NumPyro Hamiltonian samplers."""

    uses_distribution_priors = True
    # run_sampler saves both; model_log_likelihood is the more meaningful of
    # the two downstream because the Jacobian has been divided out.
    archive_log_likelihood_keys = ("model_log_likelihood", "logdensity")

    #: Name used in printed progress lines, e.g. "BlackJAX".
    backend_label: ClassVar[str] = ""
    #: Dataclass used to hold this backend's results.
    results_class: ClassVar[type] = MCMCSamplingResults

    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_jax()
        self.parameter_names = None

    @property
    def algorithm_name(self):
        """``"hmc"`` or ``"nuts"``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #
    @classmethod
    def _run_optax_warmup(
        cls,
        *,
        context: SamplerContext,
        sampler_kwargs,
        warmup_optimizer_kwargs,
        logdensity,
        transform_positions,
        free_parameter_names,
        initial_position,
        on_optimum=None,
    ):
        """Replace the prior-median start with an optax MAP estimate.

        The default ``initial_position`` is the unconstrained origin, which maps
        (via ``cube = sigmoid(z)``) to the prior median of every parameter -- a
        stiff, often ill-conditioned, divergence-prone place to start a
        Hamiltonian sampler. This runs the optax optimiser (Adam by default)
        first so sampling begins inside the bulk of the posterior, prints and
        saves the MAP point, and fans it out into per-chain starts.

        Callers can override the optimiser's own starting point through
        ``optimizer_kwargs["initial_position"]``.

        Args:
            on_optimum: optional callback receiving the ``OptimizeResult``
                before the warm-up state is released. NumPyro uses it to seed
                the inverse mass matrix from the MAP Hessian, which must happen
                while the optimum is still in hand.

        Returns:
            ``(initial_position, best_logdensity)``.
        """
        from petitRADTRANS.retrieval.samplers.optimal_estimation import (
            optimize_unconstrained_position,
            make_walker_inits,
            report_and_save_map_warmup,
        )

        walker_jitter = warmup_optimizer_kwargs.pop("walker_jitter", 0.0)
        requested_num_chains = sampler_kwargs.get("num_chains", 1)
        warmup_seed = warmup_optimizer_kwargs.pop(
            "seed", sampler_kwargs.get("seed", 42654)
        )
        warmup_initial_position = warmup_optimizer_kwargs.pop("initial_position", None)
        if warmup_initial_position is not None:
            initial_position = _opt.jnp.asarray(
                warmup_initial_position, dtype=initial_position.dtype
            )

        optimize_result = optimize_unconstrained_position(
            logdensity, initial_position, seed=warmup_seed, **warmup_optimizer_kwargs
        )
        best_logdensity = float(optimize_result.best_logdensity)

        # Print the MAP parameter values and save the MAP sample to file.
        report_and_save_map_warmup(
            optimize_result,
            transform_positions,
            free_parameter_names,
            context.output_directory,
            context.retrieval_name,
            output_subdirectory=cls.output_subdirectory,
            verbose=warmup_optimizer_kwargs.get("verbose", True),
        )

        if on_optimum is not None:
            on_optimum(optimize_result)

        initial_position = make_walker_inits(
            optimize_result.best_position,
            requested_num_chains,
            jitter=walker_jitter,
            seed=warmup_seed,
        )
        del optimize_result
        # Drop the warm-up executables (Adam/L-BFGS/Hessian jits and their
        # device constants/workspaces) before the sampler's step function.
        _opt.clear_caches()

        return initial_position, best_logdensity

    @staticmethod
    def _common_run_kwargs(
        sampler_kwargs, *, initial_position, free_parameter_names, log_jacobian, num_chains
    ):
        """The ``run_sampler`` kwargs both backends accept, with their defaults."""
        return {
            "initial_position": sampler_kwargs.pop("initial_position", initial_position),
            "parameter_names": sampler_kwargs.pop(
                "parameter_names", free_parameter_names
            ),
            "num_samples": sampler_kwargs.pop("num_samples", 1000),
            "num_warmup": sampler_kwargs.pop(
                "num_warmup", sampler_kwargs.pop("num_burnin", 1000)
            ),
            "initial_step_size": sampler_kwargs.pop("initial_step_size", 1.0),
            "target_acceptance_rate": sampler_kwargs.pop("target_acceptance_rate", 0.8),
            "is_mass_matrix_diagonal": sampler_kwargs.pop(
                "is_mass_matrix_diagonal", True
            ),
            "progress_bar": sampler_kwargs.pop("progress_bar", False),
            "seed": sampler_kwargs.pop("seed", 42654),
            "use_jit": sampler_kwargs.pop("use_jit", sampler_kwargs.pop("jit", True)),
            "log_jacobian_fn": log_jacobian,
            "num_chains": num_chains,
        }

    @staticmethod
    def _common_summary_parameters(algorithm_name, run_kwargs, extra=None):
        """The summary entries both backends report. ``extra`` is inserted mid-dict
        so each backend keeps its historical key order in the summary file."""
        return {
            "sampler_type": algorithm_name,
            "num_samples": run_kwargs["num_samples"],
            "num_warmup": run_kwargs["num_warmup"],
            "initial_step_size": run_kwargs["initial_step_size"],
            "target_acceptance_rate": run_kwargs["target_acceptance_rate"],
            "is_mass_matrix_diagonal": run_kwargs["is_mass_matrix_diagonal"],
            "num_chains": run_kwargs["num_chains"],
            **(extra or {}),
            "seed": run_kwargs["seed"],
            "use_jit": run_kwargs["use_jit"],
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_results(
        self,
        parameter_names,
        samples,
        unconstrained_samples,
        logdensity,
        model_log_likelihood,
        acceptance_rate,
        is_divergent,
        num_integration_steps,
        warmup_parameters,
        metadata,
    ):
        output_prefix = self._get_output_prefix()
        samples_file = f"{output_prefix}_samples.npz"
        params_file = f"{output_prefix}_params.json"
        metadata_file = f"{output_prefix}_metadata.json"

        arrays_to_save = {
            "samples": samples,
            "unconstrained_samples": unconstrained_samples,
            "logdensity": logdensity,
        }
        for name, array in (
            ("model_log_likelihood", model_log_likelihood),
            ("acceptance_rate", acceptance_rate),
            ("is_divergent", is_divergent),
            ("num_integration_steps", num_integration_steps),
        ):
            if array is not None:
                arrays_to_save[name] = array

        np.savez(samples_file, **arrays_to_save)

        with open(params_file, "w") as handle:
            json.dump(parameter_names, handle, indent=4)

        metadata["warmup_parameters"] = self._to_serializable(warmup_parameters)
        with open(metadata_file, "w") as handle:
            json.dump(self._to_serializable(metadata), handle, indent=4)

        return samples_file, metadata_file

    def load_results(self):
        samples_file = self._get_output_prefix() + "_samples.npz"
        params_file = self._get_output_prefix() + "_params.json"
        metadata_file = self._get_output_prefix() + "_metadata.json"

        if not os.path.exists(samples_file):
            return self._load_results_from_generic_archive()

        if os.path.exists(params_file):
            with open(params_file, "r") as handle:
                self.parameter_names = list(json.load(handle))

        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)

        with np.load(samples_file, allow_pickle=False) as archive:
            def optional(name, dtype=float):
                if name not in archive.files:
                    return None
                return np.asarray(archive[name], dtype=dtype)

            self.results = self.results_class(
                algorithm=str(metadata.get("algorithm", self.algorithm_name)),
                samples=np.asarray(archive["samples"], dtype=float),
                unconstrained_samples=np.asarray(
                    archive["unconstrained_samples"], dtype=float
                ),
                logdensity=np.asarray(archive["logdensity"], dtype=float),
                model_log_likelihood=optional("model_log_likelihood"),
                warmup_parameters=dict(metadata.get("warmup_parameters", {})),
                acceptance_rate=optional("acceptance_rate"),
                is_divergent=optional("is_divergent", dtype=None),
                num_integration_steps=optional("num_integration_steps"),
                output_file=samples_file,
                metadata_file=metadata_file if os.path.exists(metadata_file) else None,
            )

        return self.results

    def _load_results_from_generic_archive(self):
        """Fall back to the equal-weight ``out_samples`` archive.

        That archive keeps only samples and log likelihoods, so the physical
        and unconstrained draws are the same array and every per-draw
        diagnostic is lost.
        """
        archived_results, parameter_names = self._load_sample_archive_results()

        if archived_results is None:
            return None

        if parameter_names:
            self.parameter_names = list(parameter_names)

        archived_log_likelihood = np.asarray(
            archived_results["log_likelihood"], dtype=float
        )
        archived_samples = np.asarray(archived_results["samples"], dtype=float)
        self.results = self.results_class(
            algorithm=self.algorithm_name,
            samples=archived_samples,
            unconstrained_samples=archived_samples,
            logdensity=archived_log_likelihood,
            model_log_likelihood=archived_log_likelihood,
            warmup_parameters={},
            acceptance_rate=None,
            is_divergent=None,
            num_integration_steps=None,
            output_file=archived_results.get("output_file"),
            metadata_file=None,
        )
        return self.results

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def pretty_print_results(self):
        if self.results is None:
            return

        print(
            f"{self.backend_label} {self.algorithm_name.upper()} completed: "
            f"{self.results.samples.shape[0]} samples"
        )
        if self.results.acceptance_rate is not None:
            print(f"  Mean acceptance rate: {np.mean(self.results.acceptance_rate):.3f}")
        if self.results.is_divergent is not None:
            print(f"  Divergences: {int(np.sum(self.results.is_divergent))}")

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        results = self.results

        def statistic(function, array_name):
            if results is None:
                return None
            array = getattr(results, array_name)
            if array is None:
                return None
            return function(array)

        summary["diagnostics"] = {
            "num_samples": (
                int(results.samples.shape[0]) if results is not None else None
            ),
            "mean_logdensity": statistic(lambda a: float(np.mean(a)), "logdensity"),
            "max_logdensity": statistic(lambda a: float(np.max(a)), "logdensity"),
            "mean_acceptance_rate": statistic(
                lambda a: float(np.mean(a)), "acceptance_rate"
            ),
            "n_divergent": statistic(lambda a: int(np.sum(a)), "is_divergent"),
            "mean_integration_steps": statistic(
                lambda a: float(np.mean(a)), "num_integration_steps"
            ),
        }
        if results is not None:
            summary["parameters"] = dict(results.warmup_parameters)
            summary["posterior"] = self._posterior_summary_from_matrix(
                results.samples,
                free_parameter_names,
            )
        return summary
