"""Nautilus backend: importance nested sampling boosted by neural networks.

Nautilus' posterior is weighted, so :meth:`NautilusSampler._build_posterior`
draws an equal-weight posterior for downstream plotting while keeping the
weighted representation alongside it.

Three things drive the design here. Nautilus transforms every point in its main
process, so the prior transform has to be compiled, which is why it is always
run in vectorized mode. Its checkpoint holds data rather than pickled objects,
so closures are fine here, unlike for Dynesty, but it is also restored without
comparing it to the problem at hand, which :meth:`NautilusSampler._prepare_checkpoint`
makes up for. And it neither recognises nor stops on a retrieval whose every
point is invalid, which :meth:`NautilusSampler._draw_until_valid` guards against.

Parallelism is a ``spawn`` process pool on one node or an ``MPICommExecutor``
across nodes. ``spawn`` because JAX is already running by then, and forking a
multithreaded process can deadlock; ``MPICommExecutor`` because the launch mode
the Nautilus documentation recommends, ``python -m mpi4py.futures``, runs the
script on rank 0 only, which would hang on the barriers in ``Retrieval.run``.

There is no ``pin_pool_workers`` as for Dynesty. Importing pRT starts XLA, and a
``spawn`` worker imports it before it runs any code of ours, so pinning could no
longer bound XLA's thread pool. Binding MPI ranks at launch does.
"""

import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
import time
import warnings

import numpy as np
from scipy.special import logsumexp

from petitRADTRANS.retrieval._mpi import get_mpi_context
from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import (
    Sampler,
    SamplerContext,
    equal_weight_log_weights,
)
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_batched_prior_transform,
    build_device_likelihood_wrapper,
    build_jitted_batched_likelihood,
    resolve_scalar_likelihood,
    vectorization_disable_reason,
)
from petitRADTRANS.retrieval.samplers.nautilus_pool import (
    BatchedChunkLogLikelihood,
    NautilusPoolLogLikelihood,
    ScalarChunkLogLikelihood,
    WorkerPool,
    evaluate_in_worker,
    initialize_worker,
    sanitize_log_likelihoods,
)

#: Largest number of dimensions the Nautilus documentation reports testing.
_MAX_TESTED_DIMENSIONS = 60

#: Sampler settings a checkpoint must have been written with to be resumed.
#: Stopping criteria are absent on purpose: tightening them is why one resumes.
_CHECKPOINT_SETTINGS = (
    "n_live",
    "n_update",
    "enlarge_per_dim",
    "n_points_min",
    "split_threshold",
    "periodic",
    "n_networks",
    "n_like_new_bound",
)

#: The run statistics kept with the saved posterior and copied into the results.
_RESULT_METADATA_KEYS = (
    "n_like",
    "n_eff",
    "eta",
    "logz",
    "logzerr",
    "converged",
    "discard_exploration",
)


class NautilusSampler(Sampler):
    """Importance nested sampling with Nautilus."""

    sampler_type = "nautilus"
    output_subdirectory = "out_Nautilus"
    archive_log_likelihood_keys = ("log_likelihood", "weighted_log_likelihood")

    def __init__(
        self,
        log_likelihood_func,
        prior_func,
        output_directory,
        retrieval_name,
    ):
        # Nautilus is only imported to sample, so that run_mode="evaluate" can
        # reload the saved results without it.
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        self.sampler = None
        self.parameter_names = []
        self._mpi_comm = None
        self._reference_log_likelihood = None
        self._eager_prior_transform = None

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        _opt.require_nautilus()

        n_dim = context.n_free_parameters
        if n_dim < 2:
            raise ValueError(
                f"Nautilus requires at least 2 free parameters; this retrieval has {n_dim}."
            )
        if n_dim > _MAX_TESTED_DIMENSIONS:
            warnings.warn(
                f"This retrieval has {n_dim} free parameters, and Nautilus reports testing "
                f"up to {_MAX_TESTED_DIMENSIONS}. Its efficiency eventually breaks down in "
                "high dimensions; gradient-based samplers may be better suited.",
                RuntimeWarning,
            )

        if "pool" in sampler_kwargs:
            raise ValueError(
                "pRT builds the Nautilus pools itself, because a pool it does not own "
                "would receive the retrieval runtime with every task. Use pool_njobs on "
                "one node, or launch with mpiexec to run across nodes."
            )

        vectorized = sampler_kwargs.pop("vectorized", False)
        pool_njobs = sampler_kwargs.pop("pool_njobs", None)
        sampler_pool_njobs = sampler_kwargs.pop("sampler_pool_njobs", None)
        likelihood_devices = sampler_kwargs.pop("likelihood_devices", None)
        use_mpi = sampler_kwargs.pop("use_mpi", True)
        use_jit = sampler_kwargs.pop("use_jit", True)
        seed = sampler_kwargs.pop("seed", 42654)

        _, comm, rank = get_mpi_context()
        mpi_comm = comm if use_mpi and comm is not None and comm.Get_size() > 1 else None

        if vectorized:
            disable_reason = vectorization_disable_reason(context)
            if disable_reason is not None:
                warnings.warn(
                    f"Nautilus vectorized mode disabled because {disable_reason}.",
                    RuntimeWarning,
                )
                vectorized = False

        if pool_njobs is not None and sampler_pool_njobs is not None:
            raise ValueError(
                "pool_njobs already serves the sampler's own calculations; "
                "sampler_pool_njobs is for runs without a likelihood pool."
            )
        if pool_njobs is not None and mpi_comm is not None:
            raise ValueError(
                "pool_njobs cannot be combined with an MPI launch, which already "
                "parallelises over its ranks. Pass use_mpi=False to use the pool instead."
            )
        if pool_njobs is not None and vectorized:
            raise ValueError(
                "pool_njobs and vectorized cannot be combined, because every worker "
                "would batch on the same device. Launch with mpiexec to batch on several."
            )
        if pool_njobs is not None and likelihood_devices is not None:
            raise ValueError(
                "likelihood_devices cannot be combined with pool_njobs, because the "
                "devices are assigned per MPI rank. Launch with mpiexec instead."
            )

        scalar_log_likelihood, jit_enabled = resolve_scalar_likelihood(
            context, use_jit, "Nautilus"
        )
        target_device = None
        if likelihood_devices is not None:
            scalar_log_likelihood, target_device = build_device_likelihood_wrapper(
                dataclasses.replace(context, likelihood_devices=likelihood_devices)
            )
            jit_enabled = True

        if vectorized:
            log_likelihood_func = BatchedChunkLogLikelihood(
                build_jitted_batched_likelihood(context, target_device)
            )
            jit_enabled = True
        elif pool_njobs is not None and int(pool_njobs) > 1:
            log_likelihood_func = NautilusPoolLogLikelihood(
                parameter_layout=context.parameter_layout,
                runtime=context.runtime,
                uncertainties_mode=context.uncertainties_mode,
                print_log_likelihood_for_debugging=context.print_log_likelihood_for_debugging,
                use_jit=jit_enabled,
            )
        else:
            log_likelihood_func = ScalarChunkLogLikelihood(scalar_log_likelihood)

        sampler = cls(
            log_likelihood_func=log_likelihood_func,
            prior_func=build_batched_prior_transform(context),
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        sampler.parameter_names = list(context.free_parameter_names)
        sampler._mpi_comm = mpi_comm
        sampler._reference_log_likelihood = scalar_log_likelihood
        sampler._eager_prior_transform = context.parameter_layout.prior_transform_numpy

        n_live = sampler_kwargs.pop("n_live", 2000)
        init_kwargs = {
            "n_dim": n_dim,
            "n_live": n_live,
            "n_update": sampler_kwargs.pop("n_update", None),
            "enlarge_per_dim": sampler_kwargs.pop("enlarge_per_dim", 1.1),
            "n_points_min": sampler_kwargs.pop("n_points_min", None),
            "split_threshold": sampler_kwargs.pop("split_threshold", 100),
            "periodic": sampler_kwargs.pop("periodic", None),
            "n_networks": sampler_kwargs.pop("n_networks", 4),
            "neural_network_kwargs": sampler_kwargs.pop("neural_network_kwargs", {}),
            "n_batch": sampler_kwargs.pop("n_batch", None),
            "n_like_new_bound": sampler_kwargs.pop("n_like_new_bound", None),
            "seed": seed,
            # An explicit filepath=None turns checkpointing off, as in Nautilus.
            "filepath": sampler_kwargs.pop(
                "filepath", sampler._get_output_prefix() + "_checkpoint.hdf5"
            ),
            "resume": sampler_kwargs.pop("resume", True),
        }
        nautilus_run_kwargs = {
            "f_live": sampler_kwargs.pop("f_live", 0.01),
            "n_shell": sampler_kwargs.pop("n_shell", 1),
            "n_eff": sampler_kwargs.pop("n_eff", 10000),
            "n_like_max": sampler_kwargs.pop("n_like_max", np.inf),
            # Nautilus defaults to False, but its own guidance is that True is
            # what makes the posterior and the evidence fully unbiased.
            "discard_exploration": bool(sampler_kwargs.pop("discard_exploration", True)),
            "timeout": sampler_kwargs.pop("timeout", np.inf),
            "verbose": sampler_kwargs.pop("verbose", True),
        }
        run_kwargs = {
            "init_kwargs": init_kwargs,
            "pool_njobs": pool_njobs,
            "sampler_pool_njobs": sampler_pool_njobs,
            "equal_weight_boost": sampler_kwargs.pop("equal_weight_boost", None),
            "max_initial_invalid_calls": sampler_kwargs.pop(
                "max_initial_invalid_calls", 100 * n_live
            ),
            "verify_checkpoint": sampler_kwargs.pop("verify_checkpoint", True),
            **nautilus_run_kwargs,
        }

        if sampler_kwargs:
            raise TypeError(
                f"Unknown Nautilus argument(s): {sorted(sampler_kwargs)}."
            )

        summary_parameters = {
            "sampler_type": "nautilus",
            "parameter_names": list(context.free_parameter_names),
            "seed": seed,
            "use_jit": jit_enabled,
            "vectorized": vectorized,
            "n_live": n_live,
            "resume": init_kwargs["resume"],
            "n_eff": nautilus_run_kwargs["n_eff"],
            "f_live": nautilus_run_kwargs["f_live"],
            "discard_exploration": nautilus_run_kwargs["discard_exploration"],
        }
        if pool_njobs is not None:
            summary_parameters["pool_njobs"] = int(pool_njobs)
        if sampler_pool_njobs is not None:
            summary_parameters["sampler_pool_njobs"] = int(sampler_pool_njobs)
        if mpi_comm is not None:
            summary_parameters["mpi_workers"] = mpi_comm.Get_size() - 1
        if likelihood_devices is not None:
            summary_parameters["likelihood_devices"] = [str(d) for d in likelihood_devices]
            summary_parameters["likelihood_target_device"] = str(target_device)
        if rank == 0 and getattr(context.configuration, "run_mode", None) == "retrieval":
            print(f"Starting Nautilus retrieval: {context.retrieval_name}")

        return sampler, run_kwargs, summary_parameters

    def post_run(self, results):
        return self.get_results(), {}

    @staticmethod
    def _start_pool(njobs, likelihood=None):
        """Start a ``spawn`` pool, or return ``None`` when one process is enough.

        This is an executor rather than a ``multiprocessing.Pool``, because a
        worker killed by the kernel (the usual way to run out of memory with
        opacity tables loaded) makes the latter wait forever, not raise.
        """
        if njobs is None or int(njobs) < 2:
            return None

        initializer_kwargs = (
            {"initializer": initialize_worker, "initargs": (likelihood,)}
            if likelihood is not None
            else {}
        )
        return WorkerPool(
            concurrent.futures.ProcessPoolExecutor(
                max_workers=int(njobs),
                mp_context=multiprocessing.get_context("spawn"),
                **initializer_kwargs,
            ),
            int(njobs),
        )

    def run_sampler(
        self,
        init_kwargs,
        pool_njobs=None,
        sampler_pool_njobs=None,
        equal_weight_boost=None,
        max_initial_invalid_calls=None,
        verify_checkpoint=True,
        **run_kwargs,
    ):
        _opt.require_nautilus()

        run_settings = {
            "init_kwargs": dict(init_kwargs),
            "equal_weight_boost": equal_weight_boost,
            "max_initial_invalid_calls": max_initial_invalid_calls,
            "verify_checkpoint": verify_checkpoint,
            "run_kwargs": dict(run_kwargs),
        }

        if self._mpi_comm is not None:
            from mpi4py.futures import MPICommExecutor

            initialize_worker(self.log_likelihood_func)
            # Every rank enters; the workers serve here until rank 0 is done,
            # and then get None.
            with MPICommExecutor(self._mpi_comm, root=0) as executor:
                if executor is None:
                    return None
                pool = WorkerPool(executor, self._mpi_comm.Get_size() - 1)
                return self._run_nautilus(
                    likelihood=evaluate_in_worker,
                    pool=(pool, pool),
                    **run_settings,
                )

        likelihood_pool = self._start_pool(pool_njobs, likelihood=self.log_likelihood_func)
        sampler_pool = (
            likelihood_pool if likelihood_pool is not None else self._start_pool(sampler_pool_njobs)
        )
        # pRT fixes its precision when imported, and a spawned worker imports it
        # with default JAX settings unless told otherwise. The executor starts
        # its workers on demand, so this has to hold for the whole run.
        original_x64_setting = os.environ.get("JAX_ENABLE_X64")
        if sampler_pool is not None:
            _opt.require_jax()
            os.environ["JAX_ENABLE_X64"] = str(_opt.jnp.zeros(0).dtype == _opt.jnp.float64)
        try:
            return self._run_nautilus(
                likelihood=(
                    evaluate_in_worker if likelihood_pool is not None else self.log_likelihood_func
                ),
                pool=(likelihood_pool, sampler_pool),
                **run_settings,
            )
        finally:
            for pool in {likelihood_pool, sampler_pool} - {None}:
                pool.executor.shutdown(wait=True, cancel_futures=True)
            if sampler_pool is not None:
                if original_x64_setting is None:
                    os.environ.pop("JAX_ENABLE_X64", None)
                else:
                    os.environ["JAX_ENABLE_X64"] = original_x64_setting

    def _run_nautilus(
        self,
        likelihood,
        pool,
        init_kwargs,
        equal_weight_boost,
        max_initial_invalid_calls,
        verify_checkpoint,
        run_kwargs,
    ):
        self._prepare_checkpoint(init_kwargs, verify_checkpoint)

        # Nautilus would cut the batch into uneven chunks otherwise, each new
        # shape recompiling a batched likelihood.
        n_workers = 1 if pool[0] is None else pool[0].size
        if init_kwargs["n_batch"] is not None and init_kwargs["n_batch"] % n_workers != 0:
            n_batch = (init_kwargs["n_batch"] // n_workers + 1) * n_workers
            warnings.warn(
                f"Raising n_batch from {init_kwargs['n_batch']} to {n_batch}, "
                f"a multiple of the {n_workers} pool workers.",
                RuntimeWarning,
            )
            init_kwargs["n_batch"] = n_batch

        self.sampler = _opt.NautilusBackend(
            self.prior_func,
            likelihood,
            vectorized=True,
            pass_dict=False,
            pool=pool,
            **init_kwargs,
        )

        # On a resumed run the checkpoint's choice would otherwise stand, since
        # Nautilus only applies the argument when exploration ends.
        if self.sampler.explored:
            self.sampler.discard_exploration = run_kwargs["discard_exploration"]

        deadline = time.monotonic() + run_kwargs["timeout"]
        self._draw_until_valid(max_initial_invalid_calls, deadline, run_kwargs)
        run_kwargs["timeout"] = max(deadline - time.monotonic(), 0.0)

        converged = bool(self.sampler.run(**run_kwargs))

        if not self.sampler.explored:
            warnings.warn(
                f"Nautilus stopped after {self.sampler.n_like} likelihood calls, before "
                "finishing its exploration phase, so there is no usable posterior yet "
                "and none was saved."
                + (
                    " Run the same script again to resume from the checkpoint."
                    if init_kwargs["filepath"] is not None
                    else ""
                ),
                RuntimeWarning,
            )
            self.results = None
            return None

        if not converged:
            warnings.warn(
                f"Nautilus stopped after {self.sampler.n_like} likelihood calls, with "
                f"an effective sample size of {self.sampler.n_eff:.0f} instead of the "
                f"requested {run_kwargs['n_eff']}. The posterior was saved, but is "
                "marked as not converged.",
                RuntimeWarning,
            )

        self.results = self._build_posterior(equal_weight_boost, converged)
        self.pretty_print_results()
        return self.results

    def _draw_until_valid(self, max_calls, deadline, run_kwargs):
        """Sample batch by batch until a point is valid, and raise if none is.

        Nautilus keeps drawing from the prior until a point is valid, so a
        retrieval that has none would never stop on its own.
        """
        sampler = self.sampler

        while not sampler.explored and not any(
            np.any(np.isfinite(log_l)) for log_l in sampler.log_l
        ):
            if max_calls is not None and sampler.n_like >= max_calls:
                raise RuntimeError(
                    f"All of the first {sampler.n_like} likelihood evaluations of retrieval "
                    f"'{self.retrieval_name}' were invalid, so there is nothing for Nautilus "
                    "to sample. Check the forward model and the priors, or raise "
                    "max_initial_invalid_calls if valid points really are that rare."
                )

            next_n_like = sampler.n_like + sampler.n_batch
            remaining = deadline - time.monotonic()
            if next_n_like > run_kwargs["n_like_max"] or remaining <= 0:
                return

            sampler.run(
                **{
                    **run_kwargs,
                    "n_like_max": next_n_like,
                    "timeout": remaining,
                    "verbose": False,
                }
            )

    def _checkpoint_fingerprint(self, init_kwargs):
        """What a checkpoint must agree on with the problem resuming from it.

        These are values rather than descriptions of the problem, because the
        prior and the model are arbitrary callables. The log-likelihood at the
        centre of the prior covers everything behind it: the data, the model and
        the fixed parameters.
        """
        n_dim = init_kwargs["n_dim"]
        settings = {key: init_kwargs[key] for key in _CHECKPOINT_SETTINGS}
        return {
            "parameter_names": list(self.parameter_names),
            "prior_quantiles": [
                self._eager_prior_transform(np.full(n_dim, quantile)).tolist()
                for quantile in (0.1, 0.5, 0.9)
            ],
            "reference_log_likelihood": float(
                sanitize_log_likelihoods(
                    self._reference_log_likelihood(
                        self._eager_prior_transform(np.full(n_dim, 0.5))
                    )
                )
            ),
            "settings": self._to_serializable(settings),
        }

    def _prepare_checkpoint(self, init_kwargs, verify_checkpoint):
        """Refuse to resume from a checkpoint of a different problem.

        Nautilus restores a checkpoint without comparing it to the problem at
        hand: a different likelihood or ``n_live`` goes unnoticed, and a
        different dimension only fails later on, with an unrelated error.
        """
        checkpoint_file = init_kwargs["filepath"]
        if checkpoint_file is None:
            return

        fingerprint_file = os.path.splitext(checkpoint_file)[0] + "_fingerprint.json"
        resuming = init_kwargs["resume"] and os.path.exists(checkpoint_file)

        if resuming and not verify_checkpoint:
            return

        fingerprint = self._checkpoint_fingerprint(init_kwargs)

        if resuming and os.path.exists(fingerprint_file):
            with open(fingerprint_file, "r") as handle:
                saved_fingerprint = json.load(handle)

            differing = _differing_fingerprint_entries(saved_fingerprint, fingerprint)
            if differing:
                raise ValueError(
                    f"The Nautilus checkpoint '{checkpoint_file}' belongs to a different "
                    f"problem: its {', '.join(differing)} differ. Pass resume=False to start "
                    "over and overwrite it, or verify_checkpoint=False if you know it is "
                    "still valid."
                )
            print(f"Resuming Nautilus from checkpoint '{checkpoint_file}'.")
            return

        if resuming:
            warnings.warn(
                f"The Nautilus checkpoint '{checkpoint_file}' has no fingerprint, so it "
                "cannot be checked against this retrieval. Resuming from it regardless.",
                RuntimeWarning,
            )

        with open(fingerprint_file, "w") as handle:
            json.dump(fingerprint, handle, indent=4)

    def _build_posterior(self, equal_weight_boost, converged):
        sampler = self.sampler

        points, log_weights, log_likelihood = sampler.posterior()
        points = np.asarray(points, dtype=float)
        valid = np.isfinite(log_weights)
        points, log_weights, log_likelihood = (
            points[valid],
            log_weights[valid],
            log_likelihood[valid],
        )
        log_weights = log_weights - logsumexp(log_weights)

        # At a boost of 1 the equal-weight posterior holds far fewer draws than
        # the effective sample size, because the scale is set by the largest
        # weight, so aim for the effective sample size by default.
        if equal_weight_boost is None:
            equal_weight_boost = max(
                1.0,
                sampler.n_eff / np.sum(np.exp(log_weights - np.max(log_weights))),
            )

        samples, _, sample_log_likelihood = sampler.posterior(
            equal_weight=True, equal_weight_boost=equal_weight_boost
        )
        samples = np.array(samples, dtype=float)
        sample_log_likelihood = np.array(sample_log_likelihood, dtype=float)

        # As for Dynesty: the best fit is read from these samples, and the draw
        # can lose the maximum-likelihood point.
        best_index = int(np.argmax(log_likelihood))
        samples[0] = points[best_index]
        sample_log_likelihood[0] = log_likelihood[best_index]

        n_eff = float(sampler.n_eff)
        metadata = {
            "algorithm": "nautilus",
            "parameter_names": list(self.parameter_names),
            "n_like": int(sampler.n_like),
            "n_eff": n_eff,
            "eta": float(sampler.eta),
            "logz": float(sampler.log_z),
            # Nautilus' statistical error estimate; it does not cover systematics.
            "logzerr": 1.0 / np.sqrt(n_eff),
            "equal_weight_boost": float(equal_weight_boost),
            "converged": converged,
            "discard_exploration": bool(sampler.discard_exploration),
        }
        samples_file, metadata_file = self._save_results(
            parameter_names=self.parameter_names,
            samples=samples,
            log_likelihood=sample_log_likelihood,
            weighted_samples=points,
            weighted_log_likelihood=log_likelihood,
            weighted_log_weights=log_weights,
            metadata=metadata,
        )

        return {
            "samples": samples,
            "log_likelihood": sample_log_likelihood,
            "log_weights": equal_weight_log_weights(samples.shape[0]),
            "weighted_samples": {
                "points": points,
                "logl": log_likelihood,
                "log_weights": log_weights,
            },
            "output_file": samples_file,
            "metadata_file": metadata_file,
            **{key: metadata[key] for key in _RESULT_METADATA_KEYS},
        }

    def load_results(self):
        return self._load_weighted_results(_RESULT_METADATA_KEYS)

    def pretty_print_results(self):
        if not self.results:
            return

        print(
            f"Nautilus completed: {self.results['samples'].shape[0]} posterior samples"
        )
        if self.results.get("logz") is not None:
            print(
                f"  ln Z: {self.results['logz']:.3f} +- {self.results.get('logzerr', 0.0):.3f}"
            )

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if not self.results:
            return summary

        summary["evidence"] = {
            "ln_z": self._summary_scalar(self.results.get("logz")),
            "ln_z_err": self._summary_scalar(self.results.get("logzerr")),
        }
        summary["diagnostics"] = {
            key: self._summary_scalar(self.results.get(key))
            for key in ("n_like", "n_eff", "eta", "converged", "discard_exploration")
        }
        summary["posterior"] = self._posterior_summary_from_matrix(
            self.results.get("samples"),
            list(free_parameter_names or self.parameter_names),
        )
        return summary


def _differing_fingerprint_entries(saved, current):
    """Names of the entries of two checkpoint fingerprints that disagree.

    The numbers come from a model evaluation, so they are compared with a
    tolerance that lets a checkpoint move between machines.
    """
    differing = []

    # A JSON round trip makes the current entries comparable to the saved ones.
    for key in ("parameter_names", "settings"):
        if saved.get(key) != json.loads(json.dumps(current[key])):
            differing.append(key)

    for key in ("prior_quantiles", "reference_log_likelihood"):
        if np.shape(saved.get(key)) != np.shape(current[key]) or not np.allclose(
            saved[key], current[key], rtol=1e-6, atol=0.0
        ):
            differing.append(key)

    return differing
