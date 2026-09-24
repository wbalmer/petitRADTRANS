"""Dynesty backend: dynamic nested sampling.

Dynesty's posterior is weighted, so :meth:`DynestySampler._build_equal_weight_posterior`
resamples it to equal weight for downstream plotting while keeping the
weighted representation alongside it.

Two things drive the design here. Dynesty runs pure Python, so the likelihood
is worth JIT-compiling; and it pickles the sampler both for its worker pool
and for checkpointing, so the likelihood and prior handed to it must be
picklable -- hence the helper classes in
:mod:`~petitRADTRANS.retrieval.samplers.dynesty_pool` rather than closures.
"""

import json
import os
import warnings

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import (
    Sampler,
    SamplerContext,
    equal_weight_log_weights,
)
from petitRADTRANS.retrieval.samplers.dynesty_pool import (
    DynestyPoolLogLikelihood,
    DynestyPoolPriorTransform,
)


class DynestySampler(Sampler):
    """Dynamic nested sampling with Dynesty."""

    sampler_type = "dynesty"
    output_subdirectory = "out_Dynesty"
    archive_log_likelihood_keys = ("log_likelihood", "weighted_log_likelihood")

    def __init__(
        self,
        log_likelihood_func,
        prior_func,
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_dynesty()
        self.sampler = None
        self.raw_results = None
        self.parameter_names = []
        self._run_summary = {}
        self._dynesty_pool = None
        self._owns_dynesty_pool = False

    @staticmethod
    def _default_use_pool() -> dict[str, bool]:
        return {
            "prior_transform": False,
            "loglikelihood": True,
            "propose_point": True,
            "update_bound": True,
        }

    @classmethod
    def _configure_dynesty_pool(
        cls,
        log_likelihood_func,
        prior_func,
        *,
        pool,
        pool_njobs,
    ):
        if pool is not None and pool_njobs is not None:
            raise ValueError("Pass either 'pool' or 'pool_njobs' for dynesty, not both.")

        if pool is not None:
            pool_loglike = getattr(pool, "loglike", log_likelihood_func)
            pool_prior = getattr(pool, "prior_transform", prior_func)
            return pool_loglike, pool_prior, pool

        if pool_njobs is None:
            return log_likelihood_func, prior_func, None

        pool_size = max(1, int(pool_njobs))
        if pool_size < 2:
            return log_likelihood_func, prior_func, None

        dynesty_pool = None
        try:
            dynesty_pool = _opt.DynestyPool(pool_size, log_likelihood_func, prior_func)
            dynesty_pool.__enter__()
        except Exception:
            try:
                if dynesty_pool is not None:
                    dynesty_pool.__exit__(None, None, None)
            except Exception:
                pass
            raise

        return (
            dynesty_pool.loglike,
            dynesty_pool.prior_transform,
            dynesty_pool,
        )

    def _close_dynesty_pool(self):
        close_error = None

        try:
            if self._owns_dynesty_pool and self._dynesty_pool is not None:
                try:
                    self._dynesty_pool.__exit__(None, None, None)
                except Exception as error:
                    close_error = error
        finally:
            self._dynesty_pool = None
            self._owns_dynesty_pool = False

        if close_error is not None:
            warnings.warn(
                f"Failed to close Dynesty pool cleanly: {close_error}",
                RuntimeWarning,
            )

    @classmethod
    def _build_dynesty_interface(
        cls,
        context: SamplerContext,
        *,
        use_jit: bool = True,
        pool_safe: bool = False,
    ):
        prior_transform = context.prior_ultranest_func
        log_likelihood_func = context.log_likelihood_func

        if pool_safe:
            helper_use_jit = use_jit

            if helper_use_jit:
                try:
                    _opt.require_jax()
                except ImportError:
                    warnings.warn(
                        "JAX is unavailable; Dynesty pool workers will use the non-jitted scalar likelihood.",
                        RuntimeWarning,
                    )
                    helper_use_jit = False

            if helper_use_jit and any(
                group.model_contract != "differentiable"
                for group in context.runtime.model_groups
            ):
                warnings.warn(
                    "Dynesty JIT likelihood disabled because the retrieval contains legacy model groups.",
                    RuntimeWarning,
                )
                helper_use_jit = False

            return (
                DynestyPoolLogLikelihood(
                    parameter_layout=context.parameter_layout,
                    runtime=context.runtime,
                    uncertainties_mode=context.uncertainties_mode,
                    print_log_likelihood_for_debugging=context.print_log_likelihood_for_debugging,
                    use_jit=helper_use_jit,
                ),
                DynestyPoolPriorTransform(context.parameter_layout),
                helper_use_jit,
            )

        if not use_jit:
            return log_likelihood_func, prior_transform, False

        try:
            _opt.require_jax()
        except ImportError:
            warnings.warn(
                "JAX is unavailable; Dynesty will use the non-jitted scalar likelihood.",
                RuntimeWarning,
            )
            return log_likelihood_func, prior_transform, False

        if any(
            group.model_contract != "differentiable"
            for group in context.runtime.model_groups
        ):
            warnings.warn(
                "Dynesty JIT likelihood disabled because the retrieval contains legacy model groups.",
                RuntimeWarning,
            )
            return log_likelihood_func, prior_transform, False

        # Return picklable helper callables instead of local closures so that
        # Dynesty checkpointing (which pickles the whole sampler) can serialize
        # the likelihood and prior transform.
        return (
            DynestyPoolLogLikelihood(
                parameter_layout=context.parameter_layout,
                runtime=context.runtime,
                uncertainties_mode=context.uncertainties_mode,
                print_log_likelihood_for_debugging=context.print_log_likelihood_for_debugging,
                use_jit=True,
            ),
            DynestyPoolPriorTransform(context.parameter_layout),
            True,
        )

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        _opt.require_dynesty()

        pool = sampler_kwargs.pop("pool", None)
        pool_njobs = sampler_kwargs.pop("pool_njobs", None)
        queue_size = sampler_kwargs.pop("queue_size", None)
        use_pool = sampler_kwargs.pop("use_pool", None)
        use_jit = sampler_kwargs.pop("use_jit", True)
        pool_safe = pool is not None or pool_njobs is not None
        log_likelihood_func, prior_func, jit_enabled = cls._build_dynesty_interface(
            context,
            use_jit=use_jit,
            pool_safe=pool_safe,
        )
        seed = sampler_kwargs.pop("seed", 42654)

        (
            log_likelihood_func,
            prior_func,
            dynesty_pool,
        ) = cls._configure_dynesty_pool(
            log_likelihood_func,
            prior_func,
            pool=pool,
            pool_njobs=pool_njobs,
        )

        if dynesty_pool is not None and queue_size is None:
            queue_size = dynesty_pool.size

        if use_pool is None and (dynesty_pool is not None or pool is not None):
            use_pool = cls._default_use_pool()

        init_kwargs = {
            "ndim": context.n_free_parameters,
            "nlive": sampler_kwargs.pop("nlive", 500),
            "bound": sampler_kwargs.pop("bound", "multi"),
            "sample": sampler_kwargs.pop("sample", "auto"),
            "periodic": sampler_kwargs.pop("periodic", None),
            "reflective": sampler_kwargs.pop("reflective", None),
            "update_interval": sampler_kwargs.pop("update_interval", None),
            "first_update": sampler_kwargs.pop("first_update", None),
            "rstate": sampler_kwargs.pop("rstate", np.random.default_rng(seed)),
            "queue_size": queue_size,
            "pool": dynesty_pool if dynesty_pool is not None else pool,
            "use_pool": use_pool,
            "enlarge": sampler_kwargs.pop("enlarge", None),
            "bootstrap": sampler_kwargs.pop("bootstrap", None),
            "walks": sampler_kwargs.pop("walks", None),
            "facc": sampler_kwargs.pop("facc", 0.5),
            "slices": sampler_kwargs.pop("slices", None),
            "ncdim": sampler_kwargs.pop("ncdim", None),
            "history_filename": sampler_kwargs.pop("history_filename", None),
            "save_evaluation_history": sampler_kwargs.pop(
                "save_evaluation_history", False
            ),
        }

        run_kwargs = {
            "seed": seed,
            "nlive_init": sampler_kwargs.pop("nlive_init", None),
            "maxiter_init": sampler_kwargs.pop("maxiter_init", None),
            "maxcall_init": sampler_kwargs.pop("maxcall_init", None),
            "dlogz_init": sampler_kwargs.pop("dlogz_init", 0.01),
            "logl_max_init": sampler_kwargs.pop("logl_max_init", np.inf),
            "nlive_batch": sampler_kwargs.pop("nlive_batch", None),
            "wt_function": sampler_kwargs.pop("wt_function", None),
            "wt_kwargs": sampler_kwargs.pop("wt_kwargs", None),
            "maxiter_batch": sampler_kwargs.pop("maxiter_batch", None),
            "maxcall_batch": sampler_kwargs.pop("maxcall_batch", None),
            "maxiter": sampler_kwargs.pop("maxiter", None),
            "maxcall": sampler_kwargs.pop("maxcall", None),
            "maxbatch": sampler_kwargs.pop("maxbatch", None),
            "n_effective": sampler_kwargs.pop("n_effective", None),
            "stop_function": sampler_kwargs.pop("stop_function", None),
            "stop_kwargs": sampler_kwargs.pop("stop_kwargs", None),
            "use_stop": sampler_kwargs.pop("use_stop", True),
            "save_bounds": sampler_kwargs.pop("save_bounds", True),
            "print_progress": sampler_kwargs.pop("print_progress", True),
            "live_points": sampler_kwargs.pop("live_points", None),
            "resume": sampler_kwargs.pop("resume", False),
            "checkpoint_file": sampler_kwargs.pop("checkpoint_file", None),
            "checkpoint_every": sampler_kwargs.pop("checkpoint_every", 60),
        }
        run_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=log_likelihood_func,
            prior_func=prior_func,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        sampler.parameter_names = list(context.free_parameter_names)
        sampler._dynesty_pool = dynesty_pool
        sampler._owns_dynesty_pool = dynesty_pool is not None

        summary_parameters = {
            "sampler_type": "dynesty",
            "parameter_names": list(context.free_parameter_names),
            "seed": seed,
            "use_jit": jit_enabled,
            "nlive": init_kwargs["nlive"],
            "bound": init_kwargs["bound"],
            "sample": init_kwargs["sample"],
        }
        if pool_njobs is not None:
            summary_parameters["pool_njobs"] = int(pool_njobs)
        if init_kwargs["queue_size"] is not None:
            summary_parameters["queue_size"] = init_kwargs["queue_size"]
        if init_kwargs["use_pool"] is not None:
            summary_parameters["use_pool"] = init_kwargs["use_pool"]
        for key in (
            "nlive_init",
            "nlive_batch",
            "dlogz_init",
            "maxiter",
            "maxcall",
            "maxbatch",
            "n_effective",
            "use_stop",
        ):
            if run_kwargs.get(key) is not None:
                summary_parameters[key] = run_kwargs[key]

        return sampler, {"init_kwargs": init_kwargs, **run_kwargs}, summary_parameters

    def post_run(self, results):
        return self.get_results(), {}

    @staticmethod
    def _normalize_weights(weights):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size == 0:
            return weights

        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            return np.full(weights.shape, 1.0 / weights.size, dtype=float)

        return weights / weight_sum

    def _save_results(
        self,
        parameter_names,
        samples,
        log_likelihood,
        weighted_samples,
        weighted_log_likelihood,
        weighted_log_weights,
        metadata,
    ):
        output_prefix = self._get_output_prefix()
        samples_file = f"{output_prefix}_samples.npz"
        params_file = f"{output_prefix}_params.json"
        metadata_file = f"{output_prefix}_metadata.json"

        np.savez(
            samples_file,
            samples=samples,
            log_likelihood=log_likelihood,
            weighted_samples=weighted_samples,
            weighted_log_likelihood=weighted_log_likelihood,
            weighted_log_weights=weighted_log_weights,
        )

        with open(params_file, "w") as handle:
            json.dump(parameter_names, handle, indent=4)

        with open(metadata_file, "w") as handle:
            json.dump(metadata, handle, indent=4)

        return samples_file, metadata_file

    def _build_equal_weight_posterior(self, raw_results, seed):
        weighted_samples = np.asarray(raw_results.samples, dtype=float)
        weighted_log_likelihood = np.asarray(raw_results.logl, dtype=float).reshape(-1)
        weights = self._normalize_weights(raw_results.importance_weights())
        weighted_log_weights = np.log(np.clip(weights, np.finfo(float).tiny, None))

        combined = np.column_stack((weighted_samples, weighted_log_likelihood))
        equal_weight = np.asarray(
            _opt.dyfunc.resample_equal(combined, weights, rstate=np.random.default_rng(seed)),
            dtype=float,
        )

        if equal_weight.shape[0] > 0:
            max_index = int(np.argmax(weighted_log_likelihood))
            equal_weight[0, :-1] = weighted_samples[max_index]
            equal_weight[0, -1] = weighted_log_likelihood[max_index]

        return {
            "samples": equal_weight[:, :-1],
            "log_likelihood": equal_weight[:, -1],
            "log_weights": equal_weight_log_weights(equal_weight.shape[0]),
            "weighted_samples": {
                "points": weighted_samples,
                "logl": weighted_log_likelihood,
                "log_weights": weighted_log_weights,
            },
        }

    @staticmethod
    def _summarize_ncall(raw_results):
        ncall = getattr(raw_results, "ncall", 0)

        if ncall is None:
            return 0

        ncall_array = np.asarray(ncall)

        if ncall_array.size == 0:
            return 0

        if ncall_array.ndim == 0:
            return int(ncall_array.item())

        return int(np.sum(ncall_array, dtype=np.int64))

    def run_sampler(self, init_kwargs, **run_kwargs):
        _opt.require_dynesty()

        init_kwargs = dict(init_kwargs)
        run_kwargs = dict(run_kwargs)
        seed = run_kwargs.pop("seed", 42654)
        checkpoint_file = run_kwargs.get("checkpoint_file")

        try:
            if checkpoint_file is None:
                checkpoint_file = self._get_output_prefix() + "_checkpoint.save"
                run_kwargs["checkpoint_file"] = checkpoint_file

            resume = bool(run_kwargs.get("resume", False))
            pool = init_kwargs.get("pool")
            checkpoint_exists = bool(
                checkpoint_file and os.path.exists(checkpoint_file)
            )

            if resume and checkpoint_exists:
                self.sampler = _opt.DynamicNestedSampler.restore(checkpoint_file, pool=pool)
            else:
                if resume and not checkpoint_exists:
                    warnings.warn(
                        f"Dynesty checkpoint '{checkpoint_file}' was not found; "
                        "starting a fresh run with resume=False.",
                        RuntimeWarning,
                    )
                    run_kwargs["resume"] = False

                self.sampler = _opt.DynamicNestedSampler(
                    self.log_likelihood_func,
                    self.prior_func,
                    **init_kwargs,
                )

            print(f"Starting Dynesty retrieval: {self.retrieval_name}")
            self.sampler.run_nested(**run_kwargs)
            self.raw_results = self.sampler.results

            derived_results = self._build_equal_weight_posterior(
                self.raw_results,
                seed=seed,
            )

            self._run_summary = {
                "niter": int(getattr(self.raw_results, "niter", 0)),
                "ncall": self._summarize_ncall(self.raw_results),
                "eff": float(getattr(self.raw_results, "eff", 0.0)),
                "n_effective": float(getattr(self.sampler, "n_effective", 0.0)),
                "logz": float(np.asarray(self.raw_results.logz)[-1]),
                "logzerr": float(np.asarray(self.raw_results.logzerr)[-1]),
            }
            metadata = {
                "algorithm": "dynesty_dynamic",
                "parameter_names": list(self.parameter_names),
                **self._run_summary,
            }

            samples_file, metadata_file = self._save_results(
                parameter_names=self.parameter_names,
                samples=derived_results["samples"],
                log_likelihood=derived_results["log_likelihood"],
                weighted_samples=derived_results["weighted_samples"]["points"],
                weighted_log_likelihood=derived_results["weighted_samples"]["logl"],
                weighted_log_weights=derived_results["weighted_samples"]["log_weights"],
                metadata=metadata,
            )

            self.results = {
                **derived_results,
                "output_file": samples_file,
                "metadata_file": metadata_file,
                "niter": self._run_summary["niter"],
                "ncall": self._run_summary["ncall"],
                "eff": self._run_summary["eff"],
                "n_effective": self._run_summary["n_effective"],
                "logz": self._run_summary["logz"],
                "logzerr": self._run_summary["logzerr"],
            }

            self.pretty_print_results()
            self.plot_diagnostics(self.results)
            return self.results
        finally:
            self._close_dynesty_pool()

    def get_results(self):
        return self.results

    def load_results(self):
        samples_file = self._get_output_prefix() + "_samples.npz"
        params_file = self._get_output_prefix() + "_params.json"
        metadata_file = self._get_output_prefix() + "_metadata.json"

        if not os.path.exists(samples_file) or not os.path.exists(params_file):
            return super().load_results()

        with open(params_file, "r") as handle:
            self.parameter_names = list(json.load(handle))

        with np.load(samples_file, allow_pickle=False) as archive:
            samples = np.asarray(archive["samples"], dtype=float)
            log_likelihood = np.asarray(archive["log_likelihood"], dtype=float).reshape(
                -1
            )
            self.results = {
                "samples": samples,
                "log_likelihood": log_likelihood,
                "log_weights": equal_weight_log_weights(samples.shape[0]),
                "weighted_samples": {
                    "points": np.asarray(archive["weighted_samples"], dtype=float),
                    "logl": np.asarray(
                        archive["weighted_log_likelihood"], dtype=float
                    ).reshape(-1),
                    "log_weights": np.asarray(
                        archive["weighted_log_weights"], dtype=float
                    ).reshape(-1),
                },
                "output_file": samples_file,
                "metadata_file": metadata_file,
            }

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)
            for key in ("niter", "ncall", "eff", "n_effective", "logz", "logzerr"):
                if key in metadata:
                    self.results[key] = metadata[key]

        return self.results

    def pretty_print_results(self):
        if self.raw_results is not None and hasattr(self.raw_results, "summary"):
            print(self.raw_results.summary())
            return

        if not self.results:
            return

        print(
            f"Dynesty completed: {self.results['samples'].shape[0]} posterior samples"
        )
        if self.results.get("logz") is not None:
            print(
                f"  ln Z: {self.results['logz']:.3f} +- {self.results.get('logzerr', 0.0):.3f}"
            )

    def plot_diagnostics(self, results=None):
        if results is None:
            results = self.results

        if results is None or self.raw_results is None:
            return {}

        try:
            _opt.require_dynesty()
            _opt.require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        output_directory = self._get_output_directory()
        labels = list(self.parameter_names or [])
        plot_files = {}

        try:
            runplot_output = _opt.dyplot.runplot(self.raw_results, logplot=True)
            plot_files["runplot"] = self._save_plot_figure(
                runplot_output,
                os.path.join(output_directory, f"{self.retrieval_name}_runplot.pdf"),
            )
        except Exception as error:
            warnings.warn(
                f"Dynesty runplot generation failed: {error}", RuntimeWarning
            )

        try:
            traceplot_output = _opt.dyplot.traceplot(
                self.raw_results,
                labels=labels if labels else None,
                show_titles=True,
            )
            plot_files["traceplot"] = self._save_plot_figure(
                traceplot_output,
                os.path.join(
                    output_directory,
                    f"{self.retrieval_name}_traceplot.pdf",
                ),
            )
        except Exception as error:
            warnings.warn(
                f"Dynesty traceplot generation failed: {error}", RuntimeWarning
            )

        try:
            cornerpoints_output = _opt.dyplot.cornerpoints(
                self.raw_results,
                labels=labels if labels else None,
            )
            plot_files["cornerpoints"] = self._save_plot_figure(
                cornerpoints_output,
                os.path.join(
                    output_directory,
                    f"{self.retrieval_name}_cornerpoints.pdf",
                ),
            )
        except Exception as error:
            warnings.warn(
                f"Dynesty cornerpoints generation failed: {error}",
                RuntimeWarning,
            )

        return plot_files

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if not self.results:
            return summary

        summary["evidence"] = {
            "ln_z": self._summary_scalar(self.results.get("logz")),
            "ln_z_err": self._summary_scalar(self.results.get("logzerr")),
        }
        summary["diagnostics"] = {
            "niter": self._summary_scalar(self.results.get("niter")),
            "ncall": self._summary_scalar(self.results.get("ncall")),
            "efficiency": self._summary_scalar(self.results.get("eff")),
            "n_effective": self._summary_scalar(self.results.get("n_effective")),
        }
        summary["posterior"] = self._posterior_summary_from_matrix(
            self.results.get("samples"),
            list(free_parameter_names or self.parameter_names),
        )
        return summary

