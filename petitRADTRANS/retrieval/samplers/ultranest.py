"""UltraNest backend: reactive nested sampling, optionally batched through JAX.

UltraNest owns its output directory (``out_<retrieval name>_ultranest``) and
writes its chains as text, so this backend reads them back rather than saving
a pRT ``.npz``.

Two likelihood modes are available. The default drives the scalar
``Retrieval.log_likelihood`` callback one point at a time (and can dispatch it
to a chosen JAX device). ``vectorized=True`` instead hands UltraNest a batched
JAX likelihood, which is much faster but requires a fully differentiable,
non-ragged retrieval -- :meth:`UltranestSampler._resolve_vectorized` checks
that and downgrades with a warning rather than failing mid-run.
"""

import dataclasses
import json
import os
import warnings

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_device_likelihood_wrapper,
    resolve_scalar_likelihood,
)


class UltranestSampler(Sampler):
    """Reactive nested sampling with UltraNest."""

    sampler_type = "ultranest"
    # UltraNest's folder name embeds the retrieval name, so the layout comes
    # from _get_output_directory() below rather than a fixed subdirectory.
    output_subdirectory = None

    def __init__(
        self,
        log_likelihood_func,  # Retrieval.log_likelihood
        prior_func,  # Retrieval.prior_model_jax (generator)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_ultranest()
        self.results = None

    def _get_output_directory(self):
        return self._get_log_dir(self.output_directory, self.retrieval_name)

    @staticmethod
    def _get_log_dir(output_directory, retrieval_name):
        """UltraNest's own run folder, which it creates and populates itself."""
        return os.path.join(output_directory, "out_" + retrieval_name + "_ultranest")

    @classmethod
    def _resolve_vectorized(cls, context: SamplerContext, requested: bool) -> bool:
        if not requested:
            return False
        runtime = context.runtime
        if any(
            group.model_contract != "differentiable" for group in runtime.model_groups
        ):
            warnings.warn(
                "UltraNest vectorized mode disabled because the retrieval contains legacy model groups.",
                RuntimeWarning,
            )
            return False
        if any(
            getattr(obs.spectrum, "dtype", None) == "O"
            for obs in runtime.observations.values()
        ):
            warnings.warn(
                "UltraNest vectorized mode disabled because the retrieval contains ragged observation groups.",
                RuntimeWarning,
            )
            return False
        return True

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        likelihood_devices = sampler_kwargs.pop("likelihood_devices", None)
        use_jit = sampler_kwargs.pop("use_jit", True)
        vectorized = cls._resolve_vectorized(
            context, sampler_kwargs.pop("vectorized", False)
        )
        # The vectorized path always runs through JAX.
        jit_enabled = vectorized

        if likelihood_devices is not None and vectorized:
            warnings.warn(
                "likelihood_devices is ignored in UltraNest vectorized mode "
                "(the vectorized path already dispatches through JAX).",
                RuntimeWarning,
            )
            likelihood_devices = None

        if not vectorized:
            if likelihood_devices is not None:
                device_context = dataclasses.replace(context, likelihood_devices=likelihood_devices)
                log_likelihood_func, target_device = build_device_likelihood_wrapper(
                    device_context
                )
                jit_enabled = True
            else:
                log_likelihood_func, jit_enabled = resolve_scalar_likelihood(
                    context, use_jit, "UltraNest"
                )
                target_device = None
            prior_func = context.prior_ultranest_func
        else:
            parameter_layout = context.parameter_layout
            runtime = context.runtime

            def prior_transform(cube):
                return parameter_layout.prior_transform_numpy(cube)

            def log_likelihood_batch(physical_positions):
                return np.asarray(
                    runtime.evaluate_likelihood_jax_batched(
                        free_parameter_values=physical_positions,
                        uncertainties_mode=context.uncertainties_mode,
                        print_log_likelihood_for_debugging=False,
                    )
                )

            log_likelihood_func = log_likelihood_batch
            prior_func = prior_transform

        sampler = cls(
            log_likelihood_func=log_likelihood_func,
            prior_func=prior_func,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )

        run_kwargs = {
            "parameter_names": sampler_kwargs.pop(
                "parameter_names", context.free_parameter_names
            ),
            "vectorized": vectorized,
        }
        run_kwargs.update(sampler_kwargs)

        summary_parameters = {
            "sampler_type": "ultranest",
            "parameter_names": run_kwargs["parameter_names"],
            "vectorized": vectorized,
            "use_jit": jit_enabled,
        }
        if run_kwargs.get("enable_step_sampling"):
            summary_parameters["enable_step_sampling"] = True
            summary_parameters["nsteps"] = run_kwargs.get("nsteps", 100)
        if likelihood_devices is not None:
            summary_parameters["likelihood_devices"] = [str(d) for d in likelihood_devices]
            summary_parameters["likelihood_target_device"] = str(target_device)
        return sampler, run_kwargs, summary_parameters

    def load_results(self):
        log_dir = self._get_output_directory()
        results_file = os.path.join(log_dir, "info", "results.json")
        equal_weighted_file = os.path.join(
            log_dir, "chains", "equal_weighted_post.txt"
        )
        weighted_file = os.path.join(log_dir, "chains", "weighted_post.txt")

        if not any(
            os.path.exists(path)
            for path in (results_file, equal_weighted_file, weighted_file)
        ):
            return super().load_results()

        results = {}

        if os.path.exists(results_file):
            with open(results_file, "r") as handle:
                results = json.load(handle)

        if os.path.exists(equal_weighted_file):
            results["samples"] = np.loadtxt(
                equal_weighted_file,
                dtype=float,
                ndmin=2,
                skiprows=1,
            )

        if os.path.exists(weighted_file):
            weighted_rows = np.loadtxt(
                weighted_file,
                dtype=float,
                ndmin=2,
                skiprows=1,
            )

            if weighted_rows.shape[1] >= 3:
                weights = np.asarray(weighted_rows[:, 0], dtype=float)
                weighted_log_likelihood = np.asarray(weighted_rows[:, 1], dtype=float)
                weighted_points = np.asarray(weighted_rows[:, 2:], dtype=float)
                results["weighted_samples"] = {
                    "points": weighted_points,
                    "logl": weighted_log_likelihood,
                    "log_weights": np.log(
                        np.clip(weights, np.finfo(float).tiny, None)
                    ),
                }

                if "samples" not in results:
                    results["samples"] = weighted_points
                    results["log_likelihood"] = weighted_log_likelihood
                else:
                    aligned = _equal_weighted_log_likelihood(
                        results["samples"], weighted_points, weighted_log_likelihood
                    )
                    if aligned is not None:
                        results["log_likelihood"] = aligned

        self.results = results if results else None
        return self.results

    def run_sampler(self, parameter_names, **kwargs):
        log_dir = self._get_output_directory()
        os.makedirs(log_dir, exist_ok=True)

        reactive_kwargs = {
            "derived_param_names": kwargs.pop("derived_param_names", []),
            "wrapped_params": kwargs.pop("wrapped_params", None),
            "resume": kwargs.pop("resume", "subfolder"),
            "run_num": kwargs.pop("run_num", None),
            "log_dir": kwargs.pop("log_dir", log_dir),
            "num_test_samples": kwargs.pop("num_test_samples", 2),
            "draw_multiple": kwargs.pop("draw_multiple", True),
            "num_bootstraps": kwargs.pop("num_bootstraps", 30),
            "vectorized": kwargs.pop("vectorized", False),
            "ndraw_min": kwargs.pop("ndraw_min", 128),
            "ndraw_max": kwargs.pop("ndraw_max", 65536),
            "storage_backend": kwargs.pop("storage_backend", "hdf5"),
            "warmstart_max_tau": kwargs.pop("warmstart_max_tau", -1),
        }

        # UltraNest's region sampling (MLFriends) degrades quickly beyond ~10
        # dimensions; a slice step sampler is its recommended replacement there.
        # It attaches to an already-constructed ReactiveNestedSampler.
        step_sampler_kwargs = None
        if kwargs.pop("enable_step_sampling", False):
            step_sampler_kwargs = {
                "nsteps": kwargs.pop("nsteps", 100),
                "generate_direction": kwargs.pop(
                    "generate_direction",
                    _opt.ultranest_generate_mixture_random_direction,
                ),
                "scale": kwargs.pop("scale", 1.0),
                "check_nsteps": kwargs.pop("check_nsteps", "move-distance"),
                "adaptive_nsteps": kwargs.pop("adaptive_nsteps", False),
                "max_nsteps": kwargs.pop("max_nsteps", 1000),
                "region_filter": kwargs.pop("region_filter", False),
                "log": kwargs.pop("log", False),
            }

        self.sampler = _opt.ultranest.ReactiveNestedSampler(
            param_names=parameter_names,
            loglike=self.log_likelihood_func,
            transform=self.prior_func,
            **reactive_kwargs,
        )
        if step_sampler_kwargs is not None:
            self.sampler.stepsampler = _opt.UltranestSliceSampler(**step_sampler_kwargs)
        run_kwargs = {
            "update_interval_volume_fraction": kwargs.pop(
                "update_interval_volume_fraction", 0.8
            ),
            "update_interval_ncall": kwargs.pop("update_interval_ncall", None),
            "log_interval": kwargs.pop("log_interval", None),
            "show_status": kwargs.pop("show_status", True),
            "viz_callback": kwargs.pop("viz_callback", "auto"),
            "dlogz": kwargs.pop("dlogz", 0.5),
            "dKL": kwargs.pop("dKL", 0.5),
            "frac_remain": kwargs.pop("frac_remain", 0.01),
            "Lepsilon": kwargs.pop("Lepsilon", 0.001),
            "min_ess": kwargs.pop("min_ess", 400),
            "max_iters": kwargs.pop("max_iters", None),
            "max_ncalls": kwargs.pop("max_ncalls", None),
            "max_num_improvement_loops": kwargs.pop("max_num_improvement_loops", -1),
            "min_num_live_points": kwargs.pop("min_num_live_points", 400),
            "cluster_num_live_points": kwargs.pop("cluster_num_live_points", 40),
            "insertion_test_zscore_threshold": kwargs.pop(
                "insertion_test_zscore_threshold", 4
            ),
            "insertion_test_window": kwargs.pop("insertion_test_window", 10),
            "region_class": kwargs.pop("region_class", _opt.MLFriends),
            "widen_before_initial_plateau_num_warn": kwargs.pop(
                "widen_before_initial_plateau_num_warn", 10000
            ),
            "widen_before_initial_plateau_num_max": kwargs.pop(
                "widen_before_initial_plateau_num_max", 50000
            ),
        }
        # run_kwargs['max_iters'] = self.max_iters if self.max_iters > 0 else None

        self.sampler.run(**run_kwargs)
        self.results = self.sampler.results
        if self.results:
            self.sampler.print_results()
        return self.results

    def get_results(self):
        return self.results

    def pretty_print_results(self):
        self.sampler.print_results()

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if not isinstance(self.results, dict):
            return summary

        summary["evidence"] = {
            "ln_z": self._summary_scalar(self.results.get("logz")),
            "ln_z_err": self._summary_scalar(self.results.get("logzerr")),
        }
        summary["diagnostics"] = {
            "ess": self._summary_scalar(self.results.get("ess")),
            "ncall": self._summary_scalar(self.results.get("ncall")),
            "samples": (
                self._summary_scalar(self.results.get("samples").shape[0])
                if self.results.get("samples", None) is not None
                else None
            ),
        }
        summary["posterior"] = self._posterior_summary_from_matrix(
            self.results.get("samples"),
            free_parameter_names,
        )
        if not summary["posterior"] and isinstance(
            self.results.get("weighted_samples"), dict
        ):
            summary["posterior"] = self._posterior_summary_from_matrix(
                self.results["weighted_samples"].get("points"),
                free_parameter_names,
            )
        return summary

    @classmethod
    def load_samples_archive(cls, output_directory, retrieval_name):
        """Read UltraNest's weighted chain, whose points and log-likelihoods align.

        ``chains/equal_weighted_post.txt`` holds only parameter columns -- it
        carries no per-sample log likelihood -- so the weighted chain is the
        only UltraNest output from which a ``(samples, log_likelihood)`` pair
        can be read consistently. ``Retrieval`` tries the generic equal-weight
        ``out_samples`` archive before calling this, so this path is the
        fallback for a run whose generic archive is missing.
        """
        weighted_file = os.path.join(
            cls._get_log_dir(output_directory, retrieval_name),
            "chains",
            "weighted_post.txt",
        )

        if not os.path.exists(weighted_file):
            return None

        with open(weighted_file, "r") as handle:
            header = handle.readline().split()

        weighted_rows = np.loadtxt(weighted_file, dtype=float, ndmin=2, skiprows=1)

        # Columns are: weight, log-likelihood, then one per free parameter.
        if weighted_rows.shape[1] < 3:
            return None

        parameter_names = header[2:] if len(header) == weighted_rows.shape[1] else []
        return weighted_rows[:, 2:], weighted_rows[:, 1], parameter_names


def _equal_weighted_log_likelihood(equal_weighted, weighted_points, weighted_log_likelihood):
    """Log likelihoods of UltraNest's equal-weighted samples, row by row.

    ``equal_weighted_post.txt`` resamples the weighted chain (same number of rows,
    with repeats) and stores no likelihoods; its rows are exact copies of
    ``weighted_post.txt`` rows, so each is looked up there. Pairing the samples
    with the weighted chain's ``logl`` column instead mismatches every row.
    Returns None if any row is missing, so the caller can recompute them.
    """
    equal_weighted = np.ascontiguousarray(np.asarray(equal_weighted, dtype=float))
    weighted_points = np.ascontiguousarray(np.asarray(weighted_points, dtype=float))
    if equal_weighted.ndim != 2 or equal_weighted.shape[1] != weighted_points.shape[1]:
        return None
    lookup = {
        row.tobytes(): value for row, value in zip(weighted_points, weighted_log_likelihood)
    }
    values = [lookup.get(row.tobytes()) for row in equal_weighted]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)
