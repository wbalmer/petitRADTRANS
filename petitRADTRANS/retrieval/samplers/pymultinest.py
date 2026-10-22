"""PyMultiNest backend: MPI-parallel importance nested sampling.

MultiNest owns its own output format under ``out_PMN/``; results are exposed
through a ``pymultinest.Analyzer`` rather than a pRT results object, which is
why this sampler returns the analyzer from ``run_sampler`` and why
``Retrieval`` keeps a reference to it for the evidence and marginal statistics.
"""

import dataclasses
import json
import os
import warnings

import numpy as np

from petitRADTRANS.retrieval._mpi import get_mpi_context
from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_device_likelihood_wrapper,
    resolve_scalar_likelihood,
)
from petitRADTRANS.retrieval.utils import get_pymultinest_sample_dict


class PymultinestSampler(Sampler):
    """Nested sampling with MultiNest, via PyMultiNest."""

    sampler_type = "pymultinest"
    output_subdirectory = "out_PMN"

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
        _opt.require_pymultinest()
        self.seed = -1
        self.analyzer = None
        self._n_dims = None
        self.outputfiles_basename = os.path.join(
            self.output_directory, "out_PMN", f"{self.retrieval_name}_"
        )

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        likelihood_devices = sampler_kwargs.pop("likelihood_devices", None)
        use_jit = sampler_kwargs.pop("use_jit", True)

        if likelihood_devices is not None:
            device_context = dataclasses.replace(context, likelihood_devices=likelihood_devices)
            log_likelihood_func, target_device = build_device_likelihood_wrapper(device_context)
            jit_enabled = True
        else:
            log_likelihood_func, jit_enabled = resolve_scalar_likelihood(
                context, use_jit, "PyMultiNest"
            )
            target_device = None

        prefix = os.path.join(
            context.output_directory,
            "out_PMN",
            context.retrieval_name + "_",
        )
        _, _, rank = get_mpi_context()

        if len(prefix) > 100 - 21 and rank == 0:
            warnings.warn(
                "The path to the output files is very long. "
                "This can cause problems for MultiNest...",
                RuntimeWarning,
            )
        if (
            getattr(context.configuration, "run_mode", None) == "retrieval"
            and rank == 0
        ):
            print(f"Starting PyMultiNest retrieval: {context.retrieval_name}")
            print(
                f"Output directory: {os.path.join(context.output_directory, 'out_PMN')}"
            )
            os.makedirs(
                os.path.join(context.output_directory, "out_PMN"), exist_ok=True
            )
            with open(prefix + "params.json", "w") as f:
                json.dump(context.free_parameter_names, f, indent=4)

        run_kwargs = {
            "n_dims": context.n_free_parameters,
            "outputfiles_basename": sampler_kwargs.pop("outputfiles_basename", prefix),
            "sampling_efficiency": sampler_kwargs.pop("sampling_efficiency", 0.8),
            "const_efficiency_mode": sampler_kwargs.pop("const_efficiency_mode", False),
            "n_live_points": sampler_kwargs.pop("n_live_points", 400),
            "evidence_tolerance": sampler_kwargs.pop(
                "evidence_tolerance",
                sampler_kwargs.pop("log_z_convergence", 0.5),
            ),
            "n_iter_before_update": sampler_kwargs.pop("n_iter_before_update", 50),
            "resume": sampler_kwargs.pop("resume", False),
            "verbose": sampler_kwargs.pop("verbose", True),
            "max_iter": sampler_kwargs.pop("max_iter", 0),
            "seed": sampler_kwargs.pop("seed", -1),
            "n_clustering_params": sampler_kwargs.pop("n_clustering_params", None),
            "wrapped_params": sampler_kwargs.pop("wrapped_params", None),
            "importance_nested_sampling": sampler_kwargs.pop(
                "importance_nested_sampling", True
            ),
            "multimodal": sampler_kwargs.pop("multimodal", True),
            "null_log_evidence": sampler_kwargs.pop("null_log_evidence", -1e90),
            "max_modes": sampler_kwargs.pop("max_modes", 100),
            "mode_tolerance": sampler_kwargs.pop("mode_tolerance", -1e90),
            "context": sampler_kwargs.pop("context", 0),
            "write_output": sampler_kwargs.pop("write_output", True),
            "log_zero": sampler_kwargs.pop("log_zero", -1e100),
            "use_MPI": sampler_kwargs.pop("use_MPI", True),
            "init_MPI": sampler_kwargs.pop("init_MPI", False),
            "dump_callback": sampler_kwargs.pop("dump_callback", False),
        }

        sampler = cls(
            log_likelihood_func=log_likelihood_func,
            prior_func=context.prior_func,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        # Store context info needed for post_run
        sampler._free_parameter_names = context.free_parameter_names
        sampler._prefix = prefix

        summary_parameters = dict(run_kwargs)
        summary_parameters["use_jit"] = jit_enabled
        if likelihood_devices is not None:
            summary_parameters["likelihood_devices"] = [str(d) for d in likelihood_devices]
            summary_parameters["likelihood_target_device"] = str(target_device)
        return sampler, run_kwargs, summary_parameters

    def post_run(self, results):
        _, _, rank = get_mpi_context()
        analyzer = results if hasattr(results, "get_stats") else self.analyzer

        if analyzer is None and rank == 0 and self._n_dims is not None:
            try:
                analyzer = _opt.pymultinest.Analyzer(
                    outputfiles_basename=self.outputfiles_basename,
                    n_params=self._n_dims,
                )
            except Exception as error:
                warnings.warn(
                    f"Unable to construct PyMultiNest analyzer for '{self.retrieval_name}': {error}",
                    RuntimeWarning,
                )

        self.analyzer = analyzer

        if rank == 0:
            self.pretty_print(self._free_parameter_names, self._prefix)

        statistics = (
            analyzer.get_stats()
            if analyzer is not None and hasattr(analyzer, "get_stats")
            else None
        )
        extra = {}
        if statistics is not None:
            extra["stats"] = statistics
        return analyzer, extra

    def run_sampler(self, n_dims, **kwargs):
        """
        Args:
            importance_nested_sampling: bool
                If True, Multinest will use Importance Nested Sampling (INS).
                Read http://arxiv.org/abs/1306.2144 for more details on INS.
                Please read the MultiNest README file before using the
                INS in MultiNest v3.0.
            n_params: int
                Total no. of parameters, should be equal to ndims in most cases
                but if you need to store some additional
                parameters with the actual parameters then you need to pass
                them through the likelihood routine.
            sampling_efficiency: float
                defines the sampling efficiency. 0.8 and 0.3 are recommended
                for parameter estimation & evidence evalutation
                respectively.
                use 'parameter' or 'model' to select the respective default
                values
            mode_tolerance: float
                MultiNest can find multiple modes & also specify which samples belong to which mode. It might be
                desirable to have separate samples & mode statistics for modes with local
                log-evidence value greater than a particular value in which case Ztol should
                be set to that value. If there isn't any particularly interesting
                Ztol value, then Ztol should be set to a very large negative number (e.g. -1e90).
            evidence_tolerance: float
                A value of 0.5 should give good enough accuracy.
            n_clustering_params: int
                If mmodal is T, MultiNest will attempt to separate out the
                modes. Mode separation is done through a clustering
                algorithm. Mode separation can be done on all the parameters
                (in which case nCdims should be set to ndims) & it
                can also be done on a subset of parameters (in which case
                nCdims < ndims) which might be advantageous as
                clustering is less accurate as the dimensionality increases.
                If nCdims < ndims then mode separation is done on
                the first nCdims parameters.
            null_log_evidence: float
                If mmodal is T, MultiNest can find multiple modes & also specify
                which samples belong to which mode. It might be
                desirable to have separate samples & mode statistics for modes
                with local log-evidence value greater than a
                particular value in which case nullZ should be set to that
                value. If there isn't any particulrly interesting
                nullZ value, then nullZ should be set to a very large negative
                number (e.g. -1.d90).
            init_MPI: bool
                initialize MPI routines?, relevant only if compiling with MPI
                To run pymultinest with MPI, you need mpi4py installed. Then,
                the libmultinest_mpi library is loaded when you run with mpiexec
                or similar. init_MPI should be set to False, because importing
                mpi4py initialises MPI already.
            log_zero: float
                points with loglike < logZero will be ignored by MultiNest
            max_iter: int
                maximum number of iterations. 0 is unlimited.
            write_output: bool
                write output files? This is required for analysis.
            dump_callback: callable
                a callback function for dumping the current status
            use_MPI: bool
                if True (default), if run with mpiexec and mpi4py installed, use multinest MPI library.
                if False, use only a single processor.
            resume : bool
                Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
            seed : int
                Random number generator seed, -ve value for seed from the system clock (for reproducibility)
        Returns:
            analyzer
        """
        _, _, rank = get_mpi_context()

        outputfiles_basename = kwargs.get(
            "outputfiles_basename", self.outputfiles_basename
        )
        self.outputfiles_basename = outputfiles_basename
        self._n_dims = n_dims

        _opt.pymultinest.run(
            LogLikelihood=self.log_likelihood_func,
            Prior=self.prior_func,
            n_dims=n_dims,
            **kwargs,
        )

        if rank == 0:
            self.analyzer = _opt.pymultinest.Analyzer(
                outputfiles_basename=self.outputfiles_basename, n_params=n_dims
            )
        return self.analyzer

    def load_results(self):
        params_file = self.outputfiles_basename + "params.json"
        sample_file = self.outputfiles_basename + "post_equal_weights.dat"

        if not os.path.exists(params_file) or not os.path.exists(sample_file):
            return super().load_results()

        with open(params_file, "r") as handle:
            parameter_names = list(json.load(handle))

        self._n_dims = len(parameter_names)
        self._free_parameter_names = list(parameter_names)
        self._prefix = self.outputfiles_basename

        try:
            self.analyzer = _opt.pymultinest.Analyzer(
                outputfiles_basename=self.outputfiles_basename,
                n_params=self._n_dims,
            )
        except Exception as error:
            warnings.warn(
                f"Unable to construct PyMultiNest analyzer for '{self.retrieval_name}' "
                f"while loading saved results: {error}",
                RuntimeWarning,
            )
            return super().load_results()

        return self.analyzer

    def pretty_print(self, free_parameter_names, prefix):
        if self.analyzer is None:
            warnings.warn(
                f"PyMultiNest analyzer unavailable for '{self.retrieval_name}'; skipping stats output.",
                RuntimeWarning,
            )
            return

        statistics = self.analyzer.get_stats()
        with open(prefix + "stats.json", "w") as f:
            json.dump(statistics, f, indent=4)
        print("\nPyMultiNest Global Evidence:")  # ... (printing logic)
        for mode in statistics["modes"]:
            print(
                f"  log Z: {mode['local log-evidence']:.2f} +- {mode['local log-evidence error']:.2f}"
            )
        print(
            f"  Global log Z: {statistics['global evidence']:.2f} +- "
            f"{statistics['global evidence error']:.2f}"
        )

        print("\nPyMultiNest Parameter Statistics:")
        for i, name in enumerate(free_parameter_names):
            param_stats = self.analyzer.get_stats()["marginals"][i]
            print(
                f"  {name}: {param_stats['median']:.2e} "
                f"[{param_stats['1sigma'][0]:.2e} - {param_stats['1sigma'][1]:.2e}]"
            )

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.analyzer is None:
            return summary

        statistics = self.analyzer.get_stats()
        summary["evidence"] = {
            "ln_z": self._summary_scalar(statistics.get("global evidence")),
            "ln_z_err": self._summary_scalar(statistics.get("global evidence error")),
        }
        posterior = []
        marginals = statistics.get("marginals", [])
        names = free_parameter_names or []
        for index, marginal in enumerate(marginals):
            name = names[index] if index < len(names) else f"parameter_{index}"
            one_sigma = marginal.get("1sigma", [None, None])
            posterior.append(
                {
                    "name": name,
                    "median": self._summary_scalar(marginal.get("median")),
                    "lower_1sigma": self._summary_scalar(one_sigma[0]),
                    "upper_1sigma": self._summary_scalar(one_sigma[1]),
                }
            )
        summary["posterior"] = posterior
        return summary

    @classmethod
    def load_samples_archive(cls, output_directory, retrieval_name):
        """Read MultiNest's own ``post_equal_weights.dat`` posterior."""
        samples_by_name = get_pymultinest_sample_dict(
            output_dir=output_directory,
            name=retrieval_name,
            add_log_likelihood=True,
            add_stats=False,
        )
        # get_pymultinest_sample_dict appends the log likelihood as the final
        # entry, so the parameter names are everything before it.
        parameter_names = list(samples_by_name.keys())[:-1]
        rows = np.asarray(list(samples_by_name.values()), dtype=float)
        return rows[:-1].T, rows[-1], parameter_names
