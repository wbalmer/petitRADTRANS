"""The sampler abstraction shared by every petitRADTRANS retrieval backend.

A sampler is the piece that turns a configured :class:`~petitRADTRANS.retrieval.retrieval.Retrieval`
into posterior samples. ``Retrieval`` never talks to a backend library
directly; it builds a :class:`SamplerContext` and drives the sampler through
the small lifecycle below.

Workflow
---------

``Retrieval.run()`` performs, in order:

1. ``sampler_class.prepare(context, **sampler_kwargs)`` ->
   ``(sampler, run_kwargs, summary_parameters)``.
2. ``sampler.run_sampler(**run_kwargs)`` -> backend results.
3. ``sampler.post_run(results)`` -> ``(results, extra_finalize_kwargs)``.
4. ``Retrieval`` writes the generic ``out_samples/<name>_samples.npz`` archive
   and the retrieval summary (via ``sampler.get_summary()``).

In ``run_mode="evaluate"`` steps 1-3 are replaced by ``sampler.load_results()``.

Results
-------

Results objects are duck-typed rather than a single class, because the
backends' natural outputs differ (a PyMultiNest ``Analyzer``, a JAXNS/Dynesty
mapping, an MCMC dataclass). To participate in pRT plotting and archiving, a
results object -- mapping or attribute holder -- should expose:

``samples``
    Equal-weight posterior matrix, shape ``(n_samples, n_parameters)``.
``log_likelihood``
    One value per posterior sample.
``log_weights`` (optional)
    Log weights; ``-log(n_samples)`` for an equal-weight posterior.
``weighted_samples`` (optional)
    ``{"points": ..., "logl": ..., "log_weights": ...}`` for backends whose
    native posterior is weighted. Convert to equal weight for ``samples``.

Class-level declarations
------------------------

Concrete samplers declare their own identity and on-disk layout so that
``Retrieval`` never has to match on ``sampler_type`` strings:

``sampler_type`` / ``aliases``
    Canonical name and the accepted spellings of it.
``output_subdirectory``
    Name of the backend's own output folder under the retrieval output
    directory (e.g. ``"out_JAXNS"``), or ``None`` for a sampler that writes no
    native output.
``uses_distribution_priors``
    Whether free parameters must carry a JAX ``distribution`` (true for the
    gradient-based and JAXNS backends).
``archive_log_likelihood_keys``
    Preference order for the log-likelihood array inside the backend's
    ``_samples.npz``, used by :meth:`Sampler.load_samples_archive`.
"""

import abc
import json
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt

if TYPE_CHECKING:
    from petitRADTRANS.retrieval.runtime import ParameterLayout, RetrievalRuntime


# Quantiles of the standard normal at -1 sigma and +1 sigma, used for the
# posterior summaries written into the retrieval summary file.
_LOWER_1SIGMA_QUANTILE = 0.15865525393145707
_UPPER_1SIGMA_QUANTILE = 0.8413447460685429


@dataclass
class SamplerContext:
    """Everything a sampler needs from a ``Retrieval``, and nothing more.

    Built once by ``Retrieval._build_sampler_context()`` and passed to each
    sampler's :meth:`Sampler.prepare`, so sampler implementations never depend
    on the full ``Retrieval`` object.
    """

    parameter_layout: "ParameterLayout"
    runtime: "RetrievalRuntime"
    configuration: Any
    output_directory: str
    retrieval_name: str
    uncertainties_mode: str = "default"
    print_log_likelihood_for_debugging: bool = False
    # Legacy scalar callbacks (pymultinest / ultranest non-vectorized)
    log_likelihood_func: Any = None
    prior_func: Any = None
    prior_ultranest_func: Any = None
    free_parameter_names: list = field(default_factory=list)
    n_free_parameters: int = 0
    likelihood_devices: list | None = None


class Sampler(abc.ABC):
    """Abstract base class for samplers. See the module docstring for the contract."""

    #: Canonical ``sampler_type`` name for this backend.
    sampler_type: ClassVar[str] = ""
    #: Additional accepted spellings of ``sampler_type``.
    aliases: ClassVar[tuple[str, ...]] = ()
    #: Backend-owned output folder under the retrieval output directory.
    output_subdirectory: ClassVar[str | None] = None
    #: Whether free parameters must carry a JAX distribution.
    uses_distribution_priors: ClassVar[bool] = False
    #: Preference order for the log-likelihood array in ``<prefix>_samples.npz``.
    archive_log_likelihood_keys: ClassVar[tuple[str, ...]] = ("log_likelihood",)
    #: Preference order for the sample matrix in ``<prefix>_samples.npz``.
    archive_sample_keys: ClassVar[tuple[str, ...]] = ("samples", "sample_matrix")

    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name
        self.results = None

        output_subdirectory = self._get_output_directory()
        if output_subdirectory is not None:
            os.makedirs(output_subdirectory, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Workflow
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def run_sampler(self, **kwargs):
        pass

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        """Build a sampler instance, its run kwargs, and its summary parameters.

        Every concrete sampler overrides this to assemble its own run kwargs
        from the common ``SamplerContext`` instead of relying on ``Retrieval``.

        Returns:
            ``(sampler_instance, run_kwargs, summary_parameters)``
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement prepare(). "
            "Pass a pre-built sampler_object to run() instead."
        )

    def post_run(self, results):
        """Optional hook called after ``run_sampler``.

        Handles pretty-printing, results extraction, and any other
        sampler-specific finalization.

        Returns:
            ``(final_results, extra_finalize_kwargs)`` where the extra kwargs
            are forwarded to ``Retrieval._finalize_sampler_run``.
        """
        if callable(getattr(self, "pretty_print_results", None)):
            self.pretty_print_results()
        return results, {}

    def get_results(self):
        return self.results

    def plot_diagnostics(self, results=None):
        """Optional hook to persist sampler-specific diagnostics plots."""
        return {}

    def pretty_print_results(self):
        """Optional hook to print a human-readable run summary."""
        return None

    # ------------------------------------------------------------------ #
    # Output paths
    # ------------------------------------------------------------------ #
    def _get_output_directory(self):
        """Absolute path of this backend's own output folder, or ``None``."""
        if self.output_subdirectory is None:
            return None
        return os.path.join(self.output_directory, self.output_subdirectory)

    def _get_output_prefix(self):
        """``<output directory>/<subdirectory>/<retrieval name>``."""
        return os.path.join(self._get_output_directory(), self.retrieval_name)

    # ------------------------------------------------------------------ #
    # Loading saved results
    # ------------------------------------------------------------------ #
    def _get_samples_archive_path(self):
        """Path of the generic, backend-independent archive written by ``Retrieval``."""
        return os.path.join(
            self.output_directory,
            "out_samples",
            f"{self.retrieval_name}_samples.npz",
        )

    def _load_sample_archive_results(self):
        archive_path = self._get_samples_archive_path()

        if not os.path.exists(archive_path):
            return None, []

        with np.load(archive_path, allow_pickle=False) as archive:
            if "sample_matrix" not in archive.files or "log_likelihood" not in archive.files:
                return None, []

            samples = np.asarray(archive["sample_matrix"], dtype=float)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            elif samples.ndim != 2:
                return None, []

            log_likelihood = np.asarray(archive["log_likelihood"], dtype=float).reshape(-1)
            if log_likelihood.shape[0] != samples.shape[0]:
                return None, []

            parameter_names = []
            if "parameter_names" in archive.files:
                parameter_names = [
                    str(parameter_name)
                    for parameter_name in archive["parameter_names"].tolist()
                ]

        return (
            {
                "samples": samples,
                "log_likelihood": log_likelihood,
                "log_weights": equal_weight_log_weights(samples.shape[0]),
                "output_file": archive_path,
            },
            parameter_names,
        )

    def load_results(self):
        """Reload this sampler's saved results, for ``run_mode="evaluate"``.

        Prefers the generic equal-weight ``out_samples`` archive that
        ``Retrieval`` writes for every backend, and falls back to the
        backend's own archive via :meth:`load_samples_archive` when that run
        predates it or was never finalized.
        """
        results, _parameter_names = self._load_sample_archive_results()

        if results is None:
            results = self._load_results_from_own_archive()

        if results is not None:
            self.results = results

        return results

    def _load_results_from_own_archive(self):
        archive = self.load_samples_archive(self.output_directory, self.retrieval_name)

        if archive is None:
            return None

        samples, log_likelihood, parameter_names = archive
        if parameter_names:
            self.parameter_names = list(parameter_names)

        return {
            "samples": samples,
            "log_likelihood": log_likelihood,
            "log_weights": equal_weight_log_weights(samples.shape[0]),
            "output_file": f"{self._get_output_prefix()}_samples.npz",
        }

    @classmethod
    def load_samples_archive(cls, output_directory, retrieval_name):
        """Read this backend's own posterior archive, without instantiating it.

        Used by ``Retrieval.get_samples()`` to read the outputs of *other*
        retrievals (e.g. for a multi-retrieval corner plot), so it must work
        from paths alone.

        Returns:
            ``(sample_matrix, log_likelihood, parameter_names)`` with
            ``sample_matrix`` of shape ``(n_samples, n_parameters)``, or
            ``None`` when this backend has no native archive to read.
        """
        if cls.output_subdirectory is None:
            return None

        prefix = os.path.join(
            output_directory, cls.output_subdirectory, retrieval_name
        )
        samples_file = f"{prefix}_samples.npz"
        params_file = f"{prefix}_params.json"

        if not os.path.exists(samples_file):
            return None

        parameter_names = []
        if os.path.exists(params_file):
            with open(params_file, "r") as handle:
                parameter_names = [str(name) for name in json.load(handle)]

        with np.load(samples_file, allow_pickle=False) as archive:
            if not parameter_names and "parameter_names" in archive.files:
                parameter_names = [
                    str(name) for name in archive["parameter_names"].tolist()
                ]

            sample_matrix = _first_available_array(archive, cls.archive_sample_keys)
            log_likelihood = _first_available_array(
                archive, cls.archive_log_likelihood_keys
            )

        if sample_matrix is None or log_likelihood is None:
            raise FileNotFoundError(
                f"Sample file '{samples_file}' does not contain both posterior samples "
                f"(one of {cls.archive_sample_keys}) and log-likelihood values "
                f"(one of {cls.archive_log_likelihood_keys})."
            )

        if sample_matrix.ndim == 1:
            sample_matrix = sample_matrix.reshape(-1, 1)

        return sample_matrix, log_likelihood.reshape(-1), parameter_names

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _to_serializable(cls, value):
        """Recursively convert NumPy scalars/arrays to JSON-serializable values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: cls._to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_serializable(val) for val in value]

        return value

    @staticmethod
    def _summary_scalar(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_figure(plot_output):
        """Find a matplotlib figure inside whatever a plotting helper returned."""
        if plot_output is None:
            return None

        if hasattr(plot_output, "savefig"):
            return plot_output

        if hasattr(plot_output, "figure"):
            return plot_output.figure

        if isinstance(plot_output, np.ndarray):
            for entry in plot_output.reshape(-1):
                figure = Sampler._extract_figure(entry)
                if figure is not None:
                    return figure

        if isinstance(plot_output, (list, tuple)):
            for entry in plot_output:
                figure = Sampler._extract_figure(entry)
                if figure is not None:
                    return figure

        return None

    @classmethod
    def _save_plot_figure(cls, plot_output, output_file):
        _opt.require_matplotlib_pyplot()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        figure = cls._extract_figure(plot_output)

        if figure is None:
            figure = _opt.plt.gcf()

        figure.savefig(output_file, bbox_inches="tight")
        _opt.plt.close(figure)
        return output_file

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def get_summary(self, free_parameter_names=None):
        """Structured run summary consumed by ``Retrieval.generate_retrieval_summary``."""
        return {
            "sampler": self.__class__.__name__,
            "parameters": {},
            "diagnostics": {},
            "evidence": {},
            "posterior": [],
        }

    @staticmethod
    def _posterior_summary_from_matrix(samples, parameter_names):
        if samples is None or parameter_names is None:
            return []

        samples = np.asarray(samples)
        if samples.ndim != 2 or samples.shape[1] != len(parameter_names):
            return []

        return [
            _posterior_entry(parameter_name, samples[:, index])
            for index, parameter_name in enumerate(parameter_names)
        ]

    @staticmethod
    def _posterior_summary_from_mapping(
        samples_by_parameter, free_parameter_names=None
    ):
        if not isinstance(samples_by_parameter, dict):
            return []

        parameter_names = free_parameter_names or list(samples_by_parameter.keys())

        return [
            _posterior_entry(
                parameter_name, np.asarray(samples_by_parameter[parameter_name])
            )
            for parameter_name in parameter_names
            if parameter_name in samples_by_parameter
        ]


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def equal_weight_log_weights(sample_count):
    """Log weights of an equal-weight posterior of ``sample_count`` draws."""
    if sample_count <= 0:
        return np.array([], dtype=float)
    return np.full(int(sample_count), -np.log(sample_count), dtype=float)


def _posterior_entry(parameter_name, parameter_samples):
    parameter_samples = np.asarray(parameter_samples).reshape(-1)
    return {
        "name": parameter_name,
        "median": float(np.quantile(parameter_samples, 0.5)),
        "lower_1sigma": float(np.quantile(parameter_samples, _LOWER_1SIGMA_QUANTILE)),
        "upper_1sigma": float(np.quantile(parameter_samples, _UPPER_1SIGMA_QUANTILE)),
    }


def _first_available_array(archive, keys):
    for key in keys:
        if key in archive.files:
            return np.asarray(archive[key], dtype=float)
    return None
