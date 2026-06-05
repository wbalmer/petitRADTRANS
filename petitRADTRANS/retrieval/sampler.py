import abc
from dataclasses import dataclass, field, replace as _dc_replace
import gc
import os
import json
from typing import Any, TYPE_CHECKING
import warnings
import numpy as np

from petitRADTRANS.retrieval._mpi import get_mpi_context

if TYPE_CHECKING:
    from petitRADTRANS.retrieval.runtime import ParameterLayout, RetrievalRuntime

lax = None
random = None
tree = None
vmap = None
tree_util = None
jit = None
local_device_count = None
make_mesh = None
shard_map = None
set_mesh = None
devices = None
default_device = None
device_get = None
clear_caches = None
jnp = None
NamedSharding = None
P = None
tfp = None


def _require_jax():
    global lax, random, tree, vmap, tree_util, jit, local_device_count
    global make_mesh, shard_map, set_mesh, devices, default_device, device_get
    global clear_caches, jnp, NamedSharding, P, tfp

    if jnp is not None:
        return

    try:
        from jax import (
            lax as _lax,
            random as _random,
            tree as _tree,
            vmap as _vmap,
            tree_util as _tree_util,
            jit as _jit,
            local_device_count as _ldc,
            make_mesh as _mm,
            shard_map as _sm,
            set_mesh as _set_mesh,
            devices as _devices,
            default_device as _default_device,
            device_get as _device_get,
            clear_caches as _clear_caches,
        )
        import jax.numpy as _jnp
        from jax.sharding import NamedSharding as _ns, PartitionSpec as _ps
        import tensorflow_probability.substrates.jax as _tfp
    except ImportError as error:
        raise ImportError(
            "JAX (and tensorflow-probability) are required for JAX-based samplers but are not installed."
        ) from error

    lax = _lax
    random = _random
    tree = _tree
    vmap = _vmap
    tree_util = _tree_util
    jit = _jit
    local_device_count = _ldc
    make_mesh = _mm
    shard_map = _sm
    set_mesh = _set_mesh
    devices = _devices
    default_device = _default_device
    device_get = _device_get
    clear_caches = _clear_caches
    jnp = _jnp
    NamedSharding = _ns
    P = _ps
    tfp = _tfp


jaxns = None
Model = None
NestedSampler = None
TerminationCondition = None
save_results = None
load_results = None
jaxns_resample = None
ShardedStaticNestedSampler = None
UniformSampler = None

_JAXNS_TERMINATION_FIELDS = (
    "ess",
    "evidence_uncert",
    "live_evidence_frac",
    "dlogZ",
    "max_samples",
    "max_num_likelihood_evaluations",
    "log_L_contour",
    "efficiency_threshold",
    "rtol",
    "atol",
    "peak_XL_frac",
)


def _require_jaxns():
    global jaxns, Model, NestedSampler, TerminationCondition
    global save_results, load_results, jaxns_resample
    global ShardedStaticNestedSampler, UniformSampler

    if jaxns is not None:
        return

    try:
        import jaxns as jaxns_module
        from jaxns import Model as jaxns_model
        from jaxns import NestedSampler as jaxns_nested_sampler
        from jaxns import load_results as jaxns_load_results
        from jaxns import save_results as jaxns_save_results
        from jaxns.nested_samplers.common.types import (
            TerminationCondition as jaxns_termination_condition,
        )
        from jaxns.nested_samplers import (
            ShardedStaticNestedSampler as jaxns_sharded_static_nested_sampler,
        )
        from jaxns.samplers import UniformSampler as jaxns_uniform_sampler
        from jaxns.utils import resample as jaxns_utils_resample
    except Exception as error:
        raise ImportError(
            "JAXNS module or its JAX dependencies are not installed. This sampler cannot be used."
        ) from error

    jaxns = jaxns_module
    Model = jaxns_model
    NestedSampler = jaxns_nested_sampler
    TerminationCondition = jaxns_termination_condition
    save_results = jaxns_save_results
    load_results = jaxns_load_results
    jaxns_resample = jaxns_utils_resample
    ShardedStaticNestedSampler = jaxns_sharded_static_nested_sampler
    UniformSampler = jaxns_uniform_sampler


blackjax = None


def _require_blackjax():
    global blackjax

    if blackjax is not None:
        return

    try:
        import blackjax as _blackjax
    except ImportError as error:
        raise ImportError(
            "BlackJAX is not installed. This sampler cannot be used."
        ) from error

    blackjax = _blackjax


numpyro = None
NumPyroHMC = None
NumPyroNUTS = None
NumPyroMCMC = None
az = None
plt = None


def _require_numpyro():
    global numpyro, NumPyroHMC, NumPyroNUTS, NumPyroMCMC

    if numpyro is not None:
        return

    try:
        import numpyro as _numpyro
        from numpyro.infer import HMC as _hmc
        from numpyro.infer import MCMC as _mcmc
        from numpyro.infer import NUTS as _nuts
    except ImportError as error:
        raise ImportError(
            "NumPyro is not installed. This sampler cannot be used."
        ) from error

    numpyro = _numpyro
    NumPyroHMC = _hmc
    NumPyroNUTS = _nuts
    NumPyroMCMC = _mcmc


def _require_arviz():
    global az

    if az is not None:
        return

    try:
        import arviz as _az
    except ImportError as error:
        raise ImportError(
            "ArviZ is required to plot NumPyro diagnostics but is not installed."
        ) from error

    az = _az


def _require_matplotlib_pyplot():
    global plt

    if plt is not None:
        return

    try:
        import matplotlib.pyplot as _plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to save sampler diagnostics plots but is not installed."
        ) from error

    plt = _plt


pymultinest = None


def _require_pymultinest():
    global pymultinest

    if pymultinest is not None:
        return

    try:
        import pymultinest as _pymultinest
    except ImportError as error:
        raise ImportError(
            "PyMultiNest is not installed. This sampler cannot be used."
        ) from error

    pymultinest = _pymultinest


ultranest = None
MLFriends = None
RobustEllipsoidRegion = None
dynesty = None
DynamicNestedSampler = None
DynestyPool = None
dyplot = None
dyfunc = None


def _require_ultranest():
    global ultranest, MLFriends, RobustEllipsoidRegion

    if ultranest is not None:
        return

    try:
        import ultranest as _ultranest
        from ultranest.mlfriends import MLFriends as _MLFriends
        from ultranest.mlfriends import RobustEllipsoidRegion as _RobustEllipsoidRegion
    except ImportError as error:
        raise ImportError(
            "UltraNest is not installed. This sampler cannot be used."
        ) from error

    ultranest = _ultranest
    MLFriends = _MLFriends
    RobustEllipsoidRegion = _RobustEllipsoidRegion


def _require_dynesty():
    global dynesty, DynamicNestedSampler, DynestyPool, dyplot, dyfunc

    if dynesty is not None:
        return

    try:
        import dynesty as _dynesty
        from dynesty import DynamicNestedSampler as _DynamicNestedSampler
        from dynesty.pool import Pool as _DynestyPool
        from dynesty import plotting as _dyplot
        from dynesty import utils as _dyfunc
    except ImportError as error:
        raise ImportError(
            "Dynesty is not installed. This sampler cannot be used."
        ) from error

    dynesty = _dynesty
    DynamicNestedSampler = _DynamicNestedSampler
    DynestyPool = _DynestyPool
    dyplot = _dyplot
    dyfunc = _dyfunc


# ---------------------------------------------------------------------------
# Sampler context — the common interface between Retrieval and all samplers
# ---------------------------------------------------------------------------


@dataclass
class SamplerContext:
    """Lightweight container that carries everything a sampler needs from a Retrieval.

    Built once by ``Retrieval._build_sampler_context()`` and passed to each
    sampler's ``prepare()`` class method so that sampler implementations never
    depend on the full ``Retrieval`` object.
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


def _check_differentiable_support(context: SamplerContext, sampler_label: str) -> None:
    """Raise if the retrieval has only legacy (non-differentiable) model groups."""
    runtime = context.runtime
    if all(group.model_contract != "differentiable" for group in runtime.model_groups):
        raise ValueError(
            f"{sampler_label} requires at least one differentiable model group; "
            "legacy-only retrievals are not supported."
        )


def _build_device_likelihood_wrapper(context: SamplerContext):
    """Return a PyMultiNest-compatible log-likelihood that dispatches to a JAX device.

    Each MPI rank r selects context.likelihood_devices[r % len(likelihood_devices)],
    so N ranks stripe naturally across M devices. Requires differentiable model groups.

    Returns ``(device_likelihood_func, target_device)``.

    Notes
    -----
    The user is responsible for ensuring that JAX processes do not unexpectedly contend
    on the same device (e.g. set ``JAX_PLATFORMS=cuda,cpu`` or ``CUDA_VISIBLE_DEVICES``
    per MPI process for true isolation when using multiple GPUs).
    """
    _require_jax()
    _check_differentiable_support(context, "Device-dispatched likelihood (likelihood_devices)")

    likelihood_devices = context.likelihood_devices
    _, _, rank = get_mpi_context()
    target_device = likelihood_devices[rank % len(likelihood_devices)]

    parameter_layout = context.parameter_layout
    runtime = context.runtime
    uncertainties_mode = context.uncertainties_mode
    print_debug = context.print_log_likelihood_for_debugging

    _jit_evaluate = jit(
        lambda physical_params: runtime.evaluate_jax(
            physical_params=physical_params,
            uncertainties_mode=uncertainties_mode,
            print_log_likelihood_for_debugging=print_debug,
        )
    )

    def device_likelihood(cube, ndim=0, nparam=0):
        physical_params = parameter_layout.physical_params_from_vector(cube)
        with default_device(target_device):
            result = _jit_evaluate(physical_params)
        return float(device_get(result))

    return device_likelihood, target_device


def _build_unconstrained_mcmc_interface(context: SamplerContext):
    """Build the unconstrained-space logdensity and transforms for gradient-based samplers.

    Returns ``(free_parameter_names, initial_position, logdensity, transform_positions, log_jacobian)``.
    """
    _require_jax()
    from petitRADTRANS.retrieval.parameter import JAX_DTYPE

    _check_differentiable_support(context, "Gradient-based JAX samplers")
    parameter_layout = context.parameter_layout
    runtime = context.runtime
    free_parameter_names = list(parameter_layout.free_parameter_names)

    if len(free_parameter_names) < 1:
        raise ValueError("Gradient-based samplers require at least one free parameter.")

    free_parameters = [
        context.configuration.parameters[name] for name in free_parameter_names
    ]
    for parameter_name, parameter in zip(free_parameter_names, free_parameters):
        if (
            parameter.distribution is None
            and parameter.transform_prior_cube_coordinate is None
        ):
            raise ValueError(
                f"Free parameter '{parameter_name}' has neither a JAX distribution nor a prior transform."
            )

    def transform_single(unconstrained_position):
        return parameter_layout.transform_unconstrained(unconstrained_position)

    def transform_positions(unconstrained_positions):
        if jnp.ndim(unconstrained_positions) == 1:
            return transform_single(unconstrained_positions)
        return vmap(transform_single)(unconstrained_positions)

    def log_jacobian_single(unconstrained_position):
        return parameter_layout.log_jacobian_single(unconstrained_position)

    def log_jacobian(unconstrained_positions):
        return parameter_layout.log_jacobian(unconstrained_positions)

    def logdensity(unconstrained_position):
        physical_position = transform_single(unconstrained_position)
        physical_params = parameter_layout.physical_params_from_free_values(
            tuple(physical_position)
        )
        return runtime.evaluate_jax(
            physical_params=physical_params,
            uncertainties_mode=context.uncertainties_mode,
            print_log_likelihood_for_debugging=context.print_log_likelihood_for_debugging,
        ) + log_jacobian_single(unconstrained_position)

    initial_position = jnp.zeros(parameter_layout.n_free_parameters, dtype=JAX_DTYPE)

    return (
        free_parameter_names,
        initial_position,
        logdensity,
        transform_positions,
        log_jacobian,
    )


class Sampler(abc.ABC):
    """
    Abstract base class for samplers.
    """

    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name

    @abc.abstractmethod
    def run_sampler(self, **kwargs):
        pass

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        """Build a sampler instance, run-kwargs, and summary parameters.

        Every concrete sampler should override this to assemble its own
        run kwargs from the common ``SamplerContext`` instead of relying on
        the ``Retrieval`` class.

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
        if hasattr(self, "pretty_print_results") and callable(
            self.pretty_print_results
        ):
            self.pretty_print_results()
        return results, {}

    def plot_diagnostics(self, results=None):
        """Optional hook to persist sampler-specific diagnostics plots."""
        return {}

    def _get_samples_archive_path(self):
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

        sample_count = samples.shape[0]
        if sample_count > 0:
            log_weights = np.full(sample_count, -np.log(sample_count), dtype=float)
        else:
            log_weights = np.array([], dtype=float)

        return (
            {
                "samples": samples,
                "log_likelihood": log_likelihood,
                "log_weights": log_weights,
                "output_file": archive_path,
            },
            parameter_names,
        )

    def load_results(self):
        results, _parameter_names = self._load_sample_archive_results()

        if results is not None:
            self.results = results

        return results

    @staticmethod
    def _extract_figure(plot_output):
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
        _require_matplotlib_pyplot()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        figure = cls._extract_figure(plot_output)

        if figure is None:
            figure = plt.gcf()

        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        return output_file

    @staticmethod
    def _summary_scalar(value):
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def get_summary(self, free_parameter_names=None):
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

        posterior = []
        for index, parameter_name in enumerate(parameter_names):
            parameter_samples = np.asarray(samples[:, index]).reshape(-1)
            posterior.append(
                {
                    "name": parameter_name,
                    "median": float(np.quantile(parameter_samples, 0.5)),
                    "lower_1sigma": float(
                        np.quantile(parameter_samples, 0.15865525393145707)
                    ),
                    "upper_1sigma": float(
                        np.quantile(parameter_samples, 0.8413447460685429)
                    ),
                }
            )

        return posterior

    @staticmethod
    def _posterior_summary_from_mapping(
        samples_by_parameter, free_parameter_names=None
    ):
        if not isinstance(samples_by_parameter, dict):
            return []

        parameter_names = free_parameter_names or list(samples_by_parameter.keys())
        posterior = []
        for parameter_name in parameter_names:
            if parameter_name not in samples_by_parameter:
                continue

            parameter_samples = np.asarray(samples_by_parameter[parameter_name])
            if parameter_samples.ndim == 0:
                parameter_samples = parameter_samples.reshape(1)
            else:
                parameter_samples = parameter_samples.reshape(-1)

            posterior.append(
                {
                    "name": parameter_name,
                    "median": float(np.quantile(parameter_samples, 0.5)),
                    "lower_1sigma": float(
                        np.quantile(parameter_samples, 0.15865525393145707)
                    ),
                    "upper_1sigma": float(
                        np.quantile(parameter_samples, 0.8413447460685429)
                    ),
                }
            )

        return posterior


@dataclass
class BlackJAXSamplingResults:
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


@dataclass
class NumPyroSamplingResults:
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


class BlackJAXSampler(Sampler):
    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_jax()
        _require_blackjax()
        os.makedirs(os.path.join(self.output_directory, "out_BlackJAX"), exist_ok=True)
        self.results = None

    @classmethod
    def _prepare_blackjax(
        cls, context: SamplerContext, algorithm_name: str, **sampler_kwargs
    ):
        """Shared preparation logic for all BlackJAX sampler variants."""
        (
            free_parameter_names,
            initial_position,
            logdensity,
            transform_positions,
            log_jacobian,
        ) = _build_unconstrained_mcmc_interface(context)

        # Optional optax warm-up (see NumPyro variant for details): replace the
        # prior-median initial_position with a MAP estimate. Defaults to Adam.
        optimize_init = sampler_kwargs.pop("optimize_init", False)
        warmup_optimizer_kwargs = dict(sampler_kwargs.pop("optimizer_kwargs", {}))
        warmup_best_logdensity = None
        if optimize_init and "initial_position" not in sampler_kwargs:
            from petitRADTRANS.retrieval.optimal_estimation import (
                optimize_unconstrained_position,
                make_walker_inits,
                report_and_save_map_warmup,
            )

            walker_jitter = warmup_optimizer_kwargs.pop("walker_jitter", 0.0)
            requested_num_chains = sampler_kwargs.get("num_chains", 1)
            warmup_seed = warmup_optimizer_kwargs.pop(
                "seed", sampler_kwargs.get("seed", 42654)
            )
            # Optional override of the optimiser's starting point. The default
            # ``initial_position`` is the unconstrained origin, which maps (via
            # cube = sigmoid(z)) to the prior median/midpoint of every parameter
            # -- a stiff, often ill-conditioned point for these models. Callers
            # can pass a different start (e.g. the prior midpoint plus a small
            # offset) through ``optimizer_kwargs["initial_position"]``.
            warmup_initial_position = warmup_optimizer_kwargs.pop(
                "initial_position", None
            )
            if warmup_initial_position is not None:
                initial_position = jnp.asarray(
                    warmup_initial_position, dtype=initial_position.dtype
                )
            optimize_result = optimize_unconstrained_position(
                logdensity, initial_position, seed=warmup_seed, **warmup_optimizer_kwargs
            )
            warmup_best_logdensity = float(optimize_result.best_logdensity)
            # Print the MAP parameter values and save the MAP sample to file.
            report_and_save_map_warmup(
                optimize_result,
                transform_positions,
                free_parameter_names,
                context.output_directory,
                context.retrieval_name,
                output_subdirectory="out_BlackJAX",
                verbose=warmup_optimizer_kwargs.get("verbose", True),
            )
            initial_position = make_walker_inits(
                optimize_result.best_position,
                requested_num_chains,
                jitter=walker_jitter,
                seed=warmup_seed,
            )

        run_kwargs = {
            "initial_position": sampler_kwargs.pop(
                "initial_position", initial_position
            ),
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
            "num_chains": sampler_kwargs.pop("num_chains", 1),
        }
        if algorithm_name == "hmc":
            run_kwargs["num_integration_steps"] = sampler_kwargs.pop(
                "num_integration_steps", 20
            )
        run_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=logdensity,
            prior_func=transform_positions,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )

        summary_parameters = {
            "sampler_type": algorithm_name,
            "num_samples": run_kwargs["num_samples"],
            "num_warmup": run_kwargs["num_warmup"],
            "initial_step_size": run_kwargs["initial_step_size"],
            "target_acceptance_rate": run_kwargs["target_acceptance_rate"],
            "is_mass_matrix_diagonal": run_kwargs["is_mass_matrix_diagonal"],
            "num_chains": run_kwargs["num_chains"],
            "seed": run_kwargs["seed"],
            "use_jit": run_kwargs["use_jit"],
        }
        if "num_integration_steps" in run_kwargs:
            summary_parameters["num_integration_steps"] = run_kwargs[
                "num_integration_steps"
            ]
        if optimize_init:
            summary_parameters["optimize_init"] = True
            summary_parameters["warmup_best_logdensity"] = warmup_best_logdensity

        return sampler, run_kwargs, summary_parameters

    @property
    def algorithm_name(self):
        raise NotImplementedError

    @property
    def algorithm_factory(self):
        raise NotImplementedError

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                key: BlackJAXSampler._to_serializable(val) for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [BlackJAXSampler._to_serializable(val) for val in value]

        return value

    def _get_output_prefix(self):
        return os.path.join(self.output_directory, "out_BlackJAX", self.retrieval_name)

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
        if model_log_likelihood is not None:
            arrays_to_save["model_log_likelihood"] = model_log_likelihood
        if acceptance_rate is not None:
            arrays_to_save["acceptance_rate"] = acceptance_rate
        if is_divergent is not None:
            arrays_to_save["is_divergent"] = is_divergent
        if num_integration_steps is not None:
            arrays_to_save["num_integration_steps"] = num_integration_steps

        np.savez(samples_file, **arrays_to_save)

        with open(params_file, "w") as handle:
            json.dump(parameter_names, handle, indent=4)

        metadata["warmup_parameters"] = self._to_serializable(warmup_parameters)
        with open(metadata_file, "w") as handle:
            json.dump(self._to_serializable(metadata), handle, indent=4)

        return samples_file, metadata_file

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
        **blackjax_kwargs,
    ):
        # --- Single warmup run; adapted parameters are shared across all chains. ---
        # Discard legacy 'jit' alias that older code or users may pass; use_jit
        # is the named parameter that controls JIT compilation in this wrapper.
        blackjax_kwargs.pop("jit", None)
        warmup = blackjax.window_adaptation(
            self.algorithm_factory,
            self.log_likelihood_func,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            initial_step_size=initial_step_size,
            target_acceptance_rate=target_acceptance_rate,
            progress_bar=progress_bar,
            **blackjax_kwargs,
        )

        rng_key = random.PRNGKey(seed)
        rng_key, warmup_key, sample_key = random.split(rng_key, 3)
        (initial_state, warmup_parameters), _ = warmup.run(
            warmup_key,
            initial_position,
            num_steps=num_warmup,
        )

        algorithm = self.algorithm_factory(
            self.log_likelihood_func, **warmup_parameters
        )
        kernel = algorithm.step

        # --- Per-chain sampling loop (captures initial_state and kernel). ---
        def run_one_chain(chain_sample_key):
            def one_step(current_state, rng_k):
                next_state, info = kernel(rng_k, current_state)
                return next_state, (next_state, info)

            chain_keys = random.split(chain_sample_key, num_samples)
            _, (chain_states, chain_infos) = lax.scan(
                one_step, initial_state, chain_keys
            )
            return chain_states, chain_infos

        if num_chains == 1:
            # Original single-chain path.
            if use_jit:
                run_one_chain = jit(run_one_chain)
            states, infos = run_one_chain(sample_key)
        else:
            # Multi-chain path: distribute chains across devices using shard_map + vmap.
            # Each device runs chains_per_device independent chains using the same adapted
            # warmup parameters but different RNG seeds.
            n_devices = local_device_count()
            effective_devices = min(n_devices, num_chains)
            chains_per_device = max(1, num_chains // effective_devices)

            if effective_devices > 1:
                mesh = make_mesh((effective_devices,), ("devices",))

                def run_device_chains(base_key):
                    # Derive a unique sample key per device from the shared base key,
                    # then vmap over chains_per_device independent chains.
                    device_idx = lax.axis_index("devices")
                    device_key = random.fold_in(base_key, device_idx)
                    per_chain_keys = random.split(device_key, chains_per_device)
                    return vmap(run_one_chain)(per_chain_keys)

                run_all = shard_map(
                    run_device_chains,
                    mesh=mesh,
                    in_specs=P(),  # base_key replicated to every device
                    out_specs=P(
                        "devices"
                    ),
                    check_vma=False,
                )
                if use_jit:
                    run_all = jit(run_all)
                with set_mesh(mesh):
                    all_states, all_infos = run_all(sample_key)
            else:
                # Single device: vmap over all chains.
                chain_keys = random.split(sample_key, num_chains)
                run_all_single = vmap(run_one_chain)
                if use_jit:
                    run_all_single = jit(run_all_single)
                all_states, all_infos = run_all_single(chain_keys)

            # Flatten [num_chains, num_samples, ...] → [num_chains * num_samples, ...]
            def _flatten(x):
                return x.reshape(-1, *x.shape[2:]) if x.ndim >= 2 else x

            states = tree_util.tree_map(_flatten, all_states)
            infos = tree_util.tree_map(_flatten, all_infos)

        unconstrained_samples = np.asarray(states.position)
        transformed_samples = (
            states.position
            if self.prior_func is None
            else self.prior_func(states.position)
        )
        samples = np.asarray(transformed_samples)
        logdensity = np.asarray(states.logdensity)
        acceptance_rate = (
            np.asarray(infos.acceptance_rate)
            if hasattr(infos, "acceptance_rate")
            else None
        )
        is_divergent = (
            np.asarray(infos.is_divergent) if hasattr(infos, "is_divergent") else None
        )
        num_integration_steps = (
            np.asarray(infos.num_integration_steps)
            if hasattr(infos, "num_integration_steps")
            else None
        )

        model_log_likelihood = None
        if log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(
                states.logdensity - log_jacobian_fn(states.position)
            )

        warmup_parameters_np = tree.map(np.asarray, warmup_parameters)
        metadata = {
            "algorithm": self.algorithm_name,
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            "num_chains": num_chains,
            "seed": seed,
            "parameter_names": parameter_names,
        }
        samples_file, metadata_file = self._save_results(
            parameter_names=parameter_names,
            samples=samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            warmup_parameters=warmup_parameters_np,
            metadata=metadata,
        )

        self.results = BlackJAXSamplingResults(
            algorithm=self.algorithm_name,
            samples=samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            warmup_parameters=self._to_serializable(warmup_parameters_np),
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            output_file=samples_file,
            metadata_file=metadata_file,
        )

        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        samples_file = self._get_output_prefix() + "_samples.npz"
        metadata_file = self._get_output_prefix() + "_metadata.json"

        if not os.path.exists(samples_file):
            archived_results, _parameter_names = self._load_sample_archive_results()

            if archived_results is None:
                return None

            archived_log_likelihood = np.asarray(
                archived_results["log_likelihood"], dtype=float
            )
            archived_samples = np.asarray(archived_results["samples"], dtype=float)
            self.results = BlackJAXSamplingResults(
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

        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)

        with np.load(samples_file, allow_pickle=False) as archive:
            self.results = BlackJAXSamplingResults(
                algorithm=str(metadata.get("algorithm", self.algorithm_name)),
                samples=np.asarray(archive["samples"], dtype=float),
                unconstrained_samples=np.asarray(
                    archive["unconstrained_samples"], dtype=float
                ),
                logdensity=np.asarray(archive["logdensity"], dtype=float),
                model_log_likelihood=(
                    np.asarray(archive["model_log_likelihood"], dtype=float)
                    if "model_log_likelihood" in archive.files
                    else None
                ),
                warmup_parameters=dict(metadata.get("warmup_parameters", {})),
                acceptance_rate=(
                    np.asarray(archive["acceptance_rate"], dtype=float)
                    if "acceptance_rate" in archive.files
                    else None
                ),
                is_divergent=(
                    np.asarray(archive["is_divergent"])
                    if "is_divergent" in archive.files
                    else None
                ),
                num_integration_steps=(
                    np.asarray(archive["num_integration_steps"], dtype=float)
                    if "num_integration_steps" in archive.files
                    else None
                ),
                output_file=samples_file,
                metadata_file=metadata_file if os.path.exists(metadata_file) else None,
            )

        return self.results

    def pretty_print_results(self):
        if self.results is None:
            return

        print(
            f"BlackJAX {self.algorithm_name.upper()} completed: {self.results.samples.shape[0]} samples"
        )
        if self.results.acceptance_rate is not None:
            print(
                f"  Mean acceptance rate: {np.mean(self.results.acceptance_rate):.3f}"
            )
        if self.results.is_divergent is not None:
            print(f"  Divergences: {int(np.sum(self.results.is_divergent))}")

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        summary["diagnostics"] = {
            "num_samples": (
                int(self.results.samples.shape[0]) if self.results is not None else None
            ),
            "mean_logdensity": (
                float(np.mean(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None
                else None
            ),
            "max_logdensity": (
                float(np.max(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None
                else None
            ),
            "mean_acceptance_rate": (
                float(np.mean(self.results.acceptance_rate))
                if self.results is not None and self.results.acceptance_rate is not None
                else None
            ),
            "n_divergent": (
                int(np.sum(self.results.is_divergent))
                if self.results is not None and self.results.is_divergent is not None
                else None
            ),
            "mean_integration_steps": (
                float(np.mean(self.results.num_integration_steps))
                if self.results is not None
                and self.results.num_integration_steps is not None
                else None
            ),
        }
        if self.results is not None:
            summary["parameters"] = dict(self.results.warmup_parameters)
            summary["posterior"] = self._posterior_summary_from_matrix(
                self.results.samples,
                free_parameter_names,
            )
        return summary


class BlackJAXHMCSampler(BlackJAXSampler):
    @property
    def algorithm_name(self):
        return "hmc"

    @property
    def algorithm_factory(self):
        return blackjax.hmc

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_blackjax(context, algorithm_name="hmc", **sampler_kwargs)


class BlackJAXNUTSSampler(BlackJAXSampler):
    @property
    def algorithm_name(self):
        return "nuts"

    @property
    def algorithm_factory(self):
        return blackjax.nuts

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_blackjax(context, algorithm_name="nuts", **sampler_kwargs)


class NumPyroSampler(Sampler):
    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_jax()
        _require_numpyro()
        os.makedirs(os.path.join(self.output_directory, "out_NumPyro"), exist_ok=True)
        self.results = None
        self.mcmc = None
        self.parameter_names = None

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
        ) = _build_unconstrained_mcmc_interface(context)

        # Optional optax warm-up: replace the prior-median initial_position with a
        # MAP estimate so the gradient sampler starts inside the bulk of the
        # posterior. Defaults to Adam; see optimal_estimation.build_optimizer.
        optimize_init = sampler_kwargs.pop("optimize_init", False)
        warmup_optimizer_kwargs = dict(sampler_kwargs.pop("optimizer_kwargs", {}))
        warmup_best_logdensity = None
        if optimize_init and "initial_position" not in sampler_kwargs:
            from petitRADTRANS.retrieval.optimal_estimation import (
                optimize_unconstrained_position,
                make_walker_inits,
                report_and_save_map_warmup,
            )

            walker_jitter = warmup_optimizer_kwargs.pop("walker_jitter", 0.0)
            requested_num_chains = sampler_kwargs.get("num_chains", 1)
            warmup_seed = warmup_optimizer_kwargs.pop(
                "seed", sampler_kwargs.get("seed", 42654)
            )
            # Optional override of the optimiser's starting point. The default
            # ``initial_position`` is the unconstrained origin, which maps (via
            # cube = sigmoid(z)) to the prior median/midpoint of every parameter
            # -- a stiff, often ill-conditioned point for these models. Callers
            # can pass a different start (e.g. the prior midpoint plus a small
            # offset) through ``optimizer_kwargs["initial_position"]``.
            warmup_initial_position = warmup_optimizer_kwargs.pop(
                "initial_position", None
            )
            if warmup_initial_position is not None:
                initial_position = jnp.asarray(
                    warmup_initial_position, dtype=initial_position.dtype
                )
            optimize_result = optimize_unconstrained_position(
                logdensity, initial_position, seed=warmup_seed, **warmup_optimizer_kwargs
            )
            warmup_best_logdensity = float(optimize_result.best_logdensity)
            # Print the MAP parameter values and save the MAP sample to file.
            report_and_save_map_warmup(
                optimize_result,
                transform_positions,
                free_parameter_names,
                context.output_directory,
                context.retrieval_name,
                output_subdirectory="out_NumPyro",
                verbose=warmup_optimizer_kwargs.get("verbose", True),
            )
            initial_position = make_walker_inits(
                optimize_result.best_position,
                requested_num_chains,
                jitter=walker_jitter,
                seed=warmup_seed,
            )

        run_kwargs = {
            "initial_position": sampler_kwargs.pop(
                "initial_position", initial_position
            ),
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
            "num_chains": sampler_kwargs.pop("num_chains", 1),
            "thinning": sampler_kwargs.pop("thinning", 1),
            "chain_method": sampler_kwargs.pop("chain_method", "vectorized"),
        }
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

        summary_parameters = {
            "sampler_type": algorithm_name,
            "num_samples": run_kwargs["num_samples"],
            "num_warmup": run_kwargs["num_warmup"],
            "initial_step_size": run_kwargs["initial_step_size"],
            "target_acceptance_rate": run_kwargs["target_acceptance_rate"],
            "is_mass_matrix_diagonal": run_kwargs["is_mass_matrix_diagonal"],
            "num_chains": run_kwargs["num_chains"],
            "thinning": run_kwargs["thinning"],
            "chain_method": run_kwargs["chain_method"],
            "seed": run_kwargs["seed"],
            "use_jit": run_kwargs["use_jit"],
        }
        if "num_steps" in run_kwargs:
            summary_parameters["num_integration_steps"] = run_kwargs["num_steps"]
        if "trajectory_length" in run_kwargs:
            summary_parameters["trajectory_length"] = run_kwargs["trajectory_length"]
        if optimize_init:
            summary_parameters["optimize_init"] = True
            summary_parameters["warmup_best_logdensity"] = warmup_best_logdensity

        return sampler, run_kwargs, summary_parameters

    @property
    def algorithm_name(self):
        raise NotImplementedError

    @property
    def kernel_class(self):
        raise NotImplementedError

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                key: NumPyroSampler._to_serializable(val) for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [NumPyroSampler._to_serializable(val) for val in value]

        return value

    def _get_output_prefix(self):
        return os.path.join(self.output_directory, "out_NumPyro", self.retrieval_name)

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
        # Discard legacy 'jit' alias that older code or users may pass; use_jit
        # is the named parameter that controls JIT compilation in this wrapper.
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
        if model_log_likelihood is not None:
            arrays_to_save["model_log_likelihood"] = model_log_likelihood
        if acceptance_rate is not None:
            arrays_to_save["acceptance_rate"] = acceptance_rate
        if is_divergent is not None:
            arrays_to_save["is_divergent"] = is_divergent
        if num_integration_steps is not None:
            arrays_to_save["num_integration_steps"] = num_integration_steps

        np.savez(samples_file, **arrays_to_save)

        with open(params_file, "w") as handle:
            json.dump(parameter_names, handle, indent=4)

        metadata["warmup_parameters"] = self._to_serializable(warmup_parameters)
        with open(metadata_file, "w") as handle:
            json.dump(self._to_serializable(metadata), handle, indent=4)

        return samples_file, metadata_file

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
        rng_key = random.PRNGKey(seed)
        n_devices = local_device_count()
        use_multi_device = (
            n_devices > 1 and num_chains > 1 and chain_method != "sequential"
        )

        if use_multi_device:
            # Multi-device path: distribute chains across devices via shard_map.
            # Each device runs chains_per_device chains with chain_method='vectorized',
            # using a unique per-device RNG key derived from lax.axis_index + fold_in.
            effective_devices = min(n_devices, num_chains)
            chains_per_device = max(1, num_chains // effective_devices)
            _kernel_class = self.kernel_class
            _kernel_kwargs = kernel_kwargs
            _initial_position = initial_position
            mesh = make_mesh((effective_devices,), ("devices",))

            # NOTE: ``init_position`` is passed as an explicit shard_map argument
            # (replicated, in_specs=P()) rather than closed over. On JAX versions
            # that put the mesh axes in "Explicit" sharding mode (set_mesh below),
            # closing over an array inside shard_map raises
            #   NotImplementedError: Closing over inputs to shard_map where the
            #   input is sharded on `Explicit` axes is not implemented.
            # because the captured array inherits the mesh's explicit sharding.
            # Threading it through as an argument lets in_specs govern its
            # sharding. This is the only closed-over JAX array here -- everything
            # in _kernel_kwargs is a Python scalar/bool or the potential_fn
            # closure, and extra_fields is a tuple of strings.
            #
            # FUTURE: once JAX implements closing over Explicit-sharded inputs to
            # shard_map, init_position can simply be captured again and the extra
            # argument/in_specs entry dropped, i.e.:
            #
            #     def run_on_device(base_key):
            #         ...
            #         device_mcmc.run(
            #             device_key, init_params=_initial_position,
            #             extra_fields=extra_fields,
            #         )
            #         ...
            #     run_sharded = shard_map(
            #         run_on_device, mesh=mesh, in_specs=P(),
            #         out_specs=(P("devices"),) * 7,
            #     )
            #     ... = run_sharded(rng_key)
            def run_on_device(base_key, init_position):
                device_idx = lax.axis_index("devices")
                device_key = random.fold_in(base_key, device_idx)
                device_kernel = _kernel_class(**_kernel_kwargs)
                device_mcmc = NumPyroMCMC(
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

            run_sharded = shard_map(
                run_on_device,
                mesh=mesh,
                # base_key and init_position both replicated to every device.
                in_specs=(P(), P()),
                out_specs=(P("devices"),)
                * 7,  # all 7 outputs concatenated along axis 0
            )
            with set_mesh(mesh):
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
        else:
            # Single-device path (also handles num_chains=1 on any device count).
            kernel = self.kernel_class(**kernel_kwargs)
            self.mcmc = NumPyroMCMC(
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
            else np.asarray(self.prior_func(jnp.asarray(unconstrained_samples)))
        )

        model_log_likelihood = None
        if logdensity is not None and log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(
                logdensity
                - np.asarray(log_jacobian_fn(jnp.asarray(unconstrained_samples)))
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

    def get_results(self):
        return self.results

    def load_results(self):
        samples_file = self._get_output_prefix() + "_samples.npz"
        params_file = self._get_output_prefix() + "_params.json"
        metadata_file = self._get_output_prefix() + "_metadata.json"

        if not os.path.exists(samples_file):
            archived_results, parameter_names = self._load_sample_archive_results()

            if archived_results is None:
                return None

            if parameter_names:
                self.parameter_names = list(parameter_names)

            archived_log_likelihood = np.asarray(
                archived_results["log_likelihood"], dtype=float
            )
            archived_samples = np.asarray(archived_results["samples"], dtype=float)
            self.results = NumPyroSamplingResults(
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

        if os.path.exists(params_file):
            with open(params_file, "r") as handle:
                self.parameter_names = list(json.load(handle))

        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)

        with np.load(samples_file, allow_pickle=False) as archive:
            self.results = NumPyroSamplingResults(
                algorithm=str(metadata.get("algorithm", self.algorithm_name)),
                samples=np.asarray(archive["samples"], dtype=float),
                unconstrained_samples=np.asarray(
                    archive["unconstrained_samples"], dtype=float
                ),
                logdensity=np.asarray(archive["logdensity"], dtype=float),
                model_log_likelihood=(
                    np.asarray(archive["model_log_likelihood"], dtype=float)
                    if "model_log_likelihood" in archive.files
                    else None
                ),
                warmup_parameters=dict(metadata.get("warmup_parameters", {})),
                acceptance_rate=(
                    np.asarray(archive["acceptance_rate"], dtype=float)
                    if "acceptance_rate" in archive.files
                    else None
                ),
                is_divergent=(
                    np.asarray(archive["is_divergent"])
                    if "is_divergent" in archive.files
                    else None
                ),
                num_integration_steps=(
                    np.asarray(archive["num_integration_steps"], dtype=float)
                    if "num_integration_steps" in archive.files
                    else None
                ),
                output_file=samples_file,
                metadata_file=metadata_file if os.path.exists(metadata_file) else None,
            )

        return self.results

    def pretty_print_results(self):
        if self.results is None:
            return

        print(
            f"NumPyro {self.algorithm_name.upper()} completed: {self.results.samples.shape[0]} samples"
        )
        if self.results.acceptance_rate is not None:
            print(
                f"  Mean acceptance rate: {np.mean(self.results.acceptance_rate):.3f}"
            )
        if self.results.is_divergent is not None:
            print(f"  Divergences: {int(np.sum(self.results.is_divergent))}")

    def plot_diagnostics(self, results=None):
        if results is None:
            results = self.results

        if results is None:
            return {}

        try:
            _require_arviz()
            _require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        inference_data = None
        if self.mcmc is not None:
            try:
                inference_data = az.from_numpyro(self.mcmc)
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

            inference_data = az.from_dict(
                posterior={
                    name: samples[:, index][np.newaxis, :]
                    for index, name in enumerate(parameter_names)
                }
            )

        trace_output = os.path.join(
            self.output_directory,
            "out_NumPyro",
            f"{self.retrieval_name}_trace.pdf",
        )

        try:
            plot_output = az.plot_trace(
                inference_data,
                compact=True,
                figsize=(15, max(5, 2 * max(1, len(parameter_names)))),
            )
            return {"trace": self._save_plot_figure(plot_output, trace_output)}
        except Exception as error:
            warnings.warn(
                f"NumPyro diagnostics plotting failed: {error}", RuntimeWarning
            )
            return {}

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        summary["diagnostics"] = {
            "num_samples": (
                int(self.results.samples.shape[0]) if self.results is not None else None
            ),
            "mean_logdensity": (
                float(np.mean(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None
                else None
            ),
            "max_logdensity": (
                float(np.max(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None
                else None
            ),
            "mean_acceptance_rate": (
                float(np.mean(self.results.acceptance_rate))
                if self.results is not None and self.results.acceptance_rate is not None
                else None
            ),
            "n_divergent": (
                int(np.sum(self.results.is_divergent))
                if self.results is not None and self.results.is_divergent is not None
                else None
            ),
            "mean_integration_steps": (
                float(np.mean(self.results.num_integration_steps))
                if self.results is not None
                and self.results.num_integration_steps is not None
                else None
            ),
        }
        if self.results is not None:
            summary["parameters"] = dict(self.results.warmup_parameters)
            summary["posterior"] = self._posterior_summary_from_matrix(
                self.results.samples,
                free_parameter_names,
            )
        return summary


class NumPyroHMCSampler(NumPyroSampler):
    @property
    def algorithm_name(self):
        return "hmc"

    @property
    def kernel_class(self):
        return NumPyroHMC

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_numpyro(context, algorithm_name="hmc", **sampler_kwargs)


class NumPyroNUTSSampler(NumPyroSampler):
    @property
    def algorithm_name(self):
        return "nuts"

    @property
    def kernel_class(self):
        return NumPyroNUTS

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_numpyro(context, algorithm_name="nuts", **sampler_kwargs)


class PymultinestSampler(Sampler):
    """
    Sampler class for pymultinest.
    """

    def __init__(
        self,
        log_likelihood_func,  # Retrieval.log_likelihood_jax
        prior_func,  # Retrieval.prior_model_jax (generator)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_pymultinest()
        os.makedirs(os.path.join(self.output_directory, "out_PMN/"), exist_ok=True)
        self.seed = -1
        self.analyzer = None
        self._n_dims = None
        self.outputfiles_basename = os.path.join(
            self.output_directory, "out_PMN", f"{self.retrieval_name}_"
        )

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        likelihood_devices = sampler_kwargs.pop("likelihood_devices", None)

        if likelihood_devices is not None:
            device_context = _dc_replace(context, likelihood_devices=likelihood_devices)
            log_likelihood_func, target_device = _build_device_likelihood_wrapper(device_context)
        else:
            log_likelihood_func = context.log_likelihood_func
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
        if likelihood_devices is not None:
            summary_parameters["likelihood_devices"] = [str(d) for d in likelihood_devices]
            summary_parameters["likelihood_target_device"] = str(target_device)
        return sampler, run_kwargs, summary_parameters

    def post_run(self, results):
        _, _, rank = get_mpi_context()
        analyzer = results if hasattr(results, "get_stats") else self.analyzer

        if analyzer is None and rank == 0 and self._n_dims is not None:
            try:
                analyzer = pymultinest.Analyzer(
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
                desirable to have separate samples & mode statistics for modes with local log-evidence value greater than a
                particular value in which case Ztol should be set to that value. If there isn't any particularly interesting
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

        pymultinest.run(
            LogLikelihood=self.log_likelihood_func,
            Prior=self.prior_func,
            n_dims=n_dims,
            **kwargs,
        )

        if rank == 0:
            self.analyzer = pymultinest.Analyzer(
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
            self.analyzer = pymultinest.Analyzer(
                outputfiles_basename=self.outputfiles_basename,
                n_params=self._n_dims,
            )
        except Exception as error:
            warnings.warn(
                f"Unable to construct PyMultiNest analyzer for '{self.retrieval_name}' while loading saved results: {error}",
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


class UltranestSampler(Sampler):
    def __init__(
        self,
        log_likelihood_func,  # Retrieval.log_likelihood_jax
        prior_func,  # Retrieval.prior_model_jax (generator)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_ultranest()
        self.results = None

    def _get_log_dir(self):
        return os.path.join(
            self.output_directory, "out_" + self.retrieval_name + "_ultranest"
        )

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
        vectorized = cls._resolve_vectorized(
            context, sampler_kwargs.pop("vectorized", False)
        )

        if likelihood_devices is not None and vectorized:
            warnings.warn(
                "likelihood_devices is ignored in UltraNest vectorized mode "
                "(the vectorized path already dispatches through JAX).",
                RuntimeWarning,
            )
            likelihood_devices = None

        if not vectorized:
            if likelihood_devices is not None:
                device_context = _dc_replace(context, likelihood_devices=likelihood_devices)
                log_likelihood_func, target_device = _build_device_likelihood_wrapper(
                    device_context
                )
            else:
                log_likelihood_func = context.log_likelihood_func
                target_device = None
            prior_func = context.prior_ultranest_func
        else:
            parameter_layout = context.parameter_layout
            runtime = context.runtime

            def prior_transform(cube):
                return parameter_layout.prior_transform_numpy(cube)

            def log_likelihood_batch(physical_positions):
                return np.asarray(
                    runtime.evaluate_jax_batched(
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
        }
        if likelihood_devices is not None:
            summary_parameters["likelihood_devices"] = [str(d) for d in likelihood_devices]
            summary_parameters["likelihood_target_device"] = str(target_device)
        return sampler, run_kwargs, summary_parameters

    def load_results(self):
        log_dir = self._get_log_dir()
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

        self.results = results if results else None
        return self.results

    def run_sampler(self, parameter_names, **kwargs):
        log_dir = os.path.join(
            self.output_directory, "out_" + self.retrieval_name + "_ultranest"
        )
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
            "enable_step_sampling": kwargs.pop("enable_step_sampling", False),
        }

        if reactive_kwargs.get("enable_step_sampling"):
            try:
                from ultranest.stepsampler import (
                    SliceSampler,
                    generate_mixture_random_direction,
                )

                step_sampler_kwargs = {
                    "nsteps": kwargs.pop("nsteps", 100),
                    "generate_direction": kwargs.pop(
                        "generate_direction", generate_mixture_random_direction
                    ),
                    "scale": kwargs.pop("scale", 1.0),
                    "check_nsteps": kwargs.pop("check_nsteps", "move-distance"),
                    "adaptive_nsteps": kwargs.pop("adaptive_nsteps", False),
                    "max_nsteps": kwargs.pop("max_nsteps", 1000),
                    "region_filter": kwargs.pop("region_filter", False),
                    "log": kwargs.pop("log", False),
                }
                self.sampler.stepsampler = SliceSampler(**step_sampler_kwargs)
            except ImportError:
                warnings.warn(
                    "Could not import SliceSampler. Step sampler disabled.",
                    RuntimeWarning,
                )
        else:
            reactive_kwargs.pop("enable_step_sampling", None)
            self.sampler = ultranest.ReactiveNestedSampler(  # Renamed to avoid conflict with base class name
                param_names=parameter_names,
                loglike=self.log_likelihood_func,
                transform=self.prior_func,
                **reactive_kwargs,
            )
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
            "region_class": kwargs.pop("region_class", MLFriends),
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


class DynestySampler(Sampler):
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
        _require_dynesty()
        os.makedirs(os.path.join(self.output_directory, "out_Dynesty"), exist_ok=True)
        self.sampler = None
        self.results = None
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
            dynesty_pool = DynestyPool(pool_size, log_likelihood_func, prior_func)
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
            from petitRADTRANS.retrieval.dynesty_pool_helpers import (
                DynestyPoolLogLikelihood,
                DynestyPoolPriorTransform,
            )

            helper_use_jit = use_jit

            if helper_use_jit:
                try:
                    _require_jax()
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
            _require_jax()
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
        from petitRADTRANS.retrieval.dynesty_pool_helpers import (
            DynestyPoolLogLikelihood,
            DynestyPoolPriorTransform,
        )

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
        _require_dynesty()

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

    def _get_output_prefix(self):
        return os.path.join(self.output_directory, "out_Dynesty", self.retrieval_name)

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
            dyfunc.resample_equal(combined, weights, rstate=np.random.default_rng(seed)),
            dtype=float,
        )

        if equal_weight.shape[0] > 0:
            max_index = int(np.argmax(weighted_log_likelihood))
            equal_weight[0, :-1] = weighted_samples[max_index]
            equal_weight[0, -1] = weighted_log_likelihood[max_index]

        sample_count = equal_weight.shape[0]
        if sample_count > 0:
            posterior_log_weights = np.full(sample_count, -np.log(sample_count))
        else:
            posterior_log_weights = np.array([], dtype=float)

        return {
            "samples": equal_weight[:, :-1],
            "log_likelihood": equal_weight[:, -1],
            "log_weights": posterior_log_weights,
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
        _require_dynesty()

        init_kwargs = dict(init_kwargs)
        run_kwargs = dict(run_kwargs)
        seed = run_kwargs.pop("seed", 42654)
        checkpoint_file = run_kwargs.get("checkpoint_file")

        try:
            if checkpoint_file is None:
                checkpoint_file = os.path.join(
                    self.output_directory,
                    "out_Dynesty",
                    f"{self.retrieval_name}_checkpoint.save",
                )
                run_kwargs["checkpoint_file"] = checkpoint_file

            resume = bool(run_kwargs.get("resume", False))
            pool = init_kwargs.get("pool")
            checkpoint_exists = bool(
                checkpoint_file and os.path.exists(checkpoint_file)
            )

            if resume and checkpoint_exists:
                self.sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
            else:
                if resume and not checkpoint_exists:
                    warnings.warn(
                        f"Dynesty checkpoint '{checkpoint_file}' was not found; "
                        "starting a fresh run with resume=False.",
                        RuntimeWarning,
                    )
                    run_kwargs["resume"] = False

                self.sampler = DynamicNestedSampler(
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
            sample_count = samples.shape[0]
            if sample_count > 0:
                posterior_log_weights = np.full(sample_count, -np.log(sample_count))
            else:
                posterior_log_weights = np.array([], dtype=float)

            self.results = {
                "samples": samples,
                "log_likelihood": log_likelihood,
                "log_weights": posterior_log_weights,
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
            _require_dynesty()
            _require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        output_directory = os.path.join(self.output_directory, "out_Dynesty")
        labels = list(self.parameter_names or [])
        plot_files = {}

        try:
            runplot_output = dyplot.runplot(self.raw_results, logplot=True)
            plot_files["runplot"] = self._save_plot_figure(
                runplot_output,
                os.path.join(output_directory, f"{self.retrieval_name}_runplot.pdf"),
            )
        except Exception as error:
            warnings.warn(
                f"Dynesty runplot generation failed: {error}", RuntimeWarning
            )

        try:
            traceplot_output = dyplot.traceplot(
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
            cornerpoints_output = dyplot.cornerpoints(
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


class JAXNSSampler(Sampler):
    """
    Sampler class for JAXNS.
    """

    def __init__(
        self,
        log_likelihood_func,  # Retrieval.log_likelihood_jax
        prior_func,  # Retrieval.prior_model_jax (generator)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_jax()
        _require_jaxns()
        os.makedirs(os.path.join(self.output_directory, "out_JAXNS/"), exist_ok=True)
        self.jaxns_parameters = {}

        self.seed = -1
        self.sampler = None
        self.results = None
        self.raw_results = None
        self.parameter_names = []
        self.offload_results_to_cpu = True

    @classmethod
    def _build_jaxns_interface(cls, context: SamplerContext):
        """Build JAXNS prior_model generator and log_likelihood wrapper."""
        _check_differentiable_support(context, "JAXNS")
        free_parameter_names = context.free_parameter_names

        if len(free_parameter_names) < 1:
            raise ValueError("JAXNS samplers require at least one free parameter.")

        priors = []
        for parameter_name in free_parameter_names:
            parameter = context.configuration.parameters[parameter_name]
            if parameter.distribution is None:
                raise ValueError(
                    f"JAXNS samplers require distribution-backed free parameters; "
                    f"'{parameter_name}' has no distribution."
                )
            priors.append(parameter.set_prior())

        priors = tuple(priors)
        parameter_layout = context.parameter_layout
        runtime = context.runtime
        uncertainties_mode = context.uncertainties_mode

        def prior_model():
            values = []
            for prior in priors:
                value = yield prior
                values.append(value)
            return tuple(values)

        def log_likelihood_wrapper(*args):
            physical_params = parameter_layout.physical_params_from_free_values(args)
            return runtime.evaluate_jax(
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=False,
            )

        return free_parameter_names, prior_model, log_likelihood_wrapper

    @staticmethod
    def _coerce_termination_condition_mapping(values, source_name: str) -> dict[str, Any]:
        if values is None:
            return {}

        if hasattr(values, "_asdict"):
            values = values._asdict()

        if not isinstance(values, dict):
            raise TypeError(
                f"JAXNS termination condition source '{source_name}' must be a mapping or TerminationCondition, "
                f"but is '{type(values)}'"
            )

        unknown_fields = [
            field_name
            for field_name in values
            if field_name not in _JAXNS_TERMINATION_FIELDS
        ]

        if len(unknown_fields) > 0:
            raise TypeError(
                "Unknown JAXNS termination condition field(s): "
                f"{unknown_fields}. Supported fields are {_JAXNS_TERMINATION_FIELDS}."
            )

        return dict(values)

    @classmethod
    def _build_termination_condition(
        cls, sampler_kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        _require_jaxns()

        termination_condition_values: dict[str, Any] = {}

        def _add_termination_field(field_name: str, value: Any, source_name: str) -> None:
            if field_name in termination_condition_values:
                raise TypeError(
                    f"got multiple values for JAXNS termination condition field '{field_name}'"
                )

            termination_condition_values[field_name] = value

        for source_name in (
            "termination_condition",
            "term_cond",
            "termination_condition_kwargs",
        ):
            values = sampler_kwargs.pop(source_name, None)
            values = cls._coerce_termination_condition_mapping(values, source_name)

            for field_name, value in values.items():
                _add_termination_field(field_name, value, source_name)

        for field_name in _JAXNS_TERMINATION_FIELDS:
            option_names = [f"termination_{field_name}"]

            if field_name != "max_samples":
                option_names.insert(0, field_name)

            present_option_names = [
                option_name for option_name in option_names if option_name in sampler_kwargs
            ]

            if len(present_option_names) > 1:
                raise TypeError(
                    "got multiple values for JAXNS termination condition field "
                    f"'{field_name}' via {present_option_names}"
                )

            if len(present_option_names) == 0:
                continue

            option_name = present_option_names[0]
            _add_termination_field(
                field_name,
                sampler_kwargs.pop(option_name),
                option_name,
            )

        if len(termination_condition_values) == 0:
            return None, {}

        return (
            TerminationCondition(**termination_condition_values),
            dict(termination_condition_values),
        )

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        _require_jax()
        if tfp is None:
            raise ImportError(
                "TensorFlow Probability (for JAX) is required for JAXNSSampler but not installed."
            )

        free_parameter_names, prior_model, log_likelihood_wrapper = (
            cls._build_jaxns_interface(context)
        )

        termination_condition, termination_summary = cls._build_termination_condition(
            sampler_kwargs
        )

        jaxns_kwargs = {
            "num_live_points": sampler_kwargs.pop("num_live_points", 400),
            "max_samples": sampler_kwargs.pop("max_samples", 1e7),
            "s": sampler_kwargs.pop("s", 4),
            "k": sampler_kwargs.pop("k", 0),
            "c": sampler_kwargs.pop("c", 200),
            "devices": sampler_kwargs.pop("devices", devices()),
            "difficult_model": sampler_kwargs.pop("difficult_model", False),
            "parameter_estimation": sampler_kwargs.pop("parameter_estimation", False),
            "shell_fraction": sampler_kwargs.pop("shell_fraction", 0.5),
            "gradient_guided": sampler_kwargs.pop("gradient_guided", False),
            "init_efficiency_threshold": sampler_kwargs.pop(
                "init_efficiency_threshold", 0.05
            ),
            "verbose": sampler_kwargs.pop("verbose", False),
        }
        jaxns_kwargs["seed"] = sampler_kwargs.pop("seed", 42654)
        use_jit = sampler_kwargs.pop("use_jit", True)
        offload_results_to_cpu = sampler_kwargs.pop("offload_results_to_cpu", True)
        jaxns_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=log_likelihood_wrapper,
            prior_func=prior_model,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        sampler.parameter_names = list(free_parameter_names)
        sampler.offload_results_to_cpu = offload_results_to_cpu

        run_kwargs = {
            "use_jit": use_jit,
            "termination_condition": termination_condition,
            **jaxns_kwargs,
        }
        summary_parameters = {
            **jaxns_kwargs,
            "sampler_type": "jaxns",
            "parameter_names": free_parameter_names,
            "offload_results_to_cpu": offload_results_to_cpu,
        }

        if len(termination_summary) > 0:
            summary_parameters["termination_condition"] = termination_summary

        return sampler, run_kwargs, summary_parameters

    def _get_output_prefix(self):
        return os.path.join(self.output_directory, "out_JAXNS", self.retrieval_name)

    def _get_raw_results_path(self):
        return self._get_output_prefix() + "_jaxns_results.json"

    def _get_equal_weight_samples_path(self):
        return self._get_output_prefix() + "_samples.npz"

    def _get_parameter_names_path(self):
        return self._get_output_prefix() + "_params.json"

    def _get_cpu_device(self):
        if not self.offload_results_to_cpu:
            return None

        try:
            cpu_devices = devices("cpu")
        except Exception as error:
            warnings.warn(
                "No JAX CPU backend is available for result offload. "
                "Ensure the environment initializes the CPU backend, for example with JAX_PLATFORMS=cuda,cpu. "
                f"Original error: {error}",
                RuntimeWarning,
            )
            return None

        for device in cpu_devices:
            if getattr(device, "platform", None) == "cpu":
                return device

        return None

    def _materialize_raw_results(self, termination_reason, state):
        cpu_device = self._get_cpu_device()

        if cpu_device is None:
            return self.sampler.to_results(
                state=state,
                termination_reason=termination_reason,
            )

        termination_reason = device_get(termination_reason)
        state = device_get(state)

        gc.collect()
        clear_caches()

        with default_device(cpu_device):
            return self.sampler.to_results(
                state=state,
                termination_reason=termination_reason,
            )

    def _to_numpy_tree(self, pytree):
        cpu_device = self._get_cpu_device()

        if cpu_device is None:
            return tree.map(np.asarray, pytree)

        with default_device(cpu_device):
            return tree.map(np.asarray, pytree)

    @staticmethod
    def _get_named_samples_mapping(raw_results):
        for field_name in ("samples", "parametrised_samples"):
            field_value = getattr(raw_results, field_name, None)

            if isinstance(field_value, dict) and len(field_value) > 0:
                return field_value

        return None

    @staticmethod
    def _samples_mapping_to_matrix(samples_by_parameter, parameter_names=None):
        if not isinstance(samples_by_parameter, dict):
            return None, list(parameter_names or [])

        ordered_names = list(parameter_names or [])
        if ordered_names:
            ordered_names = [
                name for name in ordered_names if name in samples_by_parameter
            ]

        if not ordered_names:
            ordered_names = list(samples_by_parameter.keys())

        if not ordered_names:
            return None, []

        matrix = np.column_stack(
            [
                np.asarray(samples_by_parameter[name], dtype=float).reshape(-1)
                for name in ordered_names
            ]
        )
        return matrix, ordered_names

    def _save_equal_weight_samples(self, parameter_names, samples, log_likelihood):
        if samples is None:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because no posterior sample matrix is available.",
                RuntimeWarning,
            )
            return None, None

        sample_matrix = np.asarray(samples, dtype=float)
        log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)
        parameter_names = list(parameter_names)

        if sample_matrix.ndim == 1:
            if len(parameter_names) != 1:
                warnings.warn(
                    "Skipping JAXNS equal-weight sample export because the sample matrix is 1-D but the parameter list is not.",
                    RuntimeWarning,
                )
                return None, None
            sample_matrix = sample_matrix.reshape(-1, 1)
        elif sample_matrix.ndim != 2:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because the sample matrix is not 2-D.",
                RuntimeWarning,
            )
            return None, None

        if len(parameter_names) == 0 or sample_matrix.shape[1] != len(parameter_names):
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because parameter names do not match the sample matrix columns.",
                RuntimeWarning,
            )
            return None, None

        if log_likelihood.shape[0] != sample_matrix.shape[0]:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because log-likelihood values do not match the number of samples.",
                RuntimeWarning,
            )
            return None, None

        samples_file = self._get_equal_weight_samples_path()
        params_file = self._get_parameter_names_path()

        np.savez(
            samples_file,
            parameter_names=np.asarray(parameter_names, dtype=str),
            sample_matrix=sample_matrix,
            log_likelihood=log_likelihood,
        )

        with open(params_file, "w") as handle:
            json.dump(list(parameter_names), handle, indent=4)

        return samples_file, params_file

    def _build_processed_results(
        self,
        raw_results,
        *,
        equal_weight_samples=None,
        equal_weight_log_likelihood=None,
        output_file=None,
        params_file=None,
        raw_results_file=None,
    ):
        weighted_matrix, parameter_names = self._samples_mapping_to_matrix(
            self._get_named_samples_mapping(raw_results),
            self.parameter_names,
        )
        self.parameter_names = list(parameter_names)

        weighted_log_likelihood = np.asarray(
            getattr(raw_results, "log_L_samples", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)
        weighted_log_weights = np.asarray(
            getattr(raw_results, "log_dp_mean", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)

        if equal_weight_samples is None or equal_weight_log_likelihood is None:
            samples = weighted_matrix
            log_likelihood = weighted_log_likelihood
            log_weights = weighted_log_weights
        else:
            samples = np.asarray(equal_weight_samples, dtype=float)
            if samples.ndim == 1:
                target_width = 1 if len(self.parameter_names) <= 1 else len(self.parameter_names)
                samples = samples.reshape(-1, target_width)

            log_likelihood = np.asarray(
                equal_weight_log_likelihood,
                dtype=float,
            ).reshape(-1)
            sample_count = samples.shape[0]
            if sample_count > 0:
                log_weights = np.full(sample_count, -np.log(sample_count), dtype=float)
            else:
                log_weights = np.array([], dtype=float)

        processed_results = {
            "samples": samples,
            "log_likelihood": log_likelihood,
            "log_weights": log_weights,
            "weighted_samples": {
                "points": weighted_matrix,
                "logl": weighted_log_likelihood,
                "log_weights": weighted_log_weights,
            },
        }

        if output_file is not None:
            processed_results["output_file"] = output_file
        if params_file is not None:
            processed_results["params_file"] = params_file
        if raw_results_file is not None:
            processed_results["raw_results_file"] = raw_results_file

        for key in (
            "log_Z_mean",
            "log_Z_uncert",
            "ESS",
            "H_mean",
            "total_num_samples",
            "total_num_likelihood_evaluations",
            "log_efficiency",
        ):
            value = getattr(raw_results, key, None)
            if value is None:
                continue

            value = np.asarray(value)
            if value.size == 1:
                processed_results[key] = value.item()
            else:
                processed_results[key] = value

        return processed_results

    def _build_equal_weight_results(self, raw_results):
        named_samples = self._get_named_samples_mapping(raw_results)
        posterior_payload = {
            "samples": named_samples or {},
            "log_likelihood": getattr(raw_results, "log_L_samples", np.array([])),
        }

        cpu_device = self._get_cpu_device()

        if cpu_device is None:
            resampled_payload = jaxns_resample(
                random.PRNGKey(self.seed),
                posterior_payload,
                getattr(raw_results, "log_dp_mean"),
            )
            resampled_payload = tree.map(np.asarray, resampled_payload)
        else:
            with default_device(cpu_device):
                resampled_payload = jaxns_resample(
                    random.PRNGKey(self.seed),
                    posterior_payload,
                    getattr(raw_results, "log_dp_mean"),
                )
                resampled_payload = tree.map(np.asarray, resampled_payload)

        equal_weight_samples, parameter_names = self._samples_mapping_to_matrix(
            resampled_payload["samples"],
            self.parameter_names,
        )
        self.parameter_names = list(parameter_names)
        equal_weight_log_likelihood = np.asarray(
            resampled_payload["log_likelihood"],
            dtype=float,
        ).reshape(-1)

        weighted_matrix, _ = self._samples_mapping_to_matrix(
            named_samples,
            self.parameter_names,
        )
        weighted_log_likelihood = np.asarray(
            getattr(raw_results, "log_L_samples", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)

        if (
            equal_weight_samples is not None
            and equal_weight_samples.shape[0] > 0
            and weighted_matrix is not None
            and weighted_log_likelihood.shape[0] == weighted_matrix.shape[0]
        ):
            best_index = int(np.argmax(weighted_log_likelihood))
            equal_weight_samples[0, :] = weighted_matrix[best_index, :]
            equal_weight_log_likelihood[0] = weighted_log_likelihood[best_index]

        samples_file, params_file = self._save_equal_weight_samples(
            self.parameter_names,
            equal_weight_samples,
            equal_weight_log_likelihood,
        )

        return self._build_processed_results(
            raw_results,
            equal_weight_samples=equal_weight_samples,
            equal_weight_log_likelihood=equal_weight_log_likelihood,
            output_file=samples_file,
            params_file=params_file,
            raw_results_file=self._get_raw_results_path(),
        )

    def post_run(self, results):
        """JAXNS post-run: get_results is already called in run_sampler, pass stats=results."""
        results = self.get_results()
        return results, {"stats": results}

    def run_sampler(self, use_jit=True, parameters={}, **jaxns_kwargs):
        """
        Run the JAXNS sampler.
        """
        _require_jaxns()

        termination_condition, _termination_summary = self._build_termination_condition(
            jaxns_kwargs
        )

        self.jaxns_parameters = {
            "num_live_points": jaxns_kwargs.pop("num_live_points", 400),
            "max_samples": jaxns_kwargs.pop("max_samples", 1e7),
            "s": jaxns_kwargs.pop("s", 4),
            "k": jaxns_kwargs.pop("k", 0),
            "c": jaxns_kwargs.pop("c", 200),
            "devices": jaxns_kwargs.pop("devices", devices()),
            "difficult_model": jaxns_kwargs.pop("difficult_model", False),
            "parameter_estimation": jaxns_kwargs.pop("parameter_estimation", False),
            "shell_fraction": jaxns_kwargs.pop("shell_fraction", 0.5),
            "gradient_guided": jaxns_kwargs.pop("gradient_guided", False),
            "init_efficiency_threshold": jaxns_kwargs.pop(
                "init_efficiency_threshold", 0.05
            ),
            "verbose": jaxns_kwargs.pop("verbose", False),
        }

        self.seed = jaxns_kwargs.pop("seed", 42654)
        # The prior_func is expected to be the prior_model (a generator)

        jaxified_likelihood = self.log_likelihood_func

        model = Model(prior_model=self.prior_func, log_likelihood=jaxified_likelihood)

        self.sampler = NestedSampler(model, **self.jaxns_parameters)

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")

        if use_jit:
            termination_reason, state = jit(self.sampler)(
                jax_random_key,
                termination_condition,
            )
        else:
            termination_reason, state = self.sampler(
                jax_random_key,
                termination_condition,
            )

        self.raw_results = self._materialize_raw_results(
            termination_reason=termination_reason,
            state=state,
        )

        del termination_reason
        del state
        gc.collect()
        clear_caches()

        results_out = self._to_numpy_tree(self.raw_results)
        save_results(
            results=results_out,
            save_file=self._get_raw_results_path(),
        )

        self.raw_results = results_out

        self.results = self._build_equal_weight_results(self.raw_results)

        self.pretty_print_results()
        self.plot_diagnostics(self.results)
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        raw_results_path = self._get_raw_results_path()

        if not os.path.exists(raw_results_path):
            return super().load_results()

        _require_jaxns()
        self.raw_results = load_results(raw_results_path)

        parameter_names_path = self._get_parameter_names_path()
        if os.path.exists(parameter_names_path):
            with open(parameter_names_path, "r") as handle:
                self.parameter_names = list(json.load(handle))
        else:
            named_samples = self._get_named_samples_mapping(self.raw_results)
            self.parameter_names = list(named_samples.keys()) if named_samples else []

        samples_path = self._get_equal_weight_samples_path()
        if os.path.exists(samples_path):
            with np.load(samples_path, allow_pickle=False) as archive:
                if "parameter_names" in archive.files:
                    self.parameter_names = [
                        str(name) for name in archive["parameter_names"].tolist()
                    ]

                equal_weight_samples = np.asarray(
                    archive["sample_matrix"],
                    dtype=float,
                )
                equal_weight_log_likelihood = np.asarray(
                    archive["log_likelihood"],
                    dtype=float,
                ).reshape(-1)

            self.results = self._build_processed_results(
                self.raw_results,
                equal_weight_samples=equal_weight_samples,
                equal_weight_log_likelihood=equal_weight_log_likelihood,
                output_file=samples_path,
                params_file=parameter_names_path if os.path.exists(parameter_names_path) else None,
                raw_results_file=self._get_raw_results_path(),
            )
            return self.results

        self.results = self._build_processed_results(
            self.raw_results,
            output_file=samples_path if os.path.exists(samples_path) else None,
            params_file=parameter_names_path if os.path.exists(parameter_names_path) else None,
            raw_results_file=self._get_raw_results_path(),
        )
        return self.results

    def pretty_print_results(self):
        if self.raw_results is not None and self.sampler is not None:
            self.sampler.summary(self.raw_results)
            return

        if self.results is None:
            return

        sample_count = np.asarray(self.results.get("samples", np.array([]))).shape[0]
        print(f"JAXNS completed: {sample_count} posterior samples")

    def plot_diagnostics(self, results=None):
        if results is None:
            results = self.results

        diagnostics_results = self.raw_results if self.raw_results is not None else results

        if diagnostics_results is None or self.sampler is None:
            return {}

        try:
            _require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        output_directory = os.path.join(self.output_directory, "out_JAXNS")
        plot_files = {}

        try:
            diagnostics_output = self.sampler.plot_diagnostics(diagnostics_results)
            plot_files["diagnostics"] = self._save_plot_figure(
                diagnostics_output,
                os.path.join(
                    output_directory,
                    f"{self.retrieval_name}_diagnostics.pdf",
                ),
            )
        except Exception as error:
            warnings.warn(f"JAXNS diagnostics plotting failed: {error}", RuntimeWarning)

        try:
            corner_output = self.sampler.plot_cornerplot(diagnostics_results)
            plot_files["cornerplot"] = self._save_plot_figure(
                corner_output,
                os.path.join(
                    output_directory,
                    f"{self.retrieval_name}_cornerplot.pdf",
                ),
            )
        except Exception as error:
            warnings.warn(
                f"JAXNS corner plot generation failed: {error}", RuntimeWarning
            )

        return plot_files

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.results is None:
            return summary

        if isinstance(self.results, dict):
            summary["evidence"] = {
                "ln_z": self._summary_scalar(self.results.get("log_Z_mean")),
                "ln_z_err": self._summary_scalar(self.results.get("log_Z_uncert")),
            }
            summary["diagnostics"] = {
                "ess": self._summary_scalar(self.results.get("ESS")),
                "h_mean": self._summary_scalar(self.results.get("H_mean")),
                "total_num_samples": self._summary_scalar(
                    self.results.get("total_num_samples")
                ),
                "total_num_likelihood_evaluations": self._summary_scalar(
                    self.results.get("total_num_likelihood_evaluations")
                ),
            }
            summary["posterior"] = self._posterior_summary_from_matrix(
                self.results.get("samples"),
                list(free_parameter_names or self.parameter_names),
            )
            if not summary["posterior"] and isinstance(
                self.results.get("weighted_samples"), dict
            ):
                summary["posterior"] = self._posterior_summary_from_matrix(
                    self.results["weighted_samples"].get("points"),
                    list(free_parameter_names or self.parameter_names),
                )
            return summary

        posterior_sources = []
        if hasattr(self.results, "samples"):
            posterior_sources.append(getattr(self.results, "samples"))
        if hasattr(self.results, "parametrised_samples"):
            posterior_sources.append(getattr(self.results, "parametrised_samples"))

        for source in (self.results, getattr(self.results, "_asdict", lambda: {})()):
            if not source:
                continue
            ln_z = (
                getattr(source, "log_Z_mean", None)
                if not isinstance(source, dict)
                else source.get("log_Z_mean")
            )
            ln_z_err = (
                getattr(source, "log_Z_uncert", None)
                if not isinstance(source, dict)
                else source.get("log_Z_uncert")
            )
            ess = (
                getattr(source, "ESS", None)
                if not isinstance(source, dict)
                else source.get("ESS")
            )
            h_mean = (
                getattr(source, "H_mean", None)
                if not isinstance(source, dict)
                else source.get("H_mean")
            )
            if any(value is not None for value in (ln_z, ln_z_err, ess, h_mean)):
                summary["evidence"] = {
                    "ln_z": self._summary_scalar(ln_z),
                    "ln_z_err": self._summary_scalar(ln_z_err),
                }
                summary["diagnostics"] = {
                    "ess": self._summary_scalar(ess),
                    "h_mean": self._summary_scalar(h_mean),
                }
                break

        for posterior_source in posterior_sources:
            summary["posterior"] = self._posterior_summary_from_mapping(
                posterior_source,
                free_parameter_names=free_parameter_names,
            )
            if summary["posterior"]:
                break

        return summary


class JAXNSShardedStaticNestedSampler(JAXNSSampler):
    """
    Sampler class for JAXNS.
    """

    def __init__(
        self,
        log_likelihood_func,  # Retrieval.log_likelihood_jax
        prior_func,  # Retrieval.prior_model_jax (generator)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _require_jaxns()
        self.log_likelihood_func = log_likelihood_func
        self.prior_function = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name
        os.makedirs(os.path.join(self.output_directory, "out_JAXNS/"), exist_ok=True)
        self.jaxns_parameters = {}

        self.seed = -1
        self.sampler = None
        self.results = None

    def run_sampler(self, use_jit=True, **jaxns_kwargs):
        """
        Run the JAXNS sampler.
        """
        _require_jaxns()

        termination_condition, _termination_summary = self._build_termination_condition(
            jaxns_kwargs
        )

        self.jaxns_parameters = {
            "num_live_points": jaxns_kwargs.pop("num_live_points", 400),
            "max_samples": jaxns_kwargs.pop("max_samples", 1e7),
            "devices": jaxns_kwargs.pop("devices", devices()),
            "difficult_model": jaxns_kwargs.pop("difficult_model", False),
            "verbose": jaxns_kwargs.pop("verbose", False),
        }

        sampler_class = jaxns_kwargs.pop("sampler", UniformSampler)

        self.seed = jaxns_kwargs.pop("seed", 42654)
        # The prior_func is expected to be the prior_model (a generator)
        model = Model(
            prior_model=self.prior_func, log_likelihood=self.log_likelihood_func
        )

        self.sampler = ShardedStaticNestedSampler(
            model, sampler=sampler_class, **self.jaxns_parameters
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")

        if use_jit:
            termination_reason, state = jit(self.sampler)(
                jax_random_key,
                termination_condition,
            )
        else:
            termination_reason, state = self.sampler(
                jax_random_key,
                termination_condition,
            )

        self.raw_results = self._materialize_raw_results(
            termination_reason=termination_reason,
            state=state,
        )

        del termination_reason
        del state
        gc.collect()
        clear_caches()

        results_out = self._to_numpy_tree(self.raw_results)
        save_results(
            results=results_out,
            save_file=self._get_raw_results_path(),
        )

        self.raw_results = results_out

        self.results = self._build_equal_weight_results(self.raw_results)

        self.pretty_print_results()
        self.plot_diagnostics(self.results)
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        _require_jaxns()
        return super().load_results()

    def pretty_print_results(self):
        super().pretty_print_results()
