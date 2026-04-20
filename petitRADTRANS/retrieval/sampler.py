import abc
from dataclasses import dataclass, field
import os
import json
from typing import Any, TYPE_CHECKING
import warnings
import numpy as np

if TYPE_CHECKING:
    from petitRADTRANS.retrieval.runtime import ParameterLayout, RetrievalRuntime

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None
    comm = None
    rank = 0

lax = None
random = None
tree = None
vmap = None
tree_util = None
jit = None
local_device_count = None
make_mesh = None
shard_map = None
devices = None
jnp = None
NamedSharding = None
P = None
tfp = None


def _require_jax():
    global lax, random, tree, vmap, tree_util, jit, local_device_count
    global make_mesh, shard_map, devices, jnp, NamedSharding, P, tfp

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
            devices as _devices,
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
    devices = _devices
    jnp = _jnp
    NamedSharding = _ns
    P = _ps
    tfp = _tfp


jaxns = None
Model = None
NestedSampler = None
save_results = None
load_results = None
ShardedStaticNestedSampler = None
UniformSampler = None


def _require_jaxns():
    global jaxns, Model, NestedSampler, save_results, load_results, ShardedStaticNestedSampler, UniformSampler

    if jaxns is not None:
        return

    try:
        import jaxns as jaxns_module
        from jaxns import Model as jaxns_model
        from jaxns import NestedSampler as jaxns_nested_sampler
        from jaxns import load_results as jaxns_load_results
        from jaxns import save_results as jaxns_save_results
        from jaxns.nested_samplers import (
            ShardedStaticNestedSampler as jaxns_sharded_static_nested_sampler,
        )
        from jaxns.samplers import UniformSampler as jaxns_uniform_sampler
    except Exception as error:
        raise ImportError(
            "JAXNS module or its JAX dependencies are not installed. This sampler cannot be used."
        ) from error

    jaxns = jaxns_module
    Model = jaxns_model
    NestedSampler = jaxns_nested_sampler
    save_results = jaxns_save_results
    load_results = jaxns_load_results
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


def _check_runtime_native_support(context: SamplerContext, sampler_label: str) -> None:
    """Raise if the retrieval has only legacy (non-differentiable) model groups."""
    runtime = context.runtime
    if all(group.model_contract != "differentiable" for group in runtime.model_groups):
        raise ValueError(
            f"{sampler_label} requires at least one differentiable model group; "
            "legacy-only retrievals are not supported."
        )


def _build_unconstrained_mcmc_interface(context: SamplerContext):
    """Build the unconstrained-space logdensity and transforms for gradient-based samplers.

    Returns ``(free_parameter_names, initial_position, logdensity, transform_positions, log_jacobian)``.
    """
    _require_jax()
    from petitRADTRANS.retrieval.parameter import JAX_DTYPE

    _check_runtime_native_support(context, "Gradient-based JAX samplers")
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
                    ),  # outputs concatenated along axis 0 across devices
                )
                if use_jit:
                    run_all = jit(run_all)
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

            def run_on_device(base_key):
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
                    device_key, init_params=_initial_position, extra_fields=extra_fields
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
                in_specs=P(),  # base_key replicated to every device
                out_specs=(P("devices"),)
                * 7,  # all 7 outputs concatenated along axis 0
            )
            (
                all_unc,
                all_pe,
                all_accept,
                all_diverge,
                all_steps,
                all_step_sizes,
                all_imm,
            ) = run_sharded(rng_key)

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

        return self.results

    def get_results(self):
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
        self.outputfiles_basename = os.path.join(
            self.output_directory, "out_PMN", f"{self.retrieval_name}_"
        )

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        prefix = os.path.join(
            context.output_directory,
            "out_PMN",
            context.retrieval_name + "_",
        )
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
            log_likelihood_func=context.log_likelihood_func,
            prior_func=context.prior_func,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        # Store context info needed for post_run
        sampler._free_parameter_names = context.free_parameter_names
        sampler._prefix = prefix

        summary_parameters = dict(run_kwargs)
        return sampler, run_kwargs, summary_parameters

    def post_run(self, results):
        analyzer = results
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
            sampling_efficiency : Float
                pymultinest sampling efficiency. If const efficiency mode is true, should be set to around
                0.05. Otherwise, it should be around 0.8 for parameter estimation and 0.3 for evidence
                comparison.
            const_efficiency_mode : Bool
                pymultinest constant efficiency mode
            n_live_points : Int
                Number of live points to use in pymultinest, or the minimum number of live points to
                use for the Ultranest reactive sampler.
            log_z_convergence : float
                If ultranest is being used, the convergence criterion on log z.
            step_sampler : bool
                Use a step sampler to improve the efficiency in ultranest.
            warmstart_max_tau : float
                Warm start allows accelerated computation based on a different but similar UltraNest run.
            n_iter_before_update : int
                Number of live point replacements before printing an update to a log file.
            max_iters : int
                Maximum number of sampling iterations. If 0, will continue until convergence criteria are satisfied.
            frac_remain : float
                Ultranest convergence criterion. Halts integration if live point weights are below the specified value.
            l_epsilon : float
                Ultranest convergence criterion. Use with noisy likelihoods. Halts integration if live points are
                within l_epsilon.
            resume : bool
                Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
            error_checking : bool
                Test the model generating function for typical errors. ONLY TURN THIS OFF IF YOU KNOW WHAT YOU'RE DOING!
            force_serial_error_checking : bool
                If True, error checking will be performed process-by-process, instead of with all processes at once.
                This can prevent memory overflow.
            seed : int
                Random number generator seed, -ve value for seed from the system clock (for reproducibility)
        Returns:
            analyzer
        """
        outputfiles_basename = kwargs.get(
            "outputfiles_basename", self.outputfiles_basename
        )

        pymultinest.run(
            LogLikelihood=self.log_likelihood_func,
            Prior=self.prior_func,
            n_dims=n_dims,
            **kwargs,
        )

        if rank == 0:
            self.outputfiles_basename = outputfiles_basename
            self.analyzer = pymultinest.Analyzer(
                outputfiles_basename=self.outputfiles_basename, n_params=n_dims
            )
        return self.analyzer

    def pretty_print(self, free_parameter_names, prefix):
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
        vectorized = cls._resolve_vectorized(
            context, sampler_kwargs.pop("vectorized", False)
        )

        if not vectorized:
            log_likelihood_func = context.log_likelihood_func
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
        return sampler, run_kwargs, summary_parameters

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

    @classmethod
    def _build_jaxns_interface(cls, context: SamplerContext):
        """Build JAXNS prior_model generator and log_likelihood wrapper."""
        _check_runtime_native_support(context, "JAXNS")
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
        jaxns_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=log_likelihood_wrapper,
            prior_func=prior_model,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )

        run_kwargs = {"use_jit": use_jit, **jaxns_kwargs}
        summary_parameters = {
            **jaxns_kwargs,
            "sampler_type": "jaxns",
            "parameter_names": free_parameter_names,
        }
        return sampler, run_kwargs, summary_parameters

    def post_run(self, results):
        """JAXNS post-run: get_results is already called in run_sampler, pass stats=results."""
        results = self.get_results()
        return results, {"stats": results}

    def run_sampler(self, use_jit=True, parameters={}, **jaxns_kwargs):
        """
        Run the JAXNS sampler.
        """
        _require_jaxns()

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
            termination_reason, state = jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(
            state=state, termination_reason=termination_reason
        )

        results_out = tree.map(np.asarray, self.results)
        save_results(
            results=results_out,
            save_file=os.path.join(
                self.output_directory,
                "out_JAXNS",
                self.retrieval_name + "_jaxns_results.json",
            ),
        )

        self.pretty_print_results()
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        _require_jaxns()
        self.results = load_results(
            os.path.join(
                self.output_directory,
                "out_JAXNS/",
                self.retrieval_name + "_jaxns_results.json",
            )
        )
        return self.results

    def pretty_print_results(self):
        self.sampler.summary(self.results)

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.results is None:
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

        self.jaxns_parameters = {
            "num_live_points": jaxns_kwargs.pop("num_live_points", 400),
            "sampler": jaxns_kwargs.pop("sampler", 400),
            "max_samples": jaxns_kwargs.pop("max_samples", 1e7),
            "devices": jaxns_kwargs.pop("devices", devices()),
            "difficult_model": jaxns_kwargs.pop("difficult_model", False),
            "verbose": jaxns_kwargs.pop("verbose", False),
        }

        self.seed = jaxns_kwargs.pop("seed", 42654)
        # The prior_func is expected to be the prior_model (a generator)
        model = Model(
            prior_model=self.prior_func, log_likelihood=self.log_likelihood_func
        )

        # TODO Test Sharded parallel sampler
        # TODO test different sampler choices.
        self.sampler = ShardedStaticNestedSampler(
            model, sampler=UniformSampler, **self.jaxns_parameters
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")

        if use_jit:
            termination_reason, state = jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(
            state=state, termination_reason=termination_reason
        )

        results_out = tree.map(np.asarray, self.results)
        save_results(
            results=results_out,
            save_file=os.path.join(
                self.output_directory,
                "out_JAXNS",
                self.retrieval_name + "_jaxns_results.json",
            ),
        )

        self.pretty_print_results()
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        _require_jaxns()
        self.results = load_results(
            os.path.join(
                self.output_directory,
                "out_JAXNS/",
                self.retrieval_name + "_jaxns_results.json",
            )
        )
        return self.results

    def pretty_print_results(self):
        self.sampler.summary(self.results)
