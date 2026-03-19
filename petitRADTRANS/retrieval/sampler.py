import abc
from dataclasses import dataclass
import os
import json
import warnings
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None
    comm = None
    rank = 0

try:
    import pymultinest
except Exception:
    pymultinest = None

try:
    import ultranest
    from ultranest.mlfriends import RobustEllipsoidRegion, MLFriends
except Exception:
    ultranest = None  # Mark ultranest as not available

try:
    import jax
    from jax import lax, random, tree
    import jax.numpy as jnp
except Exception:
    jax = None
    lax = None
    random = None
    tree = None
    jnp = None

try:
    import blackjax
except Exception:
    blackjax = None

try:
    import numpyro
    from numpyro.infer import HMC as NumPyroHMC
    from numpyro.infer import MCMC as NumPyroMCMC
    from numpyro.infer import NUTS as NumPyroNUTS
except Exception:
    numpyro = None
    NumPyroHMC = None
    NumPyroMCMC = None
    NumPyroNUTS = None

try:
    import jaxns
    from jaxns import Model, NestedSampler, save_results, load_results, summary as jaxns_summary  # Renamed to avoid conflict
    from jaxns.nested_samplers import ShardedStaticNestedSampler
    from jaxns.samplers import UniformSampler
except Exception:
    jaxns = None  # Flag that jaxns is not available


class Sampler(abc.ABC):
    """
    Abstract base class for samplers.
    """
    def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name

    @abc.abstractmethod
    def run_sampler(self, **kwargs):
        pass

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
            'sampler': self.__class__.__name__,
            'parameters': {},
            'diagnostics': {},
            'evidence': {},
            'posterior': [],
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
            posterior.append({
                'name': parameter_name,
                'median': float(np.quantile(parameter_samples, 0.5)),
                'lower_1sigma': float(np.quantile(parameter_samples, 0.15865525393145707)),
                'upper_1sigma': float(np.quantile(parameter_samples, 0.8413447460685429)),
            })

        return posterior

    @staticmethod
    def _posterior_summary_from_mapping(samples_by_parameter, free_parameter_names=None):
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

            posterior.append({
                'name': parameter_name,
                'median': float(np.quantile(parameter_samples, 0.5)),
                'lower_1sigma': float(np.quantile(parameter_samples, 0.15865525393145707)),
                'upper_1sigma': float(np.quantile(parameter_samples, 0.8413447460685429)),
            })

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
    def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        if blackjax is None or jax is None:
            raise ImportError("BlackJAX and JAX are required for this sampler but are not installed.")

        os.makedirs(os.path.join(self.output_directory, "out_BlackJAX"), exist_ok=True)
        self.results = None

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
                key: BlackJAXSampler._to_serializable(val)
                for key, val in value.items()
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
            'samples': samples,
            'unconstrained_samples': unconstrained_samples,
            'logdensity': logdensity,
        }
        if model_log_likelihood is not None:
            arrays_to_save['model_log_likelihood'] = model_log_likelihood
        if acceptance_rate is not None:
            arrays_to_save['acceptance_rate'] = acceptance_rate
        if is_divergent is not None:
            arrays_to_save['is_divergent'] = is_divergent
        if num_integration_steps is not None:
            arrays_to_save['num_integration_steps'] = num_integration_steps

        np.savez(samples_file, **arrays_to_save)

        with open(params_file, 'w') as handle:
            json.dump(parameter_names, handle, indent=4)

        metadata['warmup_parameters'] = self._to_serializable(warmup_parameters)
        with open(metadata_file, 'w') as handle:
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
        jit=True,
        log_jacobian_fn=None,
        **blackjax_kwargs,
    ):
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
        _, warmup_key, sample_key = random.split(rng_key, 3)
        (state, warmup_parameters), _ = warmup.run(
            warmup_key,
            initial_position,
            num_steps=num_warmup,
        )

        algorithm = self.algorithm_factory(self.log_likelihood_func, **warmup_parameters)
        kernel = algorithm.step

        def one_step(current_state, rng_key_):
            next_state, info = kernel(rng_key_, current_state)
            return next_state, (next_state, info)

        def inference_loop(sample_rng_key, initial_state):
            keys = random.split(sample_rng_key, num_samples)
            return lax.scan(one_step, initial_state, keys)

        if jit:
            inference_loop = jax.jit(inference_loop)

        _, (states, infos) = inference_loop(sample_key, state)

        unconstrained_samples = np.asarray(states.position)
        transformed_samples = states.position if self.prior_func is None else self.prior_func(states.position)
        samples = np.asarray(transformed_samples)
        logdensity = np.asarray(states.logdensity)
        acceptance_rate = np.asarray(infos.acceptance_rate) if hasattr(infos, 'acceptance_rate') else None
        is_divergent = np.asarray(infos.is_divergent) if hasattr(infos, 'is_divergent') else None
        num_integration_steps = (
            np.asarray(infos.num_integration_steps)
            if hasattr(infos, 'num_integration_steps') else None
        )

        model_log_likelihood = None
        if log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(states.logdensity - log_jacobian_fn(states.position))

        warmup_parameters_np = tree.map(np.asarray, warmup_parameters)
        metadata = {
            'algorithm': self.algorithm_name,
            'num_samples': num_samples,
            'num_warmup': num_warmup,
            'seed': seed,
            'parameter_names': parameter_names,
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

        print(f"BlackJAX {self.algorithm_name.upper()} completed: {self.results.samples.shape[0]} samples")
        if self.results.acceptance_rate is not None:
            print(f"  Mean acceptance rate: {np.mean(self.results.acceptance_rate):.3f}")
        if self.results.is_divergent is not None:
            print(f"  Divergences: {int(np.sum(self.results.is_divergent))}")

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        summary['diagnostics'] = {
            'num_samples': int(self.results.samples.shape[0]) if self.results is not None else None,
            'mean_logdensity': (
                float(np.mean(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None else None
            ),
            'max_logdensity': (
                float(np.max(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None else None
            ),
            'mean_acceptance_rate': (
                float(np.mean(self.results.acceptance_rate))
                if self.results is not None and self.results.acceptance_rate is not None else None
            ),
            'n_divergent': (
                int(np.sum(self.results.is_divergent))
                if self.results is not None and self.results.is_divergent is not None else None
            ),
            'mean_integration_steps': (
                float(np.mean(self.results.num_integration_steps))
                if self.results is not None and self.results.num_integration_steps is not None else None
            ),
        }
        if self.results is not None:
            summary['parameters'] = dict(self.results.warmup_parameters)
            summary['posterior'] = self._posterior_summary_from_matrix(
                self.results.samples,
                free_parameter_names,
            )
        return summary


class BlackJAXHMCSampler(BlackJAXSampler):
    @property
    def algorithm_name(self):
        return 'hmc'

    @property
    def algorithm_factory(self):
        return blackjax.hmc


class BlackJAXNUTSSampler(BlackJAXSampler):
    @property
    def algorithm_name(self):
        return 'nuts'

    @property
    def algorithm_factory(self):
        return blackjax.nuts


class NumPyroSampler(Sampler):
    def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        if numpyro is None or jax is None:
            raise ImportError("NumPyro and JAX are required for this sampler but are not installed.")

        os.makedirs(os.path.join(self.output_directory, "out_NumPyro"), exist_ok=True)
        self.results = None
        self.mcmc = None

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
                key: NumPyroSampler._to_serializable(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [NumPyroSampler._to_serializable(val) for val in value]

        return value

    def _get_output_prefix(self):
        return os.path.join(self.output_directory, "out_NumPyro", self.retrieval_name)

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
            'samples': samples,
            'unconstrained_samples': unconstrained_samples,
            'logdensity': logdensity,
        }
        if model_log_likelihood is not None:
            arrays_to_save['model_log_likelihood'] = model_log_likelihood
        if acceptance_rate is not None:
            arrays_to_save['acceptance_rate'] = acceptance_rate
        if is_divergent is not None:
            arrays_to_save['is_divergent'] = is_divergent
        if num_integration_steps is not None:
            arrays_to_save['num_integration_steps'] = num_integration_steps

        np.savez(samples_file, **arrays_to_save)

        with open(params_file, 'w') as handle:
            json.dump(parameter_names, handle, indent=4)

        metadata['warmup_parameters'] = self._to_serializable(warmup_parameters)
        with open(metadata_file, 'w') as handle:
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
        jit=True,
        log_jacobian_fn=None,
        num_chains=1,
        thinning=1,
        chain_method='parallel',
        **numpyro_kwargs,
    ):
        kernel_kwargs = {
            'potential_fn': lambda position: -self.log_likelihood_func(position),
            'step_size': initial_step_size,
            'adapt_step_size': numpyro_kwargs.pop('adapt_step_size', True),
            'adapt_mass_matrix': numpyro_kwargs.pop('adapt_mass_matrix', True),
            'dense_mass': numpyro_kwargs.pop('dense_mass', not is_mass_matrix_diagonal),
            'target_accept_prob': target_acceptance_rate,
            'find_heuristic_step_size': numpyro_kwargs.pop('find_heuristic_step_size', False),
            'forward_mode_differentiation': numpyro_kwargs.pop('forward_mode_differentiation', False),
            'regularize_mass_matrix': numpyro_kwargs.pop('regularize_mass_matrix', True),
        }
        kernel_kwargs.update(numpyro_kwargs)

        if not jit:
            warnings.warn(
                "NumPyro MCMC does not expose a simple non-JIT execution mode through this wrapper; "
                "'jit=False' is ignored.",
                RuntimeWarning,
            )

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

        extra_fields = ('potential_energy', 'accept_prob', 'diverging', 'num_steps')
        rng_key = random.PRNGKey(seed)
        self.mcmc.run(
            rng_key,
            init_params=initial_position,
            extra_fields=extra_fields,
        )

        unconstrained_samples = np.asarray(self.mcmc.get_samples(group_by_chain=False))
        transformed_samples = (
            unconstrained_samples
            if self.prior_func is None
            else np.asarray(self.prior_func(jnp.asarray(unconstrained_samples)))
        )

        extra = self.mcmc.get_extra_fields(group_by_chain=False)
        potential_energy = np.asarray(extra.get('potential_energy')) if 'potential_energy' in extra else None
        logdensity = -potential_energy if potential_energy is not None else None
        acceptance_rate = np.asarray(extra.get('accept_prob')) if 'accept_prob' in extra else None
        is_divergent = np.asarray(extra.get('diverging')) if 'diverging' in extra else None
        num_integration_steps = np.asarray(extra.get('num_steps')) if 'num_steps' in extra else None

        model_log_likelihood = None
        if logdensity is not None and log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(
                logdensity - np.asarray(log_jacobian_fn(jnp.asarray(unconstrained_samples)))
            )

        adapt_state = self.mcmc.last_state.adapt_state
        warmup_parameters = {
            'step_size': np.asarray(adapt_state.step_size),
            'inverse_mass_matrix': np.asarray(adapt_state.inverse_mass_matrix),
        }
        metadata = {
            'algorithm': self.algorithm_name,
            'num_samples': num_samples,
            'num_warmup': num_warmup,
            'num_chains': num_chains,
            'thinning': thinning,
            'chain_method': chain_method,
            'seed': seed,
            'parameter_names': parameter_names,
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

        print(f"NumPyro {self.algorithm_name.upper()} completed: {self.results.samples.shape[0]} samples")
        if self.results.acceptance_rate is not None:
            print(f"  Mean acceptance rate: {np.mean(self.results.acceptance_rate):.3f}")
        if self.results.is_divergent is not None:
            print(f"  Divergences: {int(np.sum(self.results.is_divergent))}")

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        summary['diagnostics'] = {
            'num_samples': int(self.results.samples.shape[0]) if self.results is not None else None,
            'mean_logdensity': (
                float(np.mean(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None else None
            ),
            'max_logdensity': (
                float(np.max(self.results.logdensity))
                if self.results is not None and self.results.logdensity is not None else None
            ),
            'mean_acceptance_rate': (
                float(np.mean(self.results.acceptance_rate))
                if self.results is not None and self.results.acceptance_rate is not None else None
            ),
            'n_divergent': (
                int(np.sum(self.results.is_divergent))
                if self.results is not None and self.results.is_divergent is not None else None
            ),
            'mean_integration_steps': (
                float(np.mean(self.results.num_integration_steps))
                if self.results is not None and self.results.num_integration_steps is not None else None
            ),
        }
        if self.results is not None:
            summary['parameters'] = dict(self.results.warmup_parameters)
            summary['posterior'] = self._posterior_summary_from_matrix(
                self.results.samples,
                free_parameter_names,
            )
        return summary


class NumPyroHMCSampler(NumPyroSampler):
    @property
    def algorithm_name(self):
        return 'hmc'

    @property
    def kernel_class(self):
        return NumPyroHMC


class NumPyroNUTSSampler(NumPyroSampler):
    @property
    def algorithm_name(self):
        return 'nuts'

    @property
    def kernel_class(self):
        return NumPyroNUTS


class PymultinestSampler(Sampler):
    """
    Sampler class for pymultinest.
    """
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,          # Retrieval.prior_model_jax (generator)
                 output_directory,
                 retrieval_name,
                 ):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        os.makedirs(os.path.join(self.output_directory, "out_PMN/"),exist_ok=True)

        self.seed = -1
        self.analyzer = None

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
        pymultinest.run(
            LogLikelihood=self.log_likelihood_func,
            Prior=self.prior_func,
            n_dims=n_dims,
            **kwargs
        )

        if rank == 0:
            self.analyzer = pymultinest.Analyzer(outputfiles_basename=f"{self.retrieval_name}_", n_params=n_dims)
        return self.analyzer

    def pretty_print(self, free_parameter_names, prefix):
        statistics = self.analyzer.get_stats()
        with open(prefix + 'stats.json', 'w') as f:
            json.dump(statistics, f, indent=4)
        print("\nPyMultiNest Global Evidence:")  # ... (printing logic)
        for mode in statistics['modes']:
            print(f"  log Z: {mode['local log-evidence']:.2f} +- {mode['local log-evidence error']:.2f}")
        print(f"  Global log Z: {statistics['global evidence']:.2f} +- "
                f"{statistics['global evidence error']:.2f}")

        print("\nPyMultiNest Parameter Statistics:")
        for i, name in enumerate(free_parameter_names):
            param_stats = self.analyzer.get_mode_stats()['marginals'][i]
            print(f"  {name}: {param_stats['median']:.2e} "
                    f"[{param_stats['1sigma'][0]:.2e} - {param_stats['1sigma'][1]:.2e}]")

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.analyzer is None:
            return summary

        statistics = self.analyzer.get_stats()
        summary['evidence'] = {
            'ln_z': self._summary_scalar(statistics.get('global evidence')),
            'ln_z_err': self._summary_scalar(statistics.get('global evidence error')),
        }
        posterior = []
        marginals = statistics.get('marginals', [])
        names = free_parameter_names or []
        for index, marginal in enumerate(marginals):
            name = names[index] if index < len(names) else f'parameter_{index}'
            one_sigma = marginal.get('1sigma', [None, None])
            posterior.append({
                'name': name,
                'median': self._summary_scalar(marginal.get('median')),
                'lower_1sigma': self._summary_scalar(one_sigma[0]),
                'upper_1sigma': self._summary_scalar(one_sigma[1]),
            })
        summary['posterior'] = posterior
        return summary

class UltranestSampler(Sampler):
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,           # Retrieval.prior_model_jax (generator)
                 output_directory,
                 retrieval_name,
                 enable_step_sampling = False
                 ):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        if ultranest is None:
            raise ImportError("Ultranest module is not installed. This sampler cannot be used.")

        self.enable_step_sampling = enable_step_sampling

        self.results = None

    def run_sampler(self, parameter_names, **kwargs):
        if ultranest is None:  # Should have been caught by __init__
            warnings.warn("Ultranest module not found, cannot run UltranestSampler.", RuntimeWarning)
            return None
        warnings.warn("Ultranest mode is still in development.", UserWarning)
        log_dir = os.path.join(self.output_directory, "out_" +
                               self.retrieval_name + "_ultranest")
        os.makedirs(log_dir, exist_ok=True)

        reactive_kwargs = {
            'derived_param_names': kwargs.pop('derived_param_names', []),
            'wrapped_params': kwargs.pop('wrapped_params', None),
            'resume': kwargs.pop('resume', 'subfolder'),
            'run_num': kwargs.pop('run_num', None),
            'log_dir': kwargs.pop('log_dir', log_dir),
            'num_test_samples': kwargs.pop('num_test_samples', 2),
            'draw_multiple': kwargs.pop('draw_multiple', True),
            'num_bootstraps': kwargs.pop('num_bootstraps', 30),
            'vectorized': kwargs.pop('vectorized', False),
            'ndraw_min': kwargs.pop('ndraw_min', 128),
            'ndraw_max': kwargs.pop('ndraw_max', 65536),
            'storage_backend': kwargs.pop('storage_backend', 'hdf5'),
            'warmstart_max_tau': kwargs.pop('warmstart_max_tau', -1),
        }

        self.sampler = ultranest.ReactiveNestedSampler(  # Renamed to avoid conflict with base class name
            param_names=parameter_names,
            loglike=self.log_likelihood_func,
            transform=self.prior_func,
            **reactive_kwargs
        )
        if self.enable_step_sampling:
            try:
                from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
                step_sampler_kwargs = {
                    'nsteps': kwargs.pop('nsteps', 100),
                    'generate_direction': kwargs.pop('generate_direction', generate_mixture_random_direction),
                    'scale': kwargs.pop('scale', 1.0),
                    'check_nsteps': kwargs.pop('check_nsteps', 'move-distance'),
                    'adaptive_nsteps': kwargs.pop('adaptive_nsteps', False),
                    'max_nsteps': kwargs.pop('max_nsteps', 1000),
                    'region_filter': kwargs.pop('region_filter', False),
                    'log': kwargs.pop('log', False),
                }
                self.sampler.stepsampler = SliceSampler(
                    **step_sampler_kwargs
                )
            except ImportError:
                warnings.warn("Could not import SliceSampler. Step sampler disabled.", RuntimeWarning)

        run_kwargs = {
            'update_interval_volume_fraction': kwargs.pop('update_interval_volume_fraction', 0.8),
            'update_interval_ncall': kwargs.pop('update_interval_ncall', None),
            'log_interval': kwargs.pop('log_interval', None),
            'show_status': kwargs.pop('show_status', True),
            'viz_callback': kwargs.pop('viz_callback', 'auto'),
            'dlogz': kwargs.pop('dlogz', 0.5),
            'dKL': kwargs.pop('dKL', 0.5),
            'frac_remain': kwargs.pop('frac_remain', 0.01),
            'Lepsilon': kwargs.pop('Lepsilon', 0.001),
            'min_ess': kwargs.pop('min_ess', 400),
            'max_iters': kwargs.pop('max_iters', None),
            'max_ncalls': kwargs.pop('max_ncalls', None),
            'max_num_improvement_loops': kwargs.pop('max_num_improvement_loops', -1),
            'min_num_live_points': kwargs.pop('min_num_live_points', 400),
            'cluster_num_live_points': kwargs.pop('cluster_num_live_points', 40),
            'insertion_test_zscore_threshold': kwargs.pop('insertion_test_zscore_threshold', 4),
            'insertion_test_window': kwargs.pop('insertion_test_window', 10),
            'region_class': kwargs.pop('region_class', MLFriends),
            'widen_before_initial_plateau_num_warn': kwargs.pop('widen_before_initial_plateau_num_warn', 10000),
            'widen_before_initial_plateau_num_max': kwargs.pop('widen_before_initial_plateau_num_max', 50000),
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

        summary['evidence'] = {
            'ln_z': self._summary_scalar(self.results.get('logz')),
            'ln_z_err': self._summary_scalar(self.results.get('logzerr')),
        }
        summary['diagnostics'] = {
            'ess': self._summary_scalar(self.results.get('ess')),
            'ncall': self._summary_scalar(self.results.get('ncall')),
            'samples': self._summary_scalar(self.results.get('samples').shape[0])
            if self.results.get('samples', None) is not None else None,
        }
        summary['posterior'] = self._posterior_summary_from_matrix(
            self.results.get('samples'),
            free_parameter_names,
        )
        if not summary['posterior'] and isinstance(self.results.get('weighted_samples'), dict):
            summary['posterior'] = self._posterior_summary_from_matrix(
                self.results['weighted_samples'].get('points'),
                free_parameter_names,
            )
        return summary


class JAXNSSampler(Sampler):
    """
    Sampler class for JAXNS.
    """
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,          # Retrieval.prior_model_jax (generator)
                 output_directory,
                 retrieval_name,
                 ):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        if jaxns is None:
            raise ImportError("JAXNS module or its JAX dependencies are not installed. This sampler cannot be used.")
        os.makedirs(os.path.join(self.output_directory, "out_JAXNS/"), exist_ok=True)
        self.jaxns_parameters = {}

        self.seed = -1
        self.sampler = None
        self.results = None

    def run_sampler(self, jit=True, parameters={}, **jaxns_kwargs):
        """
        Run the JAXNS sampler.
        """
        if jaxns is None:  # Should have been caught by __init__
            warnings.warn("JAXNS module not found, cannot run JAXNSSampler.", RuntimeWarning)
            return None

        self.jaxns_parameters = {
            'num_live_points': jaxns_kwargs.pop('num_live_points', 400),
            'max_samples': jaxns_kwargs.pop('max_samples', 1e7),
            's': jaxns_kwargs.pop('s', 4),
            'k': jaxns_kwargs.pop('k', 0),
            'c': jaxns_kwargs.pop('c', 200),
            'devices': jaxns_kwargs.pop('devices', jax.devices()),
            'difficult_model': jaxns_kwargs.pop('difficult_model', False),
            'parameter_estimation': jaxns_kwargs.pop('parameter_estimation', False),
            'shell_fraction': jaxns_kwargs.pop('shell_fraction', 0.5),
            'gradient_guided': jaxns_kwargs.pop('gradient_guided', False),
            'init_efficiency_threshold': jaxns_kwargs.pop('init_efficiency_threshold', 0.05),
            'verbose': jaxns_kwargs.pop('verbose', False),
        }

        self.seed = jaxns_kwargs.pop('seed', 42654)
        # The prior_func is expected to be the prior_model (a generator)

        jaxified_likelihood = self.log_likelihood_func

        model = Model(prior_model=self.prior_func, log_likelihood=jaxified_likelihood)

        self.sampler = NestedSampler(
            model,
            **self.jaxns_parameters
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")

        if jit:
            termination_reason, state = jax.jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(state=state, termination_reason=termination_reason)

        results_out = tree.map(np.asarray, self.results)
        save_results(
            results=results_out,
            save_file=os.path.join(
                self.output_directory,
                "out_JAXNS",
                self.retrieval_name + "_jaxns_results.json")
            )

        self.pretty_print_results()
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        self.results = load_results(os.path.join(
            self.output_directory,
            "out_JAXNS/",
            self.retrieval_name + "_jaxns_results.json"))
        return self.results

    def pretty_print_results(self):
        self.sampler.summary(self.results)

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.results is None:
            return summary

        posterior_sources = []
        if hasattr(self.results, 'samples'):
            posterior_sources.append(getattr(self.results, 'samples'))
        if hasattr(self.results, 'parametrised_samples'):
            posterior_sources.append(getattr(self.results, 'parametrised_samples'))

        for source in (self.results, getattr(self.results, '_asdict', lambda: {})()):
            if not source:
                continue
            ln_z = getattr(source, 'log_Z_mean', None) if not isinstance(source, dict) else source.get('log_Z_mean')
            ln_z_err = getattr(source, 'log_Z_uncert', None) if not isinstance(source, dict) else source.get('log_Z_uncert')
            ess = getattr(source, 'ESS', None) if not isinstance(source, dict) else source.get('ESS')
            h_mean = getattr(source, 'H_mean', None) if not isinstance(source, dict) else source.get('H_mean')
            if any(value is not None for value in (ln_z, ln_z_err, ess, h_mean)):
                summary['evidence'] = {
                    'ln_z': self._summary_scalar(ln_z),
                    'ln_z_err': self._summary_scalar(ln_z_err),
                }
                summary['diagnostics'] = {
                    'ess': self._summary_scalar(ess),
                    'h_mean': self._summary_scalar(h_mean),
                }
                break

        for posterior_source in posterior_sources:
            summary['posterior'] = self._posterior_summary_from_mapping(
                posterior_source,
                free_parameter_names=free_parameter_names,
            )
            if summary['posterior']:
                break

        return summary


class JAXNSShardedStaticNestedSampler(JAXNSSampler):
    """
    Sampler class for JAXNS.
    """
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,          # Retrieval.prior_model_jax (generator)
                 output_directory,
                 retrieval_name,
                 ):
        super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
        if jaxns is None:
            raise ImportError("JAXNS module or its JAX dependencies are not installed. This sampler cannot be used.")
        self.log_likelihood_func = log_likelihood_func
        self.prior_function = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name
        os.makedirs(os.path.join(self.output_directory, "out_JAXNS/"), exist_ok=True)
        self.jaxns_parameters = {}

        self.seed = -1
        self.sampler = None
        self.results = None

    def run_sampler(self, jit=True, **jaxns_kwargs):
        """
        Run the JAXNS sampler.
        """
        if jaxns is None:  # Should have been caught by __init__
            warnings.warn("JAXNS module not found, cannot run JAXNSSampler.", RuntimeWarning)
            return None

        self.jaxns_parameters = {
            'num_live_points': jaxns_kwargs.pop('num_live_points', 400),
            'sampler': jaxns_kwargs.pop('sampler', 400),
            'max_samples': jaxns_kwargs.pop('max_samples', 1e7),
            'devices': jaxns_kwargs.pop('devices', jax.devices()),
            'difficult_model': jaxns_kwargs.pop('difficult_model', False),
            'verbose': jaxns_kwargs.pop('verbose', False),
        }

        self.seed = jaxns_kwargs.pop('seed', 42654)
        # The prior_func is expected to be the prior_model (a generator)
        model = Model(prior_model=self.prior_func, log_likelihood=self.log_likelihood_func)

        # TODO Test Sharded parallel sampler
        # TODO test different sampler choices.
        self.sampler = ShardedStaticNestedSampler(
            model,
            sampler=UniformSampler,
            **self.jaxns_parameters
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")

        if jit:
            termination_reason, state = jax.jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(state=state, termination_reason=termination_reason)

        results_out = tree.map(np.asarray, self.results)
        save_results(
            results=results_out,
            save_file=os.path.join(
                self.output_directory,
                "out_JAXNS",
                self.retrieval_name + "_jaxns_results.json")
            )

        self.pretty_print_results()
        return self.results

    def get_results(self):
        return self.results

    def load_results(self):
        self.results = load_results(os.path.join(
            self.output_directory,
            "out_JAXNS/",
            self.retrieval_name + "_jaxns_results.json"))
        return self.results

    def pretty_print_results(self):
        self.sampler.summary(self.results)

