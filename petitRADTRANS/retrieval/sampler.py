import abc
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
except ImportError:
    pymultinest = None

try:
    import ultranest
    from ultranest.mlfriends import RobustEllipsoidRegion, MLFriends
except ImportError:
    ultranest = None  # Mark ultranest as not available

try:
    import jax
    from jax import random, tree
    import jax.numpy as jnp
    import jaxns
    from jaxns import Model, NestedSampler, save_results, load_results, summary as jaxns_summary  # Renamed to avoid conflict
    from jaxns.nested_samplers import ShardedStaticNestedSampler
    from jaxns.samplers import UniformSampler
except ImportError:
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
        os.makedirs(os.path.join(self.output_directory, "out_PMN/"))

        self.seed = -1
        self.analyzer = None

    def run_sampler(self, ndim, **kwargs):
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
            ndim=ndim
            **kwargs
        )

        if rank == 0:
            self.analyzer = pymultinest.Analyzer(outputfiles_basename=f"{self.retrieval_name}_", n_params=ndim)
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
                               self.retrieval_configuration.retrieval_name + "_ultranest")
        os.makedirs(log_dir, exist_ok=True)

        reactive_kwargs = {
            'derived_param_names': kwargs.pop('derived_param_names', []),
            'wrapped_params': kwargs.pop('wrapped_params', None),
            'resume': kwargs.pop('resume', 'subfolder'),
            'run_num': kwargs.pop('run_num', None),
            'log_dir': kwargs.pop('log_dir', None),
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
            log_dir=log_dir,
            **reactive_kwargs
        )
        if self.enable_step_sampling:
            try:
                from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
                step_sampler_kwargs = {
                    'nsteps':kwargs.pop('nsteps',100), 
                    'generate_direction':kwargs.pop('generate_direction',generate_mixture_random_direction),
                    'scale':kwargs.pop('scale',1.0),
                    'check_nsteps':kwargs.pop('check_nsteps','move-distance'),
                    'adaptive_nsteps':kwargs.pop('adaptive_nsteps',False),
                    'max_nsteps':kwargs.pop('max_nsteps',1000),
                    'region_filter':kwargs.pop('region_filter',False),
                    'log':kwargs.pop('log',False),
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
        run_kwargs['max_iters'] = self.max_iters if self.max_iters > 0 else None

        self.sampler.run(**run_kwargs)
        self.results = self.sampler.results
        if self.results:
            self.sampler.print_results()
        return self.results

    def get_results(self):
        return self.results

    def pretty_print(self):
        self.sampler.print_results()

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

    def run_sampler(self, jit=True, **jaxns_kwargs):
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
        model = Model(prior_model=self.prior_func, log_likelihood=self.log_likelihood_func)

        self.sampler = NestedSampler(
            model,
            **self.jaxns_parameters
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_configuration.retrieval_name}")

        if jit:
            termination_reason, state = jax.jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(state=state, termination_reason=termination_reason)

        results_out = tree.map(np.asarray, self.results)
        save_results(results_out, os.path.join(
            self.output_directory,
            "out_JAXNS",
            self.retrieval_name + "_jaxns_results.json"))

        self.sampler.save_results(
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


class JAXNSShardedStaticNestedSampler(JAXNSSampler):
    """
    Sampler class for JAXNS.
    """
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,          # Retrieval.prior_model_jax (generator)
                 retrieval_configuration,
                 output_directory,
                 retrieval_name,
                 ):
        super().__init__(log_likelihood_func, prior_func, retrieval_configuration, output_directory, retrieval_name)
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

        print(f"Starting JAXNS retrieval: {self.retrieval_configuration.retrieval_name}")

        if jit:
            termination_reason, state = jax.jit(self.sampler)(jax_random_key)
        else:
            termination_reason, state = self.sampler(jax_random_key)

        self.results = self.sampler.to_results(state=state, termination_reason=termination_reason)

        results_out = tree.map(np.asarray, self.results)
        save_results(results_out, os.path.join(
            self.output_directory,
            "out_JAXNS",
            self.retrieval_name + "_jaxns_results.json"))

        self.sampler.save_results(
            )
        self.pretty_print_results()
        return self.results

