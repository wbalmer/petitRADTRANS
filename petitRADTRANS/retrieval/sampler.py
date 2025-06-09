import abc
import pymultinest
import os
import json
import warnings

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None
    comm = None
    rank = 0

# Ultranest imports
try:
    import ultranest
    from ultranest.mlfriends import RobustEllipsoidRegion
except ImportError:
    ultranest = None  # Mark ultranest as not available

# JAXNS imports
try:
    import jax
    from jax import random
    import jax.numpy as jnp
    import jaxns
    from jaxns import Model, NestedSampler, summary as jaxns_summary  # Renamed to avoid conflict
except ImportError:
    jaxns = None  # Flag that jaxns is not available


class Sampler(abc.ABC):
    """
    Abstract base class for samplers.
    """
    def __init__(self, log_likelihood_func, prior_func, retrieval_configuration, output_directory):
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.retrieval_configuration = retrieval_configuration
        self.output_directory = output_directory

    @abc.abstractmethod
    def run_sampler(self, **kwargs):
        pass


class PymultinestSampler(Sampler):
    """
    Sampler class for pymultinest.
    """
    def __init__(self,
                 log_likelihood_func,
                 prior_func,
                 retrieval_configuration,
                 output_directory,
                 sampling_efficiency=0.8,
                 const_efficiency_mode=False,
                 n_live_points=4000,
                 evidence_tolerance=0.5,
                 n_iter_before_update=50,
                 resume=False,
                 max_iter=0,
                 seed=-1):
        super().__init__(log_likelihood_func, prior_func, retrieval_configuration, output_directory)
        self.sampling_efficiency = sampling_efficiency
        self.const_efficiency_mode = const_efficiency_mode
        self.n_live_points = n_live_points
        self.evidence_tolerance = evidence_tolerance
        self.n_iter_before_update = n_iter_before_update
        self.resume = resume
        self.max_iter = max_iter
        self.seed = seed
        self.analyzer = None

    def run_sampler(self, **kwargs):
        n_params = len(self.retrieval_configuration.parameters)
        free_parameter_names = list(self.retrieval_configuration.parameters.keys())

        prefix = os.path.join(self.output_directory, 'out_PMN',
                              self.retrieval_configuration.retrieval_name + '_')
        os.makedirs(os.path.join(self.output_directory, 'out_PMN'), exist_ok=True)

        if len(prefix) > 100 - 21 and rank == 0:
            warnings.warn("The path to the output files is very long. "
                          "This can cause problems for MultiNest...", RuntimeWarning)

        if self.retrieval_configuration.run_mode == 'retrieval':
            if rank == 0:
                print(f"Starting PyMultiNest retrieval: {self.retrieval_configuration.retrieval_name}")
                print(f"Output directory: {os.path.join(self.output_directory, 'out_PMN')}")
                with open(prefix + 'params.json', 'w') as f:
                    json.dump(free_parameter_names, f, indent=4)

            pymultinest.run(
                LogLikelihood=self.log_likelihood_func,
                Prior=self.prior_func,
                n_dims=n_params,
                outputfiles_basename=prefix,
                resume=self.resume,
                verbose=True,
                sampling_efficiency=self.sampling_efficiency,
                const_efficiency_mode=self.const_efficiency_mode,
                n_live_points=self.n_live_points,
                evidence_tolerance=self.evidence_tolerance,
                n_iter_before_update=self.n_iter_before_update,
                max_iter=self.max_iter,
                seed=self.seed,
                init_MPI=False,
                use_MPI=MPI is not None
            )

        if rank == 0:
            try:
                self.analyzer = pymultinest.Analyzer(outputfiles_basename=prefix, n_params=n_params)
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
                return self.analyzer
            except Exception as e:
                warnings.warn(f"PyMultiNest analysis failed: {e}", RuntimeWarning)
                return None
        else:  # Other ranks
            if comm is not None: comm.Barrier()
            try:
                self.analyzer = pymultinest.Analyzer(outputfiles_basename=prefix, n_params=n_params)
            except Exception: self.analyzer = None
            return self.analyzer
        return None


class UltranestSampler(Sampler):
    def __init__(self,
                 log_likelihood_func,
                 prior_func,
                 retrieval_configuration,
                 output_directory,
                 n_live_points=400,
                 log_z_convergence=0.5,
                 step_sampler=False,
                 warmstart_max_tau=0.5,
                 resume='auto',
                 max_iters=0,
                 frac_remain=0.1,
                 l_epsilon=0.3):
        super().__init__(log_likelihood_func, prior_func, retrieval_configuration, output_directory)
        if ultranest is None:
            raise ImportError("Ultranest module is not installed. This sampler cannot be used.")
        self.n_live_points = n_live_points
        self.log_z_convergence = log_z_convergence
        self.step_sampler_enabled = step_sampler
        self.warmstart_max_tau = warmstart_max_tau
        self.resume_mode = resume
        self.max_iters = max_iters
        self.frac_remain = frac_remain
        self.l_epsilon = l_epsilon
        self.results = None

    def run_sampler(self, **kwargs):
        if ultranest is None:  # Should have been caught by __init__
            warnings.warn("Ultranest module not found, cannot run UltranestSampler.", RuntimeWarning)
            return None
        warnings.warn("Ultranest mode is still in development.", UserWarning)
        free_parameter_names = list(self.retrieval_configuration.parameters.keys())
        log_dir = os.path.join(self.output_directory, "out_" +
                               self.retrieval_configuration.retrieval_name + "_ultranest")
        os.makedirs(log_dir, exist_ok=True)

        un_sampler = ultranest.ReactiveNestedSampler(  # Renamed to avoid conflict with base class name
            param_names=free_parameter_names,
            loglike=self.log_likelihood_func,
            transform=self.prior_func,
            log_dir=log_dir,
            resume=self.resume_mode,
            warmstart_max_tau=self.warmstart_max_tau
        )
        if self.step_sampler_enabled:
            try:
                from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
                un_sampler.stepsampler = SliceSampler(
                    nsteps=100,
                    generate_direction=generate_mixture_random_direction,
                )
            except ImportError:
                warnings.warn("Could not import SliceSampler. Step sampler disabled.", RuntimeWarning)

        run_kwargs = {
            'min_num_live_points': self.n_live_points,
            'dlogz': self.log_z_convergence,
            'frac_remain': self.frac_remain,
            'Lepsilon': self.l_epsilon,
            'region_class': RobustEllipsoidRegion
        }
        run_kwargs['max_iters'] = self.max_iters if self.max_iters > 0 else None

        un_sampler.run(**run_kwargs)
        self.results = un_sampler.results
        if self.results:
            un_sampler.print_results()
        return self.results


class JAXNSSampler(Sampler):
    """
    Sampler class for JAXNS.
    """
    def __init__(self,
                 log_likelihood_func,  # Retrieval.log_likelihood_jax
                 prior_func,          # Retrieval.prior_model_jax (generator)
                 retrieval_configuration,
                 output_directory,
                 max_samples=1e7,
                 num_live_points=400,
                 parameter_estimation=True,
                 num_parallel_workers=1,  # Set to None for automatic core detection by jaxns?
                 verbose=False,
                 seed=42654):
        super().__init__(log_likelihood_func, prior_func, retrieval_configuration, output_directory)
        if jaxns is None:
            raise ImportError("JAXNS module or its JAX dependencies are not installed. This sampler cannot be used.")

        self.max_samples = int(max_samples)  # Ensure it's an int
        self.num_live_points = num_live_points
        self.parameter_estimation = parameter_estimation
        self.num_parallel_workers = num_parallel_workers
        self.verbose = verbose
        self.seed = seed
        self.results = None  # To store JAXNS results

    def run_sampler(self, **kwargs):
        """
        Run the JAXNS sampler.
        """
        if jaxns is None:  # Should have been caught by __init__
            warnings.warn("JAXNS module not found, cannot run JAXNSSampler.", RuntimeWarning)
            return None

        # The prior_func is expected to be the prior_model (a generator)
        model = Model(prior_model=self.prior_func, log_likelihood=self.log_likelihood_func)

        ns = NestedSampler(
            model,
            max_samples=self.max_samples,
            num_live_points=self.num_live_points,
            parameter_estimation=self.parameter_estimation,
            num_parallel_workers=self.num_parallel_workers,
            verbose=self.verbose
        )

        jax_random_key = random.PRNGKey(self.seed)

        print(f"Starting JAXNS retrieval: {self.retrieval_configuration.retrieval_name}")
        # Consider output directory for JAXNS if it produces files, for now, it's in-memory
        # output_path_jaxns = os.path.join(self.output_directory, 'out_JAXNS',
        #                                  self.retrieval_configuration.retrieval_name)
        # os.makedirs(output_path_jaxns, exist_ok=True)
        # print(f"JAXNS results (if saved) will be in: {output_path_jaxns}")

        termination_reason, state = ns(jax_random_key)
        self.results = ns.to_results(state=state, termination_reason=termination_reason)

        if self.results:
            print("\nJAXNS Summary:")
            jaxns_summary(self.results)
            # Potentially save results here if desired
            # e.g., jaxns.save_results(self.results, os.path.join(output_path_jaxns, "results.json"))
            # from jaxns.utils import save_pytree
            # save_pytree(self.results, os.path.join(output_path_jaxns, "results.pytree"))

        return self.results
