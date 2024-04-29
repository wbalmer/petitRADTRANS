import copy
import copy as cp
import json
import logging
import os
import sys
import traceback
import warnings

import numpy as np
from scipy.stats import binned_statistic

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.__file_conversion import bin_species_exok
from petitRADTRANS._input_data_loader import get_opacity_input_file, get_resolving_power_string, join_species_all_info
from petitRADTRANS.chemistry.utils import mass_fractions2volume_mixing_ratios
from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from petitRADTRANS.math import running_mean
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.parameter import Parameter, RetrievalParameter
from petitRADTRANS.retrieval.utils import get_pymultinest_sample_dict
from petitRADTRANS.utils import flatten_object

# MPI Multiprocessing
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI = None
    rank = 0
    comm = None


class Retrieval:
    def __init__(
            self,
            configuration: RetrievalConfig,
            output_directory: str = os.getcwd(),
            use_mpi: bool = False,
            evaluate_sample_spectra: bool = False,
            ultranest: bool = False,
            sampling_efficiency: float = None,
            constant_efficiency_mode: str = None,
            n_live_points: int = None,
            resume: bool = False,
            corner_plot_names: list[str] = None,
            use_prt_plot_style: bool = True,
            test_plotting: bool = False,
            uncertainties_mode: str = "default",
            print_log_likelihood_for_debugging=False
    ):
        """
        This class implements the retrieval method using petitRADTRANS and pymultinest.
        A RetrievalConfig object is passed to this class to describe the retrieval data, parameters
        and priors. The run() method then uses pymultinest to sample the parameter space, producing
        posterior distributions for parameters and bayesian evidence for models.
        Various useful plotting functions have also been included, and can be run once the retrieval is
        complete.

        Args:
            configuration : RetrievalConfig
                A RetrievalConfig object that describes the retrieval to be run. This is the user
                facing class that must be setup for every retrieval.
            output_directory : Str
                The directory in which the output folders should be written
            evaluate_sample_spectra : Bool
                Produce plots and data files for random samples drawn from the outputs of pymultinest.
            ultranest : bool
                If true, use Ultranest sampling rather than pymultinest. Provides a more accurate evidence estimate,
                but is significantly slower.
            corner_plot_names : List(Str)
                List of additional retrieval names that should be included in the corner plotlib.
            use_prt_plot_style : Bool
                Use the petitRADTRANS plotting style as described in style.py. Recommended to
                turn this parameter to false if you want to use interactive plotting, or if the
                test_plotting parameter is True.
            test_plotting : Bool
                Only use when running locally. A boolean flag that will produce plots
                for each sample when pymultinest is run.
            uncertainties_mode : Str
                Uncertainties handling method during the retrieval.
                    - "default": the uncertainties are fixed.
                    - "optimize": automatically optimize for uncertainties, following Gibson et al. 2020
                      (https://doi.org/10.1093/mnras/staa228).
                    - "retrieve": uncertainties are scaled with a coefficient, which is retrieved.
                    - "retrieve_add": a fixed scalar is added to the uncertainties, and is retrieved.
            print_log_likelihood_for_debugging : bool
                If True, the current log likelihood of a forward model run will be printed to the console.
        """
        self.configuration = configuration

        if len(self.configuration.line_species) < 1:
            for data_name, d in self.configuration.data.items():
                if d.radtrans_object is not None:
                    if len(d.radtrans_object.line_species) < 1:
                        warnings.warn("there are no line species present in the given Radtrans object")
                        break
                else:
                    warnings.warn("there are no line species present in the run definition")
                    break

        # Maybe inherit from retrieval config class?
        # Could actually be merged with RetrievalConfig
        self.ultranest = ultranest
        self.use_mpi = use_mpi

        self.uncertainties_mode = uncertainties_mode

        self.print_log_likelihood_for_debugging = print_log_likelihood_for_debugging

        self.output_directory = output_directory

        self.corner_files = corner_plot_names
        if self.corner_files is None:
            self.corner_files = [self.configuration.retrieval_name]

        # Plotting variables
        self.best_fit_spectra = {}
        self.best_fit_parameters = {}
        self.chi2 = None
        self.posterior_sample_spectra = {}
        self.test_plotting = test_plotting
        self.pt_plot_mode = False
        self.evaluate_sample_spectra = evaluate_sample_spectra

        # Pymultinest stuff
        self.ultranest = ultranest
        self.sampling_efficiency = sampling_efficiency
        self.constant_efficiency_mode = constant_efficiency_mode
        self.n_live_points = n_live_points
        self.resume = resume
        self.analyzer = None

        self.samples = {}  #: The samples produced by pymultinest.
        self.param_dictionary = {}
        # Set up pretty plotting
        if use_prt_plot_style:
            # import petitRADTRANS.retrieval.plot_style  # commented to avoid breaking of mpl
            pass

        self.prt_plot_style = use_prt_plot_style

        # Path to input opacities
        self.path = petitradtrans_config_parser.get_input_data_path()

        # Setup Directories
        if not os.path.isdir(os.path.join(self.output_directory, 'out_PMN')):
            os.makedirs(
                os.path.join(self.output_directory, 'out_PMN'),
                exist_ok=True
            )

        if not os.path.isdir(os.path.join(self.output_directory, 'evaluate_' + self.configuration.retrieval_name)):
            os.makedirs(
                os.path.join(self.output_directory, 'evaluate_' + self.configuration.retrieval_name),
                exist_ok=True
            )

        # Setup pRT Objects for each data structure.
        self.setup_data()

        try:
            self.generate_retrieval_summary()
        except (ValueError, FileNotFoundError) as e:  # TODO check if ValueError was expected here
            print(f"Could not generate summary file! Error was: {str(e)}")

    def _check_errors(self):
        data_are_valid = self._data_are_valid()

        if not data_are_valid:
            warnings.warn("Data may not be suitable for retrievals due to invalid values")

        self._error_check_model_function()

    def _data_are_valid(self, data=None):
        tested_attributes = ['wavelengths', 'spectrum', 'uncertainties']
        valid = True

        if data is None:
            data = self.configuration.data

        if isinstance(data, dict):
            for name, data_obj in data.items():
                print(f"Testing data '{name}':")

                for tested_attribute in tested_attributes:
                    print(f" {tested_attribute}:")
                    tested_attribute = data_obj.__getattribute__(tested_attribute)
                    valid = valid and self._data_are_valid(tested_attribute)

                    if valid:
                        print("  OK (no NaN, infinite, or negative value detected)")
        elif isinstance(data, np.ndarray):
            if data.dtype == 'O':
                for d in data:
                    valid = valid and self._data_are_valid(d)
            else:
                if np.any(np.isnan(data)):
                    warnings.warn(f"NaN detected ({np.nonzero(np.isnan(data))[0].size} / {data.size})")
                    valid = False

                if np.any(~np.isfinite(data)):
                    warnings.warn(f"Infinite value detected ({np.nonzero(~np.isfinite(data))[0].size} / {data.size})")
                    valid = False

                if np.any(np.less(data, 0)):
                    warnings.warn(f"Negative value detected ({np.nonzero(np.less(data))[0].size} / {data.size})\n"
                                  "Make sure that this makes sense for the data set you are considering.")
                    valid = False
        else:
            self._data_are_valid(np.array(data))

        return valid

    def run(self,
            sampling_efficiency=0.8,
            const_efficiency_mode=False,
            n_live_points=4000,
            log_z_convergence=0.5,
            step_sampler=False,
            warmstart_max_tau=0.5,
            n_iter_before_update=50,
            resume=False,
            max_iters=0,
            frac_remain=0.1,
            l_epsilon=0.3,
            error_checking=True,
            force_serial_error_checking=False,
            seed=-1):
        """
        Run mode for the class. Uses pynultinest to sample parameter space
        and produce standard PMN outputs.

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
        """
        import pymultinest

        print(f"Starting retrieval {self.configuration.retrieval_name}")

        if MPI is None:
            print(
                "Unable to import mpi4py, using slow (single process) mode\n"
                "The mpi4py module is required for faster (multi-processes) retrievals, and is strongly recommended"
            )

        self.n_live_points = n_live_points
        self.sampling_efficiency = sampling_efficiency
        self.resume = resume
        self.constant_efficiency_mode = const_efficiency_mode

        if MPI is not None and comm is not None:
            any_error_checking = comm.allreduce(error_checking, op=MPI.LOR)
        else:
            any_error_checking = error_checking

        if error_checking:
            if force_serial_error_checking and MPI is not None and comm is not None:
                for i in range(comm.Get_size()):
                    if rank == i:
                        self._check_errors()
                    else:
                        print(f"rank {rank} waiting: serial error checking in progress...")

                    comm.barrier()
            else:
                self._check_errors()
        else:
            if any_error_checking:
                print(f"Process {rank} skipping error checking, "
                      f"waiting for other processes to complete error checking...")
            else:
                print("Error checking is turned off!! You might overwrite your retrieval output files!")

        if comm is not None:
            comm.barrier()

        if self.ultranest:
            self._run_ultranest(n_live_points=n_live_points,
                                log_z_convergence=log_z_convergence,
                                step_sampler=step_sampler,
                                warmstart_max_tau=warmstart_max_tau,
                                resume=resume,
                                max_iters=max_iters,
                                frac_remain=frac_remain,
                                l_epsilon=l_epsilon)
            return

        if const_efficiency_mode and sampling_efficiency > 0.1:
            warnings.warn(
                "Sampling efficiency should be <= 0.1 (0.05 recommended) if you're using constant efficiency mode!"
            )

        prefix = os.path.join(self.output_directory, 'out_PMN', self.configuration.retrieval_name + '_')

        if len(os.path.join(self.output_directory, 'out_PMN')) > 100:
            warnings.warn("old versions of MultiNest requires output directory names to be < 100 characters "
                          f"long (current directory was {len(os.path.join(self.output_directory, 'out_PMN'))} "
                          f"characters long); "
                          "using more characters may cause MultiNest failure or filename truncation if not using the "
                          "latest MultiNest version")

        # How many free parameters?
        n_params = 0
        free_parameter_names = []

        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                free_parameter_names.append(self.configuration.parameters[pp].name)
                n_params += 1

        if self.configuration.run_mode == 'retrieval':
            print("Starting retrieval: " + self.configuration.retrieval_name + '\n')

            with open(
                    os.path.join(self.output_directory, 'out_PMN', self.configuration.retrieval_name + '_params.json'),
                    'w'
            ) as f:
                json.dump(free_parameter_names, f)

            pymultinest.run(
                LogLikelihood=self.log_likelihood,
                Prior=self.prior,
                n_dims=n_params,
                n_params=None,
                n_clustering_params=None,
                wrapped_params=None,
                importance_nested_sampling=True,
                multimodal=True,
                const_efficiency_mode=const_efficiency_mode,
                n_live_points=n_live_points,
                evidence_tolerance=log_z_convergence,  # default value is 0.5
                sampling_efficiency=sampling_efficiency,  # default value is 0.8
                n_iter_before_update=n_iter_before_update,  # default value is 100
                null_log_evidence=-1e90,  # PyM default value
                max_modes=100,  # PyM default value
                mode_tolerance=-1e90,  # PyM default value
                outputfiles_basename=prefix,
                seed=seed,
                verbose=True,
                resume=resume,
                context=0,  # PyM default value (any additional information user wants to pass)
                write_output=True,  # PyM default value
                log_zero=-1e100,  # PyM default value
                max_iter=max_iters,
                init_MPI=False,  # PyM default value (should be False, because importing mpi4py initialises MPI already)
                dump_callback=None,  # PyM default value
                use_MPI=True,  # PyM default value
            )

        # Analyze the output data
        self.analyzer = pymultinest.Analyzer(n_params=n_params,
                                             outputfiles_basename=prefix)
        s = self.analyzer.get_stats()

        if rank == 0:
            self.generate_retrieval_summary(s)

            # Save the analysis
            with open(prefix + 'stats.json', 'w') as f:
                json.dump(s, f, indent=4)

            # Informative prints
            print('  marginal likelihood:')
            print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
            print('  parameters:')

            for p, m in zip(free_parameter_names, s['marginals']):
                lo, hi = m['1sigma']
                med = m['median']
                sigma = (hi - lo) / 2

                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)

                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))

    def _run_ultranest(self,
                       n_live_points,
                       log_z_convergence,
                       step_sampler,
                       warmstart_max_tau,
                       resume,
                       max_iters,
                       frac_remain,
                       l_epsilon):
        """
        Run mode for the class. Uses ultranest to sample parameter space
        and produce standard outputs.

        Args:
            n_live_points : Int
                The minimum number of live points to use for the Ultranest reactive sampler.
            log_z_convergence : float
                The convergence criterion on log z.
            step_sampler : bool
                Use a step sampler to improve the efficiency in ultranest.
            max_iters : int
                Maximum number of sampling iterations. If 0, will continue until convergence criteria are satisfied.
            frac_remain : float
                Ultranest convergence criterion. Halts integration if live point weights are below the specified value.
            l_epsilon : float
                Ultranest convergence criterion. Use with noisy likelihoods. Halts integration if live points are
                within l_epsilon.
            resume : bool
                Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
        """

        warnings.warn("ultranest mode is still in development. Proceed with caution")
        try:
            import ultranest as un
            from ultranest.mlfriends import RobustEllipsoidRegion
        except ImportError:
            logging.error("Could not import ultranest. Exiting.")
            sys.exit(1)
        if self.configuration.run_mode == 'retrieval':
            print("Starting retrieval: " + self.configuration.retrieval_name + '\n')
            # How many free parameters?
            n_params = 0
            free_parameter_names = []
            for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    free_parameter_names.append(self.configuration.parameters[pp].name)
                    n_params += 1

            if max_iters == 0:
                max_iters = None
            sampler = un.ReactiveNestedSampler(
                free_parameter_names,
                self.log_likelihood,
                self.prior_ultranest,
                log_dir=os.path.join(self.output_directory, "out_" + self.configuration.retrieval_name),
                warmstart_max_tau=warmstart_max_tau,
                resume=resume
            )
            if step_sampler:
                import ultranest.stepsampler

                sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=n_live_points,
                    adaptive_nsteps='move-distance',
                    generate_direction=ultranest.stepsampler.generate_mixture_random_direction
                )
            sampler.run(min_num_live_points=n_live_points,
                        dlogz=log_z_convergence,
                        max_iters=max_iters,
                        frac_remain=frac_remain,
                        l_epsilon=l_epsilon,
                        region_class=RobustEllipsoidRegion)
            sampler.print_results()
            sampler.plot_corner()

    def _rebin_opacities(self, resolution):
        species = []

        for line in self.configuration.line_species:
            _line = line.split('.', 1)[0]  # remove possible previous spectral info

            matches = get_opacity_input_file(
                path_input_data=self.path,
                category='correlated_k_opacities',
                species=join_species_all_info(_line, spectral_info=get_resolving_power_string(resolution)),
                find_all=True,
                search_online=False
            )

            if len(matches) == 0:
                species.append(line)

        # If not, setup low-res c-k tables
        if len(species) > 0:
            #
            if rank == 0:
                bin_species_exok(species, resolution)

            if comm is not None:
                comm.barrier()

    @classmethod
    def from_data(cls, data: dict[str, Data], retrieved_parameters: dict[str, dict[str]],
                  retrieval_name: str = "retrieval_name",
                  run_mode="retrieval", amr=False, output_directory: str = "", use_mpi: bool = False,
                  evaluate_sample_spectra: bool = False, ultranest: bool = False,
                  corner_plot_names: list[str] = None, use_prt_plot_style: bool = True, test_plotting: bool = False,
                  uncertainties_mode: str = "default",
                  scattering_in_emission: bool = False, pressures: np.ndarray = None
                  ):
        """Instantiate a Retrieval object with a dictionary of Data objects.
        Intended to be used in couple with the SpectralModel.init_data function.

        The RetrievalConfig object is automatically generated. No fixed parameters will be used, those must be stored
        in their respective Data.model_generating_function. This is automatically done when using the
        SpectralModel.init_data function.

        Args:
            data : Dict
                A dictionary with data names as keys and Data objects as values.
            retrieved_parameters : Dict
                A dictionary with retrieved parameter names as keys and dictionaries as values. Those sub-dictionaries
                must have keys 'prior_parameters' and 'prior_type'. This can also be a list of RetrievalParameter
                objects.
            retrieval_name : Str
                Name of this retrieval. Make it informative so that you can keep track of the outputs!
            run_mode : Str
                Can be either 'retrieval', which runs the retrieval normally using pymultinest,
                or 'evaluate', which produces plots from the best fit parameters stored in the
                output post_equal_weights file.
            amr : Bool
                Use an adaptive high resolution pressure grid around the location of cloud condensation.
                This will increase the size of the pressure grid by a constant factor that can be adjusted
                in the setup_pres function.
            output_directory : Str
                The directory in which the output folders should be written
            evaluate_sample_spectra : Bool
                Produce plots and data files for random samples drawn from the outputs of pymultinest.
            ultranest : bool
                If true, use Ultranest sampling rather than pymultinest. Provides a more accurate evidence estimate,
                but is significantly slower.
            corner_plot_names : List(Str)
                List of additional retrieval names that should be included in the corner plotlib.
            use_prt_plot_style : Bool
                Use the petitRADTRANS plotting style as described in style.py. Recommended to
                turn this parameter to false if you want to use interactive plotting, or if the
                test_plotting parameter is True.
            test_plotting : Bool
                Only use when running locally. A boolean flag that will produce plots
                for each sample when pymultinest is run.
            uncertainties_mode : Str
                Uncertainties handling method during the retrieval.
                    - "default": the uncertainties are fixed.
                    - "optimize": automatically optimize for uncertainties, following Gibson et al. 2020
                      (https://doi.org/10.1093/mnras/staa228).
                    - "retrieve": uncertainties are scaled with a coefficient, which is retrieved.
                    - "retrieve_add": a fixed scalar is added to the uncertainties, and is retrieved.
            scattering_in_emission : Bool
                If using emission spectra, turn scattering on or off.
            pressures : numpy.ndarray
                A log-spaced array of pressures over which to retrieve. 100 points is standard, between
                10^-6 and 10^3.

        Returns:
            An Retrieval object instance.
        """
        if pressures is None:
            pressures = np.ones(1)

        # Instantiate an empty RetrievalConfig
        retrieval_configuration = RetrievalConfig(
            retrieval_name=retrieval_name,
            run_mode=run_mode,
            amr=amr,
            scattering_in_emission=scattering_in_emission,  # scattering is not necessary when using SpectralModels
            pressures=pressures  # pressures is not necessary when using SpectralModels
        )

        # Convert retrieved parameters ot RetrievalParameters
        if isinstance(retrieved_parameters, dict):
            retrieved_parameters = RetrievalParameter.from_dict(retrieved_parameters)

        # Add retrieved parameters to the RetrievalConfig
        for parameter in retrieved_parameters:
            if not hasattr(parameter, 'prior_function'):
                raise AttributeError(
                    f"'{type(parameter)}' object has no attribute 'prior_function': "
                    f"usage of dictionary or a '{type(RetrievalParameter)}' instance is recommended"
                )

            retrieval_configuration.add_parameter(
                name=parameter.name,
                free=True,  # fixed parameters must be stored within the Data.model_generating_function
                value=None,
                transform_prior_cube_coordinate=parameter.prior_function
            )

        # Add the data to the RetrievalConfig
        for data_name, d in data.items():
            retrieval_configuration.data[data_name] = d

        # Instantiate the new Retrieval object
        return cls(
            configuration=retrieval_configuration,
            output_directory=output_directory,
            use_mpi=use_mpi,
            evaluate_sample_spectra=evaluate_sample_spectra,
            ultranest=ultranest,
            corner_plot_names=corner_plot_names,
            use_prt_plot_style=use_prt_plot_style,
            test_plotting=test_plotting,
            uncertainties_mode=uncertainties_mode
        )

    def generate_retrieval_summary(self, stats=None):
        """
        This function produces a human-readable text file describing the retrieval.
        It includes all the fixed and free parameters, the limits of the priors (if uniform),
        a description of the data used, and if the retrieval is complete, a summary of the
        best fit parameters and model evidence.

        Args:
            stats : dict
                A Pymultinest stats dictionary, from Analyzer.get_stats().
                This contains the evidence and best fit parameters.
        """
        with open(os.path.join(
                self.output_directory,
                "evaluate_" + self.configuration.retrieval_name,
                self.configuration.retrieval_name + "_ret_summary.txt"
        ), "w+") as summary:
            from datetime import datetime
            summary.write(self.configuration.retrieval_name + '\n')
            summary.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + '\n')
            summary.write(self.output_directory + '\n')
            summary.write(f"Live points: {self.n_live_points}\n\n")
            summary.write("Fixed Parameters\n")

            for key, value in self.configuration.parameters.items():
                if key in ['pressure_simple', 'pressure_width', 'pressure_scaling']:
                    continue

                if not value.is_free_parameter:
                    if isinstance(value.value, float):
                        summary.write(f"    {key} = {value.value:.3f}\n")
                    else:
                        summary.write(f"    {key} = {value.value}\n")

            summary.write('\n')
            summary.write("Free Parameters, Prior^-1(0), Prior^-1(1)\n")

            for key, value in self.configuration.parameters.items():
                if value.is_free_parameter:
                    low = value.transform_prior_cube_coordinate(0.0000001)
                    high = value.transform_prior_cube_coordinate(0.9999999)

                    if value.corner_transform is not None:
                        low = value.corner_transform(low)
                        high = value.corner_transform(high)
                    summary.write(f"    {key} = {low:3f}, {high:3f}\n")

            summary.write('\n')
            summary.write("Data\n")

            for name, dd in self.configuration.data.items():
                summary.write(name + '\n')

                if dd.path_to_observations is not None:
                    summary.write("    " + dd.path_to_observations + '\n')

                if dd.model_generating_function is not None:
                    summary.write(f"    Model Function = {dd.model_generating_function.__name__}\n")

                if dd.scale:
                    summary.write(f"    scale factor = {dd.scale_factor:.2f}\n")

                if dd.scale_err:
                    summary.write(f"    scale err factor = {dd.scale_factor:.2f}\n")

                if dd.offset_bool:
                    summary.write("    offset = True\n")

                if dd.system_distance is not None:
                    summary.write(f"    distance = {dd.system_distance}\n")

                if dd.data_resolution is not None:
                    summary.write(f"    data resolution = {dd.data_resolution}\n")

                if dd.model_resolution is not None:
                    summary.write(f"    model resolution = {dd.model_resolution}\n")

                if dd.external_radtrans_reference is not None:
                    summary.write(f"    external_pRT_reference = {dd.external_radtrans_reference}\n")

                if dd.photometry:
                    summary.write(f"    photometric width = {dd.photometry_range[0]:.4f}"
                                  + f"--{dd.photometry_range[1]:.4f} um\n")
                    summary.write("    Photometric transform function = "
                                  + dd.photometric_transformation_function.__name__ + '\n')

            summary.write('\n')

            if stats is not None:
                summary.write("Multinest Outputs\n")
                summary.write('  marginal evidence:\n')
                summary.write('    log Z = %.1f +- %.1f\n' %
                              (stats['global evidence'] / np.log(10), stats['global evidence error'] / np.log(10)))
                summary.write('    ln Z = %.1f +- %.1f\n' % (stats['global evidence'],
                                                             stats['global evidence error']))
                summary.write("  Statistical Fit Parameters\n")

                free_params = []
                transforms = []

                for key, value in self.configuration.parameters.items():
                    if value.is_free_parameter:
                        free_params.append(key)
                        transforms.append(value.corner_transform)

                for param, marginals, transform in zip(free_params, stats['marginals'], transforms):
                    lo, hi = marginals['1sigma']
                    med = marginals['median']
                    if transform is not None:
                        lo = transform(lo)
                        hi = transform(hi)
                        med = transform(med)
                    sigma = (hi - lo) / 2
                    if sigma == 0:
                        i = 3
                    else:
                        i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                    fmt = '%%.%df' % i
                    fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                    summary.write(fmts % (param, med, sigma) + '\n')
                summary.write('\n')

            if self.configuration.run_mode == 'evaluate':
                summary.write("Best Fit Parameters\n")
                self.samples, self.param_dictionary = self.get_samples(
                    ultranest=self.ultranest,
                    names=self.corner_files,
                    output_directory=self.output_directory,
                    ret_names=None
                )
                samples_use = self.samples[self.configuration.retrieval_name]
                parameters_read = self.param_dictionary[self.configuration.retrieval_name]
                # Get best-fit index
                log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.get_max_likelihood_params(samples_use[:-1, best_fit_index], parameters_read)
                chi2_wlen = self.get_reduced_chi2(
                    sample=samples_use[:, best_fit_index],
                    subtract_n_parameters=False,
                    verbose=True,
                    show_chi2=True
                )
                chi2_d_o_f = self.get_reduced_chi2(
                    sample=samples_use[:, best_fit_index],
                    subtract_n_parameters=True,
                    verbose=True,
                    show_chi2=False  # show chi2 only once
                )

                # Get best-fit index
                summary.write(f"    𝛘^2/n_wlen = {chi2_wlen:.2f}\n")
                summary.write(f"    𝛘^2/DoF = {chi2_d_o_f:.2f}\n")
                for key, value in self.best_fit_parameters.items():
                    if key in ['pressure_simple', 'pressure_width', 'pressure_scaling', 'FstarWlenMicron']:
                        continue

                    out = value.value

                    if self.configuration.parameters[key].corner_transform is not None:
                        out = self.configuration.parameters[key].corner_transform(out)

                    if out is None:
                        continue

                    if isinstance(out, float):
                        summary.write(f"    {key} = {out:.3e}\n")
                    else:
                        summary.write(f"    {key} = {out}\n")

    def get_base_figure_name(self):
        return os.path.join(
            self.output_directory,
            'evaluate_' + self.configuration.retrieval_name,
            self.configuration.retrieval_name
        )

    def setup_data(self, scaling=10, width=3):
        """
        Creates a pRT object for each data set that asks for a unique object.
        Checks if there are low resolution c-k models from exo-k, and creates them if necessary.
        The scaling and width parameters adjust the AMR grid as described in RetrievalConfig.setup_pres
        and models.fixed_length_amr. It is recommended to keep the defaults.

        Args:
            scaling : int
                A multiplicative factor that determines the size of the full high resolution pressure grid,
                which will have length self.p_global.shape[0] * scaling.
            width : int
                The number of cells in the low pressure grid to replace with the high resolution grid.
        """
        for name, dd in self.configuration.data.items():
            if dd.radtrans_object is not None:
                print(f"Using provided Radtrans object for data '{name}'...")
                continue

            # Only create if there's no other data object using the same pRT object
            if dd.external_radtrans_reference is None:
                print(f"Setting up Radtrans object for data '{name}'...")

                if dd.line_opacity_mode == 'c-k' and dd.model_resolution is not None:
                    # Use ExoK to have low res models.
                    self._rebin_opacities(resolution=dd.model_resolution)

                    species = []

                    for spec in self.configuration.line_species:
                        spec = spec.split('.', 1)[0]  # remove possible previous spectral info

                        species.append(join_species_all_info(
                            spec,
                            spectral_info=get_resolving_power_string(dd.model_resolution)
                        ))
                else:
                    # Otherwise for 'lbl' or no model_resolution binning,
                    # we just use the default species.
                    species = copy.deepcopy(self.configuration.line_species)

                lbl_samp = None

                if dd.line_opacity_mode == 'lbl' and dd.model_resolution is not None:
                    lbl_samp = int(1e6 / dd.model_resolution)

                # Create random P-T profile to create RT arrays of the Radtrans object.
                if self.configuration.amr:
                    p = self.configuration._setup_pres(scaling, width)  # TODO this function shouldn't be protected
                else:
                    p = self.configuration.pressures

                # Set up the pRT objects for the given dataset
                rt_object = Radtrans(
                    pressures=p,
                    line_species=cp.copy(species),
                    rayleigh_species=cp.copy(self.configuration.rayleigh_species),
                    gas_continuum_contributors=cp.copy(self.configuration.continuum_opacities),
                    cloud_species=cp.copy(self.configuration.cloud_species),
                    line_opacity_mode=dd.line_opacity_mode,
                    wavelength_boundaries=dd.wavelength_boundaries,
                    scattering_in_emission=self.configuration.scattering_in_emission,
                    line_by_line_opacity_sampling=lbl_samp
                )

                dd.radtrans_object = rt_object

    def _error_check_model_function(self):
        free_params = []

        for key, val in self.configuration.parameters.items():
            if val.is_free_parameter:
                free_params.append(key)

        cube = np.ones(len(free_params)) * 0.5
        self.prior(cube)

        i_p = 0  # parameter count
        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                self.configuration.parameters[pp].set_param(cube[i_p])
                i_p += 1

        for name, data in self.configuration.data.items():
            print(f"Testing model function for data '{name}'...")
            message = None
            wlen = None
            model = None
            exc_info = None

            try:
                use_obj = data.radtrans_object

                if data.external_radtrans_reference is not None:
                    use_obj = self.configuration.data[data.external_radtrans_reference].radtrans_object

                model_returned_values = data.model_generating_function(
                    use_obj,
                    self.configuration.parameters,
                    False,
                    amr=self.configuration.amr
                )  # TODO the generating function should always return the same number of values

                if len(model_returned_values) == 3:  # handle case where beta is returned
                    wlen, model, _ = model_returned_values
                else:
                    wlen, model = model_returned_values
            except KeyError:
                exc_info = sys.exc_info()
                message = "and may be caused by an invalid parameter dictionary or invalid abundances"
            except ValueError:
                exc_info = sys.exc_info()
                message = "and may be caused by erroneous calculations or invalid inputs"
            except IndexError:
                exc_info = sys.exc_info()
                message = "and may be caused by arrays with an incorrect shape"
            except ZeroDivisionError:
                exc_info = sys.exc_info()
                message = ""
            except TypeError:
                exc_info = sys.exc_info()
                message = "and may be caused by invalid inputs"
            finally:
                if message is not None:
                    message = "this error was directly caused by the above error " + message
                    traceback.print_exception(*exc_info)
                    raise RuntimeError(message)

            if wlen is None or model is None:
                raise ValueError("unable to compute a spectrum (output wavelengths and spectrum are both None), "
                                 "check your inputs and your model function")

        print("No errors detected in the model functions!")

    def prior(self, cube, ndim=0, nparams=0):
        """
        pyMultinest Prior function. Transforms unit hypercube into physical space.
        """
        i_p = 0

        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                cube[i_p] = self.configuration.parameters[pp].get_param_uniform(cube[i_p])
                i_p += 1

    def prior_ultranest(self, cube):
        """
        pyMultinest Prior function. Transforms unit hypercube into physical space.
        """
        params = cube.copy()
        i_p = 0
        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                params[i_p] = self.configuration.parameters[pp].get_param_uniform(cube[i_p])
                i_p += 1
        return params

    def get_data_model(self, cube, dd, retrieve_uncertainties=False):
        i_p = 0

        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                self.configuration.parameters[pp].set_param(cube[i_p])
                i_p += 1

        # Compute the model
        model_returned_values = dd.model_generating_function(
            dd.radtrans_object,
            self.configuration.parameters,
            self.pt_plot_mode,
            amr=self.configuration.amr
        )  # TODO the generating function should always return the same number of values
        beta = None

        if retrieve_uncertainties:
            if len(model_returned_values) == 4:
                wlen_model, spectrum_model, beta, additional_logl = model_returned_values
            else:
                wlen_model, spectrum_model, beta = model_returned_values
                additional_logl = 0.
        else:
            if len(model_returned_values) == 3:
                wlen_model, spectrum_model, additional_logl = model_returned_values
            else:
                wlen_model, spectrum_model = model_returned_values
                additional_logl = 0.

        if additional_logl is None:
            additional_logl = 0

        return wlen_model, spectrum_model, beta, additional_logl

    def log_likelihood(self, cube, ndim=0, nparam=0, log_l_per_datapoint_dict=None, return_model=False):
        """
        pyMultiNest required likelihood function.

        This function wraps the model computation and log-likelihood calculations
        for pyMultiNest to sample. If PT_plot_mode is True, it will return
        only the pressure and temperature arrays rather than the wavelength
        and flux. If run_mode is 'evaluate', it will save the provided sample to the
        best-fit spectrum file, and add it to the best_fit_specs dictionary.
        If evaluate_sample_spectra is true, it will store the spectrum in
        posterior_sample_specs.

        Args:
            cube : numpy.ndarray
                The transformed unit hypercube, providing the parameter values
                to be passed to the model_generating_function.
            ndim : int
                The number of dimensions of the problem
            nparam : int
                The number of parameters in the fit.
            log_l_per_datapoint_dict : dict
                Dictionary with instrument-entries. If provided, log likelihood
                per datapoint is appended to existing list.
            return_model : bool
                If True, return the generated model in addition to the log-likelihood.
                This is intended for debugging purposes, and should stay False in a normal retrieval.

        Returns:
            log_likelihood : float
                The (negative) log likelihood of the model given the data.
        """
        invalid_value = -1e99
        tiny = 1e-99
        log_likelihood = 0.
        log_prior = 0.

        retrieve_uncertainties = False
        beta_mode = "multiply"

        wavelengths_models = []
        spectrum_models = []
        additional_log_ls = []
        beta = 1.0

        atmospheric_model_column_fluxes = None

        if self.uncertainties_mode == "default":
            beta = 1.0
        elif self.uncertainties_mode == "optimize":
            warnings.warn("automatically optimizing for uncertainties, be sure of what you are doing...")
            beta = None
        elif self.uncertainties_mode == "retrieve":
            warnings.warn("retrieving uncertainties (multiply mode), be sure of what you are doing...")
            retrieve_uncertainties = True
            beta_mode = "multiply"
        elif self.uncertainties_mode == "retrieve_add":
            warnings.warn("retrieving uncertainties (add mode), be sure of what you are doing...")
            retrieve_uncertainties = True
            beta_mode = "add"
        else:
            raise ValueError(f"uncertainties mode must be 'default'|'optimize'|'retrieve'|'retrieve_add', "
                             f"but was '{self.uncertainties_mode}'")

        # Store per data-object
        per_datapoint = False

        if isinstance(log_l_per_datapoint_dict, dict):
            per_datapoint = True

        # Update model parameters with the retrieval cube drawn values
        i = 0  # free parameters enumerator

        for parameter in self.configuration.parameters:
            if self.configuration.parameters[parameter].is_free_parameter:
                self.configuration.parameters[parameter].set_param(cube[i])
                i += 1

        for data_name, data in self.configuration.data.items():
            if per_datapoint:
                # Keep logL's separate per data-object
                log_likelihood = 0.

            # Only calculate spectra within a given wlen range once if data.scale or data.scale_err:
            if data_name + "_scale_factor" in self.configuration.parameters:
                data.scale_factor = self.configuration.parameters[data_name + "_scale_factor"].value

            if data.offset_bool:
                data.offset = self.configuration.parameters[data_name + "_offset"].value

            if data_name + "_b" in self.configuration.parameters.keys():
                data.bval = self.configuration.parameters[data_name + "_b"].value

            if self.pt_plot_mode and data_name == self.configuration.plot_kwargs['take_PTs_from']:
                # Get the PT profile
                use_obj = data.radtrans_object
                if data.external_radtrans_reference is not None:
                    use_obj = self.configuration.data[data.external_radtrans_reference].radtrans_object
                ret_val = \
                    data.model_generating_function(use_obj,
                                                   self.configuration.parameters,
                                                   self.pt_plot_mode,
                                                   amr=self.configuration.amr)

                if len(ret_val) == 3:
                    pressures, temperatures, __ = ret_val
                else:
                    pressures, temperatures = ret_val

                return pressures, temperatures
            elif self.pt_plot_mode:
                continue

            if data.external_radtrans_reference is None:
                # Compute the model
                model_returned_values = data.model_generating_function(
                    data.radtrans_object,
                    self.configuration.parameters,
                    self.pt_plot_mode,
                    amr=self.configuration.amr
                )  # TODO the generating function should always return the same number of values

                if retrieve_uncertainties:
                    if len(model_returned_values) == 4:
                        wavelengths_model, spectrum_model, beta, additional_log_l = model_returned_values
                    else:
                        wavelengths_model, spectrum_model, beta = model_returned_values
                        additional_log_l = 0.
                else:
                    if data.variability_atmospheric_column_model_flux_return_mode:
                        wavelengths_model, spectrum_model, additional_log_l, atmospheric_model_column_fluxes = \
                            model_returned_values
                    elif len(model_returned_values) == 3:
                        wavelengths_model, spectrum_model, additional_log_l = model_returned_values
                    else:
                        wavelengths_model, spectrum_model = model_returned_values
                        additional_log_l = 0.

                if additional_log_l is None:
                    additional_log_l = 0

                # Ensure that the spectrum model has no masked values, so that no conversion to NaN is required
                if isinstance(spectrum_model, np.ma.MaskedArray):
                    spectrum_model = spectrum_model.filled(0)

                # Sanity checks on outputs
                if spectrum_model is None:
                    # -np.infs (and anything <1e-100) seemed to cause issues with Multinest.
                    # This particular check probably wasn't the underlying cause, and could
                    # use a better default value.
                    return invalid_value

                # Calculate log likelihood
                # TODO uniformize convolve/rebin handling
                if not isinstance(data.spectrum, float) and data.spectrum.dtype == 'O':
                    if np.ndim(data.spectrum) == 1:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Second dimension of data must be a function of wavelength
                        for i, spectrum_data in enumerate(data.spectrum):
                            if np.isnan(spectrum_model[i][~data.mask[i]]).any():
                                return invalid_value

                            log_likelihood += data.log_likelihood(
                                spectrum_model[i][~data.mask[i]], spectrum_data, data.uncertainties[i],
                                beta=beta,
                                beta_mode=beta_mode
                            )
                    elif np.ndim(data.spectrum) == 2:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Third dimension of data must be a function of wavelength
                        for i, detector in enumerate(data.spectrum):
                            for j, spectrum_data in enumerate(detector):
                                if np.isnan(spectrum_model[i, j][~data.mask[i, j]]).any():
                                    return invalid_value

                                log_likelihood += data.log_likelihood(
                                    spectrum_model[i, j][~data.mask[i, j]], spectrum_data, data.uncertainties[i, j],
                                    beta=beta,
                                    beta_mode=beta_mode
                                )
                    else:
                        raise ValueError(f"observation is an array containing object, "
                                         f"and have {np.ndim(data.spectrum)} dimensions, "
                                         f"but must have 1 to 2")
                else:
                    if np.isnan(spectrum_model).any():
                        return invalid_value

                    if isinstance(data.spectrum, float) or np.ndim(data.spectrum) == 1:
                        # Convolution and rebin are cared of in get_chisq
                        log_likelihood += data.get_chisq(
                            wavelengths_model,
                            spectrum_model,  # [~dd.mask],  # TODO temporary fix until code design rework
                            self.test_plotting,
                            self.configuration.parameters,
                            per_datapoint=per_datapoint,
                            atmospheric_model_column_fluxes=atmospheric_model_column_fluxes
                        ) + additional_log_l
                    elif np.ndim(data.spectrum) == 2:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Second dimension of data must be a function of wavelength
                        for i, spectrum_data in enumerate(data.spectrum):
                            log_likelihood += data.log_likelihood(
                                spectrum_model[i, ~data.mask[i, :]], spectrum_data, data.uncertainties[i],
                                beta=beta,
                                beta_mode=beta_mode
                            )
                    elif np.ndim(data.spectrum) == 3:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Third dimension of data must be a function of wavelength
                        for i, detector in enumerate(data.spectrum):
                            for j, spectrum_data in enumerate(detector):
                                log_likelihood += data.log_likelihood(
                                    spectrum_model[i, j, ~data.mask[i, j, :]], spectrum_data, data.uncertainties[i, j],
                                    beta=beta,
                                    beta_mode=beta_mode
                                )
                    else:
                        raise ValueError(f"observations have {np.ndim(data.spectrum)} dimensions, but must have 1 to 3")
            else:
                wavelengths_model = None
                spectrum_model = None
                additional_log_l = None

            wavelengths_models.append(wavelengths_model)
            spectrum_models.append(spectrum_model)
            additional_log_ls.append(additional_log_l)

            if per_datapoint:
                log_l_per_datapoint_dict[data_name].append(log_likelihood)

            # Check for data using the same pRT object
            # Calculate log likelihood
            for data_name_2, data_2 in self.configuration.data.items():
                if data_2.external_radtrans_reference is not None:
                    if data_2.scale:
                        data_2.scale_factor = self.configuration.parameters[data_name_2 + "_scale_factor"].value

                    if data_2.external_radtrans_reference == data_name:
                        if spectrum_model is None:
                            return invalid_value

                        if np.isnan(spectrum_model).any():
                            return invalid_value

                        log_likelihood += data_2.get_chisq(
                            wavelengths_model,
                            spectrum_model,
                            self.test_plotting,
                            self.configuration.parameters,
                            per_datapoint=per_datapoint,
                            atmospheric_model_column_fluxes=atmospheric_model_column_fluxes
                        ) + additional_log_l

            # Save sampled outputs if necessary.
            if self.configuration.run_mode == 'evaluate':
                if self.evaluate_sample_spectra:
                    self.posterior_sample_spectra[data_name] = [wavelengths_model, spectrum_model]
                else:
                    np.savetxt(
                        os.path.join(
                            self.output_directory,
                            'evaluate_' + self.configuration.retrieval_name,
                            'model_spec_best_fit_' + data_name.replace('/', '_').replace('.', '_') + '.dat',
                        ),
                        np.column_stack((wavelengths_model, spectrum_model))
                    )

                    self.best_fit_spectra[data_name] = [wavelengths_model, spectrum_model]

        if per_datapoint:
            return log_l_per_datapoint_dict

        if "log_prior_weight" in self.configuration.parameters.keys():
            log_prior += self.configuration.parameters["log_prior_weight"].value  # TODO why not using log_likelihood?

        if log_likelihood + log_prior < invalid_value:
            return invalid_value

        if np.abs(log_likelihood + log_prior) < tiny:
            log_likelihood = tiny
            log_prior = 0

        if self.ultranest and np.isinf(log_likelihood + log_prior):
            return invalid_value

        if not return_model:
            if self.print_log_likelihood_for_debugging:
                print('log_likelihood + log_prior', log_likelihood + log_prior)
            return log_likelihood + log_prior
        else:
            return log_likelihood + log_prior, wavelengths_models, spectrum_models, beta, additional_log_ls

    def save_best_fit_outputs(self, parameters, only_return_best_fit_spectra=False):
        # Save sampled outputs if necessary.
        for name, dd in self.configuration.data.items():
            # Only calculate spectra within a given
            # wlen range once
            if dd.scale or dd.scale_err:
                dd.scale_factor = parameters[name + "_scale_factor"].value
            if dd.offset_bool:
                dd.offset = parameters[name + "_offset"].value
            if name + "_b" in parameters.keys():
                dd.bval = parameters[name + "_b"].value

            if dd.external_radtrans_reference is None:
                # Compute the model
                ret_val = \
                    dd.model_generating_function(dd.radtrans_object,
                                                 parameters,
                                                 False,
                                                 amr=self.configuration.amr)
                if len(ret_val) == 3:
                    wlen_model, spectrum_model, additional_logl = ret_val
                elif len(ret_val) == 4 and dd.variability_atmospheric_column_model_flux_return_mode:
                    wlen_model, spectrum_model, additional_logl, atmospheric_model_column_fluxes = ret_val
                    spectrum_model = dd.atmospheric_column_flux_mixer(atmospheric_model_column_fluxes,
                                                                      parameters,
                                                                      dd.name)
                else:
                    wlen_model, spectrum_model = ret_val
            else:
                # Compute the model
                prt_obj = self.configuration.data[dd.external_radtrans_reference].radtrans_object
                ret_val = (
                    self.configuration.data[dd.external_radtrans_reference].model_generating_function(
                        prt_obj,
                        parameters,
                        False,
                        amr=self.configuration.amr
                    )
                )

                if len(ret_val) == 3:
                    wlen_model, spectrum_model, additional_logl = ret_val
                elif len(ret_val) == 4 and dd.variability_atmospheric_column_model_flux_return_mode:
                    wlen_model, spectrum_model, additional_logl, atmospheric_model_column_fluxes = ret_val
                    spectrum_model = dd.atmospheric_column_flux_mixer(atmospheric_model_column_fluxes,
                                                                      parameters,
                                                                      dd.name)
                else:
                    wlen_model, spectrum_model = ret_val

            if self.evaluate_sample_spectra:
                self.posterior_sample_spectra[name] = [wlen_model, spectrum_model]
            else:
                # TODO: This will overwrite the best fit spectrum with
                # whatever is ran through the loglike function. Not good.
                if not only_return_best_fit_spectra:
                    np.savetxt(
                        os.path.join(
                            self.output_directory,
                            'evaluate_' + self.configuration.retrieval_name,
                            'model_spec_best_fit_'
                        )
                        + name.replace('/', '_').replace('.', '_') + '.dat',
                        np.column_stack((wlen_model, spectrum_model))
                    )

                self.best_fit_spectra[name] = [wlen_model, spectrum_model]

        return self.best_fit_spectra

    def save_best_fit_outputs_external_variability(self, parameters, only_return_best_fit_spectra=False):
        wavelengths_model = None  # prevent potential reference before assignment

        # Save sampled outputs if necessary
        for name, data in self.configuration.data.items():
            # Prevent potential reference before assignment
            atmospheric_model_column_fluxes = None
            wavelengths_model = None
            spectrum_model = None

            if data.external_radtrans_reference is None:
                # Only calculate spectra within a given
                # wlen range once
                if data.scale or data.scale_err:
                    data.scale_factor = parameters[name + "_scale_factor"].value

                if data.offset_bool:
                    data.offset = parameters[name + "_offset"].value

                if name + "_b" in parameters.keys():
                    data.bval = parameters[name + "_b"].value

                # Compute the model
                results = (
                    data.model_generating_function(
                        data.radtrans_object,
                        parameters,
                        False,
                        amr=self.configuration.amr
                    )
                )

                if len(results) == 3:
                    wavelengths_model, spectrum_model, additional_logl = results
                elif len(results) == 4 and data.variability_atmospheric_column_model_flux_return_mode:
                    wavelengths_model, spectrum_model, additional_logl, atmospheric_model_column_fluxes = results

                    spectrum_model = data.atmospheric_column_flux_mixer(
                        atmospheric_model_column_fluxes,
                        parameters,
                        data.name
                    )
                else:
                    wavelengths_model, spectrum_model = results

                self.best_fit_spectra[name] = [wavelengths_model, spectrum_model]

            for name_2, data_2 in self.configuration.data.items():
                if data_2.external_radtrans_reference is not None:
                    if data_2.external_radtrans_reference == name:
                        if (data_2.variability_atmospheric_column_model_flux_return_mode
                                and atmospheric_model_column_fluxes is not None):
                            spectrum_model_2 = data_2.atmospheric_column_flux_mixer(
                                atmospheric_model_column_fluxes,
                                parameters,
                                data_2.name
                            )
                            self.best_fit_spectra[name_2] = [wavelengths_model, spectrum_model_2]
                        else:
                            self.best_fit_spectra[name_2] = [wavelengths_model, spectrum_model]

        for name, data in self.configuration.data.items():
            if not only_return_best_fit_spectra:
                np.savetxt(
                    os.path.join(
                        self.output_directory,
                        'evaluate_' + self.configuration.retrieval_name,
                        'model_spec_best_fit_'
                    )
                    + name.replace('/', '_').replace('.', '_') + '.dat',
                    np.column_stack((wavelengths_model, self.best_fit_spectra[name]))
                )

        return self.best_fit_spectra

    def get_samples(self,
                    ultranest=False,
                    names=None,
                    output_directory=os.getcwd(),
                    ret_names=None):
        if ret_names is None:
            ret_names = []

        param_dict = {}
        samples = {}

        if names is None:
            names = [self.configuration.retrieval_name]

        if ultranest:
            for name in names:
                samples_ = np.genfromtxt(output_directory + 'out_' + name + '/chains/equal_weighted_post.txt')
                parameters_read = open(output_directory + 'out_' + name + '/chains/weighted_post.paramnames')
                samples[name] = samples_
                param_dict[name] = parameters_read

            for name in ret_names:
                samples_ = np.genfromtxt(output_directory + 'out_' + name + '/chains/qual_weighted_post.txt')
                parameters_read = open(output_directory + 'out_' + name + '/chains/weighted_post.paramnames')
                samples[name] = samples_
                param_dict[name] = parameters_read

            return samples, param_dict

        # pymultinest
        for name in names:
            samples_ = get_pymultinest_sample_dict(
                output_dir=output_directory,
                name=name,
                add_log_likelihood=True,
                add_stats=False
            )

            samples[name] = np.array(list(samples_.values()))
            param_dict[name] = list(samples_.keys())[:-1]  # do not add the likelihood

        for name in ret_names:
            samples_ = get_pymultinest_sample_dict(
                output_dir=output_directory,
                name=name,
                add_log_likelihood=False,
                add_stats=False
            )

            samples[name] = np.array(list(samples_.values()))
            param_dict[name] = list(samples_.keys())[:-1]  # do not add the likelihood

        return samples, param_dict

    def get_max_likelihood_params(self, best_fit_params, parameters_read):
        """
        This function converts the sample from the post_equal_weights file with the maximum
        log likelihood, and converts it into a dictionary of Parameters that can be used in
        a model function.

        Args:
            best_fit_params : numpy.ndarray
                An array of the best fit parameter values (or any other sample)
            parameters_read : list
                A list of the free parameter names as read from the output files.
        """
        self.best_fit_parameters = self.build_param_dict(best_fit_params, parameters_read)
        return self.best_fit_parameters

    def get_median_params(self, samples, parameters_read, return_array=False):
        """
        This function builds a parameter dictionary based on the median value
        of each parameter. This will update the best_fit_parameter dictionary!
        # TODO fix docstring
        Args:
            parameters_read : list
                A list of the free parameter names as read from the output files.
        """
        i_p = 0
        samples_use = np.zeros(len(parameters_read))
        # Take the median of each column
        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.configuration.parameters[pp].name:
                        samples_use[i_p] = np.median(samples[:, i_s])
                i_p += 1
        self.best_fit_parameters = self.build_param_dict(samples_use, parameters_read)

        if return_array:
            return self.best_fit_parameters, samples_use

        return self.best_fit_parameters

    def get_full_range_model(self,
                             parameters,
                             model_generating_function=None,
                             contribution=False,
                             prt_object=None,
                             prt_reference=None):
        """
        Retrieve a full wavelength range model based on the given parameters.

        Parameters:
            parameters (dict): A dictionary containing parameters used to generate the model.
            model_generating_function (callable, optional): A function to generate the model.
                Defaults to None.
            contribution (bool, optional): Return the emission or transmission contribution function.
                Defaults to False.
            prt_object (object, optional): RadTrans object for calculating the spectrum.
                Defaults to None.
            prt_reference (object, optional): Reference Data object for calculating the spectrum.
                Defaults to None.

        Returns:
            object: The generated full range model.
        """

        # Find the boundaries of the wavelength range to calculate
        wmin = 99999.0
        wmax = 0.0
        for name, dd in self.configuration.data.items():
            if dd.wavelength_boundaries[0] < wmin:
                wmin = dd.wavelength_boundaries[0]
            if dd.wavelength_boundaries[1] > wmax:
                wmax = dd.wavelength_boundaries[1]
        # Set up parameter dictionary
        # parameters = self.build_param_dict(params,parameters_read)
        parameters["contribution"] = Parameter("contribution", False, value=contribution)

        # Set up the pRT object
        if self.configuration.amr:
            p = self.configuration._setup_pres()
            parameters["pressure_scaling"] = self.configuration.parameters["pressure_scaling"]
            parameters["pressure_width"] = self.configuration.parameters["pressure_width"]
            parameters["pressure_simple"] = self.configuration.parameters["pressure_simple"]
        else:
            p = self.configuration.pressures

        if prt_object is not None:
            atmosphere = prt_object
        elif prt_reference is not None:
            atmosphere = self.configuration.data[prt_reference].radtrans_object
        else:
            atmosphere = Radtrans(
                pressures=p,
                line_species=cp.copy(self.configuration.line_species),
                rayleigh_species=cp.copy(self.configuration.rayleigh_species),
                gas_continuum_contributors=cp.copy(self.configuration.continuum_opacities),
                cloud_species=cp.copy(self.configuration.cloud_species),
                line_opacity_mode=self.configuration.data[
                    self.configuration.plot_kwargs["take_PTs_from"]].line_opacity_mode,
                wavelength_boundaries=np.array([wmin * 0.98, wmax * 1.02]),
                scattering_in_emission=self.configuration.scattering_in_emission
            )
        if self.configuration.amr:
            parameters["pressure_scaling"] = self.configuration.parameters["pressure_scaling"]
            parameters["pressure_width"] = self.configuration.parameters["pressure_width"]
            parameters["pressure_simple"] = self.configuration.parameters["pressure_simple"]

        # Check what model function we're using
        if model_generating_function is None:
            mg_func = self.configuration.data[self.configuration.plot_kwargs["take_PTs_from"]].model_generating_function
        else:
            mg_func = model_generating_function

        # get the spectrum
        return mg_func(atmosphere, parameters, pt_plot_mode=False, amr=self.configuration.amr)

    def get_best_fit_model(self, best_fit_params, parameters_read, ret_name=None, contribution=False,
                           prt_reference=None, model_generating_function=None, refresh=True, mode='bestfit'):
        """
        This function uses the best fit parameters to generate a pRT model that spans the entire wavelength
        range of the retrieval, to be used in plots.

        Args:
            best_fit_params : numpy.ndarray
                A numpy array containing the best fit parameters, to be passed to get_max_likelihood_params
            parameters_read : list
                A list of the free parameters as read from the output files.
            ret_name : str
                If plotting a fit from a different retrieval, input the retrieval name to be included.
            contribution : bool
                If True, calculate the emission or transmission contribution function as well as the spectrum.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plotlib will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                If "best_fit", will use the maximum likelihood parameter values to calculate the best fit model
                and contribution. If "median", uses the median parameter values.
        Returns:
            bf_wlen : numpy.ndarray
                The wavelength array of the best fit model
            bf_spectrum : numpy.ndarray
                The emission or transmission spectrum array, with the same shape as bf_wlen
        """
        if ret_name is None:
            ret_name = self.configuration.retrieval_name
        parameters = self.build_param_dict(best_fit_params, parameters_read)
        self.best_fit_parameters = parameters

        if self.configuration.amr:
            _ = self.configuration._setup_pres()  # TODO this function should not be private
            self.best_fit_parameters["pressure_scaling"] = self.configuration.parameters["pressure_scaling"]
            self.best_fit_parameters["pressure_width"] = self.configuration.parameters["pressure_width"]
            self.best_fit_parameters["pressure_simple"] = self.configuration.parameters["pressure_simple"]

        contribution_file = os.path.join(
            self.output_directory,
            f"evaluate_{self.configuration.retrieval_name}",
            f"{ret_name}_{mode}_model_contribution.npy"
        )
        full_file = contribution_file.replace('_contribution.npy', '_full.npy')

        if contribution:
            if not refresh and os.path.exists(contribution_file):
                print("Loading best fit spectrum and contribution from file")
                bf_contribution = np.load(contribution_file)
                bf_wlen, bf_spectrum = np.load(full_file).T

                return bf_wlen, bf_spectrum, bf_contribution

            bf_wlen, bf_spectrum, bf_contribution = self.get_full_range_model(
                self.best_fit_parameters,
                model_generating_function=model_generating_function,
                contribution=contribution,
                prt_reference=prt_reference
            )
            np.save(contribution_file, bf_contribution)
        else:
            if not refresh and os.path.exists(full_file):
                print("Loading best fit spectrum from file")
                bf_wlen, bf_spectrum = np.load(full_file).T
                return bf_wlen, bf_spectrum

            ret_val = self.get_full_range_model(
                self.best_fit_parameters,
                model_generating_function=model_generating_function,
                contribution=contribution,
                prt_reference=prt_reference
            )

            bf_contribution = None  # prevent eventual reference before assignment

            if len(ret_val) == 2:
                bf_wlen, bf_spectrum = ret_val
            else:
                bf_wlen, bf_spectrum, _ = ret_val

        # Add to the dictionary.
        name = f"FullRange_{mode}"

        if prt_reference is not None:
            name = prt_reference

        self.best_fit_spectra[name] = [bf_wlen, bf_spectrum]
        np.save(full_file, np.column_stack([bf_wlen, bf_spectrum]))

        if contribution:
            return bf_wlen, bf_spectrum, bf_contribution

        return bf_wlen, bf_spectrum

    def get_mass_fractions(self, sample, parameters_read=None):
        """
        This function returns the mass fraction abundances of each species as a function of pressure

        Args:
            sample : numpy.ndarray
                A sample from the pymultinest output, the abundances returned will be
                computed for this set of parameters.
            parameters_read : list
                A list of the free parameters as read from the output files.
        Returns:
            abundances : dict
                A dictionary of abundances. The keys are the species name,
                the values are the mass fraction abundances at each pressure
            MMW : numpy.ndarray
                The mean molecular weight at each pressure level in the atmosphere.
        """
        from petitRADTRANS.chemistry.core import get_abundances
        parameters = self.build_param_dict(sample, parameters_read)

        self.pt_plot_mode = True
        pressures, temps = self.log_likelihood(sample, 0, 0)
        self.pt_plot_mode = False

        if self.configuration.data[self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference is None:
            name = self.configuration.plot_kwargs["take_PTs_from"]
        else:
            name = self.configuration.data[self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference

        species = [
            spec.split(
                self.configuration.data.resolving_power_str
            )[0]
            for spec in self.configuration.data[name].radtrans_object.line_species
        ]

        abundances, mmw, _, _ = get_abundances(
            self.configuration.pressures,
            temps,
            cp.copy(species),
            cp.copy(self.configuration.data[name].radtrans_object.cloud_species),
            parameters,
            amr=False
        )
        return abundances, mmw

    def get_volume_mixing_ratios(self, sample, parameters_read=None):
        """
        This function returns the VMRs of each species as a function of pressure.

        Args:
            sample : numpy.ndarray
                A sample from the pymultinest output, the abundances returned will be
                computed for this set of parameters.
            parameters_read : list
                A list of the free parameters as read from the output files.
        Returns:
            vmr : dict
                A dictionary of abundances. The keys are the species name,
                the values are the mass fraction abundances at each pressure
            MMW : numpy.ndarray
                The mean molecular weight at each pressure level in the atmosphere.
        """
        mass_fracs, mmw = self.get_mass_fractions(sample, parameters_read)
        vmr = mass_fractions2volume_mixing_ratios(mass_fracs)

        return vmr, mmw

    def save_volume_mixing_ratios(self, sample_dict, parameter_dict, rets=None):
        """
        Save volume mixing ratios (VMRs) and line absorber species information for specified retrievals.

        Parameters:
        - self: The instance of the class containing the function.
        - sample_dict (dict): A dictionary mapping retrieval names to lists of samples.
        - parameter_dict (dict): A dictionary mapping retrieval names to parameter values.
        - rets (list, optional): List of retrieval names to process. If None, uses the default retrieval name.

        Returns:
        - vmrs (numpy.ndarray): Array containing volume mixing ratios for each sample and species.

        The function processes the specified retrievals and saves the corresponding VMRs and line absorber species
        information to files in the output directory. If 'rets' is not provided, the default retrieval name is used.
        The VMRs are saved in a numpy file, and the line absorber species are saved in a JSON file.

        Example usage:
            ```
            sample_dict = {'Retrieval1': [...], 'Retrieval2': [...]}
            parameter_dict = {'Retrieval1': {...}, 'Retrieval2': {...}}
            vmrs = save_volume_mixing_ratios(sample_dict, parameter_dict)
            ```
        """
        if rets is None:
            rets = [self.configuration.retrieval_name]

        vmrs = None

        for ret in rets:
            if self.configuration.data[
                    self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference is None:
                name = self.configuration.plot_kwargs["take_PTs_from"]
            else:
                name = self.configuration.data[
                    self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference

            species = [spec.split("_R_")[0] for spec in self.configuration.data[name].radtrans_object.line_species]

            samples_use = sample_dict[ret]
            parameters_read = parameter_dict[ret]
            vmrs = []

            for sample in samples_use:
                vmr, _ = self.get_volume_mixing_ratios(sample[:-1], parameters_read)
                vmrs.append(np.array(list(vmr.values())))

            vmrs = np.array(vmrs)
            np.save(
                os.path.join(self.output_directory, f"evaluate_{ret}", f"{ret}_volume_mixing_ratio_profiles"),
                vmrs
            )
            line_spec_file = os.path.join(
                self.output_directory,
                f"evaluate_{ret}",
                f"{ret}_line_absorber_species.json"
            )

            with open(line_spec_file, 'w+') as myFile:
                json.dump(species, myFile)

        return vmrs

    def save_mass_fractions(self, sample_dict, parameter_dict, rets=None):
        """
        Save mass fractions and line absorber species information for specified retrievals.

        Parameters:
        - self: The instance of the class containing the function.
        - sample_dict (dict): A dictionary mapping retrieval names to lists of samples.
        - parameter_dict (dict): A dictionary mapping retrieval names to parameter values.
        - rets (list, optional): List of retrieval names to process. If None, uses the default retrieval name.

        Returns:
        - mass_fractions (numpy.ndarray): Array containing mass fractions for each sample and species.

        The function processes the specified retrievals and saves the corresponding mass fracs and line absorber species
        information to files in the output directory. If 'rets' is not provided, the default retrieval name is used.
        The mass fractinos are saved in a numpy file, and the line absorber species are saved in a JSON file.

        Example usage:
            ```
            sample_dict = {'Retrieval1': [...], 'Retrieval2': [...]}
            parameter_dict = {'Retrieval1': {...}, 'Retrieval2': {...}}
            mass_fractions = save_mass_fractions(sample_dict, parameter_dict)
            ```
        """
        if rets is None:
            rets = [self.configuration.retrieval_name]

        mass_fractions = []

        for ret in rets:
            if self.configuration.data[
                    self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference is None:
                name = self.configuration.plot_kwargs["take_PTs_from"]
            else:
                name = self.configuration.data[
                    self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference

            species = [spec.split("_R_")[0] for spec in self.configuration.data[name].radtrans_object.line_species]

            samples_use = sample_dict[ret]
            parameters_read = parameter_dict[ret]

            for sample in samples_use:
                m_frac, _ = self.get_mass_fraction(sample[:-1], parameters_read)
                mass_fractions.append(np.array(list(m_frac.values())))

            mass_fractions = np.array(mass_fractions)
            np.save(
                os.path.join(self.output_directory, f"evaluate_{ret}", f"{ret}_mass_fraction_profiles"),
                mass_fractions
            )
            line_spec_file = os.path.join(
                self.output_directory,
                f"evaluate_{ret}",
                f"{ret}_line_absorber_species.json"
            )

            with open(line_spec_file, 'w+') as myFile:
                json.dump(species, myFile)

        return mass_fractions

    def get_evidence(self, ret_name=""):
        """
        Get the log10 Z and error for the retrieval

        This function uses the pymultinest analyzer to
        get the evidence for the current retrieval_name
        by default, though any retrieval_name in the
        out_PMN folder can be passed as an argument -
        useful for when you're comparing multiple similar
        models. This value is also printed in the summary file.

        Args:
            ret_name : string
                The name of the retrieval that prepends all the PMN
                output files.
        """
        analyzer = self.get_analyzer(ret_name)
        s = analyzer.get_stats()
        return s['global evidence'] / np.log(10), s['global evidence error'] / np.log(10)

    @staticmethod
    def get_best_fit_likelihood(samples):
        """
        Get the log likelihood of the best fit model

        Args:
            samples : numpy.ndarray
                An array of samples and likelihoods taken from a post_equal_weights file
        """
        log_l = samples[:, -1]
        best_fit_index = np.argmax(log_l)
        print(f"Best fit likelihood = {log_l[best_fit_index]:.2f}")
        return log_l[best_fit_index], best_fit_index

    def get_best_fit_chi2(self, samples):
        """
        Get the 𝛘^2 of the best fit model - removing normalization term from log L

        Args:
            samples : numpy.ndarray
                An array of samples and likelihoods taken from a post_equal_weights file
        """
        log_l, best_fit_index = self.get_best_fit_likelihood(samples)
        params = []
        for key, val in self.configuration.parameters.items():
            if val.is_free_parameter:
                params.append(key)

        self.get_max_likelihood_params(samples[best_fit_index:, -1], params)
        norm = 0.0

        for name, dd in self.configuration.data.items():
            sf = 1.0
            if dd.scale_err:
                sf = self.best_fit_parameters[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _, log_det = np.linalg.slogdet(2 * np.pi * dd.covariance * sf ** 2)
                add = 0.5 * log_det
            else:
                f_err = dd.uncertainties
                if dd.scale_err:
                    f_err = f_err * sf
                if f"{name}_b" in self.configuration.parameters.keys():
                    print(self.best_fit_parameters.keys())
                    f_err = np.sqrt(f_err ** 2 + 10 ** self.best_fit_parameters[f"{name}_b"].value)
                add = 0.5 * np.sum(np.log(2.0 * np.pi * f_err ** 2.))
            norm = norm + add

        return (-log_l - norm) * 2

    def get_log_likelihood_per_datapoint(self, samples_use, ret_name=None):
        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        # Set-up the dictionary structure
        log_l_per_datapoint_dict = {}
        for name in self.configuration.data.keys():
            log_l_per_datapoint_dict[name] = []

        for sample_i in samples_use:
            # Append the logL per datapoint for each posterior sample
            log_l_per_datapoint_dict = self.log_likelihood(
                cube=sample_i, log_l_per_datapoint_dict=log_l_per_datapoint_dict
            )

        # Save the logL's for each instrument
        for name in self.configuration.data.keys():
            log_l_per_datapoint_dict[name] = np.array(log_l_per_datapoint_dict[name])

            np.save(
                f"{self.output_directory}/evaluate_{ret_name}/{ret_name}_logL_per_datapoint_{name}",
                log_l_per_datapoint_dict[name]
            )

    def get_elpd_per_datapoint(self, ret_name=None):
        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        if isinstance(ret_name, str):
            ret_name = [ret_name]

        assert len(ret_name) <= 2

        from .psis import psisloo

        elpd_tot, elpd, pareto_k = {}, {}, {}
        delta_elpd = {}

        for d_name, dd in self.configuration.data.items():

            elpd_tot[d_name], elpd[d_name], pareto_k[d_name] = {}, {}, {}

            for ret_name_i in ret_name:
                log_l_per_datapoint_j = np.load(
                    f"{self.output_directory}/evaluate_{ret_name_i}/{ret_name_i}_logL_per_datapoint_{d_name}.npy",
                )

                # Compute the ELPDs with the PSIS module
                elpd_tot[d_name][ret_name_i], elpd[d_name][ret_name_i], pareto_k[d_name][ret_name_i] \
                    = psisloo(log_l_per_datapoint_j, Reff=1)

                if ret_name_i == self.configuration.retrieval_name:
                    dd.elpd_tot = elpd_tot[d_name][ret_name_i]
                    dd.elpd = elpd[d_name][ret_name_i]
                    dd.pareto_k = pareto_k[d_name][ret_name_i]

            if len(ret_name) == 2:
                delta_elpd[d_name] = elpd[d_name][ret_name[0]] - elpd[d_name][ret_name[1]]
                dd.delta_elpd = delta_elpd[d_name]

        return elpd_tot, elpd, pareto_k, delta_elpd

    def get_chi2(self, sample):
        """
        Get the 𝛘^2 of the given sample relative to the data - removing normalization term from log L

        Args:
            sample : numpy.ndarray
                A single sample and likelihood taken from a post_equal_weights file
        """
        log_l = sample[-1]
        norm = 0
        params = []

        for key, val in self.configuration.parameters.items():
            if val.is_free_parameter:
                params.append(key)

        param_dict = self.build_param_dict(sample, params)

        for name, dd in self.configuration.data.items():
            sf = 1.0
            if dd.scale_err:
                sf = param_dict[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _, log_det = np.linalg.slogdet(2 * np.pi * dd.covariance * sf ** 2)
                add = 0.5 * log_det
            else:
                f_err = dd.uncertainties
                f_err = flatten_object(f_err)

                if dd.scale_err:
                    f_err = f_err * sf

                if f"{name}_b" in self.configuration.parameters.keys():
                    f_err = np.sqrt(f_err ** 2 + 10 ** param_dict[f"{name}_b"].value)

                add = 0.5 * np.sum(np.log(2.0 * np.pi * f_err ** 2.))

            norm = norm + add

        return 2 * (-log_l - norm)

    def get_chi2_normalisation(self, sample):
        """
        Get the 𝛘^2 normalization term from log L

        Args:
            sample : numpy.ndarray
                A single sample and likelihood taken from a post_equal_weights file
        """
        norm = 0
        params = []

        for key, val in self.configuration.parameters.items():
            if val.is_free_parameter:
                params.append(key)

        param_dict = self.build_param_dict(sample, params)

        for name, dd in self.configuration.data.items():
            sf = 1.0
            if dd.scale_err:
                sf = param_dict[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _, log_det = np.linalg.slogdet(2 * np.pi * dd.covariance * sf ** 2)
                add = 0.5 * log_det
            else:
                f_err = dd.uncertainties
                if dd.scale_err:
                    f_err = f_err * sf
                if f"{name}_b" in self.configuration.parameters.keys():
                    f_err = np.sqrt(f_err ** 2 + 10 ** param_dict[f"{name}_b"].value)
                add = 0.5 * np.sum(np.log(2.0 * np.pi * f_err ** 2.))
            norm = norm + add

        return norm

    def get_reduced_chi2(self, sample, subtract_n_parameters=False, verbose=False, show_chi2=False):
        """
        Get the 𝛘^2/DoF of the given model - divide chi^2 by DoF or number of wavelength channels.

        Args:
            sample : numpy.ndarray
                A single sample and likelihoods taken from a post_equal_weights file
            subtract_n_parameters : bool
                If True, divide the Chi2 by the degrees of freedom (n_data - n_parameters). If False,
                divide only by n_data
            verbose : bool
                If True, display the calculated best fit reduced chi^2, and also the best fit chi^2 if show_chi2 is True
            show_chi2 : bool
                If True, additionally display the calculated best fit chi^2 if verbose is True
        """
        chi2 = self.get_chi2(sample)
        d_o_f = 0

        for name, dd in self.configuration.data.items():
            d_o_f += np.size(dd.spectrum)

        if subtract_n_parameters:
            for name, pp in self.configuration.parameters.items():
                if pp.is_free_parameter:
                    d_o_f -= 1

        if verbose:
            if show_chi2:
                print(f"Best fit 𝛘^2 = {chi2:.2f}")

            if subtract_n_parameters:
                print(f"Best fit 𝛘^2/DoF = {chi2 / d_o_f:.2f}")
            else:
                print(f"Best fit 𝛘^2/n_wlen = {chi2 / d_o_f:.2f}")

        self.chi2 = chi2 / d_o_f

        return chi2 / d_o_f

    def get_reduced_chi2_from_model(self, wlen_model, spectrum_model, subtract_n_parameters=False,
                                    verbose=False, show_chi2=False):
        """
        Get the 𝛘^2/DoF of the supplied spectrum - divide chi^2 by DoF

        Args:
            wlen_model : np.ndarray
                The wavelength grid of the model spectrum in micron.
            spectrum_model : np.ndarray
                The model flux in the same units as the data.
            subtract_n_parameters : bool
                If True, divide the Chi2 by the degrees of freedom (n_data - n_parameters). If False,
                divide only by n_data
            verbose : bool
                If True, display the calculated best fit chi^2 and reduced chi^2
            show_chi2 : bool
                If True, additionally display the calculated best fit chi^2 if verbose is True
        """
        log_l = 0
        norm = 0
        d_o_f = 0

        for name, dd in self.configuration.data.items():
            d_o_f += np.size(dd.spectrum)
            sf = 1
            log_l += dd.get_chisq(
                wlen_model,
                spectrum_model,
                False,
                self.configuration.parameters
            )

            if dd.covariance is not None:
                if self.best_fit_parameters:
                    if dd.scale_err:
                        sf *= self.best_fit_parameters[f"{name}_scale_factor"].value
                    _, log_det = np.linalg.slogdet(2 * np.pi * dd.covariance * sf ** 2)
                    add = 0.5 * log_det
                else:
                    if dd.scale_err:
                        sf *= dd.scale_factor
                    _, log_det = np.linalg.slogdet(2 * np.pi * dd.covariance * sf ** 2)
                    add = 0.5 * log_det
            else:
                add = 0.5 * np.sum(np.log(2.0 * np.pi * dd.uncertainties ** 2.))
            norm += add

        if subtract_n_parameters:
            for name, pp in self.configuration.parameters.items():
                if pp.is_free_parameter:
                    d_o_f -= 1

        chi2 = 2 * (-log_l - norm)

        if verbose:
            if show_chi2:
                print(f"Best fit 𝛘^2 = {chi2:.2f}")

            if subtract_n_parameters:
                print(f"Best fit 𝛘^2/DoF = {chi2 / d_o_f:.2f}")
            else:
                print(f"Best fit 𝛘^2/n_wlen = {chi2 / d_o_f:.2f}")

        return chi2 / d_o_f

    def get_analyzer(self, ret_name=""):
        """
        Get the PMN analyzer from a retrieval run

        This function uses gets the PMN analyzer object
        for the current retrieval_name by default,
        though any retrieval_name in the out_PMN folder can
        be passed as an argument - useful for when you're
        comparing multiple similar models.

        Args:
            ret_name : string
                The name of the retrieval that prepends all the PMN
                output files.
        """
        import pymultinest

        # Avoid loading if we just want the current retrievals output
        if ret_name == "" and self.analyzer is not None:
            return self.analyzer

        if ret_name == "":
            ret_name = self.configuration.retrieval_name

        prefix = os.path.join(self.output_directory, 'out_PMN', ret_name + '_')

        # How many free parameters?
        n_params = 0
        free_parameter_names = []
        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                free_parameter_names.append(self.configuration.parameters[pp].name)
                n_params += 1

        # Get the outputs
        analyzer = pymultinest.Analyzer(n_params=n_params,
                                        outputfiles_basename=prefix)
        if ret_name == self.configuration.retrieval_name:
            self.analyzer = analyzer
        return analyzer

    def build_param_dict(self, sample, free_param_names):
        """
        This function builds a dictionary of parameters that can be passed to the
        model building functions. It requires a numpy array with the same length
        as the number of free parameters, and a list of all the parameter names
        in the order they appear in the array. The returned dictionary will contain
        all of these parameters, together with the fixed retrieval parameters.

        Args:
            sample : numpy.ndarray
                An array or list of free parameter values
            free_param_names : list(string)
                A list of names for each of the free parameters.
        Returns:
            params : dict
                A dictionary of Parameters, with values set to the values
                in sample.
        """
        params = {}
        i_p = 0
        for pp in self.configuration.parameters:
            if self.configuration.parameters[pp].is_free_parameter:
                for i_s in range(len(free_param_names)):
                    if free_param_names[i_s] == self.configuration.parameters[pp].name:
                        params[self.configuration.parameters[pp].name] = \
                            Parameter(pp, False, value=sample[i_p])
                        i_p += 1
            else:
                params[pp] = Parameter(pp, False, value=self.configuration.parameters[pp].value)
        return params

    def sample_teff(self, sample_dict, param_dict, ret_names=None, nsample=None, resolution=40):
        r"""
        This function samples the outputs of a retrieval and computes Teff
        for each sample. For each sample, a model is computed at low resolution,
        and integrated to find the total radiant emittance, which is converted into
        a temperature using the stefan boltzmann law: $j^{\star} = \sigma T^{4}$.
        Teff itself is computed using util.calc_teff.

        Args:
            sample_dict : dict
                A dictionary, where each key is the name of a retrieval, and the values
                are the equal weighted samples.
            param_dict : dict
                A dictionary where each key is the name of a retrieval, and the values
                are the names of the free parameters associated with that retrieval.
            ret_names : Optional(list(string))
                A list of retrieval names, each should be included in the sample_dict.
                If left as none, it defaults to only using the current retrieval name.
            nsample : Optional(int)
                The number of times to compute Teff. If left empty, uses the "take_PTs_from"
                plot_kwarg. Recommended to use ~300 samples, probably more than is set in
                the kwarg!
            resolution : int
                The spectra resolution to compute the models at. Typically, this should be very
                low in order to enable rapid calculation.
        Returns:
            tdict : dict
                A dictionary with retrieval names for keys, and the values are the calculated
                values of Teff for each sample.
        """
        from petitRADTRANS.physics import compute_effective_temperature

        if ret_names is None:
            ret_names = [self.configuration.retrieval_name]
        if nsample is None:
            nsample = self.configuration.plot_kwargs["nsample"]

        # Set up the pRT object
        self._rebin_opacities(resolution=resolution)

        species = []

        for spec in self.configuration.line_species:
            species.append(join_species_all_info(spec, spectral_info=get_resolving_power_string(resolution)))

        if self.configuration.amr:
            p = self.configuration._setup_pres()
        else:
            p = self.configuration.pressures

        prt_object = Radtrans(
            pressures=p,
            line_species=cp.copy(self.configuration.line_species),
            rayleigh_species=cp.copy(self.configuration.rayleigh_species),
            gas_continuum_contributors=cp.copy(self.configuration.continuum_opacities),
            cloud_species=cp.copy(self.configuration.cloud_species),
            line_opacity_mode='c-k',
            wavelength_boundaries=np.array([0.5, 28]),
            scattering_in_emission=self.configuration.scattering_in_emission
        )

        tdict = {}

        for name in ret_names:
            teffs = []
            samples = sample_dict[name]
            parameters_read = param_dict[name]

            if nsample == "all":
                rands = np.linspace(0, samples.shape[0] - 1, samples.shape[0])
            else:
                rands = np.random.randint(0, samples.shape[0], int(nsample))

            duse = self.configuration.data[self.configuration.plot_kwargs["take_PTs_from"]]

            if duse.external_radtrans_reference is not None:
                duse = self.configuration.data[duse.external_radtrans_reference]

            for rint in rands:
                samp = samples[int(rint), :-1]
                params = self.build_param_dict(samp, parameters_read)
                ret_val = duse.model_generating_function(
                    prt_object,
                    params,
                    False,
                    self.configuration.amr
                )

                if len(ret_val) == 2:
                    wlen, model = ret_val
                else:
                    wlen, model, __ = ret_val

                tfit = compute_effective_temperature(wlen, model, params["D_pl"].value, params["R_pl"].value)
                teffs.append(tfit)
            tdict[name] = np.array(teffs)
            np.save(os.path.join(self.output_directory, "evaluate_" + name, "sampled_teff"), np.array(teffs))
        return tdict

    def plot_all(self,
                 output_directory=None,
                 ret_names=None,
                 contribution=False,
                 model_generating_function=None,
                 prt_reference=None,
                 mode='bestfit'):
        # TODO no plot functions outside of plotlib
        """
        Produces plots for the best fit spectrum, a sample of 100 output spectra,
        the best fit PT profile and a corner plot for parameters specified in the
        run definition.

        By default, this runs the following functions:
            plot_spectra: Plots the best fit spectrum together with the data, with an extra
                        panel showing the residuals between the model and data.
            plot_PT: plots the pressure-temperature profile contours
            plot_corner : Corner plot based on the posterior sample distributions
            plot_abundances : Abundance profiles for each line species used.

        if contribution = True:
            plot_contribution : The emission or transmission contribution function
            In addition to plotting the contribution function, the contribution
            will also be overlaid on top of the PT profiles and abundance profiles.

        if self.evaluate_sample_spectra = True
            plot_sampled : Randomly draws N samples from the posterior distribution,
            and plots the resulting spectrum overtop the data.

        Args:
            output_directory: string
                Output directory to store the plots. Defaults to selt.output_dir.
            ret_names : list(str)
                List of retrieval names. Used if multiple retrievals are to be included
                in a single corner plot.
            contribution : bool
                If true, plot the emission or transmission contribution function.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            mode : str
                If 'bestfit', consider the maximum likelihood sample for plotting,
                if median, calculate the model based on the median retrieved parameters.
        """
        # Run plotting on a single core only.
        if not self.use_mpi or rank == 0:
            if ret_names is None:
                ret_names = []

            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'

            if output_directory is None:
                output_directory = self.output_directory

            sample_dict, parameter_dict = self.get_samples(
                ultranest=self.ultranest,
                names=self.corner_files,
                output_directory=output_directory,
                ret_names=ret_names
            )

            ###########################################
            # Plot best-fit spectrum
            ###########################################
            samples_use = cp.copy(sample_dict[self.configuration.retrieval_name])
            parameters_read = cp.copy(parameter_dict[self.configuration.retrieval_name])
            i_p = 0

            # This might actually be redundant...
            for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.configuration.parameters[pp].name:
                            samples_use[:, i_p] = sample_dict[self.configuration.retrieval_name][:, i_s]
                    i_p += 1

            print("Best fit parameters")
            i_p = 0
            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Print outputs
            # TODO add verbosity
            for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.configuration.parameters[pp].name:
                            print(self.configuration.parameters[pp].name, samples_use[best_fit_index][i_p])
                            i_p += 1

            # Plotting
            #
            self.plot_spectra(samples_use,
                              parameters_read,
                              refresh=True,
                              model_generating_function=model_generating_function,
                              prt_reference=prt_reference,
                              mode=mode)

            if self.evaluate_sample_spectra:
                self.plot_sampled(samples_use,
                                  parameters_read,
                                  model_generating_function=model_generating_function,
                                  prt_reference=prt_reference, )

            self.plot_pt(sample_dict,
                         parameters_read,
                         contribution=contribution,
                         model_generating_function=model_generating_function,
                         prt_reference=prt_reference,
                         mode=mode,
                         refresh=False)
            self.plot_corner(sample_dict,
                             parameter_dict,
                             parameters_read)

            if contribution:
                self.plot_contribution(samples_use,
                                       parameters_read,
                                       model_generating_function=model_generating_function,
                                       prt_reference=prt_reference,
                                       mode=mode,
                                       refresh=False)

            self.plot_abundances(samples_use,
                                 parameters_read,
                                 contribution=contribution,
                                 model_generating_function=model_generating_function,
                                 prt_reference=prt_reference,
                                 mode=mode,
                                 refresh=False)
            print("Finished generating all plots!")

        if self.use_mpi and comm is not None:
            comm.barrier()
        return

    def plot_spectra(self, samples_use, parameters_read, model_generating_function=None, prt_reference=None,
                     refresh=True, mode="bestfit", marker_color_type=None, marker_cmap=None, marker_label='',
                     only_save_best_fit_spectra=False):
        """
        Plot the best fit spectrum, the data from each dataset and the residuals between the two.
        Saves a file to OUTPUT_DIR/evaluate_RETRIEVAL_NAME/RETRIEVAL_NAME_MODE_spec.pdf

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            model_generating_function : method
                A function that will take in the standard 'model' arguments
                (pRT_object, params, pt_plot_mode, amr, resolution)
                and will return the wavlength and flux arrays as calculated by petitRadTrans.
                If no argument is given, it uses the method of the first dataset included in the retrieval.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                Use 'bestfit' (minimum likelihood) parameters, or median parameter values.
            marker_color_type : str
                Data-attribute to plot as marker colors. Use 'delta_elpd', 'elpd', or 'pareto_k'.
            marker_cmap : matplotlib colormap
                Colormap to use for marker colors.
            marker_label : str
                Label to add to colorbar corresponding to marker colors.
            only_save_best_fit_spectra : bool
                If False (default value), the plot_spectra routine will run as per usual, producing a plot
                of the best-fit or median-parameter forward model and the data.
                If True, only the best-fit models will be calculated for every dataset, and saved in the
                "evaluate_"+retrieval_name folder. No figure will be generated.
        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

        if marker_cmap is None:
            marker_cmap = plt.cm.bwr

        if not self.use_mpi or rank == 0:
            # Avoiding saving the model spectrum to the sampled spectrum dictionary.
            check = self.evaluate_sample_spectra

            if self.evaluate_sample_spectra:
                self.evaluate_sample_spectra = False

            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'
            print("\nPlotting Best-fit spectrum")

            fig, axes = plt.subplots(
                nrows=2, ncols=1, sharex='col', sharey=False,
                gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.1},
                figsize=(18, 9)
            )
            ax = axes[0]  # Normal Spectrum axis
            ax_r = axes[1]  # residual axis

            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
            sample_use = samples_use[:-1, best_fit_index]

            # Then get the full wavelength range
            # Generate the best fit spectrum using the set of parameters with the lowest log-likelihood
            if mode.lower() == "median":
                med_param, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)

            # Setup best fit spectrum
            # First get the fit for each dataset for the residual plots
            # self.log_likelihood(sample_use, 0, 0)
            if only_save_best_fit_spectra:
                self.save_best_fit_outputs(self.best_fit_parameters)
                return None, None, None
            bf_wlen, bf_spectrum = self.get_best_fit_model(
                sample_use,  # set of parameters with the lowest log-likelihood (best-fit)
                parameters_read,  # name of the parameters
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                refresh=refresh,
                mode=mode
            )
            chi2 = self.get_reduced_chi2_from_model(
                wlen_model=bf_wlen,
                spectrum_model=bf_spectrum,
                subtract_n_parameters=True,
                verbose=True,
                show_chi2=True
            )
            if not only_save_best_fit_spectra:
                self.save_best_fit_outputs(self.best_fit_parameters)

            markersize = None
            if marker_color_type is not None:

                markersize = 10

                l, b, w, h = ax.get_position().bounds
                cax = fig.add_axes([l + w + 0.015 * w, b, 0.025 * w, h])

                markerfacecolors = {}
                vmin, vmax = np.inf, -np.inf
                for name, dd in self.configuration.data.items():
                    assert hasattr(dd, marker_color_type)

                    markerfacecolors[name] = getattr(dd, marker_color_type)
                    vmin = min([markerfacecolors[name].min(), vmin])
                    vmax = max([markerfacecolors[name].max(), vmax])

                if marker_color_type.startswith('delta_'):
                    vmax = np.max(np.abs([vmin, vmax]))
                    vmin = -1 * vmax

                from matplotlib.colors import Normalize
                norm = Normalize(vmin=vmin, vmax=vmax)

                for name, dd in self.configuration.data.items():
                    markerfacecolors[name] = norm(markerfacecolors[name])
                    markerfacecolors[name] = marker_cmap(markerfacecolors[name])

                from matplotlib.cm import ScalarMappable
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=marker_cmap),
                    ax=ax, cax=cax, orientation='vertical'
                )
                cax.set_ylabel(marker_label)

                if marker_color_type == 'pareto_k':
                    cax.axhline(0.7, c='k', ls='--')

            # Iterate through each dataset, plotting the data and the residuals.
            for name, dd in self.configuration.data.items():
                # If the user has specified a resolution, rebin to that
                if not dd.photometry:
                    resolution_data = np.mean(dd.wavelengths[1:] / np.diff(dd.wavelengths))
                    if self.configuration.plot_kwargs["resolution"] is not None and \
                            self.configuration.plot_kwargs["resolution"] < resolution_data:
                        ratio = resolution_data / self.configuration.plot_kwargs["resolution"]
                        flux, edges, _ = binned_statistic(
                            dd.wavelengths, dd.spectrum, 'mean', dd.wavelengths.shape[0] / ratio
                        )
                        error, _, _ = binned_statistic(
                            dd.wavelengths, dd.uncertainties,
                            'mean', dd.wavelengths.shape[0] / ratio
                        ) / np.sqrt(ratio)

                        wlen = np.array([(edges[i] + edges[i + 1]) / 2.0 for i in range(edges.shape[0] - 1)])
                        wlen_bins = np.zeros_like(wlen)
                        wlen_bins[:-1] = np.diff(wlen)
                        wlen_bins[-1] = wlen_bins[-2]
                    else:
                        wlen = dd.wavelengths
                        error = dd.uncertainties
                        flux = dd.spectrum
                        wlen_bins = dd.wavelength_bin_widths
                else:
                    wlen = np.mean(dd.photometric_bin_edges)
                    flux = dd.spectrum
                    error = dd.uncertainties
                    wlen_bins = dd.wavelength_bin_widths

                # If the data has an arbitrary retrieved scaling factor
                scale = 1.0

                if dd.scale:
                    scale = self.best_fit_parameters[f"{name}_scale_factor"].value

                if dd.scale_err:
                    errscale = self.best_fit_parameters[f"{name}_scale_factor"].value
                    error = error * errscale

                offset = 0.0
                if dd.offset_bool:
                    offset = self.best_fit_parameters[f"{name}_offset"].value

                flux = (flux * scale) - offset
                if f"{dd.name}_b" in self.configuration.parameters.keys():
                    # TODO best_fit_parameters is not an attribute of Retrieval
                    raise ValueError("undefined attribute 'Retrieval.best_fit_parameters'")
                    # error = np.sqrt(error + 10 ** (self.best_fit_parameters["{dd.name}_b"]))

                if not dd.photometry:
                    if dd.external_radtrans_reference is None:
                        spectrum_model = self.best_fit_spectra[name][1]
                        if dd.data_resolution is not None:
                            spectrum_model = dd.convolve(self.best_fit_spectra[name][0],
                                                         self.best_fit_spectra[name][1],
                                                         dd.data_resolution)
                        best_fit_binned = frebin.rebin_spectrum_bin(
                            self.best_fit_spectra[name][0],
                            spectrum_model,
                            wlen,
                            wlen_bins
                        )
                    else:
                        if dd.data_resolution is not None:
                            spectrum_model = dd.convolve(self.best_fit_spectra[dd.external_radtrans_reference][0],
                                                         self.best_fit_spectra[dd.external_radtrans_reference][1],
                                                         dd.data_resolution)
                        else:
                            spectrum_model = None  # TODO prevent reference before assignment

                        best_fit_binned = frebin.rebin_spectrum_bin(
                            self.best_fit_spectra[name][0],
                            spectrum_model,
                            wlen,
                            wlen_bins
                        )
                else:
                    if dd.external_radtrans_reference is None:
                        best_fit_binned = dd.photometric_transformation_function(self.best_fit_spectra[name][0],
                                                                                 self.best_fit_spectra[name][1])
                        # Species functions give tuples of (flux,error)
                        try:
                            best_fit_binned = best_fit_binned[0]
                        except Exception:  # TODO find exception expected here
                            pass

                    else:
                        best_fit_binned = \
                            dd.photometric_transformation_function(
                                self.best_fit_spectra[dd.external_radtrans_reference][0],
                                self.best_fit_spectra[dd.external_radtrans_reference][1]
                            )
                        try:
                            best_fit_binned = best_fit_binned[0]
                        except Exception:  # TODO find exception expected here
                            pass
                # Plot the data
                marker = 'o'
                if dd.photometry:
                    marker = 's'
                if not dd.photometry:
                    label = dd.name
                    for i in range(len(flux)):
                        color_i = 'C0'
                        ecolor = 'C0'
                        if marker_color_type is not None:
                            color_i = markerfacecolors[name][i]
                            ecolor = 'k'

                        ax.errorbar(wlen[i],
                                    (flux[i] * self.configuration.plot_kwargs["y_axis_scaling"]),
                                    yerr=error[i] * self.configuration.plot_kwargs["y_axis_scaling"],
                                    marker=marker,
                                    ecolor=ecolor,
                                    markersize=markersize,
                                    markerfacecolor=color_i,
                                    markeredgecolor='k',
                                    linewidth=0,
                                    elinewidth=2,
                                    label=label,
                                    zorder=10,
                                    alpha=0.9)

                        # Plot the residuals
                        ax_r.errorbar(
                            wlen[i],
                            ((flux - best_fit_binned) / error)[i],
                            yerr=(error / error)[i],
                            marker=marker,
                            ecolor=ecolor,
                            markersize=markersize,
                            markerfacecolor=color_i,
                            markeredgecolor='k',
                            linewidth=0,
                            elinewidth=2,
                            zorder=10,
                            alpha=0.9
                        )

                        label = None
                else:
                    # Don't label photometry?

                    for i in range(len(flux)):
                        color_i = 'grey'
                        ecolor = 'grey'
                        if marker_color_type is not None:
                            color_i = markerfacecolors[name][i]
                            ecolor = 'k'

                        ax.errorbar(wlen[i],
                                    (flux[i] * self.configuration.plot_kwargs["y_axis_scaling"]),
                                    yerr=error[i] * self.configuration.plot_kwargs["y_axis_scaling"],
                                    xerr=dd.wlen_bins / 2.,
                                    linewidth=0,
                                    elinewidth=2,
                                    marker=marker,
                                    ecolor=ecolor,
                                    markersize=markersize,
                                    markerfacecolor=color_i,
                                    markeredgecolor='k',
                                    color='grey',
                                    zorder=10,
                                    label=None,
                                    alpha=0.6)

                        # Plot the residuals
                        ax_r.errorbar(
                            wlen[i],
                            ((flux - best_fit_binned) / error)[i],
                            yerr=(error / error)[i],
                            xerr=dd.wlen_bins / 2.,
                            color='grey',
                            marker=marker,
                            ecolor=ecolor,
                            markersize=markersize,
                            markerfacecolor=color_i,
                            markeredgecolor='k',
                            linewidth=0,
                            elinewidth=2,
                            zorder=10,
                            alpha=0.6
                        )

            # Plot the best fit model
            ax.plot(bf_wlen,
                    bf_spectrum * self.configuration.plot_kwargs["y_axis_scaling"],
                    label=rf'Best Fit Model, $\chi^2=${chi2:.2f}',
                    linewidth=4,
                    alpha=0.5,
                    color='r')

            # Plot the shading in the residual plot
            yabs_max = abs(max(ax_r.get_ylim(), key=abs))
            lims = ax.get_xlim()
            lim_y = ax.get_ylim()
            lim_y = [lim_y[0], lim_y[1] * 1.05]

            if self.configuration.plot_kwargs.get('flux_lim') is not None:
                ax.set_ylim(self.configuration.plot_kwargs.get('flux_lim'))
            else:
                ax.set_ylim(lim_y)

            # weird scaling to get axis to look ok on log plots
            if self.configuration.plot_kwargs["xscale"] == 'log':
                lims = [bf_wlen[0] * 0.98, lims[1] * 1.02]
            else:
                lims = [bf_wlen[0] * 0.98, bf_wlen[-1] * 1.02]

            if self.configuration.plot_kwargs.get('wavelength_lim') is not None:
                ax.set_xlim(self.configuration.plot_kwargs.get('wavelength_lim'))
                ax_r.set_xlim(self.configuration.plot_kwargs.get('wavelength_lim'))
            else:
                ax.set_xlim(lims)
                ax_r.set_xlim(lims)

            ax_r.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax_r.fill_between(lims, -1, 1, color='dimgrey', alpha=0.4, zorder=-10)
            ax_r.fill_between(lims, -3, 3, color='darkgrey', alpha=0.3, zorder=-9)
            ax_r.fill_between(lims, -5, 5, color='lightgrey', alpha=0.3, zorder=-8)
            ax_r.axhline(linestyle='--', color='k', alpha=0.8, linewidth=2)

            # Making the plots pretty
            if "xscale" in self.configuration.plot_kwargs.keys():
                ax.set_xscale(self.configuration.plot_kwargs["xscale"])
            try:
                ax.set_yscale(self.configuration.plot_kwargs["yscale"])
            except Exception:  # TODO find exception expected here
                pass

            # Fancy ticks for upper pane
            ax.tick_params(axis="both", direction="in", length=10, bottom=True, top=True, left=True, right=True)
            try:
                ax.xaxis.set_major_formatter('{x:.1f}')
            except Exception:  # TODO find exception expected here
                warnings.warn("Please update to matplotlib 3.3.4 or greater")
                pass

            min_wlen = bf_wlen[0]
            max_wlen = bf_wlen[-1]

            if self.configuration.plot_kwargs["xscale"] == 'log':
                if min_wlen < 0:
                    min_wlen = 0.08
                # For the minor ticks, use no labels; default NullFormatter.
                x_major = LogLocator(base=10.0, subs=np.linspace(min_wlen, max_wlen, 4, dtype=int), numticks=4)
                ax.xaxis.set_major_locator(x_major)
                x_minor = LogLocator(base=10.0, subs=np.linspace(min_wlen, max_wlen, 40), numticks=100)
                ax.xaxis.set_minor_locator(x_minor)
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(axis='both', which='minor',
                               bottom=True, top=True, left=True, right=True,
                               direction='in', length=5)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='minor',
                           bottom=True, top=True, left=True, right=True,

                           direction='in', length=5)
            ax.set_ylabel(self.configuration.plot_kwargs["spec_ylabel"])

            # Fancy ticks for lower pane
            ax_r.tick_params(axis="both", direction="in", length=10, bottom=True, top=True, left=True, right=True)
            try:
                ax_r.xaxis.set_major_formatter('{x:.1f}')
            except Exception:  # TODO find exception expected here
                warnings.warn("Please update to matplotlib 3.3.4 or greater")
                pass

            if self.configuration.plot_kwargs["xscale"] == 'log':
                # For the minor ticks, use no labels; default NullFormatter.
                x_major = LogLocator(base=10.0, subs=(1, 2, 3, 4), numticks=4)
                ax_r.xaxis.set_major_locator(x_major)
                x_minor = LogLocator(base=10.0, subs=np.arange(0.1, 10.1, 0.1) * 0.1, numticks=100)
                ax_r.xaxis.set_minor_locator(x_minor)
                ax_r.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax_r.xaxis.set_minor_locator(AutoMinorLocator())
                ax_r.tick_params(axis='both', which='minor',
                                 bottom=True, top=True, left=True, right=True,
                                 direction='in', length=5)
            ax_r.yaxis.set_minor_locator(AutoMinorLocator())
            ax_r.tick_params(axis='both', which='minor',
                             bottom=True, top=True, left=True, right=True,
                             direction='in', length=5)

            ax_r.set_ylabel(r"Residuals [$\sigma$]")
            ax_r.set_xlabel(self.configuration.plot_kwargs["spec_xlabel"])
            ax.legend(loc='upper center', ncol=len(self.configuration.data.keys()) + 1).set_zorder(1002)
            plt.tight_layout()
            plt.savefig(
                self.get_base_figure_name() + '_' + mode + '_spec.pdf'
            )
            self.evaluate_sample_spectra = check
        else:
            fig = None
            ax = None
            ax_r = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig, ax, ax_r

    def plot_sampled(self, samples_use, parameters_read, downsample_factor=None, save_outputs=False,
                     nsample=None, model_generating_function=None, prt_reference=None, refresh=True):
        """
        Plot a set of randomly sampled output spectra for each dataset in
        the retrieval.

        This will save nsample files for each dataset included in the retrieval.
        Note that if you change the model_resolution of your Data and rerun this
        function, the files will NOT be updated - if the files exists the function
        defaults to reading from file rather than recomputing. Delete all of the
        sample functions and run it again.

        Args:
            samples_use : np.ndarray
                posterior samples from pynmultinest outputs (post_equal_weights)
            parameters_read : list(str)
                list of free parameters as read from the output files.
            downsample_factor : int
                Factor by which to reduce the resolution of the sampled model,
                for smoother plotting. Defaults to None. A value of None will result
                in the full resolution spectrum. Note that this factor can only
                reduce the resolution from the underlying model_resolution of the
                data.
            nsample : int
                Number of samples to draw from the posterior distribution. Defaults to the
                value of self.rd.plot_kwargs["nsample"].
            save_outputs : bool
                If true, saves each calculated spectrum as a .npy file. The name of the file indicates the
                index from the post_equal_weights file that was used to generate the sample.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
        """
        import matplotlib.pyplot as plt
        from petitRADTRANS.plotlib.plotlib import plot_data

        if not self.use_mpi or rank == 0:

            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'

            self.configuration.plot_kwargs["nsample"] = int(self.configuration.plot_kwargs["nsample"])

            print("\nPlotting Best-fit spectrum with " + str(self.configuration.plot_kwargs["nsample"]) + " samples.")
            print("This could take some time...")
            len_samples = samples_use.shape[0]
            path = os.path.join(self.output_directory, 'evaluate_' + self.configuration.retrieval_name)

            wmin = 99999.0
            wmax = 0.0
            for name, dd in self.configuration.data.items():
                if dd.wavelength_boundaries[0] < wmin:
                    wmin = dd.wavelength_boundaries[0]
                if dd.wavelength_boundaries[1] > wmax:
                    wmax = dd.wavelength_boundaries[1]

            # Set up parameter dictionary
            atmosphere = Radtrans(line_species=cp.copy(self.configuration.line_species),
                                  rayleigh_species=cp.copy(self.configuration.rayleigh_species),
                                  gas_continuum_contributors=cp.copy(self.configuration.continuum_opacities),
                                  cloud_species=cp.copy(self.configuration.cloud_species),
                                  line_opacity_mode='c-k',
                                  wavelength_boundaries=np.array([wmin * 0.98, wmax * 1.02]),
                                  scattering_in_emission=self.configuration.scattering_in_emission)
            fig, ax = plt.subplots(figsize=(16, 10))
            nsamp = nsample
            if nsample is None:
                nsamp = self.configuration.plot_kwargs["nsample"]
            random_ints = np.random.randint(low=0, high=len_samples, size=int(nsamp))
            for i_sample, random_index in enumerate(random_ints):
                file = os.path.join(path, "posterior_sampled_spectra_" + str(random_index).zfill(5))

                if os.path.exists(file):
                    wlen, model = np.load(file)
                else:
                    print(f"Generating sampled spectrum {i_sample} / {self.configuration.plot_kwargs['nsample']}...")

                    parameters = self.build_param_dict(samples_use[random_index, :-1], parameters_read)
                    parameters["contribution"] = Parameter("contribution", False, value=False)
                    ret_val = self.get_full_range_model(parameters, prt_object=atmosphere)

                    wlen = None
                    model = None

                    if len(ret_val) == 2:
                        wlen, model = ret_val
                    elif len(ret_val) == 3:
                        wlen, model, __ = ret_val
                    else:
                        ValueError(f"expected 2 or 3 values to unpack from full range model, "
                                   f"but got {len(ret_val)}")

                if downsample_factor is not None:
                    model = running_mean(model, downsample_factor)[::downsample_factor]
                    wlen = wlen[::downsample_factor]

                if save_outputs:
                    np.save(
                        file,
                        np.column_stack((wlen, model))
                    )
                ax.plot(wlen, model * self.configuration.plot_kwargs["y_axis_scaling"],
                        color="#00d2f3", alpha=1 / self.configuration.plot_kwargs["nsample"] + 0.1, linewidth=0.2,
                        marker=None)
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Setup best fit spectrum
            # First get the fit for each dataset for the residual plots
            # self.log_likelihood(samples_use[best_fit_index, :-1], 0, 0)
            # Then get the full wavelength range
            bf_wlen, bf_spectrum = self.get_best_fit_model(
                samples_use[:-1, best_fit_index],
                parameters_read,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                refresh=refresh
            )
            chi2 = self.get_reduced_chi2_from_model(
                wlen_model=bf_wlen,
                spectrum_model=bf_spectrum,
                subtract_n_parameters=True,
                verbose=True,
                show_chi2=True
            )
            ax.plot(bf_wlen,
                    bf_spectrum * self.configuration.plot_kwargs["y_axis_scaling"],
                    marker=None,
                    label=rf"Best fit, $\chi^{2}=${chi2:.2f}",
                    linewidth=4,
                    alpha=0.5,
                    color='r')

            for name, dd in self.configuration.data.items():
                fig, ax = plot_data(fig, ax, dd,
                                    resolution=self.configuration.plot_kwargs["resolution"],
                                    scaling=self.configuration.plot_kwargs["y_axis_scaling"])
            ax.set_xlabel('Wavelength [micron]')
            ax.set_ylabel(self.configuration.plot_kwargs["spec_ylabel"])
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(os.path.join(path, self.configuration.retrieval_name + '_sampled.pdf'), bbox_inches=0.)
        else:
            fig = None
            ax = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig, ax

    def plot_pt(self, sample_dict, parameters_read, contribution=False, refresh=False, model_generating_function=None,
                prt_reference=None, mode='bestfit'):
        """
        Plot the PT profile with error contours

        Args:
            sample_dict : np.ndarray
                posterior samples from pynmultinest outputs (post_equal_weights)
            parameters_read : List
                List of free parameters as read from the output file.
            contribution : bool
                Weight the opacity of the pt profile by the emission contribution function,
                and overplot the contribution curve.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            mode : str
                'bestfit' or 'median', indicating which set of values should be used to calculate the contribution
                function.

        Returns:
            fig : matplotlib.figure
            ax : matplotlib.axes
        """
        import matplotlib.pyplot as plt

        if not self.use_mpi or rank == 0:

            print("\nPlotting PT profiles")
            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'

            # Choose what samples we want to use
            samples_use = cp.copy(sample_dict[self.configuration.retrieval_name])

            # This is probably obsolete
            # i_p = 0
            self.pt_plot_mode = True
            """for pp in self.configuration.parameters:
                if self.configuration.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.configuration.parameters[pp].name:
                            samples_use[:, i_p] = sample_dict[self.configuration.retrieval_name][:, i_s]

                    i_p += 1"""

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.configuration.amr
            self.configuration.amr = False
            temps = []

            pressures = self.configuration.pressures  # prevent eventual reference before assignment
            if amr:
                for name, dd in self.configuration.data.items():
                    if dd.external_radtrans_reference is None:
                        dd.radtrans_object._pressures = pressures * 1e6
            press_file = f"{self.get_base_figure_name()}_pressures"
            temp_file = f"{self.get_base_figure_name()}_temps"

            if os.path.exists(press_file + ".npy") and os.path.exists(temp_file + ".npy") and not refresh:
                pressures = np.load(press_file + ".npy")
                temps_sort = np.load(temp_file + ".npy")
            else:
                length_samples = np.shape(samples_use)[1]
                for i_sample in range(length_samples):
                    press, t = self.log_likelihood(samples_use[:-1, i_sample], 0, 0)

                    if t is None:
                        continue

                    temps.append(t)

                temps = np.array(temps, dtype=float)
                temps_sort = np.sort(temps, axis=0)
                np.save(press_file, pressures)
                np.save(temp_file, temps_sort)

            len_samp = temps_sort.shape[0]
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.fill_betweenx(pressures,
                             x1=temps_sort[0, :],
                             x2=temps_sort[-1, :],
                             color='cyan',
                             label='all',
                             zorder=0)
            ax.fill_betweenx(pressures,
                             x1=temps_sort[int(len_samp * (0.5 - 0.997 / 2.)), :],
                             x2=temps_sort[int(len_samp * (0.5 + 0.997 / 2.)), :],
                             color='brown',
                             label='3 sig',
                             zorder=1)
            ax.fill_betweenx(pressures,
                             x1=temps_sort[int(len_samp * (0.5 - 0.95 / 2.)), :],
                             x2=temps_sort[int(len_samp * (0.5 + 0.95 / 2.)), :],
                             color='orange',
                             label='2 sig',
                             zorder=2)
            ax.fill_betweenx(pressures,
                             x1=temps_sort[int(len_samp * (0.5 - 0.68 / 2.)), :],
                             x2=temps_sort[int(len_samp * (0.5 + 0.68 / 2.)), :],
                             color='red',
                             label='1 sig',
                             zorder=3)

            '''
            np.savetxt('pRT_PT_envelopes.dat',
                    np.column_stack((pressures,
                                        temps_sort[int(len_samp * (0.5 - 0.997 / 2.)), :],
                                        temps_sort[int(len_samp * (0.5 - 0.95 / 2.)), :],
                                        temps_sort[int(len_samp * (0.5 - 0.68 / 2.)), :],
                                        temps_sort[int(len_samp * 0.5), :],
                                        temps_sort[int(len_samp * (0.5 + 0.68 / 2.)), :],
                                        temps_sort[int(len_samp * (0.5 + 0.95 / 2.)), :],
                                        temps_sort[int(len_samp * (0.5 + 0.997 / 2.)), :])))
            '''
            # Plot limits
            if self.configuration.plot_kwargs["temp_limits"] is not None:
                tlims = self.configuration.plot_kwargs["temp_limits"]
                ax.set_xlim(self.configuration.plot_kwargs["temp_limits"])
            else:
                tlims = (np.min(temps) * 0.97, np.max(temps) * 1.03)
                ax.set_xlim(tlims)

            # Check if we're weighting by the contribution function.
            if contribution:
                self.pt_plot_mode = False
                if mode.strip('-').strip("_").lower() == "bestfit":
                    # Get best-fit index
                    log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                    self.get_max_likelihood_params(samples_use[:-1, best_fit_index], parameters_read)
                    sample_use = samples_use[:-1, best_fit_index]
                elif mode.lower() == "median":
                    med_param, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)
                else:
                    sample_use = None  # TODO prevent reference before assignment

                bf_wlen, bf_spectrum, bf_contribution = self.get_best_fit_model(
                    sample_use,
                    parameters_read,
                    model_generating_function=model_generating_function,
                    prt_reference=prt_reference,
                    refresh=refresh,
                    contribution=True,
                    mode=mode
                )
                nu = cst.c / bf_wlen

                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = bf_spectrum * diff_nu / np.sum(bf_spectrum * diff_nu)

                if self.test_plotting:
                    plt.clf()
                    plt.plot(bf_wlen / 1e-4, spectral_weights)
                    plt.show()
                    print(np.shape(bf_contribution))

                pressure_weights = np.diff(np.log10(pressures))
                weights = np.ones_like(pressures)
                weights[:-1] = pressure_weights
                weights[-1] = weights[-2]
                weights = weights / np.sum(weights)
                weights = weights.reshape(len(weights), 1)

                contr_em = bf_contribution / weights

                # This probably doesn't need to be in a loop
                for i_str in range(bf_contribution.shape[0]):
                    contr_em[i_str, :] = bf_contribution[i_str, :] * spectral_weights

                contr_em = np.sum(bf_contribution, axis=1)
                contr_em = contr_em / np.sum(contr_em)

                if self.test_plotting:
                    plt.clf()
                    plt.yscale('log')
                    plt.ylim([pressures[-1], pressures[0]])
                    plt.plot(contr_em, pressures)
                    plt.show()

                #####
                # Use contribution function to weigh alphas
                #####
                contr_em_weigh = contr_em / np.max(contr_em)
                from scipy.interpolate import interp1d
                contr_em_weigh_intp = interp1d(pressures, contr_em_weigh)

                yborders = pressures
                for i_p in range(len(yborders) - 1):
                    mean_press = (yborders[i_p + 1] + yborders[i_p]) / 2.
                    # print(1.-contr_em_weigh_intp(mean_press))
                    ax.fill_between(tlims,
                                    yborders[i_p + 1],
                                    yborders[i_p],
                                    color='white',
                                    alpha=min(1. - contr_em_weigh_intp(mean_press), 0.9),
                                    linewidth=0,
                                    rasterized=True,
                                    zorder=4)

                ax.plot(contr_em_weigh * (
                        tlims[1] - tlims[0])
                        + tlims[0],
                        pressures, '--',
                        color='black',
                        linewidth=1.,
                        label='Spectrally weighted contribution',
                        zorder=5)

                # np.savetxt('spectrally_weighted_constribution.dat', np.column_stack((pressures, contr_em_weigh)))
            ax.set_yscale('log')
            try:
                ax.set_ylim(self.configuration.plot_kwargs["press_limits"])
            except Exception:  # TODO find what is expected here
                ax.set_ylim([pressures[-1] * 1.03, pressures[0] / 1.03])

            # Labelling and output
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            ax.legend(loc='best')
            plt.savefig(f"{self.get_base_figure_name()}_PT_envelopes.pdf",
                        bbox_inches='tight')
            self.configuration.amr = amr
            if amr:
                for name, dd in self.configuration.data.items():
                    if dd.external_radtrans_reference is None:
                        dd.radtrans_object._pressures = self.configuration.amr_pressure * 1e6
        else:
            fig = None
            ax = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig, ax

    def plot_corner(self, sample_dict, parameter_dict, parameters_read, plot_best_fit=True, true_values=None, **kwargs):
        """
        Make the corner plots.

        Args:
            sample_dict : Dict
                Dictionary of samples from PMN outputs, with keys being retrieval names
            parameter_dict : Dict
                Dictionary of parameters for each of the retrievals to be plotted.
            parameters_read : List
                Used to plot correct parameters, as some in self.configuration.parameters are not free, and
                aren't included in the PMN outputs
            plot_best_fit : bool
                If true, plot vertical lines to indicate the maximum likelihood parameter values.
            true-values : np.ndarray
                An array of values for each plotted parameter, where a vertical line will be plotted
                for each value. Can be used to indicate true values if retrieving on synthetic data,
                or to overplot additional measurements.
            kwargs : dict
                Each kwarg can be one of the kwargs used in corner.corner. These can be used to adjust
                the title_kwargs,label_kwargs,hist_kwargs, hist2d_kawargs or the contour kwargs. Each
                kwarg must be a dictionary with the arguments as keys and values as the values.
            true_values :
                # TODO complete docstring
        """
        from petitRADTRANS.plotlib.plotlib import contour_corner

        if not self.use_mpi or rank == 0:

            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'
            print("\nMaking corner plot")
            sample_use_dict = {}
            p_plot_inds = {}
            p_ranges = {}
            p_use_dict = {}

            for name, params in parameter_dict.items():
                samples_use = cp.copy(sample_dict[name]).T
                parameters_use = cp.copy(params)
                parameter_plot_indices = []
                parameter_ranges = []
                i_p = 0

                for pp in parameters_read:
                    if self.configuration.parameters[pp].plot_in_corner:
                        parameter_plot_indices.append(i_p)

                    if self.configuration.parameters[pp].corner_label is not None:
                        parameters_use[i_p] = self.configuration.parameters[pp].corner_label

                    if self.configuration.parameters[pp].corner_transform is not None:
                        samples_use[:, i_p] = \
                            self.configuration.parameters[pp].corner_transform(samples_use[:, i_p])
                    parameter_ranges.append(self.configuration.parameters[pp].corner_ranges)

                    i_p += 1

                p_plot_inds[name] = parameter_plot_indices
                p_ranges[name] = parameter_ranges
                p_use_dict[name] = parameters_use
                sample_use_dict[name] = cp.copy(samples_use)

            output_file = self.get_base_figure_name() + '_corner_plot.pdf'
            # from Plotting
            fig = contour_corner(
                sample_use_dict,
                p_use_dict,
                output_file,
                parameter_plot_indices=p_plot_inds,
                parameter_ranges=p_ranges,
                prt_plot_style=self.prt_plot_style,
                plot_best_fit=plot_best_fit,
                true_values=true_values,
                **kwargs
            )
        else:
            fig = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig

    def plot_data(self, yscale='linear'):
        """
        Plot the data used in the retrieval.
        """
        import matplotlib.pyplot as plt

        if not self.use_mpi or rank == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, dd in self.configuration.data.items():
                if dd.photometry:
                    wlen = np.mean(dd.photometric_bin_edges)
                else:
                    wlen = dd.wavelengths

                # If the data has an arbitrary retrieved scaling factor
                scale = 1.0
                errscale = 1.0
                offset = 0.0
                spectrum = dd.spectrum
                error = dd.uncertainties
                if self.configuration.run_mode == 'evaluate':
                    if dd.scale:
                        scale = self.best_fit_parameters[f"{name}_scale_factor"].value

                    if dd.scale_err:
                        errscale = self.best_fit_parameters[f"{name}_scale_factor"].value
                        error = error * errscale

                    if dd.offset_bool:
                        offset = self.best_fit_parameters[f"{name}_offset"].value

                spectrum = (spectrum * scale) - offset
                ax.errorbar(wlen, spectrum, yerr=error, label=name, marker='o')
                ax.set_yscale(yscale)
            ax.legend(fontsize=6)

            plt.savefig(self.get_base_figure_name() + "_Data.pdf")

        if self.use_mpi and comm is not None:
            comm.barrier()

    def plot_contribution(self, samples_use, parameters_read, model_generating_function=None, prt_reference=None,
                          log_scale_contribution=False, n_contour_levels=30, refresh=True, mode='bestfit'):
        """
        Plot the contribution function of the bestfit or median model from a retrieval. This plot indicates the
        relative contribution from each wavelength and each pressure level in the atmosphere to the spectrum.

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            log_scale_contribution : bool
                If true, take the log10 of the contribution function to visualise faint features.
            n_contour_levels : int
                Number of contour levels to pass to the matplotlib contourf function.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                'bestfit' or 'median', indicating which set of values should be used to calculate the contribution
                function.

        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data
        """
        import matplotlib.pyplot as plt

        if not self.use_mpi or rank == 0:

            self.evaluate_sample_spectra = False
            if not self.configuration.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.configuration.run_mode = 'evaluate'
            print("\nPlotting Best-fit contribution function")

            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.configuration.amr
            self.configuration.amr = False
            # Store old pressure array so that we can put it back later.
            p_global_keep = self.configuration.pressures
            pressures = self.configuration.pressures  # prevent eventual reference before assignment

            if amr:
                for name, dd in self.configuration.data.items():
                    dd.radtrans_object.setup_opa_structure(pressures)

            # Calculate the temperature structure
            self.pt_plot_mode = True
            pressures, t = self.log_likelihood(samples_use[:-1, best_fit_index], 0, 0)
            self.pt_plot_mode = False

            # Calculate the best fit/median spectrum contribution
            if mode.strip('-').strip("_").lower() == "bestfit":
                # Get best-fit index
                log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.get_max_likelihood_params(samples_use[:-1, best_fit_index], parameters_read)
                sample_use = samples_use[:-1, best_fit_index]
            elif mode.lower() == "median":
                med_params, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)
            else:
                sample_use = None  # TODO prevent reference before assignment

            bf_wlen, bf_spectrum, bf_contribution = self.get_best_fit_model(
                sample_use,
                parameters_read,
                model_generating_function=model_generating_function,
                prt_reference=prt_reference,
                refresh=refresh,
                contribution=True,
                mode=mode
            )
            # Normalization
            index = (bf_contribution < 1e-16) & np.isnan(bf_contribution)
            bf_contribution[index] = 1e-16

            pressure_weights = np.diff(np.log10(pressures))
            weights = np.ones_like(pressures)
            weights[:-1] = pressure_weights
            weights[-1] = weights[-2]
            weights = weights / np.sum(weights)
            weights = weights.reshape(len(weights), 1)

            x, y = np.meshgrid(bf_wlen, pressures)

            # Plotting
            fig, ax = plt.subplots()
            if log_scale_contribution:
                plot_cont = -np.log10(bf_contribution * self.configuration.plot_kwargs["y_axis_scaling"] / weights)
                label = "-Log Weighted Flux"
            else:
                plot_cont = bf_contribution * self.configuration.plot_kwargs["y_axis_scaling"] / weights
                label = "Weighted Flux"

            im = ax.contourf(x,
                             y,
                             plot_cont,
                             n_contour_levels,
                             cmap='magma')
            ax.set_xlabel(self.configuration.plot_kwargs["spec_xlabel"])
            ax.set_ylabel("Pressure [bar]")
            ax.set_xscale(self.configuration.plot_kwargs["xscale"])
            ax.set_yscale("log")
            ax.set_ylim(pressures[-1] * 1.03, pressures[0] / 1.03)
            plt.colorbar(im, ax=ax, label=label)

            plt.savefig(
                f"{self.get_base_figure_name()}_{mode}_contribution.pdf",
                bbox_inches='tight'
            )

            # Restore the correct pressure arrays.
            # *1e6 for units (cgs from bar)
            self.configuration.pressures = p_global_keep
            self.configuration.amr = amr
            if amr:
                for name, dd in self.configuration.data.items():
                    dd.radtrans_object.setup_opa_structure(self.configuration.amr_pressure * 1e6)
        else:
            fig = None
            ax = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig, ax

    def plot_abundances(self,
                        samples_use,
                        parameters_read,
                        species_to_plot=None,
                        contribution=False,
                        refresh=True,
                        model_generating_function=None,
                        prt_reference=None,
                        mode='bestfit',
                        sample_posteriors=False,
                        volume_mixing_ratio=False):
        """
        Plot the abundance profiles in mass fractions or volume mixing ratios as a function of pressure.

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            species_to_plot : list
                A list of which molecular species to include in the plot.
            contribution : bool
                If true, overplot the emission or transmission contribution function.
            prt_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            model_generating_function : (callable, optional):
                A function that returns the wavelength and spectrum, and takes a pRT_Object and the
                current set of parameters stored in self.configuration.parameters. This should be the same model
                function used in the retrieval.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                'bestfit' or 'median', indicating which set of values should be used for plotting the abundances.
            sample_posteriors : bool
                If true, sample the posterior distribtions to calculate confidence intervales for the retrieved
                abundance profiles.
            volume_mixing_ratio : bool
                If true, plot in units of volume mixing ratio (number fraction) instead of mass fractions.

        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data
        """
        import matplotlib.pyplot as plt

        if not self.use_mpi or rank == 0:
            print("\nPlotting Abundances profiles")
            # if self.prt_plot_style:
            #     import petitRADTRANS.retrieval.plot_style as ps  # TODO never used

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.configuration.amr
            self.configuration.amr = False
            # Store old pressure array so that we can put it back later.
            p_global_keep = self.configuration.pressures
            pressures = self.configuration.pressures  # prevent eventual reference before assignment
            if amr:
                for name, dd in self.configuration.data.items():
                    dd.radtrans_object.setup_opa_structure(pressures)

            self.pt_plot_mode = True
            if mode.strip('-').strip("_").lower() == "bestfit":
                # Get best-fit index
                log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.get_max_likelihood_params(samples_use[:-1, best_fit_index], parameters_read)
                sample_use = samples_use[:-1, best_fit_index]
            elif mode.lower() == "median":
                med_params, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)
            else:
                sample_use = None  # TODO prevent reference before assignment

            pressures, t = self.log_likelihood(sample_use, 0, 0)
            self.pt_plot_mode = False

            # Check if we're only plotting a few species
            if species_to_plot is None:
                if self.configuration.data[
                        self.configuration.plot_kwargs["take_PTs_from"]].external_radtrans_reference is not None:
                    species_to_plot = self.configuration.data[
                        self.configuration.data[
                            self.configuration.plot_kwargs["take_PTs_from"]
                        ].external_radtrans_reference
                    ].radtrans_object.line_species
                else:
                    species_to_plot = self.configuration.data[
                        self.configuration.plot_kwargs["take_PTs_from"]
                    ].radtrans_object.line_species

            # Set up colours - abundances usually have a lot of species,
            # so let's use the default matplotlib colour scheme rather
            # than the pRT colours.
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            # Figure
            fig, ax = plt.subplots(figsize=(12, 7))

            # Check to see if we're plotting contour regions
            if sample_posteriors:
                abundances = {}
                for species in species_to_plot:
                    abundances[species.split()[0]] = []

                # Go through EVERY sample to find the abundance distribution.
                # Very slow.
                for sample in samples_use:
                    if volume_mixing_ratio:
                        abund_dict, mmw = self.get_volume_mixing_ratios(sample[:-1], parameters_read)
                    else:
                        abund_dict, mmw = self.get_mass_fractions(sample[:-1], parameters_read)
                    for species in species_to_plot:
                        abundances[species.split(self.configuration.data.resolving_power_str)[0]].append(
                            abund_dict[species.split(self.configuration.data.resolving_power_str)[0]]
                        )

                # Plot median and 1sigma contours
                for i, species in enumerate(species_to_plot):
                    low, med, high = np.quantile(
                        np.array(abundances[species.split(self.configuration.data.resolving_power_str)[0]]),
                        [0.159, 0.5, 0.841],
                        axis=0
                    )
                    ax.plot(med,
                            pressures,
                            label=species.split('_')[0],
                            color=colors[i % len(colors)],
                            zorder=0,
                            linewidth=2)
                    ax.plot(low,
                            pressures,
                            color=colors[i % len(colors)],
                            linewidth=0.4,
                            zorder=0)
                    ax.plot(high,
                            pressures,
                            color=colors[i % len(colors)],
                            linewidth=0.4,
                            zorder=0)

                    ax.fill_betweenx(pressures,
                                     x1=low,
                                     x2=high,
                                     color=colors[i % len(colors)],
                                     alpha=0.15,
                                     zorder=-1)
            else:
                # Plot only the best fit abundances.
                # Default to this for speed.
                if volume_mixing_ratio:
                    abund_dict, mmw = self.get_volume_mixing_ratios(sample_use, parameters_read)
                else:
                    abund_dict, mmw = self.get_mass_fractions(sample_use, parameters_read)
                for i, spec in enumerate(species_to_plot):
                    ax.plot(abund_dict[spec.split(self.configuration.data.resolving_power_str)[0]],
                            pressures,
                            label=spec.split('_')[0],
                            color=colors[i % len(colors)],
                            zorder=0,
                            linewidth=2)

            # Check to see if we're weighting by the emission contribution.
            if contribution:
                bf_wlen, bf_spectrum, bf_contribution = self.get_best_fit_model(
                    sample_use,
                    parameters_read,
                    model_generating_function=model_generating_function,
                    prt_reference=prt_reference, refresh=refresh,
                    contribution=True
                )
                nu = cst.c / bf_wlen
                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = bf_spectrum * diff_nu / np.sum(bf_spectrum * diff_nu)

                if self.test_plotting:
                    plt.clf()
                    plt.plot(bf_wlen / 1e-4, spectral_weights)
                    plt.show()
                    print(np.shape(bf_contribution))

                pressure_weights = np.diff(np.log10(pressures))
                weights = np.ones_like(pressures)
                weights[:-1] = pressure_weights
                weights[-1] = weights[-2]
                weights = weights / np.sum(weights)
                weights = weights.reshape(len(weights), 1)

                contr_em = bf_contribution / weights

                # This probably doesn't need to be in a loop
                for i_str in range(bf_contribution.shape[0]):
                    contr_em[i_str, :] = bf_contribution[i_str, :] * spectral_weights

                contr_em = np.sum(bf_contribution, axis=1)
                contr_em = contr_em / np.sum(contr_em)

                if self.test_plotting:
                    plt.clf()
                    plt.yscale('log')
                    plt.ylim([pressures[-1], pressures[0]])
                    plt.plot(contr_em, pressures)
                    plt.show()

                #####
                # Use contribution function to weigh alphas
                #####

                contr_em_weigh = contr_em / np.max(contr_em)
                from scipy.interpolate import interp1d
                contr_em_weigh_intp = interp1d(pressures, contr_em_weigh)
                yborders = pressures
                for i_p in range(len(yborders) - 1):
                    mean_press = (yborders[i_p + 1] + yborders[i_p]) / 2.
                    # print(1.-contr_em_weigh_intp(mean_press))
                    ax.fill_between([1e-7, 3],
                                    yborders[i_p + 1],
                                    yborders[i_p],
                                    color='white',
                                    alpha=min(1. - contr_em_weigh_intp(mean_press), 0.9),
                                    linewidth=0,
                                    rasterized=True,
                                    zorder=100)

                ax.plot(
                    contr_em_weigh * (3 - 1e-7) + 1e-7,
                    pressures, '--',
                    color='black',
                    linewidth=1.,
                    zorder=120,
                    label='Contribution'
                )
            xlabel = "Mass Fraction Abundance"
            if volume_mixing_ratio:
                xlabel = "Volume Mixing Ratio"

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Pressure [bar]")
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=12)

            ax.invert_yaxis()
            ax.set_xlim(8e-13, 3)
            ax.set_axisbelow(False)
            ax.tick_params(zorder=2)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
            plt.tight_layout()

            if not sample_posteriors:
                plt.savefig(
                    self.get_base_figure_name() + '_' + mode + '_abundance_profiles.pdf',
                    bbox_inches='tight'
                )
            else:
                plt.savefig(
                    self.get_base_figure_name() + '_sampled_abundance_profiles.pdf',
                    bbox_inches='tight'
                )
            # Restore the correct pressure arrays.
            self.configuration.pressures = p_global_keep
            self.configuration.amr = amr
            if amr:
                for name, dd in self.configuration.data.items():
                    dd.radtrans_object.setup_opa_structure(self.configuration.amr_pressure * 1e6)
        else:
            fig = None
            ax = None

        if self.use_mpi and comm is not None:
            comm.barrier()

        return fig, ax
