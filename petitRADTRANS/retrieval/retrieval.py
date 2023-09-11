# Input / output, general run definitions
import sys
import os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"
# Read external packages
import numpy as np
import copy as cp
import pymultinest
import json
import logging
from scipy.stats import binned_statistic

# MPI Multiprocessing
RANK = 0
COMM = None
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except ImportError:
    logging.warning("MPI is required to run retrievals across multiple cores. Using single core mode only!")

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

# Read own packages
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from .parameter import Parameter
from .data import Data
from .plotting import plot_specs,plot_data,contour_corner
from .rebin_give_width import rebin_give_width as rgw
from .util import bin_species_exok, mass_to_number


class Retrieval:
    """
    This class implements the retrieval method using petitRADTRANS and pymultinest.
    A RetrievalConfig object is passed to this class to describe the retrieval data, parameters
    and priors. The run() method then uses pymultinest to sample the parameter space, producing
    posterior distributions for parameters and bayesian evidence for models.
    Various useful plotting functions have also been included, and can be run once the retrieval is
    complete.

    Args:
        run_definition : RetrievalConfig
            A RetrievalConfig object that describes the retrieval to be run. This is the user
            facing class that must be setup for every retrieval.
        output_dir : Str
            The directory in which the output folders should be written
        sample_spec : Bool
            Produce plots and data files for random samples drawn from the outputs of pymultinest.
        ultranest : bool
            If true, use Ultranest sampling rather than pymultinest. Provides a more accurate evidence estimate,
            but is significantly slower.
        bayes_factor_species : Str
            A pRT species that should be removed to test for the bayesian evidence for its presence.
        corner_plot_names : List(Str)
            List of additional retrieval names that should be included in the corner plot.
        short_names : List(Str)
            For each corner_plot_name, a shorter name to be included when plotting.
        pRT_plot_style : Bool
            Use the petitRADTRANS plotting style as described in plot_style.py. Recommended to
            turn this parameter to false if you want to use interactive plotting, or if the
            test_plotting parameter is True.
        test_plotting : Bool
            Only use when running locally. A boolean flag that will produce plots
            for each sample when pymultinest is run.
    """

    def __init__(self,
                 run_definition,
                 output_dir="",
                 use_MPI = False,
                 sample_spec=False,
                 ultranest=False,
                 bayes_factor_species=None,
                 corner_plot_names=None,
                 short_names=None,
                 pRT_plot_style=True,
                 test_plotting=False,):
        self.rd = run_definition

        print(f"Starting retrieval {self.rd.retrieval_name}")
        if len(self.rd.line_species) < 1:
            logging.warning("There are no line species present in the run definition!")

        # Maybe inherit from retrieval config class?
        self.retrieval_name = self.rd.retrieval_name
        self.data = self.rd.data
        self.run_mode = self.rd.run_mode
        self.parameters = self.rd.parameters
        self.use_MPI = use_MPI
        self.output_dir = output_dir
        if self.output_dir != "" and not self.output_dir.endswith("/"):
            self.output_dir += "/"

        self.remove_species = bayes_factor_species
        self.corner_files = corner_plot_names
        if self.corner_files is None:
            self.corner_files = [self.retrieval_name]

        self.short_names = short_names

        # Plotting variables
        self.best_fit_specs = {}
        self.best_fit_params = {}
        self.chi2 = None
        self.posterior_sample_specs = {}
        self.plotting = test_plotting
        self.PT_plot_mode = False
        self.evaluate_sample_spectra = sample_spec

        # Pymultinest stuff
        self.ultranest = ultranest
        self.analyzer = None
        self.n_live_points = 4000
        self.sampling_efficiency = False
        self.resume = False
        self.const_efficiency_mode = False

        self.samples = {}  #: The samples produced by pymultinest.
        self.param_dict = {}
        # Set up pretty plotting
        if pRT_plot_style:
            # import petitRADTRANS.retrieval.plot_style  # commented to avoid breaking of mpl
            pass
        self.prt_plot_style = pRT_plot_style

        # Path to input opacities
        self.path = os.environ.get("pRT_input_data_path")

        if self.path is None:
            raise OSError(f"Path to input data not specified!\n"
                          f"Please set pRT_input_data_path variable in .bashrc / .bash_profile or specify path via\n"
                          f">>> import os"
                          f">>> os.environ['pRT_input_data_path'] = 'absolute/path/of/the/folder/input_data'\n"
                          f"before creating a Radtrans object or loading the nat_cst module.\n"
                          f"(this will become unnecessary in a future update)"
                          )
        if not self.path.endswith("/"):
            self.path += "/"
        # Setup Directories
        if not os.path.isdir(self.output_dir + 'out_PMN/'):
            os.makedirs(self.output_dir + 'out_PMN', exist_ok=True)
        if not os.path.isdir(self.output_dir + 'evaluate_' + self.retrieval_name + '/'):
            os.makedirs(self.output_dir + 'evaluate_' + self.retrieval_name, exist_ok=True)

        # Setup pRT Objects for each data structure.
        print("Setting up PRT Objects")
        self.setup_data()
        try:
            self.generate_retrieval_summary()
        except ValueError as e:  # TODO check if ValueError was expected here
            print(f"Could not generate summary file! Error was: {str(e)}")

    def run(self,
            sampling_efficiency=0.8,
            const_efficiency_mode=False,
            n_live_points=4000,
            log_z_convergence=0.5,
            step_sampler=False,
            warmstart_max_tau=0.5,
            n_iter_before_update=50,
            resume=True,
            max_iters=0,
            frac_remain=0.1,
            importance_nested_sampling = True,
            Lepsilon=0.3,
            error_checking = True):
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
            Lepsilon : float
                Ultranest convergence criterion. Use with noisy likelihoods. Halts integration if live points are wihin Lepsilon.
            resume : bool
                Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
            error_checking : bool
                Test the model generating function for typical errors. ONLY TURN THIS OFF IF YOU KNOW WHAT YOU'RE DOING!
        """
        import pymultinest
        self.n_live_points = n_live_points
        self.sampling_efficiency = sampling_efficiency
        self.resume = resume
        self.const_efficiency_mode = const_efficiency_mode

        if error_checking:
            self._error_check_model_function()
        else:
            print("Error checking is turned off!! You might overwrite your retrieval output files!")

        if self.ultranest:
            self._run_ultranest(n_live_points=n_live_points,
                                log_z_convergence=log_z_convergence,
                                step_sampler=step_sampler,
                                warmstart_max_tau=warmstart_max_tau,
                                resume=resume,
                                max_iters=max_iters,
                                frac_remain=frac_remain,
                                Lepsilon=Lepsilon)
            return

        if const_efficiency_mode and sampling_efficiency > 0.1:
            logging.warning("Sampling efficiency should be ~ 0.05 if you're using constant efficiency mode!")

        prefix = self.output_dir + 'out_PMN/' + self.retrieval_name + '_'

        if len(self.output_dir + 'out_PMN/') > 200:
            logging.error("PyMultinest requires output directory names to be <200 characters.")
            sys.exit(3)
        # How many free parameters?
        n_params = 0
        free_parameter_names = []

        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                free_parameter_names.append(self.parameters[pp].name)
                n_params += 1

        if self.run_mode == 'retrieval':
            print("Starting retrieval: " + self.retrieval_name + '\n')
            with open(self.output_dir + 'out_PMN/' + self.retrieval_name + '_params.json', 'w') as f:
                json.dump(free_parameter_names, f)

            pymultinest.run(
                LogLikelihood=self.log_likelihood,
                Prior=self.prior,
                n_dims=n_params,
                const_efficiency_mode=const_efficiency_mode,
                n_live_points=n_live_points,
                evidence_tolerance=log_z_convergence,  # default value is 0.5
                sampling_efficiency=sampling_efficiency,  # default value is 0.8
                n_iter_before_update=n_iter_before_update,  # default value is 100
                outputfiles_basename=prefix,
                verbose=True,
                resume=resume,
                max_iter=max_iters
            )

        # Analyze the output data
        self.analyzer = pymultinest.Analyzer(n_params=n_params,
                                             outputfiles_basename=prefix)
        s = self.analyzer.get_stats()

        self.run_mode = 'evaluate'
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
                       Lepsilon):
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
            Lepsilon : float
                Ultranest convergence criterion. Use with noisy likelihoods. Halts integration if live points are wihin Lepsilon.
            resume : bool
                Continue existing retrieval. If FALSE THIS WILL OVERWRITE YOUR EXISTING RETRIEVAL.
        """

        logging.warning("ultranest mode is still in development. Proceed with caution")
        try:
            import ultranest as un
            from ultranest.mlfriends import RobustEllipsoidRegion
        except ImportError:
            logging.error("Could not import ultranest. Exiting.")
            sys.exit(1)
        if self.run_mode == 'retrieval':
            print("Starting retrieval: " + self.retrieval_name + '\n')
            # How many free parameters?
            n_params = 0
            free_parameter_names = []
            for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    free_parameter_names.append(self.parameters[pp].name)
                    n_params += 1

            if max_iters == 0:
                max_iters = None
            sampler = un.ReactiveNestedSampler(free_parameter_names,
                                               self.log_likelihood,
                                               self.prior_ultranest,
                                               log_dir=self.output_dir + "out_" + self.retrieval_name,
                                               warmstart_max_tau=warmstart_max_tau,
                                               resume=resume)
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
                        Lepsilon=Lepsilon,
                        region_class=RobustEllipsoidRegion)
            sampler.print_results()
            sampler.plot_corner()

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
        with open(
                self.output_dir + "evaluate_" + self.retrieval_name + "/" + self.retrieval_name + "_ret_summary.txt",
                "w+"
        ) as summary:
            from datetime import datetime
            summary.write(self.retrieval_name + '\n')
            summary.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + '\n')
            summary.write(self.output_dir + '\n')
            summary.write(f"Live points: {self.n_live_points}\n\n")
            summary.write("Fixed Parameters\n")

            for key, value in self.parameters.items():
                if key in ['pressure_simple', 'pressure_width', 'pressure_scaling']:
                    continue

                if not value.is_free_parameter:
                    if isinstance(value.value, float):
                        summary.write(f"    {key} = {value.value:.3f}\n")
                    else:
                        summary.write(f"    {key} = {value.value}\n")

            summary.write('\n')
            summary.write("Free Parameters, Prior^-1(0), Prior^-1(1)\n")

            for key, value in self.parameters.items():
                if value.is_free_parameter:
                    low = value.transform_prior_cube_coordinate(0.0000001)
                    high = value.transform_prior_cube_coordinate(0.9999999)

                    if value.corner_transform is not None:
                        low = value.corner_transform(low)
                        high = value.corner_transform(high)
                    summary.write(f"    {key} = {low:3f}, {high:3f}\n")

            summary.write('\n')
            summary.write("Data\n")

            for name, dd in self.data.items():
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
                    summary.write(f"    offset = True\n")

                if dd.distance is not None:
                    summary.write(f"    distance = {dd.distance}\n")

                if dd.data_resolution is not None:
                    summary.write(f"    data resolution = {dd.data_resolution}\n")

                if dd.model_resolution is not None:
                    summary.write(f"    model resolution = {dd.model_resolution}\n")

                if dd.external_pRT_reference is not None:
                    summary.write(f"    external_pRT_reference = {dd.external_pRT_reference}\n")

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
                for key, value in self.parameters.items():
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

            if self.run_mode == 'evaluate':
                summary.write("Best Fit Parameters\n")
                self.get_samples(self.output_dir)
                samples_use = self.samples[self.retrieval_name]
                parameters_read = self.param_dict[self.retrieval_name]
                # Get best-fit index
                logL ,best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.get_max_likelihood_params(samples_use[best_fit_index,:-1],parameters_read)
                chi2_wlen = self.get_reduced_chi2(samples_use[best_fit_index],subtract_n_parameters=False)
                chi2_DoF = self.get_reduced_chi2(samples_use[best_fit_index],subtract_n_parameters=True)

                    # Get best-fit index
                summary.write(f"    𝛘^2/n_wlen = {chi2_wlen:.2f}\n")
                summary.write(f"    𝛘^2/DoF = {chi2_DoF:.2f}\n")
                for key, value in self.best_fit_params.items():
                    if key in ['pressure_simple', 'pressure_width', 'pressure_scaling', 'FstarWlenMicron']:
                        continue

                    out = value.value

                    if self.parameters[key].corner_transform is not None:
                        out = self.parameters[key].corner_transform(out)

                    if out is None:
                        continue

                    if isinstance(out, float):
                        summary.write(f"    {key} = {out:.3e}\n")
                    else:
                        summary.write(f"    {key} = {out}\n")

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
        exo_k_check = False
        for name, dd in self.data.items():
            if dd.pRT_object is not None:
                continue

            # Only create if there's no other data
            # object using the same pRT object
            if dd.external_pRT_reference is None:
                if dd.opacity_mode == 'c-k' and dd.model_resolution is not None:
                    # Use ExoK to have low res models.
                    species = []
                    # Check if low res opacities already exist
                    for line in self.rd.line_species:
                        if not os.path.isdir(
                                self.path + "opacities/lines/corr_k/" + line + "_R_" + str(dd.model_resolution)
                        ):
                            species.append(line)
                    # If not, setup low-res c-k tables
                    if len(species) > 0:
                        exo_k_check = True
                        print("Exo-k should only be run on a single thread.")
                        #print("The retrieval should be run once on a single core to build the c-k\n"
                        #    "tables, and then again with multiple cores for the remainder of the retrieval.")
                        # Automatically build the entire table
                        if RANK == 0:
                            bin_species_exok(species, dd.model_resolution)
                        if COMM is not None:
                            comm.barrier()
                    species = []
                    for spec in self.rd.line_species:
                        species.append(spec + "_R_" + str(dd.model_resolution))
                else:
                    # Otherwise for 'lbl' or no model_resolution binning,
                    # we just use the default species.
                    species = cp.copy(self.rd.line_species)
                lbl_samp = None
                if dd.opacity_mode == 'lbl' and dd.model_resolution is not None:
                    lbl_samp = int(1e6 / dd.model_resolution)

                # Set up the pRT objects for the given dataset
                rt_object = Radtrans(
                    line_species=cp.copy(species),
                    rayleigh_species=cp.copy(self.rd.rayleigh_species),
                    continuum_opacities=cp.copy(self.rd.continuum_opacities),
                    cloud_species=cp.copy(self.rd.cloud_species),
                    mode=dd.opacity_mode,
                    wlen_bords_micron=dd.wlen_range_pRT,
                    do_scat_emis=self.rd.scattering,
                    lbl_opacity_sampling=lbl_samp
                )

                # Create random P-T profile to create RT arrays of the Radtrans object.
                if self.rd.AMR:
                    p = self.rd._setup_pres(scaling, width)  # TODO this function shouldn't be protected
                else:
                    p = self.rd.p_global
                rt_object.setup_opa_structure(p)
                dd.pRT_object = rt_object

    def _error_check_model_function(self):
        free_params = []
        for key, val in self.parameters.items():
            if val.is_free_parameter:
                free_params.append(key)
        cube = np.ones(len(free_params))*0.5
        self.prior(cube)

        i_p = 0  # parameter count
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                self.parameters[pp].set_param(cube[i_p])
                i_p += 1

        wlen = None
        model = None
        for name, data in self.rd.data.items():
            try:
                use_obj = data.pRT_object
                if data.external_pRT_reference is not None:
                    use_obj = self.data[data.external_pRT_reference].pRT_object
                wlen, model = data.model_generating_function(use_obj,
                                                             self.parameters,
                                                             False,
                                                             AMR=self.rd.AMR)
            except KeyError as error:
                logging.error(error)
                logging.error("There is a KeyError in your model function. Please check your parameter dictionary and abundances!")
                sys.exit(20)
            except ValueError as error:
                logging.error(error)
                logging.error("There is a ValueError in your model function. Please check that your calculations and inputs are correct!")
                sys.exit(21)
            except IndexError as error:
                logging.error(error)
                logging.error("There is an IndexError in your model function. Please check that arrays are the correct shape!")
                sys.exit(22)
            except ZeroDivisionError as error:
                logging.error(error)
                logging.error("There is an ZeroDivisionError in your model function. Don't divide by zero!")
                sys.exit(23)
            except TypeError as error:
                logging.error(error)
                logging.error("There is a TypeError in your model function. Are you sure your inputs are correct?")
                sys.exit(24)
            if wlen is None or model is None:
                logging.error("Unable to compute a spectrum! Check your inputs and your model function!")
                sys.exit(30)
        print("No errors detected in the model function!")
        return 

    def prior(self, cube, ndim=0, nparams=0):
        """
        pyMultinest Prior function. Transforms unit hypercube into physical space.
        """
        i_p = 0

        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                cube[i_p] = self.parameters[pp].get_param_uniform(cube[i_p])
                i_p += 1

    def prior_ultranest(self, cube):
        """
        pyMultinest Prior function. Transforms unit hypercube into physical space.
        """
        params = cube.copy()
        i_p = 0
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                params[i_p] = self.parameters[pp].get_param_uniform(cube[i_p])
                i_p += 1
        return params

    def log_likelihood(self, cube, ndim=0, nparam=0):
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

        Returns:
            log_likelihood : float
                The (negative) log likelihood of the model given the data.
        """
        log_likelihood = 0.
        log_prior = 0.
        additional_logl = 0.

        i_p = 0  # parameter count

        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                self.parameters[pp].set_param(cube[i_p])
                i_p += 1

        #for name, dd in self.data.items():
        #    #if dd.scale or dd.scale_error:
        #    #    try:
        #    #        dd.scale_factor = self.parameters[name + "_scale_factor"].value
        #    #    except KeyError:
        #    #        # If multiple datasets should be scaled by the same value (need to be called "Instrument_1", "Instrument_2", etc.)
        #    #        name_use = name[:-2] # cut off the _1, _2, etc.
        #    #        dd.scale_factor = self.parameters[name_use + "_scale_factor_multiple"].value

        for name, dd in self.data.items():
            # Only calculate spectra within a given
            # wlen range once
            if dd.scale or dd.scale_err:
                dd.scale_factor = self.parameters[name + "_scale_factor"].value
            if dd.offset_bool:
                dd.offset = self.parameters[name + "_offset"].value
            if name + "_b" in self.parameters.keys():
                dd.bval = self.parameters[name + "_b"].value
            
            if self.PT_plot_mode and name == self.rd.plot_kwargs['take_PTs_from']:
                # Get the PT profile
                use_obj = dd.pRT_object
                if dd.external_pRT_reference is not None:
                    use_obj = self.data[dd.external_pRT_reference].pRT_object
                pressures, temperatures = \
                    dd.model_generating_function(use_obj,
                                                 self.parameters,
                                                 self.PT_plot_mode,
                                                 AMR=self.rd.AMR)
                return pressures, temperatures
            elif self.PT_plot_mode:
                continue

            if dd.external_pRT_reference is None:
                # Compute the model
                retVal = \
                    dd.model_generating_function(dd.pRT_object,
                                                    self.parameters,
                                                    self.PT_plot_mode,
                                                    AMR=self.rd.AMR)
                if len(retVal) == 3:
                    wlen_model, spectrum_model, additional_logl = retVal
                else:
                    wlen_model, spectrum_model = retVal
                    additional_logl = 0.

                # Sanity checks on outputs
                if spectrum_model is None:
                    # -np.infs (and anything <1e-100) seemed to cause issues with Multinest.
                    # This particular check probably wasn't the underlying cause, and could 
                    # use a better default value.
                    return -1e98  # TODO why switching to 1e98 ?

                # if np.isnan(spectrum_model).any():  # TODO make it work with jagged arrays
                #     return -1e98

                # Calculate log likelihood
                # TODO uniformize convolve/rebin handling
                if not isinstance(dd.flux,float) and dd.flux.dtype == 'O':
                    if np.ndim(dd.flux) == 1:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Second dimension of data must be a function of wavelength
                        for i, data in enumerate(dd.flux):
                            log_likelihood += dd.log_likelihood_gibson(
                                spectrum_model[i][~dd.mask[i]], data, dd.flux_error[i],
                                alpha=1.0,
                                beta=1.0
                            )
                    elif np.ndim(dd.flux) == 2:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Third dimension of data must be a function of wavelength
                        for i, detector in enumerate(dd.flux):
                            for j, data in enumerate(detector):
                                log_likelihood += dd.log_likelihood_gibson(
                                    spectrum_model[i, j][~dd.mask[i, j]], data, dd.flux_error[i, j],
                                    alpha=1.0,
                                    beta=1.0
                                )
                    else:
                        raise ValueError(f"observation is an array containing object, "
                                            f"and have {np.ndim(dd.flux)} dimensions, "
                                            f"but must have 1 to 2")
                elif isinstance(dd.flux,float) or np.ndim(dd.flux) == 1:
                    # Convolution and rebin are cared of in get_chisq
                    log_likelihood += dd.get_chisq(
                            wlen_model,
                            spectrum_model, #[~dd.mask],
                            self.plotting,
                            self.parameters
                        ) + additional_logl
                elif np.ndim(dd.flux) == 2:
                    # Convolution and rebin are *not* cared of in get_log_likelihood
                    # Second dimension of data must be a function of wavelength
                    for i, data in enumerate(dd.flux):
                        log_likelihood += dd.log_likelihood_gibson(
                            spectrum_model[i, ~dd.mask[i, :]], data, dd.flux_error[i],
                            alpha=1.0,
                            beta=1.0
                        )
                elif np.ndim(dd.flux) == 3:
                    # Convolution and rebin are *not* cared of in get_log_likelihood
                    # Third dimension of data must be a function of wavelength
                    for i, detector in enumerate(dd.flux):
                        for j, data in enumerate(detector):
                            log_likelihood += dd.log_likelihood_gibson(
                                spectrum_model[i, j, ~dd.mask[i, j, :]], data, dd.flux_error[i, j],
                                alpha=1.0,
                                beta=1.0
                            )

                # Save sampled outputs if necessary.
                if self.run_mode == 'evaluate':
                    if self.evaluate_sample_spectra:
                        self.posterior_sample_specs[name] = [wlen_model, spectrum_model]
                    else:
                        # TODO: This will overwrite the best fit spectrum with
                        # whatever is ran through the loglike function. Not good.
                        np.savetxt(
                            self.output_dir + 'evaluate_' + self.retrieval_name
                            + '/model_spec_best_fit_'
                            + name.replace('/', '_').replace('.', '_') + '.dat',
                            np.column_stack((wlen_model, spectrum_model))
                        )

                        self.best_fit_specs[name] = [wlen_model, spectrum_model]                    
            else:
                # TODO what exactly is going on here? Why the double loop on the same items?
                # Definition here to avoid possible reference before assignment
                #
                # The double loop is making sure that the spectrum and wavelength are 
                # calculated before being requested by a different pRT object.
                # Since the spectrum isn't stored, we need to immediately check if 
                # any other data objects are requesting to use this spectrum, before
                # moving on to the next item. 
                spectrum_model = None
                wlen_model = None

            # Check for data using the same pRT object
            # Calculate log likelihood
            for de_name, dede in self.data.items():
                if dede.external_pRT_reference is not None:
                    if dede.scale:
                        dede.scale_factor = self.parameters[de_name + "_scale_factor"].value
                    if dede.external_pRT_reference == name:
                        if spectrum_model is None:
                            return -1e99

                        if np.isnan(spectrum_model).any():
                            return -1e99
                        log_likelihood += dede.get_chisq(
                            wlen_model,
                            spectrum_model,
                            self.plotting,
                            self.parameters
                        ) + additional_logl

        if "log_prior_weight" in self.parameters.keys():
            log_prior += self.parameters["log_prior_weight"].value

        if log_likelihood + log_prior < -9e98:
            return -1e98

        if np.abs(log_likelihood + log_prior) < 1e-98:
            return 1e-98

        if self.ultranest and np.isinf(log_likelihood + log_prior):
            return -1e98
        return log_likelihood + log_prior

    @staticmethod
    def _get_samples(ultranest, names, output_dir=None, ret_names=None):
        if ret_names is None:
            ret_names = []

        param_dict = {}
        samples = {}

        if ultranest:
            for name in names:
                samples_ = np.genfromtxt(output_dir + 'out_' + name + '/chains/equal_weighted_post.txt')
                parameters_read = open(output_dir + 'out_' + name + '/chains/weighted_post.paramnames')
                samples[name] = samples_
                param_dict[name] = parameters_read

            for name in ret_names:
                samples_ = np.genfromtxt(output_dir + 'out_' + name + '/chains/qual_weighted_post.txt')
                parameters_read = open(output_dir + 'out_' + name + '/chains/weighted_post.paramnames')
                samples[name] = samples_
                param_dict[name] = parameters_read

            return samples, param_dict

        # pymultinest
        for name in names:
            samples_ = np.genfromtxt(output_dir + 'out_PMN/' + name + '_post_equal_weights.dat')

            with open(output_dir + 'out_PMN/' + name + '_params.json', 'r') as f:
                parameters_read = json.load(f)

            samples[name] = samples_
            param_dict[name] = parameters_read

        for name in ret_names:
            samples_ = np.genfromtxt(output_dir + 'out_PMN/' + name + '_post_equal_weights.dat')

            with open(output_dir + 'out_PMN/' + name + '_params.json', 'r') as f:
                parameters_read = json.load(f)

            samples[name] = samples_
            param_dict[name] = parameters_read

        return samples, param_dict

    def get_samples(self, output_dir=None, ret_names=None):
        """
        This function looks in the given output directory and finds the post_equal_weights
        file associated with the current retrieval name.

        Args:
            output_dir : str
                Parent directory of the out_PMN/RETRIEVALNAME_post_equal_weights.dat file
            ret_names : List(str)
                A list of retrieval names to add to the sample and parameter dictionary.
                Functions the same as setting corner_files during initialisation.

        Returns:
            sample_dict : dict
                A dictionary with keys being the name of the retrieval, and values are a numpy
                ndarray containing the samples in the post_equal_weights file
            parameter_dict : dict
                A dictionary with keys being the name of the retrieval, and values are a list of names
                of the parameters used in the retrieval. The first name corresponds to the first column
                of the samples, and so on.
        """
        # TODO could be static
        if output_dir is None:
            output_dir = self.output_dir

        if ret_names is None:
            ret_names = []

        if self.ultranest:
            for name in self.corner_files:
                samples = np.genfromtxt(output_dir + 'out_' + name + '/chains/equal_weighted_post.txt')
                # TODO formatting of paramname file
                parameters_read = open(output_dir + 'out_' + name + '/chains/weighted_post.paramnames')
                self.samples[name] = samples
                self.param_dict[name] = parameters_read
            for name in ret_names:
                samples = np.genfromtxt(output_dir + 'out_' + name + '/chains/qual_weighted_post.txt')
                parameters_read = open(output_dir + 'out_' + name + '/chains/weighted_post.paramnames')
                self.samples[name] = samples
                self.param_dict[name] = parameters_read
            return self.samples, self.param_dict

        # pymultinest
        for name in self.corner_files:
            samples = np.genfromtxt(output_dir + 'out_PMN/' + name + '_post_equal_weights.dat')

            with open(output_dir + 'out_PMN/' + name + '_params.json', 'r') as f:
                parameters_read = json.load(f)

            self.samples[name] = samples
            self.param_dict[name] = parameters_read


        for name in ret_names:
            samples = np.genfromtxt(output_dir + 'out_PMN/' + name + '_post_equal_weights.dat')

            with open(output_dir + 'out_PMN/' + name + '_params.json', 'r') as f:
                parameters_read = json.load(f)
            self.samples[name] = samples
            self.param_dict[name] = parameters_read

        return self.samples, self.param_dict

    def get_max_likelihood_params(self,best_fit_params,parameters_read):
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
        self.best_fit_params = self.build_param_dict(best_fit_params, parameters_read)
        return self.best_fit_params

    def get_median_params(self,samples,parameters_read, return_array=False):
        """
        This function builds a parameter dictionary based on the median value
        of each parameter. This will update the best_fit_parameter dictionary!

        Args:
            best_fit_params : numpy.ndarray
                An array of the best fit parameter values (or any other sample)
            parameters_read : list
                A list of the free parameter names as read from the output files.
        """
        i_p = 0
        samples_use = np.zeros(len(parameters_read))
        # Take the median of each column
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(parameters_read)):
                    if parameters_read[i_s] == self.parameters[pp].name:
                        samples_use[i_p] = np.median(samples[:, i_s])
                i_p += 1
        self.best_fit_params = self.build_param_dict(samples_use,parameters_read)
        if return_array:
            return self.best_fit_params, samples_use
        return self.best_fit_params

    def get_full_range_model(self,
                             parameters,
                             model_generating_func=None,
                             ret_name=None,  # TODO remove unused parameter
                             contribution=False,
                             pRT_object=None,
                             pRT_reference=None):
        """
        Retrieve a full wavelength range model based on the given parameters.

        Parameters:
            parameters (dict): A dictionary containing parameters used to generate the model.
            model_generating_func (callable, optional): A function to generate the model.
                Defaults to None.
            ret_name (str, optional): Name of the model to be returned.
                TODO: Remove this parameter as it's currently unused.
                Defaults to None.
            contribution (bool, optional): Return the emission or transmission contribution function.
                Defaults to False.
            pRT_object (object, optional): RadTrans object for calculating the spectrum.
                Defaults to None.
            pRT_reference (object, optional): Reference Data object for calculating the spectrum.
                Defaults to None.

        Returns:
            object: The generated full range model.
        """

        # Find the boundaries of the wavelength range to calculate
        wmin = 99999.0
        wmax = 0.0
        for name, dd in self.data.items():
            if dd.wlen_range_pRT[0] < wmin:
                wmin = dd.wlen_range_pRT[0]
            if dd.wlen_range_pRT[1] > wmax:
                wmax = dd.wlen_range_pRT[1]
        # Set up parameter dictionary
        # parameters = self.build_param_dict(params,parameters_read)
        parameters["contribution"] = Parameter("contribution", False, value=contribution)

        # Setup the pRT object
        if pRT_object is not None:
            atmosphere = pRT_object
        elif pRT_reference is not None:
            atmosphere = self.data[pRT_reference].pRT_object
        else:
            atmosphere = Radtrans(line_species=cp.copy(self.rd.line_species),
                                  rayleigh_species=cp.copy(self.rd.rayleigh_species),
                                  continuum_opacities=cp.copy(self.rd.continuum_opacities),
                                  cloud_species=cp.copy(self.rd.cloud_species),
                                  mode='c-k',
                                  wlen_bords_micron=[wmin * 0.98, wmax * 1.02],
                                  do_scat_emis=self.rd.scattering)
        if self.rd.AMR:
            p = self.rd._setup_pres()
            parameters["pressure_scaling"] = self.parameters["pressure_scaling"]
            parameters["pressure_width"] = self.parameters["pressure_width"]
            parameters["pressure_simple"] = self.parameters["pressure_simple"]
        else:
            p = self.rd.p_global
        atmosphere.setup_opa_structure(p)

        # Check what model function we're using
        if model_generating_func is None:
            mg_func = self.data[self.rd.plot_kwargs["take_PTs_from"]].model_generating_function
        else:
            mg_func = model_generating_func

        # get the spectrum
        return mg_func(atmosphere, parameters, PT_plot_mode=False, AMR=self.rd.AMR)


    def get_best_fit_model(self, best_fit_params, parameters_read, ret_name=None, contribution=False,
                           pRT_reference=None, refresh=True, mode = 'bestfit'):
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
            pRT_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
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
            ret_name = self.retrieval_name
        parameters = self.build_param_dict(best_fit_params, parameters_read)
        self.best_fit_params = parameters

        use_reference = self.rd.plot_kwargs["take_PTs_from"]
        if pRT_reference is not None:
            use_reference = pRT_reference
        if self.rd.AMR:
            _ = self.rd._setup_pres()  # TODO this function should not be private
            self.best_fit_params["pressure_scaling"] = self.parameters["pressure_scaling"]
            self.best_fit_params["pressure_width"] = self.parameters["pressure_width"]
            self.best_fit_params["pressure_simple"] = self.parameters["pressure_simple"]

        if contribution:
            if not refresh and os.path.exists(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_contribution.npy"):
                print("Loading best fit spectrum and contribution from file")
                bf_contribution = np.load(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_contribution.npy")
                bf_wlen, bf_spectrum = np.load(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_full.npy").T
                return bf_wlen, bf_spectrum, bf_contribution

            bf_wlen, bf_spectrum, bf_contribution = self.get_full_range_model(self.best_fit_params,
                                                                              model_generating_func=None,
                                                                              ret_name=ret_name,
                                                                              contribution=contribution,
                                                                              pRT_reference=use_reference)
            np.save(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_contribution",bf_contribution)
        else:
            if not refresh and os.path.exists(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_full.npy"):
                print("Loading best fit spectrum from file")
                bf_wlen, bf_spectrum = np.load(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_full.npy").T
                return bf_wlen, bf_spectrum

            retVal = self.get_full_range_model(
                self.best_fit_params,
                model_generating_func=None,
                ret_name=ret_name,
                contribution=contribution,
                pRT_reference=use_reference
            )
            bf_contribution = None  # prevent eventual reference before assignment

            if len(retVal) == 2:
                bf_wlen, bf_spectrum = retVal
            else:
                bf_wlen, bf_spectrum, _ = retVal

        # Add to the dictionary.
        self.best_fit_specs[use_reference] = [bf_wlen, bf_spectrum]
        np.save(f"{self.output_dir}evaluate_{self.retrieval_name}/{ret_name}_{mode}_model_full",
                    np.column_stack([bf_wlen,bf_spectrum]))

        if contribution:
            return bf_wlen, bf_spectrum, bf_contribution

        return bf_wlen, bf_spectrum

    def get_mass_fractions(self,sample,parameters_read=None):
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
        from petitRADTRANS.retrieval.chemistry import get_abundances
        parameters = self.build_param_dict(sample, parameters_read)

        self.PT_plot_mode = True
        pressures, temps = self.log_likelihood(sample, 0, 0)
        self.PT_plot_mode = False

        if self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference is None:
            name = self.rd.plot_kwargs["take_PTs_from"]
        else:
            name = self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference
        species = [spec.split("_R_")[0] for spec in self.data[name].pRT_object.line_species]
        abundances, MMW, _, _ = get_abundances(pressures,
                                            temps,
                                            cp.copy(species),
                                            cp.copy(self.data[name].pRT_object.cloud_species),
                                            parameters,
                                            AMR=False)
        return abundances, MMW

    def get_volume_mixing_ratios(self,sample,parameters_read=None):
        """
        This function returns the VNRs of each species as a function of pressure

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
        mass_fracs, MMW = self.get_mass_fractions(sample,parameters_read)
        vmr = mass_to_number(mass_fracs)
        return vmr, MMW
    
    def save_volume_mixing_ratios(self, sample_dict, parameter_dict, rets = None):
        if rets is None:
            rets = [self.retrieval_name]
        for ret in rets:
            samples_use = sample_dict[ret]
            parameters_read = parameter_dict[ret]
            vmrs = []
            for sample in samples_use:
                vmr,_ = self.get_volume_mixing_ratios(sample[:-1],parameters_read)
                vmrs.append(np.array(list(vmr.values())))
            vmrs = np.array(vmrs)
            np.save(f"{self.output_dir}{ret}_volume_mixing_ratio_profiles", vmrs)
        return vmrs

    def get_evidence(self, ret_name = ""):

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
        for key,val in self.parameters.items():
            if val.is_free_parameter: params.append(key) 
        self.get_max_likelihood_params(samples[best_fit_index:,-1],params)
        norm = 0.0
        for name, dd in self.data.items():
            sf = 1.0
            if dd.scale_err:
                sf=self.best_fit_params[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _,log_det = np.linalg.slogdet(2*np.pi*dd.covariance*sf**2)
                add = 0.5 * log_det
            else:
                f_err = dd.flux_error
                if dd.scale_err:
                    f_err = f_err * sf
                if f"{name}_b" in self.parameters.keys():
                    print(self.best_fit_params.keys())
                    f_err = np.sqrt(f_err**2 + 10**self.best_fit_params[f"{name}_b"].value)
                add = 0.5*np.sum(np.log(2.0*np.pi*f_err**2.))
            norm = norm + add
        print(f"Best fit 𝛘^2 = {-log_l - norm:.2f}")
        return (-log_l - norm) * 2


    def get_chi2(self,sample):
        """
        Get the 𝛘^2 of the given sample relative to the data - removing normalization term from log L

        Args:
            sample : numpy.ndarray
                A single sample and likelihood taken from a post_equal_weights file
        """
        logL = sample[-1]
        norm = 0
        params = []
        for key,val in self.parameters.items():
            if val.is_free_parameter: params.append(key) 
        param_dict = self.build_param_dict(sample,params)
        for name, dd in self.data.items():
            sf = 1.0
            if dd.scale_err:
                sf=param_dict[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _,log_det = np.linalg.slogdet(2*np.pi*dd.covariance*sf**2)
                add = 0.5 * log_det
            else:
                f_err = dd.flux_error
                if dd.scale_err:
                    f_err = f_err * sf
                if f"{name}_b" in self.parameters.keys():
                    f_err = np.sqrt(f_err**2 + 10**param_dict[f"{name}_b"].value)
                add = 0.5*np.sum(np.log(2.0*np.pi*f_err**2.))
            norm = norm + add
        print(f"Best fit 𝛘^2 = {2*(-logL - norm):.2f}")
        return 2*(-logL - norm)

    def get_chi2_normalisation(self,sample):
        """
        Get the 𝛘^2 normalization term from log L

        Args:
            sample : numpy.ndarray
                A single sample and likelihood taken from a post_equal_weights file
        """
        logL = sample[-1]
        norm = 0
        params = []
        for key,val in self.parameters.items():
            if val.is_free_parameter: params.append(key) 
        param_dict = self.build_param_dict(sample,params)
        for name, dd in self.data.items():
            sf = 1.0
            if dd.scale_err:
                sf=param_dict[f"{name}_scale_factor"].value
            if dd.covariance is not None:
                _,log_det = np.linalg.slogdet(2*np.pi*dd.covariance*sf**2)
                add = 0.5 * log_det
            else:
                f_err = dd.flux_error
                if dd.scale_err:
                    f_err = f_err * sf
                if f"{name}_b" in self.parameters.keys():
                    f_err = np.sqrt(f_err**2 + 10**param_dict[f"{name}_b"].value)
                add = 0.5*np.sum(np.log(2.0*np.pi*f_err**2.))
            norm = norm + add
        return norm

    def get_reduced_chi2(self,sample,subtract_n_parameters = False):
        """
        Get the 𝛘^2/DoF of the given model - divide chi^2 by DoF or number of wavelength channels.

        Args:
            sample : numpy.ndarray
                A single sample and likelihoods taken from a post_equal_weights file
            subtract_n_parameters : bool
                If True, divide the Chi2 by the degrees of freedom (n_data - n_parameters). If False, 
                divide only by n_data
        """
        chi2 = self.get_chi2(sample)
        DoF = 0
        for name, dd in self.data.items():
            DoF += np.size(dd.flux)
        if subtract_n_parameters:
            for name, pp in self.parameters.items():
                if pp.is_free_parameter:
                    DoF -= 1
        if subtract_n_parameters:
            print(f"Best fit 𝛘^2/DoF = {chi2/DoF:.2f}")
        else:
            print(f"Best fit 𝛘^2/n_wlen = {chi2/DoF:.2f}")
        self.chi2 = chi2/DoF
        return chi2/DoF

    def get_reduced_chi2_from_model(self,wlen_model,spectrum_model,subtract_n_parameters = False):
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
        """
        logL = 0
        add = 0
        norm = 0
        DoF = 0
        for name, dd in self.data.items():
            DoF += np.size(dd.flux)
            sf = 1
            logL += dd.get_chisq(wlen_model,
                                 spectrum_model,
                                 False,
                                 self.parameters)
            if dd.covariance is not None:
                if self.best_fit_params:
                    if dd.scale_err:
                        sf*=self.best_fit_params[f"{name}_scale_factor"].value
                    _,log_det = np.linalg.slogdet(2*np.pi*dd.covariance*sf**2)
                    add = 0.5 * log_det
                else:
                    if dd.scale_err:
                        sf*=dd.scale_factor
                    _,log_det = np.linalg.slogdet(2*np.pi*dd.covariance*sf**2)
                    add = 0.5 * log_det
            else:
                add = 0.5*np.sum(np.log(2.0*np.pi*dd.flux_error**2.))
            norm += add
        if subtract_n_parameters:
            for name, pp in self.parameters.items():
                if pp.is_free_parameter:
                    DoF -= 1
        chi2 = 2*(-logL-norm)
        if subtract_n_parameters:
            print(f"Best fit 𝛘^2/DoF = {chi2/DoF:.2f}")
        else:
            print(f"Best fit 𝛘^2/n_wlen = {chi2/DoF:.2f}")        
        return chi2/DoF

    def get_analyzer(self,ret_name = ""):
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
            ret_name = self.retrieval_name

        prefix = self.output_dir + 'out_PMN/' + ret_name + '_'

        # How many free parameters?
        n_params = 0
        free_parameter_names = []
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                free_parameter_names.append(self.parameters[pp].name)
                n_params += 1

        # Get the outputs
        analyzer = pymultinest.Analyzer(n_params=n_params,
                                        outputfiles_basename=prefix)
        if ret_name == self.retrieval_name:
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
        for pp in self.parameters:
            if self.parameters[pp].is_free_parameter:
                for i_s in range(len(free_param_names)):
                    if free_param_names[i_s] == self.parameters[pp].name:
                        params[self.parameters[pp].name] = \
                            Parameter(pp, False, value=sample[i_p])
                        i_p += 1
            else:
                params[pp] = Parameter(pp, False, value=self.parameters[pp].value)
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
        from .util import teff_calc
        if ret_names is None:
            ret_names = [self.retrieval_name]
        if nsample is None:
            nsample = self.rd.plot_kwargs["nsample"]
        # Setup the pRT object
        species = []
        for line in self.rd.line_species:
            if not os.path.isdir(self.path + "opacities/lines/corr_k/"
                                 + line + "_R_"
                                 + str(resolution)):
                species.append(line)
        # If not, setup low-res c-k tables
        if len(species) > 0:
            print("Exo-k should only be run on a single thread.")
            print("The retrieval should be run once on a single core to build the c-k\n"
                  "tables, and then again with multiple cores for the remainder of the retrieval.")
            # Automatically build the entire table
            bin_species_exok(species, resolution)

        species = []

        for spec in self.rd.line_species:
            species.append(spec + "_R_" + str(resolution))

        prt_object = Radtrans(line_species=cp.copy(species),
                              rayleigh_species=cp.copy(self.rd.rayleigh_species),
                              continuum_opacities=cp.copy(self.rd.continuum_opacities),
                              cloud_species=cp.copy(self.rd.cloud_species),
                              mode='c-k',
                              wlen_bords_micron=[0.5, 28],
                              do_scat_emis=self.rd.scattering)
        if self.rd.AMR:
            p = self.rd._setup_pres()
        else:
            p = self.rd.p_global

        prt_object.setup_opa_structure(p)
        tdict = {}

        for name in ret_names:
            teffs = []
            samples = sample_dict[name]
            parameters_read = param_dict[name]
            rands = 0
            if nsample == "all":
                rands = np.linspace(0,samples.shape[0]-1,samples.shape[0])
            else:
                rands = np.random.randint(0, samples.shape[0], int(nsample))
            duse = self.data[self.rd.plot_kwargs["take_PTs_from"]]
            if duse.external_pRT_reference is not None:
                duse = self.data[duse.external_pRT_reference]
            for rint in rands:
                samp = samples[int(rint), :-1]
                params = self.build_param_dict(samp, parameters_read)
                retVal = duse.model_generating_function(prt_object,
                                                             params,
                                                             False,
                                                             self.rd.AMR)
                if len(retVal) == 2:
                    wlen, model = retVal
                else:
                    wlen, model, __ = retVal

                tfit = teff_calc(wlen, model, params["D_pl"].value, params["R_pl"].value)
                teffs.append(tfit)
            tdict[name] = np.array(teffs)
            np.save(self.output_dir + "evaluate_" + name + "/sampled_teff", np.array(teffs))
        return tdict

    def plot_all(self, output_dir=None, ret_names=None, contribution=False, mode = 'bestfit'):
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
            output_dir: string
                Output directory to store the plots. Defaults to selt.output_dir.
            ret_names : list(str)
                List of retrieval names. Used if multiple retrievals are to be included 
                in a single corner plot.
            contribution : bool
                If true, plot the emission or transmission contribution function.
            mode : str
                If 'bestfit', consider the maximum likelihood sample for plotting,
                if median, calculate the model based on the median retrieved parameters.
        """
        # Run plotting on a single core only.
        if not self.use_MPI or RANK == 0:
            if ret_names is None:
                ret_names = []

            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'

            if output_dir is None:
                output_dir = self.output_dir

            sample_dict, parameter_dict = self.get_samples(output_dir, ret_names=ret_names)

            ###########################################
            # Plot best-fit spectrum
            ###########################################
            samples_use = cp.copy(sample_dict[self.retrieval_name])
            parameters_read = cp.copy(parameter_dict[self.retrieval_name])
            i_p = 0

            # This might actually be redundant...
            for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.parameters[pp].name:
                            samples_use[:, i_p] = sample_dict[self.retrieval_name][:, i_s]
                    i_p += 1

            print("Best fit parameters")
            i_p = 0
            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Print outputs
            # TODO add verbosity
            for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.parameters[pp].name:
                            print(self.parameters[pp].name, samples_use[best_fit_index][i_p])
                            i_p += 1

            # Plotting
            # 
            self.plot_spectra(samples_use, parameters_read, refresh = True, mode = mode)

            if self.evaluate_sample_spectra:
                self.plot_sampled(samples_use, parameters_read)

            self.plot_PT(sample_dict, parameters_read, contribution=contribution, mode = mode, refresh = False)
            self.plot_corner(sample_dict, parameter_dict, parameters_read)

            if contribution:
                self.plot_contribution(samples_use, parameters_read, mode = mode, refresh = False)

            self.plot_abundances(samples_use, parameters_read, contribution=contribution, mode = mode, refresh = False)
            print("Finished generating all plots!")
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return


    def plot_spectra(self, samples_use, parameters_read, model_generating_func=None, pRT_reference=None,
                     refresh=True,mode = "bestfit"):
        """
        Plot the best fit spectrum, the data from each dataset and the residuals between the two.
        Saves a file to OUTPUT_DIR/evaluate_RETRIEVAL_NAME/RETRIEVAL_NAME_MODE_spec.pdf

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            model_generating_func : method
                A function that will take in the standard 'model' arguments
                (pRT_object, params, pt_plot_mode, AMR, resolution)
                and will return the wavlength and flux arrays as calculated by petitRadTrans.
                If no argument is given, it uses the method of the first dataset included in the retrieval.
            pRT_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
            mode : str
                Use 'bestfit' (minimum likelihood) parameters, or median parameter values.
        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data
        """
        if not self.use_MPI or RANK==0:
            # Avoiding saving the model spectrum to the sampled spectrum dictionary.
            check = self.evaluate_sample_spectra

            if self.evaluate_sample_spectra:
                self.evaluate_sample_spectra = False

            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'
            print("\nPlotting Best-fit spectrum")

            fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', sharey=False,
                                    gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.1},
                                    figsize=(18, 9))
            ax = axes[0]  # Normal Spectrum axis
            ax_r = axes[1]  # residual axis

            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
            sample_use = samples_use[best_fit_index,:-1]

            # Then get the full wavelength range
            # Generate the best fit spectrum using the set of parameters with the lowest log-likelihood
            if mode.lower() == "median":
                med_param, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)

            # Setup best fit spectrum
            # First get the fit for each dataset for the residual plots
            self.log_likelihood(sample_use, 0, 0)

            bf_wlen, bf_spectrum = self.get_best_fit_model(
                sample_use,  # set of parameters with the lowest log-likelihood (best-fit)
                parameters_read,  # name of the parameters
                model_generating_func,
                pRT_reference=pRT_reference,
                refresh=refresh,
                mode = mode
            )
            chi2 = self.get_reduced_chi2_from_model(bf_wlen, bf_spectrum, subtract_n_parameters = True)

            # Iterate through each dataset, plotting the data and the residuals.
            for name,dd in self.data.items():
                # If the user has specified a resolution, rebin to that
                if not dd.photometry:
                    resolution_data = np.mean(dd.wlen[1:]/np.diff(dd.wlen))
                    if self.rd.plot_kwargs["resolution"] is not None and\
                        self.rd.plot_kwargs["resolution"] < resolution_data:
                        ratio = resolution_data/self.rd.plot_kwargs["resolution"]
                        flux,edges,_ = binned_statistic(dd.wlen,dd.flux,'mean',dd.wlen.shape[0]/ratio)
                        error,_,_ = binned_statistic(dd.wlen,dd.flux_error,\
                                                    'mean',dd.wlen.shape[0]/ratio)/np.sqrt(ratio)
                        wlen = np.array([(edges[i]+edges[i+1])/2.0 for i in range(edges.shape[0]-1)])
                        wlen_bins = np.zeros_like(wlen)
                        wlen_bins[:-1] = np.diff(wlen)
                        wlen_bins[-1] = wlen_bins[-2]
                    else:
                        wlen = dd.wlen
                        error = dd.flux_error
                        flux = dd.flux
                        wlen_bins = dd.wlen_bins
                else:
                    wlen = np.mean(dd.width_photometry)
                    flux = dd.flux
                    error = dd.flux_error
                    wlen_bins = dd.wlen_bins

                # If the data has an arbitrary retrieved scaling factor
                scale  = 1.0
                if dd.scale:
                    scale = self.best_fit_params[f"{name}_scale_factor"].value

                errscale = 1.0
                if dd.scale_err:
                    errscale = self.best_fit_params[f"{name}_scale_factor"].value
                    error = error * errscale

                offset = 0.0
                if dd.offset_bool:
                    offset = self.best_fit_params[f"{name}_offset"].value

                flux = (flux*scale) - offset
                if f"{dd.name}_b" in self.parameters.keys():
                    error = np.sqrt(error + 10**(self.best_fit_parameters["{dd.name}_b"]))

                if not dd.photometry:
                    if dd.external_pRT_reference is None:
                        spectrum_model = self.best_fit_specs[name][1]
                        if dd.data_resolution is not None:
                            spectrum_model = dd.convolve(self.best_fit_specs[name][0],
                                                        self.best_fit_specs[name][1],
                                                        dd.data_resolution)
                        best_fit_binned = rgw(self.best_fit_specs[name][0],
                                            spectrum_model,
                                            wlen,
                                            wlen_bins)
                    else:
                        if dd.data_resolution is not None:
                            spectrum_model = dd.convolve(self.best_fit_specs[dd.external_pRT_reference][0],
                                                        self.best_fit_specs[dd.external_pRT_reference][1],
                                                        dd.data_resolution)
                        best_fit_binned = rgw(self.best_fit_specs[name][0],
                                            spectrum_model,
                                            wlen,
                                            wlen_bins)
                else:
                    if dd.external_pRT_reference is None:
                        best_fit_binned = dd.photometric_transformation_function(self.best_fit_specs[name][0],
                                                                                self.best_fit_specs[name][1])
                        # Species functions give tuples of (flux,error)
                        try:
                            best_fit_binned = best_fit_binned[0]
                        except:  # TODO find exception expected here
                            pass

                    else:
                        best_fit_binned = \
                            dd.photometric_transformation_function(self.best_fit_specs[dd.external_pRT_reference][0],
                                                                self.best_fit_specs[dd.external_pRT_reference][1])
                        try:
                            best_fit_binned = best_fit_binned[0]
                        except:  # TODO find exception expected here
                            pass
                # Plot the data
                marker = 'o'
                if dd.photometry:
                    marker = 's'
                if not dd.photometry:
                    label = dd.name
                    ax.errorbar(wlen,
                                (flux * self.rd.plot_kwargs["y_axis_scaling"]),
                                yerr=error * self.rd.plot_kwargs["y_axis_scaling"],
                                marker=marker, 
                                markeredgecolor='k', 
                                linewidth=0, 
                                elinewidth=2,
                                label=label, 
                                zorder=10, 
                                alpha=0.9)
                else:
                    # Don't label photometry?
                    ax.errorbar(wlen,
                                (flux * self.rd.plot_kwargs["y_axis_scaling"]),
                                yerr=error * self.rd.plot_kwargs["y_axis_scaling"],
                                xerr=dd.wlen_bins / 2., 
                                linewidth=0, 
                                elinewidth=2,
                                marker=marker, 
                                markeredgecolor='k', 
                                color='grey', 
                                zorder=10,
                                label=None, 
                                alpha=0.6)
                    
                # Plot the residuals
                col = ax.get_lines()[-1].get_color()
                if dd.external_pRT_reference is None:

                    ax_r.errorbar(wlen,
                                (flux - best_fit_binned) / (error),
                                yerr=error / error,
                                color=col,
                                linewidth=0, elinewidth=2,
                                marker=marker, markeredgecolor='k', zorder=10,
                                alpha=0.9)
                else:
                    ax_r.errorbar(wlen,
                                (flux - best_fit_binned) / (error),
                                yerr= error / error,
                                color=col,
                                linewidth=0, 
                                elinewidth=2,
                                marker=marker, 
                                markeredgecolor='k', 
                                zorder=10,
                                alpha=0.9)
                    
            # Plot the best fit model
            ax.plot(bf_wlen,
                    bf_spectrum * self.rd.plot_kwargs["y_axis_scaling"],
                    label=rf'Best Fit Model, $\chi^2=${chi2:.2f}',
                    linewidth=4,
                    alpha=0.5,
                    color='r')
            
            # Plot the shading in the residual plot
            yabs_max = abs(max(ax_r.get_ylim(), key=abs))
            lims = ax.get_xlim()
            lim_y = ax.get_ylim()
            lim_y = [lim_y[0], lim_y[1] * 1.05]

            if self.rd.plot_kwargs.get('flux_lim') is not None:
                ax.set_ylim(self.rd.plot_kwargs.get('flux_lim'))
            else:
                ax.set_ylim(lim_y)

            # weird scaling to get axis to look ok on log plots
            if self.rd.plot_kwargs["xscale"] == 'log':
                lims = [lims[0] * 1.09, lims[1] * 1.02]
            else:
                lims = [bf_wlen[0] * 0.98, bf_wlen[-1] * 1.02]

            if self.rd.plot_kwargs.get('wavelength_lim') is not None:
                ax.set_xlim(self.rd.plot_kwargs.get('wavelength_lim'))
                ax_r.set_xlim(self.rd.plot_kwargs.get('wavelength_lim'))
            else:
                ax.set_xlim(lims)
                ax_r.set_xlim(lims)

            ax_r.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax_r.fill_between(lims, -1, 1, color='dimgrey', alpha=0.4, zorder=-10)
            ax_r.fill_between(lims, -3, 3, color='darkgrey', alpha=0.3, zorder=-9)
            ax_r.fill_between(lims, -5, 5, color='lightgrey', alpha=0.3, zorder=-8)
            ax_r.axhline(linestyle='--', color='k', alpha=0.8, linewidth=2)

            # Making the plots pretty
            if "xscale" in self.rd.plot_kwargs.keys():
                ax.set_xscale(self.rd.plot_kwargs["xscale"])
            try:
                ax.set_yscale(self.rd.plot_kwargs["yscale"])
            except:  # TODO find exception expected here
                pass

            # Fancy ticks for upper pane
            ax.tick_params(axis="both", direction="in", length=10, bottom=True, top=True, left=True, right=True)
            try:
                ax.xaxis.set_major_formatter('{x:.1f}')
            except:  # TODO find exception expected here
                logging.warning("Please update to matplotlib 3.3.4 or greater")
                pass

            maxwlen = np.floor(np.max(bf_wlen))
            ntick_div = 1
            if maxwlen >6:
                ntick_div = 2
            if self.rd.plot_kwargs["xscale"] == 'log':
                # For the minor ticks, use no labels; default NullFormatter.
                x_major = LogLocator(base=10.0, subs=(1, 2, 3, 4), numticks=4)
                ax.xaxis.set_major_locator(x_major)
                x_minor = LogLocator(base=10.0, subs=np.arange(0.1, 10.1, 0.1) * 0.1, numticks=100)
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
            ax.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])

            # Fancy ticks for lower pane
            ax_r.tick_params(axis="both", direction="in", length=10, bottom=True, top=True, left=True, right=True)
            try:
                ax_r.xaxis.set_major_formatter('{x:.1f}')
            except:  # TODO find exception expected here
                logging.warning("Please update to matplotlib 3.3.4 or greater")
                pass
            if self.rd.plot_kwargs["xscale"] == 'log':
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
            ax_r.set_xlabel(self.rd.plot_kwargs["spec_xlabel"])
            ax.legend(loc='upper center', ncol=len(self.data.keys()) + 1).set_zorder(1002)
            plt.tight_layout()
            plt.savefig(
                self.output_dir + 'evaluate_' + self.rd.retrieval_name + '/' + self.retrieval_name + '_' + mode + '_spec.pdf')
            self.evaluate_sample_spectra = check
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig, ax, ax_r

    def plot_sampled(self, samples_use, parameters_read, downsample_factor=None, save_outputs=False,
                     nsample = None, pRT_reference=None, refresh=True):
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
            pRT_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for plotting,
                instead of generating a new pRT object at R = 1000.
            refresh : bool
                If True (default value) the .npy files in the evaluate_[retrieval_name] folder will be replaced
                by recalculating the best fit model. This is useful if plotting intermediate results from a
                retrieval that is still running. If False no new spectrum will be calculated and the plot will
                be generated from the .npy files in the evaluate_[retrieval_name] folder.
        """
        if not self.use_MPI or RANK==0:

            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'

            self.rd.plot_kwargs["nsample"] = int(self.rd.plot_kwargs["nsample"])

            print("\nPlotting Best-fit spectrum with " + str(self.rd.plot_kwargs["nsample"]) + " samples.")
            print("This could take some time...")
            len_samples = samples_use.shape[0]
            path = self.output_dir + 'evaluate_' + self.retrieval_name + "/"

            wmin = 99999.0
            wmax = 0.0
            for name, dd in self.data.items():
                if dd.wlen_range_pRT[0] < wmin:
                    wmin = dd.wlen_range_pRT[0]
                if dd.wlen_range_pRT[1] > wmax:
                    wmax = dd.wlen_range_pRT[1]

            # Set up parameter dictionary
            atmosphere = Radtrans(line_species=cp.copy(self.rd.line_species),
                                rayleigh_species=cp.copy(self.rd.rayleigh_species),
                                continuum_opacities=cp.copy(self.rd.continuum_opacities),
                                cloud_species=cp.copy(self.rd.cloud_species),
                                mode='c-k',
                                wlen_bords_micron=[wmin * 0.98, wmax * 1.02],
                                do_scat_emis=self.rd.scattering)
            fig, ax = plt.subplots(figsize=(16, 10))
            nsamp = nsample
            if nsample is None:
                nsamp = self.rd.plot_kwargs["nsample"]
            random_ints = np.random.randint(low=0,high = len_samples, size = int(nsamp))
            for i_sample,random_index in enumerate(random_ints):
                if os.path.exists(path + "posterior_sampled_spectra_" + str(random_index).zfill(5)):
                    wlen, model = np.load(path + "posterior_sampled_spectra_" + str(random_index).zfill(5) + ".npy")
                else:
                    print(f"Generating sampled spectrum {i_sample} / {self.rd.plot_kwargs['nsample']}...")

                    parameters = self.build_param_dict(samples_use[random_index, :-1], parameters_read)
                    parameters["contribution"] = Parameter("contribution", False, value=False)
                    retVal = self.get_full_range_model(parameters, pRT_object=atmosphere)

                    wlen = None
                    model = None

                    if len(retVal) == 2:
                        wlen, model = retVal
                    elif len(retVal) == 3:
                        wlen, model, __ = retVal
                    else:
                        ValueError(f"expected 2 or 3 values to unpack from full range model, "
                                f"but got {len(retVal)}")

                if downsample_factor is not None:
                    model = nc.running_mean(model, downsample_factor)[::downsample_factor]
                    wlen = wlen[::downsample_factor]

                if save_outputs:
                    np.save(path + "posterior_sampled_spectra_" +
                            str(random_index).zfill(5),
                            np.column_stack((wlen, model)))
                ax.plot(wlen, model * self.rd.plot_kwargs["y_axis_scaling"],
                        color="#00d2f3", alpha=1 / self.rd.plot_kwargs["nsample"] + 0.1, linewidth=0.2,
                        marker=None)
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Setup best fit spectrum
            # First get the fit for each dataset for the residual plots
            # self.log_likelihood(samples_use[best_fit_index, :-1], 0, 0)
            # Then get the full wavelength range
            bf_wlen, bf_spectrum = self.get_best_fit_model(
                samples_use[best_fit_index, :-1],
                parameters_read,
                pRT_reference=pRT_reference,
                refresh=refresh
            )
            chi2 = self.get_reduced_chi2_from_model(bf_wlen, bf_spectrum,subtract_n_parameters = True)
            ax.plot(bf_wlen,
                    bf_spectrum * self.rd.plot_kwargs["y_axis_scaling"],
                    marker=None,
                    label=rf"Best fit, $\chi^{2}=${chi2:.2f}",
                    linewidth=4,
                    alpha=0.5,
                    color='r')

            for name, dd in self.data.items():
                fig, ax = plot_data(fig, ax, dd,
                                    resolution=self.rd.plot_kwargs["resolution"],
                                    scaling=self.rd.plot_kwargs["y_axis_scaling"])
            ax.set_xlabel('Wavelength [micron]')
            ax.set_ylabel(self.rd.plot_kwargs["spec_ylabel"])
            ax.legend(loc='best')
            plt.tight_layout()
            plt.savefig(path + self.retrieval_name + '_sampled.pdf', bbox_inches=0.)
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig, ax

    def plot_PT(self, sample_dict, parameters_read, contribution=False, refresh = False, pRT_reference = None, mode = 'bestfit'):
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
            pRT_reference : str
                If specified, the pRT object of the data with name pRT_reference will be used for calculating
                the contribution function, instead of generating a new pRT object at R = 1000.
            mode : str
                'bestfit' or 'median', indicating which set of values should be used to calculate the contribution function.

        Returns:
            fig : matplotlib.figure
            ax : matplotlib.axes
        """
        if not self.use_MPI or RANK==0:

            print("\nPlotting PT profiles")
            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'

            # Choose what samples we want to use
            samples_use = cp.copy(sample_dict[self.retrieval_name])
            len_samp = len(samples_use)

            logL, best_fit_index = self.get_best_fit_likelihood(samples_use)
            # This is probably obsolete
            #i_p = 0
            self.PT_plot_mode = True
            """for pp in self.parameters:
                if self.parameters[pp].is_free_parameter:
                    for i_s in range(len(parameters_read)):
                        if parameters_read[i_s] == self.parameters[pp].name:
                            samples_use[:, i_p] = sample_dict[self.retrieval_name][:, i_s]

                    i_p += 1"""

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.rd.AMR
            self.rd.AMR = False
            temps = []

            pressures = self.rd.p_global  # prevent eventual reference before assignment
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(pressures)
            press_file = f"{self.output_dir}evaluate_{self.retrieval_name}/{self.retrieval_name}_pressures"
            temp_file = f"{self.output_dir}evaluate_{self.retrieval_name}/{self.retrieval_name}_temps"

            if os.path.exists(press_file + ".npy") and os.path.exists(temp_file + ".npy") and not refresh:
                pressures = np.load(press_file+ ".npy")
                temps_sort = np.load(temp_file+ ".npy")
            else:
                for sample in samples_use:
                    pressures, t = self.log_likelihood(sample[:-1], 0, 0)
                    temps.append(t)

                temps = np.array(temps)
                temps_sort = np.sort(temps, axis=0)
                np.save(press_file,pressures)
                np.save(temp_file,temps_sort)
            
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.fill_betweenx(pressures,
                            x1=temps_sort[0, :],
                            x2=temps_sort[-1, :],
                            color='cyan', label='all',
                            zorder=0)
            ax.fill_betweenx(pressures,
                            x1=temps_sort[int(len_samp * (0.5 - 0.997 / 2.)), :],
                            x2=temps_sort[int(len_samp * (0.5 + 0.997 / 2.)), :],
                            color='brown', label='3 sig',
                            zorder=1)
            ax.fill_betweenx(pressures,
                            x1=temps_sort[int(len_samp * (0.5 - 0.95 / 2.)), :],
                            x2=temps_sort[int(len_samp * (0.5 + 0.95 / 2.)), :],
                            color='orange', label='2 sig',
                            zorder=2)
            ax.fill_betweenx(pressures,
                            x1=temps_sort[int(len_samp * (0.5 - 0.68 / 2.)), :],
                            x2=temps_sort[int(len_samp * (0.5 + 0.68 / 2.)), :],
                            color='red', label='1 sig',
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
            if self.rd.plot_kwargs["temp_limits"] is not None:
                tlims = self.rd.plot_kwargs["temp_limits"]
                ax.set_xlim(self.rd.plot_kwargs["temp_limits"])
            else:
                tlims = (np.min(temps)*0.97,np.max(temps)*1.03)
                ax.set_xlim(tlims)

            # Check if we're weighting by the contribution function.
            if contribution:
                self.PT_plot_mode = False
                if mode.strip('-').strip("_").lower() == "bestfit":
                    # Get best-fit index
                    logL, best_fit_index = self.get_best_fit_likelihood(samples_use)
                    self.get_max_likelihood_params(samples_use[best_fit_index, :-1], parameters_read)
                    sample_use = samples_use[best_fit_index,:-1]
                elif mode.lower() == "median":
                    med_param, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)

                bf_wlen, bf_spectrum, bf_contribution = self.get_best_fit_model(
                    sample_use,
                    parameters_read,
                    pRT_reference=pRT_reference,
                    refresh=refresh,
                    contribution = True,
                    mode = mode
                )
                nu = nc.c/bf_wlen

                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = bf_spectrum * diff_nu / np.sum(bf_spectrum * diff_nu)

                if self.plotting:
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

                if self.plotting:
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

                #np.savetxt('spectrally_weighted_constribution.dat', np.column_stack((pressures, contr_em_weigh)))
            ax.set_yscale('log')
            try:
                ax.set_ylim(self.rd.plot_kwargs["press_limits"])
            except:
                ax.set_ylim([pressures[-1]*1.03, pressures[0]/1.03])

            # Labelling and output
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Pressure [bar]')
            ax.legend(loc='best')
            plt.savefig(f"{self.output_dir}evaluate_{self.retrieval_name}/{self.retrieval_name}_PT_envelopes.pdf",
                        bbox_inches = 'tight')
            self.rd.AMR = amr
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(self.rd.amr_pressure*1e6)
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig, ax

    def plot_corner(self, sample_dict, parameter_dict, parameters_read, plot_best_fit=True, true_values = None, **kwargs):
        """
        Make the corner plots

        Args:
            sample_dict : Dict
                Dictionary of samples from PMN outputs, with keys being retrieval names
            parameter_dict : Dict
                Dictionary of parameters for each of the retrievals to be plotted.
            parameters_read : List
                Used to plot correct parameters, as some in self.parameters are not free, and
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
        """
        if not self.use_MPI or RANK==0:

            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'
            print("\nMaking corner plot")
            sample_use_dict = {}
            p_plot_inds = {}
            p_ranges = {}
            p_use_dict = {}
            bf_index = None

            if plot_best_fit:
                bf_index = {}

            for name, params in parameter_dict.items():
                samples_use = cp.copy(sample_dict[name])
                parameters_use = cp.copy(params)
                parameter_plot_indices = []
                parameter_ranges = []
                i_p = 0

                for pp in parameters_read:
                    if self.parameters[pp].plot_in_corner:
                        parameter_plot_indices.append(i_p)

                    if self.parameters[pp].corner_label is not None:
                        parameters_use[i_p] = self.parameters[pp].corner_label

                    if self.parameters[pp].corner_transform is not None:
                        samples_use[:, i_p] = \
                            self.parameters[pp].corner_transform(samples_use[:, i_p])

                    parameter_ranges.append(self.parameters[pp].corner_ranges)

                    i_p += 1

                p_plot_inds[name] = parameter_plot_indices
                p_ranges[name] = parameter_ranges
                p_use_dict[name] = parameters_use
                sample_use_dict[name] = samples_use

            output_file = self.output_dir + 'evaluate_' + self.retrieval_name + '/' + self.retrieval_name \
                + '_corner_plot.pdf'

            # from Plotting
            fig = contour_corner(
                sample_use_dict,
                p_use_dict,
                output_file,
                parameter_plot_indices=p_plot_inds,
                parameter_ranges=p_ranges,
                prt_plot_style=self.prt_plot_style,
                plot_best_fit=plot_best_fit,
                true_values = true_values,
                **kwargs
            )
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig

    def plot_data(self):
        """
        Plot the data used in the retrieval. 
        """
        if not self.use_MPI or RANK==0:
            fig, ax = plt.subplots(figsize = (10,6))
            for name, dd in self.rd.data.items():
                if dd.photometry:
                    wlen = np.mean(dd.width_photometry)
                else:
                    wlen = dd.wlen
                ax.errorbar(wlen, dd.flux, yerr=dd.flux_error, label=name, marker='o')
            ax.legend()
            plt.savefig(self.output_dir + "evaluate_" + self.retrieval_name + "/" + self.retrieval_name + "_Data.pdf")
        if self.use_MPI and COMM is not None:
            comm.barrier()

    def plot_contribution(self, samples_use, parameters_read, model_generating_func=None, log_scale_contribution=False,
                          n_contour_levels=30, refresh=True, mode = 'bestfit',):
        """
        Plot the contribution function of the bestfit or median model from a retrieval. This plot indicates the
        relative contribution from each wavelength and each pressure level in the atmosphere to the spectrum.

        Args:
            samples_use : numpy.ndarray
                An array of the samples from the post_equal_weights file, used to find the best fit sample
            parameters_read : list
                A list of the free parameters as read from the output files.
            model_generating_func : method
                A function that will take in the standard 'model' arguments
                (pRT_object, params, pt_plot_mode, AMR, resolution)
                and will return the wavlength and flux arrays as calculated by petitRadTrans.
                If no argument is given, it uses the method of the first dataset included in the retrieval.
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
                'bestfit' or 'median', indicating which set of values should be used to calculate the contribution function.

        Returns:
            fig : matplotlib.figure
                The matplotlib figure, containing the data, best fit spectrum and residuals.
            ax : matplotlib.axes
                The upper pane of the plot, containing the best fit spectrum and data
            ax_r : matplotlib.axes
                The lower pane of the plot, containing the residuals between the fit and the data
        """
        if not self.use_MPI or RANK==0:

            self.evaluate_sample_spectra = False
            if not self.run_mode == 'evaluate':
                logging.warning("Not in evaluate mode. Changing run mode to evaluate.")
                self.run_mode = 'evaluate'
            print("\nPlotting Best-fit contribution function")

            # Get best-fit index
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.rd.AMR
            self.rd.AMR = False
            # Store old pressure array so that we can put it back later.
            p_global_keep = self.rd.p_global
            pressures = self.rd.p_global  # prevent eventual reference before assignment
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(pressures)
            #if self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference is not None:
            #    p_keep = self.data[self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference].pRT_object.press
            #    self.data[self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference].pRT_object.setup_opa_structure(temp_pres)
            #else:
            #    p_keep = self.data[self.rd.plot_kwargs["take_PTs_from"]].pRT_object.press
            #    self.data[self.rd.plot_kwargs["take_PTs_from"]].pRT_object.setup_opa_structure(temp_pres)

            # Calculate the temperature structure
            self.PT_plot_mode = True
            pressures, t = self.log_likelihood(samples_use[best_fit_index, :-1], 0, 0)
            self.PT_plot_mode = False

            # Calculate the best fit/median spectrum contribution
            if mode.strip('-').strip("_").lower() == "bestfit":
                # Get best-fit index
                logL, best_fit_index = self.get_best_fit_likelihood(samples_use)
                self.get_max_likelihood_params(samples_use[best_fit_index, :-1], parameters_read)
                sample_use = samples_use[best_fit_index,:-1]
            elif mode.lower() == "median":
                med_params,sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)
            bf_wlen, bf_spectrum, bf_contribution= self.get_best_fit_model(
                                                                        sample_use,
                                                                        parameters_read,
                                                                        refresh=refresh,
                                                                        contribution = True,
                                                                        mode = mode
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
                plot_cont = -np.log10(bf_contribution * self.rd.plot_kwargs["y_axis_scaling"] / weights)
                label = "-Log Weighted Flux"
            else:
                plot_cont = bf_contribution * self.rd.plot_kwargs["y_axis_scaling"] / weights
                label = "Weighted Flux"

            im = ax.contourf(x,
                            y,
                            plot_cont,
                            n_contour_levels,
                            cmap='magma')
            ax.set_xlabel(self.rd.plot_kwargs["spec_xlabel"])
            ax.set_ylabel("Pressure [bar]")
            ax.set_xscale(self.rd.plot_kwargs["xscale"])
            ax.set_yscale("log")
            ax.set_ylim(pressures[-1] * 1.03, pressures[0] / 1.03)
            plt.colorbar(im, ax=ax, label=label)
            plt.savefig(f"{self.output_dir}evaluate_{self.retrieval_name}/{self.retrieval_name}_{mode}_contribution.pdf",
                        bbox_inches = 'tight'
            )

            # Restore the correct pressure arrays.
            # *1e6 for units (cgs from bar)
            self.rd.p_global = p_global_keep
            self.rd.AMR = amr
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(self.rd.amr_pressure*1e6)
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig, ax

    def plot_abundances(self, 
                        samples_use, 
                        parameters_read, 
                        species_to_plot=None, 
                        contribution=False, 
                        refresh=True, 
                        mode = 'bestfit', 
                        sample_posteriors = False, 
                        volume_mixing_ratio = False):
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
        if not self.use_MPI or RANK==0:

            print("\nPlotting Abundances profiles")
            if self.prt_plot_style:
                import petitRADTRANS.retrieval.plot_style as ps

            # Let's set up a standardized pressure array, regardless of AMR stuff.
            amr = self.rd.AMR
            self.rd.AMR = False
            # Store old pressure array so that we can put it back later.
            p_global_keep = self.rd.p_global
            pressures = self.rd.p_global  # prevent eventual reference before assignment
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(pressures)

            self.PT_plot_mode = True
            if mode.strip('-').strip("_").lower() == "bestfit":
                    # Get best-fit index
                    logL, best_fit_index = self.get_best_fit_likelihood(samples_use)
                    self.get_max_likelihood_params(samples_use[best_fit_index, :-1], parameters_read)
                    sample_use = samples_use[best_fit_index, :-1]
            elif mode.lower() == "median":
                    med_params, sample_use = self.get_median_params(samples_use, parameters_read, return_array=True)
            pressures, t = self.log_likelihood(sample_use, 0, 0)
            self.PT_plot_mode = False

            # Check if we're only plotting a few species
            if species_to_plot is None:
                if self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference is not None:
                    species_to_plot = self.data[self.data[self.rd.plot_kwargs["take_PTs_from"]].external_pRT_reference].pRT_object.line_species
                else:
                    species_to_plot = self.data[self.rd.plot_kwargs["take_PTs_from"]].pRT_object.line_species

            # Set up colours - abundances usually have a lot of species,
            # so let's use the default matplotlib colour scheme rather
            # than the pRT colours.
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            # Figure
            fig,ax = plt.subplots(figsize=(12, 7))

            # Check to see if we're plotting contour regions
            if sample_posteriors:
                abundances = {}
                for species in species_to_plot:
                    abundances[species.split("_R_")[0]] = []

                # Go through EVERY sample to find the abundance distribution.
                # Very slow.
                for sample in samples_use:
                    if volume_mixing_ratio:
                        abund_dict, MMW = self.get_volume_mixing_ratios(sample[:-1], parameters_read)
                    else:
                        abund_dict, MMW = self.get_mass_fractions(sample[:-1], parameters_read)
                    for species in species_to_plot:
                        abundances[species.split("_R_")[0]].append(abund_dict[species.split("_R_")[0]])

                # Plot median and 1sigma contours
                for i,species in enumerate(species_to_plot):
                    medians = np.median(np.array(abundances[species.split("_R_")[0]]),axis=0)
                    low,med,high = np.quantile(np.array(abundances[species.split("_R_")[0]]),[0.159,0.5,0.841],axis=0)
                    ax.plot(med,
                            pressures,
                            label=species.split('_')[0],
                            color = colors[i%len(colors)],
                            zorder = 0,
                            linewidth = 2)
                    ax.plot(low,
                        pressures,
                        color = colors[i%len(colors)],
                        linewidth = 0.4,
                        zorder = 0)
                    ax.plot(high,
                        pressures,
                        color = colors[i%len(colors)],
                        linewidth = 0.4,
                        zorder = 0)

                    ax.fill_betweenx(pressures,
                                    x1 = low,
                                    x2 = high,
                                    color = colors[i%len(colors)],
                                    alpha = 0.15,
                                    zorder = -1)
            else:
                # Plot only the best fit abundances.
                # Default to this for speed.
                if volume_mixing_ratio:
                    abund_dict, MMW = self.get_volume_mixing_ratios(sample_use, parameters_read)
                else:
                    abund_dict, MMW = self.get_mass_fractions(sample_use, parameters_read)
                for i,spec in enumerate(species_to_plot):
                    ax.plot(abund_dict[spec.split("_R_")[0]],
                            pressures,
                            label=spec.split('_')[0],
                            color = colors[i%len(colors)],
                            zorder = 0,
                            linewidth = 2)

            # Check to see if we're weighting by the emission contribution.
            if contribution:
                bf_wlen, bf_spectrum,bf_contribution = self.get_best_fit_model(
                                                sample_use,
                                                parameters_read,
                                                refresh=refresh,
                                                contribution = True
                                                )
                nu = nc.c/bf_wlen
                mean_diff_nu = -np.diff(nu)
                diff_nu = np.zeros_like(nu)
                diff_nu[:-1] = mean_diff_nu
                diff_nu[-1] = diff_nu[-2]
                spectral_weights = bf_spectrum * diff_nu / np.sum(bf_spectrum * diff_nu)

                if self.plotting:
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

                if self.plotting:
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

                # plt.plot(temp, p, color = 'white', linewidth = 3.)
                # plt.plot(temp, p, '-', color = 'black', linewidth = 1.,label='Input')
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
            ax.set_xlim(8e-13,3)
            ax.set_axisbelow(False)
            ax.tick_params(zorder =2)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize =18)
            plt.tight_layout()
            if not sample_posteriors:
                plt.savefig(self.output_dir + 'evaluate_'+self.retrieval_name +'/' +  self.retrieval_name  + '_' + mode + '_abundance_profiles.pdf',
                            bbox_inches = 'tight')
            else:
                plt.savefig(self.output_dir + 'evaluate_'+self.retrieval_name +'/' +  self.retrieval_name  + '_sampled_abundance_profiles.pdf',
                            bbox_inches = 'tight')
            # Restore the correct pressure arrays.
            self.rd.p_global = p_global_keep
            self.rd.AMR = amr
            if amr:
                for name, dd in self.data.items():
                    dd.pRT_object.setup_opa_structure(self.rd.amr_pressure*1e6)
        if self.use_MPI and COMM is not None:
            comm.barrier()
        return fig,ax
