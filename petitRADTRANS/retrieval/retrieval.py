import os
import sys
import copy
import importlib
import json
import traceback
import warnings
import dill
import numpy as np
import numpy.typing as npt
from typing import Callable

from scipy.stats import binned_statistic
from jax import vmap, lax, numpy as jnp

import petitRADTRANS
from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity, CorrelatedKOpacity
from petitRADTRANS.chemistry.utils import mass_fractions2volume_mixing_ratios
from petitRADTRANS.config.configuration import petitradtrans_config_parser
from petitRADTRANS.radtrans_core.spectrum_utils import (
    convolve,
    convolve_and_sample_variable_resolution_breads,
    filter_spectrum_with_spline,
    rebin_spectrum_bin,
)
from petitRADTRANS.physics import (
    shift_wavelengths_by_radial_velocity,
)
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.data import Data, PhotometryData, TimeSeriesData
from petitRADTRANS.retrieval.parameter import Parameter, RetrievalParameter
from petitRADTRANS.retrieval.models import (
    MODEL_CONTRACT_DIFFERENTIABLE,
    _build_model_context_from_parameters,
    _model_result_to_legacy_output,
    coerce_model_generating_function,
    run_runtime_model_adapter,
)
from petitRADTRANS.retrieval._mpi import get_mpi_context
from petitRADTRANS.retrieval.plotting import RetrievalPlotter
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.runtime import (
    ParameterLayout,
    PhysicalParams,
    RetrievalRuntime,
)
from petitRADTRANS.retrieval.utils import (
    get_pymultinest_sample_dict,
    unpack_model_output,
    extract_spectrum_parameters,
    process_spectrum_pipeline,
    validate_model_output,
    ensure_jax_array,
)
from petitRADTRANS.retrieval.sampler import (
    Sampler,
    SamplerContext,
)
from petitRADTRANS.retrieval.log_likelihood import (
    LogLikelihood,
    get_aic,
    get_bic,
    get_bpics,
)
from petitRADTRANS.utils import list_str2str

# noinspection PyUnresolvedReferences

from petitRADTRANS.jax_dtype import JAX_DTYPE

_SAMPLER_CLASS_NAMES: dict[str, str] = {
    "pymultinest": "PymultinestSampler",
    "ultranest": "UltranestSampler",
    "dynesty": "DynestySampler",
    "blackjaxhmc": "BlackJAXHMCSampler",
    "blackjax_hmc": "BlackJAXHMCSampler",
    "hmc": "BlackJAXHMCSampler",
    "blackjaxnuts": "BlackJAXNUTSSampler",
    "blackjax_nuts": "BlackJAXNUTSSampler",
    "nuts": "BlackJAXNUTSSampler",
    "numpyrohmc": "NumPyroHMCSampler",
    "numpyro_hmc": "NumPyroHMCSampler",
    "numpyronuts": "NumPyroNUTSSampler",
    "numpyro_nuts": "NumPyroNUTSSampler",
    "jaxnestedsampler": "JAXNSSampler",
    "jaxns": "JAXNSSampler",
    "jaxnsshardedstaticnestedsampler": "JAXNSShardedStaticNestedSampler",
    "jaxnsshardedstaticsestedsampler": "JAXNSShardedStaticNestedSampler",
}


def _resolve_sampler_class(sampler_type: str) -> type[Sampler] | None:
    sampler_class_name = _SAMPLER_CLASS_NAMES.get(sampler_type)

    if sampler_class_name is None:
        return None

    sampler_module = importlib.import_module("petitRADTRANS.retrieval.sampler")
    return getattr(sampler_module, sampler_class_name)


class Retrieval:
    """Implement the retrieval method using petitRADTRANS.

    A RetrievalConfig object is passed to this class to describe the retrieval data, parameters
    and priors. The run() method then samples the parameter space, producing
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
            Produce plots and data files for random samples drawn from the outputs of the sampler.
        corner_plot_names : List(Str)
            List of additional retrieval names that should be included in the corner plotlib.
        reference_data_name : str
            Name of the dataset to use as the central plotting reference. This controls which dataset supplies the
            default Radtrans object and forward model for pressure-temperature plots, abundance plots, and related
            full-range evaluation utilities. If None, the first key in ``configuration.data`` is used.
        use_prt_plot_style : Bool
            Use the petitRADTRANS plotting style as described in style.py. Recommended to
            turn this parameter to false if you want to use interactive plotting, or if the
            test_plotting parameter is True.
        test_plotting : Bool
            Only use when running locally. A boolean flag that will produce plots
            for each sample.
        uncertainties_mode : Str
            Uncertainties handling method during the retrieval.
                - "default": the uncertainties are fixed.
                - "optimize": automatically optimize for uncertainties, following Gibson et al. 2020
                  (https://doi.org/10.1093/mnras/staa228).
                - "retrieve": uncertainties are scaled with a coefficient, which is retrieved.
                - "retrieve_add": a fixed scalar is added to the uncertainties, and is retrieved.
        print_log_likelihood_for_debugging : bool
            If True, the current log likelihood of a forward model run will be printed to the console.
        generate_mock_data : bool
            If True, the retrieval will generate a mock data set by sampling the prior distributions and bring
            it into the exact same shape as the input data. This is useful for testing the retrieval setup in input
            = output tests. The mock data will be saved in the mock_data folder in the run directory, with the
            following file names: data.name + '_mock_data.dat'.
    """

    def __init__(
        self,
        configuration: RetrievalConfig,
        output_directory: str = os.getcwd(),
        evaluate_sample_spectra: bool = False,
        corner_plot_names: list[str] = None,
        reference_data_name: str = None,
        use_prt_plot_style: bool = True,
        test_plotting: bool = False,
        uncertainties_mode: str = "default",
        print_log_likelihood_for_debugging: bool = False,
        generate_mock_data: bool = False,
    ):
        self.configuration = configuration

        if len(self.configuration.line_species) < 1:
            for data_name, d in self.configuration.data.items():
                if d.radtrans_object is not None:
                    if len(d.radtrans_object.line_species) < 1:
                        warnings.warn(
                            "there are no line species present in the given Radtrans object"
                        )
                        break
                else:
                    warnings.warn(
                        "there are no line species present in the run definition"
                    )
                    break

        # Could actually be merged with RetrievalConfig
        _, self.comm, self.rank = get_mpi_context()
        self.uncertainties_mode = uncertainties_mode
        self.print_log_likelihood_for_debugging = print_log_likelihood_for_debugging
        self.generate_mock_data = generate_mock_data
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
        self.evaluate_sample_spectra = evaluate_sample_spectra

        # PMN analysis
        self.analyzer = None
        self.sampler = None

        self.samples = {}
        self.param_dictionary = {}
        # TODO reference_data_name should be in retrieval not in plotter
        self.plotter = RetrievalPlotter(self, reference_data_name=reference_data_name)

        # Set up pretty plotting
        if use_prt_plot_style:
            # import petitRADTRANS.retrieval.plot_style  # commented to avoid breaking of mpl
            pass
        self.prt_plot_style = use_prt_plot_style

        # Path to input opacities
        self.path = petitradtrans_config_parser.get_input_data_path()

        # Setup Directories
        if "pymultinest" in self.configuration.sampler_type.lower():
            os.makedirs(os.path.join(self.output_directory, "out_PMN"), exist_ok=True)

        os.makedirs(
            os.path.join(
                self.output_directory, "evaluate_" + self.configuration.retrieval_name
            ),
            exist_ok=True,
        )

        # Setup pRT Objects for each data structure.
        self.initialise_radtrans_objects()
        self.parameter_layout = ParameterLayout.from_configuration(self.configuration)
        self._n_free_params_total_retrieval = self.parameter_layout.n_free_parameters
        self._free_parameter_names_cache = list(
            self.parameter_layout.free_parameter_names
        )
        self.runtime = RetrievalRuntime.from_configuration(
            self.configuration,
            parameter_layout=self.parameter_layout,
            n_free_parameters=self._n_free_params_total_retrieval,
        )
        self.latest_sampler_results = None
        if self.configuration.run_mode == "evaluate":
            sample_dict, parameters_dict = self.get_samples(
                output_directory=self.output_directory
            )
            samples_use = copy.copy(sample_dict[self.configuration.retrieval_name])
            parameters_read = parameters_dict[self.configuration.retrieval_name]
            log_l, best_fit_index = self.get_best_fit_likelihood(samples_use)
            best_fit_sample = samples_use[:-1, best_fit_index]
            self.best_fit_parameters = self.build_param_dict(
                best_fit_sample, parameters_read
            )
        try:
            self.generate_retrieval_summary(None)
        except (ValueError, FileNotFoundError) as e:
            print(f"Could not generate initial summary file! Error was: {str(e)}")

    def _check_errors(self):
        data_are_valid = self._data_are_valid()

        if not data_are_valid:
            warnings.warn(
                "Data may not be suitable for retrievals due to invalid values"
            )

        self._error_check_model_function()

    def _data_are_valid(self, data=None):
        tested_attributes = ["wavelengths", "spectrum", "uncertainties"]
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
            if data.dtype == "O":
                for d in data:
                    valid = valid and self._data_are_valid(d)
            else:
                if np.any(np.isnan(data)):
                    warnings.warn(
                        f"NaN detected ({np.nonzero(np.isnan(data))[0].size} / {data.size})"
                    )
                    valid = False

                if np.any(~np.isfinite(data)):
                    warnings.warn(
                        f"Infinite value detected ({np.nonzero(~np.isfinite(data))[0].size} / {data.size})"
                    )
                    valid = False

                if np.any(np.less(data, 0)):
                    warnings.warn(
                        f"Negative value detected ({np.nonzero(np.less(data, 0))[0].size} / {data.size})\n"
                        "Make sure that this makes sense for the data set you are considering."
                    )
                    valid = False
        else:
            self._data_are_valid(np.array(data))

        return valid

    def _error_check_model_function(self):
        for data_name in self.configuration.data:
            print(f"Testing model function for data '{data_name}'...")
            message = None
            wavelengths = None
            model = None
            exc_info = None

            cube = self.get_parameters_prior_mid_range_values()

            try:
                model_returned_values = self.calculate_forward_model(
                    parameters=cube,
                    data=data_name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                    copy_configuration_parameters=False,
                )

                # TODO the generating function should always return the same number of values
                if (
                    len(model_returned_values) == 3
                ):  # handle case where beta is returned
                    wavelengths, model, _ = model_returned_values
                else:
                    wavelengths, model = model_returned_values
            except KeyError:
                exc_info = sys.exc_info()
                message = "and may be caused by an invalid parameter dictionary or invalid abundances"
            except ValueError:
                exc_info = sys.exc_info()
                message = (
                    "and may be caused by erroneous calculations or invalid inputs"
                )
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
                    message = (
                        "this error was directly caused by the above error " + message
                    )
                    traceback.print_exception(*exc_info)
                    raise RuntimeError(message)

            if wavelengths is None or model is None:
                raise ValueError(
                    "unable to compute a spectrum (output wavelengths and spectrum are both None), "
                    "check your inputs and your model function"
                )

        print("No errors detected in the model functions!")

    def _rebin_opacities(self, resolution: float):
        species = []

        for line in self.configuration.line_species:
            # _line = line.split('.', 1)[0]  # remove possible previous spectral info
            _line = Opacity.get_species_isotopologue_name(line)
            matches = CorrelatedKOpacity.find(
                path_input_data=self.path,
                category="correlated_k_opacities",
                species=CorrelatedKOpacity.join_species_all_info(
                    species_name=_line,
                    spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                        resolution
                    ),
                ),
                find_all=True,
                search_online=False,
            )

            if len(matches) == 0:
                species.append(line)

        # If not, setup low-res c-k tables
        if len(species) > 0:
            if self.rank == 0:
                CorrelatedKOpacity.exo_k_multiple_rebin_from_species(
                    species=species, resolving_power=resolution
                )

            if self.comm is not None:
                self.comm.barrier()

    def build_param_dict(
        self, sample: npt.NDArray[float], free_param_names: list[str]
    ) -> dict[str, Parameter]:
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
            parameters : dict
                A dictionary of Parameters, with values set to the values
                in sample.
        """
        sample_values = self._flatten_free_parameter_values(sample)

        if sample_values.size < len(free_param_names):
            raise ValueError(
                f"Expected at least {len(free_param_names)} free-parameter values, got {sample_values.size}"
            )

        free_parameter_values = {
            parameter_name: sample_values[index]
            for index, parameter_name in enumerate(free_param_names)
        }
        parameters = {}

        for parameter_name, parameter in self.configuration.parameters.items():
            if parameter.is_free_parameter and parameter_name in free_parameter_values:
                parameters[parameter_name] = Parameter(
                    name=parameter_name,
                    is_free_parameter=False,
                    value=free_parameter_values[parameter_name],
                )
            else:
                parameters[parameter_name] = Parameter(
                    name=parameter_name, is_free_parameter=False, value=parameter.value
                )
        return parameters

    @staticmethod
    def _call_model_generating_function(
        model_generating_function: Callable,
        radtrans_object,
        parameters: dict[str, Parameter],
        *,
        data_name: str,
        data_object: Data | None = None,
        pt_plot_mode: bool = False,
        adaptive_mesh_refinement: bool = False,
    ):
        _, runtime_ready_model_function = coerce_model_generating_function(
            model_generating_function
        )

        physical_params = PhysicalParams.from_mapping(
            {name: parameter.value for name, parameter in parameters.items()}
        )
        model_metadata = None
        variability_mode = False

        if data_object is not None:
            observation_state = data_object.to_observation_state()
            model_metadata = observation_state.metadata
            variability_mode = (
                observation_state.variability_atmospheric_column_model_flux_return_mode
            )

        model_context = _build_model_context_from_parameters(
            data_name,
            "unknown",
            radtrans_object,
            parameters,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            variability_atmospheric_column_model_flux_return_mode=variability_mode,
            model_metadata=model_metadata,
        )
        model_result = run_runtime_model_adapter(
            runtime_ready_model_function,
            model_context,
            physical_params,
            pt_plot_mode=pt_plot_mode,
        )

        return _model_result_to_legacy_output(model_result)

    def calculate_forward_model(
        self,
        parameters=None,
        data=None,
        pt_plot_mode: bool = False,
        adaptive_mesh_refinement: bool = False,
        copy_configuration_parameters: bool = True,
        **kwargs,
    ):
        """Calculate the forward model associated with the given data, for the given parameters.

        Args:
            parameters:
                Parameters of the forward models.
                Can be a dictionary with parameter names as keys, or a string, or None.
                Possible string values are:
                    - 'best fit': return the forward model with the retrieved best fit parameters.
                    - 'median': return the forward model with the retrieved median parameters.
                    - 'quantile': return the forward model with retrieved parameters at the given quantile.
                If ``parameter='quantile'``, then a ``quantile`` (float) keyword argument must be added.
                If None, the free parameters of the retrieval are set to their prior mid-range value.
            data:
                Can be a string, or a dictionary with data name as keys and Data objects as values, or None.
                If None, the forward models of all the data in the retrieval configuration are calculated.
            pt_plot_mode:
                If True, the model function should return the pressure and temperature arrays before computing the flux.
            adaptive_mesh_refinement:
                If True, use an adaptive high resolution pressure grid around the location of cloud condensation.
                This will increase the size of the pressure grid by a constant factor that can be adjusted in the
                setup_pres function.
            copy_configuration_parameters:
                If True, copy the configuration parameters to prevent unwanted modifications.
        Returns:
            A dictionary with the outputs of the forward models for the requested data, or, if data is a string, the
            output of the forward model corresponding to these data.
        """
        if parameters is None:
            parameters = {}
        elif isinstance(parameters, str):
            if parameters == "best fit":
                parameters = self.get_best_fit_parameters(return_max_likelihood=False)
            elif parameters == "median":
                parameters = self.get_quantile_parameters(0.5)
            elif parameters == "quantile":
                parameters = self.get_quantile_parameters(**kwargs)

        if copy_configuration_parameters:
            configuration_parameters = copy.deepcopy(
                self.configuration.parameters
            )  # copy to prevent unwanted modifs
        else:
            configuration_parameters = self.configuration.parameters

        i_p = 0  # parameter count

        if isinstance(parameters, dict):
            cube = self.get_parameters_prior_mid_range_values()

            # Check if the given parameters are all set in the configuration
            for parameter_name in parameters:
                if parameter_name not in configuration_parameters:
                    available_parameters = list_str2str(configuration_parameters)

                    raise KeyError(
                        f"retrieved parameter '{parameter_name}' is not a configured parameter of this retrieval\n"
                        f"Available parameters are: {available_parameters}"
                    )

            # Set the parameters to their requested values
            for parameter_name in configuration_parameters:
                if configuration_parameters[parameter_name].is_free_parameter:
                    if parameter_name not in parameters and len(parameters) > 0:
                        print(
                            f"No value given to retrieved parameter '{parameter_name}', "
                            f"assuming the prior mid-range value ({cube[i_p]})"
                        )
                    else:
                        if parameter_name in parameters:
                            lower_boundary = configuration_parameters[
                                parameter_name
                            ].get_param_uniform(0)
                            upper_boundary = configuration_parameters[
                                parameter_name
                            ].get_param_uniform(1)

                            if lower_boundary > upper_boundary:
                                upper_boundary, lower_boundary = (
                                    lower_boundary,
                                    upper_boundary,
                                )  # swap boundaries

                            if not (
                                lower_boundary
                                <= parameters[parameter_name]
                                <= upper_boundary
                            ):
                                warnings.warn(
                                    f"value given for parameter '{parameter_name}' ({parameters[parameter_name]}) "
                                    f"is not within its prior boundaries ({lower_boundary} to {upper_boundary})"
                                )

                            cube[i_p] = parameters[parameter_name]

                    configuration_parameters[parameter_name].set_param(cube[i_p])
                    i_p += 1
                elif parameter_name in parameters:
                    warnings.warn(
                        f"parameter '{parameter_name}' is not a free parameter"
                    )
                    configuration_parameters[parameter_name].set_param(
                        parameters[parameter_name]
                    )
        else:  # assume that the 'parameters' argument is the list of parameters values in the correct order
            for parameter_name in configuration_parameters:
                if configuration_parameters[parameter_name].is_free_parameter:
                    if (
                        configuration_parameters[
                            parameter_name
                        ].transform_prior_cube_coordinate
                        is not None
                    ):
                        # TODO: tfp distributions enforce this natively?
                        lower_boundary = configuration_parameters[
                            parameter_name
                        ].get_param_uniform(0)
                        upper_boundary = configuration_parameters[
                            parameter_name
                        ].get_param_uniform(1)

                        if lower_boundary > upper_boundary:
                            upper_boundary, lower_boundary = (
                                lower_boundary,
                                upper_boundary,
                            )  # swap boundaries

                        if not (lower_boundary <= parameters[i_p] <= upper_boundary):
                            warnings.warn(
                                f"value given for parameter '{parameter_name}' ({parameters[i_p]}) "
                                f"is not within its prior boundaries ({lower_boundary} to {upper_boundary})"
                            )

                    configuration_parameters[parameter_name].set_param(parameters[i_p])
                    i_p += 1

        model_returned_values = {}
        return_dict = False

        if data is None:
            data = self.configuration.data
            return_dict = True
        elif isinstance(data, str):
            data = {data: self.configuration.data[data]}

        for name, _data in data.items():
            if (
                return_dict or len(data) > 1
            ):
            if _data.external_radtrans_reference is not None:
                radtrans_object = self.configuration.data[
                    _data.external_radtrans_reference
                ].radtrans_object
            else:
                radtrans_object = _data.radtrans_object

            model_returned_values[name] = self._call_model_generating_function(
                _data.model_generating_function,
                radtrans_object,
                configuration_parameters,
                data_name=name,
                data_object=_data,
                pt_plot_mode=pt_plot_mode,
                adaptive_mesh_refinement=adaptive_mesh_refinement,
            )

        if return_dict or len(data) > 1:
            return model_returned_values
        else:
            return model_returned_values[list(data.keys())[0]]

    def calculate_temperature_profile(
        self,
        parameters=None,
        data=None,
        adaptive_mesh_refinement: bool = False,
        copy_configuration_parameters: bool = True,
        **kwargs,
    ):
        """Calculate the temperature profile associated with the given data, for the given parameters.
        Wrapper for calculate_forward_model.

        Args:
            parameters:
                Parameters of the forward models.
                Can be a dictionary with parameter names as keys, or a string, or None.
                Possible string values are:
                    - 'best fit': return the forward model with the retrieved best fit parameters.
                    - 'median': return the forward model with the retrieved median parameters.
                    - 'quantile': return the forward model with retrieved parameters at the given quantile.
                If ``parameter='quantile'``, then a ``quantile`` (float) keyword argument must be added.
                If None, the free parameters of the retrieval are set to their prior mid-range value.
            data:
                Can be a string, or a dictionary with data name as keys and Data objects as values, or None.
                If None, the forward models of all the data in the retrieval configuration are calculated.
            pt_plot_mode:
                If True, the model function should return the pressure and temperature arrays before computing the flux.
            adaptive_mesh_refinement:
                If True, use an adaptive high resolution pressure grid around the location of cloud condensation.
                This will increase the size of the pressure grid by a constant factor that can be adjusted in the
                setup_pres function.
            copy_configuration_parameters:
                If True, copy the configuration parameters to prevent unwanted modifications.
        Returns:
            A dictionary with the outputs of the forward models for the requested data, or, if data is a string, the
            output of the forward model corresponding to these data.
        """
        if parameters is not None and not isinstance(parameters, (dict, str)):
            parameter_array = np.asarray(parameters)

            if parameter_array.ndim == 2:
                free_parameter_names = list(self._get_parameter_layout().free_parameter_names)
                sample_matrix, free_parameter_names = self._coerce_parameter_sample_matrix(
                    parameters,
                    free_parameter_names,
                )

                if sample_matrix is None:
                    raise ValueError(
                        "expected batched temperature-profile parameters with shape "
                        "(n_samples, n_free_parameters) or (n_free_parameters, n_samples)"
                    )

                model_returned_values = {}
                return_dict = False

                if data is None:
                    data = self.configuration.data
                    return_dict = True
                elif isinstance(data, str):
                    data = {data: self.configuration.data[data]}

                for name, _data in data.items():
                    if _data.external_radtrans_reference is not None:
                        radtrans_object = self.configuration.data[
                            _data.external_radtrans_reference
                        ].radtrans_object
                    else:
                        radtrans_object = _data.radtrans_object

                    model_returned_values[name] = self._get_batched_full_range_model_outputs(
                        sample_matrix,
                        free_parameter_names,
                        model_generating_function=_data.model_generating_function,
                        prt_object=radtrans_object,
                        prt_reference=_data,
                        pt_plot_mode=True,
                        adaptive_mesh_refinement=adaptive_mesh_refinement,
                    )

                if return_dict or len(data) > 1:
                    return model_returned_values

                return model_returned_values[list(data.keys())[0]]

        return self.calculate_forward_model(
            parameters,
            data,
            True,
            adaptive_mesh_refinement,
            copy_configuration_parameters,
            **kwargs,
        )

    @classmethod
    def from_data(
        cls,
        data: dict[str, Data],
        retrieved_parameters: dict[str, dict],
        retrieval_name: str = "retrieval_name",
        run_mode: str = "retrieval",
        adaptive_mesh_refinement: bool = False,
        output_directory: str = "",
        evaluate_sample_spectra: bool = False,
        corner_plot_names: list[str] = None,
        reference_data_name: str = None,
        use_prt_plot_style: bool = True,
        test_plotting: bool = False,
        uncertainties_mode: str = "default",
        scattering_in_emission: bool = False,
        pressures: np.ndarray = None,
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
                Can be either 'retrieval', which runs the retrieval, or 'evaluate', which produces plots
                from the best fit parameters stored in the samples file.
            adaptive_mesh_refinement : Bool
                Use an adaptive high resolution pressure grid around the location of cloud condensation.
                This will increase the size of the pressure grid by a constant factor that can be adjusted
                in the setup_pres function.
            output_directory : Str
                The directory in which the output folders should be written
            evaluate_sample_spectra : Bool
                Produce plots and data files for random samples drawn from the outputs of the sampler.
            corner_plot_names : List(Str)
                List of additional retrieval names that should be included in the corner plotlib.
            reference_data_name : str
                Name of the dataset to use as the central plotting reference. If None, the first key in ``data``
                is used.
            use_prt_plot_style : Bool
                Use the petitRADTRANS plotting style as described in style.py. Recommended to
                turn this parameter to false if you want to use interactive plotting, or if the
                test_plotting parameter is True.
            test_plotting : Bool
                Only use when running locally. A boolean flag that will produce plots
                for each sample.
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
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            scattering_in_emission=scattering_in_emission,  # scattering is not necessary when using SpectralModels
            pressures=pressures,  # pressures is not necessary when using SpectralModels
        )

        # Convert retrieved parameters to RetrievalParameters
        if isinstance(retrieved_parameters, dict):
            retrieved_parameters = RetrievalParameter.from_dict(retrieved_parameters)

        # Add retrieved parameters to the RetrievalConfig
        for parameter in retrieved_parameters:
            if not hasattr(parameter, "prior_function"):
                raise AttributeError(
                    f"'{type(parameter)}' object has no attribute 'prior_function': "
                    f"usage of dictionary or a '{type(RetrievalParameter)}' instance is recommended"
                )

            retrieval_configuration.add_parameter(
                name=parameter.name,
                is_free_parameter=True,  # fixed parameters must be stored within the Data.model_generating_function
                value=None,
                transform_prior_cube_coordinate=parameter.prior_function,
            )

        # Add the data to the RetrievalConfig
        for data_name, d in data.items():
            retrieval_configuration.data[data_name] = d

        # Instantiate the new Retrieval object
        return cls(
            configuration=retrieval_configuration,
            output_directory=output_directory,
            evaluate_sample_spectra=evaluate_sample_spectra,
            corner_plot_names=corner_plot_names,
            reference_data_name=reference_data_name,
            use_prt_plot_style=use_prt_plot_style,
            test_plotting=test_plotting,
            uncertainties_mode=uncertainties_mode,
        )

    def generate_retrieval_summary(
        self, stats: dict = None, sampler_parameters: dict = None
    ) -> None:
        """
        This function produces a human-readable text file describing the retrieval.
        It includes all the fixed and free parameters, the limits of the priors (if uniform),
        a description of the data used, and if the retrieval is complete, a summary of the
        best fit parameters and model evidence.

        Args:
            stats : dict
                Optional sampler-specific statistics object. If the active sampler exposes
                a standardized summary payload, that is preferred.
        """
        sampler_summary = self._collect_sampler_summary(
            stats=stats,
            sampler_parameters=sampler_parameters,
        )

        with open(
            os.path.join(
                self.output_directory,
                "evaluate_" + self.configuration.retrieval_name,
                self.configuration.retrieval_name + "_ret_summary.txt",
            ),
            "w+",
        ) as summary:
            from datetime import datetime
            from importlib.metadata import version, PackageNotFoundError

            summary.write(self.configuration.retrieval_name + "\n")
            summary.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + "\n")
            try:
                prt_version = version("petitRADTRANS")
            except PackageNotFoundError:
                prt_version = petitRADTRANS.__version__
            summary.write(f"pRT v{prt_version}\n")
            summary.write(self.output_directory + "\n")

            summary.write("\n")
            self._write_sampler_summary_section(
                summary=summary, sampler_summary=sampler_summary
            )
            summary.write("\n")

            summary.write("Fixed Parameters\n")

            for key, value in self.configuration.parameters.items():
                if not value.is_free_parameter:
                    if isinstance(value.value, float):
                        summary.write(f"    {key} = {value.value:.3f}\n")

                if not value.is_free_parameter:
                    if isinstance(value.value, float):
                        summary.write(f"    {key} = {value.value:.3f}\n")
                    else:
                        summary.write(f"    {key} = {value.value}\n")

            radtrans_object = self.plotter.get_reference_data().radtrans_object

            summary.write("    line_species = ")
            summary.write(f"{radtrans_object.line_species}")
            summary.write("\n")
            summary.write("    cloud_species = ")
            summary.write(f"{radtrans_object.cloud_species}")
            summary.write("\n")
            summary.write("Free Parameters, Prior^-1(0), Prior^-1(1)\n")

            for key, value in self.configuration.parameters.items():
                if value.is_free_parameter:
                    if value.transform_prior_cube_coordinate is not None:
                        low = value.transform_prior_cube_coordinate(0.0000001)
                        high = value.transform_prior_cube_coordinate(0.9999999)

                        if value.corner_transform is not None:
                            low = value.corner_transform(low)
                            high = value.corner_transform(high)
                        summary.write(f"    {key} = {low:3f}, {high:3f}\n")

                    elif value.distribution is not None:
                        summary.write(f"    {key} = {value.distribution.name}\n")
                        for (
                            dist_param,
                            dist_param_value,
                        ) in value.distribution.parameters.items():
                            summary.write(
                                f"        {dist_param} = {dist_param_value}\n"
                            )

            summary.write("\n")
            summary.write("Data\n")

            for name, data in self.configuration.data.items():
                summary.write(name + "\n")

                if data.path_to_observations is not None:
                    summary.write("    " + data.path_to_observations + "\n")

                if data.model_generating_function is not None:
                    summary.write(
                        f"    Model Function = {data.model_generating_function.__name__}\n"
                    )

                if data.scale_flux:
                    summary.write(f"    scale factor = {data.scale_factor:.2f}\n")

                if data.scale_uncertainties:
                    summary.write(f"    scale err factor = {data.scale_factor:.2f}\n")

                if data.fit_flux_offset:
                    summary.write("    offset = True\n")

                if data.system_distance is not None:
                    summary.write(f"    distance = {data.system_distance}\n")

                if data.data_resolution is not None:
                    summary.write(f"    data resolution = {data.data_resolution}\n")

                if data.model_resolution is not None:
                    summary.write(f"    model resolution = {data.model_resolution}\n")

                if data.external_radtrans_reference is not None:
                    summary.write(
                        f"    external_pRT_reference = {data.external_radtrans_reference}\n"
                    )

                if isinstance(data, PhotometryData):
                    summary.write(
                        f"    photometric width = {data.photometric_bin_edges[0]:.4f}"
                        + f"--{data.photometric_bin_edges[1]:.4f} um\n"
                    )
                    summary.write(
                        "    Photometric transform function = "
                        + data.photometric_transformation_function.__name__
                        + "\n"
                    )

            summary.write("\n")

            best_fit_summary = self._collect_best_fit_summary()
            if best_fit_summary is not None:
                evidence_approximations = best_fit_summary.get(
                    "evidence_approximations", {}
                )
                summary.write("Model Comparison Statistics\n")
                summary.write(
                    f"    log_likelihood = {best_fit_summary['log_likelihood']:.6g}\n"
                )
                summary.write(
                    f"    𝛘^2/n_wlen = {best_fit_summary['chi2_wavelengths']:.2f}\n"
                )
                summary.write(f"    𝛘^2/DoF = {best_fit_summary['chi2_d_o_f']:.2f}\n")
                if evidence_approximations:
                    for metric_name in ("bic", "aic", "bpics"):
                        metric_value = evidence_approximations.get(metric_name)
                        if metric_value is None:
                            continue
                        summary.write(
                            f"    {metric_name.upper()} = {metric_value:.6g}\n"
                        )
                summary.write("\n")
                summary.write("Best Fit Parameters\n")
                for key, value in best_fit_summary["parameters"].items():
                    if key in ["FstarWlenMicron"]:
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

    def _collect_best_fit_summary(self) -> dict | None:
        try:
            samples_dict, parameter_dict = self.get_samples(
                names=[self.configuration.retrieval_name],
                output_directory=self.output_directory,
                ret_names=None,
            )
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

        samples_use = samples_dict.get(self.configuration.retrieval_name)
        parameters_read = parameter_dict.get(self.configuration.retrieval_name)

        if samples_use is None or parameters_read is None:
            return None

        samples_use = np.asarray(samples_use)
        if (
            samples_use.ndim != 2
            or samples_use.shape[0] == 0
            or samples_use.shape[1] == 0
        ):
            return None

        log_l, best_fit_index = self.get_best_fit_likelihood(
            samples_use, print_value=False
        )
        best_fit_sample = samples_use[:, best_fit_index]
        best_fit_parameters = self.build_param_dict(
            best_fit_sample[:-1], parameters_read
        )

        chi2_wavelengths = self.get_reduced_chi2_from_sample(
            sample=best_fit_sample,
            subtract_n_parameters=False,
            verbose=False,
            show_chi2=False,
        )
        chi2_d_o_f = self.get_reduced_chi2_from_sample(
            sample=best_fit_sample,
            subtract_n_parameters=True,
            verbose=False,
            show_chi2=False,
        )
        evidence_approximations = self._collect_evidence_approximations(
            max_logl=log_l,
            logl_samples=samples_use[-1, :],
        )

        self.samples = samples_dict
        self.param_dictionary = parameter_dict
        self.best_fit_parameters = best_fit_parameters

        return {
            "log_likelihood": log_l,
            "chi2_wavelengths": chi2_wavelengths,
            "chi2_d_o_f": chi2_d_o_f,
            "evidence_approximations": evidence_approximations,
            "parameters": best_fit_parameters,
        }

    def _count_data_points(self) -> int:
        return sum(np.size(data.spectrum) for data in self.configuration.data.values())

    def _count_free_parameters(self) -> int:
        return sum(
            1
            for parameter in self.configuration.parameters.values()
            if parameter.is_free_parameter
        )

    def _get_log_weight_samples(self, results=None) -> np.ndarray | None:
        if results is None:
            results = self._get_active_sampler_results()

        log_weight_samples = self._get_results_field(
            results,
            "log_weights",
            "log_posterior_weights",
            "sample_log_weights",
        )
        if log_weight_samples is not None:
            return np.asarray(log_weight_samples, dtype=float).reshape(-1)

        if isinstance(results, dict):
            weighted_samples = results.get("weighted_samples")
            if isinstance(weighted_samples, dict):
                weighted_logl = self._get_results_field(
                    weighted_samples, "logl", "logL"
                )
                weighted_weights = self._get_results_field(
                    weighted_samples,
                    "log_weights",
                    "weights",
                    "sample_weights",
                )

                if weighted_weights is None:
                    return None

                weighted_weights = np.asarray(weighted_weights, dtype=float).reshape(-1)
                if weighted_weights.size == 0:
                    return None

                if np.all(weighted_weights > 0):
                    weighted_log_weights = np.log(weighted_weights)
                else:
                    weighted_log_weights = weighted_weights

                if weighted_logl is None:
                    return weighted_log_weights

                weighted_logl = np.asarray(weighted_logl, dtype=float).reshape(-1)
                if weighted_logl.shape == weighted_log_weights.shape:
                    return weighted_log_weights

        return None

    def _collect_evidence_approximations(
        self,
        max_logl: float,
        logl_samples: npt.ArrayLike,
        log_weights: npt.ArrayLike = None,
    ) -> dict[str, float] | None:
        logl_samples = np.asarray(logl_samples, dtype=float).reshape(-1)

        if logl_samples.size == 0:
            return None

        n_data = self._count_data_points()
        n_params = self._count_free_parameters()

        if n_data < 1 or n_params < 0:
            return None

        if log_weights is None:
            log_weights = self._get_log_weight_samples()

        if log_weights is not None:
            log_weights = np.asarray(log_weights, dtype=float).reshape(-1)
            if log_weights.size != logl_samples.size:
                log_weights = None

        return {
            "bic": get_bic(max_logl=max_logl, n_data=n_data, n_params=n_params),
            "aic": get_aic(max_logl=max_logl, n_params=n_params),
            "bpics": get_bpics(
                logl_samples=logl_samples, n_params=n_params, log_weights=log_weights
            ),
        }

    def _collect_sampler_summary(
        self, stats: dict = None, sampler_parameters: dict = None
    ) -> dict:
        sampler_summary = {
            "sampler": (
                self.sampler.__class__.__name__ if self.sampler is not None else None
            ),
            "parameters": {},
            "diagnostics": {},
            "evidence": {},
            "posterior": [],
        }

        if self.sampler is not None and hasattr(self.sampler, "get_summary"):
            sampler_summary = self.sampler.get_summary(
                free_parameter_names=self._free_parameter_names_cache
            )

        if sampler_parameters is not None:
            sampler_summary.setdefault("parameters", {}).update(sampler_parameters)

        if (
            stats
            and isinstance(stats, dict)
            and "global evidence" in stats
            and "marginals" in stats
        ):
            sampler_summary.setdefault("evidence", {})
            sampler_summary["evidence"].setdefault("ln_z", stats.get("global evidence"))
            sampler_summary["evidence"].setdefault(
                "ln_z_err", stats.get("global evidence error")
            )
            if not sampler_summary.get("posterior"):
                posterior = []
                for param_name, marginal in zip(
                    self._free_parameter_names_cache, stats["marginals"]
                ):
                    lo, hi = marginal["1sigma"]
                    med = marginal["median"]
                    posterior.append(
                        {
                            "name": param_name,
                            "median": med,
                            "lower_1sigma": lo,
                            "upper_1sigma": hi,
                        }
                    )

                sampler_summary["posterior"] = posterior

        return sampler_summary

    def _write_sampler_summary_section(self, summary, sampler_summary: dict) -> None:
        summary.write("Sampler\n")
        if sampler_summary.get("sampler") is not None:
            summary.write(f"    sampler = {sampler_summary['sampler']}\n")

        if sampler_summary.get("parameters"):
            summary.write("Sampler Parameters\n")
            for key, value in sampler_summary["parameters"].items():
                summary.write(f"    {key} = {value}\n")

        if sampler_summary.get("diagnostics"):
            summary.write("Sampler Diagnostics\n")
            for key, value in sampler_summary["diagnostics"].items():
                if value is None:
                    continue
                summary.write(f"    {key} = {value}\n")

        evidence = sampler_summary.get("evidence", {})
        if evidence:

            summary.write("Posterior Summary\n")
            if evidence.get("ln_z") is not None:
                ln_z = evidence["ln_z"]
                ln_z_err = evidence.get("ln_z_err")
                if ln_z_err is not None:
                    summary.write(f"    ln Z = {ln_z:.6g} +- {ln_z_err:.6g}\n")
                    summary.write(
                        f"    log Z = {ln_z / np.log(10):.6g} +- {ln_z_err / np.log(10):.6g}\n"
                    )
                else:
                    summary.write(f"    ln Z = {ln_z:.6g}\n")

        if sampler_summary.get("posterior"):
            if not evidence:
                summary.write("Posterior Summary\n")
            summary.write("Statistical Fit Parameters\n")
            for posterior_entry in sampler_summary["posterior"]:
                name = posterior_entry["name"]
                median = posterior_entry.get("median")
                lo = posterior_entry.get("lower_1sigma")
                hi = posterior_entry.get("upper_1sigma")

                if (
                    name not in self.configuration.parameters
                    or median is None
                    or lo is None
                    or hi is None
                ):
                    continue

                transform = self.configuration.parameters[name].corner_transform
                if transform is not None:
                    median = transform(median)
                    lo = transform(lo)
                    hi = transform(hi)

                sigma = (hi - lo) / 2
                if sigma == 0:
                    precision = 3
                else:
                    precision = max(0, int(-np.floor(np.log10(abs(sigma)))) + 1)

                fmt = f"{{:.{precision}f}}"
                summary.write(
                    f"    {name:<15}{fmt.format(median)} +- {fmt.format(sigma)}\n"
                )

    @staticmethod
    def _get_results_field(results, *field_names):
        if results is None:
            return None

        for field_name in field_names:
            if isinstance(results, dict) and field_name in results:
                return results[field_name]
            if hasattr(results, field_name):
                return getattr(results, field_name)

        return None

    def _get_active_sampler_results(self):
        if getattr(self, "latest_sampler_results", None) is not None:
            return self.latest_sampler_results

        if self.sampler is None:
            return None

        if hasattr(self.sampler, "get_results"):
            try:
                sampler_results = self.sampler.get_results()
            except TypeError:
                sampler_results = None

            if sampler_results is not None:
                return sampler_results

        return getattr(self.sampler, "results", None)

    def _coerce_parameter_sample_matrix(self, parameter_samples, parameter_names):
        if parameter_samples is None:
            return None, parameter_names

        if isinstance(parameter_samples, dict):
            ordered_names = [
                name for name in parameter_names if name in parameter_samples
            ]
            if not ordered_names:
                ordered_names = list(parameter_samples.keys())

            if not ordered_names:
                return None, parameter_names

            matrix = np.column_stack(
                [
                    np.asarray(parameter_samples[name]).reshape(-1)
                    for name in ordered_names
                ]
            )
            return matrix, ordered_names

        matrix = np.asarray(parameter_samples)
        if matrix.ndim == 1:
            if len(parameter_names) != 1:
                return None, parameter_names
            matrix = matrix.reshape(-1, 1)
        elif matrix.ndim != 2:
            return None, parameter_names

        if matrix.shape[1] == len(parameter_names):
            return matrix, parameter_names

        if matrix.shape[0] == len(parameter_names):
            return matrix.T, parameter_names

        if len(parameter_names) == 1:
            return matrix.reshape(-1, 1), parameter_names

        return None, parameter_names

    def _evaluate_sample_log_likelihoods(self, parameter_samples):
        return np.asarray(
            [
                self.log_likelihood(np.asarray(sample, dtype=float))
                for sample in np.asarray(parameter_samples)
            ],
            dtype=float,
        )

    def _build_samples_from_sampler_results(self, results=None, parameter_names=None):
        if parameter_names is None:
            parameter_names = list(self._free_parameter_names_cache)

        if results is None:
            results = self._get_active_sampler_results()

        if results is None:
            return None, parameter_names

        parameter_samples = self._get_results_field(
            results, "samples", "parametrised_samples"
        )
        matrix, parameter_names = self._coerce_parameter_sample_matrix(
            parameter_samples, parameter_names
        )

        if matrix is None and isinstance(results, dict):
            weighted_samples = results.get("weighted_samples")
            if isinstance(weighted_samples, dict):
                matrix, parameter_names = self._coerce_parameter_sample_matrix(
                    weighted_samples.get("points"),
                    parameter_names,
                )

        if matrix is None:
            return None, parameter_names

        log_likelihood = self._get_results_field(
            results,
            "model_log_likelihood",
            "log_likelihood",
            "log_likelihoods",
            "log_L_samples",
            "logL",
            "logl",
            "log_L",
        )

        if log_likelihood is None and isinstance(results, dict):
            weighted_samples = results.get("weighted_samples")
            if isinstance(weighted_samples, dict):
                log_likelihood = self._get_results_field(
                    weighted_samples,
                    "logl",
                    "logL",
                    "log_likelihood",
                    "log_likelihoods",
                    "log_L",
                )

        if log_likelihood is None:
            log_likelihood = self._evaluate_sample_log_likelihoods(matrix)
        else:
            log_likelihood = np.asarray(log_likelihood).reshape(-1)
            if log_likelihood.shape[0] != matrix.shape[0]:
                log_likelihood = self._evaluate_sample_log_likelihoods(matrix)

        return np.vstack((matrix.T, log_likelihood[np.newaxis, :])), parameter_names

    def _get_samples_archive_path(self, ret_name: str | None = None) -> str:
        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        return os.path.join(
            self.output_directory,
            "out_samples",
            f"{ret_name}_samples.npz",
        )

    def _load_samples_archive(self, ret_name: str | None = None):
        archive_path = self._get_samples_archive_path(ret_name=ret_name)

        if not os.path.exists(archive_path):
            return None, None

        with np.load(archive_path, allow_pickle=False) as archive:
            if "parameter_names" in archive.files:
                parameter_names = [
                    str(name) for name in archive["parameter_names"].tolist()
                ]
            else:
                parameter_names = [
                    file_name
                    for file_name in archive.files
                    if file_name not in {"log_likelihood", "sample_matrix"}
                ]

            if not parameter_names:
                return None, None

            if "sample_matrix" in archive.files:
                parameter_matrix = np.asarray(archive["sample_matrix"], dtype=float)
                parameter_matrix, parameter_names = (
                    self._coerce_parameter_sample_matrix(
                        parameter_matrix,
                        parameter_names,
                    )
                )
            else:
                parameter_matrix, parameter_names = (
                    self._coerce_parameter_sample_matrix(
                        {name: archive[name] for name in parameter_names},
                        parameter_names,
                    )
                )

            if parameter_matrix is None:
                return None, None

            if "log_likelihood" not in archive.files:
                return None, None

            log_likelihood = np.asarray(archive["log_likelihood"], dtype=float).reshape(
                -1
            )
            if log_likelihood.shape[0] != parameter_matrix.shape[0]:
                return None, None

        return (
            np.vstack((parameter_matrix.T, log_likelihood[np.newaxis, :])),
            parameter_names,
        )

    def _load_jaxns_raw_results_samples(
        self,
        *,
        output_directory: str,
        ret_name: str,
        parameter_names: list[str],
    ):
        raw_results_path = os.path.join(
            output_directory,
            "out_JAXNS",
            f"{ret_name}_jaxns_results.json",
        )

        if not os.path.exists(raw_results_path):
            return None, parameter_names, None

        try:
            from jaxns import load_results as jaxns_load_results
        except Exception as error:
            warnings.warn(
                f"Unable to import JAXNS while recovering samples for '{ret_name}': {error}",
                RuntimeWarning,
            )
            return None, parameter_names, None

        try:
            raw_results = jaxns_load_results(raw_results_path)
        except Exception as error:
            warnings.warn(
                f"Unable to load raw JAXNS results for '{ret_name}': {error}",
                RuntimeWarning,
            )
            return None, parameter_names, None

        parameter_matrix, parameter_names = self._coerce_parameter_sample_matrix(
            self._get_results_field(raw_results, "samples", "parametrised_samples"),
            parameter_names,
        )

        if parameter_matrix is None:
            return None, parameter_names, None

        log_likelihood = self._get_results_field(
            raw_results,
            "log_L_samples",
            "log_likelihood",
            "log_likelihoods",
            "logL",
            "logl",
            "log_L",
        )

        if log_likelihood is None:
            return None, parameter_names, None

        log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)

        if log_likelihood.shape[0] != parameter_matrix.shape[0]:
            return None, parameter_names, None

        return parameter_matrix, parameter_names, log_likelihood

    def _build_samples_for_export(self):
        samples, parameter_names = self._build_samples_from_sampler_results()

        if samples is not None and parameter_names is not None:
            return samples, parameter_names

        if self.configuration.sampler_type.lower() == "pymultinest":
            samples_dict = get_pymultinest_sample_dict(
                output_dir=self.output_directory,
                name=self.configuration.retrieval_name,
                add_log_likelihood=True,
                add_stats=False,
            )
            parameter_names = list(samples_dict.keys())[:-1]
            samples = np.array(list(samples_dict.values()), dtype=float)
            return samples, parameter_names

        return None, None

    def _save_samples_archive(self) -> str | None:
        samples, parameter_names = self._build_samples_for_export()

        if samples is None or parameter_names is None:
            return None

        os.makedirs(os.path.join(self.output_directory, "out_samples"), exist_ok=True)
        archive_path = self._get_samples_archive_path()
        archive_payload = {
            "parameter_names": np.asarray(parameter_names, dtype=str),
            "sample_matrix": np.asarray(samples[:-1, :].T, dtype=float),
            "log_likelihood": np.asarray(samples[-1, :], dtype=float),
        }

        for index, parameter_name in enumerate(parameter_names):
            archive_payload[parameter_name] = np.asarray(samples[index, :], dtype=float)

        np.savez(archive_path, **archive_payload)

        return archive_path

    def _finalize_sampler_run(
        self, sampler, results, stats=None, sampler_parameters=None
    ):
        self.sampler = sampler
        self.latest_sampler_results = results

        if self.configuration.sampler_type.lower() == "pymultinest":
            analyzer = None

            if results is not None and hasattr(results, "get_stats") and callable(
                getattr(results, "get_stats")
            ):
                analyzer = results
            elif getattr(sampler, "analyzer", None) is not None:
                analyzer = sampler.analyzer

            self.analyzer = analyzer
        else:
            self.analyzer = None

        if self.rank == 0:
            try:
                self._save_samples_archive()
            except (FileNotFoundError, OSError, ValueError) as error:
                warnings.warn(
                    f"Failed to save sample archive for '{self.configuration.retrieval_name}': {error}",
                    RuntimeWarning,
                )
            self.generate_retrieval_summary(
                stats=stats,
                sampler_parameters=sampler_parameters,
            )

        return results

    def get_pymultinest_analyzer(self, ret_name: str = ""):
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
        if "pymultinest" not in self.configuration.sampler_type.lower():
            return None

        if not ret_name and self.analyzer is not None:
            return self.analyzer

        current_ret_name = self.configuration.retrieval_name
        if not ret_name:
            ret_name = current_ret_name

        is_current_retrieval = ret_name == current_ret_name

        if (
            hasattr(self, "latest_sampler_results")
            and self.latest_sampler_results is not None
        ):
            if hasattr(self.latest_sampler_results, "get_stats") and callable(
                getattr(self.latest_sampler_results, "get_stats")
            ):
                if is_current_retrieval:
                    self.analyzer = self.latest_sampler_results
                    return self.analyzer

        try:
            import pymultinest

            n_params = getattr(self, "_n_free_params_total_retrieval", None)

            if n_params is None:
                n_params = sum(
                    parameter.is_free_parameter
                    for parameter in self.configuration.parameters.values()
                )

            prefix = os.path.join(self.output_directory, "out_PMN", ret_name + "_")

            if not os.path.exists(
                os.path.join(self.output_directory, "out_PMN", ret_name + "_params.json")
            ):
                warnings.warn(
                    f"No params file found for retrieval '{ret_name}' in out_PMN. "
                    "Cannot construct analyzer."
                )
                return None

            analyzer = pymultinest.Analyzer(
                n_params=n_params, outputfiles_basename=prefix
            )

            if is_current_retrieval:
                self.analyzer = analyzer

            return analyzer
        except ImportError:
            warnings.warn(
                "pymultinest is not installed. Cannot load PymultinestAnalyzer from disk."
            )
            return None
        except Exception as e:
            warnings.warn(f"Failed to load PyMultiNest analyzer for '{ret_name}': {e}")
            return None

        warnings.warn(
            "No sampler results found or specified PyMultiNest retrieval name not loadable."
        )
        return None

    def get_base_figure_name(self) -> str:
        return os.path.join(
            self.output_directory,
            "evaluate_" + self.configuration.retrieval_name,
            self.configuration.retrieval_name,
        )

    def get_best_fit_chi2(self, samples: npt.NDArray[np.floating]) -> float:
        """
        Get the 𝛘^2 of the best fit model - removing normalization term from log L

        Args:
            samples : numpy.ndarray
                An array of samples and likelihoods taken from a post_equal_weights file.
        """
        log_l, best_fit_index = self.get_best_fit_likelihood(samples)
        return self.get_chi2_from_sample(samples[:, best_fit_index])

    @staticmethod
    def get_best_fit_likelihood(
        samples: npt.NDArray[np.floating], print_value: bool = True
    ) -> tuple[float, int]:
        """
        Get the log likelihood of the best fit model

        Args:
            samples : numpy.ndarray
                An array of samples and likelihoods taken from a post_equal_weights file.
            print_value : bool
                If True, print the best fit likelihood value.
        """
        log_l = samples[-1, :]
        best_fit_index = int(np.argmax(log_l))

        if print_value:
            print(f"Best fit likelihood = {log_l[best_fit_index]:.2f}")

        return log_l[best_fit_index], best_fit_index

    def get_best_fit_model(
        self,
        best_fit_params: npt.NDArray[np.floating],
        parameters_read: list[str],
        ret_name: str = None,
        contribution: bool = False,
        prt_reference: str = None,
        model_generating_function: Callable = None,
        refresh: bool = True,
        mode: str = "bestfit",
        save: bool = True,
    ):
        """
        This function uses the best fit parameters to generate a pRT model that spans the entire wavelength
        range of the retrieval, to be used in plots.

        Args:
            best_fit_params : numpy.ndarray
                A numpy array containing the best fit parameters, to be passed to build_param_dict
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
                A function that returns the wavelength and spectrum, and takes a Radtrans object and the
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
            save : bool
                If True, save the best fit spectrum.
        Returns:
            best_fit_wavelengths : numpy.ndarray
                The wavelength array of the best fit model
            best_fit_spectrum : numpy.ndarray
                The emission or transmission spectrum array, with the same shape as best_fit_wavelengths
        """
        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        parameters = self.build_param_dict(best_fit_params, parameters_read)
        self.best_fit_parameters = parameters

        contribution_file = os.path.join(
            self.output_directory,
            f"evaluate_{self.configuration.retrieval_name}",
            f"{ret_name}_{mode}_model_contribution.npy",
        )
        full_file = contribution_file.replace("_contribution.npy", "_full.npy")

        if contribution:
            if not refresh and os.path.exists(contribution_file):
                print("Loading best fit spectrum and contribution from file")
                cached_auxiliary_outputs = np.load(
                    contribution_file,
                    allow_pickle=True,
                )
                best_fit_wavelengths, best_fit_spectrum = np.load(full_file).T

                return (
                    best_fit_wavelengths,
                    best_fit_spectrum,
                    self._coerce_best_fit_auxiliary_outputs(cached_auxiliary_outputs),
                )

            best_fit_wavelengths, best_fit_spectrum, best_fit_auxiliary_outputs = (
                self.get_full_range_model(
                    self.best_fit_parameters,
                    model_generating_function=model_generating_function,
                    contribution=contribution,
                    prt_reference=prt_reference,
                )
            )
            np.save(
                contribution_file,
                self._coerce_best_fit_auxiliary_outputs(best_fit_auxiliary_outputs),
                allow_pickle=True,
            )
        else:
            if not refresh and os.path.exists(full_file):
                print("Loading best fit spectrum from file")
                best_fit_wavelengths, best_fit_spectrum = np.load(full_file).T

                return best_fit_wavelengths, best_fit_spectrum

            ret_val = self.get_full_range_model(
                self.best_fit_parameters,
                model_generating_function=model_generating_function,
                contribution=contribution,
                prt_reference=prt_reference,
            )

            best_fit_auxiliary_outputs = None  # prevent eventual reference before assignment

            if len(ret_val) == 2:
                best_fit_wavelengths, best_fit_spectrum = ret_val
            else:
                best_fit_wavelengths, best_fit_spectrum, _ = ret_val

        # Add to the dictionary.
        name = f"FullRange_{mode}"

        if prt_reference is not None:
            name = prt_reference

        self.best_fit_spectra[name] = [best_fit_wavelengths, best_fit_spectrum]

        if save:
            np.save(
                full_file, np.column_stack([best_fit_wavelengths, best_fit_spectrum])
            )

        if contribution:
            return (
                best_fit_wavelengths,
                best_fit_spectrum,
                self._coerce_best_fit_auxiliary_outputs(best_fit_auxiliary_outputs),
            )

        return best_fit_wavelengths, best_fit_spectrum

    @staticmethod
    def _coerce_best_fit_auxiliary_outputs(auxiliary_outputs):
        if auxiliary_outputs is None:
            return {}

        if isinstance(auxiliary_outputs, np.ndarray):
            if auxiliary_outputs.dtype == object and auxiliary_outputs.shape == ():
                auxiliary_outputs = auxiliary_outputs.item()
            else:
                return {"contribution": np.asarray(auxiliary_outputs, dtype=float)}

        if isinstance(auxiliary_outputs, dict):
            normalized_outputs = {}

            for key, value in auxiliary_outputs.items():
                if value is None:
                    normalized_outputs[key] = None
                    continue

                normalized_outputs[key] = np.asarray(value, dtype=float)

            return normalized_outputs

        return {"contribution": np.asarray(auxiliary_outputs, dtype=float)}

    def get_best_fit_parameters(self, return_max_likelihood: bool = False):
        """Get the retrieved best fit parameters.

        Args:
            return_max_likelihood:
                if True, also return the max (log) likelihood and the chi2.

        Returns:
            A dict containing the free parameter names as keys and their retrieved best-fit values as values.
            If return_max_likelihood is True, return in addition the max (log) likelihood and the chi2.
        """
        samples = self.get_samples_dict(return_likelihood=True)

        max_likelihood, max_likelihood_index = self.get_best_fit_likelihood(
            samples=np.array(list(samples.values())), print_value=False
        )

        del samples["log_likelihood"]

        if return_max_likelihood:
            chi2 = list(self.configuration.data.values())[0].log_likelihood2chi2(
                max_likelihood
            )

            return (
                {
                    parameter_name: value[max_likelihood_index]
                    for parameter_name, value in samples.items()
                },
                max_likelihood,
                chi2,
            )

        return {
            parameter_name: value[max_likelihood_index]
            for parameter_name, value in samples.items()
        }

    def get_chi2_from_sample(self, sample: npt.NDArray[np.floating]) -> float:
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

        for name, data in self.configuration.data.items():
            # Prepare spectrum data
            data_spectrum = data.spectrum
            data_spectrum = data.apply_flux_scaling(data_spectrum, param_dict)
            data_spectrum = data.offset_flux(data_spectrum, param_dict)
            data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(
                param_dict
            )

            if data_covariance is not None:
                _, log_det = np.linalg.slogdet(2 * np.pi * data_covariance)
                add = 0.5 * log_det
            else:
                add = 0.5 * np.sum(np.log(2.0 * np.pi * data_uncertainties**2.0))

            norm = norm + add

        return 2 * (-log_l - norm)

    def get_chi2_normalisation_from_sample(
        self, sample: npt.NDArray[np.floating]
    ) -> float:
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

        for name, data in self.configuration.data.items():
            # Prepare spectrum data
            data_spectrum = data.spectrum
            data_spectrum = data.apply_flux_scaling(data_spectrum, param_dict)
            data_spectrum = data.offset_flux(data_spectrum, param_dict)
            data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(
                param_dict
            )

            if data_covariance is not None:
                _, log_det = np.linalg.slogdet(2 * np.pi * data_covariance)
                add = 0.5 * log_det
            else:
                add = 0.5 * np.sum(np.log(2.0 * np.pi * data_uncertainties**2.0))

            norm = norm + add

        return norm

    def get_elpd_per_datapoint(self, ret_name: str = None):
        from .psis import psisloo

        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        if isinstance(ret_name, str):
            ret_name = [ret_name]

        assert len(ret_name) <= 2

        elpd_tot, elpd, pareto_k = {}, {}, {}
        delta_elpd = {}

        for d_name, data in self.configuration.data.items():

            elpd_tot[d_name], elpd[d_name], pareto_k[d_name] = {}, {}, {}

            for ret_name_i in ret_name:
                log_l_per_datapoint_j = np.load(
                    f"{self.output_directory}/evaluate_{ret_name_i}/{ret_name_i}_logL_per_datapoint_{d_name}.npy",
                )

                # Compute the ELPDs with the PSIS module
                (
                    elpd_tot[d_name][ret_name_i],
                    elpd[d_name][ret_name_i],
                    pareto_k[d_name][ret_name_i],
                ) = psisloo(log_l_per_datapoint_j, Reff=1)

                if ret_name_i == self.configuration.retrieval_name:
                    data.elpd_tot = elpd_tot[d_name][ret_name_i]
                    data.elpd = elpd[d_name][ret_name_i]
                    data.pareto_k = pareto_k[d_name][ret_name_i]

            if len(ret_name) == 2:
                delta_elpd[d_name] = (
                    elpd[d_name][ret_name[0]] - elpd[d_name][ret_name[1]]
                )
                data.delta_elpd = delta_elpd[d_name]

        return elpd_tot, elpd, pareto_k, delta_elpd

    def get_evidence(self, ret_name: str = "") -> tuple[float, float]:
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
        current_ret_name = self.configuration.retrieval_name
        if not ret_name or ret_name == current_ret_name:
            sampler_summary = self._collect_sampler_summary()
            evidence = sampler_summary.get("evidence", {})
            ln_z = evidence.get("ln_z")
            if ln_z is not None:
                ln_z_err = evidence.get("ln_z_err")
                if ln_z_err is None:
                    ln_z_err = np.nan
                return ln_z / np.log(10), ln_z_err / np.log(10)

        analyzer = self.get_pymultinest_analyzer(ret_name)
        if analyzer is None:
            return np.nan, np.nan

        s = analyzer.get_stats()

        return s["global evidence"] / np.log(10), s["global evidence error"] / np.log(
            10
        )

    def get_full_range_model(
        self,
        parameters: dict[str, Parameter],
        model_generating_function: Callable = None,
        contribution: bool = False,
        prt_object: Radtrans = None,
        prt_reference: Data = None,
    ) -> tuple:
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
        parameters = copy.deepcopy(parameters)
        parameters["contribution"] = Parameter(
            "contribution", False, value=contribution
        )

        # Set up the pRT object
        p = self.configuration.pressures

        if prt_object is not None:
            atmosphere = prt_object
        elif prt_reference is not None:
            atmosphere = self.configuration.data[prt_reference].radtrans_object
        else:
            # Find the boundaries of the wavelength range to calculate
            wavelength_min = np.inf
            wavelength_max = -np.inf

            for name, data in self.configuration.data.items():
                if data.wavelength_boundaries[0] < wavelength_min:
                    wavelength_min = data.wavelength_boundaries[0]

                if data.wavelength_boundaries[1] > wavelength_max:
                    wavelength_max = data.wavelength_boundaries[1]

                species = copy.copy(
                    self.plotter.get_reference_radtrans_data().radtrans_object.line_species
                )

                line_opacity_mode = (
                    self.plotter.get_reference_radtrans_data().line_opacity_mode
                )
                resolution = self.plotter.get_reference_radtrans_data().model_resolution
            line_by_line_sampling = None

            if line_opacity_mode == "lbl" and resolution is not None:
                line_by_line_sampling = int(1e6 / resolution)

            atmosphere = Radtrans(
                pressures=p,
                line_species=tuple(species),
                rayleigh_species=copy.copy(self.configuration.rayleigh_species),
                gas_continuum_contributors=copy.copy(
                    self.configuration.continuum_opacities
                ),
                cloud_species=copy.copy(self.configuration.cloud_species),
                line_opacity_mode=line_opacity_mode,
                wavelength_boundaries=np.array(
                    [wavelength_min * 0.98, wavelength_max * 1.02]
                ),
                scattering_in_emission=self.configuration.scattering_in_emission,
                line_by_line_opacity_sampling=line_by_line_sampling,
            )

        if prt_reference is not None:
            reference_data = self.configuration.data[prt_reference]
        else:
            reference_data = self.plotter.get_reference_data()

        if model_generating_function is None:
            mg_func = reference_data.model_generating_function
        else:
            mg_func = model_generating_function

        results = self._call_model_generating_function(
            mg_func,
            atmosphere,
            parameters,
            data_name=reference_data.name,
            data_object=reference_data,
            pt_plot_mode=False,
            adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
        )
        if len(results) == 2:
            return results

        wavelength, spectrum, additional_outputs = results
        return wavelength, spectrum, self._coerce_best_fit_auxiliary_outputs(
            additional_outputs
        )

    def _get_batched_full_range_model_outputs(
        self,
        free_parameter_values: npt.ArrayLike,
        free_param_names: list[str],
        *,
        model_generating_function: Callable = None,
        prt_object: Radtrans = None,
        prt_reference: Data | str = None,
        contribution: bool = False,
        pt_plot_mode: bool = False,
        adaptive_mesh_refinement: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return batched model outputs for a batch of sampled free parameters.

        Runtime-native model functions are evaluated with a single static model
        context and ``jax.vmap`` over the leading sample dimension. Legacy model
        adapters fall back to scalar evaluation to preserve compatibility.

        When ``pt_plot_mode=False`` this returns batched wavelength and spectrum
        arrays. When ``pt_plot_mode=True`` this returns a shared pressure grid and
        a temperature matrix with the leading dimension matching the sample batch.
        """
        sample_matrix = np.asarray(free_parameter_values, dtype=float)

        if adaptive_mesh_refinement is None:
            adaptive_mesh_refinement = self.configuration.adaptive_mesh_refinement

        if sample_matrix.ndim == 1:
            sample_matrix = sample_matrix[np.newaxis, :]

        if sample_matrix.ndim != 2:
            raise ValueError(
                "expected free_parameter_values with shape (n_samples, n_free_parameters)"
            )

        if sample_matrix.shape[0] == 0:
            return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)

        if prt_object is None:
            raise ValueError(
                "prt_object must be provided for batched full-range model evaluation"
            )

        if isinstance(prt_reference, str):
            reference_data = self.configuration.data[prt_reference]
        elif prt_reference is not None:
            reference_data = prt_reference
        else:
            reference_data = self.plotter.get_reference_data()

        if model_generating_function is None:
            model_function = reference_data.model_generating_function
        else:
            model_function = model_generating_function

        resolved_contract, runtime_ready_model_function = (
            coerce_model_generating_function(model_function)
        )

        parameter_layout = self._get_parameter_layout()
        parameter_index_by_name = {
            name: index for index, name in enumerate(free_param_names)
        }

        try:
            reorder_indices = [
                parameter_index_by_name[name]
                for name in parameter_layout.free_parameter_names
            ]
        except KeyError as error:
            raise ValueError(
                f"sample matrix is missing free parameter '{error.args[0]}'"
            ) from error

        ordered_sample_matrix = np.ascontiguousarray(
            sample_matrix[:, reorder_indices],
            dtype=float,
        )

        if resolved_contract != MODEL_CONTRACT_DIFFERENTIABLE:
            batched_first_output = []
            batched_second_output = []

            for sample in sample_matrix:
                parameters = self.build_param_dict(sample, free_param_names)
                parameters["contribution"] = Parameter(
                    "contribution", False, value=contribution
                )
                ret_val = self._call_model_generating_function(
                    model_function,
                    prt_object,
                    parameters,
                    data_name=reference_data.name,
                    data_object=reference_data,
                    pt_plot_mode=pt_plot_mode,
                    adaptive_mesh_refinement=adaptive_mesh_refinement,
                )

                if len(ret_val) == 2:
                    first_output, second_output = ret_val
                elif len(ret_val) == 3 and not pt_plot_mode:
                    first_output, second_output, __ = ret_val
                else:
                    raise ValueError(
                        "expected 2 or 3 values to unpack from full range model, "
                        f"but got {len(ret_val)}"
                    )

                if first_output is None or second_output is None:
                    if pt_plot_mode:
                        continue

                    raise ValueError(
                        "batched full-range model evaluation returned an invalid sample"
                    )

                batched_first_output.append(np.asarray(first_output, dtype=float))
                batched_second_output.append(np.asarray(second_output, dtype=float))

            if not batched_first_output:
                return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)

            if pt_plot_mode:
                pressure_grid = np.asarray(batched_first_output[0], dtype=float)

                if any(
                    pressure_grid.shape != grid.shape
                    or not np.allclose(pressure_grid, grid, equal_nan=True)
                    for grid in batched_first_output[1:]
                ):
                    raise ValueError(
                        "batched PT-profile evaluation requires a shared pressure grid"
                    )

                return pressure_grid, np.asarray(batched_second_output, dtype=float)

            return (
                np.asarray(batched_first_output, dtype=float),
                np.asarray(batched_second_output, dtype=float),
            )

        context_parameters = self.build_param_dict(sample_matrix[0], free_param_names)
        context_parameters["contribution"] = Parameter(
            "contribution", False, value=contribution
        )

        observation_state = reference_data.to_observation_state()
        model_context = _build_model_context_from_parameters(
            reference_data.name,
            "unknown",
            prt_object,
            context_parameters,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            variability_atmospheric_column_model_flux_return_mode=(
                observation_state.variability_atmospheric_column_model_flux_return_mode
            ),
            model_metadata=observation_state.metadata,
        )

        batched_physical_params = vmap(
            parameter_layout.physical_params_from_free_values
        )(jnp.asarray(ordered_sample_matrix, dtype=JAX_DTYPE))

        batched_model_results = vmap(
            lambda physical_params: run_runtime_model_adapter(
                runtime_ready_model_function,
                model_context,
                physical_params,
                pt_plot_mode=pt_plot_mode,
            )
        )(batched_physical_params)

        if pt_plot_mode:
            batched_pressures = np.asarray(batched_model_results.pressures, dtype=float)
            batched_temperatures = np.asarray(
                batched_model_results.temperatures,
                dtype=float,
            )

            if batched_pressures.ndim == 1:
                return batched_pressures, batched_temperatures

            pressure_grid = batched_pressures[0, :]

            if not np.allclose(
                batched_pressures,
                pressure_grid[np.newaxis, :],
                equal_nan=True,
            ):
                raise ValueError(
                    "batched PT-profile evaluation requires a shared pressure grid"
                )

            return pressure_grid, batched_temperatures

        return (
            np.asarray(batched_model_results.wavelengths, dtype=float),
            np.asarray(batched_model_results.spectrum, dtype=float),
        )

    def get_log_likelihood_per_datapoint(
        self, samples_use: npt.NDArray[np.floating], ret_name: str = None
    ):
        if ret_name is None:
            ret_name = self.configuration.retrieval_name

        # Set up the dictionary structure
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
                log_l_per_datapoint_dict[name],
            )

    def get_mass_fractions(
        self, sample: npt.NDArray[np.floating], parameters_read: list[str] = None
    ):
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

        pressures, temps = self.calculate_temperature_profile(sample)

        name = self.plotter.get_reference_radtrans_name()

        species = [
            Opacity([species]).get_full_name().split(".R")[0]
            for species in self.configuration.data[name].radtrans_object.line_species
        ]
        abundances, mmw, _, _ = get_abundances(
            pressures,
            temps,
            copy.copy(species),
            copy.copy(self.configuration.data[name].radtrans_object.cloud_species),
            parameters,
            adaptive_mesh_refinement=False,
        )

        return abundances, mmw

    def get_median_params(
        self,
        samples: npt.NDArray[np.floating],
        parameters_read: list[str],
        return_array=False,
    ):
        """
        Build a parameter dictionary from the marginal median of each parameter.

        Args:
            samples : numpy.ndarray
                An array of samples and likelihoods taken from a post_equal_weights file.
            parameters_read : list
                A list of the free parameter names as read from the output files.
            return_array : bool
                If True, also return the median parameters as an array
        """
        samples_use = np.array(
            [np.median(samples[index, :]) for index in range(len(parameters_read))],
            dtype=float,
        )

        median_parameters = self.build_param_dict(samples_use, parameters_read)

        if return_array:
            return median_parameters, samples_use

        return median_parameters

    def get_parameters_prior_mid_range_values(self) -> npt.NDArray[np.floating]:
        """Return the prior mid-range values of the free parameters."""
        cube = []
        use_distribution = False
        sampler_type = getattr(self.configuration, "sampler_type", "")

        if (
            ("jaxns" in sampler_type.lower())
            or ("blackjax" in sampler_type.lower())
            or ("numpyro" in sampler_type.lower())
        ):
            use_distribution = True

        for parameter_name, parameter in self.configuration.parameters.items():
            if parameter.is_free_parameter:
                if use_distribution:
                    if parameter.distribution.name.lower() == "uniform":
                        cube.append(
                            (parameter.distribution.low + parameter.distribution.high)
                            / 2.0
                        )
                    else:
                        cube.append(parameter.distribution.mean())
                else:
                    cube.append(0.5)

        if use_distribution:
            return cube
        else:
            return self.prior(cube)

    def get_quantile_parameters(self, quantile: float) -> dict[str, float]:
        """Return the given quantile of the retrieved parameters' posteriors.
        For example, if quantile = 0.84, the 84th percentile (~+1 sigma) of the retrieved parameters' posterior is
        returned.

        Args:
            quantile:
                Quantile of the retrieved parameter's posterior.

        Returns:
            A dict with the free parameter names as keys and their value at the requested quantile.
        """
        samples = self.get_samples_dict(return_likelihood=False)

        return {
            parameter_name: float(np.quantile(value, quantile))
            for parameter_name, value in samples.items()
        }

    def get_reduced_chi2_from_sample(
        self, sample, subtract_n_parameters=False, verbose=False, show_chi2=False
    ):
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
        chi2 = self.get_chi2_from_sample(sample)
        d_o_f = 0

        for name, data in self.configuration.data.items():
            d_o_f += np.size(data.spectrum)

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

    def get_reduced_chi2_from_model(
        self,
        wavelengths_model,
        spectrum_model,
        parameters,
        subtract_n_parameters=False,
        verbose=False,
        show_chi2=False,
    ):
        """
        Get the 𝛘^2/DoF of the supplied spectrum - divide chi^2 by DoF

        Args:
            wavelengths_model : np.ndarray
                The wavelength grid of the model spectrum in micron.
            spectrum_model : np.ndarray
                The model flux in the same units as the data.
            parameters : dict[Parameter]
                Dictionary of Parameters passed to the forward model used to calculate the spectrum
            subtract_n_parameters : bool
                If True, divide the Chi2 by the degrees of freedom (n_data - n_parameters). If False,
                divide only by n_data
            verbose : bool
                If True, display the calculated best fit chi^2 and reduced chi^2
            show_chi2 : bool
                If True, additionally display the calculated best fit chi^2 if verbose is True
        """
        chi2 = 0
        d_o_f = 0

        for name, data in self.configuration.data.items():
            if isinstance(data.data_resolution, np.ndarray):
                data.initialise_data_resolution(wavelengths_model)

            d_o_f += np.size(data.spectrum)

            spectrum_params = extract_spectrum_parameters(parameters, name)

            data_spectrum = data.spectrum
            data_spectrum = data.apply_flux_scaling(data_spectrum, parameters)
            data_spectrum = data.offset_flux(data_spectrum, parameters)
            data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(
                parameters
            )

            model_spectrum_processed = process_spectrum_pipeline(
                wavelengths_model,
                spectrum_model,
                data,
                parameters,
                name,
                spectrum_params,
            )

            chi2 += LogLikelihood(
                model_spectrum=model_spectrum_processed,
                data_spectrum=data_spectrum,
                data_uncertainties=data_uncertainties,
                data_covariance=data_covariance,
                n_free_parameters=self._n_free_params_total_retrieval,
            ).get_chi_squared()

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

        return chi2 / d_o_f

    def get_samples(
        self,
        names: list[str] = None,
        output_directory: str = os.getcwd(),
        ret_names: list[str] = None,
    ) -> tuple[dict[str, npt.NDArray[np.floating]], dict[str, list[str]]]:
        """Get the samples of the requested retrievals.

        Args:
            names:
                Names of the retrievals from which to get the samples.
            output_directory :
                Directories in which the retrievals are stored.
            ret_names : list(str)
                Additional list of retrieval names. Used if multiple retrievals are to be included in a corner plot.

        Returns:
            A dict containing the retrieval names as keys and the samples of each requested retrievals as values, and
            a dict containing the retrieval names as keys and the free parameters names as values.
        """
        if ret_names is None:
            ret_names = []

        param_dict = {}
        samples = {}

        if names is None:
            names = [self.configuration.retrieval_name]

        current_ret_name = self.configuration.retrieval_name
        current_sampler_samples = None
        current_parameter_names = None
        if current_ret_name in names:
            current_sampler_samples, current_parameter_names = (
                self._build_samples_from_sampler_results()
            )

        def _add_archived_samples_if_available(name):
            archived_samples, archived_parameter_names = self._load_samples_archive(
                ret_name=name
            )

            if archived_samples is None or archived_parameter_names is None:
                return False

            samples[name] = archived_samples
            param_dict[name] = list(archived_parameter_names)
            return True

        sampler_type = self.configuration.sampler_type.lower()
        if sampler_type in {
            "blackjaxhmc",
            "blackjax_hmc",
            "hmc",
            "blackjaxnuts",
            "blackjax_nuts",
            "nuts",
        }:
            for name in names:
                if name == current_ret_name and current_sampler_samples is not None:
                    samples[name] = current_sampler_samples
                    param_dict[name] = list(current_parameter_names)
                    continue

                if _add_archived_samples_if_available(name):
                    continue

                params_file = os.path.join(
                    output_directory, "out_BlackJAX", name + "_params.json"
                )
                samples_file = os.path.join(
                    output_directory, "out_BlackJAX", name + "_samples.npz"
                )

                with open(params_file, "r") as handle:
                    parameters_read = json.load(handle)

                chain = np.load(samples_file)
                log_values = (
                    chain["model_log_likelihood"]
                    if "model_log_likelihood" in chain.files
                    else chain["logdensity"]
                )
                samples[name] = np.vstack(
                    (chain["samples"].T, log_values[np.newaxis, :])
                )
                param_dict[name] = parameters_read

            return samples, param_dict

        if sampler_type in {"numpyrohmc", "numpyro_hmc", "numpyronuts", "numpyro_nuts"}:
            for name in names:
                if name == current_ret_name and current_sampler_samples is not None:
                    samples[name] = current_sampler_samples
                    param_dict[name] = list(current_parameter_names)
                    continue

                if _add_archived_samples_if_available(name):
                    continue

                params_file = os.path.join(
                    output_directory, "out_NumPyro", name + "_params.json"
                )
                samples_file = os.path.join(
                    output_directory, "out_NumPyro", name + "_samples.npz"
                )

                with open(params_file, "r") as handle:
                    parameters_read = json.load(handle)

                chain = np.load(samples_file)
                log_values = (
                    chain["model_log_likelihood"]
                    if "model_log_likelihood" in chain.files
                    else chain["logdensity"]
                )
                samples[name] = np.vstack(
                    (chain["samples"].T, log_values[np.newaxis, :])
                )
                param_dict[name] = parameters_read

            return samples, param_dict

        if sampler_type in {"dynesty"}:
            for name in names:
                if name == current_ret_name and current_sampler_samples is not None:
                    samples[name] = current_sampler_samples
                    param_dict[name] = list(current_parameter_names)
                    continue

                if _add_archived_samples_if_available(name):
                    continue

                params_file = os.path.join(
                    output_directory, "out_Dynesty", name + "_params.json"
                )
                samples_file = os.path.join(
                    output_directory, "out_Dynesty", name + "_samples.npz"
                )

                with open(params_file, "r") as handle:
                    parameters_read = json.load(handle)

                chain = np.load(samples_file)
                log_values = (
                    chain["log_likelihood"]
                    if "log_likelihood" in chain.files
                    else chain["weighted_log_likelihood"]
                )
                samples[name] = np.vstack(
                    (chain["samples"].T, np.asarray(log_values)[np.newaxis, :])
                )
                param_dict[name] = parameters_read

            return samples, param_dict

        if sampler_type in {
            "ultranest",
            "jaxnestedsampler",
            "jaxns",
            "jaxnsshardedstaticnestedsampler",
            "jaxnsshardedstaticsestedsampler",
        }:
            for name in names:
                if name == current_ret_name and current_sampler_samples is not None:
                    samples[name] = current_sampler_samples
                    param_dict[name] = list(current_parameter_names)
                    continue

                if _add_archived_samples_if_available(name):
                    continue

                params_file = os.path.join(
                    output_directory, "out_JAXNS", name + "_params.json"
                )
                samples_file = os.path.join(
                    output_directory, "out_JAXNS", name + "_samples.npz"
                )

                if not os.path.exists(params_file) or not os.path.exists(samples_file):
                    raise FileNotFoundError(
                        f"Sampler outputs for '{name}' are not available in memory for sampler '{sampler_type}'."
                    )

                with open(params_file, "r") as handle:
                    parameters_read = json.load(handle)

                with np.load(samples_file, allow_pickle=False) as chain:
                    if "sample_matrix" in chain.files:
                        sample_matrix = np.asarray(chain["sample_matrix"], dtype=float)
                    elif "samples" in chain.files:
                        sample_matrix = np.asarray(chain["samples"], dtype=float)
                    else:
                        raise FileNotFoundError(
                            f"JAXNS sample file '{samples_file}' does not contain posterior samples."
                        )

                    if "log_likelihood" in chain.files:
                        log_values = np.asarray(chain["log_likelihood"], dtype=float)
                    elif "weighted_log_likelihood" in chain.files:
                        log_values = np.asarray(
                            chain["weighted_log_likelihood"], dtype=float
                        )
                    else:
                        raise FileNotFoundError(
                            f"JAXNS sample file '{samples_file}' does not contain log likelihood values."
                        )

                sample_matrix, parameters_read = self._coerce_parameter_sample_matrix(
                    sample_matrix,
                    parameters_read,
                )
                log_values = np.asarray(log_values, dtype=float).reshape(-1)

                if sample_matrix is None or log_values.shape[0] != sample_matrix.shape[0]:
                    recovered_matrix, recovered_parameters, recovered_log_values = (
                        self._load_jaxns_raw_results_samples(
                            output_directory=output_directory,
                            ret_name=name,
                            parameter_names=parameters_read,
                        )
                    )

                    if recovered_matrix is None or recovered_log_values is None:
                        raise ValueError(
                            f"JAXNS samples for '{name}' are malformed: posterior sample matrix shape "
                            f"{None if sample_matrix is None else sample_matrix.shape} does not match "
                            f"log-likelihood shape {log_values.shape}."
                        )

                    sample_matrix = recovered_matrix
                    parameters_read = recovered_parameters
                    log_values = recovered_log_values

                    try:
                        np.savez(
                            samples_file,
                            parameter_names=np.asarray(parameters_read, dtype=str),
                            sample_matrix=np.asarray(sample_matrix, dtype=float),
                            log_likelihood=np.asarray(log_values, dtype=float),
                        )
                        with open(params_file, "w") as handle:
                            json.dump(list(parameters_read), handle, indent=4)
                    except OSError as error:
                        warnings.warn(
                            f"Recovered JAXNS samples for '{name}' but failed to rewrite the repaired archive: {error}",
                            RuntimeWarning,
                        )

                samples[name] = np.vstack((sample_matrix.T, log_values[np.newaxis, :]))
                param_dict[name] = parameters_read
                continue

            return samples, param_dict

        # pymultinest
        for name in names:
            if name == current_ret_name and current_sampler_samples is not None:
                samples[name] = current_sampler_samples
                param_dict[name] = list(current_parameter_names)
                continue

            if _add_archived_samples_if_available(name):
                continue

            samples_ = get_pymultinest_sample_dict(
                output_dir=output_directory,
                name=name,
                add_log_likelihood=True,
                add_stats=False,
            )

            samples[name] = np.array(list(samples_.values()))
            param_dict[name] = list(samples_.keys())[:-1]  # do not add the likelihood

        # TODO: JAXNS
        return samples, param_dict

    def get_samples_dict(
        self, return_likelihood: bool = False
    ) -> dict[str, npt.NDArray[np.floating]]:
        """Return the samples of this retrieval as a dict.

        Args:
            return_likelihood:
                If True, the samples dictionary also contains the log likelihood of each sample.
        Returns:
            A dict containing the free parameters names as keys, and their sampled values as values.
        """
        samples, parameter_names = self.get_samples(
            names=None, output_directory=self.output_directory, ret_names=None
        )

        samples = samples[self.configuration.retrieval_name]
        param_dict = list(parameter_names[self.configuration.retrieval_name])

        # Add likelihood
        if return_likelihood:
            param_dict.append("log_likelihood")

        return {
            parameter_name: samples[i, :] for i, parameter_name in enumerate(param_dict)
        }

    @staticmethod
    def get_special_parameters() -> set[str]:
        # TODO: naming convention
        return {"_spectrum_scaling", "_spectrum_offset", "_uncertainty_scaling"}

    def get_volume_mixing_ratios(
        self, sample: npt.NDArray[np.floating], parameters_read: list[str] = None
    ):
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
        mass_fractions, mean_molar_masses = self.get_mass_fractions(
            sample, parameters_read
        )
        mass_fractions_no_clouds = copy.deepcopy(mass_fractions)
        for cloud in self.configuration.cloud_species:
            cloud_opacity = CloudOpacity([cloud])
            cloud_name = cloud_opacity.species_isotopologue_name.split("_")[0]
            if cloud_name in mass_fractions_no_clouds.keys():
                del mass_fractions_no_clouds[cloud_name]
        volume_mixing_ratios = mass_fractions2volume_mixing_ratios(
            mass_fractions_no_clouds
        )

        return volume_mixing_ratios, mean_molar_masses

    def log_likelihood(
        self,
        cube: npt.NDArray[np.floating],
        ndim: int = 0,
        nparam: int = 0,
        log_l_per_datapoint_dict: dict = None,
        return_model: bool = False,
    ):
        """Sampler-facing log-likelihood entry point.

        The normal scalar retrieval path now executes through ``RetrievalRuntime``.
        Less common debug and per-datapoint modes still fall back to the legacy
        implementation until the entire retrieval stack has been migrated.
        """
        if isinstance(log_l_per_datapoint_dict, dict) or return_model:
            return self._log_likelihood_legacy(
                cube,
                ndim=ndim,
                nparam=nparam,
                log_l_per_datapoint_dict=log_l_per_datapoint_dict,
                return_model=return_model,
            )

        physical_params = self.parameter_layout.physical_params_from_vector(cube)
        return self.runtime.evaluate_scalar(
            physical_params=physical_params,
            uncertainties_mode=self.uncertainties_mode,
            print_log_likelihood_for_debugging=self.print_log_likelihood_for_debugging,
        )

    def _log_likelihood_legacy(
        self,
        cube: npt.NDArray[np.floating],
        ndim: int = 0,  # noqa: F811 keeping unused parameters for Multinest compatibility.
        nparam: int = 0,  # noqa: F811 keeping unused parameters for Multinest compatibility.
        log_l_per_datapoint_dict: dict = None,
        return_model: bool = False,
    ):
        """Legacy scalar likelihood implementation retained for fallback modes."""
        invalid_value = -1e99
        tiny = 1e-99
        log_likelihood = 0.0
        log_prior = 0.0

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
            warnings.warn(
                "automatically optimizing for uncertainties, be sure of what you are doing..."
            )
            beta = None
        elif self.uncertainties_mode == "retrieve":
            warnings.warn(
                "retrieving uncertainties (multiply mode), be sure of what you are doing..."
            )
            retrieve_uncertainties = True
            beta_mode = "multiply"
        elif self.uncertainties_mode == "retrieve_add":
            warnings.warn(
                "retrieving uncertainties (add mode), be sure of what you are doing..."
            )
            retrieve_uncertainties = True
            beta_mode = "add"
        else:
            raise ValueError(
                f"uncertainties mode must be 'default'|'optimize'|'retrieve'|'retrieve_add', "
                f"but was '{self.uncertainties_mode}'"
            )

        # Store per data-object
        per_datapoint = False

        if isinstance(log_l_per_datapoint_dict, dict):
            per_datapoint = True

        # Update model parameters with the sampled free-parameter values.
        self._set_free_parameter_values(cube)

        for data_name, data in self.configuration.data.items():
            data_spectrum = data.spectrum

            if per_datapoint:
                # Keep logL's separate per data-object
                log_likelihood = 0.0

            # Only calculate spectra within a given wavelengths range once if data.scale_flux or data.scale_uncertainties:
            data_spectrum = data.apply_flux_scaling(
                data_spectrum, self.configuration.parameters
            )
            data_spectrum = data.offset_flux(
                data_spectrum, self.configuration.parameters
            )
            data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(
                self.configuration.parameters
            )

            if data.external_radtrans_reference is None:
                # Compute the model
                model_returned_values = self._call_model_generating_function(
                    data.model_generating_function,
                    data.radtrans_object,
                    self.configuration.parameters,
                    data_name=data_name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                )  # TODO the generating function should always return the same number of values

                if retrieve_uncertainties:
                    if len(model_returned_values) == 4:
                        wavelengths_model, spectrum_model, beta, additional_log_l = (
                            model_returned_values
                        )
                    else:
                        wavelengths_model, spectrum_model, beta = model_returned_values
                        additional_log_l = 0.0
                else:
                    if data.variability_atmospheric_column_model_flux_return_mode:
                        (
                            wavelengths_model,
                            spectrum_model,
                            additional_log_l,
                            atmospheric_model_column_fluxes,
                        ) = model_returned_values
                    elif len(model_returned_values) == 3:
                        wavelengths_model, spectrum_model, additional_log_l = (
                            model_returned_values
                        )
                    else:
                        wavelengths_model, spectrum_model = model_returned_values
                        additional_log_l = 0.0

                if additional_log_l is None:
                    additional_log_l = 0

                # Ensure that the spectrum model has no masked values, so that no conversion to NaN is required
                if isinstance(spectrum_model, np.ma.MaskedArray):
                    spectrum_model = spectrum_model.filled(0)

                # Sanity checks on outputs
                if spectrum_model is None:
                    # -np.inf (and anything <1e-100) seemed to cause issues with Multinest.
                    # This particular check probably wasn't the underlying cause, and could
                    # use a better default value.
                    return invalid_value

                # Calculate log likelihood
                if not isinstance(data.spectrum, float) and data.spectrum.dtype == "O":
                    if jnp.ndim(data.spectrum) == 1:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Second dimension of data must be a function of wavelength
                        for i, spectrum_data in enumerate(data.spectrum):
                            if jnp.isnan(spectrum_model[i][~data.mask[i]]).any():
                                return invalid_value

                            log_likelihood += data.log_likelihood(
                                spectrum_model[i][~data.mask[i]],
                                spectrum_data,
                                data.uncertainties[i],
                                beta=beta,
                                beta_mode=beta_mode,
                            )
                    elif jnp.ndim(data.spectrum) == 2:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Third dimension of data must be a function of wavelength
                        for i, detector in enumerate(data.spectrum):
                            for j, spectrum_data in enumerate(detector):
                                if jnp.isnan(
                                    spectrum_model[i, j][~data.mask[i, j]]
                                ).any():
                                    return invalid_value

                                log_likelihood += data.log_likelihood(
                                    spectrum_model[i, j][~data.mask[i, j]],
                                    spectrum_data,
                                    data.uncertainties[i, j],
                                    beta=beta,
                                    beta_mode=beta_mode,
                                )
                    else:
                        raise ValueError(
                            f"observation is an array containing object, "
                            f"and have {jnp.ndim(data.spectrum)} dimensions, "
                            f"but must have 1 to 2"
                        )
                else:
                    if jnp.isnan(spectrum_model).any():
                        return invalid_value

                    if (
                        isinstance(data.spectrum, float) or jnp.ndim(data.spectrum) == 1
                    ) and jnp.ndim(spectrum_model) == 1:

                        if data.fit_instrumental_resolution:
                            # Convolve and rebin if fitting for resolution
                            resolution_slope = self.configuration.parameters[
                                data.name + "_R_slope"
                            ].value
                            resolution_intersect = self.configuration.parameters[
                                data.name + "_R_int"
                            ].value
                            resolution_array = (
                                data.wavelengths * resolution_slope
                            ) + resolution_intersect
                            # TODO: interpolate resolution_array and use standard convolve function.
                            # TODO: compare different convolution and binning methods
                            model_spectrum_rebinned = (
                                convolve_and_sample_variable_resolution_breads(
                                    data.wavelengths,
                                    resolution_array,
                                    data.wavelengths,
                                    spectrum_model,
                                )
                            )
                        model_spectrum_convolved = spectrum_model
                        if data.data_resolution is not None:
                            # Convolve and rebin the model spectrum to the data resolution
                            model_spectrum_convolved = convolve(
                                wavelengths_model, spectrum_model, data.data_resolution
                            )
                        if (
                            data_name + "_radial_velocity"
                            in self.configuration.parameters.keys()
                        ):
                            # RV in km/s -> multiply by 1e5 to cm/s
                            # wavelengths_model in micron
                            # cst.c in cm/s
                            radial_velocity = (
                                self.configuration.parameters[
                                    data.name + "_radial_velocity"
                                ].value
                                * 1e5
                            )
                            wavelengths_model *= shift_wavelengths_by_radial_velocity(
                                radial_velocity
                            )

                        elif (
                            "system_radial_velocity"
                            in self.configuration.parameters.keys()
                        ):
                            radial_velocity = (
                                self.configuration.parameters[
                                    "system_radial_velocity"
                                ].value
                                * 1e5
                            )
                            wavelengths_model *= shift_wavelengths_by_radial_velocity(
                                radial_velocity
                            )

                        if np.size(wavelengths_model) == np.size(
                            data.wavelengths
                        ) and np.all(wavelengths_model == data.wavelengths):
                            model_spectrum_rebinned = copy.deepcopy(
                                model_spectrum_convolved
                            )
                        else:
                            model_spectrum_rebinned = rebin_spectrum_bin(
                                wavelengths_model,
                                model_spectrum_convolved,
                                data.wavelengths,
                                data.wavelength_bin_widths,
                            )

                        # Convolution and rebin are cared of in get_chisq
                        log_likelihood += (
                            LogLikelihood(
                                model_spectrum=model_spectrum_rebinned,
                                data_spectrum=data_spectrum,
                                data_uncertainties=data_uncertainties,
                                data_covariance=data_covariance,
                                n_free_parameters=self._n_free_params_total_retrieval,
                            ).calculate_log_likelihood()
                            + additional_log_l
                        )

                    elif jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 2:
                        log_likelihood += data.log_likelihood(
                            spectrum_model[0, :],
                            data.spectrum,
                            data.uncertainties,
                            beta=beta,
                            beta_mode=beta_mode,
                        )
                    elif jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 3:
                        log_likelihood += data.log_likelihood(
                            spectrum_model[0, 0, :],
                            data.spectrum,
                            data.uncertainties,
                            beta=beta,
                            beta_mode=beta_mode,
                        )
                    elif jnp.ndim(data.spectrum) == 2:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Second dimension of data must be a function of wavelength
                        for i, spectrum_data in enumerate(data.spectrum):
                            log_likelihood += data.log_likelihood(
                                spectrum_model[i, ~data.mask[i, :]],
                                spectrum_data,
                                data.uncertainties[i],
                                beta=beta,
                                beta_mode=beta_mode,
                            )
                    elif jnp.ndim(data.spectrum) == 3:
                        # Convolution and rebin are *not* cared of in get_log_likelihood
                        # Third dimension of data must be a function of wavelength
                        for i, detector in enumerate(data.spectrum):
                            for j, spectrum_data in enumerate(detector):
                                log_likelihood += data.log_likelihood(
                                    spectrum_model[i, j, ~data.mask[i, j, :]],
                                    spectrum_data,
                                    data.uncertainties[i, j],
                                    beta=beta,
                                    beta_mode=beta_mode,
                                )
            else:
                wavelengths_model = None
                spectrum_model = None
                additional_log_l = None

            if per_datapoint:
                log_l_per_datapoint_dict[data_name].append(log_likelihood)

            # Check for data using the same pRT object
            # Calculate log likelihood
            for data_name_2, data_2 in self.configuration.data.items():
                if data_2.external_radtrans_reference is not None:
                    data_2_spectrum = data_2.spectrum

                    data_2_spectrum = data_2.apply_flux_scaling(
                        data_2_spectrum, self.configuration.parameters
                    )
                    data_2_spectrum = data_2.offset_flux(
                        data_2_spectrum, self.configuration.parameters
                    )
                    data_2_uncertainties, data_2_covariance = (
                        data.scale_and_inflate_uncertainties(
                            self.configuration.parameters
                        )
                    )

                    if data_2.external_radtrans_reference == data_name:
                        if spectrum_model is None:
                            return invalid_value

                        if np.isnan(spectrum_model).any():
                            return invalid_value

                        if data.fit_instrumental_resolution:
                            # Convolve and rebin if fitting for resolution
                            resolution_slope = self.configuration.parameters[
                                data.name + "_R_slope"
                            ].value
                            resolution_intersect = self.configuration.parameters[
                                data.name + "_R_int"
                            ].value
                            resolution_array = (
                                data.wavelengths * resolution_slope
                            ) + resolution_intersect
                            model_spectrum_rebinned = (
                                convolve_and_sample_variable_resolution_breads(
                                    data.wavelengths,
                                    resolution_array,
                                    data.wavelengths,
                                    spectrum_model,
                                )
                            )

                        model_spectrum_convolved = spectrum_model
                        if data.data_resolution is not None:
                            # Convolve and rebin the model spectrum to the data resolution
                            model_spectrum_convolved = convolve(
                                wavelengths_model, spectrum_model, data.data_resolution
                            )
                        if (
                            data_name + "_radial_velocity"
                            in self.configuration.parameters.keys()
                        ):
                            # RV in km/s -> multiply by 1e5 to cm/s
                            # wavelengths_model in micron
                            # cst.c in cm/s
                            radial_velocity = (
                                self.configuration.parameters[
                                    data.name + "_radial_velocity"
                                ].value
                                * 1e5
                            )
                            wavelengths_model *= shift_wavelengths_by_radial_velocity(
                                radial_velocity
                            )

                        elif (
                            "system_radial_velocity"
                            in self.configuration.parameters.keys()
                        ):
                            radial_velocity = (
                                self.configuration.parameters[
                                    "system_radial_velocity"
                                ].value
                                * 1e5
                            )
                            wavelengths_model *= shift_wavelengths_by_radial_velocity(
                                radial_velocity
                            )

                        if np.size(wavelengths_model) == np.size(
                            data.wavelengths
                        ) and np.all(wavelengths_model == data.wavelengths):
                            model_spectrum_rebinned = copy.deepcopy(spectrum_model)
                        else:
                            model_spectrum_rebinned = rebin_spectrum_bin(
                                wavelengths_model,
                                model_spectrum_convolved,
                                data.wavelengths,
                                data.wavelength_bin_widths,
                            )

                        # Convolution and rebin are cared of in get_chisq
                        log_likelihood += (
                            LogLikelihood(
                                model_spectrum=model_spectrum_rebinned,
                                data_spectrum=data_2_spectrum,
                                data_uncertainties=data_2_uncertainties,
                                data_covariance=data_2_covariance,
                                n_free_parameters=self._n_free_params_total_retrieval,
                            ).calculate_log_likelihood()
                            + additional_log_l
                        )

        if per_datapoint:
            return log_l_per_datapoint_dict

        if "log_prior_weight" in self.configuration.parameters.keys():
            log_prior += self.configuration.parameters[
                "log_prior_weight"
            ].value  # TODO why not using log_likelihood?

        if log_likelihood + log_prior < invalid_value:
            return invalid_value

        if jnp.abs(log_likelihood + log_prior) < tiny:
            log_likelihood = tiny
            log_prior = 0

        if not return_model:
            if self.print_log_likelihood_for_debugging:
                print("log_likelihood + log_prior", log_likelihood + log_prior)
            return log_likelihood + log_prior
        else:
            return (
                log_likelihood + log_prior,
                wavelengths_models,
                spectrum_models,
                beta,
                additional_log_ls,
            )

    def _set_free_parameter_values(self, values: npt.ArrayLike) -> None:
        """Copy sampled free-parameter values into the active retrieval configuration."""
        flat_values = self._flatten_free_parameter_values(values)
        index = 0

        for parameter in self.configuration.parameters.values():
            if not parameter.is_free_parameter:
                continue

            if index >= flat_values.size:
                raise ValueError(
                    f"Expected at least {index + 1} free-parameter values, got {flat_values.size}"
                )

            parameter.set_param(flat_values[index])
            index += 1

        if index != flat_values.size:
            raise ValueError(
                f"Expected {index} free-parameter values, got {flat_values.size}"
            )

    @staticmethod
    def _flatten_free_parameter_values(values: npt.ArrayLike) -> np.ndarray:
        """Return a 1D numpy array from sampler-provided parameter values.

        PyMultiNest can pass a ctypes-backed buffer that NumPy 2 refuses to wrap
        directly via np.asarray because of an unsupported PEP 3118 format string.
        Falling back to plain indexing avoids that incompatibility while keeping
        support for numpy, JAX, and Python sequence inputs.
        """
        if isinstance(values, np.ndarray):
            return values.reshape(-1)

        if hasattr(values, "tolist"):
            converted_values = values.tolist()
            if isinstance(converted_values, list):
                return np.array(converted_values, dtype=float).reshape(-1)

        try:
            return np.array(values, dtype=float)
        except ValueError:
            return values

    def _get_parameter_layout(self) -> ParameterLayout:
        parameter_layout = getattr(self, "parameter_layout", None)

        if parameter_layout is None:
            parameter_layout = ParameterLayout.from_configuration(self.configuration)
            self.parameter_layout = parameter_layout

        return parameter_layout

    def _get_runtime(self) -> RetrievalRuntime:
        runtime = getattr(self, "runtime", None)

        if runtime is None:
            parameter_layout = self._get_parameter_layout()
            runtime = RetrievalRuntime.from_configuration(
                self.configuration,
                parameter_layout=parameter_layout,
                n_free_parameters=parameter_layout.n_free_parameters,
            )
            self.runtime = runtime

        return runtime

    def _has_only_legacy_model_groups(self) -> bool:
        runtime = self._get_runtime()
        return bool(runtime.model_groups) and all(
            group.model_contract != "differentiable" for group in runtime.model_groups
        )

    def _ensure_jax_sampler_model_support(self, sampler_name: str) -> None:
        if self._has_only_legacy_model_groups():
            raise ValueError(
                f"{sampler_name} requires at least one runtime-native model group; legacy-only retrievals are not supported."
            )

    def _get_ultranest_vectorization_disable_reason(self) -> str | None:
        runtime = self._get_runtime()

        if any(
            group.model_contract != "differentiable" for group in runtime.model_groups
        ):
            return "the retrieval contains legacy model groups"

        if any(
            getattr(observation.spectrum, "dtype", None) == "O"
            for observation in runtime.observations.values()
        ):
            return "the retrieval contains ragged observation groups"

        return None

    def _resolve_ultranest_vectorized_mode(self, requested: bool) -> bool:
        if not requested:
            return False

        disable_reason = self._get_ultranest_vectorization_disable_reason()

        if disable_reason is not None:
            warnings.warn(
                f"UltraNest vectorized mode disabled because {disable_reason}.",
                RuntimeWarning,
            )
            return False

        return True

    def log_likelihood_jax(
        self,
        parameters=None,
        configuration=None,
        log_l_per_datapoint_dict: dict = None,
        return_model: bool = False,
        uncertainties_mode: str = "default",
        generate_mock_data: bool = False,
        print_log_likelihood_for_debugging: bool = False,
        *,
        physical_params: "PhysicalParams | None" = None,
    ):
        """JAX-compatible log likelihood computation.

        Uses the pre-built ``RetrievalRuntime`` stored on ``self`` instead of
        reconstructing it on every call.  Accepts either a legacy *parameters*
        dict (for backward compatibility) or a *physical_params* instance.

        Falls back to the legacy implementation when *log_l_per_datapoint_dict*,
        *return_model*, or *generate_mock_data* are requested.
        """
        if (
            isinstance(log_l_per_datapoint_dict, dict)
            or return_model
            or generate_mock_data
        ):
            if configuration is None:
                configuration = self.configuration
            if parameters is None and physical_params is not None:
                parameters = self._get_parameter_layout().build_legacy_parameter_dict(
                    physical_params
                )
            return Retrieval._log_likelihood_jax_legacy(
                parameters=parameters,
                configuration=configuration,
                log_l_per_datapoint_dict=log_l_per_datapoint_dict,
                return_model=return_model,
                uncertainties_mode=uncertainties_mode,
                generate_mock_data=generate_mock_data,
                print_log_likelihood_for_debugging=print_log_likelihood_for_debugging,
            )

        runtime = self._get_runtime()

        if physical_params is None:
            if parameters is None:
                raise ValueError(
                    "Either 'parameters' or 'physical_params' must be provided."
                )
            physical_params = PhysicalParams.from_mapping(
                {
                    parameter_name: (
                        parameter.value if hasattr(parameter, "value") else parameter
                    )
                    for parameter_name, parameter in parameters.items()
                }
            )

        return runtime.evaluate_jax(
            physical_params=physical_params,
            uncertainties_mode=uncertainties_mode,
            print_log_likelihood_for_debugging=print_log_likelihood_for_debugging,
        )

    @staticmethod
    def _log_likelihood_jax_legacy(
        parameters,
        configuration,
        log_l_per_datapoint_dict: dict = None,
        return_model: bool = False,
        uncertainties_mode: str = "default",
        generate_mock_data: bool = False,
        print_log_likelihood_for_debugging: bool = False,
    ):
        # === Constants and Configuration ===
        invalid_value = -1e99
        tiny = 1e-99

        # === Uncertainties Mode Parse ===
        retrieve_uncertainties = False
        beta_mode = "multiply"
        beta = 1.0

        if uncertainties_mode == "default":
            beta = 1.0
        elif uncertainties_mode == "optimize":
            warnings.warn(
                "automatically optimizing for uncertainties, be sure of what you are doing..."
            )
            beta = None
        elif uncertainties_mode == "retrieve":
            warnings.warn(
                "retrieving uncertainties (multiply mode), be sure of what you are doing..."
            )
            retrieve_uncertainties = True
            beta_mode = "multiply"
        elif uncertainties_mode == "retrieve_add":
            warnings.warn(
                "retrieving uncertainties (add mode), be sure of what you are doing..."
            )
            retrieve_uncertainties = True
            beta_mode = "add"
        else:
            raise ValueError(
                f"uncertainties mode must be 'default'|'optimize'|'retrieve'|'retrieve_add', "
                f"but was '{uncertainties_mode}'"
            )

        # === Setup ===
        per_datapoint = isinstance(log_l_per_datapoint_dict, dict)

        # Count free parameters (outside loops)
        n_free_parameters = sum(
            1 for parameter in parameters if parameters[parameter].is_free_parameter
        )

        # Collect log likelihood contributions (functional approach)
        log_likelihood_contributions = []
        log_prior = 0.0

        # === Main Data Iteration ===
        for data_name, data in configuration.data.items():
            # Pre-extract spectrum parameters to avoid repeated lookups
            spectrum_params = extract_spectrum_parameters(parameters, data_name)

            # Prepare spectrum data
            data_spectrum = data.spectrum
            data_spectrum = data.apply_flux_scaling(data_spectrum, parameters)
            data_spectrum = data.offset_flux(data_spectrum, parameters)
            data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(
                parameters
            )

            per_datapoint_contribution = 0.0

            # === Process Primary Data Source ===
            if data.external_radtrans_reference is None:
                # Compute model spectrum
                model_returned_values = Retrieval._call_model_generating_function(
                    data.model_generating_function,
                    data.radtrans_object,
                    parameters,
                    data_name=data_name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=configuration.adaptive_mesh_refinement,
                )

                # Unpack model output using helper
                wavelengths_model, spectrum_model, beta_model, additional_log_l, _ = (
                    unpack_model_output(
                        model_returned_values,
                        retrieve_uncertainties,
                        data.variability_atmospheric_column_model_flux_return_mode,
                    )
                )

                # Update beta if it was retrieved
                if retrieve_uncertainties and beta_model is not None:
                    beta = beta_model

                # Validate model output
                is_valid = validate_model_output(spectrum_model, wavelengths_model)
                # if not is_valid:
                #    warnings.warn("spectrum_model contains NaN values")
                #    return invalid_value
                invalid_log_like = lax.cond(
                    is_valid,
                    lambda: 0.0,  # valid output contributes 0 to logL (neutral element)
                    lambda: invalid_value,  # invalid output contributes -1e99 to logL
                )
                log_likelihood_contributions.append(invalid_log_like)

                # Handle masked arrays (JAX compatibility)
                if isinstance(spectrum_model, np.ma.MaskedArray):
                    spectrum_model = spectrum_model.filled(0)

                spectrum_model = ensure_jax_array(spectrum_model)

                # === Compute Log Likelihood Based on Data Structure ===
                if (
                    not isinstance(data.spectrum, float)
                    and hasattr(data.spectrum, "dtype")
                    and data.spectrum.dtype == "O"
                ):
                    # Object array: loop over object entries
                    per_datapoint_contribution += (
                        Retrieval._compute_object_array_log_likelihood(
                            data, spectrum_model, beta, beta_mode
                        )
                    )
                else:
                    # Standard array: use spectrum processing pipeline
                    if (
                        isinstance(data.spectrum, float) or jnp.ndim(data.spectrum) == 1
                    ) and jnp.ndim(spectrum_model) == 1:
                        # Process spectrum through full pipeline
                        model_spectrum_processed = process_spectrum_pipeline(
                            wavelengths_model,
                            spectrum_model,
                            data,
                            parameters,
                            data_name,
                            spectrum_params,
                        )

                        # Compute log likelihood
                        log_l = LogLikelihood(
                            model_spectrum=model_spectrum_processed,
                            data_spectrum=data_spectrum,
                            data_uncertainties=data_uncertainties,
                            data_covariance=data_covariance,
                            n_free_parameters=n_free_parameters,
                        ).calculate_log_likelihood()

                        per_datapoint_contribution += log_l + additional_log_l

                    elif jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 2:
                        # 1D data, 2D model: take first row
                        per_datapoint_contribution += data.log_likelihood(
                            spectrum_model[0, :],
                            data.spectrum,
                            data.uncertainties,
                            beta=beta,
                            beta_mode=beta_mode,
                        )
                    elif jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 3:
                        # 1D data, 3D model: take first [0, 0, :]
                        per_datapoint_contribution += data.log_likelihood(
                            spectrum_model[0, 0, :],
                            data.spectrum,
                            data.uncertainties,
                            beta=beta,
                            beta_mode=beta_mode,
                        )
                    elif jnp.ndim(data.spectrum) == 2:
                        # 2D data: loop over first dimension
                        for i, spectrum_data in enumerate(data.spectrum):
                            if ~data.mask[i, :] is not None:
                                per_datapoint_contribution += data.log_likelihood(
                                    spectrum_model[i, ~data.mask[i, :]],
                                    spectrum_data,
                                    data.uncertainties[i],
                                    beta=beta,
                                    beta_mode=beta_mode,
                                )
                    elif jnp.ndim(data.spectrum) == 3:
                        # 3D data: loop over first two dimensions
                        for i, detector in enumerate(data.spectrum):
                            for j, spectrum_data in enumerate(detector):
                                if ~data.mask[i, j, :] is not None:
                                    per_datapoint_contribution += data.log_likelihood(
                                        spectrum_model[i, j, ~data.mask[i, j, :]],
                                        spectrum_data,
                                        data.uncertainties[i, j],
                                        beta=beta,
                                        beta_mode=beta_mode,
                                    )

            # === Process External Data Using Same pRT Object ===
            for data_name_2, data_2 in configuration.data.items():
                if data_2.external_radtrans_reference == data_name and data_name in [
                    dn for dn, _ in configuration.data.items()
                ]:
                    # This data references the spectrum computed above
                    if wavelengths_model is None or spectrum_model is None:
                        return invalid_value

                    # Prepare secondary data
                    data_2_spectrum = data_2.spectrum
                    data_2_spectrum = data_2.apply_flux_scaling(
                        data_2_spectrum, parameters
                    )
                    data_2_spectrum = data_2.offset_flux(data_2_spectrum, parameters)
                    data_2_uncertainties, data_2_covariance = (
                        data_2.scale_and_inflate_uncertainties(parameters)
                    )

                    # Process spectrum for secondary data
                    model_spectrum_processed_2 = process_spectrum_pipeline(
                        wavelengths_model,
                        spectrum_model,
                        data_2,
                        parameters,
                        data_name_2,
                        spectrum_params,
                    )

                    # Compute log likelihood for secondary data
                    log_l_2 = LogLikelihood(
                        model_spectrum=model_spectrum_processed_2,
                        data_spectrum=data_2_spectrum,
                        data_uncertainties=data_2_uncertainties,
                        data_covariance=data_2_covariance,
                        n_free_parameters=n_free_parameters,
                    ).calculate_log_likelihood()

                    per_datapoint_contribution += log_l_2

            # === Handle Per-Datapoint Tracking ===
            if per_datapoint:
                log_l_per_datapoint_dict[data_name].append(per_datapoint_contribution)
            else:
                log_likelihood_contributions.append(per_datapoint_contribution)

        # === Return Per-Datapoint if Requested ===
        if per_datapoint:
            return log_l_per_datapoint_dict

        # === Compute Total Log Likelihood ===
        log_likelihood = jnp.sum(jnp.array(log_likelihood_contributions))

        # === Apply Prior Weight ===
        if "log_prior_weight" in parameters.keys():
            log_prior += parameters["log_prior_weight"].value

        # === JAX-Compatible Clamping ===
        log_likelihood, log_prior = lax.cond(
            log_likelihood + log_prior < invalid_value,
            lambda: (invalid_value, 0.0),
            lambda: (log_likelihood, log_prior),
        )

        log_likelihood, log_prior = lax.cond(
            jnp.abs(log_likelihood + log_prior) < tiny,
            lambda: (tiny, 0.0),
            lambda: (log_likelihood, log_prior),
        )

        return log_likelihood + log_prior

    @staticmethod
    def _compute_object_array_log_likelihood(
        data, spectrum_model: jnp.ndarray, beta: float, beta_mode: str
    ) -> float:
        """
        Helper method to compute log likelihood for object array spectra.

        This extracts the complex object array handling logic to improve clarity.

        Args:
            data: Data object with object array spectra
            spectrum_model: Model spectrum
            beta: Uncertainty scaling parameter
            beta_mode: Mode for applying beta

        Returns:
            Total log likelihood for all object array entries
        """
        log_l = 0.0
        spectrum_ndim = jnp.ndim(data.spectrum)

        if spectrum_ndim == 1:
            # 1D object array
            for i, spectrum_data in enumerate(data.spectrum):
                if jnp.isnan(spectrum_model[i][~data.mask[i]]).any():
                    return -1e99
                log_l += data.log_likelihood(
                    spectrum_model[i][~data.mask[i]],
                    spectrum_data,
                    data.uncertainties[i],
                    beta=beta,
                    beta_mode=beta_mode,
                )
        elif spectrum_ndim == 2:
            # 2D object array
            for i, detector in enumerate(data.spectrum):
                for j, spectrum_data in enumerate(detector):
                    if jnp.isnan(spectrum_model[i, j][~data.mask[i, j]]).any():
                        return -1e99
                    log_l += data.log_likelihood(
                        spectrum_model[i, j][~data.mask[i, j]],
                        spectrum_data,
                        data.uncertainties[i, j],
                        beta=beta,
                        beta_mode=beta_mode,
                    )
        else:
            raise ValueError(
                f"Object array spectrum has {spectrum_ndim} dimensions, "
                f"but must have 1 or 2"
            )

        return log_l

    def prior(
        self, cube: npt.NDArray[float], ndim: int = 0, nparams: int = 0
    ) -> npt.NDArray[float]:
        """
        pyMultinest Prior function. Transforms unit hypercube into physical space.
        """
        del ndim, nparams
        return self._get_parameter_layout().prior_transform_inplace(cube)

    def prior_ultranest(
        self, cube: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """
        UltraNest prior function. Transforms unit hypercube into physical space.
        """
        return self._get_parameter_layout().prior_transform_numpy(cube)

    def prior_model_jaxns(self):
        _, prior_model, _ = self._build_jaxns_sampler_interface()
        return (yield from prior_model())

    def _build_ultranest_sampler_interface(self, vectorized: bool = False):
        if not vectorized:
            return self.log_likelihood, self.prior_ultranest

        parameter_layout = self._get_parameter_layout()
        runtime = self._get_runtime()

        def prior_transform(cube):
            return parameter_layout.prior_transform_numpy(cube)

        def log_likelihood_batch(physical_positions):
            return np.asarray(
                runtime.evaluate_jax_batched(
                    free_parameter_values=physical_positions,
                    uncertainties_mode=self.uncertainties_mode,
                    print_log_likelihood_for_debugging=False,
                )
            )

        return log_likelihood_batch, prior_transform

    def _build_sampler_context(self) -> SamplerContext:
        """Build a SamplerContext from the current Retrieval state."""
        parameter_layout = self._get_parameter_layout()
        runtime = self._get_runtime()
        return SamplerContext(
            parameter_layout=parameter_layout,
            runtime=runtime,
            configuration=self.configuration,
            output_directory=self.output_directory,
            retrieval_name=self.configuration.retrieval_name,
            uncertainties_mode=self.uncertainties_mode,
            print_log_likelihood_for_debugging=self.print_log_likelihood_for_debugging,
            log_likelihood_func=self.log_likelihood,
            prior_func=self.prior,
            prior_ultranest_func=self.prior_ultranest,
            free_parameter_names=list(parameter_layout.free_parameter_names),
            n_free_parameters=parameter_layout.n_free_parameters,
        )

    def _build_jaxns_sampler_interface(self):
        self._ensure_jax_sampler_model_support("JAXNS")
        parameter_layout = self._get_parameter_layout()
        runtime = self._get_runtime()
        free_parameter_names = list(parameter_layout.free_parameter_names)

        if len(free_parameter_names) < 1:
            raise ValueError("JAXNS samplers require at least one free parameter.")

        priors = []
        for parameter_name in free_parameter_names:
            parameter = self.configuration.parameters[parameter_name]
            if parameter.distribution is None:
                raise ValueError(
                    f"JAXNS samplers require distribution-backed free parameters; '{parameter_name}' has no distribution."
                )
            priors.append(parameter.set_prior())

        priors = tuple(priors)

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
                uncertainties_mode=self.uncertainties_mode,
                print_log_likelihood_for_debugging=False,
            )

        return free_parameter_names, prior_model, log_likelihood_wrapper

    def _build_unconstrained_mcmc_interface(self):
        self._ensure_jax_sampler_model_support("Gradient-based JAX samplers")
        parameter_layout = self._get_parameter_layout()
        runtime = self._get_runtime()
        free_parameter_names = list(parameter_layout.free_parameter_names)

        if len(free_parameter_names) < 1:
            raise ValueError(
                "Gradient-based samplers require at least one free parameter."
            )

        free_parameters = [
            self.configuration.parameters[name] for name in free_parameter_names
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
                uncertainties_mode=self.uncertainties_mode,
                print_log_likelihood_for_debugging=self.print_log_likelihood_for_debugging,
            ) + log_jacobian_single(unconstrained_position)

        initial_position = jnp.zeros(
            parameter_layout.n_free_parameters, dtype=JAX_DTYPE
        )

        return (
            free_parameter_names,
            initial_position,
            logdensity,
            transform_positions,
            log_jacobian,
        )

    def _build_blackjax_logdensity_interface(self):
        return self._build_unconstrained_mcmc_interface()

    def _build_blackjax_sampler_kwargs(
        self,
        sampler_kwargs: dict,
        *,
        free_parameter_names: list[str],
        initial_position,
        log_jacobian,
        algorithm_name: str,
    ) -> dict:
        blackjax_kwargs = {
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
            "use_jit": sampler_kwargs.pop("use_jit", True),
            "log_jacobian_fn": log_jacobian,
        }
        if algorithm_name == "hmc":
            blackjax_kwargs["num_integration_steps"] = sampler_kwargs.pop(
                "num_integration_steps", 20
            )
        blackjax_kwargs.update(sampler_kwargs)
        return blackjax_kwargs

    def _build_numpyro_sampler_kwargs(
        self,
        sampler_kwargs: dict,
        *,
        free_parameter_names: list[str],
        initial_position,
        log_jacobian,
        algorithm_name: str,
    ) -> dict:
        numpyro_kwargs = {
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
            "use_jit": sampler_kwargs.pop("use_jit", True),
            "log_jacobian_fn": log_jacobian,
            "num_chains": sampler_kwargs.pop("num_chains", 1),
            "thinning": sampler_kwargs.pop("thinning", 1),
            "chain_method": sampler_kwargs.pop("chain_method", "parallel"),
        }

        if algorithm_name == "hmc":
            numpyro_num_steps = sampler_kwargs.pop("num_integration_steps", None)
            if numpyro_num_steps is None:
                numpyro_num_steps = sampler_kwargs.pop("num_steps", None)
            if numpyro_num_steps is None and "trajectory_length" not in sampler_kwargs:
                numpyro_num_steps = 20
            if numpyro_num_steps is not None:
                numpyro_kwargs["num_steps"] = numpyro_num_steps

        numpyro_kwargs.update(sampler_kwargs)
        return numpyro_kwargs

    @staticmethod
    def _build_ultranest_summary_parameters(
        *,
        sampler_type: str,
        parameter_names: list[str],
        vectorized: bool,
    ) -> dict:
        return {
            "sampler_type": sampler_type,
            "parameter_names": parameter_names,
            "vectorized": vectorized,
        }

    @staticmethod
    def _build_blackjax_summary_parameters(
        *,
        sampler_type: str,
        sampler_kwargs: dict,
    ) -> dict:
        summary_parameters = {
            "sampler_type": sampler_type,
            "num_samples": sampler_kwargs["num_samples"],
            "num_warmup": sampler_kwargs["num_warmup"],
            "initial_step_size": sampler_kwargs["initial_step_size"],
            "target_acceptance_rate": sampler_kwargs["target_acceptance_rate"],
            "is_mass_matrix_diagonal": sampler_kwargs["is_mass_matrix_diagonal"],
            "seed": sampler_kwargs["seed"],
            "use_jit": sampler_kwargs["use_jit"],
        }
        if "num_integration_steps" in sampler_kwargs:
            summary_parameters["num_integration_steps"] = sampler_kwargs[
                "num_integration_steps"
            ]
        return summary_parameters

    @staticmethod
    def _build_numpyro_summary_parameters(
        *,
        sampler_type: str,
        sampler_kwargs: dict,
    ) -> dict:
        summary_parameters = {
            "sampler_type": sampler_type,
            "num_samples": sampler_kwargs["num_samples"],
            "num_warmup": sampler_kwargs["num_warmup"],
            "initial_step_size": sampler_kwargs["initial_step_size"],
            "target_acceptance_rate": sampler_kwargs["target_acceptance_rate"],
            "is_mass_matrix_diagonal": sampler_kwargs["is_mass_matrix_diagonal"],
            "num_chains": sampler_kwargs["num_chains"],
            "thinning": sampler_kwargs["thinning"],
            "chain_method": sampler_kwargs["chain_method"],
            "seed": sampler_kwargs["seed"],
            "use_jit": sampler_kwargs["use_jit"],
        }
        if "num_steps" in sampler_kwargs:
            summary_parameters["num_integration_steps"] = sampler_kwargs["num_steps"]
        if "trajectory_length" in sampler_kwargs:
            summary_parameters["trajectory_length"] = sampler_kwargs[
                "trajectory_length"
            ]
        return summary_parameters

    @staticmethod
    def _build_jaxns_summary_parameters(
        *,
        sampler_type: str,
        sampler_kwargs: dict,
        parameter_names: list[str],
    ) -> dict:
        return {
            **sampler_kwargs,
            "sampler_type": sampler_type,
            "parameter_names": parameter_names,
        }

    def run(self, sampler_object: Sampler = None, **sampler_kwargs):
        """
        Run mode for the class. Uses pynultinest to sample parameter space
        and produce standard PMN outputs.
        Args:
            sampler_type : str
                The type of sampler to use. Can be one of the following:
                    pymultinest - requires Multinest installation.
                    dynesty - requires Dynesty and runs dynamic nested sampling.
                    blackjaxnuts - requires BlackJAX and runs NUTS with window adaptation.
                    blackjaxhmc - requires BlackJAX and runs HMC with window adaptation.
                    numpyronuts - requires NumPyro and runs NUTS with warmup adaptation.
                    numpyrohmc - requires NumPyro and runs HMC with warmup adaptation.
                    jaxns - Requires JAXNS. Default choice for JAXNS
                    jaxnsshardedstaticnestedsampler - Requires JAXNS. Advanced option for JAXNS
                    ultranest - Requires ultranest. Implements.
        """
        sampler_type = self.configuration.sampler_type
        sampler_type_normalized = sampler_type.lower()
        print(
            f"Starting retrieval {self.configuration.retrieval_name} with {sampler_type}"
        )
        mpi, comm, rank = get_mpi_context()

        if mpi is None and sampler_type_normalized == "pymultinest":
            print(
                "Unable to import mpi4py, using slow (single process) mode for pymultinest."
            )
        sampler = self.sampler
        error_checking = sampler_kwargs.pop("error_checking", True)
        force_serial_error_checking = sampler_kwargs.pop(
            "force_serial_error_checking", False
        )
        # any_error_checking = comm.allreduce(error_checking, op=MPI.LOR)
        # %if MPI is not None and comm is not None else error_checking

        if error_checking:
            if force_serial_error_checking and mpi is not None and comm is not None:
                for i in range(comm.Get_size()):
                    if rank == i:
                        self._check_errors()
                    comm.barrier()
            else:
                self._check_errors()
        if comm is not None:
            comm.barrier()

        if sampler_object is not None:
            sampler = sampler_object
            results = sampler.run_sampler(**sampler_kwargs)
            return self._finalize_sampler_run(sampler, results)

        sampler_cls = _resolve_sampler_class(sampler_type_normalized)
        if sampler_cls is None:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        context = self._build_sampler_context()
        sampler, run_kwargs, summary_parameters = sampler_cls.prepare(
            context, **sampler_kwargs
        )
        results = sampler.run_sampler(**run_kwargs)
        results, extra_finalize_kwargs = sampler.post_run(results)
        return self._finalize_sampler_run(
            sampler,
            results,
            sampler_parameters=summary_parameters,
            **extra_finalize_kwargs,
        )

    def sample_effective_temperature(
        self, sample_dict, param_dict, ret_names=None, nsample=None, resolution=40
    ):
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
                The number of times to compute Teff. If left empty, uses the configured
                reference_data_name. Recommended to use ~300 samples, probably more than
                the default plotting sample count.
            resolution : int
                The spectra resolution to compute the models at. Typically, this should be very
                low in order to enable rapid calculation.
        Returns:
            temperature_dict : dict
                A dictionary with retrieval names for keys, and the values are the calculated
                values of Teff for each sample.
        """
        from petitRADTRANS.physics import compute_effective_temperature

        if ret_names is None:
            ret_names = [self.configuration.retrieval_name]
        if nsample is None:
            nsample = self.plotter.nsample

        # Set up the pRT object
        self._rebin_opacities(resolution=resolution)

        species = ()

        for _species in self.configuration.line_species:
            name, natural_abundance, charge, cloud_info, source, spectral_info = (
                CorrelatedKOpacity.split_species_all_info(_species, full=False)
            )
            species.append(
                CorrelatedKOpacity.join_species_all_info(
                    species_name=name,
                    natural_abundance=natural_abundance,
                    charge=charge,
                    cloud_info=cloud_info,
                    source=source,
                    spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                        resolution
                    ),
                ),
            )

        pressures = self.configuration.pressures

        radtrans_object = Radtrans(
            pressures=pressures,
            line_species=copy.copy(species),
            rayleigh_species=copy.copy(self.configuration.rayleigh_species),
            gas_continuum_contributors=copy.copy(
                self.configuration.continuum_opacities
            ),
            cloud_species=copy.copy(self.configuration.cloud_species),
            line_opacity_mode="c-k",
            wavelength_boundaries=np.array([0.5, 28]),
            scattering_in_emission=self.configuration.scattering_in_emission,
        )

        temperature_dict = {}

        for name in ret_names:
            temperature_fits = []
            samples = sample_dict[name]
            parameters_read = param_dict[name]

            if nsample == "all":
                random_numbers = np.linspace(0, samples.shape[1] - 1, samples.shape[1])
            else:
                random_numbers = np.random.randint(0, samples.shape[1], int(nsample))

            data_use = self.plotter.get_reference_data()

            if data_use.external_radtrans_reference is not None:
                data_use = self.configuration.data[data_use.external_radtrans_reference]

            for random_number in random_numbers:
                sample = samples[:-1, int(random_number)]
                parameters = self.build_param_dict(sample, parameters_read)
                ret_val = self._call_model_generating_function(
                    data_use.model_generating_function,
                    radtrans_object,
                    parameters,
                    data_name=data_use.name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                )

                if len(ret_val) == 2:
                    wavelengths, model = ret_val
                else:
                    wavelengths, model, __ = ret_val

                temperature_fit = compute_effective_temperature(
                    wavelengths=wavelengths,
                    flux=model,
                    orbit_semi_major_axis=parameters["D_pl"].value,
                    planet_radius=parameters["planet_radius"].value,
                    use_si_units=True,
                )
                temperature_fits.append(temperature_fit)

            temperature_dict[name] = np.array(temperature_fits)
            np.save(
                f"{self.output_directory}/evaluate_{name}/{name}_sampled_teff",
                np.array(temperature_fits),
            )

        return temperature_dict

    def save_best_fit_outputs(
        self,
        parameters,
        only_return_best_fit_spectra=False,
        retain_best_fit_spectra=True,
    ):
        best_fit_spectra = self.best_fit_spectra if retain_best_fit_spectra else {}

        if retain_best_fit_spectra:
            best_fit_spectra.clear()
        else:
            self.best_fit_spectra = {}

        # Save sampled outputs if necessary.
        for name, data in self.configuration.data.items():
            # Only calculate spectra within a given
            # Wavelengths range once
            if data.scale_flux or data.scale_uncertainties:
                data.scale_factor = parameters[name + "_scale_factor"].value
            if data.fit_flux_offset:
                data.offset = parameters[name + "_offset"].value
            if name + "_b" in parameters.keys():
                data.bval = parameters[name + "_b"].value

            if data.external_radtrans_reference is None:
                # Compute the model
                ret_val = self._call_model_generating_function(
                    data.model_generating_function,
                    data.radtrans_object,
                    parameters,
                    data_name=name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                )
                if len(ret_val) == 3:
                    wavelengths_model, spectrum_model, additional_logl = ret_val
                elif (
                    len(ret_val) == 4
                    and data.variability_atmospheric_column_model_flux_return_mode
                ):
                    (
                        wavelengths_model,
                        spectrum_model,
                        additional_logl,
                        atmospheric_model_column_fluxes,
                    ) = ret_val
                    spectrum_model = data.atmospheric_column_flux_mixer(
                        atmospheric_model_column_fluxes, parameters, data.name
                    )
                else:
                    wavelengths_model, spectrum_model = ret_val
            else:
                # Compute the model
                prt_obj = self.configuration.data[
                    data.external_radtrans_reference
                ].radtrans_object
                ret_val = self._call_model_generating_function(
                    self.configuration.data[
                        data.external_radtrans_reference
                    ].model_generating_function,
                    prt_obj,
                    parameters,
                    data_name=data.external_radtrans_reference,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                )

                if len(ret_val) == 3:
                    wavelengths_model, spectrum_model, additional_logl = ret_val
                elif (
                    len(ret_val) == 4
                    and data.variability_atmospheric_column_model_flux_return_mode
                ):
                    (
                        wavelengths_model,
                        spectrum_model,
                        additional_logl,
                        atmospheric_model_column_fluxes,
                    ) = ret_val
                    spectrum_model = data.atmospheric_column_flux_mixer(
                        atmospheric_model_column_fluxes, parameters, data.name
                    )
                else:
                    wavelengths_model, spectrum_model = ret_val

            spectrum_params = extract_spectrum_parameters(parameters, name)

            if not isinstance(data, PhotometryData):
                binned = process_spectrum_pipeline(
                    wavelengths_model=np.asarray(wavelengths_model),
                    spectrum_model=np.asarray(spectrum_model),
                    data=data,
                    parameters=parameters,
                    data_name=name,
                    spectrum_params=spectrum_params,
                )
            else:
                if data.data_resolution_array_model is not None:
                    data.initialise_data_resolution(wavelengths_model)
                    spectrum_model = convolve(
                        wavelengths_model,
                        spectrum_model,
                        data.data_resolution_array_model,
                    )
                elif data.data_resolution is not None:
                    spectrum_model = convolve(
                        wavelengths_model, spectrum_model, data.data_resolution
                    )

                binned = data.photometric_transformation_function(
                    wavelengths_model, spectrum_model
                )

            if self.evaluate_sample_spectra:
                self.posterior_sample_spectra[name] = [data.wavelengths, binned]
            else:
                data_safe_name = name.replace("/", "_").replace(".", "_")

                if (not only_return_best_fit_spectra) and (
                    not isinstance(data, PhotometryData)
                ):
                    np.savetxt(
                        os.path.join(
                            self.output_directory,
                            "evaluate_" + self.configuration.retrieval_name,
                            f"{self.configuration.retrieval_name}_model_spec_best_fit_{data_safe_name}.dat",
                        ),
                        np.column_stack((data.wavelengths, binned)),
                    )

                best_fit_spectra[name] = [data.wavelengths, binned]

        return best_fit_spectra

    def save_best_fit_outputs_external_variability(
        self,
        parameters,
        only_return_best_fit_spectra=False,
        retain_best_fit_spectra=True,
    ):
        best_fit_spectra = self.best_fit_spectra if retain_best_fit_spectra else {}

        if retain_best_fit_spectra:
            best_fit_spectra.clear()
        else:
            self.best_fit_spectra = {}

        wavelengths_model = None  # prevent potential reference before assignment

        # Save sampled outputs if necessary
        for name, data in self.configuration.data.items():
            # Prevent potential reference before assignment
            atmospheric_model_column_fluxes = None
            wavelengths_model = None
            spectrum_model = None

            if data.external_radtrans_reference is None:
                # Only calculate spectra within a given wavelengths range once
                if data.scale_flux or data.scale_uncertainties:
                    data.scale_factor = parameters[name + "_scale_factor"].value

                if data.fit_flux_offset:
                    data.offset = parameters[name + "_offset"].value

                if name + "_b" in parameters.keys():
                    data.bval = parameters[name + "_b"].value

                # Compute the model
                results = self._call_model_generating_function(
                    data.model_generating_function,
                    data.radtrans_object,
                    parameters,
                    data_name=name,
                    pt_plot_mode=False,
                    adaptive_mesh_refinement=self.configuration.adaptive_mesh_refinement,
                )

                if len(results) == 3:
                    wavelengths_model, spectrum_model, additional_logl = results
                elif (
                    len(results) == 4
                    and data.variability_atmospheric_column_model_flux_return_mode
                ):
                    (
                        wavelengths_model,
                        spectrum_model,
                        additional_logl,
                        atmospheric_model_column_fluxes,
                    ) = results

                    spectrum_model = data.atmospheric_column_flux_mixer(
                        atmospheric_model_column_fluxes, parameters, data.name
                    )
                else:
                    wavelengths_model, spectrum_model = results

                if data.data_resolution_array_model is not None:
                    data.initialise_data_resolution(wavelengths_model)
                    spectrum_model = convolve(
                        wavelengths_model,
                        spectrum_model,
                        data.data_resolution_array_model,
                    )
                elif data.data_resolution is not None:
                    spectrum_model = convolve(
                        wavelengths_model, spectrum_model, data.data_resolution
                    )
                binned = rebin_spectrum_bin(
                    wavelengths_model,
                    spectrum_model,
                    data.wavelengths,
                    data.wavelength_bin_widths,
                )
                best_fit_spectra[name] = [data.wavelengths, binned]

            for name_2, data_2 in self.configuration.data.items():
                if data_2.external_radtrans_reference is not None:
                    if data_2.external_radtrans_reference == name:
                        if (
                            data_2.variability_atmospheric_column_model_flux_return_mode
                            and atmospheric_model_column_fluxes is not None
                        ):
                            spectrum_model_2 = data_2.atmospheric_column_flux_mixer(
                                atmospheric_model_column_fluxes, parameters, data_2.name
                            )

                            if data.data_resolution_array_model is not None:
                                data.initialise_data_resolution(wavelengths_model)
                                spectrum_model_2 = convolve(
                                    wavelengths_model,
                                    spectrum_model_2,
                                    data.data_resolution_array_model,
                                )
                            elif data.data_resolution is not None:
                                spectrum_model_2 = convolve(
                                    wavelengths_model,
                                    spectrum_model_2,
                                    data.data_resolution,
                                )
                            binned_2 = rebin_spectrum_bin(
                                wavelengths_model,
                                spectrum_model_2,
                                data.wavelengths,
                                data.wavelength_bin_widths,
                            )
                            best_fit_spectra[name_2] = [data.wavelengths, binned_2]
                        else:
                            # TODO @Evert 'binned' may not be assigned if data.external_radtrans_reference is not None
                            #  , fix the condition and the other now useless (?) variables marked "prevent reference
                            #  befor assignment"
                            best_fit_spectra[name_2] = [data.wavelengths, binned]

        for name, data in self.configuration.data.items():
            data_safe_name = name.replace("/", "_").replace(".", "_")
            if not only_return_best_fit_spectra:
                np.savetxt(
                    os.path.join(
                        self.output_directory,
                        "evaluate_" + self.configuration.retrieval_name,
                        f"{self.configuration.retrieval_name}_model_spec_best_fit_{data_safe_name}.dat",
                    ),
                    np.column_stack((best_fit_spectra[name])),
                )
        return best_fit_spectra

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
            name = self.plotter.get_reference_radtrans_name()

            species = [
                Opacity([species]).get_full_name().split(".R")[0]
                for species in self.configuration.data[
                    name
                ].radtrans_object.line_species
            ]

            samples_use = sample_dict[ret].T
            parameters_read = parameter_dict[ret]

            for sample in samples_use:
                m_frac, _ = self.get_mass_fractions(sample[:-1], parameters_read)
                mass_fractions.append(np.array(list(m_frac.values())))

            mass_fractions = np.array(mass_fractions)
            np.save(
                os.path.join(
                    self.output_directory,
                    f"evaluate_{ret}",
                    f"{ret}_mass_fraction_profiles",
                ),
                mass_fractions,
            )
            line_spec_file = os.path.join(
                self.output_directory,
                f"evaluate_{ret}",
                f"{ret}_line_absorber_species.json",
            )

            with open(line_spec_file, "w+") as myFile:
                json.dump(species, myFile)

        return mass_fractions

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
            name = self.plotter.get_reference_radtrans_name()

            species = [
                Opacity([species]).get_full_name().split(".R")[0]
                for species in self.configuration.data[
                    name
                ].radtrans_object.line_species
            ]

            samples_use = sample_dict[ret].T
            parameters_read = parameter_dict[ret]
            vmrs = []

            for sample in samples_use:
                vmr, _ = self.get_volume_mixing_ratios(sample[:-1], parameters_read)
                vmrs.append(np.array(list(vmr.values())))

            vmrs = np.array(vmrs)
            np.save(
                os.path.join(
                    self.output_directory,
                    f"evaluate_{ret}",
                    f"{ret}_volume_mixing_ratio_profiles",
                ),
                vmrs,
            )
            line_spec_file = os.path.join(
                self.output_directory,
                f"evaluate_{ret}",
                f"{ret}_line_absorber_species.json",
            )

            with open(line_spec_file, "w+") as myFile:
                json.dump(species, myFile)

        return vmrs

    def initialise_radtrans_objects(self, scaling=10, width=3):
        """
        Creates a pRT object for each data set that asks for a unique object.
        Checks if there are low resolution c-k models from exo-k, and creates them if necessary.
        The scaling and width parameters are retained for API compatibility and
        are ignored by the fixed-size adaptive mesh refinement path.

        Args:
            scaling : int
                A multiplicative factor that determines the size of the full high resolution pressure grid,
                which will have length self.p_global.shape[0] * scaling.
            width : int
                The number of cells in the low pressure grid to replace with the high resolution grid.
        """
        radtrans = None

        for name, data in self.configuration.data.items():
            if data.radtrans_object is not None:
                print(f"Using provided Radtrans object for data '{name}'...")
                continue

            # Only create if there's no other data object using the same pRT object
            if data.external_radtrans_reference is None:
                print(f"Setting up Radtrans object for data '{name}'...")

                if (
                    data.line_opacity_mode == "c-k"
                    and data.model_resolution is not None
                ):
                    # Use ExoK to have low res models.
                    self._rebin_opacities(resolution=data.model_resolution)

                    species = ()

                    for line_species in self.configuration.line_species:
                        line_species, spectral_info = (
                            CorrelatedKOpacity.split_species_spectral_info(line_species)
                        )
                        opacity_wavelength_range = (
                            CorrelatedKOpacity.split_spectral_info(spectral_info)[1]
                        )

                        species = species + (
                            CorrelatedKOpacity.join_species_all_info(
                                line_species,
                                spectral_sampling=CorrelatedKOpacity.get_resolving_power_string(
                                    data.model_resolution
                                ),
                                wavelength_range=opacity_wavelength_range,
                            ),
                        )
                else:
                    # Otherwise for 'lbl' or no model_resolution binning,
                    # we just use the default species.
                    species = copy.deepcopy(self.configuration.line_species)

                line_by_line_sampling = None

                if (
                    data.line_opacity_mode == "lbl"
                    and data.model_resolution is not None
                ):
                    line_by_line_sampling = int(1e6 / data.model_resolution)

                # Create random P-T profile to create RT arrays of the Radtrans object.
                pressures = self.configuration.pressures

                # Set up the pRT objects for the given dataset
                radtrans = Radtrans(
                    pressures=pressures,
                    line_species=copy.copy(tuple(species)),
                    rayleigh_species=copy.copy(self.configuration.rayleigh_species),
                    gas_continuum_contributors=copy.copy(
                        self.configuration.continuum_opacities
                    ),
                    cloud_species=copy.copy(self.configuration.cloud_species),
                    line_opacity_mode=data.line_opacity_mode,
                    wavelength_boundaries=data.wavelength_boundaries,
                    scattering_in_emission=self.configuration.scattering_in_emission,
                    line_by_line_opacity_sampling=line_by_line_sampling,
                )
                data.radtrans_object = radtrans

            # Setup for non-uniform spectral resolution
            if isinstance(data.data_resolution, np.ndarray):
                data.initialise_data_resolution(1e4 * radtrans.get_wavelengths())

    def plot_abundances(
        self,
        samples_use,
        parameters_read,
        species_to_plot=None,
        contribution=False,
        refresh=True,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
        sample_posteriors=False,
        volume_mixing_ratio=False,
    ):
        """
        Plot abundance profiles (mass fractions or volume mixing ratios) as a function of pressure.
        This is a wrapper for RetrievalPlotter.plot_abundances.

        Args:
            samples_use: Array of samples from the posterior.
            parameters_read: List of free parameter names.
            species_to_plot: List of species to plot (optional).
            contribution: If True, overplot the contribution function.
            refresh: If True, recalculate and overwrite cached results.
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            mode: 'bestfit' or 'median' for which sample to plot.
            sample_posteriors: If True, plot posterior intervals.
            volume_mixing_ratio: If True, plot volume mixing ratios instead of mass fractions.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        return self.plotter.plot_abundances(
            samples_use=samples_use,
            parameters_read=parameters_read,
            species_to_plot=species_to_plot,
            contribution=contribution,
            refresh=refresh,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            mode=mode,
            sample_posteriors=sample_posteriors,
            volume_mixing_ratio=volume_mixing_ratio,
        )

    def plot_all(
        self,
        output_directory=None,
        ret_names=None,
        contribution=False,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
    ):
        """
        Generate all standard plots for a retrieval, including best-fit spectrum, sample spectra, PT profile, corner plot, and abundances.
        This is a wrapper for RetrievalPlotter.plot_all.

        Args:
            output_directory: Directory to save plots (optional).
            ret_names: List of retrieval names for plotting (optional).
            contribution: If True, plot contribution function.
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            mode: 'bestfit' or 'median' for which sample to plot.
        Returns:
            None
        """
        return self.plotter.plot_all(
            output_directory=output_directory,
            ret_names=ret_names,
            contribution=contribution,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            mode=mode,
        )

    def plot_contribution(
        self,
        samples_use,
        parameters_read,
        model_generating_function=None,
        prt_reference=None,
        log_scale_contribution=False,
        n_contour_levels=30,
        refresh=True,
        mode="bestfit",
    ):
        """
        Plot the contribution function of the best-fit or median model from a retrieval.
        This is a wrapper for RetrievalPlotter.plot_contribution.

        Args:
            samples_use: Array of samples from the posterior.
            parameters_read: List of free parameter names.
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            log_scale_contribution: If True, plot -log10(weighted flux).
            n_contour_levels: Number of contour levels in the plot.
            refresh: If True, recalculate and overwrite cached results.
            mode: 'bestfit' or 'median' for which sample to plot.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        return self.plotter.plot_contribution(
            samples_use=samples_use,
            parameters_read=parameters_read,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            log_scale_contribution=log_scale_contribution,
            n_contour_levels=n_contour_levels,
            refresh=refresh,
            mode=mode,
        )

    def plot_corner(
        self,
        sample_dict,
        parameter_dict,
        parameters_read,
        plot_best_fit=True,
        true_values=None,
        **kwargs,
    ):
        """
        Make a corner plot of the posterior samples for the retrieved parameters.
        This is a wrapper for RetrievalPlotter.plot_corner.

        Args:
            sample_dict: Dictionary of posterior samples for each retrieval.
            parameter_dict: Dictionary of parameter names for each retrieval.
            parameters_read: List of free parameter names.
            plot_best_fit: If True, mark the best-fit point on the plot.
            true_values: Optional dictionary of true parameter values for reference.
            **kwargs: Additional keyword arguments passed to the plotting function.
        Returns:
            fig: Matplotlib figure object.
        """
        return self.plotter.plot_corner(
            sample_dict=sample_dict,
            parameter_dict=parameter_dict,
            parameters_read=parameters_read,
            plot_best_fit=plot_best_fit,
            true_values=true_values,
            **kwargs,
        )

    def plot_data(self, yscale="linear"):
        """
        Plot the observational data used in the retrieval.
        This is a wrapper for RetrievalPlotter.plot_data.

        Args:
            yscale: Y-axis scaling for the plot (default 'linear').
        Returns:
            None
        """
        return self.plotter.plot_data(yscale=yscale)

    def plot_pt(
        self,
        sample_dict,
        parameters_read,
        contribution=False,
        refresh=False,
        model_generating_function=None,
        prt_reference=None,
        mode="bestfit",
    ):
        """
        Plot the pressure-temperature (PT) profile with error contours for the retrieval.
        This is a wrapper for RetrievalPlotter.plot_pt.

        Args:
            sample_dict: Dictionary of posterior samples for each retrieval.
            parameters_read: List of free parameter names.
            contribution: If True, overplot the contribution function.
            refresh: If True, recalculate and overwrite cached results.
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            mode: 'bestfit' or 'median' for which sample to plot.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        return self.plotter.plot_pt(
            sample_dict=sample_dict,
            parameters_read=parameters_read,
            contribution=contribution,
            refresh=refresh,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            mode=mode,
        )

    def plot_sampled(
        self,
        samples_use,
        parameters_read,
        downsample_factor=None,
        save_outputs=False,
        nsample=None,
        model_generating_function=None,
        prt_reference=None,
        refresh=True,
    ):
        """
        Plot a set of randomly sampled output spectra for each dataset in the retrieval.
        This is a wrapper for RetrievalPlotter.plot_sampled.

        Args:
            samples_use: Array of samples from the posterior.
            parameters_read: List of free parameter names.
            downsample_factor: Optional factor to downsample the spectra.
            save_outputs: If True, save the sampled spectra to disk.
            nsample: Number of samples to plot (optional).
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            refresh: If True, recalculate and overwrite cached results.
        Returns:
            fig, ax: Matplotlib figure and axis objects.
        """
        return self.plotter.plot_sampled(
            samples_use=samples_use,
            parameters_read=parameters_read,
            downsample_factor=downsample_factor,
            save_outputs=save_outputs,
            nsample=nsample,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            refresh=refresh,
        )

    def plot_spectra(
        self,
        samples_use,
        parameters_read,
        model_generating_function=None,
        prt_reference=None,
        refresh=True,
        mode="bestfit",
        marker_color_type=None,
        marker_cmap=None,
        marker_label="",
        only_save_best_fit_spectra=False,
    ):
        """
        Plot the best-fit spectrum, the data from each dataset, and the residuals between the two.
        This is a wrapper for RetrievalPlotter.plot_spectra.

        Args:
            samples_use: Array of samples from the posterior.
            parameters_read: List of free parameter names.
            model_generating_function: Optional model function for spectrum generation.
            prt_reference: Optional reference for pRT object.
            refresh: If True, recalculate and overwrite cached results.
            mode: 'bestfit' or 'median' for which sample to plot.
            marker_color_type: Optional marker color type for data points.
            marker_cmap: Optional colormap for markers.
            marker_label: Optional label for markers.
            only_save_best_fit_spectra: If True, only save the best-fit spectra to disk.
        Returns:
            fig, ax, ax_r: Matplotlib figure and axis objects for the spectrum and residuals.
        """
        return self.plotter.plot_spectra(
            samples_use=samples_use,
            parameters_read=parameters_read,
            model_generating_function=model_generating_function,
            prt_reference=prt_reference,
            refresh=refresh,
            mode=mode,
            marker_color_type=marker_color_type,
            marker_cmap=marker_cmap,
            marker_label=marker_label,
            only_save_best_fit_spectra=only_save_best_fit_spectra,
        )

    def save_configuration(self):
        """
        Save the retrieval_config configuration to file. Warning, can take up a significant amount of storage
        space, as the opacities for each Radtrans object will also be saved to file.
        """
        filename = f"{self.configuration.retrieval_name}_configuration.pkl"
        with open(
            f"{self.output_directory}/evaluate_{self.configuration.retrieval_name}/{filename}",
            "wb",
        ) as db:
            dill.dump(self.configuration, db)
        print(f"Saving configuration to file: {filename}")
