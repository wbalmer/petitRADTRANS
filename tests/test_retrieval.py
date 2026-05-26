"""Test the petitRADTRANS retrieval module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).
"""

import ctypes
import json
import os
import sys
from types import ModuleType, SimpleNamespace
import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

jax.config.update("jax_enable_x64", True)
os.environ["DYLD_LIBRARY_PATH"] = (
    "/Users/nasedkin/software/MultiNest/lib/:$DYLD_LIBRARY_PATH"
)
os.environ["LD_LIBRARY_PATH"] = (
    "/Users/nasedkin/software/MultiNest/lib/:$LD_LIBRARY_PATH"
)

import numpy as np

tfpd = tfp.distributions

from petitRADTRANS.chemistry.utils import (
    build_adaptive_pressure_grid,
    compute_mean_molar_masses,
)
from petitRADTRANS.opacities.opacities import Opacity, CorrelatedKOpacity
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.plotting import RetrievalPlotter
from petitRADTRANS.retrieval.models import (
    MODEL_CONTRACT_LEGACY_ADAPTER,
    MODEL_CONTRACT_DIFFERENTIABLE,
    _resolve_runtime_abundance_state,
    calculate_emission_spectrum,
    calculate_emission_spectrum_runtime,
    calculate_transmission_spectrum,
    calculate_transmission_spectrum_runtime,
    coerce_model_generating_function,
    initialize_pressure,
    mark_legacy_model_function,
    run_legacy_model_adapter,
    run_runtime_model_adapter,
    molliere_2020_two_column,
    gradient_profile_emission,
    guillot_emission,
    guillot_emission_add_gaussian_temperature,
    guillot_patchy_emission,
    guillot_patchy_transmission,
    guillot_transmission,
    interpolated_profile_emission,
    isothermal_transmission,
    madhushudhan_seager_emission,
    madhushudhan_seager_transmission,
    madhu_seager_patchy_transmission,
    power_law_profile_emission,
    power_law_profile_transmission,
    molliere_2020_emission,
)
from petitRADTRANS.retrieval.runtime import ParameterLayout, RetrievalRuntime
from petitRADTRANS.retrieval.runtime import (
    LikelihoodState,
    ModelContext,
    ModelGroupState,
    ModelInputs,
    ModelResult,
    ObservationState,
    PhysicalParams,
    compute_diagonal_observation_log_likelihood,
    project_model_to_observation_space,
)
from petitRADTRANS.retrieval.sampler import (
    BlackJAXNUTSSampler,
    BlackJAXSamplingResults,
    JAXNSSampler,
    JAXNSShardedStaticNestedSampler,
    NumPyroNUTSSampler,
    NumPyroSamplingResults,
    NumPyroSampler,
    UltranestSampler,
)
from petitRADTRANS.retrieval.utils import gaussian_prior

from .context import petitRADTRANS
from .utils import tests_results_directory, reference_filenames, test_parameters

# TODO MultiNest's random generator is computer-dependent (on Mac only?), the tolerance is temporarily increased to match this until a fix is found  # noqa E501
relative_tolerance = 1e0  # relative tolerance when comparing with older results


class _DummyDistribution:
    name = "dummy"
    parameters = {}

    @staticmethod
    def quantile(x):
        return 10.0 + 2.0 * x


class _DummyRadtrans:
    def __init__(self):
        self.pressures = jnp.logspace(-6, 2, 16) * 1e6
        self.line_species = ("H2O",)
        self.cloud_species = ("MgSiO3(c)",)
        self.calculate_flux_kwargs = None
        self.calculate_transit_radii_kwargs = None

    def calculate_flux(self, **kwargs):
        self.calculate_flux_kwargs = kwargs
        return (
            jnp.array([1.0e-4, 2.0e-4]),
            jnp.array([2.0, 4.0]),
            {"source": "emission"},
        )

    def calculate_transit_radii(self, **kwargs):
        self.calculate_transit_radii_kwargs = kwargs
        return (
            jnp.array([1.0e-4, 2.0e-4]),
            jnp.array([3.0, 6.0]),
            {"source": "transmission"},
        )


class _RecordingUltranestSampler:
    last_instance = None

    def __init__(
        self,
        log_likelihood_func,
        prior_func,
        output_directory,
        retrieval_name,
        enable_step_sampling=False,
    ):
        del output_directory, retrieval_name, enable_step_sampling
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.run_kwargs = None
        type(self).last_instance = self

    @classmethod
    def prepare(cls, context, **kwargs):
        vectorized = UltranestSampler._resolve_vectorized(
            context, kwargs.pop("vectorized", False)
        )

        if vectorized:
            runtime = context.runtime
            uncertainties_mode = context.uncertainties_mode
            parameter_layout = context.parameter_layout

            def log_likelihood_batch(physical_positions):
                return np.asarray(
                    runtime.evaluate_jax_batched(
                        free_parameter_values=physical_positions,
                        uncertainties_mode=uncertainties_mode,
                        print_log_likelihood_for_debugging=False,
                    )
                )

            def prior_transform(cube):
                return parameter_layout.prior_transform_numpy(cube)

            log_likelihood_func = log_likelihood_batch
            prior_func = prior_transform
        else:
            log_likelihood_func = context.log_likelihood_func
            prior_func = context.prior_ultranest_func

        instance = cls(
            log_likelihood_func=log_likelihood_func,
            prior_func=prior_func,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        run_kwargs = {"vectorized": vectorized}
        run_kwargs.update(kwargs)
        return instance, run_kwargs, {}

    def run_sampler(self, **kwargs):
        self.run_kwargs = kwargs
        return {"status": "ok", "run_kwargs": kwargs}

    def post_run(self, results):
        return results, {}

    def pretty_print_results(self):
        return None


class _RecordingPosteriorSampler:
    def __init__(self, samples=None, log_likelihood=None, summary_parameters=None):
        if samples is None:
            samples = np.array([[1.0], [3.0], [2.0]])
        if log_likelihood is None:
            log_likelihood = np.array([-3.0, -1.0, -2.0])
        if summary_parameters is None:
            summary_parameters = {"num_live_points": 32}

        self.results = SimpleNamespace(
            samples=np.asarray(samples, dtype=float),
            model_log_likelihood=np.asarray(log_likelihood, dtype=float),
        )
        self.summary_parameters = summary_parameters

    def run_sampler(self, **kwargs):
        return self.results

    def get_results(self):
        return self.results

    def get_summary(self, free_parameter_names=None):
        parameter_name = (
            free_parameter_names[0] if free_parameter_names else "parameter_0"
        )
        return {
            "sampler": self.__class__.__name__,
            "parameters": dict(self.summary_parameters),
            "diagnostics": {"num_samples": int(self.results.samples.shape[0])},
            "evidence": {"ln_z": 1.5, "ln_z_err": 0.2},
            "posterior": [
                {
                    "name": parameter_name,
                    "median": 2.0,
                    "lower_1sigma": 1.0,
                    "upper_1sigma": 3.0,
                },
            ],
        }


class _EvaluateLoadingSampler:
    last_instance = None
    results_to_load = {"status": "loaded"}

    def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
        self.log_likelihood_func = log_likelihood_func
        self.prior_func = prior_func
        self.output_directory = output_directory
        self.retrieval_name = retrieval_name
        self.load_results_calls = 0
        self.run_sampler_calls = 0
        type(self).last_instance = self

    def load_results(self):
        self.load_results_calls += 1
        return type(self).results_to_load

    def run_sampler(self, **kwargs):
        del kwargs
        self.run_sampler_calls += 1
        raise AssertionError("evaluate mode must not start sampling")


class _MissingEvaluateSampler:
    def __init__(self):
        self.load_results_calls = 0
        self.run_sampler_calls = 0

    def load_results(self):
        self.load_results_calls += 1
        return None

    def run_sampler(self, **kwargs):
        del kwargs
        self.run_sampler_calls += 1
        raise AssertionError("evaluate mode must not start sampling")


class _JAXNSLikeResults:
    def __init__(self):
        self.parametrised_samples = {"amplitude": np.array([1.0, 3.0, 2.0])}
        self.log_L_samples = np.array([-3.0, -1.0, -2.0])
        self.log_Z_mean = 1.5
        self.log_Z_uncert = 0.2


class _JAXNSSamplesOnlyResults:
    def __init__(self):
        self.samples = {"amplitude": np.array([1.0, 3.0, 2.0])}
        self.parametrised_samples = {}
        self.log_L_samples = np.array([-3.0, -1.0, -2.0])
        self.log_dp_mean = np.array([-0.3, -0.2, -0.1])
        self.log_Z_mean = 1.5
        self.log_Z_uncert = 0.2
        self.ESS = 3.0
        self.H_mean = 0.1
        self.total_num_samples = 3
        self.total_num_likelihood_evaluations = 3
        self.log_efficiency = -1.0


def _build_runtime(
    configuration, model_function, source_name="simple", observation_names=None
):
    if observation_names is None:
        observation_names = (source_name,)

    parameter_layout = ParameterLayout.from_parameters(configuration.parameters)

    observations = {}
    likelihoods = {}
    for name in observation_names:
        data = configuration.data[name]
        observations[name] = data.to_observation_state(model_contract="differentiable")
        likelihoods[name] = data.to_likelihood_state(
            n_free_parameters=parameter_layout.n_free_parameters
        )

    source_data = configuration.data[source_name]
    return RetrievalRuntime(
        parameter_layout=parameter_layout,
        observations=observations,
        likelihoods=likelihoods,
        model_groups=(
            ModelGroupState(
                source_name=source_name,
                observation_names=tuple(observation_names),
                model_contract="differentiable",
                model_generating_function=model_function,
                radtrans_object=source_data.radtrans_object,
                metadata={
                    "line_opacity_mode": source_data.line_opacity_mode,
                    "has_external_references": len(observation_names) > 1,
                },
            ),
        ),
    )


def _mark_differentiable_model_function(model_function):
    setattr(model_function, "_prt_model_contract", MODEL_CONTRACT_DIFFERENTIABLE)
    return model_function


def _seed_retrieval_mpi_state(retrieval):
    retrieval.comm = None
    retrieval.rank = 0
    return retrieval


def test_parameter_layout_supports_ordered_prior_transforms_and_legacy_adapter():
    parameters = {
        "transform_parameter": Parameter(
            name="transform_parameter",
            is_free_parameter=True,
            transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
        ),
        "distribution_parameter": Parameter(
            name="distribution_parameter",
            is_free_parameter=True,
            distribution=_DummyDistribution(),
        ),
        "fixed_string_parameter": Parameter(
            name="fixed_string_parameter",
            is_free_parameter=False,
            value="runtime-static",
        ),
        "fixed_numeric_parameter": Parameter(
            name="fixed_numeric_parameter",
            is_free_parameter=False,
            value=7.0,
        ),
    }

    layout = ParameterLayout.from_parameters(parameters)

    assert layout.free_parameter_names == (
        "transform_parameter",
        "distribution_parameter",
    )
    np.testing.assert_array_equal(layout.free_parameter_indices, np.array([0, 1]))
    np.testing.assert_allclose(
        layout.prior_transform_numpy(np.array([0.25, 0.75])), np.array([1.75, 11.5])
    )

    physical_params = layout.physical_params_from_vector(np.array([3.5, 4.5]))
    assert physical_params.ordered_names == (
        "transform_parameter",
        "distribution_parameter",
        "fixed_string_parameter",
        "fixed_numeric_parameter",
    )
    assert physical_params.get("fixed_string_parameter") == "runtime-static"
    assert physical_params.get("fixed_numeric_parameter") == 7.0

    legacy_parameters = layout.build_legacy_parameter_dict(physical_params)
    assert legacy_parameters["transform_parameter"].value == 3.5
    assert legacy_parameters["distribution_parameter"].value == 4.5
    assert legacy_parameters["fixed_string_parameter"].value == "runtime-static"
    assert legacy_parameters["fixed_numeric_parameter"].value == 7.0


def test_parameter_layout_prior_transform_accepts_ctypes_pointer_inputs():
    parameters = {
        "transform_parameter": Parameter(
            name="transform_parameter",
            is_free_parameter=True,
            transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
        ),
        "distribution_parameter": Parameter(
            name="distribution_parameter",
            is_free_parameter=True,
            distribution=_DummyDistribution(),
        ),
    }
    layout = ParameterLayout.from_parameters(parameters)

    cube_array = (ctypes.c_double * 2)(0.25, 0.75)
    cube_pointer = ctypes.cast(cube_array, ctypes.POINTER(ctypes.c_double))

    np.testing.assert_allclose(
        layout.prior_transform_numpy(cube_pointer), np.array([1.75, 11.5])
    )

    layout.prior_transform_inplace(cube_pointer)

    np.testing.assert_allclose(
        np.array([cube_pointer[0], cube_pointer[1]]), np.array([1.75, 11.5])
    )


def test_parameter_layout_supports_unconstrained_transforms_and_log_jacobian():
    layout = ParameterLayout.from_parameters(
        {
            "transform_parameter": Parameter(
                name="transform_parameter",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 2.0 * x - 1.0,
            ),
            "distribution_parameter": Parameter(
                name="distribution_parameter",
                is_free_parameter=True,
                distribution=_DummyDistribution(),
            ),
        }
    )

    unconstrained_position = jnp.array([0.0, jnp.log(3.0)], dtype=jnp.float64)
    cube_position = layout.cube_from_unconstrained(unconstrained_position)
    transformed = layout.transform_unconstrained(unconstrained_position)
    expected_cube_position = jnp.array([0.5, 0.75], dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(cube_position), np.asarray(expected_cube_position)
    )
    np.testing.assert_allclose(np.asarray(transformed), np.array([0.0, 11.5]))

    expected_log_jacobian = np.sum(
        -jax.nn.softplus(-np.asarray(unconstrained_position))
        - jax.nn.softplus(np.asarray(unconstrained_position))
    )
    np.testing.assert_allclose(
        layout.log_jacobian_single(unconstrained_position), expected_log_jacobian
    )


def test_data_materializes_observation_and_likelihood_state():
    data = Data(
        name="test_dataset",
        model_generating_function=lambda *args, **kwargs: None,
        radtrans_object=SimpleNamespace(name="radtrans"),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
        mask=jnp.array([False, True]),
        scale_flux=True,
        fit_flux_offset=True,
    )

    observation_state = data.to_observation_state(model_contract="differentiable")
    likelihood_state = data.to_likelihood_state(n_free_parameters=3)

    assert observation_state.name == "test_dataset"
    assert observation_state.model_contract == "differentiable"
    assert observation_state.model_generating_function is data.model_generating_function
    assert observation_state.scale_flux is True
    np.testing.assert_array_equal(observation_state.mask, jnp.array([False, True]))

    assert likelihood_state.n_free_parameters == 3
    np.testing.assert_array_equal(
        likelihood_state.data_uncertainties, jnp.array([0.01, 0.02])
    )
    np.testing.assert_array_equal(likelihood_state.mask, jnp.array([False, True]))
    assert likelihood_state.metadata["has_covariance"] is False


def test_runtime_scores_shared_grid_time_series_by_projecting_model_times_first():
    observation_times = jnp.array([0.0, 0.5, 1.5, 2.0])
    observation_wavelengths = jnp.array([1.5, 2.5])
    uncertainties = jnp.full((4, 2), 0.1)
    mask = jnp.zeros((4, 2), dtype=bool)

    model_times = jnp.array([0.0, 1.0, 2.0])
    model_wavelengths = jnp.array([1.0, 2.0, 3.0])
    model_spectra = jnp.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
            [4.0, 6.0, 8.0],
        ]
    )

    interpolated_model = jax.vmap(
        lambda wavelength_bin: jnp.interp(observation_times, model_times, wavelength_bin),
        in_axes=1,
        out_axes=1,
    )(model_spectra)

    reference_observation = ObservationState(
        name="time_series",
        wavelengths=observation_wavelengths,
        spectrum=jnp.zeros((4, 2)),
        mask=mask,
        uncertainties=uncertainties,
        metadata={"observation_times": observation_times},
    )
    projected_reference = jax.vmap(
        lambda epoch_spectrum: project_model_to_observation_space(
            model_wavelengths,
            epoch_spectrum,
            reference_observation,
            {},
            "time_series",
        )
    )(interpolated_model)

    observation = ObservationState(
        name="time_series",
        wavelengths=observation_wavelengths,
        spectrum=projected_reference,
        mask=mask,
        uncertainties=uncertainties,
        metadata={"observation_times": observation_times},
    )
    likelihood = LikelihoodState(
        data_uncertainties=uncertainties,
        n_free_parameters=0,
    )
    runtime = RetrievalRuntime(
        parameter_layout=SimpleNamespace(n_free_parameters=0),
        observations={"time_series": observation},
        likelihoods={"time_series": likelihood},
    )

    reference_ll = compute_diagonal_observation_log_likelihood(
        projected_reference,
        observation.spectrum,
        observation.uncertainties,
        mask=observation.mask,
    )
    optimized_ll = runtime._score_dataset(
        "time_series",
        {},
        model_wavelengths,
        jnp.full_like(interpolated_model, -10.0),
        0.0,
        model_auxiliary_outputs={
            "model_times": model_times,
            "model_spectra": model_spectra,
        },
    )

    np.testing.assert_allclose(
        np.asarray(optimized_ll),
        np.asarray(reference_ll),
        rtol=1e-8,
        atol=1e-8,
    )


def test_data_loadfits_accepts_structured_spectrum_tables(tmp_path):
    spectrum_path = tmp_path / "structured_spectrum.fits"
    wavelengths = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    fluxes = np.array([2.0, 2.5, 3.0], dtype=np.float64)
    covariance = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.09],
        ],
        dtype=np.float64,
    )

    table_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVELENGTH", format="D", array=wavelengths),
            fits.Column(name="FLUX", format="D", array=fluxes),
            fits.Column(name="COVARIANCE", format="3D", array=covariance),
        ],
        name="SPECTRUM",
    )
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(spectrum_path)

    data = Data(
        name="structured_dataset",
        path_to_observations=str(spectrum_path),
        model_generating_function=lambda *args, **kwargs: None,
    )

    np.testing.assert_allclose(np.asarray(data.wavelengths), wavelengths)
    np.testing.assert_allclose(np.asarray(data.spectrum), fluxes)
    np.testing.assert_allclose(np.asarray(data.covariance), covariance)
    np.testing.assert_allclose(
        np.asarray(data.uncertainties), np.sqrt(np.diag(covariance))
    )


def test_retrieval_runtime_groups_external_radtrans_references():
    source_data = Data(
        name="source",
        model_generating_function=lambda *args, **kwargs: None,
        radtrans_object=SimpleNamespace(name="source_radtrans"),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
    )
    shared_data = Data(
        name="shared",
        model_generating_function=lambda *args, **kwargs: None,
        radtrans_object=SimpleNamespace(name="shared_radtrans"),
        external_radtrans_reference="source",
        wavelengths=jnp.array([1.5, 2.5]),
        spectrum=jnp.array([0.3, 0.4]),
        uncertainties=jnp.array([0.03, 0.04]),
    )
    independent_data = Data(
        name="independent",
        model_generating_function=lambda *args, **kwargs: None,
        radtrans_object=SimpleNamespace(name="independent_radtrans"),
        wavelengths=jnp.array([3.0, 4.0]),
        spectrum=jnp.array([0.5, 0.6]),
        uncertainties=jnp.array([0.05, 0.06]),
    )

    config = SimpleNamespace(
        parameters={
            "free_parameter": Parameter(name="free_parameter", is_free_parameter=True),
        },
        data={
            "source": source_data,
            "shared": shared_data,
            "independent": independent_data,
        },
    )

    runtime = RetrievalRuntime.from_configuration(config)

    assert set(runtime.observations.keys()) == {"source", "shared", "independent"}
    assert set(runtime.likelihoods.keys()) == {"source", "shared", "independent"}
    assert len(runtime.model_groups) == 2

    groups_by_source = {group.source_name: group for group in runtime.model_groups}
    assert set(groups_by_source["source"].observation_names) == {"source", "shared"}
    assert groups_by_source["source"].metadata["has_external_references"] is True
    assert groups_by_source["source"].radtrans_object is source_data.radtrans_object
    assert groups_by_source["source"].model_contract == MODEL_CONTRACT_LEGACY_ADAPTER
    assert (
        groups_by_source["independent"].model_contract == MODEL_CONTRACT_LEGACY_ADAPTER
    )
    assert groups_by_source["independent"].observation_names == ("independent",)


def test_retrieval_runtime_initializes_missing_radtrans_objects(monkeypatch):
    initialized = []

    def fake_initialize_radtrans_objects(self, scaling=10, width=3):
        initialized.append((scaling, width))

        for name, data in self.configuration.data.items():
            if (
                data.external_radtrans_reference is None
                and data.radtrans_object is None
            ):
                data.radtrans_object = SimpleNamespace(name=f"{name}_radtrans")

    monkeypatch.setattr(
        petitRADTRANS.retrieval.retrieval.Retrieval,
        "initialise_radtrans_objects",
        fake_initialize_radtrans_objects,
    )

    source_data = Data(
        name="source",
        model_generating_function=lambda *args, **kwargs: None,
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([0.1, 0.2]),
        uncertainties=jnp.array([0.01, 0.02]),
        wavelength_boundaries=(0.95, 2.05),
    )
    shared_data = Data(
        name="shared",
        model_generating_function=lambda *args, **kwargs: None,
        external_radtrans_reference="source",
        wavelengths=jnp.array([1.5, 2.5]),
        spectrum=jnp.array([0.3, 0.4]),
        uncertainties=jnp.array([0.03, 0.04]),
        wavelength_boundaries=(1.45, 2.55),
    )

    config = SimpleNamespace(
        parameters={
            "free_parameter": Parameter(name="free_parameter", is_free_parameter=True),
        },
        data={
            "source": source_data,
            "shared": shared_data,
        },
        adaptive_mesh_refinement=False,
    )

    runtime = RetrievalRuntime.from_configuration(config)

    assert initialized == [(10, 3)]
    assert source_data.radtrans_object is not None
    assert shared_data.radtrans_object is None
    assert runtime.model_groups[0].source_name == "source"
    assert runtime.model_groups[0].radtrans_object is source_data.radtrans_object


def test_runtime_construction_accepts_explicit_differentiable_model_functions():
    def model_generating_function(context, params, pt_plot_mode=False):
        del context, pt_plot_mode
        amplitude = params.get("amplitude")
        return ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array([amplitude, amplitude + 1.0]),
        )

    _mark_differentiable_model_function(model_generating_function)

    config = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0),
        },
        data={
            "simple": Data(
                name="simple",
                model_generating_function=model_generating_function,
                radtrans_object=SimpleNamespace(),
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([0.5, 0.5]),
            ),
        },
        adaptive_mesh_refinement=False,
    )

    runtime = RetrievalRuntime.from_configuration(config)

    assert runtime.model_groups[0].model_contract == MODEL_CONTRACT_DIFFERENTIABLE
    assert (
        runtime.model_groups[0].model_generating_function is model_generating_function
    )


def test_calculate_forward_model_accepts_model_functions():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )

    def model_generating_function(context, params, pt_plot_mode=False):
        del context, pt_plot_mode
        amplitude = params.get("amplitude")
        return ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array([amplitude, amplitude + 1.0]),
        )

    _mark_differentiable_model_function(model_generating_function)

    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={
            "simple": Data(
                name="simple",
                model_generating_function=model_generating_function,
                radtrans_object=SimpleNamespace(),
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([0.5, 0.5]),
            ),
        },
        adaptive_mesh_refinement=False,
    )

    wavelengths, spectrum = retrieval.calculate_forward_model(
        parameters={"amplitude": 2.0},
        data="simple",
    )

    np.testing.assert_array_equal(wavelengths, jnp.array([1.0, 2.0]))
    np.testing.assert_array_equal(spectrum, jnp.array([2.0, 3.0]))


def test_run_legacy_model_adapter_normalizes_legacy_outputs():
    model_context = ModelContext(
        name="legacy_model",
        mode="emission",
        radtrans=SimpleNamespace(),
        adaptive_mesh_refinement=False,
    )
    physical_params = PhysicalParams.from_mapping({"alpha": 1.2, "label": "fixed"})

    def legacy_model(
        prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        assert parameters["alpha"].value == 1.2
        assert parameters["label"].value == "fixed"
        return jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), 0.5

    result = run_legacy_model_adapter(legacy_model, model_context, physical_params)

    assert result.kind == "spectrum"
    np.testing.assert_array_equal(result.wavelengths, jnp.array([1.0, 2.0]))
    np.testing.assert_array_equal(result.spectrum, jnp.array([3.0, 4.0]))
    assert result.additional_log_likelihood == 0.5


def test_run_runtime_model_adapter_accepts_model_result():
    model_context = ModelContext(
        name="runtime_model",
        mode="transmission",
        radtrans=SimpleNamespace(),
        adaptive_mesh_refinement=False,
    )
    physical_params = PhysicalParams.from_mapping({"alpha": 1.2})

    def runtime_model(context, params, pt_plot_mode=False):
        assert context.name == "runtime_model"
        assert params.get("alpha") == 1.2
        return ModelResult(
            kind="spectrum", wavelengths=jnp.array([1.0]), spectrum=jnp.array([2.0])
        )

    result = run_runtime_model_adapter(runtime_model, model_context, physical_params)

    assert result.kind == "spectrum"
    np.testing.assert_array_equal(result.wavelengths, jnp.array([1.0]))
    np.testing.assert_array_equal(result.spectrum, jnp.array([2.0]))


def test_calculate_emission_spectrum_runtime_returns_model_result():
    radtrans = _DummyRadtrans()
    model_context = ModelContext(
        name="emission",
        mode="emission",
        radtrans=radtrans,
        adaptive_mesh_refinement=False,
    )
    model_inputs = ModelInputs(
        pressures=jnp.array([1.0]),
        temperatures=jnp.array([1000.0]),
        abundances={"H2O": jnp.array([1e-3])},
        mean_molar_masses=jnp.array([2.3]),
        gravity=jnp.array(1e3),
        planet_radius=jnp.array(2.0),
        distance_to_system=jnp.array(4.0),
    )

    result = calculate_emission_spectrum_runtime(model_context, model_inputs)

    assert result.kind == "spectrum"
    np.testing.assert_allclose(result.wavelengths, jnp.array([1.0, 2.0]))
    assert result.auxiliary_outputs["source"] == "emission"
    np.testing.assert_allclose(
        np.asarray(radtrans.calculate_flux_kwargs["pressures"]),
        np.asarray(model_inputs.pressures),
    )


def test_calculate_transmission_spectrum_runtime_returns_model_result():
    radtrans = _DummyRadtrans()
    model_context = ModelContext(
        name="transmission",
        mode="transmission",
        radtrans=radtrans,
        adaptive_mesh_refinement=False,
        return_contribution=True,
    )
    model_inputs = ModelInputs(
        pressures=jnp.array([1.0]),
        temperatures=jnp.array([1000.0]),
        abundances={"H2O": jnp.array([1e-3])},
        mean_molar_masses=jnp.array([2.3]),
        gravity=jnp.array(1e3),
        planet_radius=jnp.array(2.0),
        stellar_radius=jnp.array(2.0),
    )

    result = calculate_transmission_spectrum_runtime(model_context, model_inputs)

    assert result.kind == "spectrum"
    np.testing.assert_allclose(result.wavelengths, jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(result.spectrum, jnp.array([2.25, 9.0]))
    assert result.auxiliary_outputs["source"] == "transmission"
    np.testing.assert_allclose(
        np.asarray(radtrans.calculate_transit_radii_kwargs["pressures"]),
        np.asarray(model_inputs.pressures),
    )


def test_resolve_runtime_abundance_state_rebuilds_amr_grid_without_mutating_radtrans(
    monkeypatch,
):
    radtrans = _DummyRadtrans()
    base_pressures = radtrans.pressures * 1e-6
    base_temperatures = jnp.linspace(900.0, 1200.0, base_pressures.shape[0])
    model_context = ModelContext(
        name="adaptive",
        mode="emission",
        radtrans=radtrans,
        adaptive_mesh_refinement=True,
    )
    physical_params = PhysicalParams.from_mapping({})

    get_abundances_calls = []
    cloud_base_resolver_calls = []
    temperature_builder_calls = []

    def fake_get_abundances(
        pressures,
        temperatures,
        parameters,
        line_species,
        cloud_species,
        adaptive_mesh_refinement,
    ):
        del parameters, line_species, cloud_species
        get_abundances_calls.append(
            {
                "pressures": np.asarray(pressures),
                "temperatures": np.asarray(temperatures),
                "adaptive_mesh_refinement": adaptive_mesh_refinement,
            }
        )
        return (
            {"H2O": jnp.full_like(pressures, 1e-3)},
            jnp.full_like(pressures, 2.3),
            jnp.arange(pressures.shape[0]),
            {"MgSiO3(c)": jnp.asarray(1e-2)},
        )

    def temperature_builder(pressures):
        temperature_builder_calls.append(np.asarray(pressures))
        return 900.0 + 30.0 * jnp.log10(pressures / pressures[0])

    def fake_resolve_cloud_base_pressures(
        pressures,
        temperatures,
        parameters,
        line_species,
        cloud_species,
    ):
        del parameters, line_species, cloud_species
        cloud_base_resolver_calls.append(
            {
                "pressures": np.asarray(pressures),
                "temperatures": np.asarray(temperatures),
            }
        )
        return {"MgSiO3(c)": jnp.asarray(1e-2)}

    monkeypatch.setattr(
        petitRADTRANS.retrieval.models,
        "get_abundances",
        fake_get_abundances,
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.models,
        "resolve_cloud_base_pressures",
        fake_resolve_cloud_base_pressures,
    )

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        base_pressures,
        base_pressures,
        base_temperatures,
        temperature_builder=temperature_builder,
    )

    expected_pressures = build_adaptive_pressure_grid(
        base_pressures,
        jnp.array([1e-2]),
    )

    assert result is None
    assert len(cloud_base_resolver_calls) == 1
    assert len(get_abundances_calls) == 1
    assert len(temperature_builder_calls) == 1
    assert get_abundances_calls[0]["adaptive_mesh_refinement"] is False
    np.testing.assert_allclose(
        np.asarray(temperature_builder_calls[0]),
        np.asarray(expected_pressures),
    )
    np.testing.assert_allclose(
        np.asarray(abundance_state["pressures"]),
        np.asarray(expected_pressures),
    )
    np.testing.assert_allclose(
        np.asarray(get_abundances_calls[0]["pressures"]),
        np.asarray(expected_pressures),
    )
    np.testing.assert_allclose(
        np.asarray(cloud_base_resolver_calls[0]["pressures"]),
        np.asarray(base_pressures),
    )
    np.testing.assert_allclose(
        np.asarray(cloud_base_resolver_calls[0]["temperatures"]),
        np.asarray(base_temperatures),
    )
    np.testing.assert_allclose(
        np.asarray(radtrans.pressures),
        np.asarray(base_pressures * 1e6),
    )


def test_initialize_pressure_returns_base_grid_when_legacy_amr_is_requested():
    pressures = jnp.logspace(-6, 2, 16)

    p_use, p_global = initialize_pressure(
        pressures,
        {"pressure_scaling": 10, "pressure_simple": pressures.shape[0]},
        adaptive_mesh_refinement=True,
    )

    np.testing.assert_allclose(np.asarray(p_use), np.asarray(pressures))
    np.testing.assert_allclose(np.asarray(p_global), np.asarray(pressures))


def test_retrieval_config_setup_pres_returns_base_grid_without_legacy_amr_parameters():
    config = petitRADTRANS.retrieval.RetrievalConfig(
        adaptive_mesh_refinement=True,
        pressures=jnp.logspace(-6, 2, 16),
    )

    returned_pressures = config._setup_pres(scaling=10, width=3)

    np.testing.assert_allclose(
        np.asarray(returned_pressures),
        np.asarray(config.pressures),
    )
    assert "pressure_scaling" not in config.parameters
    assert "pressure_width" not in config.parameters
    assert "pressure_simple" not in config.parameters


def test_log_likelihood_routes_normal_scalar_path_through_runtime():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )

    def simple_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    data = Data(
        name="simple",
        model_generating_function=simple_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )

    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        retrieval.configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )

    runtime_value = retrieval.log_likelihood(np.array([2.0]))
    legacy_value = retrieval._log_likelihood_legacy(np.array([2.0]))

    np.testing.assert_allclose(runtime_value, legacy_value)


def test_log_likelihood_jax_routes_normal_path_through_runtime():
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )

    def simple_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    data = Data(
        name="simple",
        model_generating_function=simple_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
    )
    parameters = {
        "amplitude": Parameter(
            name="amplitude", is_free_parameter=True, value=jnp.array(2.0)
        )
    }

    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = configuration
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )

    runtime_value = retrieval.log_likelihood_jax(
        parameters=parameters,
        uncertainties_mode="default",
    )
    legacy_value = (
        petitRADTRANS.retrieval.retrieval.Retrieval._log_likelihood_jax_legacy(
            parameters=parameters,
            configuration=configuration,
            uncertainties_mode="default",
        )
    )

    np.testing.assert_allclose(runtime_value, legacy_value)


def test_prior_ultranest_supports_transform_and_distribution_priors():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = SimpleNamespace(
        parameters={
            "transform_parameter": Parameter(
                name="transform_parameter",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
            ),
            "distribution_parameter": Parameter(
                name="distribution_parameter",
                is_free_parameter=True,
                distribution=_DummyDistribution(),
            ),
            "fixed_parameter": Parameter(
                name="fixed_parameter",
                is_free_parameter=False,
                value=7.0,
            ),
        }
    )

    cube = np.array([0.25, 0.75], dtype=float)
    transformed = retrieval.prior_ultranest(cube)

    np.testing.assert_allclose(transformed, np.array([1.75, 11.5]))


def test_prior_ultranest_supports_vectorized_cube_transforms():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = SimpleNamespace(
        parameters={
            "transform_parameter": Parameter(
                name="transform_parameter",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
            ),
            "distribution_parameter": Parameter(
                name="distribution_parameter",
                is_free_parameter=True,
                distribution=_DummyDistribution(),
            ),
        }
    )

    cube = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=float)
    transformed = retrieval.prior_ultranest(cube)

    np.testing.assert_allclose(transformed, np.array([[1.75, 11.5], [2.5, 11.0]]))


def test_build_ultranest_sampler_interface_vectorized_routes_through_runtime():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )

    def simple_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    data = Data(
        name="simple",
        model_generating_function=simple_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )

    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        retrieval.configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )

    log_likelihood_batch, prior_transform = (
        retrieval._build_ultranest_sampler_interface(vectorized=True)
    )

    np.testing.assert_allclose(
        prior_transform(np.array([[0.25], [0.5]])), np.array([[1.0], [2.0]])
    )

    batched = log_likelihood_batch(np.array([[2.0], [1.0]], dtype=float))
    expected = np.array(
        [
            retrieval.log_likelihood(np.array([2.0])),
            retrieval.log_likelihood(np.array([1.0])),
        ]
    )
    np.testing.assert_allclose(batched, expected)


def test_build_blackjax_sampler_kwargs_adds_hmc_steps():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    kwargs = retrieval._build_blackjax_sampler_kwargs(
        {"num_integration_steps": 17, "seed": 7},
        free_parameter_names=["alpha"],
        initial_position=jnp.array([0.0]),
        log_jacobian=lambda position: position,
        algorithm_name="hmc",
    )

    assert kwargs["parameter_names"] == ["alpha"]
    assert kwargs["num_integration_steps"] == 17
    assert kwargs["seed"] == 7


def test_build_numpyro_sampler_kwargs_prefers_trajectory_length_for_hmc():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    kwargs = retrieval._build_numpyro_sampler_kwargs(
        {"trajectory_length": 3.5, "num_steps": 20},
        free_parameter_names=["alpha"],
        initial_position=jnp.array([0.0]),
        log_jacobian=lambda position: position,
        algorithm_name="hmc",
    )

    assert kwargs["parameter_names"] == ["alpha"]
    assert kwargs["trajectory_length"] == 3.5
    assert kwargs["num_steps"] == 20


def test_sampler_summary_helpers_build_expected_payloads():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    ultranest_summary = retrieval._build_ultranest_summary_parameters(
        sampler_type="ultranest",
        parameter_names=["alpha"],
        vectorized=True,
    )
    assert ultranest_summary == {
        "sampler_type": "ultranest",
        "parameter_names": ["alpha"],
        "vectorized": True,
    }

    blackjax_summary = retrieval._build_blackjax_summary_parameters(
        sampler_type="nuts",
        sampler_kwargs={
            "num_samples": 100,
            "num_warmup": 50,
            "initial_step_size": 0.1,
            "target_acceptance_rate": 0.9,
            "is_mass_matrix_diagonal": True,
            "seed": 7,
            "use_jit": True,
        },
    )
    assert blackjax_summary["sampler_type"] == "nuts"
    assert blackjax_summary["num_samples"] == 100

    numpyro_summary = retrieval._build_numpyro_summary_parameters(
        sampler_type="numpyrohmc",
        sampler_kwargs={
            "num_samples": 100,
            "num_warmup": 50,
            "initial_step_size": 0.1,
            "target_acceptance_rate": 0.9,
            "is_mass_matrix_diagonal": True,
            "num_chains": 2,
            "thinning": 1,
            "chain_method": "parallel",
            "seed": 7,
            "use_jit": True,
            "num_steps": 15,
        },
    )
    assert numpyro_summary["sampler_type"] == "numpyrohmc"
    assert numpyro_summary["num_integration_steps"] == 15

    jaxns_summary = retrieval._build_jaxns_summary_parameters(
        sampler_type="jaxns",
        sampler_kwargs={"num_live_points": 64},
        parameter_names=["alpha"],
    )
    assert jaxns_summary == {
        "num_live_points": 64,
        "sampler_type": "jaxns",
        "parameter_names": ["alpha"],
    }


def test_jaxns_sampler_builds_termination_condition_from_sampler_kwargs():
    sampler_kwargs = {
        "termination_condition_kwargs": {"ess": 256, "rtol": 1e-4},
        "dlogZ": 0.05,
        "termination_max_samples": 1024,
    }

    termination_condition, termination_summary = JAXNSSampler._build_termination_condition(
        sampler_kwargs
    )

    assert sampler_kwargs == {}
    assert termination_condition is not None
    assert termination_condition.ess == 256
    assert termination_condition.rtol == 1e-4
    assert termination_condition.dlogZ == 0.05
    assert termination_condition.max_samples == 1024
    assert termination_summary == {
        "ess": 256,
        "rtol": 1e-4,
        "dlogZ": 0.05,
        "max_samples": 1024,
    }


def test_jaxns_run_sampler_passes_termination_condition(monkeypatch, tmp_path):
    captured = {}

    class FakeNestedSampler:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["init_kwargs"] = dict(kwargs)

        def __call__(self, key, termination_condition):
            captured["key"] = key
            captured["termination_condition"] = termination_condition
            return "termination_reason", "state"

    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "Model",
        lambda prior_model, log_likelihood: {
            "prior_model": prior_model,
            "log_likelihood": log_likelihood,
        },
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "NestedSampler",
        FakeNestedSampler,
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "save_results",
        lambda results, save_file: captured.update(
            {"saved_results": results, "save_file": save_file}
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "clear_caches",
        lambda: None,
    )

    sampler = JAXNSSampler(
        log_likelihood_func=lambda *_args: 0.0,
        prior_func=lambda: (),
        output_directory=str(tmp_path),
        retrieval_name="jaxns_termination",
    )
    sampler._materialize_raw_results = (
        lambda termination_reason, state: {
            "termination_reason": termination_reason,
            "state": state,
        }
    )
    sampler._to_numpy_tree = lambda raw_results: raw_results
    sampler._build_equal_weight_results = lambda raw_results: {"samples": np.array([[1.0]])}
    sampler.pretty_print_results = lambda: None
    sampler.plot_diagnostics = lambda results=None: {}

    results = sampler.run_sampler(
        use_jit=False,
        num_live_points=32,
        max_samples=4096,
        seed=7,
        dlogZ=0.1,
        termination_max_samples=256,
    )

    assert captured["init_kwargs"]["num_live_points"] == 32
    assert captured["init_kwargs"]["max_samples"] == 4096
    assert captured["termination_condition"] is not None
    assert captured["termination_condition"].dlogZ == 0.1
    assert captured["termination_condition"].max_samples == 256
    assert results == {"samples": np.array([[1.0]])}


def test_jaxns_sharded_run_sampler_passes_termination_condition(monkeypatch, tmp_path):
    captured = {}

    class FakeShardedStaticNestedSampler:
        def __init__(self, model, sampler, **kwargs):
            captured["model"] = model
            captured["sampler"] = sampler
            captured["init_kwargs"] = dict(kwargs)

        def __call__(self, key, termination_condition):
            captured["key"] = key
            captured["termination_condition"] = termination_condition
            return "termination_reason", "state"

    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "Model",
        lambda prior_model, log_likelihood: {
            "prior_model": prior_model,
            "log_likelihood": log_likelihood,
        },
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "ShardedStaticNestedSampler",
        FakeShardedStaticNestedSampler,
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "UniformSampler",
        object(),
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "save_results",
        lambda results, save_file: captured.update(
            {"saved_results": results, "save_file": save_file}
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "clear_caches",
        lambda: None,
    )

    sampler = JAXNSShardedStaticNestedSampler(
        log_likelihood_func=lambda *_args: 0.0,
        prior_func=lambda: (),
        output_directory=str(tmp_path),
        retrieval_name="jaxns_sharded_termination",
    )
    sampler._materialize_raw_results = (
        lambda termination_reason, state: {
            "termination_reason": termination_reason,
            "state": state,
        }
    )
    sampler._to_numpy_tree = lambda raw_results: raw_results
    sampler._build_equal_weight_results = lambda raw_results: {"samples": np.array([[1.0]])}
    sampler.pretty_print_results = lambda: None
    sampler.plot_diagnostics = lambda results=None: {}

    results = sampler.run_sampler(
        use_jit=False,
        num_live_points=24,
        max_samples=2048,
        seed=11,
        termination_condition_kwargs={"evidence_uncert": 0.2, "atol": 1e-3},
    )

    assert captured["init_kwargs"]["num_live_points"] == 24
    assert captured["init_kwargs"]["max_samples"] == 2048
    assert captured["termination_condition"] is not None
    assert captured["termination_condition"].evidence_uncert == 0.2
    assert captured["termination_condition"].atol == 1e-3
    assert results == {"samples": np.array([[1.0]])}


def test_parameter_layout_free_parameter_ordering_matches_all_sampler_interfaces():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    parameters = {
        "alpha": Parameter(
            name="alpha",
            is_free_parameter=True,
            distribution=tfpd.Uniform(low=0.0, high=1.0),
        ),
        "beta": Parameter(
            name="beta",
            is_free_parameter=True,
            distribution=tfpd.Uniform(low=1.0, high=2.0),
        ),
        "fixed": Parameter(name="fixed", is_free_parameter=False, value=7.0),
    }

    data = Data(
        name="simple",
        model_generating_function=lambda context, params, pt_plot_mode=False: ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array(
                [
                    params.get("alpha") + params.get("beta"),
                    params.get("alpha") + params.get("beta"),
                ]
            ),
        ),
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 2.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    configuration = SimpleNamespace(
        parameters=parameters, data={"simple": data}, adaptive_mesh_refinement=False
    )

    retrieval.configuration = configuration
    retrieval.parameter_layout = ParameterLayout.from_parameters(parameters)
    retrieval.runtime = _build_runtime(
        configuration,
        model_function=data.model_generating_function,
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval.uncertainties_mode = "default"
    retrieval.generate_mock_data = False
    retrieval.print_log_likelihood_for_debugging = False

    pymultinest_names = tuple(retrieval._free_parameter_names_cache)
    ultranest_names = tuple(retrieval._free_parameter_names_cache)
    blackjax_names, *_ = retrieval._build_unconstrained_mcmc_interface()
    numpyro_names, *_ = retrieval._build_unconstrained_mcmc_interface()
    jaxns_names, *_ = retrieval._build_jaxns_sampler_interface()

    assert pymultinest_names == ("alpha", "beta")
    assert ultranest_names == pymultinest_names
    assert tuple(blackjax_names) == pymultinest_names
    assert tuple(numpyro_names) == pymultinest_names
    assert tuple(jaxns_names) == pymultinest_names


def test_run_legacy_model_adapter_reproduces_builtin_emission_model_output():
    model_context = ModelContext(
        name="legacy_emission_builtin",
        mode="emission",
        radtrans=_DummyRadtrans(),
        adaptive_mesh_refinement=False,
    )
    physical_params = PhysicalParams.from_mapping(
        {"distance_to_system": jnp.array(4.0)}
    )

    def legacy_emission_model(
        prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        return calculate_emission_spectrum(
            prt_object=prt_object,
            parameters=parameters,
            temperatures=jnp.array([1000.0]),
            abundances={"H2O": jnp.array([1e-3])},
            gravity=jnp.array(1e3),
            mean_molar_masses=jnp.array([2.3]),
            planet_radius=jnp.array(2.0),
            sigma_lnorm=None,
            cloud_particle_mean_radii=None,
            cloud_f_sed=None,
            eddy_diffusion_coefficients=None,
            cloud_hansen_b=None,
            cloud_fraction=None,
            patchy_clouds=None,
            distribution=None,
        )

    direct_output = legacy_emission_model(
        _DummyRadtrans(), {"distance_to_system": SimpleNamespace(value=jnp.array(4.0))}
    )
    adapter_result = run_legacy_model_adapter(
        legacy_emission_model, model_context, physical_params
    )

    np.testing.assert_allclose(adapter_result.wavelengths, direct_output[0])
    np.testing.assert_allclose(adapter_result.spectrum, direct_output[1])
    assert (
        adapter_result.auxiliary_outputs["legacy_output_2"]["source"]
        == direct_output[2]["source"]
    )


def test_run_legacy_model_adapter_reproduces_builtin_transmission_model_output():
    model_context = ModelContext(
        name="legacy_transmission_builtin",
        mode="transmission",
        radtrans=_DummyRadtrans(),
        adaptive_mesh_refinement=False,
    )
    physical_params = PhysicalParams.from_mapping({"stellar_radius": jnp.array(2.0)})

    def legacy_transmission_model(
        prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        return calculate_transmission_spectrum(
            prt_object=prt_object,
            parameters=parameters,
            temperatures=jnp.array([1000.0]),
            abundances={"H2O": jnp.array([1e-3])},
            gravity=jnp.array(1e3),
            mean_molar_masses=jnp.array([2.3]),
            planet_radius=jnp.array(2.0),
            reference_pressure=None,
            opaque_cloud_top_pressure=None,
            sigma_lnorm=None,
            cloud_particle_mean_radii=None,
            cloud_f_sed=None,
            eddy_diffusion_coefficients=None,
            haze_factor=None,
            power_law_opacity_coefficient=None,
            power_law_opacity_350nm=None,
            cloud_hansen_b=None,
            cloud_fraction=None,
            patchy_clouds=None,
            distribution=None,
        )

    direct_output = legacy_transmission_model(
        _DummyRadtrans(), {"stellar_radius": SimpleNamespace(value=jnp.array(2.0))}
    )
    adapter_result = run_legacy_model_adapter(
        legacy_transmission_model, model_context, physical_params
    )

    np.testing.assert_allclose(adapter_result.wavelengths, direct_output[0])
    np.testing.assert_allclose(adapter_result.spectrum, direct_output[1])
    assert (
        adapter_result.auxiliary_outputs["legacy_output_2"]["source"]
        == direct_output[2]["source"]
    )


def test_builtin_models_expose_stable_model_result_fields():
    emission_context = ModelContext(
        name="emission",
        mode="emission",
        radtrans=_DummyRadtrans(),
        adaptive_mesh_refinement=False,
    )
    transmission_context = ModelContext(
        name="transmission",
        mode="transmission",
        radtrans=_DummyRadtrans(),
        adaptive_mesh_refinement=False,
    )
    emission_inputs = ModelInputs(
        pressures=jnp.array([1.0]),
        temperatures=jnp.array([1000.0]),
        abundances={"H2O": jnp.array([1e-3])},
        mean_molar_masses=jnp.array([2.3]),
        gravity=jnp.array(1e3),
        planet_radius=jnp.array(2.0),
        distance_to_system=jnp.array(4.0),
    )
    transmission_inputs = ModelInputs(
        pressures=jnp.array([1.0]),
        temperatures=jnp.array([1000.0]),
        abundances={"H2O": jnp.array([1e-3])},
        mean_molar_masses=jnp.array([2.3]),
        gravity=jnp.array(1e3),
        planet_radius=jnp.array(2.0),
        stellar_radius=jnp.array(2.0),
    )

    emission_result = calculate_emission_spectrum_runtime(
        emission_context, emission_inputs
    )
    transmission_result = calculate_transmission_spectrum_runtime(
        transmission_context, transmission_inputs
    )

    for result, source_name in (
        (emission_result, "emission"),
        (transmission_result, "transmission"),
    ):
        assert isinstance(result, ModelResult)
        assert result.kind == "spectrum"
        assert result.additional_log_likelihood == 0.0
        assert result.pressures is None
        assert result.temperatures is None
        assert result.atmospheric_column_fluxes is None
        assert result.auxiliary_outputs == {"source": source_name}


def test_builtin_model_exports_are_detected_as_native():
    unified_models = {
        "molliere_2020_emission": molliere_2020_emission,
        "molliere_2020_two_column": molliere_2020_two_column,
        "guillot_emission": guillot_emission,
        "guillot_emission_add_gaussian_temperature": guillot_emission_add_gaussian_temperature,
        "guillot_patchy_emission": guillot_patchy_emission,
        "interpolated_profile_emission": interpolated_profile_emission,
        "gradient_profile_emission": gradient_profile_emission,
        "power_law_profile_emission": power_law_profile_emission,
        "guillot_transmission": guillot_transmission,
        "guillot_patchy_transmission": guillot_patchy_transmission,
        "madhushudhan_seager_emission": madhushudhan_seager_emission,
        "madhushudhan_seager_transmission": madhushudhan_seager_transmission,
        "madhu_seager_patchy_transmission": madhu_seager_patchy_transmission,
        "isothermal_transmission": isothermal_transmission,
        "power_law_profile_transmission": power_law_profile_transmission,
    }

    for name, model in unified_models.items():
        resolved_contract, coerced_model = coerce_model_generating_function(model)

        assert model.__name__ == name
        assert resolved_contract == MODEL_CONTRACT_DIFFERENTIABLE
        assert coerced_model is model


def test_isothermal_transmission_runs_through_runtime_helpers(monkeypatch):
    monkeypatch.setattr(
        petitRADTRANS.retrieval.models,
        "get_abundances",
        lambda **kwargs: (
            {"H2O": jnp.array([1e-3])},
            jnp.array([2.3]),
            jnp.array([0]),
            None,
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.models.clouds,
        "setup_clouds",
        lambda pressures, parameters, cloud_species: (
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            None,
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.retrieval.models.clouds,
        "setup_simple_clouds_hazes",
        lambda parameters: (None, None, None, None),
    )

    radtrans = _DummyRadtrans()
    radtrans.pressures = jnp.array([1e6])
    radtrans.line_species = ()
    radtrans.cloud_species = ()
    model_context = ModelContext(
        name="transmission",
        mode="transmission",
        radtrans=radtrans,
        adaptive_mesh_refinement=False,
    )
    physical_params = PhysicalParams.from_mapping(
        {
            "temperature": 900.0,
            "log_g": 3.0,
            "planet_radius": 2.0,
            "stellar_radius": 2.0,
        }
    )

    result = isothermal_transmission(model_context, physical_params)

    assert result.kind == "spectrum"
    np.testing.assert_allclose(result.wavelengths, jnp.array([1.0, 2.0]))
    assert result.auxiliary_outputs == {"source": "transmission"}


def test_parameter_layout_physical_params_from_vector_accepts_jax_tracers():
    parameters = {
        "transform_parameter": Parameter(
            name="transform_parameter",
            is_free_parameter=True,
            transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
        ),
        "distribution_parameter": Parameter(
            name="distribution_parameter",
            is_free_parameter=True,
            distribution=_DummyDistribution(),
        ),
        "fixed_numeric_parameter": Parameter(
            name="fixed_numeric_parameter",
            is_free_parameter=False,
            value=7.0,
        ),
    }
    layout = ParameterLayout.from_parameters(parameters)

    @jax.jit
    def extract_parameter_values(values):
        physical_params = layout.physical_params_from_vector(values)
        return jnp.stack(
            (
                jnp.asarray(physical_params["transform_parameter"], dtype=jnp.float64),
                jnp.asarray(physical_params["distribution_parameter"], dtype=jnp.float64),
                jnp.asarray(physical_params["fixed_numeric_parameter"], dtype=jnp.float64),
            )
        )

    np.testing.assert_allclose(
        np.asarray(extract_parameter_values(jnp.array([3.5, 4.5]))),
        np.array([3.5, 4.5, 7.0]),
    )


def test_runtime_evaluation_adds_runtime_model_log_prior_weight_once():
    base_model = lambda context, params, pt_plot_mode=False: ModelResult(
        kind="spectrum",
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
    )
    weighted_model = lambda context, params, pt_plot_mode=False: ModelResult(
        kind="spectrum",
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        auxiliary_outputs={"log_prior_weight": 1.25},
    )

    base_configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0),
            "log_prior_weight": Parameter(
                name="log_prior_weight", is_free_parameter=False, value=0.0
            ),
        },
        data={
            "source": Data(
                name="source",
                model_generating_function=base_model,
                radtrans_object=SimpleNamespace(),
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([1.0, 1.0]),
            ),
            "shared": Data(
                name="shared",
                model_generating_function=base_model,
                radtrans_object=SimpleNamespace(),
                external_radtrans_reference="source",
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([1.0, 1.0]),
            ),
        },
        adaptive_mesh_refinement=False,
    )
    configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0),
            "log_prior_weight": Parameter(
                name="log_prior_weight", is_free_parameter=False, value=0.0
            ),
        },
        data={
            "source": Data(
                name="source",
                model_generating_function=weighted_model,
                radtrans_object=SimpleNamespace(),
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([1.0, 1.0]),
            ),
            "shared": Data(
                name="shared",
                model_generating_function=base_model,
                radtrans_object=SimpleNamespace(),
                external_radtrans_reference="source",
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([1.0, 1.0]),
            ),
        },
        adaptive_mesh_refinement=False,
    )
    base_runtime = RetrievalRuntime.from_configuration(base_configuration)
    runtime = RetrievalRuntime.from_configuration(configuration)
    physical_params = ParameterLayout.from_parameters(
        configuration.parameters
    ).physical_params_from_vector(np.array([0.0]))

    base_log_likelihood = base_runtime.evaluate_scalar(physical_params)
    log_likelihood = runtime.evaluate_scalar(physical_params)

    assert np.isfinite(log_likelihood)
    np.testing.assert_allclose(log_likelihood - base_log_likelihood, 1.25)


def test_model_group_evaluation_is_reused_across_external_references():
    call_counter = {"count": 0}

    def shared_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        call_counter["count"] += 1
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    source_data = Data(
        name="source",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    shared_data = Data(
        name="shared",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        external_radtrans_reference="source",
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0)
        },
        data={"source": source_data, "shared": shared_data},
        adaptive_mesh_refinement=False,
    )
    runtime = RetrievalRuntime.from_configuration(configuration)
    physical_params = ParameterLayout.from_parameters(
        configuration.parameters
    ).physical_params_from_vector(np.array([2.0]))

    log_likelihood = runtime.evaluate_scalar(physical_params)

    assert np.isfinite(log_likelihood)
    assert call_counter["count"] == 1


def test_log_likelihood_routes_external_reference_setup_through_runtime():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    def shared_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    source_data = Data(
        name="source",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    shared_data = Data(
        name="shared",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        external_radtrans_reference="source",
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )

    retrieval.configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0)
        },
        data={"source": source_data, "shared": shared_data},
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        retrieval.configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )

    runtime_value = retrieval.log_likelihood(np.array([2.0]))
    legacy_value = retrieval._log_likelihood_legacy(np.array([2.0]))

    np.testing.assert_allclose(runtime_value, legacy_value)


def test_log_likelihood_jax_routes_external_reference_setup_through_runtime():
    def shared_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    source_data = Data(
        name="source",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    shared_data = Data(
        name="shared",
        model_generating_function=shared_model,
        radtrans_object=SimpleNamespace(),
        external_radtrans_reference="source",
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(name="amplitude", is_free_parameter=True, value=0.0)
        },
        data={"source": source_data, "shared": shared_data},
        adaptive_mesh_refinement=False,
    )
    parameters = {
        "amplitude": Parameter(
            name="amplitude", is_free_parameter=True, value=jnp.array(2.0)
        )
    }

    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = configuration
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )

    runtime_value = retrieval.log_likelihood_jax(
        parameters=parameters,
        uncertainties_mode="default",
    )
    legacy_value = (
        petitRADTRANS.retrieval.retrieval.Retrieval._log_likelihood_jax_legacy(
            parameters=parameters,
            configuration=configuration,
            uncertainties_mode="default",
        )
    )

    np.testing.assert_allclose(runtime_value, legacy_value)


def test_build_jaxns_sampler_interface_routes_through_runtime_jax_likelihood():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        distribution=tfpd.Uniform(low=0.0, high=4.0),
    )

    def simple_model(
        _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
    ):
        del pt_plot_mode, adaptive_mesh_refinement
        amplitude = parameters["amplitude"].value
        return jnp.array([1.0, 2.0]), jnp.array([amplitude, amplitude + 1.0])

    data = Data(
        name="simple",
        model_generating_function=simple_model,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )

    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = _build_runtime(
        retrieval.configuration,
        model_function=lambda context, params, pt_plot_mode=False: ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array(
                [params.get("amplitude"), params.get("amplitude") + 1.0]
            ),
        ),
    )
    retrieval.uncertainties_mode = "default"
    retrieval.generate_mock_data = False

    free_parameter_names, _prior_model, log_likelihood_wrapper = (
        retrieval._build_jaxns_sampler_interface()
    )

    assert free_parameter_names == ["amplitude"]
    expected = retrieval.log_likelihood_jax(
        parameters={
            "amplitude": Parameter(
                name="amplitude", is_free_parameter=True, value=jnp.array(2.0)
            )
        },
        uncertainties_mode="default",
    )
    np.testing.assert_allclose(log_likelihood_wrapper(jnp.array(2.0)), expected)


def test_build_jaxns_sampler_interface_rejects_non_distribution_priors():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    data = Data(
        name="simple",
        model_generating_function=lambda context, params, pt_plot_mode=False: ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array(
                [
                    params.get("transform_parameter"),
                    params.get("transform_parameter") + 1.0,
                ]
            ),
        ),
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={
            "transform_parameter": Parameter(
                name="transform_parameter",
                is_free_parameter=True,
                transform_prior_cube_coordinate=lambda x: 1.0 + 3.0 * x,
            ),
        },
        data={"simple": data},
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = _build_runtime(
        retrieval.configuration,
        model_function=data.model_generating_function,
    )

    with pytest.raises(ValueError, match="distribution-backed free parameters"):
        retrieval._build_jaxns_sampler_interface()


def test_jax_sampler_interfaces_fail_clearly_on_legacy_only_models():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = SimpleNamespace(
        parameters={
            "amplitude": Parameter(
                name="amplitude",
                is_free_parameter=True,
                distribution=tfpd.Uniform(low=0.0, high=4.0),
            ),
        },
        data={
            "simple": Data(
                name="simple",
                model_generating_function=lambda _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False: (
                    jnp.array([1.0, 2.0]),
                    jnp.array(
                        [
                            parameters["amplitude"].value,
                            parameters["amplitude"].value + 1.0,
                        ]
                    ),
                ),
                radtrans_object=SimpleNamespace(),
                wavelengths=jnp.array([1.0, 2.0]),
                spectrum=jnp.array([2.0, 3.0]),
                uncertainties=jnp.array([0.5, 0.5]),
            ),
        },
        adaptive_mesh_refinement=False,
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = RetrievalRuntime.from_configuration(
        retrieval.configuration,
        parameter_layout=retrieval.parameter_layout,
        n_free_parameters=retrieval.parameter_layout.n_free_parameters,
    )

    with pytest.raises(ValueError, match="legacy-only retrievals are not supported"):
        retrieval._build_unconstrained_mcmc_interface()

    with pytest.raises(ValueError, match="legacy-only retrievals are not supported"):
        retrieval._build_jaxns_sampler_interface()


def test_run_ultranest_vectorized_mode_stays_enabled_for_retrieval(
    monkeypatch,
):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )

    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )

    def model_generating_function(context, params, pt_plot_mode=False):
        del context, pt_plot_mode
        amplitude = params.get("amplitude")
        return ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array([amplitude, amplitude + 1.0]),
        )

    data = Data(
        name="simple",
        model_generating_function=model_generating_function,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="ultranest",
        retrieval_name="ultranest",
        run_mode="retrieval",
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval.runtime = _build_runtime(
        retrieval.configuration, model_generating_function
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )
    retrieval.output_directory = tests_results_directory
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None

    monkeypatch.setitem(
        petitRADTRANS.retrieval.retrieval.SAMPLER_DISPATCH,
        "ultranest",
        _RecordingUltranestSampler,
    )
    retrieval.generate_retrieval_summary = lambda *args, **kwargs: None

    retrieval.run(error_checking=False, vectorized=True)

    assert _RecordingUltranestSampler.last_instance.run_kwargs["vectorized"] is True
    batched = _RecordingUltranestSampler.last_instance.log_likelihood_func(
        np.array([[2.0], [1.0]], dtype=float)
    )
    np.testing.assert_allclose(
        batched,
        np.array(
            [
                retrieval.log_likelihood(np.array([2.0])),
                retrieval.log_likelihood(np.array([1.0])),
            ]
        ),
    )


@pytest.mark.parametrize(
    ("runtime", "warning_text"),
    [
        (
            "legacy",
            "legacy model groups",
        ),
        (
            "ragged",
            "ragged observation groups",
        ),
    ],
)
def test_ultranest_vectorized_mode_is_disabled_for_unsupported_runtime_groups(
    monkeypatch, runtime, warning_text
):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    data = Data(
        name="simple",
        model_generating_function=lambda _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False: (
            jnp.array([1.0, 2.0]),
            jnp.array(
                [parameters["amplitude"].value, parameters["amplitude"].value + 1.0]
            ),
        ),
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="ultranest",
        retrieval_name="ultranest_disabled",
        run_mode="retrieval",
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )
    retrieval.output_directory = tests_results_directory
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None

    if runtime == "legacy":
        retrieval.runtime = RetrievalRuntime.from_configuration(
            retrieval.configuration,
            parameter_layout=retrieval.parameter_layout,
            n_free_parameters=retrieval.parameter_layout.n_free_parameters,
        )
    else:
        retrieval.runtime = RetrievalRuntime(
            parameter_layout=retrieval.parameter_layout,
            observations={
                "simple": ObservationState(
                    name="simple",
                    wavelengths=jnp.array([1.0, 2.0]),
                    spectrum=np.array(
                        [np.array([2.0, 3.0]), np.array([1.5])], dtype=object
                    ),
                    mask=jnp.array([False, False]),
                    radtrans_object=SimpleNamespace(),
                    model_generating_function=lambda context, params, pt_plot_mode=False: ModelResult(
                        kind="spectrum",
                        wavelengths=jnp.array([1.0, 2.0]),
                        spectrum=jnp.array(
                            [params.get("amplitude"), params.get("amplitude") + 1.0]
                        ),
                    ),
                    model_contract="differentiable",
                ),
            },
            likelihoods={"simple": LikelihoodState(n_free_parameters=1)},
            model_groups=(
                ModelGroupState(
                    source_name="simple",
                    observation_names=("simple",),
                    model_contract="differentiable",
                    model_generating_function=lambda context, params, pt_plot_mode=False: ModelResult(
                        kind="spectrum",
                        wavelengths=jnp.array([1.0, 2.0]),
                        spectrum=jnp.array(
                            [params.get("amplitude"), params.get("amplitude") + 1.0]
                        ),
                    ),
                    radtrans_object=SimpleNamespace(),
                    metadata={},
                ),
            ),
        )

    monkeypatch.setitem(
        petitRADTRANS.retrieval.retrieval.SAMPLER_DISPATCH,
        "ultranest",
        _RecordingUltranestSampler,
    )
    retrieval.generate_retrieval_summary = lambda *args, **kwargs: None

    with pytest.warns(RuntimeWarning, match=warning_text):
        retrieval.run(error_checking=False, vectorized=True)

    assert _RecordingUltranestSampler.last_instance.run_kwargs["vectorized"] is False


def test_run_regenerates_summary_for_non_pymultinest_sampler(tmp_path):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    data = Data(
        name="simple",
        model_generating_function=lambda _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False: (
            jnp.array([1.0, 2.0]),
            jnp.array(
                [parameters["amplitude"].value, parameters["amplitude"].value + 1.0]
            ),
        ),
        radtrans_object=SimpleNamespace(line_species=(), cloud_species=()),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="blackjaxnuts",
        retrieval_name="non_pmn_summary",
        run_mode="retrieval",
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.ultranest = False
    retrieval.plotter = RetrievalPlotter(retrieval)

    os.makedirs(tmp_path / "evaluate_non_pmn_summary", exist_ok=True)

    sampler = _RecordingPosteriorSampler()
    retrieval.run(sampler_object=sampler, error_checking=False)

    summary_file = (
        tmp_path / "evaluate_non_pmn_summary" / "non_pmn_summary_ret_summary.txt"
    )
    assert summary_file.exists()

    summary_text = summary_file.read_text()
    assert "sampler = _RecordingPosteriorSampler" in summary_text
    assert "num_live_points = 32" in summary_text
    assert "amplitude" in summary_text


def test_run_evaluate_mode_loads_saved_results_without_sampling(monkeypatch, tmp_path):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    retrieval.configuration = SimpleNamespace(
        sampler_type="blackjaxnuts",
        retrieval_name="evaluate_only",
        run_mode="evaluate",
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.analyzer = None
    retrieval._check_errors = lambda: pytest.fail(
        "evaluate mode must not run error checks"
    )

    expected_results = {
        "samples": np.array([[1.0]], dtype=float),
        "log_likelihood": np.array([-1.0], dtype=float),
    }
    _EvaluateLoadingSampler.results_to_load = expected_results
    monkeypatch.setattr(
        petitRADTRANS.retrieval.retrieval,
        "_resolve_sampler_class",
        lambda _sampler_type: _EvaluateLoadingSampler,
    )

    results = retrieval.run(error_checking=True)

    assert results is expected_results
    assert retrieval.sampler is _EvaluateLoadingSampler.last_instance
    assert retrieval.latest_sampler_results is expected_results
    assert _EvaluateLoadingSampler.last_instance.load_results_calls == 1
    assert _EvaluateLoadingSampler.last_instance.run_sampler_calls == 0


def test_run_evaluate_mode_raises_when_saved_results_are_missing(tmp_path):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    retrieval.configuration = SimpleNamespace(
        sampler_type="blackjaxnuts",
        retrieval_name="evaluate_missing",
        run_mode="evaluate",
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.analyzer = None
    retrieval._check_errors = lambda: pytest.fail(
        "evaluate mode must not run error checks"
    )

    sampler = _MissingEvaluateSampler()

    with pytest.raises(FileNotFoundError, match="no saved results were found"):
        retrieval.run(sampler_object=sampler, error_checking=True)

    assert sampler.load_results_calls == 1
    assert sampler.run_sampler_calls == 0


def test_blackjax_sampler_load_results_reads_saved_outputs(tmp_path):
    sampler = object.__new__(BlackJAXNUTSSampler)
    sampler.output_directory = str(tmp_path)
    sampler.retrieval_name = "blackjax_saved"
    sampler.results = None

    output_directory = tmp_path / "out_BlackJAX"
    output_directory.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_directory / "blackjax_saved_samples.npz",
        samples=np.array([[1.0], [2.0]], dtype=float),
        unconstrained_samples=np.array([[0.1], [0.2]], dtype=float),
        logdensity=np.array([-1.0, -2.0], dtype=float),
        model_log_likelihood=np.array([-1.5, -2.5], dtype=float),
        acceptance_rate=np.array([0.8, 0.9], dtype=float),
        is_divergent=np.array([False, True]),
        num_integration_steps=np.array([3.0, 4.0], dtype=float),
    )
    (output_directory / "blackjax_saved_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "nuts",
                "warmup_parameters": {"step_size": 0.5},
            }
        )
    )

    results = sampler.load_results()

    assert isinstance(results, BlackJAXSamplingResults)
    assert results.algorithm == "nuts"
    np.testing.assert_allclose(results.samples, np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(results.logdensity, np.array([-1.0, -2.0]))
    np.testing.assert_allclose(
        results.model_log_likelihood, np.array([-1.5, -2.5])
    )
    np.testing.assert_allclose(results.acceptance_rate, np.array([0.8, 0.9]))
    np.testing.assert_array_equal(results.is_divergent, np.array([False, True]))


def test_numpyro_sampler_load_results_reads_saved_outputs(tmp_path):
    sampler = object.__new__(NumPyroNUTSSampler)
    sampler.output_directory = str(tmp_path)
    sampler.retrieval_name = "numpyro_saved"
    sampler.results = None
    sampler.parameter_names = None

    output_directory = tmp_path / "out_NumPyro"
    output_directory.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_directory / "numpyro_saved_samples.npz",
        samples=np.array([[1.0], [2.0]], dtype=float),
        unconstrained_samples=np.array([[0.1], [0.2]], dtype=float),
        logdensity=np.array([-1.0, -2.0], dtype=float),
        model_log_likelihood=np.array([-1.5, -2.5], dtype=float),
        acceptance_rate=np.array([0.8, 0.9], dtype=float),
        is_divergent=np.array([False, True]),
        num_integration_steps=np.array([3.0, 4.0], dtype=float),
    )
    (output_directory / "numpyro_saved_params.json").write_text(
        json.dumps(["amplitude"])
    )
    (output_directory / "numpyro_saved_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "nuts",
                "warmup_parameters": {"step_size": 0.5},
            }
        )
    )

    results = sampler.load_results()

    assert isinstance(results, NumPyroSamplingResults)
    assert results.algorithm == "nuts"
    assert sampler.parameter_names == ["amplitude"]
    np.testing.assert_allclose(results.samples, np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(results.logdensity, np.array([-1.0, -2.0]))
    np.testing.assert_allclose(
        results.model_log_likelihood, np.array([-1.5, -2.5])
    )
    np.testing.assert_allclose(results.acceptance_rate, np.array([0.8, 0.9]))
    np.testing.assert_array_equal(results.is_divergent, np.array([False, True]))


def test_ultranest_sampler_load_results_reads_saved_outputs(tmp_path):
    sampler = object.__new__(UltranestSampler)
    sampler.output_directory = str(tmp_path)
    sampler.retrieval_name = "ultranest_saved"
    sampler.results = None

    log_directory = tmp_path / "out_ultranest_saved_ultranest"
    (log_directory / "info").mkdir(parents=True, exist_ok=True)
    (log_directory / "chains").mkdir(parents=True, exist_ok=True)

    (log_directory / "info" / "results.json").write_text(
        json.dumps({"logz": 1.2, "logzerr": 0.3, "ess": 7, "ncall": 9})
    )
    np.savetxt(
        log_directory / "chains" / "equal_weighted_post.txt",
        np.array([[1.0], [2.0]], dtype=float),
        header="amplitude",
        comments="",
    )
    np.savetxt(
        log_directory / "chains" / "weighted_post.txt",
        np.array([[0.6, -1.0, 1.0], [0.4, -2.0, 2.0]], dtype=float),
        header="weight logl amplitude",
        comments="",
    )

    results = sampler.load_results()

    assert results["logz"] == 1.2
    assert results["logzerr"] == 0.3
    assert results["ess"] == 7
    assert results["ncall"] == 9
    np.testing.assert_allclose(results["samples"], np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(
        results["weighted_samples"]["points"], np.array([[1.0], [2.0]])
    )
    np.testing.assert_allclose(
        results["weighted_samples"]["logl"], np.array([-1.0, -2.0])
    )


def test_run_saves_named_sample_archive_for_completed_retrieval(tmp_path):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    data = Data(
        name="simple",
        model_generating_function=lambda _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False: (
            jnp.array([1.0, 2.0]),
            jnp.array(
                [parameters["amplitude"].value, parameters["amplitude"].value + 1.0]
            ),
        ),
        radtrans_object=SimpleNamespace(line_species=(), cloud_species=()),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="blackjaxnuts",
        retrieval_name="archived_summary",
        run_mode="retrieval",
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.ultranest = False
    retrieval.plotter = RetrievalPlotter(retrieval)

    os.makedirs(tmp_path / "evaluate_archived_summary", exist_ok=True)

    sampler = _RecordingPosteriorSampler()
    retrieval.run(sampler_object=sampler, error_checking=False)

    archive_path = tmp_path / "out_samples" / "archived_summary_samples.npz"
    assert archive_path.exists()

    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) >= {
            "parameter_names",
            "sample_matrix",
            "amplitude",
            "log_likelihood",
        }
        np.testing.assert_array_equal(
            archive["parameter_names"], np.array(["amplitude"])
        )
        np.testing.assert_allclose(
            archive["sample_matrix"], np.array([[1.0], [3.0], [2.0]])
        )
        np.testing.assert_allclose(archive["amplitude"], np.array([1.0, 3.0, 2.0]))
        np.testing.assert_allclose(
            archive["log_likelihood"], np.array([-3.0, -1.0, -2.0])
        )


def test_get_samples_loads_named_archive_for_non_pymultinest_sampler(tmp_path):
    retrieval = _seed_retrieval_mpi_state(
        object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    )
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    data = Data(
        name="simple",
        model_generating_function=lambda _prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False: (
            jnp.array([1.0, 2.0]),
            jnp.array(
                [parameters["amplitude"].value, parameters["amplitude"].value + 1.0]
            ),
        ),
        radtrans_object=SimpleNamespace(line_species=(), cloud_species=()),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 3.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="jaxns",
        retrieval_name="archived_jaxns",
        run_mode="retrieval",
    )
    retrieval.parameter_layout = ParameterLayout.from_parameters(
        retrieval.configuration.parameters
    )
    retrieval._free_parameter_names_cache = list(
        retrieval.parameter_layout.free_parameter_names
    )
    retrieval._n_free_params_total_retrieval = (
        retrieval.parameter_layout.n_free_parameters
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.ultranest = False
    retrieval.plotter = RetrievalPlotter(retrieval)

    os.makedirs(tmp_path / "evaluate_archived_jaxns", exist_ok=True)

    sampler = _RecordingPosteriorSampler()
    retrieval.run(sampler_object=sampler, error_checking=False)

    retrieval.sampler = None
    retrieval.latest_sampler_results = None

    sample_dict, parameter_dict = retrieval.get_samples(
        names=["archived_jaxns"],
        output_directory=str(tmp_path),
    )

    np.testing.assert_allclose(
        sample_dict["archived_jaxns"], np.array([[1.0, 3.0, 2.0], [-3.0, -1.0, -2.0]])
    )
    assert parameter_dict["archived_jaxns"] == ["amplitude"]


def test_get_samples_dict_supports_jaxns_style_results_without_pymultinest():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        sampler_type="jaxns",
        retrieval_name="jaxns_samples",
    )
    retrieval._free_parameter_names_cache = ["amplitude"]
    retrieval.output_directory = tests_results_directory
    retrieval.ultranest = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = _JAXNSLikeResults()

    samples = retrieval.get_samples_dict(return_likelihood=True)

    np.testing.assert_allclose(samples["amplitude"], np.array([1.0, 3.0, 2.0]))
    np.testing.assert_allclose(samples["log_likelihood"], np.array([-3.0, -1.0, -2.0]))
    assert retrieval.get_best_fit_parameters() == {"amplitude": 3.0}
    assert retrieval.get_quantile_parameters(0.5) == {"amplitude": 2.0}


def test_jaxns_equal_weight_results_fall_back_to_samples_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(
        petitRADTRANS.retrieval.sampler,
        "jaxns_resample",
        lambda key, posterior_payload, log_dp_mean: {
            "samples": posterior_payload["samples"],
            "log_likelihood": posterior_payload["log_likelihood"],
        },
    )

    sampler = JAXNSSampler(
        log_likelihood_func=lambda *_args: 0.0,
        prior_func=lambda: (),
        output_directory=str(tmp_path),
        retrieval_name="jaxns_samples_only",
    )
    sampler.parameter_names = ["amplitude"]
    sampler.seed = 7

    results = sampler._build_equal_weight_results(_JAXNSSamplesOnlyResults())

    assert results["samples"].shape == (3, 1)
    np.testing.assert_allclose(
        results["weighted_samples"]["points"],
        np.array([[1.0], [3.0], [2.0]]),
    )

    with np.load(
        tmp_path / "out_JAXNS" / "jaxns_samples_only_samples.npz",
        allow_pickle=False,
    ) as archive:
        np.testing.assert_array_equal(
            archive["parameter_names"], np.array(["amplitude"])
        )
        np.testing.assert_allclose(
            archive["sample_matrix"],
            results["samples"],
        )
        np.testing.assert_allclose(
            archive["log_likelihood"],
            results["log_likelihood"],
        )


def test_get_samples_recovers_malformed_jaxns_archive_from_raw_results(tmp_path, monkeypatch):
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    parameter = Parameter(
        name="amplitude",
        is_free_parameter=True,
        value=0.0,
        transform_prior_cube_coordinate=lambda x: 4.0 * x,
    )
    retrieval.configuration = SimpleNamespace(
        parameters={"amplitude": parameter},
        sampler_type="jaxns",
        retrieval_name="current_jaxns",
    )
    retrieval.output_directory = str(tmp_path)
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.ultranest = False

    out_jaxns = tmp_path / "out_JAXNS"
    out_jaxns.mkdir(parents=True, exist_ok=True)

    params_path = out_jaxns / "archived_jaxns_params.json"
    samples_path = out_jaxns / "archived_jaxns_samples.npz"
    raw_results_path = out_jaxns / "archived_jaxns_jaxns_results.json"

    params_path.write_text("[]")
    raw_results_path.write_text("{}")
    np.savez(
        samples_path,
        sample_matrix=np.array(np.nan),
        log_likelihood=np.array([-3.0, -1.0, -2.0]),
    )

    fake_jaxns = ModuleType("jaxns")
    fake_jaxns.load_results = lambda _path: _JAXNSSamplesOnlyResults()
    monkeypatch.setitem(sys.modules, "jaxns", fake_jaxns)

    sample_dict, parameter_dict = retrieval.get_samples(
        names=["archived_jaxns"],
        output_directory=str(tmp_path),
    )

    np.testing.assert_allclose(
        sample_dict["archived_jaxns"],
        np.array([[1.0, 3.0, 2.0], [-3.0, -1.0, -2.0]]),
    )
    assert parameter_dict["archived_jaxns"] == ["amplitude"]

    with np.load(samples_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["parameter_names"], np.array(["amplitude"])
        )
        np.testing.assert_allclose(
            archive["sample_matrix"], np.array([[1.0], [3.0], [2.0]])
        )
        np.testing.assert_allclose(
            archive["log_likelihood"], np.array([-3.0, -1.0, -2.0])
        )

    assert json.loads(params_path.read_text()) == ["amplitude"]


def test_set_free_parameter_values_updates_non_pymultinest_retrievals():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = SimpleNamespace(
        parameters={
            "free_parameter": Parameter(
                name="free_parameter",
                is_free_parameter=True,
            ),
            "fixed_parameter": Parameter(
                name="fixed_parameter",
                is_free_parameter=False,
                value=7.0,
            ),
        }
    )

    retrieval._set_free_parameter_values(np.array([3.5]))

    assert retrieval.configuration.parameters["free_parameter"].value == 3.5
    assert retrieval.configuration.parameters["fixed_parameter"].value == 7.0


def test_set_free_parameter_values_accepts_ctypes_sequences():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)
    retrieval.configuration = SimpleNamespace(
        parameters={
            "free_parameter": Parameter(
                name="free_parameter",
                is_free_parameter=True,
            ),
        }
    )

    cube = (ctypes.c_double * 1)(4.25)

    retrieval._set_free_parameter_values(cube)

    assert retrieval.configuration.parameters["free_parameter"].value == 4.25


def test_get_full_range_model_uses_model_adapter_for_plotting():
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    def legacy_direct_call(*args, **kwargs):
        raise AssertionError("plotting should not call the legacy model path directly")

    reference_data = SimpleNamespace(
        name="reference_dataset",
        model_generating_function=legacy_direct_call,
    )

    retrieval.configuration = SimpleNamespace(
        pressures=np.array([1.0, 10.0]),
        adaptive_mesh_refinement=False,
        data={"reference_dataset": reference_data},
    )
    retrieval.plotter = SimpleNamespace(
        get_reference_data=lambda: reference_data,
    )

    expected_wavelengths = np.array([1.0, 2.0])
    expected_spectrum = np.array([0.1, 0.2])
    adapter_call = {}

    def fake_call_model_generating_function(
        model_generating_function,
        radtrans_object,
        parameters,
        *,
        data_name,
        data_object=None,
        pt_plot_mode=False,
        adaptive_mesh_refinement=False,
    ):
        adapter_call["model_generating_function"] = model_generating_function
        adapter_call["radtrans_object"] = radtrans_object
        adapter_call["parameters"] = parameters
        adapter_call["data_name"] = data_name
        adapter_call["data_object"] = data_object
        adapter_call["pt_plot_mode"] = pt_plot_mode
        adapter_call["adaptive_mesh_refinement"] = adaptive_mesh_refinement
        return expected_wavelengths, expected_spectrum

    retrieval._call_model_generating_function = fake_call_model_generating_function

    parameters = {
        "amplitude": Parameter(
            name="amplitude",
            is_free_parameter=False,
            value=1.0,
        )
    }
    atmosphere = object()

    wavelengths, spectrum = retrieval.get_full_range_model(
        parameters,
        prt_object=atmosphere,
    )

    np.testing.assert_allclose(wavelengths, expected_wavelengths)
    np.testing.assert_allclose(spectrum, expected_spectrum)
    assert adapter_call["model_generating_function"] is legacy_direct_call
    assert adapter_call["radtrans_object"] is atmosphere
    assert adapter_call["data_name"] == "reference_dataset"
    assert adapter_call["data_object"] is reference_data
    assert adapter_call["pt_plot_mode"] is False
    assert adapter_call["adaptive_mesh_refinement"] is False
    assert "contribution" not in parameters


def test_save_best_fit_outputs_can_skip_in_memory_retention(tmp_path, monkeypatch):
    retrieval = object.__new__(petitRADTRANS.retrieval.retrieval.Retrieval)

    wavelengths = np.array([1.0, 2.0, 3.0])
    spectrum = np.array([0.5, 1.0, 1.5])
    output_directory = tmp_path / "retrieval_outputs"
    (output_directory / "evaluate_test").mkdir(parents=True)

    data = SimpleNamespace(
        scale_flux=False,
        scale_uncertainties=False,
        fit_flux_offset=False,
        external_radtrans_reference=None,
        variability_atmospheric_column_model_flux_return_mode=False,
        data_resolution_array_model=None,
        data_resolution=None,
        wavelengths=wavelengths,
        wavelength_bin_widths=np.ones_like(wavelengths),
        subtract_continuum=False,
        model_generating_function=object(),
        radtrans_object=object(),
        name="data.1",
    )

    retrieval.configuration = SimpleNamespace(
        data={"data.1": data},
        retrieval_name="test",
        adaptive_mesh_refinement=False,
    )
    retrieval.output_directory = str(output_directory)
    retrieval.evaluate_sample_spectra = False
    retrieval.best_fit_spectra = {"stale": [np.array([0.0]), np.array([0.0])]}
    retrieval.posterior_sample_spectra = {}
    retrieval._call_model_generating_function = lambda *args, **kwargs: (
        wavelengths,
        spectrum,
    )

    monkeypatch.setattr(
        petitRADTRANS.retrieval.retrieval,
        "rebin_spectrum_bin",
        lambda wavelengths_model, spectrum_model, data_wavelengths, data_bin_widths: np.asarray(
            spectrum_model
        ),
    )

    best_fit_spectra = retrieval.save_best_fit_outputs(
        parameters={},
        retain_best_fit_spectra=False,
    )

    output_file = (
        output_directory
        / "evaluate_test"
        / "test_model_spec_best_fit_data_1.dat"
    )

    assert retrieval.best_fit_spectra == {}
    assert list(best_fit_spectra) == ["data.1"]
    np.testing.assert_allclose(best_fit_spectra["data.1"][0], wavelengths)
    np.testing.assert_allclose(best_fit_spectra["data.1"][1], spectrum)
    assert output_file.is_file()


def test_retrieval_parameter_interface_copies_fixed_parameter_mutables():
    interface = (
        petitRADTRANS.spectral_model.SpectralModel._SpectralModel__retrieval_parameters_interface
    )

    fixed_imposed_mass_fractions = {"CO": np.array([0.2, 0.2])}
    fixed_other = np.array([1.0, 2.0])
    fixed_parameters = {
        "imposed_mass_fractions": fixed_imposed_mass_fractions,
        "other": fixed_other,
    }

    parameters, _, _, _ = interface(
        retrieved_parameters={"H2O": -3.0},
        fixed_parameters=fixed_parameters,
        line_species=("H2O",),
        n_layers=2,
    )

    assert set(fixed_imposed_mass_fractions) == {"CO"}
    assert set(parameters["imposed_mass_fractions"]) == {"CO", "H2O"}
    assert parameters["imposed_mass_fractions"] is not fixed_imposed_mass_fractions
    assert (
        parameters["imposed_mass_fractions"]["CO"]
        is not fixed_imposed_mass_fractions["CO"]
    )
    np.testing.assert_allclose(
        parameters["imposed_mass_fractions"]["CO"],
        fixed_imposed_mass_fractions["CO"],
    )

    parameters["other"][0] = 99.0
    assert fixed_other[0] == 1.0


def test_numpyro_hmc_kernel_kwargs_prefer_trajectory_length_over_num_steps():
    kernel_kwargs = NumPyroSampler._prepare_kernel_kwargs(
        algorithm_name="hmc",
        log_likelihood_func=lambda position: jnp.sum(position**2),
        initial_step_size=1.0,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=0.8,
        numpyro_kwargs={
            "num_steps": 20,
            "trajectory_length": 3.5,
        },
    )

    assert "trajectory_length" in kernel_kwargs
    assert "num_steps" not in kernel_kwargs


def init_run():
    # Since our retrieval has already run before, we'll set the mode to 'evaluate' so we can make some plots.
    run_definition_simple = petitRADTRANS.retrieval.RetrievalConfig(
        retrieval_name="test",
        run_mode="retrieval",  # This must be 'retrieval' to run PyMultiNest
        adaptive_mesh_refinement=False,  # We won't be using adaptive mesh refinement for the pressure grid
        pressures=test_parameters["pressures"],
        scattering_in_emission=False,  # This would turn on scattering when calculating emission spectra
    )
    # Scattering is automatically included for transmission spectra

    # Fixed parameters
    run_definition_simple.add_parameter(
        "Rstar",
        False,
        value=test_parameters["stellar_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_sun,
    )

    # Log of the surface gravity
    run_definition_simple.add_parameter(
        "log_g",
        False,
        value=np.log10(test_parameters["planetary_parameters"]["reference_gravity"]),
    )

    # Retrieved parameters
    # Prior functions
    def prior_planet_radius(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters["retrieval_parameters"]["planetary_radius_bounds"][0]
            * petitRADTRANS.physical_constants.r_jup_mean,
            x2=test_parameters["retrieval_parameters"]["planetary_radius_bounds"][1]
            * petitRADTRANS.physical_constants.r_jup_mean,
        )

    def prior_temperature(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters["retrieval_parameters"]["intrinsic_temperature_bounds"][
                0
            ],
            x2=test_parameters["retrieval_parameters"]["intrinsic_temperature_bounds"][
                1
            ],
        )

    def prior_cloud_pressure(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters["retrieval_parameters"]["log10_cloud_pressure_bounds"][
                0
            ],
            x2=test_parameters["retrieval_parameters"]["log10_cloud_pressure_bounds"][
                1
            ],
        )

    # Planet radius
    run_definition_simple.add_parameter(
        "R_pl", True, transform_prior_cube_coordinate=prior_planet_radius
    )

    # Intrinsic temperature
    run_definition_simple.add_parameter(
        "Temperature", True, transform_prior_cube_coordinate=prior_temperature
    )

    # Include a gray cloud as well to see what happens
    run_definition_simple.add_parameter(
        "log_Pcloud", True, transform_prior_cube_coordinate=prior_cloud_pressure
    )

    # Spectrum parameters
    run_definition_simple.set_rayleigh_species(
        test_parameters["spectrum_parameters"]["rayleigh_species"]
    )
    run_definition_simple.set_continuum_opacities(
        test_parameters["spectrum_parameters"]["continuum_opacities"]
    )

    run_definition_simple.set_line_species(
        test_parameters["spectrum_parameters"]["line_species_correlated_k"],
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(
            test_parameters["retrieval_parameters"][
                "log10_species_mass_fractions_bounds"
            ][0],
            test_parameters["retrieval_parameters"][
                "log10_species_mass_fractions_bounds"
            ][1],
        ),  # prior: min = abund_lim[0], max = min + abund_lim[1]
    )

    # Remove old binned down opacities to test rebinning function
    for species in test_parameters["spectrum_parameters"]["line_species_correlated_k"]:
        file = CorrelatedKOpacity.find(
            path_input_data=petitRADTRANS.config.petitradtrans_config_parser.get_input_data_path(),
            category="correlated_k_opacities",
            species=species,
            find_all=False,
            search_online=True,
        ).replace(
            CorrelatedKOpacity.join_species_all_info(
                "",
                spectral_info=CorrelatedKOpacity.get_resolving_power_string(
                    resolving_power=int(
                        CorrelatedKOpacity.get_default_resolving_power()
                    )
                ),
            ),
            CorrelatedKOpacity.join_species_all_info(
                "",
                CorrelatedKOpacity.get_resolving_power_string(
                    resolving_power=test_parameters["mock_observation_parameters"][
                        "resolving_power"
                    ]
                    * 2
                ),
            ),
        )

        if os.path.isfile(file):
            os.remove(file)

    # Load data
    run_definition_simple.add_data(
        name="test",
        path_to_observations=reference_filenames["mock_observation_transmission"],
        model_generating_function=retrieval_model_spec_iso,
        line_opacity_mode="c-k",
        data_resolution=test_parameters["mock_observation_parameters"][
            "resolving_power"
        ],
        model_resolution=test_parameters["mock_observation_parameters"][
            "resolving_power"
        ]
        * 2,
    )

    # Plot parameters
    # Corner plot
    run_definition_simple.parameters["R_pl"].plot_in_corner = True
    run_definition_simple.parameters["R_pl"].corner_label = (
        r"$R_{\rm P}$ ($\rm R_{Jup}$)"
    )
    run_definition_simple.parameters["R_pl"].corner_transform = (
        lambda x: x / petitRADTRANS.physical_constants.r_jup_mean
    )
    run_definition_simple.parameters["Temperature"].plot_in_corner = True
    run_definition_simple.parameters["Temperature"].corner_label = "Temp"
    run_definition_simple.parameters["log_Pcloud"].plot_in_corner = True
    run_definition_simple.parameters["log_Pcloud"].corner_label = r"log P$_{\rm cloud}$"
    run_definition_simple.parameters["log_Pcloud"].corner_ranges = [-6, 2]

    # for spec in run_definition_simple.line_species:
    #    spec = Opacity([spec]).get_full_name().split('.R')[0]  # deal with the naming scheme for binned down opacities
    #    run_definition_simple.parameters[spec].plot_in_corner = True
    #    run_definition_simple.parameters[spec].corner_ranges = [-6.0, 0.0]

    return run_definition_simple


# Atmospheric model
# noinspection PyUnusedLocal
def retrieval_model_spec_iso(
    prt_object, parameters, pt_plot_mode=None, adaptive_mesh_refinement=False
):
    """
    This model computes a transmission spectrum based on free retrieval chemistry
    and an isothermal temperature-pressure profile.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                Rstar : Radius of the host star [cm]
                log_g : Log of surface gravity
                R_pl : planet radius [cm]
                Temperature : Isothermal temperature [K]
                species : Log abundances for each species used in the retrieval
                log_Pcloud : optional, cloud base pressure of a gray cloud deck.
        pt_plot_mode:
            Return only the pressure-temperature profile for plotting. Evaluate mode only. Mandatory.
        amr:
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base. Mandatory.

    Returns:
        wavelength_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    # Make the P-T profile
    pressures = prt_object.pressures * 1e-6  # bar to cgs
    temperatures = parameters["Temperature"].value * np.ones_like(pressures)

    # Make the abundance profiles
    abundances = {}
    m_sum = 0.0  # Check that the total mass fraction of all species is <1

    for species in prt_object.line_species:
        abundances[species] = 10 ** parameters[species].value * np.ones_like(pressures)
        m_sum += 10 ** parameters[species].value

    abundances["H2"] = (
        test_parameters["mass_fractions_correlated_k"]["H2"]
        * (1.0 - m_sum)
        * np.ones_like(pressures)
    )
    abundances["He"] = (
        test_parameters["mass_fractions_correlated_k"]["He"]
        * (1.0 - m_sum)
        * np.ones_like(pressures)
    )

    # Find the mean molecular weight in each layer
    mmw = compute_mean_molar_masses(abundances)

    # Calculate the spectrum
    wavelengths, transit_radii, _ = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=10 ** parameters["log_g"].value,
        mean_molar_masses=mmw,
        planet_radius=parameters["R_pl"].value,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        opaque_cloud_top_pressure=10 ** parameters["log_Pcloud"].value,
        frequencies_to_wavelengths=True,  # True by default
    )

    # Transform the outputs into the units of our data.
    wavelengths *= 1e4  # cm to um
    spectrum_model = (transit_radii / parameters["Rstar"].value) ** 2.0

    return wavelengths, spectrum_model


run_definition = init_run()


def test_list_available_species():
    run_definition.list_available_line_species()


def test_simple_retrieval():
    retrieval = petitRADTRANS.retrieval.Retrieval(
        run_definition,
        output_directory=tests_results_directory,
        evaluate_sample_spectra=test_parameters["retrieval_parameters"][
            "sample_spectrum_output"
        ],
        reference_data_name="test",
    )

    retrieval.plotter.spec_xlabel = "Wavelength [micron]"
    retrieval.plotter.spec_ylabel = r"$(R_{\rm P}/R_*)^2$ [ppm]"
    retrieval.plotter.y_axis_scaling = 1e6
    retrieval.plotter.xscale = "linear"
    retrieval.plotter.yscale = "linear"
    retrieval.plotter.nsample = 10
    retrieval.plotter.temp_limits = [150, 3000]
    retrieval.plotter.press_limits = [1e2, 1e-5]

    retrieval.run(
        sampling_efficiency=test_parameters["retrieval_parameters"][
            "sampling_efficiency"
        ],
        n_live_points=test_parameters["retrieval_parameters"]["n_live_points"],
        const_efficiency_mode=test_parameters["retrieval_parameters"][
            "const_efficiency_mode"
        ],
        resume=test_parameters["retrieval_parameters"]["resume"],
        seed=test_parameters["retrieval_parameters"]["seed"],
    )

    # Check if 'get_samples' works
    # noinspection PyUnusedLocal
    sample_dict, parameter_dict = retrieval.get_samples(
        names=retrieval.corner_files, output_directory=retrieval.output_directory
    )

    # Get results and reference
    with open(reference_filenames["pymultinest_parameter_analysis"]) as f:
        reference = json.load(f)

    new_result_file = os.path.join(
        tests_results_directory,
        "out_PMN",
        os.path.basename(reference_filenames["pymultinest_parameter_analysis"]),
    )

    with open(new_result_file) as f:
        new_results = json.load(f)

    # Check if retrieved parameters are in +/- 1 sigma of the previous retrieved parameters
    for i, marginal in enumerate(new_results["marginals"]):
        assert (
            marginal["median"] - relative_tolerance * reference["marginals"][i]["sigma"]
            <= reference["marginals"][i]["median"]
            <= marginal["median"]
            + relative_tolerance * reference["marginals"][i]["sigma"]
        )
