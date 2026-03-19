from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, TYPE_CHECKING
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import tree_util

from petitRADTRANS.retrieval.jax_parameter import JAX_DTYPE, Parameter

if TYPE_CHECKING:
    from petitRADTRANS.radtrans import Radtrans


def _flatten_parameter_values(values: npt.ArrayLike) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.reshape(-1)

    if hasattr(values, 'tolist'):
        converted_values = values.tolist()
        if isinstance(converted_values, list):
            return np.array(converted_values, dtype=float).reshape(-1)

    try:
        return np.array([values[index] for index in range(len(values))], dtype=float)
    except TypeError:
        return np.array([values], dtype=float)


def _is_static_parameter_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, (str, bytes, dict)):
        return True

    if isinstance(value, (list, tuple)):
        return all(_is_static_parameter_value(item) for item in value)

    return False


def _clone_parameter(parameter: Parameter, value: Any) -> Parameter:
    return parameter.__class__(
        name=parameter.name,
        is_free_parameter=parameter.is_free_parameter,
        value=value,
        distribution=parameter.distribution,
        transform_prior_cube_coordinate=parameter.transform_prior_cube_coordinate,
        plot_in_corner=parameter.plot_in_corner,
        corner_ranges=copy.deepcopy(parameter.corner_ranges),
        corner_transform=parameter.corner_transform,
        corner_label=parameter.corner_label,
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PhysicalParams:
    ordered_names: tuple[str, ...]
    dynamic_names: tuple[str, ...]
    dynamic_values: tuple[Any, ...]
    static_items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> 'PhysicalParams':
        ordered_names = tuple(mapping.keys())
        dynamic_names = []
        dynamic_values = []
        static_items = []

        for name, value in mapping.items():
            if _is_static_parameter_value(value):
                static_items.append((name, value))
            else:
                dynamic_names.append(name)
                dynamic_values.append(value)

        return cls(
            ordered_names=ordered_names,
            dynamic_names=tuple(dynamic_names),
            dynamic_values=tuple(dynamic_values),
            static_items=tuple(static_items),
        )

    def tree_flatten(self):
        children = self.dynamic_values
        aux_data = {
            'ordered_names': self.ordered_names,
            'dynamic_names': self.dynamic_names,
            'static_items': self.static_items,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            ordered_names=aux_data['ordered_names'],
            dynamic_names=aux_data['dynamic_names'],
            dynamic_values=tuple(children),
            static_items=aux_data['static_items'],
        )

    def as_dict(self) -> dict[str, Any]:
        dynamic_mapping = dict(zip(self.dynamic_names, self.dynamic_values, strict=True))
        static_mapping = dict(self.static_items)
        return {
            name: dynamic_mapping[name] if name in dynamic_mapping else static_mapping[name]
            for name in self.ordered_names
        }

    def get(self, name: str, default: Any = None) -> Any:
        return self.as_dict().get(name, default)

    def ordered_items(self) -> tuple[tuple[str, Any], ...]:
        mapping = self.as_dict()
        return tuple((name, mapping[name]) for name in self.ordered_names)


@dataclass(frozen=True)
class ObservationState:
    name: str
    wavelengths: Any
    spectrum: Any
    mask: Any
    radtrans_object: Any = None
    model_generating_function: Any = None
    model_resolution: Any = None
    data_resolution: Any = None
    wavelength_bin_widths: Any = None
    photometry: bool = False
    external_radtrans_reference: str | None = None
    line_opacity_mode: str | None = None
    model_contract: str = 'legacy'
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LikelihoodState:
    data_uncertainties: Any = None
    data_covariance: Any = None
    mask: Any = None
    inv_covariance: Any = None
    log_covariance_determinant: Any = None
    n_free_parameters: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelContext:
    name: str
    mode: str
    radtrans: 'Radtrans'
    adaptive_mesh_refinement: bool
    variability_atmospheric_column_model_flux_return_mode: bool = False
    return_contribution: bool = False
    return_opacities: bool = False
    return_photosphere_radius: bool = False
    return_rosseland_optical_depths: bool = False
    return_radius_hydrostatic_equilibrium: bool = False
    return_cloud_contribution: bool = False
    return_abundances: bool = False
    model_metadata: Mapping[str, Any] = field(default_factory=dict)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelInputs:
    pressures: Any
    temperatures: Any
    abundances: Mapping[str, Any]
    mean_molar_masses: Any
    gravity: Any
    planet_radius: Any
    reference_pressure: Any = None
    opaque_cloud_top_pressure: Any = None
    sigma_lnorm: Any = None
    cloud_particles_mean_radii: Any = None
    cloud_f_sed: Any = None
    eddy_diffusion_coefficients: Any = None
    haze_factor: Any = None
    power_law_opacity_coefficient: Any = None
    power_law_opacity_350nm: Any = None
    cloud_hansen_b: Any = None
    cloud_fraction: Any = None
    complete_coverage_clouds: tuple[str, ...] = ()
    cloud_particles_radius_distribution: str | None = None
    distance_to_system: Any = None
    stellar_radius: Any = None
    disk_blackbody_temperature: Any = None
    disk_radius: Any = None
    v_band_extinction: Any = None
    v_band_reddening: Any = None
    additional_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.complete_coverage_clouds is None:
            normalized_complete_coverage_clouds = ()
        elif isinstance(self.complete_coverage_clouds, tuple):
            normalized_complete_coverage_clouds = self.complete_coverage_clouds
        elif isinstance(self.complete_coverage_clouds, (str, bytes)):
            normalized_complete_coverage_clouds = (self.complete_coverage_clouds,)
        else:
            normalized_complete_coverage_clouds = tuple(self.complete_coverage_clouds)

        object.__setattr__(
            self,
            'complete_coverage_clouds',
            normalized_complete_coverage_clouds,
        )

    def tree_flatten(self):
        children = (
            self.pressures,
            self.temperatures,
            self.abundances,
            self.mean_molar_masses,
            self.gravity,
            self.planet_radius,
            self.reference_pressure,
            self.opaque_cloud_top_pressure,
            self.sigma_lnorm,
            self.cloud_particles_mean_radii,
            self.cloud_f_sed,
            self.eddy_diffusion_coefficients,
            self.haze_factor,
            self.power_law_opacity_coefficient,
            self.power_law_opacity_350nm,
            self.cloud_hansen_b,
            self.cloud_fraction,
            self.distance_to_system,
            self.stellar_radius,
            self.disk_blackbody_temperature,
            self.disk_radius,
            self.v_band_extinction,
            self.v_band_reddening,
        )
        aux_data = {
            'complete_coverage_clouds': self.complete_coverage_clouds,
            'cloud_particles_radius_distribution': self.cloud_particles_radius_distribution,
            'additional_inputs': self.additional_inputs,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            pressures=children[0],
            temperatures=children[1],
            abundances=children[2],
            mean_molar_masses=children[3],
            gravity=children[4],
            planet_radius=children[5],
            reference_pressure=children[6],
            opaque_cloud_top_pressure=children[7],
            sigma_lnorm=children[8],
            cloud_particles_mean_radii=children[9],
            cloud_f_sed=children[10],
            eddy_diffusion_coefficients=children[11],
            haze_factor=children[12],
            power_law_opacity_coefficient=children[13],
            power_law_opacity_350nm=children[14],
            cloud_hansen_b=children[15],
            cloud_fraction=children[16],
            distance_to_system=children[17],
            stellar_radius=children[18],
            disk_blackbody_temperature=children[19],
            disk_radius=children[20],
            v_band_extinction=children[21],
            v_band_reddening=children[22],
            complete_coverage_clouds=aux_data['complete_coverage_clouds'],
            cloud_particles_radius_distribution=aux_data['cloud_particles_radius_distribution'],
            additional_inputs=aux_data['additional_inputs'],
        )


@dataclass(frozen=True)
class ModelResult:
    kind: str
    wavelengths: Any = None
    spectrum: Any = None
    pressures: Any = None
    temperatures: Any = None
    additional_log_likelihood: Any = 0.0
    auxiliary_outputs: Mapping[str, Any] = field(default_factory=dict)
    atmospheric_column_fluxes: Any = None


@dataclass(frozen=True)
class ModelGroupState:
    source_name: str
    observation_names: tuple[str, ...]
    model_contract: str
    model_generating_function: Any = None
    radtrans_object: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRuntime:
    parameter_layout: 'ParameterLayout'
    observations: Mapping[str, ObservationState] = field(default_factory=dict)
    likelihoods: Mapping[str, LikelihoodState] = field(default_factory=dict)
    model_groups: tuple[ModelGroupState, ...] = ()

    @classmethod
    def from_data(
        cls,
        parameter_layout: 'ParameterLayout',
        data: Mapping[str, Any],
        n_free_parameters: int | None = None,
    ) -> 'RetrievalRuntime':
        if n_free_parameters is None:
            n_free_parameters = parameter_layout.n_free_parameters

        observations = {
            data_name: data_object.to_observation_state()
            for data_name, data_object in data.items()
        }
        likelihoods = {
            data_name: data_object.to_likelihood_state(n_free_parameters=n_free_parameters)
            for data_name, data_object in data.items()
        }

        grouped_observations: dict[str, list[str]] = {}
        for data_name, observation in observations.items():
            source_name = observation.external_radtrans_reference or data_name
            grouped_observations.setdefault(source_name, []).append(data_name)

        model_groups = []
        for source_name, observation_names in grouped_observations.items():
            source_observation = observations[source_name]
            model_groups.append(
                ModelGroupState(
                    source_name=source_name,
                    observation_names=tuple(observation_names),
                    model_contract=source_observation.model_contract,
                    model_generating_function=source_observation.model_generating_function,
                    radtrans_object=source_observation.radtrans_object,
                    metadata={
                        'line_opacity_mode': source_observation.line_opacity_mode,
                        'has_external_references': any(
                            observations[name].external_radtrans_reference is not None
                            for name in observation_names
                        ),
                    },
                )
            )

        return cls(
            parameter_layout=parameter_layout,
            observations=observations,
            likelihoods=likelihoods,
            model_groups=tuple(model_groups),
        )

    @classmethod
    def from_configuration(
        cls,
        configuration: Any,
        parameter_layout: 'ParameterLayout' | None = None,
        n_free_parameters: int | None = None,
    ) -> 'RetrievalRuntime':
        if parameter_layout is None:
            parameter_layout = ParameterLayout.from_configuration(configuration)

        return cls.from_data(
            parameter_layout=parameter_layout,
            data=configuration.data,
            n_free_parameters=n_free_parameters,
        )

    @staticmethod
    def _parse_uncertainties_mode(uncertainties_mode: str) -> tuple[bool, str, float | None]:
        retrieve_uncertainties = False
        beta_mode = 'multiply'
        beta = 1.0

        if uncertainties_mode == 'default':
            beta = 1.0
        elif uncertainties_mode == 'optimize':
            warnings.warn('automatically optimizing for uncertainties, be sure of what you are doing...')
            beta = None
        elif uncertainties_mode == 'retrieve':
            warnings.warn('retrieving uncertainties (multiply mode), be sure of what you are doing...')
            retrieve_uncertainties = True
            beta_mode = 'multiply'
        elif uncertainties_mode == 'retrieve_add':
            warnings.warn('retrieving uncertainties (add mode), be sure of what you are doing...')
            retrieve_uncertainties = True
            beta_mode = 'add'
        else:
            raise ValueError(
                "uncertainties mode must be 'default'|'optimize'|'retrieve'|'retrieve_add', "
                f"but was '{uncertainties_mode}'"
            )

        return retrieve_uncertainties, beta_mode, beta

    def _build_model_context(
        self,
        configuration: Any,
        group: ModelGroupState,
        parameters: Mapping[str, Any],
    ) -> ModelContext:
        from petitRADTRANS.retrieval.utils import get_calculate_flux_return_values

        source_data = configuration.data[group.source_name]
        (
            return_contribution,
            return_opacities,
            return_photosphere_radius,
            return_rosseland_optical_depths,
            return_radius_hydrostatic_equilibrium,
            return_cloud_contribution,
            return_abundances,
            _return_any,
        ) = get_calculate_flux_return_values(parameters)

        return ModelContext(
            name=group.source_name,
            mode=group.metadata.get('mode', 'unknown'),
            radtrans=group.radtrans_object,
            adaptive_mesh_refinement=configuration.adaptive_mesh_refinement,
            variability_atmospheric_column_model_flux_return_mode=(
                source_data.variability_atmospheric_column_model_flux_return_mode
            ),
            return_contribution=return_contribution,
            return_opacities=return_opacities,
            return_photosphere_radius=return_photosphere_radius,
            return_rosseland_optical_depths=return_rosseland_optical_depths,
            return_radius_hydrostatic_equilibrium=return_radius_hydrostatic_equilibrium,
            return_cloud_contribution=return_cloud_contribution,
            return_abundances=return_abundances,
            model_metadata=group.metadata,
        )

    def _evaluate_model_group(
        self,
        configuration: Any,
        group: ModelGroupState,
        physical_params: PhysicalParams,
    ) -> ModelResult:
        from petitRADTRANS.retrieval.models import run_legacy_model_adapter, run_runtime_model_adapter

        model_context = self._build_model_context(configuration, group, self.parameter_layout.build_legacy_parameter_dict(physical_params))

        if group.model_contract == 'runtime_native':
            return run_runtime_model_adapter(
                group.model_generating_function,
                model_context,
                physical_params,
                pt_plot_mode=False,
            )

        return run_legacy_model_adapter(
            group.model_generating_function,
            model_context,
            physical_params,
            pt_plot_mode=False,
        )

    def _score_dataset_scalar(
        self,
        data_name: str,
        data: Any,
        parameters: Mapping[str, Any],
        wavelengths_model: Any,
        spectrum_model: Any,
        additional_log_likelihood: Any,
        beta: float | None,
        beta_mode: str,
    ) -> float:
        from petitRADTRANS.retrieval.log_likelihood import LogLikelihood
        from petitRADTRANS.retrieval.utils import extract_spectrum_parameters, process_spectrum_pipeline

        invalid_value = jnp.asarray(-1e99, dtype=JAX_DTYPE)

        data_spectrum = data.spectrum
        data_spectrum = data.scale_flux(data_spectrum, parameters)
        data_spectrum = data.offset_flux(data_spectrum, parameters)
        data_uncertainties, data_covariance = data.scale_and_inflate_uncertainties(parameters)

        if isinstance(spectrum_model, np.ma.MaskedArray):
            spectrum_model = spectrum_model.filled(0)

        if spectrum_model is None:
            return invalid_value

        if (not isinstance(data.spectrum, float)
                and hasattr(data.spectrum, 'dtype')
                and data.spectrum.dtype == 'O'):
            log_likelihood = 0.0
            if jnp.ndim(data.spectrum) == 1:
                for i, spectrum_data in enumerate(data.spectrum):
                    masked_model = spectrum_model[i][~data.mask[i]]
                    log_likelihood += jax.lax.cond(
                        jnp.isnan(masked_model).any(),
                        lambda _: invalid_value,
                        lambda _: data.log_likelihood(
                            masked_model,
                            spectrum_data,
                            data.uncertainties[i],
                            beta=beta,
                            beta_mode=beta_mode,
                        ),
                        operand=None,
                    )
            elif jnp.ndim(data.spectrum) == 2:
                for i, detector in enumerate(data.spectrum):
                    for j, spectrum_data in enumerate(detector):
                        masked_model = spectrum_model[i, j][~data.mask[i, j]]
                        log_likelihood += jax.lax.cond(
                            jnp.isnan(masked_model).any(),
                            lambda _: invalid_value,
                            lambda _: data.log_likelihood(
                                masked_model,
                                spectrum_data,
                                data.uncertainties[i, j],
                                beta=beta,
                                beta_mode=beta_mode,
                            ),
                            operand=None,
                        )
            else:
                raise ValueError(
                    f"observation is an array containing object, and has {jnp.ndim(data.spectrum)} dimensions, "
                    "but must have 1 to 2"
                )

            return log_likelihood + additional_log_likelihood

        if ((isinstance(data.spectrum, float) or jnp.ndim(data.spectrum) == 1)
                and jnp.ndim(spectrum_model) == 1):
            spectrum_params = extract_spectrum_parameters(parameters, data_name)
            model_spectrum_processed = process_spectrum_pipeline(
                wavelengths_model,
                spectrum_model,
                data,
                parameters,
                data_name,
                spectrum_params,
            )
            return jax.lax.cond(
                jnp.isnan(model_spectrum_processed).any(),
                lambda _: invalid_value,
                lambda _: (
                    LogLikelihood(
                        model_spectrum=model_spectrum_processed,
                        data_spectrum=data_spectrum,
                        data_uncertainties=data_uncertainties,
                        data_covariance=data_covariance,
                        n_free_parameters=self.parameter_layout.n_free_parameters,
                    ).calculate_log_likelihood()
                    + additional_log_likelihood
                ),
                operand=None,
            )

        if jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 2:
            return jax.lax.cond(
                jnp.isnan(spectrum_model[0, :]).any(),
                lambda _: invalid_value,
                lambda _: data.log_likelihood(
                    spectrum_model[0, :],
                    data.spectrum,
                    data.uncertainties,
                    beta=beta,
                    beta_mode=beta_mode,
                ),
                operand=None,
            )

        if jnp.ndim(data.spectrum) == 1 and jnp.ndim(spectrum_model) == 3:
            return jax.lax.cond(
                jnp.isnan(spectrum_model[0, 0, :]).any(),
                lambda _: invalid_value,
                lambda _: data.log_likelihood(
                    spectrum_model[0, 0, :],
                    data.spectrum,
                    data.uncertainties,
                    beta=beta,
                    beta_mode=beta_mode,
                ),
                operand=None,
            )

        if jnp.ndim(data.spectrum) == 2:
            log_likelihood = jnp.asarray(0.0, dtype=JAX_DTYPE)
            for i, spectrum_data in enumerate(data.spectrum):
                masked_model = spectrum_model[i, ~data.mask[i, :]]
                log_likelihood += jax.lax.cond(
                    jnp.isnan(masked_model).any(),
                    lambda _: invalid_value,
                    lambda _: data.log_likelihood(
                        masked_model,
                        spectrum_data,
                        data.uncertainties[i],
                        beta=beta,
                        beta_mode=beta_mode,
                    ),
                    operand=None,
                )
            return log_likelihood

        if jnp.ndim(data.spectrum) == 3:
            log_likelihood = jnp.asarray(0.0, dtype=JAX_DTYPE)
            for i, detector in enumerate(data.spectrum):
                for j, spectrum_data in enumerate(detector):
                    masked_model = spectrum_model[i, j, ~data.mask[i, j, :]]
                    log_likelihood += jax.lax.cond(
                        jnp.isnan(masked_model).any(),
                        lambda _: invalid_value,
                        lambda _: data.log_likelihood(
                            masked_model,
                            spectrum_data,
                            data.uncertainties[i, j],
                            beta=beta,
                            beta_mode=beta_mode,
                        ),
                        operand=None,
                    )
            return log_likelihood

        raise ValueError('unsupported spectrum dimensionality for scalar runtime evaluation')

    def evaluate_scalar(
        self,
        configuration: Any,
        physical_params: PhysicalParams,
        *,
        uncertainties_mode: str = 'default',
        print_log_likelihood_for_debugging: bool = False,
    ) -> float:
        invalid_value = -1e99
        tiny = 1e-99
        log_likelihood = 0.0
        log_prior = 0.0

        _retrieve_uncertainties, beta_mode, beta = self._parse_uncertainties_mode(uncertainties_mode)
        parameters = self.parameter_layout.build_legacy_parameter_dict(physical_params)

        for group in self.model_groups:
            model_result = self._evaluate_model_group(configuration, group, physical_params)

            if model_result.kind == 'invalid' or model_result.spectrum is None:
                return invalid_value

            additional_log_likelihood = model_result.additional_log_likelihood
            wavelengths_model = model_result.wavelengths
            spectrum_model = model_result.spectrum

            for observation_name in group.observation_names:
                data = configuration.data[observation_name]
                contribution = self._score_dataset_scalar(
                    observation_name,
                    data,
                    parameters,
                    wavelengths_model,
                    spectrum_model,
                    additional_log_likelihood,
                    beta,
                    beta_mode,
                )

                if contribution <= invalid_value:
                    return invalid_value

                log_likelihood += contribution

        if 'log_prior_weight' in parameters.keys():
            log_prior += parameters['log_prior_weight'].value

        if log_likelihood + log_prior < invalid_value:
            return invalid_value

        if jnp.abs(log_likelihood + log_prior) < tiny:
            log_likelihood = tiny
            log_prior = 0.0

        if print_log_likelihood_for_debugging:
            print('log_likelihood + log_prior', log_likelihood + log_prior)

        return log_likelihood + log_prior

    def evaluate_jax(
        self,
        configuration: Any,
        physical_params: PhysicalParams,
        *,
        uncertainties_mode: str = 'default',
        print_log_likelihood_for_debugging: bool = False,
    ) -> float:
        invalid_value = jnp.asarray(-1e99, dtype=JAX_DTYPE)
        tiny = 1e-99
        log_likelihood_contributions = []
        log_prior = 0.0
        has_invalid_contribution = False

        _retrieve_uncertainties, beta_mode, beta = self._parse_uncertainties_mode(uncertainties_mode)
        parameters = self.parameter_layout.build_legacy_parameter_dict(physical_params)

        for group in self.model_groups:
            model_result = self._evaluate_model_group(configuration, group, physical_params)

            if model_result.kind == 'invalid' or model_result.spectrum is None:
                return invalid_value

            group_contribution = 0.0
            for observation_name in group.observation_names:
                data = configuration.data[observation_name]
                contribution = self._score_dataset_scalar(
                    observation_name,
                    data,
                    parameters,
                    model_result.wavelengths,
                    model_result.spectrum,
                    model_result.additional_log_likelihood,
                    beta,
                    beta_mode,
                )

                contribution_is_invalid = contribution <= invalid_value
                has_invalid_contribution = jnp.logical_or(has_invalid_contribution, contribution_is_invalid)
                group_contribution = jnp.where(
                    contribution_is_invalid,
                    invalid_value,
                    group_contribution + contribution,
                )

            log_likelihood_contributions.append(group_contribution)

        log_likelihood = jnp.sum(jnp.asarray(log_likelihood_contributions, dtype=JAX_DTYPE))
        log_likelihood = jnp.where(has_invalid_contribution, invalid_value, log_likelihood)

        if 'log_prior_weight' in parameters.keys():
            log_prior += parameters['log_prior_weight'].value

        log_likelihood, log_prior = jax.lax.cond(
            log_likelihood + log_prior < invalid_value,
            lambda: (invalid_value, 0.0),
            lambda: (log_likelihood, log_prior),
        )

        log_likelihood, log_prior = jax.lax.cond(
            jnp.abs(log_likelihood + log_prior) < tiny,
            lambda: (tiny, 0.0),
            lambda: (log_likelihood, log_prior),
        )

        if print_log_likelihood_for_debugging:
            print('log_likelihood + log_prior', log_likelihood + log_prior)

        return log_likelihood + log_prior

    def evaluate_vectorized_jax(
        self,
        configuration: Any,
        free_parameter_values: Any,
        *,
        uncertainties_mode: str = 'default',
        print_log_likelihood_for_debugging: bool = False,
    ) -> Any:
        def evaluate_single(single_values: Any) -> Any:
            physical_params = self.parameter_layout.physical_params_from_free_values(single_values)
            return self.evaluate_jax(
                configuration=configuration,
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=print_log_likelihood_for_debugging,
            )

        free_parameter_values = jnp.asarray(free_parameter_values, dtype=JAX_DTYPE)

        if jnp.ndim(free_parameter_values) == 1:
            return evaluate_single(free_parameter_values)

        return jax.vmap(evaluate_single)(free_parameter_values)


@dataclass(frozen=True)
class ParameterLayout:
    parameter_names: tuple[str, ...]
    free_parameter_names: tuple[str, ...]
    free_parameter_indices: tuple[int, ...]
    parameter_templates: tuple[Parameter, ...]
    cube_epsilon: float = field(default_factory=lambda: float(1e-7 if JAX_DTYPE == jnp.float32 else 1e-12))

    @classmethod
    def from_configuration(cls, configuration: Any) -> 'ParameterLayout':
        return cls.from_parameters(configuration.parameters)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Parameter]) -> 'ParameterLayout':
        parameter_names = tuple(parameters.keys())
        parameter_templates = tuple(
            _clone_parameter(parameters[name], parameters[name].value)
            for name in parameter_names
        )
        free_parameter_names = tuple(
            name for name, parameter in zip(parameter_names, parameter_templates, strict=True)
            if parameter.is_free_parameter
        )
        free_parameter_indices = tuple(
            index for index, parameter in enumerate(parameter_templates)
            if parameter.is_free_parameter
        )

        return cls(
            parameter_names=parameter_names,
            free_parameter_names=free_parameter_names,
            free_parameter_indices=free_parameter_indices,
            parameter_templates=parameter_templates,
        )

    @property
    def n_free_parameters(self) -> int:
        return len(self.free_parameter_names)

    def _transform_parameter(self, parameter: Parameter, cube_value: Any) -> Any:
        if parameter.distribution is not None:
            return parameter.distribution.quantile(cube_value)

        if parameter.transform_prior_cube_coordinate is not None:
            return parameter.transform_prior_cube_coordinate(cube_value)

        raise ValueError(
            f"Free parameter '{parameter.name}' has neither a distribution nor a prior transform."
        )

    def _validate_free_parameter_count(self, values: npt.ArrayLike) -> np.ndarray:
        flat_values = _flatten_parameter_values(values)

        if flat_values.size != self.n_free_parameters:
            raise ValueError(
                f"Expected {self.n_free_parameters} free-parameter values, got {flat_values.size}"
            )

        return flat_values

    def prior_transform_numpy(self, cube: npt.ArrayLike) -> np.ndarray:
        cube_values = np.asarray(cube, dtype=float)

        if cube_values.ndim == 2:
            if cube_values.shape[1] != self.n_free_parameters:
                raise ValueError(
                    f"Expected cube positions with {self.n_free_parameters} free parameters, got {cube_values.shape[1]}"
                )

            return np.stack([self.prior_transform_numpy(row) for row in cube_values], axis=0)

        cube_values = self._validate_free_parameter_count(cube_values)
        transformed = np.empty_like(cube_values, dtype=float)

        for index, parameter_name in enumerate(self.free_parameter_names):
            parameter = self.parameter_templates[self.parameter_names.index(parameter_name)]
            transformed[index] = np.asarray(self._transform_parameter(parameter, cube_values[index]), dtype=float)

        return transformed

    def prior_transform_inplace(self, cube: npt.ArrayLike) -> npt.ArrayLike:
        transformed = self.prior_transform_numpy(cube)
        for index, value in enumerate(transformed):
            cube[index] = value
        return cube

    def transform_cube_to_physical(self, cube_position: Any) -> jnp.ndarray:
        cube_position = jnp.asarray(cube_position, dtype=JAX_DTYPE)
        transformed = []

        for index, free_parameter_name in enumerate(self.free_parameter_names):
            parameter = self.parameter_templates[self.parameter_names.index(free_parameter_name)]
            transformed.append(jnp.asarray(self._transform_parameter(parameter, cube_position[index]), dtype=JAX_DTYPE))

        if not transformed:
            return jnp.zeros((0,), dtype=JAX_DTYPE)

        return jnp.stack(transformed)

    def cube_from_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        unconstrained_position = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        return jnp.clip(
            jax.nn.sigmoid(unconstrained_position),
            self.cube_epsilon,
            1.0 - self.cube_epsilon,
        )

    def transform_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        return self.transform_cube_to_physical(self.cube_from_unconstrained(unconstrained_position))

    def log_jacobian_single(self, unconstrained_position: Any) -> jnp.ndarray:
        unconstrained_position = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        return jnp.sum(
            -jax.nn.softplus(-unconstrained_position) - jax.nn.softplus(unconstrained_position)
        )

    def log_jacobian(self, unconstrained_positions: Any) -> jnp.ndarray:
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)

        if jnp.ndim(unconstrained_positions) == 1:
            return self.log_jacobian_single(unconstrained_positions)

        return jax.vmap(self.log_jacobian_single)(unconstrained_positions)

    def physical_params_from_vector(self, values: npt.ArrayLike) -> PhysicalParams:
        flat_values = self._validate_free_parameter_count(values)
        return self.physical_params_from_free_values(tuple(flat_values))

    def physical_params_from_free_values(self, values: Any) -> PhysicalParams:
        if len(values) != self.n_free_parameters:
            raise ValueError(
                f"Expected {self.n_free_parameters} free-parameter values, got {len(values)}"
            )

        physical_mapping = {}
        free_index = 0

        for parameter_name, parameter in zip(self.parameter_names, self.parameter_templates, strict=True):
            if parameter.is_free_parameter:
                physical_mapping[parameter_name] = values[free_index]
                free_index += 1
            else:
                physical_mapping[parameter_name] = parameter.value

        return PhysicalParams.from_mapping(physical_mapping)

    def physical_params_from_cube(self, cube_position: npt.ArrayLike) -> PhysicalParams:
        return self.physical_params_from_vector(self.prior_transform_numpy(cube_position))

    def physical_params_from_unconstrained(self, unconstrained_position: Any) -> PhysicalParams:
        transformed = self.transform_unconstrained(unconstrained_position)
        return self.physical_params_from_vector(np.asarray(transformed))

    def free_parameter_vector(self, physical_params: PhysicalParams) -> jnp.ndarray:
        mapping = physical_params.as_dict()
        return jnp.asarray([mapping[name] for name in self.free_parameter_names], dtype=JAX_DTYPE)

    def build_legacy_parameter_dict(self, physical_params: PhysicalParams) -> dict[str, Parameter]:
        mapping = physical_params.as_dict()
        return {
            parameter_name: _clone_parameter(parameter, mapping[parameter_name])
            for parameter_name, parameter in zip(self.parameter_names, self.parameter_templates, strict=True)
        }


__all__ = [
    'LikelihoodState',
    'ModelContext',
    'ModelGroupState',
    'ModelInputs',
    'ModelResult',
    'ObservationState',
    'ParameterLayout',
    'PhysicalParams',
    'RetrievalRuntime',
]
