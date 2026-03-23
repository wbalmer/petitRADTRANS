"""Runtime data model and evaluation helpers for retrievals.

This module defines the runtime-facing representation used by the retrieval
package when executing sampler evaluations, especially for JAX-compatible and
sampler-agnostic workflows.

The core responsibilities of the module are:

- represent parameter values in a pytree-friendly form
- normalize observations and likelihood inputs into immutable runtime state
- group shared model evaluations across observations
- evaluate retrieval likelihoods through scalar, JAX, and batched JAX paths
- translate between sampler coordinates and the legacy ``Parameter`` objects

The classes in this file are intentionally small, immutable containers with a
few conversion helpers. Together they provide the bridge between the legacy
retrieval configuration objects and the newer runtime-native execution model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import logging
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


logger = logging.getLogger(__name__)


def _flatten_parameter_values(
    values: npt.ArrayLike,
    expected_size: int | None = None,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.reshape(-1)

    if hasattr(values, 'tolist'):
        converted_values = values.tolist()
        if isinstance(converted_values, list):
            return np.array(converted_values, dtype=float).reshape(-1)

    if expected_size is not None:
        try:
            return np.array([values[index] for index in range(expected_size)], dtype=float)
        except (IndexError, TypeError, ValueError):
            pass

    try:
        return np.array(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        pass

    try:
        return np.array([values[index] for index in range(len(values))], dtype=float)
    except (TypeError, ValueError):
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


def _serialize_runtime_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_runtime_metadata_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, tuple):
        return [_serialize_runtime_metadata_value(item) for item in value]

    if isinstance(value, list):
        return [_serialize_runtime_metadata_value(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if callable(value):
        module = getattr(value, '__module__', '__callable__')
        qualname = getattr(value, '__qualname__', repr(value))
        return f'{module}.{qualname}'

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def _configuration_requires_radtrans_initialization(configuration: Any) -> bool:
    for data in getattr(configuration, 'data', {}).values():
        if getattr(data, 'external_radtrans_reference', None) is None and getattr(data, 'radtrans_object', None) is None:
            return True

    return False


def _ensure_configuration_radtrans_objects_initialized(configuration: Any) -> None:
    if not _configuration_requires_radtrans_initialization(configuration):
        return

    from petitRADTRANS.config.configuration import petitradtrans_config_parser
    from petitRADTRANS.retrieval.retrieval import Retrieval

    retrieval = object.__new__(Retrieval)
    retrieval.configuration = configuration
    retrieval.path = petitradtrans_config_parser.get_input_data_path()

    Retrieval.initialise_radtrans_objects(retrieval)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PhysicalParams:
    """Immutable parameter values used by the runtime evaluator.

    The runtime stores free and fixed parameter values separately so that the
    dynamic portion can participate cleanly in JAX pytree flattening while the
    original parameter ordering remains preserved.
    """

    ordered_names: tuple[str, ...]
    dynamic_names: tuple[str, ...]
    dynamic_values: tuple[Any, ...]
    static_items: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> 'PhysicalParams':
        """Build runtime parameters from an ordered name-to-value mapping."""
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
        """Return the pytree representation used by JAX transforms."""
        children = self.dynamic_values
        aux_data = {
            'ordered_names': self.ordered_names,
            'dynamic_names': self.dynamic_names,
            'static_items': self.static_items,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a ``PhysicalParams`` instance from pytree parts."""
        return cls(
            ordered_names=aux_data['ordered_names'],
            dynamic_names=aux_data['dynamic_names'],
            dynamic_values=tuple(children),
            static_items=aux_data['static_items'],
        )

    def as_dict(self) -> dict[str, Any]:
        """Return all parameters as a plain dictionary in original order."""
        dynamic_mapping = dict(zip(self.dynamic_names, self.dynamic_values, strict=True))
        static_mapping = dict(self.static_items)
        return {
            name: dynamic_mapping[name] if name in dynamic_mapping else static_mapping[name]
            for name in self.ordered_names
        }

    def get(self, name: str, default: Any = None) -> Any:
        """Return a parameter value by name with an optional default."""
        return self.as_dict().get(name, default)

    def ordered_items(self) -> tuple[tuple[str, Any], ...]:
        """Return ordered ``(name, value)`` pairs for all parameters."""
        mapping = self.as_dict()
        return tuple((name, mapping[name]) for name in self.ordered_names)


@dataclass(frozen=True)
class ObservationState:
    """Immutable observation description used by the runtime.

    This mirrors the pieces of a retrieval ``Data`` object that are needed when
    evaluating models and likelihoods, while stripping away mutable behavior.
    """

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
    """Precomputed likelihood inputs associated with one observation.

    The runtime stores the uncertainty representation separately from the raw
    observation so samplers can reuse the same normalized bookkeeping across
    repeated evaluations.
    """

    data_uncertainties: Any = None
    data_covariance: Any = None
    mask: Any = None
    inv_covariance: Any = None
    log_covariance_determinant: Any = None
    n_free_parameters: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelContext:
    """Execution context passed to runtime-native model functions.

    A runtime-native model receives this context instead of the legacy mutable
    ``Parameter`` dictionary plus a collection of loosely coupled flags.
    """

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
    """Structured physical inputs consumed by low-level spectrum calculators.

    Runtime-native model builders populate this container once they have
    derived thermodynamic, chemical, and cloud state from the current
    parameters.
    """

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
        """Return the pytree representation used by JAX transforms."""
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
        """Reconstruct a ``ModelInputs`` instance from pytree parts."""
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
    """Normalized output emitted by runtime model evaluation.

    The result can represent a valid spectrum, a pressure-temperature profile,
    or an invalid evaluation. Named fields replace the legacy tuple protocol so
    downstream code can handle auxiliary outputs consistently.
    """

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
    """Description of one shared model evaluation group.

    Multiple observations can reuse the same underlying model call when they
    point at the same radiative-transfer source. The runtime tracks that shared
    evaluation through this grouping object.
    """

    source_name: str
    observation_names: tuple[str, ...]
    model_contract: str
    model_generating_function: Any = None
    radtrans_object: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRuntime:
    """Immutable runtime view of a retrieval configuration.

    ``RetrievalRuntime`` centralizes the normalized observation graph, model
    grouping, and likelihood evaluation code used by scalar and JAX-backed
    samplers.
    """

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
        """Build a runtime from retrieval data objects.

        This normalizes observation state, resolves model contracts, and groups
        observations that share a single model evaluation.
        """
        from petitRADTRANS.retrieval.models import coerce_model_generating_function

        if n_free_parameters is None:
            n_free_parameters = parameter_layout.n_free_parameters

        observations = {}
        for data_name, data_object in data.items():
            observation = data_object.to_observation_state()
            model_contract, model_generating_function = coerce_model_generating_function(
                observation.model_generating_function,
                observation.model_contract,
            )
            observations[data_name] = replace(
                observation,
                model_generating_function=model_generating_function,
                model_contract=model_contract,
            )

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
        parameter_layout: ParameterLayout | None = None,
        n_free_parameters: int | None = None,
    ) -> 'RetrievalRuntime':
        """Build a runtime directly from a retrieval configuration object."""
        _ensure_configuration_radtrans_objects_initialized(configuration)

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
        from petitRADTRANS.retrieval.models import (
            MODEL_CONTRACT_LEGACY,
            run_legacy_model_adapter,
            run_runtime_model_adapter,
        )

        model_context = self._build_model_context(configuration, group, self.parameter_layout.build_legacy_parameter_dict(physical_params))

        if group.model_contract != MODEL_CONTRACT_LEGACY:
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
        data_spectrum = data.apply_flux_scaling(data_spectrum, parameters)
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
        """Evaluate the posterior contribution for one parameter point.

        This path is primarily used by non-JAX samplers and debugging code.
        It returns the log likelihood plus any runtime-provided prior weight.
        """
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

            if model_result.auxiliary_outputs:
                log_prior += model_result.auxiliary_outputs.get('log_prior_weight', 0.0)

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
        """Evaluate the posterior through a JAX-compatible execution path.

        The structure mirrors :meth:`evaluate_scalar`, but keeps the control
        flow and array operations compatible with JAX transformations.
        """
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

            if model_result.auxiliary_outputs:
                log_prior += model_result.auxiliary_outputs.get('log_prior_weight', 0.0)

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
        """Evaluate one or many parameter vectors with JAX vectorization.

        A one-dimensional input is evaluated as a single sample. Higher-rank
        inputs are mapped over the leading dimension with ``jax.vmap``.
        """
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

    def _project_dataset_observation(
        self,
        data_name: str,
        data: Any,
        parameters: Mapping[str, Any],
        wavelengths_model: Any,
        spectrum_model: Any,
    ) -> Any:
        """Project one model spectrum into the observation space of a dataset."""
        from petitRADTRANS.retrieval.utils import (
            ensure_jax_array,
            extract_spectrum_parameters,
            process_spectrum_pipeline,
        )

        parameter_dict = dict(parameters)

        if data.photometry:
            projected = data.photometric_transformation_function(wavelengths_model, spectrum_model)
            if isinstance(projected, (tuple, list)):
                projected = projected[0]
            return ensure_jax_array(projected)

        spectrum_params = extract_spectrum_parameters(parameter_dict, data_name)
        projected = process_spectrum_pipeline(
            wavelengths_model,
            spectrum_model,
            data,
            parameter_dict,
            data_name,
            spectrum_params,
        )
        return ensure_jax_array(projected)

    def simulate_observations_scalar(
        self,
        configuration: Any,
        physical_params: PhysicalParams,
    ) -> dict[str, dict[str, Any]]:
        """Return deterministic observation-space model outputs for one parameter point."""
        parameters = self.parameter_layout.build_legacy_parameter_dict(physical_params)
        simulated_observations: dict[str, dict[str, Any]] = {}
        parameter_payload = {
            name: float(value)
            for name, value in zip(
                self.parameter_layout.free_parameter_names,
                np.asarray(self.parameter_layout.free_parameter_vector(physical_params), dtype=float),
                strict=True,
            )
        }

        for group in self.model_groups:
            model_result = self._evaluate_model_group(configuration, group, physical_params)

            if model_result.kind == 'invalid' or model_result.spectrum is None:
                raise ValueError(
                    f"Model evaluation for source '{group.source_name}' returned an invalid result."
                )

            for observation_name in group.observation_names:
                data = configuration.data[observation_name]
                projected_values = self._project_dataset_observation(
                    observation_name,
                    data,
                    parameters,
                    model_result.wavelengths,
                    model_result.spectrum,
                )
                projected_values = data.apply_flux_scaling(projected_values, parameters)
                projected_values = data.offset_flux(projected_values, parameters)
                projected_values_array = np.asarray(projected_values, dtype=float)
                finite_mask = np.isfinite(projected_values_array)
                if not finite_mask.all():
                    non_finite_count = int(finite_mask.size - finite_mask.sum())
                    logger.error(
                        "Non-finite projected observation values detected for source '%s' and dataset '%s' "
                        "(non_finite_count=%d, total_values=%d, parameters=%s)",
                        group.source_name,
                        observation_name,
                        non_finite_count,
                        int(finite_mask.size),
                        parameter_payload,
                    )
                    raise ValueError(
                        "Projected observation values contain non-finite entries for "
                        f"source '{group.source_name}' and dataset '{observation_name}'."
                    )
                projected_uncertainties, projected_covariance = data.scale_and_inflate_uncertainties(parameters)
                simulated_observations[observation_name] = {
                    'values': projected_values,
                    'uncertainties': projected_uncertainties,
                    'covariance': projected_covariance,
                    'mask': data.mask,
                    'coordinates': data.wavelengths,
                    'metadata': {
                        'photometry': data.photometry,
                        'source_name': group.source_name,
                        'line_opacity_mode': data.line_opacity_mode,
                        'scale_flux': data.scale_flux,
                        'scale_uncertainties': data.scale_uncertainties,
                        'fit_flux_offset': data.fit_flux_offset,
                    },
                }

        return simulated_observations


@dataclass(frozen=True)
class ParameterLayout:
    """Translation layer between sampler coordinates and runtime parameters.

    ``ParameterLayout`` owns the ordered parameter metadata needed to transform
    from unit-cube or unconstrained sampler coordinates into physical values,
    and to reconstruct legacy ``Parameter`` dictionaries when required.
    """

    parameter_names: tuple[str, ...]
    free_parameter_names: tuple[str, ...]
    free_parameter_indices: tuple[int, ...]
    parameter_templates: tuple[Parameter, ...]
    cube_epsilon: float = field(default_factory=lambda: float(1e-7 if JAX_DTYPE == jnp.float32 else 1e-12))

    @classmethod
    def from_configuration(cls, configuration: Any) -> 'ParameterLayout':
        """Build a parameter layout from a retrieval configuration."""
        return cls.from_parameters(configuration.parameters)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Parameter]) -> 'ParameterLayout':
        """Build a parameter layout from an ordered parameter mapping."""
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
        """Return the number of free parameters in the layout."""
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
        flat_values = _flatten_parameter_values(values, expected_size=self.n_free_parameters)

        if flat_values.size != self.n_free_parameters:
            raise ValueError(
                f"Expected {self.n_free_parameters} free-parameter values, got {flat_values.size}"
            )

        return flat_values

    def prior_transform_numpy(self, cube: npt.ArrayLike) -> np.ndarray:
        """Transform unit-cube samples into physical parameter values.

        Both single samples and batched samples are supported. The return value
        is always a NumPy array to keep non-JAX sampler interfaces simple.
        """
        if isinstance(cube, np.ndarray):
            cube_values = cube.astype(float, copy=False)
        elif isinstance(cube, (list, tuple)):
            cube_values = np.asarray(cube, dtype=float)
        elif hasattr(cube, 'tolist'):
            converted_cube = cube.tolist()
            if isinstance(converted_cube, list):
                cube_values = np.asarray(converted_cube, dtype=float)
            else:
                cube_values = self._validate_free_parameter_count(cube)
        else:
            cube_values = self._validate_free_parameter_count(cube)

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
        """Apply :meth:`prior_transform_numpy` and write the result back in place."""
        transformed = self.prior_transform_numpy(cube)
        for index, value in enumerate(transformed):
            cube[index] = value
        return cube

    def transform_cube_to_physical(self, cube_position: Any) -> jnp.ndarray:
        """Transform one unit-cube position into JAX physical coordinates."""
        cube_position = jnp.asarray(cube_position, dtype=JAX_DTYPE)
        transformed = []

        for index, free_parameter_name in enumerate(self.free_parameter_names):
            parameter = self.parameter_templates[self.parameter_names.index(free_parameter_name)]
            transformed.append(jnp.asarray(self._transform_parameter(parameter, cube_position[index]), dtype=JAX_DTYPE))

        if not transformed:
            return jnp.zeros((0,), dtype=JAX_DTYPE)

        return jnp.stack(transformed)

    def transform_cube_to_physical_batch(self, cube_positions: Any) -> jnp.ndarray:
        """Transform one or many unit-cube positions into physical coordinates."""
        cube_positions = jnp.asarray(cube_positions, dtype=JAX_DTYPE)

        if jnp.ndim(cube_positions) == 1:
            return self.transform_cube_to_physical(cube_positions)[jnp.newaxis, :]

        return jax.vmap(self.transform_cube_to_physical)(cube_positions)

    def cube_from_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        """Map unconstrained coordinates onto the open unit cube."""
        unconstrained_position = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        return jnp.clip(
            jax.nn.sigmoid(unconstrained_position),
            self.cube_epsilon,
            1.0 - self.cube_epsilon,
        )

    def cube_from_unconstrained_batch(self, unconstrained_positions: Any) -> jnp.ndarray:
        """Map one or many unconstrained positions onto the open unit cube."""
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)
        return jnp.clip(
            jax.nn.sigmoid(unconstrained_positions),
            self.cube_epsilon,
            1.0 - self.cube_epsilon,
        )

    def transform_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        """Transform unconstrained coordinates directly into physical values."""
        return self.transform_cube_to_physical(self.cube_from_unconstrained(unconstrained_position))

    def transform_unconstrained_batch(self, unconstrained_positions: Any) -> jnp.ndarray:
        """Transform one or many unconstrained positions into physical values."""
        return self.transform_cube_to_physical_batch(self.cube_from_unconstrained_batch(unconstrained_positions))

    def log_jacobian_single(self, unconstrained_position: Any) -> jnp.ndarray:
        """Return the log-Jacobian term for one unconstrained position."""
        unconstrained_position = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        return jnp.sum(
            -jax.nn.softplus(-unconstrained_position) - jax.nn.softplus(unconstrained_position)
        )

    def log_jacobian(self, unconstrained_positions: Any) -> jnp.ndarray:
        """Return log-Jacobian terms for one or many unconstrained positions."""
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)

        if jnp.ndim(unconstrained_positions) == 1:
            return self.log_jacobian_single(unconstrained_positions)

        return jax.vmap(self.log_jacobian_single)(unconstrained_positions)

    def physical_params_from_vector(self, values: npt.ArrayLike) -> PhysicalParams:
        """Build ``PhysicalParams`` from a flat vector of free values."""
        flat_values = self._validate_free_parameter_count(values)
        return self.physical_params_from_free_values(tuple(flat_values))

    def physical_params_from_matrix(self, values: npt.ArrayLike) -> tuple[PhysicalParams, ...]:
        """Build ``PhysicalParams`` objects from a batch of free-parameter vectors."""
        values = np.asarray(values, dtype=float)

        if values.ndim == 1:
            return (self.physical_params_from_vector(values),)

        if values.shape[1] != self.n_free_parameters:
            raise ValueError(
                f"Expected value matrix with {self.n_free_parameters} free parameters, got {values.shape[1]}"
            )

        return tuple(self.physical_params_from_vector(row) for row in values)

    def physical_params_from_free_values(self, values: Any) -> PhysicalParams:
        """Build ``PhysicalParams`` from ordered free-parameter values."""
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
        """Build ``PhysicalParams`` from a unit-cube position."""
        return self.physical_params_from_vector(self.prior_transform_numpy(cube_position))

    def physical_params_from_unconstrained(self, unconstrained_position: Any) -> PhysicalParams:
        """Build ``PhysicalParams`` from unconstrained sampler coordinates."""
        transformed = self.transform_unconstrained(unconstrained_position)
        return self.physical_params_from_vector(np.asarray(transformed))

    def free_parameter_vector(self, physical_params: PhysicalParams) -> jnp.ndarray:
        """Extract the ordered free-parameter vector from ``PhysicalParams``."""
        mapping = physical_params.as_dict()
        return jnp.asarray([mapping[name] for name in self.free_parameter_names], dtype=JAX_DTYPE)

    def build_legacy_parameter_dict(self, physical_params: PhysicalParams) -> dict[str, Parameter]:
        """Reconstruct legacy ``Parameter`` objects for compatibility paths."""
        mapping = physical_params.as_dict()
        return {
            parameter_name: _clone_parameter(parameter, mapping[parameter_name])
            for parameter_name, parameter in zip(self.parameter_names, self.parameter_templates, strict=True)
        }

    def serialize_metadata(self) -> dict[str, Any]:
        """Return a deterministic metadata payload describing the layout."""
        parameter_metadata = []

        for parameter_name, parameter in zip(self.parameter_names, self.parameter_templates, strict=True):
            distribution = parameter.distribution
            distribution_name = None
            distribution_parameters = None

            if distribution is not None:
                distribution_name = getattr(distribution, 'name', distribution.__class__.__name__)
                distribution_parameters = _serialize_runtime_metadata_value(
                    getattr(distribution, 'parameters', None)
                )

            parameter_metadata.append(
                {
                    'name': parameter_name,
                    'is_free_parameter': parameter.is_free_parameter,
                    'distribution_name': distribution_name,
                    'distribution_parameters': distribution_parameters,
                    'transform_prior_cube_coordinate': _serialize_runtime_metadata_value(
                        parameter.transform_prior_cube_coordinate
                    ),
                    'value': _serialize_runtime_metadata_value(parameter.value),
                }
            )

        return {
            'parameter_names': list(self.parameter_names),
            'free_parameter_names': list(self.free_parameter_names),
            'free_parameter_indices': list(self.free_parameter_indices),
            'cube_epsilon': float(self.cube_epsilon),
            'parameters': parameter_metadata,
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
