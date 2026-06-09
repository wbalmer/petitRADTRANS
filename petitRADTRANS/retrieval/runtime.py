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

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, core as jax_core, lax, vmap, nn, tree_util, debug

from petitRADTRANS.retrieval.parameter import JAX_DTYPE, Parameter
from petitRADTRANS.retrieval.utils import _get_parameter_value

if TYPE_CHECKING:
    from petitRADTRANS.radtrans import Radtrans


logger = logging.getLogger(__name__)

# Sentinel value returned for invalid model evaluations.  Use the most
# negative finite value representable in the active JAX dtype so that the
# sentinel is safely representable in both float32 and float64 without
# overflowing to -inf.
INVALID_LOG_LIKELIHOOD = jnp.asarray(jnp.finfo(JAX_DTYPE).min, dtype=JAX_DTYPE)


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
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PhysicalParams":
        """Build runtime parameters from an ordered name-to-value mapping."""
        from petitRADTRANS.retrieval.parameter_aliases import normalize_parameter_mapping
        mapping = normalize_parameter_mapping(mapping)
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
            "ordered_names": self.ordered_names,
            "dynamic_names": self.dynamic_names,
            "static_items": self.static_items,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a ``PhysicalParams`` instance from pytree parts."""
        return cls(
            ordered_names=aux_data["ordered_names"],
            dynamic_names=aux_data["dynamic_names"],
            dynamic_values=tuple(children),
            static_items=aux_data["static_items"],
        )

    def as_dict(self) -> dict[str, Any]:
        """Return all parameters as a plain dictionary in original order."""
        dynamic_mapping = dict(
            zip(self.dynamic_names, self.dynamic_values, strict=True)
        )
        static_mapping = dict(self.static_items)
        return {
            name: (
                dynamic_mapping[name]
                if name in dynamic_mapping
                else static_mapping[name]
            )
            for name in self.ordered_names
        }

    def get(self, name: str, default: Any = None) -> Any:
        """Return a parameter value by name with an optional default."""
        if name in self.dynamic_names:
            return self.dynamic_values[self.dynamic_names.index(name)]
        for k, v in self.static_items:
            if k == name:
                return v
        return default

    def __getitem__(self, name: str) -> Any:
        """Return a parameter value by name, raising ``KeyError`` if missing."""
        if name in self.dynamic_names:
            return self.dynamic_values[self.dynamic_names.index(name)]
        for k, v in self.static_items:
            if k == name:
                return v
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        """Return ``True`` if *name* is a known parameter."""
        return name in self.ordered_names

    def keys(self):
        """Return the parameter names in original order."""
        return self.ordered_names

    def ordered_items(self) -> tuple[tuple[str, Any], ...]:
        """Return ordered ``(name, value)`` pairs for all parameters."""
        dynamic = dict(zip(self.dynamic_names, self.dynamic_values, strict=True))
        static = dict(self.static_items)
        return tuple(
            (n, dynamic[n] if n in dynamic else static[n]) for n in self.ordered_names
        )

    def as_legacy_dict(self) -> dict[str, Any]:
        """Return parameters wrapped as ``SimpleNamespace(value=...)`` objects.

        This provides backward compatibility with shared helper functions
        (chemistry, clouds, pressure initialization) that expect the legacy
        ``parameters['name'].value`` access pattern.
        """
        from types import SimpleNamespace

        return {
            name: SimpleNamespace(value=value) for name, value in self.ordered_items()
        }


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
    uncertainties: Any = None
    covariance: Any = None
    radtrans_object: Any = None
    model_generating_function: Any = None
    model_resolution: Any = None
    data_resolution: Any = None
    data_resolution_array_model: Any = None
    wavelength_bin_widths: Any = None
    external_radtrans_reference: str | None = None
    line_opacity_mode: str | None = None
    model_contract: str = "legacy"
    scale_flux: bool = False
    scale_uncertainties: bool = False
    fit_flux_offset: bool = False
    fit_instrumental_resolution: bool = False
    subtract_continuum: bool = False
    photometric_transformation_function: Any = None
    fit_covariance: bool = False
    covariance_mode: str = "none"
    global_covariance_kernel: str = "squared_exponential"
    local_covariance_kernel: str = "squared_exponential"
    n_local_covariance_kernels: int = 0
    covariance_jitter: float = 0.0
    variability_atmospheric_column_model_flux_return_mode: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhotometryObservationState(ObservationState):
    """ObservationState subtype used for photometric datasets."""


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
    covariance_cholesky_factor: Any = None
    log_covariance_determinant: Any = None
    log_likelihood_normalization: Any = None
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
    radtrans: "Radtrans"
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
    reference_pressure: Any = 0.1  # bar
    opaque_cloud_top_pressure: Any = None
    sigma_lnorm: Any = None
    cloud_particle_mean_radii: Any = None
    cloud_f_sed: Any = None
    eddy_diffusion_coefficients: Any = None
    haze_factor: Any = 1.0
    power_law_opacity_coefficient: Any = None
    power_law_opacity_350nm: Any = None
    cloud_hansen_b: Any = None
    cloud_fraction: Any = 1.0
    patchy_clouds: tuple[str, ...] | None = None
    cloud_particle_radius_distribution: str | None = None
    distance_to_system: Any = None
    stellar_radius: Any = None
    disk_blackbody_temperature: Any = None
    disk_radius: Any = None
    v_band_extinction: Any = None
    v_band_reddening: Any = None
    additional_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.patchy_clouds is None:
            normalized_patchy_clouds = None
        elif isinstance(self.patchy_clouds, tuple):
            normalized_patchy_clouds = self.patchy_clouds
        elif isinstance(self.patchy_clouds, (str, bytes)):
            normalized_patchy_clouds = (self.patchy_clouds,)
        else:
            normalized_patchy_clouds = tuple(self.patchy_clouds)

        object.__setattr__(
            self,
            "patchy_clouds",
            normalized_patchy_clouds,
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
            self.cloud_particle_mean_radii,
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
            "patchy_clouds": self.patchy_clouds,
            "cloud_particle_radius_distribution": self.cloud_particle_radius_distribution,
            "additional_inputs": self.additional_inputs,
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
            cloud_particle_mean_radii=children[9],
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
            patchy_clouds=aux_data["patchy_clouds"],
            cloud_particle_radius_distribution=aux_data[
                "cloud_particle_radius_distribution"
            ],
            additional_inputs=aux_data["additional_inputs"],
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelResult:
    """Normalized output emitted by runtime model evaluation.

    The result can represent a valid spectrum, a pressure-temperature profile,
    or an invalid evaluation. Named fields replace the legacy tuple protocol so
    downstream code can handle auxiliary outputs consistently.

    For JIT compatibility, invalid results carry NaN-filled spectrum arrays of
    the expected shape rather than ``None`` so that downstream tracing always
    sees consistent types.

    The class is registered as a JAX pytree so that ``jax.vmap`` and ``jax.jit``
    can propagate batched spectra through model evaluation pipelines.  The
    ``kind`` field and ``auxiliary_outputs`` keys are treated as static
    (aux_data); all array-valued fields — including the values inside
    ``auxiliary_outputs`` — are dynamic leaves.
    """

    kind: str
    wavelengths: Any = None
    spectrum: Any = None
    pressures: Any = None
    temperatures: Any = None
    additional_log_likelihood: Any = 0.0
    auxiliary_outputs: Mapping[str, Any] = field(default_factory=dict)
    atmospheric_column_fluxes: Any = None

    # ------------------------------------------------------------------
    # JAX pytree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        """Flatten for JAX pytree operations (jit, vmap, grad, …).

        ``kind`` and the keys of ``auxiliary_outputs`` go into ``aux_data``
        (static — must be identical across a vmap batch).  All array-valued
        fields, including the *values* of ``auxiliary_outputs``, are leaves.
        """
        aux_keys = tuple(self.auxiliary_outputs.keys())
        children = (
            self.wavelengths,
            self.spectrum,
            self.pressures,
            self.temperatures,
            self.additional_log_likelihood,
            self.atmospheric_column_fluxes,
            *(self.auxiliary_outputs[k] for k in aux_keys),
        )
        aux_data = {"kind": self.kind, "aux_keys": aux_keys}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a ``ModelResult`` from pytree parts."""
        aux_keys = aux_data["aux_keys"]
        n_base = 6
        return cls(
            kind=aux_data["kind"],
            wavelengths=children[0],
            spectrum=children[1],
            pressures=children[2],
            temperatures=children[3],
            additional_log_likelihood=children[4],
            atmospheric_column_fluxes=children[5],
            auxiliary_outputs=dict(zip(aux_keys, children[n_base:])),
        )

    # ------------------------------------------------------------------
    # JIT-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_nan_result(wavelengths: Any) -> "ModelResult":
        """Create an invalid result with NaN-filled spectrum of matching shape.

        ``wavelengths`` should be a JAX array (typically from the Radtrans
        wavelength grid).  The returned result has ``kind='invalid'`` for
        backward compatibility, but the all-NaN spectrum is the primary
        JIT-safe sentinel.
        """
        nan_spectrum = jnp.full_like(wavelengths, jnp.nan)
        return ModelResult(
            kind="invalid",
            wavelengths=wavelengths,
            spectrum=nan_spectrum,
        )

    def spectrum_is_valid(self) -> Any:
        """JIT-safe check: True when spectrum is a real, non-NaN array."""
        if self.spectrum is None:
            return False
        return ~jnp.isnan(self.spectrum).all()


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
    model_context: Any = (
        None  # pre-built ModelContext; populated by RetrievalRuntime.from_data
    )


@dataclass(frozen=True)
class RetrievalRuntime:
    """Immutable runtime view of a retrieval configuration.

    ``RetrievalRuntime`` centralizes the normalized observation graph, model
    grouping, and likelihood evaluation code used by scalar and JAX-backed
    samplers.
    """

    parameter_layout: "ParameterLayout"
    observations: Mapping[str, ObservationState] = field(default_factory=dict)
    likelihoods: Mapping[str, LikelihoodState] = field(default_factory=dict)
    model_groups: tuple[ModelGroupState, ...] = ()
    adaptive_mesh_refinement: bool = False

    @classmethod
    def from_data(
        cls,
        parameter_layout: "ParameterLayout",
        data: Mapping[str, Any],
        n_free_parameters: int | None = None,
        adaptive_mesh_refinement: bool = False,
    ) -> "RetrievalRuntime":
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
            model_contract, model_generating_function = (
                coerce_model_generating_function(
                    observation.model_generating_function,
                    observation.model_contract,
                )
            )
            observations[data_name] = replace(
                observation,
                model_generating_function=model_generating_function,
                model_contract=model_contract,
            )

        likelihoods = {
            data_name: data_object.to_likelihood_state(
                n_free_parameters=n_free_parameters,
                parameter_names=parameter_layout.parameter_names,
            )
            for data_name, data_object in data.items()
        }

        grouped_observations: dict[str, list[str]] = {}
        for data_name, observation in observations.items():
            source_name = observation.external_radtrans_reference or data_name
            grouped_observations.setdefault(source_name, []).append(data_name)

        from petitRADTRANS.retrieval.utils import get_calculate_flux_return_values

        # Compute diagnostic-output flags once from static parameter template values.
        # These flags only depend on which parameter *names* are present (never on
        # sampled values), so they can be resolved at construction time and stored
        # on each ModelGroupState rather than re-scanned on every evaluation.
        template_params = {
            name: template.value
            for name, template in zip(
                parameter_layout.parameter_names,
                parameter_layout.parameter_templates,
                strict=True,
            )
        }
        (
            _return_contribution,
            _return_opacities,
            _return_photosphere_radius,
            _return_rosseland_optical_depths,
            _return_radius_hydrostatic_equilibrium,
            _return_cloud_contribution,
            _return_abundances,
            _return_any,
        ) = get_calculate_flux_return_values(template_params)

        model_groups = []
        for source_name, observation_names in grouped_observations.items():
            source_observation = observations[source_name]
            # Merge source observation metadata so model functions can
            # access time-series configuration, observation times, etc.
            group_metadata = {
                "line_opacity_mode": source_observation.line_opacity_mode,
                "has_external_references": any(
                    observations[name].external_radtrans_reference is not None
                    for name in observation_names
                ),
            }
            obs_meta = source_observation.metadata or {}
            for key in (
                "observation_times",
                "time_series_config",
                "atmospheric_column_flux_mixer",
                "concatenate_flux_epochs_variability",
            ):
                if key in obs_meta and obs_meta[key] is not None:
                    group_metadata[key] = obs_meta[key]
            prebuilt_context = ModelContext(
                name=source_name,
                mode=group_metadata.get("mode", "unknown"),
                radtrans=source_observation.radtrans_object,
                adaptive_mesh_refinement=adaptive_mesh_refinement,
                variability_atmospheric_column_model_flux_return_mode=(
                    source_observation.variability_atmospheric_column_model_flux_return_mode
                ),
                return_contribution=_return_contribution,
                return_opacities=_return_opacities,
                return_photosphere_radius=_return_photosphere_radius,
                return_rosseland_optical_depths=_return_rosseland_optical_depths,
                return_radius_hydrostatic_equilibrium=_return_radius_hydrostatic_equilibrium,
                return_cloud_contribution=_return_cloud_contribution,
                return_abundances=_return_abundances,
                model_metadata=group_metadata,
            )
            model_groups.append(
                ModelGroupState(
                    source_name=source_name,
                    observation_names=tuple(observation_names),
                    model_contract=source_observation.model_contract,
                    model_generating_function=source_observation.model_generating_function,
                    radtrans_object=source_observation.radtrans_object,
                    metadata=group_metadata,
                    model_context=prebuilt_context,
                )
            )

        return cls(
            parameter_layout=parameter_layout,
            observations=observations,
            likelihoods=likelihoods,
            model_groups=tuple(model_groups),
            adaptive_mesh_refinement=adaptive_mesh_refinement,
        )

    @classmethod
    def from_configuration(
        cls,
        configuration: Any,
        parameter_layout: ParameterLayout | None = None,
        n_free_parameters: int | None = None,
    ) -> "RetrievalRuntime":
        """Build a runtime directly from a retrieval configuration object."""
        _ensure_configuration_radtrans_objects_initialized(configuration)

        if parameter_layout is None:
            parameter_layout = ParameterLayout.from_configuration(configuration)

        return cls.from_data(
            parameter_layout=parameter_layout,
            data=configuration.data,
            n_free_parameters=n_free_parameters,
            adaptive_mesh_refinement=getattr(
                configuration, "adaptive_mesh_refinement", False
            ),
        )

    def _evaluate_model_group(
        self,
        group: ModelGroupState,
        physical_params: PhysicalParams,
    ) -> ModelResult:
        from petitRADTRANS.retrieval.models import _normalize_model_output

        # All model functions are coerced at build time (via
        # coerce_model_generating_function) so they share the same
        # (model_context, physical_params, pt_plot_mode) calling convention.
        # model_context is pre-built on the group; no per-evaluation reconstruction.
        model_output = group.model_generating_function(
            group.model_context,
            physical_params,
            pt_plot_mode=False,
        )
        result = _normalize_model_output(model_output, pt_plot_mode=False)

        # Ensure the result always carries arrays so that downstream JAX
        # tracing sees consistent types.  Legacy models may still return
        # ``spectrum=None`` on failure; convert to NaN sentinel here.
        if result.spectrum is None:
            wavelengths = _radtrans_wavelength_grid(group.radtrans_object)
            result = ModelResult.make_nan_result(wavelengths)

        return result

    def _score_dataset(
        self,
        data_name: str,
        parameters: Mapping[str, Any],
        wavelengths_model: Any,
        spectrum_model: Any,
        additional_log_likelihood: Any,
        model_auxiliary_outputs: Mapping[str, Any] | None = None,
    ) -> float:
        """Score one observation against a model spectrum using ``LogLikelihood``.

        This uses the pre-built ``ObservationState`` stored on the runtime
        rather than reaching back into mutable ``Data`` objects.

        The method decomposes the work into three composable stages:

        1. **Prepare observation data** — apply flux scaling, offset, and
           uncertainty inflation (``_apply_flux_scaling``, ``_offset_flux``,
           ``_scale_and_inflate_uncertainties``).
        2. **Project model** — convolve, rebin, and RV-shift the model onto the
           observation wavelength grid (``project_model_to_observation_space``).
        3. **Compute log-likelihood** — delegate to
           ``compute_observation_log_likelihood`` which handles NaN-sentinel
           detection internally.

        Multi-dimensional observations (2-D orders, 3-D detectors×orders, and
        object-dtype ragged arrays) iterate over the static outer dimensions
        before invoking the same composable primitives.
        """
        observation = self.observations[data_name]
        likelihood_state = self.likelihoods[data_name]
        n_free = self.parameter_layout.n_free_parameters

        # --- Stage 1: prepare observation data ---
        data_spectrum = observation.spectrum
        data_spectrum = _apply_flux_scaling(data_spectrum, observation, parameters)
        data_spectrum = _offset_flux(data_spectrum, observation, parameters)
        data_uncertainties, data_covariance = _scale_and_inflate_uncertainties(
            observation, parameters
        )
        covariance_cholesky_factor = likelihood_state.covariance_cholesky_factor
        log_likelihood_normalization = likelihood_state.log_likelihood_normalization

        if covariance_cholesky_factor is not None:
            data_covariance = None

        if isinstance(spectrum_model, np.ma.MaskedArray):
            spectrum_model = spectrum_model.filled(0)

        # --- Object-dtype spectra (ragged arrays — e.g. time series) ---
        if (
            not isinstance(observation.spectrum, float)
            and hasattr(observation.spectrum, "dtype")
            and observation.spectrum.dtype == "O"
        ):
            log_likelihood = 0.0
            if jnp.ndim(observation.spectrum) == 1:
                for i, spectrum_data in enumerate(observation.spectrum):
                    masked_model = spectrum_model[i][~observation.mask[i]]
                    log_likelihood += compute_observation_log_likelihood(
                        masked_model,
                        spectrum_data,
                        observation.uncertainties[i],
                        None,
                        n_free,
                    )
            elif jnp.ndim(observation.spectrum) == 2:
                for i, detector in enumerate(observation.spectrum):
                    for j, spectrum_data in enumerate(detector):
                        masked_model = spectrum_model[i, j][~observation.mask[i, j]]
                        log_likelihood += compute_observation_log_likelihood(
                            masked_model,
                            spectrum_data,
                            observation.uncertainties[i, j],
                            None,
                            n_free,
                        )
            else:
                raise ValueError(
                    f"observation is an array containing object, and has "
                    f"{jnp.ndim(observation.spectrum)} dimensions, but must have 1 to 2"
                )
            return log_likelihood + additional_log_likelihood

        # --- Standard 1-D spectrum path (primary JIT target) ---
        if (
            isinstance(observation.spectrum, float)
            or jnp.ndim(observation.spectrum) == 1
        ) and jnp.ndim(spectrum_model) == 1:
            # Stage 2: project model into observation space
            model_processed = project_model_to_observation_space(
                wavelengths_model,
                spectrum_model,
                observation,
                parameters,
                data_name,
            )
            # Stage 3: compute log-likelihood
            ll = compute_observation_log_likelihood(
                model_processed,
                data_spectrum,
                data_uncertainties,
                data_covariance,
                n_free,
                covariance_cholesky_factor=covariance_cholesky_factor,
                log_likelihood_normalization=log_likelihood_normalization,
            )
            return jnp.where(
                ll <= INVALID_LOG_LIKELIHOOD,
                INVALID_LOG_LIKELIHOOD,
                ll + additional_log_likelihood,
            )

        # --- 1-D data, higher-dim model (take leading slice) ---
        if jnp.ndim(observation.spectrum) == 1 and jnp.ndim(spectrum_model) == 2:
            ll = compute_observation_log_likelihood(
                spectrum_model[0, :],
                data_spectrum,
                data_uncertainties,
                data_covariance,
                n_free,
                covariance_cholesky_factor=covariance_cholesky_factor,
                log_likelihood_normalization=log_likelihood_normalization,
            )
            return jnp.where(
                ll <= INVALID_LOG_LIKELIHOOD,
                INVALID_LOG_LIKELIHOOD,
                ll + additional_log_likelihood,
            )

        if jnp.ndim(observation.spectrum) == 1 and jnp.ndim(spectrum_model) == 3:
            ll = compute_observation_log_likelihood(
                spectrum_model[0, 0, :],
                data_spectrum,
                data_uncertainties,
                data_covariance,
                n_free,
                covariance_cholesky_factor=covariance_cholesky_factor,
                log_likelihood_normalization=log_likelihood_normalization,
            )
            return jnp.where(
                ll <= INVALID_LOG_LIKELIHOOD,
                INVALID_LOG_LIKELIHOOD,
                ll + additional_log_likelihood,
            )

        # --- 2-D spectrum (e.g. multi-order spectroscopy or time series) ---
        if jnp.ndim(observation.spectrum) == 2:
            if (
                jnp.ndim(observation.wavelengths) == 1
                and data_covariance is None
                and data_uncertainties is not None
            ):
                projected_model = None

                if model_auxiliary_outputs is not None:
                    model_times = model_auxiliary_outputs.get("model_times")

                    if model_times is not None and jnp.ndim(spectrum_model) == 2:
                        projected_model = project_model_time_series_to_observation_space(
                            wavelengths_model,
                            spectrum_model,
                            model_times,
                            observation,
                            parameters,
                            data_name,
                        )

                if projected_model is None:
                    if jnp.ndim(spectrum_model) == 1:
                        spectrum_model = jnp.broadcast_to(
                            spectrum_model,
                            observation.spectrum.shape[:-1] + (spectrum_model.shape[-1],),
                        )

                    project_epoch = lambda spectrum_epoch: project_model_to_observation_space(
                        wavelengths_model,
                        spectrum_epoch,
                        observation,
                        parameters,
                        data_name,
                    )

                    if _uses_variable_resolution_convolution(observation):
                        projected_model = lax.map(project_epoch, spectrum_model)
                    else:
                        projected_model = vmap(project_epoch)(spectrum_model)

                ll = compute_diagonal_observation_log_likelihood(
                    projected_model,
                    data_spectrum,
                    data_uncertainties,
                    mask=observation.mask,
                )

                return jnp.where(
                    ll <= INVALID_LOG_LIKELIHOOD,
                    INVALID_LOG_LIKELIHOOD,
                    ll + additional_log_likelihood,
                )

            log_likelihood = jnp.asarray(0.0, dtype=JAX_DTYPE)

            if jnp.ndim(spectrum_model) == 1:
                spectrum_model = jnp.broadcast_to(
                    spectrum_model,
                    observation.spectrum.shape[:-1] + (spectrum_model.shape[-1],),
                )

            for i, spectrum_data in enumerate(observation.spectrum):
                epoch_observation = _slice_observation_epoch(observation, i)

                # Project each epoch/order model spectrum into observation space
                model_epoch = project_model_to_observation_space(
                    wavelengths_model,
                    spectrum_model[i],
                    epoch_observation,
                    parameters,
                    data_name,
                )

                if epoch_observation.uncertainties is not None:
                    log_likelihood += compute_diagonal_observation_log_likelihood(
                        model_epoch,
                        spectrum_data,
                        epoch_observation.uncertainties,
                        mask=epoch_observation.mask,
                    )
                else:
                    log_likelihood += compute_observation_log_likelihood(
                        model_epoch,
                        spectrum_data,
                        epoch_observation.uncertainties,
                        epoch_observation.covariance,
                        n_free,
                    )
            return log_likelihood + additional_log_likelihood

        # --- 3-D spectrum (e.g. multi-detector multi-order) ---
        if jnp.ndim(observation.spectrum) == 3:
            log_likelihood = jnp.asarray(0.0, dtype=JAX_DTYPE)
            for i, detector in enumerate(observation.spectrum):
                for j, spectrum_data in enumerate(detector):
                    masked_model = spectrum_model[i, j, ~observation.mask[i, j, :]]
                    log_likelihood += compute_observation_log_likelihood(
                        masked_model,
                        spectrum_data,
                        observation.uncertainties[i, j],
                        None,
                        n_free,
                    )
            return log_likelihood + additional_log_likelihood

        raise ValueError("unsupported spectrum dimensionality for runtime evaluation")

    def evaluate_scalar(
        self,
        physical_params: PhysicalParams,
        *,
        uncertainties_mode: str = "default",
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

        for group in self.model_groups:
            model_result = self._evaluate_model_group(group, physical_params)

            if not model_result.spectrum_is_valid():
                return invalid_value

            additional_log_likelihood = model_result.additional_log_likelihood
            wavelengths_model = model_result.wavelengths
            spectrum_model = model_result.spectrum

            for observation_name in group.observation_names:
                contribution = self._score_dataset(
                    observation_name,
                    physical_params,
                    wavelengths_model,
                    spectrum_model,
                    additional_log_likelihood,
                    model_auxiliary_outputs=model_result.auxiliary_outputs,
                )

                if contribution <= invalid_value:
                    return invalid_value

                log_likelihood += contribution

            if model_result.auxiliary_outputs:
                log_prior += model_result.auxiliary_outputs.get("log_prior_weight", 0.0)

        if "log_prior_weight" in physical_params:
            log_prior += physical_params["log_prior_weight"]

        if log_likelihood + log_prior < invalid_value:
            return invalid_value

        if jnp.abs(log_likelihood + log_prior) < tiny:
            log_likelihood = tiny
            log_prior = 0.0

        if print_log_likelihood_for_debugging:
            print("log_likelihood + log_prior", log_likelihood + log_prior)

        return log_likelihood + log_prior

    def evaluate_jax(
        self,
        physical_params: PhysicalParams,
        *,
        uncertainties_mode: str = "default",
        print_log_likelihood_for_debugging: bool = False,
    ) -> float:
        """Evaluate the posterior through a JAX-compatible execution path.

        All control flow uses ``lax.cond`` / ``jnp.where`` so the function
        is safe to pass through ``jax.jit`` and ``jax.grad``.  Invalid model
        evaluations are detected via the NaN-spectrum sentinel produced by
        ``ModelResult.make_nan_result`` rather than Python-level string
        comparisons.
        """
        invalid_value = INVALID_LOG_LIKELIHOOD
        tiny = jnp.finfo(JAX_DTYPE).tiny
        log_likelihood_contributions = []
        log_prior = jnp.asarray(0.0, dtype=JAX_DTYPE)
        has_invalid_contribution = jnp.asarray(False)

        for group in self.model_groups:
            model_result = self._evaluate_model_group(group, physical_params)

            # JIT-safe validity check — models return NaN-filled spectra on
            # failure instead of Python-level ``kind == 'invalid'``.
            model_valid = model_result.spectrum_is_valid()
            has_invalid_contribution = jnp.logical_or(
                has_invalid_contribution,
                ~model_valid,
            )

            group_contribution = jnp.asarray(0.0, dtype=JAX_DTYPE)
            for observation_name in group.observation_names:
                contribution = self._score_dataset(
                    observation_name,
                    physical_params,
                    model_result.wavelengths,
                    model_result.spectrum,
                    model_result.additional_log_likelihood,
                    model_auxiliary_outputs=model_result.auxiliary_outputs,
                )

                contribution_is_invalid = contribution <= invalid_value
                has_invalid_contribution = jnp.logical_or(
                    has_invalid_contribution, contribution_is_invalid
                )
                group_contribution = jnp.where(
                    contribution_is_invalid,
                    invalid_value,
                    group_contribution + contribution,
                )

            log_likelihood_contributions.append(group_contribution)

            # Always use .get() to avoid Python-level truthiness check on
            # the auxiliary_outputs dict during JIT tracing.
            aux = model_result.auxiliary_outputs or {}
            log_prior = log_prior + aux.get("log_prior_weight", 0.0)

        log_likelihood = jnp.sum(
            jnp.asarray(log_likelihood_contributions, dtype=JAX_DTYPE)
        )
        log_likelihood = jnp.where(
            has_invalid_contribution, invalid_value, log_likelihood
        )

        if "log_prior_weight" in physical_params:
            log_prior = log_prior + physical_params["log_prior_weight"]

        total = log_likelihood + log_prior

        total = jnp.where(total < invalid_value, invalid_value, total)
        total = jnp.where(
            jnp.abs(total) < tiny, jnp.asarray(tiny, dtype=JAX_DTYPE), total
        )

        if print_log_likelihood_for_debugging:
            debug.print("log_likelihood + log_prior {}", total)

        return total

    def evaluate_vectorized_jax(
        self,
        free_parameter_values: Any,
        *,
        uncertainties_mode: str = "default",
        print_log_likelihood_for_debugging: bool = False,
    ) -> Any:
        """Evaluate one or many parameter vectors with JAX vectorization.

        A one-dimensional input is evaluated as a single sample. Higher-rank
        inputs are mapped over the leading dimension with ``vmap``.
        """

        def evaluate_single(single_values: Any) -> Any:
            physical_params = self.parameter_layout.physical_params_from_free_values(
                single_values
            )
            return self.evaluate_jax(
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=print_log_likelihood_for_debugging,
            )

        free_parameter_values = jnp.asarray(free_parameter_values, dtype=JAX_DTYPE)

        if jnp.ndim(free_parameter_values) == 1:
            return evaluate_single(free_parameter_values)

        return vmap(evaluate_single)(free_parameter_values)

    def evaluate_jax_batched(
        self,
        free_parameter_values: Any,
        *,
        uncertainties_mode: str = "default",
        print_log_likelihood_for_debugging: bool = False,
    ) -> Any:
        """Evaluate a batch of parameter vectors by vmapping the scalar JAX path.

        For model groups using ``MODEL_CONTRACT_DIFFERENTIABLE`` this method
        applies ``jax.vmap`` to the single-sample :meth:`evaluate_jax` path.
        This keeps model generation and scoring adjacent inside each mapped
        sample, which avoids retaining a fully materialized batched
        ``ModelResult`` tree before likelihood accumulation.

        For non-native groups the method falls back to
        :meth:`evaluate_vectorized_jax`, which applies a single vmap over the
        full ``evaluate_jax`` pipeline.

        A one-dimensional input is handled as a single sample (no batching).
        Higher-rank inputs are mapped over the leading dimension.

        Args:
            free_parameter_values: Shape ``(n_samples, n_free_params)`` or
                ``(n_free_params,)`` for a single sample.

        Returns:
            Log-posterior contributions, shape ``(n_samples,)`` or scalar.
        """
        from petitRADTRANS.retrieval.models import MODEL_CONTRACT_DIFFERENTIABLE

        if jnp.ndim(free_parameter_values) == 1:
            physical_params = self.parameter_layout.physical_params_from_free_values(
                free_parameter_values
            )
            return self.evaluate_jax(
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=(
                    print_log_likelihood_for_debugging
                ),
            )

        all_native = all(
            g.model_contract == MODEL_CONTRACT_DIFFERENTIABLE for g in self.model_groups
        )
        if not all_native:
            return self.evaluate_vectorized_jax(
                free_parameter_values,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=(
                    print_log_likelihood_for_debugging
                ),
            )

        batched_physical_params = vmap(
            self.parameter_layout.physical_params_from_free_values
        )(free_parameter_values)

        return vmap(
            lambda physical_params_single: self.evaluate_jax(
                physical_params=physical_params_single,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=(
                    print_log_likelihood_for_debugging
                ),
            )
        )(batched_physical_params)

    def _project_dataset_observation(
        self,
        data_name: str,
        parameters: PhysicalParams,
        wavelengths_model: Any,
        spectrum_model: Any,
    ) -> Any:
        """Project one model spectrum into the observation space of a dataset."""
        from petitRADTRANS.retrieval.utils import (
            ensure_jax_array,
            extract_spectrum_parameters,
            process_spectrum_pipeline,
        )

        observation = self.observations[data_name]

        if isinstance(observation, PhotometryObservationState):
            projected = observation.photometric_transformation_function(
                wavelengths_model, spectrum_model
            )
            if isinstance(projected, (tuple, list)):
                projected = projected[0]
            return ensure_jax_array(projected)

        spectrum_params = extract_spectrum_parameters(parameters, data_name)
        projected = process_spectrum_pipeline(
            wavelengths_model,
            spectrum_model,
            observation,
            parameters,
            data_name,
            spectrum_params,
        )
        return ensure_jax_array(projected)


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
    cube_epsilon: float = field(
        default_factory=lambda: float(1e-7 if JAX_DTYPE == jnp.float32 else 1e-12)
    )
    # Per-free-parameter routing for the *unconstrained* (gradient-sampler) path.
    # Free parameters with a Normal prior use an affine map physical = loc + scale*z
    # with an implicit N(0, 1) prior on z (constant, O(1) curvature -- ideal for
    # NUTS/HMC). Every other free parameter keeps the bounded cube map
    # physical = quantile(sigmoid(z)). These tuples are aligned with
    # ``free_parameter_names``; empty tuples (legacy direct construction) disable
    # the affine path and fall back to the pure cube map. The cube path used by
    # nested samplers (``transform_cube_to_physical``) is unaffected.
    free_affine_mask: tuple[bool, ...] = ()
    free_affine_loc: tuple[float, ...] = ()
    free_affine_scale: tuple[float, ...] = ()

    @classmethod
    def from_configuration(cls, configuration: Any) -> "ParameterLayout":
        """Build a parameter layout from a retrieval configuration."""
        return cls.from_parameters(configuration.parameters)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Parameter]) -> "ParameterLayout":
        """Build a parameter layout from an ordered parameter mapping."""
        parameter_names = tuple(parameters.keys())
        parameter_templates = tuple(
            _clone_parameter(parameters[name], parameters[name].value)
            for name in parameter_names
        )
        free_parameter_names = tuple(
            name
            for name, parameter in zip(
                parameter_names, parameter_templates, strict=True
            )
            if parameter.is_free_parameter
        )
        free_parameter_indices = tuple(
            index
            for index, parameter in enumerate(parameter_templates)
            if parameter.is_free_parameter
        )

        # Route free parameters with a Normal prior onto the affine unconstrained
        # map (physical = loc + scale * z); everything else keeps the cube map.
        affine_mask = []
        affine_loc = []
        affine_scale = []
        for free_index in free_parameter_indices:
            parameter = parameter_templates[free_index]
            distribution = getattr(parameter, "distribution", None)
            is_normal = (
                distribution is not None
                and type(distribution).__name__ == "Normal"
                and hasattr(distribution, "loc")
                and hasattr(distribution, "scale")
            )
            if is_normal:
                affine_mask.append(True)
                affine_loc.append(float(np.asarray(distribution.loc)))
                affine_scale.append(float(np.asarray(distribution.scale)))
            else:
                affine_mask.append(False)
                affine_loc.append(0.0)
                affine_scale.append(1.0)

        return cls(
            parameter_names=parameter_names,
            free_parameter_names=free_parameter_names,
            free_parameter_indices=free_parameter_indices,
            parameter_templates=parameter_templates,
            free_affine_mask=tuple(affine_mask),
            free_affine_loc=tuple(affine_loc),
            free_affine_scale=tuple(affine_scale),
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
        flat_values = _flatten_parameter_values(
            values, expected_size=self.n_free_parameters
        )

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
        elif hasattr(cube, "tolist"):
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

            return np.stack(
                [self.prior_transform_numpy(row) for row in cube_values], axis=0
            )

        cube_values = self._validate_free_parameter_count(cube_values)
        transformed = np.empty_like(cube_values, dtype=float)

        for index, parameter_name in enumerate(self.free_parameter_names):
            parameter = self.parameter_templates[
                self.parameter_names.index(parameter_name)
            ]
            transformed[index] = np.asarray(
                self._transform_parameter(parameter, cube_values[index]), dtype=float
            )

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
            parameter = self.parameter_templates[
                self.parameter_names.index(free_parameter_name)
            ]
            transformed.append(
                jnp.asarray(
                    self._transform_parameter(parameter, cube_position[index]),
                    dtype=JAX_DTYPE,
                )
            )

        if not transformed:
            return jnp.zeros((0,), dtype=JAX_DTYPE)

        return jnp.stack(transformed)

    def transform_cube_to_physical_batch(self, cube_positions: Any) -> jnp.ndarray:
        """Transform one or many unit-cube positions into physical coordinates."""
        cube_positions = jnp.asarray(cube_positions, dtype=JAX_DTYPE)

        if jnp.ndim(cube_positions) == 1:
            return self.transform_cube_to_physical(cube_positions)[jnp.newaxis, :]

        return vmap(self.transform_cube_to_physical)(cube_positions)

    def cube_from_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        """Map unconstrained coordinates onto the open unit cube."""
        unconstrained_position = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        return jnp.clip(
            nn.sigmoid(unconstrained_position),
            self.cube_epsilon,
            1.0 - self.cube_epsilon,
        )

    def cube_from_unconstrained_batch(
        self, unconstrained_positions: Any
    ) -> jnp.ndarray:
        """Map one or many unconstrained positions onto the open unit cube."""
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)
        return jnp.clip(
            nn.sigmoid(unconstrained_positions),
            self.cube_epsilon,
            1.0 - self.cube_epsilon,
        )

    def _affine_arrays(self):
        """Return (mask, loc, scale) jnp arrays for the affine free parameters.

        ``mask`` is ``None`` when no free parameter uses the affine map (legacy
        construction or no Normal priors), signalling the pure cube path.
        """
        if not any(self.free_affine_mask):
            return None, None, None
        mask = jnp.asarray(self.free_affine_mask, dtype=bool)
        loc = jnp.asarray(self.free_affine_loc, dtype=JAX_DTYPE)
        scale = jnp.asarray(self.free_affine_scale, dtype=JAX_DTYPE)
        return mask, loc, scale

    def transform_unconstrained(self, unconstrained_position: Any) -> jnp.ndarray:
        """Transform unconstrained coordinates directly into physical values.

        Normal-prior free parameters use the affine map physical = loc + scale*z;
        all others use the bounded cube map physical = quantile(sigmoid(z)).
        """
        z = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        cube_physical = self.transform_cube_to_physical(self.cube_from_unconstrained(z))
        mask, loc, scale = self._affine_arrays()
        if mask is None:
            return cube_physical
        affine_physical = loc + scale * z
        return jnp.where(mask, affine_physical, cube_physical)

    def transform_unconstrained_batch(
        self, unconstrained_positions: Any
    ) -> jnp.ndarray:
        """Transform one or many unconstrained positions into physical values."""
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)
        if jnp.ndim(unconstrained_positions) == 1:
            return self.transform_unconstrained(unconstrained_positions)[jnp.newaxis, :]
        return vmap(self.transform_unconstrained)(unconstrained_positions)

    def log_jacobian_single(self, unconstrained_position: Any) -> jnp.ndarray:
        """Return the additive log-density term for one unconstrained position.

        For cube-mapped parameters this is the sigmoid log-Jacobian
        ``log sigmoid'(z)`` (the prior is carried by the quantile transform). For
        affine-mapped Normal parameters it is the implicit standard-normal prior
        ``-0.5 z**2`` (constant dropped), which exactly reproduces the
        Normal(loc, scale) prior on the physical value.
        """
        z = jnp.asarray(unconstrained_position, dtype=JAX_DTYPE)
        cube_term = -nn.softplus(-z) - nn.softplus(z)
        mask, _loc, _scale = self._affine_arrays()
        if mask is None:
            return jnp.sum(cube_term)
        affine_term = -0.5 * jnp.square(z)
        return jnp.sum(jnp.where(mask, affine_term, cube_term))

    def log_jacobian(self, unconstrained_positions: Any) -> jnp.ndarray:
        """Return log-Jacobian terms for one or many unconstrained positions."""
        unconstrained_positions = jnp.asarray(unconstrained_positions, dtype=JAX_DTYPE)

        if jnp.ndim(unconstrained_positions) == 1:
            return self.log_jacobian_single(unconstrained_positions)

        return vmap(self.log_jacobian_single)(unconstrained_positions)

    def physical_params_from_vector(self, values: npt.ArrayLike) -> PhysicalParams:
        """Build ``PhysicalParams`` from a flat vector of free values."""
        flat_values = self._validate_free_parameter_count(values)
        return self.physical_params_from_free_values(tuple(flat_values))

    def physical_params_from_matrix(
        self, values: npt.ArrayLike
    ) -> tuple[PhysicalParams, ...]:
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

        for parameter_name, parameter in zip(
            self.parameter_names, self.parameter_templates, strict=True
        ):
            if parameter.is_free_parameter:
                physical_mapping[parameter_name] = values[free_index]
                free_index += 1
            else:
                physical_mapping[parameter_name] = parameter.value

        return PhysicalParams.from_mapping(physical_mapping)

    def physical_params_from_cube(self, cube_position: npt.ArrayLike) -> PhysicalParams:
        """Build ``PhysicalParams`` from a unit-cube position."""
        return self.physical_params_from_vector(
            self.prior_transform_numpy(cube_position)
        )

    def physical_params_from_unconstrained(
        self, unconstrained_position: Any
    ) -> PhysicalParams:
        """Build ``PhysicalParams`` from unconstrained sampler coordinates."""
        transformed = self.transform_unconstrained(unconstrained_position)
        return self.physical_params_from_vector(np.asarray(transformed))

    def free_parameter_vector(self, physical_params: PhysicalParams) -> jnp.ndarray:
        """Extract the ordered free-parameter vector from ``PhysicalParams``."""
        mapping = physical_params.as_dict()
        return jnp.asarray(
            [mapping[name] for name in self.free_parameter_names], dtype=JAX_DTYPE
        )

    def build_legacy_parameter_dict(
        self, physical_params: PhysicalParams
    ) -> dict[str, Parameter]:
        """Reconstruct legacy ``Parameter`` objects for compatibility paths."""
        mapping = physical_params.as_dict()
        return {
            parameter_name: _clone_parameter(parameter, mapping[parameter_name])
            for parameter_name, parameter in zip(
                self.parameter_names, self.parameter_templates, strict=True
            )
        }

    def serialize_metadata(self) -> dict[str, Any]:
        """Return a deterministic metadata payload describing the layout."""
        parameter_metadata = []

        for parameter_name, parameter in zip(
            self.parameter_names, self.parameter_templates, strict=True
        ):
            distribution = parameter.distribution
            distribution_name = None
            distribution_parameters = None

            if distribution is not None:
                distribution_name = getattr(
                    distribution, "name", distribution.__class__.__name__
                )
                distribution_parameters = _serialize_runtime_metadata_value(
                    getattr(distribution, "parameters", None)
                )

            parameter_metadata.append(
                {
                    "name": parameter_name,
                    "is_free_parameter": parameter.is_free_parameter,
                    "distribution_name": distribution_name,
                    "distribution_parameters": distribution_parameters,
                    "transform_prior_cube_coordinate": _serialize_runtime_metadata_value(
                        parameter.transform_prior_cube_coordinate
                    ),
                    "value": _serialize_runtime_metadata_value(parameter.value),
                }
            )

        return {
            "parameter_names": list(self.parameter_names),
            "free_parameter_names": list(self.free_parameter_names),
            "free_parameter_indices": list(self.free_parameter_indices),
            "cube_epsilon": float(self.cube_epsilon),
            "parameters": parameter_metadata,
        }


def _is_jax_parameter_values(values: Any) -> bool:
    if isinstance(values, (Array, jax_core.Tracer)):
        return True

    if isinstance(values, (list, tuple)):
        return any(_is_jax_parameter_values(item) for item in values)

    return False


def _flatten_parameter_values(
    values: npt.ArrayLike,
    expected_size: int | None = None,
) -> npt.ArrayLike:
    if isinstance(values, np.ndarray):
        return values.reshape(-1)

    if _is_jax_parameter_values(values):
        return jnp.asarray(values, dtype=JAX_DTYPE).reshape(-1)

    if hasattr(values, "tolist"):
        converted_values = values.tolist()
        if isinstance(converted_values, list):
            return np.array(converted_values, dtype=float).reshape(-1)

    if expected_size is not None:
        try:
            return np.array(
                [values[index] for index in range(expected_size)], dtype=float
            )
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

    if isinstance(value, (bool, np.bool_, int, np.integer)):
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
        module = getattr(value, "__module__", "__callable__")
        qualname = getattr(value, "__qualname__", repr(value))
        return f"{module}.{qualname}"

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


# ---------------------------------------------------------------------------
# Pure functions for parameter-dependent observation transforms.
# These replace the mutable ``Data`` methods with free functions operating on
# ``ObservationState`` fields and a parameter dictionary.
# ---------------------------------------------------------------------------


def _apply_flux_scaling(
    input_spectrum: Any,
    observation: ObservationState,
    parameters: Mapping[str, Any],
) -> Any:
    """Apply multiplicative flux scaling to a spectrum."""
    if not observation.scale_flux:
        return input_spectrum
    scale_factor = _get_parameter_value(
        parameters, f"{observation.name}_scale_factor", default=1.0
    )
    return input_spectrum * scale_factor


def _offset_flux(
    input_spectrum: Any,
    observation: ObservationState,
    parameters: Mapping[str, Any],
) -> Any:
    """Apply additive flux offset to a spectrum."""
    if not observation.fit_flux_offset:
        return input_spectrum
    offset_val = _get_parameter_value(
        parameters, f"{observation.name}_offset", default=0.0
    )
    return input_spectrum - offset_val


def _line_b_uncertainty_scaling(
    observation_name: str,
    parameters: Mapping[str, Any],
) -> Any:
    """Compute the Line (2015) 10^b uncertainty inflation term."""
    b_val = -jnp.inf
    if parameters is not None:
        if f"{observation_name}_b" in parameters:
            b_val = _get_parameter_value(
                parameters, f"{observation_name}_b", default=-jnp.inf
            )
        elif "uncertainty_scaling_b" in parameters:
            b_val = _get_parameter_value(
                parameters, "uncertainty_scaling_b", default=-jnp.inf
            )
    return b_val


def _scale_and_inflate_uncertainties(
    observation: ObservationState,
    parameters: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Compute scaled uncertainties and covariance from observation state.

    This is the pure-function equivalent of ``Data.scale_and_inflate_uncertainties``.
    For observations with ``fit_covariance=True``, this builds the fitted
    covariance matrix from parameters. Otherwise it scales the static
    uncertainties or covariance stored on the observation.
    """
    err_scale_factor = 1.0
    if observation.scale_uncertainties:
        err_scale_factor = _get_parameter_value(
            parameters,
            f"{observation.name}_scale_factor",
            default=1.0,
        )

    raw_b_val = _line_b_uncertainty_scaling(observation.name, parameters)
    additive_variance_term = jnp.where(
        jnp.isfinite(raw_b_val),
        10**raw_b_val,
        0.0,
    )

    if observation.fit_covariance:
        from petitRADTRANS.retrieval.covariance import (
            build_global_covariance_kernel,
            build_local_covariance_kernel,
        )

        scaled_covariance = _build_fitted_covariance_from_observation(
            observation,
            parameters,
            err_scale_factor,
            additive_variance_term,
        )
        scaled_uncertainties = jnp.sqrt(jnp.diag(scaled_covariance))
        return scaled_uncertainties, scaled_covariance

    if observation.covariance is not None:
        scaled_covariance = observation.covariance * (err_scale_factor**2)
        scaled_covariance = scaled_covariance + additive_variance_term * jnp.eye(
            scaled_covariance.shape[0],
            dtype=scaled_covariance.dtype,
        )
        scaled_uncertainties = jnp.sqrt(jnp.diag(scaled_covariance))
    elif observation.uncertainties is not None:
        scaled_uncertainties = observation.uncertainties * err_scale_factor
        scaled_uncertainties = jnp.sqrt(
            scaled_uncertainties**2 + additive_variance_term
        )
        scaled_covariance = None
    else:
        raise ValueError(
            f"Observation '{observation.name}' has neither uncertainties nor covariance."
        )

    return scaled_uncertainties, scaled_covariance


def _build_fitted_covariance_from_observation(
    observation: ObservationState,
    parameters: Mapping[str, Any],
    err_scale_factor: float,
    additive_variance_term: Any,
) -> Any:
    """Build a fitted covariance matrix from observation state and parameters.

    This is the pure-function equivalent of ``Data._build_fitted_covariance``.
    """
    from petitRADTRANS.retrieval.covariance import (
        build_global_covariance_kernel,
        build_local_covariance_kernel,
    )

    wavelength_vector = jnp.asarray(observation.wavelengths)

    if observation.covariance is not None:
        base_covariance = jnp.asarray(observation.covariance) * (err_scale_factor**2)
    elif observation.uncertainties is not None:
        scaled_uncertainties = jnp.asarray(observation.uncertainties) * err_scale_factor
        base_covariance = jnp.diag(scaled_uncertainties**2)
    else:
        raise ValueError(
            f"Observation '{observation.name}' has neither uncertainties nor covariance."
        )

    identity = jnp.eye(wavelength_vector.shape[0], dtype=base_covariance.dtype)
    total_covariance = base_covariance + additive_variance_term * identity

    if observation.covariance_mode in {"global", "global_local"}:
        global_amplitude = _get_parameter_value(
            parameters,
            f"{observation.name}_gp_global_amp",
            default=0.0,
        )
        global_length = _get_parameter_value(
            parameters,
            f"{observation.name}_gp_global_length",
            default=1.0,
        )
        total_covariance = total_covariance + build_global_covariance_kernel(
            wavelength_vector,
            amplitude=global_amplitude,
            length_scale=global_length,
            kernel_name=observation.global_covariance_kernel,
        )

    if observation.covariance_mode in {"local", "global_local"}:
        for kernel_index in range(1, observation.n_local_covariance_kernels + 1):
            local_amplitude = _get_parameter_value(
                parameters,
                f"{observation.name}_gp_local_{kernel_index}_amp",
                default=0.0,
            )
            local_length = _get_parameter_value(
                parameters,
                f"{observation.name}_gp_local_{kernel_index}_length",
                default=1.0,
            )
            default_center = jnp.mean(wavelength_vector)
            local_center = _get_parameter_value(
                parameters,
                f"{observation.name}_gp_local_{kernel_index}_center",
                default=default_center,
            )
            total_covariance = total_covariance + build_local_covariance_kernel(
                wavelength_vector,
                amplitude=local_amplitude,
                length_scale=local_length,
                center=local_center,
                kernel_name=observation.local_covariance_kernel,
            )

    if observation.covariance_jitter > 0.0:
        total_covariance = total_covariance + observation.covariance_jitter * jnp.eye(
            total_covariance.shape[0],
            dtype=total_covariance.dtype,
        )

    return total_covariance


def _slice_observation_epoch(
    observation: ObservationState,
    epoch_index: int,
) -> ObservationState:
    def slice_epoch(value: Any) -> Any:
        if value is None:
            return None

        ndim = getattr(value, "ndim", None)

        if ndim is None or ndim < 2:
            return value

        return value[epoch_index]

    return replace(
        observation,
        wavelengths=slice_epoch(observation.wavelengths),
        spectrum=slice_epoch(observation.spectrum),
        mask=slice_epoch(observation.mask),
        uncertainties=slice_epoch(observation.uncertainties),
        covariance=slice_epoch(observation.covariance),
        data_resolution=slice_epoch(observation.data_resolution),
        data_resolution_array_model=slice_epoch(
            observation.data_resolution_array_model
        ),
        wavelength_bin_widths=slice_epoch(observation.wavelength_bin_widths),
    )


def _uses_variable_resolution_convolution(observation: ObservationState) -> bool:
    for resolution in (
        observation.data_resolution_array_model,
        observation.data_resolution,
    ):
        if resolution is None:
            continue

        resolution_ndim = getattr(resolution, "ndim", None)

        if resolution_ndim is not None and resolution_ndim > 0:
            return True

        if isinstance(resolution, (np.ndarray, jnp.ndarray)):
            return True

    return False


def _configuration_requires_radtrans_initialization(configuration: Any) -> bool:
    for data in getattr(configuration, "data", {}).values():
        if (
            getattr(data, "external_radtrans_reference", None) is None
            and getattr(data, "radtrans_object", None) is None
        ):
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


def _radtrans_wavelength_grid(radtrans_object: Any) -> Any:
    """Return the wavelength-space bin centres for a Radtrans object (in µm)."""
    edges = radtrans_object.frequency_bins_edges
    # bin centres in frequency → convert to wavelength (cm → µm)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # speed of light in cm/s → wavelength in cm, then to µm
    wavelengths_cm = 2.99792458e10 / centres
    return wavelengths_cm * 1e4  # cm → µm


def project_model_to_observation_space(
    wavelengths_model: Any,
    spectrum_model: Any,
    observation: "ObservationState",
    parameters: Mapping[str, Any],
    data_name: str,
) -> Any:
    """Convolve, rebin, and shift a 1-D model spectrum onto the observation grid.

    This is a thin composition of ``extract_spectrum_parameters`` and
    ``process_spectrum_pipeline`` from :mod:`petitRADTRANS.retrieval.utils`,
    gathered here so that ``_score_dataset`` can delegate to a single call.
    """
    from petitRADTRANS.retrieval.utils import (
        extract_spectrum_parameters,
        ensure_jax_array,
        process_spectrum_pipeline,
    )

    if isinstance(observation, PhotometryObservationState):
        projected = observation.photometric_transformation_function(
            wavelengths_model, spectrum_model
        )
        if isinstance(projected, (tuple, list)):
            projected = projected[0]
        return ensure_jax_array(projected)

    spectrum_params = extract_spectrum_parameters(parameters, data_name)
    return process_spectrum_pipeline(
        wavelengths_model,
        spectrum_model,
        observation,
        parameters,
        data_name,
        spectrum_params,
    )


def project_model_time_series_to_observation_space(
    wavelengths_model: Any,
    model_spectra: Any,
    model_times: Any,
    observation: "ObservationState",
    parameters: Mapping[str, Any],
    data_name: str,
) -> Any | None:
    """Project shared-grid time-series spectra before interpolating in time."""
    observation_times = (observation.metadata or {}).get("observation_times")
    if observation_times is None:
        return None

    observation_times = jnp.asarray(observation_times)
    model_times = jnp.asarray(model_times)
    model_spectra = jnp.asarray(model_spectra)

    project_epoch = lambda spectrum_epoch: project_model_to_observation_space(
        wavelengths_model,
        spectrum_epoch,
        observation,
        parameters,
        data_name,
    )

    if _uses_variable_resolution_convolution(observation):
        projected_model_spectra = lax.map(project_epoch, model_spectra)
    else:
        projected_model_spectra = vmap(project_epoch)(model_spectra)

    return vmap(
        lambda wavelength_bin: jnp.interp(
            observation_times,
            model_times,
            wavelength_bin,
        ),
        in_axes=1,
        out_axes=1,
    )(projected_model_spectra)


def compute_diagonal_observation_log_likelihood(
    model_spectrum: Any,
    data_spectrum: Any,
    data_uncertainties: Any,
    *,
    mask: Any = None,
) -> Any:
    model_spectrum = jnp.asarray(model_spectrum)
    data_spectrum = jnp.asarray(data_spectrum)
    data_uncertainties = jnp.asarray(data_uncertainties)

    if model_spectrum.shape != data_spectrum.shape:
        raise ValueError("The data and model model must have the same shape.")

    if data_uncertainties.shape != data_spectrum.shape:
        raise ValueError(
            "data_uncertainties must have the same shape as the data_spectrum"
        )

    if mask is None:
        valid_mask = jnp.ones_like(data_spectrum, dtype=bool)
    else:
        valid_mask = ~jnp.asarray(mask, dtype=bool)

        if valid_mask.shape != data_spectrum.shape:
            raise ValueError("mask must have the same shape as the data_spectrum")

    safe_uncertainties = jnp.where(valid_mask, data_uncertainties, 1.0)
    residual = jnp.where(valid_mask, data_spectrum - model_spectrum, 0.0)
    variance = safe_uncertainties**2

    log_norm = -0.5 * jnp.sum(
        jnp.where(valid_mask, jnp.log(2.0 * jnp.pi * variance), 0.0)
    )
    chi_squared = jnp.sum((residual / safe_uncertainties) ** 2)

    return log_norm - 0.5 * chi_squared


def compute_observation_log_likelihood(
    model_spectrum: Any,
    data_spectrum: Any,
    data_uncertainties: Any | None,
    data_covariance: Any | None,
    n_free_parameters: int,
    covariance_cholesky_factor: Any | None = None,
    log_likelihood_normalization: Any | None = None,
) -> Any:
    """Compute the log-likelihood for a single 1-D observation slice.

    Returns ``INVALID_LOG_LIKELIHOOD`` when the model contains NaN values.
    Otherwise delegates to :class:`~petitRADTRANS.retrieval.log_likelihood.LogLikelihood`.
    """
    from petitRADTRANS.retrieval.log_likelihood import LogLikelihood

    return lax.cond(
        jnp.isnan(model_spectrum).any(),
        lambda _: INVALID_LOG_LIKELIHOOD,
        lambda _: LogLikelihood(
            model_spectrum=model_spectrum,
            data_spectrum=data_spectrum,
            data_uncertainties=data_uncertainties,
            data_covariance=data_covariance,
            covariance_cholesky_factor=covariance_cholesky_factor,
            log_likelihood_normalization=log_likelihood_normalization,
            n_free_parameters=n_free_parameters,
        ).calculate_log_likelihood(),
        operand=None,
    )


__all__ = [
    "LikelihoodState",
    "ModelContext",
    "ModelGroupState",
    "ModelInputs",
    "ModelResult",
    "ObservationState",
    "ParameterLayout",
    "PhysicalParams",
    "RetrievalRuntime",
]
