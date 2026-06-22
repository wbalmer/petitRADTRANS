"""Task definitions for simulation-based inference workflows.

These classes bridge the user-facing retrieval configuration objects and the
runtime-native simulation interfaces used by amortized inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import inspect
from typing import Any, Mapping

from petitRADTRANS.sbi.compatibility import (
    CompatibilityReport,
    TaskFingerprint,
    compare_named_mappings,
)
from petitRADTRANS.retrieval.data import Data, PhotometryData
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.runtime import (
    ObservationState,
    ParameterLayout,
    RetrievalRuntime,
)


@dataclass(frozen=True)
class NoiseModelConfig:
    """Describe how observational noise is injected during simulation.

    Attributes:
        mode:
            Short identifier of the noise model implementation.
        parameters:
            Backend-specific parameters used to instantiate the noise model.
        seed:
            Optional deterministic seed for repeatable simulation pipelines.
    """

    mode: str = "observational"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None


@dataclass(frozen=True)
class ObservationValueConstraint:
    """Admissible range for simulated observation values.

    Attributes:
        min_value:
            Optional lower bound for valid simulated values.
        max_value:
            Optional upper bound for valid simulated values.
        min_inclusive:
            Whether ``min_value`` is included in the valid interval.
        max_inclusive:
            Whether ``max_value`` is included in the valid interval.
    """

    min_value: float | None = None
    max_value: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_inclusive": bool(self.min_inclusive),
            "max_inclusive": bool(self.max_inclusive),
        }


@dataclass(frozen=True)
class ObservationSchema:
    """Capture the supported observation family for an amortized task.

    Attributes:
        dataset_names:
            Dataset identifiers included in the task.
        modalities:
            Mapping from dataset name to a short modality label such as
            ``'spectrum'``, ``'photometry'``, or ``'time_series'``.
        metadata:
            Additional task-level information required by encoders or
            benchmarks, such as instrument names or wavelength coverage.
    """

    dataset_names: tuple[str, ...]
    modalities: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    """Control how a task generates prior-predictive simulations.

    Attributes:
        batch_size:
            Number of parameter points produced per simulator call.
        n_simulations:
            Optional target number of simulations for dataset generation jobs.
        use_vectorized_runtime:
            Whether to prefer the JAX-vectorized runtime path when available.
        store_per_datapoint_log_likelihood:
            Persist log-likelihood components alongside simulated observations.
        noise_model:
            Noise model configuration applied after forward-model evaluation.
        observation_value_constraints:
            Optional per-dataset admissible value ranges enforced on
            deterministic projected observations before simulations are accepted.
    """

    batch_size: int = 256
    n_simulations: int | None = None
    use_vectorized_runtime: bool = True
    store_per_datapoint_log_likelihood: bool = False
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)
    observation_value_constraints: Mapping[str, ObservationValueConstraint] = field(
        default_factory=dict
    )


# Version of the fingerprint *scheme* itself. Bump this whenever the structure of
# ``fingerprint_payload`` changes so old and new digests are guaranteed distinct.
_FINGERPRINT_SCHEMA_VERSION = "1.0.0"

# Fixed interior cube grid used to fingerprint each prior. Endpoints (0, 1) are
# avoided so unbounded priors (e.g. Normal) stay finite while bounded priors
# still resolve their edges closely (a Uniform(0.9, 2.0) vs Uniform(6.4e9,
# 1.4e10) differ by ~1e10 at every grid point -> the digest changes).
_PRIOR_FINGERPRINT_CUBE_GRID = (0.001, 0.25, 0.5, 0.75, 0.999)


def _model_function_digest(function: Any) -> str:
    """Return a stable digest of a model-generating function's implementation.

    Prefers the source text (so editing the function body invalidates a cached
    dataset), falling back to bytecode and then the qualified name for callables
    whose source is unavailable. Never raises -- fingerprinting must not fail.
    """
    if function is None:
        return "none"
    try:
        source = inspect.getsource(function)
        return "src:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        pass
    code = getattr(function, "__code__", None)
    if code is not None:
        return "code:" + hashlib.sha256(bytes(code.co_code)).hexdigest()
    module = getattr(function, "__module__", "?")
    qualname = getattr(function, "__qualname__", repr(function))
    return f"name:{module}.{qualname}"


@dataclass(frozen=True)
class SBITask:
    """Bundle the immutable ingredients required for an SBI problem.

    The task owns the retrieval configuration, canonical parameter layout,
    observation schema, and simulation policy needed to generate training data
    and evaluate amortized posteriors.
    """

    name: str
    retrieval_config: RetrievalConfig
    parameter_layout: ParameterLayout
    observation_schema: ObservationSchema
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    task_version: str = "0.1.0"
    preprocessing_version: str = "0.5.0"
    forward_model_family: str = "unknown"
    petitradtrans_version: str = field(
        default_factory=lambda: importlib.metadata.version("petitRADTRANS")
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_config(
        cls,
        retrieval_config: RetrievalConfig,
        simulation_config: SimulationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SBITask":
        """Create an SBI task from an existing retrieval configuration.

        Parameters
        ----------
        retrieval_config:
            Retrieval configuration describing the free parameters, observation
            datasets, and runtime-native forward model family.
        simulation_config:
            Optional simulation policy overriding the default prior-predictive
            batch configuration.
        metadata:
            Optional task-level metadata. Reserved keys such as
            ``task_version``, ``preprocessing_version``, and
            ``forward_model_family`` override the default inferred values.

        Returns
        -------
        SBITask
            Immutable SBI task description with parameter layout, observation
            schema, and deterministic task fingerprint payload.

        Notes
        -----
        The task inspects the configured retrieval datasets to infer modalities
        and a forward-model-family string that later participates in artifact
        compatibility checks.
        """
        parameter_layout = ParameterLayout.from_configuration(retrieval_config)
        dataset_names = tuple(retrieval_config.data.keys())
        modalities = {
            name: cls._infer_modality(data)
            for name, data in retrieval_config.data.items()
        }
        observation_schema = ObservationSchema(
            dataset_names=dataset_names,
            modalities=modalities,
            metadata={
                name: {
                    "line_opacity_mode": data.line_opacity_mode,
                    "photometry": isinstance(data, PhotometryData),
                }
                for name, data in retrieval_config.data.items()
            },
        )
        metadata_dict = dict(metadata or {})

        return cls(
            name=retrieval_config.retrieval_name,
            retrieval_config=retrieval_config,
            parameter_layout=parameter_layout,
            observation_schema=observation_schema,
            simulation_config=simulation_config or SimulationConfig(),
            task_version=str(metadata_dict.pop("task_version", "0.1.0")),
            preprocessing_version=str(
                metadata_dict.pop("preprocessing_version", "0.5.0")
            ),
            forward_model_family=str(
                metadata_dict.pop(
                    "forward_model_family",
                    cls._infer_forward_model_family(retrieval_config),
                )
            ),
            metadata=metadata_dict,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the free parameter names in canonical task order."""
        return tuple(self.parameter_layout.free_parameter_names)

    @property
    def line_opacity_modes(self) -> Mapping[str, str]:
        """Return the line-opacity mode for each configured dataset."""
        return {
            name: data.line_opacity_mode
            for name, data in self.retrieval_config.data.items()
        }

    def _forward_model_signature(self) -> dict[str, Any]:
        """Signature of everything that determines the *simulated spectra*.

        Captures prior bounds, declared parameter units, and the source of each
        model-generating function. These must invalidate a cached dataset:
        historically a prior bound (``0.9 * r_jup`` -> ``0.9``) and unit
        (``None`` -> ``jupiterRad``) edit left the digest unchanged, so stale --
        and in one case physically corrupt (~1e-20 flux) -- simulations were
        silently reused across runs.
        """
        layout = self.parameter_layout
        priors: dict[str, Any] = {}
        units: dict[str, Any] = {}
        for template in layout.parameter_templates:
            name = str(template.name)
            units[name] = (
                None
                if getattr(template, "units", None) is None
                else str(template.units)
            )
            is_free = bool(getattr(template, "is_free_parameter", False))
            has_prior = (
                getattr(template, "distribution", None) is not None
                or getattr(template, "transform_prior_cube_coordinate", None)
                is not None
            )
            if is_free and has_prior:
                try:
                    quantiles: Any = [
                        round(float(layout._transform_parameter(template, q)), 10)
                        for q in _PRIOR_FINGERPRINT_CUBE_GRID
                    ]
                except Exception:  # pragma: no cover - defensive: never fail here
                    quantiles = None
                priors[name] = {
                    "dist": (
                        type(template.distribution).__name__
                        if getattr(template, "distribution", None) is not None
                        else "transform"
                    ),
                    "q": quantiles,
                }
            else:
                fixed_value: Any = getattr(template, "value", None)
                try:
                    fixed_value = round(float(fixed_value), 10)
                except (TypeError, ValueError):
                    fixed_value = repr(fixed_value)
                priors[name] = {"fixed": fixed_value}

        model_sources = {
            str(data_name): _model_function_digest(
                getattr(data, "model_generating_function", None)
            )
            for data_name, data in self.retrieval_config.data.items()
        }

        return {"priors": priors, "units": units, "model_sources": model_sources}

    @property
    def fingerprint_payload(self) -> Mapping[str, Any]:
        """Return the deterministic payload used to fingerprint the task family."""
        return {
            "fingerprint_schema_version": _FINGERPRINT_SCHEMA_VERSION,
            "name": self.name,
            "task_version": self.task_version,
            "preprocessing_version": self.preprocessing_version,
            "forward_model_family": self.forward_model_family,
            "petitradtrans_version": self.petitradtrans_version,
            "parameter_names": self.parameter_names,
            "dataset_names": self.observation_schema.dataset_names,
            "modalities": dict(self.observation_schema.modalities),
            "line_opacity_modes": dict(self.line_opacity_modes),
            "observation_value_constraints": {
                name: constraint.to_payload()
                for name, constraint in sorted(
                    self.simulation_config.observation_value_constraints.items()
                )
            },
            # Forward-model-determining config: prior bounds, parameter units,
            # and model-function source. See _forward_model_signature.
            "forward_model_signature": self._forward_model_signature(),
        }

    @property
    def task_fingerprint(self) -> TaskFingerprint:
        """Return the deterministic fingerprint of the task family."""
        return TaskFingerprint.from_payload(self.fingerprint_payload)

    def validate_observation_schema(
        self, observation_schema: ObservationSchema
    ) -> CompatibilityReport:
        """Compare an external observation schema with the task expectation.

        Parameters
        ----------
        observation_schema:
            Candidate schema to compare against the task's required datasets and
            modalities.

        Returns
        -------
        CompatibilityReport
            Structured compatibility report containing mismatches, if any.
        """
        expected_payload = {
            "dataset_names": self.observation_schema.dataset_names,
            "modalities": dict(self.observation_schema.modalities),
        }
        observed_payload = {
            "dataset_names": observation_schema.dataset_names,
            "modalities": dict(observation_schema.modalities),
        }
        return compare_named_mappings(
            expected=expected_payload,
            observed=observed_payload,
            prefix="observation_schema",
        )

    def validate_artifact_metadata(self, artifact_metadata: Any) -> CompatibilityReport:
        """Compare a task with persisted artifact metadata.

        Parameters
        ----------
        artifact_metadata:
            Artifact-like object exposing task and preprocessing provenance
            fields. ``Any`` is accepted to avoid a hard dependency cycle.

        Returns
        -------
        CompatibilityReport
            Structured report describing whether the artifact is compatible with
            the current task definition.

        The method accepts ``Any`` to avoid a hard import dependency back from
        the task module to the artifact module.
        """
        expected_payload = {
            "task_version": self.task_version,
            "preprocessing_version": self.preprocessing_version,
            "forward_model_family": self.forward_model_family,
            "petitradtrans_version": self.petitradtrans_version,
            "task_fingerprint": self.task_fingerprint.digest,
            "observation_schema": {
                "dataset_names": self.observation_schema.dataset_names,
                "modalities": dict(self.observation_schema.modalities),
            },
        }
        observed_payload = {
            "task_version": getattr(artifact_metadata, "task_version", None),
            "preprocessing_version": getattr(
                artifact_metadata, "preprocessing_version", None
            ),
            "forward_model_family": getattr(
                artifact_metadata, "forward_model_family", None
            ),
            "petitradtrans_version": getattr(
                artifact_metadata, "petitradtrans_version", None
            ),
            "task_fingerprint": getattr(artifact_metadata, "task_fingerprint", None),
            "observation_schema": getattr(artifact_metadata, "observation_schema", {}),
        }
        return compare_named_mappings(
            expected=expected_payload,
            observed=observed_payload,
            prefix="artifact_metadata",
        )

    def build_observation_states(
        self, model_contract: str = "differentiable"
    ) -> dict[str, ObservationState]:
        """Materialize immutable runtime observations for the task datasets.

        Parameters
        ----------
        model_contract:
            Runtime contract requested from each retrieval dataset.

        Returns
        -------
        dict[str, ObservationState]
            Mapping from dataset name to immutable runtime observation state.
        """
        return {
            name: data.to_observation_state(model_contract=model_contract)
            for name, data in self.retrieval_config.data.items()
        }

    def build_runtime(self) -> RetrievalRuntime:
        """Construct a retrieval runtime matching the task definition.

        Returns
        -------
        RetrievalRuntime
            Runtime object configured with the task's parameter layout and
            retrieval datasets.

        The default implementation mirrors the retrieval-runtime grouping logic
        used by the exact retrieval package and is sufficient for initial SBI
        simulation backends.
        """
        return RetrievalRuntime.from_configuration(
            self.retrieval_config,
            parameter_layout=self.parameter_layout,
            n_free_parameters=self.parameter_layout.n_free_parameters,
        )

    @staticmethod
    def _infer_modality(data: Data) -> str:
        if isinstance(data, PhotometryData):
            return "photometry"
        if data.__class__.__name__ == "TimeSeriesData":
            return "time_series"
        return "spectrum"

    @staticmethod
    def _infer_forward_model_family(retrieval_config: RetrievalConfig) -> str:
        model_names = []

        for data in retrieval_config.data.values():
            model_function = data.model_generating_function
            if model_function is None:
                model_names.append("external_radtrans_reference")
                continue

            module = getattr(model_function, "__module__", "__callable__")
            qualname = getattr(model_function, "__qualname__", repr(model_function))
            model_names.append(f"{module}.{qualname}")

        return "|".join(sorted(set(model_names))) if model_names else "unknown"
