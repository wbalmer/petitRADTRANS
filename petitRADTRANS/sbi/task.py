"""Task definitions for simulation-based inference workflows.

These classes bridge the user-facing retrieval configuration objects and the
runtime-native simulation interfaces used by amortized inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
from typing import Any, Mapping

from petitRADTRANS.sbi.compatibility import CompatibilityReport, TaskFingerprint, compare_named_mappings
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.runtime import ObservationState, ParameterLayout, RetrievalRuntime


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

    mode: str = 'observational'
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None


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
    """

    batch_size: int = 256
    n_simulations: int | None = None
    use_vectorized_runtime: bool = True
    store_per_datapoint_log_likelihood: bool = False
    noise_model: NoiseModelConfig = field(default_factory=NoiseModelConfig)


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
    task_version: str = '0.1.0'
    preprocessing_version: str = '0.1.0'
    forward_model_family: str = 'unknown'
    petitradtrans_version: str = field(
        default_factory=lambda: importlib.metadata.version('petitRADTRANS')
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_retrieval_config(
        cls,
        retrieval_config: RetrievalConfig,
        simulation_config: SimulationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> 'SBITask':
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
                    'line_opacity_mode': data.line_opacity_mode,
                    'photometry': data.photometry,
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
            task_version=str(metadata_dict.pop('task_version', '0.1.0')),
            preprocessing_version=str(metadata_dict.pop('preprocessing_version', '0.1.0')),
            forward_model_family=str(
                metadata_dict.pop(
                    'forward_model_family',
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

    @property
    def fingerprint_payload(self) -> Mapping[str, Any]:
        """Return the deterministic payload used to fingerprint the task family."""
        return {
            'name': self.name,
            'task_version': self.task_version,
            'preprocessing_version': self.preprocessing_version,
            'forward_model_family': self.forward_model_family,
            'petitradtrans_version': self.petitradtrans_version,
            'parameter_names': self.parameter_names,
            'dataset_names': self.observation_schema.dataset_names,
            'modalities': dict(self.observation_schema.modalities),
            'line_opacity_modes': dict(self.line_opacity_modes),
        }

    @property
    def task_fingerprint(self) -> TaskFingerprint:
        """Return the deterministic fingerprint of the task family."""
        return TaskFingerprint.from_payload(self.fingerprint_payload)

    def validate_observation_schema(self, observation_schema: ObservationSchema) -> CompatibilityReport:
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
            'dataset_names': self.observation_schema.dataset_names,
            'modalities': dict(self.observation_schema.modalities),
        }
        observed_payload = {
            'dataset_names': observation_schema.dataset_names,
            'modalities': dict(observation_schema.modalities),
        }
        return compare_named_mappings(
            expected=expected_payload,
            observed=observed_payload,
            prefix='observation_schema',
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
            'task_version': self.task_version,
            'preprocessing_version': self.preprocessing_version,
            'forward_model_family': self.forward_model_family,
            'petitradtrans_version': self.petitradtrans_version,
            'task_fingerprint': self.task_fingerprint.digest,
            'observation_schema': {
                'dataset_names': self.observation_schema.dataset_names,
                'modalities': dict(self.observation_schema.modalities),
            },
        }
        observed_payload = {
            'task_version': getattr(artifact_metadata, 'task_version', None),
            'preprocessing_version': getattr(artifact_metadata, 'preprocessing_version', None),
            'forward_model_family': getattr(artifact_metadata, 'forward_model_family', None),
            'petitradtrans_version': getattr(artifact_metadata, 'petitradtrans_version', None),
            'task_fingerprint': getattr(artifact_metadata, 'task_fingerprint', None),
            'observation_schema': getattr(artifact_metadata, 'observation_schema', {}),
        }
        return compare_named_mappings(
            expected=expected_payload,
            observed=observed_payload,
            prefix='artifact_metadata',
        )

    def build_observation_states(self, model_contract: str = 'runtime_native') -> dict[str, ObservationState]:
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
        if getattr(data, 'photometry', False):
            return 'photometry'
        if data.__class__.__name__ == 'TimeSeriesData':
            return 'time_series'
        return 'spectrum'

    @staticmethod
    def _infer_forward_model_family(retrieval_config: RetrievalConfig) -> str:
        model_names = []

        for data in retrieval_config.data.values():
            model_function = data.model_generating_function
            if model_function is None:
                model_names.append('external_radtrans_reference')
                continue

            module = getattr(model_function, '__module__', '__callable__')
            qualname = getattr(model_function, '__qualname__', repr(model_function))
            model_names.append(f'{module}.{qualname}')

        return '|'.join(sorted(set(model_names))) if model_names else 'unknown'
