"""Simulation interfaces for prior-predictive and proposal-driven generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping

from jax import Array, random, vmap
import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.parameter import JAX_DTYPE
from petitRADTRANS.retrieval.runtime import (
    RetrievalRuntime,
    PhysicalParams,
    _apply_flux_scaling,
    _offset_flux,
    _scale_and_inflate_uncertainties,
)

from .task import SBITask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationBatch:
    """Container for one simulated batch.

    Attributes:
        parameters:
            Array-like free-parameter matrix with leading dimension equal to the
            number of simulated samples.
        observations:
            Task-conditioned simulated observations.
        log_likelihood:
            Optional scalar likelihood values associated with each sample.
        diagnostics:
            Additional runtime diagnostics such as clipping flags or NaN counts.
    """

    parameters: Any
    observations: Any
    cube_parameters: Any = None
    unconstrained_parameters: Any = None
    log_likelihood: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Return the number of simulated samples represented by the batch."""
        parameter_array = np.asarray(self.parameters)

        if parameter_array.ndim == 0:
            return 1

        return int(parameter_array.shape[0])


def _stack_simulation_output_payloads(payloads: list[dict]) -> dict:
    """Stack per-sample observation payloads from simulate_observations_scalar.

    Array fields are stacked into a leading (n_samples, ...) dimension.
    The ``"metadata"`` field is kept as a ``list`` (one dict per sample).
    Fields that are ``None`` for every sample remain ``None``.
    """
    if not payloads:
        return {}
    stacked: dict = {}
    for key in payloads[0]:
        values = [p[key] for p in payloads]
        if key == "metadata":
            stacked[key] = values
        elif all(v is None for v in values):
            stacked[key] = None
        else:
            stacked[key] = jnp.stack(
                [
                    (
                        jnp.asarray(v, dtype=JAX_DTYPE)
                        if v is not None
                        else jnp.asarray(jnp.nan, dtype=JAX_DTYPE)
                    )
                    for v in values
                ]
            )
    return stacked


class SimulationRuntime(RetrievalRuntime):
    """Extension of :class:`~petitRADTRANS.retrieval.runtime.RetrievalRuntime`
    that adds deterministic forward-model simulation methods.

    ``RetrievalRuntime`` covers likelihood evaluation.  This subclass layers
    the *simulation* surface on top — projecting parameters into observation
    space without noise so that :class:`RuntimeSimulator` can inject noise
    afterwards and assemble :class:`SimulationBatch` objects.
    """

    def simulate_observations_scalar(
        self,
        physical_params: PhysicalParams,
    ) -> dict[str, dict[str, Any]]:
        """Return deterministic observation-space model outputs for one parameter point."""
        simulated_observations: dict[str, dict[str, Any]] = {}
        parameter_payload = {
            name: float(value)
            for name, value in zip(
                self.parameter_layout.free_parameter_names,
                np.asarray(
                    self.parameter_layout.free_parameter_vector(physical_params),
                    dtype=float,
                ),
                strict=True,
            )
        }

        for group in self.model_groups:
            model_result = self._evaluate_model_group(group, physical_params)

            if not model_result.spectrum_is_valid():
                raise ValueError(
                    f"Model evaluation for source '{group.source_name}' returned an invalid result."
                )

            for observation_name in group.observation_names:
                observation = self.observations[observation_name]
                projected_values = self._project_dataset_observation(
                    observation_name,
                    physical_params,
                    model_result.wavelengths,
                    model_result.spectrum,
                )
                projected_values = _apply_flux_scaling(
                    projected_values, observation, physical_params
                )
                projected_values = _offset_flux(
                    projected_values, observation, physical_params
                )
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
                projected_uncertainties, projected_covariance = (
                    _scale_and_inflate_uncertainties(observation, physical_params)
                )
                simulated_observations[observation_name] = {
                    "values": projected_values,
                    "uncertainties": projected_uncertainties,
                    "covariance": projected_covariance,
                    "mask": observation.mask,
                    "coordinates": observation.wavelengths,
                    "metadata": {
                        "photometry": observation.photometry,
                        "source_name": group.source_name,
                        "line_opacity_mode": observation.line_opacity_mode,
                        "scale_flux": observation.scale_flux,
                        "scale_uncertainties": observation.scale_uncertainties,
                        "fit_flux_offset": observation.fit_flux_offset,
                    },
                }

        return simulated_observations

    def simulate_observations_batched(
        self,
        parameter_matrix: Any,
    ) -> dict[str, dict[str, Any]]:
        """Return deterministic batched observation outputs for a parameter matrix.

        For model groups using ``MODEL_CONTRACT_RUNTIME_NATIVE`` this method
        vmaps the expensive RT kernel over the whole batch in one XLA call,
        then performs the cheap projection step (interpolation, scaling, etc.)
        in a Python loop.  Non-native groups fall back to a scalar-loop path
        equivalent to calling :meth:`simulate_observations_scalar` per row.

        Args:
            parameter_matrix: Shape ``(n_samples, n_free_params)``.

        Returns:
            Mapping from observation name to a payload dict where ``"values"``
            has shape ``(n_samples, n_wavelengths)`` and ``"mask"``,
            ``"coordinates"``, and ``"metadata"`` are static / per-sample lists.
        """
        from petitRADTRANS.retrieval.models import MODEL_CONTRACT_RUNTIME_NATIVE

        parameter_matrix = jnp.asarray(parameter_matrix, dtype=JAX_DTYPE)
        if parameter_matrix.ndim == 1:
            parameter_matrix = parameter_matrix[jnp.newaxis, :]

        n_samples = int(parameter_matrix.shape[0])

        all_native = all(
            g.model_contract == MODEL_CONTRACT_RUNTIME_NATIVE for g in self.model_groups
        )

        if not all_native:
            # Scalar fallback — equivalent to calling simulate_observations_scalar per row.
            result_payloads: list[dict[str, dict]] = []
            for row in np.asarray(parameter_matrix):
                physical_params = self.parameter_layout.physical_params_from_vector(row)
                result_payloads.append(
                    self.simulate_observations_scalar(physical_params=physical_params)
                )
            if not result_payloads:
                return {}
            obs_names = list(result_payloads[0].keys())
            return {
                name: _stack_simulation_output_payloads(
                    [rp[name] for rp in result_payloads]
                )
                for name in obs_names
            }

        # Build batched PhysicalParams: each leaf has shape (n_samples, ...).
        batched_physical_params = vmap(
            self.parameter_layout.physical_params_from_free_values
        )(parameter_matrix)

        # Vmap the RT kernel once per model group — the expensive step.
        group_batched_results = [
            (
                group,
                vmap(group.model_generating_function, in_axes=(None, 0))(
                    group.model_context, batched_physical_params
                ),
            )
            for group in self.model_groups
        ]

        # Python loop over samples for cheap projection (interpolation, scaling).
        obs_payloads: dict[str, list[dict]] = {name: [] for name in self.observations}

        for i in range(n_samples):
            # Extract scalar PhysicalParams for sample i.
            physical_params_i = PhysicalParams(
                ordered_names=batched_physical_params.ordered_names,
                dynamic_names=batched_physical_params.dynamic_names,
                dynamic_values=tuple(
                    v[i] for v in batched_physical_params.dynamic_values
                ),
                static_items=batched_physical_params.static_items,
            )

            for group, batched_result in group_batched_results:
                wavelengths_i = batched_result.wavelengths[i]
                spectrum_i = batched_result.spectrum[i]

                for obs_name in group.observation_names:
                    observation = self.observations[obs_name]
                    projected = self._project_dataset_observation(
                        obs_name, physical_params_i, wavelengths_i, spectrum_i
                    )
                    projected = _apply_flux_scaling(
                        projected, observation, physical_params_i
                    )
                    projected = _offset_flux(projected, observation, physical_params_i)
                    projected_uncertainties, projected_covariance = (
                        _scale_and_inflate_uncertainties(observation, physical_params_i)
                    )
                    obs_payloads[obs_name].append(
                        {
                            "values": projected,
                            "uncertainties": projected_uncertainties,
                            "covariance": projected_covariance,
                            "mask": observation.mask,
                            "coordinates": observation.wavelengths,
                            "metadata": {
                                "photometry": observation.photometry,
                                "source_name": group.source_name,
                                "line_opacity_mode": observation.line_opacity_mode,
                                "scale_flux": observation.scale_flux,
                                "scale_uncertainties": observation.scale_uncertainties,
                                "fit_flux_offset": observation.fit_flux_offset,
                            },
                        }
                    )

        return {
            name: _stack_simulation_output_payloads(payloads)
            for name, payloads in obs_payloads.items()
        }


class ProposalSampler(abc.ABC):
    """Interface for simulation proposals beyond the prior distribution."""

    @abc.abstractmethod
    def sample(self, n_samples: int, task: SBITask) -> Any:
        """Draw free-parameter vectors for the given task."""


class Simulator(abc.ABC):
    """Base simulator for SBI dataset generation and validation."""

    def __init__(self, task: SBITask):
        self.task = task

    @abc.abstractmethod
    def sample_parameters(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> Any:
        """Sample free-parameter vectors from the task prior or a proposal."""

    @abc.abstractmethod
    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        """Run the forward model and noise pipeline for pre-specified parameters."""

    def simulate_batch(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> SimulationBatch:
        """Sample parameters and simulate one batch in a single call."""
        parameters = self.sample_parameters(n_samples=n_samples, proposal=proposal)
        return self.simulate_from_parameters(parameters)


class BatchedSimulator(Simulator):
    """Runtime-backed simulator that prefers vectorized task execution.

    The class intentionally exposes only interfaces at this stage. Concrete
    implementations can back this with ``RetrievalRuntime.evaluate_vectorized_jax``
    or a fallback scalar loop when JAX batching is unavailable.
    """

    def __init__(self, task: SBITask, runtime: Any | None = None):
        super().__init__(task=task)
        self.runtime = runtime

    def sample_parameters(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> Any:
        if proposal is not None:
            return proposal.sample(n_samples=n_samples, task=self.task)
        raise NotImplementedError(
            "Sampling from RetrievalConfig priors should be implemented by the concrete backend."
        )

    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        raise NotImplementedError(
            "Concrete batched simulators must connect task parameters to the retrieval runtime."
        )


class RuntimeSimulator(BatchedSimulator):
    """Concrete simulator backed by the retrieval runtime.

    The simulator samples from the retrieval prior using JAX random utilities
    and projects deterministic forward-model outputs into the observation space
    using ``RetrievalRuntime``.
    """

    def __init__(
        self, task: SBITask, runtime: Any | None = None, seed: int | None = None
    ):
        if runtime is None:
            runtime = SimulationRuntime.from_configuration(
                task.retrieval_config,
                parameter_layout=task.parameter_layout,
                n_free_parameters=task.parameter_layout.n_free_parameters,
            )
        elif not isinstance(runtime, SimulationRuntime):
            # Wrap a pre-built RetrievalRuntime in a SimulationRuntime.
            runtime = SimulationRuntime(
                parameter_layout=runtime.parameter_layout,
                observations=runtime.observations,
                likelihoods=runtime.likelihoods,
                model_groups=runtime.model_groups,
                adaptive_mesh_refinement=runtime.adaptive_mesh_refinement,
            )
        super().__init__(task=task, runtime=runtime)
        initial_seed = seed
        if initial_seed is None:
            initial_seed = task.simulation_config.noise_model.seed
        if initial_seed is None:
            initial_seed = 0

        self._rng_key = random.PRNGKey(int(initial_seed))
        self._validate_task_support()

    def _validate_task_support(self) -> None:
        for data in self.task.retrieval_config.data.values():
            if data.__class__.__name__ == "TimeSeriesData":
                raise NotImplementedError(
                    "RuntimeSimulator does not support time series data in the current phase."
                )

    def _next_key(self) -> Array:
        self._rng_key, next_key = random.split(self._rng_key)
        return next_key

    def _sample_prior_parameter_batch(self, n_samples: int) -> dict[str, jnp.ndarray]:
        if self.task.parameter_layout.n_free_parameters == 0:
            empty = jnp.zeros((n_samples, 0), dtype=JAX_DTYPE)
            return {
                "parameters": empty,
                "cube_parameters": empty,
                "unconstrained_parameters": empty,
            }

        cube_key = self._next_key()
        cube_samples = random.uniform(
            cube_key,
            shape=(n_samples, self.task.parameter_layout.n_free_parameters),
            dtype=JAX_DTYPE,
            minval=self.task.parameter_layout.cube_epsilon,
            maxval=1.0 - self.task.parameter_layout.cube_epsilon,
        )
        unconstrained_samples = jnp.log(cube_samples) - jnp.log1p(-cube_samples)

        return {
            "parameters": self.task.parameter_layout.transform_cube_to_physical_batch(
                cube_samples
            ),
            "cube_parameters": cube_samples,
            "unconstrained_parameters": unconstrained_samples,
        }

    def sample_parameters(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> Any:
        if proposal is not None:
            return proposal.sample(n_samples=n_samples, task=self.task)

        return self._sample_prior_parameter_batch(n_samples)["parameters"]

    def _apply_noise(
        self,
        values: Any,
        uncertainties: Any,
        covariance: Any,
    ) -> tuple[Any, Any, Any]:
        mode = self.task.simulation_config.noise_model.mode
        values = jnp.asarray(values, dtype=JAX_DTYPE)

        if mode in {"none", "no_noise"}:
            return values, uncertainties, covariance

        if mode not in {"observational", "gaussian"}:
            raise ValueError(f"Unsupported SBI noise mode: {mode!r}")

        key = self._next_key()

        if covariance is not None:
            covariance = jnp.asarray(covariance, dtype=JAX_DTYPE)
            mean = jnp.zeros(covariance.shape[0], dtype=JAX_DTYPE)
            noise = random.multivariate_normal(key, mean=mean, cov=covariance)
            return values + noise, uncertainties, covariance

        if uncertainties is None:
            return values, uncertainties, covariance

        uncertainties = jnp.asarray(uncertainties, dtype=JAX_DTYPE)
        noise = random.normal(key, shape=values.shape, dtype=JAX_DTYPE) * uncertainties
        return values + noise, uncertainties, covariance

    def _apply_noise_batched(
        self,
        values_batch: Any,
        uncertainties_batch: Any,
        covariance_batch: Any,
    ) -> tuple[Any, Any, Any]:
        """Apply noise to a batch of simulation outputs.

        Args:
            values_batch: Shape ``(n_samples, n_wavelengths)``.
            uncertainties_batch: Shape ``(n_samples, n_wavelengths)`` or ``None``.
            covariance_batch: Shape ``(n_samples, n_wl, n_wl)`` or ``None``.

        Returns:
            Noisy values, uncertainties, and covariance with the same leading
            batch dimension as ``values_batch``.
        """
        from jax import vmap as _vmap

        mode = self.task.simulation_config.noise_model.mode
        values_batch = jnp.asarray(values_batch, dtype=JAX_DTYPE)

        if mode in {"none", "no_noise"}:
            return values_batch, uncertainties_batch, covariance_batch

        if mode not in {"observational", "gaussian"}:
            raise ValueError(f"Unsupported SBI noise mode: {mode!r}")

        n_samples = int(values_batch.shape[0])

        if covariance_batch is not None:
            covariance_batch = jnp.asarray(covariance_batch, dtype=JAX_DTYPE)
            n = int(covariance_batch.shape[-1])
            mean = jnp.zeros(n, dtype=JAX_DTYPE)
            key = self._next_key()
            subkeys = random.split(key, n_samples)
            noise = _vmap(lambda k, c: random.multivariate_normal(k, mean, c))(
                subkeys, covariance_batch
            )
            return values_batch + noise, uncertainties_batch, covariance_batch

        if uncertainties_batch is None:
            return values_batch, uncertainties_batch, covariance_batch

        uncertainties_batch = jnp.asarray(uncertainties_batch, dtype=JAX_DTYPE)
        key = self._next_key()
        noise = (
            random.normal(key, shape=values_batch.shape, dtype=JAX_DTYPE)
            * uncertainties_batch
        )
        return values_batch + noise, uncertainties_batch, covariance_batch

    @staticmethod
    def _stack_observation_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        keys = payloads[0].keys()
        stacked_payload: dict[str, Any] = {}

        for key in keys:
            values = [payload[key] for payload in payloads]

            if key == "metadata":
                stacked_payload[key] = values
                continue

            if all(value is None for value in values):
                stacked_payload[key] = None
                continue

            stacked_payload[key] = jnp.stack(
                [
                    (
                        jnp.asarray(value, dtype=JAX_DTYPE)
                        if value is not None
                        else jnp.asarray(jnp.nan, dtype=JAX_DTYPE)
                    )
                    for value in values
                ]
            )

        return stacked_payload

    def _simulate_parameter_matrix(
        self,
        parameter_matrix: Any,
        cube_parameters: Any = None,
        unconstrained_parameters: Any = None,
    ) -> SimulationBatch:
        parameter_matrix = jnp.asarray(parameter_matrix, dtype=JAX_DTYPE)

        if jnp.ndim(parameter_matrix) == 1:
            parameter_matrix = parameter_matrix[jnp.newaxis, :]

        if parameter_matrix.shape[-1] != self.task.parameter_layout.n_free_parameters:
            raise ValueError(
                f"Expected parameter vectors with {self.task.parameter_layout.n_free_parameters} free parameters, "
                f"got {parameter_matrix.shape[-1]}"
            )

        # --- Batched path (vmap the RT kernel) ---
        if self.task.simulation_config.use_vectorized_runtime:
            det_obs = self.runtime.simulate_observations_batched(parameter_matrix)

            stacked_observations_batched: dict[str, Any] = {}
            for obs_name, payload in det_obs.items():
                noisy_values, uncertainties, covariance = self._apply_noise_batched(
                    payload["values"],
                    payload["uncertainties"],
                    payload["covariance"],
                )
                stacked_observations_batched[obs_name] = {
                    "values": noisy_values,
                    "deterministic_values": payload["values"],
                    "uncertainties": uncertainties,
                    "covariance": covariance,
                    "mask": payload["mask"],
                    "coordinates": payload["coordinates"],
                    "metadata": payload["metadata"],
                }

            batched_log_likelihood = None
            if self.task.simulation_config.store_per_datapoint_log_likelihood:
                # Reuse the batched evaluator for per-sample log-likelihoods.
                batched_log_likelihood = jnp.asarray(
                    self.runtime.evaluate_jax_batched(parameter_matrix), dtype=JAX_DTYPE
                )

            diagnostics_batched = {
                "parameter_layout": self.task.parameter_layout.serialize_metadata(),
                "task_fingerprint": self.task.task_fingerprint.digest,
            }
            return SimulationBatch(
                parameters=parameter_matrix,
                observations=stacked_observations_batched,
                cube_parameters=(
                    None
                    if cube_parameters is None
                    else jnp.asarray(cube_parameters, dtype=JAX_DTYPE)
                ),
                unconstrained_parameters=(
                    None
                    if unconstrained_parameters is None
                    else jnp.asarray(unconstrained_parameters, dtype=JAX_DTYPE)
                ),
                log_likelihood=batched_log_likelihood,
                diagnostics=diagnostics_batched,
            )

        # --- Scalar fallback path ---
        batched_observations: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.task.retrieval_config.data.keys()
        }
        log_likelihood = None

        if self.task.simulation_config.store_per_datapoint_log_likelihood:
            log_likelihood = []

        for parameter_row in np.asarray(parameter_matrix):
            physical_params = self.runtime.parameter_layout.physical_params_from_vector(
                parameter_row
            )
            simulated_observations = self.runtime.simulate_observations_scalar(
                physical_params=physical_params,
            )

            for observation_name, observation_payload in simulated_observations.items():
                noisy_values, uncertainties, covariance = self._apply_noise(
                    observation_payload["values"],
                    observation_payload["uncertainties"],
                    observation_payload["covariance"],
                )
                batched_observations[observation_name].append(
                    {
                        "values": noisy_values,
                        "deterministic_values": observation_payload["values"],
                        "uncertainties": uncertainties,
                        "covariance": covariance,
                        "mask": observation_payload["mask"],
                        "coordinates": observation_payload["coordinates"],
                        "metadata": observation_payload["metadata"],
                    }
                )

            if log_likelihood is not None:
                log_likelihood.append(
                    self.runtime.evaluate_scalar(
                        physical_params=physical_params,
                    )
                )

        stacked_observations = {
            name: self._stack_observation_payloads(payloads)
            for name, payloads in batched_observations.items()
        }

        diagnostics = {
            "parameter_layout": self.task.parameter_layout.serialize_metadata(),
            "task_fingerprint": self.task.task_fingerprint.digest,
        }

        return SimulationBatch(
            parameters=parameter_matrix,
            observations=stacked_observations,
            cube_parameters=(
                None
                if cube_parameters is None
                else jnp.asarray(cube_parameters, dtype=JAX_DTYPE)
            ),
            unconstrained_parameters=(
                None
                if unconstrained_parameters is None
                else jnp.asarray(unconstrained_parameters, dtype=JAX_DTYPE)
            ),
            log_likelihood=(
                None
                if log_likelihood is None
                else jnp.asarray(log_likelihood, dtype=JAX_DTYPE)
            ),
            diagnostics=diagnostics,
        )

    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        return self._simulate_parameter_matrix(parameters)

    def simulate_batch(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> SimulationBatch:
        """Sample parameters and preserve prior-space coordinates when available."""
        if proposal is not None:
            return self.simulate_from_parameters(
                self.sample_parameters(n_samples=n_samples, proposal=proposal)
            )

        parameter_batch = self._sample_prior_parameter_batch(n_samples)
        return self._simulate_parameter_matrix(
            parameter_batch["parameters"],
            cube_parameters=parameter_batch["cube_parameters"],
            unconstrained_parameters=parameter_batch["unconstrained_parameters"],
        )
