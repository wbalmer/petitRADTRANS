"""Simulation interfaces for prior-predictive and proposal-driven generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.jax_parameter import JAX_DTYPE

from .task import SBITask


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
    def sample_parameters(self, n_samples: int, proposal: ProposalSampler | None = None) -> Any:
        """Sample free-parameter vectors from the task prior or a proposal."""

    @abc.abstractmethod
    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        """Run the forward model and noise pipeline for pre-specified parameters."""

    def simulate_batch(self, n_samples: int, proposal: ProposalSampler | None = None) -> SimulationBatch:
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

    def sample_parameters(self, n_samples: int, proposal: ProposalSampler | None = None) -> Any:
        if proposal is not None:
            return proposal.sample(n_samples=n_samples, task=self.task)
        raise NotImplementedError('Sampling from RetrievalConfig priors should be implemented by the concrete backend.')

    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        raise NotImplementedError('Concrete batched simulators must connect task parameters to the retrieval runtime.')


class RuntimeSimulator(BatchedSimulator):
    """Concrete simulator backed by the retrieval runtime.

    The simulator samples from the retrieval prior using JAX random utilities
    and projects deterministic forward-model outputs into the observation space
    using ``RetrievalRuntime``.
    """

    def __init__(self, task: SBITask, runtime: Any | None = None, seed: int | None = None):
        super().__init__(task=task, runtime=runtime or task.build_runtime())
        initial_seed = seed
        if initial_seed is None:
            initial_seed = task.simulation_config.noise_model.seed
        if initial_seed is None:
            initial_seed = 0

        self._rng_key = jax.random.PRNGKey(int(initial_seed))
        self._validate_task_support()

    def _validate_task_support(self) -> None:
        for data in self.task.retrieval_config.data.values():
            if data.__class__.__name__ == 'TimeSeriesData':
                raise NotImplementedError('RuntimeSimulator does not support time series data in the current phase.')

    def _next_key(self) -> jax.Array:
        self._rng_key, next_key = jax.random.split(self._rng_key)
        return next_key

    def _sample_prior_parameter_batch(self, n_samples: int) -> dict[str, jnp.ndarray]:
        if self.task.parameter_layout.n_free_parameters == 0:
            empty = jnp.zeros((n_samples, 0), dtype=JAX_DTYPE)
            return {
                'parameters': empty,
                'cube_parameters': empty,
                'unconstrained_parameters': empty,
            }

        cube_key = self._next_key()
        cube_samples = jax.random.uniform(
            cube_key,
            shape=(n_samples, self.task.parameter_layout.n_free_parameters),
            dtype=JAX_DTYPE,
            minval=self.task.parameter_layout.cube_epsilon,
            maxval=1.0 - self.task.parameter_layout.cube_epsilon,
        )
        unconstrained_samples = jnp.log(cube_samples) - jnp.log1p(-cube_samples)

        return {
            'parameters': self.task.parameter_layout.transform_cube_to_physical_batch(cube_samples),
            'cube_parameters': cube_samples,
            'unconstrained_parameters': unconstrained_samples,
        }

    def sample_parameters(self, n_samples: int, proposal: ProposalSampler | None = None) -> Any:
        if proposal is not None:
            return proposal.sample(n_samples=n_samples, task=self.task)

        return self._sample_prior_parameter_batch(n_samples)['parameters']

    def _apply_noise(
        self,
        values: Any,
        uncertainties: Any,
        covariance: Any,
    ) -> tuple[Any, Any, Any]:
        mode = self.task.simulation_config.noise_model.mode
        values = jnp.asarray(values, dtype=JAX_DTYPE)

        if mode in {'none', 'no_noise'}:
            return values, uncertainties, covariance

        if mode not in {'observational', 'gaussian'}:
            raise ValueError(f"Unsupported SBI noise mode: {mode!r}")

        key = self._next_key()

        if covariance is not None:
            covariance = jnp.asarray(covariance, dtype=JAX_DTYPE)
            mean = jnp.zeros(covariance.shape[0], dtype=JAX_DTYPE)
            noise = jax.random.multivariate_normal(key, mean=mean, cov=covariance)
            return values + noise, uncertainties, covariance

        if uncertainties is None:
            return values, uncertainties, covariance

        uncertainties = jnp.asarray(uncertainties, dtype=JAX_DTYPE)
        noise = jax.random.normal(key, shape=values.shape, dtype=JAX_DTYPE) * uncertainties
        return values + noise, uncertainties, covariance

    @staticmethod
    def _stack_observation_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        keys = payloads[0].keys()
        stacked_payload: dict[str, Any] = {}

        for key in keys:
            values = [payload[key] for payload in payloads]

            if key == 'metadata':
                stacked_payload[key] = values
                continue

            if all(value is None for value in values):
                stacked_payload[key] = None
                continue

            stacked_payload[key] = jnp.stack([
                jnp.asarray(value, dtype=JAX_DTYPE) if value is not None else jnp.asarray(jnp.nan, dtype=JAX_DTYPE)
                for value in values
            ])

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

        batched_observations: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.task.retrieval_config.data.keys()
        }
        log_likelihood = None

        if self.task.simulation_config.store_per_datapoint_log_likelihood:
            log_likelihood = []

        for parameter_row in np.asarray(parameter_matrix):
            physical_params = self.runtime.parameter_layout.physical_params_from_vector(parameter_row)
            simulated_observations = self.runtime.simulate_observations_scalar(
                physical_params=physical_params,
            )

            for observation_name, observation_payload in simulated_observations.items():
                noisy_values, uncertainties, covariance = self._apply_noise(
                    observation_payload['values'],
                    observation_payload['uncertainties'],
                    observation_payload['covariance'],
                )
                batched_observations[observation_name].append(
                    {
                        'values': noisy_values,
                        'deterministic_values': observation_payload['values'],
                        'uncertainties': uncertainties,
                        'covariance': covariance,
                        'mask': observation_payload['mask'],
                        'coordinates': observation_payload['coordinates'],
                        'metadata': observation_payload['metadata'],
                    }
                )

            if log_likelihood is not None:
                log_likelihood.append(
                    self.runtime.evaluate_scalar(
                        configuration=self.task.retrieval_config,
                        physical_params=physical_params,
                    )
                )

        stacked_observations = {
            name: self._stack_observation_payloads(payloads)
            for name, payloads in batched_observations.items()
        }

        diagnostics = {
            'parameter_layout': self.task.parameter_layout.serialize_metadata(),
            'task_fingerprint': self.task.task_fingerprint.digest,
        }

        return SimulationBatch(
            parameters=parameter_matrix,
            observations=stacked_observations,
            cube_parameters=None if cube_parameters is None else jnp.asarray(cube_parameters, dtype=JAX_DTYPE),
            unconstrained_parameters=(
                None
                if unconstrained_parameters is None
                else jnp.asarray(unconstrained_parameters, dtype=JAX_DTYPE)
            ),
            log_likelihood=None if log_likelihood is None else jnp.asarray(log_likelihood, dtype=JAX_DTYPE),
            diagnostics=diagnostics,
        )

    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        return self._simulate_parameter_matrix(parameters)

    def simulate_batch(self, n_samples: int, proposal: ProposalSampler | None = None) -> SimulationBatch:
        """Sample parameters and preserve prior-space coordinates when available."""
        if proposal is not None:
            return self.simulate_from_parameters(self.sample_parameters(n_samples=n_samples, proposal=proposal))

        parameter_batch = self._sample_prior_parameter_batch(n_samples)
        return self._simulate_parameter_matrix(
            parameter_batch['parameters'],
            cube_parameters=parameter_batch['cube_parameters'],
            unconstrained_parameters=parameter_batch['unconstrained_parameters'],
        )
