"""Simulation interfaces for prior-predictive and proposal-driven generation.

This module is similar in spirit to petitRADTRANS.retrieval.runtime and the
log likelihood calculations of the retrieval module. It takes arrays of parameters,
computes forward models, and projects them into observation space via binning, scaling,
convolutions and other transformations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping

import jax
from jax import Array, random, vmap
import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.parameter import JAX_DTYPE
from petitRADTRANS.retrieval.runtime import (
    PhotometryObservationState,
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
    """Container for one batch of simulations.

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


def _concatenate_batched_observation_outputs(
    chunks: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Concatenate the per-chunk outputs of ``simulate_observations_batched``.

    Each chunk maps observation name -> payload dict; the array fields carry a
    leading ``(chunk_n_samples, ...)`` axis and ``"metadata"`` is a per-sample
    list. The merged result is bit-identical to running the whole batch in a
    single call, so splitting a large batch into chunks purely to bound peak
    device memory is transparent to every caller.
    """
    if not chunks:
        return {}
    if len(chunks) == 1:
        return chunks[0]

    merged: dict[str, dict[str, Any]] = {}
    for obs_name, first_payload in chunks[0].items():
        merged_payload: dict[str, Any] = {}
        for key in first_payload:
            values = [chunk[obs_name][key] for chunk in chunks]
            if key == "metadata":
                merged_list: list[Any] = []
                for sub_list in values:
                    merged_list.extend(sub_list)
                merged_payload[key] = merged_list
            elif all(v is None for v in values):
                merged_payload[key] = None
            elif any(isinstance(v, np.ndarray) for v in values):
                # ``mask`` is a host (numpy) array; keep it on the host.
                merged_payload[key] = np.concatenate(
                    [np.asarray(v) for v in values if v is not None], axis=0
                )
            else:
                merged_payload[key] = jnp.concatenate(
                    [
                        jnp.asarray(v, dtype=JAX_DTYPE)
                        for v in values
                        if v is not None
                    ],
                    axis=0,
                )
        merged[obs_name] = merged_payload
    return merged


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

        # Build parameters
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

        # Groups in case a Data object uses an external prt reference
        for group in self.model_groups:
            model_result = self._evaluate_model_group(group, physical_params)

            if not model_result.spectrum_is_valid():
                raise ValueError(
                    f"Model evaluation for source '{group.source_name}' returned an invalid result."
                )

            for observation_name in group.observation_names:
                observation = self.observations[observation_name]

                # Calculate the forward model
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
                        "photometry": isinstance(
                            observation, PhotometryObservationState
                        ),
                        "source_name": group.source_name,
                        "line_opacity_mode": observation.line_opacity_mode,
                        "scale_flux": observation.scale_flux,
                        "scale_uncertainties": observation.scale_uncertainties,
                        "fit_flux_offset": observation.fit_flux_offset,
                    },
                }

        return simulated_observations

    def _run_batched_rt_kernel(
        self,
        parameter_matrix: Any,
    ) -> tuple[Any, list[tuple[Any, Any]]]:
        """Run the expensive vmapped RT kernel and return raw results.

        Returns
        -------
        tuple[batched_physical_params, list[(group, batched_model_result)]]
            The batched physical parameters and per-group raw model outputs
            before observation projection.
        """
        batched_physical_params = vmap(
            self.parameter_layout.physical_params_from_free_values
        )(parameter_matrix)

        group_batched_results = [
            (
                group,
                vmap(group.model_generating_function, in_axes=(None, 0))(
                    group.model_context, batched_physical_params
                ),
            )
            for group in self.model_groups
        ]
        return batched_physical_params, group_batched_results

    def _run_batched_rt_kernel_multi_device(
        self,
        parameter_matrix: Any,
    ) -> tuple[Any, list[tuple[Any, Any]]]:
        """Multi-device version of :meth:`_run_batched_rt_kernel`.

        Distributes the batch across all available JAX devices by placing
        each chunk on a separate device with ``jax.device_put`` and then
        calling the single-device ``_run_batched_rt_kernel`` per chunk.
        JAX's asynchronous GPU dispatch allows the per-device computations
        to overlap.  Results are gathered and concatenated before return.

        The parameter matrix **must** have its leading dimension equal to a
        multiple of the device count (caller is responsible for padding).
        """
        devices = jax.devices()
        n_devices = len(devices)
        n_total = int(parameter_matrix.shape[0])
        per_device = n_total // n_devices

        # Dispatch each chunk to a different device; the vmap inside
        # _run_batched_rt_kernel follows the data placement.
        device_results: list[tuple[Any, list[tuple[Any, Any]]]] = []
        for i in range(n_devices):
            chunk = parameter_matrix[i * per_device : (i + 1) * per_device]
            chunk = jax.device_put(chunk, devices[i])
            device_results.append(self._run_batched_rt_kernel(chunk))

        # Move all results to device 0 so that concatenation doesn't mix
        # arrays residing on different GPUs (which JAX forbids).
        gather_device = devices[0]
        device_results = [
            (
                jax.device_put(pp, gather_device),
                [(g, jax.device_put(r, gather_device)) for g, r in gr],
            )
            for pp, gr in device_results
        ]

        # Concatenate physical params from all devices.
        flat_pp = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *(dr[0] for dr in device_results),
        )

        # Concatenate per-group model results.
        groups = [g for g, _ in device_results[0][1]]
        flat_group_results = [
            (
                group,
                jax.tree_util.tree_map(
                    lambda *xs: jnp.concatenate(xs, axis=0),
                    *(device_results[d][1][g_idx][1] for d in range(n_devices)),
                ),
            )
            for g_idx, group in enumerate(groups)
        ]
        return flat_pp, flat_group_results

    def _project_batched_rt_results(
        self,
        batched_physical_params: Any,
        group_batched_results: list[tuple[Any, Any]],
        n_samples: int,
    ) -> dict[str, dict[str, Any]]:
        """Project raw RT kernel outputs into observation space.

        This is the cheap projection step (interpolation, scaling, binning).
        When possible the projection and flux-transform functions are
        vectorized with ``jax.vmap`` so that the Python-level loop over
        samples is replaced by a single traced call.
        """
        simulated_observations: dict[str, dict[str, Any]] = {}

        for group, batched_result in group_batched_results:
            for obs_name in group.observation_names:
                observation = self.observations[obs_name]

                # --- Vectorize projection + flux scaling + offset ----------
                def _project_and_scale(pp, wl, sp):
                    proj = self._project_dataset_observation(obs_name, pp, wl, sp)
                    proj = _apply_flux_scaling(proj, observation, pp)
                    proj = _offset_flux(proj, observation, pp)
                    return proj

                projected_batch = jnp.asarray(
                    vmap(_project_and_scale)(
                        batched_physical_params,
                        batched_result.wavelengths,
                        batched_result.spectrum,
                    ),
                    dtype=JAX_DTYPE,
                )

                # --- Uncertainties -----------------------------------------
                # When ``scale_uncertainties`` is active, the error-bar
                # parameters vary per sample so we must evaluate per row.
                # Otherwise the result is observation-static and we broadcast.
                first_pp = PhysicalParams(
                    ordered_names=batched_physical_params.ordered_names,
                    dynamic_names=batched_physical_params.dynamic_names,
                    dynamic_values=tuple(
                        v[0] for v in batched_physical_params.dynamic_values
                    ),
                    static_items=batched_physical_params.static_items,
                )
                base_unc, base_cov = _scale_and_inflate_uncertainties(
                    observation, first_pp
                )

                if observation.scale_uncertainties or observation.fit_covariance:
                    unc_list = [base_unc]
                    cov_list = [base_cov]
                    for i in range(1, n_samples):
                        pp_i = PhysicalParams(
                            ordered_names=batched_physical_params.ordered_names,
                            dynamic_names=batched_physical_params.dynamic_names,
                            dynamic_values=tuple(
                                v[i] for v in batched_physical_params.dynamic_values
                            ),
                            static_items=batched_physical_params.static_items,
                        )
                        u, c = _scale_and_inflate_uncertainties(observation, pp_i)
                        unc_list.append(u)
                        cov_list.append(c)
                    stacked_unc = (
                        jnp.stack([jnp.asarray(u, dtype=JAX_DTYPE) for u in unc_list])
                        if base_unc is not None
                        else None
                    )
                    stacked_cov = (
                        jnp.stack([jnp.asarray(c, dtype=JAX_DTYPE) for c in cov_list])
                        if base_cov is not None
                        else None
                    )
                else:
                    if base_unc is not None:
                        stacked_unc = jnp.broadcast_to(
                            jnp.asarray(base_unc, dtype=JAX_DTYPE)[jnp.newaxis, :],
                            (n_samples,) + jnp.asarray(base_unc).shape,
                        )
                    else:
                        stacked_unc = None
                    if base_cov is not None:
                        stacked_cov = jnp.broadcast_to(
                            jnp.asarray(base_cov, dtype=JAX_DTYPE)[jnp.newaxis, :, :],
                            (n_samples,) + jnp.asarray(base_cov).shape,
                        )
                    else:
                        stacked_cov = None

                # --- Per-observation metadata (static, replicated) ---------
                metadata_entry = {
                    "photometry": isinstance(
                        observation, PhotometryObservationState
                    ),
                    "source_name": group.source_name,
                    "line_opacity_mode": observation.line_opacity_mode,
                    "scale_flux": observation.scale_flux,
                    "scale_uncertainties": observation.scale_uncertainties,
                    "fit_flux_offset": observation.fit_flux_offset,
                }

                _mask_arr = (
                    np.asarray(observation.mask)
                    if observation.mask is not None
                    else None
                )
                _coord_arr = jnp.asarray(observation.wavelengths, dtype=JAX_DTYPE)

                simulated_observations[obs_name] = {
                    "values": projected_batch,
                    "uncertainties": stacked_unc,
                    "covariance": stacked_cov,
                    "mask": (
                        np.broadcast_to(
                            _mask_arr[np.newaxis],
                            (n_samples,) + _mask_arr.shape,
                        )
                        if _mask_arr is not None
                        else None
                    ),
                    "coordinates": jnp.broadcast_to(
                        _coord_arr[jnp.newaxis],
                        (n_samples,) + _coord_arr.shape,
                    ),
                    "metadata": [metadata_entry] * n_samples,
                }

        return simulated_observations

    def simulate_observations_batched(
        self,
        parameter_matrix: Any,
        multi_device: bool = False,
        max_chunk_size: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return deterministic batched observation outputs for a parameter matrix.

        For model groups using ``MODEL_CONTRACT_DIFFERENTIABLE`` this method
        vmaps the expensive RT kernel over the whole batch in one XLA call,
        then performs the cheap projection step (interpolation, scaling, etc.)
        in a Python loop.  Non-native groups fall back to a scalar-loop path
        equivalent to calling :meth:`simulate_observations_scalar` per row.

        Args:
            parameter_matrix: Shape ``(n_samples, n_free_params)``.
            multi_device: When ``True`` and more than one JAX device is
                available, distribute the vmapped RT kernel across devices
                using ``jax.device_put`` and ``jax.tree_util.tree_map``.
            max_chunk_size: When set, the native batched path is evaluated in
                sub-batches of at most this many samples and the (binned)
                outputs are concatenated. The multi-device kernel gathers the
                raw, high-resolution forward-model spectra onto a single device
                before projection, so a large batch can exceed device memory;
                chunking bounds that peak while leaving the result identical.

        Returns:
            Mapping from observation name to a payload dict where ``"values"``
            has shape ``(n_samples, n_wavelengths)`` and ``"mask"``,
            ``"coordinates"``, and ``"metadata"`` are static / per-sample lists.
        """
        from petitRADTRANS.retrieval.models import MODEL_CONTRACT_DIFFERENTIABLE

        parameter_matrix = jnp.asarray(parameter_matrix, dtype=JAX_DTYPE)
        if parameter_matrix.ndim == 1:
            parameter_matrix = parameter_matrix[jnp.newaxis, :]

        n_samples = int(parameter_matrix.shape[0])

        all_native = all(
            g.model_contract == MODEL_CONTRACT_DIFFERENTIABLE for g in self.model_groups
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

        if max_chunk_size is not None and max_chunk_size > 0 and n_samples > max_chunk_size:
            # Round the chunk up to a multiple of the device count: the
            # multi-device kernel processes only ``(n // n_devices) * n_devices``
            # rows and silently drops the remainder, so every chunk handed to it
            # must be a clean multiple to avoid losing samples.
            chunk = int(max_chunk_size)
            n_devices = len(jax.devices())
            if multi_device and n_devices > 1:
                chunk = max(n_devices, ((chunk + n_devices - 1) // n_devices) * n_devices)
            chunk_outputs = [
                self._simulate_observations_batched_native(
                    parameter_matrix[start : start + chunk], multi_device
                )
                for start in range(0, n_samples, chunk)
            ]
            return _concatenate_batched_observation_outputs(chunk_outputs)

        return self._simulate_observations_batched_native(parameter_matrix, multi_device)

    def _simulate_observations_batched_native(
        self,
        parameter_matrix: Any,
        multi_device: bool,
    ) -> dict[str, dict[str, Any]]:
        """Run the native (vmapped) RT kernel + projection for one (sub-)batch."""
        n_samples = int(parameter_matrix.shape[0])
        use_multi_device = (
            multi_device and len(jax.devices()) > 1 and n_samples >= len(jax.devices())
        )
        if use_multi_device:
            batched_pp, group_results = self._run_batched_rt_kernel_multi_device(
                parameter_matrix
            )
        else:
            batched_pp, group_results = self._run_batched_rt_kernel(parameter_matrix)

        return self._project_batched_rt_results(batched_pp, group_results, n_samples)


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
        self,
        task: SBITask,
        runtime: Any | None = None,
        seed: int | None = None,
        data_parallel: bool | None = None,
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
        if data_parallel is None:
            self._data_parallel = len(jax.devices()) > 1
        else:
            self._data_parallel = data_parallel
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

    @staticmethod
    def _row_invalid_value_mask(
        values: Any,
        mask: Any,
        constraint: Any,
    ) -> np.ndarray:
        """
        Sanity checking simulated observations.
        Uses the ObservationValueConstraints to check if the values are within the specified bounds.
        Failing simulations will be rejected and resampled. This function returns a boolean mask
        indicating which rows are invalid.
        """
        values_array = np.asarray(values, dtype=float)
        if values_array.ndim == 0:
            values_array = values_array.reshape(1, 1)
        elif values_array.ndim == 1:
            values_array = values_array[np.newaxis, :]

        value_mask = np.zeros_like(values_array, dtype=bool)
        if mask is not None:
            raw_mask = np.asarray(mask, dtype=bool)
            if raw_mask.ndim == 0:
                raw_mask = np.broadcast_to(raw_mask.reshape(1, 1), values_array.shape)
            elif raw_mask.ndim == 1:
                if (
                    values_array.ndim > 1
                    and raw_mask.shape[0] >= values_array.shape[0]
                    and raw_mask.shape[0] != values_array.shape[-1]
                ):
                    raw_mask = raw_mask[: values_array.shape[0]]
                if (
                    values_array.ndim > 1
                    and raw_mask.shape[0] == values_array.shape[0]
                    and raw_mask.shape[0] != values_array.shape[-1]
                ):
                    reshape_shape = (raw_mask.shape[0],) + (1,) * (values_array.ndim - 1)
                else:
                    reshape_shape = (1,) * (values_array.ndim - 1) + (raw_mask.shape[0],)
                raw_mask = np.broadcast_to(raw_mask.reshape(reshape_shape), values_array.shape)
            elif raw_mask.shape != values_array.shape:
                raw_mask = np.broadcast_to(raw_mask, values_array.shape)
            value_mask = raw_mask

        valid_positions = ~value_mask
        invalid_positions = valid_positions & ~np.isfinite(values_array)

        if constraint is not None:
            if constraint.min_value is not None:
                if constraint.min_inclusive:
                    lower_violation = values_array < float(constraint.min_value)
                else:
                    lower_violation = values_array <= float(constraint.min_value)
                invalid_positions |= valid_positions & lower_violation
            if constraint.max_value is not None:
                if constraint.max_inclusive:
                    upper_violation = values_array > float(constraint.max_value)
                else:
                    upper_violation = values_array >= float(constraint.max_value)
                invalid_positions |= valid_positions & upper_violation

        return np.any(invalid_positions, axis=tuple(range(1, invalid_positions.ndim)))

    def _invalid_sample_mask(
        self,
        observations: Mapping[str, Mapping[str, Any]],
        n_samples: int,
    ) -> tuple[np.ndarray, dict[str, int]]:
        invalid_mask = np.zeros((n_samples,), dtype=bool)
        invalid_counts: dict[str, int] = {}
        constraints = self.task.simulation_config.observation_value_constraints

        for observation_name, payload in observations.items():
            deterministic_values = payload.get("deterministic_values", payload.get("values"))
            if deterministic_values is None:
                continue
            row_invalid = self._row_invalid_value_mask(
                deterministic_values,
                payload.get("mask"),
                constraints.get(observation_name),
            )
            invalid_mask |= row_invalid
            invalid_counts[observation_name] = int(np.sum(row_invalid))

        return invalid_mask, invalid_counts

    @staticmethod
    def _slice_batch_rows(
        batch: SimulationBatch,
        row_indices: np.ndarray,
    ) -> SimulationBatch:
        row_indices = np.asarray(row_indices, dtype=int)
        original_n_samples = batch.n_samples

        def _slice_optional_array(value: Any) -> Any:
            if value is None:
                return None
            return jnp.asarray(value)[row_indices]

        sliced_observations: dict[str, dict[str, Any]] = {}
        for observation_name, payload in batch.observations.items():
            sliced_payload: dict[str, Any] = {}
            for key, value in payload.items():
                if value is None:
                    sliced_payload[key] = None
                elif key == "metadata":
                    sliced_payload[key] = [value[int(index)] for index in row_indices]
                elif key == "mask":
                    mask_array = np.asarray(value)
                    if mask_array.ndim >= 1 and mask_array.shape[0] == original_n_samples:
                        sliced_payload[key] = mask_array[row_indices]
                    else:
                        sliced_payload[key] = mask_array
                else:
                    sliced_payload[key] = jnp.asarray(value)[row_indices]
            sliced_observations[observation_name] = sliced_payload

        diagnostics = dict(batch.diagnostics)
        diagnostics["n_samples"] = int(row_indices.shape[0])

        return SimulationBatch(
            parameters=_slice_optional_array(batch.parameters),
            observations=sliced_observations,
            cube_parameters=_slice_optional_array(batch.cube_parameters),
            unconstrained_parameters=_slice_optional_array(batch.unconstrained_parameters),
            log_likelihood=_slice_optional_array(batch.log_likelihood),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _concatenate_simulation_batches(
        batches: list[SimulationBatch],
        diagnostics: Mapping[str, Any] | None = None,
    ) -> SimulationBatch:
        if not batches:
            raise ValueError("At least one simulation batch is required for concatenation.")

        def _concat_optional_arrays(values: list[Any]) -> Any:
            present_values = [value for value in values if value is not None]
            if not present_values:
                return None
            return jnp.concatenate([jnp.asarray(value) for value in present_values], axis=0)

        concatenated_observations: dict[str, dict[str, Any]] = {}
        observation_names = list(batches[0].observations.keys())
        for observation_name in observation_names:
            payload_keys = list(batches[0].observations[observation_name].keys())
            concatenated_payload: dict[str, Any] = {}
            for key in payload_keys:
                field_values = [batch.observations[observation_name][key] for batch in batches]
                if all(value is None for value in field_values):
                    concatenated_payload[key] = None
                elif key == "metadata":
                    metadata_values: list[Any] = []
                    for value in field_values:
                        metadata_values.extend(list(value))
                    concatenated_payload[key] = metadata_values
                elif key == "mask":
                    mask_arrays = [np.asarray(value) for value in field_values]
                    if all(
                        mask_array.ndim >= 1 and mask_array.shape[0] == batch.n_samples
                        for mask_array, batch in zip(mask_arrays, batches, strict=True)
                    ):
                        concatenated_payload[key] = np.concatenate(mask_arrays, axis=0)
                    else:
                        concatenated_payload[key] = mask_arrays[0]
                else:
                    concatenated_payload[key] = jnp.concatenate(
                        [jnp.asarray(value) for value in field_values],
                        axis=0,
                    )
            concatenated_observations[observation_name] = concatenated_payload

        merged_diagnostics = dict(batches[0].diagnostics)
        if diagnostics is not None:
            merged_diagnostics.update(dict(diagnostics))

        return SimulationBatch(
            parameters=_concat_optional_arrays([batch.parameters for batch in batches]),
            observations=concatenated_observations,
            cube_parameters=_concat_optional_arrays(
                [batch.cube_parameters for batch in batches]
            ),
            unconstrained_parameters=_concat_optional_arrays(
                [batch.unconstrained_parameters for batch in batches]
            ),
            log_likelihood=_concat_optional_arrays(
                [batch.log_likelihood for batch in batches]
            ),
            diagnostics=merged_diagnostics,
        )

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

    def _advance_rng_for_batch(self, n_samples: int) -> None:
        """Advance the PRNG state as if ``simulate_batch(n_samples)`` were called.

        This performs only cheap key-splitting – no forward-model evaluation –
        making it suitable for fast dataset-resume skipping.

        The number of ``_next_key`` calls matches the pattern inside
        ``simulate_batch`` → ``_sample_prior_parameter_batch`` (1 key) and
        ``_simulate_parameter_matrix`` (1 key per observation that adds noise).
        """
        # _sample_prior_parameter_batch always consumes one key.
        self._next_key()

        noise_mode = self.task.simulation_config.noise_model.mode
        if noise_mode in {"none", "no_noise"}:
            return

        if self.task.simulation_config.use_vectorized_runtime:
            # Batched noise path: 1 key per observation regardless of batch size.
            for obs in self.runtime.observations.values():
                if obs.covariance is not None or obs.uncertainties is not None:
                    self._next_key()
        else:
            # Scalar fallback: 1 key per observation *per sample*.
            n_noise_obs = sum(
                1
                for obs in self.runtime.observations.values()
                if obs.covariance is not None or obs.uncertainties is not None
            )
            for _ in range(n_samples * n_noise_obs):
                self._next_key()

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
        data_parallel: bool = False,
    ) -> SimulationBatch:
        parameter_matrix = jnp.asarray(parameter_matrix, dtype=JAX_DTYPE)

        if jnp.ndim(parameter_matrix) == 1:
            parameter_matrix = parameter_matrix[jnp.newaxis, :]

        if parameter_matrix.shape[-1] != self.task.parameter_layout.n_free_parameters:
            raise ValueError(
                f"Expected parameter vectors with {self.task.parameter_layout.n_free_parameters} free parameters, "
                f"got {parameter_matrix.shape[-1]}"
            )

        n_samples = int(parameter_matrix.shape[0])

        # When data_parallel is requested we pad the batch to a multiple of
        # the device count so that jax can split it evenly.  The extra
        # samples are trimmed from the results after simulation.
        multi_device = (
            data_parallel
            and self.task.simulation_config.use_vectorized_runtime
            and len(jax.devices()) > 1
            and n_samples >= len(jax.devices())
        )
        n_devices = len(jax.devices()) if multi_device else 1
        padded_n = n_samples
        if multi_device:
            remainder = n_samples % n_devices
            if remainder != 0:
                pad_count = n_devices - remainder
                parameter_matrix = jnp.concatenate(
                    [
                        parameter_matrix,
                        jnp.repeat(parameter_matrix[-1:], pad_count, axis=0),
                    ],
                    axis=0,
                )
                if cube_parameters is not None:
                    cube_parameters = jnp.concatenate(
                        [
                            jnp.asarray(cube_parameters, dtype=JAX_DTYPE),
                            jnp.repeat(
                                jnp.asarray(cube_parameters, dtype=JAX_DTYPE)[-1:],
                                pad_count,
                                axis=0,
                            ),
                        ],
                        axis=0,
                    )
                if unconstrained_parameters is not None:
                    unconstrained_parameters = jnp.concatenate(
                        [
                            jnp.asarray(unconstrained_parameters, dtype=JAX_DTYPE),
                            jnp.repeat(
                                jnp.asarray(unconstrained_parameters, dtype=JAX_DTYPE)[
                                    -1:
                                ],
                                pad_count,
                                axis=0,
                            ),
                        ],
                        axis=0,
                    )
                padded_n = int(parameter_matrix.shape[0])

        # --- Batched path (vmap the RT kernel) ---
        if self.task.simulation_config.use_vectorized_runtime:
            # Bound the forward-model sub-batch to the configured simulation
            # batch size so a single device never holds the whole batch's
            # high-resolution spectra at once (the multi-device kernel gathers
            # raw outputs onto device 0 before projection). This is the footprint
            # proven safe during dataset generation; the result is identical.
            det_obs = self.runtime.simulate_observations_batched(
                parameter_matrix,
                multi_device=multi_device,
                max_chunk_size=int(self.task.simulation_config.batch_size),
            )

            # Trim padding introduced for even device splitting.
            if padded_n > n_samples:
                parameter_matrix = parameter_matrix[:n_samples]
                if cube_parameters is not None:
                    cube_parameters = jnp.asarray(cube_parameters, dtype=JAX_DTYPE)[
                        :n_samples
                    ]
                if unconstrained_parameters is not None:
                    unconstrained_parameters = jnp.asarray(
                        unconstrained_parameters, dtype=JAX_DTYPE
                    )[:n_samples]
                for obs_name in det_obs:
                    payload = det_obs[obs_name]
                    for key in ("values", "uncertainties", "covariance", "coordinates", "mask"):
                        if payload.get(key) is not None and hasattr(
                            payload[key], "__getitem__"
                        ):
                            arr = payload[key]
                            if (
                                hasattr(arr, "shape")
                                and arr.ndim >= 1
                                and arr.shape[0] == padded_n
                            ):
                                payload[key] = arr[:n_samples]

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
            batched_batch = SimulationBatch(
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
            invalid_sample_mask, invalid_counts = self._invalid_sample_mask(
                batched_batch.observations,
                batched_batch.n_samples,
            )
            batched_batch.diagnostics.update(
                {
                    "invalid_sample_mask": invalid_sample_mask,
                    "invalid_sample_count": int(np.sum(invalid_sample_mask)),
                    "invalid_sample_counts_by_observation": invalid_counts,
                }
            )
            return batched_batch

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
        scalar_batch = SimulationBatch(
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
        invalid_sample_mask, invalid_counts = self._invalid_sample_mask(
            scalar_batch.observations,
            scalar_batch.n_samples,
        )
        scalar_batch.diagnostics.update(
            {
                "invalid_sample_mask": invalid_sample_mask,
                "invalid_sample_count": int(np.sum(invalid_sample_mask)),
                "invalid_sample_counts_by_observation": invalid_counts,
            }
        )
        return scalar_batch

    def simulate_from_parameters(self, parameters: Any) -> SimulationBatch:
        return self._simulate_parameter_matrix(
            parameters, data_parallel=self._data_parallel
        )

    def simulate_batch(
        self, n_samples: int, proposal: ProposalSampler | None = None
    ) -> SimulationBatch:
        """Sample parameters and preserve prior-space coordinates when available."""
        accepted_batches: list[SimulationBatch] = []
        accepted_count = 0
        attempted_count = 0
        rejected_count = 0
        invalid_counts_by_observation: dict[str, int] = {}
        max_attempted = max(int(n_samples) * 20, int(n_samples) + 1024)

        while accepted_count < n_samples:
            remaining = int(n_samples - accepted_count)
            if attempted_count >= max_attempted:
                acceptance = (
                    float(accepted_count) / float(attempted_count)
                    if attempted_count > 0
                    else 0.0
                )
                raise RuntimeError(
                    "Failed to generate enough valid simulations within the allowed "
                    f"resampling budget: accepted {accepted_count}/{n_samples} after "
                    f"{attempted_count} attempts (acceptance rate {acceptance:.1%}, "
                    f"budget {max_attempted}). Invalid rows per observation: "
                    f"{invalid_counts_by_observation}. A row is invalid if any value "
                    "is non-finite or violates its ObservationValueConstraint -- a "
                    "collapsed acceptance rate usually means the constraint rejects "
                    "physically valid simulations (e.g. a strict 'value > 0' bound "
                    "rejecting fluxes that underflow to 0 in float32) or the forward "
                    "model produces non-finite values over part of the prior."
                )

            if proposal is None:
                parameter_batch = self._sample_prior_parameter_batch(remaining)
                candidate_batch = self._simulate_parameter_matrix(
                    parameter_batch["parameters"],
                    cube_parameters=parameter_batch["cube_parameters"],
                    unconstrained_parameters=parameter_batch["unconstrained_parameters"],
                    data_parallel=self._data_parallel,
                )
            else:
                candidate_batch = self._simulate_parameter_matrix(
                    self.sample_parameters(n_samples=remaining, proposal=proposal),
                    data_parallel=self._data_parallel,
                )

            attempted_count += candidate_batch.n_samples
            invalid_mask = np.asarray(
                candidate_batch.diagnostics.get("invalid_sample_mask"),
                dtype=bool,
            )
            if invalid_mask.ndim == 0:
                invalid_mask = np.full((candidate_batch.n_samples,), bool(invalid_mask))
            rejected_count += int(np.sum(invalid_mask))

            counts_by_observation = candidate_batch.diagnostics.get(
                "invalid_sample_counts_by_observation",
                {},
            )
            for observation_name, count in counts_by_observation.items():
                invalid_counts_by_observation[observation_name] = (
                    invalid_counts_by_observation.get(observation_name, 0) + int(count)
                )

            valid_indices = np.flatnonzero(~invalid_mask)
            if valid_indices.size == 0:
                continue

            accepted_batch = self._slice_batch_rows(candidate_batch, valid_indices)
            accepted_batches.append(accepted_batch)
            accepted_count += accepted_batch.n_samples

        final_batch = self._concatenate_simulation_batches(
            accepted_batches,
            diagnostics={
                "parameter_layout": self.task.parameter_layout.serialize_metadata(),
                "task_fingerprint": self.task.task_fingerprint.digest,
                "accepted_sample_count": int(accepted_count),
                "attempted_sample_count": int(attempted_count),
                "rejected_sample_count": int(rejected_count),
                "rejection_rate": (
                    float(rejected_count) / float(attempted_count)
                    if attempted_count > 0
                    else 0.0
                ),
                "invalid_sample_counts_by_observation": invalid_counts_by_observation,
            },
        )
        return final_batch
