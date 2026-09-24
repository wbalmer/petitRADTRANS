"""End-to-end SBI validation workflow.

:func:`run_sbi` drives the full amortized-SBI validation run --
build the task, generate or load the simulation dataset, train (or load) the
conditional-flow posterior, save checkpoint guardrails, compute SBC and
posterior-predictive reports on held-out simulations, run amortized inference on
the target observation, and persist samples, plots, a summary, and a run
manifest. Everything experiment-specific (priors, model-generating function,
non-default knobs) is supplied through a single :class:`SBIConfig`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import (
    build_run_label,
    compute_run_config_hash,
    count_predictive_case_checkpoints,
    resolve_latest_posterior_run,
    resolve_posterior_dir,
    resolve_run_manifest,
    save_predictive_spectrum,
    update_latest_run_pointer,
    write_json,
    write_run_manifest,
)
from .calibration import (
    _block_log_spectrum_scale,
    classify_posterior_informativeness,
    generate_embedding_information_report,
    generate_encoder_stage_information_report,
    generate_posterior_predictive_report,
    generate_sbc_report,
    serialize_ood_diagnostic,
    summarize_parameter_units,
    summarize_posterior_width_ratios,
    summarize_prior_edge_occupancy,
    summarize_posterior_samples,
    summarize_predictive_report,
    summarize_training_diagnostics,
)
from .config import SBIConfig, build_conditional_flow_posterior
from .dataset import (
    DatasetSplit,
    HDF5SimulationDatasetStore,
    NormalizedObservationDatasetReader,
    estimate_preloaded_split_memory_gib,
    generate_simulation_dataset,
    infer_split_counts,
)
from .flows import load_posterior_estimator
from .inference import AmortizedRetrieval
from .observation import build_observation_block_batch, build_raw_observation_blocks
from .plotting import plot_posterior_predictive_report, save_validation_plots
from .preprocessing import (
    TaskPreprocessingMetadata,
    denormalize_spectral_values,
    fit_task_preprocessing,
)
from .task import NoiseModelConfig, ObservationValueConstraint, SBITask, SimulationConfig
from .training import load_trainer_checkpoint_state


def _resolve_spectrum_y_label(config: SBIConfig) -> str:
    """Plot y-axis label: explicit override, else derived from the observable."""
    if config.spectrum_y_label is not None:
        return config.spectrum_y_label
    if getattr(config.retrieval_config, "scattering_in_emission", False):
        return r"Flux [W m$^{-2}$ $\mu$m$^{-1}$]"
    return "Transit depth"


def _spectral_inverse_terms(stats, n_wavelengths: int):
    """``(center, scale)`` such that ``raw = center + scale * sinh(normalized)``.

    JAX-friendly mirror of
    :func:`petitRADTRANS.experimental.sbi.preprocessing.denormalize_spectral_values` (same
    round-trip test covers both).
    """
    from .preprocessing import (
        _global_spectral_scale,
        _NORMALIZATION_SCALE_FLOOR,
        _per_wavelength_stats_arrays,
    )

    per_wavelength = _per_wavelength_stats_arrays(stats, n_wavelengths)
    if per_wavelength is not None:
        center, scale = per_wavelength
        return (
            jnp.asarray(center, dtype=jnp.float32),
            jnp.asarray(scale, dtype=jnp.float32),
        )
    global_scale = max(float(_global_spectral_scale(stats)), _NORMALIZATION_SCALE_FLOOR)
    return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(
        global_scale, dtype=jnp.float32
    )


def _build_elbo_log_likelihood_fn(task: SBITask, preprocessing_metadata=None):
    """Build a differentiable forward-model log-likelihood for the ELBO objective.

    Returns ``log_likelihood_fn(theta_cube, observations) -> (batch,)``: it maps
    unit-cube parameters to physical values, runs the *same* differentiable pRT
    forward model used to generate the dataset (and used by JAXNS), projects the
    model spectrum onto the observation grid, and scores the batch's observed
    spectra under the known Gaussian noise model. The observed values reaching
    the loss are reader-normalized, so they are inverted here -- via the single
    round-trip-tested
    :func:`petitRADTRANS.experimental.sbi.preprocessing.denormalize_spectral_values` -- so
    the likelihood is evaluated in raw units where the noise is Gaussian.
    ``preprocessing_metadata`` is required whenever per-wavelength
    normalization is in use (the current default).
    """
    from petitRADTRANS.experimental.sbi.simulator import SimulationRuntime
    from petitRADTRANS.retrieval.runtime import _apply_flux_scaling, _offset_flux

    runtime = SimulationRuntime.from_configuration(
        task.retrieval_config,
        parameter_layout=task.parameter_layout,
        n_free_parameters=task.parameter_layout.n_free_parameters,
    )
    parameter_layout = task.parameter_layout
    if len(runtime.model_groups) != 1:
        raise ValueError(
            "ELBO forward-model likelihood currently supports a single model "
            f"group; found {len(runtime.model_groups)}."
        )
    group = runtime.model_groups[0]
    model_context = group.model_context
    model_fn = group.model_generating_function
    if len(group.observation_names) != 1:
        raise ValueError(
            "ELBO forward-model likelihood currently supports a single "
            f"observation; found {group.observation_names}."
        )
    obs_name = group.observation_names[0]
    observation = runtime.observations[obs_name]
    log_2pi = float(np.log(2.0 * np.pi))

    def spectrum_fn(theta_physical):
        """(batch, n_free) physical -> (batch, n_wl_obs) model spectrum."""
        batched_pp = jax.vmap(parameter_layout.physical_params_from_free_values)(
            theta_physical
        )
        batched_result = jax.vmap(lambda pp: model_fn(model_context, pp))(batched_pp)

        def _project(pp, wavelengths, spectrum):
            projected = runtime._project_dataset_observation(
                obs_name, pp, wavelengths, spectrum
            )
            projected = _apply_flux_scaling(projected, observation, pp)
            projected = _offset_flux(projected, observation, pp)
            return projected

        return jax.vmap(_project)(
            batched_pp, batched_result.wavelengths, batched_result.spectrum
        )

    block_stats = None
    if preprocessing_metadata is not None:
        block_stats = preprocessing_metadata.blocks.get(obs_name)
    if block_stats is None:
        raise ValueError(
            "The ELBO likelihood needs the task's preprocessing statistics to "
            "recover raw observation units from the reader-normalized batch; "
            "pass preprocessing_metadata."
        )
    inverse_center, inverse_scale = _spectral_inverse_terms(
        block_stats, int(np.size(observation.wavelengths))
    )

    def log_likelihood_fn(theta_cube, observations):
        # observations is a PreStackedObservations; block 0 is the spectrum.
        block = observations.stacked_blocks[0]
        value_norm = jnp.asarray(block[0], dtype=jnp.float32)
        uncert_norm = jnp.asarray(block[1], dtype=jnp.float32)
        mask = jnp.asarray(block[3])                            # True == masked out

        # Exact inverse of the applied normalization (see
        # preprocessing.denormalize_spectral_values).
        raw_values = inverse_center + inverse_scale * jnp.sinh(value_norm)
        raw_uncert = uncert_norm * inverse_scale * jnp.cosh(value_norm)

        theta_physical = parameter_layout.transform_cube_to_physical_batch(theta_cube)
        model_spectrum = spectrum_fn(theta_physical)
        model_spectrum = jnp.reshape(model_spectrum, raw_values.shape)

        valid = (~mask.astype(bool)).astype(jnp.float32)
        sigma = jnp.where(raw_uncert > 0.0, raw_uncert, 1.0)
        inv_var = 1.0 / (sigma * sigma)
        residual = raw_values - model_spectrum
        quadratic = jnp.sum(valid * residual * residual * inv_var, axis=-1)
        normalizer = jnp.sum(valid * (2.0 * jnp.log(sigma) + log_2pi), axis=-1)
        return -0.5 * (quadratic + normalizer)

    return log_likelihood_fn


def _build_residual_fn_factory(task: SBITask):
    """Build the differentiable forward model once and return a residual factory.

    Returns ``make_residual(raw_values, raw_uncertainties, mask=None) ->
    residual(theta_cube)`` where ``residual`` is the noise-standardized
    ``(x_obs - model(theta))/sigma`` (masked channels set to 0) in RAW physical
    units. The expensive ``SimulationRuntime`` (opacity load + model context) is
    constructed a single time, so the per-observation closures are cheap --
    enabling test-time refinement over many held-out cases without reloading
    opacities.

    The caller supplies the observation in raw units. This function deliberately
    does NOT invert the reader's normalization: a hardcoded inverse of the
    global-scale transform silently corrupted the target by ~100x (with a sign
    flip) once per-wavelength normalization became the default, making every
    refinement optimise a meaningless objective. Callers that only hold
    normalized data must go through
    :func:`petitRADTRANS.experimental.sbi.preprocessing.denormalize_spectral_values`, the
    single round-trip-tested inverse.
    """
    from petitRADTRANS.experimental.sbi.simulator import SimulationRuntime
    from petitRADTRANS.retrieval.runtime import _apply_flux_scaling, _offset_flux

    runtime = SimulationRuntime.from_configuration(
        task.retrieval_config,
        parameter_layout=task.parameter_layout,
        n_free_parameters=task.parameter_layout.n_free_parameters,
    )
    parameter_layout = task.parameter_layout
    group = runtime.model_groups[0]
    model_context = group.model_context
    model_fn = group.model_generating_function
    obs_name = group.observation_names[0]
    observation = runtime.observations[obs_name]

    def spectrum_fn(theta_physical):
        batched_pp = jax.vmap(parameter_layout.physical_params_from_free_values)(
            theta_physical
        )
        batched_result = jax.vmap(lambda pp: model_fn(model_context, pp))(batched_pp)

        def _project(pp, wavelengths, spectrum):
            projected = runtime._project_dataset_observation(
                obs_name, pp, wavelengths, spectrum
            )
            projected = _apply_flux_scaling(projected, observation, pp)
            projected = _offset_flux(projected, observation, pp)
            return projected

        return jax.vmap(_project)(
            batched_pp, batched_result.wavelengths, batched_result.spectrum
        )

    def make_residual(raw_values, raw_uncertainties, mask=None):
        raw_values = jnp.asarray(raw_values, dtype=jnp.float32).reshape(-1)
        raw_uncert = jnp.abs(
            jnp.asarray(raw_uncertainties, dtype=jnp.float32)
        ).reshape(-1)
        if mask is None:
            valid = jnp.ones_like(raw_values)
        else:
            valid = (
                ~jnp.asarray(mask).reshape(-1).astype(bool)
            ).astype(jnp.float32)
        sigma = jnp.where(raw_uncert > 0.0, raw_uncert, 1.0)

        def residual(theta_cube):
            theta_physical = parameter_layout.transform_cube_to_physical_batch(
                theta_cube[None, :]
            )
            model_spectrum = jnp.reshape(spectrum_fn(theta_physical), (-1,))
            return valid * (raw_values - model_spectrum) / sigma

        return residual

    return make_residual


def _refinement_verdict(
    method: str,
    refinement_config: Any,
    importance_sampling_diagnostics: Mapping[str, Any] | None,
    test_time_validation_report: Mapping[str, Any] | None,
    interval_level: float = 0.9,
) -> dict[str, Any]:
    """Decide whether a refined posterior is fit to be delivered.

    Refinement is a per-observation optimisation against the exact likelihood,
    and it can fail in ways that still produce a confident-looking artifact:
    importance sampling can put all the weight on one particle, and a MAP point
    can carry a Laplace covariance that is far too narrow. Both have shipped
    here as "the posterior" -- 18 of 18 and 19 of 19 zero-width marginals --
    while the diagnostics that named the failure sat unread on disk.

    Returns a verdict mapping with ``accepted`` plus the numbers behind the
    decision, so the summary records why a run delivered what it delivered.
    """
    verdict: dict[str, Any] = {
        "method": method,
        "accepted": True,
        "reasons": [],
        "diagnostics": {},
    }
    if method == "none":
        verdict["status"] = "not_requested"
        return verdict

    if method == "importance_sampling":
        if not importance_sampling_diagnostics:
            verdict["status"] = "accepted_without_diagnostics"
            return verdict
        ess = float(importance_sampling_diagnostics.get("effective_sample_size", 0.0))
        efficiency = float(
            importance_sampling_diagnostics.get("sampling_efficiency", 0.0)
        )
        minimum = float(refinement_config.min_effective_sample_size)
        verdict["diagnostics"] = {
            "effective_sample_size": ess,
            "sampling_efficiency": efficiency,
            "min_effective_sample_size": minimum,
        }
        if ess < minimum:
            verdict["accepted"] = False
            verdict["reasons"].append(
                f"effective sample size {ess:.1f} is below the required "
                f"{minimum:.0f}: the importance weights collapsed onto too few "
                "proposal draws to describe a distribution"
            )

    elif method == "map_laplace":
        if not test_time_validation_report:
            # Without held-out validation there is nothing to test the delivered
            # method against; deliver it but say so.
            verdict["status"] = "accepted_without_validation"
            return verdict
        coverage = test_time_validation_report.get("overall_coverage_90")
        median_chi2 = test_time_validation_report.get("median_chi2_per_n")
        max_coverage_error = float(refinement_config.max_test_time_coverage_error)
        max_chi2 = float(refinement_config.max_test_time_median_chi2_per_n)
        verdict["diagnostics"] = {
            "overall_coverage": coverage,
            "nominal_coverage": interval_level,
            "median_chi2_per_n": median_chi2,
            "max_test_time_coverage_error": max_coverage_error,
            "max_test_time_median_chi2_per_n": max_chi2,
        }
        if coverage is not None:
            coverage_error = abs(float(coverage) - interval_level)
            verdict["diagnostics"]["coverage_error"] = coverage_error
            if coverage_error > max_coverage_error:
                verdict["accepted"] = False
                verdict["reasons"].append(
                    f"held-out interval coverage {float(coverage):.3f} misses the "
                    f"nominal {interval_level:.2f} by {coverage_error:.3f}: the "
                    "refined covariance is not calibrated"
                )
        if median_chi2 is not None and float(median_chi2) > max_chi2:
            verdict["accepted"] = False
            verdict["reasons"].append(
                f"median chi2/N over held-out cases is {float(median_chi2):.2f} "
                f"(limit {max_chi2:.1f}): the refinement optimizer is not "
                "converging case-to-case"
            )

    verdict["status"] = "accepted" if verdict["accepted"] else "rejected"
    return verdict


def _predictive_crps(result: Any) -> float | None:
    """Mean CRPS across observation blocks of a target inference result.

    CRPS is a *proper* scoring rule, which is why the delivery comparison uses
    it rather than the mean absolute error or the interval width on their own.
    A posterior can win on MAE by collapsing to a biased point, and win on
    coverage by inflating until it contains everything; CRPS penalises both
    directions at once, so "better CRPS" cannot be bought by mis-stating the
    uncertainty. Returns ``None`` when no predictive was computed.
    """
    predictive = getattr(result, "posterior_predictive", None)
    if predictive is None:
        return None
    scores = [
        float(value)
        for value in getattr(predictive, "crps", {}).values()
        if np.isfinite(float(value))
    ]
    if not scores:
        return None
    return float(np.mean(scores))


def _refinement_predictive_comparison(
    refined_result: Any,
    amortized_result: Any,
    max_crps_ratio: float,
    refined_samples: Any = None,
) -> dict[str, Any]:
    """Compare a refined target predictive against the amortized one.

    The threshold gate in :func:`_refinement_verdict` asks whether a refinement
    is bad in absolute terms. It never asked the question that actually decides
    what to ship: *is it better than what it replaces?* On run
    20260821T162601Z the refinement passed every threshold (held-out coverage
    error 0.164 < 0.25, median chi2/N 1.12 < 10) and was delivered, while being
    measurably worse than the amortized posterior it displaced -- CRPS
    8.65e-06 against 3.65e-06, MAE 1.74 sigma against 0.83, and a predictive
    interval 28.5 sigma wide against 6.8. The run shipped the worse of the two
    artifacts it had computed.

    Returns the numbers behind the comparison and whether the refinement earns
    delivery.
    """
    # Degeneracy is checked BEFORE the score, and vetoes it. A refinement that
    # has collapsed onto a single point produces an *excellent* predictive: the
    # maximum-likelihood parameter pushed through the forward model, with the
    # observational noise supplying all of the spread. CRPS scores the
    # observable forecast and cannot see that the parameter posterior has zero
    # width, so it actively prefers the collapse. Measured on emission run
    # 20260729T121612Z, whose importance sampling put all 16384 weights on one
    # particle: CRPS 1.70e-17 for the collapsed artifact against 6.39e-17 for
    # the honest amortized posterior -- 3.7x "better". Every credible interval
    # in that artifact was meaningless.
    degenerate = (
        0 if refined_samples is None else _count_degenerate_marginals(refined_samples)
    )
    refined_crps = _predictive_crps(refined_result)
    amortized_crps = _predictive_crps(amortized_result)
    comparison: dict[str, Any] = {
        "refined_degenerate_marginals": int(degenerate),
        "criterion": "mean predictive CRPS (proper scoring rule; lower is better)",
        "refined_crps": refined_crps,
        "amortized_crps": amortized_crps,
        "max_crps_ratio": float(max_crps_ratio),
    }
    if degenerate:
        comparison["refined_is_better"] = False
        comparison["note"] = (
            f"{degenerate} delivered marginal(s) have zero width: a repeated "
            "point estimate, not a distribution"
        )
        return comparison

    if refined_crps is None or amortized_crps is None or amortized_crps <= 0.0:
        # Nothing to compare against: keep the threshold gate's decision rather
        # than inventing one.
        comparison["refined_is_better"] = True
        comparison["note"] = "no predictive available on one side; comparison skipped"
        return comparison

    ratio = refined_crps / amortized_crps
    comparison["crps_ratio"] = float(ratio)
    comparison["refined_is_better"] = bool(ratio <= float(max_crps_ratio))
    return comparison


def _resolve_truncation(
    config: Any,
    task: Any,
    output_root: Path,
    raw_observation_blocks: Any,
) -> tuple[Any, dict[str, Any], Path]:
    """Build and verify the truncation box for a sequential round.

    Loads the source round's posterior, samples it at the target observation,
    and derives a box in absolute cube coordinates -- composing with the source
    round's own box when that round was itself truncated, so boxes never drift
    into relative coordinates.

    Raises ``RuntimeError`` if the box is not safe to truncate with. That check
    is the whole safety argument for the method: mass the box cuts away cannot
    be recovered by any later round, so an unsafe box must stop the run rather
    than quietly produce a confident wrong answer.

    Returns
    -------
    tuple
        The box, its verification report, and the source run root.
    """
    from .artifacts import resolve_latest_posterior_run, resolve_posterior_dir
    from .flows.base import load_posterior_estimator
    from .preprocessing import TaskPreprocessingMetadata, normalize_observation_blocks
    from .truncation import build_truncation_box, truncation_box_report

    truncation = config.truncation
    if truncation.box_from_run_label is not None:
        # Reuse a completed round's box exactly. Deriving it again could give a
        # slightly different box, hence a different task fingerprint, hence a
        # regenerated dataset -- the whole point of resuming is to avoid that.
        from .truncation import TruncationBox

        reuse_root = (output_root / "runs" / truncation.box_from_run_label).resolve()
        reuse_manifest = resolve_run_manifest(reuse_root)
        payload = reuse_manifest.get("truncation_box")
        if not payload:
            raise RuntimeError(
                f"Run {truncation.box_from_run_label} has no stored truncation box, "
                "so its round cannot be reproduced. Check the run label."
            )
        box = TruncationBox.from_payload(payload)
        print(f"Truncated round: reusing the stored box from {reuse_root.name}")
        report = {
            "reused_box_from_run": reuse_root.name,
            "safe": True,
            "reasons": [],
            "round": int(box.metadata.get("round", 1)),
            "volume_fraction": box.volume_fraction,
        }
        return box, report, reuse_root

    if truncation.source_run_label is not None:
        source_run_root = (output_root / "runs" / truncation.source_run_label).resolve()
        source_manifest = resolve_run_manifest(source_run_root)
    else:
        source_run_root, source_manifest = resolve_latest_posterior_run(output_root)
    source_posterior_dir = resolve_posterior_dir(source_run_root, source_manifest)

    print(f"Truncated round: deriving the box from {source_run_root.name}")
    source_posterior = load_posterior_estimator(Path(source_posterior_dir).as_posix())
    source_metadata = json.loads(
        (Path(source_posterior_dir) / "metadata.json").read_text()
    )
    source_preprocessing = TaskPreprocessingMetadata.from_payload(
        source_metadata["preprocessing_metadata_payload"]
    )

    normalized = normalize_observation_blocks(
        raw_observation_blocks, source_preprocessing
    )
    encoded = source_posterior.encode_observation(normalized)
    # Seeded: the box determines the task fingerprint, so re-deriving it must
    # reproduce it exactly. An unseeded draw would give a slightly different
    # box on a resume, and with it a different fingerprint -- which reads as a
    # cache miss and silently regenerates the round's dataset.
    cube_samples = np.asarray(
        source_posterior.sample_posterior(
            encoded,
            n_samples=max(int(config.n_posterior_samples), 20000),
            seed=int(config.seed) + 17,
        ).samples
    )

    parameter_names = tuple(task.parameter_layout.free_parameter_names)
    box = build_truncation_box(
        cube_samples,
        parameter_names,
        coverage=truncation.coverage,
        safety_factor=truncation.safety_factor,
        min_box_fraction=truncation.min_box_fraction,
        source=source_run_root.name,
    )

    # Verify in the source round's own coordinates, then lift to absolute cube
    # coordinates if that round was itself truncated.
    report = truncation_box_report(
        box,
        cube_samples,
        truth_cube=None,
        min_captured_fraction=truncation.min_captured_fraction,
    )
    parent_payload = source_manifest.get("truncation_box")
    if parent_payload:
        from .truncation import TruncationBox

        box = TruncationBox.from_payload(parent_payload).compose(box)
        report["composed_with_source_box"] = True
    report["round"] = int(box.metadata.get("round", 1))
    report["source_run"] = source_run_root.name

    # Check any known truths in physical units. Each parameter's cube-to-physical
    # map is monotone, so the box corners map to the endpoints of the retained
    # physical interval -- no inverse prior transform is needed.
    truths = dict(config.truth_parameter_values or {})
    if truths:
        corners = np.asarray(
            task.parameter_layout.transform_cube_to_physical_batch(
                np.stack([box.lower, box.upper])
            ),
            dtype=float,
        )
        physical_lower = np.minimum(corners[0], corners[1])
        physical_upper = np.maximum(corners[0], corners[1])
        excluded = []
        retained = {}
        for index, name in enumerate(parameter_names):
            if name not in truths:
                continue
            truth_value = float(truths[name])
            low, high = float(physical_lower[index]), float(physical_upper[index])
            retained[name] = [low, high]
            if not low <= truth_value <= high:
                excluded.append(
                    f"{name} (truth {truth_value:.6g} outside [{low:.6g}, {high:.6g}])"
                )
        report["retained_physical_intervals"] = retained
        report["truth_inside_box"] = not excluded
        if excluded:
            report["reasons"].append(
                "the box excludes the known truth for: " + ", ".join(excluded)
            )
            if truncation.require_truth_inside_box:
                report["safe"] = False

    print(
        f"  box retains {report['captured_fraction']:.5f} of the source posterior "
        f"mass; prior volume fraction {box.volume_fraction:.4e} (round "
        f"{report['round']})"
    )
    if not report["safe"]:
        raise RuntimeError(
            "Refusing to truncate: "
            + "; ".join(report["reasons"])
            + ". Widen the box (raise truncation.coverage or safety_factor) or "
            "improve the source posterior before running a truncated round."
        )
    return box, report, source_run_root


def _count_degenerate_marginals(posterior_samples: Any) -> int:
    """Number of parameters whose delivered marginal has zero width.

    A duplicated point estimate serialises exactly like a posterior: same
    shape, same file names, plausible-looking medians. Counting the collapsed
    marginals puts the distinction in the summary where a reader will see it.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        return 0
    spread = np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)
    return int(np.count_nonzero(spread <= 0.0))


def _save_rejected_refinement_samples(
    *,
    run_root: Path,
    posterior,
    task,
    raw_observation_blocks,
    preprocessing_metadata,
    n_posterior_samples: int,
    seed: int,
) -> None:
    """Write a rejected refinement's draws aside for inspection.

    Stored in physical units, matching the delivered posterior's samples, so
    the two can be compared directly; posteriors sample in cube coordinates and
    an archive left in that space cannot be read against the truth values.

    Only the parameter samples are stored: the posterior-predictive step runs
    the forward model and is reserved for the posterior actually delivered.
    Never raises -- failing to archive a rejected artifact must not fail a run
    that has already produced a good one.
    """
    from .preprocessing import normalize_observation_blocks

    try:
        normalized = normalize_observation_blocks(
            raw_observation_blocks, preprocessing_metadata
        )
        encoded = posterior.encode_observation(normalized)
        cube_samples = np.asarray(
            posterior.sample_posterior(
                encoded, n_samples=n_posterior_samples, seed=seed
            ).samples
        )
        samples = np.asarray(
            task.parameter_layout.transform_cube_to_physical_batch(cube_samples),
            dtype=float,
        )
        np.save(run_root / "rejected_refined_posterior_samples.npy", samples)
    except Exception as error:  # pragma: no cover - diagnostics must not break a run
        print(f"  (could not archive the rejected refinement samples: {error})")


def _generate_test_time_validation_report(
    posterior,
    reader,
    make_residual,
    *,
    split,
    max_cases,
    n_posterior_samples,
    map_kwargs,
    parameter_names,
    seed,
):
    """Validate the *actual* test-time inference (MAP+Laplace) on held-out cases.

    The amortized posterior is only a warm-start; the delivered prediction is the
    per-observation MAP+Laplace. This runs that same method on a subset of held-out
    simulations (where the truth is known) and reports its accuracy (chi2/N at the
    MAP) and calibration (90% coverage and mean SBC rank per parameter).
    """
    from petitRADTRANS.experimental.sbi.observation import PreStackedObservations
    from .training import _prestack_observation_blocks

    batch = next(
        iter(
            reader.iter_batches(
                split=split,
                batch_size=int(max_cases),
                shuffle=True,
                seed=int(seed),
                parameter_space="cube",
            )
        )
    )
    true_cube = np.asarray(batch.parameters, dtype=float)  # (cases, P)
    # The reader yields observations as a list of normalized ObservationBlock lists
    # (encoder=None). Pre-stack them into a PreStackedObservations so they match the
    # dense form make_residual / map_laplace_posterior_on_observation expect (the
    # same one the target inference uses) and can be sliced per case below.
    observations = batch.observations
    if not hasattr(observations, "stacked_blocks"):
        observations = _prestack_observation_blocks(observations)
    if observations is None or not hasattr(observations, "stacked_blocks"):
        raise RuntimeError(
            "Test-time (MAP+Laplace) validation requires pre-stackable observation "
            "blocks, but the reader returned a batch that could not be pre-stacked."
        )
    n_cases = min(int(max_cases), true_cube.shape[0])

    def _slice_case(index):
        return PreStackedObservations(
            stacked_blocks=tuple(
                tuple(np.asarray(arr)[index:index + 1] for arr in block)
                for block in observations.stacked_blocks
            ),
            modalities=observations.modalities,
        )

    param_dim = true_cube.shape[1]
    block_name = next(iter(reader.dataset.manifest.observation_schema["modalities"]))
    rank_fraction = np.zeros((n_cases, param_dim))      # P(sample < true)
    inside_90 = np.zeros((n_cases, param_dim), dtype=bool)
    # Standardized error of the truth under the delivered posterior. Coverage
    # alone cannot say *why* it falls short; (truth - mean) / sd separates a
    # displaced posterior from a too-narrow one, and those call for opposite
    # fixes.
    z_score = np.zeros((n_cases, param_dim))
    chi2_per_n = np.zeros(n_cases)
    map_diagnostics: list[dict] = []
    n_wavelengths = None

    for case_index in range(n_cases):
        case_obs = _slice_case(case_index)
        case_block = case_obs.stacked_blocks[0]
        case_raw_values, case_raw_uncert = denormalize_spectral_values(
            np.asarray(case_block[0])[0],
            np.asarray(case_block[1])[0],
            reader.preprocessing_metadata.blocks[block_name],
        )
        residual = make_residual(
            case_raw_values, case_raw_uncert, np.asarray(case_block[3])[0]
        )
        refined = posterior.map_laplace_posterior_on_observation(
            case_obs,
            residual,
            n_posterior_samples=int(n_posterior_samples),
            seed=int(seed) + case_index,
            verbose=False,
            **map_kwargs,
        )
        samples = np.asarray(refined._fixed_cube_samples, dtype=float)  # (S, P)
        map_cube = np.asarray(
            getattr(refined, "_fixed_point_estimate_cube", samples.mean(0)),
            dtype=float,
        )
        residual_at_map = np.asarray(residual(jnp.asarray(map_cube, jnp.float32)))
        n_wavelengths = residual_at_map.shape[0]
        chi2_per_n[case_index] = float(np.sum(residual_at_map ** 2)) / max(
            n_wavelengths, 1
        )
        truth = true_cube[case_index]
        rank_fraction[case_index] = np.mean(samples < truth[None, :], axis=0)
        lo = np.percentile(samples, 5.0, axis=0)
        hi = np.percentile(samples, 95.0, axis=0)
        inside_90[case_index] = (truth >= lo) & (truth <= hi)
        z_score[case_index] = (truth - samples.mean(axis=0)) / np.maximum(
            samples.std(axis=0), 1e-12
        )
        case_diagnostics = getattr(refined, "_map_diagnostics", None)
        if isinstance(case_diagnostics, dict):
            map_diagnostics.append(case_diagnostics)
        print(
            f"  test-time case {case_index + 1}/{n_cases}: "
            f"chi2/N(MAP) {chi2_per_n[case_index]:.2f}",
            flush=True,
        )

    coverage_90 = inside_90.mean(axis=0)        # per-parameter (target 0.90)
    mean_rank = rank_fraction.mean(axis=0)      # per-parameter (target 0.50)
    report = {
        "n_cases": int(n_cases),
        "n_wavelengths": int(n_wavelengths or 0),
        "mean_chi2_per_n": float(np.mean(chi2_per_n)),
        "median_chi2_per_n": float(np.median(chi2_per_n)),
        "overall_coverage_90": float(coverage_90.mean()),
        "per_parameter_coverage_90": {
            parameter_names[i]: float(coverage_90[i]) for i in range(param_dim)
        },
        "per_parameter_mean_rank": {
            parameter_names[i]: float(mean_rank[i]) for i in range(param_dim)
        },
    }

    # Bias/width decomposition of the coverage shortfall. For a calibrated
    # Gaussian posterior z has mean 0 and standard deviation 1 per parameter, so
    # z_bias near 0 with z_spread of k says the widths are k times too narrow,
    # while z_spread near 1 with a large z_bias says the location is wrong and
    # widening would be the wrong fix. ``implied_width_inflation`` is the factor
    # the reported standard deviations would have to be multiplied by.
    z_bias = z_score.mean(axis=0)
    z_spread = z_score.std(axis=0)
    rms_z = np.sqrt(np.mean(z_score**2, axis=0))
    report.update(
        {
            "overall_rms_z": float(np.sqrt(np.mean(z_score**2))),
            "overall_mean_abs_z_bias": float(np.mean(np.abs(z_bias))),
            "overall_mean_z_spread": float(np.mean(z_spread)),
            "implied_width_inflation": float(np.median(rms_z)),
            "per_parameter_z_bias": {
                parameter_names[i]: float(z_bias[i]) for i in range(param_dim)
            },
            "per_parameter_z_spread": {
                parameter_names[i]: float(z_spread[i]) for i in range(param_dim)
            },
        }
    )
    if map_diagnostics:
        def _median_of(key: str) -> float | None:
            values = [
                float(entry[key])
                for entry in map_diagnostics
                if entry.get(key) is not None
            ]
            return float(np.median(values)) if values else None

        report["map_covariance_method"] = str(
            map_diagnostics[0].get("covariance_method", "")
        )
        report["median_covariance_width_ratio_vs_gauss_newton"] = _median_of(
            "covariance_width_ratio_vs_gauss_newton"
        )
        report["median_chi2_inflation_factor"] = _median_of("chi2_inflation_factor")
    print(
        f"Test-time (MAP+Laplace) validation over {n_cases} held-out cases: "
        f"mean chi2/N(MAP) {report['mean_chi2_per_n']:.2f} | "
        f"median {report['median_chi2_per_n']:.2f} | "
        f"mean 90% coverage {report['overall_coverage_90']:.2f} (target 0.90)"
    )
    print(
        f"  calibration decomposition: mean |z bias| "
        f"{report['overall_mean_abs_z_bias']:.2f} | mean z spread "
        f"{report['overall_mean_z_spread']:.2f} (target 1.00) | widths would "
        f"need x{report['implied_width_inflation']:.2f}"
    )
    return report


def _resolve_amplitude_reparameterization(
    config: Any,
    task: SBITask,
    reader: Any,
    preprocessing_metadata: Any,
    parameter_names: Any,
    embedding_dim: int,
    seed: int,
) -> tuple[Any, Any]:
    """Fit the observation-conditioned coordinate for one amplitude-driven parameter.

    Returns ``(reparameterization, report)``, both ``None`` when the feature is
    off or cannot be fitted. Refuses rather than degrades when the prerequisites
    are missing: the coordinate reads the exact log-amplitude out of a known slot
    in the conditioning vector, so it is meaningless without the exact-summary
    channel, and the fitted ratio box has to contain essentially every training
    draw or the flow will pile mass on a cube edge.
    """
    from .flows.reparameterization import fit_amplitude_conditioned_parameter

    parameter_name = getattr(config, "amplitude_conditioned_parameter", None)
    if not parameter_name:
        return None, None
    if not config.encoder.exact_summary_channel:
        raise ValueError(
            "amplitude_conditioned_parameter requires encoder.exact_summary_channel: "
            "the coordinate is defined by the exact log-amplitude scalar that "
            "channel puts in the conditioning vector."
        )
    names = list(parameter_names)
    if parameter_name not in names:
        raise ValueError(
            f"amplitude_conditioned_parameter {parameter_name!r} is not a free "
            f"parameter of this task; available: {names}"
        )
    parameter_index = names.index(parameter_name)

    block_name = next(iter(reader.dataset.manifest.observation_schema["modalities"]))
    stats = preprocessing_metadata.blocks[block_name]
    if stats.exact_summary_center is None:
        raise ValueError(
            "The preprocessing sidecar carries no exact-summary constants; refit "
            "preprocessing before enabling amplitude_conditioned_parameter."
        )

    # Physical prior bounds, from the cube corners of a monotone prior transform.
    corners = np.asarray(
        task.parameter_layout.transform_cube_to_physical_batch(
            np.stack(
                [
                    np.full(len(names), 1e-6),
                    np.full(len(names), 1.0 - 1e-6),
                ]
            )
        ),
        dtype=float,
    )
    prior_lower = float(min(corners[0, parameter_index], corners[1, parameter_index]))
    prior_upper = float(max(corners[0, parameter_index], corners[1, parameter_index]))

    # A contiguous leading chunk, not a shuffled sample: the split assignment is
    # already randomised at generation time, so the first rows are an unbiased
    # draw, and a shuffled read of a 24M-row split means scattered HDF5 indexing
    # for no statistical gain.
    del seed
    batch = next(
        iter(
            reader.iter_batches(
                split=DatasetSplit.TRAIN,
                batch_size=8192,
                shuffle=False,
                parameter_space="physical",
            )
        )
    )
    physical_values = np.asarray(batch.parameters, dtype=float)[:, parameter_index]
    log_amplitudes = np.asarray(
        [_block_log_spectrum_scale(blocks) for blocks in batch.observations],
        dtype=float,
    )

    reparameterization, report = fit_amplitude_conditioned_parameter(
        parameter_name=parameter_name,
        parameter_index=parameter_index,
        physical_values=physical_values,
        log_amplitudes=log_amplitudes,
        prior_lower=prior_lower,
        prior_upper=prior_upper,
        # The exact-summary channel is appended after the learned embedding, and
        # the log amplitude is its first entry.
        context_index=int(embedding_dim),
        amplitude_center=float(stats.exact_summary_center[0]),
        amplitude_scale=float(stats.exact_summary_scale[0]),
        amplitude_exponent=float(config.amplitude_conditioned_exponent),
    )
    if report["captured_fraction"] < 0.995:
        raise ValueError(
            "The amplitude-conditioned ratio box captures only "
            f"{report['captured_fraction']:.4f} of training draws; a coordinate "
            "that clips this often piles posterior mass on a cube edge."
        )
    print(
        f"Amplitude-conditioned coordinate for {parameter_name}: ratio in "
        f"[{report['ratio_lower']:.4f}, {report['ratio_upper']:.4f}] "
        f"(sd {report['ratio_sd']:.4f}, captured {report['captured_fraction']:.4f}, "
        f"{report['compression_vs_prior']:.2f}x narrower than the log prior range)"
    )
    return reparameterization, report


def _ensure_exact_summary_constants(
    task: SBITask,
    dataset: Any,
    preprocessing_metadata: Any,
    sidecar_path: Path,
    max_samples: int = 262144,
) -> Any:
    """Fill in the exact-summary constants of a sidecar written before they existed.

    Only the new constants are fitted; every pre-existing statistic is left
    byte-for-byte alone, so a dataset keeps the normalization its earlier runs
    used and results stay comparable. Refitting the whole sidecar would reread
    the entire training split -- 42 GB at 24M simulations -- and would perturb
    the per-wavelength transform for no reason.

    Without this the channel would silently emit zeros for every existing
    dataset: the constants default to ``None``, and a disabled channel looks
    exactly like a working one from the outside.
    """
    from .preprocessing import (
        _apply_per_wavelength_value_transform,
        _fit_exact_summary_constants,
        _per_wavelength_stats_arrays,
        _positive_scale,
        exact_summary_feature_matrix,
    )

    missing = [
        name
        for name, stats in preprocessing_metadata.blocks.items()
        if stats.modality == "spectrum" and stats.exact_summary_center is None
    ]
    if not missing:
        return preprocessing_metadata

    print(
        "Fitting exact-summary constants for existing preprocessing sidecar "
        f"({', '.join(missing)}) from up to {max_samples} training rows..."
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in missing}
    n_rows = 0
    for chunk in dataset.iter_split_chunks(
        split=DatasetSplit.TRAIN, chunk_size=8192
    ):
        for name in missing:
            payload = chunk["observations"][name]
            values = np.asarray(payload["values"], dtype=float)
            uncertainties = payload.get("uncertainties")
            stats = preprocessing_metadata.blocks[name]
            per_wavelength = _per_wavelength_stats_arrays(stats, values.shape[-1])
            if per_wavelength is None:
                raise RuntimeError(
                    f"Block {name!r} has no per-wavelength statistics; the "
                    "exact-summary constants cannot be fitted against a "
                    "normalization that does not exist."
                )
            centre_wl, scale_wl = per_wavelength
            normalized, normalized_uncertainties = (
                _apply_per_wavelength_value_transform(
                    values,
                    None if uncertainties is None else np.asarray(uncertainties, dtype=float),
                    centre_wl,
                    scale_wl,
                )
            )
            with np.errstate(all="ignore"):
                sample_scale = np.nanmedian(np.abs(np.where(np.isfinite(values), values, np.nan)), axis=1)
            sample_scale = np.where(
                np.isfinite(sample_scale) & (sample_scale > 0.0),
                sample_scale,
                _positive_scale(stats.value_median, stats.value_mean, default=1.0),
            )
            collected[name].append(
                exact_summary_feature_matrix(
                    normalized,
                    normalized_uncertainties,
                    payload.get("mask"),
                    np.log10(sample_scale),
                )
            )
        n_rows += int(np.asarray(chunk["parameters"]).shape[0])
        if n_rows >= max_samples:
            break

    updated_blocks = dict(preprocessing_metadata.blocks)
    for name in missing:
        centre, scale = _fit_exact_summary_constants(
            np.concatenate(collected[name], axis=0)[:max_samples]
        )
        updated_blocks[name] = dataclasses.replace(
            updated_blocks[name],
            exact_summary_center=centre,
            exact_summary_scale=scale,
        )
        print(
            f"  {name}: centre {np.round(centre, 4).tolist()} "
            f"scale {np.round(scale, 4).tolist()} (from {n_rows} rows)"
        )

    preprocessing_metadata = dataclasses.replace(
        preprocessing_metadata, blocks=updated_blocks
    )
    write_json(sidecar_path, preprocessing_metadata.to_payload())
    return preprocessing_metadata


def _refit_preprocessing(task: SBITask, dataset: Any, sidecar_path: Path) -> Any:
    """Rebuild and persist preprocessing metadata from a dataset's train split."""
    training_split = dataset.read_split(DatasetSplit.TRAIN)
    training_blocks = build_observation_block_batch(
        training_split["observations"],
        task.observation_schema.modalities,
    )
    preprocessing_metadata = fit_task_preprocessing(
        training_blocks,
        version=task.preprocessing_version,
        metadata={
            "task_name": task.name,
            "task_fingerprint": task.task_fingerprint.digest,
            "split_policy": dataset.manifest.metadata.get("split_policy", "shuffled"),
        },
        spectral_value_space=config.spectral_value_space,
    )
    write_json(sidecar_path, preprocessing_metadata.to_payload())
    return preprocessing_metadata


def _save_checkpoint_guardrails(
    config: SBIConfig,
    run_root: Path,
    checkpoint_dir: Path,
    task: SBITask,
    posterior: Any,
    raw_observation_blocks: list,
    spectrum_y_label: str,
) -> dict[str, Any]:
    """Run inference for each saved checkpoint kind and persist a guardrail
    spectrum / plot / summary, so a regression in checkpoint selection is visible
    without re-running the model."""
    if not config.save_checkpoint_guardrails:
        return {}

    retrieval_config = config.retrieval_config
    guardrail_artifacts: dict[str, Any] = {}
    for offset, checkpoint_kind in enumerate(
        ("best_selection", "best_loss", "best_stable")
    ):
        restored = load_trainer_checkpoint_state(
            checkpoint_dir, checkpoint_kind=checkpoint_kind
        )
        if restored is None:
            continue
        restored_state, checkpoint_metadata = restored
        checkpoint_model = restored_state.get("model")
        if checkpoint_model is None:
            continue

        original_model = posterior.model
        posterior.model = checkpoint_model
        try:
            checkpoint_result = AmortizedRetrieval(
                task=task, posterior_estimator=posterior
            ).infer(
                raw_observation_blocks,
                n_posterior_samples=min(
                    config.guardrail_posterior_samples, config.n_posterior_samples
                ),
                include_posterior_predictive=True,
                posterior_predictive_interval_level=0.9,
                n_predictive_forward_model_samples=min(
                    config.guardrail_predictive_forward_model_samples,
                    config.n_predictive_forward_model_samples,
                ),
                seed=config.seed + 100 + offset,
            )
        except Exception as error:
            guardrail_artifacts[checkpoint_kind] = {
                "checkpoint_metadata": checkpoint_metadata,
                "error": str(error),
            }
            continue
        finally:
            posterior.model = original_model

        if checkpoint_result.posterior_predictive is None:
            continue

        base = run_root / f"{checkpoint_kind}_target_posterior_predictive_guardrail"
        plot_path = base.with_suffix(".png")
        spectrum_path = base.with_suffix(".txt")
        summary_path = base.with_suffix(".json")
        save_predictive_spectrum(
            spectrum_path, retrieval_config, checkpoint_result.posterior_predictive
        )

        predictive = checkpoint_result.posterior_predictive
        name = next(iter(predictive.observed_values))
        plot_saved = False
        try:
            figure, _ = plot_posterior_predictive_report(
                predictive,
                dataset_names=[name],
                x_coordinates=np.asarray(
                    retrieval_config.data[name].wavelengths, dtype=float
                ),
                x_label=config.spectrum_x_label,
                y_label=spectrum_y_label,
            )
            figure.savefig(plot_path, dpi=config.plot_dpi, bbox_inches="tight")
            plot_saved = True
        except ImportError as error:
            print(
                "Skipping checkpoint guardrail plot generation because "
                f"matplotlib is unavailable: {error}"
            )

        guardrail_payload = {
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_metadata": checkpoint_metadata,
            "ood_diagnostic": serialize_ood_diagnostic(
                checkpoint_result.ood_diagnostic
            ),
            "posterior_predictive_plot_path": (
                None if not plot_saved else plot_path.as_posix()
            ),
            "posterior_predictive_spectrum_path": spectrum_path.as_posix(),
            "posterior_predictive_summary": summarize_predictive_report(predictive),
        }
        write_json(summary_path, guardrail_payload)
        guardrail_artifacts[checkpoint_kind] = {
            **guardrail_payload,
            "summary_path": summary_path.as_posix(),
        }

    return guardrail_artifacts


def run_sbi(config: SBIConfig) -> Path:
    """Run the full amortized-SBI validation workflow and return the run directory.

    This is the single end-to-end entry point: given one :class:`SBIConfig` (the
    built ``RetrievalConfig`` plus the grouped estimator/workflow knobs), it
    builds the task, obtains a simulation dataset, trains or loads the conditional
    flow posterior, validates it, runs inference on the real target observation,
    and writes every artifact to a self-contained run directory. It is designed to
    be deterministic (seeded) and resumable, and to fail loudly on stale or
    mismatched inputs via the task fingerprint.

    Workflow stages
    ---------------
    1. **Task construction.** Build an :class:`SBITask` from
       ``config.retrieval_config``, attaching the noise model and the
       per-dataset prior-predictive value constraint (default: strictly positive
       flux / transit depth). The task's fingerprint is derived from the priors,
       units, and forward-model source code.
    2. **Run/dataset resolution.** Resolve the output layout under
       ``config.output_root``: a fingerprint-keyed dataset path under
       ``datasets/`` and a per-run directory under ``runs/<run_label>``. When
       ``config.load_completed_model`` is set, locate the source run (explicit
       ``run_label`` or the latest completed run) and, if
       ``config.resume_predictive_checkpoints`` is set, reuse its directory.
    3. **Dataset generation or reuse.** Reuse an existing ``.h5`` dataset when its
       fingerprint matches (rebuilding the preprocessing sidecar if missing);
       otherwise generate ``config.n_simulations`` simulations split into
       train/validation/test, fitting preprocessing on the train split. Dataset
       *resume* is disabled when value-constraint rejection sampling is active
       (rejection changes RNG consumption).
    4. **Reader construction.** Wrap the dataset in a
       :class:`NormalizedObservationDatasetReader`, which on construction runs the
       observation/training flux-scale guard and aborts before training on a
       units/scale mismatch or a stale/corrupt dataset.
    5. **Train or load the posterior.** Either load a previously trained,
       fingerprint-compatible posterior, or build and ``fit`` a
       :class:`ConditionalFlowPosterior` (the ``npe`` objective, or ``elbo`` with
       a lazily-built differentiable forward-model likelihood), then save it.
    6. **Checkpoint guardrails.** For each saved checkpoint kind
       (best-selection / best-loss / best-stable), run inference on the target and
       persist a predictive spectrum, plot, and summary so a checkpoint-selection
       regression is visible without retraining.
    7. **Held-out validation.** Compute a simulation-based calibration (SBC)
       report and a posterior-predictive report on the held-out *test* split.
    8. **Optional test-time refinement** (any objective). When
       ``config.refinement.method`` is set, sharpen the posterior for the specific
       target spectrum with the differentiable forward model -- ``map_laplace``
       (MAP + Laplace / Gaussian-VI, optionally validated on held-out cases) or
       ``elbo`` (flow-ELBO). Validation/SBC stay amortized; only the delivered
       target inference uses the refined posterior.
    9. **Target inference.** Run :class:`AmortizedRetrieval` on the real
       observation to produce posterior samples (mapped to physical parameter
       space), a posterior-predictive summary, and an OOD diagnostic.
    10. **Reporting & persistence.** Classify per-parameter informativeness
        (width ratio cross-checked against SBC calibration), optionally render the
        validation plot set, and write the samples, predictive spectrum, summary
        JSON, and run manifest; finally mark the run completed and update the
        ``latest`` pointer.

    Parameters
    ----------
    config:
        Fully-populated :class:`SBIConfig`. It carries the experiment definition
        (the built ``RetrievalConfig`` with priors and model-generating function),
        the grouped estimator configs (``flow``, ``encoder``, ``training``,
        ``stability``, ``refinement``), and the workflow knobs (simulation count,
        sample counts, plotting, guardrails, resume/load flags, labels, and
        optional known truths).

    Returns
    -------
    pathlib.Path
        The run directory (``output_root/runs/<run_label>``) containing all
        artifacts produced by the run.

    Side effects
    ------------
    Writes to disk under ``config.output_root``:

    * ``datasets/<task>_<n>_<fingerprint>.h5`` and a ``.preprocessing.json``
      sidecar (shared across runs of the same task family).
    * ``runs/<run_label>/`` containing ``posterior_model/`` (weights +
      ``metadata.json``), ``checkpoints/`` and ``predictive_checkpoints/``, the
      target ``*.npy`` samples/log-probabilities, the target predictive spectrum,
      ``amortized_summary.json``, the checkpoint-guardrail artifacts, the optional
      ``test_time_validation_report.json`` and validation plots, and a run
      manifest. The ``latest`` run pointer under ``output_root`` is updated on
      completion.

    Raises
    ------
    FileExistsError
        If the target run directory already exists and is not being reused for a
        predictive-checkpoint resume.
    RuntimeError
        If a cached dataset or a loaded posterior fingerprint does not match the
        current task, or if inference unexpectedly returns non-physical samples.
    ValueError
        If the ELBO objective is requested for a task whose runtime has more than
        one model group or observation (not yet supported by the ELBO likelihood).

    Notes
    -----
    Progress is reported to stdout throughout (device list, split counts, memory
    estimates, per-stage status, validation metrics, and the per-parameter
    informativeness verdict). The function is the orchestration layer only: the
    science lives in the simulator, encoder, flow, objective, and calibration
    modules it calls.
    """
    devices = jax.devices()
    print(f"JAX devices: {devices}")

    retrieval_config = config.retrieval_config
    observation_name = next(iter(retrieval_config.data))
    transit_data = retrieval_config.data[observation_name]
    observation_file = (
        None
        if transit_data.path_to_observations is None
        else Path(str(transit_data.path_to_observations))
    )
    print(f"Using observation file: {observation_file}")

    output_root = Path(config.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {output_root}")
    print(f"Model function: {transit_data.model_generating_function.__name__}")
    print(f"Loaded points: {int(np.size(transit_data.wavelengths))}")
    print(
        "Wavelength range: "
        f"{float(np.min(transit_data.wavelengths)):.3f}--"
        f"{float(np.max(transit_data.wavelengths)):.3f} micron"
    )

    value_constraint = config.observation_value_constraint or ObservationValueConstraint(
        min_value=0.0, min_inclusive=False
    )
    task = SBITask.from_retrieval_config(
        retrieval_config,
        simulation_config=SimulationConfig(
            batch_size=config.simulation_batch_size,
            noise_model=NoiseModelConfig(
                mode="observational",
                seed=config.seed,
                resample_during_training=config.noise_resample_during_training,
            ),
            observation_value_constraints={
                name: value_constraint for name in retrieval_config.data
            },
        ),
        metadata={"preprocessing_version": config.preprocessing_version},
    )

    # Sequential (truncated) round: re-express the cube over a box around the
    # source round's posterior. Wrapping the parameter layout is the whole
    # mechanism -- the simulator draws the unit cube exactly as before, and
    # every consumer that turns a cube coordinate into a physical value now
    # goes through the box. The prior quantiles the fingerprint records change
    # with it, so a truncated round cannot reuse its parent's dataset.
    truncation_box = None
    truncation_report = None
    if config.truncation.enabled:
        from dataclasses import replace as _dataclass_replace

        from .truncation import TruncatedParameterLayout

        truncation_box, truncation_report, _truncation_source = _resolve_truncation(
            config, task, output_root, build_raw_observation_blocks(retrieval_config)
        )
        task = _dataclass_replace(
            task,
            parameter_layout=TruncatedParameterLayout(
                base=task.parameter_layout, box=truncation_box
            ),
            metadata={**dict(task.metadata), "truncation_box": truncation_box.to_payload()},
        )

    split_counts = infer_split_counts(config.n_simulations)
    dataset_dir = output_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{task.name}_{config.n_simulations}_{task.task_fingerprint.digest[:12]}"
    dataset_path = dataset_dir / f"{basename}.h5"
    preprocessing_sidecar = dataset_dir / f"{basename}.preprocessing.json"
    dataset_uri = dataset_path.as_posix()
    run_config = config.to_run_config()

    source_run_root: Path | None = None
    source_run_label: str | None = None
    source_manifest: dict[str, Any] | None = None
    if config.load_completed_model:
        if config.run_label is not None and config.resume_predictive_checkpoints:
            source_run_root = (output_root / "runs" / config.run_label).resolve()
            source_manifest = resolve_run_manifest(source_run_root)
            resolve_posterior_dir(source_run_root, source_manifest)
        else:
            source_run_root, source_manifest = resolve_latest_posterior_run(output_root)
        source_run_label = str(source_manifest.get("run_label") or source_run_root.name)

    # Resuming training reuses the run directory too: the trainer's own
    # checkpoints live there, and a walltime-limited run must be able to pick
    # up where it stopped rather than start a fresh 8M-simulation training.
    resume_training_run = bool(config.resume_training and config.run_label is not None)
    reuse_existing_run = bool(
        config.load_completed_model
        and source_run_root is not None
        and source_manifest is not None
        and source_run_label is not None
        and config.resume_predictive_checkpoints
    ) or resume_training_run
    if resume_training_run and source_run_root is None:
        run_label = str(config.run_label)
        run_root = (output_root / "runs" / run_label).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"Resuming training in existing run directory: {run_root}")
    elif reuse_existing_run:
        assert source_run_root is not None and source_run_label is not None
        run_label = source_run_label
        run_root = source_run_root
        run_root.mkdir(parents=True, exist_ok=True)
        print(f"Reusing run directory for predictive-checkpoint resume: {run_root}")
    else:
        run_label = config.run_label or build_run_label(run_config)
        run_root = output_root / "runs" / run_label
        if run_root.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_root}. Set run_label "
                "explicitly or remove the existing directory before rerunning."
            )
        run_root.mkdir(parents=True, exist_ok=False)

    saved_posterior_dir = run_root / "posterior_model"
    if source_run_root is not None and source_manifest is not None:
        saved_posterior_dir = resolve_posterior_dir(source_run_root, source_manifest)

    checkpoint_dir = run_root / "checkpoints"
    predictive_checkpoint_dir = run_root / "predictive_checkpoints"
    summary_path = run_root / "amortized_summary.json"
    parameter_names = list(task.parameter_names)
    n_wavelength_points = int(np.size(transit_data.wavelengths))
    estimated_train_preload_gib = estimate_preloaded_split_memory_gib(
        n_samples=int(split_counts[DatasetSplit.TRAIN.value]),
        n_wavelength_points=n_wavelength_points,
        n_parameters=len(parameter_names),
    )
    estimated_val_preload_gib = estimate_preloaded_split_memory_gib(
        n_samples=int(split_counts[DatasetSplit.VALIDATION.value]),
        n_wavelength_points=n_wavelength_points,
        n_parameters=len(parameter_names),
    )
    n_devices = max(1, len(devices))
    effective_generation_batch = config.simulation_batch_size * n_devices
    train_batch_size = config.training.batch_size

    print("Production split counts:", split_counts)
    print(
        "Effective simulator batch per generation step:",
        effective_generation_batch,
        f"(configured per-device batch: {config.simulation_batch_size})",
    )
    print(
        "Effective global training batch:",
        train_batch_size,
        f"(~{train_batch_size // n_devices} per GPU on {n_devices} devices)",
    )
    print(
        "Estimated host RAM for eager split preload: "
        f"train ~{estimated_train_preload_gib:.1f} GiB, "
        f"validation ~{estimated_val_preload_gib:.2f} GiB"
    )
    existing_predictive_checkpoints = count_predictive_case_checkpoints(
        predictive_checkpoint_dir
    )
    if reuse_existing_run and existing_predictive_checkpoints:
        print(
            "Existing predictive checkpoints found:",
            existing_predictive_checkpoints,
            f"/ {config.validation_max_cases} held-out cases",
        )

    write_run_manifest(
        run_root=run_root,
        run_label=run_label,
        run_config=run_config,
        task=task,
        dataset_uri=dataset_uri,
        posterior_dir=saved_posterior_dir,
        summary_path=None,
        status="validating"
        if (config.load_completed_model or reuse_existing_run)
        else "training",
    )

    store = HDF5SimulationDatasetStore(chunk_size=train_batch_size)
    dataset = None
    if dataset_path.exists():
        dataset = store.open(dataset_uri)
        expected_fingerprint = task.task_fingerprint.digest
        cached_fingerprint = str(getattr(dataset.manifest, "task_fingerprint", "") or "")
        if cached_fingerprint and cached_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "Existing simulation dataset fingerprint does not match the current "
                f"task. Expected {expected_fingerprint}, found {cached_fingerprint}. "
                f"Remove the stale dataset and rerun: {dataset_path}"
            )
        # Completeness guard: an aborted generation leaves a partial file whose
        # manifest records fewer rows than requested (e.g. train=768 with no
        # validation split). Reusing it silently starves the trainer -- every
        # sharded batch is dropped and the loss is nan for the whole run -- so
        # move the partial artifact aside and regenerate.
        manifest_splits = {
            name: int(count)
            for name, count in dict(getattr(dataset.manifest, "splits", {}) or {}).items()
        }
        missing = {
            name: (manifest_splits.get(name, 0), int(count))
            for name, count in split_counts.items()
            if manifest_splits.get(name, 0) < int(count)
        }
        if missing:
            import time as _time

            suffix = _time.strftime("partial-%Y%m%dT%H%M%SZ", _time.gmtime())
            partial_path = dataset_path.with_name(f"{dataset_path.name}.{suffix}")
            dataset = None
            dataset_path.rename(partial_path)
            if preprocessing_sidecar.exists():
                preprocessing_sidecar.rename(
                    preprocessing_sidecar.with_name(
                        f"{preprocessing_sidecar.name}.{suffix}"
                    )
                )
            print(
                "WARNING: existing simulation dataset is incomplete "
                f"(rows found vs requested per split: {missing}); it is likely "
                "left over from an aborted generation run. Moved it aside to "
                f"{partial_path} and regenerating from scratch."
            )
    if dataset is not None:
        if preprocessing_sidecar.exists():
            preprocessing_metadata = TaskPreprocessingMetadata.from_payload(
                json.loads(preprocessing_sidecar.read_text())
            )
        else:
            print("Rebuilding preprocessing metadata for existing dataset...")
            preprocessing_metadata = _refit_preprocessing(
                task, dataset, preprocessing_sidecar
            )
        if config.encoder.exact_summary_channel:
            preprocessing_metadata = _ensure_exact_summary_constants(
                task, dataset, preprocessing_metadata, preprocessing_sidecar
            )
        print("Reusing existing simulation dataset:", dataset_path)
    else:
        print("Generating simulation dataset...")
        dataset_resume_enabled = not bool(
            task.simulation_config.observation_value_constraints
        )
        if not dataset_resume_enabled:
            print(
                "Dataset resume disabled because simulation-value rejection sampling "
                "changes RNG consumption."
            )
        generated = generate_simulation_dataset(
            task=task,
            storage_uri=dataset_uri,
            n_simulations=config.n_simulations,
            chunk_size=config.simulation_batch_size,
            split_counts=split_counts,
            dataset_version="0.1.0",
            resume=dataset_resume_enabled,
            store_covariance=False,
        )
        print("Simulation dataset splits:", generated.manifest.splits)
        dataset = store.open(generated.manifest)
        assert generated.preprocessing_metadata is not None
        preprocessing_metadata = generated.preprocessing_metadata
        write_json(preprocessing_sidecar, preprocessing_metadata.to_payload())

    # The reader runs the observation/training flux-scale guard automatically on
    # construction (NormalizedObservationDatasetReader.__post_init__): a
    # units/scale mismatch or stale/corrupt dataset aborts here, before training.
    target_value_arrays = {
        name: np.asarray(data.spectrum, dtype=float)
        for name, data in retrieval_config.data.items()
        if getattr(data, "spectrum", None) is not None
    }
    focal_target_log10_median = None
    if target_value_arrays:
        first_target = next(iter(target_value_arrays.values()))
        focal_target_log10_median = float(
            np.log10(max(float(np.median(np.abs(first_target))), 1e-300))
        )
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=preprocessing_metadata,
        target_observations=target_value_arrays,
        scale_check=config.startup_scale_check,
        noise_resampling=task.simulation_config.noise_model.resample_during_training
        and task.simulation_config.noise_model.mode not in ("none", "no_noise"),
        focal_target_log10_median=focal_target_log10_median,
    )

    amplitude_reparameterization, amplitude_reparameterization_report = (
        _resolve_amplitude_reparameterization(
            config,
            task,
            reader,
            preprocessing_metadata,
            parameter_names,
            embedding_dim=int(config.encoder.embedding_dim),
            seed=config.seed + 23,
        )
    )

    posterior: Any
    elbo_log_likelihood_fn = None  # built lazily for ELBO training / refinement
    # Keyed on load_completed_model alone: reuse_existing_run is also set when
    # only the run DIRECTORY is being reused (resuming interrupted training),
    # which must still train rather than load a finished posterior.
    if config.load_completed_model:
        print("Loading previously trained posterior and skipping training...")
        assert source_manifest is not None
        posterior = load_posterior_estimator(saved_posterior_dir.as_posix())
        expected_fingerprint = task.task_fingerprint.digest
        saved_fingerprint = str(
            source_manifest.get("task_fingerprint")
            or getattr(posterior, "task_fingerprint", "")
            or ""
        )
        if saved_fingerprint and saved_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "The saved posterior is incompatible with the rebuilt task. Expected "
                f"task fingerprint {expected_fingerprint}, found {saved_fingerprint}. "
                "Use a completed model trained for the same task definition."
            )
        training_artifacts = getattr(posterior, "training_artifacts", None)
        if training_artifacts is not None:
            print(
                "Loaded trained epochs:",
                len(training_artifacts.history.get("train_loss", [])),
            )
        if source_run_label is not None:
            print("Loaded posterior from completed run:", source_run_label)
        print("Loaded posterior from:", saved_posterior_dir)
    else:
        print("Training conditional flow posterior...")
        posterior = build_conditional_flow_posterior(
            parameter_dim=len(parameter_names),
            n_wavelengths=n_wavelength_points,
            flow=config.flow,
            encoder=config.encoder,
            training=config.training,
            stability=config.stability,
            seed=config.seed,
            checkpoint_directory=checkpoint_dir.as_posix(),
            resume_from_checkpoint=config.resume_training,
            verbose_diagnostics=config.verbose_training_diagnostics,
            diagnostics_output_directory=run_root.as_posix(),
            diagnostics_plot_interval=config.training_diagnostics_plot_interval,
            reparameterization=amplitude_reparameterization,
        )
        if config.training.training_objective == "elbo":
            import warnings

            warnings.warn(
                "training_objective='elbo' is EXPERIMENTAL: the amortized "
                "variational (reverse-KL) objective is mode-seeking and has not "
                "converged on production retrieval benchmarks (validation loss "
                "plateaus at hundreds of sigma per point; SBC is miscalibrated, "
                "and any apparent fit comes from test-time refinement alone). "
                "Use the default NPE objective unless you are explicitly "
                "experimenting with the ELBO path.",
                stacklevel=2,
            )
            print(
                "WARNING: ELBO objective is experimental and has not converged on "
                "production benchmarks; prefer training_objective='npe'."
            )
            print(
                "Objective: amortized variational (ELBO) -- building "
                "differentiable forward-model likelihood (loads opacities)..."
            )
            elbo_log_likelihood_fn = _build_elbo_log_likelihood_fn(task, preprocessing_metadata)
            training_artifacts = posterior.fit(
                reader,
                elbo_log_likelihood_fn=elbo_log_likelihood_fn,
                elbo_num_samples=config.training.elbo_num_samples,
            )
        else:
            training_artifacts = posterior.fit(reader)
        posterior.save(saved_posterior_dir.as_posix())
        print("Finished training epochs:", len(training_artifacts.history["train_loss"]))
        print("Posterior saved to:", saved_posterior_dir)

    raw_observation_blocks = build_raw_observation_blocks(retrieval_config)
    spectrum_y_label = _resolve_spectrum_y_label(config)
    guardrail_checkpoint_dir = checkpoint_dir
    if (config.load_completed_model or reuse_existing_run) and source_run_root is not None:
        guardrail_checkpoint_dir = source_run_root / "checkpoints"
    checkpoint_guardrails = _save_checkpoint_guardrails(
        config,
        run_root=run_root,
        checkpoint_dir=guardrail_checkpoint_dir,
        task=task,
        posterior=posterior,
        raw_observation_blocks=raw_observation_blocks,
        spectrum_y_label=spectrum_y_label,
    )

    write_run_manifest(
        run_root=run_root,
        run_label=run_label,
        run_config=run_config,
        task=task,
        dataset_uri=dataset_uri,
        posterior_dir=saved_posterior_dir,
        summary_path=None,
        status="validating",
    )

    from .preprocessing import normalize_observation_blocks

    print("Computing validation reports on held-out simulations...")
    validation_sbc_report = generate_sbc_report(
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(config.n_posterior_samples, 1024),
        batch_size=train_batch_size,
        max_cases=config.validation_max_cases,
        seed=config.seed + 1,
    )
    # What does the embedding actually know? A broad posterior marginal is
    # ambiguous on its own -- it can mean the flow failed to decode the
    # embedding, or that the encoder never carried the information. Only the
    # second is an encoder problem, and the two call for opposite fixes.
    try:
        embedding_information_report = generate_embedding_information_report(
            posterior,
            reader,
            parameter_names,
            split=DatasetSplit.TEST,
            max_cases=8192,
            batch_size=train_batch_size,
            seed=config.seed + 5,
            # Also measure the scatter over held-out spectra resembling the
            # target, which is the only figure comparable with a reference
            # retrieval's width for this observation.
            target_normalized_values=np.ravel(
                np.asarray(
                    normalize_observation_blocks(
                        raw_observation_blocks, preprocessing_metadata
                    )[0].values,
                    dtype=float,
                )
            ),
        )
        uninformative = sorted(
            name
            for name, value in embedding_information_report["r2_nonlinear"].items()
            if value < 0.05
        )
        if uninformative:
            print(
                f"  embedding carries no usable information on {len(uninformative)}/"
                f"{len(parameter_names)} parameters: {', '.join(uninformative)}"
            )
        statistic = embedding_information_report.get("observation_statistic", {}).get(
            "log10_median_spectrum_scale"
        )
        if statistic is not None:
            print(
                "  conditioning recovers log10(median depth) at R2 "
                f"{statistic['r2_nonlinear']:.6f} "
                f"(residual {statistic['residual_sd']:.5f} of a prior spread "
                f"{statistic['prior_sd']:.4f})"
            )
    except Exception as error:  # pragma: no cover - a probe must not fail a run
        print(f"  (embedding information probe skipped: {error})")
        embedding_information_report = None

    # Where, layer by layer, does that statistic stop being recoverable? This
    # separates "the encoder never had it" from "a specific layer removed it",
    # and only the second names something to change.
    try:
        encoder_stage_report = generate_encoder_stage_information_report(
            posterior,
            reader,
            split=DatasetSplit.TEST,
            max_cases=2048,
            batch_size=train_batch_size,
            seed=config.seed + 6,
        )
        for stage_name, stage in encoder_stage_report.get("stages", {}).items():
            print(
                f"  stage {stage_name:<22s} width {stage['width']:>6d} | "
                f"R2 {stage['r2_best']:.6f} | residual {stage['residual_sd']:.5f}"
            )
    except Exception as error:  # pragma: no cover - a probe must not fail a run
        print(f"  (encoder stage audit skipped: {error})")
        encoder_stage_report = None

    validation_predictive_report = generate_posterior_predictive_report(
        task,
        posterior,
        reader,
        split=DatasetSplit.TEST,
        n_posterior_samples=min(config.n_posterior_samples, 1024),
        interval_level=0.9,
        max_cases=config.validation_max_cases,
        seed=config.seed + 2,
        n_predictive_forward_model_samples=config.n_predictive_forward_model_samples,
        checkpoint_directory=predictive_checkpoint_dir.as_posix(),
    )
    print("SBC mean rank:", validation_sbc_report.mean_rank)
    print(
        "SBC normalized mean-rank error:",
        validation_sbc_report.normalized_mean_rank_error,
    )

    # Optional test-time, per-observation refinement. The amortized posterior
    # is a warm-start; the delivered target inference uses the refined
    # posterior, while validation/SBC stay amortized. Available for BOTH
    # objectives: it was historically gated to ELBO runs, but an NPE warm
    # start benefits equally -- the amortized flow struggles to represent the
    # razor-thin joint degeneracy ridges a low-noise spectrum demands
    # (transmission: radius vs cloud-top height), and the likelihood-based
    # MAP+Laplace step enforces exact data consistency for the delivered fit.
    target_posterior = posterior
    refinement = config.refinement
    is_diagnostics = None
    test_time_validation_report = None
    if refinement.method != "none":
        from .preprocessing import normalize_observation_blocks
        from .training import _prestack_observation_blocks

        normalized_target_blocks = normalize_observation_blocks(
            raw_observation_blocks, preprocessing_metadata
        )
        target_prestacked = _prestack_observation_blocks([normalized_target_blocks])
        # The likelihood is scored in raw physical units, taken straight from
        # the observation (never re-derived from the normalized copy).
        target_raw_block = raw_observation_blocks[0]
        target_raw_arrays = (
            np.asarray(target_raw_block.values, dtype=float),
            np.asarray(target_raw_block.uncertainties, dtype=float),
            None
            if target_raw_block.mask is None
            else np.asarray(target_raw_block.mask),
        )

        if refinement.method == "map_laplace":
            print(
                "MAP + Laplace test-time inference on the target spectrum "
                f"({refinement.map_num_steps} steps, lr={refinement.map_learning_rate}; "
                "loads opacities)..."
            )
            # Build the differentiable forward model once; reuse it for the target
            # and for the held-out test-time validation cases.
            make_residual = _build_residual_fn_factory(task)
            target_residual_fn = make_residual(*target_raw_arrays)
            map_kwargs = dict(
                num_steps=refinement.map_num_steps,
                learning_rate=refinement.map_learning_rate,
                beta_start=refinement.map_beta_start,
                beta_anneal_fraction=refinement.map_beta_anneal_fraction,
                covariance_method=refinement.map_covariance_method,
                covariance_chi2_inflation=refinement.map_covariance_chi2_inflation,
                vi_num_steps=refinement.map_vi_num_steps,
                vi_learning_rate=refinement.map_vi_learning_rate,
                vi_num_samples=refinement.map_vi_num_samples,
                num_starts=refinement.map_num_starts,
                start_selection=refinement.map_start_selection,
                num_scored_starts=refinement.map_num_scored_starts,
                start_forward_batch_size=int(
                    refinement.is_forward_batch_size
                    if refinement.is_forward_batch_size is not None
                    else config.simulation_batch_size
                ),
                convergence_chi2_per_n=refinement.map_convergence_chi2_per_n,
            )
            if refinement.test_time_validation:
                print(
                    "Test-time (MAP+Laplace) validation on "
                    f"{refinement.test_time_validation_cases} held-out cases..."
                )
                # Bound the held-out validation cost: fewer starts per case.
                ttv_map_kwargs = dict(map_kwargs)
                ttv_map_kwargs["num_starts"] = min(
                    2, int(refinement.map_num_starts)
                )
                test_time_validation_report = _generate_test_time_validation_report(
                    posterior,
                    reader,
                    make_residual,
                    split=DatasetSplit.TEST,
                    max_cases=refinement.test_time_validation_cases,
                    n_posterior_samples=refinement.test_time_validation_samples,
                    map_kwargs=ttv_map_kwargs,
                    parameter_names=parameter_names,
                    seed=config.seed + 11,
                )
                write_json(
                    run_root / "test_time_validation_report.json",
                    test_time_validation_report,
                )
            target_posterior = posterior.map_laplace_posterior_on_observation(
                target_prestacked,
                target_residual_fn,
                n_posterior_samples=max(2 * config.n_posterior_samples, 8192),
                seed=config.seed + 7,
                **map_kwargs,
            )
        elif refinement.method == "importance_sampling":
            # Inherit the per-device forward-model batch validated for this
            # task during dataset generation unless explicitly overridden.
            is_forward_batch_size = int(
                refinement.is_forward_batch_size
                if refinement.is_forward_batch_size is not None
                else config.simulation_batch_size
            )
            print(
                "Importance-sampling refinement on the target spectrum "
                f"({refinement.is_num_proposal_samples} amortized proposals, "
                f"forward batch {is_forward_batch_size}; loads "
                "opacities)..."
            )
            make_residual = _build_residual_fn_factory(task)
            target_residual_fn = make_residual(*target_raw_arrays)
            target_posterior = posterior.importance_sampling_posterior_on_observation(
                target_prestacked,
                target_residual_fn,
                n_proposal_samples=refinement.is_num_proposal_samples,
                forward_batch_size=is_forward_batch_size,
                n_posterior_samples=config.n_posterior_samples,
                seed=config.seed + 13,
            )
            is_diagnostics = getattr(
                target_posterior, "_importance_sampling_diagnostics", None
            )
            if is_diagnostics is not None:
                write_json(
                    run_root / "importance_sampling_diagnostics.json",
                    is_diagnostics,
                )
        else:  # "elbo"
            if elbo_log_likelihood_fn is None:
                print(
                    "Semi-amortized refinement -- building differentiable "
                    "forward-model likelihood (loads opacities)..."
                )
                elbo_log_likelihood_fn = _build_elbo_log_likelihood_fn(task, preprocessing_metadata)
            print(
                "Semi-amortized (flow-ELBO) refinement on the target spectrum "
                f"({refinement.elbo_num_steps} steps, lr={refinement.elbo_learning_rate}, "
                f"{refinement.elbo_num_samples} samples/step)..."
            )
            target_posterior = posterior.refine_posterior_on_observation(
                target_prestacked,
                elbo_log_likelihood_fn,
                num_steps=refinement.elbo_num_steps,
                learning_rate=refinement.elbo_learning_rate,
                num_samples=refinement.elbo_num_samples,
                beta_start=config.training.elbo_beta_start,
                beta_anneal_fraction=refinement.elbo_beta_anneal_fraction,
                seed=config.seed + 7,
            )

    # Gate the delivered posterior on the refinement's own diagnostics. A
    # refinement that collapsed onto a single particle, or whose covariance is
    # uncalibrated on held-out cases, is written aside rather than shipped as
    # the answer.
    refinement_verdict = _refinement_verdict(
        refinement.method,
        refinement,
        is_diagnostics,
        test_time_validation_report,
    )
    if not refinement_verdict["accepted"]:
        for reason in refinement_verdict["reasons"]:
            print(f"  REFINEMENT REJECTED ({refinement.method}): {reason}")
        print(
            "  Delivering the amortized posterior instead; the refined result is "
            "kept as rejected_refined_posterior_samples.npy for inspection."
        )
        rejected_posterior = target_posterior
        target_posterior = posterior
        _save_rejected_refinement_samples(
            run_root=run_root,
            posterior=rejected_posterior,
            task=task,
            raw_observation_blocks=raw_observation_blocks,
            preprocessing_metadata=preprocessing_metadata,
            n_posterior_samples=config.n_posterior_samples,
            seed=config.seed + 3,
        )
    write_json(run_root / "refinement_verdict.json", refinement_verdict)
    if truncation_box is not None:
        write_json(run_root / "truncation_box.json", truncation_box.to_payload())
        write_json(run_root / "truncation_report.json", truncation_report)

    def _infer_target(estimator: Any) -> Any:
        """Target inference + predictive for one candidate posterior.

        Both delivery candidates go through this same call -- same sample
        counts, same seed, same interval level -- so the CRPS comparison below
        is like-for-like and not an artefact of differing settings.
        """
        return AmortizedRetrieval(task=task, posterior_estimator=estimator).infer(
            raw_observation_blocks,
            n_posterior_samples=config.n_posterior_samples,
            include_posterior_predictive=True,
            posterior_predictive_interval_level=0.9,
            n_predictive_forward_model_samples=config.n_predictive_forward_model_samples,
            seed=config.seed + 3,
        )

    print("Running amortized inference on the target spectrum...")
    result = _infer_target(target_posterior)

    # Comparative delivery test. The thresholds above ask whether the refinement
    # is bad in absolute terms; this asks whether it beats the amortized
    # posterior it would replace. Only runs when a refinement actually survived
    # the thresholds, so a "none" or already-rejected refinement costs nothing
    # extra.
    if (
        refinement.method != "none"
        and refinement_verdict["accepted"]
        and target_posterior is not posterior
    ):
        print(
            "  Comparing the refined target predictive against the amortized one "
            "(proper scoring rule: CRPS)..."
        )
        amortized_result = _infer_target(posterior)
        comparison = _refinement_predictive_comparison(
            result,
            amortized_result,
            max_crps_ratio=float(refinement.max_refinement_crps_ratio),
            refined_samples=np.asarray(result.posterior.samples, dtype=float),
        )
        refinement_verdict["predictive_comparison"] = comparison
        if not comparison["refined_is_better"]:
            refinement_verdict["accepted"] = False
            refinement_verdict["status"] = "rejected"
            if comparison.get("refined_degenerate_marginals"):
                reason = (
                    f"{comparison['refined_degenerate_marginals']} refined "
                    "marginal(s) have zero width: the refinement collapsed to a "
                    "point estimate, which scores well on any predictive metric "
                    "while carrying no parameter uncertainty"
                )
            else:
                reason = (
                    f"the refined predictive scores worse than the amortized one: "
                    f"CRPS {comparison['refined_crps']:.4g} vs "
                    f"{comparison['amortized_crps']:.4g} "
                    f"(ratio {comparison['crps_ratio']:.2f} > "
                    f"{comparison['max_crps_ratio']:.2f}) -- delivering the "
                    "refinement would ship the worse of the two artifacts computed"
                )
            refinement_verdict["reasons"].append(reason)
            print(f"  REFINEMENT REJECTED ({refinement.method}): {reason}")
            _save_rejected_refinement_samples(
                run_root=run_root,
                posterior=target_posterior,
                task=task,
                raw_observation_blocks=raw_observation_blocks,
                preprocessing_metadata=preprocessing_metadata,
                n_posterior_samples=config.n_posterior_samples,
                seed=config.seed + 3,
            )
            target_posterior = posterior
            result = amortized_result
        else:
            print(
                f"  refinement accepted: CRPS {comparison['refined_crps']:.4g} vs "
                f"amortized {comparison['amortized_crps']:.4g}"
            )
        write_json(run_root / "refinement_verdict.json", refinement_verdict)
    if result.metadata.get("parameter_space") != "physical":
        raise RuntimeError(
            "AmortizedRetrieval is expected to return posterior samples in physical "
            f"space. Received parameter_space={result.metadata.get('parameter_space')!r}."
        )

    posterior_samples = np.asarray(result.posterior.samples, dtype=float)
    degenerate_marginals = _count_degenerate_marginals(posterior_samples)
    if degenerate_marginals:
        print(
            f"  WARNING: {degenerate_marginals}/{posterior_samples.shape[-1]} "
            "delivered marginals have zero width -- the delivered artifact is a "
            "repeated point estimate, not a distribution. Treat every credible "
            "interval below as meaningless."
        )
    np.save(run_root / "target_posterior_samples.npy", posterior_samples)
    np.save(
        run_root / "target_posterior_log_probabilities.npy",
        np.asarray(result.posterior.log_probabilities, dtype=float),
    )
    target_predictive_spectrum_path = save_predictive_spectrum(
        run_root / "target_posterior_predictive_spectrum.txt",
        retrieval_config,
        result.posterior_predictive,
    )

    print("Posterior sample matrix shape:", posterior_samples.shape)
    print("First posterior draw:", posterior_samples[0])
    print("Inference preprocessing applied:", result.metadata["preprocessing_applied"])
    if result.ood_diagnostic is not None:
        print(
            "OOD diagnostic:",
            {
                "score": float(result.ood_diagnostic.score),
                "threshold": result.ood_diagnostic.threshold,
                "passed": result.ood_diagnostic.passed,
            },
        )
    print(
        "Observed values:",
        np.asarray(
            result.posterior_predictive.observed_values[observation_name], dtype=float
        )[:5],
        "...",
    )
    print(
        "Posterior-predictive mean values:",
        np.asarray(
            result.posterior_predictive.predictive_mean[observation_name], dtype=float
        )[:5],
        "...",
    )

    posterior_width_ratios = summarize_posterior_width_ratios(
        posterior_samples, parameter_names, retrieval_config
    )
    # Cross-check the width ratio against the validation SBC calibration: a tiny
    # width ratio with a high SBC rank error is an OVERCONFIDENT collapse
    # (confidently wrong), not a genuine constraint.
    width_ratio_array = [
        (entry["width_ratio"] if entry["width_ratio"] is not None else float("nan"))
        for entry in posterior_width_ratios.values()
    ]
    informativeness = classify_posterior_informativeness(
        width_ratio_array,
        validation_sbc_report.normalized_mean_rank_error,
        list(posterior_width_ratios.keys()),
    )
    n_constrained = sum(1 for v in informativeness.verdict if v == "constrained")
    n_overconfident = sum(1 for v in informativeness.verdict if v == "overconfident")
    print(
        "Posterior informativeness (width ratio + validation-SBC calibration): "
        f"{n_constrained} constrained, {n_overconfident} OVERCONFIDENT/collapsed "
        f"of {len(parameter_names)}"
    )
    verdict_tag = {
        "constrained": "  <-- constrained",
        "overconfident": "  <-- OVERCONFIDENT (collapsed, miscalibrated)",
        "uninformative": "  <-- ~prior (uninformative)",
    }
    for name, width_ratio, verdict in zip(
        informativeness.parameter_names,
        informativeness.width_ratio,
        informativeness.verdict,
    ):
        width_ratio_text = (
            "n/a" if not np.isfinite(width_ratio) else f"{width_ratio:.2f}"
        )
        print(
            f"  {name:28s} width_ratio={width_ratio_text}"
            f"{verdict_tag.get(verdict, '')}"
        )

    # Prior-edge guard. SBC draws its truths from the prior, so it is calibrated
    # by construction exactly where this fails; only the delivered posterior's
    # own position against the bounds can reveal it.
    parameter_units = summarize_parameter_units(retrieval_config, parameter_names)
    prior_edge_report = summarize_prior_edge_occupancy(
        posterior_samples, parameter_names, retrieval_config
    )
    if prior_edge_report["n_parameters_at_prior_edge"]:
        print(
            "WARNING: posterior mass is piled against a prior bound for "
            f"{prior_edge_report['n_parameters_at_prior_edge']} parameter(s); "
            "the prior may be truncating the solution:"
        )
        for name in prior_edge_report["parameters_at_prior_edge"]:
            entry = prior_edge_report["per_parameter"][name]
            print(
                f"  {name:28s} {entry['edge']} bound, "
                f"{max(entry['lower_edge_mass'], entry['upper_edge_mass']):.1%} of mass "
                f"in the outer {prior_edge_report['edge_width']:.0%} "
                f"({entry['enrichment']:.1f}x the prior)"
            )
    else:
        print("Prior-edge guard: no parameter piles against a prior bound.")

    if config.generate_plots:
        save_validation_plots(
            run_root=run_root,
            posterior_samples=posterior_samples,
            parameter_names=parameter_names,
            validation_sbc_report=validation_sbc_report,
            validation_predictive_report=validation_predictive_report,
            target_predictive_report=result.posterior_predictive,
            dataset_names=[observation_name],
            spectrum_wavelengths=np.asarray(transit_data.wavelengths, dtype=float),
            x_label=config.spectrum_x_label,
            y_label=config.spectrum_y_label,
            dpi=config.plot_dpi,
        )

    summary_payload = {
        "run_label": run_label,
        "run_root": run_root.as_posix(),
        "config_hash": compute_run_config_hash(run_config),
        "observation_file": None if observation_file is None else observation_file.as_posix(),
        "dataset_uri": dataset_uri,
        "load_completed_model": config.load_completed_model,
        "resume_predictive_checkpoints": config.resume_predictive_checkpoints,
        "loaded_from_run_root": (
            None if source_run_root is None else source_run_root.as_posix()
        ),
        "loaded_from_run_label": source_run_label,
        "loaded_posterior_dir": saved_posterior_dir.as_posix(),
        "existing_predictive_checkpoints": existing_predictive_checkpoints,
        "checkpoint_guardrails": checkpoint_guardrails,
        "task_name": task.name,
        "task_fingerprint": task.task_fingerprint.digest,
        "parameter_names": parameter_names,
        "split_counts": split_counts,
        "truth_parameters": {
            name: config.truth_parameter_values[name]
            for name in parameter_names
            if name in config.truth_parameter_values
        },
        "n_simulations": config.n_simulations,
        "simulation_batch_size": config.simulation_batch_size,
        "train_batch_size": train_batch_size,
        "num_epochs": config.training.num_epochs,
        "early_stopping_patience": config.training.early_stopping_patience,
        "n_posterior_samples": config.n_posterior_samples,
        "n_predictive_forward_model_samples": config.n_predictive_forward_model_samples,
        "effective_generation_batch": effective_generation_batch,
        "estimated_train_preload_gib": estimated_train_preload_gib,
        "estimated_validation_preload_gib": estimated_val_preload_gib,
        "preprocessing_applied": result.metadata["preprocessing_applied"],
        "preprocessing_version": result.metadata["preprocessing_version"],
        "ood_diagnostic": serialize_ood_diagnostic(result.ood_diagnostic),
        "training_diagnostics": summarize_training_diagnostics(posterior),
        "posterior_summary": summarize_posterior_samples(
            posterior_samples, parameter_names, config.truth_parameter_values
        ),
        "truncation_box": (
            None if truncation_box is None else truncation_box.to_payload()
        ),
        "truncation_report": truncation_report,
        "embedding_information": embedding_information_report,
        "encoder_stage_information": encoder_stage_report,
        "amplitude_reparameterization": amplitude_reparameterization_report,
        "refinement_verdict": refinement_verdict,
        "delivered_posterior": (
            "amortized"
            if refinement.method == "none" or not refinement_verdict["accepted"]
            else f"refined_{refinement.method}"
        ),
        "delivered_posterior_degenerate_marginals": _count_degenerate_marginals(
            posterior_samples
        ),
        "posterior_prior_width_ratio": posterior_width_ratios,
        "parameter_units": parameter_units,
        "prior_edge_report": prior_edge_report,
        "n_parameters_constrained": int(n_constrained),
        "sbc_mean_rank": np.asarray(
            validation_sbc_report.mean_rank, dtype=float
        ).tolist(),
        "sbc_normalized_mean_rank_error": np.asarray(
            validation_sbc_report.normalized_mean_rank_error, dtype=float
        ).tolist(),
        "validation_predictive_mean_absolute_error": {
            str(name): float(value)
            for name, value in validation_predictive_report.mean_absolute_error.items()
        },
        "validation_predictive_interval_coverage": {
            str(name): np.asarray(value, dtype=float).tolist()
            for name, value in validation_predictive_report.interval_coverage.items()
        },
        "validation_predictive_summary": summarize_predictive_report(
            validation_predictive_report
        ),
        "target_predictive_summary": summarize_predictive_report(
            result.posterior_predictive
        ),
        "target_predictive_spectrum_path": target_predictive_spectrum_path.as_posix(),
    }
    write_json(summary_path, summary_payload)

    write_run_manifest(
        run_root=run_root,
        run_label=run_label,
        run_config=run_config,
        task=task,
        dataset_uri=dataset_uri,
        posterior_dir=saved_posterior_dir,
        summary_path=summary_path,
        status="completed",
    )
    update_latest_run_pointer(output_root, run_root, run_label)

    print("Wrote summary to:", summary_path)
    print("Wrote target predictive spectrum to:", target_predictive_spectrum_path)
    return run_root


__all__ = ["run_sbi"]
