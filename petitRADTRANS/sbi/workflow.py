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
    classify_posterior_informativeness,
    generate_posterior_predictive_report,
    generate_sbc_report,
    serialize_ood_diagnostic,
    summarize_posterior_contraction,
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
from .estimator_registry import load_posterior_estimator
from .inference import AmortizedRetrieval
from .observation import build_observation_block_batch, build_raw_observation_blocks
from .plotting import plot_posterior_predictive_report, save_validation_plots
from .preprocessing import TaskPreprocessingMetadata, fit_task_preprocessing
from .task import NoiseModelConfig, ObservationValueConstraint, SBITask, SimulationConfig
from .training import load_trainer_checkpoint_state


def _resolve_spectrum_y_label(config: SBIConfig) -> str:
    """Plot y-axis label: explicit override, else derived from the observable."""
    if config.spectrum_y_label is not None:
        return config.spectrum_y_label
    if getattr(config.retrieval_config, "scattering_in_emission", False):
        return r"Flux [W m$^{-2}$ $\mu$m$^{-1}$]"
    return "Transit depth"


def _build_elbo_log_likelihood_fn(task: SBITask):
    """Build a differentiable forward-model log-likelihood for the ELBO objective.

    Returns ``log_likelihood_fn(theta_cube, observations) -> (batch,)``: it maps
    unit-cube parameters to physical values, runs the *same* differentiable pRT
    forward model used to generate the dataset (and used by JAXNS), projects the
    model spectrum onto the observation grid, and scores the batch's observed
    spectra under the known Gaussian noise model. The observed values reaching
    the loss are reader-normalized (``arcsinh(raw/scale)``); we invert that here
    so the likelihood is evaluated in raw units where the noise is Gaussian.
    """
    from petitRADTRANS.sbi.simulator import SimulationRuntime
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

    def log_likelihood_fn(theta_cube, observations):
        # observations is a PreStackedObservations; block 0 is the spectrum.
        block = observations.stacked_blocks[0]
        value_norm = jnp.asarray(block[0], dtype=jnp.float32)   # arcsinh(raw/scale)
        uncert_norm = jnp.asarray(block[1], dtype=jnp.float32)  # raw_sigma/(scale*cosh)
        mask = jnp.asarray(block[3])                            # True == masked out
        scale = jnp.asarray(block[6], dtype=jnp.float32)[:, None]  # (batch, 1)

        # Invert the reader's arcsinh normalization to raw units.
        raw_values = scale * jnp.sinh(value_norm)
        raw_uncert = uncert_norm * scale * jnp.cosh(value_norm)

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

    Returns ``make_residual(target_observation) -> residual(theta_cube)`` where
    ``residual`` is the noise-standardized ``(x_obs - model(theta))/sigma`` (masked
    channels set to 0) in raw units. The expensive ``SimulationRuntime`` (opacity
    load + model context) is constructed a single time, so the per-observation
    closures are cheap -- enabling test-time MAP+Laplace over many held-out cases
    without reloading opacities.
    """
    from petitRADTRANS.sbi.simulator import SimulationRuntime
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

    def make_residual(target_observation):
        block = target_observation.stacked_blocks[0]
        value_norm = jnp.asarray(block[0], dtype=jnp.float32)[0]
        uncert_norm = jnp.asarray(block[1], dtype=jnp.float32)[0]
        mask = jnp.asarray(block[3])[0]
        scale = jnp.asarray(block[6], dtype=jnp.float32)[0]
        raw_values = scale * jnp.sinh(value_norm)
        raw_uncert = jnp.abs(uncert_norm) * scale * jnp.cosh(value_norm)
        valid = (~mask.astype(bool)).astype(jnp.float32)
        sigma = jnp.where(raw_uncert > 0.0, raw_uncert, 1.0)

        def residual(theta_cube):
            theta_physical = parameter_layout.transform_cube_to_physical_batch(
                theta_cube[None, :]
            )
            model_spectrum = jnp.reshape(spectrum_fn(theta_physical), (-1,))
            return valid * (raw_values - model_spectrum) / sigma

        return residual

    return make_residual


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
    from petitRADTRANS.sbi.observation import PreStackedObservations

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
    observations = batch.observations
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
    rank_fraction = np.zeros((n_cases, param_dim))      # P(sample < true)
    inside_90 = np.zeros((n_cases, param_dim), dtype=bool)
    chi2_per_n = np.zeros(n_cases)
    n_wavelengths = None

    for case_index in range(n_cases):
        case_obs = _slice_case(case_index)
        residual = make_residual(case_obs)
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
    print(
        f"Test-time (MAP+Laplace) validation over {n_cases} held-out cases: "
        f"mean chi2/N(MAP) {report['mean_chi2_per_n']:.2f} | "
        f"median {report['median_chi2_per_n']:.2f} | "
        f"mean 90% coverage {report['overall_coverage_90']:.2f} (target 0.90)"
    )
    return report


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
    """Run the full SBI validation workflow and return the run directory."""
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
            noise_model=NoiseModelConfig(mode="observational", seed=config.seed),
            observation_value_constraints={
                name: value_constraint for name in retrieval_config.data
            },
        ),
        metadata={"preprocessing_version": config.preprocessing_version},
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

    reuse_existing_run = bool(
        config.load_completed_model
        and source_run_root is not None
        and source_manifest is not None
        and source_run_label is not None
        and config.resume_predictive_checkpoints
    )
    if reuse_existing_run:
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
        if preprocessing_sidecar.exists():
            preprocessing_metadata = TaskPreprocessingMetadata.from_payload(
                json.loads(preprocessing_sidecar.read_text())
            )
        else:
            print("Rebuilding preprocessing metadata for existing dataset...")
            preprocessing_metadata = _refit_preprocessing(
                task, dataset, preprocessing_sidecar
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
    reader = NormalizedObservationDatasetReader(
        dataset=dataset,
        preprocessing_metadata=preprocessing_metadata,
        target_observations={
            name: np.asarray(data.spectrum, dtype=float)
            for name, data in retrieval_config.data.items()
            if getattr(data, "spectrum", None) is not None
        },
        scale_check=config.startup_scale_check,
    )

    posterior: Any
    elbo_log_likelihood_fn = None  # built lazily for ELBO training / refinement
    if config.load_completed_model or reuse_existing_run:
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
            verbose_diagnostics=config.verbose_training_diagnostics,
            diagnostics_output_directory=run_root.as_posix(),
            diagnostics_plot_interval=config.training_diagnostics_plot_interval,
        )
        if config.training.training_objective == "elbo":
            print(
                "Objective: amortized variational (ELBO) -- building "
                "differentiable forward-model likelihood (loads opacities)..."
            )
            elbo_log_likelihood_fn = _build_elbo_log_likelihood_fn(task)
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

    # Optional test-time, per-observation refinement (ELBO runs only). The
    # amortized posterior is a warm-start; the delivered target inference uses
    # the refined posterior, while validation/SBC stay amortized.
    target_posterior = posterior
    refinement = config.refinement
    if refinement.method != "none" and config.training.training_objective == "elbo":
        from .preprocessing import normalize_observation_blocks
        from .training import _prestack_observation_blocks

        normalized_target_blocks = normalize_observation_blocks(
            raw_observation_blocks, preprocessing_metadata
        )
        target_prestacked = _prestack_observation_blocks([normalized_target_blocks])

        if refinement.method == "map_laplace":
            print(
                "MAP + Laplace test-time inference on the target spectrum "
                f"({refinement.map_num_steps} steps, lr={refinement.map_learning_rate}; "
                "loads opacities)..."
            )
            # Build the differentiable forward model once; reuse it for the target
            # and for the held-out test-time validation cases.
            make_residual = _build_residual_fn_factory(task)
            map_kwargs = dict(
                num_steps=refinement.map_num_steps,
                learning_rate=refinement.map_learning_rate,
                beta_start=refinement.map_beta_start,
                beta_anneal_fraction=refinement.map_beta_anneal_fraction,
                covariance_method=refinement.map_covariance_method,
                vi_num_steps=refinement.map_vi_num_steps,
                vi_learning_rate=refinement.map_vi_learning_rate,
                vi_num_samples=refinement.map_vi_num_samples,
            )
            if refinement.test_time_validation:
                print(
                    "Test-time (MAP+Laplace) validation on "
                    f"{refinement.test_time_validation_cases} held-out cases..."
                )
                test_time_validation_report = _generate_test_time_validation_report(
                    posterior,
                    reader,
                    make_residual,
                    split=DatasetSplit.TEST,
                    max_cases=refinement.test_time_validation_cases,
                    n_posterior_samples=refinement.test_time_validation_samples,
                    map_kwargs=map_kwargs,
                    parameter_names=parameter_names,
                    seed=config.seed + 11,
                )
                write_json(
                    run_root / "test_time_validation_report.json",
                    test_time_validation_report,
                )
            target_residual_fn = make_residual(target_prestacked)
            target_posterior = posterior.map_laplace_posterior_on_observation(
                target_prestacked,
                target_residual_fn,
                n_posterior_samples=max(2 * config.n_posterior_samples, 8192),
                seed=config.seed + 7,
                **map_kwargs,
            )
        else:  # "elbo"
            if elbo_log_likelihood_fn is None:
                print(
                    "Semi-amortized refinement -- building differentiable "
                    "forward-model likelihood (loads opacities)..."
                )
                elbo_log_likelihood_fn = _build_elbo_log_likelihood_fn(task)
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

    print("Running amortized inference on the target spectrum...")
    result = AmortizedRetrieval(task=task, posterior_estimator=target_posterior).infer(
        raw_observation_blocks,
        n_posterior_samples=config.n_posterior_samples,
        include_posterior_predictive=True,
        posterior_predictive_interval_level=0.9,
        n_predictive_forward_model_samples=config.n_predictive_forward_model_samples,
        seed=config.seed + 3,
    )
    if result.metadata.get("parameter_space") != "physical":
        raise RuntimeError(
            "AmortizedRetrieval is expected to return posterior samples in physical "
            f"space. Received parameter_space={result.metadata.get('parameter_space')!r}."
        )

    posterior_samples = np.asarray(result.posterior.samples, dtype=float)
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

    posterior_contraction = summarize_posterior_contraction(
        posterior_samples, parameter_names, retrieval_config
    )
    # Cross-check contraction against the validation SBC calibration: a tiny
    # contraction with a high SBC rank error is an OVERCONFIDENT collapse
    # (confidently wrong), not a genuine constraint.
    contraction_array = [
        (entry["contraction"] if entry["contraction"] is not None else float("nan"))
        for entry in posterior_contraction.values()
    ]
    informativeness = classify_posterior_informativeness(
        contraction_array,
        validation_sbc_report.normalized_mean_rank_error,
        list(posterior_contraction.keys()),
    )
    n_constrained = sum(1 for v in informativeness.verdict if v == "constrained")
    n_overconfident = sum(1 for v in informativeness.verdict if v == "overconfident")
    print(
        "Posterior informativeness (contraction + validation-SBC calibration): "
        f"{n_constrained} constrained, {n_overconfident} OVERCONFIDENT/collapsed "
        f"of {len(parameter_names)}"
    )
    verdict_tag = {
        "constrained": "  <-- constrained",
        "overconfident": "  <-- OVERCONFIDENT (collapsed, miscalibrated)",
        "uninformative": "  <-- ~prior (uninformative)",
    }
    for name, contraction, verdict in zip(
        informativeness.parameter_names,
        informativeness.contraction,
        informativeness.verdict,
    ):
        contraction_text = (
            "n/a" if not np.isfinite(contraction) else f"{contraction:.2f}"
        )
        print(f"  {name:28s} contraction={contraction_text}{verdict_tag.get(verdict, '')}")

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
        "posterior_contraction": posterior_contraction,
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
