# `petitRADTRANS.experimental.sbi` — Module Architecture Overview

Simulation-based inference (SBI) for petitRADTRANS. This package provides an
**amortized** inference workflow built on top of the existing retrieval runtime:
instead of running one MCMC/nested-sampling fit per observed spectrum, it trains a
single conditional neural density estimator (a normalizing flow) once on a large
bank of prior-predictive simulations, and then performs near-instant posterior
inference on any compatible observation.

The end-to-end pipeline is:

```
RetrievalConfig                 (priors + forward model, from the retrieval package)
  │
  └─► SBITask                   (immutable problem definition + fingerprint)
        │
        ├─► RuntimeSimulator    (sample prior → forward model → add noise)
        │      │
        │      └─► generate_simulation_dataset → HDF5 dataset on disk
        │              │
        │              └─► fit_task_preprocessing  (normalization statistics)
        │                      │
        │                      └─► NormalizedObservationDatasetReader (training stream)
        │                              │
        ├──────────────────────────────┴─► ConditionalFlowPosterior.fit()  (trains the flow)
        │                                       │  (encoder + flow + SBITrainer loop)
        │                                       ▼
        │                                  saved posterior_model/  (weights + metadata.json)
        │
        └─► AmortizedRetrieval    (serve trained model: encode → sample → predictive → OOD)
                  │
                  └─► calibration / plotting   (SBC, posterior-predictive, sensitivity reports)

run_sbi(SBIConfig)  orchestrates all of the above as a single benchmark entry point.
```

Everything is JAX-native (flows and encoders are `equinox.Module`s; the simulator
and runtime use `jax.numpy`), so the forward model, encoder, and flow are all
differentiable — which is what enables the optional ELBO objective and test-time
posterior refinement.

---

## 1. Problem definition & provenance

### `task.py`
Bridges the user-facing **retrieval configuration** to the SBI runtime.

- **`SBITask`** (frozen dataclass) — the central immutable description of an SBI
  problem. It owns the `RetrievalConfig`, the canonical `ParameterLayout` (free
  parameter names/order), the `ObservationSchema`, and a `SimulationConfig`.
  `SBITask.from_retrieval_config(...)` is the standard constructor: it infers
  dataset modalities, the forward-model family, and version strings.
  - `build_runtime()` / `build_observation_states()` materialize the retrieval
    runtime and immutable observation states.
  - **`task_fingerprint`** / `fingerprint_payload` produce a deterministic SHA-256
    digest of *everything that determines the simulated spectra* — prior bounds
    (sampled on a fixed cube grid), parameter units, and the **source code** of
    each model-generating function (`_forward_model_signature`). This digest is
    the cache key: editing a prior bound, a unit, or the model function
    invalidates any cached dataset and any saved posterior, preventing silent
    reuse of stale/corrupt simulations.
  - `validate_observation_schema()` / `validate_artifact_metadata()` produce
    `CompatibilityReport`s so a loaded artifact can be checked against the task.
- **`SimulationConfig`** — how prior-predictive simulations are generated: batch
  size, count, noise model, and per-dataset `ObservationValueConstraint`s
  (rejection bounds, e.g. "flux > 0").
- **`NoiseModelConfig`**, **`ObservationSchema`**, **`ObservationValueConstraint`** —
  supporting frozen config dataclasses.

### `compatibility.py`
Pure helpers for provenance and version-safety. No SBI logic.

- **`TaskFingerprint`** — deterministic digest wrapper (`make_digest` →
  canonicalized JSON → SHA-256).
- **`CompatibilityReport`** — structured comparison result; `raise_for_incompatibility()`
  throws **`TaskCompatibilityError`**.
- `compare_named_mappings()` — diffs two payloads key-by-key.

---

## 2. Observation representation

### `observation.py`
Defines the data containers that flow through the entire pipeline.

- **`ObservationModality`** (enum): `spectrum`, `photometry`, `time_series`, `auxiliary`.
- **`ObservationBlock`** (frozen) — one modality-specific block: `values`,
  `uncertainties`, `coordinates` (wavelengths), `mask`, `metadata`. This is the
  universal currency: a real observed spectrum and one simulated draw are both
  represented as lists of `ObservationBlock`s.
- **`EncodedObservation`** (frozen) — the dense embedding produced by an encoder
  and consumed by the flow.
- **`ObservationEncoder`** (ABC) — the encode interface (`encode` /
  `batch_encode`); concrete encoders live in `encoders.py`.
- **`PreStackedObservations`** — a registered JAX pytree holding per-sample block
  data already stacked into dense arrays, so the encoder can be JIT-traced without
  retracing every batch. The key performance bridge across the JIT boundary.
- Builder functions (`build_observation_block`, `build_observation_blocks_from_sample`,
  `build_observation_block_batch`, `build_raw_observation_blocks`) convert
  simulator payloads / retrieval datasets into `ObservationBlock` lists.

---

## 3. Simulation

### `simulator.py`
Generates `(parameters, observation)` pairs — the training signal.

- **`SimulationRuntime`** (extends `retrieval.runtime.RetrievalRuntime`) — adds the
  *simulation* surface (deterministic, noise-free projection of parameters into
  observation space) on top of the runtime's likelihood machinery. Provides
  scalar, batched, and multi-device (`pmap`/sharded) kernels.
- **`Simulator`** / **`BatchedSimulator`** / **`ProposalSampler`** (ABCs) — the
  abstract sampling+forward-model interface.
- **`RuntimeSimulator`** (the concrete workhorse) — samples parameters from the
  retrieval prior with JAX RNG, runs the batched forward model, applies the noise
  model, enforces `ObservationValueConstraint`s via rejection (resampling invalid
  rows), and assembles **`SimulationBatch`** objects. Manages its own splittable
  RNG key for reproducibility, and disables dataset *resume* when rejection
  sampling is active (because rejection changes RNG consumption).
- **`SimulationBatch`** — one batch of sampled parameters + projected/noised
  observation payloads.

---

## 4. Dataset generation & storage

### `dataset.py`
Persists simulations to disk and streams them back for training.

- **`generate_simulation_dataset(...)`** — top-level driver: deterministically
  assigns simulations to train/validation/test splits, runs the simulator in
  chunks, writes them via a store, fits preprocessing on the train split, and
  returns a **`GeneratedSimulationDataset`** (manifest + preprocessing metadata).
  Supports resume.
- **`DatasetSplit`** / **`SplitSamplingPolicy`** (enums); `infer_split_counts()`,
  `estimate_preloaded_split_memory_gib()` — split sizing / RAM estimation helpers.
- **`SimulationDatasetStore`** / **`SimulationDatasetWriter`** (ABCs) and the HDF5
  implementations **`HDF5SimulationDatasetStore`**, **`HDF5SimulationDatasetWriter`**,
  **`HDF5StoredSimulationDataset`** — chunked HDF5 read/write; the stored dataset
  can iterate split chunks or read a whole split.
- **`SimulationDatasetManifest`** — on-disk provenance (carries the task
  fingerprint, used to detect stale caches).
- **`NormalizedObservationDatasetReader`** — the **training-time bridge**: wraps a
  stored dataset + `TaskPreprocessingMetadata` and yields *normalized*
  `ObservationBlock` batches (cached or streaming). On construction it runs the
  **observation/training flux-scale guard** (via `validation.py`) so a units/scale
  mismatch or corrupt dataset aborts *before* training, not after.

---

## 5. Preprocessing (normalization)

### `preprocessing.py`
Computes and applies the normalization that makes training numerically stable.

- **`BlockNormalizationStats`** — per-block statistics: robust per-wavelength
  median/IQR vectors, global value/uncertainty mean/std/median/IQR, and
  `log_median_flux` (an absolute-amplitude feature). Robust statistics are used so
  outliers don't dominate.
- **`TaskPreprocessingMetadata`** — serializable bundle of per-block stats +
  version; round-trips via `to_payload` / `from_payload` (saved as a JSON sidecar
  next to the dataset).
- `compute_block_normalization_stats()`, **`fit_task_preprocessing()`** — fit stats
  from simulated data.
- `normalize_observation_block()` / **`normalize_observation_blocks()`** — apply the
  transform (amplitude/shape split: standardize the shape, keep a separate
  log-amplitude scalar) to observation blocks at both training and inference time.

---

## 6. Encoders (observation → embedding)

### `encoders.py`
JAX/Equinox neural encoders that turn variable observation blocks into a fixed
embedding vector for the flow's conditioning input.

- **`SpectralConvolutionEncoder`** — the **default** (`spectrum_encoder_type="convolution"`).
  Explicitly factors a spectrum into **amplitude** (low-dim: global log-scale +
  amplitude features) and **shape** (within-spectrum standardized structure),
  so the network can never collapse to an amplitude-only summary — the failure
  mode that previously left composition/temperature/cloud marginals stuck at the
  prior.
- **`SpectralConv1DEncoder`** — 1D-convolution encoder retaining positional info.
- **`SpectralPatchEncoder`** — patch-summary + MLP-pool encoder.
- **`PhotometryPointEncoder`** — per-point MLP + pooling for photometry blocks.
- **`DatasetSetAggregator`** — permutation-invariant pooling across multiple
  block embeddings.
- **`ObservationEncoder`** — the top-level encoder (implements
  `ObservationEncoder`). Dispatches each block to the right modality sub-encoder,
  then aggregates. Exposes `encode`, batched `encode_stacked_batch`, and
  `encode_from_prestacked` (the JIT-friendly path consuming `PreStackedObservations`).

---

## 7–8. Flows & posterior estimators — the `flows/` subpackage

Everything flow- and posterior-related lives in the **`flows/`** subpackage. Its
`__init__.py` is a façade that re-exports the public transforms, the base
interfaces/dataclasses, the estimator-family registry + generic loader, and the
concrete estimators; importing it registers both estimator families. (Historical
note: this subpackage replaced the former top-level `flows.py`, `posterior.py`,
`posterior_base.py`, `flow_posterior.py`, `flow_matching_posterior.py`, and
`estimator_registry.py` — all consolidated here.)

### `flows/transforms.py` — low-level flow primitives (was `flows.py`)
Backend-agnostic conditional normalizing-flow components in JAX/Equinox; each
transform implements `forward`/`inverse` with log-determinant tracking. The module
docstring is the canonical guide to the families and when to use each.

- **Coupling transforms**: `ConditionalAffineCoupling`,
  `ConditionalRationalQuadraticSplineCoupling` (+ 1-D `ConditionalScalarSplineTransform`).
- **Autoregressive transforms**: `ConditionalAutoregressiveAffineTransform`,
  **`ConditionalNeuralAutoregressiveTransform`** (NAF — monotonic neural map,
  inverted by bisection + Newton polish; `inverse_with_diagnostics` returns
  **`NeuralInverseDiagnostics`** for invertibility monitoring).
- **Conditioners**: `CouplingConditioner`, `_ScalarSplineConditioner`,
  `_AutoregressiveDimensionConditioner`, `_NeuralAutoregressiveDimensionConditioner`.
- Spline math: `_rational_quadratic_spline_forward/_inverse`, `_normalize_bin_sizes`,
  `_spline_boundary_derivatives`.
- **Flow stacks** (`_ConditionalFlowStack` base + gaussian/logistic base sampling/log-prob):
  `ConditionalAffineFlow`, `ConditionalAutoregressiveFlow`,
  **`ConditionalNeuralAutoregressiveFlow`**, `ConditionalSplineFlow`, plus the
  auxiliary `ConditionalDiagonalAffine` pre-layer. Each exposes `log_prob`,
  `sample`, `forward`, `inverse`.

### `flows/base.py` — interfaces, dataclasses, persistence, registry (was `posterior_base.py` + `estimator_registry.py`)
- **`PosteriorEstimator`** (ABC) — the contract: `fit`, `encode_observation`,
  `sample_posterior`, optional `log_prob`, `save`/`load`.
- **`PersistentPosteriorEstimator`** — shared on-disk persistence: writes
  `metadata.json` (schema-versioned, with task fingerprint, observation schema,
  preprocessing payload, estimator config, training artifacts) plus
  backend-specific weights; rebuilds via the abstract `from_serialized_metadata` +
  `load_backend_state`.
- Data carriers: **`PosteriorBatch`**, **`PosteriorSamples`**, **`TrainingArtifacts`**.
- **Estimator registry** (moved here from the old `estimator_registry.py`):
  `register_posterior_estimator()` / `get_posterior_estimator_class()` (family →
  class map) and **`load_posterior_estimator(dir)`** — reads `metadata.json`,
  resolves the estimator family (with legacy fallbacks), and dispatches to the
  right `.load(...)`. The concrete estimator modules self-register at import.

### `flows/posterior.py` — the core model (was `flow_posterior.py`)
- **`ConditionalFlowPosterior`** (extends `PersistentPosteriorEstimator`,
  `estimator_family="conditional_flow"`) — couples an `ObservationEncoder` with a
  conditional flow (`_PosteriorModel`, defined in `flows/objectives.py`, bundles
  encoder + flow as one Equinox module). The main trainable object; now slimmed to
  construction (`__init__`, `_build_flow`), `fit` orchestration, the inference
  surface (`encode_observation`, `sample_posterior` + batched/jitted variants,
  `log_prob`), persistence, and thin delegators for the objective/refinement/
  diagnostics that were moved to sibling modules. Its class docstring documents
  every constructor parameter.
- **Thin subclasses** fixing the flow family: `ConditionalSplineFlowPosterior`,
  `ConditionalAutoregressiveFlowPosterior`,
  `ConditionalNeuralAutoregressiveFlowPosterior`.

### `flows/cube.py` — parameter-space transforms (extracted helper)
Pure change-of-variables maps between physical / unit-cube / unconstrained
coordinates **with the Jacobian terms** (`_cube_to_unconstrained`,
`_cube_log_abs_det_jacobian`, `_parameter_space_log_prob`, `_flow_space_parameters`,
`_minimum_cube_spline_bound`).

### `flows/regularization.py` — spline regularization + checkpoint diagnostics (extracted helper)
The spline bin/derivative penalties (`_spline_regularization_terms`,
`_spline_layer_regularization_terms`), per-checkpoint spline statistics
(`_spline_parameter_statistics`), the **checkpoint-stability gate**
(`_flow_validation_stability_metrics`, `_flow_validation_selection_metric`), and the
diagnostics batch-slicing helpers.

### `flows/objectives.py` — the training objective (extracted)
**`_PosteriorModel`** (the encoder+flow Equinox container) and the loss functions:
`loss_from_observations` (NPE conditional NLL with log-prob clipping, optional spline
+ invertibility regularization, aux-scale term), `compute_loss`/`training_loss`
(eval vs train), `invertibility_penalty`, `make_elbo_loss` (amortized-variational
objective), `batch_embeddings`, and `compute_validation_diagnostics` (the
flow-invertibility/collapse metrics feeding checkpoint selection).

### `flows/refinement.py` — test-time refinement (extracted)
`refine_posterior_on_observation` (flow-ELBO) and
`map_laplace_posterior_on_observation` (MAP + Laplace / Gaussian-VI) as free
functions taking the posterior; `ConditionalFlowPosterior` exposes them as thin
methods. They sharpen the amortized posterior for one specific target spectrum using
the differentiable forward model.

### `flows/matching.py` — alternative estimator (was `flow_matching_posterior.py`)
- **`FlowMatchingPosterior`** (`estimator_family="flow_matching"`, **experimental**)
  — a continuous-time (flow-matching / velocity-field) estimator;
  `_FlowMatchingVelocityField` + `_FlowMatchingModel`; same `fit` / `encode` /
  `sample_posterior` interface with ODE-integrated sampling (no `log_prob`).

---

## 9. Training loop

### `training.py`
The o
gine used by the posterior estimators' `fit` methods.

- **`SBITrainer`** — reusable JAX/Optax training loop: cosine LR schedule with
  warmup, gradient clipping, weight decay, early stopping
  (**`EarlyStoppingConfig`**), multi-device sharding (`_shard_batch`,
  `_get_device_mesh`), per-epoch metric logging, and optional verbose diagnostic
  artifacts/plots.
- **`TrainingConfig`** (trainer-internal — *not* the public `config.TrainingConfig`).
- **Checkpointing**: `CheckpointBackend` (ABC) with **`EquinoxCheckpointBackend`**
  and **`OrbaxCheckpointBackend`**; `resolve_checkpoint_backend("auto")` prefers
  Orbax and falls back to Equinox. `load_trainer_checkpoint_state` /
  resume support.
- Helpers to prestack observation blocks (`_prestack_observation_blocks` → the
  `PreStackedObservations` pytree) and extract metadata-derived scale features.

---

## 10. Grouped configuration & high-level entry points

### `config.py`
Tames `ConditionalFlowPosterior`'s ~40 flat constructor kwargs into themed
dataclasses, each with sensible defaults.

- **`FlowConfig`** (flow family + geometry), **`EncoderConfig`** (encoder type +
  embedding dims), **`TrainingConfig`** (public schedule/objective config),
  **`StabilityConfig`** (checkpoint-selection thresholds), **`RefinementConfig`**
  (test-time refinement).
- **`build_conditional_flow_posterior(...)`** — expands the grouped configs back
  into the flat constructor.
- **`SBIConfig`** — the single end-to-end config for `run_sbi`: bundles the
  `RetrievalConfig`, all grouped configs, and workflow knobs (simulation count,
  sample counts, plotting, guardrails, labels, truths). `to_run_config()` yields a
  JSON-serializable provenance dict that also drives the run hash.

### `inference.py`
The user-facing serving layer.

- **`AmortizedRetrieval`** — serves a trained model through a retrieval-like API.
  `from_retrieval_config(...)` builds the `SBITask` + an untrained posterior in
  one call; `.fit(reader)` trains; `.infer(blocks)` runs inference. `infer`:
  1. `diagnose_domain` → **OOD check** (robust blockwise z-score vs preprocessing
     stats),
  2. normalize the raw blocks (`prepare_observation_blocks`),
  3. encode + `sample_posterior`,
  4. optional `posterior_predictive`,
  5. transform samples back to **physical** parameter space,
  returning an **`AmortizedRetrievalResult`** (posterior + predictive + OOD +
  metadata).
- **`OODDiagnostic`** — in/out-of-distribution score, threshold, pass/fail.

### `workflow.py`
The orchestrator.

- **`run_sbi(config: SBIConfig) -> Path`** — the one-call benchmark driver. In
  order: build the `SBITask`; resolve/sanity-check run directories and the
  fingerprint-keyed dataset path; generate **or** reuse the simulation dataset
  (+ preprocessing sidecar); build the `NormalizedObservationDatasetReader` (which
  runs the startup scale guard); train **or** load the `ConditionalFlowPosterior`
  (NPE or ELBO, the latter building a differentiable likelihood); save the model;
  write **checkpoint guardrails**; compute **SBC** and **posterior-predictive**
  validation reports on the held-out test split; optionally run test-time
  refinement; run amortized inference on the real target observation; and persist
  samples, plots, a summary JSON, and a run manifest. Supports loading a completed
  model and resuming predictive checkpoints.
- Internal helpers build the ELBO log-likelihood / residual functions, generate
  test-time validation reports, refit preprocessing for legacy datasets, and save
  guardrails.

---

## 11. Artifacts, calibration, plotting, validation

### `artifacts.py`
Provenance, registry, and run-directory bookkeeping.

- **`ArtifactMetadata`** / **`ArtifactRole`** — typed metadata (role, task
  name/version/fingerprint, observation schema, preprocessing) attached to saved
  models.
- **`ArtifactRegistry`** (ABC) + **`FileArtifactRegistry`** / **`RegisteredArtifact`**
  — register and resolve artifacts on disk.
- Run-directory helpers: `compute_run_config_hash`, `build_run_label`,
  `write_run_manifest`/`resolve_run_manifest`, `resolve_latest_posterior_run`,
  `resolve_latest_completed_run`, `update_latest_run_pointer`,
  `count_predictive_case_checkpoints`, and **`save_predictive_spectrum`**.

### `calibration.py`
The diagnostics suite (validating that the amortized posterior is trustworthy).

- **`SimulationBasedCalibrationReport`** + `generate_sbc_report()` — rank-based SBC
  over held-out simulations (a well-calibrated posterior gives uniform ranks);
  also `central_credible_interval`, `empirical_interval_coverage`,
  `CoverageLevelReport`.
- **`PosteriorPredictiveReport`** + `generate_posterior_predictive_report()` /
  `posterior_predictive_report_from_samples()` — pushes posterior draws back
  through the forward model and compares to the observation (with CRPS, intervals,
  checkpointing; `PredictiveResamplingExhaustedError` when valid draws can't be
  collected).
- **`LocalSensitivityReport`** / **`LocalSensitivityPointReport`** +
  `generate_local_sensitivity_report()` — local linear-identifiability /
  Fisher-information diagnostics (Jacobians, singular values, parameter
  correlations) around representative posterior points.
- **`PosteriorInformativenessReport`** + `classify_posterior_informativeness()` —
  flags parameters whose posterior is no tighter than the prior.
- Summarizers (`summarize_posterior_samples`, `summarize_posterior_contraction`,
  `summarize_predictive_report`, `summarize_training_diagnostics`,
  `serialize_ood_diagnostic`) and the parameter-space → physical transform
  (`_posterior_space_to_physical`).

### `plotting.py`
Matplotlib visualizations in the petitRADTRANS house style: `plot_posterior_corner`,
`plot_posterior_marginals`, `plot_sbc_rank_histograms`,
`plot_posterior_predictive_report`, the local-sensitivity plots
(`plot_local_sensitivity_jacobians` / `_fisher_correlations` / `_singular_values`),
and `save_validation_plots` (batch-saves the validation figure set).

### `validation.py`
Pre-training data-sanity guard (distinct from the post-training calibration suite).

- **`ObservationScaleReport`** + **`check_observation_in_training_range()`** —
  compares the real target observation's flux scale against the simulated training
  distribution and aborts (or warns) on a units/scale mismatch. Invoked
  automatically by `NormalizedObservationDatasetReader` at construction.

---

## How the pieces interact (summary)

- **`SBITask`** is the hub: it is derived from a `RetrievalConfig`, fingerprints
  the whole problem, and is consumed by the simulator, dataset generator,
  posterior (for metadata), inference layer, and calibration.
- **`ObservationBlock`** is the universal data unit; **`PreStackedObservations`**
  is its JIT-efficient batched form.
- **Simulator → dataset → preprocessing → reader** is the offline data path that
  feeds **`ConditionalFlowPosterior.fit`** (encoder + flow trained by
  **`SBITrainer`**).
- **`AmortizedRetrieval`** is the online path that reuses the same encoder/flow to
  serve posteriors, with OOD guarding and optional refinement.
- **`config.py`** dataclasses + **`run_sbi`** package the whole thing into a single
  declarative benchmark call.
- **Fingerprints + metadata** (`task.py`, `compatibility.py`, `artifacts.py`,
  `flows/base.py`) thread provenance through every persisted file so stale
  datasets or mismatched models fail loudly rather than silently.
- **`validation.py`** guards inputs before training; **`calibration.py`** +
  **`plotting.py`** validate outputs after training.

---

## Scientific core vs. infrastructure

This section separates the functions and classes that **determine the numerical
result** — what the simulations are, how data is transformed, how the flow is
parametrized and trained, and how the posterior is produced — from the much larger
body of code that exists only to make that math run fast, persist to disk, or stay
reproducible. **A reader who wants to understand or audit the science only needs
the items in §A–§E.** Everything in §F can be treated as a black box: changing it
should never change the answer (only the speed, memory, or filenames).

A practical test was applied to each item: *"if I changed this function's body,
could the posterior samples for a fixed seed change?"* If yes → scientific core. If
it only re-shapes, shards, stacks, caches, or serializes data that is later
recombined identically → infrastructure.

### A. How the simulations are computed (the forward model & training data)

These define the `(parameters, observation)` pairs the flow learns from. They *are*
the likelihood/data-generating process.

| Function / class | File | Scientific role |
| --- | --- | --- |
| `RuntimeSimulator._sample_prior_parameter_batch` | `simulator.py` | **Prior sampling.** Draws unit-cube points and maps them to physical parameters (`transform_cube_to_physical_batch`); also stores the cube + unconstrained coordinates the flow can be trained in. This is the prior π(θ). |
| `SimulationRuntime.simulate_observations_scalar` | `simulator.py` | **Forward model + projection** for one point: evaluate the radiative-transfer model, then project to observation space (binning/interpolation), apply flux scaling and offset. The deterministic, noise-free mean observation. |
| `SimulationRuntime.simulate_observations_batched` / `_project_batched_rt_results` | `simulator.py` | Vectorized form of the same physics (the projection, flux scaling, offset, and uncertainty scaling/inflation). Identical math to the scalar path, batched. |
| `RuntimeSimulator._apply_noise` / `_apply_noise_batched` | `simulator.py` | **Noise model.** Adds Gaussian / multivariate-Gaussian observational noise from the per-point uncertainties or covariance. Defines the stochastic part of the likelihood. |
| `RuntimeSimulator._row_invalid_value_mask` / `_invalid_sample_mask` | `simulator.py` | **Rejection-sampling constraint.** Decides which simulated draws are physically admissible (finiteness + `ObservationValueConstraint`, e.g. flux > 0). Truncates the effective prior, so it shapes the training distribution. |
| `RuntimeSimulator.simulate_batch` | `simulator.py` | The accept/resample loop that turns the above into exactly `n_samples` valid simulations. Its rejection logic is scientific; its batching/concatenation is not. |
| `SimulationConfig`, `NoiseModelConfig`, `ObservationValueConstraint` | `task.py` | User-facing knobs that select the noise mode/seed and the admissibility bounds — i.e. they parametrize the items above. |

> Note: the priors themselves (distributions, bounds) and the radiative-transfer
> model function live in the **retrieval** package (`RetrievalConfig`,
> `ParameterLayout`, `model_generating_function`); `sbi` consumes them. They are the
> single largest scientific input but are *outside* this module.

### B. How the data is encoded (observation → conditioning vector)

The encoder is a learned function whose architecture directly controls what
information the flow can condition on. The default deliberately separates spectral
**amplitude** from **shape** so the posterior cannot collapse to the prior on
shape-determined parameters (composition, temperature, clouds).

| Function / class | File | Scientific role |
| --- | --- | --- |
| `SpectralConvolutionEncoder` (default) | `encoders.py` | The production spectral encoder; explicit amplitude/shape factorization. |
| `SpectralConv1DEncoder`, `SpectralPatchEncoder` | `encoders.py` | Alternative spectral encoder architectures (user-selectable via `EncoderConfig.spectrum_encoder_type`). |
| `PhotometryPointEncoder` | `encoders.py` | Encoder architecture for photometric points. |
| `DatasetSetAggregator` | `encoders.py` | Permutation-invariant pooling across blocks — defines how multiple datasets are combined into one embedding. |
| `ObservationEncoder` | `encoders.py` | Top-level encoder: modality dispatch + aggregation. The object that produces the conditioning vector. |
| `_amplitude_features`, `_generic_amplitude_features`, `_generic_shape_tensor`, `_within_spectrum_standardized`, `_spectral_local_structure`, `_robust_center_scale`, `_normalized_coordinate_channel` | `encoders.py` | The feature-construction primitives that *define* the amplitude/shape split. Changing these changes what the network sees. |

### C. How the data is normalized (preprocessing transform)

Applied identically at training time and inference time; it sets the numerical
representation the encoder/flow actually operate on.

| Function / class | File | Scientific role |
| --- | --- | --- |
| `compute_block_normalization_stats`, `fit_task_preprocessing` | `preprocessing.py` | Fit the robust per-wavelength median/IQR, global scale, and `log_median_flux` from the simulated training set. |
| `_apply_spectral_value_transform`, `_apply_relative_spectral_value_transform` | `preprocessing.py` | The actual value transform — an `arcsinh((x − center)/scale)` (a signed, zero-safe soft-log). Determines how fluxes are compressed. |
| `_transform_spectral_uncertainties` | `preprocessing.py` | Maps error bars through the same local scaling so uncertainties stay consistent with the transformed values. |
| `_global_spectral_scale`, `_sample_log_median_flux`, `_global_/relative` scale helpers | `preprocessing.py` | Define the amplitude scalar(s) carried alongside the normalized shape. |
| `normalize_observation_block` / `normalize_observation_blocks` | `preprocessing.py` | Apply the transform to blocks. The bridge that guarantees train/inference parity. |
| `BlockNormalizationStats`, `TaskPreprocessingMetadata` | `preprocessing.py` | The fitted statistics themselves (scientific content); their `to_payload`/`from_payload` are bookkeeping. |

### D. How the normalizing flow is set up by the user

The flow's family, depth, and conditioning are the density-estimator hypothesis
class. These are the dials a user turns to define the model.

| Function / class | File | Scientific role |
| --- | --- | --- |
| `FlowConfig` | `config.py` | **Primary user surface for the flow:** family (`neural_autoregressive` / `spline` / `affine` / `autoregressive`), number of transforms, hidden width, spline bins/bound, base distribution, NAF slope/residual, spline regularization weights. |
| `EncoderConfig` | `config.py` | Selects the encoder type and embedding sizes (links to §B). |
| `TrainingConfig` (public, `config.py`) | `config.py` | **The objective and schedule:** `training_objective` (`npe` vs `elbo`), `parameter_space` (`cube`/`unconstrained`), learning rate, epochs, regularization/invertibility weights, ELBO β-annealing. Directly sets what is optimized. |
| `RefinementConfig` | `config.py` | Selects test-time per-observation refinement (`none`/`map_laplace`/`elbo`) and its hyperparameters (links to §E). |
| `StabilityConfig` | `config.py` | Thresholds for the checkpoint-selection gate (links to §E model selection). |
| `build_conditional_flow_posterior` | `config.py` | Expands the configs into the estimator — the one call a user makes to instantiate the model. |
| `ConditionalFlowPosterior.__init__` / `_build_flow` | `flows/posterior.py` | Constructs the chosen flow family + encoder. The actual model-assembly logic. |
| Flow transforms & stacks: `ConditionalNeuralAutoregressiveFlow`, `ConditionalSplineFlow`, `ConditionalAffineFlow`, `ConditionalAutoregressiveFlow`, and their transform classes | `flows/transforms.py` | The invertible maps and their log-dets — i.e. the parametric density family. |
| Spline/NAF math: `_rational_quadratic_spline_forward/_inverse`, `_neural_autoregressive_scalar_forward` | `flows/transforms.py` | The mathematical form of the transforms. |

### E. How the posterior is trained, selected, and produced

| Function / class | File | Scientific role |
| --- | --- | --- |
| `loss_from_observations` / `compute_loss` / `training_loss` | `flows/objectives.py` | **The training objective.** Conditional negative log-likelihood (NPE) with log-prob clipping, plus optional spline and invertibility regularization and the aux-scale term. |
| `make_elbo_loss` | `flows/objectives.py` | The alternative amortized-variational (ELBO) objective. |
| `invertibility_penalty` (`flows/objectives.py`); `_spline_regularization_terms`, `_spline_layer_regularization_terms` (`flows/regularization.py`) | `flows/objectives.py` + `flows/regularization.py` | Extra loss terms that shape the learned density (keep NAF invertible, keep spline bins/derivatives well-conditioned). |
| Parameter-space maps: `_parameter_space_log_prob`, `_cube_to_unconstrained`, `_cube_log_abs_det_jacobian`, `_flow_space_parameters` | `flows/cube.py` | Change of variables (physical ↔ cube ↔ unconstrained) **with Jacobian terms** — affects the actual density values, not just bookkeeping. |
| `_flow_validation_stability_metrics`, `_flow_validation_selection_metric`, `compute_validation_diagnostics` | `flows/regularization.py` (+ `flows/objectives.py` for the diagnostics) | **Model selection.** Reject non-invertible / collapsed checkpoints when picking the "best" model — changes which trained model is returned. |
| `ConditionalFlowPosterior.sample_posterior` / `log_prob` | `flows/posterior.py` | Produce the posterior draws / densities (the scientific output). Batched/jitted variants are infrastructure around this. |
| `refine_posterior_on_observation`, `map_laplace_posterior_on_observation` | `flows/refinement.py` | **Test-time refinement** of the posterior for a specific target spectrum (ELBO/MAP+Laplace). Alters the delivered posterior. |
| `FlowMatchingPosterior` (`_loss`, `sample_posterior`) | `flows/matching.py` | Alternative estimator family with its own flow-matching objective and ODE sampling. |
| `SBITrainer.fit` | `training.py` | The optimization loop that minimizes the objective above. Its sharding/checkpointing is infrastructure; the gradient step / schedule / early-stopping decision is scientific. |
| `AmortizedRetrieval.infer` | `inference.py` | The end-to-end inference call: OOD check → normalize → encode → sample → map to physical space. Orchestrates §B/§C/§E for one observation. |
| `AmortizedRetrieval.diagnose_domain` (`OODDiagnostic`) | `inference.py` | Out-of-distribution score that gates whether a result should be trusted. |

**Validation / diagnostics (measure results, don't change them).** Everything in
`calibration.py` (`generate_sbc_report`, `generate_posterior_predictive_report`,
`generate_local_sensitivity_report`, `classify_posterior_informativeness`, the
`summarize_*` helpers) and `plotting.py` is scientifically *meaningful* — it
determines what is reported about calibration and information content — but it reads
a trained posterior; it never alters the posterior itself. `validation.py`
(`check_observation_in_training_range`) is a guard that can abort a run but does not
change samples. Treat these as the "did it work?" layer, separate from the
"what is the answer?" core above.

### F. Infrastructure — wrappers, parallelization, I/O, bookkeeping

These can be read last or skipped. Each is designed to be **result-invariant**:
its contract is to reproduce, bit-for-bit, what the single-device / unchunked /
in-memory path would produce.

- **Parallelization & batching (speed/memory only):**
  `SimulationRuntime._run_batched_rt_kernel_multi_device`,
  `_simulate_observations_batched_native`, `max_chunk_size` chunking and
  `_concatenate_batched_observation_outputs`, `_stack_simulation_output_payloads`,
  `RuntimeSimulator._simulate_parameter_matrix` device padding/trimming,
  `_slice_batch_rows`, `_concatenate_simulation_batches`,
  `_stack_observation_payloads` (`simulator.py`); `SBITrainer._shard_batch`,
  `_get_device_mesh`, `_pad_to_multiple` (`training.py`);
  `ObservationEncoder.encode_stacked_batch` / `encode_from_prestacked`,
  `_pad_to_fixed`, `_resize_embedding` (`encoders.py`);
  `batch_embeddings` (`flows/objectives.py`), `_jit_batch_sample` /
  `batch_sample_posterior` (`flows/posterior.py`), `_tile_prestacked_observation`
  (`flows/refinement.py`);
  `PreStackedObservations` + its pytree (un)flatten (`observation.py`).
- **RNG accounting (reproducibility plumbing):** `RuntimeSimulator._next_key`,
  `_advance_rng_for_batch` (`simulator.py`) — split keys to match consumption for
  resume; do not generate new physics.
- **File I/O & dataset storage:** all of `dataset.py`'s HDF5 classes
  (`HDF5SimulationDatasetStore/Writer`, `HDF5StoredSimulationDataset`,
  `NormalizedObservationDatasetReader`'s iteration/caching), `SimulationDatasetManifest`,
  split-assignment helpers (`infer_split_counts`, `_build_split_assignments`,
  `estimate_preloaded_split_memory_gib`).
- **Checkpointing:** `CheckpointBackend`, `EquinoxCheckpointBackend`,
  `OrbaxCheckpointBackend`, `resolve_checkpoint_backend`,
  `load_trainer_checkpoint_state` (`training.py`).
- **Persistence & generic loading of models:** `PersistentPosteriorEstimator.save`/`load`,
  `_build_serialized_metadata`, `from_serialized_metadata`, `save/load_backend_state`,
  `hydrate_loaded_metadata` (`flows/base.py`, `flows/posterior.py`,
  `flows/matching.py`); the estimator registry + generic loader now in `flows/base.py`.
- **Provenance, fingerprints & compatibility (correctness guards, not math):**
  `SBITask.task_fingerprint` / `_forward_model_signature` / `fingerprint_payload`,
  `validate_observation_schema`, `validate_artifact_metadata` (`task.py`); all of
  `compatibility.py`; `ArtifactMetadata`, `ArtifactRegistry`, run-manifest and
  run-label helpers, `save_predictive_spectrum` (`artifacts.py`). These decide
  *whether* a cached dataset or saved model may be reused; they never change a
  computed value.
- **Workflow orchestration:** `run_sbi` and its private helpers (`workflow.py`)
  wire §A–§E together and handle directories/resume/plot generation. The *choices*
  it hard-codes (default value constraint, validation split usage) matter; the
  plumbing around them does not.

### One-paragraph reading guide

To understand *what the SBI module computes*, read, in order: the prior +
forward-model + noise + rejection functions in §A, the normalization transform in
§C, the encoder in §B, the flow family and training objective in §D–§E, and finally
`AmortizedRetrieval.infer` in §E for how a single posterior is produced. Everything
in §F is there so those ~20 functions can run on many GPUs, survive interruption and
resume, persist to disk, and refuse to mix incompatible artifacts — none of it
should alter the result.
</content>
