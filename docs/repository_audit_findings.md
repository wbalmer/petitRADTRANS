# petitRADTRANS Repository Audit Findings

Date: 2026-05-19

Scope:
- Reviewed the current repository with emphasis on runtime performance, peak memory usage, redundant internal code, and package/interface import latency.
- Treated the previous audit as stale input only and revalidated claims against the current code before carrying them forward.
- Examined the package root, core radiative-transfer path, chemistry, opacities, retrieval, SpectralModel, Planet, photometry, plotting, docs helpers, and SBI at a high level.
- Measured import-time behavior in the `jaxprt` environment with `conda run -n jaxprt python -X importtime ...`.
- This document reports findings and an implementation plan only. No code changes are implemented here.

## Executive Summary

The repository still contains a large number of `TODO` markers, but the remaining backlog is no longer dominated by top-tier performance work. Most of the current `TODO` surface is now deprecation bookkeeping, dead-code notes, or opacity-reference metadata cleanup rather than unresolved core runtime problems.

The highest-value remaining findings are:

1. The retrieval/model adapter contract is still split between legacy compatibility and the newer runtime normalization path. The `pt_plot_mode` and `beta` deprecation markers are still real and now anchor the largest remaining cleanup slice.
2. The variable-resolution convolution path in retrieval is still special-cased and carries open benchmarking and unification TODOs. This is one of the few remaining user-visible performance TODO clusters.
3. A smaller set of JAX-facing kernels still carry plausible performance TODOs around array normalization, vectorization, and branching (`spectrum_utils`, `cloud_core`, `hminus`, selected chemistry helpers), but these should be treated as profile-first candidates rather than assumed bottlenecks.
4. Several older TODOs are now stale or deprioritized. In particular, the old line-opacity pressure/temperature slice-loading TODO is no longer the right memory target after the recent profiling and loader refactor.

## Current Measurements

Import-time measurements were taken in the `jaxprt` environment after the latest cleanup pass.

| Import target | Observed total |
| --- | --- |
| `import petitRADTRANS` | about 84.8 ms |
| `from petitRADTRANS.spectral_model import SpectralModel` | about 2.00 s |
| `from petitRADTRANS.retrieval.retrieval import Retrieval` | about 2.75 s |
| `docs/utils.py` via `importlib` | about 3 ms |

Interpretation:

- The package root import remains cheap.
- `SpectralModel` import is materially better than the earlier baseline, but the remaining cost is still concentrated in the retrieval/runtime/model stack and transitive astronomy dependencies.
- `docs/utils.py` is no longer a meaningful import-surface problem after the loader migration.

## Status Of The Previous Audit

The previous audit is now partially outdated because several of the earlier TODO-backed findings have already been addressed.

- Lazy sampler dispatch, removal of the unused `jaxns` summary import, PHOENIX consolidation, registry deduplication, `docs/utils.py` migration off `_input_data_loader.py`, removal of `get_max_likelihood_params()`, photosphere-radius vectorization, and the main SpectralModel copy-contract cleanup are no longer outstanding TODO themes.
- This update focuses specifically on the remaining in-tree `TODO` inventory and distinguishes between still-actionable work and stale or low-value comments.

## Outstanding TODO Inventory

Repo-wide snapshot for `petitRADTRANS/**/*.py`:

- 176 `TODO` markers total.
- Highest concentrations are `opacities/file_conversion.py` (49), `plotlib/plotlib.py` (13), `spectral_model.py` (12), `retrieval/retrieval.py` (10), `retrieval/data.py` (10), `chemistry/clouds.py` (10), and `radtrans.py` (8).
- About 52 TODOs are deprecation/removal/unused/not-in-docs markers.
- About 22 TODOs directly mention performance/runtime work such as JIT, vectorization, interpolation, loading, convolution, or array normalization.

The raw TODO count therefore overstates the amount of genuinely important engineering debt.

## Findings

### 1. High: retrieval model-output compatibility is still the largest real TODO cluster

Primary files:
- `petitRADTRANS/retrieval/retrieval.py`
- `petitRADTRANS/retrieval/models.py`
- `petitRADTRANS/retrieval/runtime.py`
- `petitRADTRANS/spectral_model.py`

Relevant TODOs:
- `retrieval/retrieval.py`: model-generating functions should always return the same number of values.
- `spectral_model.py`: remove `pt_plot_mode` from the compatibility template in v4.0.0.
- `spectral_model.py`: remove deprecated `beta` handling in favor of `_uncertainty_scaling`.

Assessment:
- Still important and still real.
- This is now the highest-value remaining cleanup slice because it keeps legacy tuple-shape branching and compatibility kwargs alive through the retrieval surface.

Required implementation changes:
- Standardize one model-generation return contract for retrieval, runtime, and model adapters.
- Remove `pt_plot_mode` from public and adapter signatures once all call sites normalize through one output adapter.
- Remove `beta` alias support in the same versioned cleanup pass and update notebooks/tests/docs together.

### 2. High: variable-resolution convolution remains special-cased and under-benchmarked

Primary files:
- `petitRADTRANS/retrieval/retrieval.py`
- `petitRADTRANS/radtrans_core/spectrum_utils.py`

Relevant TODOs:
- interpolate the resolution array and use the standard convolution path.
- compare different convolution and binning methods.

Assessment:
- Important and user-visible.
- This branch still carries bespoke logic for variable-resolution observations, which makes benchmarking and maintenance harder and likely leaves performance on the table.

Required implementation changes:
- Project `resolution_array` onto the model grid explicitly.
- Benchmark the BREADS-specific path against the standard convolution path on representative grids.
- Unify on one kernel where accuracy is acceptable, or retain both paths behind a clearer, measured contract.

### 3. Medium-High: likelihood/data TODOs still expose real API and correctness debt

Primary files:
- `petitRADTRANS/retrieval/log_likelihood.py`
- `petitRADTRANS/retrieval/data.py`

Relevant TODOs:
- covariance handling assumes a 1D spectrum.
- interpolation helpers should raise when interpolation is impossible.

Assessment:
- Important, mostly for correctness and robustness rather than raw speed.
- These TODOs are still active and can surface user-visible failures on multi-dimensional or edge-case data.

Required implementation changes:
- Define the supported ND data contract explicitly.
- Extend covariance handling beyond 1D or reject unsupported shapes earlier in `Data`.
- Make interpolation failures explicit instead of silently accepting impossible transforms.

### 4. Medium: JAX-native kernel cleanup remains plausible but should stay profile-first

Primary files:
- `petitRADTRANS/radtrans_core/spectrum_utils.py`
- `petitRADTRANS/radtrans_core/cloud_core.py`
- `petitRADTRANS/opacities/hminus.py`
- `petitRADTRANS/chemistry/utils.py`
- `petitRADTRANS/chemistry/prt_easychem.py`

Relevant TODOs:
- ensure everything uses JAX arrays instead of NumPy arrays.
- decide whether internal functions need JIT.
- vectorize on/off code.
- `vmap`/`lax.cond` cleanup in branchy kernels.

Assessment:
- Important only once backed by profiling on the actual call sites.
- These TODOs are plausible wins, but they are still hypotheses rather than demonstrated dominant bottlenecks.

Required implementation changes:
- Profile the cloud, continuum-subtraction, H-minus opacity, and chemistry helper paths under realistic workloads.
- Normalize array types at API boundaries.
- Replace Python branching with `lax`/`vmap` only where traces show meaningful wins.
- Add targeted microbenchmarks before and after each kernel rewrite.

### 5. Medium: remaining deprecation TODOs are real, but they are version-bound cleanup rather than urgent runtime work

Primary files:
- `petitRADTRANS/spectral_model.py`
- `petitRADTRANS/radtrans.py`
- `petitRADTRANS/opacities/cia.py`
- `petitRADTRANS/_input_data_loader.py`
- `petitRADTRANS/retrieval/data.py`

Relevant TODOs:
- remove compatibility aliases and deprecated attributes in v4.0.0.
- convert legacy factory-style helpers in `retrieval/data.py` to proper constructors/classmethods.
- remove old CIA alias helpers and compatibility shells.

Assessment:
- Still real, but mostly release-timed API cleanup.
- These TODOs matter because they keep the public surface larger and the internal logic branchier than necessary, but they are not the current runtime bottleneck.

Required implementation changes:
- Decide whether the v4 deprecation window is now open.
- Remove compatibility aliases and deprecated kwargs in one coordinated pass.
- Update docs, notebooks, and tests in the same change so stale compatibility markers do not linger.

## Stale Or Low-Value TODOs

### 1. The old pressure/temperature slice-loading TODO is now stale as the main memory target

Primary file:
- `petitRADTRANS/radtrans.py`

Relevant TODO:
- load only the relevant pressure-temperature slice instead of the full grid.

Assessment:
- Stale or at least strongly deprioritized.
- Recent profiling and implementation work showed that staging copies, padding, and host/device duplication were the real setup-memory drivers. The user also explicitly deprioritized partial PT loading because the full pressure range is usually needed anyway.

Required changes if revisited:
- Re-profile with a workload that truly uses partial grids.
- Prove that slice loading saves more than it costs in complexity and HDF5 access overhead before reviving this TODO.

### 2. `file_conversion.py` contains a large metadata/doc backlog, not a hot-path engineering backlog

Primary file:
- `petitRADTRANS/opacities/file_conversion.py`

Relevant TODOs:
- many inline "not in docs", "outdated", or reference-ambiguity comments.

Assessment:
- Mostly low-value for runtime and likely stale as inline code comments.
- These are data provenance and bibliography curation notes, not performance or memory work.

Required changes:
- Move the remaining reference anomalies to a dedicated opacity-reference audit or manifest.
- Remove inline TODO noise from runtime code once the reference data source is tracked elsewhere.

### 3. Several helpers in `physics.py` look like genuine dead code

Primary file:
- `petitRADTRANS/physics.py`

Relevant TODOs:
- delete `make_press_temp`.
- delete `make_press_temp_iso`.
- `planck_function_hz_temperature_derivative` unused?

Assessment:
- Likely stale and low risk.
- These functions have no internal call sites in the current tree.

Required changes:
- Confirm there is no external or documentation usage that still relies on them.
- Delete the dead helpers or move them to a private compatibility module if an external escape hatch is still required.

### 4. `_input_data_loader.py` is now mostly a compatibility shell

Primary file:
- `petitRADTRANS/_input_data_loader.py`

Assessment:
- Still real, but largely deprecation bookkeeping rather than a live internal dependency.
- The important internal migrations away from this module are already done; what remains is release-timed removal.

Required changes:
- Verify whether any documented external compatibility promise still exists.
- Remove the module and its remaining compatibility helpers at the next version boundary, with release-note coverage.

### 5. `Opacity.load()` and `Opacity.save()` are TODO stubs, but not pressing

Primary file:
- `petitRADTRANS/opacities/opacities.py`

Assessment:
- Low priority unless persisted `Opacity` objects become a supported feature.
- Today these methods are explicit no-op stubs rather than missing hot-path functionality.

Required changes:
- Either implement real serialization semantics with a stable file contract, or remove the public methods to avoid implying support that does not exist.

### 6. `plotlib.py` TODOs are mostly hygiene, not performance blockers

Primary file:
- `petitRADTRANS/plotlib/plotlib.py`

Relevant TODOs:
- complete docstrings.
- replace bare exceptions.
- question the optional seaborn dependency.

Assessment:
- Real cleanup work, but not a core runtime or memory priority.

Required changes:
- Narrow exception types, make seaborn optional at the call site, and complete docstrings as standalone maintenance work.

## Resolved Or No Longer Outstanding TODO Themes

The following earlier TODO-backed findings are no longer active items in the current tree:

- lazy sampler dispatch and the unused `jaxns` summary import
- `docs/utils.py` depending on `_input_data_loader.py`
- duplicated `_get_base_*` registries and duplicated default rebinning wavelength range helper
- `Retrieval.get_max_likelihood_params()`
- the main SpectralModel fixed-parameter deep-copy hotspot
- Python-loop photosphere-radius interpolation

## Existing Validation Anchors

Current repository assets that should be used to validate future performance work:

- `tests/_performance_test_correlated_k.py`
- `tests/_performance_test_line_by_line.py`
- `tests/_performance_test_planck_float_precision_core.py`
- `tests/_performance_test_spectral_model_line_by_line.py`
- `tests/_performance_test_transit_radii_core.py`
- `tests/_performance_test_guillot_emission_amr.py`
- `test_jax_prt/benchmark_prt_implementations.py`

Additional validation that should be added during implementation:

- Dedicated import-time checks for `Radtrans`, `SpectralModel`, `Planet`, and the retrieval subpackages.
- Peak-memory benchmarks for `Radtrans` initialization and line-opacity loading.
- A dedicated benchmark for `return_photosphere_radius=True`.
- A memory-churn benchmark for `SpectralModel.init_data`, `SpectralModel.init_retrieval`, and retrieval export paths.

## Detailed Implementation Plan

### Phase A: finish the retrieval contract cleanup

1. Normalize model-generating functions to one return contract across retrieval, runtime, and model adapters.
2. Remove `pt_plot_mode` plumbing once the normalized adapter path is the only path left.
3. Remove `beta` alias handling in the same versioned cleanup pass.
4. Re-run the existing retrieval and SpectralModel validation suite after each substep.

Success criteria:
- No remaining tuple-shape branching in the retrieval execution path.
- `pt_plot_mode` and `beta` compatibility code is gone from the public API surface.

### Phase B: benchmark and simplify the variable-resolution convolution path

1. Benchmark the current BREADS-specific branch against a resolution-array interpolation plus standard convolution path.
2. Choose one kernel where possible, or document the split with a measured crossover.
3. Add a dedicated benchmark and focused regression tests for variable-resolution observations.

Success criteria:
- One clearly documented variable-resolution contract.
- Measured behavior rather than TODO comments driving the implementation choice.

### Phase C: harden data and likelihood contracts

1. Decide the supported ND-data contract for retrieval likelihoods.
2. Reject impossible interpolation earlier in `Data`.
3. Extend covariance handling beyond 1D only if there is a real supported use case.

Success criteria:
- No silent acceptance of impossible interpolation requests.
- ND limitations are either implemented or made explicit at the API boundary.

### Phase D: profile-first JAX kernel cleanup

1. Profile `continuum_subtract_spectrum`, cloud-core kernels, H-minus opacity helpers, and the remaining chemistry JIT markers.
2. Normalize array types at the boundaries of those kernels.
3. Apply `vmap`, `lax.cond`, or JIT only where the profiling data shows a real win.

Success criteria:
- Targeted performance wins backed by benchmarks.
- Fewer mixed NumPy/JAX array boundaries in optional hot paths.

### Phase E: remove stale compatibility and dead-code TODOs

1. Remove or isolate dead helpers in `physics.py` after confirming no external/documented use.
2. Remove `_input_data_loader.py` and remaining CIA compatibility aliases when the version boundary allows it.
3. Move opacity-reference metadata TODOs out of `file_conversion.py` into a dedicated audit/manifest.
4. Decide whether `Opacity.load()` and `Opacity.save()` should be implemented or deleted.

Success criteria:
- Fewer stale TODOs in runtime modules.
- The remaining TODO count better reflects live engineering work.

## Suggested Priority Order

1. Finish the retrieval contract cleanup (`pt_plot_mode`, `beta`, model-output normalization).
2. Benchmark and simplify the variable-resolution convolution path.
3. Harden `Data`/likelihood ND handling and impossible-interpolation behavior.
4. Profile and clean up the remaining JAX-kernel TODOs in optional hot paths.
5. Delete or relocate stale compatibility/dead-code TODOs so the remaining backlog reflects real work.

## Notable Non-Findings

- The package root import is already lightweight, so `petitRADTRANS/__init__.py` itself is not the main optimization target.
- The earlier chemistry pressure-node issue does not appear to be an active current problem.
- SBI did not surface a top-tier performance or memory regression relative to the core RT/retrieval/import path in this pass.
- No code changes or benchmark refactors were implemented during this audit.