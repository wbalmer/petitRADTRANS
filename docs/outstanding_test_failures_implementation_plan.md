# Outstanding petitRADTRANS Test Failures: Implementation Plan

## Current State

- Source of truth: `pRTTestsv13.txt` — **19 failed, 263 passed, 1 skipped** (petitRADTRANS-4.0.0a42), plus the fixes implemented since (see below).
- **Implemented and verified in this pass (2026-06-11):**
  - The three live product defects from the v13 findings (§1–§3 below) are fixed; all 7 affected tests pass, and a regression sweep of `test_retrieval.py`, `test_sbi_interfaces.py`, and `test_cloud_parameter_plumbing.py` (109 tests) passes with no new failures.
  - The benchmark suites (`test_cloud_core.py`, `test_spectral_model_correlated_k.py`, `test_spectral_model_line_by_line_custom.py`, `test_radtrans_correlated_k.py`) show only the pre-existing v13 failures — the fixes introduced no regressions.
- **Implemented and verified (2026-06-13):**
  - The SpectralModel metallicity unit regression is fixed, with a new optional `log10_metallicities` argument (§4).
  - The transmission contribution function is fixed (internal baseline + scattering flag); its test passes against the **original** 3.2.0a8 reference — removed from the refresh list (§7).
  - Regression sweep: `test_radtrans_correlated_k.py`, `test_fortran_modules.py`, `test_retrieval_spectral_model.py` all pass; `test_spectral_model_correlated_k.py` is down to the two known drift-residual failures.
- **Reference refresh completed (2026-06-13):** all 11 remaining drift failures resolved by regenerating 10 reference files (chemistry 5 + SpectralModel chemistry 2, photospheric radius 1 file shared by 2 tests, adaptive Feautrier 1, CCF 1). Each was regenerated on 4.0.0a42 via an allow-listed monkeypatch of `Benchmark.write_reference_file`, so no other reference was touched (verified: `git status` shows exactly those 10 `*_ref.h5` modified). All 11 tests now pass; a 41-test sweep over the affected files is green.
  - One harness fix was required alongside the refresh: `tests/benchmark.py::_iter_output_lookup_keys` now also yields the integer form of a numeric string key. `test_mass_fractions2metallicities` returns `h_ratios` keyed by integer atomic number; HDF5 stringifies these on save, so a reloaded reference key (`"11"`) could not match a freshly-computed integer key (`11`). This was previously masked by the stale-parameter check failing first. The fix uses the existing key-aliasing mechanism and changes no reference file.
- Remaining failures: none in the drift buckets.

## 1. Runtime Projection Rebin Defect — IMPLEMENTED

Affected tests (all passing now):

- `test_retrieval.py::test_log_likelihood_routes_normal_scalar_path_through_runtime`
- `test_retrieval.py::test_log_likelihood_routes_external_reference_setup_through_runtime`
- `test_sbi_interfaces.py::test_runtime_simulator_generates_deterministic_observations_without_noise`
- `test_sbi_interfaces.py::test_runtime_simulator_supports_nuisance_parameters`
- `test_sbi_interfaces.py::test_generate_local_sensitivity_report_tracks_linear_information_content`

Changes applied:

1. **Identical-grid passthrough** (`petitRADTRANS/retrieval/utils.py`): added `_eager_grids_match()` and an extra branch in `process_spectrum_pipeline` — when the (RV-shifted) model wavelength grid is concrete (not a JAX tracer) and exactly equals the observation grid, the spectrum is passed through unchanged, matching the legacy likelihood short-circuit (`retrieval.py:4128-4140`). Under JIT tracing the check conservatively returns False and the rebin branch is taken.
2. **Covered-fraction renormalization** (`petitRADTRANS/radtrans_core/spectrum_utils.py`, `rebinning_interpolation`): each output bin is now normalized by the width of the bin actually covered by the input grid instead of the nominal bin width. Fully covered bins are bit-identical to before (the segment overlaps telescope to the bin width); partially covered edge bins now return the mean flux over the covered interval instead of silently losing the uncovered fraction; uncovered bins return 0. This protects real retrievals whose model grid does not extend half a bin beyond the data grid, including jitted paths the eager passthrough cannot reach.
3. No change to the `Data` auto-bin-width behaviour — it is the intended new contract.

## 2. Cloud Per-Species Width Transpose Bug — IMPLEMENTED

Affected test (passing now):

- `test_cloud_parameter_plumbing.py::test_compute_cloud_particle_mean_radius_accepts_per_species_width_grid`

Change applied (`petitRADTRANS/radtrans_core/cloud_core.py`, 2-D branch of `compute_cloud_particle_mean_radius`): the documented `(n_layers, n_clouds)` layout is now checked **first**; the transpose is applied only when the shape matches `(n_clouds, n_layers)` and not the documented layout. Square matrices are therefore interpreted in the documented layout, consistent with `f_seds`.

## 3. Evaluate-Mode Crashes — IMPLEMENTED

Affected test (passing now):

- `test_retrieval.py::test_run_evaluate_mode_loads_saved_results_without_sampling`

Three changes applied to `petitRADTRANS/retrieval/retrieval.py`:

1. `generate_retrieval_summary` now creates the `evaluate_<name>` output directory (`os.makedirs(..., exist_ok=True)`) before opening the summary file, covering evaluate runs whose output directory was moved or deleted after `__init__`.
2. The evaluate branch of `run()` wraps `generate_retrieval_summary()` in a best-effort try/except with a warning: loading saved results is the contract of evaluate mode, and a summary failure (e.g. partially initialized retrieval state) must not discard successfully loaded results.
3. The evaluate branch now returns the loaded **results** object instead of the sampler, matching the contract of the sampling branches (both return `results` via `_finalize_sampler_run`).

Known cosmetic issue noted in passing (not fixed): `generate_retrieval_summary` writes each fixed parameter twice (duplicated block at `retrieval.py:1110-1119`).

## 4. SpectralModel Metallicity Unit Regression — IMPLEMENTED

Affected tests:

- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry`
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry_imposed`

Root cause (findings §2A.1): commit `5c765518` rewired `SpectralModel.compute_mass_fractions` to `chemistry.core.calculate_gas_mass_fractions`, which treats `metallicity` as dex (aliased with `Fe/H`/`M/H`), while SpectralModel's `metallicity` is documented linear — every SpectralModel equilibrium-chemistry run executed at `10^Z`× solar.

Changes applied (`petitRADTRANS/spectral_model.py`, `compute_mass_fractions`):

1. The linear `metallicity` is converted to dex (`jnp.log10`) before delegating to `calculate_gas_mass_fractions`.
2. New optional `log10_metallicities` argument: dex metallicity passed to the chemistry table as-is (also retrievable directly as a model parameter without a transform). Mutually exclusive with `metallicity`; supplying both raises `ValueError`.
3. Verified: `metallicity=1.0` and `log10_metallicities=0.0` produce identical solar chemistry (H₂O = 5.1×10⁻³, μ = 2.3557 matching the reference), and `log10_metallicities=1.0` reproduces the 10× solar state.

Post-fix status: outputs `'0'`, `'1'`, `'2'`, `'4'` are within tolerance; the tests now fail only on trace-species mass-fraction drift (≤1.8%, majors at ~10⁻⁶) — the same table-interpolation drift as §5. **These two references move into the §5 refresh batch.**

Remaining follow-ups (small, unscheduled):

- Remove or re-wire the dead `SpectralModel.compute_equilibrium_mass_fractions` (no remaining callers; contains the old conversion).
- Audit SpectralModel-based retrievals that sampled `metallicity` since `5c765518` for results calibrated against the buggy 10^Z behaviour.

## 5. Refresh The Chemistry Bucket (Direct + SpectralModel)

Affected tests (all compare against `3.2.0a8 (2024-10-26)`; pure table-interpolation drift):

- `test_chemistry.py::test_chemistry_atmosphere`
- `test_chemistry.py::test_chemistry_atmosphere_quench`
- `test_chemistry.py::test_chemistry_c_o_ratios`
- `test_chemistry.py::test_chemistry_metallicities`
- `test_chemistry.py::test_mass_fractions2metallicities`
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry` (post-§4 residual: trace species ≤1.8%, majors ~10⁻⁶)
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry_imposed` (same)

Planned steps:

1. Regenerate the five direct chemistry references and the `mass_fractions2metallicities` upstream parameter payload together on current code.
2. Verify all five pass.
3. Regenerate the two SpectralModel equilibrium-chemistry references (only after §4's fix, which is already in place).

Focused validation:

```
pytest -c setup.cfg tests/test_chemistry.py tests/test_spectral_model_correlated_k.py -vv
```

## 6. Refresh The Photospheric Radius Bucket

- `test_cloud_core.py::test_correlated_k_photospheric_radius_calculation`
- `test_radtrans_correlated_k_scattering.py::test_correlated_k_photospheric_radius_calculation`

Unchanged from the v11 plan: a28-era references, ~0.13% drift, no API errors. Regenerate both references on current code and verify.

## 7. Transmission Contribution — IMPLEMENTED (No Refresh Needed)

- `test_radtrans_correlated_k.py::test_correlated_k_transmission_contribution_cloud_calculated_radius` — **passes against the original 3.2.0a8 reference.**

The "contract drift" theory was disproved (both old and new code sum-normalize; the reference sums to 1). Two genuine defects were fixed instead (findings §3A):

1. `compute_transmission_spectrum_contribution` now computes its baseline transit radii internally with the same working-precision numerics as the leave-one-out evaluations, eliminating the layer-independent offset that flattened the contribution toward 1/n_layers. The `transit_radii_squared` parameter was removed; the single caller (`radtrans.py`) and the algebraic unit test (`test_fortran_modules.py`) were updated.
2. The contribution call in `Radtrans._calculate_transit_radii` no longer passes the stale `self.__scattering_in_transmission` flag (always False on this path); it passes `True`, matching the main transit-radii calculation which includes `continuum_opacities_scattering` unconditionally.

Optional follow-up: tighten the test's `rtol` from `1000` to ~`1e-3` now that the function reproduces the reference.

## 8. Refresh The Adaptive Surface-Scattering Reference

- `test_radtrans_correlated_k_surface_scattering.py::test_correlated_k_emission_spectrum_surface_scattering_with_adaptive_feautrier_iterations`

Investigation completed in the v11 pass: pure benchmark drift from a stale 3.1.3 reference (0.089% accumulated JAX-rewrite drift). Regenerate the reference on current code; run the non-adaptive companion at the same time to confirm it stays green.

Caution from the v11 pass: do not "fix" the `jnp.sqrt(TINIEST)` convergence guard in `radiative_transfer_core.py` as part of this — replacing it with bare `TINIEST` triggers float64 promotion that breaks `lax.cond`/`lax.scan` dtype invariants under the mixed-precision configuration.

## 9. Refresh The CCF Reference

- `test_spectral_model_line_by_line_custom.py::test_line_by_line_spectral_model_transmission_ccf`

Unchanged from the v11 plan: the inner mock-transmission benchmark passes; only the CCF surface drifted. Refresh last — the CCF amplifies any upstream spectrum shift, and the §1 rebin renormalization touched the projection code it depends on (covered bins are unchanged, but confirm with the refreshed reference).

## 10. Address JAX Deprecation Warnings (Non-Blocking)

Same three categories as v11 (67 warnings in v13): `jnp.size`/`jnp.shape` on lists (`validation.py:300`, `retrieval/parameter.py`), and the float64→float32 scatter `FutureWarning`. Not failures today; schedule before the next JAX upgrade.

## Suggested Execution Order

1. ~~Fix the runtime projection rebin defect~~ — **done**.
2. ~~Fix the cloud width transpose bug~~ — **done**.
3. ~~Fix the evaluate-mode crashes~~ — **done**.
4. ~~Fix the SpectralModel metallicity unit regression (§4)~~ — **done** (plus `log10_metallicities` API).
5. ~~Fix the transmission contribution function (§7)~~ — **done**; test passes on the original reference.
6. Refresh the chemistry bucket, direct + SpectralModel (§5) — 7 tests.
7. Refresh the photospheric radius references (§6) — 2 tests.
8. Refresh the adaptive surface-scattering reference (§8) — 1 test.
9. Refresh the CCF reference (§9) — 1 test.
10. JAX deprecation warnings (§10) — non-blocking.

## Exit Criteria

- ~~Both log-likelihood routing tests pass unmodified~~ — **met**.
- ~~The three SBI runtime tests pass with the projection fix~~ — **met**.
- ~~`rebin_spectrum_bin` no longer silently truncates flux for partially covered bins~~ — **met** (covered-fraction renormalization; covered-bin results unchanged).
- ~~The per-species cloud width test passes~~ — **met**.
- ~~Evaluate mode writes its summary without requiring pre-existing directories and returns the results object~~ — **met**.
- ~~SpectralModel equilibrium chemistry runs at the requested metallicity, with `metallicity` (linear) and `log10_metallicities` (dex) disambiguated in the API~~ — **met**.
- ~~The transmission contribution test passes against its original reference~~ — **met**.
- Chemistry (direct + SpectralModel), photospheric radius, adaptive Feautrier, and CCF references refreshed.
- A full-suite rerun shows 0 failures.
