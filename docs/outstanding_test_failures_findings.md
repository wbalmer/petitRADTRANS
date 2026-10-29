# Outstanding petitRADTRANS Test Failures: Findings

## Scope

- Source of truth for this pass: `pRTTestsv13.txt`.
- Raw v13 summary: **19 failed, 263 passed, 1 skipped, 67 warnings** (petitRADTRANS-4.0.0a42).
- Comparison baselines: the v11-based findings (13 failed / 233 passed) and the intermediate v12 run (19 failed / 227 passed).
- This pass:
  - read `pRTTestsv13.txt` in full,
  - classified every failure as a live product defect, benchmark drift, benchmark contract drift, or test-harness update required,
  - identified the concrete root cause of every new failure, verified against the source.
- No production code was changed in this pass.

## Executive Summary

1. The 19 failures split into **12 carried-over benchmark-drift failures** (unchanged from v11, same error signatures verified) and **7 new failures**.
2. Five of the seven new failures share a **single root cause**: the recent `Data` input reorganisation (commits `792d59ed` → `61ad0acf` → `4c5f808f`, June 1–11 2026) makes `Data.__init__` always auto-fill `wavelength_bin_widths`. This silently reroutes the runtime model projection from interpolation to flux rebinning, and the rebin function silently truncates flux for bins that extend beyond the model's wavelength coverage. This exposes a **genuine product defect**: the runtime and legacy likelihood paths now produce different values for the same inputs, and edge bins can silently lose flux in real retrievals.
3. One new failure is a **genuine product bug** in `compute_cloud_particle_mean_radius`: for square per-species width matrices (n_layers == n_clouds), the shape-disambiguation logic transposes input that is already in the documented `(n_layers, n_clouds)` layout, silently discarding per-species particle-width information.
4. One new failure is a missing-directory crash in the new evaluate-mode path: `generate_retrieval_summary()` writes into `evaluate_<name>/` without creating it.
5. The SBI flow-matching JSON serialization bug from v11 is **fixed and verified passing** (`bool()` wrap applied in `training.py`).
6. The v12 intermediate failures (`Data` attribute errors, `system_distance` KeyError, float32 subprocess failure, `test_simple_retrieval` RuntimeError) were all resolved by the Data reorganisation commits and pass in v13.

## Delta From v11/v12

### Resolved since v11

- `test_sbi_interfaces.py::test_flow_matching_posterior_fit_sample_and_generic_load` — fixed by wrapping `np.isfinite(validation_metric)` in `bool()` at `petitRADTRANS/sbi/training.py` (the `is_stable_checkpoint` assignment).

### Resolved since v12 (intermediate run)

- `test_retrieval.py::test_log_likelihood_jax_routes_normal_path_through_runtime` (AttributeError fixed)
- `test_retrieval.py::test_log_likelihood_jax_routes_external_reference_setup_through_runtime` (AttributeError fixed)
- `test_retrieval.py::test_run_legacy_model_adapter_reproduces_builtin_emission_model_output` (`system_distance` KeyError fixed)
- `test_retrieval.py::test_simple_retrieval` (RuntimeError fixed)
- `test_float32_scattering_regressions.py::test_float32_scattering_flux_stays_finite` (subprocess failure fixed)

### New failing test names in v13

- `test_cloud_parameter_plumbing.py::test_compute_cloud_particle_mean_radius_accepts_per_species_width_grid`
- `test_retrieval.py::test_log_likelihood_routes_normal_scalar_path_through_runtime` (failed in v12 with AttributeError; now fails with a numerical mismatch — different root cause)
- `test_retrieval.py::test_log_likelihood_routes_external_reference_setup_through_runtime` (same evolution as above)
- `test_retrieval.py::test_run_evaluate_mode_loads_saved_results_without_sampling`
- `test_sbi_interfaces.py::test_runtime_simulator_generates_deterministic_observations_without_noise`
- `test_sbi_interfaces.py::test_runtime_simulator_supports_nuisance_parameters`
- `test_sbi_interfaces.py::test_generate_local_sensitivity_report_tracks_linear_information_content`

## Status Map

- Raw failures in `pRTTestsv13.txt`: 19
- Live product defects: 3 distinct defects affecting 7 tests
  - runtime projection rebin defect: 5 tests (§1A)
  - cloud per-species width transpose defect: 1 test (§1B)
  - evaluate-mode summary directory defect: 1 test (§1C)
- Benchmark drift (reference refresh required): 11 (§2, unchanged from v11)
- Benchmark contract drift (validate before refresh): 1 (§3, unchanged from v11)

## 1. Live Product Defects (New In v13)

### 1A. Runtime Model Projection Silently Rebins And Truncates Flux

Affected tests (all share one root cause):

- `test_retrieval.py::test_log_likelihood_routes_normal_scalar_path_through_runtime`
- `test_retrieval.py::test_log_likelihood_routes_external_reference_setup_through_runtime`
- `test_sbi_interfaces.py::test_runtime_simulator_generates_deterministic_observations_without_noise`
- `test_sbi_interfaces.py::test_runtime_simulator_supports_nuisance_parameters`
- `test_sbi_interfaces.py::test_generate_local_sensitivity_report_tracks_linear_information_content`

Observed pattern:

- The two routing tests assert `Retrieval.log_likelihood()` (runtime path) equals `Retrieval._log_likelihood_legacy()`. They now disagree: runtime −7.264 vs legacy −0.452 (scalar test) and −14.528 vs −0.903 (external-reference test).
- The simulator tests get `[1.125, 1.375]` where the model returns `[2.0, 3.0]` on the data's own wavelength grid `[1.0, 2.0]`.
- The sensitivity-report test gets a whitened Jacobian of exactly half the expected value (`[5.0, 2.5]` vs `[10.0, 5.0]`).

Concrete root-cause chain (verified in source):

1. **`Data` now always sets bin widths.** Commits `792d59ed` ("ensuring wavelength boundaries are correctly set in Data"), `61ad0acf` ("Reorganise inputs for Data objects"), and `4c5f808f` ("Bug fix", 2026-06-11) make `Data.__init__` auto-compute `wavelength_bin_widths` from `jnp.diff(wavelengths)` whenever the caller does not provide them (`petitRADTRANS/retrieval/data.py:269-288`). Previously these toy `Data` objects had `wavelength_bin_widths = None`.
2. **Bin widths select the projection branch.** `process_spectrum_pipeline` (`petitRADTRANS/retrieval/utils.py:431-444`) uses `jnp.interp` when `data.wavelength_bin_widths is None` (identity when model and data grids match) and `rebin_spectrum_bin` otherwise. With bin widths now always set, the flux-rebin branch is always taken.
3. **No identical-grid short-circuit in the runtime path.** The legacy likelihood (`petitRADTRANS/retrieval/retrieval.py:4128-4140`) explicitly skips rebinning when the model grid equals the data grid. The runtime pipeline has no such check, so the two paths diverge — which is precisely what the routing-equivalence tests detect.
4. **The rebin coverage guard is disabled.** `rebin_spectrum_bin` (`petitRADTRANS/radtrans_core/spectrum_utils.py:749-760`) builds bins `[center − width/2, center + width/2]`. The original input-coverage check (raise if the model grid does not extend half a bin beyond the output grid) has been commented out since the 2025-06 JAX migration (`4ef6ff4c`), because a value-dependent `raise` is not jit-compatible. Bins that extend beyond the model's coverage now silently lose the uncovered fraction of their flux.
5. For the toy tests (model grid == data grid `[1, 2]`, auto bin widths `[1, 1]`), each bin is exactly half covered. The effective projection matrix is `[[0.375, 0.125], [0.125, 0.375]]` — rows summing to 0.5 — which reproduces every observed number: the simulator values (`W·[2,3] = [1.125, 1.375]`), the halved Jacobian (row sums 0.5), and the log-likelihood gaps (residuals `[0.875, 1.625]` → χ² = 13.625 → ΔlogL = −6.8125, matching exactly).

Interpretation:

- The auto-bin-width API change is intentional and physically sensible (flux rebinning is more correct than point interpolation for binned data). The tests must be updated to reflect the new `Data` contract.
- However, two genuine product defects are exposed and must be fixed before the tests are updated:
  - **Defect 1 — runtime/legacy divergence:** both likelihood paths must produce identical values. The runtime pipeline needs the same identical-grid passthrough the legacy path has (or both should rebin identically).
  - **Defect 2 — silent flux truncation:** any real retrieval whose forward-model wavelength grid does not extend at least half a bin beyond the data grid now silently loses flux in the first and last bins. This needs either a jit-compatible renormalisation by the covered bin fraction, or an eager-time coverage validation when the `Data`/observation state is constructed (where Python control flow is still allowed).

### 1B. Per-Species Cloud Particle Width Grid Silently Transposed For Square Inputs

Affected test:

- `test_cloud_parameter_plumbing.py::test_compute_cloud_particle_mean_radius_accepts_per_species_width_grid`

Observed pattern:

- The test passes a `(n_layers=2, n_clouds=2)` width grid `[[1.05, 1.5], [1.05, 1.5]]` (different width per species, same per layer) and asserts that the two species columns yield different mean radii. They come out identical.

Concrete root cause:

- In `compute_cloud_particle_mean_radius` (`petitRADTRANS/radtrans_core/cloud_core.py:177-186`), the 2-D disambiguation checks `shape == (n_clouds, n_layers)` **first** and transposes on match. For square matrices (`n_layers == n_clouds`) this branch always matches, so input already in the documented `(n_layers, n_clouds)` layout — the same convention `f_seds` uses — is wrongly transposed. The per-species widths `[1.05, 1.5]` become per-layer widths, making both species columns identical.
- The width genuinely affects the result (`_compute_mean_radius_single` uses `std_dev` to set the radius fit range), so the transpose silently changes physics whenever square per-species width grids are used.

Interpretation:

- Genuine product bug in shape-precedence logic. The documented layout `(n_layers, n_clouds)` must be checked first; only transpose when the shape matches `(n_clouds, n_layers)` and *not* `(n_layers, n_clouds)`. Square matrices are then interpreted in the documented layout, which is the only consistent choice.

### 1C. Evaluate Mode Crashes Writing The Retrieval Summary

Affected test:

- `test_retrieval.py::test_run_evaluate_mode_loads_saved_results_without_sampling`

Observed pattern:

- `retrieval.run()` in evaluate mode loads saved results successfully, then crashes with `FileNotFoundError` when `generate_retrieval_summary()` opens `output_directory/evaluate_<name>/<name>_ret_summary.txt` for writing.

Concrete root cause:

- The evaluate-mode branch of `run()` (`petitRADTRANS/retrieval/retrieval.py:5058`, recently added — "now self.sampler.get_summary() populates everything") calls `generate_retrieval_summary()`.
- `generate_retrieval_summary` (`retrieval.py:1074-1081`) opens the summary file without creating the `evaluate_<name>` directory.
- That directory is only created in `Retrieval.__init__` (`retrieval.py:349-354`). The test builds the object via `object.__new__(Retrieval)`, bypassing `__init__`, so the directory never exists.

Interpretation:

- Part test-harness artifact (bypassing `__init__`), part robustness gap: any user who deletes or relocates the output directory between sampling and a later evaluate run hits the same crash. The defensive fix — `os.makedirs(..., exist_ok=True)` before opening the summary file — is cheap and covers both.

## 2. Benchmark Drift (Carried Over From v11, Signatures Verified Unchanged)

### 2A. Chemistry Bucket

Affected tests (all compare against `petitRADTRANS-3.2.0a8 (2024-10-26)`):

- `test_chemistry.py::test_chemistry_atmosphere` — outputs `'0'`, `'1'`, `'2'` fail at `rtol=1e-6`
- `test_chemistry.py::test_chemistry_atmosphere_quench` — outputs `'0'`, `'1'`, `'2'` fail
- `test_chemistry.py::test_chemistry_c_o_ratios` — output `'0'` fails
- `test_chemistry.py::test_chemistry_metallicities` — output `'0'` fails
- `test_chemistry.py::test_mass_fractions2metallicities` — parameter check fails at `rtol=1e-14` (stale upstream payload)
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry` — outputs `'1'`, `'3'`, `'4'` fail
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry_imposed` — outputs `'1'`, `'3'`, `'4'` fail

v13 signatures are identical to v11:

- Direct chemistry differences remain tiny (first failing element of the quench test: 0.22144061 vs 0.22126345, ~8×10⁻⁴ relative, against `rtol=1e-6`). The five direct chemistry tests remain a pure reference-refresh family.
- The two SpectralModel tests originally failed because of a genuine metallicity unit-convention regression (§2A.1), which is now **fixed**. Their residual failure is the same table-interpolation drift as the direct bucket; refresh their references together with it.

#### 2A.1. SpectralModel Equilibrium Chemistry Runs At 10× The Requested Metallicity (Root Cause Identified, Fix Not Yet Applied)

Affected tests:

- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry`
- `test_spectral_model_correlated_k.py::test_correlated_k_spectral_model_emission_equilibrium_chemistry_imposed`

Observed: mean molar masses ~2.98 versus reference ~2.36 (26%); fluxes and temperatures (outputs `'1'`, `'3'`) fail in consequence.

Investigation (verified by instrumenting the failing run):

1. The MMW is **consistent with the stored mass fractions** — recomputing μ from the returned mass fractions reproduces 2.975–2.982 exactly. The MMW computation is not the problem; the mass fractions are.
2. The mass fractions are unphysical for the test's solar-metallicity setup: H₂O = 0.224 and CO = 0.039, where solar equilibrium gives H₂O ≈ 5×10⁻³. H₂/He are correspondingly diluted (0.553/0.179 instead of ~0.74/0.24).
3. The chemistry table itself is correct: calling `interpolate_mass_fractions` directly with `log10_metallicities=0.0` (solar), C/O = 0.55 gives H₂O = 4.7×10⁻³, μ = 2.325.
4. Scanning the table reproduces the corruption exactly: **C/O = 0.1, log10Z = 1.0 gives H₂O = 0.2237, μ = 3.095** — matching the failing output (the test uses C/O = 0.1; μ is then lowered to ~2.98 by the filling-species renormalisation).
5. Tracing the failing run confirms: `model_parameters['metallicity'] = 1.0` (linear, as documented for SpectralModel) arrives at the chemistry table as `log10_metallicities = 1.0` — i.e. **the linear metallicity is passed through without `log10()`**, so the model runs at 10× solar.

Concrete root cause — a unit-convention collision between two modules:

- `SpectralModel`'s `metallicity` model parameter is **linear** ("ratio between heavy elements and H2 + He compared to solar", `spectral_model.py` docstring). The pre-rewire code path (`SpectralModel.compute_equilibrium_mass_fractions`, still present at `spectral_model.py:2095-2144` but now dead code) correctly applied `np.log10(metallicity)` before calling the table. The 3.2.0a8 reference was generated with that conversion (μ ≈ 2.355 ✓).
- `chemistry/core.py`'s `calculate_gas_mass_fractions` treats `metallicity` as **log10** — consistent with its own parameter resolution, which aliases `metallicity` with the dex quantities `Fe/H` and `M/H` (`chemistry/core.py:564-569`), and passes it to `interpolate_mass_fractions(log10_metallicities=...)` unconverted (`chemistry/core.py:955-971`).
- Commit `5c765518` ("Spectral model compatibility with chemistry.core", 2026-02-26) rewired `SpectralModel.compute_mass_fractions` (`spectral_model.py:2215`) to delegate to `calculate_gas_mass_fractions`, forwarding the linear `metallicity` straight into the log10-convention function. No conversion was added at the boundary.

Impact beyond the tests:

- Every `SpectralModel` run with `use_equilibrium_chemistry=True` since `5c765518` computes chemistry at `10^Z`× solar instead of `Z`× solar: a user requesting solar (`metallicity=1`) silently gets 10× solar; `metallicity=10` requests 10^10× solar (clamped to the table edge).
- Retrieval-path users who pass `Fe/H`/`M/H` in dex are **unaffected** — for them the pass-through is correct.

Status: **fixed (2026-06-13).** `SpectralModel.compute_mass_fractions` now converts the linear `metallicity` to dex (`jnp.log10`) before delegating to `calculate_gas_mass_fractions`, and additionally accepts an optional `log10_metallicities` argument (dex, passed to the chemistry table as-is; mutually exclusive with `metallicity` — supplying both raises `ValueError`). Verified: `metallicity=1.0` and `log10_metallicities=0.0` produce identical solar chemistry (H₂O = 5.1×10⁻³).

Post-fix test status: the mean molar masses now match the reference (2.3557 vs 2.3557, max rel. diff 6×10⁻⁶), and outputs `'0'`, `'1'`, `'2'`, `'4'` are all within the 1e-4 tolerance. The two tests still fail, but only on output `'3'` (the mass-fraction profiles): trace species (CH₄, C₂H₂, HCN, NH₃, PH₃, FeH, TiO, H, H₂S, H⁻, K, e⁻) drift by up to ~1.8% while the major species (H₂O, CO, H₂, He) agree to ~10⁻⁶. This is the same chemistry-table interpolation drift as the direct `test_chemistry.py` bucket (§2A) — these two references should now simply be refreshed together with that bucket.

### 2B. Photospheric Radius Bucket

- `test_cloud_core.py::test_correlated_k_photospheric_radius_calculation` (`rtol=1e-5`)
- `test_radtrans_correlated_k_scattering.py::test_correlated_k_photospheric_radius_calculation` (`rtol=1e-4`)

Both compare against `petitRADTRANS-4.0.0a28 (2026-05-12)`; the a28→current code drift (~0.13%) persists unchanged. Reference refresh required.

### 2C. CCF Downstream Drift

- `test_spectral_model_line_by_line_custom.py::test_line_by_line_spectral_model_transmission_ccf` — output `'2'` fails at `rtol=1e-4` against `3.2.0a8`; the inner mock-transmission benchmark passes. Unchanged from v11; refresh after upstream transmission is stable.

## 3. Transmission Contribution — RESOLVED (Was Misdiagnosed As Contract Drift)

### 3A. Two Genuine Defects, Not A Normalization Change (Fixed 2026-06-13)

- `test_radtrans_correlated_k.py::test_correlated_k_transmission_contribution_cloud_calculated_radius` — **now passes against the original 3.2.0a8 reference; no refresh needed.**

The earlier "contract drift" theory was wrong. Verified directly: the 3.2.0a8-era Fortran (`compute_transmission_spectrum_contribution`, checked at commit `49652134` and tag `3.2.0`) sum-normalizes over layers exactly like the current JAX code, and the stored reference layer sums equal 1 to machine precision. The normalization never changed.

The actual discrepancy: the reference contribution is a physically sensible peaked profile (0.38 at the photospheric layer 11, falling to ~10⁻⁷ in the deep layers), while the current output was nearly **uniform ≈ 1/27** with a faint bump 2–3 layers too deep. Two genuine defects, both fixed:

1. **Stale scattering flag.** The contribution call in `Radtrans._calculate_transit_radii` passed `self.__scattering_in_transmission`, which is initialized `False` and only ever updated by the photospheric-radius code path — so the contribution ignored all continuum scattering (Rayleigh + clouds), probing too deep. The locally computed flag from `compute_continuum_scattering_opacities` (True when Rayleigh species exist) was never threaded through, and the cloud-path flag is converted to a traced value by `lax.cond`, so it cannot drive the static branch anyway. The main `compute_transit_radii` includes `continuum_opacities_scattering` unconditionally (the array is zero when nothing scatters); the contribution call now does the same (`scattering_in_transmission=True`).
2. **Baseline mismatch flattening the contribution.** The leave-one-out differences used the externally supplied `transit_radii_squared` from the main (float64) transmission path, while the leave-one-out evaluations run in `WORKING_JAX_DTYPE` (float32). The systematic offset between the two baselines is layer-independent and ~40× larger than the true per-layer signal, so after sum-normalization every layer collapsed toward 1/n_layers. `compute_transmission_spectrum_contribution` now computes its baseline internally with the same numerics as the leave-one-out evaluations (the offset cancels exactly); the `transit_radii_squared` parameter was removed and the single caller updated.

After both fixes the contribution matches the 3.2.0a8 reference to ~4 significant figures at the peak (0.3787 vs 0.3787 at layer 11); the only residual difference is the deep-layer floor (reference ~1.0×10⁻⁷ double-precision floor vs current exact 0 in float32), far inside the test tolerance. The algebraic unit test in `test_fortran_modules.py` was updated to the internal-baseline contract.

Remaining (optional) follow-up: the test's `rtol=1000` is no longer meaningful now that the function reproduces the reference; it could be tightened to ~1e-3.

## 4. Adaptive Solver Drift (Carried Over)

### 4A. Adaptive Feautrier Reference Is A Stale 3.1.3 Artefact

- `test_radtrans_correlated_k_surface_scattering.py::test_correlated_k_emission_spectrum_surface_scattering_with_adaptive_feautrier_iterations` — output `'1'` fails at `rtol=1e-4` against `petitRADTRANS-3.1.3 (2025-05-12)`; max relative difference ~0.089%.
- Root cause established in the v11 pass: the reference predates the entire JAX refactor; the non-adaptive companion passes because its reference is from the JAX branch (`3.2.0a8`). The convergence-guard change in `792d59ed` was investigated and ruled out (the `sqrt(TINIEST)` guard never fires at physical flux levels; an attempted "fix" reverting it broke `lax.cond`/`lax.scan` dtype invariants and was rolled back).
- Pure benchmark drift; regenerate the reference on current code.

## 5. Warnings Noted (Not Failures)

The v13 run produces 67 warnings; the same three recurring categories from v11 remain and will become errors in future JAX releases:

- `DeprecationWarning: size requires ndarray or scalar arguments, got <class 'list'>` (`validation.py:300`, `retrieval/parameter.py`).
- `DeprecationWarning: shape requires ndarray or scalar arguments` (same surface).
- `FutureWarning: scatter inputs have incompatible types (float64 → float32 cast)` — the same mixed-precision surface that produced the v10 NaN regression.
