# Native correlated-k mixing kernel (`prt_ck_mixing_ffi.cc`)

`prt_ck_mixing_ffi.cc` is the CPU implementation of petitRADTRANS's correlated-k
(c-k) opacity mixing. It is the only C++ source in the package. It is built
into the extension module `petitRADTRANS.prt_ck_mixing_ffi` and called from JAX
as an [XLA FFI](https://docs.jax.dev/en/latest/ffi.html) custom call. That
keeps it inside the compiled XLA program, with no host round trip and no GIL.

The kernel provides three operations, each for float32 and float64 inputs:

| Operation | FFI target | Computes |
|---|---|---|
| value | `prt_combine_ck_mix_{f32,f64}` | the mixed opacities |
| forward mode | `prt_combine_ck_mix_jvp_{f32,f64}` | tangents of the mixed opacities |
| reverse mode | `prt_combine_ck_mix_vjp_{f32,f64}` | cotangents of the inputs |

On GPU and other non-CPU backends, the mixing runs entirely in JAX, in
`_combine_ck_opacities_jax` in `radtrans_core/radiative_transfer_core.py`. Both
implementations compute the same mixing, and the tests hold them to agree to
1e-12.

## When the mixing runs

The mixing combines the c-k opacities of all line species into one
k-distribution per (frequency, layer) cell. pRT needs it only for **emission
spectra with `scattering_in_emission=True`** (`Radtrans._set_sum_opacities`).
Transmission spectra, emission without scattering, and line-by-line mode never
call this kernel.

## Algorithm

For each cell, species are added to the running mixture one at a time with the
random-overlap approximation:

1. **Pairwise sums.** Form the `n_g * n_g` sums `current_i + species_j`, each
   with weight `w_i * w_j` (normalised to sum to 1, in `WORKING_JAX_DTYPE`).
2. **Order.** Sort the sums by opacity; their cumulative weights then define a
   quantile function `Q(g)` on `(0, 1]`.
3. **Window means.** At each g node, average `Q` over a window centred on the
   node that reaches the nearer edge of its quadrature bin. Two sums only swap
   places in the sort when they are equal, which leaves `Q` unchanged, so the
   window means are continuous in the opacities.
4. **Threshold ramp.** Blend the result in as `current + lambda * (mean - current)`,
   where `lambda` depends on the species' strongest opacity relative to the
   threshold:
   - **Threshold:** 1% of the cell's largest weakest-g opacity, continuum
     included.
   - **Skipped** (`lambda = 0`): the species is below 0.5 of the threshold.
   - **Fully mixed** (`lambda = 1`): the species is at or above the threshold.
   - **Smoothstep** in between.

Steps 3 and 4 replace the Fortran scheme. The sorted sums were interpolated
at the midpoints of their weight intervals and species were switched on and off
at once. Both made the mixed opacities jump, so automatic differentiation missed
part of the gradient. See `CHANGELOG.md` (unreleased) for the measured effect.

The three constants `kMixingThreshold`, `kMixingRampStart` and
`kMixingWindowFraction` at the top of the file must match `CK_MIXING_THRESHOLD`,
`CK_MIXING_RAMP_START` and `CK_MIXING_WINDOW_FRACTION` in
`radiative_transfer_core.py`.

## How it is called from Python

```
Radtrans.calculate_flux (scattering_in_emission=True)
└─ opacities/continuum.py: compute_combined_opacities
   └─ radtrans_core/radiative_transfer_core.py: combine_ck_opacities
      ├─ GPU and other backends: _combine_ck_opacities_jax            (pure JAX)
      └─ CPU: _combine_ck_opacities_ffi                               (jax.custom_jvp)
         ├─ value:   ck_mixing_ffi.combine_ck_mixing_ffi      → prt_combine_ck_mix_*
         └─ JVP rule _combine_ck_ffi_jvp
            └─ primitive _ck_mixing_jvp_p
               ├─ forward mode: ck_mixing_ffi.combine_ck_mixing_jvp_ffi → prt_combine_ck_mix_jvp_*
               └─ reverse mode: its transpose, primitive _ck_mixing_vjp_p
                  └─ ck_mixing_ffi.combine_ck_mixing_vjp_ffi            → prt_combine_ck_mix_vjp_*
```

- **`radtrans_core/ck_mixing_ffi.py`** registers the six handlers with
  `jax.ffi.register_ffi_target` on first use, and wraps each in `jax.ffi.ffi_call`.
- **The two primitives** in `radiative_transfer_core.py` are each the other's
  transpose, so `jax.jvp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev` all run on
  the native kernels.
- **Second and higher derivatives** (for example Hessians for a NUTS mass matrix)
  come from the JAX implementation, through the primitives' own JVP rules.
- **`vmap`** is folded into the cell axis (a batch of inputs) or the tangent
  axis (a batch of tangents) instead of looping. Only a batch of g grids is
  looped over.

All arrays use the layout pRT already holds, so no transpose is needed. Here
`n_fl = n_frequencies * n_layers` and the arrays are row-major:

| Operand | Shape |
|---|---|
| weighted opacities (opacity times mass fraction; continuum *not* added) | `(n_fl, n_species, n_g)` |
| continuum opacities | `(n_fl,)` |
| g nodes and quadrature weights | `(n_g,)` each |
| forward-mode tangents (in) | `(n_tangents, n_fl, n_species, n_g)` and `(n_tangents, n_fl)` |
| forward-mode tangents (out) | `(n_tangents, n_fl, n_g)` |
| reverse-mode cotangents (in) | `(n_tangents, n_fl, n_g)` |
| reverse-mode cotangents (out) | `(n_tangents, n_fl, n_species, n_g)` and `(n_tangents, n_fl)` |

The continuum is passed separately because it also enters the threshold.

## Program flow through the file

The file reads top-down in the order below. Everything except the module init
function is in an anonymous namespace.

### 1. Entry: FFI handlers

`XLA_FFI_DEFINE_HANDLER_SYMBOL` (via the `PRT_CK_MIX_SYMBOLS` macro) declares the
operand and result types of each handler. The macro `PRT_CK_MIX_HANDLERS` turns
them into the typed entry points `PrtCombineCkMix{,Jvp,Vjp}{F32,F64}Impl`, which
forward to the three templated implementations:

- `PrtCombineCkMixImpl` (value)
- `PrtCombineCkMixJvpImpl` (forward mode)
- `PrtCombineCkMixVjpImpl` (reverse mode)

Each implementation:

1. calls **`PrepareInputs`**, which checks the shapes of the primal inputs, fills
   a `Buffers` struct with pointers and sizes, and calls **`BuildGrid`**;
2. checks the shapes of its own tangent, cotangent or result buffers;
3. calls **`RunMixCells<Mode>`** with `Mode::kValue`, `kForward` or `kReverse`.

Errors are returned as `ffi::Error::InvalidArgument`, which JAX raises as a
Python exception.

### 2. Per-call setup: `BuildGrid`

Everything that depends only on the g grid is computed once per call and
shared by all cells in a `Grid`:

- the `n_g * n_g` pairwise weights, normalised to sum to 1. They are formed exactly as
  `build_ck_combination_state` forms them: products in single precision
  (`WORKING_JAX_DTYPE`), divided by their sum accumulated in double precision and
  rounded back to single precision, so both implementations use the same values;
- each node's averaging window `[window_low, window_high]`, from the bin edges
  given by the cumulative quadrature weights.

### 3. Parallel split: `RunMixCells`

The cells are split into about `8 * threads` tasks on XLA's intra-op thread pool
(`ffi::ThreadPool`). Several tasks per thread balance the load, because a cell's
cost depends on how many species it skips.

- **Workspace:** each task builds its own `Workspace` (its scratch memory),
  then calls `MixCells` on its range of cells.
- **First task:** runs on the calling thread.
- **Waiting:** the call returns once an atomic counter shows that every task has
  finished.

### 4. Per-cell loop: `MixCells<Mode>`

For each cell:

1. **Threshold.** Find the largest weakest-g opacity: species 0 plus the
   continuum, or another species. Remember which species it was, because the
   derivatives need it, and scale it by `kMixingThreshold`.
2. **Start.** Set the mixture to species 0 plus the continuum. In forward mode,
   also initialise the mixture's tangents and the threshold's tangent.
3. **For each further species:**
   1. **`MixingWeight`** gives `lambda` and its slope from the ratio of the
      species' strongest opacity to the threshold. If `lambda` is 0, skip the
      species.
   2. **`WindowMeans`** forms the pairwise sums. It orders them with
      `OrderPairSums`, then accumulates the cumulative weights and weighted
      opacities. It finds the sample covering each window bound
      (`CoveringSample`, a binary search) and computes each window mean as a
      difference of the integral of `Q`.
   3. **Forward mode:** carry each tangent through the same step. Gather the
      tangents of the sorted sums, average them over each window with
      `WindowSum`, and add the derivative of the ramp. This is done for all
      `n_tangents` tangents at once, so the step is ordered only once.
   4. **Reverse mode:** record the step in `ws.steps[s]` (a `StepRecord`). It
      holds the sort order, cumulative weights, covering samples, the mixture
      before the step, the window means, and `lambda` with its slope.
   5. **Blend:** `current += lambda * (mean - current)`.
4. **Write out:**
   - **value mode:** the mixture;
   - **forward mode:** the tangents;
   - **reverse mode:** call **`BackPropagateCell`**.

`OrderPairSums` exploits the fact that k-table opacities ascend along g. Each
row `current_i + species_j` is then already a sorted run, and the `n_g` runs are
merged pairwise, which is about 1.4x faster than sorting. If a species'
opacities are not sorted, it falls back to `std::stable_sort`. Both orderings
break ties by sample index, exactly like JAX's stable `argsort`, so the two
implementations order samples, and therefore differentiate, identically.

### 5. Reverse sweep: `BackPropagateCell`

For each output cotangent, walk the recorded steps from the last species back
to the first:

1. **Blend:** split the cotangent into the part that bypasses the step,
   `(1 - lambda)`, and the part that reaches the window means, `lambda`. Also
   collect the cotangent of `lambda`.
2. **Window means:** spread each window's cotangent over the sorted sums it
   covers, in proportion to the part of the window each covers. This is the
   transpose of `WindowSum`.
3. **Back to the inputs:** map each sorted sum back through the recorded order
   to its mixture node `i` and species node `j`.
4. **Threshold:** route the ramp's derivative to the species' strongest opacity
   and to the threshold.

Finally, the mixture's cotangent goes to species 0 and the continuum, and the
threshold's cotangent goes to whichever input set it.

### 6. Python module

`PyInit_prt_ck_mixing_ffi` creates the module. Its six functions
(`combine_ck_mix_{,jvp_,vjp_}{f32,f64}_capsule`) return the handler symbols as
`PyCapsule`s for `jax.ffi.register_ffi_target`.

## Numerical notes

- **Precision:** all per-cell arithmetic is in double precision, whatever the
  input dtype.
- **No cancellation in derivatives:** `WindowSum` adds up only the samples
  inside a window instead of subtracting two cumulative sums. Differencing
  cumulative sums would lose precision in small derivative entries, and made
  forward and reverse mode disagree by up to 5e-9.
- **Ramp limit:** for a species inside the threshold ramp, the derivative
  includes the small change it makes to the mixture, `mean - current`. That
  difference is accurate to only about 1e-8 relative, in this kernel and in the
  JAX implementation alike.

## Building

Meson builds the module with the rest of the package (`meson.build` at the
repository root, target `prt_ck_mixing_ffi`), for example with
`pip install -e .`. It needs a C++17 compiler and the XLA FFI headers that ship
with jaxlib (`jax.ffi.include_dir()`); it does not link against XLA.

If petitRADTRANS is imported from the source tree via `PYTHONPATH` rather than
installed, the compiled `prt_ck_mixing_ffi.cpython-*.so` must sit in
`petitRADTRANS/` next to `__init__.py`. A manual build:

```bash
c++ -std=c++17 -O3 -fPIC -shared -fvisibility=hidden \
    -I"$(python -c 'import jax.ffi; print(jax.ffi.include_dir())')" \
    -I"$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
    petitRADTRANS/cpp_src/prt_ck_mixing_ffi.cc \
    -o petitRADTRANS/prt_ck_mixing_ffi$(python3-config --extension-suffix)
```

At run time the module needs a libstdc++ at least as new as the compiler's. A
Python module that is already loaded is not reloaded, so restart the process
after rebuilding. Replace the `.so` with a move (`mv`), not a copy over the old
file, if running processes may have it loaded.

## Changing the kernel

- **Keep the JAX path in step.** Any change to the mixing must be made in both
  `prt_ck_mixing_ffi.cc` and `_combine_ck_opacities_jax`. The JAX path is used
  on GPU and for derivatives beyond first order.
- **Run the tests** in `tests/test_fortran_modules.py`:
  - `test_combine_ck_opacities_cpu_kernel_matches_jax_implementation`: values
    agree to 1e-12.
  - `test_combine_ck_opacities_cpu_kernel_derivatives_match_jax_implementation`:
    native forward- and reverse-mode derivatives, also under `vmap`.
  - `test_combine_ck_opacities_cpu_kernel_derivatives_agree_across_modes`:
    forward against reverse mode, first and second order.
  - `test_combine_ck_opacities_derivative_integrates_to_value_change` and
    `test_combine_ck_opacities_switches_species_on_continuously`: continuity.
  - `test_combine_ck_opacities_close_to_exact_random_overlap`: accuracy.
- **Regenerate references if the values change.** The reference spectra of the
  emission tests with scattering depend on the mixing.
