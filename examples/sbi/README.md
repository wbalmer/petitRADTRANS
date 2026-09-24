# SBI examples

Simulation-based inference for petitRADTRANS. See
`petitRADTRANS/experimental/sbi/ARCHITECTURE.md` for the module layout and
`docs/sbi/sbi_module.tex` for the method write-up.

| Script | What it shows |
| --- | --- |
| `basic_sbi_workflow.py` | The shortest path: declare an `SBIConfig`, call `run_sbi`. Analytic toy model, runs in minutes on a laptop. |
| `advanced_sbi_features.py` | The lower-level building blocks `run_sbi` orchestrates. |
| `transmission_sbi_example.py` | A real transmission retrieval: 355 points, 0.90-5.29 um, 60 ppm. |
| `emission_sbi_example.py` | The same grid in emission, 2e-17 flux uncertainties. |

Both spectral examples use the observations in `observations/` and default to a
reduced simulation budget so they finish in about an hour on one GPU. That is an
example budget, not a production one -- see each script's docstring.

## Reading a run

Every run writes `amortized_summary.json`. Three numbers first:

1. `sbc_normalized_mean_rank_error` -- calibration. Well under ~0.15 per
   parameter, or the posterior has the wrong shape.
2. `posterior_prior_width_ratio` -- informativeness. This is posterior width
   divided by prior width, so LOW means informative; near 1.0 means that
   parameter came back at the prior.
3. `target_predictive_summary.median_interval_width_over_uncertainty` -- does the
   posterior reproduce the data? Low single digits are healthy.

These are complementary. A posterior that returns the prior is perfectly
calibrated and useless, so calibration alone never demonstrates success.

Then read `prior_edge_report`. It flags parameters whose posterior mass piles
against a prior bound, which is the failure SBC structurally cannot see: SBC
draws its truths from the prior, so it is calibrated by construction in exactly
the regime where a truth near or outside a bound goes wrong. A flag means check
the prior range, not that the run is broken.

`parameter_units` records the unit of each posterior column. Priors declared
with `units=` yield samples in that unit (e.g. R_jup) while undeclared ones
yield cgs, and the saved arrays do not carry that distinction otherwise.

Do not use the validation loss as a quality proxy: in this module's benchmarks it
has repeatedly moved opposite to delivered posterior quality.

## Before a long run

Dataset generation and training take hours to days. Intercept `run_sbi` and check
the configuration and task fingerprint first:

```python
import runpy
import petitRADTRANS.experimental.sbi as sbi

captured = {}
def capture(config):
    captured["config"] = config
    raise SystemExit(0)

sbi.run_sbi = capture
runpy.run_path("my_run.py", run_name="__main__")   # __main__ matters for the fingerprint
```

Confirm the log says *Reusing existing simulation dataset* when you expect reuse.
The fingerprint covers the model function's source text and its `__module__`, so
a script that imports its model function rather than defining it will
fingerprint differently and silently regenerate the dataset.
