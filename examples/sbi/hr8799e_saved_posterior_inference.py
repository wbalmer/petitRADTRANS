"""Run fast amortized inference on real HR8799e data using a saved SBI posterior.

This script is the companion to
``examples/sbi/hr8799e_runtime_native_real_data.py``. It assumes that a
conditional spline posterior has already been trained and saved, then:

1. rebuilds the real-data retrieval configuration;
2. loads the saved posterior artifact from disk;
3. runs preprocessing-aware amortized inference on the observed GRAVITY data;
4. writes a compact JSON summary and saves a few diagnostic plots.

This is the path you would use once a posterior model is already trained and you
want repeated low-latency inference rather than retraining each time.

Configuration is controlled through editable constants near the top of the
script rather than command-line arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hr8799e_runtime_native_real_data import (  # noqa: E402
    build_raw_observation_blocks,
    build_retrieval_config,
    build_sbi_task,
    locate_input_files,
)
from petitRADTRANS.sbi import (  # noqa: E402
    AmortizedRetrieval,
    ConditionalSplineFlowPosterior,
    plot_posterior_marginals,
    plot_posterior_predictive_report,
)


# User-editable configuration
POSTERIOR_DIRECTORY: Path | None = None
OUTPUT_DIRECTORY: Path | None = None
N_POSTERIOR_SAMPLES = 512
POSTERIOR_PREDICTIVE_INTERVAL_LEVEL = 0.9
INFERENCE_SEED = 2026
GENERATE_PLOTS = True
PLOT_DPI = 180


def main() -> None:
    example_data_file, _ = locate_input_files()
    retrieval_config, pressures = build_retrieval_config(example_data_file)
    del pressures
    task = build_sbi_task(retrieval_config)

    if POSTERIOR_DIRECTORY is None:
        posterior_dir = (
            example_data_file.parents[4]
            / 'runs'
            / 'retrieval_jax_runtime_emission'
            / 'sbi'
            / 'posterior_model'
        )
    else:
        posterior_dir = POSTERIOR_DIRECTORY.resolve()

    if not posterior_dir.exists():
        raise FileNotFoundError(
            f'Saved posterior directory not found: {posterior_dir}. Run hr8799e_runtime_native_real_data.py first.'
        )

    if OUTPUT_DIRECTORY is None:
        output_dir = posterior_dir.parent / 'inference_outputs'
    else:
        output_dir = OUTPUT_DIRECTORY.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior = ConditionalSplineFlowPosterior.load(posterior_dir.as_posix())
    raw_observation_blocks = build_raw_observation_blocks(retrieval_config)
    retrieval = AmortizedRetrieval(task=task, posterior_estimator=posterior)
    result = retrieval.infer(
        raw_observation_blocks,
        n_posterior_samples=N_POSTERIOR_SAMPLES,
        include_posterior_predictive=True,
        posterior_predictive_interval_level=POSTERIOR_PREDICTIVE_INTERVAL_LEVEL,
        seed=INFERENCE_SEED,
    )

    posterior_samples = np.asarray(result.posterior.samples, dtype=float)
    parameter_names = list(task.parameter_names)

    print('Loaded posterior from:', posterior_dir)
    print('Posterior sample shape:', posterior_samples.shape)
    print('Preprocessing applied during inference:', result.metadata['preprocessing_applied'])
    print('Observed GRAVITY points:', result.posterior_predictive.observed_values['GRAVITY'].shape[0])

    summary_path = output_dir / 'saved_posterior_inference_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'task_name': task.name,
                'posterior_dir': posterior_dir.as_posix(),
                'parameter_names': parameter_names,
                'n_posterior_samples': int(N_POSTERIOR_SAMPLES),
                'posterior_predictive_interval_level': float(POSTERIOR_PREDICTIVE_INTERVAL_LEVEL),
                'inference_seed': int(INFERENCE_SEED),
                'generate_plots': bool(GENERATE_PLOTS),
                'posterior_means': np.mean(posterior_samples, axis=0).tolist(),
                'posterior_stds': np.std(posterior_samples, axis=0).tolist(),
                'preprocessing_applied': result.metadata['preprocessing_applied'],
                'preprocessing_version': result.metadata['preprocessing_version'],
                'predictive_interval_coverage': dict(result.posterior_predictive.interval_coverage),
                'predictive_mean_absolute_error': dict(result.posterior_predictive.mean_absolute_error),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print('Wrote inference summary to:', summary_path)

    if GENERATE_PLOTS:
        posterior_figure, _ = plot_posterior_marginals(
            posterior_samples,
            parameter_names=parameter_names,
        )
        posterior_plot_path = output_dir / 'posterior_marginals.png'
        posterior_figure.savefig(posterior_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        print('Saved posterior marginals plot to:', posterior_plot_path)

        predictive_figure, _ = plot_posterior_predictive_report(
            result.posterior_predictive,
            dataset_names=['GRAVITY'],
        )
        predictive_plot_path = output_dir / 'posterior_predictive_gravity.png'
        predictive_figure.savefig(predictive_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        print('Saved posterior-predictive plot to:', predictive_plot_path)
    else:
        print('Plot generation disabled by configuration.')


if __name__ == '__main__':
    main()
