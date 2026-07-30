"""Minimal end-to-end SBI workflow for petitRADTRANS.

This script shows the shortest useful path through
``petitRADTRANS.experimental.sbi``: describe the experiment with a
:class:`RetrievalConfig`, bundle it into an :class:`SBIConfig`, and hand the
whole thing to :func:`run_sbi`. That single call

1. builds the SBI task and fingerprints it,
2. generates a prior-predictive simulation dataset (with observational noise),
3. fits a conditional normalizing-flow posterior,
4. validates it on held-out simulations (SBC + posterior-predictive checks),
5. runs amortized inference on the target observation, and
6. writes samples, plots, a summary JSON, and a run manifest to disk.

Everything that differs from the library defaults is expressed through the
grouped configuration dataclasses (:class:`FlowConfig`, :class:`EncoderConfig`,
:class:`TrainingConfig`), so the script itself stays declarative. See
``advanced_sbi_features.py`` for the lower-level building blocks that
:func:`run_sbi` orchestrates.

The forward model here is a fast analytic toy so the example runs in a couple of
minutes on a laptop; it is a reference for the control flow and the API, not for
a scientifically meaningful retrieval.

.. warning::

   ``petitRADTRANS.experimental.sbi`` is under active development. Its public
   API, saved-artifact formats, and numerical behaviour may change without a
   deprecation cycle.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.runtime import ModelResult
from petitRADTRANS.experimental.sbi import (
    EncoderConfig,
    FlowConfig,
    ObservationValueConstraint,
    SBIConfig,
    TrainingConfig,
    run_sbi,
)

tfpd = tfp.distributions

OBSERVATION_NAME = 'synthetic_spectrum'
WAVELENGTHS = np.linspace(1.0, 2.5, 24)

# Ground truth behind the synthetic observation, so the recovered posterior can
# be compared against a known answer at the end of the run.
TRUTH_PARAMETER_VALUES: dict[str, float] = {
    'amplitude': 1.30,
    'tilt': 0.40,
    'line_depth': 0.25,
}
NOISE_SIGMA = 0.01


def synthetic_spectrum_model(
    model_context: Any,
    physical_params: Any,
    pt_plot_mode: bool = False,
) -> ModelResult:
    """Fast analytic stand-in for a radiative-transfer forward model.

    Model-generating functions receive the runtime's ``model_context`` and a
    mapping of physical parameter values, and return a :class:`ModelResult`.
    This one builds a sloped continuum with a single Gaussian absorption
    feature, which is enough to make all three parameters identifiable.
    """
    del model_context, pt_plot_mode

    amplitude = physical_params['amplitude']
    tilt = physical_params['tilt']
    line_depth = physical_params['line_depth']

    wavelengths = jnp.asarray(WAVELENGTHS)
    continuum = amplitude + tilt * (wavelengths - 1.75)
    line = line_depth * jnp.exp(-0.5 * ((wavelengths - 1.60) / 0.06) ** 2)
    return ModelResult(
        kind='spectrum',
        wavelengths=wavelengths,
        spectrum=continuum - line,
    )


# The runtime needs to know the model is JAX-differentiable to use the fast
# vmapped simulation path.
setattr(synthetic_spectrum_model, '_prt_model_contract', 'differentiable')


def simulate_target_observation() -> np.ndarray:
    """Generate the 'observed' spectrum from the known truth, plus noise."""
    truth_result = synthetic_spectrum_model(None, TRUTH_PARAMETER_VALUES)
    noiseless = np.asarray(truth_result.spectrum, dtype=float)
    rng = np.random.default_rng(20260730)
    return noiseless + rng.normal(0.0, NOISE_SIGMA, size=noiseless.shape)


def build_retrieval_config(observed_spectrum: np.ndarray) -> RetrievalConfig:
    """Describe the experiment: priors, forward model, and the observation."""
    configuration = RetrievalConfig(retrieval_name='basic_sbi_demo')

    # Free parameters and their priors. The prior defines what the simulation
    # bank covers, so it also defines where the amortized posterior is valid.
    configuration.add_parameter(
        'amplitude', is_free_parameter=True, distribution=tfpd.Uniform(low=0.5, high=2.0)
    )
    configuration.add_parameter(
        'tilt', is_free_parameter=True, distribution=tfpd.Uniform(low=-1.0, high=1.0)
    )
    configuration.add_parameter(
        'line_depth', is_free_parameter=True, distribution=tfpd.Uniform(low=0.0, high=0.6)
    )

    configuration.add_data(
        name=OBSERVATION_NAME,
        path_to_observations=None,
        model_generating_function=synthetic_spectrum_model,
        radtrans_object=object(),  # unused by the analytic toy model
        wavelengths=jnp.asarray(WAVELENGTHS),
        spectrum=jnp.asarray(observed_spectrum),
        uncertainties=jnp.full(WAVELENGTHS.shape, NOISE_SIGMA),
    )
    return configuration


def main() -> None:
    observed_spectrum = simulate_target_observation()

    # A temporary workspace keeps the example self-contained; point
    # ``output_root`` at a persistent directory to keep the run artifacts.
    with tempfile.TemporaryDirectory(prefix='prt_sbi_basic_') as tmp_dir:
        config = SBIConfig(
            retrieval_config=build_retrieval_config(observed_spectrum),
            output_root=Path(tmp_dir),
            # --- simulation budget -------------------------------------
            # Real problems need 1e6+ simulations; this toy model is cheap
            # and low-dimensional, so a small bank suffices.
            n_simulations=4096,
            simulation_batch_size=256,
            # The toy spectrum is strictly positive, so reject any draw that
            # is not (this is the prior-predictive admissibility constraint).
            observation_value_constraint=ObservationValueConstraint(
                min_value=0.0, min_inclusive=False
            ),
            # --- estimator ----------------------------------------------
            # Rational-quadratic spline coupling flow: its inverse is
            # closed-form, so posterior sampling stays exact however sharp
            # the posterior becomes.
            flow=FlowConfig(
                family='spline',
                num_transforms=6,
                num_spline_bins=8,
                base_distribution='logistic',
            ),
            encoder=EncoderConfig(embedding_dim=32, spectrum_embedding_dim=32, hidden_dim=64),
            training=TrainingConfig(
                num_epochs=30,
                batch_size=256,
                learning_rate=1e-3,
                lr_schedule_total_epochs=30.0,
            ),
            # --- validation / reporting budget --------------------------
            n_posterior_samples=1024,
            n_predictive_forward_model_samples=64,
            validation_max_cases=16,
            truth_parameter_values=TRUTH_PARAMETER_VALUES,
        )

        run_root = run_sbi(config)

        # run_sbi returns the run directory; everything it produced lives there.
        print('\nRun directory contents:')
        for path in sorted(run_root.iterdir()):
            print(f'  {path.name}')

        summary = json.loads((run_root / 'amortized_summary.json').read_text())
        print('\nRecovered posterior vs truth:')
        for name, statistics in summary['posterior_summary'].items():
            truth = TRUTH_PARAMETER_VALUES.get(name)
            mean, std = statistics['mean'], statistics['std']
            deviation = '' if truth is None else f'  ({abs(mean - truth) / max(std, 1e-12):.1f} sigma from truth)'
            print(f'  {name:12s} {mean:+.3f} +- {std:.3f}   truth {truth}{deviation}')

        print(
            '\nSBC normalized mean-rank error (0 = perfectly calibrated):',
            np.round(summary['sbc_normalized_mean_rank_error'], 3),
        )


if __name__ == '__main__':
    main()
