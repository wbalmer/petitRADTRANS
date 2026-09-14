"""Emission-spectrum SBI: a complete, runnable example.

The companion to ``transmission_sbi_example.py``, on the same wavelength grid
(355 points, 0.90-5.29 um) but in emission: self-luminous flux at 2e-17
uncertainties, a per-point signal-to-noise of about 20.

Run it::

    python emission_sbi_example.py

WHY EMISSION IS THE EASIER PROBLEM
----------------------------------
The two examples look similar and behave very differently, which is worth
understanding before choosing a simulation budget.

What matters is not per-point precision but *feature* signal-to-noise -- how
large the molecular features are compared with the noise. Emission spectra have
a feature contrast of order unity (the flux varies by factors across a band),
while a transmission spectrum is a sub-percent ripple on an almost flat
baseline. The transmission benchmark has 180x better per-point precision and
still carries less usable information about the atmosphere.

Practically: emission converges with far fewer simulations, and emission runs in
this module have been well calibrated since early on, while transmission needed
an exact-amplitude conditioning channel and an autoregressive flow before its
marginals stopped sitting at the prior.

The high-level API is identical either way -- ``scattering_in_emission=True`` on
the retrieval config is the only structural difference, and the workflow adapts
plot labels and value constraints from it.

WHAT TO SET FOR REAL WORK
-------------------------
``N_SIMULATIONS`` defaults to 50_000 for a runnable example. Scattering emission
radiative transfer is far more expensive per sample than transmission and holds
much larger intermediates, which is why ``simulation_batch_size`` is small here:
the same batch size that is comfortable for transmission will exhaust a 40 GB
GPU in emission.

Read ``sbc_normalized_mean_rank_error``, ``posterior_prior_width_ratio``,
``prior_edge_report`` and ``target_predictive_summary`` from the run summary
before trusting any result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

# Emission fluxes span ~1e-25..1e-12 cgs across the prior: in float32 the cold
# tail underflows to exactly 0.0 at short wavelengths, and the strict
# "flux > 0" rejection then discards ~95% of prior draws, exhausting the
# dataset-generation resampling budget (the SLURM wrapper's JAX_ENABLE_X64
# line is not exported, so the script must enable x64 itself -- the
# transmission scripts already do).
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

jax.config.update('jax_enable_x64', True)

from petitRADTRANS.chemistry import clouds
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval.models import (
    _build_runtime_emission_model_inputs,
    _compute_gravity,
    _initialize_runtime_pressure_grid,
    _resolve_guillot_gamma,
    _resolve_runtime_abundance_state,
    calculate_emission_spectrum_runtime,
)
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.experimental.sbi import (
    EncoderConfig,
    FlowConfig,
    ObservationValueConstraint,
    RefinementConfig,
    SBIConfig,
    TrainingConfig,
    run_sbi,
)
from petitRADTRANS.temperature_profiles import guillot_global_temperature_profile

tfpd = tfp.distributions

RETRIEVAL_NAME = 'emission_spectrum_perturbed_2e-17_sbi_validation'
OBSERVATION_NAME = 'EmissionSynthetic2e17'
OBSERVATION_FILE = (
    Path(__file__).resolve().parent
    / 'observations'
    / 'emission_spectrum_perturbed_2e-17.txt'
)
OUTPUT_ROOT = Path(__file__).resolve().parent / 'runs' / 'sbi_emission_example'

FIXED_DISTANCE_TO_SYSTEM = float(20.0 * cst.pc)
# The exact synthetic-truth parameters were not available in the benchmarking
# folder, so only the fixed distance is recorded.
TRUTH_PARAMETER_VALUES: dict[str, float] = {
    'distance_to_system': FIXED_DISTANCE_TO_SYSTEM,
    # Values that generated emission_spectrum_perturbed_2e-17.txt. Keys use the
    # *declared parameter names*, which differ from the generating script's:
    # 'gamma' -> 'log_gamma' (the model takes 10**log_gamma), 'CH4' -> 'CH4__MM',
    # 'fsed_' -> 'f_sed_'. planet_radius is in R_jup because its prior declares
    # units=u.jupiterRad; do NOT multiply by cst.r_jup here.
    'log_g': 4.50,
    'planet_radius': 1.0,
    'T_int': 500.0,
    # Generated with T_equ = 0.1, below this task's Uniform(10, 200) prior. The
    # Guillot irradiation term scales as T_equ**4, so against T_int = 500 K it
    # contributes ~1.6e-7 of the internal flux at the prior floor: 0.1 and 10
    # give indistinguishable spectra. Recorded at the floor so the value is
    # scoreable; T_equ is unidentifiable here either way.
    'T_equ': 10.0,
    'log_gamma': 0.0,
    'log_kappa_IR': 1.0,
    'H2O__POKAZATEL': -1.9,
    'CO-NatAbund': -2.3,
    'CH4__MM': -3.9,
    'CO2': -4.3,
    'H2S': -3.9,
    'Na__NewAllard': -5.0,
    'K__Allard': -5.0,
    'log_X_cb_MgSiO3(s)': -2.0,
    'log_Pbase_MgSiO3(s)': -0.5,
    'sigma_lnorm': 1.200,
    'f_sed_MgSiO3(s)': 5.0,
    'log_cloud_radius_MgSiO3(s)': -4.0,
    'cloud_fraction': 0.8,
}


def guillot_emission_mixed_only(
    model_context: Any,
    physical_params: Any,
    pt_plot_mode: bool = False,
):
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    p_use = cast(Any, p_use)
    # Pass parameter_units so the declared planet_radius unit (u.jupiterRad) is
    # converted to cgs, exactly as every built-in model function does. Without
    # this the radius (~1.45) was read as 1.45 cm instead of 1.45 R_jup, so the
    # forward model emitted a ~1 cm planet and the predicted flux collapsed to
    # ~0 (orders of magnitude below the observation).
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    gamma = cast(Any, _resolve_guillot_gamma(physical_params))
    temperatures = guillot_global_temperature_profile(
        p_use,
        10 ** physical_params['log_kappa_IR'],
        gamma,
        gravity,
        physical_params['T_int'],
        physical_params['T_equ'],
    )

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result
    assert abundance_state is not None

    cloud_properties = clouds.setup_clouds(
        abundance_state['pressures'],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


setattr(guillot_emission_mixed_only, '_prt_model_contract', 'differentiable')


def build_retrieval_config(observation_file: Path) -> RetrievalConfig:
    from astropy import units as u

    pressures = jnp.logspace(-5.0, 2.0, 100)
    config = RetrievalConfig(
        retrieval_name=RETRIEVAL_NAME,
        run_mode='retrieval',
        sampler_type='jaxns',
        adaptive_mesh_refinement=False,
        scattering_in_emission=True,
        pressures=pressures,
    )

    config.add_data(
        name=OBSERVATION_NAME,
        path_to_observations=str(observation_file),
        model_generating_function=guillot_emission_mixed_only,
        data_resolution=None,
        model_resolution=200,
        subtract_continuum=False,
        line_opacity_mode='c-k',
    )

    config.add_parameter(
        name='distance_to_system',
        is_free_parameter=False,
        value=FIXED_DISTANCE_TO_SYSTEM,
    )
    config.add_parameter(
        name='log_g',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=2.5, high=5.5),
    )
    config.add_parameter(
        name='planet_radius',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.9, high=2.0),
        units=u.jupiterRad,
    )
    config.add_parameter(
        name='T_int',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=50.0, high=1500.0),
    )
    config.add_parameter(
        name='T_equ',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=10.0, high=200.0),
    )
    config.add_parameter(
        name='log_gamma',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-2.0, high=3.0),
    )
    config.add_parameter(
        name='log_kappa_IR',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-4.0, high=2.0),
    )

    config.set_line_species(
        ['H2O__POKAZATEL', 'CO-NatAbund', 'CH4__MM', 'CO2', 'H2S', 'Na__NewAllard', "K__Allard"],
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-10.0, -1.0),
    )
    config.set_rayleigh_species(['H2', 'He'])
    config.set_continuum_opacities(['H2--H2', 'H2--He'])

    config.add_cloud_species(
        'MgSiO3(s)_crystalline_000__Mie',
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-8.0, -1.0),
        cloud_base_pressure_limits=(-4.0, 2.0),
    )
    config.add_parameter(
        name='sigma_lnorm',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=1.05, high=3.0),
    )
    config.add_parameter(
        name='f_sed_MgSiO3(s)',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.5, high=10.0),
    )
    config.add_parameter(
        name='log_cloud_radius_MgSiO3(s)',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-6.0, high=-2.0),
    )
    config.add_parameter(
        name='cloud_fraction',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.0, high=1.0),
    )
    config.add_parameter(
        name='patchy_clouds',
        is_free_parameter=False,
        value=('MgSiO3(s)_crystalline_000__Mie',),  # type: ignore[arg-type]
    )

    return config


# Example budget, not a production one -- see the module docstring.
N_SIMULATIONS = 50_000


def main() -> None:
    run_sbi(
        SBIConfig(
            retrieval_config=build_retrieval_config(OBSERVATION_FILE),
            output_root=OUTPUT_ROOT,
            # Zero flux is physically valid (cold atmosphere, short wavelength);
            # only negative or non-finite values indicate a solver failure.
            observation_value_constraint=ObservationValueConstraint(
                min_value=0.0, min_inclusive=True
            ),
            n_simulations=N_SIMULATIONS,
            # Deliberately small: scattering emission RT holds far larger
            # per-sample intermediates than transmission. A batch size that is
            # comfortable there has OOMed a 40 GB A100 here.
            simulation_batch_size=64,
            flow=FlowConfig(
                family='neural_autoregressive',
                num_transforms=4,
                hidden_dim=256,
                autoregressive_transform_units=16,
            ),
            encoder=EncoderConfig(
                embedding_dim=128,
                spectrum_embedding_dim=128,
                hidden_dim=256,
                exact_summary_channel=True,
            ),
            training=TrainingConfig(
                learning_rate=5.0e-4,
                num_epochs=60,
                batch_size=512,
                weight_decay=1.0e-2,
                early_stopping_patience=20,
            ),
            # Refinement is off. Both per-observation refiners were measured
            # against the amortized posterior they would replace and both lost
            # every time: importance sampling collapsed to ESS ~1 of 16384
            # proposals, and MAP+Laplace reached only 0.73 coverage at a nominal
            # 0.90 (intervals ~1.7x too narrow), so the delivery gate rejected
            # them and shipped the amortized posterior anyway. Set
            # method='map_laplace' to re-enable and re-measure.
            refinement=RefinementConfig(method='none'),
            truth_parameter_values=TRUTH_PARAMETER_VALUES,
        )
    )


if __name__ == '__main__':
    main()
