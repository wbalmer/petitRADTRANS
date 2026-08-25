"""Transmission-spectrum SBI: a complete, runnable example.

Trains one amortized posterior on a cloudy hot-Jupiter transmission spectrum
(355 points, 0.90-5.29 um, 60 ppm uncertainties -- a JWST-like precision) and
runs inference on it. The forward model, priors and truth values are the same
ones used for the module's transmission benchmark.

Run it::

    python transmission_sbi_example.py

WHAT TO SET FOR REAL WORK
-------------------------
``N_SIMULATIONS`` below defaults to 50_000 so the example finishes in roughly an
hour on one GPU. That is *not* a production budget. How many simulations a run
needs is set by how narrow the posterior is relative to the prior: the module's
own transmission benchmark needed 8 million before the marginals stopped sitting
at the prior. Expect to scale the budget with the sharpness of your data, and
read the SBC and posterior-contraction numbers in the summary rather than
assuming a budget was enough.

WHAT THE RUN PRODUCES
---------------------
``runs/sbi_transmission_example/runs/<label>/`` contains

* ``amortized_summary.json``  -- SBC, contraction, predictive scores, verdicts
* ``target_posterior_samples.npy`` -- the delivered posterior, physical units
* ``target_posterior_corner.png``  -- and the rest of the plot set
* ``refinement_verdict.json`` -- whether test-time refinement was delivered, and why
* ``run_manifest.json`` -- provenance, including the task fingerprint

THE THREE NUMBERS TO READ FIRST
-------------------------------
1. ``sbc_normalized_mean_rank_error`` -- calibration. Should be well under ~0.15
   for every parameter. Large values mean the posterior is the wrong shape.
2. ``posterior_contraction`` -- informativeness. A value near 1.0 means that
   parameter came back at the prior, i.e. the run learned nothing about it.
3. ``target_predictive_summary.median_interval_width_over_uncertainty`` -- does
   the posterior reproduce the data? Values in the low single digits are healthy;
   hundreds mean the posterior is not respecting the observation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

jax.config.update('jax_enable_x64', True)

from petitRADTRANS.chemistry import clouds
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.models import (
    _build_runtime_transmission_model_inputs,
    _compute_gravity,
    _initialize_runtime_pressure_grid,
    _resolve_guillot_gamma,
    _resolve_runtime_abundance_state,
)
from petitRADTRANS.retrieval.runtime import ModelResult
from petitRADTRANS.experimental.sbi import (
    EncoderConfig,
    FlowConfig,
    ObservationValueConstraint,
    RefinementConfig,
    SBIConfig,
    StabilityConfig,
    TrainingConfig,
    run_sbi,
)
from petitRADTRANS.temperature_profiles import guillot_global_temperature_profile


tfpd = tfp.distributions


RETRIEVAL_NAME = 'transit_spectrum_v2_60ppm_sbi_validation'
OBSERVATION_NAME = 'TransitSyntheticV2_60ppm'
OBSERVATION_FILE = (
    Path(__file__).resolve().parent
    / 'observations'
    / 'transit_spectrum_v2_perturbed_60ppm.txt'
)
OUTPUT_ROOT = (
    Path(__file__).resolve().parent / 'runs' / 'sbi_transmission_example'
)
TRANSIT_VALUE_CONSTRAINT = ObservationValueConstraint(
    min_value=0.0,
    max_value=1.0,
    min_inclusive=False,
    max_inclusive=True,
)
# 0.6 Rsun (M dwarf). With R_p = 1 Rjup this gives a transit depth of
# (R_p/R_s)^2 = 0.0293, matching the observed 0.0301 median.
FIXED_STELLAR_RADIUS = float(0.6 * cst.r_sun)


# Truth values used to generate the perturbed target spectrum.
TRUTH_PARAMETER_VALUES: dict[str, float] = {
    'distance_to_system': float(20.0 * cst.pc),
    'log_g': 3.50,
    'planet_radius': float(1.0 * cst.r_jup),
    'T_int': 400.0,
    'T_equ': 500.0,
    'gamma': 1e2,
    'log_gamma': 2.0,
    'log_kappa_IR': -2.0,
    'H2O__POKAZATEL': -1.9,
    'CO-NatAbund': -2.3,
    'CH4__MM': -3.9,
    'CO2': -4.3,
    'H2S': -3.9,
    'sigma_lnorm': 1.8,
    'log_kzz': 8.0,
    'f_sed_MgSiO3(s)': 5.0,
    'log_cloud_radius_MgSiO3(s)': -4.0,
    'log_X_cb_MgSiO3(s)': -3.0,
    'log_Pbase_MgSiO3(s)': -0.5,
    'stellar_radius': FIXED_STELLAR_RADIUS,
    # NOTE: 1.0 sits exactly on the upper edge of the U(0, 1) prior
    # (cube coordinate 1.0000). The spectrum is what it is, so the run
    # proceeds, but cloud_fraction's SBC rank and interval coverage are
    # degenerate by construction and must be disregarded for it; every
    # posterior draw lies at or below the truth. See the run notes.
    'cloud_fraction': 1.0,
}


def guillot_transmission_mixed_only(
    model_context: Any,
    physical_params: Any,
    pt_plot_mode: bool = False,
):
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    p_use = cast(Any, p_use)
    reference_pressure = 100.0
    if 'reference_pressure' in physical_params.keys():
        reference_pressure = physical_params['reference_pressure']

    gravity, planet_radius = _compute_gravity(physical_params)

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

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    cloud_properties = clouds.setup_clouds(
        abundance_state['pressures'],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_transmission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_properties=cloud_properties,
    )

    wavelengths_model, transit_radii, additional_outputs = (
        model_context.radtrans.calculate_transit_radii(
            temperatures=model_inputs.temperatures,
            mass_fractions=model_inputs.abundances,
            reference_gravity=model_inputs.gravity,
            mean_molar_masses=model_inputs.mean_molar_masses,
            planet_radius=model_inputs.planet_radius,
            reference_pressure=model_inputs.reference_pressure,
            opaque_cloud_top_pressure=model_inputs.opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=model_inputs.sigma_lnorm,
            cloud_particle_mean_radii=model_inputs.cloud_particle_mean_radii,
            cloud_f_sed=model_inputs.cloud_f_sed,
            eddy_diffusion_coefficients=model_inputs.eddy_diffusion_coefficients,
            haze_factor=model_inputs.haze_factor,
            power_law_opacity_coefficient=model_inputs.power_law_opacity_coefficient,
            power_law_opacity_350nm=model_inputs.power_law_opacity_350nm,
            cloud_hansen_b=model_inputs.cloud_hansen_b,
            cloud_fraction=model_inputs.cloud_fraction,
            patchy_clouds=model_inputs.patchy_clouds,
            cloud_particle_radius_distribution=model_inputs.cloud_particle_radius_distribution,
            return_contribution=model_context.return_contribution,
            return_clear_spectrum=False,
            return_cloud_contribution=model_context.return_cloud_contribution,
            return_radius_hydrostatic_equilibrium=model_context.return_radius_hydrostatic_equilibrium,
            return_abundances=model_context.return_abundances,
            return_opacities=model_context.return_opacities,
        )
    )
    wavelengths_model = wavelengths_model * 1e4

    if model_inputs.stellar_radius is None:
        raise ValueError(
            'stellar_radius is required for transmission spectrum evaluation'
        )

    spectrum_model = (transit_radii / model_inputs.stellar_radius) ** 2.0
    auxiliary_outputs: dict[str, Any]
    if additional_outputs is None:
        auxiliary_outputs = {}
    elif isinstance(additional_outputs, dict):
        auxiliary_outputs = dict(additional_outputs)
    else:
        auxiliary_outputs = {'runtime_output': additional_outputs}

    return ModelResult(
        kind='spectrum',
        wavelengths=wavelengths_model,
        spectrum=spectrum_model,
        auxiliary_outputs=auxiliary_outputs,
    )


setattr(guillot_transmission_mixed_only, '_prt_model_contract', 'differentiable')


def build_retrieval_config(observation_file: Path) -> RetrievalConfig:
    pressures = jnp.logspace(-6.0, 2.0, 100)
    config = RetrievalConfig(
        retrieval_name=RETRIEVAL_NAME,
        run_mode='retrieval',
        sampler_type='jaxns',
        adaptive_mesh_refinement=False,
        scattering_in_emission=False,
        pressures=pressures,
    )

    config.add_data(
        name=OBSERVATION_NAME,
        path_to_observations=str(observation_file),
        model_generating_function=guillot_transmission_mixed_only,
        data_resolution=None,
        model_resolution=200,
        subtract_continuum=False,
        line_opacity_mode='c-k',
    )

    config.add_parameter(
        name='distance_to_system',
        is_free_parameter=False,
        value=TRUTH_PARAMETER_VALUES['distance_to_system'],
    )
    config.add_parameter(
        name='log_g',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=2.5, high=5.5),
    )
    config.add_parameter(
        name='planet_radius',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.6 * cst.r_jup, high=2.0 * cst.r_jup),
    )
    config.add_parameter(
        name='T_int',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=50.0, high=1500.0),
    )
    config.add_parameter(
        name='T_equ',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=50.0, high=2000.0),
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
    config.add_parameter(
        name='stellar_radius',
        is_free_parameter=False,
        value=FIXED_STELLAR_RADIUS,
    )

    config.set_line_species(
        ['H2O__POKAZATEL', 'CO-NatAbund', 'CH4__MM', 'CO2', 'H2S'],
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-10.0, -1.0),
    )
    config.set_rayleigh_species(['H2', 'He'])
    config.set_continuum_opacities(['H2--H2', 'H2--He'])

    config.add_cloud_species(
        'MgSiO3(s)_crystalline_000__Mie',
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-8.0, -1.0),
        cloud_base_pressure_limits=(-6.0, 2.0),
    )
    config.add_parameter(
        name='sigma_lnorm',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=1.05, high=3.0),
    )
    config.add_parameter(
        name='log_kzz',
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=5.0, high=12.0),
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
    config.add_parameter(
        name='return_clear_spectrum',
        is_free_parameter=False,
        value=False,
    )
    return config


# 50k keeps the example to about an hour on one GPU. See the module docstring:
# this is an example budget, not a production one.
N_SIMULATIONS = 50_000


def main() -> None:
    run_sbi(
        SBIConfig(
            retrieval_config=build_retrieval_config(OBSERVATION_FILE),
            output_root=OUTPUT_ROOT,
            observation_value_constraint=TRANSIT_VALUE_CONSTRAINT,
            n_simulations=N_SIMULATIONS,
            simulation_batch_size=256,
            # Neural autoregressive, not spline coupling. On this benchmark the
            # coupling flow returned the prior for log_g even though the
            # conditioning vector carried it at R^2 0.93: a coupling layer
            # transforms half its dimensions through one conditioner shared
            # across all parameters, while a NAF gives every dimension its own
            # path from the context. Swapping the family (at matched capacity)
            # moved val_loss from -12.4 to -19.7 and took the run from 4/18 to
            # 16/18 constrained parameters.
            #
            # NAF cost scales as num_transforms x n_parameters conditioner MLPs,
            # so do NOT port num_transforms/hidden_dim from a coupling config.
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
                # Appends the observation's amplitude to the conditioning vector
                # analytically instead of asking the network to preserve it.
                exact_summary_channel=True,
            ),
            training=TrainingConfig(
                learning_rate=5.0e-4,
                num_epochs=60,
                batch_size=512,
                weight_decay=1.0e-2,
                early_stopping_patience=20,
            ),
            stability=StabilityConfig(),
            # Test-time refinement is gated twice: on absolute thresholds, and on
            # whether it actually beats the amortized posterior by CRPS. It is
            # only delivered if it wins.
            refinement=RefinementConfig(
                method='map_laplace',
                map_covariance_method='hessian',
                test_time_validation=True,
                test_time_validation_cases=8,
            ),
            truth_parameter_values=TRUTH_PARAMETER_VALUES,
        )
    )


if __name__ == '__main__':
    main()
