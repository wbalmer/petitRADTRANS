"""Leave-one-out cross-validation analysis (Welbanks et al. 2023).

Compares two transmission retrievals of the same HST spectrum with the Dynesty
sampler: a baseline free-chemistry model (CH4, CO2, CO) and the same model with
H2O added back in. Uses a custom differentiable ``retrieval_model_spec_iso``
forward model, then computes per-datapoint ELPD and Pareto-k diagnostics to
assess the impact of H2O.

Equivalent to the ``retrieval_leave_one_out_analysis`` documentation notebook.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import petitRADTRANS as prt
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.chemistry import clouds
from petitRADTRANS.chemistry.core import get_abundances
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import (
    MODEL_CONTRACT_DIFFERENTIABLE,
    _compute_gravity,
    calculate_transmission_spectrum_runtime,
    initialize_pressure,
)
from petitRADTRANS.retrieval.runtime import (
    ModelContext,
    ModelInputs,
    ModelResult,
    PhysicalParams,
    _radtrans_wavelength_grid,
)
from petitRADTRANS.temperature_profiles import isothermal_temperature_profile

tfpd = tfp.distributions

PRESSURES = jnp.logspace(-6.0, 2.0, 100)
CORNER_LABELS = {
    "1H2-16O__POKAZATEL": r"$\log X_{\rm H_2O}$",
    "12C-1H4__HITEMP": r"$\log X_{\rm CH_4}$",
    "12C-16O2__UCL-4000": r"$\log X_{\rm CO_2}$",
    "C-O-NatAbund__HITEMP": r"$\log X_{\rm CO}$",
}


def retrieval_model_spec_iso(model_context, physical_params, pt_plot_mode=False):
    """Isothermal transmission model, mirroring the built-in ``isothermal_transmission``."""
    p_use, p_global = initialize_pressure(
        model_context.radtrans.pressures / 1e6,
        physical_params,
        model_context.adaptive_mesh_refinement,
    )

    reference_pressure = 100.0
    if "reference_pressure" in physical_params.keys():
        reference_pressure = physical_params["reference_pressure"]

    temperatures = isothermal_temperature_profile(p_use, physical_params["temperature"])
    gravity, planet_radius = _compute_gravity(physical_params)

    abundances, mean_molar_masses, small_index, _p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=physical_params,
        line_species=model_context.radtrans.line_species,
        cloud_species=model_context.radtrans.cloud_species,
        adaptive_mesh_refinement=model_context.adaptive_mesh_refinement,
    )

    if abundances is None:
        return ModelResult.make_nan_result(_radtrans_wavelength_grid(model_context.radtrans))

    if pt_plot_mode:
        return ModelResult(
            kind="pt_profile",
            pressures=p_use[small_index],
            temperatures=temperatures[small_index],
        )

    if model_context.adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = p_global[small_index]
        mean_molar_masses = mean_molar_masses[small_index]
        model_context.radtrans.pressures = pressures * 1e6
    else:
        pressures = p_use

    if pressures.shape[0] != model_context.radtrans.pressures.shape[0]:
        return ModelResult.make_nan_result(_radtrans_wavelength_grid(model_context.radtrans))

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = clouds.setup_clouds(pressures, physical_params, model_context.radtrans.cloud_species)

    model_inputs = ModelInputs(
        pressures=pressures,
        temperatures=temperatures,
        abundances=abundances,
        mean_molar_masses=mean_molar_masses,
        gravity=gravity,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        cloud_particle_radius_distribution=distribution,
        stellar_radius=physical_params["stellar_radius"],
    )

    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


# Mark the custom model as differentiable so Dynesty uses the JAX-aware runtime path.
setattr(retrieval_model_spec_iso, "_prt_model_contract", MODEL_CONTRACT_DIFFERENTIABLE)


def build_leave_one_out_config(retrieval_name, line_species, example_data_file, wavelength_boundaries):
    retrieval_config = RetrievalConfig(
        retrieval_name=retrieval_name,
        run_mode="retrieval",
        sampler_type="dynesty",
        pressures=PRESSURES,
        adaptive_mesh_refinement=False,
        scattering_in_emission=False,
    )

    retrieval_config.add_parameter(
        name="stellar_radius", is_free_parameter=False, value=0.651 * cst.r_sun, plot_in_corner=False
    )
    retrieval_config.add_parameter(
        name="reference_pressure", is_free_parameter=False, value=0.01, plot_in_corner=False
    )
    retrieval_config.add_parameter(
        name="log_g", is_free_parameter=True, distribution=tfpd.Uniform(low=2.0, high=5.5),
        plot_in_corner=True, corner_ranges=(2.0, 5.5), corner_label=r"$\log(g)$",
    )
    retrieval_config.add_parameter(
        name="planet_radius", is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.1 * cst.r_jup_mean, high=0.4 * cst.r_jup_mean),
        plot_in_corner=True, corner_ranges=(0.1, 0.4),
        corner_label=r"$R_{\rm P}$ ($R_{\rm Jup}$)",
        corner_transform=lambda radius: radius / cst.r_jup_mean,
    )
    retrieval_config.add_parameter(
        name="temperature", is_free_parameter=True, distribution=tfpd.Uniform(low=300.0, high=2000.0),
        plot_in_corner=True, corner_ranges=(300.0, 2000.0), corner_label="Temperature [K]",
    )
    retrieval_config.add_parameter(
        name="log_Pcloud", is_free_parameter=True, distribution=tfpd.Uniform(low=-6.0, high=2.0),
        plot_in_corner=True, corner_ranges=(-6.0, 2.0), corner_label=r"$\log P_{\rm cloud}$",
    )

    retrieval_config.set_rayleigh_species(["H2", "He"])
    retrieval_config.set_continuum_opacities(["H2--H2", "H2--He"])
    retrieval_config.set_line_species(
        list(line_species),
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-6.0, 0.0),
        plot_in_corner=True,
    )
    for species in retrieval_config.line_species:
        retrieval_config.parameters[species].corner_label = CORNER_LABELS.get(species, species)
        retrieval_config.parameters[species].corner_ranges = (-6.2, 0.2)

    retrieval_config.add_data(
        name="HST",
        path_to_observations=str(example_data_file),
        model_generating_function=retrieval_model_spec_iso,
        data_resolution=60,
        model_resolution=120,
        wavelength_boundaries=wavelength_boundaries,
        line_opacity_mode="c-k",
    )

    return retrieval_config


def configure_plotter(retrieval):
    retrieval.plotter.spec_xlabel = "Wavelength [micron]"
    retrieval.plotter.spec_ylabel = r"$(R_{\rm P}/R_*)^2$ [ppm]"
    retrieval.plotter.y_axis_scaling = 1e6
    retrieval.plotter.xscale = "linear"
    retrieval.plotter.yscale = "linear"
    retrieval.plotter.reference_data_name = "HST"
    retrieval.plotter.temp_limits = [150, 3000]
    retrieval.plotter.press_limits = [1e1, 1e-6]


def run_dynesty_retrieval(retrieval):
    return retrieval.run(
        nlive=40,
        dlogz_init=0.5,
        bound="multi",
        sample="rwalk",
        use_jit=True,
        resume=True,
    )


def load_results_if_available(retrieval):
    ret_name = retrieval.configuration.retrieval_name
    dynesty_dir = Path(retrieval.output_directory) / "out_Dynesty"
    if not (dynesty_dir / f"{ret_name}_samples.npz").exists() or \
            not (dynesty_dir / f"{ret_name}_params.json").exists():
        print(f"Missing Dynesty outputs for {ret_name}.")
        return None, None, None, None
    sample_dict, parameter_dict = retrieval.get_samples(output_directory=retrieval.output_directory)
    return sample_dict, parameter_dict, sample_dict[ret_name], parameter_dict[ret_name]


def main():
    package_root = Path(prt.__file__).resolve().parent
    repo_root = package_root.parent
    example_data_file = Path("retrievals/transmission/observations/HST/hst_example_clear_spec.txt")

    output_root = repo_root / "retrieval_outputs" / "leave_one_out_dynesty"
    output_root.mkdir(parents=True, exist_ok=True)

    observations = np.loadtxt(example_data_file.resolve())
    wavelengths = observations[:, 0]
    wavelength_boundaries = (0.95 * float(wavelengths.min()), 1.05 * float(wavelengths.max()))

    line_species_no_h2o = ("12C-1H4__HITEMP", "12C-16O2__UCL-4000", "C-O-NatAbund__HITEMP")
    line_species_with_h2o = ("1H2-16O__POKAZATEL", *line_species_no_h2o)

    retrieval_config_no_h2o = build_leave_one_out_config(
        "hst_example_clear_spec_no_h2o_v4", line_species_no_h2o, example_data_file, wavelength_boundaries
    )
    retrieval_config_h2o = build_leave_one_out_config(
        "hst_example_clear_spec_with_h2o_v4", line_species_with_h2o, example_data_file, wavelength_boundaries
    )

    retrieval_no_h2o = Retrieval(
        configuration=retrieval_config_no_h2o, output_directory=str(output_root),
        evaluate_sample_spectra=False, reference_data_name="HST", use_prt_plot_style=True,
    )
    retrieval_h2o = Retrieval(
        configuration=retrieval_config_h2o, output_directory=str(output_root),
        evaluate_sample_spectra=False, reference_data_name="HST", use_prt_plot_style=True,
    )
    for retrieval in (retrieval_no_h2o, retrieval_h2o):
        configure_plotter(retrieval)

    # Run both fits (set flags to False to skip).
    run_no_h2o_retrieval = True
    run_h2o_retrieval = True
    if run_no_h2o_retrieval:
        run_dynesty_retrieval(retrieval_no_h2o)
    if run_h2o_retrieval:
        run_dynesty_retrieval(retrieval_h2o)

    # Load posterior samples.
    _, _, samples_no_h2o, parameters_no_h2o = load_results_if_available(retrieval_no_h2o)
    _, _, samples_h2o, parameters_h2o = load_results_if_available(retrieval_h2o)

    # Leave-one-out diagnostics.
    delta_elpd = None
    pareto_k = None
    if samples_no_h2o is not None and samples_h2o is not None:
        samples_no_h2o = samples_no_h2o.T  # (num_samples, num_parameters + 1)
        samples_h2o = samples_h2o.T
        retrieval_no_h2o.get_log_likelihood_per_datapoint(
            samples_no_h2o[:, :-1], ret_name=retrieval_no_h2o.configuration.retrieval_name
        )
        retrieval_h2o.get_log_likelihood_per_datapoint(
            samples_h2o[:, :-1], ret_name=retrieval_h2o.configuration.retrieval_name
        )
        comparison_names = [
            retrieval_h2o.configuration.retrieval_name,
            retrieval_no_h2o.configuration.retrieval_name,
        ]
        elpd_tot, _elpd, pareto_k, delta_elpd = retrieval_h2o.get_elpd_per_datapoint(
            ret_name=comparison_names,
        )
        for data_name, totals in elpd_tot.items():
            print(f"{data_name}: total ELPD: {totals}")

    # Best-fit spectrum coloured by Pareto-k and by per-point delta-ELPD.
    if samples_h2o is not None and pareto_k is not None:
        retrieval_h2o.plot_spectra(
            samples_h2o.T, parameters_h2o, refresh=True, mode="bestfit",
            marker_color_type="pareto_k", marker_cmap=mpl.colormaps["plasma"],
            marker_label=r"Pareto $k$",
        )
    if samples_h2o is not None and delta_elpd is not None:
        retrieval_h2o.plot_spectra(
            samples_h2o.T, parameters_h2o, refresh=True, mode="bestfit",
            marker_color_type="delta_elpd", marker_cmap=mpl.colormaps["bwr"],
            marker_label=r"$\mathrm{ELPD}_{\mathrm{with\ H_2O}} - \mathrm{ELPD}_{\mathrm{no\ H_2O}}$",
        )
    plt.show()


if __name__ == "__main__":
    main()
