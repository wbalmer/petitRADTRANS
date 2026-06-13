"""Transmission retrieval with multiple datasets.

JWST observations of WASP-39 b (NIRSpec PRISM + NIRISS SOSS orders 1 & 2) are
fit simultaneously with a single shared Radtrans object via external references.
Uses a Guillot temperature profile, free chemistry, patchy grey clouds, an
enhanced power-law haze, and the pymultinest sampler. Also demonstrates
extracting retrieval outputs for manual plotting.

Equivalent to the ``retrieval_multiple_data`` documentation notebook.
"""
import copy as cp

import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.plotlib.style import set_petitradtrans_plot_style
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import guillot_transmission
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.utils import gaussian_prior


def setup_configuration(path_to_data="./"):
    retrieval_config = RetrievalConfig(
        retrieval_name="WASP39b_Guillot_FreeChem_PatchyGreyHaze",
        run_mode="retrieval",
        pressures=jnp.logspace(-8, 3, 100),
        adaptive_mesh_refinement=False,
        scattering_in_emission=False,
        sampler_type="pymultinest",
    )

    # Data. PRISM covers the full NIRISS SOSS range, so it serves as the external
    # Radtrans reference for both SOSS orders: the model is computed once.
    transmission_directory = "retrievals/transmission/observations/"
    retrieval_config.add_data(
        "JWST/NIRSPEC/PRISM",
        f"{path_to_data}{transmission_directory}JWST/WASP39b_NIRSpec_PRISM_bins_scale1.dat",
        data_resolution=100,
        model_resolution=400,
        fit_flux_offset=True,
        model_generating_function=guillot_transmission,
    )
    retrieval_config.add_data(
        "JWST/NIRISSSOSS/O1",
        f"{path_to_data}{transmission_directory}JWST/WASP39b_NIRISS_SOSS_Order1_R100.dat",
        data_resolution=700,
        model_resolution=400,
        model_generating_function=guillot_transmission,
        external_radtrans_reference="JWST/NIRSPEC/PRISM",
    )
    retrieval_config.add_data(
        "JWST/NIRISSSOSS/O2",
        f"{path_to_data}{transmission_directory}JWST/WASP39b_NIRISS_SOSS_Order2_R100.dat",
        data_resolution=700,
        model_resolution=400,
        fit_flux_offset=True,
        model_generating_function=guillot_transmission,
        external_radtrans_reference="JWST/NIRSPEC/PRISM",
    )

    # WASP-39 system parameters.
    retrieval_config.add_parameter(name="stellar_radius", is_free_parameter=False, value=0.9324 * cst.r_sun)
    retrieval_config.add_parameter("reference_pressure", False, value=0.01)
    retrieval_config.add_parameter("log_g", True, transform_prior_cube_coordinate=lambda x: 2.0 + 3.5 * x)
    retrieval_config.add_parameter(
        "planet_radius", True,
        transform_prior_cube_coordinate=lambda x: 0.8 * cst.r_jup_mean + (x * 0.8 * cst.r_jup_mean),
    )

    # Guillot 2010 temperature profile.
    retrieval_config.add_parameter("T_int", True, transform_prior_cube_coordinate=lambda x: 100 + 3500 * x)
    retrieval_config.add_parameter("T_equ", True, transform_prior_cube_coordinate=lambda x: 100 + 3500 * x)
    retrieval_config.add_parameter("gamma", True, transform_prior_cube_coordinate=lambda x: 10 ** (-((x / 2) ** 2) / 2))
    retrieval_config.add_parameter("log_kappa_IR", True, transform_prior_cube_coordinate=lambda x: -4.0 + 6.0 * x)

    # Grey cloud deck and enhanced haze scattering slope.
    retrieval_config.add_parameter("log_Pcloud", True, transform_prior_cube_coordinate=lambda x: -8 + 11 * x)
    retrieval_config.add_parameter("haze_factor", True, transform_prior_cube_coordinate=lambda x: -4 + 14 * x)
    retrieval_config.add_parameter(
        "power_law_opacity_350nm", True, transform_prior_cube_coordinate=lambda x: 10 ** (-20 + 40 * x)
    )
    retrieval_config.add_parameter(
        "power_law_opacity_coefficient", True, transform_prior_cube_coordinate=lambda x: -20 + 22 * x
    )
    retrieval_config.add_parameter("cloud_fraction", True, transform_prior_cube_coordinate=lambda x: x)

    # Data offsets.
    retrieval_config.add_parameter(
        "JWST/NIRSPEC/PRISM_offset", True, transform_prior_cube_coordinate=lambda x: gaussian_prior(x, 0, 1e-4)
    )
    retrieval_config.add_parameter(
        "JWST/NIRISSSOSS/O2_offset", True, transform_prior_cube_coordinate=lambda x: gaussian_prior(x, 0, 1e-4)
    )

    # Opacities (free chemistry).
    retrieval_config.set_rayleigh_species(["H2", "He"])
    retrieval_config.set_continuum_opacities(["H2--H2", "H2--He"])
    retrieval_config.set_line_species(
        ["H2O__POKAZATEL", "CO-NatAbund", "CO2", "CH4__MM", "SO2"],
        use_equilibrium_chemistry=False,
        abund_lim=(-8.0, 0.0),
    )

    # Corner plot labels.
    corner_labels = {
        "log_g": r"$\log g$",
        "planet_radius": r"R$_{pl}$",
        "T_int": r"T$_{int}$",
        "T_equ": r"T$_{equ}$",
        "gamma": r"$\gamma$",
        "log_kappa_IR": r"$\log\kappa_{IR}$",
        "log_Pcloud": r"$\log P_{cl}$",
        "haze_factor": r"$\log\kappa_{haze}$",
        "power_law_opacity_350nm": r"$\log\gamma_{350nm}$",
        "power_law_opacity_coefficient": r"$\log\gamma_{350nm}$",
        "cloud_fraction": r"$f_{cloud}$",
        "JWST/NIRSPEC/PRISM_offset": r"PRISM offset",
        "JWST/NIRISSSOSS/O2_offset": r"SOSS O2 offset",
        "H2O__POKAZATEL": r"$\log$ H$_{2}$O",
        "CO-NatAbund": r"$\log$ CO",
        "CO2": r"$\log$ CO$_{2}$",
        "CH4__MM": r"$\log$ CH$_{4}$",
        "SO2": r"$\log$ SO$_{2}$",
    }
    for name, label in corner_labels.items():
        retrieval_config.parameters[name].plot_in_corner = True
        retrieval_config.parameters[name].corner_label = label
    retrieval_config.parameters["planet_radius"].corner_transform = lambda x: x / cst.r_jup_mean

    return retrieval_config


def configure_retrieval_plotter(retrieval):
    retrieval.plotter.spec_xlabel = "Wavelength [micron]"
    retrieval.plotter.spec_ylabel = r"$(R_{\rm P}/R_*)^2$ [ppm]"
    retrieval.plotter.y_axis_scaling = 1e6
    retrieval.plotter.xscale = "linear"
    retrieval.plotter.yscale = "linear"
    retrieval.plotter.nsample = 10
    retrieval.plotter.reference_data_name = "JWST/NIRSPEC/PRISM"
    retrieval.plotter.temp_limits = [150, 3000]
    retrieval.plotter.press_limits = [1e1, 1e-7]


def post_process(retrieval, output_dir):
    """Load results and demonstrate manual plotting of best-fit outputs."""
    sample_dict, parameter_dict = retrieval.get_samples(
        ultranest=False, output_directory=retrieval.output_directory
    )
    samples_use = cp.copy(sample_dict[retrieval.configuration.retrieval_name])
    parameters_read = parameter_dict[retrieval.configuration.retrieval_name]

    # The posterior can also be loaded directly with pandas.
    columns = cp.copy(parameters_read)
    columns.append("log likelihood")
    pd.read_csv(
        f"{output_dir}out_PMN/{retrieval.configuration.retrieval_name}_post_equal_weights.dat",
        delim_whitespace=True,
        index_col=False,
        names=columns,
    )

    # Best fit model and parameters.
    _, best_fit_index = retrieval.get_best_fit_likelihood(samples_use)
    best_fit_wavelengths, best_fit_spectrum = retrieval.get_best_fit_model(
        sample_dict[retrieval.configuration.retrieval_name][:-1, best_fit_index],
        parameters_read,
        prt_reference=None,
        refresh=True,
        contribution=False,
        mode="bestfit",
    )
    best_fit_parameters = retrieval.best_fit_parameters
    for key, param in best_fit_parameters.items():
        print(key, f"{param.value:.2f}")

    # Best fit spectrum vs data.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(best_fit_wavelengths, best_fit_spectrum, label="Best Fit Model", linewidth=2)
    for name, data in retrieval.configuration.data.items():
        ax.errorbar(
            data.wavelengths, data.spectrum - data.offset, yerr=data.uncertainties,
            xerr=data.wavelength_bin_widths, fmt="o", label=name,
        )
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"$(R_{\rm P}/R_*)^2$")
    ax.legend(fontsize=16, frameon=False)

    # Convolved and binned model per instrument.
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in retrieval.configuration.data.items():
        label = "Convolved and binned model" if name == retrieval.plotter.reference_data_name else None
        ax.plot(
            retrieval.best_fit_spectra[name][0], retrieval.best_fit_spectra[name][1],
            label=label, color="k", linewidth=4,
        )
        ax.errorbar(
            data.wavelengths, data.spectrum, yerr=data.uncertainties,
            xerr=data.wavelength_bin_widths, fmt="o", label=name,
        )
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"$(R_{\rm P}/R_*)^2$")
    ax.legend(fontsize=16, frameon=False)

    # Best fit temperature profile.
    data_to_use = retrieval.configuration.data["JWST/NIRSPEC/PRISM"]
    best_fit_pressure, best_fit_temperature = data_to_use.model_generating_function(
        data_to_use.radtrans_object,
        best_fit_parameters,
        adaptive_mesh_refinement=retrieval.configuration.adaptive_mesh_refinement,
        pt_plot_mode=True,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_fit_temperature, best_fit_pressure, linewidth=4)
    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e-7)
    ax.set_xlim(150, 3000)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")

    # Temperature confidence interval from the saved grid.
    temperatures = np.load(
        f"{output_dir}evaluate_{retrieval.configuration.retrieval_name}/"
        f"{retrieval.configuration.retrieval_name}_temps.npy"
    )
    lower, median, upper = np.quantile(temperatures, [0.005, 0.5, 0.995], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_betweenx(best_fit_pressure, lower, upper, alpha=0.3, color="C0")
    ax.plot(lower, best_fit_pressure, color="C0", linewidth=1)
    ax.plot(upper, best_fit_pressure, color="C0", linewidth=1)
    ax.plot(median, best_fit_pressure, color="C0", linewidth=3)
    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e-7)
    ax.set_xlim(150, 3000)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")

    # Save various outputs for portability between machines.
    retrieval.save_volume_mixing_ratios(sample_dict, parameter_dict)
    retrieval.save_mass_fractions(sample_dict, parameter_dict)
    retrieval.save_best_fit_outputs(best_fit_parameters)
    retrieval.save_configuration()

    # Request additional outputs: opacities, optical depths, contribution functions.
    retrieval.configuration.parameters["return_opacities"] = Parameter("return_opacities", False, True)
    retrieval.configuration.parameters["return_cloud_contribution"] = Parameter("return_cloud_contribution", False, True)
    retrieval.configuration.parameters["return_optical_depths"] = Parameter("return_optical_depths", False, True)
    retrieval.configuration.parameters["return_contribution"] = Parameter("return_contribution", False, True)
    _, _, outputs = data_to_use.model_generating_function(
        data_to_use.radtrans_object,
        retrieval.configuration.parameters,
        adaptive_mesh_refinement=retrieval.configuration.adaptive_mesh_refinement,
        pt_plot_mode=False,
    )
    for key in outputs.keys():
        print(key)


def main():
    set_petitradtrans_plot_style()

    retrieval_config = setup_configuration()

    output_dir = "retrievals/transmission/runs/"
    retrieval = Retrieval(
        retrieval_config,
        output_directory=output_dir,
        evaluate_sample_spectra=False,
        use_prt_plot_style=True,
    )
    configure_retrieval_plotter(retrieval)

    # This retrieval is expensive (days on many cores). Set to True to attempt it.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(
            sampling_efficiency=0.05,
            const_efficiency_mode=True,
            n_live_points=400,
            resume=True,
        )
        retrieval.plot_all(contribution=False, mode="bestfit")
        post_process(retrieval, output_dir)

    plt.show()


if __name__ == "__main__":
    main()
