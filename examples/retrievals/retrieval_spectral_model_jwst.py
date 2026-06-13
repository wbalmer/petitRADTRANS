"""Low-resolution JWST retrieval workflow with SpectralModel.

A transmission retrieval of WASP-39 b JWST data using the SpectralModel
interface (isothermal profile, constant mass fractions, patchy grey cloud).
Demonstrates a single-dataset retrieval (NIRSpec PRISM) and a multi-dataset
retrieval that adds two NIRISS SOSS orders, shares opacities between forward
models, and fits per-instrument spectrum offsets.

Equivalent to the ``retrieval_spectral_model_jwst`` documentation notebook.
"""
import copy
import os

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.plotlib import plot_result_corner
from petitRADTRANS.retrieval.retrieval import Retrieval
from petitRADTRANS.spectral_model import SpectralModel

PATH_TO_DATA = os.path.join("./", "retrievals", "transmission", "observations", "JWST")


def base_retrieved_parameters():
    return {
        "temperature": {
            "prior_parameters": [100, 4000], "prior_type": "uniform",
            "figure_title": r"T", "figure_label": r"T (K)",
        },
        "planet_radius": {
            "prior_parameters": np.array([0.8, 1.6]) * cst.r_jup, "prior_type": "uniform",
            "figure_title": r"$R_p$", "figure_label": r"$R_p$)",
        },
        "CH4.R300": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[CH4]", "figure_label": r"$\log_{10}$(MMR) CH4",
        },
        "CO-NatAbund.R300": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[CO]", "figure_label": r"$\log_{10}$(MMR) CO",
        },
        "CO2.R300": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[CO$_2$]", "figure_label": r"$\log_{10}$(MMR) CO$_2$",
        },
        "H2O__POKAZATEL.R300": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[H$_2$O]", "figure_label": r"$\log_{10}$(MMR) H$_2$O",
        },
        "SO2.R300": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[SO$_2$]", "figure_label": r"$\log_{10}$(MMR) SO$_2$",
        },
        "log10_opaque_cloud_top_pressure": {
            "prior_parameters": [-10, 2], "prior_type": "uniform",
            "figure_title": r"[$P_c$]", "figure_label": r"$\log_{10}(P_c)$ ([Pa])",
            "figure_offset": 5,  # [bar] to [Pa]
        },
        "cloud_fraction": {
            "prior_parameters": [0, 1], "prior_type": "uniform",
            "figure_title": r"$c_f$", "figure_label": r"$c_f$",
        },
    }


def build_nirspec_forward_model(planet, nirspec_wavelengths, nirspec_wavelength_bin_widths):
    """Isothermal, constant-abundance transmission model binned to R=300 opacities."""
    return SpectralModel(
        pressures=jnp.logspace(-6, 2, 100),
        line_species=("CH4.R300", "CO-NatAbund.R300", "CO2.R300", "H2O__POKAZATEL.R300", "SO2.R300"),
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        planet_radius=planet.radius,
        reference_gravity=planet.reference_gravity,
        reference_pressure=1e-2,
        star_radius=planet.star_radius,
        temperature_profile_mode="isothermal",
        temperature=planet.equilibrium_temperature,
        opaque_cloud_top_pressure=1e2,
        cloud_fraction=1.0,
        use_equilibrium_chemistry=False,
        imposed_mass_fractions={
            "CH4.R300": 1e-6,
            "CO-NatAbund.R300": 1e-2,
            "CO2.R300": 1e-4,
            "H2O__POKAZATEL.R300": 1e-3,
            "SO2.R300": 1e-4,
        },
        filling_species={"H2": 37, "He": 12},
        rebinned_wavelengths=nirspec_wavelengths * 1e-4,  # um to cm
        bin_widths=nirspec_wavelength_bin_widths * 1e-4,
        rebin_range_margin_power=1,
        convolve_resolving_power=100,
    )


def run_single_dataset(planet):
    """Retrieve the NIRSpec PRISM dataset alone."""
    nirspec_data_file = os.path.join(PATH_TO_DATA, "WASP39b_nirspec_prism.txt")
    nirspec_wavelengths, nirspec_wavelength_bin_widths, nirspec_data, nirspec_uncertainties = np.loadtxt(
        nirspec_data_file
    ).T

    forward_model_nirspec = build_nirspec_forward_model(
        planet, nirspec_wavelengths, nirspec_wavelength_bin_widths
    )
    retrieved_parameters = base_retrieved_parameters()

    # init_data combines the data with its forward model. SpectralModel uses 1 - (Rp/R*)^2.
    # Uncertainties are artificially inflated x20 here to make the example run faster.
    data = {
        "NIRSpec": forward_model_nirspec.init_data(
            data_spectrum=1 - nirspec_data,
            data_wavelengths=nirspec_wavelengths,
            data_uncertainties=nirspec_uncertainties * 20,
            data_name="NIRSpec",
            retrieved_parameters=retrieved_parameters,
            mode="transmission",
            scale=True,
            convolve=True,
            rebin=True,
        )
    }

    retrieval_name = "spectral_model_jwst_example_1_dataset"
    retrieval_directory = os.path.join(".", "results", retrieval_name)
    retrieval = Retrieval.from_data(
        data=data,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        output_directory=retrieval_directory,
        run_mode="retrieval",
    )

    # Few live points for speed; use a few hundred for real runs. ~1 hour here.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(n_live_points=15, resume=False, seed=12345)

    plot_result_corner(
        retrieval_directory=[retrieval_directory],
        retrieved_parameters=retrieved_parameters,
        figure_name="corner_plot",
        label_kwargs={"fontsize": 16},
        title_kwargs={"fontsize": 14},
        figure_font_size=12,
        smooth=1,
        save=True,
        figure_directory=retrieval_directory,
        image_format="png",
    )

    # Best fit model vs data.
    _, best_fit_spectra_nirspec, _ = retrieval.calculate_forward_model(parameters="best fit", data="NIRSpec")
    fig, axes = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
    axes[0].plot(nirspec_wavelengths, 1 - best_fit_spectra_nirspec[0, 0], ls="-", color="r", label="Best-fit spectrum")
    axes[0].errorbar(
        nirspec_wavelengths, nirspec_data, xerr=nirspec_wavelength_bin_widths,
        yerr=nirspec_uncertainties, ls="", label="NIRSpec", color="k",
    )
    axes[1].plot(
        nirspec_wavelengths, (1 - best_fit_spectra_nirspec[0, 0] - nirspec_data) / nirspec_uncertainties,
        ls="-", color="r",
    )
    axes[0].set_title("Comparison with best fit model")
    axes[0].set_ylabel("Spectrum [A.U.]")
    axes[1].set_xlabel("Wavelength [microns]")
    axes[1].set_ylabel(r"Residuals ($\sigma$)")
    fig.legend()

    return forward_model_nirspec


def run_multiple_datasets(planet, forward_model_nirspec):
    """Retrieve NIRSpec PRISM plus two NIRISS SOSS orders with per-instrument offsets."""
    nirspec_data_file = os.path.join(PATH_TO_DATA, "WASP39b_nirspec_prism.txt")
    nirspec_wavelengths, _, nirspec_data, nirspec_uncertainties = np.loadtxt(nirspec_data_file).T

    niriss_wl_1, niriss_bw_1, niriss_data_1, niriss_unc_1 = np.loadtxt(
        os.path.join(PATH_TO_DATA, "WASP39b_niriss_soss1.txt")
    ).T
    niriss_wl_2, niriss_bw_2, niriss_data_2, niriss_unc_2 = np.loadtxt(
        os.path.join(PATH_TO_DATA, "WASP39b_niriss_soss2.txt")
    ).T

    # The NIRISS forward models differ from NIRSpec only in wavelength range. Copy the
    # NIRSpec parameters and adjust. from_radtrans shares opacities (saves memory, costs time).
    niriss_1_params = copy.deepcopy(forward_model_nirspec.model_parameters)
    niriss_1_params["rebinned_wavelengths"] = niriss_wl_1 * 1e-4
    niriss_1_params["bin_widths"] = niriss_bw_1 * 1e-4
    niriss_2_params = copy.deepcopy(forward_model_nirspec.model_parameters)
    niriss_2_params["rebinned_wavelengths"] = niriss_wl_2 * 1e-4
    niriss_2_params["bin_widths"] = niriss_bw_2 * 1e-4

    forward_model_niriss_1 = SpectralModel.from_radtrans(radtrans=forward_model_nirspec, **niriss_1_params)
    forward_model_niriss_2 = SpectralModel.from_radtrans(radtrans=forward_model_nirspec, **niriss_2_params)

    # Special parameter _spectrum_offset adjusts for instrumental flux imprecisions.
    retrieved_parameters = base_retrieved_parameters()
    retrieved_parameters.update({
        "_spectrum_offset_niriss_1": {
            "prior_parameters": [0, 1e-4], "prior_type": "gaussian",
            "figure_title": r"Offset 1", "figure_label": r"Offset NIRISS 1",
        },
        "_spectrum_offset_niriss_2": {
            "prior_parameters": [0, 1e-4], "prior_type": "gaussian",
            "figure_title": r"Offset 2", "figure_label": r"Offset NIRISS 2",
        },
    })

    # fixed_special_parameters indicates which offsets must NOT apply to each dataset.
    data = {
        "NIRSpec": forward_model_nirspec.init_data(
            data_spectrum=1 - nirspec_data,
            data_wavelengths=nirspec_wavelengths,
            data_uncertainties=nirspec_uncertainties * 20,
            data_name="NIRSpec",
            retrieved_parameters=retrieved_parameters,
            fixed_special_parameters={"_spectrum_offset_niriss_1": 0.0, "_spectrum_offset_niriss_2": 0.0},
            mode="transmission", scale=True, convolve=True, rebin=True,
        ),
        "NIRISS1": forward_model_niriss_1.init_data(
            data_spectrum=1 - niriss_data_1,
            data_wavelengths=niriss_wl_1,
            data_uncertainties=niriss_unc_1 * 20,
            data_name="NIRISS1",
            retrieved_parameters=retrieved_parameters,
            fixed_special_parameters={"_spectrum_offset_niriss_2": 0.0},
            mode="transmission", scale=True, convolve=True, rebin=True,
        ),
        "NIRISS2": forward_model_niriss_2.init_data(
            data_spectrum=1 - niriss_data_2,
            data_wavelengths=niriss_wl_2,
            data_uncertainties=niriss_unc_2 * 20,
            data_name="NIRISS2",
            retrieved_parameters=retrieved_parameters,
            fixed_special_parameters={"_spectrum_offset_niriss_1": 0.0},
            mode="transmission", scale=True, convolve=True, rebin=True,
        ),
    }

    retrieval_name = "spectral_model_jwst_example_3_datasets"
    retrieval_directory = os.path.join(".", "results", retrieval_name)
    retrieval = Retrieval.from_data(
        data=data,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        output_directory=retrieval_directory,
        run_mode="retrieval",
    )

    run_retrieval = False
    if run_retrieval:
        retrieval.run(n_live_points=15, resume=False, max_iters=30, n_iter_before_update=1, seed=12345)

    plot_result_corner(
        retrieval_directory=[retrieval_directory],
        retrieved_parameters=retrieved_parameters,
        figure_name="corner_plot",
        label_kwargs={"fontsize": 16},
        title_kwargs={"fontsize": 14},
        figure_font_size=12,
        smooth=1,
        save=True,
        figure_directory=retrieval_directory,
        image_format="png",
    )

    best_fit_parameters, max_likelihood, chi2 = retrieval.get_best_fit_parameters(return_max_likelihood=True)
    data_size = nirspec_data.size + niriss_data_1.size + niriss_data_2.size
    chi2 *= 20
    print(
        f"Best fit reduced chi2 (with 20x uncertainties): "
        f"{chi2 / (data_size - len(retrieved_parameters)):.2e} (Log(L)/20 = {max_likelihood:.2f})"
    )
    print("\nBest fit parameters:")
    for parameter_name, value in best_fit_parameters.items():
        print(f"    - '{parameter_name}': {value:.5f}")

    # Best fit models, with offsets removed from the models and applied to the data.
    _, best_fit_nirspec, _ = retrieval.calculate_forward_model(parameters="best fit", data="NIRSpec")
    _, best_fit_niriss_1, _ = retrieval.calculate_forward_model(parameters="best fit", data="NIRISS1")
    _, best_fit_niriss_2, _ = retrieval.calculate_forward_model(parameters="best fit", data="NIRISS2")
    best_fit_niriss_1 -= best_fit_parameters["_spectrum_offset_niriss_1"]
    best_fit_niriss_2 -= best_fit_parameters["_spectrum_offset_niriss_2"]
    corrected_data_1 = niriss_data_1 + best_fit_parameters["_spectrum_offset_niriss_1"]
    corrected_data_2 = niriss_data_2 + best_fit_parameters["_spectrum_offset_niriss_2"]

    fig, axes = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
    axes[0].plot(nirspec_wavelengths, 1 - best_fit_nirspec[0, 0], ls="", marker="x", color="r")
    axes[0].plot(niriss_wl_1, 1 - best_fit_niriss_1[0, 0], ls="-", color="r")
    axes[0].plot(niriss_wl_2, 1 - best_fit_niriss_2[0, 0], ls="-", color="r", label="Best-fit spectrum")
    axes[0].errorbar(nirspec_wavelengths, nirspec_data, yerr=nirspec_uncertainties, ls="", label="NIRSpec", color="k")
    axes[0].errorbar(niriss_wl_1, corrected_data_1, yerr=niriss_unc_1, ls="", color="C2", label="NIRISS1 (corrected)")
    axes[0].errorbar(niriss_wl_2, corrected_data_2, yerr=niriss_unc_2, ls="", color="C0", label="NIRISS2 (corrected)")
    axes[1].plot(
        nirspec_wavelengths, (1 - best_fit_nirspec[0, 0] - nirspec_data) / nirspec_uncertainties,
        ls="", marker="x", color="r",
    )
    axes[1].plot(niriss_wl_1, (1 - best_fit_niriss_1[0, 0] - corrected_data_1) / niriss_unc_1, ls="-", color="r")
    axes[1].plot(niriss_wl_2, (1 - best_fit_niriss_2[0, 0] - corrected_data_2) / niriss_unc_2, ls="-", color="r")
    axes[0].set_title("Comparison with best fit model")
    axes[0].set_ylabel("Spectrum [A.U.]")
    axes[1].set_xlabel("Wavelength [microns]")
    axes[1].set_ylabel(r"Residuals ($\sigma$)")
    fig.legend()


def main():
    # Planet is a convenient source of system parameters, but not required.
    planet = Planet.get("WASP-39 b")

    forward_model_nirspec = run_single_dataset(planet)
    run_multiple_datasets(planet, forward_model_nirspec)

    plt.show()


if __name__ == "__main__":
    main()
