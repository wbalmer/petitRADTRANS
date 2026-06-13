"""Using the retrieval model functions directly.

Demonstrates calling the runtime-native model-function interface outside a full
retrieval loop to compute spectra and temperature profiles. Each model takes a
``ModelContext`` (Radtrans object + evaluation flags) and a ``PhysicalParams``
mapping, and returns a ``ModelResult``. Includes built-in emission and
transmission models plus a custom model example.

Equivalent to the ``retrieval_model_functions_tutorial`` documentation notebook.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt

import petitRADTRANS as prt
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.plotlib.style import set_petitradtrans_plot_style
from petitRADTRANS.retrieval.models import (
    molliere_2020_emission,
    guillot_emission,
    guillot_relative_emission,
    gradient_profile_emission,
    guillot_transmission,
    isothermal_transmission,
    calculate_transmission_spectrum_runtime,
)
from petitRADTRANS.retrieval.runtime import ModelContext, ModelInputs, ModelResult, PhysicalParams


PRESSURES = jnp.logspace(-6, 2, 100)


def build_emission_atmosphere():
    return prt.radtrans.Radtrans(
        pressures=PRESSURES,
        line_species=(
            "H2O__POKAZATEL.R300", "CO-NatAbund.R300", "CH4.R300",
            "CO2.R300", "H2S.R300", "NH3.R300",
        ),
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        cloud_species=(),
        wavelength_boundaries=(1, 5),
    )


def build_transmission_atmosphere():
    return prt.radtrans.Radtrans(
        pressures=PRESSURES,
        line_species=("H2O__POKAZATEL.R300", "CO-NatAbund.R300", "CH4.R300", "CO2.R300"),
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        cloud_species=(),
        wavelength_boundaries=(1, 5),
    )


def molliere_emission_example(atmosphere):
    """Mollière 2020 disequilibrium-chemistry emission for an HR 8799-like planet."""
    context = ModelContext(
        name="hr8799_emission", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False, return_contribution=False,
    )
    params = PhysicalParams.from_mapping({
        "distance_to_system": 41.3 * cst.pc,
        "mass": 5.0 * cst.m_jup,
        "planet_radius": 1.0 * cst.r_jup_mean,
        "T_int": 1500.0, "T3": 0.8, "T2": 0.4, "T1": 0.5, "log_delta": 0.65, "alpha": 1.70,
        "Fe/H": 1.0, "C/O": 0.7, "log_pquench": 2.5,
    })

    result = molliere_2020_emission(context, params, pt_plot_mode=False)
    pt_result = molliere_2020_emission(context, params, pt_plot_mode=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(result.wavelengths, result.spectrum, linewidth=2)
    axes[0].set_xlabel("Wavelength [micron]")
    axes[0].set_ylabel(r"F$_{\lambda}$ [W m$^{-2}$ $\mu$m$^{-1}$]")
    axes[0].set_title("Emission spectrum")
    axes[1].plot(pt_result.temperatures, pt_result.pressures, linewidth=2)
    axes[1].set_xlabel("Temperature [K]")
    axes[1].set_ylabel("Pressure [bar]")
    axes[1].set_ylim(1e2, 1e-6)
    axes[1].set_yscale("log")
    axes[1].set_title("T-P profile")
    fig.tight_layout()


def guillot_emission_example(atmosphere):
    """Guillot temperature profile emission."""
    context = ModelContext(
        name="guillot_emission_demo", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False,
    )
    params = PhysicalParams.from_mapping({
        "distance_to_system": 41.3 * cst.pc,
        "mass": 5.0 * cst.m_jup,
        "planet_radius": 1.0 * cst.r_jup_mean,
        "T_int": 700.0, "T_equ": 1300.0, "gamma": 0.4, "log_kappa_IR": -1.5,
        "Fe/H": 1.0, "C/O": 0.7, "log_pquench": 2.5,
    })
    result = guillot_emission(context, params)
    print(f"Guillot emission: computed {result.wavelengths.shape[0]} wavelength points.")


def guillot_relative_emission_example(atmosphere):
    """Planet-to-star flux ratio using the Guillot profile and a Sun-like host."""
    context = ModelContext(
        name="guillot_relative_emission_demo", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False,
    )
    params = PhysicalParams.from_mapping({
        "distance_to_system": 41.3 * cst.pc,
        "mass": 5.0 * cst.m_jup,
        "planet_radius": 1.0 * cst.r_jup_mean,
        "T_int": 700.0, "T_equ": 1300.0, "gamma": 0.4, "log_kappa_IR": -1.5,
        "Fe/H": 1.0, "C/O": 0.7, "log_pquench": 2.5,
        "star_effective_temperature": 5778.0,
        "stellar_radius": cst.r_sun,
    })
    result = guillot_relative_emission(context, params)
    print(f"Relative emission: peak contrast {float(result.spectrum.max()):.2e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.wavelengths, result.spectrum * 1e6, linewidth=2)
    ax.set_xlabel("Wavelength [micron]", fontsize=18)
    ax.set_ylabel(r"$F_\mathrm{planet} / F_\mathrm{star}$ [ppm]", fontsize=18)
    ax.set_title("Guillot relative emission (planet-to-star flux ratio)", fontsize=22)
    fig.tight_layout()


def gradient_emission_example(atmosphere):
    """Gradient (dT/d log P) profile emission from Zhang et al. (2023)."""
    n_layers = 5
    context = ModelContext(
        name="gradient_emission_demo", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False,
    )
    params = PhysicalParams.from_mapping({
        "distance_to_system": 41.3 * cst.pc,
        "mass": 5.0 * cst.m_jup,
        "planet_radius": 1.0 * cst.r_jup_mean,
        "N_layers": n_layers,
        "top_of_atmosphere_pressure": -3,
        "bottom_of_atmosphere_pressure": 2,
        "T_bottom": 2200.0,
        "PTslope_1": 0.2, "PTslope_2": 0.2, "PTslope_3": 0.2, "PTslope_4": 0.1, "PTslope_5": 0.0,
        "Fe/H": 0.5, "C/O": 0.55, "log_pquench": 1,
    })
    pt = gradient_profile_emission(context, params, pt_plot_mode=True)

    fig, ax = plt.subplots()
    ax.plot(pt.temperatures, pt.pressures, linewidth=2)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_ylim(1e2, 1e-6)
    ax.set_yscale("log")
    ax.set_title("Gradient profile T-P")
    fig.tight_layout()


def isothermal_transmission_example(atmosphere):
    """Isothermal transmission for a WASP-39b-like planet."""
    context = ModelContext(
        name="wasp39_transmission", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False, return_contribution=False,
    )
    params = PhysicalParams.from_mapping({
        "stellar_radius": 0.9324 * cst.r_sun,
        "planet_radius": 1.3 * cst.r_jup_mean,
        "log_g": 2.75,
        "reference_pressure": 100.0,
        "temperature": 750.0,
        "Fe/H": 1.0, "C/O": 0.55,
    })
    result = isothermal_transmission(context, params, pt_plot_mode=False)
    pt = isothermal_transmission(context, params, pt_plot_mode=True)
    _plot_transmission(result, pt, "Transmission spectrum (isothermal)")


def guillot_transmission_example(atmosphere):
    """Guillot transmission with haze and a gray cloud deck."""
    context = ModelContext(
        name="wasp39_transmission", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False, return_contribution=False,
    )
    params = PhysicalParams.from_mapping({
        "stellar_radius": 0.9324 * cst.r_sun,
        "planet_radius": 1.3 * cst.r_jup_mean,
        "log_g": 2.75,
        "reference_pressure": 100.0,
        "T_int": 750.0, "T_equ": 600.0, "log_kappa_IR": -1.0, "gamma": 1.0,
        "log_Pcloud": -2.0,
        "power_law_opacity_350nm": 1e-4,
        "power_law_opacity_coefficient": -3,
        "Fe/H": 1.0, "C/O": 0.55,
    })
    result = guillot_transmission(context, params, pt_plot_mode=False)
    pt = guillot_transmission(context, params, pt_plot_mode=True)
    _plot_transmission(result, pt, "Transmission spectrum (Guillot)")


def custom_isothermal_free_chemistry_transmission(model_context, physical_params, pt_plot_mode=False):
    """Isothermal transmission model with one free-chemistry species (H2O)."""
    p = model_context.radtrans.pressures / 1e6  # bar
    temperatures = jnp.full_like(p, physical_params["temperature"])

    if pt_plot_mode:
        return ModelResult(kind="pt_profile", pressures=p, temperatures=temperatures)

    x_h2o = jnp.power(10.0, physical_params["log_H2O"])
    remainder = jnp.clip(1.0 - x_h2o, 1e-12, 1.0)

    abundances = {
        species: jnp.full_like(p, x_h2o) if "H2O" in species or "POKAZATEL" in species else jnp.full_like(p, 0.0)
        for species in model_context.radtrans.line_species
    }
    abundances["H2"] = jnp.full_like(p, 0.74 * remainder)
    abundances["He"] = jnp.full_like(p, 0.26 * remainder)

    model_inputs = ModelInputs(
        pressures=p,
        temperatures=temperatures,
        abundances=abundances,
        mean_molar_masses=jnp.full_like(p, 2.33),
        gravity=jnp.power(10.0, physical_params["log_g"]),
        planet_radius=physical_params["planet_radius"],
        stellar_radius=physical_params["stellar_radius"],
        reference_pressure=physical_params["reference_pressure"],
        opaque_cloud_top_pressure=jnp.power(10.0, physical_params["log_Pcloud"]),
    )
    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


def custom_model_example(atmosphere):
    context = ModelContext(
        name="custom_isothermal", mode="evaluate", radtrans=atmosphere,
        adaptive_mesh_refinement=False,
    )
    params = PhysicalParams.from_mapping({
        "temperature": 900.0,
        "log_g": 2.75,
        "planet_radius": 1.3 * cst.r_jup_mean,
        "stellar_radius": 0.9324 * cst.r_sun,
        "reference_pressure": 100.0,
        "log_H2O": -3.0,
        "log_Pcloud": 0.0,
    })
    result = custom_isothermal_free_chemistry_transmission(context, params)

    fig, ax = plt.subplots()
    ax.plot(result.wavelengths, result.spectrum * 1e6, linewidth=2)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"$(R_p/R_*)^2$ [ppm]")
    ax.set_title("Custom isothermal free-chemistry model")
    fig.tight_layout()


def _plot_transmission(result, pt, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(result.wavelengths, result.spectrum * 1e6, linewidth=2)
    axes[0].set_xlabel("Wavelength [micron]")
    axes[0].set_ylabel(r"$(R_p/R_*)^2$ [ppm]")
    axes[0].set_title(title)
    axes[1].plot(pt.temperatures, pt.pressures, linewidth=2)
    axes[1].set_xlabel("Temperature [K]")
    axes[1].set_ylabel("Pressure [bar]")
    axes[1].set_ylim(1e2, 1e-6)
    axes[1].set_yscale("log")
    axes[1].set_title("T-P profile")
    fig.tight_layout()


def main():
    set_petitradtrans_plot_style()
    print(f"JAX devices: {jax.devices()}")

    emission_atmosphere = build_emission_atmosphere()
    molliere_emission_example(emission_atmosphere)
    guillot_emission_example(emission_atmosphere)
    guillot_relative_emission_example(emission_atmosphere)
    gradient_emission_example(emission_atmosphere)

    transmission_atmosphere = build_transmission_atmosphere()
    isothermal_transmission_example(transmission_atmosphere)
    guillot_transmission_example(transmission_atmosphere)
    custom_model_example(transmission_atmosphere)

    plt.show()


if __name__ == "__main__":
    main()
