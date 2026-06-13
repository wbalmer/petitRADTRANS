"""Exploring the legacy built-in retrieval models.

Demonstrates the legacy model-function interface (kept for pRT3 backwards
compatibility), which takes a ``Radtrans`` object and a dict of ``Parameter``
objects. The new interface (see retrieval_model_functions_tutorial.py) is
recommended for new work. Here the models are called directly to compute a
spectrum and T-P profile in isolation, outside any retrieval loop.

Equivalent to the ``legacy_retrieval_models`` documentation notebook.
"""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt

import petitRADTRANS as prt
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.plotlib.style import set_petitradtrans_plot_style
from petitRADTRANS.retrieval.models import molliere_2020_emission, guillot_transmission
from petitRADTRANS.retrieval.parameter import Parameter

# Line species binned to R=300 opacities, shared by both examples.
LINE_SPECIES = (
    "H2O__POKAZATEL.R300", "CO-NatAbund.R300", "CH4.R300", "CO2.R300", "HCN.R300",
    "FeH.R300", "H2S.R300", "NH3.R300", "PH3.R300", "Na.R300", "K.R300",
    "TiO.R300", "VO.R300", "SiO.R300",
)


def emission_example():
    """Mollière 2020 emission model for an HR 8799-like planet.

    Requires (dis)equilibrium chemistry and two of mass/planet_radius/log_g.
    """
    parameters = {
        "D_pl": Parameter("D_pl", False, value=10.0 * cst.pc),
        "mass": Parameter("mass", False, value=5 * cst.m_jup),
        "planet_radius": Parameter("planet_radius", False, value=1 * cst.r_jup_mean),
        "T_int": Parameter("T_int", False, value=1500.0),
        "T1": Parameter("T1", False, value=0.5),
        "T2": Parameter("T2", False, value=0.4),
        "T3": Parameter("T3", False, value=0.8),
        "log_delta": Parameter("log_delta", False, value=0.65),
        "alpha": Parameter("alpha", False, value=1.70),
        "Fe/H": Parameter("[Fe/H]", False, value=1.0),
        "C/O": Parameter("C/O", False, value=0.7),
        "log_pquench": Parameter("log_pquench", False, value=2.5),
        "sigma_lnorm": Parameter("sigma_lnorm", False, value=1.25),
        "fsed_MgSiO3(s)": Parameter("fsed_MgSiO3(s)", False, value=0.05),
        "fsed_Fe(s)": Parameter("fsed_Fe(s)", False, value=7.5),
        "log_kzz": Parameter("log_kzz", False, value=5),
        "eq_scaling_Fe(s)": Parameter("eq_scaling_Fe(s)", False, value=-0.3),
        "eq_scaling_MgSiO3(s)": Parameter("eq_scaling_MgSiO3(s)", False, value=-1.00),
    }

    atmosphere = prt.radtrans.Radtrans(
        pressures=jnp.logspace(-6, 2, 100),
        line_species=LINE_SPECIES,
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        cloud_species=("MgSiO3(s)_crystalline_000__DHS", "Fe(s)_crystalline_000__DHS"),
        wavelength_boundaries=(1, 5),
    )

    wavelength, model = molliere_2020_emission(atmosphere, parameters, adaptive_mesh_refinement=False, pt_plot_mode=False)
    pressure, temperature = molliere_2020_emission(atmosphere, parameters, adaptive_mesh_refinement=False, pt_plot_mode=True)

    fig, ax = plt.subplots()
    ax.plot(wavelength, model, linewidth=3)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"F$_{\lambda}$ [W/m$^{2}/\mu$m]")

    fig, ax = plt.subplots()
    ax.plot(temperature, pressure, linewidth=3)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_ylim(1e2, 1e-6)
    ax.set_yscale("log")


def transmission_example():
    """Guillot transmission model for a WASP-39b-like planet."""
    parameters = {
        "stellar_radius": Parameter("stellar_radius", False, value=0.9324 * cst.r_sun),
        "planet_radius": Parameter("planet_radius", False, value=1.3 * cst.r_jup_mean),
        "log_g": Parameter("log_g", False, value=2.75),
        "T_int": Parameter("T_int", False, value=750),
        "T_equ": Parameter("T_equ", False, value=600),
        "log_kappa_IR": Parameter("log_kappa_IR", False, value=-1.0),
        "gamma": Parameter("gamma", False, value=1.0),
        "haze_factor": Parameter("haze_factor", False, value=0.0),
        "power_law_opacity_350nm": Parameter("power_law_opacity_350nm", False, value=1e-4),
        "power_law_opacity_coefficient": Parameter("power_law_opacity_coefficient", False, value=-1.9),
        "log_Pcloud": Parameter("log_Pcloud", False, value=0),
        "Fe/H": Parameter("Fe/H", False, value=1.0),
        "C/O": Parameter("C/O", False, value=0.1),
    }

    atmosphere = prt.radtrans.Radtrans(
        pressures=jnp.logspace(-6, 2, 100),
        line_species=LINE_SPECIES,
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        cloud_species=(),
        wavelength_boundaries=(1, 5),
    )

    wavelength, model = guillot_transmission(atmosphere, parameters, amr=False, pt_plot_mode=False)
    pressure, temperature = guillot_transmission(atmosphere, parameters, amr=False, pt_plot_mode=True)

    fig, ax = plt.subplots()
    ax.plot(wavelength, model * 1e6, linewidth=3)
    ax.set_xlabel("Wavelength [micron]")
    ax.set_ylabel(r"F$_{p}$/F$_{\rm star}$ [ppm]")

    fig, ax = plt.subplots()
    ax.plot(temperature, pressure, linewidth=3)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_ylim(1e2, 1e-6)
    ax.set_yscale("log")


def main():
    set_petitradtrans_plot_style()
    emission_example()
    transmission_example()
    plt.show()


if __name__ == "__main__":
    main()
