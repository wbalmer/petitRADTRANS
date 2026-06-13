"""Emission spectra retrieval of HR 8799 e with NumPyro NUTS.

Same Mollière et al. (2020) forward model as the standard emission retrieval,
but using the differentiable ``molliere_2020_emission`` model with the
gradient-based NumPyro NUTS sampler. Priors are defined with TensorFlow
Probability distributions, the recommended JAX-native approach. NUTS adapts the
Hamiltonian trajectory during warmup but does not estimate the Bayesian
evidence (use a nested sampler for model comparison).

Equivalent to the ``retrieval_NUTS_emission_tutorial`` documentation notebook.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

from jax import config
config.update("jax_enable_x64", True)

import tensorflow_probability.substrates.jax as tfp

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import molliere_2020_emission

tfpd = tfp.distributions


def setup_configuration(path_to_data="./"):
    retrieval_config = RetrievalConfig(
        retrieval_name="HR8799e_example",
        run_mode="retrieval",
        sampler_type="numpyro_nuts",
        adaptive_mesh_refinement=True,
        scattering_in_emission=True,
    )

    # GRAVITY spectroscopy (Mollière et al. 2020 also used SPHERE and GPI data).
    retrieval_config.add_data(
        name="GRAVITY",
        path_to_observations=f"{path_to_data}/retrievals/emission/observations/HR8799e_Spectra.fits",
        data_resolution=500,
        model_resolution=1000,
        model_generating_function=molliere_2020_emission,
    )
    retrieval_config.add_photometry(
        path_to_data + "retrievals/emission/observations/hr8799e_photometry.txt",
        molliere_2020_emission,
        model_resolution=40,
    )

    # Parameters and priors. For NUTS, priors are TFP distributions.
    retrieval_config.add_parameter(name="system_distance", is_free_parameter=False, value=41.2925 * cst.pc)
    retrieval_config.add_parameter("log_g", True, distribution=tfpd.Uniform(low=2.0, high=5.5))
    retrieval_config.add_parameter(
        "planet_radius", True, distribution=tfpd.Uniform(low=0.7 * cst.r_jup_mean, high=2.0 * cst.r_jup_mean)
    )
    retrieval_config.add_parameter("T_int", True, distribution=tfpd.Uniform(low=300.0, high=2300.0), value=0.0)

    # Spline temperature structure (T1 < T2 < T3 enforced in the model function).
    retrieval_config.add_parameter("T3", True, distribution=tfpd.Uniform(low=0.0, high=1.0), value=0.0)
    retrieval_config.add_parameter("T2", True, distribution=tfpd.Uniform(low=0.0, high=1.0), value=0.0)
    retrieval_config.add_parameter("T1", True, distribution=tfpd.Uniform(low=0.0, high=1.0))

    # Optical depth model: tau = delta * press_cgs**alpha.
    retrieval_config.add_parameter("alpha", True, distribution=tfpd.Uniform(low=1.0, high=2.0))
    retrieval_config.add_parameter("log_delta", True, distribution=tfpd.Uniform(low=0.0, high=1.0))

    # Chemistry.
    retrieval_config.add_parameter("log_pquench", True, distribution=tfpd.Uniform(low=-6.0, high=3.0))
    retrieval_config.add_parameter("Fe/H", True, distribution=tfpd.Uniform(low=-1.5, high=1.5))
    retrieval_config.add_parameter("C/O", True, distribution=tfpd.Uniform(low=0.1, high=1.6))

    # Clouds (Ackermann-Marley 2001 model).
    retrieval_config.add_parameter("sigma_lnorm", True, distribution=tfpd.Uniform(low=1.05, high=3.0))
    retrieval_config.add_parameter("log_kzz", True, distribution=tfpd.Uniform(low=5.0, high=13.0))
    retrieval_config.add_parameter("f_sed", True, distribution=tfpd.Uniform(low=1.0, high=11.0))

    # Opacities (equilibrium chemistry).
    retrieval_config.set_rayleigh_species(("H2", "He"))
    retrieval_config.set_continuum_opacities(("H2--H2", "H2--He"))
    retrieval_config.set_line_species(
        (
            "H2O__POKAZATEL", "CO-NatAbund", "CH4", "CO2", "HCN", "FeH", "H2S",
            "NH3", "PH3", "Na__NewAllard", "K__Allard", "TiO", "VO", "SiO",
        ),
        use_equilibrium_chemistry=True,
    )
    retrieval_config.add_cloud_species(
        "Fe(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=True,
        equilbrium_mass_fraction_scaling_factor=(-3.5, 1.0),
    )
    retrieval_config.add_cloud_species(
        "MgSiO3(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=True,
        equilbrium_mass_fraction_scaling_factor=(-3.5, 1.0),
    )

    # Corner plot setup.
    retrieval_config.parameters["planet_radius"].plot_in_corner = True
    retrieval_config.parameters["planet_radius"].corner_label = r"$R_{\rm P}$ ($\rm R_{Jup}$)"
    retrieval_config.parameters["planet_radius"].corner_transform = lambda x: x / cst.r_jup_mean
    retrieval_config.parameters["log_g"].plot_in_corner = True
    retrieval_config.parameters["log_g"].corner_ranges = [2.0, 5.0]
    retrieval_config.parameters["log_g"].corner_label = "log g"
    retrieval_config.parameters["fsed"].plot_in_corner = True
    retrieval_config.parameters["log_kzz"].plot_in_corner = True
    retrieval_config.parameters["log_kzz"].corner_label = "log Kzz"
    retrieval_config.parameters["C/O"].plot_in_corner = True
    retrieval_config.parameters["Fe/H"].plot_in_corner = True
    retrieval_config.parameters["log_pquench"].plot_in_corner = True
    retrieval_config.parameters["log_pquench"].corner_label = "log pquench"
    for spec in retrieval_config.cloud_species:
        cname = spec.split("_")[0]
        retrieval_config.parameters["eq_scaling_" + cname].plot_in_corner = True
        retrieval_config.parameters["eq_scaling_" + cname].corner_label = cname

    return retrieval_config


def configure_retrieval_plotter(retrieval):
    retrieval.plotter.spec_xlabel = "Wavelength [micron]"
    retrieval.plotter.spec_ylabel = "Flux [W/m2/micron]"
    retrieval.plotter.y_axis_scaling = 1.0
    retrieval.plotter.xscale = "log"
    retrieval.plotter.yscale = "linear"
    retrieval.plotter.resolution = 100.0
    retrieval.plotter.nsample = 100
    retrieval.plotter.reference_data_name = "GRAVITY"
    retrieval.plotter.temp_limits = [150, 3000]
    retrieval.plotter.press_limits = [1e2, 1e-5]


def main():
    retrieval_config = setup_configuration()

    retrieval = Retrieval(retrieval_config, output_directory="./", evaluate_sample_spectra=False)
    configure_retrieval_plotter(retrieval)

    # NUTS controls: warmup length, posterior sample count, number of chains.
    # Reduce num_chains if memory-limited. Set to True to attempt a run.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(
            num_warmup=1000,
            num_samples=2000,
            num_chains=4,
            chain_method="vectorized",
            target_acceptance_rate=0.9,
            progress_bar=True,
            use_jit=True,
            seed=12345,  # remove or randomize for a production retrieval
        )
        retrieval.plot_all(contribution=True)


if __name__ == "__main__":
    main()
