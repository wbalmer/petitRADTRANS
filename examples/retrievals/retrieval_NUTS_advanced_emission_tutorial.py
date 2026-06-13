"""Warm-started, multi-chain HR 8799 e emission retrieval with NumPyro NUTS.

Builds on the NUTS emission tutorial with two advanced features of pRT's
gradient-based path:

1. Optimal-estimation warm-starts: an optax optimisation locates the MAP point
   and the Hessian there preconditions the NUTS mass matrix (Rodgers/Laplace).
2. Multi-chain orchestration via ``Retrieval.run_multichain``.

The configuration is wrapped in a top-level ``build_config`` function, the
pattern required by ``run_multichain`` (for ``chain_execution="processes"`` the
builder must be importable and picklable by reference).

Equivalent to the ``retrieval_NUTS_advanced_emission_tutorial`` documentation notebook.
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


def build_config():
    """Build the HR 8799 e retrieval configuration.

    Defined at module level so ``Retrieval.run_multichain`` can construct a
    fresh configuration in every chain worker. For ``chain_execution="processes"``
    this function must live in an importable module.
    """
    path_to_data = "./"

    retrieval_config = RetrievalConfig(
        retrieval_name="HR8799e_advanced_example",
        run_mode="retrieval",
        sampler_type="numpyro_nuts",
        adaptive_mesh_refinement=True,
        scattering_in_emission=True,
    )

    # Data.
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

    # Parameters and priors (TFP distributions).
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

    # Clouds: per-species sedimentation fractions.
    retrieval_config.add_parameter("sigma_lnorm", True, distribution=tfpd.Uniform(low=1.05, high=3.0))
    retrieval_config.add_parameter("log_kzz", True, distribution=tfpd.Uniform(low=5.0, high=13.0))
    retrieval_config.add_parameter("f_sed_MgSiO3(s)", True, distribution=tfpd.Uniform(low=1.0, high=11.0))
    retrieval_config.add_parameter("f_sed_Fe(s)", True, distribution=tfpd.Uniform(low=1.0, high=11.0))

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

    return retrieval_config


def configure_corner(retrieval_config):
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


def run_warm_started(retrieval):
    """Single-process NUTS run with MAP warm-start and Hessian preconditioning."""
    retrieval.run(
        num_warmup=500,  # warm-started, preconditioned chains need less adaptation
        num_samples=2000,
        num_chains=4,
        chain_method="vectorized",
        target_acceptance_rate=0.9,
        progress_bar=True,
        use_jit=True,
        seed=12345,  # remove or randomize for a production retrieval
        # Optimal estimation warm-up: find the MAP with optax before sampling.
        optimize_init=True,
        optimizer_kwargs={
            "optimizer": "adam",
            "learning_rate": 1e-2,
            "clip_norm": 1.0,
            "n_steps": 400,
            "n_starts": 2,  # random restarts to guard against local optima
            "lbfgs_polish_steps": 50,  # second-order polish of the best optimum
            "walker_jitter": 0.05,  # spread chains 1-3 around the MAP (chain 0 stays at it)
            "seed": 12345,
        },
        # Precondition the NUTS metric with the inverse Hessian at the MAP.
        init_inverse_mass_matrix_from_hessian=True,
        is_mass_matrix_diagonal=True,  # False for a dense (correlated) metric
    )
    retrieval.plot_all(contribution=True)


def run_multichain_demo():
    """Orchestrate independent single-chain runs and merge their posteriors."""
    result = Retrieval.run_multichain(
        build_config,
        num_chains=4,
        output_directory="./",
        chain_execution="sequential",  # notebook-safe; use "processes" from a script
        chain_seed=12345,
        # Forwarded to each per-chain Retrieval.run call:
        num_warmup=500,
        num_samples=1000,
        target_acceptance_rate=0.9,
        progress_bar=True,
        optimize_init=True,
        optimizer_kwargs={"n_steps": 400, "lbfgs_polish_steps": 50},
        init_inverse_mass_matrix_from_hessian=True,
    )
    print(f"execution mode   : {result['execution']}")
    print(f"succeeded chains : {result['succeeded_chain_ids']}")
    print(f"total samples    : {result['diagnostics']['n_samples_total']}")
    print(f"max R-hat        : {result['diagnostics']['max_rhat']:.4f}")
    print(f"converged        : {result['diagnostics']['converged']}")


def main():
    retrieval_config = build_config()
    configure_corner(retrieval_config)

    retrieval = Retrieval(retrieval_config, output_directory="./", evaluate_sample_spectra=False)
    configure_retrieval_plotter(retrieval)

    # Both runs are expensive; enable one at a time.
    run_retrieval = False
    run_multichain = False

    if run_retrieval:
        run_warm_started(retrieval)
    if run_multichain:
        run_multichain_demo()


if __name__ == "__main__":
    main()
