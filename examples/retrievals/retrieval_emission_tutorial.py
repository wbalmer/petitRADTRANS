"""JAX emission retrieval tutorial.

A compact emission-spectrum retrieval on the HR8799e GRAVITY spectrum, built on
the JAX runtime architecture. It uses the built-in differentiable
``guillot_emission`` model with a clear atmosphere, equilibrium chemistry,
correlated-k opacities, and the JAXNS sampler.

Equivalent to the ``retrieval_emission_tutorial`` documentation notebook.
"""
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import guillot_emission

tfpd = tfp.distributions


def setup_configuration(example_data_file, pressures):
    retrieval_config = RetrievalConfig(
        retrieval_name="hr8799e_emission_jaxns",
        run_mode="retrieval",
        sampler_type="jaxns",
        adaptive_mesh_refinement=False,
        scattering_in_emission=False,
        pressures=pressures,
    )

    retrieval_config.add_parameter(
        name="system_distance", is_free_parameter=False, value=41.2925 * cst.pc, plot_in_corner=False
    )
    retrieval_config.add_parameter(
        name="log_g", is_free_parameter=True, distribution=tfpd.Uniform(low=3.0, high=5.0),
        plot_in_corner=True, corner_ranges=(3.0, 5.0), corner_label=r"$\log(g)$",
    )
    retrieval_config.add_parameter(
        name="planet_radius", is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.8 * cst.r_jup, high=1.6 * cst.r_jup),
        plot_in_corner=True, corner_ranges=(0.8, 1.6), corner_label=r"R$_{pl}$",
        corner_transform=lambda radius: radius / cst.r_jup_mean,
    )
    retrieval_config.add_parameter(
        name="T_int", is_free_parameter=True, distribution=tfpd.Uniform(low=200.0, high=2000.0),
        plot_in_corner=True, corner_ranges=(200.0, 2000.0), corner_label=r"$T_{int}$",
    )
    retrieval_config.add_parameter(
        name="T_equ", is_free_parameter=True, distribution=tfpd.Uniform(low=600.0, high=2200.0),
        plot_in_corner=True, corner_ranges=(600.0, 2200.0), corner_label=r"$T_{eq}$",
    )
    retrieval_config.add_parameter(
        name="gamma", is_free_parameter=True, distribution=tfpd.Uniform(low=0.1, high=2.0),
        plot_in_corner=True, corner_ranges=(0.1, 2.0), corner_label=r"$\gamma$",
    )
    retrieval_config.add_parameter(
        name="log_kappa_IR", is_free_parameter=True, distribution=tfpd.Uniform(low=-5.0, high=0.0),
        plot_in_corner=True, corner_ranges=(-5.0, 0.0), corner_label=r"$\log \kappa_{IR}$",
    )
    retrieval_config.add_parameter(
        name="Fe/H", is_free_parameter=True, distribution=tfpd.Uniform(low=-1.0, high=1.5),
        plot_in_corner=True, corner_ranges=(-1.0, 1.5), corner_label=r"[Fe/H]",
    )
    retrieval_config.add_parameter(
        name="C/O", is_free_parameter=True, distribution=tfpd.Uniform(low=0.2, high=1.2),
        plot_in_corner=True, corner_ranges=(0.2, 1.2), corner_label=r"C/O",
    )

    retrieval_config.set_rayleigh_species(["H2", "He"])
    retrieval_config.set_continuum_opacities(["H2--H2", "H2--He"])
    retrieval_config.set_line_species(
        ["1H2-16O__POKAZATEL", "C-O-NatAbund__HITEMP"],
        use_equilibrium_chemistry=True,
    )

    retrieval_config.add_data(
        name="GRAVITY",
        path_to_observations=str(example_data_file),
        model_generating_function=guillot_emission,
        data_resolution=500,
        model_resolution=300,
        wavelength_boundaries=(1.0, 2.5),
        line_opacity_mode="c-k",
    )

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
    print(f"JAX devices: {jax.devices()}")

    example_data_file = Path("retrievals/emission/observations/HR8799e_Spectra.fits")
    output_root = example_data_file.parents[0] / "runs" / "retrieval_jax_runtime_emission"
    output_root.mkdir(parents=True, exist_ok=True)
    pressures = jnp.logspace(-6.0, 2.0, 100)

    retrieval_config = setup_configuration(example_data_file, pressures)

    # Quick look at the loaded data.
    gravity_data = retrieval_config.data["GRAVITY"]
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        gravity_data.wavelengths, gravity_data.spectrum, yerr=gravity_data.uncertainties,
        fmt="o", ms=4, label="GRAVITY",
    )
    plt.xlabel("Wavelength [micron]")
    plt.ylabel("Flux [W/m2/micron]")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()

    retrieval = Retrieval(
        configuration=retrieval_config,
        output_directory=str(output_root),
        evaluate_sample_spectra=False,
        use_prt_plot_style=True,
    )
    configure_retrieval_plotter(retrieval)

    # Settings below are small for interactive testing; increase for science runs.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(
            devices=[jax.devices()[0]],
            num_live_points=80,
            max_samples=60_000,
            s=4,
            k=0,
            c=200,
            shell_fraction=0.5,
            gradient_guided=False,
            init_efficiency_threshold=0.05,
            verbose=True,
            use_jit=True,
            seed=12345,
        )

        free_parameter_names = [
            name for name, parameter in retrieval.configuration.parameters.items() if parameter.is_free_parameter
        ]
        summary = retrieval.sampler.get_summary(free_parameter_names=free_parameter_names)
        print("Evidence:", summary.get("evidence", {}))
        print("Diagnostics:", summary.get("diagnostics", {}))
        print("Posterior summary:")
        for name, stats in summary.get("posterior", {}).items():
            print(f"{name:>16s}: mean={stats.get('mean')}, std={stats.get('std')}")


if __name__ == "__main__":
    main()
