"""Basic retrieval example.

Runs a simple free-chemistry, isothermal transmission retrieval on a simulated
HST spectrum of a sub-Neptune using the built-in ``isothermal_transmission``
model and the JAXNS nested sampler.

Equivalent to the ``retrieval_basic`` documentation notebook.
"""
import os

# Prevent numpy from parallelising on its own when using MPI.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

# 64-bit precision must be set before importing petitRADTRANS.
from jax import config
config.update("jax_enable_x64", True)

import tensorflow_probability.substrates.jax as tfp

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import isothermal_transmission

tfpd = tfp.distributions


def setup_configuration():
    """Set up the RetrievalConfig with parameters, opacities and data."""
    retrieval_config = RetrievalConfig(
        retrieval_name="hst_example_clear_spec",
        run_mode="retrieval",  # set to 'evaluate' to only make plots from a finished run
        sampler_type="jaxns",
        adaptive_mesh_refinement=False,
        scattering_in_emission=False,
    )

    # Fixed parameters.
    retrieval_config.add_parameter(
        name="stellar_radius",
        is_free_parameter=False,
        value=0.651 * cst.r_sun,
        plot_in_corner=False,
    )

    # Free parameters: priors transform a uniform sample in [0, 1] to physical units.
    retrieval_config.add_parameter(
        name="log_g",
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=2.0, high=5.5),
        plot_in_corner=True,
        corner_ranges=(2.0, 5.5),
        corner_label=r"$\log(g)$",
    )
    retrieval_config.add_parameter(
        name="planet_radius",
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=0.1 * cst.r_jup, high=0.4 * cst.r_jup),
        plot_in_corner=True,
        corner_ranges=(0.1, 0.4),
        corner_label=r"R$_{pl}$",
        corner_transform=lambda x: x / cst.r_jup_mean,
    )
    retrieval_config.add_parameter(
        name="temperature",
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=300, high=2000),
        plot_in_corner=True,
        corner_ranges=(300, 2000),
        corner_label=r"Temp",
    )
    retrieval_config.add_parameter(
        "log_Pcloud",
        is_free_parameter=True,
        distribution=tfpd.Uniform(low=-6, high=2),
        plot_in_corner=True,
        corner_ranges=(-6, 2),
        corner_label=r"$\log P_{cl}$",
    )

    # Opacities.
    retrieval_config.set_rayleigh_species(["H2", "He"])
    retrieval_config.set_continuum_opacities(["H2--H2", "H2--He"])
    retrieval_config.set_line_species(
        ["1H2-16O__POKAZATEL", "12C-1H4__HITEMP", "12C-16O2__UCL-4000", "C-O-NatAbund__HITEMP"],
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-6.0, 0.0),
        plot_in_corner=True,
    )

    # Data. ASCII format: wavelength [micron], flux [(Rp/Rstar)^2], flux error.
    path_to_data = "./"
    retrieval_config.add_data(
        name="HST",
        path_to_observations=f"{path_to_data}retrievals/transmission/observations/HST/hst_example_clear_spec.txt",
        model_generating_function=isothermal_transmission,
        line_opacity_mode="c-k",
        data_resolution=60,
        model_resolution=120,
    )

    return retrieval_config


def configure_retrieval_plotter(retrieval):
    """Set up plot axis properties used in evaluation mode."""
    retrieval.plotter.spec_xlabel = "Wavelength [micron]"
    retrieval.plotter.spec_ylabel = "Transit Depth [ppm]"
    retrieval.plotter.y_axis_scaling = 1e6
    retrieval.plotter.xscale = "linear"
    retrieval.plotter.yscale = "linear"
    retrieval.plotter.resolution = None
    retrieval.plotter.reference_data_name = "HST"
    retrieval.plotter.temp_limits = [150, 3000]
    retrieval.plotter.press_limits = [1e2, 1e-5]


def main():
    retrieval_config = setup_configuration()

    output_directory = "./retrievals/runs"
    retrieval = Retrieval(
        configuration=retrieval_config,
        output_directory=output_directory,
        evaluate_sample_spectra=False,
        use_prt_plot_style=True,
    )
    configure_retrieval_plotter(retrieval)

    # Few live points are used here so the example runs quickly; use more for real runs.
    retrieval.run(
        num_live_points=40,
        max_samples=1e5,
        verbose=True,
        parameter_estimation=True,
    )

    # Read the samples (dictionaries keyed by retrieval name in case of multiple runs).
    sample_dict, parameter_dict = retrieval.get_samples(output_directory=output_directory)
    samples_use = sample_dict[retrieval.configuration.retrieval_name]
    parameters_read = parameter_dict[retrieval.configuration.retrieval_name]

    # Plots: best fit spectrum, P-T profile, corner plot.
    retrieval.plot_spectra(samples_use, parameters_read, refresh=True, mode="bestfit")
    retrieval.plot_pt(sample_dict, parameters_read, contribution=True)
    retrieval.plot_corner(sample_dict, parameter_dict, parameters_read, title_kwargs={"fontsize": 10}, smooth=1)


if __name__ == "__main__":
    main()
