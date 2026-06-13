"""Emission spectra retrieval of HR 8799 e.

Based on the forward model of Mollière et al. (2020): a spline temperature
structure, (dis)equilibrium chemistry with quenching, an Ackermann-Marley cloud
model with equilibrium-condensate clouds, scattering, GRAVITY spectroscopy plus
photometry, and the pymultinest sampler.

This is a realistic, expensive retrieval (days on hundreds of cores). Set
``run_retrieval = True`` in ``main`` to attempt running it.

Equivalent to the ``retrieval_emission`` documentation notebook.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

from jax import config
config.update("jax_enable_x64", True)

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import molliere_2020_emission


def setup_configuration(path_to_data="./"):
    retrieval_config = RetrievalConfig(
        retrieval_name="HR8799e_example",
        run_mode="retrieval",
        adaptive_mesh_refinement=True,
        scattering_in_emission=True,
        sampler_type="pymultinest",
    )

    # GRAVITY spectroscopy (Mollière et al. 2020 also used SPHERE and GPI data).
    retrieval_config.add_data(
        name="GRAVITY",
        path_to_observations=f"{path_to_data}/retrievals/emission/observations/HR8799e_Spectra.fits",
        data_resolution=500,
        model_resolution=1000,
        model_generating_function=molliere_2020_emission,
    )

    # Photometry. Format: name, lower edge [um], upper edge [um], flux [W/m2/um], error.
    retrieval_config.add_photometry(
        path_to_data + "retrievals/emission/observations/hr8799e_photometry.txt",
        model_generating_function=molliere_2020_emission,
        model_resolution=40,
    )

    # Parameters and priors (Mollière 2020 model). Lambda functions give uniform priors.
    retrieval_config.add_parameter(name="system_distance", is_free_parameter=False, value=41.2925 * cst.pc)
    retrieval_config.add_parameter("log_g", True, transform_prior_cube_coordinate=lambda x: 2 + 3.5 * x)
    retrieval_config.add_parameter(
        "planet_radius", True, transform_prior_cube_coordinate=lambda x: (0.7 + 1.3 * x) * cst.r_jup_mean
    )

    # Spline temperature structure (T1 < T2 < T3 enforced in the model function).
    retrieval_config.add_parameter(
        "T_int", True, transform_prior_cube_coordinate=lambda x: 300.0 + 2000.0 * x, value=0.0
    )
    retrieval_config.add_parameter("T3", True, transform_prior_cube_coordinate=lambda x: x, value=0.0)
    retrieval_config.add_parameter("T2", True, transform_prior_cube_coordinate=lambda x: x, value=0.0)
    retrieval_config.add_parameter("T1", True, transform_prior_cube_coordinate=lambda x: x)

    # Optical depth model: tau = delta * press_cgs**alpha.
    retrieval_config.add_parameter("alpha", True, transform_prior_cube_coordinate=lambda x: 1.0 + x)
    retrieval_config.add_parameter("log_delta", True, transform_prior_cube_coordinate=lambda x: x)

    # Chemistry (bulk parameters for (dis)equilibrium chemistry).
    retrieval_config.add_parameter("log_pquench", True, transform_prior_cube_coordinate=lambda x: -6.0 + 9.0 * x)
    retrieval_config.add_parameter("Fe/H", True, transform_prior_cube_coordinate=lambda x: -1.5 + 3.0 * x)
    retrieval_config.add_parameter("C/O", True, transform_prior_cube_coordinate=lambda x: 0.1 + 1.5 * x)

    # Clouds (Ackermann-Marley 2001 model).
    retrieval_config.add_parameter("sigma_lnorm", True, transform_prior_cube_coordinate=lambda x: 1.05 + 1.95 * x)
    retrieval_config.add_parameter("log_kzz", True, transform_prior_cube_coordinate=lambda x: 5.0 + 8.0 * x)
    retrieval_config.add_parameter("f_sed", True, transform_prior_cube_coordinate=lambda x: 1.0 + 10.0 * x)

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

    # This retrieval takes days on hundreds of cores; set to True to attempt it.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(
            n_live_points=400,
            sampling_efficiency=0.8,
            const_efficiency_mode=False,
            resume=False,
            seed=12345,  # use -1 for a real retrieval; fixed here for reproducibility
        )
        retrieval.plot_all(contribution=True)


if __name__ == "__main__":
    main()
