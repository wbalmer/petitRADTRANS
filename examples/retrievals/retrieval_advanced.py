"""Advanced retrieval example.

A complex emission retrieval of HR 8799 e following Mollière et al. (2020):
gradient (Zhang+ 2023) temperature profile, (dis)equilibrium chemistry with a
pressure-varying CO profile, an Ackermann-Marley cloud model, patchy clouds,
free cloud species, photometry, and the pymultinest sampler.

This retrieval is computationally expensive (days on many cores). Set
``run_retrieval = True`` in ``main`` to attempt running it.

Equivalent to the ``retrieval_advanced`` documentation notebook.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

from jax import config
config.update("jax_enable_x64", True)

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import gradient_profile_emission
from petitRADTRANS.retrieval.utils import gaussian_prior


def setup_configuration(path_to_data="./"):
    retrieval_config = RetrievalConfig(
        retrieval_name="HR8799e_example",
        run_mode="retrieval",
        adaptive_mesh_refinement=True,
        scattering_in_emission=True,
        sampler_type="pymultinest",
    )

    # Spectral data (GRAVITY, Mollière et al. 2020).
    retrieval_config.add_data(
        name="GRAVITY",
        path_to_observations=f"{path_to_data}/retrievals/emission/observations/HR8799e_Spectra.fits",
        data_resolution=500,
        model_resolution=1000,
        model_generating_function=gradient_profile_emission,
        scale_flux=True,
        scale_uncertainties=True,
        fit_flux_offset=False,
        subtract_continuum=False,
        line_opacity_mode="c-k",
        radtrans_object=None,
        wavelengths=None,
    )

    # Photometry. Format: name, lower edge [um], upper edge [um], flux [W/m2/um], error.
    # With no transformation function, RetrievalConfig uses species to fetch the filter.
    retrieval_config.add_photometry(
        path_to_data + "retrievals/emission/observations/hr8799e_photometry.txt",
        gradient_profile_emission,
        model_resolution=40,
        scale_flux=False,
        photometric_transformation_function=None,
    )

    # Fixed and free parameters.
    retrieval_config.add_parameter(name="system_distance", is_free_parameter=False, value=41.2925 * cst.pc)
    retrieval_config.add_parameter("log_g", True, transform_prior_cube_coordinate=lambda x: 2 + 3.5 * x)
    retrieval_config.add_parameter(
        "planet_radius", True, transform_prior_cube_coordinate=lambda x: (0.7 + 1.3 * x) * cst.r_jup_mean
    )

    # Temperature profile, based on Zhang+ 2023.
    n_layers = 10
    retrieval_config.add_parameter("top_of_atmosphere_pressure", False, value=-6)
    retrieval_config.add_parameter("N_layers", False, value=n_layers)
    retrieval_config.add_parameter("T_bottom", True, transform_prior_cube_coordinate=lambda x: 2000.0 + 10000.0 * x)

    pt_slope_priors = [
        (0.25, 0.1), (0.25, 0.1), (0.26, 0.1), (0.2, 0.1), (0.12, 0.1),
        (0.07, 0.1), (0.0, 0.5), (0.0, 0.5), (0.0, 0.05), (0.0, 0.05),
    ]
    for i, (mu, sigma) in enumerate(pt_slope_priors, start=1):
        retrieval_config.add_parameter(
            f"PTslope_{i}",
            True,
            transform_prior_cube_coordinate=lambda x, mu=mu, sigma=sigma: gaussian_prior(x, mu, sigma),
        )

    # Chemistry: (dis)equilibrium with bulk parameters.
    retrieval_config.add_parameter("log_pquench", True, transform_prior_cube_coordinate=lambda x: -6.0 + 9.0 * x)
    retrieval_config.add_parameter("Fe/H", True, transform_prior_cube_coordinate=lambda x: -1.5 + 3.0 * x)
    retrieval_config.add_parameter("C/O", True, transform_prior_cube_coordinate=lambda x: 0.1 + 1.5 * x)

    # Let CO vary as a function of altitude.
    retrieval_config.add_pressure_varying_line_species(
        "CO-NatAbund__HITEMP",
        mode="linear",
        pressure_spacing="relative",
        n_nodes=3,
        free_mass_fraction_limits=(-7.0, 0.0),
        log_pressure_range_prior=(0, 9),
    )

    # Clouds: Ackermann-Marley (2001) model.
    retrieval_config.add_parameter("sigma_lnorm", True, transform_prior_cube_coordinate=lambda x: 1.05 + 1.95 * x)
    retrieval_config.add_parameter("fsed_MgSiO3(s)", True, transform_prior_cube_coordinate=lambda x: 10.0 * x)
    retrieval_config.add_parameter("fsed_Fe(s)", True, transform_prior_cube_coordinate=lambda x: 10.0 * x)
    retrieval_config.add_parameter(
        "log_cloud_radius_MgSiO3(s)", True, transform_prior_cube_coordinate=lambda x: -7 + 10.0 * x
    )
    retrieval_config.add_parameter(
        "log_cloud_radius_Fe(s)", True, transform_prior_cube_coordinate=lambda x: -7 + 10.0 * x
    )
    retrieval_config.add_parameter("cloud_coverage", True, transform_prior_cube_coordinate=lambda x: x)
    retrieval_config.add_parameter("patchy_clouds", False, ("MgSiO3(s)_crystalline_000__DHS",))

    # Opacity species.
    retrieval_config.set_rayleigh_species(("H2", "He"))
    retrieval_config.set_continuum_opacities(("H2--H2", "H2--He"))
    retrieval_config.set_line_species(
        (
            "H2O__POKAZATEL",
            "CO-NatAbund",  # freely retrieving the abundance profile of CO
            "CH4__MM",
            "CO2",
            "HCN",
            "FeH",
            "H2S",
            "NH3",
            "PH3",
            "Na__NewAllard",
            "K__Allard",
            "TiO",
            "VO",
            "SiO",
        ),
        eq=False,
    )
    # CO varies with altitude, so remove it as a vertically-constant parameter.
    del retrieval_config.parameters["CO-NatAbund"]

    # Free cloud parameterisation: retrieve cloud base pressure and abundance directly.
    retrieval_config.add_cloud_species(
        "Fe(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-7.0, -1.0),
        cloud_base_pressure_limits=(-6, 3),
    )
    retrieval_config.add_cloud_species(
        "MgSiO3(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits=(-7.0, -1.0),
        cloud_base_pressure_limits=(-6, 3),
    )

    # Plot every parameter in the corner plot.
    for parameter in retrieval_config.parameters:
        retrieval_config.parameters[parameter].plot_in_corner = True

    return retrieval_config


def main():
    retrieval_config = setup_configuration()

    retrieval = Retrieval(
        retrieval_config,
        output_directory="./",
        evaluate_sample_spectra=False,
    )

    # This retrieval takes days on hundreds of cores; set to True to attempt it.
    run_retrieval = False
    if run_retrieval:
        retrieval.run(
            sampling_efficiency=0.8,
            const_efficiency_mode=False,
            n_live_points=400,
            log_z_convergence=0.5,
            step_sampler=False,
            warmstart_max_tau=0.5,
            n_iter_before_update=50,
            resume=False,
            max_iters=0,
            frac_remain=0.1,
            l_epsilon=0.3,
            error_checking=True,
            force_serial_error_checking=False,
            seed=12345,  # use -1 for a real retrieval; fixed here for reproducibility
        )
        retrieval.plot_all(contribution=True)


if __name__ == "__main__":
    main()
