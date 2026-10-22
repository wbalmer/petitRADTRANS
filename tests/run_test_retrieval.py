import os
# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # keep hdf5 file locking up across processes 

from jax import config
config.update("jax_enable_x64", True)

import tensorflow_probability.substrates.jax as tfp
from petitRADTRANS.retrieval import Retrieval, RetrievalConfig
from petitRADTRANS.retrieval.models import molliere_2020_emission, gradient_profile_emission
from petitRADTRANS import physical_constants as cst

tfpd = tfp.distributions

run_retrieval = True

if __name__ == '__main__':
    # Define the pRT run setup
    retrieval_config = RetrievalConfig(
        retrieval_name="2m0624_example",  # give a useful name for your retrieval
        run_mode="retrieval",  # 'retrieve' to run, or 'evaluate' to make plots
        sampler_type="numpyro_nuts",  # use NumPyro NUTS for this differentiable retrieval
        adaptive_mesh_refinement=True,  # adaptive mesh refinement, slower if True
        scattering_in_emission=True,  # add scattering for emission spectra clouds
    )

    # Read in Data

    # Here we import petitRADTRANS to find the path of the example files on your machine
    # In general this is not required, you just put the files in the folder that you are running
    # Your script in, for example
    path_to_data = "/projects/b1094/balmer/spectral_data"

    retrieval_config.add_data(
        name="JWST_BIN",
        path_to_observations=f"{path_to_data}/2M0624_bin300.txt",
        # data_resolution=300,
        model_resolution=1000,
        model_generating_function=gradient_profile_emission,
    )

    # Note that in Mollière et al. (2020), additional data sets from SPHERE and GPI were used

    #retrieval_config.add_photometry(
    #    path_to_data + "/aflepb_photometry.txt", gradient_profile_emission, model_resolution=40
    #)

    # Add parameters and priors for free parameters

    # This run uses the model of Molliere (2020) for HR8799e
    # For NumPyro NUTS we define JAX-native priors with TensorFlow Probability distributions.

    # Distance to the planet in cm
    retrieval_config.add_parameter(name="system_distance", is_free_parameter=False, value=26.84 * cst.pc)

    # Log of the surface gravity in cgs units.
    retrieval_config.add_parameter(
        "log_g",
        True,
        distribution=tfpd.Uniform(low=3.0, high=5.0),
    )

    # Planet radius in cm
    retrieval_config.add_parameter(
        "planet_radius",
        True,
        distribution=tfpd.Uniform(low=0.7 * cst.r_jup_mean, high=2.0 * cst.r_jup_mean),
    )

    # P-T profile
    n_layers = 7 
    retrieval_config.add_parameter('N_layers', is_free_parameter=False, value = n_layers)
    retrieval_config.add_parameter('T_bottom', True, distribution=tfpd.Uniform(low=2000.0, high=20000.0))

    # dts_exorem = [0.1, 0.16, 0.25, 0.23, 0.17, 0.23] # first value from Zhang paper, others from LT transition temp range (800-1400K) exorem PT profile 
    # dts_err_exorem = [0.25, 0.04, 0.04, 0.05, 0.08, 0.13]

    dts_exorem = {1e3:(0.05, 0.30), # from teff range pt priors 
                    1e2:(0.15, 0.04), 
                    1e1:(0.23, 0.06), 
                    1e0:(0.25, 0.04), 
                    1e-1:(0.21, 0.05), 
                    1e-2:(0.2, 0.09), 
                    1e-3:(0.23, 0.13)}

    # using truncated normals for now to avoid inversions
    retrieval_config.add_parameter(f'PTslope_1',
                                True,
                                distribution=tfpd.Uniform(low=dts_exorem[1e3][0],
                                                        high=dts_exorem[1e3][1]))
    retrieval_config.add_parameter(f'PTslope_2',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e2][0], scale=dts_exorem[1e2][1], low=0.0, high=0.5))

    retrieval_config.add_parameter(f'PTslope_3',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e1][0], scale=dts_exorem[1e1][1], low=0.0, high=0.5))

    retrieval_config.add_parameter(f'PTslope_4',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e0][0], scale=dts_exorem[1e0][1], low=0.0, high=0.5))
    retrieval_config.add_parameter(f'PTslope_5',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e-1][0], scale=dts_exorem[1e-1][1], low=0.0, high=0.5))
    retrieval_config.add_parameter(f'PTslope_6',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e-2][0], scale=dts_exorem[1e-2][1], low=0.0, high=0.5))
    retrieval_config.add_parameter(f'PTslope_7',
                                True,
                                distribution=tfpd.TruncatedNormal(loc=dts_exorem[1e-3][0], scale=dts_exorem[1e-3][1], low=0.0, high=0.5))

    # Chemistry
    # A 'free retrieval' would have each line species as a parameter
    # Using a (dis)equilibrium model, we only supply bulk parameters.
    # Carbon quench pressure
    retrieval_config.add_parameter("log_pquench", True, distribution=tfpd.Uniform(low=-6.0, high=3.0))
    # Metallicity [Fe/H]
    retrieval_config.add_parameter("Fe/H", True, distribution=tfpd.Uniform(low=-1.5, high=1.5))
    # C/O ratio
    retrieval_config.add_parameter("C/O", True, distribution=tfpd.Uniform(low=0.1, high=1.6))
    # Clouds
    # Based on an Ackermann-Marley (2001) cloud model
    # Width of particle size distribution
    retrieval_config.add_parameter("sigma_lnorm", is_free_parameter=False, value=2.0)#True, distribution=tfpd.Uniform(low=1.05, high=3.0))
    # Vertical mixing parameters
    # retrieval_config.add_parameter("log_kzz", True, distribution=tfpd.Uniform(low=5.0, high=13.0))
    # Sedimentation parameter
    # retrieval_config.add_parameter("f_sed", True, distribution=tfpd.Uniform(low=0.1, high=11.0))

    # Define opacity species to be included
    retrieval_config.set_rayleigh_species(("H2", "He"))
    retrieval_config.set_continuum_opacities(("H2--H2", "H2--He"))
    retrieval_config.set_line_species(
        (
            "H2O__POKAZATEL",
            "CO-NatAbund",
            "CH4",
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
        use_equilibrium_chemistry=True,
    )

    retrieval_config.add_cloud_species(
        "Fe(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits = (-10.0, 0.0),
        cloud_base_pressure_limits = (-3.0, 3.0)
        # equilbrium_mass_fraction_scaling_factor=(-3.5, 1.0),
    )

    retrieval_config.add_cloud_species(
        "MgSiO3(s)_crystalline_000__DHS",
        use_equilibrium_chemistry=False,
        free_mass_fraction_limits = (-10.0, 0.0),
        cloud_base_pressure_limits = (-3.0, 3.0)
        # equilbrium_mass_fraction_scaling_factor=(-3.5, 1.0),
    )

    retrieval_config.add_cloud_species(
            "SiO2(s)_amorphous__Mie",
            use_equilibrium_chemistry=False,
            free_mass_fraction_limits = (-10.0, 0.0),
            cloud_base_pressure_limits = (-6.0, -2.0)
            # equilbrium_mass_fraction_scaling_factor=(-3.5, 1.0),
        )

    for specie in retrieval_config.cloud_species:
    
            retrieval_config.add_parameter('log_cloud_radius_'+specie.split('_')[0],
                                           True,\
                                           distribution=tfpd.Uniform(low=-8.0, high=2.0))
            
            
            retrieval_config.add_parameter(
               'f_sed_'+specie.split('_')[0],
                True,
                distribution=tfpd.Uniform(low=0.0, high=10.0))

    # Before we run the retrieval, let's set up plotting.
    # Define what to put into corner plot if run_mode == 'evaluate'
    # We could also do this with the plot_in_corner argument for add_parameter
    retrieval_config.parameters["planet_radius"].plot_in_corner = True
    retrieval_config.parameters["planet_radius"].corner_label = r"$R_{\rm P}$ ($\rm R_{Jup}$)"
    retrieval_config.parameters["planet_radius"].corner_transform = lambda x: x / cst.r_jup_mean
    retrieval_config.parameters["log_g"].plot_in_corner = True
    retrieval_config.parameters["log_g"].corner_ranges = [2.0, 5.0]
    retrieval_config.parameters["log_g"].corner_label = "log g"
    retrieval_config.parameters["C/O"].plot_in_corner = True
    retrieval_config.parameters["Fe/H"].plot_in_corner = True
    retrieval_config.parameters["log_pquench"].plot_in_corner = True
    retrieval_config.parameters["log_pquench"].corner_label = "log pquench"

    for spec in retrieval_config.cloud_species:
        cname = spec.split('_')[0]
        retrieval_config.parameters['log_X_cb_' + cname].plot_in_corner = True
        retrieval_config.parameters['log_X_cb_' + cname].corner_label = cname
        retrieval_config.parameters['fsed_' + cname].plot_in_corner = True
        retrieval_config.parameters['fsed_' + cname].corner_label = 'fsed_'+cname    
        retrieval_config.parameters['log_cloud_radius_' + cname].plot_in_corner = True
        retrieval_config.parameters['log_cloud_radius_' + cname].corner_label = 'log_cloud_rad_'+cname
        retrieval_config.parameters['log_Pbase_' + cname].plot_in_corner = True
        retrieval_config.parameters['log_Pbase_' + cname].corner_label = 'log_Pbase_'+cname



    def configure_retrieval_plotter(retrieval):
        retrieval.plotter.spec_xlabel = "Wavelength [micron]"
        retrieval.plotter.spec_ylabel = "Flux [W/m2/micron]"
        retrieval.plotter.y_axis_scaling = 1.0
        retrieval.plotter.xscale = "log"
        retrieval.plotter.yscale = "linear"
        retrieval.plotter.resolution = 100.0  # maximum resolution, will rebin the data
        retrieval.plotter.nsample = 100  # if we want a plot with many sampled spectra
        retrieval.plotter.reference_data_name = "GRAVITY"
        retrieval.plotter.temp_limits = [150, 3000]
        retrieval.plotter.press_limits = [1e2, 1e-5]

    retrieval = Retrieval(retrieval_config, output_directory="./", evaluate_sample_spectra=False)

    configure_retrieval_plotter(retrieval)

    if run_retrieval:
        retrieval.run(
            # NUTS sampling configuration
            num_warmup=500,  # warm-started, preconditioned chains need less adaptation
            num_samples=2000,
            num_chains=1,
            chain_method="vectorized",
            target_acceptance_rate=0.9,
            progress_bar=True,
            use_jit=True,
            seed=12345,  # remove or randomize for a production retrieval
            # Optimal estimation warm-up: find the MAP with optax before sampling
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
            # Precondition the NUTS metric with the inverse Hessian at the MAP
            init_inverse_mass_matrix_from_hessian=True,
            is_mass_matrix_diagonal=True,  # False for a dense (correlated) metric
        )

    if run_retrieval:
        retrieval.plot_all(contribution=True)

