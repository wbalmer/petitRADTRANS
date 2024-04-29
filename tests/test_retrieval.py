"""Test the petitRADTRANS retrieval module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).
"""
import json
import os

import numpy as np

from petitRADTRANS._input_data_loader import (get_default_correlated_k_resolution, get_opacity_input_file,
                                              get_resolving_power_string, join_species_all_info)
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.utils import gaussian_prior

from .context import petitRADTRANS
from .utils import tests_results_directory, reference_filenames, test_parameters

# TODO MultiNest's random generator is computer-dependent (on Mac only?), the tolerance is increased to match this
relative_tolerance = 1e0  # relative tolerance when comparing with older results
max_number_of_tests = 3


def init_run():
    # Since our retrieval has already run before, we'll set the mode to 'evaluate' so we can make some plots.
    run_definition_simple = petitRADTRANS.retrieval.RetrievalConfig(
        retrieval_name="test",
        run_mode="retrieval",  # This must be 'retrieval' to run PyMultiNest
        amr=False,  # We won't be using adaptive mesh refinement for the pressure grid
        pressures=test_parameters['pressures'],
        scattering_in_emission=False  # This would turn on scattering when calculating emission spectra
    )
    # Scattering is automatically included for transmission spectra

    # Fixed parameters
    run_definition_simple.add_parameter(
        'Rstar',
        False,
        value=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun
    )

    # Log of the surface gravity
    run_definition_simple.add_parameter(
        'log_g',
        False,
        value=np.log10(test_parameters['planetary_parameters']['reference_gravity'])
    )

    # Retrieved parameters
    # Prior functions
    def prior_planet_radius(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters['retrieval_parameters']['planetary_radius_bounds'][0]
               * petitRADTRANS.physical_constants.r_jup_mean,
            x2=test_parameters['retrieval_parameters']['planetary_radius_bounds'][1]
               * petitRADTRANS.physical_constants.r_jup_mean,
        )

    def prior_temperature(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters['retrieval_parameters']['intrinsic_temperature_bounds'][0],
            x2=test_parameters['retrieval_parameters']['intrinsic_temperature_bounds'][1]
        )

    def prior_cloud_pressure(x):
        return petitRADTRANS.retrieval.utils.uniform_prior(
            cube=x,
            x1=test_parameters['retrieval_parameters']['log10_cloud_pressure_bounds'][0],
            x2=test_parameters['retrieval_parameters']['log10_cloud_pressure_bounds'][1]
        )

    # Planet radius
    run_definition_simple.add_parameter(
        'R_pl',
        True,
        transform_prior_cube_coordinate=prior_planet_radius
    )

    # Intrinsic temperature
    run_definition_simple.add_parameter(
        'Temperature',
        True,
        transform_prior_cube_coordinate=prior_temperature
    )

    # Include a grey cloud as well to see what happens
    run_definition_simple.add_parameter(
        "log_Pcloud",
        True,
        transform_prior_cube_coordinate=prior_cloud_pressure
    )

    # Spectrum parameters
    run_definition_simple.set_rayleigh_species(test_parameters['spectrum_parameters']['rayleigh_species'])
    run_definition_simple.set_continuum_opacities(test_parameters['spectrum_parameters']['continuum_opacities'])

    run_definition_simple.set_line_species(
        test_parameters['spectrum_parameters']['line_species_correlated_k'],
        eq=False,
        abund_lim=(
            test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][0],
            test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][1]
        )  # prior: min = abund_lim[0], max = min + abund_lim[1]
    )

    # Remove old binned down opacities to test rebinning function
    for species in test_parameters['spectrum_parameters']['line_species_correlated_k']:
        file = get_opacity_input_file(
            path_input_data=petitRADTRANS.config.petitradtrans_config_parser.get_input_data_path(),
            category='correlated_k_opacities',
            species=species
        ).replace(
            join_species_all_info('', spectral_info=get_default_correlated_k_resolution()),
            join_species_all_info(
                '',
                get_resolving_power_string(test_parameters['mock_observation_parameters']['resolving_power'] * 2)
            )
        )

        if os.path.isfile(file):
            os.remove(file)

    # Load data
    run_definition_simple.add_data(
        name='test',
        path_to_observations=reference_filenames['mock_observation_transmission'],
        model_generating_function=retrieval_model_spec_iso,
        line_opacity_mode='c-k',
        data_resolution=test_parameters['mock_observation_parameters']['resolving_power'],
        model_resolution=test_parameters['mock_observation_parameters']['resolving_power'] * 2
    )

    # Plot parameters
    # Corner plot
    run_definition_simple.parameters['R_pl'].plot_in_corner = True
    run_definition_simple.parameters['R_pl'].corner_label = r'$R_{\rm P}$ ($\rm R_{Jup}$)'
    run_definition_simple.parameters['R_pl'].corner_transform = \
        lambda x: x / petitRADTRANS.physical_constants.r_jup_mean
    run_definition_simple.parameters['Temperature'].plot_in_corner = True
    run_definition_simple.parameters['Temperature'].corner_label = "Temp"
    run_definition_simple.parameters['log_Pcloud'].plot_in_corner = True
    run_definition_simple.parameters['log_Pcloud'].corner_label = r"log P$_{\rm cloud}$"
    run_definition_simple.parameters['log_Pcloud'].corner_ranges = [-6, 2]

    for spec in run_definition_simple.line_species:
        spec = spec.split(Data.resolving_power_str)[0]  # deal with the naming scheme for binned down opacities
        run_definition_simple.parameters[spec].plot_in_corner = True
        run_definition_simple.parameters[spec].corner_ranges = [-6.0, 0.0]

    # Spectrum plot
    run_definition_simple.plot_kwargs["spec_xlabel"] = 'Wavelength [micron]'
    run_definition_simple.plot_kwargs["spec_ylabel"] = r'$(R_{\rm P}/R_*)^2$ [ppm]'
    run_definition_simple.plot_kwargs["y_axis_scaling"] = 1e6  # so we have units of ppm
    run_definition_simple.plot_kwargs["xscale"] = 'linear'
    run_definition_simple.plot_kwargs["yscale"] = 'linear'

    run_definition_simple.plot_kwargs["nsample"] = 10

    # Temperature profile plot
    run_definition_simple.plot_kwargs["take_PTs_from"] = 'test'
    run_definition_simple.plot_kwargs["temp_limits"] = [150, 3000]
    run_definition_simple.plot_kwargs["press_limits"] = [1e2, 1e-5]

    return run_definition_simple


# Atmospheric model
def retrieval_model_spec_iso(prt_object, parameters, pt_plot_mode=None, amr=False):
    """
    This model computes a transmission spectrum based on free retrieval chemistry
    and an isothermal temperature-pressure profile.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                Rstar : Radius of the host star [cm]
                log_g : Log of surface gravity
                R_pl : planet radius [cm]
                Temperature : Isothermal temperature [K]
                species : Log abundances for each species used in the retrieval
                log_Pcloud : optional, cloud base pressure of a grey cloud deck.
        pt_plot_mode:
            Return only the pressure-temperature profile for plotting. Evaluate mode only. Mandatory.
        amr:
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base. Mandatory.

    Returns:
        wavelength_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    # Make the P-T profile
    pressures = prt_object.pressures * 1e-6  # bar to cgs
    temperatures = parameters['Temperature'].value * np.ones_like(pressures)

    # Make the abundance profiles
    abundances = {}
    m_sum = 0.0  # Check that the total mass fraction of all species is <1

    for species in prt_object.line_species:
        spec = species.split(Data.resolving_power_str)[0]  # deal with the naming scheme for binned down opacities
        abundances[species] = 10 ** parameters[spec].value * np.ones_like(pressures)
        m_sum += 10 ** parameters[spec].value

    abundances['H2'] = test_parameters['mass_fractions_correlated_k']['H2'] * (1.0 - m_sum) * np.ones_like(pressures)
    abundances['He'] = test_parameters['mass_fractions_correlated_k']['He'] * (1.0 - m_sum) * np.ones_like(pressures)

    # Find the mean molecular weight in each layer
    mmw = compute_mean_molar_masses(abundances)

    # Calculate the spectrum
    wavelengths, transit_radii, _ = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=10 ** parameters['log_g'].value,
        mean_molar_masses=mmw,
        planet_radius=parameters['R_pl'].value,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        opaque_cloud_top_pressure=10 ** parameters['log_Pcloud'].value,
        frequencies_to_wavelengths=True  # True by default
    )

    # Transform the outputs into the units of our data.
    wavelengths *= 1e4  # cm to um
    spectrum_model = (transit_radii / parameters['Rstar'].value) ** 2.

    return wavelengths, spectrum_model


run_definition = init_run()


def test_list_available_species():
    run_definition.list_available_line_species()


def test_simple_retrieval():
    retrieval = petitRADTRANS.retrieval.Retrieval(
        run_definition,
        output_directory=tests_results_directory,
        evaluate_sample_spectra=test_parameters['retrieval_parameters']['sample_spectrum_output'],
        ultranest=test_parameters['retrieval_parameters']['ultranest']  # if False, use PyMultiNest
    )

    retrieval.run(
        sampling_efficiency=test_parameters['retrieval_parameters']['sampling_efficiency'],
        n_live_points=test_parameters['retrieval_parameters']['n_live_points'],
        const_efficiency_mode=test_parameters['retrieval_parameters']['const_efficiency_mode'],
        resume=test_parameters['retrieval_parameters']['resume'],
        seed=test_parameters['retrieval_parameters']['seed']
    )

    # Just check if get_samples works
    sample_dict, parameter_dict = retrieval.get_samples(
        ultranest=False,
        names=retrieval.corner_files,
        output_directory=retrieval.output_directory
    )

    # Get results and reference
    with open(reference_filenames['pymultinest_parameter_analysis']) as f:
        reference = json.load(f)

    new_result_file = os.path.join(
        tests_results_directory,
        'out_PMN',
        os.path.basename(reference_filenames['pymultinest_parameter_analysis'])
    )

    with open(new_result_file) as f:
        new_results = json.load(f)

    # Check if retrieved parameters are in +/- 1 sigma of the previous retrieved parameters
    for i, marginal in enumerate(new_results['marginals']):
        assert marginal['median'] - relative_tolerance * reference['marginals'][i]['sigma'] \
               <= reference['marginals'][i]['median'] \
               <= marginal['median'] + relative_tolerance * reference['marginals'][i]['sigma']
