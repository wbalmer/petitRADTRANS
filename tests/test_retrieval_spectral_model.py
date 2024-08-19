import json
import os

import numpy as np

from .context import petitRADTRANS
from .utils import reference_filenames, temperature_isothermal, tests_results_directory, test_parameters

relative_tolerance = 1e0  # relative tolerance when comparing with older results


def init_retrieval():
    retrieved_parameters = {
        'temperature': {
            'prior_parameters': [
                test_parameters['retrieval_parameters']['intrinsic_temperature_bounds'][0],
                test_parameters['retrieval_parameters']['intrinsic_temperature_bounds'][1]
            ],  # (K)
            'prior_type': 'uniform',
            'figure_title': r'T',
            'figure_label': r'T (K)'
        },
        'CO-NatAbund': {
            'prior_parameters': [
                test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][0],
                test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][1]
            ],  # (MMR)
            'prior_type': 'uniform',
            'figure_title': r'[CO]',
            'figure_label': r'$\log_{10}$(MMR) CO'
        },
        'H2O': {
            'prior_parameters': [
                test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][0],
                test_parameters['retrieval_parameters']['log10_species_mass_fractions_bounds'][1]
            ],  # (MMR)
            'prior_type': 'uniform',
            'figure_title': r'[H$_2$O]',
            'figure_label': r'$\log_{10}$(MMR) H$_2$O'
        },
        'log10_opaque_cloud_top_pressure': {
            'prior_parameters': [
                test_parameters['retrieval_parameters']['log10_cloud_pressure_bounds'][0],
                test_parameters['retrieval_parameters']['log10_cloud_pressure_bounds'][1]
            ],  # (bar)
            'prior_type': 'uniform',
            'figure_title': r'[$P_c$]',
            'figure_label': r'$\log_{10}(P_c)$ ([Pa])',
            'figure_offset': 5  # [bar] to [Pa]
        },
        'radial_velocity_semi_amplitude': {
            'prior_parameters': np.array([
                test_parameters['retrieval_parameters']['radial_velocity_semi_amplitude_bounds'][0],
                test_parameters['retrieval_parameters']['radial_velocity_semi_amplitude_bounds'][1],
            ]),  # (cm.s-1)
            'prior_type': 'uniform',
            'figure_title': r'$K_p$',
            'figure_label': r'$K_p$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5  # cm.s-1 to km.s-1
        },
        'rest_frame_velocity_shift': {
            'prior_parameters': np.array([
                test_parameters['retrieval_parameters']['rest_frame_velocity_shift_bounds'][0],
                test_parameters['retrieval_parameters']['rest_frame_velocity_shift_bounds'][1],
            ]),  # (cm.s-1)
            'prior_type': 'uniform',
            'figure_title': r'$V_\mathrm{rest}$',
            'figure_label': r'$V_\mathrm{rest}$ (km$\cdot$s$^{-1}$)',
            'figure_coefficient': 1e-5  # cm.s-1 to km.s-1
        },
        'mid_transit_time': {
            'prior_parameters': np.array([
                test_parameters['retrieval_parameters']['mid_transit_time_bounds'][0],
                test_parameters['retrieval_parameters']['mid_transit_time_bounds'][1],
            ]),  # (cm.s-1)
            'prior_type': 'uniform',
            'figure_title': r'$T_0$',
            'figure_label': r'$T_0$ (s)'
        }
    }

    rebinned_wavelengths = petitRADTRANS.math.resolving_space(
        test_parameters['spectrum_parameters']['wavelength_range_line_by_line'][0] * 1e-4,
        test_parameters['spectrum_parameters']['wavelength_range_line_by_line'][1] * 1e-4,
        test_parameters['mock_observation_parameters']['high_resolution_resolving_power']
    )
    data_uncertainties = np.ones(rebinned_wavelengths.shape)

    mass_fractions = {
        species: mass_fraction
        for species, mass_fraction in test_parameters['mass_fractions_line_by_line'].items()
        if species not in test_parameters['filling_species']
    }

    forward_model = petitRADTRANS.spectral_model.SpectralModel(
        # Radtrans parameters
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        line_opacity_mode='lbl',
        # SpectralModel parameters
        # Planet parameters
        planet_radius=test_parameters[
                          'planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,  # cm
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,  # cm
        transit_duration=test_parameters['planetary_parameters']['transit_duration'],
        orbital_period=test_parameters['planetary_parameters']['orbital_period'],
        # Velocity paramters
        star_mass=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.m_sun,  # g
        orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],  # cm
        orbital_inclination=test_parameters['planetary_parameters']['orbital_inclination'],
        rest_frame_velocity_shift=test_parameters['mock_observation_parameters'][
            "rest_frame_velocity_shift"],  # cm.s-1
        system_observer_radial_velocities=np.linspace(
            test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][0],
            test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][1],
            test_parameters['mock_observation_parameters']['n_exposures']
        ),  # cm.s-1
        # Temperature profile parameters
        temperature_profile_mode='isothermal',
        temperature=temperature_isothermal,
        # Mass fractions
        use_equilibrium_chemistry=False,
        imposed_mass_fractions=mass_fractions,
        filling_species=test_parameters['filling_species'],
        # Observation parameters
        rebinned_wavelengths=rebinned_wavelengths,
        rebin_range_margin_power=3,  # TODO this should not be necessary
        # (cm) used for the rebinning, and also to set the wavelengths boundaries
        convolve_resolving_power=test_parameters['mock_observation_parameters']['high_resolution_resolving_power'],
        mid_transit_time=0,
        times=2 * test_parameters['planetary_parameters']['transit_duration'] * (
                np.linspace(0, 1, test_parameters['mock_observation_parameters']['n_exposures']) - 0.5),
        # Preparation parameters
        tellurics_mask_threshold=test_parameters['preparing_parameters']['tellurics_mask_threshold'],
        polynomial_fit_degree=test_parameters['preparing_parameters']['polynomial_fit_degree'],
        uncertainties=data_uncertainties
    )

    data_wavelengths, data = forward_model.calculate_spectrum(
        mode='transmission',
        update_parameters=True,  # the parameters that we set will be properly initialised
        telluric_transmittances_wavelengths=None,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,  # no noise so we can see the lines
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False  # if the data were loaded, a preparation could not be done rigth away
    )

    # The data and its uncertainties must be masked arrays
    data = np.ma.masked_array(data)
    data_uncertainties = 1e-3 * np.ma.ones(data.shape)

    data.mask = np.zeros(data.shape, dtype=bool)
    data_uncertainties.mask = np.zeros(data_uncertainties.shape, dtype=bool)

    forward_model.model_parameters['uncertainties'] = data_uncertainties

    prepared_data, preparation_matrix, prepared_data_uncertainties = petitRADTRANS.retrieval.preparing.polyfit(
        spectrum=data,
        uncertainties=data_uncertainties,
        wavelengths=data_wavelengths,
        airmass=None,
        tellurics_mask_threshold=forward_model.model_parameters['tellurics_mask_threshold'],
        polynomial_fit_degree=forward_model.model_parameters['polynomial_fit_degree'],
        full=True,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True
    )

    data = {  # multiple data can be retrieved by adding multiple keys
        'data_1': forward_model.init_data(
            # Data parameters
            data_spectrum=prepared_data,
            data_wavelengths=data_wavelengths,
            data_uncertainties=data_uncertainties,
            data_name='data_1',
            # Retrieved parameters
            retrieved_parameters=retrieved_parameters,
            # Forward model post-processing parameters
            mode='transmission',
            update_parameters=True,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            prepare=True
        )
    }

    retrieval_name = 'test_spectral_model'
    retrieval_directory = tests_results_directory

    _retrieval = petitRADTRANS.retrieval.Retrieval.from_data(
        data=data,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        output_directory=retrieval_directory,
        run_mode='retrieval'
    )

    return _retrieval


retrieval = init_retrieval()


def test_spectral_model_retrieval():
    retrieval.run(
        sampling_efficiency=test_parameters['retrieval_parameters']['sampling_efficiency'],
        n_live_points=test_parameters['retrieval_parameters']['n_live_points_spectral_model'],
        const_efficiency_mode=test_parameters['retrieval_parameters']['const_efficiency_mode'],
        resume=test_parameters['retrieval_parameters']['resume'],
        seed=test_parameters['retrieval_parameters']['seed']
    )

    # Get results and reference
    with open(reference_filenames['pymultinest_parameter_analysis_spectral_model']) as f:
        reference = json.load(f)

    new_result_file = os.path.join(
        tests_results_directory,
        'out_PMN',
        os.path.basename(reference_filenames['pymultinest_parameter_analysis_spectral_model'])
    )

    with open(new_result_file) as f:
        new_results = json.load(f)

    # Check if retrieved parameters are in +/- 1 sigma of the previous retrieved parameters
    for i, marginal in enumerate(new_results['marginals']):
        assert (
            marginal['median'] - relative_tolerance * reference['marginals'][i]['sigma']
            <= reference['marginals'][i]['median']
            <= marginal['median'] + relative_tolerance * reference['marginals'][i]['sigma']
        )
