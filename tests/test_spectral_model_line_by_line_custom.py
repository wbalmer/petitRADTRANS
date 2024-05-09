import copy
import warnings

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def init_spectral_model_line_by_line():
    with warnings.catch_warnings():
        # Expect UserWarning caused by yet_another_useless_parameter42, SpectralModel should work fine
        warnings.filterwarnings("ignore", category=UserWarning)

        spectral_model = petitRADTRANS.spectral_model.SpectralModel(
            # Radtrans object parameters
            pressures=test_parameters['pressures'],  # bar
            line_species=test_parameters['spectrum_parameters']['line_species_line_by_line'],
            rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
            gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
            scattering_in_emission=False,  # is False by default on Radtrans but True by default on SpectralModel
            line_opacity_mode='lbl',
            # Temperature profile parameters: generate a Guillot temperature profile
            temperature_profile_mode='guillot',
            temperature=test_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature'],  # K
            intrinsic_temperature=test_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
            metallicity=10 ** test_parameters['chemical_parameters']['metallicities'][1],
            guillot_temperature_profile_gamma=test_parameters['temperature_guillot_2010_parameters']['gamma'],
            guillot_temperature_profile_kappa_ir_z0=test_parameters['temperature_guillot_2010_parameters'][
                'infrared_mean_opacity'],
            # Chemical parameters
            use_equilibrium_chemistry=False,
            imposed_mass_fractions=test_parameters['mass_fractions_line_by_line'],
            # Transmission spectrum parameters (radtrans.calc_transm)
            planet_radius=test_parameters[
                'planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,  # cm
            reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
            reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],  # bar
            # cloud_pressure=1e-1,
            # Instrument parameters
            convolve_resolving_power=test_parameters['mock_observation_parameters']['high_resolution_resolving_power'],
            rebinned_wavelengths=np.arange(
                test_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][0] * 1e-4,
                test_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][1] * 1e-4,
                test_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][0] * 1e-4 /
                test_parameters['mock_observation_parameters']['high_resolution_resolving_power'] / 2
            ),  # um
            wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
            # Scaling parameters
            star_radius=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,  # cm
            # Orbital parameters
            star_mass=test_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.m_sun,  # g
            orbit_semi_major_axis=test_parameters['planetary_parameters']['orbit_semi_major_axis'],  # cm
            orbital_longitudes=np.linspace(
                test_parameters['mock_observation_parameters']['orbital_phase_range'][0] * 360,
                test_parameters['mock_observation_parameters']['orbital_phase_range'][1] * 360,
                test_parameters['mock_observation_parameters']['n_exposures']
            ),
            system_observer_radial_velocities=np.linspace(
                test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][0],
                test_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][1],
                test_parameters['mock_observation_parameters']['n_exposures']
            ),  # cm.s-1
            rest_frame_velocity_shift=test_parameters['mock_observation_parameters'][
                "rest_frame_velocity_shift"],  # cm.s-1
            # Reprocessing parameters
            # uncertainties=model_uncertainties,
            # airmass=airmasses,
            # tellurics_mask_threshold=tellurics_mask_threshold,
            # polynomial_fit_degree=2,
            # apply_throughput_removal=True,
            # apply_telluric_lines_removal=True,
            # Special parameters
            # Test addition of a useless parameter
            yet_another_useless_parameter42="irrelevant string"
        )

    def calculate_mean_molar_masses(pressures, **kwargs):
        return test_parameters['mean_molar_mass'] * np.ones(pressures.size)

    def calculate_mass_mixing_ratios(pressures, **kwargs):
        """Template for mass mixing ratio profile function.
        Here, generate iso-abundant mass mixing ratios profiles.

        Args:
            pressures: (bar) pressures of the temperature profile
            **kwargs: other parameters needed to generate the temperature profile

        Returns:
            A 1D-array containing the temperatures as a function of pressures
        """
        return {
            species: mass_mixing_ratio * np.ones(np.size(pressures))
            for species, mass_mixing_ratio in kwargs['imposed_mass_fractions'].items()
        }

    # Test custom function
    spectral_model.compute_mass_fractions = calculate_mass_mixing_ratios
    spectral_model.compute_mean_molar_masses = calculate_mean_molar_masses
    spectral_model.update_model_functions_map()

    return spectral_model


spectral_model_lbl = init_spectral_model_line_by_line()


def test_line_by_line_spectral_model_emission():
    spectral_model = copy.deepcopy(spectral_model_lbl)
    spectral_model.model_parameters['is_orbiting'] = False

    benchmark = Benchmark(
        function=spectral_model.calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='emission',
        parameters=None,
        update_parameters=True,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=False,
        shift=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    assert np.allclose(
        spectral_model.temperatures,
        temperature_guillot_2010,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_fractions[species],
            test_parameters['mass_fractions_line_by_line'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        test_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )


def test_line_by_line_spectral_model_transmission():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    spectral_model.model_parameters['temperature_profile_mode'] = 'isothermal'
    spectral_model.model_parameters['temperature'] = temperature_isothermal

    benchmark = Benchmark(
        function=spectral_model.calculate_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mode='transmission',
        parameters=None,
        update_parameters=True,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=False,
        shift=False,
        convolve=False,
        rebin=False,
        prepare=False
    )

    assert np.allclose(
        spectral_model.temperatures,
        temperature_isothermal,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_fractions[species],
            test_parameters['mass_fractions_line_by_line'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        test_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )


def test_line_by_line_spectral_model_transmission_ccf():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    spectral_model.model_parameters['temperature_profile_mode'] = 'isothermal'
    spectral_model.model_parameters['temperature'] = temperature_isothermal

    def _calculate_mock_transmission_data():
        _test_case = Benchmark(
            function=spectral_model.calculate_spectrum,
            relative_tolerance=relative_tolerance
        )

        _test_case.run(
            mode='transmission',
            parameters=None,
            update_parameters=True,
            telluric_transmittances=None,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=False
        )

        assert np.allclose(
            spectral_model.temperatures,
            temperature_isothermal,
            atol=0,
            rtol=relative_tolerance
        )

        for species in spectral_model.line_species:
            assert np.allclose(
                spectral_model.mass_fractions[species],
                test_parameters['mass_fractions_line_by_line'][species],
                atol=0,
                rtol=relative_tolerance
            )

        assert np.allclose(
            spectral_model.mean_molar_masses,
            test_parameters['mean_molar_mass'],
            atol=0,
            rtol=relative_tolerance
        )

        return _test_case._run(
            mode='transmission',
            parameters=None,
            update_parameters=True,
            telluric_transmittances=None,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=False
        ).values()

    def _calculate_ccf():
        # Get models
        wavelengths_model, model = spectral_model.calculate_spectrum(
            mode='transmission',
            parameters=None,
            update_parameters=True,
            telluric_transmittances=None,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=False,
            prepare=False
        )

        # Cross-correlate spectrum with itself
        line_spread_function_fwhm = (
            petitRADTRANS.physical_constants.c
            / test_parameters['mock_observation_parameters']['high_resolution_resolving_power']
        )

        co_added_cross_correlations_snr, co_added_cross_correlations, \
            v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths \
            = petitRADTRANS.ccf.ccf.ccf_analysis(
                wavelengths_data=wavelengths,
                data=mock_transmission_data,
                wavelengths_model=wavelengths_model,
                model=model,
                velocities_ccf=None,
                model_velocities=spectral_model.model_parameters['relative_velocities'],
                normalize_ccf=test_parameters['ccf_analysis_parameters']['normalize_ccf'],
                calculate_ccf_snr=test_parameters['ccf_analysis_parameters']['calculate_ccf_snr'],
                ccf_sum_axes=test_parameters['ccf_analysis_parameters']['ccf_sum_axes'],
                radial_velocity_semi_amplitude=spectral_model.model_parameters['radial_velocity_semi_amplitude'],
                system_observer_radial_velocities=spectral_model.model_parameters['system_observer_radial_velocities'],
                orbital_longitudes=spectral_model.model_parameters['orbital_longitudes'],
                line_spread_function_fwhm=line_spread_function_fwhm,
                pixels_per_resolution_element=test_parameters['ccf_analysis_parameters'][
                    'pixels_per_resolution_element'],
                co_added_ccf_peak_width=line_spread_function_fwhm * test_parameters[
                    'ccf_analysis_parameters']['peak_lsf_factor'],
                velocity_interval_extension_factor=test_parameters['ccf_analysis_parameters'][
                    'velocity_interval_extension_factor'],
                kp_factor=test_parameters['ccf_analysis_parameters']['kp_factor'],
                n_kp=None,
                n_vr=None,
                radial_velocity_function=None
            )

        co_added_cross_correlations_max, max_kp, max_v_rest, n_around_peak \
            = petitRADTRANS.ccf.ccf.get_co_added_ccf_peak_properties(
                co_added_cross_correlation=co_added_cross_correlations,
                kp_space=kps,
                vr_space=v_rest,
                peak_cutoff=test_parameters['ccf_analysis_parameters']['peak_cutoff']
            )

        return v_rest, kps, co_added_cross_correlations, max_kp, max_v_rest, n_around_peak

    wavelengths, mock_transmission_data = _calculate_mock_transmission_data()

    benchmark = Benchmark(
        function=_calculate_ccf,
        relative_tolerance=relative_tolerance
    )

    benchmark.run()
