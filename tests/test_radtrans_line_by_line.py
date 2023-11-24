"""Test the radtrans module in line-by-line mode.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=radtrans_parameters['pressures'],
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        wavelength_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        line_opacity_mode='lbl'
    )

    return atmosphere


def init_spectral_model_line_by_line():
    spectral_model = petitRADTRANS.spectral_model.SpectralModel(
        # Radtrans object parameters
        pressures=radtrans_parameters['pressures'],  # bar
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        scattering_in_emission=False,  # is False by default on Radtrans but True by default on SpectralModel
        line_opacity_mode='lbl',
        # Temperature profile parameters: generate a Guillot temperature profile
        temperature_profile_mode='guillot',
        temperature=radtrans_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature'],  # K
        intrinsic_temperature=radtrans_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
        metallicity=1.0,
        guillot_temperature_profile_gamma=radtrans_parameters['temperature_guillot_2010_parameters']['gamma'],
        guillot_temperature_profile_kappa_ir_z0=radtrans_parameters['temperature_guillot_2010_parameters'][
            'infrared_mean_opacity'],
        # Chemical parameters
        use_equilibrium_chemistry=False,
        imposed_mass_fractions=radtrans_parameters['mass_fractions'],
        # Transmission spectrum parameters (radtrans.calc_transm)
        planet_radius=radtrans_parameters['planetary_parameters']['radius']
        * petitRADTRANS.physical_constants.r_jup_mean,  # cm
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        reference_pressure=radtrans_parameters['planetary_parameters']['reference_pressure'],  # bar
        # cloud_pressure=1e-1,
        # Instrument parameters
        convolve_resolving_power=radtrans_parameters['mock_observation_parameters']['high_resolution_resolving_power'],
        rebinned_wavelengths=np.arange(
            radtrans_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][0],
            radtrans_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][1],
            radtrans_parameters['mock_observation_parameters']['wavelength_range_high_resolution'][0] /
            radtrans_parameters['mock_observation_parameters']['high_resolution_resolving_power'] / 2
        ),  # um
        wavelength_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        # Scaling parameters
        star_radius=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,  # cm
        # Orbital parameters
        star_mass=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.m_sun,  # g
        orbit_semi_major_axis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],  # cm
        orbital_longitudes=np.linspace(
            radtrans_parameters['mock_observation_parameters']['orbital_phase_range'][0] * 360,
            radtrans_parameters['mock_observation_parameters']['orbital_phase_range'][1] * 360,
            radtrans_parameters['mock_observation_parameters']['n_exposures']
        ),
        system_observer_radial_velocities=np.linspace(
            radtrans_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][0],
            radtrans_parameters['mock_observation_parameters']['system_observer_radial_velocities_range'][1],
            radtrans_parameters['mock_observation_parameters']['n_exposures']
        ),  # cm.s-1
        rest_frame_velocity_shift=radtrans_parameters['mock_observation_parameters'][
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
        return radtrans_parameters['mean_molar_mass'] * np.ones(pressures.size)

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
    spectral_model.compute_mass_fractions = \
        calculate_mass_mixing_ratios
    spectral_model.compute_mean_molar_masses = \
        calculate_mean_molar_masses

    return spectral_model


atmosphere_lbl = init_radtrans_line_by_line()
spectral_model_lbl = init_spectral_model_line_by_line()


def test_line_by_line_emission_spectrum():
    # Calculate an emission spectrum
    frequencies, flux, _ = atmosphere_lbl.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=radtrans_parameters['mass_fractions'],
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_transmission_spectrum():
    # Calculate a transmission spectrum
    frequencies, transit_radii, _ = atmosphere_lbl.calculate_transit_radii(
        temperatures=temperature_isothermal,
        mass_fractions=radtrans_parameters['mass_fractions'],
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        planet_radius=radtrans_parameters['planetary_parameters']['radius']
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=radtrans_parameters['planetary_parameters']['reference_pressure'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'transit_radius': transit_radii / petitRADTRANS.physical_constants.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_spectral_model_emission():
    spectral_model = copy.deepcopy(spectral_model_lbl)
    spectral_model.model_parameters['is_orbiting'] = False

    wavelengths, spectral_radiosities = spectral_model.calculate_spectrum(
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

    wavelengths *= 1e4
    spectral_radiosities *= 1e-7

    assert np.allclose(
        spectral_model.temperatures,
        temperature_guillot_2010,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_fractions[species],
            radtrans_parameters['mass_fractions'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        radtrans_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_emission'],
        comparison_dict={
            'wavelength': wavelengths,
            'spectral_radiosity': petitRADTRANS.physics.flux_cm2flux_hz(
                flux_cm=spectral_radiosities * 1e7,  # W to erg
                wavelength=wavelengths * 1e-4  # um to cm
            )
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_spectral_model_transmission():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    spectral_model.model_parameters['temperature_profile_mode'] = 'isothermal'
    spectral_model.model_parameters['temperature'] = temperature_isothermal

    wavelengths, transit_radii = spectral_model.calculate_spectrum(
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

    wavelengths *= 1e4

    assert np.allclose(
        spectral_model.temperatures,
        temperature_isothermal,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_fractions[species],
            radtrans_parameters['mass_fractions'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        radtrans_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission'],
        comparison_dict={
            'wavelength': wavelengths,
            'transit_radius': transit_radii / petitRADTRANS.physical_constants.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_spectral_model_transmission_ccf():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    spectral_model.model_parameters['temperature_profile_mode'] = 'isothermal'
    spectral_model.model_parameters['temperature'] = temperature_isothermal

    wavelengths, mock_transmission_data = spectral_model.calculate_spectrum(
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

    wavelengths *= 1e4

    assert np.allclose(
        spectral_model.temperatures,
        temperature_isothermal,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_fractions[species],
            radtrans_parameters['mass_fractions'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        radtrans_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission_2d'],
        comparison_dict={
            'wavelengths': wavelengths,
            'relative_velocities': spectral_model.model_parameters['relative_velocities'],
            'transit_radii': mock_transmission_data
        },
        relative_tolerance=relative_tolerance
    )

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

    wavelengths_model *= 1e4

    # Cross-correlate spectrum with itself
    line_spread_function_fwhm = petitRADTRANS.physical_constants.c /\
        radtrans_parameters['mock_observation_parameters']['high_resolution_resolving_power']

    co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths \
        = petitRADTRANS.ccf.ccf.ccf_analysis(
            wavelengths_data=wavelengths,
            data=mock_transmission_data,
            wavelengths_model=wavelengths_model,
            model=model,
            velocities_ccf=None,
            model_velocities=spectral_model.model_parameters['relative_velocities'],
            normalize_ccf=radtrans_parameters['ccf_analysis_parameters']['normalize_ccf'],
            calculate_ccf_snr=radtrans_parameters['ccf_analysis_parameters']['calculate_ccf_snr'],
            ccf_sum_axes=radtrans_parameters['ccf_analysis_parameters']['ccf_sum_axes'],
            radial_velocity_semi_amplitude=spectral_model.model_parameters['radial_velocity_semi_amplitude'],
            system_observer_radial_velocities=spectral_model.model_parameters['system_observer_radial_velocities'],
            orbital_longitudes=spectral_model.model_parameters['orbital_longitudes'],
            orbital_inclination=spectral_model.model_parameters['orbital_inclination'],
            line_spread_function_fwhm=line_spread_function_fwhm,
            pixels_per_resolution_element=radtrans_parameters['ccf_analysis_parameters'][
                'pixels_per_resolution_element'],
            co_added_ccf_peak_width=line_spread_function_fwhm * radtrans_parameters[
                'ccf_analysis_parameters']['peak_lsf_factor'],
            velocity_interval_extension_factor=radtrans_parameters['ccf_analysis_parameters'][
                'velocity_interval_extension_factor'],
            kp_factor=radtrans_parameters['ccf_analysis_parameters']['kp_factor'],
            n_kp=None,
            n_vr=None,
            radial_velocity_function=None
        )

    co_added_cross_correlations_max, max_kp, max_v_rest, n_around_peak \
        = petitRADTRANS.ccf.ccf.get_co_added_ccf_peak_properties(
            co_added_cross_correlation=co_added_cross_correlations,
            kp_space=kps,
            vr_space=v_rest,
            peak_cutoff=radtrans_parameters['ccf_analysis_parameters']['peak_cutoff']
        )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['co_added_cross_correlation'],
        comparison_dict={
            'rest_velocities': v_rest,
            'orbital_radial_velocity_amplitudes': kps,
            'co_added_cross_correlations': co_added_cross_correlations,
            'max_kp': max_kp,
            'max_v_rest': max_v_rest,
            'n_around_peak': n_around_peak
        },
        relative_tolerance=3e-4  # TODO put back relative_tolerance here (loading from HDF5 add numerical noise (1e-15) to wavelengths, spreading here to create relative difference > 1e-6; reference files should be re-generated for 3.0.0) # noqa: E501
        # a89: multiplied by 3, caused by the change (*1e-4) in output_wavelengths
    )
