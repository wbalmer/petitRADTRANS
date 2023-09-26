import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from scipy.interpolate import interp1d
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.physics import doppler_shift
from petitRADTRANS.ccf.ccf_core import cross_correlate, co_add_cross_correlation
from petitRADTRANS.ccf.ccf import ccf_analysis, get_ccf_velocity_space, get_co_added_ccf_velocity_space
from petitRADTRANS.containers.planet import Planet
from petitRADTRANS.containers.spectral_model import SpectralModel
from petitRADTRANS.retrieval.preparing import preparing_pipeline, preparing_pipeline_sysrem, trim_spectrum
from scripts.load_spectral_matrix import construct_spectral_matrix
from petitRADTRANS.cli.eso_skycalc_cli import get_tellurics_npz


module_dir = os.path.abspath(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data')


def _ccf_analysis(ccf_norm_select, detector_selection, rvs, orbital_phases, v_sys, kp,
                  kp_range, v_rest_range, max_area, max_percentage_area, area_increase_tolerance=1.1):
    ccf_norm_select = ccf_norm_select[detector_selection]
    ccf_norm_select_sum = np.ma.sum(ccf_norm_select, axis=0)

    # Collapsed CCF
    ccf_tot, v_rests, kps = simple_co_added_ccf(np.array([ccf_norm_select_sum]), rvs, orbital_phases, v_sys, kp)

    wh_max = np.where(ccf_tot == np.max(ccf_tot))
    max_kp_ccf = kps[wh_max[1]][0]
    max_v_rest_ccf = v_rests[wh_max[2]][0]

    if not (kp_range[0] <= max_kp_ccf <= kp_range[1]) or not (v_rest_range[0] <= max_v_rest_ccf <= v_rest_range[1]):
        return np.inf, max_kp_ccf, max_v_rest_ccf, ccf_tot, 0, np.zeros(ccf_tot.shape), \
            np.inf, max_kp_ccf, max_v_rest_ccf

    area = np.size(np.where(ccf_tot[0] >= max_percentage_area * np.max(ccf_tot[0])))

    if area > max_area * area_increase_tolerance:
        return np.inf, max_kp_ccf, max_v_rest_ccf, ccf_tot, area, np.zeros(ccf_tot.shape), \
            np.inf, max_kp_ccf, max_v_rest_ccf

    # Calculate S/N
    ccf_tot_sn = np.zeros(ccf_tot.shape)
    exclude = 15.6  # exclusion region in km/s (+/-):
    max_sn = 0
    max_kp = 0
    max_v_rest = 0

    for k, ccf_tot_kp in enumerate(ccf_tot[0]):
        # Finding the maximum in the CCF, wherever it may be located:
        aux = np.argmax(ccf_tot_kp)

        # Select the v_rest range far from "detected" signal
        std_pts_a = np.where(v_rests < (v_rests[aux] - exclude))[0]
        std_pts_b = np.where(v_rests > (v_rests[aux] + exclude))[0]
        std_pts = np.concatenate((std_pts_a, std_pts_b))

        # Compute the S/N
        ccf_tot_sn[0, k, :] = ccf_tot_kp / np.std(ccf_tot_kp[std_pts])

        id_max = np.argmax(ccf_tot_sn[0, k, :])

        if ccf_tot_sn[0, k, id_max] > max_sn:
            max_sn = ccf_tot_sn[0, k, id_max]
            max_kp = kps[k]
            max_v_rest = v_rests[id_max]

    max_sn_peak = ccf_tot_sn[wh_max][0]

    return max_sn, max_kp, max_v_rest, ccf_tot, area, ccf_tot_sn, max_sn_peak, max_kp_ccf, max_v_rest_ccf


def _test_rico(node='B'):
    data_dir = r"C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data\crires\hd_209458_b\hd209_crires_v3\2022_10_12\correct_wavelengths"
    rico_loaded_data = np.load(os.path.join(data_dir, f'rico_loaded_data_{node}.npz'))

    wavelengths_instrument_rico = rico_loaded_data['wavelengths']
    observations_rico = rico_loaded_data['spec']
    uncertainties_rico = rico_loaded_data['errors']
    dates_rico = rico_loaded_data['obstimes']

    sorted_orders = np.argsort(wavelengths_instrument_rico[0, :, 0])
    wavelengths_instrument_rico_ = np.moveaxis(wavelengths_instrument_rico[:, sorted_orders], 1, 0) * 1e-3  # nm to um
    observations_rico_ = np.moveaxis(observations_rico[:, sorted_orders], 1, 0)
    uncertainties_rico_ = np.moveaxis(uncertainties_rico[:, sorted_orders], 1, 0)
    dates_rico_ = dates_rico

    truncate = 5
    wavelengths_instrument, observations, uncertainties, dates = load_rico_data(
        data_directory=data_dir,
        interpolate_to_common_wl=False,
        nodes='B',
        truncate=truncate
    )

    wh_keep = []
    j = 0

    for i, w in enumerate(wavelengths_instrument):
        if np.allclose(w[0, 0], wavelengths_instrument_rico_[j, 0, 0], rtol=1e-3, atol=0):
            wh_keep.append(i)
            j += 1

    wavelengths_instrument = copy.deepcopy(wavelengths_instrument_rico_[:, :, 5+truncate:-truncate])  # TODO wavelengths are incorrect?
    observations = observations[wh_keep]
    uncertainties = uncertainties[wh_keep]
    dates = copy.deepcopy(dates_rico_)  # TODO 1e-5 absolute diff with Rico's time
    times = (dates - np.floor(dates[0])) * cst.s_cst.day

    planet = Planet.get('HD 209458 b')
    lsf_fwhm = 1.9e5
    pixels_per_resolution_element = 2
    kp_factor = 1.5
    extra_factor = -0.25

    # Exofop 20230725 parameters
    epoch = 2459826.781018  # BJD (day)
    epoch_error = 0.00006454682  # BJD (day)
    planet.orbital_period = 3.5247404585539 * cst.s_cst.day
    planet.orbital_period_error_lower = 0.000015326508 * cst.s_cst.day
    planet.orbital_period_error_upper = 0.000015326508 * cst.s_cst.day

    mid_transit_time = planet.calculate_mid_transit_time_from_source(
        np.floor(dates[0]), epoch, epoch_error, epoch_error, planet.orbital_period / cst.s_cst.day,
        planet.orbital_period_error_lower / cst.s_cst.day, planet.orbital_period_error_upper / cst.s_cst.day
    )[0]  # +/- 15.6 s (20230725)

    orbital_phases = (times - mid_transit_time) / planet.orbital_period

    airmass = planet.get_airmass(
        ra=planet.ra,
        dec=planet.dec,
        time=dates,
        site_name='Paranal',
        time_format='jd'
    )

    berv = planet.get_barycentric_velocities(
        ra=planet.ra,
        dec=planet.dec,
        time=dates,
        site_name='Paranal',
        time_format='jd'
    )

    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)
    v_sys = planet.star_radial_velocity - berv * 1e2

    observations = trim_spectrum(
        spectrum=observations,
        uncertainties=uncertainties,
        wavelengths=wavelengths_instrument[:, 0],
        airmass=airmass,
        threshold_low=0.8,
        threshold_high=1.2,
        threshold_outlier=4,
        polynomial_fit_degree=3,
        relative_to_continnum=True
    )
    uncertainties = np.ma.masked_where(observations.mask, uncertainties)

    ccf_velocities = ccf_radial_velocity(
        v_sys=v_sys,
        kp=kp,
        lsf_fwhm=lsf_fwhm,  # cm.s-1
        pixels_per_resolution_element=pixels_per_resolution_element,
        kp_factor=kp_factor,
        extra_factor=extra_factor
    )

    wavelengths, model, spectral_model, radtrans = get_model(
        planet, wavelengths_instrument, v_sys, np.array([-kp, kp]) * 1.5, ccf_velocities,
        times, mid_transit_time, airmass, uncertainties, wh_keep, 0.8,
        'transmission',
        scale=True, shift=False, use_transit_light_loss=False, convolve=True, rebin=False, reduce=False
    )

    rs_sys, ru_sys, rm_sys = preparing_pipeline_sysrem(
        spectrum=observations,
        uncertainties=uncertainties,
        wavelengths=wavelengths_instrument[:, 0],
        n_iterations_max=10,
        convergence_criterion=1e-3,
        tellurics_mask_threshold=0.8,
        polynomial_fit_degree=3,
        full=True
    )

    rs, ru, rm = preparing_pipeline(
        spectrum=observations,
        uncertainties=uncertainties,
        wavelengths=wavelengths_instrument[:, 0],
        airmass=airmass,
        tellurics_mask_threshold=0.9,
        polynomial_fit_degree=3,
        full=True
    )

    rico_loaded_prepared_data = np.load(os.path.join(data_dir, f'rico_prepared_data_{node}.npz'))

    rs_rico = rico_loaded_prepared_data['before_ccf_spec']

    rs_rico = np.moveaxis(rs_rico, 1, 0)[sorted_orders]
    rs_rico = rs_rico[:, :, truncate+5:-truncate]

    co_added_cross_correlations_snr_r, co_added_cross_correlations_r, \
        v_rest_r, kps_r, ccf_sum_r, ccfs_r, velocities_cc_rf, ccf_models_r, ccf_model_wavelengths_r = \
        ccf_analysis(
            wavelengths_data=wavelengths_instrument,
            data=np.ma.masked_invalid(rs_rico),
            wavelengths_model=wavelengths[0],
            model=model[0],
            velocities_ccf=None,
            model_velocities=None,
            normalize_ccf=True,
            calculate_ccf_snr=True,
            ccf_sum_axes=None,
            planet_radial_velocity_amplitude=kp,
            system_observer_radial_velocities=v_sys,
            orbital_longitudes=np.rad2deg(orbital_phases * 2 * np.pi),
            planet_orbital_inclination=planet.orbital_inclination,
            line_spread_function_fwhm=lsf_fwhm,
            pixels_per_resolution_element=pixels_per_resolution_element,
            co_added_ccf_peak_width=None,
            velocity_interval_extension_factor=extra_factor,
            kp_factor=kp_factor
        )


def ccf_radial_velocity(v_sys, kp, lsf_fwhm, pixels_per_resolution_element, kp_factor=1.0, extra_factor=0.25):
    # Calculate star_radial_velocity interval, add extra coefficient just to be sure
    # Effectively, we are moving along the spectral pixels
    radial_velocity_lag_min = np.min(v_sys) - kp * kp_factor
    radial_velocity_lag_max = np.max(v_sys) + kp * kp_factor
    radial_velocity_interval = radial_velocity_lag_max - radial_velocity_lag_min
    radial_velocity_lag_min -= extra_factor * radial_velocity_interval
    radial_velocity_lag_max += extra_factor * radial_velocity_interval

    lag_step = lsf_fwhm / pixels_per_resolution_element

    n_low = int(np.floor(radial_velocity_lag_min / lag_step))
    n_high = int(np.ceil(radial_velocity_lag_max / lag_step))

    radial_velocity_lag = np.linspace(
        n_low * lag_step,
        n_high * lag_step,
        n_high - n_low + 1
    )

    return radial_velocity_lag


def find_best_detector_selection(first_guess, detector_list, ccf_norm_select, rvs, orbital_phases, v_sys, kp, kp_range,
                                 v_rest_range, max_percentage_area=0.68, area_increase_tolerance=1.1, use_peak=True):
    detector_selection = first_guess
    ccf_norm_select_tmp = ccf_norm_select[detector_selection]
    ccf_norm_select_sum = np.ma.sum(ccf_norm_select_tmp, axis=0)

    # Collapsed CCF
    ccf_tot, v_rests, kps = simple_co_added_ccf(np.array([ccf_norm_select_sum]), rvs, orbital_phases, v_sys, kp)

    # Calculate S/N
    ccf_tot_sn = np.zeros(ccf_tot.shape)
    exclude = 15.6  # exclusion region in km/s (+/-):
    max_sn_sn = 0
    max_kp_sn = 0
    max_v_rest_sn = 0
    max_area = np.size(np.where(ccf_tot[0] >= max_percentage_area * np.max(ccf_tot[0])))

    for k, ccf_tot_kp in enumerate(ccf_tot[0]):
        # Finding the maximum in the CCF, wherever it may be located:
        aux = np.argmax(ccf_tot_kp)

        # Select the v_rest range far from "detected" signal
        std_pts_a = np.where(v_rests < (v_rests[aux] - exclude))[0]
        std_pts_b = np.where(v_rests > (v_rests[aux] + exclude))[0]
        std_pts = np.concatenate((std_pts_a, std_pts_b))

        # Compute the S/N
        ccf_tot_sn[0, k, :] = ccf_tot_kp / np.std(ccf_tot_kp[std_pts])

        id_max = np.argmax(ccf_tot_sn[0, k, :])

        if ccf_tot_sn[0, k, id_max] > max_sn_sn:
            max_sn_sn = ccf_tot_sn[0, k, id_max]
            max_kp_sn = kps[k]
            max_v_rest_sn = v_rests[id_max]

    wh_max = np.where(ccf_tot == np.max(ccf_tot))

    max_sn_peak = ccf_tot_sn[wh_max][0]
    max_kp_peak = kps[wh_max[1]][0]
    max_v_peak = v_rests[wh_max[2]][0]

    if use_peak:
        max_sn = max_sn_peak
        max_kp = max_kp_peak
        max_v_rest = max_v_peak
    else:
        max_sn = max_sn_sn
        max_kp = max_kp_sn
        max_v_rest = max_v_rest_sn

    if not (kp_range[0] <= max_kp <= kp_range[1]) or not (v_rest_range[0] <= max_v_rest <= v_rest_range[1]):
        raise ValueError(
            f'bad first guess: \n'
            f' max kp {max_kp};     kp range {kp_range}\n'
            f' max vr {max_v_rest}; v_rest range {v_rest_range}\n'
        )

    # Add
    for i in detector_list:
        if i not in detector_selection:
            print(f"Try adding detector {i}")
            detector_selection_tmp = np.append(detector_selection, [i])
            max_sn_sn_tmp, max_kp_sn_tmp, max_v_rest_sn_tmp, ccf_tot_tmp, area, ccf_tot_sn_tmp, \
                max_sn_peak_tmp, max_kp_peak_tmp, max_v_peak_tmp = ccf_analysis(
                    ccf_norm_select=ccf_norm_select,
                    detector_selection=detector_selection_tmp,
                    rvs=rvs,
                    orbital_phases=orbital_phases,
                    v_sys=v_sys,
                    kp=kp,
                    kp_range=kp_range,
                    v_rest_range=v_rest_range,
                    max_area=max_area,
                    max_percentage_area=max_percentage_area,
                    area_increase_tolerance=area_increase_tolerance
                )

            if use_peak:
                max_sn_tmp = max_sn_peak_tmp
                max_kp_tmp = max_kp_peak_tmp
                max_v_rest_tmp = max_v_peak_tmp
            else:
                max_sn_tmp = max_sn_sn_tmp
                max_kp_tmp = max_kp_sn_tmp
                max_v_rest_tmp = max_v_rest_sn_tmp

            if max_sn_tmp > max_sn:
                if area > max_area * area_increase_tolerance:
                    print(f"Rejecting detector {i}: "
                          f"max area increased ({area} > {max_area})")
                elif (kp_range[0] <= max_kp_tmp <= kp_range[1]) \
                        and (v_rest_range[0] <= max_v_rest_tmp <= v_rest_range[1]):
                    print(f"Added detector {i} (S/N: {max_sn_tmp}, kp: {max_kp_tmp}, v_rest: {max_v_rest_tmp})")
                    detector_selection = detector_selection_tmp
                    max_sn = max_sn_tmp
                    max_kp = max_kp_tmp
                    max_v_rest = max_v_rest_tmp
                    max_area = area
                    ccf_tot = ccf_tot_tmp
                    ccf_tot_sn = ccf_tot_sn_tmp
                    max_sn_sn = max_sn_sn_tmp
                    max_kp_sn = max_kp_sn_tmp
                    max_v_rest_sn = max_v_rest_sn_tmp
                    max_sn_peak = max_sn_peak_tmp
                    max_kp_peak = max_kp_peak_tmp
                    max_v_peak = max_v_peak_tmp
                else:
                    print(f"Rejecting detector {i}: "
                          f"better CCF S/N ({max_sn_tmp} > {max_sn}), but at invalid kp and/or v_rest:\n"
                          f"  kp: \t{max_kp_tmp}, must be within {kp_range}\n"
                          f"  v_rest: \t{max_v_rest_tmp}, must be within {v_rest_range}")
            else:
                print(f"Rejecting detector {i}: "
                      f"worst CCF S/N ({max_sn_tmp} < {max_sn})")
    # Remove
    for i in detector_list:
        if i in detector_selection:
            print(f"Try removing detector {i}")
            detector_selection_tmp = np.delete(detector_selection, np.where(detector_selection == i))
            max_sn_sn_tmp, max_kp_sn_tmp, max_v_rest_sn_tmp, ccf_tot_tmp, area, ccf_tot_sn_tmp, \
                max_sn_peak_tmp, max_kp_peak_tmp, max_v_peak_tmp = ccf_analysis(
                    ccf_norm_select=ccf_norm_select,
                    detector_selection=detector_selection_tmp,
                    rvs=rvs,
                    orbital_phases=orbital_phases,
                    v_sys=v_sys,
                    kp=kp,
                    kp_range=kp_range,
                    v_rest_range=v_rest_range,
                    max_area=max_area,
                    max_percentage_area=max_percentage_area,
                    area_increase_tolerance=area_increase_tolerance
                )

            if use_peak:
                max_sn_tmp = max_sn_peak_tmp
                max_kp_tmp = max_kp_peak_tmp
                max_v_rest_tmp = max_v_peak_tmp
            else:
                max_sn_tmp = max_sn_sn_tmp
                max_kp_tmp = max_kp_sn_tmp
                max_v_rest_tmp = max_v_rest_sn_tmp

            if max_sn_tmp > max_sn:
                if area > max_area * area_increase_tolerance:
                    print(f"Rejecting detector {i}: "
                          f"max area increased ({area} > {max_area})")
                elif (kp_range[0] <= max_kp_tmp <= kp_range[1]) \
                        and (v_rest_range[0] <= max_v_rest_tmp <= v_rest_range[1]):
                    print(f"Removed detector {i}")
                    detector_selection = detector_selection_tmp
                    max_sn = max_sn_tmp
                    max_kp = max_kp_tmp
                    max_v_rest = max_v_rest_tmp
                    max_area = area
                    ccf_tot = ccf_tot_tmp
                    ccf_tot_sn = ccf_tot_sn_tmp
                else:
                    print(f"Re-adding detector {i}: "
                          f"better CCF S/N ({max_sn_tmp} > {max_sn}), but at invalid kp and/or v_rest:\n"
                          f"  kp: \t{max_kp_tmp}, must be within {kp_range}\n"
                          f"  v_rest: \t{max_v_rest_tmp}, must be within {v_rest_range}")
            else:
                print(f"Re-adding detector {i}: "
                      f"worse CCF S/N ({max_sn_tmp} < {max_sn})")

    print(f"\nCCF S/N: {max_sn}\n"
          f"kp: {max_kp * 1e-5} km.s-1\n"
          f"v_rest: {max_v_rest * 1e-5} km.s-1")

    return ccf_tot, max_sn_sn, max_kp_sn, max_v_rest_sn, ccf_tot_sn, detector_selection, \
        max_sn_peak, max_kp_peak, max_v_peak, kps, v_rests


def get_model(planet, wavelengths_instrument, system_observer_radial_velocities, kp_range, ccf_velocities,
              times, mid_transit_time, airmass, uncertainties, detector_selection,
              tellurics_mask_threshold=0.5, mode='transmission',
              scale=False, shift=False, use_transit_light_loss=False, convolve=False, rebin=False, reduce=False):
    # Mock_observations
    print('Initializing model...')
    spectral_model = SpectralModel(
        # Radtrans object parameters
        pressures=np.logspace(-10, 2, 100),  # bar
        line_species=[
            # 'CH4_hargreaves_main_iso',
            'CO_all_iso',
            'H2O_main_iso',
            # 'H2S_main_iso',
            # 'HCN_main_iso',
            # 'NH3_main_iso'
        ],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        opacity_mode='lbl',
        lbl_opacity_sampling=4,
        # Temperature profile parameters
        temperature_profile_mode='isothermal',
        temperature=planet.equilibrium_temperature,  # K
        # Chemical parameters
        use_equilibrium_chemistry=False,
        imposed_mass_mixing_ratios={
            'CH4_hargreaves_main_iso': 3.4e-5,
            'CO_all_iso': 1.8e-2,
            'H2O_main_iso': 5.4e-3,
            'H2S_main_iso': 1.0e-3,
            'HCN_main_iso': 2.7e-7,
            'NH3_main_iso': 7.9e-6
        },
        fill_atmosphere=True,
        # Transmission spectrum parameters (radtrans.calc_transm)
        planet_radius=planet.radius,  # cm
        planet_surface_gravity=planet.reference_gravity,  # cm.s-2
        reference_pressure=1e-2,  # bar
        cloud_pressure=1e2,
        # haze_factor=1,
        # scattering_opacity_350nm=1e-6,
        # scattering_opacity_coefficient=-12,
        # Instrument parameters
        new_resolving_power=9.2e4,
        output_wavelengths=wavelengths_instrument[:, 0],  # um
        # Scaling parameters
        star_radius=planet.star_radius,  # cm
        # Orbital parameters
        times=times,
        star_mass=planet.star_mass,  # g
        orbit_semi_major_axis=planet.orbit_semi_major_axis,  # cm
        mid_transit_time=mid_transit_time,  # TODO +0
        orbital_period=planet.orbital_period,
        orbital_inclination=planet.orbital_inclination,
        transit_duration=planet.transit_duration,
        system_observer_radial_velocities=system_observer_radial_velocities,  # cm.s-1
        planet_rest_frame_velocity_shift=0.0,  # cm.s-1
        planet_orbital_inclination=planet.orbital_inclination,
        # Reprocessing parameters
        uncertainties=uncertainties,
        airmass=airmass,
        tellurics_mask_threshold=tellurics_mask_threshold,
        polynomial_fit_degree=2,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True,
        # Special parameters
        rebin_range_margin_power=3,
        constance_tolerance=1e300,  # force constant convolve
        detector_selection=detector_selection
    )

    retrieval_velocities = spectral_model.get_retrieval_velocities(
        planet_radial_velocity_amplitude_range=kp_range,
        planet_rest_frame_velocity_shift_range=(np.min(ccf_velocities), np.max(ccf_velocities)),
        mid_transit_times_range=(0, 0)
    )

    spectral_model.wavelengths_boundaries = spectral_model.get_optimal_wavelength_boundaries(
        relative_velocities=retrieval_velocities
    )
    print(kp_range, (np.min(ccf_velocities), np.max(ccf_velocities)), spectral_model.wavelengths_boundaries)

    radtrans = spectral_model.get_radtrans()

    wavelengths, model = spectral_model.get_spectrum_model(
        radtrans=radtrans,
        mode=mode,
        update_parameters=True,
        scale=scale,
        shift=shift,
        use_transit_light_loss=use_transit_light_loss,
        convolve=convolve,
        rebin=rebin,
        reduce=reduce
    )

    return wavelengths, model, spectral_model, radtrans


def load_variable_throughput_brogi(file, times_size, wavelengths_size):
    variable_throughput = np.load(file)

    variable_throughput = np.max(variable_throughput[0], axis=1)
    variable_throughput = variable_throughput / np.max(variable_throughput)

    xp = np.linspace(0, 1, np.size(variable_throughput))
    x = np.linspace(0, 1, times_size)
    variable_throughput = np.interp(x, xp, variable_throughput)

    return np.tile(variable_throughput, (wavelengths_size, 1)).T


def load_rico_data(data_directory, interpolate_to_common_wl, nodes='both',
                   truncate=5, mask_threshold=1e-15, snr_threshold=1, mask_of_mask_threshold=0.02):
    def __mask_columns_lines(matrix, mask_lines=False, mask_columns=False):
        for i, matrix_ccd in enumerate(matrix):
            # Mask lines where too many points are masked
            if mask_lines:
                for j, arr_exposure in enumerate(matrix_ccd):
                    if np.nonzero(arr_exposure.mask)[0].size / arr_exposure.mask.size > mask_of_mask_threshold:
                        matrix[i, j, :].mask = np.ones(matrix.shape[2], dtype=bool)

            # Mask columns where too many points are masked
            if mask_columns:
                for k, arr_wavelength in enumerate(matrix_ccd.T):
                    if np.nonzero(arr_wavelength.mask)[0].size / arr_wavelength.mask.size > mask_of_mask_threshold:
                        matrix[i, :, k].mask = np.ones(matrix.shape[1], dtype=bool)

        return matrix

    wavelengths_instrument, observations, uncertainties, times = construct_spectral_matrix(
        os.path.join(data_directory, ''),
        interpolate_to_common_wl=interpolate_to_common_wl,  # TODO shift happening here? Not the same results as Rico's
        nodes=nodes
    )

    times -= 0.5
    # times = (times - np.floor(times[0])) * cst.s_cst.day

    if truncate is not None:
        # mask = np.ones(wavelengths_instrument.shape, dtype=bool)
        # mask[:, :, truncate + 5:-truncate] = False
        observations = observations[:, :, truncate + 5:-truncate]
        uncertainties = uncertainties[:, :, truncate + 5:-truncate]
        wavelengths_instrument = wavelengths_instrument[:, :, truncate + 5:-truncate]

    nan_orders = np.all(np.all(np.isnan(observations), axis=-1), axis=-1)  # all wavelengths, all exposures

    if np.any(nan_orders):
        print(f"Removing all-nans orders {list(np.nonzero(nan_orders)[0])}...")
        wavelengths_instrument = np.delete(
            wavelengths_instrument,
            nan_orders,
            axis=0
        )
        observations = np.delete(
            observations,
            nan_orders,
            axis=0
        )
        uncertainties = np.delete(
            uncertainties,
            nan_orders,
            axis=0
        )

    # Mask invalid points
    observations = np.ma.masked_invalid(observations)
    uncertainties = np.ma.masked_invalid(uncertainties)

    # Propagate mask to observations and uncertainties
    observations = np.ma.masked_where(uncertainties.mask, observations)
    uncertainties = np.ma.masked_where(observations.mask, uncertainties)

    # Mask invalid lines
    observations = __mask_columns_lines(observations, mask_lines=True)
    uncertainties = __mask_columns_lines(uncertainties, mask_lines=True)

    # Mask points where SNR is too low
    snr_mask = np.less(observations / uncertainties, snr_threshold)

    observations = np.ma.masked_where(snr_mask, observations)
    uncertainties = np.ma.masked_where(snr_mask, uncertainties)

    # Mask observations that are too low
    mask_threshold = np.less(observations, mask_threshold)
    observations = np.ma.masked_where(mask_threshold, observations)
    uncertainties = np.ma.masked_where(mask_threshold, uncertainties)

    # Mask invalid columns
    observations = __mask_columns_lines(observations, mask_columns=True)
    uncertainties = __mask_columns_lines(uncertainties, mask_columns=True)

    # Propagate mask to observations and unceratainties
    observations = np.ma.masked_where(uncertainties.mask, observations)
    uncertainties = np.ma.masked_where(observations.mask, uncertainties)

    return wavelengths_instrument, observations, uncertainties, times


def load_jason_data(data_directory, n_transits=1, i0=1, quantile_mask=0.005):
    times = []
    wavelengths = []
    spectra = []
    uncertainties = []

    for i in range(n_transits):
        with open(os.path.join(os.path.abspath(data_directory), f'transit_{i + i0}.pickle'), 'rb') as f:
            data = pickle.load(f)
            times.append(np.array(data[0]))
            wavelengths.append(np.array(data[1]) * 1e-3)  # nm to um
            spectra.append(np.array(data[2]))
            uncertainties.append(np.array(data[3]))

    if n_transits == 1:
        times = times[0]
        wavelengths = wavelengths[0]
        spectra = spectra[0]
        uncertainties = uncertainties[0]

        order_cuts = np.nonzero(
            np.greater(np.diff(wavelengths[0]),
                       np.quantile(np.diff(wavelengths[0]), 0.99955))
        )[0]

        n_wavelengths = 2048
        n_times = wavelengths.shape[0]
        n_ccds = int(wavelengths.shape[1] / n_wavelengths)

        wavelengths = np.moveaxis(np.reshape(wavelengths, (n_times, n_ccds, n_wavelengths)), 1, 0)
        spectra = np.moveaxis(np.reshape(spectra, (n_times, n_ccds, n_wavelengths)), 1, 0)
        uncertainties = np.moveaxis(np.reshape(uncertainties, (n_times, n_ccds, n_wavelengths)), 1, 0)

        mask_threshold = np.quantile(spectra, quantile_mask)
        mask_threshold = np.less(spectra, mask_threshold)
        spectra = np.ma.masked_where(mask_threshold, spectra)
        uncertainties = np.ma.masked_where(mask_threshold, uncertainties)
        invalids = np.ma.masked_invalid(spectra * uncertainties).mask
        spectra = np.ma.masked_where(invalids, spectra)
        uncertainties = np.ma.masked_where(invalids, uncertainties)

    return wavelengths, spectra, uncertainties, times


def plot_ccf(ccf, x, y, true_coord=None):
    plt.imshow(ccf, aspect='auto', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])

    if true_coord is not None:
        plt.scatter(true_coord[0], true_coord[1], color='r', marker='+')


def plot_obs(wavelengths_instrument, orbital_phases, observations):
    nrows = int(np.ceil(observations.shape[0] / np.sqrt(observations.shape[0])))
    ncols = int(np.ceil(observations.shape[0] / nrows))

    fig_size = 6.4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size / ncols * nrows))
    data_i = -1

    for i in range(nrows):
        for j in range(ncols):
            data_i += 1

            if data_i >= observations.shape[0]:
                break

            axes[i, j].pcolormesh(
                wavelengths_instrument[data_i], orbital_phases, observations[data_i]
            )
            axes[i, j].set_title(f"CCD {data_i}")

    fig.tight_layout()


def plot_ccfs(rvs, orbital_phases, ccf):
    nrows = int(np.ceil(ccf.shape[0] / np.sqrt(ccf.shape[0])))
    ncols = int(np.ceil(ccf.shape[0] / nrows))

    fig_size = 6.4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size / ncols * nrows))
    data_i = -1

    for i in range(nrows):
        for j in range(ncols):
            data_i += 1

            if data_i >= ccf.shape[0]:
                break

            axes[i, j].pcolormesh(
                rvs, orbital_phases, ccf[data_i]
            )
            axes[i, j].set_title(f"CCD {data_i}")

    fig.tight_layout()


def get_data(planet, planet_transit_duration, dates, night, mid_transit_time, data_source='rico', nodes='both'):
    if data_source == 'rico':
        wavelengths_instrument, observations, uncertainties, times = load_rico_data(
            os.path.join(module_dir, 'crires', planet.name.lower().replace(' ', '_'),
                         'hd209_crires_v3', dates[night], 'correct_wavelengths'),
            interpolate_to_common_wl=True,
            nodes=nodes
        )
    elif data_source == 'rico1':
        wavelengths_instrument, observations, uncertainties, times = load_rico_data(
            os.path.join(module_dir, 'crires', planet.name.lower().replace(' ', '_'),
                         'hd209_crires_v1', dates[night], 'correct_wavelengths'),
            interpolate_to_common_wl=True,
            nodes=nodes
        )
    elif data_source == 'jason_v1':
        wavelengths_instrument, observations, uncertainties, times = load_jason_data(
                os.path.join(module_dir, 'crires', planet.name.lower().replace(' ', '_'),
                             'jason'),
                n_transits=3
            )
    else:
        raise ValueError(f"data_source must be 'jason' or 'rico', but was '{data_source}'")

    berv = planet.get_barycentric_velocities(
        ra=planet.ra,
        dec=planet.dec,
        time=times,
        site_name='Paranal'
    )

    airmass = planet.get_airmass(
        ra=planet.ra,
        dec=planet.dec,
        time=times,
        site_name='Paranal'
    )

    #times = (times - mid_transit_time) * cst.s_cst.day
    orbital_phases = planet.get_orbital_phases(0, planet.orbital_period, times)

    # wh = np.where(np.logical_and(times >= -planet_transit_duration / 2,
    #                              times <= planet_transit_duration / 2))[0]
    #
    # if len(wh) == 0:
    #     raise ValueError(f"no in-transit data found "
    #                      f"(times: [{np.min(times)}, {np.max(times)}], mid transit: {mid_transit_time})")
    #
    # wavelengths_instrument = wavelengths_instrument[:, wh, :]
    # observations = observations[:, wh, :]
    # uncertainties = uncertainties[:, wh, :]
    # times = times[wh]
    #
    # orbital_phases = orbital_phases[wh]
    # berv = berv[wh]
    # airmass = airmass[wh]

    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)
    v_sys = planet.star_radial_velocity - berv * 1e2

    return wavelengths_instrument, observations, uncertainties, times, berv, airmass, orbital_phases, v_sys, kp


def main():
    use_t23 = False  # use full eclipse transit time instead of total transit time
    use_t1535 = False
    load_rico = True
    load_jason = False
    remove_5sig_outsiders = True
    get_mock_obs = True
    planet_name = 'HD 209458 b'

    output_directory = os.path.abspath(os.path.abspath('./')
                                       + '../../../../work/run_outputs/petitRADTRANS')
    additional_data_directory = os.path.join(output_directory, 'data')

    planet = Planet.get(planet_name)
    night = 2
    lsf_fwhm = 1.9e5
    pixels_per_resolution_element = 2
    kp_factor = 1.5
    extra_factor = -0.25

    tellurics_mask_threshold = 0.5

    dates = [
        '2022_06_29',
        '2022_07_06',
        '2022_10_12',
    ]

    mid_transit_time = [
        59759.8109 - 0.5,
        59766.8604 - 0.5,
        59865.5532 - 0.5
    ]  # https://astro.swarthmore.edu/transits/print_transits.cgi?single_object=0&ra=&dec=&epoch=&period=&duration=&depth=&target=&observatory_string=-24.625%3B-70.403333%3BAmerica%2FSantiago%3BEuropean+Southern+Observatory%3A+Paranal&use_utc=0&observatory_latitude=37.223611&observatory_longitude=-2.54625&timezone=UTC&start_date=05-29-2022&days_to_print=180&days_in_past=0&minimum_start_elevation=30&and_vs_or=or&minimum_end_elevation=30&minimum_ha=-12&maximum_ha=12&baseline_hrs=1&show_unc=1&minimum_depth=0&maximum_V_mag=&target_string=HD+209458+b&print_html=1&twilight=-12&max_airmass=2.4
    mid_transit_time = mid_transit_time[night]

    if use_t23:
        print(f"Using full transit time (T23), not total transit time (T14)")
        planet_transit_duration = planet.calculate_full_transit_duration(
            total_transit_duration=planet.transit_duration,
            planet_radius=planet.radius,
            star_radius=planet.star_radius,
            impact_parameter=planet.calculate_impact_parameter(
                planet_orbit_semi_major_axis=planet.orbit_semi_major_axis,
                planet_orbital_inclination=planet.orbital_inclination,
                star_radius=planet.star_radius
            )
        )

        if use_t1535:
            print(f"Adding exposures of half-eclipses")
            planet_transit_duration += (planet.transit_duration - planet_transit_duration) / 2
    else:
        planet_transit_duration = planet.transit_duration

    # Load data
    if load_rico:
        print(f"Loading Rico data...")
        wavelengths_instrument, observations, uncertainties, times, berv, airmass, \
            orbital_phases, v_sys, kp = get_data(
                planet=planet,
                planet_transit_duration=planet_transit_duration,
                dates=dates,
                night=night,
                mid_transit_time=mid_transit_time,
                data_source='rico',
                nodes='B'
            )

        ccf_velocities = ccf_radial_velocity(
            v_sys=v_sys,
            kp=kp,
            lsf_fwhm=lsf_fwhm,  # cm.s-1
            pixels_per_resolution_element=pixels_per_resolution_element,
            kp_factor=kp_factor,
            extra_factor=extra_factor
        )
    else:
        wavelengths_instrument = None
        observations = None
        uncertainties = None
        times = None
        berv = None
        airmass = None
        orbital_phases = None
        v_sys = None
        kp = None
        ccf_velocities = None

    if load_jason:
        print(f"Loading Jason data...")
        wavelengths_instrumentj, observationsj, uncertaintiesj, timesj, bervj, airmassj, \
            orbital_phasesj, v_sysj, kpj = get_data(
                planet=planet,
                planet_transit_duration=planet_transit_duration,
                dates=dates,
                night=night,
                mid_transit_time=mid_transit_time,
                data_source='rico'
            )

        ccf_velocitiesj = ccf_radial_velocity(
            v_sys=v_sysj,
            kp=kpj,
            lsf_fwhm=lsf_fwhm,  # cm.s-1
            pixels_per_resolution_element=pixels_per_resolution_element,
            kp_factor=kp_factor,
            extra_factor=extra_factor
        )
    else:
        wavelengths_instrumentj = None
        observationsj = None
        uncertaintiesj = None
        timesj = None
        bervj = None
        airmassj = None
        orbital_phasesj = None
        v_sysj = None
        kpj = None

    print("Generating model...")
    print(times)
    wavelengths_model, model, spectral_model, radtrans = get_model(
        planet=planet,
        wavelengths_instrument=wavelengths_instrument,
        system_observer_radial_velocities=v_sys,
        kp_range=np.array([-kp, kp]) * kp_factor,
        ccf_velocities=ccf_velocities,
        times=times,
        mid_transit_time=mid_transit_time,
        airmass=airmass,
        uncertainties=uncertainties,
        detector_selection=None,
        tellurics_mask_threshold=tellurics_mask_threshold,
        mode='transmission'
    )

    if get_mock_obs:
        wavelengths_telluric, telluric_transmittances = \
            get_tellurics_npz(
                './tellurics_crires.npz',
                np.array([
                    spectral_model.wavelengths_boundaries[0] - 1e-6,
                    spectral_model.wavelengths_boundaries[1] + 1e-6
                ])
            )

        telluric_transmittances[np.less(telluric_transmittances, 1e-15)] = 1e-15

        variable_throughput = load_variable_throughput_brogi(
            os.path.join(additional_data_directory, 'metis', 'brogi_crires_test', 'algn.npy'),
            times_size=times.size,
            wavelengths_size=wavelengths_instrument.shape[-1]
        )

        # uncertainties_simple = np.moveaxis(
        #     np.ma.mean(uncertainties, axis=1) * np.moveaxis(np.ones(uncertainties.shape), 1, 0), 0, 1)
        # observations_simple = np.moveaxis(
        #     np.ma.mean(observations, axis=1) * np.moveaxis(np.ones(observations.shape), 1, 0), 0, 1)

        noise_matrix = np.random.default_rng().normal(
            loc=0,
            scale=np.ma.masked_array(uncertainties / observations).filled(0),
            size=uncertainties.shape
        )

        _, mock_observations = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True,
            telluric_transmittances_wavelengths=wavelengths_telluric,
            telluric_transmittances=telluric_transmittances,
            instrumental_deformations=variable_throughput,
            noise_matrix=noise_matrix,
            use_transit_light_loss=True,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True
        )

        print("Preparing model...")
        prepared_mock_data, reprocessing_mock_matrix, prepared_mock_data_uncertainties = spectral_model.pipeline(
            spectrum=mock_observations,
            wavelength=wavelengths_instrument,
            **spectral_model.model_parameters
        )

    print("Preparing data...")
    prepared_data, reprocessing_matrix, prepared_data_uncertainties = spectral_model.pipeline(
        spectrum=observations,
        wavelength=wavelengths_instrument,
        **spectral_model.model_parameters
    )

    if remove_5sig_outsiders:
        print("Removing 5 sigmas outsiders...")
        stds = np.moveaxis(
            np.ma.median(np.ma.std(prepared_data, axis=-1), axis=1)
            * np.ones((prepared_data.shape[1], prepared_data.shape[2], prepared_data.shape[0])),
            -1,
            0
        )
        masked_data = np.ma.masked_where(
            np.abs(prepared_data - 1) > 5 * stds, observations
        )
        uncertainties = np.ma.masked_where(masked_data.mask, uncertainties)
        spectral_model.model_parameters['uncertainties'] = uncertainties

        print("Re-preparing data after 5 sigmas outsiders removal...")
        masked_prepared_data, reprocessing_matrix_m, masked_prepared_data_uncertainties = spectral_model.pipeline(
            spectrum=masked_data,
            wavelength=wavelengths_instrument,
            **spectral_model.model_parameters
        )

    print("CCF...")
    co_added_cross_correlations_snr, co_added_cross_correlations, \
        v_rest, kps, ccf_sum, ccfs, velocities_ccf, ccf_models, ccf_model_wavelengths = \
        ccf_analysis(
            wavelengths_data=wavelengths_instrument,
            data=prepared_data,
            wavelengths_model=wavelengths_model[0],
            model=model[0],
            velocities_ccf=ccf_velocities,
            model_velocities=None,
            normalize_ccf=True,
            calculate_ccf_snr=True,
            ccf_sum_axes=None,
            planet_radial_velocity_amplitude=kp,
            system_observer_radial_velocities=v_sys,
            orbital_longitudes=np.rad2deg(orbital_phases * 2 * np.pi),
            planet_orbital_inclination=planet.orbital_inclination,
            line_spread_function_fwhm=lsf_fwhm,
            pixels_per_resolution_element=pixels_per_resolution_element,
            co_added_ccf_peak_width=None,
            velocity_interval_extension_factor=extra_factor,
            kp_factor=kp_factor
        )

    #
    co_added_velocities, kps, v_rest = get_co_added_ccf_velocity_space(
        planet_radial_velocity_amplitude=kp,
        velocities_ccf=velocities_ccf,
        system_observer_radial_velocities=v_sys,
        orbital_longitudes=orbital_phases * 360,  # phase to deg
        planet_orbital_inclination=planet.orbital_inclination,
        kp_factor=kp_factor
    )

    detector_selection = np.arange(0, observations.shape[0])

    ccf_sum = np.array([np.sum(ccfs[detector_selection], axis=0)])

    co_added_cross_correlations_notsplit = np.zeros((ccf_sum.shape[0], kps.size, v_rest.size))

    for i, ccf_ in enumerate(ccf_sum):
        co_added_cross_correlations_notsplit[i] = co_add_cross_correlation(
            cross_correlation=ccf_,
            velocities_ccf=velocities_ccf,
            co_added_velocities=co_added_velocities
        )


def remove_mask(observed_spectra, observations_uncertainties):
    print('Taking care of mask...')
    data_ = []
    error_ = []
    mask_ = copy.copy(observed_spectra.mask)
    lengths = []

    for i in range(observed_spectra.shape[0]):
        data_.append([])
        error_.append([])

        for j in range(observed_spectra.shape[1]):
            data_[i].append(np.array(
                observed_spectra[i, j, ~mask_[i, j, :]]
            ))
            error_[i].append(np.array(observations_uncertainties[i, j, ~mask_[i, j, :]]))
            lengths.append(data_[i][j].size)

    # Handle jagged arrays
    if np.all(np.asarray(lengths) == lengths[0]):
        data_ = np.asarray(data_)
        error_ = np.asarray(error_)
    else:
        print("Array is jagged, generating object array...")
        data_ = np.asarray(data_, dtype=object)
        error_ = np.asarray(error_, dtype=object)

    return data_, error_, mask_


def simple_ccf(wavelength_data, data, wavelength_model, model,
               lsf_fwhm, pixels_per_resolution_element, radial_velocity, kp,
               data_uncertainties=None):
    n_detectors, n_integrations, n_spectral_pixels = np.shape(data)

    radial_velocity_lag = ccf_radial_velocity(
        v_sys=radial_velocity,
        kp=kp,
        lsf_fwhm=lsf_fwhm,
        pixels_per_resolution_element=pixels_per_resolution_element,
        kp_factor=1.0,
        extra_factor=0.25
    )

    ccf_ = np.zeros((n_detectors, n_integrations, np.size(radial_velocity_lag)))

    # Shift the wavelengths
    wavelength_shift = np.zeros((np.size(radial_velocity_lag), np.size(wavelength_model)))
    models_shift = np.zeros((n_detectors, np.size(radial_velocity_lag), n_spectral_pixels))

    if np.ndim(wavelength_data) == 3:  # Alex's time-dependent wavelengths
        wavelengths_ = copy.copy(wavelength_data[:, 0, :])
    else:
        wavelengths_ = copy.copy(wavelength_data)

    # Calculate Doppler shifted model wavelengths
    for j in range(np.size(radial_velocity_lag)):
        wavelength_shift[j, :] = doppler_shift(
            wavelength_0=wavelength_model,
            velocity=radial_velocity_lag[j]
        )

    # Rebin model to data wavelengths and shift
    for i in range(n_detectors):
        for k in range(np.size(radial_velocity_lag)):
            models_shift[i, k, :] = \
                frebin.rebin_spectrum(wavelength_shift[k, :], model, wavelengths_[i, :])

    # Mask management
    if hasattr(data, 'mask'):
        data, data_uncertainties, mask = remove_mask(
            data, data_uncertainties
        )
    else:
        mask = np.zeros((n_detectors, len(radial_velocity_lag), n_spectral_pixels), dtype=bool)

    # def xcorr(f_vec, g_vec, n):
    #     # Compute variances of model and data
    #     # N is kept for compatibility when using the old version
    #     sf2 = (f_vec @ f_vec)
    #     sg2 = (g_vec @ g_vec)
    #     r = (f_vec @ g_vec)
    #
    #     return r / np.sqrt(sf2 * sg2)  # Data-model cross-correlation

    # Perform cross correlation
    for i in range(n_detectors):
        for k in range(len(radial_velocity_lag)):
            for j in range(n_integrations):
                ccf_[i, j, k] = cross_correlate(data[i, j], models_shift[i, k, ~mask[i, j, :]])

    return ccf_, models_shift, radial_velocity_lag


def simple_co_added_ccf(ccf, velocities_ccf, orbital_phases_ccf, system_radial_velocities, kp, kp_factor=2, shape=None):
    if shape is None:
        shape = (ccf.shape[0], velocities_ccf.size, velocities_ccf.size)
    elif np.size(shape) < 2:
        raise ValueError(f"co added CCF shape must be of size 2")
    elif np.size(shape) == 2:
        shape = list(shape)
        shape.insert(0, ccf.shape[0])
        shape = tuple(shape)
    elif np.size(shape) == 3 and shape[0] != ccf.shape[0]:
        raise ValueError(f"co added CCF dimension 0 ({shape[0]}) must match CCF dimension 0 ({ccf.shape[0]})")

    kps = np.linspace(
        -kp * kp_factor, kp * kp_factor, shape[1]
    )

    v_kp_boundaries = (
        kps[0] * np.sin(2.0 * np.pi * orbital_phases_ccf),
        kps[-1] * np.sin(2.0 * np.pi * orbital_phases_ccf)
    )

    # Prevent out of bound integration
    v_planet_min = np.min(system_radial_velocities + np.min(v_kp_boundaries))
    v_planet_max = np.max(system_radial_velocities + np.max(v_kp_boundaries))
    v_rest_min = np.min(velocities_ccf) - v_planet_min
    v_rest_max = np.max(velocities_ccf) - v_planet_max
    v_rest = np.linspace(v_rest_min, v_rest_max, shape[2])

    # Defining matrix containing the co-added CCFs
    ccf_tot = np.zeros(shape)

    for i in range(ccf.shape[0]):
        for ikp in range(kps.size):
            planet_radial_velocities = system_radial_velocities + kps[ikp] * np.sin(2.0 * np.pi * orbital_phases_ccf)

            for j in range(ccf.shape[1]):
                radial_velocities_interp = v_rest + planet_radial_velocities[j]
                ccf_interp = interp1d(velocities_ccf, ccf[i, j, :])

                try:
                    ccf_tot[i, ikp, :] += ccf_interp(radial_velocities_interp)
                except ValueError as err:
                    print(planet_radial_velocities[j], np.min(v_rest), np.max(v_rest), kps[ikp], np.max(kps), np.min(kps))
                    print(np.min(radial_velocities_interp), np.min(velocities_ccf))
                    print(np.max(radial_velocities_interp), np.max(velocities_ccf))
                    raise ValueError(err)

    return ccf_tot, v_rest, kps
