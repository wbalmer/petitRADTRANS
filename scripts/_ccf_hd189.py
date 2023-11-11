import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.physics import doppler_shift, rebin_spectrum
from petitRADTRANS.ccf.ccf_core import cross_correlate
from petitRADTRANS.planet import Planet
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.retrieval.preparing import preparing_pipeline, remove_throughput_fit, remove_telluric_lines_fit
from scripts.high_resolution_retrieval_carmenes import load_carmenes_data

module_dir = os.path.abspath(r'C:\Users\Doriann\Documents\work\run_outputs\petitRADTRANS\data')


def ccf_analysis(ccf_norm_select, detector_selection, rvs, orbital_phases, v_sys, kp,
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


def find_best_detector_selection_noa_nosnr(first_guess, detector_list, ccf_norm_select, rvs, orbital_phases, v_sys, kp,
                                           kp_range, v_rest_range, max_percentage_area=0.68,
                                           area_increase_tolerance=1.1, use_peak=True):
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

            if (kp_range[0] <= max_kp_tmp <= kp_range[1]) \
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
                      f"Max CCF S/N ({max_sn_tmp} > {max_sn}) at invalid kp and/or v_rest:\n"
                      f"  kp: \t{max_kp_tmp}, must be within {kp_range}\n"
                      f"  v_rest: \t{max_v_rest_tmp}, must be within {v_rest_range}")
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

            if (kp_range[0] <= max_kp_tmp <= kp_range[1]) \
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
                      f"max CCF S/N ({max_sn_tmp} > {max_sn}) at invalid kp and/or v_rest:\n"
                      f"  kp: \t{max_kp_tmp}, must be within {kp_range}\n"
                      f"  v_rest: \t{max_v_rest_tmp}, must be within {v_rest_range}")

    print(f"\nCCF S/N: {max_sn}\n"
          f"kp: {max_kp * 1e-5} km.s-1\n"
          f"v_rest: {max_v_rest * 1e-5} km.s-1")

    return ccf_tot, max_sn_sn, max_kp_sn, max_v_rest_sn, ccf_tot_sn, detector_selection, \
        max_sn_peak, max_kp_peak, max_v_peak, kps, v_rests


def find_best_detector_selection_alex(ccf_norm, ccf_norm_untouched, rvs, orbital_phases, v_sys, kp, alex_kp_factor=-1.0,
                                      planet_rest_velocity=0.0):
    exclude = 15.6e5  # exclusion region in cm.s-1 (+/-):

    ccf_tot_sn = np.zeros(ccf_norm.shape[0])
    ccf_tot_sn_untouched = np.zeros(ccf_norm.shape[0])

    detector_selection = np.array([])
    ccf_tot_ = []
    ccf_tot_untouched_ = []

    kps = None
    v_rests = None

    kp *= alex_kp_factor

    for i, ccf in enumerate(ccf_norm):
        ccf = np.ma.array([ccf]).filled(0)
        ccf_untouched = np.ma.array([ccf_norm_untouched[i]]).filled(0)

        ccf_tot, v_rests, kps = simple_co_added_ccf(ccf, rvs, orbital_phases, v_sys, kp)
        ccf_tot_untouched, _, _ = simple_co_added_ccf(
            ccf_untouched, rvs, orbital_phases, v_sys, kp
        )

        ccf_tot_.append(ccf_tot)
        ccf_tot_untouched_.append(ccf_tot_untouched)

        wh_kp = np.where(np.abs(kps - kp) == np.min(np.abs(kps - kp)))[0][0]
        wh_vr = np.where(np.abs(v_rests - planet_rest_velocity) == np.min(np.abs(v_rests - planet_rest_velocity)))[0][0]

        std_pts_a = np.where(v_rests < (v_rests[wh_vr] - exclude))[0]
        std_pts_b = np.where(v_rests > (v_rests[wh_vr] + exclude))[0]
        std_pts = np.concatenate((std_pts_a, std_pts_b))

        # Compute the S/N
        ccf_tot_sn[i] = ccf_tot[0, wh_kp, wh_vr] / np.ma.std(ccf_tot[0, wh_kp, std_pts])
        ccf_tot_sn_untouched[i] = ccf_tot_untouched[0, wh_kp, wh_vr] / np.ma.std(ccf_tot_untouched[0, wh_kp, std_pts])

        if ccf_tot_sn[i] - ccf_tot_sn_untouched[i] > 3:
            print(f"Adding detector {i}: delta CCF > 3 ({ccf_tot_sn[i] - ccf_tot_sn_untouched[i]})")
            detector_selection = np.append(detector_selection, i)
        else:
            print(f"Rejecting detector {i}: delta CCF <= 3 ({ccf_tot_sn[i] - ccf_tot_sn_untouched[i]})")

    return detector_selection, ccf_tot_sn, ccf_tot_sn_untouched, np.array(ccf_tot_), np.array(ccf_tot_untouched_), \
        kps, v_rests


def get_model(planet, wavelengths_instrument, kp, v_sys, ccf_velocities,
              orbital_phases, airmass, observations, uncertainties,
              generate_mock_obs=False, tellurics_mask_threshold=0.2, use_alex_detector_selection=False,
              alex_k_factor=-1.0, use_5kms_shift=False):
    # Initialization
    if use_alex_detector_selection:
        print(f"Kp * {alex_k_factor} for Alex's detector selection")
        kp *= alex_k_factor

    if use_5kms_shift:
        rest_frame_velocity_shift = -5e5
    else:
        rest_frame_velocity_shift = 0.0

    spectral_model = SpectralModel(
        pressures=np.logspace(-10, 2, 100),
        line_species=[
            'H2O_main_iso',
            'CH4_hargreaves_main_iso',
            'CO-NatAbund',
            'H2S_main_iso',
            'HCN_main_iso',
            'NH3_main_iso'
        ],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        line_by_line_opacity_sampling=4,
        reference_pressure=1e-2,
        temperature_profile_mode='isothermal',
        temperature=planet.equilibrium_temperature,
        imposed_mass_fractions={
            'CH4_hargreaves_main_iso': 3.4e-5,
            'CO-NatAbund': 1.8e-2,
            'H2O_main_iso': 5.4e-3,
            'H2S_main_iso': 1.0e-3,
            'HCN_main_iso': 2.7e-7,
            'NH3_main_iso': 7.9e-6
        },
        use_equilibrium_chemistry=False,
        fill_atmosphere=True,
        cloud_pressure=2.0,
        planet_radius=planet.radius,
        planet_surface_gravity=planet.reference_gravity,
        star_effective_temperature=planet.star_effective_temperature,
        star_radius=planet.star_radius,
        semi_major_axis=planet.orbit_semi_major_axis,
        planet_orbital_inclination=planet.orbital_inclination,
        radial_velocity_semi_amplitude=kp,
        system_observer_radial_velocities=v_sys,
        rest_frame_velocity_shift=rest_frame_velocity_shift,
        orbital_phases=orbital_phases,
        airmass=airmass,
        new_resolving_power=8.04e4,
        output_wavelengths=wavelengths_instrument[:, 0, :],
        uncertainties=uncertainties,
        tellurics_mask_threshold=tellurics_mask_threshold,
        polynomial_fit_degree=2,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True
    )
    spectral_model.wavelengths_boundaries = spectral_model.calculate_optimal_wavelength_boundaries(
        relative_velocities=ccf_velocities
    )

    wavelengths_ref = copy.deepcopy(wavelengths_instrument[:, 0, :])
    wavelengths_ref = np.vstack((
        np.linspace(
            spectral_model.wavelengths_boundaries[0],
            np.min(wavelengths_instrument),
            wavelengths_instrument.shape[2]
        ),
        wavelengths_ref,
        np.linspace(
            np.max(wavelengths_instrument),
            spectral_model.wavelengths_boundaries[1],
            wavelengths_instrument.shape[2]
        ),
    ))

    uncertainties_ref = np.vstack((
        np.ones((1, wavelengths_instrument.shape[1], wavelengths_instrument.shape[2])) * uncertainties[0, :, :],
        uncertainties,
        np.ones((1, wavelengths_instrument.shape[1], wavelengths_instrument.shape[2])) * uncertainties[-1, :, :],
    ))

    uncertainties_ref = np.ma.masked_invalid(uncertainties_ref)

    spectral_model_ref = copy.deepcopy(spectral_model)
    spectral_model_ref.model_parameters['output_wavelengths'] = wavelengths_ref
    spectral_model_ref.model_parameters['uncertainties'] = uncertainties_ref
    spectral_model_ref.model_parameters['system_observer_radial_velocities'] = 0.0
    spectral_model_ref.wavelengths_boundaries[0] -= 0.002
    spectral_model_ref.wavelengths_boundaries[1] += 0.002

    radtrans_ref = spectral_model_ref.get_radtrans()

    print('----\n')
    if generate_mock_obs:
        radtrans = spectral_model.get_radtrans()
        model_wavelengths, ccf_model = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=False,
            convolve=True,
            rebin=False,
            prepare=False
        )
    else:
        ccf_model = None
        model_wavelengths = None
        radtrans = None

    true_parameters = copy.deepcopy(spectral_model.model_parameters)

    print('Data reduction...')
    if use_alex_detector_selection:
        print("Adding model with negative kp to observations for Alex's detector selection")
        _, ccf_model_alex = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=False
        )

        ccf_model_alex = np.ma.masked_where(uncertainties.mask, ccf_model_alex)

        ccf_model_alex, rma, rua = remove_throughput_fit(
            spectrum=ccf_model_alex,
            reduction_matrix=np.ma.ones(ccf_model_alex.shape),
            wavelengths=wavelengths_instrument,
            uncertainties=uncertainties,
            polynomial_fit_degree=2
        )

        reduced_data, reduction_matrix, reduced_uncertainties = remove_throughput_fit(
            spectrum=observations,
            reduction_matrix=np.ma.ones(observations.shape),
            wavelengths=wavelengths_instrument,
            uncertainties=uncertainties,
            polynomial_fit_degree=2
        )

        reduced_data = reduced_data + (ccf_model_alex - 1)

        reduced_data, reduction_matrix, reduced_uncertainties = remove_telluric_lines_fit(
            spectrum=reduced_data,
            reduction_matrix=reduction_matrix,
            airmass=airmass,
            uncertainties=reduced_uncertainties,
            polynomial_fit_degree=2
        )
    else:
        reduced_data, reduction_matrix, reduced_uncertainties = preparing_pipeline(
            spectrum=observations,
            uncertainties=uncertainties,
            wavelengths=wavelengths_instrument,
            airmass=airmass,
            tellurics_mask_threshold=tellurics_mask_threshold,
            full=True,
            polynomial_fit_degree=2,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True
        )

    if generate_mock_obs:
        noise = np.random.default_rng().normal(loc=0, scale=uncertainties, size=reduced_data.shape)
        telluric_transmittance = os.path.join(module_dir, 'andes', 'sky', 'transmission',
                                              f"transmission.dat")
        variable_throughput = os.path.join(module_dir, 'andes', 'brogi_crires', "algn.npy")

        print('Adding telluric transmittance...')
        telluric_data = np.loadtxt(telluric_transmittance)
        telluric_wavelengths = telluric_data[:, 0] * 1e-3  # nm to um
        telluric_transmittance = np.zeros(reduced_data.shape)

        for i, detector_wavelengths in enumerate(wavelengths_instrument[:, 0, :]):
            telluric_transmittance[i, :, :] = np.ones((reduced_data.shape[1], reduced_data.shape[2])) * \
                                              rebin_spectrum(telluric_wavelengths, telluric_data[:, 1],
                                                             detector_wavelengths)

        print('Adding variable throughput...')
        variable_throughput = np.load(variable_throughput)
        variable_throughput = np.max(variable_throughput[0], axis=1)
        variable_throughput = variable_throughput / np.max(variable_throughput)
        xp = np.linspace(0, 1, np.size(variable_throughput))
        x = np.linspace(0, 1, np.size(orbital_phases))
        variable_throughput = np.interp(x, xp, variable_throughput)

        deformation_matrix = np.zeros(reduced_data.shape)

        for i in range(reduced_data.shape[0]):
            for k in range(reduced_data.shape[2]):
                deformation_matrix[i, :, k] = telluric_transmittance[i, :, k] * variable_throughput

        print('Generating mock observations...')
        _, mock_observations = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=False,
            instrumental_deformations=deformation_matrix,
            noise_matrix=noise,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=True
        )

        print('Generating noiseless mock observations...')
        _, mock_observations_nn = spectral_model.calculate_spectrum(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=False,
            instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            prepare=True
        )
    else:
        mock_observations = None
        mock_observations_nn = None

    print('Generating reference model...')
    _, ccf_model_red = spectral_model_ref.calculate_spectrum(
        radtrans=radtrans_ref,
        mode='transmission',
        update_parameters=True,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=True,
        shift=True,
        convolve=True,
        rebin=True,
        prepare=True
    )

    return reduced_data, ccf_model, model_wavelengths, true_parameters, mock_observations, mock_observations_nn, \
        ccf_model_red, wavelengths_ref, spectral_model, spectral_model_ref, radtrans_ref


def plot_ccf(ccf, x, y, true_coord=None):
    plt.imshow(ccf, aspect='auto', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])

    if true_coord is not None:
        plt.scatter(true_coord[0], true_coord[1], color='r', marker='+')


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
               relative_velocities=None, data_uncertainties=None):
    n_detectors, n_integrations, n_spectral_pixels = np.shape(data)

    # Get rest velocity grid
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
    models_shift = np.zeros((n_detectors, np.size(radial_velocity_lag), n_spectral_pixels))

    if np.ndim(wavelength_data) == 3:  # Alex's time-dependent wavelengths
        wavelengths_ = copy.copy(wavelength_data[:, 0, :])
    else:
        wavelengths_ = copy.copy(wavelength_data)

    # Calculate Doppler shifted model wavelengths
    if np.ndim(model) == 2:
        print(f"2D model, assuming it was given for 0 cm.s-1 relative velocity...")
        wavelength_shift = np.zeros((np.size(radial_velocity_lag), np.size(wavelength_model)))

        if relative_velocities is None:
            relative_velocities = 0

        for j in range(np.size(radial_velocity_lag)):
            wavelength_shift[j, :] = doppler_shift(
                wavelength_0=wavelength_model,
                velocity=radial_velocity_lag[j] - relative_velocities
            )

        model_flat = model.flatten()

        # Rebin model to data wavelengths and shift
        for i in range(n_detectors):
            for k in range(np.size(radial_velocity_lag)):
                models_shift[i, k, :] = \
                    rebin_spectrum(wavelength_shift[k, :], model_flat, wavelengths_[i, :])
    elif np.ndim(model) == 3:
        print(f"3D model, following input relative velocity...")
        wavelength_shift = np.zeros((model.shape[1], np.size(radial_velocity_lag), np.size(wavelength_model)))
        model_flat = np.zeros((model.shape[1], model.shape[0] * model.shape[2]))

        for i in range(model.shape[1]):
            for j in range(np.size(radial_velocity_lag)):
                wavelength_shift[i, j, :] = doppler_shift(
                    wavelength_0=wavelength_model,
                    velocity=radial_velocity_lag[j] - relative_velocities[i]
                )

                model_flat[i] = model[:, i, :].flatten()

        # Rebin model to data wavelengths and shift
        for i in range(n_detectors):
            for j, ws in enumerate(wavelength_shift):
                for k in range(np.size(radial_velocity_lag)):
                    models_shift[i, k, :] = \
                        rebin_spectrum(ws[k, :], model_flat[j], wavelengths_[i, :])
    else:
        raise ValueError(f"number of dimension for model must be 2 or 3, but is {np.ndim(model)}")

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
            radial_velocities = system_radial_velocities + kps[ikp] * np.sin(2.0 * np.pi * orbital_phases_ccf)

            for j in range(ccf.shape[1]):
                radial_velocities_interp = v_rest + radial_velocities[j]
                ccf_interp = interp1d(velocities_ccf, ccf[i, j, :])

                ccf_tot[i, ikp, :] += ccf_interp(radial_velocities_interp)

    return ccf_tot, v_rest, kps


def sysrem(n_data, n_spectra, data_in, errors_in):
    """
    SYSREM procedure by Alejandro Sanchez-Lopez, 26-09-17 2017
    """
    # We first define an array holding the time evolution
    # of a certain pixel. Ignas uses (n_data/3). However,
    # this value might have to be changed if it falls inside a mask.
    a = data_in[int(n_data / 3), :]  # TODO but a is the airmass, why taking only one value and why using the data?

    # Resize to intialize an entire matrix
    # 'aj' is the airmass in Tamuz et al. 2004; it's
    # here called a1.
    a1 = np.matlib.repmat(a, n_data, 1)

    # 'c' which is the extinction coefficient in Tamuz et al. 2004; it's
    # calculated as in equation (2) of the paper.
    # For the calculation of c, data_in is the r_ij of Tamuz et al. 2004.
    c = np.sum(data_in * a1 / errors_in ** 2., 1) / \
        np.sum(a1 ** 2. / errors_in ** 2., 1)
    # Right now c is an array. We're going to resize it to obtain
    # a matrix 'c1'. It's like having the t evolution of this
    # 'extinction coeff' for each pixel.
    c1 = np.transpose(np.matlib.repmat(c, n_spectra, 1))

    # As in the paper, we aim to remove the product c_i*a_j
    # (c1 * a1) from each r_ij (data_in(i,j)). So, we are
    # basically assuming that the average variation of the flux
    # over time is mainly due to airmass variations (a1). We
    # assume that the variation is linear and the slope is c1.

    # Defining the correction factors 'cor':
    cor1 = c1 * a1  # in essence, the thing we want to substract to the data.
    cor0 = np.zeros(cor1.shape, dtype=float)

    # Now that c has been calculated using the prior a1 value from the
    # data_in, we can turn around the problem and calculate the set of 'a'
    # values that minimize eq. (3) in the paper using the previously
    # calculated c values.
    # Hence, we repeat the loop until the convergence criterion is achieved.
    while np.sum(np.abs(cor0 - cor1)) / np.sum(np.abs(cor0)) >= 1e-3:
        # Start with the first value calculated before for the correction
        cor0 = cor1
        # Apply eq. (4) in the paper, which gives us 'a':
        a = np.sum(data_in * c1 / errors_in ** 2., 0) / \
            np.sum(c1 ** 2. / errors_in ** 2., 0)

        # We transform it into a matrix
        a1 = np.matlib.repmat(np.transpose(a), n_data, 1)

        # Now we recalculate the best fitting coefficients 'c' using the
        # latest 'a' values. By iterating this process until the
        # convergence criterion is achieved, we obtained the two sets
        # (a & c) that BEST account for the variations during the
        # observations (airmass variations mostly).
        c = np.sum(data_in * a1 / errors_in ** 2., 1) / \
            np.sum(a1 ** 2. / errors_in ** 2., 1)
        c1 = np.transpose(np.matlib.repmat(c, n_spectra, 1))

        # Recalculate the correction using the latest values of the
        # two sets a & c.
        cor1 = a1 * c1

    data_out = np.copy(data_in / cor1)  # divide instead of subtract to respect Eq. 16

    return data_out, cor1


def main_alex():
    def ccf(lag, n_spectra, obs, ccf_iterations, wave, wave_CC,
            ccf_values, template):

        # CC a synthetic spectrum with data.

        for m in range(ccf_iterations):
            for i in range(n_spectra):
                syn_spec_shifted = np.interp(wave, wave_CC *
                                             (1. + lag[m] / 3e5), template[i, :])
                # Keep only values not masked (!= 1)
                keep = list()
                keep = np.array(np.where(obs[i, :] != 1)[0, :])
                # print(keep.shape)
                # if m == 0 and i == 0: print(keep)
                aux_obs = obs[:, keep]
                aux_model = syn_spec_shifted[keep]
                xd = aux_model - np.mean(aux_model)
                yd = aux_obs[i, :] - np.mean(aux_obs[i, :])
                cross = np.sum(yd * xd)
                ccf_values[m, i] = cross / np.sqrt(np.sum(xd ** 2)
                                                   * np.sum(yd ** 2))
        return ccf_values

    def mask_nan(self, n_spectra, n_orders, n_data, spec, sig, wave_all):
        """
         Masking Nan values in the data
        """
        # Look for points to correct
        for i in range(n_spectra):
            for j in range(n_orders):
                nans = np.array(np.where(np.isfinite(
                    spec[i, :, j]) == False))[0, :]
                no_nans = np.array(np.where(np.isfinite(
                    spec[i, :, j]) == True))[0, :]
                # print str(nans)
                # print str(no_nans)
                if nans.shape != (0,):
                    for n in range(len(nans)):
                        spec[i, nans[n], j] = \
                            np.median(spec[i, no_nans, j])
                        sig[i, nans[n], j] = \
                            np.median(sig[i, no_nans, j])
        return spec, sig

    # syn_spec is your model, which I plug into a matrix called mat_cc
    # (which includes both in and out-of-transit=1.
    # We do not really care about OOT but it's just to be able to use the rest of my old code since it does not matter
    # anyway)

    # Calculate relative velocities
    v_cc = data['K_p'] * np.sin(2. * np.pi * phase) + data['V_sys'] - data['berv'] + data['V_rest']

    # Fill
    mat_cc = np.zeros((46, 2040))

    for n in range(n_spectra):
        if n in in_transit:
            mat_cc[n, :] = syn_spec[h, n - in_transit[0], :]
        else:
            mat_cc[n, :] = 1.

    syn_mat_res, good, mask = pipeline_carmenes(phase, wave_carmenes, mat_cc, sig)

    # Undo the shift to bring back to vacuum wvls
    mat_back = np.zeros((len(v_cc), len(wave_carmenes)))
    for i in range(len(v_cc)):
        mat_back[i, :] = np.interp(wave_carmenes,
                                   wave_carmenes * (1. - v_cc[i] / 3e5),
                                   syn_mat_res[i, :])

    mean_step_v = 1.3  # Step for CCF lags

    # Define interval
    ccf_v_interval = 308.  # km/s
    ccf_v_step = np.round(mean_step_v, 1)
    ccf_iterations = int(2 * ccf_v_interval / ccf_v_step + 1)
    v_ccf = np.zeros(ccf_iterations)
    for i in range(ccf_iterations):
        v_ccf[i] = -ccf_v_interval + float(i) * ccf_v_step

    ccf_values = np.zeros((ccf_iterations, n_spectra, data['n_orders']), float)

    ccf_values[:, :, h] = ccf(lag=v_ccf, n_spectra=n_spectra, obs=mat_res,
                              ccf_iterations=ccf_iterations, wave=wave_carmenes,
                              wave_CC=wave_carmenes, ccf_values=ccf_values[:, :, h],  # h runs over CCDs
                              template=mat_back)

    # Subtracting the median value to each row (gets rid of broad S/N differences)
    for n in range(n_spectra):
        ccf_values[:, n, h] -= np.median(ccf_values[:, n, h])

    # Merge orders
    ccf_merged = np.sum(ccf_values, 2)

    # For the shifts in Kp to the Kp-dependent planet rest-frame, I put expected V-pixel at central position. Pixels_lef_right is a stupid name for the actual number of pixels we take around expected_v_pixel
    for kp in kp_range:
        # Calculate planetary velocities during the night
        v_planet = kp * np.sin(2. * np.pi * phase) + data['V_sys'] - data['berv'] + data['V_rest']
        for i in in_transit:
            v_aux = np.arange(v_planet[i] - pixels_left_right * ccf_v_step,
                              v_planet[i] + pixels_left_right * ccf_v_step,
                              ccf_v_step)
            v_aux = np.append(v_aux, v_planet[i] +
                              pixels_left_right * ccf_v_step)
            if len(v_aux) == len(v_wind) + 1:
                v_aux = v_aux[0:-1]

            # Locate the pixel where the planetary signal should be
            ccf_values_shift[:, cont, np.int(kp + (n_kp - 1) / 2 - 1)] = np.interp(v_aux, v_ccf, ccf_merged[:, i])
            cont += 1

    ccf_tot = np.sum(ccf_values_shift, 1)

    # Maximum Vrest for std calculations and plots
    max_ccf_v = 255.2
    plot_step = 50.
    pixels_left_right = int(max_ccf_v / ccf_v_step)

    # Vrest grid for the plots
    v_wind = ccf_v_step * (np.arange(2 * pixels_left_right + 1) -
                           float(pixels_left_right))

    # Calculate S/N
    ccf_tot_sn = np.copy(ccf_tot)
    max_sn = 0
    max_kp = 0
    max_v_wind = 0
    exclude = 15

    for kp in kp_range:
        # Finding the maximum in the CCF, wherever it may be located:
        aux = np.array(np.where(ccf_tot[:, kp + np.int((n_kp - 1) / 2)] == \
                                np.amax(ccf_tot[:, np.int(kp + (n_kp - 1) / 2)]))[0])
        # print(v_wind[aux])

        # Select the v_rest range far from "detected" signal
        std_pts_a = np.array(np.where(v_wind < \
                                      (v_wind[aux] - exclude)))[0]
        std_pts_b = np.array(np.where(v_wind > \
                                      (v_wind[aux] + exclude)))[0]
        std_pts = np.concatenate((std_pts_a, std_pts_b))

        # Compute the S/N
        ccf_tot_sn[:, np.int(kp + (n_kp - 1) / 2)] = (ccf_tot[:,
                                                      np.int(kp + (n_kp - 1) / 2)] - np.mean(ccf_tot[std_pts,
            np.int(kp + (n_kp - 1) / 2)])) / np.std(ccf_tot[std_pts,
            np.int(kp + (n_kp - 1) / 2)])
        if np.amax(ccf_tot_sn[:, np.int(kp + (n_kp - 1) / 2)]) > max_sn:
            max_sn = np.amax(ccf_tot_sn[:, np.int(kp + (n_kp - 1) / 2)])
            max_kp = np.int(kp + (n_kp - 1) / 2)
            max_v_wind = v_wind[np.array(np.where(
                ccf_tot_sn[:, np.int(kp + (n_kp - 1) / 2)] == max_sn))]


def reprocessing_pipeline_carmenes(phase, wave, mat, noise):
    ##Step1. Normalisation
    result1 = np.zeros_like(mat)
    error1 = np.zeros_like(noise)

    for n in range(len(phase)):
        fit_coeffs = np.polynomial.Polynomial.fit(x=wave, y=mat[n, :], deg=2,
                                                  w=noise[n, :]).convert().coef
        result1[n, :] = mat[n, :] / fit_coeffs[0] + fit_coeffs[1] * wave + fit_coeffs[2] * wave ** 2.
        # Assuming the fit has no error
        error1[n, :] = result1[n, :] * np.sqrt((noise[n, :] / mat[n, :]) ** 2.)

    # Masking now before correction to avoid fit explosion. I do it this ugly way but anyway
    mask = []
    n_spectra = len(phase)
    n_data = len(wave)

    for n in range(n_spectra):
        for k in range(n_data):
            if result1[n, k] < 0.5:
                result1[:, k] = 1.
                mask.append(k)

    mask = np.sort(np.asarray(mask))

    if mask.shape != (0,):
        result1[:, mask] = 1

    # Brogi extra step
    telluric_spec = np.zeros_like(wave)
    result2 = np.zeros_like(result1)
    error2 = np.zeros_like(error1)
    telluric_fit_log = np.zeros_like(wave)

    for k in range(n_data):
        telluric_spec[k] = np.median(result1[:, k])

    for n in range(n_spectra):
        c1 = np.polyfit(telluric_spec, np.log(result1[n, :]), 2)
        telluric_fit_log = np.polyval(c1, telluric_spec)
        result2[n, :] = result1[n, :] / np.exp(telluric_fit_log)
        error2[n, :] = result2[n, :] * np.sqrt((error1[n, :] / result1[n, :]) ** 2.)

    # Step2
    result2 = np.zeros_like(result1)
    error2 = np.zeros_like(error1)
    # Second correction in each spectral pixel
    x = np.asarray(range(n_spectra))

    for k in range(n_data):
        if np.sum(result1[:, k]) != n_spectra:  # Avoiding masks, which are set to 1
            fit_coeffs = np.polynomial.Polynomial.fit(x=x,
                                                      y=np.log(result1[:, k]),
                                                      deg=2,
                                                      w=1. / error1[:, k]).convert().coef

            result2[:, k] = result1[:, k] / np.exp(fit_coeffs[0] + fit_coeffs[1] * x + fit_coeffs[2] * x ** 2.)
            error2[:, k] = result2[:, k] * np.sqrt((error1[:, k] / result1[:, k]) ** 2.)
        else:
            result2[:, k] = result1[:, k]  # we dont care, it will not be used
            error2[:, k] = error1[:, k]  # we dont care, it will not be used

    if mask.shape != (0,):  # Just a remasking, for extra safety
        result2[:, mask] = 1

    return result2, error2, mask


def main():
    use_t23 = True  # use full eclipse transit time instead of total transit time
    use_t1535 = True  # use intermediate eclipse ("T_transit")
    use_all_models = False  # use all reprocessed models instead of the one closest to 0 cm.s-1 relative velocity
    use_alex_ttransit = False  # add orbital phase 27 to T1.53.5
    planet_name = 'HD 189733 b'
    planet = Planet.get(planet_name)
    # Overriding to Rosenthal et al. 2021
    planet.star_radius = 0.78271930158600000 * cst.r_sun
    planet.star_radius_error_upper = +0.01396094224705000 * cst.r_sun
    planet.star_radius_error_lower = -0.01396094224705000 * cst.r_sun

    tellurics_mask_threshold = 0.5

    use_alex_detector_selection = False

    # Load data
    wavelengths_instrument, observations, instrument_snr, uncertainties, orbital_phases, airmass, berv, times, \
        mid_transit_time = \
        load_carmenes_data(
            os.path.join(module_dir, 'carmenes', planet_name.lower().replace(' ', '_')),
            planet.orbital_period, mid_transit_jd=58004.42319302507  # 58004.425291  # previous value
        )

    if use_t23:
        print(f"Using full transit time (T23), not total transit time (T14)")
        planet_transit_duration = planet.calculate_full_transit_duration(
            total_transit_duration=planet.transit_duration,
            planet_radius=planet.radius,
            star_radius=planet.star_radius,
            impact_parameter=planet.calculate_impact_parameter(
                orbit_semi_major_axis=planet.orbit_semi_major_axis,
                orbital_inclination=planet.orbital_inclination,
                star_radius=planet.star_radius
            )
        )

        if use_t1535:
            print(f"Adding exposures of half-eclipses")
            planet_transit_duration += (planet.transit_duration - planet_transit_duration) / 2
    else:
        planet_transit_duration = planet.transit_duration

    if not use_alex_ttransit:
        print(f"Select orbital phases...")
        exposure_selection = np.where(np.logical_and(times >= mid_transit_time - planet_transit_duration / 2,
                                                     times <= mid_transit_time + planet_transit_duration / 2))[0]
    else:
        print(f"Select Alex's orbital phases...")
        exposure_selection = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])

    wavelengths_instrument = wavelengths_instrument[:, exposure_selection, :]
    orbital_phases = orbital_phases[exposure_selection]
    berv = berv[exposure_selection]
    observations = observations[:, exposure_selection, :]
    uncertainties = uncertainties[:, exposure_selection, :]
    airmass = airmass[exposure_selection]

    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)
    v_sys = planet.star_radial_velocity - berv * 1e5

    # Load Alex's reduced data
    with fits.open(
            os.path.join(os.path.join(module_dir, 'carmenes', planet_name.lower().replace(' ', '_')),
                         'spec_sysrem_iteration_10.fits')
    ) as f:
        reduced_data = f[0].data

    data_shape = reduced_data.shape
    reduced_data = np.moveaxis(
        np.reshape(
            np.moveaxis(reduced_data, 0, 2), (data_shape[1], int(data_shape[2] / 2), int(data_shape[0] * 2)), order='F'
        ),
        2,
        0
    )
    reduced_data_alex = reduced_data[:, exposure_selection[0], :]

    # Get models and reduce data
    # Get CCF rest velocity grid
    ccf_velocities = ccf_radial_velocity(
        v_sys=v_sys,
        kp=kp,
        lsf_fwhm=2.6e5,  # cm.s-1
        pixels_per_resolution_element=2,
        kp_factor=1.0,
        extra_factor=0.25
    )

    # Get spectral models
    reduced_data, ccf_model_pre, model_wavelengths, true_parameters, mock_observations, mock_observations_nn, \
        ccf_model_ref_pre, wavelengths_ref, spectral_model, spectral_model_ref, radtrans = get_model(
            planet=planet,
            wavelengths_instrument=wavelengths_instrument,
            kp=kp,
            v_sys=v_sys,
            ccf_velocities=ccf_velocities,
            orbital_phases=orbital_phases,
            airmass=airmass,
            observations=observations,
            uncertainties=uncertainties,
            tellurics_mask_threshold=tellurics_mask_threshold,
            use_alex_detector_selection=use_alex_detector_selection,
            alex_k_factor=-1.0,
            use_5kms_shift=False
        )

    if use_alex_detector_selection:
        reduced_data_untouched, _, _ = preparing_pipeline(
            spectrum=observations,
            uncertainties=uncertainties,
            wavelengths=wavelengths_instrument,
            airmass=airmass,
            tellurics_mask_threshold=tellurics_mask_threshold,
            full=True,
            polynomial_fit_degree=2,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True
        )
    else:
        reduced_data_untouched = None

    ccf_model = ccf_model_ref_pre  # no need to use norm_sys since the model is already reprocessed

    # Reprocessed model selection
    if not use_all_models:
        # Search where the orbital phase is closest to 0 to get a model where kp~0, not ideal
        wh_0 = np.where(np.abs(orbital_phases) == np.min(np.abs(orbital_phases)))[0][0]
        ccf_model_ = ccf_model[:, wh_0, :]
        relative_velocities = spectral_model_ref.model_parameters['relative_velocities'][wh_0]
    else:
        ccf_model_ = copy.deepcopy(ccf_model)
        relative_velocities = spectral_model_ref.model_parameters['relative_velocities']

    # Calculate CCF
    ccf, ccf_models, rvs = simple_ccf(
        wavelength_data=wavelengths_instrument,
        data=reduced_data,
        wavelength_model=wavelengths_ref.flatten(),
        model=ccf_model_,
        lsf_fwhm=2.6e5,  # cm.s-1
        pixels_per_resolution_element=2,  # v_rest step is lsf_fwhm / pixels_per_resolution_element
        radial_velocity=v_sys,
        kp=kp,
        data_uncertainties=uncertainties,
        relative_velocities=relative_velocities
    )

    if use_alex_detector_selection:
        ccf_untouched, _, _ = simple_ccf(
            wavelength_data=wavelengths_instrument[-1],
            data=reduced_data_untouched[-1],
            wavelength_model=wavelengths_ref.flatten(),
            model=ccf_model_,
            lsf_fwhm=2.6e5,  # cm.s-1
            pixels_per_resolution_element=2,  # v_rest step is lsf_fwhm / pixels_per_resolution_element
            radial_velocity=v_sys,
            kp=kp,
            data_uncertainties=uncertainties,
            relative_velocities=spectral_model_ref.model_parameters['relative_velocities']
        )
    else:
        ccf_untouched = None

    # Nice CCF
    ccf_norm = np.transpose(np.transpose(ccf) - np.transpose(np.nanmedian(ccf, axis=2)))
    ccf_norm = np.ma.masked_invalid(ccf_norm)
    ccf_norm_select = copy.copy(ccf_norm)

    if use_alex_detector_selection:
        ccf_norm_untouched = np.transpose(
            np.transpose(ccf_untouched) - np.transpose(np.nanmedian(ccf_untouched, axis=2))
        )
        ccf_norm_untouched = np.ma.masked_invalid(ccf_norm_untouched)
        detector_selection, ccf_test, ccf_test_untouched, ccf_tot, ccf_tot_untouched, kps, v_rests = \
            find_best_detector_selection_alex(
                ccf_norm_select,
                ccf_norm_untouched,
                rvs,
                orbital_phases,
                v_sys,
                kp
            )
    else:
        detector_list = np.linspace(0, wavelengths_instrument.shape[0] - 1, wavelengths_instrument.shape[0], dtype=int)
        # Beware: this is a step of uttermost importance! Carefully chose the first guess, sometimes even *1* bad CCD is enough to completely mess uo an otherwise good selection
        # detector_selection = np.array([3, 9, 25, 26, 46, 47])  # works with telluric threshold = 0.5
        # detector_selection = np.array([4, 6, 12, 25, 26, 46, 47])  # chosen by adding them one by one and looking at the collapsed CCF
        # detector_selection = np.array([3, 9, 25, 26, 46, 47])  # TT0.5, start for T14
        # detector_selection = np.array([3, 9, 25, 26, 46, 54])  # TT0.5, new start for T14
        detector_selection = np.array([11, 18, 19, 25, 26, 32, 33, 34])  # start for bad
        # detector_selection = np.array([3,  7,  9, 13, 25, 28, 29, 46, 54])  # TT0.5, T1535 (from T14, but added 54)
        # detector_selection = np.array([1,  3,  9, 14, 25, 28, 29, 30, 46, 54])  # TT0.5, T23 (from T1535)
        # detector_selection = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28,
        #                               29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54])

        # Find best detector selection and calculate collapsed CCF
        kp_range = np.array([145e5, 160e5])

        max_percentage_area = 0.68
        ccf_tot, max_sn_sn, max_kp_sn, max_v_rest_sn, ccf_tot_sn, detector_selection, \
            max_sn_peak, max_kp_peak, max_v_rest_peak, kps, v_rests = \
            find_best_detector_selection(
                first_guess=detector_selection,
                detector_list=detector_list,
                ccf_norm_select=ccf_norm_select,
                rvs=rvs,
                orbital_phases=orbital_phases,
                v_sys=v_sys,
                kp=kp,
                kp_range=kp_range,
                v_rest_range=[-10e5, 10e5],
                max_percentage_area=max_percentage_area,
                area_increase_tolerance=1.0,
                use_peak=True
            )

        # Plots
        plt.imshow(ccf_tot[0], aspect='auto', origin='lower',
                   extent=np.array([v_rests[0], v_rests[-1], kps[0], kps[-1]]) * 1e-5)
        plt.colorbar(label='collapsed CCF')
        plt.vlines(0, np.min(kps) * 1e-5, np.max(kps) * 1e-5, color='r', alpha=0.3)
        plt.hlines(kp * 1e-5, np.min(v_rests) * 1e-5, np.max(v_rests) * 1e-5, color='r', alpha=0.3)
        plt.scatter(max_v_rest_peak * 1e-5, max_kp_peak * 1e-5, color='r', marker='+')
        plt.contour(v_rests * 1e-5, kps * 1e-5, ccf_tot[0], levels=[max_percentage_area * np.max(ccf_tot[0])],
                    colors='r', alpha=0.3)
        plt.xlabel(r'$V_r$ (km$\cdot$s$^{-1}$)')
        plt.ylabel(r'$K_p$ (km$\cdot$s$^{-1}$)')
        plt.title(f"TT mask threshold: {tellurics_mask_threshold}, SNR = {max_sn_sn:.3f}")


def figure_slide():
    detector_selection = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]) - 1
    n_exposures = 19
    uncertainties_ref = np.vstack((
        np.ma.ones((1, wavelengths_instrument.shape[1], wavelengths_instrument.shape[2])) * uncertainties[0, :, :],
        uncertainties,
        np.ma.ones((1, wavelengths_instrument.shape[1], wavelengths_instrument.shape[2])) * uncertainties[-1, :, :],
    ))
    uncertainties_ref = np.ma.masked_array(uncertainties_ref)
    uncertainties_ref[1:-1] = np.ma.masked_where(uncertainties.mask, uncertainties_ref[1:-1])
    uncertainties_ref[0] = np.ma.masked_where(uncertainties[0].mask, uncertainties_ref[0])
    uncertainties_ref[-1] = np.ma.masked_where(uncertainties[-1].mask, uncertainties_ref[-1])
    uncertainties_ref.set_fill_value(0)

    observations = np.ma.masked_where(uncertainties.mask, observations)

    wh_0 = np.where(np.abs(orbital_phases) == np.min(np.abs(orbital_phases)))[0][0]
    relative_velocities = spectral_model_ref.model_parameters['relative_velocities'][wh_0]
    _, ccf_model_red = spectral_model_ref.calculate_spectrum(
        radtrans=radtrans,
        mode='transmission',
        update_parameters=True,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=True,
        shift=True,
        convolve=True,
        rebin=True,
        prepare=False
    )
    ccf_tots = []
    kpss = []
    vrss = []
    rvs_mask_between = np.array([0, 0])
    print(f"RV masked between {rvs_mask_between}")
    print(ccf_model_red.shape)

    for i in range(15):
        print(f"{i + 1}/{len(orbital_phases) - n_exposures}")
        print(f" Preparing model...")
        ccf_m = np.ma.masked_array(ccf_model_red)
        ccf_m.mask = np.zeros(ccf_m.shape, dtype=bool)
        ccf_m.mask[1:-1] = copy.deepcopy(observations.mask)
        ccf_m = np.ma.masked_where(uncertainties_ref.mask, ccf_m)
        ccf_model__, reduction_matrix__, reduced_uncertainties__ = preparing_pipeline(
            spectrum=ccf_m[:, i:i + n_exposures, :],
            uncertainties=uncertainties_ref[:, i:i + n_exposures, :],
            wavelengths=wavelengths_ref,
            airmass=airmass[i:i + n_exposures],
            tellurics_mask_threshold=tellurics_mask_threshold,
            full=True,
            polynomial_fit_degree=2,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True
        )

        print(f" Preparing data...")
        observations = np.ma.masked_where(uncertainties.mask, observations)
        reduced_data, reduction_matrix, reduced_uncertainties = preparing_pipeline(
            spectrum=observations[:, i:i + n_exposures, :],
            uncertainties=uncertainties[:, i:i + n_exposures, :],
            wavelengths=wavelengths_instrument[:, i:i + n_exposures, :],
            airmass=airmass[i:i + n_exposures],
            tellurics_mask_threshold=tellurics_mask_threshold,
            full=True,
            polynomial_fit_degree=2,
            apply_throughput_removal=True,
            apply_telluric_lines_removal=True
        )
        wh_0 = np.where(np.abs(orbital_phases[i:i + n_exposures]) == np.min(np.abs(orbital_phases)))[0][0]
        ccf_model__ = ccf_model__[:, wh_0, :]
        print(f" CCF...")
        ccf, ccf_models, rvs = simple_ccf(
            wavelength_data=wavelengths_instrument[1:-1, i:i + n_exposures, :],
            data=reduced_data[1:-1, :, :],
            wavelength_model=wavelengths_ref.flatten(),
            model=ccf_model__,
            lsf_fwhm=2.6e5,  # cm.s-1
            pixels_per_resolution_element=2,  # v_rest step is lsf_fwhm / pixels_per_resolution_element
            radial_velocity=v_sys[i:i + n_exposures],
            kp=kp,
            data_uncertainties=uncertainties[1:-1, i:i + n_exposures, :],
            relative_velocities=relative_velocities
        )
        ccf_norm = np.transpose(np.transpose(ccf) - np.transpose(np.nanmedian(ccf, axis=2)))
        ccf_norm = np.ma.masked_invalid(ccf_norm)
        ccf_norm_select = copy.copy(ccf_norm)
        ccf_norm_select_tmp = ccf_norm_select[detector_selection]
        ccf_norm_select_sum = np.ma.sum(ccf_norm_select_tmp, axis=0)

        rvs_keep = np.nonzero(np.logical_not(
            np.logical_and(rvs > rvs_mask_between[0], rvs < rvs_mask_between[1])
        ))
        rvs_ = rvs[rvs_keep]
        ccf_norm_select_sum_ = np.array([ccf_norm_select_sum[:, rvs_keep[0]]])
        print(np.shape(ccf_norm_select_sum), rvs.shape)
        ccf_tot, v_rests, kps = simple_co_added_ccf(
            ccf_norm_select_sum_, rvs_, orbital_phases[i:i + n_exposures],
            v_sys[i:i + n_exposures], kp)
        ccf_tots.append(ccf_tot)
        kpss.append(kps)
        vrss.append(v_rests)

    ccf_tot_sns = []
    max_sns = []
    max_ccs = []
    max_kps = []
    max_v_rests = []

    for i, ct in enumerate(ccf_tots):
        print(i)
        # Calculate S/N
        ccf_tot_sn = np.zeros(ct.shape)
        exclude = 15.6  # exclusion region in km/s (+/-):
        max_sn = 0
        max_cc = 0

        for k, ccf_tot_kp in enumerate(ct[0]):
            # Finding the maximum in the CCF, wherever it may be located:
            aux = np.argmax(ccf_tot_kp)

            # Select the v_rest range far from "detected" signal
            std_pts_a = np.where(vrss[i] < (vrss[i][aux] - exclude))[0]
            std_pts_b = np.where(vrss[i] > (vrss[i][aux] + exclude))[0]
            std_pts = np.concatenate((std_pts_a, std_pts_b))

            # Compute the S/N
            ccf_tot_sn[0, k, :] = ccf_tot_kp / np.std(ccf_tot_kp[std_pts])

            id_max = np.argmax(ccf_tot_sn[0, k, :])

            if ccf_tot_kp[id_max] > max_cc:
                max_sn = ccf_tot_sn[0, k, id_max]
                max_cc = ct[0, k, id_max]
                max_kp = kpss[i][k]
                max_v_rest = vrss[i][id_max]

        ccf_tot_sns.append(ccf_tot_sn)
        max_sns.append(max_sn)
        max_ccs.append(max_cc)
        max_kps.append(max_kp)
        max_v_rests.append(max_v_rest)

    fig, axes = plt.subplots(4, sharex='col', figsize=(6.4, 4 * 1.6))

    axes[0].plot(max_ccs, ls='', marker='+')
    axes[0].scatter(7, max_ccs[7], marker='+', color='r', s=80, linewidth=3)
    axes[1].plot(max_sns, ls='', marker='+')
    axes[1].scatter(7, max_sns[7], marker='+', color='r', s=80, linewidth=3)
    axes[2].plot(np.array(max_kps) * 1e-5, ls='', marker='+')
    axes[2].scatter(7, max_kps[7] * 1e-5, marker='+', color='r', s=80, linewidth=3)
    axes[3].plot(np.array(max_v_rests) * 1e-5, ls='', marker='+')
    axes[3].scatter(7, max_v_rests[7] * 1e-5, marker='+', color='r', s=80, linewidth=3)

    axes[2].plot([0, 14], [152, 152], color='k', ls=':')
    axes[3].plot([0, 14], [-5, -5], color='k', ls=':')
    axes[0].set_ylabel('Max co-added CCF (A.U.)')
    axes[1].set_ylabel('Peak SNR')
    axes[2].set_ylabel(r'Peak $K_p$ (km$\cdot$s$^{-1}$)')
    axes[3].set_ylabel(r'Peak $V_\mathrm{rest}$ (km$\cdot$s$^{-1}$)')
    axes[3].set_xlabel(r'Starting exposure')
    axes[3].set_xticks(np.arange(15))
    plt.tight_layout()
