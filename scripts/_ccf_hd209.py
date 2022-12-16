import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS.fort_rebin import fort_rebin as fr
from scipy.interpolate import interp1d
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.physics import doppler_shift
from petitRADTRANS.ccf.ccf_core import cross_correlate
from petitRADTRANS.containers.planet import Planet
from petitRADTRANS.containers.spectral_model import SpectralModel
from petitRADTRANS.retrieval.reprocessing import reprocessing_pipeline
from scripts.load_spectral_matrix import construct_spectral_matrix


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
              generate_mock_obs=False, tellurics_mask_threshold=0.5):

    planet_rest_frame_velocity_shift = 0.0

    spectral_model = SpectralModel(
        pressures=np.logspace(-10, 2, 100),
        line_species=[
            'H2O_main_iso',
            # 'CH4_hargreaves_main_iso',
            'CO_all_iso',
            # 'H2S_main_iso',
            # 'HCN_main_iso',
            # 'NH3_main_iso'
        ],
        rayleigh_species=['H2', 'He'],
        continuum_opacities=['H2-H2', 'H2-He'],
        lbl_opacity_sampling=4,
        reference_pressure=1e-2,
        temperature_profile_mode='isothermal',
        temperature=planet.equilibrium_temperature,
        imposed_mass_mixing_ratios={
            'CH4_hargreaves_main_iso': 3.4e-5,
            'CO_all_iso': 1.8e-2,
            'H2O_main_iso': 5.4e-3,
            'H2S_main_iso': 1.0e-3,
            'HCN_main_iso': 2.7e-7,
            'NH3_main_iso': 7.9e-6
        },
        use_equilibrium_chemistry=False,
        fill_atmosphere=True,
        cloud_pressure=2.0,
        planet_radius=planet.radius,
        planet_surface_gravity=planet.surface_gravity,
        star_effective_temperature=planet.star_effective_temperature,
        star_radius=planet.star_radius,
        semi_major_axis=planet.orbit_semi_major_axis,
        planet_orbital_inclination=planet.orbital_inclination,
        planet_radial_velocity_amplitude=kp,
        system_observer_radial_velocities=v_sys,
        planet_rest_frame_velocity_shift=planet_rest_frame_velocity_shift,
        orbital_phases=orbital_phases,
        airmass=airmass,
        new_resolving_power=9.2e4,
        output_wavelengths=wavelengths_instrument[:, 0, :],
        uncertainties=uncertainties,
        tellurics_mask_threshold=tellurics_mask_threshold,
        polynomial_fit_degree=2,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True
    )
    spectral_model.wavelengths_boundaries = spectral_model.get_optimal_wavelength_boundaries(
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

    print(f"Boundaries: {spectral_model_ref.wavelengths_boundaries}")
    print(f"Radtrans...")
    radtrans_ref = spectral_model_ref.get_radtrans()

    print('----\n')
    if generate_mock_obs:
        radtrans = spectral_model.get_radtrans()
        model_wavelengths, ccf_model = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=True,
            instrumental_deformations=None,
            noise_matrix=None,
            scale=True,
            shift=False,
            convolve=True,
            rebin=False,
            reduce=False
        )
    else:
        ccf_model = None
        model_wavelengths = None

    true_parameters = copy.deepcopy(spectral_model.model_parameters)

    print('Data reduction...')
    reduced_data, reduction_matrix, reduced_uncertainties = reprocessing_pipeline(
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
                                              fr.rebin_spectrum(telluric_wavelengths, telluric_data[:, 1],
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
        _, mock_observations = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=False,
            instrumental_deformations=deformation_matrix,
            noise_matrix=noise,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=True
        )

        print('Generating noiseless mock observations...')
        _, mock_observations_nn = spectral_model.get_spectrum_model(
            radtrans=radtrans,
            mode='transmission',
            update_parameters=False,
            instrumental_deformations=deformation_matrix,
            noise_matrix=None,
            scale=True,
            shift=True,
            convolve=True,
            rebin=True,
            reduce=True
        )
    else:
        mock_observations = None
        mock_observations_nn = None

    print('Generating reference model...')
    _, ccf_model_red = spectral_model_ref.get_spectrum_model(
        radtrans=radtrans_ref,
        mode='transmission',
        update_parameters=True,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=True,
        shift=True,
        convolve=True,
        rebin=True,
        reduce=True
    )

    return reduced_data, ccf_model, model_wavelengths, true_parameters, mock_observations, mock_observations_nn, \
        ccf_model_red, wavelengths_ref, spectral_model, spectral_model_ref, radtrans_ref


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

    data_out = np.copy(data_in - cor1)

    return data_out, cor1


def load_rico_data(data_directory, interpolate_to_common_wl, nodes='both', truncate=5, quantile_mask=0.005):
    # lacking: mid_transit_time, orbital_phases, airmass, berv
    wavelengths_instrument, observations, uncertainties, times = construct_spectral_matrix(
        os.path.join(data_directory, ''),
        interpolate_to_common_wl=interpolate_to_common_wl,
        nodes=nodes,
        #planet.orbital_period,
        #mid_transit_jd=58004.425291
    )

    if truncate is not None:
        mask = np.ones(wavelengths_instrument.shape, dtype=bool)
        mask[:, :, truncate:-truncate] = False
        # wavelengths_instrument = np.ma.masked_where(mask, wavelengths_instrument)
        observations = np.ma.masked_where(mask, observations)
        uncertainties = np.ma.masked_where(mask, uncertainties)

    mask_threshold = np.quantile(observations.data, quantile_mask)
    mask_threshold = np.less(observations, mask_threshold)
    # wavelengths_instrument = np.ma.masked_where(mask_threshold, wavelengths_instrument)
    observations = np.ma.masked_where(mask_threshold, observations)
    uncertainties = np.ma.masked_where(mask_threshold, uncertainties)

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


def main():
    use_t23 = False  # use full eclipse transit time instead of total transit time
    use_t1535 = False
    planet_name = 'HD 209458 b'
    planet = Planet.get(planet_name)
    night = 0
    lsf_fwhm = 1.9e5

    tellurics_mask_threshold = 0.5

    use_alex_detector_selection = False

    dates = [
        '2022_06_29',
        '2022_07_06',
        '2022_10_12',
    ]

    # Load data
    wavelengths_instrument, observations, uncertainties, times = load_rico_data(
            os.path.join(module_dir, 'crires', planet_name.lower().replace(' ', '_'),
                         'hd209_crires_reduced', dates[night], 'correct_wavelengths'),
            interpolate_to_common_wl=True,
            nodes='A'
        )

    wavelengths_instrumentj, observationsj, uncertaintiesj, timesj = load_jason_data(
            os.path.join(module_dir, 'crires', planet_name.lower().replace(' ', '_'),
                         'jason'),
            n_transits=1
        )

    mid_transit_time = [
        59759.8109 - 0.5,
        59766.8604 - 0.5,
        59865.5532 - 0.5
    ]  # https://astro.swarthmore.edu/transits/print_transits.cgi?single_object=0&ra=&dec=&epoch=&period=&duration=&depth=&target=&observatory_string=-24.625%3B-70.403333%3BAmerica%2FSantiago%3BEuropean+Southern+Observatory%3A+Paranal&use_utc=0&observatory_latitude=37.223611&observatory_longitude=-2.54625&timezone=UTC&start_date=05-29-2022&days_to_print=180&days_in_past=0&minimum_start_elevation=30&and_vs_or=or&minimum_end_elevation=30&minimum_ha=-12&maximum_ha=12&baseline_hrs=1&show_unc=1&minimum_depth=0&maximum_V_mag=&target_string=HD+209458+b&print_html=1&twilight=-12&max_airmass=2.4
    mid_transit_time = mid_transit_time[night]

    berv = planet.get_barycentric_velocities(
        ra=planet.ra,
        dec=planet.dec,
        time=times,
        site_name='Paranal'
    )
    bervj = planet.get_barycentric_velocities(
        ra=planet.ra,
        dec=planet.dec,
        time=timesj,
        site_name='Paranal'
    )
    airmass = planet.get_airmass(
        ra=planet.ra,
        dec=planet.dec,
        time=times,
        site_name='Paranal'
    )
    airmassj = planet.get_airmass(
        ra=planet.ra,
        dec=planet.dec,
        time=timesj,
        site_name='Paranal'
    )

    times = (times - mid_transit_time) * nc.snc.day
    timesj = (timesj - mid_transit_time) * nc.snc.day

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

    wh = np.where(np.logical_and(times >= -planet_transit_duration / 2,
                                 times <= planet_transit_duration / 2))[0]
    whj = np.where(np.logical_and(timesj >= -planet_transit_duration / 2,
                                  timesj <= planet_transit_duration / 2))[0]

    orbital_phases = planet.get_orbital_phases(0, planet.orbital_period, times)
    orbital_phasesj = planet.get_orbital_phases(0, planet.orbital_period, timesj)

    wavelengths_instrument = wavelengths_instrument[:, wh, :]
    wavelengths_instrumentj = wavelengths_instrumentj[:, whj, :]
    observations = observations[:, wh, :]
    observationsj = observationsj[:, whj, :]
    uncertainties = uncertainties[:, wh, :]
    uncertaintiesj = uncertaintiesj[:, whj, :]
    times = times[wh]
    timesj = timesj[whj]

    orbital_phases = orbital_phases[wh]
    orbital_phasesj = orbital_phasesj[whj]
    berv = berv[wh]
    bervj = bervj[whj]
    airmass = airmass[wh]
    airmassj = airmassj[whj]

    kp = planet.calculate_orbital_velocity(planet.star_mass, planet.orbit_semi_major_axis)
    v_sys = planet.star_radial_velocity - berv * 1e2
    v_sysj = planet.star_radial_velocity - bervj * 1e2

    # Get models and reduce data
    ccf_velocities = ccf_radial_velocity(
        v_sys=v_sys,
        kp=kp,
        lsf_fwhm=lsf_fwhm,  # cm.s-1
        pixels_per_resolution_element=2,
        kp_factor=1.0,
        extra_factor=0.25
    )

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
            tellurics_mask_threshold=tellurics_mask_threshold
        )

    reduced_dataj, ccf_model_prej, model_wavelengthsj, true_parametersj, mock_observationsj, mock_observations_nnj, \
        ccf_model_ref_prej, wavelengths_refj, spectral_modelj, spectral_model_refj, radtransj = get_model(
            planet=planet,
            wavelengths_instrument=wavelengths_instrumentj,
            kp=kp,
            v_sys=v_sysj,
            ccf_velocities=ccf_velocities,
            orbital_phases=orbital_phasesj,
            airmass=airmassj,
            observations=observationsj,
            uncertainties=uncertaintiesj,
            tellurics_mask_threshold=tellurics_mask_threshold
        )

    ccf_model = ccf_model_ref_pre  # no need to use norm_sys since the model is already reduced
    wh_0 = np.where(np.abs(orbital_phases) == np.min(np.abs(orbital_phases)))[0][0]  # search where the orbital phase is colsest to 0 to get a model where kp~0, not ideal

    # Calculate CCF
    ccf, ccf_models, rvs = simple_ccf(
        wavelengths_instrument,
        reduced_data,
        wavelengths_ref.flatten(),
        ccf_model[:, wh_0, :].flatten(),
        lsf_fwhm=2.6e5,  # cm.s-1
        pixels_per_resolution_element=2,  # v_rest step is lsf_fwhm / pixels_per_resolution_element
        radial_velocity=v_sys,
        kp=kp,
        data_uncertainties=uncertainties
    )

    # Nice CCF
    ccf_norm = np.transpose(np.transpose(ccf) - np.transpose(np.nanmedian(ccf, axis=2)))
    ccf_norm = np.ma.masked_invalid(ccf_norm)

    detector_list = np.linspace(0, wavelengths_instrument.shape[0] - 1, wavelengths_instrument.shape[0], dtype=int)
    detector_selection = detector_list
    detector_selection = np.arange(1, 23, 1)
    # detector_selection = [5, 7, 8, 9, 14, 15, 17, 18, 19, 21]
    # detector_selection = [5, 6, 9, 14, 18, 19, 20, 21]

    # Find best detector selection and calculate collapsed CCF
    kp_range = np.array([145e5, 160e5])

    ccf_norm_select = copy.copy(ccf_norm)
    ccf_norm_select = ccf_norm_select[detector_selection]
    ccf_norm_select_sum = np.ma.sum(ccf_norm_select, axis=0)
    ccf_tot, v_rests, kps = simple_co_added_ccf(np.array([ccf_norm_select_sum]), rvs, orbital_phases, v_sys, kp)

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
                fr.rebin_spectrum(wavelength_shift[k, :], model, wavelengths_[i, :])

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
