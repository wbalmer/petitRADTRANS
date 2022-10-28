"""Useful functions to analyze data using cross-correlation.
Data and model are expected to be 3D (n_detectors, n_integrations, n_wavelengths).
"""

import copy

import numpy as np

from petitRADTRANS.containers.planet import Planet
from petitRADTRANS.fort_rebin import fort_rebin as fr
from petitRADTRANS.physics import doppler_shift
from petitRADTRANS.ccf.ccf_core import cross_correlate_3d, co_add_cross_correlation


def calculate_co_added_cross_correlation(cross_correlation, orbital_phases_ccf, velocities_ccf,
                                         system_radial_velocities, planet_max_radial_orbital_velocity, kp_factor=2.0,
                                         planet_orbital_inclination=90.0,
                                         n_kp=None, n_vr=None):
    n_detectors = np.shape(cross_correlation)[0]

    co_added_velocities, kps, v_rest = co_added_ccf_velocity_space(
        system_radial_velocities=system_radial_velocities,
        planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
        orbital_longitudes=orbital_phases_ccf * 360,  # phase to deg
        velocities_ccf=velocities_ccf,
        planet_orbital_inclination=planet_orbital_inclination,
        kp_factor=kp_factor,
        n_kp=n_kp,
        n_vr=n_vr
    )

    # Calculate the co-added CCFs
    ccf_tot = np.zeros((n_detectors, np.size(kps), np.size(v_rest)))

    for i, ccf in enumerate(cross_correlation):
        ccf_tot[i] = co_add_cross_correlation(
            cross_correlation=ccf,
            velocities_ccf=velocities_ccf,
            co_added_velocities=co_added_velocities
        )

    return ccf_tot, kps, v_rest


def calculate_co_added_ccf_snr(co_added_cross_correlation, vr_space, vr_peak_width):
    ccf_tot_sn = np.zeros(co_added_cross_correlation.shape)

    for i, ccf_tot in enumerate(co_added_cross_correlation):
        for k, ccf_tot_kp in enumerate(ccf_tot):
            # Find the maximum of the co-added CCF at a given Kp
            index_max = np.argmax(ccf_tot_kp)

            # Select co-added CCF points around the "detected peak", hopefully far enough to not include the peak itself
            noise_indices_lower = np.where(vr_space < (vr_space[index_max] - vr_peak_width))[0]
            noise_indices_greater = np.where(vr_space > (vr_space[index_max] + vr_peak_width))[0]
            noise_indices = np.concatenate((noise_indices_lower, noise_indices_greater))

            # Calculate the "signal" to "noise" ratio
            ccf_tot_sn[i, k, :] = ccf_tot_kp / np.std(ccf_tot_kp[noise_indices])

    return ccf_tot_sn


def calculate_cross_correlation(wavelength_data, data, wavelength_model, model,
                                planet_max_radial_orbital_velocity, line_spread_function_fwhm,
                                system_radial_velocity=0.0, pixels_per_resolution_element=2,
                                kp_factor=1.0, extra_velocity_factor=0.25, normalize=True, full=False):
    velocities_ccf = ccf_velocity_space(
        system_radial_velocity=system_radial_velocity,
        planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
        line_spread_function_fwhm=line_spread_function_fwhm,
        pixels_per_resolution_element=pixels_per_resolution_element,
        kp_factor=kp_factor,
        extra_velocity_factor=extra_velocity_factor
    )

    if full:
        ccf, models_shift, wavelength_shift = shift_cross_correlate(
            wavelength_data=wavelength_data,
            data=data,
            wavelength_model=wavelength_model,
            model=model,
            velocities_ccf=velocities_ccf,
            full=full
        )
    else:
        ccf = shift_cross_correlate(
            wavelength_data=wavelength_data,
            data=data,
            wavelength_model=wavelength_model,
            model=model,
            velocities_ccf=velocities_ccf,
            full=full
        )

        models_shift = None
        wavelength_shift = None

    if normalize:
        ccf = normalize_cross_correlation(ccf)

    if full:
        return ccf, velocities_ccf, models_shift, wavelength_shift
    else:
        return ccf, velocities_ccf


def ccf_velocity_space(system_radial_velocity, planet_max_radial_orbital_velocity,
                       line_spread_function_fwhm, pixels_per_resolution_element,
                       kp_factor=1.0, extra_velocity_factor=0.25):
    # Get min and max velocities based on the planet parameters
    velocity_min = np.min(system_radial_velocity) - planet_max_radial_orbital_velocity * kp_factor
    velocity_max = np.max(system_radial_velocity) + planet_max_radial_orbital_velocity * kp_factor

    # Add a margin to the boundaries
    velocity_interval = velocity_max - velocity_min
    velocity_min -= extra_velocity_factor * velocity_interval
    velocity_max += extra_velocity_factor * velocity_interval

    # Set the velocity step as the equivalent to the instrument spectral resolution
    # A finer step won't be fully resolved by the instrument
    velocity_step = line_spread_function_fwhm / pixels_per_resolution_element

    # Get the velocities
    n_low = int(np.floor(velocity_min / velocity_step))
    n_high = int(np.ceil(velocity_max / velocity_step))

    velocities = np.linspace(
        n_low * velocity_step,
        n_high * velocity_step,
        n_high - n_low + 1
    )

    return velocities


def co_added_ccf_analysis(co_added_cross_correlation, kp_space, vr_space, max_percentage_area):
    """Get the location of the co-added cross-correlation ("SNR") peak and the number of points around the peak.
    """
    ccf_tot_max = np.max(co_added_cross_correlation)

    max_coord = np.nonzero(co_added_cross_correlation == ccf_tot_max)
    max_kp = kp_space[max_coord[0]][0]
    max_v_rest = vr_space[max_coord[1]][0]

    n_around_peak = np.size(np.nonzero(co_added_cross_correlation >= max_percentage_area * ccf_tot_max))

    return ccf_tot_max, max_kp, max_v_rest, n_around_peak


def co_added_ccf_velocity_space(system_radial_velocities, planet_max_radial_orbital_velocity,
                                orbital_longitudes, velocities_ccf,
                                planet_orbital_inclination=90.0, kp_factor=2.0,
                                n_kp=None, n_vr=None):
    # Initializations
    if n_kp is None:
        n_kp = np.size(velocities_ccf)

    if n_vr is None:
        n_vr = np.size(velocities_ccf)

    n_integrations = np.size(orbital_longitudes)

    # Get Kp space
    kps = np.linspace(
        -planet_max_radial_orbital_velocity * kp_factor,
        planet_max_radial_orbital_velocity * kp_factor,
        n_kp
    )

    # Calculate the planet relative velocities in the Kp space
    planet_relative_velocities = system_radial_velocities + np.array([
        Planet.calculate_planet_radial_velocity(
            planet_max_radial_orbital_velocity=kp,
            planet_orbital_inclination=planet_orbital_inclination,
            orbital_longitude=orbital_longitudes  # phase to longitude (deg)
        ) for kp in kps
    ])

    # Set rest velocity space boundaries taking the CCF velocities into account to prevent out-of-bound interpolation
    v_planet_min = np.min(planet_relative_velocities)
    v_planet_max = np.max(planet_relative_velocities)

    v_rest_min = np.max((
        -planet_max_radial_orbital_velocity * kp_factor,
        np.min(velocities_ccf) - v_planet_min
    ))
    v_rest_max = np.min((
        planet_max_radial_orbital_velocity * kp_factor,
        np.max(velocities_ccf) - v_planet_max
    ))
    v_rest = np.linspace(v_rest_min, v_rest_max, n_vr)

    # Get velocity space
    velocities = np.zeros((n_kp, n_integrations, n_vr))

    for i in range(n_kp):
        for j in range(n_integrations):
            velocities[i, j] = v_rest + planet_relative_velocities[i, j]

    return velocities, kps, v_rest


def get_co_added_cross_correlation(wavelength_data, data, wavelength_model, model,
                                   planet_max_radial_orbital_velocity, line_spread_function_fwhm,
                                   orbital_phases_ccf,
                                   planet_orbital_inclination=90.0,
                                   system_radial_velocity=0.0, sum_ccf=False, normalize_ccf=True, calculate_snr=False,
                                   co_added_ccf_peak_width=15.6e5, pixels_per_resolution_element=2,
                                   kp_factor_ccf=1.0, kp_factor_co_added_ccf=2.0, extra_velocity_factor_ccf=0.25,
                                   n_kp=None, n_vr=None, full=False):
    """Calculate the cross correlation, then the co-added cross correlation.

    Args:
        wavelength_data:
        data:
        wavelength_model:
        model:
        planet_max_radial_orbital_velocity:
        line_spread_function_fwhm: (cm.s-1)
        orbital_phases_ccf:
        planet_orbital_inclination:
        system_radial_velocity:
        sum_ccf:
        normalize_ccf:
        calculate_snr:
        co_added_ccf_peak_width
        pixels_per_resolution_element:
        kp_factor_ccf:
        kp_factor_co_added_ccf:
        extra_velocity_factor_ccf:
        n_kp:
        n_vr:
        full:

    Returns:

    """
    ccf, velocities_ccf = calculate_cross_correlation(
        wavelength_data=wavelength_data,
        data=data,
        wavelength_model=wavelength_model,
        model=model,
        planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
        line_spread_function_fwhm=line_spread_function_fwhm,
        system_radial_velocity=system_radial_velocity,
        pixels_per_resolution_element=pixels_per_resolution_element,
        kp_factor=kp_factor_ccf,
        extra_velocity_factor=extra_velocity_factor_ccf,
        normalize=normalize_ccf,
        full=False
    )

    ccf_ = copy.deepcopy(ccf)

    if sum_ccf:
        ccf_ = np.array([np.ma.sum(ccf_, axis=0)])

    ccf_tot, kps, v_rest = calculate_co_added_cross_correlation(
        cross_correlation=ccf_,
        orbital_phases_ccf=orbital_phases_ccf,
        velocities_ccf=velocities_ccf,
        system_radial_velocities=system_radial_velocity,
        planet_max_radial_orbital_velocity=planet_max_radial_orbital_velocity,
        kp_factor=kp_factor_co_added_ccf,
        planet_orbital_inclination=planet_orbital_inclination,
        n_kp=n_kp,
        n_vr=n_vr
    )

    if calculate_snr:
        ccf_tot_sn = calculate_co_added_ccf_snr(
            co_added_cross_correlation=ccf_tot,
            vr_space=v_rest,
            vr_peak_width=co_added_ccf_peak_width
        )
    else:
        ccf_tot_sn = None

    if full:
        return ccf_tot, kps, v_rest, ccf_tot_sn, ccf, velocities_ccf, ccf_
    else:
        return ccf_tot, kps, v_rest, ccf_tot_sn


def normalize_cross_correlation(cross_correlation):
    return np.transpose(np.transpose(cross_correlation) - np.transpose(np.mean(cross_correlation, axis=2)))


def shift_cross_correlate(wavelength_data, data, wavelength_model, model, velocities_ccf, full=False):
    # Initialization
    n_detectors, n_integrations, n_spectral_pixels = np.shape(data)
    n_velocities = np.size(velocities_ccf)

    # Shift the wavelengths
    wavelength_shift = np.zeros((n_velocities, np.size(wavelength_model)))
    models_shift = np.zeros((n_detectors, n_velocities, n_spectral_pixels))

    if np.ndim(wavelength_data) == 3:  # time-dependent wavelengths
        print(f"3D data wavelengths detected: assuming no time (axis 1) dependency")
        wavelengths = copy.copy(wavelength_data[:, 0, :])
    else:
        wavelengths = copy.copy(wavelength_data)

    # Calculate Doppler shifted model wavelengths
    for j in range(n_velocities):
        wavelength_shift[j, :] = doppler_shift(
            wavelength_0=wavelength_model,
            velocity=velocities_ccf[j]
        )

    # Rebin model to data wavelengths and shift
    for i in range(n_detectors):
        for k in range(n_velocities):
            models_shift[i, k, :] = \
                fr.rebin_spectrum(wavelength_shift[k, :], model, wavelengths[i, :])

    if isinstance(data, np.ma.masked_array):
        data_ = data.filled(np.nan)  # fill masked values with NaN to be handled properly by cross_correlate_3d
    else:
        data_ = copy.deepcopy(data)

    data_ = np.moveaxis(np.tile(data_, (n_velocities, 1, 1, 1)), 0, 1)
    model_ = np.moveaxis(np.tile(models_shift, (n_integrations, 1, 1, 1)), 0, 2)

    # Calculate cross correlation
    ccf = np.moveaxis(cross_correlate_3d(
        matrix_1=data_,
        matrix_2=model_
    ), 1, 2)

    if full:
        return ccf, models_shift, wavelength_shift
    else:
        return ccf
