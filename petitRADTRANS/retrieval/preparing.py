"""
Useful functions for data reduction.
"""
import copy
import sys
import warnings

import numpy as np


def __init_pipeline_outputs(spectrum, reduction_matrix, uncertainties):
    if reduction_matrix is None:
        reduction_matrix = np.ma.ones(spectrum.shape)
        reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        spectral_data_corrected = np.ma.zeros(spectrum.shape)
        spectral_data_corrected.mask = copy.copy(spectrum.mask)

        if uncertainties is not None:
            pipeline_uncertainties = np.ma.masked_array(copy.copy(uncertainties))
            pipeline_uncertainties.mask = copy.copy(spectrum.mask)
        else:
            pipeline_uncertainties = None
    else:
        spectral_data_corrected = np.zeros(spectrum.shape)

        if uncertainties is not None:
            pipeline_uncertainties = copy.copy(uncertainties)
        else:
            pipeline_uncertainties = None

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def pipeline_validity_test(reduced_true_model, reduced_mock_observations,
                           mock_observations_reduction_matrix=None, mock_noise=None):
    if mock_observations_reduction_matrix is None:
        mock_observations_reduction_matrix = np.ones(reduced_true_model.shape)

    if mock_noise is None:
        mock_noise = np.zeros(reduced_true_model.shape)

    return 1 - (reduced_true_model - mock_noise * mock_observations_reduction_matrix) / reduced_mock_observations


def remove_noisy_wavelength_channels(spectrum, reduction_matrix, mean_subtract=False):
    for i, data in enumerate(spectrum):
        # Get standard deviation over time, for each wavelength channel
        time_standard_deviation = np.asarray([np.ma.std(data, axis=0)] * np.size(data, axis=0))

        # Mask channels where the standard deviation is greater than the total standard deviation
        data = np.ma.masked_where(
            time_standard_deviation > 3 * np.ma.std(data), data
        )

        spectrum[i, :, :] = data

    if mean_subtract:
        mean_spectra = np.mean(spectrum, axis=2)  # mean over wavelengths of each individual spectrum
        spectrum -= mean_spectra
        reduction_matrix -= mean_spectra

    return spectrum, reduction_matrix


def remove_telluric_lines_fit(spectrum, reduction_matrix, airmass, uncertainties=None, mask_threshold=1e-16,
                              polynomial_fit_degree=2, correct_uncertainties=True):
    """Remove telluric lines with a polynomial function.
    The telluric transmittance can be written as:
        T = exp(-airmass * optical_depth),
    hence the log of the transmittance can be written as a first order polynomial:
        log(T) ~ b * airmass + a.
    Using a 1st order polynomial might be not enough, as the atmospheric composition can change slowly over time. Using
    a second order polynomial, as in:
        log(T) ~ c * airmass ** 2 + b * airmass + a,
    might be safer.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        airmass: airmass of the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        correct_uncertainties:

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    if spectrum.shape[1] <= polynomial_fit_degree + 1:
        warnings.warn(f"not enough points in airmass axis ({spectrum.shape[1]}) "
                      f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                      f"At least {polynomial_fit_degree + 2} airmass axis points are required. "
                      f"Increase the number of airmass axis points to decrease correction bias, "
                      f"or decrease the polynomial fit degree.")

    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        # The log of the spectrum is fitted, so the weights must be the weights of the log
        spectrum = np.ma.masked_equal(spectrum, 0)
        uncertainties = np.ma.masked_equal(uncertainties, 0)
        weights = 1 / np.abs(uncertainties / spectrum)  # 1 / uncertainties of the log
        weights = weights.filled(0)
    else:
        weights = np.ones(spectrum.shape)

    spectrum[np.nonzero(np.equal(weights, 0))] = 0  # ensure no invalid values are hidden where weight = 0

    telluric_lines_fits = np.ma.zeros(spectral_data_corrected.shape)

    # Correction
    for i, det in enumerate(spectrum):
        # Mask wavelength columns where at least one value is lower or equal to 0, to avoid invalid log values
        masked_det = np.ma.masked_where(np.ones(det.shape) * np.min(det, axis=0) <= 0, det)
        log_det_t = np.ma.log(np.transpose(masked_det))
        weights[i][masked_det.mask] = 0  # polyfit doesn't take masks into account, so set weight of masked values to 0

        # Fit each wavelength column
        for k, log_wavelength_column in enumerate(log_det_t):
            if np.allclose(weights[i, :, k], 0, atol=sys.float_info.min):  # skip fully masked columns
                telluric_lines_fits[i, :, k] = 0

                continue

            fit_parameters = np.polynomial.Polynomial.fit(
                x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
            )
            fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)
            telluric_lines_fits[i, :, k] = fit_function(airmass)

        # Calculate telluric transmittance estimate
        telluric_lines_fits[i, :, :] = np.exp(telluric_lines_fits[i, :, :])

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        telluric_lines_fits[i, :, :] = np.ma.masked_where(
            np.ones(telluric_lines_fits[i].shape) * np.min(telluric_lines_fits[i, :, :], axis=0) < mask_threshold,
            telluric_lines_fits[i, :, :]
        )
        telluric_lines_fits[i, :, :] = np.ma.masked_where(
            masked_det.mask, telluric_lines_fits[i, :, :]
        )

        # Apply correction
        spectral_data_corrected[i, :, :] = det
        spectral_data_corrected[i, :, :] /= telluric_lines_fits[i, :, :]
        reduction_matrix[i, :, :] /= telluric_lines_fits[i, :, :]

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties /= np.abs(telluric_lines_fits)

        if correct_uncertainties:
            degrees_of_freedom = 1 + polynomial_fit_degree

            if np.ndim(pipeline_uncertainties.mask) != np.ndim(pipeline_uncertainties):
                raise ValueError(f"number of dimensions of the mask of pipeline uncertainties "
                                 f"({np.ndim(pipeline_uncertainties.mask)}) does not matches pipeline uncertainties"
                                 f"number of dimension ({np.ndim(pipeline_uncertainties)})")

            # Count number of non-masked points minus degrees of freedom in each time axes
            valid_points = airmass.size - np.sum(pipeline_uncertainties.mask, axis=1) - degrees_of_freedom
            valid_points[np.less(valid_points, 0)] = 0

            # Correct from fitting effect
            # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
            # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
            # This way the uncertainties truly reflect the standard deviation of the data
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 0, 1)
            pipeline_uncertainties *= np.sqrt(valid_points / airmass.size)
            pipeline_uncertainties = np.ma.masked_less_equal(pipeline_uncertainties, 0)
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 1, 0)

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_telluric_lines_mean(spectrum, reduction_matrix, uncertainties=None, mask_threshold=1e-16):
    """Remove the telluric lines using the weighted arithmetic mean over time.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    mean_spectrum_time = np.ma.average(spectrum, axis=1, weights=weights)
    mean_spectrum_time = np.ma.masked_array(mean_spectrum_time)  # ensure that it is a masked array

    if np.ndim(mean_spectrum_time.mask) == 0:
        mean_spectrum_time.mask = np.zeros(mean_spectrum_time.shape, dtype=bool)

    if np.ndim(spectral_data_corrected.mask) == 0:
        spectral_data_corrected.mask = np.zeros(spectral_data_corrected.shape, dtype=bool)

    # Correction
    if isinstance(spectral_data_corrected, np.ma.core.MaskedArray):
        for i, data in enumerate(spectrum):
            mean_spectrum_time[i] = np.ma.masked_where(
                mean_spectrum_time[i] < mask_threshold, mean_spectrum_time[i]
            )
            spectral_data_corrected.mask[i, :, :] = mean_spectrum_time.mask[i]
            reduction_matrix.mask[i, :, :] = mean_spectrum_time.mask[i]
            spectral_data_corrected[i, :, :] = data / mean_spectrum_time[i]
            reduction_matrix[i, :, :] /= mean_spectrum_time[i]
    else:
        for i, data in enumerate(spectrum):
            spectral_data_corrected[i, :, :] = data / mean_spectrum_time[i]
            reduction_matrix[i, :, :] /= mean_spectrum_time[i]

    if uncertainties is not None:
        for i, data in enumerate(spectrum):
            pipeline_uncertainties[i, :, :] /= np.abs(mean_spectrum_time[i])

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_throughput_fit(spectrum, reduction_matrix, wavelengths, uncertainties=None, mask_threshold=1e-16,
                          polynomial_fit_degree=2, correct_uncertainties=True):
    """Remove variable throughput with a polynomial function.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        wavelengths: wavelengths of the data
        uncertainties: uncertainties on the data
        mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        correct_uncertainties:

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    if spectrum.shape[2] <= polynomial_fit_degree + 1:
        warnings.warn(f"not enough points in wavelengths axis ({spectrum.shape[2]}) "
                      f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                      f"At least {polynomial_fit_degree + 2} wavelengths axis points are required. "
                      f"Increase the number of wavelengths axis points to decrease correction bias, "
                      f"or decrease the polynomial fit degree.")

    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = copy.deepcopy(uncertainties)  # ensure low weights within tellurics, but gives more weight to noisy non-telluric wvls
        weights = weights.filled(0)  # polyfit doesn't take masks into account, so set weight of masked values to 0
    else:
        weights = np.ones(spectrum.shape)

    spectrum[np.nonzero(np.equal(weights, 0))] = 0  # ensure no invalid values are hidden where weight = 0

    throughput_fits = np.ma.zeros(spectral_data_corrected.shape)

    if np.ndim(wavelengths) == 3:
        print('Assuming same wavelength solution for each observations, taking wavelengths of observation 0')

    # Correction
    for i, det in enumerate(spectrum):
        if np.ndim(wavelengths) == 1:
            wvl = wavelengths
        elif np.ndim(wavelengths) == 2:
            wvl = wavelengths[i, :]
        elif np.ndim(wavelengths) == 3:
            wvl = wavelengths[i, 0, :]
        else:
            raise ValueError(f"wavelengths must have at most 3 dimensions, but has {np.ndim(wavelengths)}")

        # Fit each observation
        for j, observation in enumerate(det):
            fit_parameters = np.polynomial.Polynomial.fit(
                x=wvl, y=observation, deg=polynomial_fit_degree, w=weights[i, j, :]
            )
            fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)
            throughput_fits[i, j, :] = fit_function(wvl)

        # Apply mask where estimate is lower than the threshold, as well as the data mask
        throughput_fits[i, :, :] = np.ma.masked_where(
            np.ones(throughput_fits[i].shape) * np.min(throughput_fits[i, :, :], axis=0) < mask_threshold,
            throughput_fits[i, :, :]
        )
        throughput_fits[i, :, :] = np.ma.masked_where(
            det.mask, throughput_fits[i, :, :]
        )

        # Apply correction
        spectral_data_corrected[i, :, :] = det
        spectral_data_corrected[i, :, :] /= throughput_fits[i, :, :]
        reduction_matrix[i, :, :] /= throughput_fits[i, :, :]

    # Propagation of uncertainties
    if uncertainties is not None:
        pipeline_uncertainties /= np.abs(throughput_fits)

        if correct_uncertainties:
            degrees_of_freedom = 1 + polynomial_fit_degree

            # Count number of non-masked points minus degrees of freedom in each wavelength axes
            if np.ndim(pipeline_uncertainties.mask) != np.ndim(pipeline_uncertainties):
                raise ValueError(f"number of dimensions of the mask of pipeline uncertainties "
                                 f"({np.ndim(pipeline_uncertainties.mask)}) does not matches pipeline uncertainties"
                                 f"number of dimension ({np.ndim(pipeline_uncertainties)})")

            valid_points = wavelengths.size - np.sum(pipeline_uncertainties.mask, axis=2) - degrees_of_freedom
            valid_points[np.less(valid_points, 0)] = 0

            # Correct from fitting effect
            # Uncertainties are assumed unbiased, but fitting induces a bias, so here the uncertainties are voluntarily
            # biased (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
            # This way the uncertainties truly reflect the standard deviation of the data
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 2, 0)
            pipeline_uncertainties *= np.sqrt(valid_points / wavelengths.size)
            pipeline_uncertainties = np.ma.masked_less_equal(pipeline_uncertainties, 0)
            pipeline_uncertainties = np.moveaxis(pipeline_uncertainties, 0, 2)

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def remove_throughput_mean(spectrum, reduction_matrix=None, uncertainties=None):
    """Correct for the variable throughput using the weighted arithmetic mean over wavelength.

    Args:
        spectrum: spectral data to correct
        reduction_matrix: matrix storing all the operations made to reduce the data
        uncertainties: uncertainties on the data

    Returns:
        Corrected spectral data, reduction matrix and uncertainties after correction
    """
    # Initialization
    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        weights = 1 / uncertainties
        weights[weights.mask] = 0
    else:
        weights = np.ones(spectrum.shape)

    # Correction
    for i, data in enumerate(spectrum):
        if isinstance(spectrum, np.ma.core.MaskedArray):
            correction_coefficient = np.ma.average(data, axis=1, weights=weights[i])
        elif isinstance(spectrum, np.ndarray):
            correction_coefficient = np.average(data, axis=1, weights=weights[i])
        else:
            raise ValueError(f"spectral_data must be a numpy.ndarray or a numpy.ma.core.MaskedArray, "
                             f"but is of type '{type(spectrum)}'")

        spectral_data_corrected[i, :, :] = np.transpose(np.transpose(data) / correction_coefficient)
        reduction_matrix[i, :, :] = np.transpose(np.transpose(reduction_matrix[i, :, :]) / correction_coefficient)

        if uncertainties is not None:
            pipeline_uncertainties[i, :, :] = np.transpose(
                np.transpose(pipeline_uncertainties[i, :, :]) / np.abs(correction_coefficient)
            )

    return spectral_data_corrected, reduction_matrix, pipeline_uncertainties


def preparing_pipeline(spectrum, uncertainties=None,
                       wavelengths=None, airmass=None, tellurics_mask_threshold=0.1, polynomial_fit_degree=1,
                       apply_throughput_removal=True, apply_telluric_lines_removal=True, correct_uncertainties=True,
                       full=False, **kwargs):
    """Removes the telluric lines and variable throughput of some data.
    If airmass is None, the Earth atmospheric transmittance is assumed to be time-independent, so telluric transmittance
    will be fitted using the weighted arithmetic mean. Otherwise, telluric transmittance are fitted with a polynomial.

    Args:
        spectrum: spectral data to correct
        uncertainties: uncertainties on the data
        wavelengths: wavelengths of the data
        airmass: airmass of the data
        tellurics_mask_threshold: mask wavelengths where the Earth atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        apply_throughput_removal: if True, apply the throughput removal correction
        apply_telluric_lines_removal: if True, apply the telluric lines removal correction
        correct_uncertainties:
        full: if True, return the reduced matrix and reduced uncertainties in addition to the reduced spectrum

    Returns:
        Reduced spectral data (and reduction matrix and uncertainties after reduction if full is True)
    """
    # Initialize reduction matrix
    reduction_matrix = np.ma.ones(spectrum.shape)
    reduction_matrix.mask = np.zeros(spectrum.shape, dtype=bool)

    # Initialize reduced data and pipeline noise
    reduced_data = copy.copy(spectrum)

    if isinstance(spectrum, np.ma.core.MaskedArray):
        reduced_data.mask = copy.copy(spectrum.mask)

        if uncertainties is not None:
            reduced_data_uncertainties = np.ma.masked_array(copy.copy(uncertainties))
            reduced_data_uncertainties.mask = copy.copy(spectrum.mask)
        else:
            reduced_data_uncertainties = None
    else:
        if uncertainties is not None:
            reduced_data_uncertainties = copy.copy(uncertainties)
        else:
            reduced_data_uncertainties = None

    # Apply corrections
    if apply_throughput_removal:
        if wavelengths is None:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_mean(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                uncertainties=reduced_data_uncertainties
            )
        else:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_fit(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                wavelengths=wavelengths,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=sys.float_info.min,
                polynomial_fit_degree=2,
                correct_uncertainties=correct_uncertainties
            )

    if apply_telluric_lines_removal:
        if airmass is None:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_telluric_lines_mean(
                spectrum=reduced_data,
                reduction_matrix=reduction_matrix,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=tellurics_mask_threshold
            )
        else:
            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_telluric_lines_fit(
                spectrum=reduced_data,
                reduction_matrix=reduction_matrix,
                airmass=airmass,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=tellurics_mask_threshold,
                polynomial_fit_degree=polynomial_fit_degree,
                correct_uncertainties=correct_uncertainties
            )

    if full:
        return reduced_data, reduction_matrix, reduced_data_uncertainties
    else:
        return reduced_data
