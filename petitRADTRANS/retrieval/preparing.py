"""
Useful functions for data reduction.
"""
import copy
import sys
import warnings

import numpy as np


def __init_pipeline(spectrum, uncertainties):
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

    return reduced_data, reduction_matrix, reduced_data_uncertainties


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


def __sysrem_iteration(spectrum_uncertainties_squared, uncertainties_squared_inverted, c, shape_a, shape_c):
    """SYSREM iteration.
    For the first iteration, c should be 1.
    The inputs are chosen in order to maximize speed.

    Args:
        spectrum_uncertainties_squared: spectral data to correct over the uncertainties ** 2 (..., exposure, wavelength)
        uncertainties_squared_inverted: invers of the squared uncertainties on the data (..., exposure, wavelength)
        c: 2-D matrix (..., exposures, wavelengths) containing the a-priori "extinction coefficients"
        shape_a: intermediate shape for "airmass" estimation (wavelength, ..., exposure)
        shape_c: intermediate shape for "extinction coefficients" estimation (exposure, ..., wavelength)

    Returns:
        The lower-rank estimation of the spectrum (systematics), and the estimated "extinction coefficients"
    """
    # Get the "airmass" (time variation of a pixel), not related to the true airmass
    a = np.sum(c * spectrum_uncertainties_squared, axis=-1) / \
        np.sum(c ** 2 * uncertainties_squared_inverted, axis=-1)

    # Tile into a (..., exposure, wavelength) matrix
    a = np.moveaxis(
        a * np.ones(shape_a),
        0,
        -1
    )

    # Recalculate the best fitting "extinction coefficients", not related to the true extinction coefficients
    c = np.sum(a * spectrum_uncertainties_squared, axis=-2) / \
        np.sum(a ** 2 * uncertainties_squared_inverted, axis=-2)

    # Tile into a (..., exposure, wavelength) matrix
    c = np.moveaxis(
        c * np.ones(shape_c),
        0,
        -2
    )

    return a * c, c


def __sysrem_iteration2(spectrum_uncertainties_squared, uncertainties_squared_inverted, a, shape_a, shape_c):
    """SYSREM iteration.
    For the first iteration, c should be 1.
    The inputs are chosen in order to maximize speed.

    Args:
        spectrum_uncertainties_squared: spectral data to correct over the uncertainties ** 2 (..., exposure, wavelength)
        uncertainties_squared_inverted: invers of the squared uncertainties on the data (..., exposure, wavelength)
        c: 2-D matrix (..., exposures, wavelengths) containing the a-priori "extinction coefficients"
        shape_a: intermediate shape for "airmass" estimation (wavelength, ..., exposure)
        shape_c: intermediate shape for "extinction coefficients" estimation (exposure, ..., wavelength)

    Returns:
        The lower-rank estimation of the spectrum (systematics), and the estimated "extinction coefficients"
    """
    # Recalculate the best fitting "extinction coefficients", not related to the true extinction coefficients
    c = np.sum(a * spectrum_uncertainties_squared, axis=-2) / \
        np.sum(a ** 2 * uncertainties_squared_inverted, axis=-2)

    # Tile into a (..., exposure, wavelength) matrix
    c = np.moveaxis(
        c * np.ones(shape_c),
        0,
        -2
    )

    # Get the "airmass" (time variation of a pixel), not related to the true airmass
    a = np.sum(c * spectrum_uncertainties_squared, axis=-1) / \
        np.sum(c ** 2 * uncertainties_squared_inverted, axis=-1)

    # Tile into a (..., exposure, wavelength) matrix
    a = np.moveaxis(
        a * np.ones(shape_a),
        0,
        -1
    )

    return a * c, a


def bias_pipeline_metric(reduced_true_model, reduced_mock_observations,
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
    degrees_of_freedom = polynomial_fit_degree + 1

    if spectrum.shape[1] <= degrees_of_freedom:
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
        masked_det = np.ma.masked_less_equal(det, 0)
        log_det_t = np.ma.log(np.transpose(masked_det))
        weights[i][masked_det.mask] = 0  # polyfit doesn't take masks into account, so set weight of masked values to 0

        # Fit each wavelength column
        for k, log_wavelength_column in enumerate(log_det_t):
            if weights[i, np.nonzero(weights[i, :, k]), k].size > degrees_of_freedom:
                fit_parameters = np.polynomial.Polynomial.fit(
                    x=airmass, y=log_wavelength_column, deg=polynomial_fit_degree, w=weights[i, :, k]
                )
                fit_function = np.polynomial.Polynomial(fit_parameters.convert().coef)
                telluric_lines_fits[i, :, k] = fit_function(airmass)
            else:
                telluric_lines_fits[i, :, k] = 0

                warnings.warn("not all columns have enough valid points for fitting")

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
    degrees_of_freedom = polynomial_fit_degree + 1

    if spectrum.shape[2] <= degrees_of_freedom:
        warnings.warn(f"not enough points in wavelengths axis ({spectrum.shape[2]}) "
                      f"for a meaningful correction with the requested fit degree ({polynomial_fit_degree}). "
                      f"At least {polynomial_fit_degree + 2} wavelengths axis points are required. "
                      f"Increase the number of wavelengths axis points to decrease correction bias, "
                      f"or decrease the polynomial fit degree.")

    spectral_data_corrected, reduction_matrix, pipeline_uncertainties = __init_pipeline_outputs(
        spectrum, reduction_matrix, uncertainties
    )

    if uncertainties is not None:
        # Ensure low weights within tellurics, but gives more weight to noisy non-telluric wavelengths
        weights = copy.deepcopy(uncertainties)
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
            throughput_fits[i, :, :] < mask_threshold,
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
            # Count number of non-masked points minus degrees of freedom in each wavelength axes
            if np.ndim(pipeline_uncertainties.mask) != np.ndim(pipeline_uncertainties):
                raise ValueError(f"number of dimensions of the mask of pipeline uncertainties "
                                 f"({np.ndim(pipeline_uncertainties.mask)}) does not matches pipeline uncertainties "
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
        tellurics_mask_threshold: mask wavelengths where the atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the Earth atmospheric transmittance
        apply_throughput_removal: if True, apply the throughput removal correction
        apply_telluric_lines_removal: if True, apply the telluric lines removal correction
        correct_uncertainties:
        full: if True, return the reduced matrix and reduced uncertainties in addition to the reduced spectrum

    Returns:
        Reduced spectral data (and reduction matrix and uncertainties after reduction if full is True)
    """
    reduced_data, reduction_matrix, reduced_data_uncertainties = __init_pipeline(spectrum, uncertainties)

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


def preparing_pipeline_sysrem(spectrum, uncertainties, wavelengths, n_iterations_max=10, convergence_criterion=1e-3,
                              tellurics_mask_threshold=0.8, polynomial_fit_degree=1,
                              apply_throughput_removal=True, apply_telluric_lines_removal=True,
                              correct_uncertainties=True,
                              subtract=False, remove_mean=True, full=False, verbose=False, **kwargs):
    """SYSREM preparing pipeline.
    SYSREM tries to find the coefficients a and c such as:
        S**2 = sum_ij ((spectrum_ij - a_j * c_i) / uncertainties)**2
    is minimized. Several iterations can be performed. This assumes that the spectrum is deformed by a combination of
    linear effects.
    The coefficients a and c can be seen as estimates for any strong (linear) systematic effect in the data, they are
    not necessarily related to the airmass and extinction coefficients.

    Source: Tamuz et al. 2005 (doi:10.1111/j.1365-2966.2004.08585.x).
    Thanks to Alejandro Sanchez-Lopez (26-09-2017) for sharing his version of the algorithm.

    Args:
        spectrum: spectral data to correct
        uncertainties: uncertainties on the data
        wavelengths: wavelengths of the data
        n_iterations_max: maximum number of SYSREM iterations
        convergence_criterion: SYSREM convergence criterion
        tellurics_mask_threshold: mask wavelengths where the atmospheric transmittance estimate is below this value
        polynomial_fit_degree: degree of the polynomial fit of the instrumental deformations
        apply_throughput_removal: if True, divide the spectrum by its mean over wavelengths
        apply_telluric_lines_removal: if True, apply the telluric lines removal correction
        correct_uncertainties:
        subtract: if True, subtract the fitted systematics to the spectrum instead of dividing them
        remove_mean:
        full: if True, return the reduced matrix and reduced uncertainties in addition to the reduced spectrum
        verbose: if True, print the convergence status at each iteration

    Returns:
        Reduced spectral data (and reduction matrix and uncertainties after reduction if full is True)
    """
    # Pre-preparation
    reduction_matrix = np.ones(spectrum.shape)
    reduced_data_uncertainties = copy.deepcopy(uncertainties)
    reduced_data = copy.deepcopy(spectrum)

    if apply_throughput_removal:
        if wavelengths is None:
            if verbose:
                print("Dividing by the mean...")

            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_mean(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                uncertainties=reduced_data_uncertainties
            )
        else:
            if verbose:
                print(f"Dividing by an order-{polynomial_fit_degree} polynomial...")

            reduced_data, reduction_matrix, reduced_data_uncertainties = remove_throughput_fit(
                spectrum=spectrum,
                reduction_matrix=reduction_matrix,
                wavelengths=wavelengths,
                uncertainties=reduced_data_uncertainties,
                mask_threshold=sys.float_info.min,
                polynomial_fit_degree=polynomial_fit_degree,
                correct_uncertainties=correct_uncertainties
            )

    reduced_data = np.ma.masked_where(reduced_data < tellurics_mask_threshold, reduced_data)

    if remove_mean and subtract:
        if verbose:
            print("Subtracting mean...")

        reduced_data = np.moveaxis(
            np.moveaxis(reduced_data, -1, 0) - np.ma.average(reduced_data, axis=-1, weights=uncertainties),
            0,
            -1
        )

    if not apply_telluric_lines_removal:
        if full:
            return reduced_data, reduction_matrix, reduced_data_uncertainties
        else:
            return reduced_data

    # Initialize SYSREM meaningful variables
    if verbose:
        print("Starting SysRem...")

    spectrum_shape = list(reduced_data.shape)
    shape_a = copy.copy(spectrum_shape)
    shape_c = copy.copy(spectrum_shape)

    shape_a.insert(0, shape_a.pop(-1))
    shape_c.insert(0, shape_c.pop(-2))

    uncertainties_squared_inverted = 1 / reduced_data_uncertainties ** 2
    spectrum_uncertainties_squared = reduced_data * uncertainties_squared_inverted

    # Handle masked values
    if isinstance(spectrum_uncertainties_squared, np.ma.core.MaskedArray):
        uncertainties_squared_inverted[spectrum_uncertainties_squared.mask] = 0
        spectrum_uncertainties_squared = spectrum_uncertainties_squared.filled(0)

    # Iterate
    i = 0
    c = 1
    systematics_0 = np.zeros(reduced_data.shape)
    systematics = np.zeros(reduced_data.shape)

    for i in range(n_iterations_max):
        systematics, c = __sysrem_iteration(
            spectrum_uncertainties_squared=spectrum_uncertainties_squared,
            uncertainties_squared_inverted=uncertainties_squared_inverted,
            c=c,
            shape_a=shape_a,
            shape_c=shape_c
        )
        systematics[np.nonzero(np.logical_not(np.isfinite(systematics)))] = 0

        # Check for convergence
        if np.sum(np.abs(systematics_0 - systematics)) <= convergence_criterion * np.sum(np.abs(systematics_0)):
            if verbose:
                print(f"Iteration {i + 1} (max {n_iterations_max}): "
                      f"{np.sum(np.abs(systematics_0 - systematics)) / np.sum(np.abs(systematics_0))} "
                      f"(> {convergence_criterion})")
                print("Convergence reached!")

            break
        elif verbose and i > 0:
            print(f"Iteration {i} (max {n_iterations_max}): "
                  f"{np.sum(np.abs(systematics_0 - systematics)) / np.sum(np.abs(systematics_0))} "
                  f"(> {convergence_criterion})")

        systematics_0 = systematics

    if i == n_iterations_max - 1 \
            and np.sum(np.abs(systematics_0 - systematics)) > convergence_criterion * np.sum(np.abs(systematics_0)) \
            and convergence_criterion > 0:
        warnings.warn(
            f"convergence not reached in {n_iterations_max} iterations "
            f"({np.sum(np.abs(systematics_0 - systematics)) > convergence_criterion * np.sum(np.abs(systematics_0))} "
            f"> {convergence_criterion})"
        )

    # Mask where systematics are 0 to prevent division by 0 error
    systematics = np.ma.masked_equal(systematics, 0)

    # Remove the systematics from the spectrum
    '''
    This can also be done by subtracting the systematics from the spectrum, but dividing give almost the same results
    and this way the pipeline can be used in retrievals more effectively.
    '''
    if subtract:
        reduced_data -= systematics
    else:
        reduced_data /= systematics

    if full:
        if subtract:
            # With the subtractions, uncertainties should not be affected
            # TODO it can be argued that the uncertainties on the systematics should be taken into account
            reduction_matrix -= systematics
        else:
            reduction_matrix /= systematics
            reduced_data_uncertainties = reduced_data_uncertainties * np.abs(reduction_matrix)

        return reduced_data, reduction_matrix, reduced_data_uncertainties
    else:
        return reduced_data
