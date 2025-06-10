"""Stores useful mathematical functions.
"""
import numpy as np
import jax
import jax.numpy.typing as jnpt
import jax.numpy as jnp
from scipy.special import erf, erfinv, lambertw
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from petitRADTRANS.fortran_convolve import fortran_convolve as fconvolve

def bayes_factor2sigma(bayes_factor: float) -> float | jax.typing.ArrayLike[jnp.floating]:
    """
    Convert a Bayes factor, or "evidence", into a sigma significance. For Bayes factor higher than exp(25), the function
    is approximated with a square root function.
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    Molliere part by Paul Mollière.
    :param bayes_factor: Bayes factor (aka "evidence")
    :return: sigma significance
    """
    molliere_threshold = 72004899337  # ~exp(25), roughly where numerical errors start to be significant
    is_scalar = False

    if not hasattr(bayes_factor, '__iter__'):
        is_scalar = True
        bayes_factor = jnp.array([bayes_factor])
    elif not isinstance(bayes_factor, jnp.ndarray):
        bayes_factor = jnp.array(bayes_factor)

    benneke_part = jnp.nonzero(jnp.less_equal(bayes_factor, molliere_threshold))
    molliere_part = jnp.nonzero(jnp.greater(bayes_factor, molliere_threshold))

    sigma = jnp.zeros(jnp.shape(bayes_factor))

    if jnp.size(benneke_part) > 0:
        a = -1 / (jnp.exp(1) * bayes_factor[benneke_part])
        rho = jnp.real(jnp.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        sigma[benneke_part] = jnp.sqrt(2) * erfinv(1 - rho)

    if jnp.size(molliere_part) > 0:
        a = -1 / (jnp.exp(1) * molliere_threshold)
        rho = jnp.real(jnp.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        offset = jnp.sqrt(2) * erfinv(1 - rho) - jnp.sqrt(2 * jnp.log(molliere_threshold))

        sigma[molliere_part] = jnp.sqrt(2 * jnp.log(bayes_factor[molliere_part])) + offset

    if is_scalar:
        return sigma[0]
    else:
        return sigma


def box_car_conv(array: jax.typing.ArrayLike, points: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
    res = jnp.zeros_like(array)
    len_arr = len(array)

    for i in range(len(array)):
        if (i - points / 2 >= 0) and (i + points / 2 <= len_arr + 1):
            smooth_val = array[i - points / 2:i + points / 2]
            res[i] = jnp.sum(smooth_val) / len(smooth_val)
        elif i + points / 2 > len_arr + 1:
            len_use = len_arr + 1 - i
            smooth_val = array[i - len_use:i + len_use]
            res[i] = jnp.sum(smooth_val) / len(smooth_val)
        elif i - points / 2 < 0:
            smooth_val = array[:max(2 * i, 1)]
            res[i] = jnp.sum(smooth_val) / len(smooth_val)

    return res


def calculate_chi2(
    data: float | jax.typing.ArrayLike,
    model: float | jax.typing.ArrayLike,
    uncertainties: float | jax.typing.ArrayLike
) -> float:
    return float(jnp.sum(((data - model) / uncertainties) ** 2))


def calculate_reduced_chi2(
    data: float | jax.typing.ArrayLike,
    model: float | jax.typing.ArrayLike,
    uncertainties: float | jax.typing.ArrayLike,
    degrees_of_freedom: int = 0
) -> float | jax.typing.ArrayLike:
    return calculate_chi2(data, model, uncertainties) / (jnp.size(data) - degrees_of_freedom)


def calculate_uncertainty(
    derivatives: jax.typing.ArrayLike,
    uncertainties: jax.typing.ArrayLike,
    covariance_matrix: jax.typing.ArrayLike = None
) -> jax.typing.ArrayLike | None:
    """
    Calculate the uncertainty of a function f(x, y, ...) with uncertainties on x, y, ... and Pearson's correlation
    coefficients between x, y, ...
    The function must be (approximately) linear with its variables within the uncertainties of said variables.
    For independent variables, set the covariance matrix to identity.
    Uncertainties can be asymmetric, in that case for N variables, use a (N, 2) array for the uncertainties.
    Asymmetric uncertainties are handled **the wrong way** (see source 2), but it is better than nothing.

    Sources:
        1. https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        2. https://phas.ubc.ca/~oser/p509/Lec_10.pdf
        3. http://math.jacobs-university.de/oliver/teaching/jacobs/fall2015/esm106/handouts/error-propagation.pdf
    Args:
        derivatives:
            Partial derivatives of the function with respect to each variable (df/dx, df/dy, ...)
        uncertainties:
            Uncertainties of each variable (either a 1D-array or a 2D-array containing - and + unc.)
        covariance_matrix:
            Covariance matrix between the variables, by default set to the identity matrix

    Returns:
        A size-2 array containing the - and + uncertainties of the function
    """
    if covariance_matrix is None:
        covariance_matrix = jnp.identity(jnp.size(derivatives))

    if jnp.ndim(uncertainties) == 1:
        sigmas = derivatives * uncertainties

        return jnp.sqrt(jnp.matmul(sigmas, jnp.matmul(covariance_matrix, jnp.transpose(sigmas))))
    elif jnp.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return jnp.sqrt(jnp.array([  # beware, this is not strictly correct
            jnp.matmul(sigma_less, jnp.matmul(covariance_matrix, jnp.transpose(sigma_less))),
            jnp.matmul(sigma_more, jnp.matmul(covariance_matrix, jnp.transpose(sigma_more)))
        ]))

    return None


def compute_resolving_power(array: jax.typing.ArrayLike[float]) -> float:
    """Compute the mean resolving power of an array.

    Args:
        array:
            A 1-D array.

    Returns:
        The mean resolving power of the array.
    """
    return jnp.mean(
        array[:-1] / jnp.diff(array) + 0.5,
        dtype=float
    )


def convolve(input_wavelength, input_flux, instrument_res):
    r"""
    This function convolves a model spectrum to the instrumental wavelength
    using the provided data_resolution
    Args:
        input_wavelength : numpy.ndarray
            The wavelength grid of the model spectrum
        input_flux : numpy.ndarray
            The flux as computed by the model
        instrument_res : float
            :math:`\\lambda/\\Delta \\lambda`, the width of the gaussian kernel to convolve with the model spectrum.

    Returns:
        flux_lsf
            The convolved spectrum.
    """
    if isinstance(instrument_res, jnp.ndarray):
        return fconvolve.variable_width_convolution(input_wavelength, input_flux, instrument_res)
    
    # From talking to Ignas: delta lambda of resolution element
    # is FWHM of the LSF's standard deviation, hence:
    sigma_lsf = 1. / instrument_res / (2. * jnp.sqrt(2. * jnp.log(2.)))

    # The input spacing of petitRADTRANS is 1e3, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the spacing is constant
    # as a function of wavelength
    spacing = jnp.mean(2. * jnp.diff(input_wavelength) / (input_wavelength[1:] + input_wavelength[:-1]))

    # Calculate the sigma to be used in the gauss filter in units
    # of input wavelength bins
    sigma_lsf_gauss_filter = sigma_lsf / spacing

    flux_lsf = gaussian_filter(input_flux,
                                sigma=sigma_lsf_gauss_filter,
                                mode='nearest')

    return flux_lsf


def convolve_and_sample_Rvers(
        wavelengths,
        resolutions,
        model_wavelengths,
        model_fluxes,
        channel_width=None,
        num_sigma=3):
    """
    From Jerry Xuan circa 2024
    
    Simulate the observations of a model. 
    Convolves the model with a variable Gaussian LSF, sampled at each desired spectral channel.

    Args:
        wavelengths: the wavelengths desired (length of N_output)
        resolutions: the R of each wavelengths (length of N_output)
        model_wavelengths: the wavelengths of the model (length of N_model)
        model_fluxes: the fluxes of the model (length of N_model)
        channel_width: (optional) the full width of each wavelength channel in units of wavelengths (length of N_output)
        num_sigma (float): number of +/- sigmas to evaluate the LSF to. 

    Returns:
        output_model: the fluxes in each of the wavelength channels (length of N_output)
    """

    # JX added to use function with input R, instead of LSF FWHM. 
    # first get FWHM of LSF from lambda / R. Then convert FWHM to stddev of Gaussian
    sigmas_wvs = wavelengths / resolutions / (2*jnp.sqrt(2*jnp.log(2)))  # corrected a math error here, July 15 2024

    model_in_range = jnp.where((model_wavelengths >= jnp.min(wavelengths)) & (model_wavelengths < jnp.max(wavelengths)))
    dwv_model = jnp.abs(model_wavelengths[model_in_range] - jnp.roll(model_wavelengths[model_in_range], 1))
    dwv_model[0] = dwv_model[1]

    filter_size = int(jnp.ceil(jnp.max((2 * num_sigma * sigmas_wvs)/jnp.min(dwv_model)) ))
    filter_coords = jnp.linspace(-num_sigma, num_sigma, filter_size)
    #print(jnp.min(wavelengths), jnp.max(wavelengths), jnp.min(model_wavelengths), jnp.max(model_wavelengths), dwv_model, jnp.min(dwv_model))
    
    try:
        filter_coords = jnp.tile(filter_coords, [wavelengths.shape[0], 1]) #  shape of (N_output, filter_size)
    except Exception as e:
        print(e)
        print('reached exception for ' + str(filter_size))

    filter_wv_coords = filter_coords * sigmas_wvs[:,None] + wavelengths[:,None] # model wavelengths we want
    lsf = jnp.exp(-filter_coords**2/2)/jnp.sqrt(2*jnp.pi)

    model_interp = interp1d(model_wavelengths, model_fluxes, kind='cubic', bounds_error=False)
    filter_model = model_interp(filter_wv_coords)

    output_model = jnp.nansum(filter_model * lsf, axis=1)/jnp.sum(lsf, axis=1)
    return output_model


def feature_scaling(array: jax.typing.ArrayLike, min_value: float = 0.0, max_value: float = 1.0) -> jax.typing.ArrayLike:
    """Bring all values of array between a min and max value.

    Args:
        array: array to normalize
        min_value: target minimum value
        max_value: target maximum value

    Returns:
        The normalized array with values between min_value and max_value.
    """
    return min_value + ((array - jnp.min(array)) * (max_value - min_value)) / (jnp.max(array) - jnp.min(array))


def filter_spectrum_with_spline(wavelengths, fluxes, uncertainties=None, x_nodes=None, M_spline=None):
    """
    From BREADS, BSD 3-Clause License

    Copyright (c) 2024, jruffio
    """
    if uncertainties is None:
        uncertainties = jnp.ones(fluxes.shape)

    if M_spline is None:
        M_spline = linear_spline_interpolation(x_nodes, wavelengths, spline_degree=3)

    M = M_spline/uncertainties[:,None]
    d = fluxes/uncertainties
    where_finite = jnp.where(jnp.isfinite(d))
    M = M[where_finite[0],:]
    d = d[where_finite]

    paras = lsq_linear(M,d).x
    m = jnp.dot(M, paras)
    r = d - m

    # LPF_fluxes = jnp.zeros(fluxes.shape)+jnp.nan
    HPF_fluxes = jnp.zeros(fluxes.shape)+jnp.nan
    # LPF_fluxes[where_finite] = m*uncertainties[where_finite]
    HPF_fluxes[where_finite] = r*uncertainties[where_finite]

    return HPF_fluxes#,LPF_spec


def gaussian_weights1d(sigma: float, truncate: float = 4.0) -> jax.typing.ArrayLike:
    """Compute a 1D Gaussian convolution kernel.
    To be used with scipy.ndimage.convolve1d.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigma:
            Standard deviation for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sigma + 0.5)

    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma ** 2 * x ** 2)

    return phi_x / phi_x.sum()


def gaussian_weights_running(sigmas: jax.typing.ArrayLike, truncate: float = 4.0) -> jax.typing.ArrayLike:
    """Compute 1D Gaussian convolution kernels for an array of standard deviations.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigmas:
            Standard deviations for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * jnp.max(sigmas) + 0.5)

    x = jnp.arange(-radius, radius + 1)
    sd = jnp.tile(sigmas, (x.size, 1)).T

    phi_x = jnp.exp(-0.5 / sd ** 2 * x ** 2)

    return jnp.transpose(phi_x.T / phi_x.sum(axis=1))


def linear_spline_interpolation(x_knots, x_samples, spline_degree=3):
    """
    From BREADS, BSD 3-Clause License

    Copyright (c) 2024, jruffio
    
    Compute a spline based linear model.
    If Y=[y1,y2,..] are the values of the function at the location of the node [x1,x2,...].
    jnp.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)


    Args:
        x_knots: List of nodes for the spline interpolation as jnp.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: Vector of x values. ie, the sampling of the data.
        spline_degree: Degree of the spline interpolation (default: 3).
            if jnp.size(x_knots) <= spline_degree, then spline_degree = jnp.size(x_knots)-1

    Returns:
        M: Matrix of size (D,N) with D the size of x_samples and N the total number of nodes.
    """
    if type(x_knots[0]) is list or type(x_knots[0]) is jnp.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]

    if jnp.size(x_knots_list) <= 1:
        return jnp.ones((jnp.size(x_samples),1))
    if jnp.size(x_knots_list) <= spline_degree:
        spline_degree = jnp.size(x_knots)-1

    M_list = []
    for nodes in x_knots_list:
        M = jnp.zeros((jnp.size(x_samples), jnp.size(nodes)))
        min,max = jnp.min(nodes),jnp.max(nodes)
        inbounds = jnp.where((min<x_samples)&(x_samples<max))
        _x = x_samples[inbounds]

        for chunk in range(jnp.size(nodes)):
            tmp_y_vec = jnp.zeros(jnp.size(nodes))
            tmp_y_vec[chunk] = 1
            spl = InterpolatedUnivariateSpline(nodes, tmp_y_vec, k=spline_degree, ext=0)
            M[inbounds[0], chunk] = spl(_x)
        M_list.append(M)
    return jnp.concatenate(M_list, axis=1)


def longitude2phase(longitude: float):
    return longitude / (2 * jnp.pi)


def mean_uncertainty(uncertainties: jax.typing.ArrayLike) -> float:
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return jnp.sqrt(jnp.sum(uncertainties ** 2)) / jnp.size(uncertainties)


def median_uncertainties(uncertainties: jax.typing.ArrayLike) -> float:
    """Calculate the uncertainty of the median of an array.

    Demonstration:
        uncertainty ~ standard deviation = sqrt(variance) = sqrt(V)
        V_mean / V_median = 2 * (N - 1) / (pi * N); (see source)
        => V_median = V_mean * pi * N / (2 * (N - 1))
        => uncertainty_median = uncertainty_mean * sqrt(pi * N / (2 * (N - 1)))

    Source:
        https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        uncertainties: individual uncertainties of the median of the array

    Returns:
        The uncertainty of the median of the array
    """
    return mean_uncertainty(uncertainties) \
        * jnp.sqrt(jnp.pi * jnp.size(uncertainties) / (2 * (jnp.size(uncertainties) - 1)))

def median_filter_spectrum(spectrum, size=10, mode = 'nearest'):
    median_filtered_spectrum = median_filter(spectrum, size=size, mode = mode)
    return spectrum - median_filtered_spectrum

def normalize(array: jax.typing.ArrayLike, axis: int = None) -> jax.typing.ArrayLike:
    return (array - jnp.min(array, axis=axis)) / (jnp.max(array, axis=axis) - jnp.min(array, axis=axis))


def phase2longitude(phase: float, rad2deg: bool = False):
    longitude = phase * 2 * jnp.pi

    if rad2deg:
        longitude = jnp.rad2deg(longitude)

    return longitude


def prt_resolving_space(start, stop, resolving_power) -> jax.typing.ArrayLike:
    """Return numbers evenly spaced at the specified resolving power.

    Args:
        start:
            The starting value of the sequence.
        stop:
            The end value of the sequence.
        resolving_power:
            Resolving power of the sample

    Returns:
        Samples spaced following the specified resolving power.

    """
    # Check for inputs validity
    if start > stop:
        raise ValueError(f"start ({start}) must be lower than stop {stop}")

    if resolving_power <= 0:
        raise ValueError(f"resolving power ({resolving_power}) must be strictly positive")

    inverse_resolving_power = 1 / resolving_power

    # Get maximum space length (much higher than required)
    size_max = int(jnp.ceil((stop - start) / (start / resolving_power)))

    # Start generating space
    samples = [start]
    i = 0

    for i in range(size_max):
        samples.append(samples[-1] * jnp.exp(inverse_resolving_power))

        if samples[-1] >= stop:
            break

    if i == size_max - 1 and samples[-1] < stop:
        raise ValueError(f"maximum size ({size_max}) reached before reaching stop ({samples[-1]} < {stop})")

    return jnp.array(samples)


def resolving_space(start, stop, resolving_power):
    # Check for inputs validity
    if start > stop:
        raise ValueError(f"start ({start}) must be lower than stop {stop}")

    if resolving_power <= 0:
        raise ValueError(f"resolving power ({resolving_power}) must be strictly positive")

    # Get maximum space length (much higher than required)
    size_max = int(jnp.ceil((stop - start) / (start / resolving_power)))

    if not jnp.isfinite(size_max) or size_max < 0:
        raise ValueError(f"invalid maximum size ({size_max})")

    # Start generating space
    samples = [start]
    i = 0

    for i in range(size_max):
        if samples[-1] >= stop:
            break

        samples.append(samples[-1] + samples[-1] / resolving_power)

    if i == size_max - 1 and samples[-1] < stop:
        raise ValueError(f"maximum size ({size_max}) reached before reaching stop ({samples[-1]} < {stop})")
    elif samples[-1] > stop:
        del samples[-1]  # ensure that the space is within the [start, stop] interval

    return jnp.array(samples)


def running_mean(x: jax.typing.ArrayLike, n: int) -> jax.typing.ArrayLike:
    cum_sum = jnp.cumsum(jnp.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)


def savgol_filter_spectrum(spectrum, window_length: int = 10, polyorder: int = 3, mode: str = 'nearest'):
    savgol_filtered_spectrum = savgol_filter(
        spectrum,
        window_length=window_length,
        polyorder=polyorder,
        mode=mode
    )
    return spectrum - savgol_filtered_spectrum


def sigma2bayes_factor(sigma: float) -> float:
    """
    Convert a sigma significance into a Bayes factor, or "evidence".
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    :param sigma: sigma significance
    :return: Bayes factor (aka "evidence")
    """
    rho = 1 - erf(sigma / jnp.sqrt(2))
    return - 1 / (jnp.exp(1) * rho * jnp.log(rho))


