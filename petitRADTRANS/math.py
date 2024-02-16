"""Stores useful mathematical functions.
"""
import numpy as np
from scipy.special import erf, erfinv, lambertw


def bayes_factor2sigma(bayes_factor: float) -> float:
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
        bayes_factor = np.array([bayes_factor])
    elif not isinstance(bayes_factor, np.ndarray):
        bayes_factor = np.array(bayes_factor)

    benneke_part = np.nonzero(np.less_equal(bayes_factor, molliere_threshold))
    molliere_part = np.nonzero(np.greater(bayes_factor, molliere_threshold))

    sigma = np.zeros(np.shape(bayes_factor))

    if np.size(benneke_part) > 0:
        a = -1 / (np.exp(1) * bayes_factor[benneke_part])
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        sigma[benneke_part] = np.sqrt(2) * erfinv(1 - rho)

    if np.size(molliere_part) > 0:
        a = -1 / (np.exp(1) * molliere_threshold)
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        offset = np.sqrt(2) * erfinv(1 - rho) - np.sqrt(2 * np.log(molliere_threshold))

        sigma[molliere_part] = np.sqrt(2 * np.log(bayes_factor[molliere_part])) + offset

    if is_scalar:
        return sigma[0]
    else:
        return sigma


def box_car_conv(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    res = np.zeros_like(array)
    len_arr = len(array)

    for i in range(len(array)):
        if (i - points / 2 >= 0) and (i + points / 2 <= len_arr + 1):
            smooth_val = array[i - points / 2:i + points / 2]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i + points / 2 > len_arr + 1:
            len_use = len_arr + 1 - i
            smooth_val = array[i - len_use:i + len_use]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i - points / 2 < 0:
            smooth_val = array[:max(2 * i, 1)]
            res[i] = np.sum(smooth_val) / len(smooth_val)

    return res


def calculate_chi2(data: [float, np.ndarray], model: [float, np.ndarray], uncertainties: [float, np.ndarray]) \
        -> [float, np.ndarray]:
    return np.sum(((data - model) / uncertainties) ** 2)


def calculate_reduced_chi2(data: [float, np.ndarray], model: [float, np.ndarray], uncertainties: [float, np.ndarray],
                           degrees_of_freedom: int = 0) -> [float, np.ndarray]:
    return calculate_chi2(data, model, uncertainties) / (np.size(data) - degrees_of_freedom)


def calculate_uncertainty(derivatives: np.ndarray, uncertainties: np.ndarray, covariance_matrix: np.ndarray = None)\
        -> np.ndarray:
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
            Partial derivatives of the function with respect to each variables (df/dx, df/dy, ...)
        uncertainties:
            Uncertainties of each variable (either a 1D-array or a 2D-array containing - and + unc.)
        covariance_matrix:
            Covariance matrix between the variables, by default set to the identity matrix

    Returns:
        A size-2 array containing the - and + uncertainties of the function
    """
    if covariance_matrix is None:
        covariance_matrix = np.identity(np.size(derivatives))

    if np.ndim(uncertainties) == 1:
        sigmas = derivatives * uncertainties

        return np.sqrt(np.matmul(sigmas, np.matmul(covariance_matrix, np.transpose(sigmas))))
    elif np.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return np.sqrt(np.array([  # beware, this is not strictly correct
            np.matmul(sigma_less, np.matmul(covariance_matrix, np.transpose(sigma_less))),
            np.matmul(sigma_more, np.matmul(covariance_matrix, np.transpose(sigma_more)))
        ]))


def feature_scaling(array: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    """Bring all values of array between a min and max value.

    Args:
        array: array to normalize
        min_value: target minimum value
        max_value: target maximum value

    Returns:
        The normalized array with values between min_value and max_value.
    """
    return min_value + ((array - np.min(array)) * (max_value - min_value)) / (np.max(array) - np.min(array))


def gaussian_weights1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
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

    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma ** 2 * x ** 2)

    return phi_x / phi_x.sum()


def gaussian_weights_running(sigmas: np.ndarray, truncate: float = 4.0) -> np.ndarray:
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
    radius = int(truncate * np.max(sigmas) + 0.5)

    x = np.arange(-radius, radius + 1)
    sd = np.tile(sigmas, (x.size, 1)).T

    phi_x = np.exp(-0.5 / sd ** 2 * x ** 2)

    return np.transpose(phi_x.T / phi_x.sum(axis=1))


def longitude2phase(longitude: float):
    return longitude / (2 * np.pi)


def mean_uncertainty(uncertainties: np.ndarray) -> float:
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return np.sqrt(np.sum(uncertainties ** 2)) / np.size(uncertainties)


def median_uncertainties(uncertainties: np.ndarray) -> float:
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
        * np.sqrt(np.pi * np.size(uncertainties) / (2 * (np.size(uncertainties) - 1)))


def normalize(array: np.ndarray, axis: int = None) -> np.ndarray:
    return (array - np.min(array, axis=axis)) / (np.max(array, axis=axis) - np.min(array, axis=axis))


def phase2longitude(phase: float, rad2deg: bool = False):
    longitude = phase * 2 * np.pi

    if rad2deg:
        longitude = np.rad2deg(longitude)

    return longitude


def resolving_space(start, stop, resolving_power):
    # Check for inputs validity
    if start > stop:
        raise ValueError(f"start ({start}) must be lower than stop {stop}")

    if resolving_power <= 0:
        raise ValueError(f"resolving power ({resolving_power}) must be strictly positive")

    # Get maximum space length
    size_max = int(np.ceil((stop - start) / (start / resolving_power)))

    if not np.isfinite(size_max) or size_max < 0:
        raise ValueError(f"invalid maximum size ({size_max})")

    # Start generating space
    space = [start]
    i = 0

    for i in range(size_max):
        if space[-1] >= stop:
            break

        space.append(space[-1] + space[-1] / resolving_power)

    if i == size_max - 1 and space[-1] < stop:
        raise ValueError(f"maximum size ({size_max}) reached before reaching stop ({space[-1]} < {stop})")
    elif space[-1] > stop:
        del space[-1]  # ensure that the space is within the [start, stop] interval

    return np.array(space)


def running_mean(x: np.ndarray, n: int) -> np.ndarray:
    cum_sum = np.cumsum(np.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)


def sigma2bayes_factor(sigma: float) -> float:
    """
    Convert a sigma significance into a Bayes factor, or "evidence".
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    :param sigma: sigma significance
    :return: Bayes factor (aka "evidence")
    """
    rho = 1 - erf(sigma / np.sqrt(2))
    return - 1 / (np.exp(1) * rho * np.log(rho))
