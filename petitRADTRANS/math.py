"""Stores useful mathematical functions.
"""
from typing import Literal

import numpy as np
import jax
from jax import lax, jit
from jax.typing import ArrayLike
from jaxtyping import Array, Float
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv
from scipy.special import lambertw
from jax.scipy.signal import correlate

# noinspection PyUnresolvedReferences
from petitRADTRANS.fortran_convolve import fortran_convolve as fconvolve


# TODO: replace scipy and fortran functions with JAX


def bayes_factor2sigma(bayes_factor: float) -> [float, ArrayLike]:
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


def box_car_conv(array: ArrayLike, points: ArrayLike) -> ArrayLike:
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
    data: [float, ArrayLike],
    model: [float, ArrayLike],
    uncertainties: [float, ArrayLike]
) -> float:
    return float(jnp.sum(((data - model) / uncertainties) ** 2))


def calculate_reduced_chi2(
    data: [float, ArrayLike],
    model: [float, ArrayLike],
    uncertainties: [float, ArrayLike],
    degrees_of_freedom: int = 0
) -> [float, ArrayLike]:
    return calculate_chi2(data, model, uncertainties) / (jnp.size(data) - degrees_of_freedom)


def calculate_uncertainty(
    derivatives: ArrayLike,
    uncertainties: ArrayLike,
    covariance_matrix: ArrayLike = None
) -> [ArrayLike, None]:
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


def compute_resolving_power(array: ArrayLike) -> float:
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


def feature_scaling(array: ArrayLike, min_value: float = 0.0, max_value: float = 1.0) -> ArrayLike:
    """Bring all values of array between a min and max value.

    Args:
        array: array to normalize
        min_value: target minimum value
        max_value: target maximum value

    Returns:
        The normalized array with values between min_value and max_value.
    """
    return min_value + ((array - jnp.min(array)) * (max_value - min_value)) / (jnp.max(array) - jnp.min(array))


def gaussian_kernel_1d(sigma: float, radius: int) -> Float[Array, " 2*{radius}"]:
    """JAX version of the gaussian kernel to be used in the gaussian_filter1d function.

    Note that the implementation parallels that of scipy.ndimage at
    https://github.com/scipy/scipy/blob/v1.16.2/scipy/ndimage/_filters.py#L656."""
    x = jnp.arange(-radius, radius + 1)
    sigma2 = sigma * sigma
    phi_x = jnp.exp(-0.5 * x**2 / sigma2)
    return phi_x / jnp.sum(phi_x)


@jit(static_argnames=['max_radius'])
def gaussian_weights_running(sigmas: ArrayLike, truncate: float = 4.0, max_radius: int = 20) -> ArrayLike:
    """Compute 1D Gaussian convolution kernels for an array of standard deviations.

    This version supports JIT compilation by avoiding arrays whose shape depends on
    values computed at runtime. To guarantee static shapes when JIT-compiling,
    pass an integer ``max_radius`` (half-kernel size). If ``max_radius`` is not
    provided the radius will be computed in Python (non-JIT use).

    Args:
        sigmas:
            1-D array of standard deviations for each position (shape: (N,)).
        truncate:
            Truncate the filter at this many standard deviations (used only when
            ``max_radius`` is None).
        max_radius:
            Optional integer radius to force a static kernel size of
            ``2*max_radius+1``. Recommended for JIT compilation.

    Returns:
        weights: array of shape (N, 2*radius+1) containing the normalized
            Gaussian kernel for each sigma.
    """
    # Determine radius. If max_radius is provided it will be used (static
    # Python integer). If not provided, compute it from data (works outside JIT).
    radius = max_radius
    #= lax.cond(
    #    max_radius is None,
    #    lambda: jnp.ceil(truncate * jnp.max(sigmas) + 0.5).astype(int),
    #    lambda: max_radius
    #)

    # Build fixed-position offsets (shape: (M,)) and broadcast against sigmas
    x = jnp.arange(-radius, radius + 1)

    # Ensure sigmas is an array of shape (N,)
    sigmas = jnp.asarray(sigmas)
    sd = jnp.expand_dims(sigmas, axis=1)  # (N, 1)
    x_row = jnp.expand_dims(x, axis=0)    # (1, M)

    # Compute unnormalized Gaussian and normalize per-row
    phi_x = jnp.exp(-0.5 * (x_row ** 2) / (sd ** 2))
    weights = phi_x / jnp.sum(phi_x, axis=1, keepdims=True)

    return weights


def linear_spline_interpolation(
    x_knots: ArrayLike,
    x_samples: ArrayLike,
    spline_degree: Literal[1, 2, 3, 4, 5] = 3
):
    """Compute a spline based linear model.

    If Y = [y1, y2, ...] are the values of the function at the location of the node [x1, x2, ...].
    np.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)

    From BREADS, BSD 3-Clause License
    Copyright (c) 2024, jruffio
    https://github.com/jruffio/breads

    Args:
        x_knots: np.ndarray
            List of nodes for the spline interpolation as np.ndarray in the same units as x_samples.
            x_knots can also be a list of ndarrays/list to model discontinous functions.
        x_samples: np.ndarray
            Vector of x values. ie, the sampling of the data.
        spline_degree: int
            Degree of the spline interpolation (default: 3).
            if np.size(x_knots) <= spline_degree, then spline_degree = np.size(x_knots)-1

    Returns:
        M: Matrix of shape (D, N) with D the size of x_samples and N the total number of nodes.
    """
    if type(x_knots[0]) is list or type(x_knots[0]) is np.ndarray:
        x_knots_list = x_knots
    else:
        x_knots_list = [x_knots]

    if np.size(x_knots_list) <= 1:
        return np.ones((np.size(x_samples), 1))

    if np.size(x_knots_list) <= spline_degree:
        spline_degree: int = np.size(x_knots) - 1

    if spline_degree > 5:
        raise ValueError(
            f"spline degree cannot be greater than 5, but is ({spline_degree})"
        )

    spline_degree: Literal[1, 2, 3, 4, 5]
    spline_matrix: list[ArrayLike] = []

    for nodes in x_knots_list:
        _spline_matrix = np.zeros((np.size(x_samples), np.size(nodes)))

        in_bounds = np.where((np.min(nodes) < x_samples) & (x_samples < np.max(nodes)))
        _x = x_samples[in_bounds]

        for chunk in range(np.size(nodes)):
            # TODO @Evert add comment on what is going on with y
            y = np.zeros(np.size(nodes))
            y[chunk] = 1

            spline = InterpolatedUnivariateSpline(
                x=nodes,
                y=y,
                k=spline_degree,
                ext=0
            )
            _spline_matrix[in_bounds[0], chunk] = spline(_x)

        spline_matrix.append(_spline_matrix)

    return np.concatenate(spline_matrix, axis=1)


def longitude2phase(longitude: float):
    return longitude / (2 * jnp.pi)


def mean_uncertainty(uncertainties: ArrayLike) -> float:
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return jnp.sqrt(jnp.sum(uncertainties ** 2)) / jnp.size(uncertainties)


def median_uncertainties(uncertainties: ArrayLike) -> float:
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


def normalize(array: ArrayLike, axis: int = None) -> ArrayLike:
    return (array - jnp.min(array, axis=axis)) / (jnp.max(array, axis=axis) - jnp.min(array, axis=axis))


def phase2longitude(phase: float, rad2deg: bool = False):
    longitude = phase * 2 * jnp.pi

    if rad2deg:
        longitude = jnp.rad2deg(longitude)

    return longitude


def prt_resolving_space(start, stop, resolving_power) -> ArrayLike:
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
        raise ValueError(
            f"maximum size ({size_max}) reached before reaching stop ({samples[-1]} < {stop})"
            )
    elif samples[-1] > stop:
        del samples[-1]  # ensure that the space is within the [start, stop] interval

    return jnp.array(samples)


def running_mean(x: ArrayLike, n: int) -> ArrayLike:
    cum_sum = jnp.cumsum(jnp.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)


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

