import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, lax
from jax.typing import ArrayLike
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.math import gaussian_weights_running

from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.optimize import lsq_linear
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

"""
JAX Implementation of fortran_convolve and fortran_rebin
"""


def continuum_subtract_spectrum(
        wavelengths: ArrayLike,
        spectrum: ArrayLike,
        parameters: dict,
        method: str = 'spline',
        name: str = None) -> ArrayLike:
    """
    """

    if method == 'spline':
        nodes = wavelengths[0]
        if name is not None and + (name + "_nodes" in parameters.keys()):
            nodes = parameters[name + "_nodes"].value
        elif "continuum_subtract_nodes" in parameters.keys():
            nodes = parameters["continuum_subtract_nodes"].value
        else:
            raise ValueError("No nodes available to create spline.")
        x_nodes = jnp.linspace(wavelengths[0], wavelengths[-1], nodes)
        return filter_spectrum_with_spline(wavelengths, spectrum, x_nodes=x_nodes)
    elif method == 'medfilt':
        median_filter_width = 10
        if name is not None and + (name + "_median_filter_width" in parameters.keys()):
            median_filter_width = parameters[name + "_median_filter_width"].value
        elif "median_filter_width" in parameters.keys():
            median_filter_width = parameters["median_filter_width"].value
        return median_filter_spectrum(np.ndarray(spectrum), size=median_filter_width)
    elif method == 'savgol':
        window_length = 10
        polyorder = 3
        if name is not None and + (name + "_savgol_window_length" in parameters.keys()):
            window_length = parameters[name + "_savgol_window_length"].value
        elif "savgol_window_length" in parameters.keys():
            window_length = parameters["savgol_window_length"].value

        if name is not None and + (name + "_savgol_polyorder" in parameters.keys()):
            polyorder = parameters[name + "_savgol_polyorder"].value
        elif "savgol_polyorder" in parameters.keys():
            polyorder = parameters["savgol_polyorder"].value
        return savgol_filter_spectrum(np.ndarray(spectrum), window_length=window_length, polyorder=polyorder)


#@jit
def convolve_constant_resolution(input_wavelength: ArrayLike, input_flux: ArrayLike, instrument_res: any) -> ArrayLike:
    sigma_lsf = input_wavelength / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * instrument_res)
    # Calculate the spacing of the wavelength array
    spacing = jnp.abs(input_wavelength - jnp.roll(input_wavelength, 1))
    spacing = spacing.at[0].set(spacing[1])  # Fix the first element due to jnp.roll

    # Calculate sigma_lsf for gaussian_filter
    sigma_lsf_gauss_filter = sigma_lsf / spacing

    # Convolve with a Gaussian LSF
    flux_lsf = _convolve_running(input_flux, sigma_lsf_gauss_filter)
    return flux_lsf


#@jit
def convolve_variable_resolution(input_wavelength, input_flux, resolutions):
    """
    Convolve a spectrum with a variable with gaussian kernel.
    Based on implementation from Ben Burningham for Brewster.
    https://github.com/fwang23nh/brewster_v2/blob/master/bbconv.f90

    Args:
        input_wavelength - units of micron
        input_flux - arbitrary units
        resolutions - spectral resolving power, R

    Returns:
        convolved_spectrum - same units as input_flux
    """

    input_spectrum_length = input_flux.shape[0]
    convolved_spectrum = jnp.zeros(input_spectrum_length)
    #if not isinstance(resolutions.dtype)
    # TODO probably need vmap or something to be more efficient
    for i in range(input_spectrum_length):
        #sigma is FWHM / 2.355
        sigma = (input_wavelength[i] / resolutions[i]) / 2.355
        gauss = jnp.exp(-(input_wavelength-input_wavelength[i])**2/(2*sigma**2))
        gauss = gauss/jnp.sum(gauss)
        convolved_spectrum = convolved_spectrum.at[i].set(jnp.sum(gauss*input_flux))

    return convolved_spectrum


def convolve(input_wavelength: ArrayLike, input_flux: ArrayLike, instrument_res: any) -> ArrayLike:
    """
    Convolve a model spectrum with a given instrument resolution.
    This method replicates the logic from petitRADTRANS.retrieval.data.Data.convolve().

    Args:
        input_wavelength : ArrayLike
            The wavelength array of the model spectrum.
        input_flux : ArrayLike
            The flux array of the model spectrum.
        instrument_res : float or ArrayLike
            The instrument resolution. Can be a single float (constant resolution)
            or an array (variable resolution per wavelength point).

    Returns:
        flux_lsf : ArrayLike
            The convolved flux array.
    """
    if isinstance(instrument_res, jnp.ndarray) or isinstance(instrument_res, np.ndarray):
        # Variable width convolution
        flux_lsf = convolve_variable_resolution(
            input_wavelength,
            input_flux,
            instrument_res  # This is expected to be sigma_lsf values per wavelength by variable_width_convolution
        )
    else:
        # Constant resolution convolution using Gaussian filter
        # Calculate sigma_lsf (standard deviation of the LSF)
        flux_lsf = convolve_constant_resolution(
            input_wavelength,
            input_flux,
            instrument_res  # This is expected to be sigma_lsf values per wavelength by variable_width_convolution
        )

    return flux_lsf


def _convolve_running(input_spectrum, sigma_lsf_gauss_filter, **kwargs):
    """Convolve a spectrum to a new resolving power.
    The spectrum is convolved using Gaussian filters with a standard deviation
        std_dev = R_in(lambda) / R_new(lambda) * input_wavelengths_bins.
    Both the input resolving power and output resolving power can vary with wavelength.
    The input resolving power is given by:
        lambda / Delta_lambda
    where lambda is the center of a wavelength bin and Delta_lambda is the difference between the edges of the bin.

    The weights of the convolution are stored in a (N, M) matrix, with N being the size of the input, and M the size
    of the convolution kernels.
    To speed-up calculations, a matrix A of shape (N, M) is built from the inputs such as:
        A[i, :] = s[i - M/2], s[i - M/2 + 1], ..., s[i - M/2 + M],
    with s the input spectrum.
    The definition of the convolution C of s by constant weights with wavelength is:
        C[i] = sum_{j=0}^{j=M-1} s[i - M/2 + j] * weights[j].
    Thus, the convolution of s by weights at index i is:
        C[i] = sum_{j=0}^{j=M-1} A[i, j] * weights[i, j].

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_spectrum: input spectrum
        convolve_resolving_power: resolving power of output spectrum
        input_resolving_power: if not None, skip its calculation using input_wavelengths

    Returns:
        convolved_spectrum: the convolved spectrum at the new resolving power
    """
    _ = kwargs  # kwargs is not used in this function, this is intended

    # Expect weights to have static second dimension for JIT; allow passing a
    # precomputed weights array or compute weights with a user-provided
    # ``max_radius`` via kwargs.
    if isinstance(sigma_lsf_gauss_filter, jnp.ndarray) and sigma_lsf_gauss_filter.ndim == 2:
        weights = sigma_lsf_gauss_filter
    else:
        # Allow callers to pass max_radius for static sizing
        max_radius = kwargs.pop('max_radius', 20)
        weights = gaussian_weights_running(sigma_lsf_gauss_filter, max_radius=max_radius)

    input_length = weights.shape[1]
    central_index = int(input_length // 2)

    # Create a matrix of rolled versions of the input spectrum. The comprehension
    # iterates over a static range (input_length) so it is JIT-friendly when
    # input_length is fixed at compile time.
    rolled = jnp.stack([jnp.roll(input_spectrum, shift=i - central_index, axis=-1) for i in range(input_length)], axis=0)
    # Move the rolled axis to the last position to get shape (..., input_length)
    input_matrix = jnp.moveaxis(rolled, 0, -1)

    convolved_spectrum = jnp.sum(input_matrix * weights, axis=-1)
    n_dims_non_wavelength = len(input_spectrum.shape[:-1])

    # Replace non-valid convolved values by non-convolved values (inaccurate but better than 'reflect' or 0 padding)
    # For edge bins where the full kernel would extend beyond the array bounds,
    # replace the convolved values with the original spectrum values. We build
    # an index mask for the left and right edges and write them back. This
    # implementation avoids changing array shapes and uses operations that are
    # compatible with JIT when ``input_length`` is static.
    # Use lax.cond to handle the case where central_index > 0 in a JIT-friendly way.
    predicate = jnp.asarray(central_index > 0)

    def _true_fun(operand):
        convolved, inp, cidx, _ndims = operand
        ar = jnp.arange(convolved.shape[-1])
        leading = tuple(range(convolved.ndim - 1))

        left_mask = jnp.expand_dims(ar, leading) < cidx
        convolved = jnp.where(left_mask, inp, convolved)

        right_mask = jnp.expand_dims(ar, leading) >= (convolved.shape[-1] - cidx)
        convolved = jnp.where(right_mask, inp, convolved)

        return convolved

    def _false_fun(operand):
        convolved, *_ = operand
        return convolved

    operand = (convolved_spectrum, input_spectrum, central_index, n_dims_non_wavelength)
    convolved_spectrum = jax.lax.cond(predicate, _true_fun, _false_fun, operand)

    return convolved_spectrum


def convolve_and_sample_variable_resolution_breads(
        wavelengths,
        resolutions,
        model_wavelengths,
        model_fluxes,
        num_sigma=3
        ):
    """
    Simulate the observations of a model.
    Convolves the model with a variable Gaussian LSF, sampled at each desired spectral channel.

    From Jerry Xuan circa 2024, https://github.com/jruffio/breads

    Args:
        wavelengths: jnp.ndarray
            the wavelengths desired (length of N_output)
        resolutions: np.ndarray
            the spectral resolving power at each wavelengths (length of N_output)
        model_wavelengths: np.ndarray
            the wavelengths of the model (length of N_model)
        model_fluxes: np.ndarray
            the fluxes of the model (length of N_model)
        num_sigma: float
            number of +/- sigmas to evaluate the LSF to.

    Returns:
        output_model: np.ndarray
            the fluxes in each of the wavelength channels (length of N_output)
    """

    # JX added to use function with input R, instead of LSF FWHM.
    # first get FWHM of LSF from lambda / R. Then convert FWHM to stddev of Gaussian
    sigmas_wvs = wavelengths / (resolutions * 2 * np.sqrt(2 * np.log(2)))  # corrected a math error here, July 15 2024

    model_in_range = np.where((model_wavelengths >= np.min(wavelengths)) & (model_wavelengths < np.max(wavelengths)))
    dwv_model = np.abs(model_wavelengths[model_in_range] - np.roll(model_wavelengths[model_in_range], 1))
    dwv_model[0] = dwv_model[1]

    filter_size = int(np.ceil(np.max((2 * num_sigma * sigmas_wvs)/np.min(dwv_model))))
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)

    # Commenting out try-except block from breads, doesn't seem to do anything.
    # try:
    filter_coords = np.tile(filter_coords, [wavelengths.shape[0], 1])  # shape of (N_output, filter_size)
    # except Exception as e:
    # print(e)
    # print('reached exception for ' + str(filter_size))

    filter_wv_coords = filter_coords * sigmas_wvs[:, None] + wavelengths[:, None]  # model wavelengths we want
    lsf = np.exp(-filter_coords**2/2)/np.sqrt(2*np.pi)

    model_interp = interp1d(model_wavelengths, model_fluxes, kind='cubic', bounds_error=False)
    filter_model = model_interp(filter_wv_coords)

    output_model = np.nansum(filter_model * lsf, axis=1)/np.sum(lsf, axis=1)

    return output_model


def filter_spectrum_with_spline(wavelengths, fluxes, uncertainties=None, x_nodes=None, m_spline=None):
    """
    Use spline interpolation produce a high-pass filtered version of the input spectrum.

    From BREADS, BSD 3-Clause License
    Copyright (c) 2024, jruffio
    https://github.com/jruffio/breads

    Args:
        wavelengths: np.ndarray
            Wavelengths of the input spectrum
        fluxes: np.ndarray
            Fluxes of the input spectrum
        uncertainties: np.ndarray
            Uncertainties of the input spectrum
        x_nodes: np.ndarray
            Nodes for the spline interpolation as np.ndarray in the same units as wavelengths.
            x_nodes can also be a list of ndarrays/list to model discontinous functions.
        m_spline: np.ndarray
            Pre-computed spline matrix. If None, it will be computed from x_nodes and wavelengths.
    Returns:
        high_pass_filtered_fluxes: np.ndarray
            High-pass filtered fluxes of the input spectrum
    """
    if uncertainties is None:
        uncertainties = np.ones(fluxes.shape)

    if m_spline is None:
        m_spline = linear_spline_interpolation(x_nodes, wavelengths, spline_degree=3)

    normalised_spline_matrix = m_spline/uncertainties[:, None]
    weighted_fluxes = fluxes/uncertainties
    where_finite = np.where(np.isfinite(weighted_fluxes))
    normalised_spline_matrix = normalised_spline_matrix[where_finite[0], :]
    weighted_fluxes = weighted_fluxes[where_finite]

    paras = lsq_linear(normalised_spline_matrix, weighted_fluxes).x
    spline_model = np.dot(normalised_spline_matrix, paras)
    residuals = weighted_fluxes - spline_model

    # LPF_fluxes = np.zeros(fluxes.shape)+np.nan
    high_pass_filtered_fluxes = np.zeros(fluxes.shape)+np.nan
    # LPF_fluxes[where_finite] = m*uncertainties[where_finite]
    high_pass_filtered_fluxes[where_finite] = residuals*uncertainties[where_finite]

    return high_pass_filtered_fluxes  # ,LPF_spec


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


@jit
def median_filter_spectrum(spectrum, size=10, mode = 'nearest'):
    median_filtered_spectrum = median_filter(spectrum, size=size, mode = mode)
    return spectrum - median_filtered_spectrum


def rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin):
    rebinned_spectrum = jnp.zeros(n_rebin)
    i_input = 0

    def rebin_step(i, carry):
        i_input, rebinned_spectrum = carry

        def while_input_count_initialisation(carry):
            i_input = carry
            return i_input + 1

        def while_cond_initialisation(carry):
            i_input = carry
            return input_wavelengths[i_input] < rebin_bin_low[i]

        def while_cond(carry):
            i_input, _, _ = carry
            return input_wavelengths[i_input] < rebin_bin_high[i]

        def while_body(carry):
            i_input, rebinned_spectrum, del_nu_in = carry
            nu1 = jnp.maximum(input_wavelengths[i_input - 1], rebin_bin_low[i])
            nu2 = jnp.minimum(input_wavelengths[i_input], rebin_bin_high[i])
            ddel_nu = nu2 - nu1

            slope = (input_spectrum[i_input] - input_spectrum[i_input - 1]) / (input_wavelengths[i_input] - input_wavelengths[i_input - 1])
            y_ax = input_spectrum[i_input - 1]

            add_fl = (y_ax - slope * input_wavelengths[i_input - 1]) * (nu2 - nu1) + slope * (nu2 - nu1) * (nu2 + nu1) / 2.0
            rebinned_spectrum = rebinned_spectrum.at[i].add(add_fl)

            del_nu_out = del_nu_in + ddel_nu
            i_input = i_input + 1
            return i_input, rebinned_spectrum, del_nu_out

        del_nu = 0.0
        i_input = lax.while_loop(while_cond_initialisation, while_input_count_initialisation,(i_input))
        i_input, rebinned_spectrum, del_nu = lax.while_loop(while_cond, while_body, (i_input, rebinned_spectrum, del_nu))

        nu1 = jnp.maximum(input_wavelengths[i_input - 1], rebin_bin_low[i])
        nu2 = jnp.minimum(input_wavelengths[i_input], rebin_bin_high[i])
        ddel_nu = nu2 - nu1

        slope = (input_spectrum[i_input] - input_spectrum[i_input - 1]) / (input_wavelengths[i_input] - input_wavelengths[i_input - 1])
        y_ax = input_spectrum[i_input - 1]

        add_fl = (y_ax - slope * input_wavelengths[i_input - 1]) * (nu2 - nu1) + slope * (nu2 ** 2.0 - nu1 ** 2.0) / 2.0
        rebinned_spectrum = rebinned_spectrum.at[i].add(add_fl)

        del_nu = del_nu + ddel_nu
        rebinned_spectrum = rebinned_spectrum.at[i].set(rebinned_spectrum[i] / del_nu)

        return i_input, rebinned_spectrum

    _, rebinned_spectrum = lax.fori_loop(0, n_rebin, rebin_step, (i_input, rebinned_spectrum))

    return rebinned_spectrum


def _raise_small_size(x):
    if x < 2:
        raise ValueError(f"Error: output wavelength array size must be >= 2, but is of size {x}")


def _raise_incomplete_wavelength_range(input_array, output_array):
     raise ValueError(f"Error: input wavelength array needs to extend at least half a bin width further than output wavelength array on both sides:\n"
                      f"input interval: {input_array[0]} -- {input_array[-1]}\n"
                      f"required interval: {output_array[0]} -- {output_array[-1]}")


@jit
def rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths):
    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin)

    jax.debug.callback(_raise_small_size, n_rebin)

    rebin_bin_low = jnp.zeros(n_rebin)
    rebin_bin_high = jnp.zeros(n_rebin)

    rebin_bin_low = rebin_bin_low.at[0].set(rebinned_wavelengths[0] - (rebinned_wavelengths[1] - rebinned_wavelengths[0]) / 2.0)
    rebin_bin_low = rebin_bin_low.at[1:].set(rebinned_wavelengths[1:] - (rebinned_wavelengths[1:] - rebinned_wavelengths[:-1]) / 2.0)

    rebin_bin_high = rebin_bin_high.at[:-1].set(rebin_bin_low[1:])
    rebin_bin_high = rebin_bin_high.at[-1].set(rebinned_wavelengths[-1] + (rebinned_wavelengths[-1] - rebinned_wavelengths[-2]) / 2.0)

    jax.debug.callback(_raise_incomplete_wavelength_range, (input_wavelengths, rebin_bin_low))

    rebinned_spectrum = rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin)
    return rebinned_spectrum


@jit
def rebin_spectrum_bin(input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths):
    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin)

    if n_rebin < 2:
        raise ValueError(f"Error: output wavelength array size must be >= 2, but is of size {n_rebin}")

    rebin_bin_low = rebinned_wavelengths - (bin_widths / 2.0)
    rebin_bin_high = rebinned_wavelengths + (bin_widths / 2.0)

    #if (input_wavelengths[0] >= rebin_bin_low[0]) or (input_wavelengths[-1] <= rebin_bin_high[-1]):
    #    raise ValueError(f"Error: input wavelength array needs to extend at least half a bin width further than output wavelength array on both sides:\n"
    #                     f"input interval: {input_wavelengths[0]} -- {input_wavelengths[-1]}\n"
    #                     f"required interval: {rebin_bin_low[0]} -- {rebin_bin_high[-1]}")

    rebinned_spectrum = rebinning_interpolation(input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin)
    return rebinned_spectrum


@jit
def savgol_filter_spectrum(spectrum, window_length: int = 10, polyorder: int = 3, mode: str = 'nearest'):
    savgol_filtered_spectrum = savgol_filter(
        spectrum,
        window_length=window_length,
        polyorder=polyorder,
        mode=mode
    )
    return spectrum - savgol_filtered_spectrum
