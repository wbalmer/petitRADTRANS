import jax.numpy as jnp
from jax import jit, lax, vmap, debug
from jax.typing import ArrayLike
import numpy as np
from interpax import interp1d

from petitRADTRANS.math import gaussian_weights_running, median_filter, savgol_filter

"""
JAX Implementation of fortran_convolve and fortran_rebin
"""


def compute_bins_edges(middle_bin_points: ArrayLike) -> ArrayLike:
    """Calculate bin edges for middle bin points.

    Args:
        middle_bin_points: array of size N containing the middle bin points to calculate the bin edges of

    Returns:
        array of size N+1 containing the bins edges
    """
    bin_edges = [
        middle_bin_points[0] - (middle_bin_points[1] - middle_bin_points[0]) / 2
    ]

    for i in range(int(len(middle_bin_points)) - 1):
        bin_edges.append(
            middle_bin_points[i] + (middle_bin_points[i + 1] - middle_bin_points[i]) / 2
        )

    bin_edges.append(
        middle_bin_points[len(middle_bin_points) - 1]
        + (
            middle_bin_points[len(middle_bin_points) - 1]
            - middle_bin_points[len(middle_bin_points) - 2]
        )
        / 2
    )

    return jnp.array(bin_edges)


def continuum_subtract_spectrum(
    wavelengths: ArrayLike,
    spectrum: ArrayLike,
    parameters: dict,
    method: str = "spline",
    name: str = None,
) -> ArrayLike:
    """
    # TODO Docstring
    # TODO ensure everything uses JAX arrays instead of numpy arrays.
    """

    if method == "spline":
        nodes = wavelengths[0]
        if name is not None and +(name + "_nodes" in parameters.keys()):
            nodes = getattr(
                parameters[name + "_nodes"], "value", parameters[name + "_nodes"]
            )
            x_nodes = jnp.linspace(wavelengths[0], wavelengths[-1], nodes)
        elif name is not None and +(name + "_node_array" in parameters.keys()):
            x_nodes = getattr(
                parameters[name + "_node_array"],
                "value",
                parameters[name + "_node_array"],
            )
        elif "continuum_subtract_nodes" in parameters.keys():
            nodes = getattr(
                parameters["continuum_subtract_nodes"],
                "value",
                parameters["continuum_subtract_nodes"],
            )
            x_nodes = jnp.linspace(wavelengths[0], wavelengths[-1], nodes)
        else:
            raise ValueError("No nodes available to create spline.")

        return filter_spectrum_with_spline(wavelengths, spectrum, x_nodes=x_nodes)
    elif method == "medfilt":
        median_filter_width = 10
        if name is not None and +(name + "_median_filter_width" in parameters.keys()):
            median_filter_width = getattr(
                parameters[name + "_median_filter_width"],
                "value",
                parameters[name + "_median_filter_width"],
            )
        elif "median_filter_width" in parameters.keys():
            median_filter_width = getattr(
                parameters["median_filter_width"],
                "value",
                parameters["median_filter_width"],
            )
        return median_filter_spectrum(jnp.array(spectrum), size=median_filter_width)
    elif method == "savgol":
        window_length = 10
        polyorder = 3
        if name is not None and +(name + "_savgol_window_length" in parameters.keys()):
            window_length = getattr(
                parameters[name + "_savgol_window_length"],
                "value",
                parameters[name + "_savgol_window_length"],
            )
        elif "savgol_window_length" in parameters.keys():
            window_length = getattr(
                parameters["savgol_window_length"],
                "value",
                parameters["savgol_window_length"],
            )

        if name is not None and +(name + "_savgol_polyorder" in parameters.keys()):
            polyorder = getattr(
                parameters[name + "_savgol_polyorder"],
                "value",
                parameters[name + "_savgol_polyorder"],
            )
        elif "savgol_polyorder" in parameters.keys():
            polyorder = getattr(
                parameters["savgol_polyorder"], "value", parameters["savgol_polyorder"]
            )
        return savgol_filter_spectrum(
            jnp.array(spectrum), window_length=window_length, polyorder=polyorder
        )


def convolve_constant_resolution(
    input_wavelength: ArrayLike, input_flux: ArrayLike, instrument_res: any
) -> ArrayLike:
    sigma_lsf = input_wavelength / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * instrument_res)
    # Calculate the spacing of the wavelength array
    spacing = jnp.abs(input_wavelength - jnp.roll(input_wavelength, 1))
    spacing = spacing.at[0].set(spacing[1])  # Fix the first element due to jnp.roll

    # Calculate sigma_lsf for gaussian_filter
    sigma_lsf_gauss_filter = sigma_lsf / spacing

    # Convolve with a Gaussian LSF
    flux_lsf = _convolve_running(input_flux, sigma_lsf_gauss_filter)
    return flux_lsf


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

    # Vectorized implementation using vmap to replace the explicit Python loop.
    # For each output channel (center wavelength and corresponding R) compute
    # the Gaussian LSF over the full input wavelength grid, normalize it and
    # take the dot-product with the input flux.
    def _convolve_at(center_wv, res):
        # sigma is FWHM / 2.355
        sigma = (center_wv / res) / 2.355
        gauss = jnp.exp(-((input_wavelength - center_wv) ** 2) / (2 * sigma**2))
        gauss = gauss / jnp.sum(gauss)
        return jnp.sum(gauss * input_flux)

    convolved_spectrum = vmap(_convolve_at)(input_wavelength, resolutions)

    return convolved_spectrum


def convolve(
    input_wavelength: ArrayLike, input_flux: ArrayLike, instrument_res: any
) -> ArrayLike:
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
    if isinstance(instrument_res, jnp.ndarray) or isinstance(
        instrument_res, np.ndarray
    ):
        # Variable width convolution
        flux_lsf = convolve_variable_resolution(
            input_wavelength,
            input_flux,
            instrument_res,  # This is expected to be sigma_lsf values per wavelength by variable_width_convolution
        )
    else:
        # Constant resolution convolution using Gaussian filter
        # Calculate sigma_lsf (standard deviation of the LSF)
        flux_lsf = convolve_constant_resolution(
            input_wavelength,
            input_flux,
            instrument_res,  # This is expected to be sigma_lsf values per wavelength by variable_width_convolution
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
    if (
        isinstance(sigma_lsf_gauss_filter, jnp.ndarray)
        and sigma_lsf_gauss_filter.ndim == 2
    ):
        weights = sigma_lsf_gauss_filter
    else:
        # Allow callers to pass max_radius for static sizing
        max_radius = kwargs.pop("max_radius", 20)
        weights = gaussian_weights_running(
            sigma_lsf_gauss_filter, max_radius=max_radius
        )

    input_length = weights.shape[1]
    central_index = int(input_length // 2)

    # Create a matrix of rolled versions of the input spectrum. The comprehension
    # iterates over a static range (input_length) so it is JIT-friendly when
    # input_length is fixed at compile time.
    rolled = jnp.stack(
        [
            jnp.roll(input_spectrum, shift=i - central_index, axis=-1)
            for i in range(input_length)
        ],
        axis=0,
    )
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
    convolved_spectrum = lax.cond(predicate, _true_fun, _false_fun, operand)

    return convolved_spectrum


def convolve_and_sample_variable_resolution_breads(
    wavelengths,
    resolutions,
    model_wavelengths,
    model_fluxes,
    num_sigma=3,
    filter_size=None,
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
        filter_size: int or None
            Number of points in the LSF filter. Must be provided as a static Python int
            when this function is JIT-compiled (declare it via static_argnums/static_argnames).
            If None, computed automatically from the data; this path is not JIT-compatible.

    Returns:
        output_model: np.ndarray
            the fluxes in each of the wavelength channels (length of N_output)
    """

    sigmas_wvs = wavelengths / (
        resolutions * 2 * jnp.sqrt(2 * jnp.log(2))
    )  # corrected a math error here, July 15 2024

    in_range_mask = jnp.logical_and(
        model_wavelengths >= jnp.min(wavelengths),
        model_wavelengths < jnp.max(wavelengths),
    )
    dwv_model = jnp.abs(model_wavelengths - jnp.roll(model_wavelengths, 1))
    dwv_model = dwv_model.at[0].set(dwv_model[1])
    dwv_model_min = jnp.min(jnp.where(in_range_mask, dwv_model, jnp.inf))

    if filter_size is None:
        filter_size = int(jnp.ceil(jnp.max(2 * num_sigma * sigmas_wvs) / dwv_model_min))
    filter_coords = jnp.linspace(-num_sigma, num_sigma, filter_size)

    filter_coords = jnp.tile(
        filter_coords, [wavelengths.shape[0], 1]
    )  # shape of (N_output, filter_size)

    filter_wv_coords = (
        filter_coords * sigmas_wvs[:, None] + wavelengths[:, None]
    )  # model wavelengths we want
    lsf = jnp.exp(-(filter_coords**2) / 2) / jnp.sqrt(2 * jnp.pi)

    # interpax expects 1D query points for 1D interpolation values.
    filter_model = interp1d(
        filter_wv_coords.reshape(-1),
        model_wavelengths,
        model_fluxes,
        method="cubic",
    ).reshape(filter_wv_coords.shape)

    output_model = jnp.nansum(filter_model * lsf, axis=1) / jnp.sum(lsf, axis=1)

    return output_model


def filter_spectrum_with_spline(
    wavelengths, fluxes, uncertainties=None, x_nodes=None, m_spline=None
):
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
        uncertainties = jnp.ones(fluxes.shape)

    if m_spline is None:
        m_spline = linear_spline_interpolation(x_nodes, wavelengths, spline_degree=3)

    normalised_spline_matrix = m_spline / uncertainties[:, None]
    weighted_fluxes = fluxes / uncertainties

    # Use jnp.where (not multiplication) to zero out non-finite rows so that
    # IEEE-754 arithmetic ``0.0 * inf = NaN`` cannot poison the lstsq.
    finite_mask = jnp.isfinite(weighted_fluxes)
    normalised_spline_matrix_m = jnp.where(
        finite_mask[:, None], normalised_spline_matrix, 0.0
    )
    weighted_fluxes_m = jnp.where(finite_mask, weighted_fluxes, 0.0)

    paras, _, _, _ = jnp.linalg.lstsq(normalised_spline_matrix_m, weighted_fluxes_m)
    # Evaluate the spline model over all wavelengths (not just finite ones).
    spline_model = normalised_spline_matrix_m @ paras
    residuals = weighted_fluxes - spline_model

    residuals_finite = jnp.where(finite_mask, residuals, 0.0)
    high_pass_filtered_fluxes = jnp.where(
        finite_mask, residuals_finite * uncertainties, jnp.nan
    )

    return high_pass_filtered_fluxes  # ,LPF_spec


def linear_spline_interpolation(x_knots, x_samples, spline_degree=3):
    """
    From BREADS, BSD 3-Clause License

    Copyright (c) 2024, jruffio

    Compute a spline based linear model.
    If Y=[y1,y2,..] are the values of the function at the location of the node [x1,x2,...].
    jnp.dot(M,Y) is the interpolated spline corresponding to the sampling of the x-axis (x_samples)

    Uses interpax (JAX-native) for spline basis evaluation instead of SciPy's InterpolatedUnivariateSpline.
    This enables JIT compilation and GPU acceleration. Note: interpax uses natural cubic splines while
    the original SciPy implementation used B-splines, resulting in different basis functions. However,
    the practical impact on spectrum fitting is negligible (~0.01 mean difference in fitted spectra).

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

    # Adjust spline degree based on number of nodes in first knot set
    first_knot_set = jnp.asarray(x_knots_list[0])
    if jnp.size(first_knot_set) <= spline_degree:
        spline_degree = jnp.size(first_knot_set) - 1

    def _process_one_knot_set(nodes):
        """Process a single knot set. Vectorizes basis evaluation using vmap."""
        nodes = jnp.asarray(nodes)
        x_samples_arr = jnp.asarray(x_samples)
        n_nodes = nodes.size

        # Create mask for in-bounds samples
        min_val, max_val = jnp.min(nodes), jnp.max(nodes)
        mask = (min_val < x_samples_arr) & (x_samples_arr < max_val)

        def _eval_basis(chunk_idx):
            """Evaluate the basis function for index chunk_idx using interpax cubic spline."""
            tmp_y_vec = jnp.zeros(n_nodes).at[chunk_idx].set(1.0)
            evals = interp1d(
                x_samples_arr, nodes, tmp_y_vec, method="cubic", extrap=0.0
            )
            return jnp.where(mask, evals, 0.0)

        # Vmap over chunk indices to evaluate all basis functions
        chunk_indices = jnp.arange(n_nodes)
        basis_evals = vmap(_eval_basis)(chunk_indices)  # shape: (n_nodes, n_samples)

        return basis_evals.T  # (n_samples, n_nodes)

    M_list = [_process_one_knot_set(nodes) for nodes in x_knots_list]
    return jnp.concatenate(M_list, axis=1)


def median_filter_spectrum(spectrum, size=10, mode="nearest"):
    """
    High-pass filter spectrum using median filtering.
    Returns the residuals after subtracting the median-filtered (low-pass) version.

    Args:
        spectrum: Input spectrum array
        size: Median filter window size
        mode: Padding mode for edges ('nearest', 'reflect', 'constant', etc.)

    Returns:
        High-pass filtered spectrum (original - median_filtered)
    """
    median_filtered_spectrum = median_filter(spectrum, size=size, mode=mode)
    return spectrum - median_filtered_spectrum


def rebinning_interpolation(
    input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin
):
    del n_rebin  # inferred from the output bin arrays

    rebin_dtype = jnp.result_type(
        input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high
    )
    input_wavelengths = jnp.asarray(input_wavelengths, dtype=rebin_dtype)
    input_spectrum = jnp.asarray(input_spectrum, dtype=rebin_dtype)
    rebin_bin_low = jnp.asarray(rebin_bin_low, dtype=rebin_dtype)
    rebin_bin_high = jnp.asarray(rebin_bin_high, dtype=rebin_dtype)

    segment_low = input_wavelengths[:-1]
    segment_high = input_wavelengths[1:]
    segment_width = segment_high - segment_low
    segment_slope = (input_spectrum[1:] - input_spectrum[:-1]) / segment_width
    segment_intercept = input_spectrum[:-1] - segment_slope * segment_low
    bin_widths = rebin_bin_high - rebin_bin_low

    def accumulate_bin_integrals(integrals, segment_data):
        low, high, slope, intercept = segment_data
        overlap_low = jnp.maximum(low, rebin_bin_low)
        overlap_high = jnp.minimum(high, rebin_bin_high)
        overlap_width = jnp.maximum(overlap_high - overlap_low, 0.0)

        segment_integral = intercept * overlap_width + 0.5 * slope * (
            overlap_high**2.0 - overlap_low**2.0
        )
        segment_integral = jnp.where(overlap_width > 0.0, segment_integral, 0.0)

        return integrals + segment_integral, None

    initial_integrals = jnp.zeros_like(rebin_bin_low)
    scan_inputs = (segment_low, segment_high, segment_slope, segment_intercept)
    integrated_flux, _ = lax.scan(
        accumulate_bin_integrals, initial_integrals, scan_inputs
    )

    return integrated_flux / bin_widths


def _raise_small_size(x):
    if x < 2:
        raise ValueError(
            f"Error: output wavelength array size must be >= 2, but is of size {x}"
        )


def _raise_incomplete_wavelength_range(args):
    input_array, output_array = args
    half_bin_width_low = (output_array[1] - output_array[0]) * 0.5
    half_bin_width_high = (output_array[-1] - output_array[-2]) * 0.5

    if (input_array[0] >= (output_array[0] - half_bin_width_low)) or (
        input_array[-1] <= (output_array[-1] + half_bin_width_high)
    ):
        raise ValueError(
            f"Error: input wavelength array needs to extend at least half a bin width further "
            f"than output wavelength array on both sides:\n"
            f"input interval: {input_array[0]} -- {input_array[-1]}\n"
            f"required interval: {output_array[0]-half_bin_width_low} "
            f"-- {output_array[-1]+half_bin_width_high}"
        )


@jit
def rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths):
    rebin_dtype = jnp.result_type(
        input_wavelengths, input_spectrum, rebinned_wavelengths
    )
    input_wavelengths = jnp.asarray(input_wavelengths, dtype=rebin_dtype)
    input_spectrum = jnp.asarray(input_spectrum, dtype=rebin_dtype)
    rebinned_wavelengths = jnp.asarray(rebinned_wavelengths, dtype=rebin_dtype)

    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin, dtype=rebin_dtype)

    debug.callback(_raise_small_size, n_rebin)

    rebin_bin_low = jnp.zeros(n_rebin, dtype=rebin_dtype)
    rebin_bin_high = jnp.zeros(n_rebin, dtype=rebin_dtype)

    rebin_bin_low = rebin_bin_low.at[0].set(
        rebinned_wavelengths[0]
        - (rebinned_wavelengths[1] - rebinned_wavelengths[0]) * 0.5
    )
    rebin_bin_low = rebin_bin_low.at[1:].set(
        rebinned_wavelengths[1:]
        - (rebinned_wavelengths[1:] - rebinned_wavelengths[:-1]) * 0.5
    )

    rebin_bin_high = rebin_bin_high.at[:-1].set(rebin_bin_low[1:])
    rebin_bin_high = rebin_bin_high.at[-1].set(
        rebinned_wavelengths[-1]
        + (rebinned_wavelengths[-1] - rebinned_wavelengths[-2]) * 0.5
    )

    debug.callback(
        _raise_incomplete_wavelength_range, (input_wavelengths, rebinned_wavelengths)
    )

    rebinned_spectrum = rebinning_interpolation(
        input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin
    )
    return rebinned_spectrum


def rebin_spectrum_bin(
    input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths
):
    rebin_dtype = jnp.result_type(
        input_wavelengths, input_spectrum, rebinned_wavelengths, bin_widths
    )
    input_wavelengths = jnp.asarray(input_wavelengths, dtype=rebin_dtype)
    input_spectrum = jnp.asarray(input_spectrum, dtype=rebin_dtype)
    rebinned_wavelengths = jnp.asarray(rebinned_wavelengths, dtype=rebin_dtype)
    bin_widths = jnp.asarray(bin_widths, dtype=rebin_dtype)

    n_rebin = rebinned_wavelengths.size
    rebinned_spectrum = jnp.zeros(n_rebin, dtype=rebin_dtype)

    if n_rebin < 2:
        raise ValueError(
            f"Error: output wavelength array size must be >= 2, but is of size {n_rebin}"
        )

    rebin_bin_low = rebinned_wavelengths - (bin_widths * 0.5)
    rebin_bin_high = rebinned_wavelengths + (bin_widths * 0.5)

    # if (input_wavelengths[0] >= rebin_bin_low[0]) or (input_wavelengths[-1] <= rebin_bin_high[-1]):
    #    raise ValueError(f"Error: input wavelength array needs to extend at least half a bin width further than output wavelength array on both sides:\n"
    #                     f"input interval: {input_wavelengths[0]} -- {input_wavelengths[-1]}\n"
    #                     f"required interval: {rebin_bin_low[0]} -- {rebin_bin_high[-1]}")

    rebinned_spectrum = rebinning_interpolation(
        input_wavelengths, input_spectrum, rebin_bin_low, rebin_bin_high, n_rebin
    )
    return rebinned_spectrum


def savgol_filter_spectrum(
    spectrum, window_length: int = 11, polyorder: int = 3, mode: str = "nearest"
):
    """
    High-pass filter spectrum using Savitzky-Golay filtering.
    Returns the residuals after subtracting the smoothed (low-pass) version.

    Args:
        spectrum: Input spectrum array
        window_length: Window size for polynomial fitting (will be made odd)
        polyorder: Order of polynomial to fit (must be < window_length)
        mode: Padding mode for edges ('nearest', 'reflect', 'constant', etc.)

    Returns:
        High-pass filtered spectrum (original - smoothed)
    """
    savgol_filtered_spectrum = savgol_filter(
        spectrum, window_length=window_length, polyorder=polyorder, mode=mode
    )
    return spectrum - savgol_filtered_spectrum
