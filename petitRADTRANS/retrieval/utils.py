"""This module contains a set of useful functions that don't really fit anywhere
else. This includes flux conversions, prior functions, mean molecular weight
calculations, transforms from mass to number fractions, and fits file output.
Helper functions for JAX-compatible log likelihood computation.

These functions are designed to be JIT-compilable and to improve the clarity
and maintainability of the log_likelihood_jax method.

"""

import json
import os
import warnings
import numpy as np
from scipy.special import erfcinv, gamma
from typing import Tuple, Union, Optional
from jax import lax, numpy as jnp
from petitRADTRANS.physics import shift_wavelengths_by_radial_velocity
from petitRADTRANS.radtrans_core.spectrum_utils import (
    continuum_subtract_spectrum,
    convolve,
    convolve_and_sample_variable_resolution_breads,
    rebin_spectrum_bin,
)
from petitRADTRANS.retrieval.log_likelihood import LogLikelihood

SQRT2 = jnp.sqrt(2.0)


# Prior Functions
# Stolen from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
def log_prior(cube, lx1, lx2):
    return 10 ** (lx1 + cube * (lx2 - lx1))


def uniform_prior(cube, x1, x2):
    return x1 + cube * (x2 - x1)


def gaussian_prior(cube, mu, sigma):
    return mu + sigma * SQRT2 * erfcinv(2.0 * (1.0 - cube))
    # return -(((cube-mu)/sigma)**2.)/2.


def log_gaussian_prior(cube, mu, sigma):
    bracket = sigma * sigma + sigma * SQRT2 * erfcinv(2.0 * cube)
    return mu * np.exp(bracket)


def delta_prior(cube, x1, x2):
    return x1


def inverse_gamma_prior(cube, a, b):
    return ((b**a) / gamma(a)) * (1 / cube) ** (a + 1) * np.exp(-b / cube)


# Sanity checks on parameter ranges
def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.0


def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.0


# File Formatting
def get_pymultinest_sample_dict(
    output_dir, name=None, add_log_likelihood=False, add_stats=False
):
    if name is None:
        name = output_dir.rsplit(os.sep, 1)[1]

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"output directory '{output_dir}' does not exist")

    sample_file = os.path.join(output_dir, "out_PMN", name + "_post_equal_weights.dat")
    parameter_names_file = os.path.join(output_dir, "out_PMN", name + "_params.json")

    if not os.path.isfile(sample_file):
        raise FileNotFoundError(f"sample file '{sample_file}' does not exist")

    if not os.path.isfile(parameter_names_file):
        raise FileNotFoundError(
            f"parameter names file '{parameter_names_file}' does not exist"
        )

    samples = np.genfromtxt(
        str(os.path.join(output_dir, "out_PMN", name + "_post_equal_weights.dat"))
    )

    with open(parameter_names_file, "r") as f:
        parameters_read = json.load(f)

    samples_dict = {}

    for i, key in enumerate(parameters_read):
        if np.ndim(samples) == 1:
            warnings.warn(
                f"samples in '{output_dir}' has only one live point\n"
                f"Ensure that the retrieval ran correctly and that it had enough live points "
                f"(>~ the number of free parameters)."
            )
            samples = samples[np.newaxis, :]

        samples_dict[key] = samples[:, i]

    if add_log_likelihood:
        samples_dict["log_likelihood"] = samples[:, -1]

    if add_stats:
        with open(os.path.join(output_dir, "out_PMN", name + "_stats.json"), "r") as f:
            parameters_read = json.load(f)

        samples_dict["stats"] = parameters_read

    return samples_dict


def get_calculate_flux_return_values(parameters):
    return_contribution = False
    return_opacities = False
    return_photosphere_radius = False
    return_rosseland_optical_depths = False
    return_radius_hydrostatic_equilibrium = False
    return_cloud_contribution = False
    return_abundances = False
    return_any = False
    if "contribution" in parameters.keys():
        return_contribution = parameters["contribution"].value
        return_any = True
    if "return_contribution" in parameters.keys():
        return_contribution = parameters["return_contribution"].value
        return_any = True
    if "return_opacities" in parameters.keys():
        return_opacities = parameters["return_opacities"].value
        return_any = True
    if "return_photosphere_radius" in parameters.keys():
        return_photosphere_radius = parameters["return_photosphere_radius"].value
        return_any = True
    if "return_abundances" in parameters.keys():
        return_abundances = parameters["return_abundances"].value
        return_any = True
    if "return_radius_hydrostatic_equilibrium" in parameters.keys():
        return_radius_hydrostatic_equilibrium = parameters[
            "return_radius_hydrostatic_equilibrium"
        ].value
        return_any = True
    if "return_rosseland_optical_depths" in parameters.keys():
        return_rosseland_optical_depths = parameters[
            "return_rosseland_optical_depths"
        ].value
        return_any = True
    if "return_cloud_contribution" in parameters.keys():
        return_cloud_contribution = parameters["return_cloud_contribution"].value
        return_any = True
    return (
        return_contribution,
        return_opacities,
        return_photosphere_radius,
        return_rosseland_optical_depths,
        return_radius_hydrostatic_equilibrium,
        return_cloud_contribution,
        return_abundances,
        return_any,
    )


def unpack_model_output(
    model_returned_values: Union[tuple, list],
    retrieve_uncertainties: bool,
    variability_atmospheric_column_model_flux_return_mode: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, Optional[jnp.ndarray]]:
    """
    Unified handler for parsing model_returned_values from the model generating function.

    The model function can return different numbers and types of values depending on
    configuration. This function normalizes the output to a standard tuple.

    Args:
        model_returned_values: The raw output from model_generating_function
        retrieve_uncertainties: Whether uncertainties are being retrieved
        variability_atmospheric_column_model_flux_return_mode: Whether variability column data is returned

    Returns:
        Tuple of (wavelengths_model, spectrum_model, beta, additional_log_l, atmospheric_model_column_fluxes)
        where atmospheric_model_column_fluxes is None if not needed
    """
    wavelengths_model = None
    spectrum_model = None
    beta = 1.0
    additional_log_l = 0.0
    atmospheric_model_column_fluxes = None

    if retrieve_uncertainties:
        if len(model_returned_values) == 4:
            wavelengths_model, spectrum_model, beta, additional_log_l = (
                model_returned_values
            )
        else:
            wavelengths_model, spectrum_model, beta = model_returned_values
            additional_log_l = 0.0
    else:
        if variability_atmospheric_column_model_flux_return_mode:
            (
                wavelengths_model,
                spectrum_model,
                additional_log_l,
                atmospheric_model_column_fluxes,
            ) = model_returned_values
        elif len(model_returned_values) == 3:
            wavelengths_model, spectrum_model, additional_log_l = model_returned_values
        else:
            wavelengths_model, spectrum_model = model_returned_values
            additional_log_l = 0.0

    # Ensure additional_log_l is never None
    if additional_log_l is None:
        additional_log_l = 0.0

    return (
        wavelengths_model,
        spectrum_model,
        beta,
        additional_log_l,
        atmospheric_model_column_fluxes,
    )


def extract_spectrum_parameters(parameters: dict, data_name: str) -> dict:
    """
    Extract and pre-compute spectrum-related parameters to avoid repeated dictionary lookups
    in loops.

    Args:
        parameters: Full parameters dictionary
        data_name: Name of the data object

    Returns:
        Dictionary with extracted parameters
    """
    param_dict = {}

    # Resolution parameters
    param_dict["resolution_slope"] = parameters.get(data_name + "_R_slope", None)
    param_dict["resolution_intersect"] = parameters.get(data_name + "_R_int", None)

    # Radial velocity parameters - check both data-specific and system-wide
    if data_name + "_radial_velocity" in parameters:
        param_dict["radial_velocity"] = (
            parameters[data_name + "_radial_velocity"].value * 1e5
        )
    elif "system_radial_velocity" in parameters:
        param_dict["radial_velocity"] = parameters["system_radial_velocity"].value * 1e5
    else:
        param_dict["radial_velocity"] = None

    # Continuum subtraction method
    param_dict["continuum_subtraction_method"] = parameters.get(
        "continuum_subtraction_method", None
    )

    return param_dict


def process_spectrum_pipeline(
    wavelengths_model: jnp.ndarray,
    spectrum_model: jnp.ndarray,
    data,
    parameters: dict,
    data_name: str,
    spectrum_params: Optional[dict] = None,
) -> jnp.ndarray:
    """
    Execute the spectrum processing pipeline: convolution, rebin, RV shift, continuum subtraction.

    This consolidates all the repetitive spectrum processing logic that appears multiple times
    in log_likelihood_jax.

    Args:
        wavelengths_model: Model wavelength array
        spectrum_model: Model spectrum array
        data: Data object containing wavelengths and processing configuration
        parameters: Full parameters dictionary
        data_name: Name of the data object
        spectrum_params: Pre-extracted spectrum parameters (from extract_spectrum_parameters)

    Returns:
        Processed spectrum (rebinned, convolved, continuum-subtracted as needed)
    """
    if spectrum_params is None:
        spectrum_params = extract_spectrum_parameters(parameters, data_name)

    # Handle variable resolution convolution and rebin
    if data.fit_instrumental_resolution:
        resolution_slope = spectrum_params["resolution_slope"]
        resolution_intersect = spectrum_params["resolution_intersect"]
        if resolution_slope is not None and resolution_intersect is not None:
            resolution_slope = resolution_slope.value
            resolution_intersect = resolution_intersect.value
            resolution_array = (
                data.wavelengths * resolution_slope
            ) + resolution_intersect
            model_spectrum_rebinned = convolve_and_sample_variable_resolution_breads(
                data.wavelengths, resolution_array, data.wavelengths, spectrum_model
            )
        else:
            model_spectrum_rebinned = spectrum_model
    else:
        model_spectrum_rebinned = None

    # Standard convolution to data resolution if needed
    model_spectrum_convolved = spectrum_model
    if data.data_resolution is not None:
        model_spectrum_convolved = convolve(
            wavelengths_model, spectrum_model, data.data_resolution
        )

    # Apply radial velocity shift
    wavelengths_model_shifted = wavelengths_model.copy()
    if spectrum_params["radial_velocity"] is not None:
        wavelengths_model_shifted = (
            wavelengths_model_shifted
            * shift_wavelengths_by_radial_velocity(spectrum_params["radial_velocity"])
        )

    # Rebin to data wavelengths if not already done
    if model_spectrum_rebinned is None:
        if data.wavelength_bin_widths is None:
            model_spectrum_rebinned = jnp.interp(
                data.wavelengths,
                wavelengths_model_shifted,
                model_spectrum_convolved,
            )
        else:
            model_spectrum_rebinned = rebin_spectrum_bin(
                wavelengths_model_shifted,
                model_spectrum_convolved,
                data.wavelengths,
                data.wavelength_bin_widths,
            )

    # Continuum subtraction if needed
    if data.subtract_continuum:
        method = "spline"
        if spectrum_params["continuum_subtraction_method"] is not None:
            method = spectrum_params["continuum_subtraction_method"].value
        model_spectrum_rebinned = continuum_subtract_spectrum(
            wavelengths=data.wavelengths,
            spectrum=model_spectrum_rebinned,
            parameters=parameters,
            method=method,
            name=data_name,
        )

    return model_spectrum_rebinned


def broadcast_spectrum_to_mask(
    spectrum_model: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply mask to spectrum, handling different dimensionality combinations.

    JAX-compatible masking that handles 1D, 2D, and 3D spectra without conditional branching.

    Args:
        spectrum_model: Model spectrum of any dimensionality
        mask: Mask array of same dimensionality

    Returns:
        Masked spectrum
    """
    return spectrum_model[~mask]


def compute_log_likelihood_for_spectrum(
    spectrum_model: jnp.ndarray,
    data_spectrum: jnp.ndarray,
    data_uncertainties: jnp.ndarray,
    data_covariance: Optional[jnp.ndarray],
    data_mask: Optional[jnp.ndarray],
    n_free_parameters: int,
    use_log_likelihood_object: bool = False,
    beta: float = 1.0,
    beta_mode: str = "multiply",
) -> float:
    """
    Compute log likelihood for a single spectrum, handling different approaches.

    This function handles both direct log_likelihood method calls (for object data)
    and LogLikelihood class instantiation (for simple data).

    Args:
        spectrum_model: Model spectrum
        data_spectrum: Observed spectrum
        data_uncertainties: Uncertainties on observations
        data_covariance: Covariance matrix (optional)
        data_mask: Mask for valid data points (optional)
        n_free_parameters: Number of free parameters in fit
        use_log_likelihood_object: If True, use LogLikelihood class; else use data.log_likelihood
        beta: Uncertainty scaling parameter
        beta_mode: Mode for applying beta ('multiply' or 'add')

    Returns:
        Log likelihood value
    """
    # Apply mask if provided
    if data_mask is not None:
        spectrum_model_masked = spectrum_model[~data_mask]
        data_spectrum_masked = data_spectrum[~data_mask]
        if data_uncertainties is not None:
            data_uncertainties_masked = data_uncertainties[~data_mask]
        else:
            data_uncertainties_masked = None
    else:
        spectrum_model_masked = spectrum_model
        data_spectrum_masked = data_spectrum
        data_uncertainties_masked = data_uncertainties

    # Check for NaNs (JAX-compatible)
    has_nans = jnp.isnan(spectrum_model_masked).any()

    # Use conditional to avoid NaN propagation
    def compute_likelihood():
        if use_log_likelihood_object:
            return LogLikelihood(
                model_spectrum=spectrum_model,  # Note: pass unmasked for LogLikelihood
                data_spectrum=data_spectrum,
                data_uncertainties=data_uncertainties,
                data_covariance=data_covariance,
                n_free_parameters=n_free_parameters,
            ).calculate_log_likelihood()
        else:
            # This assumes data has a log_likelihood method
            return data_spectrum.log_likelihood(
                spectrum_model_masked,
                data_spectrum_masked,
                data_uncertainties_masked,
                beta=beta,
                beta_mode=beta_mode,
            )

    # Return NaN compensation or computed value
    return lax.cond(has_nans, lambda: -1e99, compute_likelihood)


def handle_dimenstionality_dispatch(
    data_spectrum: jnp.ndarray,
    spectrum_model: jnp.ndarray,
    data_mask: jnp.ndarray,
    data_uncertainties: jnp.ndarray,
    use_data_log_likelihood_method: bool = False,
    beta: float = 1.0,
    beta_mode: str = "multiply",
) -> float:
    """
    Dispatch log likelihood computation based on spectrum dimensionality.

    Replaces the long if/elif chains for handling 1D, 2D, 3D spectra with
    a more scalable approach. Currently computes straightforward contributions
    but can be extended for more complex logic.

    Args:
        data_spectrum: Observed spectrum (can be 1D, 2D, or 3D)
        spectrum_model: Model spectrum (can be 1D, 2D, or 3D)
        data_mask: Mask for valid data points (matching dimensionality)
        data_uncertainties: Uncertainties (matching dimensionality)
        use_data_log_likelihood_method: Whether to use data.log_likelihood method
        beta: Uncertainty scaling parameter
        beta_mode: Mode for applying beta

    Returns:
        Total log likelihood across all dimensions
    """
    spectrum_ndim = jnp.ndim(data_spectrum)
    model_ndim = jnp.ndim(spectrum_model)

    # Compute log likelihood based on dimensionality combinations
    # Note: This is a simplified version; complex masking logic may need refinement

    if spectrum_ndim == 1 and model_ndim == 1:
        # 1D-1D case
        return compute_log_likelihood_for_spectrum(
            spectrum_model,
            data_spectrum,
            data_uncertainties,
            None,
            data_mask,
            0,
            use_data_log_likelihood_method,
            beta,
            beta_mode,
        )
    elif spectrum_ndim == 1 and model_ndim == 2:
        # Take first row of model
        return compute_log_likelihood_for_spectrum(
            spectrum_model[0, :],
            data_spectrum,
            data_uncertainties,
            None,
            None,
            0,
            use_data_log_likelihood_method,
            beta,
            beta_mode,
        )
    elif spectrum_ndim == 1 and model_ndim == 3:
        # Take first element of first two dimensions
        return compute_log_likelihood_for_spectrum(
            spectrum_model[0, 0, :],
            data_spectrum,
            data_uncertainties,
            None,
            None,
            0,
            use_data_log_likelihood_method,
            beta,
            beta_mode,
        )
    elif spectrum_ndim == 2 and model_ndim == 2:
        # 2D-2D case: loop over first dimension
        def compute_log_l_2d():
            log_l_contributions = []
            for i in range(data_spectrum.shape[0]):
                log_l_contributions.append(
                    compute_log_likelihood_for_spectrum(
                        spectrum_model[i, :],
                        data_spectrum[i, :],
                        data_uncertainties[i, :],
                        None,
                        data_mask[i, :],
                        0,
                        use_data_log_likelihood_method,
                        beta,
                        beta_mode,
                    )
                )
            return jnp.sum(jnp.array(log_l_contributions))

        return compute_log_l_2d()
    elif spectrum_ndim == 3 and model_ndim == 3:
        # 3D-3D case: loop over first two dimensions
        def compute_log_l_3d():
            log_l_contributions = []
            for i in range(data_spectrum.shape[0]):
                for j in range(data_spectrum.shape[1]):
                    log_l_contributions.append(
                        compute_log_likelihood_for_spectrum(
                            spectrum_model[i, j, :],
                            data_spectrum[i, j, :],
                            data_uncertainties[i, j, :],
                            None,
                            data_mask[i, j, :],
                            0,
                            use_data_log_likelihood_method,
                            beta,
                            beta_mode,
                        )
                    )
            return jnp.sum(jnp.array(log_l_contributions))

        return compute_log_l_3d()
    else:
        # Unhandled dimensionality combination
        raise ValueError(
            f"Unhandled spectrum dimensionality combination: "
            f"data_spectrum ndim={spectrum_ndim}, spectrum_model ndim={model_ndim}"
        )


def validate_model_output(
    spectrum_model: Optional[jnp.ndarray], wavelengths_model: Optional[jnp.ndarray]
) -> bool:
    """
    Validate model output for NaNs and None values.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if spectrum_model is None:
        return False, "spectrum_model is None"

    if wavelengths_model is None:
        return False, "wavelengths_model is None"

    # Check for masked arrays (not ideal for JAX)
    if isinstance(spectrum_model, np.ma.MaskedArray):
        spectrum_model = spectrum_model.filled(0)

    value = lax.cond(jnp.isnan(spectrum_model).any(), lambda: False, lambda: True)

    return value


def ensure_jax_array(array: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Ensure array is in JAX format for JIT compilation.

    Args:
        array: Input array (numpy or JAX)

    Returns:
        JAX array
    """
    if isinstance(array, np.ndarray):
        return jnp.array(array)
    return array
