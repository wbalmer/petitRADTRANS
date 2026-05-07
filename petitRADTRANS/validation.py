"""
This module provides sanity checks on inputs for the Radtrans class.
"""

import os
from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import lax, debug
from jax.typing import ArrayLike

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.jax_dtype import JAX_DTYPE
from petitRADTRANS.stellar_spectra.phoenix import compute_star_spectrum


def _check_anisotropic_cloud_scattering(mode: Union[str, bool]):
    """Check if the anisotropic cloud scattering mode is valid.

    Args:
        mode: The anisotropic cloud scattering mode.
    """
    if mode not in ["auto", True, False]:
        raise ValueError(
            f"anisotropic cloud scattering must be 'auto'|True|False, but was '{mode}'"
        )


def _check_line_opacity_mode(mode: str):
    """Check if the line opacity mode is valid.

    Args:
        mode: The line opacity mode.
    """
    if mode not in ["c-k", "lbl"]:
        raise ValueError(f"opacity mode must be 'c-k'|'lbl', but was '{mode}'")


def _check_input_data_file_existence(path: str):
    """Check if an input data file exists.

    Args:
        path: The path to the input data file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no such file or directory: '{path}'\n"
            f"This may be caused by an incorrect input_data path, outdated file formatting, or a missing file\n\n"
            f"To set the input_data path, execute: \n"
            f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
            f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
            f"replacing 'path/to/' with the path to the input_data directory (setting "
            f"pRT_input_data_path by an environment variable is no longer supported, "
            f"this environment variable can be removed safely from your setup)\n\n"
            f"To update the outdated .dat files to HDF5, execute:\n"
            f">>> from petitRADTRANS._file_conversion import convert_all\n"
            f">>> convert_all()\n\n"
            f"To download the missing file, "
            f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
        )


def _check_path_input_data(path: str):
    """Check if the input data path exists.

    Args:
        path: The path to the input data directory.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no such directory: '{path}'\n"
            f"This may be caused by an incorrect input_data path, or a missing basic input_data directory\n\n"
            f"To set the input_data path, execute: \n"
            f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
            f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
            f"replacing 'path/to/' with the path to the input_data directory (setting "
            f"pRT_input_data_path by an environment variable is no longer supported, "
            f"this environment variable can be removed safely from your setup)\n\n"
            f"To download the basic input_data directory, "
            f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
        )


def _check_pressures(pressures: ArrayLike):
    """Check if the pressures are in increasing order.

    Args:
        pressures: The array of pressures.
    """
    if not jnp.all(jnp.diff(pressures) > 0):
        raise ValueError("pressures must be an array in increasing order")


def _check_return_clear_spectrum_relevance(
    return_clear_spectrum: bool,
    clouds_have_effect: bool,
    opaque_cloud_top_pressure: Optional[float],
    cloud_fraction: float,
):
    """Check if returning a clear spectrum is relevant and issue warnings if not.

    Args:
        return_clear_spectrum: If True, a clear spectrum will be returned in addition to the cloudy one.
        clouds_have_effect: If True, clouds have an effect on the spectrum.
        opaque_cloud_top_pressure: The pressure of the opaque cloud top.
        cloud_fraction: The cloud fraction.
    """
    # When callers pass a concrete Python false, do not build the JAX debug
    # branches at all. This avoids spurious warning prints under transformed
    # call paths that trace through the effectful lax.cond branches even
    # though no clear spectrum was requested.
    if isinstance(return_clear_spectrum, (bool, np.bool_)) and not return_clear_spectrum:
        return

    lax.cond(  # TODO: revert to clearer pythonic statements
        jnp.logical_and(
            jnp.logical_and(return_clear_spectrum, jnp.logical_not(clouds_have_effect)),
            opaque_cloud_top_pressure is None,
        ),
        lambda: debug.print(
            "requested to return a clear spectrum, but clouds have no effect with this setup\n"
            "To remove this warning, add non-zero cloud mass fractions, set 'cloud_fraction' to a value > 0, "
            "add an opaque cloud layer, or set 'return_clear_spectrum' to False."
        ),
        lambda: None,
    )

    lax.cond(
        return_clear_spectrum & (cloud_fraction < 1),
        lambda: debug.print(
            "both the clear and cloudy spectra will be returned, but the returned cloudy spectrum will have "
            "a partial cloud coverage\n"
            "To remove this warning, set 'cloud_fraction' to 1, or set 'return_clear_spectrum' to False."
        ),
        lambda: None,
    )


def _check_wavelength_boundaries(boundaries: ArrayLike):
    """Check if the wavelength boundaries are valid.

    Args:
        boundaries: The wavelength boundaries.
    """
    if jnp.size(boundaries) != 2:
        raise ValueError(
            f"wavelengths boundaries must be an array of 2 floats, but was {boundaries}"
        )

    if not boundaries[0] < boundaries[1]:
        raise ValueError(
            f"wavelengths boundaries must be an array of 2 floats in increasing order, "
            f"but was {boundaries}"
        )


def _sanitize_frequency_surface_input(
    values, frequencies, default_value, quantity_name
):
    if values is None:
        return jnp.full_like(frequencies, default_value, dtype=JAX_DTYPE)

    values = jnp.array(values, dtype=JAX_DTYPE)

    if jnp.ndim(values) == 0:
        return values * jnp.ones_like(frequencies, dtype=JAX_DTYPE)

    def _check_input_size(_values, _frequencies):
        if jnp.size(_values) != _frequencies.size:
            raise ValueError(
                f"{quantity_name} must be a scalar or of the same size than frequencies "
                f"({_frequencies.size}), but is of size {jnp.size(_values)}"
            )

    debug.callback(_check_input_size, values, frequencies)

    return values


def _sanitize_emission_surface_inputs(
    frequencies,
    scattering_in_emission,
    star_irradiation_angle,
    stellar_intensities,
    star_effective_temperature,
    star_radius,
    orbit_semi_major_axis,
    reflectances,
    emissivities,
):
    star_irradiation_cos_angle = jnp.cos(jnp.deg2rad(star_irradiation_angle))
    star_irradiation_cos_angle = jnp.where(
        star_irradiation_cos_angle <= 0.0, 1e-8, star_irradiation_cos_angle
    )

    if stellar_intensities is None:
        stellar_intensities = jnp.zeros_like(frequencies, dtype=JAX_DTYPE)

        if (
            scattering_in_emission
            and star_effective_temperature is not None
            and orbit_semi_major_axis is not None
        ):
            stellar_intensities = compute_star_spectrum(
                star_effective_temperature=star_effective_temperature,
                orbit_semi_major_axis=orbit_semi_major_axis,
                frequencies=frequencies,
                star_radius=star_radius,
            )
    else:
        stellar_intensities = jnp.array(stellar_intensities, dtype=JAX_DTYPE)

    reflectances = _sanitize_frequency_surface_input(
        values=reflectances,
        frequencies=frequencies,
        default_value=0.0,
        quantity_name="reflectance",
    )
    emissivities = _sanitize_frequency_surface_input(
        values=emissivities,
        frequencies=frequencies,
        default_value=1.0,
        quantity_name="emissivity",
    )

    return {
        "star_irradiation_cos_angle": star_irradiation_cos_angle,
        "stellar_intensities": stellar_intensities,
        "reflectances": reflectances,
        "emissivities": emissivities,
    }


def _normalize_optional_mapping(mapping):
    if mapping is None or isinstance(mapping, (int, float)):
        return mapping

    if isinstance(mapping, (np.ndarray, jnp.ndarray)):
        return jnp.array(mapping, dtype=JAX_DTYPE)

    normalized_mapping = {}

    for key, value in list(mapping.items()):
        normalized_mapping[key] = jnp.array(value, dtype=JAX_DTYPE)

    return normalized_mapping


def _normalize_optional_scalar(value, missing_value):
    if value is None:
        return missing_value

    return jnp.array(value, dtype=JAX_DTYPE)


def _sanitize_flux_opacity_inputs(
    complete_coverage_clouds,
    cloud_particle_mean_radii,
    cloud_hansen_a,
    cloud_hansen_b,
    clouds_particle_porosity_factor,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    opaque_cloud_top_pressure,
    cloud_particle_radius_distribution_std,
    power_law_opacity_350nm,
    power_law_opacity_coefficient,
    gray_opacity,
    cloud_photosphere_median_optical_depth,
    cloud_particle_number_density_grid,
    additional_absorption_opacities_function,
    additional_scattering_opacities_function,
    target_particle_radii,
    target_pressure,
    cloud_species,
    missing_float,
    missing_neg_inf,
    noop_opacity_fn: Callable,
):
    if cloud_f_sed is None:
        cloud_f_sed_norm = {}
    elif isinstance(cloud_f_sed, dict):
        cloud_f_sed_norm = {
            key: jnp.array(value, dtype=JAX_DTYPE) for key, value in cloud_f_sed.items()
        }
    else:
        cloud_f_sed_norm = {
            species: jnp.array(cloud_f_sed, dtype=JAX_DTYPE)
            for species in cloud_species
        }

    clouds_particle_porosity_factor_norm = {
        species: missing_neg_inf for species in cloud_species
    }
    if clouds_particle_porosity_factor is not None:
        for key, value in list(clouds_particle_porosity_factor.items()):
            clouds_particle_porosity_factor_norm[key] = jnp.array(
                value, dtype=JAX_DTYPE
            )

    if eddy_diffusion_coefficients is None:
        eddy_diffusion_coefficients_norm = None
    elif isinstance(eddy_diffusion_coefficients, dict):
        eddy_diffusion_coefficients_norm = {
            key: jnp.array(value, dtype=JAX_DTYPE)
            for key, value in eddy_diffusion_coefficients.items()
        }
    else:
        eddy_diffusion_coefficients_norm = jnp.array(
            eddy_diffusion_coefficients, dtype=JAX_DTYPE
        )

    return {
        "complete_coverage_clouds": (
            () if complete_coverage_clouds is None else tuple(complete_coverage_clouds)
        ),
        "cloud_particle_mean_radii": _normalize_optional_mapping(
            cloud_particle_mean_radii
        ),
        "cloud_hansen_a": _normalize_optional_mapping(cloud_hansen_a),
        "cloud_hansen_b": _normalize_optional_mapping(cloud_hansen_b),
        "clouds_particle_porosity_factor": clouds_particle_porosity_factor_norm,
        "cloud_f_sed": cloud_f_sed_norm,
        "eddy_diffusion_coefficients": eddy_diffusion_coefficients_norm,
        "opaque_cloud_top_pressure": _normalize_optional_scalar(
            opaque_cloud_top_pressure, missing_float
        ),
        "cloud_particle_radius_distribution_std": _normalize_optional_scalar(
            cloud_particle_radius_distribution_std,
            missing_float,
        ),
        "power_law_opacity_350nm": _normalize_optional_scalar(
            power_law_opacity_350nm, missing_float
        ),
        "power_law_opacity_coefficient": _normalize_optional_scalar(
            power_law_opacity_coefficient,
            missing_float,
        ),
        "gray_opacity": (
            jnp.array(0.0, dtype=JAX_DTYPE)
            if gray_opacity is None
            else jnp.array(gray_opacity, dtype=JAX_DTYPE)
        ),
        "cloud_photosphere_median_optical_depth": _normalize_optional_scalar(
            cloud_photosphere_median_optical_depth,
            missing_float,
        ),
        "cloud_particle_number_density_grid": _normalize_optional_mapping(
            cloud_particle_number_density_grid
        ),
        "additional_absorption_opacities_function": (
            additional_absorption_opacities_function
            if additional_absorption_opacities_function is not None
            else noop_opacity_fn
        ),
        "additional_scattering_opacities_function": (
            additional_scattering_opacities_function
            if additional_scattering_opacities_function is not None
            else noop_opacity_fn
        ),
        "target_particle_radii": (
            None
            if target_particle_radii is None
            else jnp.array(target_particle_radii, dtype=JAX_DTYPE)
        ),
        "target_pressure": (
            None
            if target_pressure is None
            else jnp.array(target_pressure, dtype=JAX_DTYPE)
        ),
    }


def _handle_grid_misalignment(
    indices: npt.NDArray[bool],
    frequencies_reference: ArrayLike,
    frequencies_test: ArrayLike,
) -> npt.NDArray[bool]:
    """Handle frequency grid misalignment by extending the selection of indices.

    This function is a workaround for cases where the frequency grid of the model does not perfectly align with the
    frequency grid of the opacity data, even after selecting the closest frequencies. It extends the selection of
    indices to include one more frequency to the left or right to ensure that the model's frequency range is
    fully covered by the opacity data.

    Args:
        indices: The boolean array of indices selecting the frequencies from the opacity data.
        frequencies_reference: The frequency grid of the model.
        frequencies_test: The frequency grid of the opacity data.

    Returns:
        The updated boolean array of indices.
    """
    _indices: npt.NDArray[np.signedinteger] = np.nonzero(indices)[0]

    if np.abs(frequencies_test[indices][0] - frequencies_reference[0]) < np.abs(
        frequencies_test[indices][0] - frequencies_reference[1]
    ):  # right misalignment
        # Situation is as follows:
        # 0   1   2   3
        # |...|...|...|  (ref.)
        #  |...|...|     (test)
        #  0   1   2
        # Include one more frequency to the right
        # 0   1   2   3
        # |...|...|...|  (ref.)
        #  |...|...|...| (test)
        #  0   1   2   3
        index_include = _indices[-1] + 1
    else:  # left misalignment
        # Situation is as follows:
        #  0   1   2   3
        #  |...|...|...|  (ref.)
        #     |...|...|   (test)
        #     1   2   3
        # Include one more frequency to the right
        #  0   1   2   3
        #  |...|...|...|  (ref.)
        # |...|...|...|   (test)
        # 0   1   2   3
        index_include = _indices[0] - 1

    if (
        index_include < 0 or index_include == frequencies_test.size
    ):  # handle out-of-bounds corrections
        _wavelengths = 1e4 * cst.c / frequencies_test
        wavelengths = 1e4 * cst.c / frequencies_reference

        raise ValueError(
            f"unrecoverable frequency grid misalignment: "
            f"reference frequency array is of size {frequencies_reference.size}, "
            f"but corresponding frequency array "
            f"is of size {_indices.size}; "
            f"a grid realignment was attempted, but required a frequency interval larger than stored in "
            f"the opacity file ({_wavelengths[0]} -- {_wavelengths[-1]} um in file, "
            f"{wavelengths[0]} -- {wavelengths[-1]} um in model)\n"
            f"Use a slightly smaller wavelength range, or use different opacity files"
        )

    indices[index_include] = True

    return indices
