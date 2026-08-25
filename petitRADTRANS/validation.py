"""
This module provides sanity checks on inputs for the Radtrans class.
"""

import os
import warnings
from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import core as jax_core
from jax import lax, debug
from jax.typing import ArrayLike

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE
from petitRADTRANS.stellar_spectra.phoenix import compute_star_spectrum

_UNSET = object()


def _warn_scattering_in_emission_without_scattering_species(
    scattering_in_emission: bool,
    cloud_species: list[str] | tuple[str] | None,
    rayleigh_species: list[str] | tuple[str] | None
):
    """ Returns warning if scattering is on but no scatterers are loaded.
    """

    if isinstance(cloud_species, (str, bytes)):
        cloud_species = (cloud_species,)
    elif cloud_species is None:
        cloud_species = ()
    else:
        cloud_species = tuple(cloud_species)

    if isinstance(rayleigh_species, (str, bytes)):
        rayleigh_species = (rayleigh_species,)
    elif rayleigh_species is None:
        rayleigh_species = ()
    else:
        rayleigh_species = tuple(rayleigh_species)

    if scattering_in_emission and (len(cloud_species) == 0 and len(rayleigh_species) == 0):
        warnings.warn(
            "scattering_in_emission=True, but 'cloud_species' and"
            " 'rayleigh_species' is empty. Consider re-initializing a Radtrans object"
            " with 'scattering_in_emission' set to False to calculate emission spectra"
            " more quickly.",
            stacklevel=3
        )


def _warn_partial_cloud_coverage_without_patchy_clouds(
    cloud_fraction: float,
    patchy_clouds: tuple[str, ...],
    has_partial_coverage_background_cloud: bool = False,
):
    if len(patchy_clouds) > 0 or has_partial_coverage_background_cloud:
        return

    try:
        has_partial_cloud_coverage = float(np.asarray(cloud_fraction)) < 1.0
    except (TypeError, ValueError):
        return

    if has_partial_cloud_coverage:
        warnings.warn(
            "cloud_fraction is less than 1, but there are no patchy clouds. "
            "Please add cloud species with partial coverage to 'patchy_clouds'.",
            UserWarning,
            stacklevel=3,
        )


def _check_anisotropic_cloud_scattering(mode: Union[str, bool]):
    """Check if the anisotropic cloud scattering mode is valid.

    Args:
        mode: The anisotropic cloud scattering mode.
    """
    if mode not in ["auto", True, False]:
        raise ValueError(
            f"anisotropic_cloud_scattering must be 'auto'|True|False, but was '{mode}'"
        )


def _check_line_opacity_mode(mode: str):
    """Check if the line opacity mode is valid.

    Args:
        mode: The line opacity mode.
    """
    if mode not in ["c-k", "lbl"]:
        raise ValueError(f"line_opacity_mode must be 'c-k'|'lbl', but was '{mode}'")


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


def _warn_clear_spectrum_irrelevant():
    debug.print(
        "requested to return a clear spectrum, but clouds have no effect with this setup\n"
        "To remove this warning, add non-zero cloud mass fractions, set 'cloud_fraction' to a value > 0, "
        "add an opaque cloud layer, or set 'return_clear_spectrum' to False."
    )


def _warn_clear_spectrum_partial_coverage():
    debug.print(
        "both the clear and cloudy spectra will be returned, but the returned cloudy spectrum will have "
        "a partial cloud coverage\n"
        "To remove this warning, set 'cloud_fraction' to 1, or set 'return_clear_spectrum' to False."
    )


def _validation_noop():
    return None


def _check_return_clear_spectrum_relevance(
    return_clear_spectrum: bool,
    clouds_have_effect_fn: Callable[[], ArrayLike],
    opaque_cloud_top_pressure: Optional[float],
    cloud_fraction: float,
):
    """Check if returning a clear spectrum is relevant and issue warnings if not.

    Args:
        return_clear_spectrum: If True, a clear spectrum will be returned in addition to the cloudy one.
        clouds_have_effect_fn: Zero-argument callable returning True if clouds have an effect on the spectrum.
            Only called when the check is not skipped, so callers pay no cost in the common case.
        opaque_cloud_top_pressure: The pressure of the opaque cloud top.
        cloud_fraction: The cloud fraction.
    """
    # When callers pass a concrete Python false, do not build the JAX debug
    # branches at all. This avoids spurious warning prints under transformed
    # call paths that trace through the effectful lax.cond branches even
    # though no clear spectrum was requested.
    if isinstance(return_clear_spectrum, (bool, np.bool_)) and not return_clear_spectrum:
        return

    clouds_have_effect = clouds_have_effect_fn()

    lax.cond(
        jnp.logical_and(
            jnp.logical_and(return_clear_spectrum, jnp.logical_not(clouds_have_effect)),
            opaque_cloud_top_pressure is None,
        ),
        _warn_clear_spectrum_irrelevant,
        _validation_noop,
    )

    lax.cond(
        return_clear_spectrum & (cloud_fraction < 1),
        _warn_clear_spectrum_partial_coverage,
        _validation_noop,
    )


def _check_wavelength_boundaries(boundaries: ArrayLike):
    """Check if the wavelength boundaries are valid.

    Args:
        boundaries: The wavelength boundaries.
    """
    if jnp.size(jnp.asarray(boundaries)) != 2:
        raise ValueError(
            f"wavelengths boundaries must be an array of 2 floats, but was {boundaries}"
        )

    if not boundaries[0] < boundaries[1]:
        raise ValueError(
            f"wavelengths boundaries must be an array of 2 floats in increasing order, "
            f"but was {boundaries}"
        )


def _check_clouds_particle_porosity_factors(clouds_particle_porosity_factor):
    """Check caller-supplied cloud porosity factors lie in [0, 1).

    Must be called from the un-jitted entry point: inside the spectral kernels
    these values are tracers and cannot be inspected. Non-finite values are the
    "unset" sentinel that the kernel replaces with a valid per-source default
    (0.25 for DHS, 0.0 otherwise), so they are skipped.
    """
    if clouds_particle_porosity_factor is None:
        return

    for species_name, porosity in clouds_particle_porosity_factor.items():
        if isinstance(porosity, jax_core.Tracer):
            continue

        porosity_value = float(jnp.asarray(porosity))

        if not np.isfinite(porosity_value):
            continue

        if porosity_value < 0 or porosity_value >= 1:
            raise ValueError(
                f"clouds particles porosity factor "
                f"must be >= 0 (full particles) and < 1 (empty particles), "
                f"but is {porosity_value} for species '{species_name}'"
            )


def _check_power_law_opacity_completeness(
    power_law_opacity_350nm, power_law_opacity_coefficient
):
    """Check the power-law opacity parameters are supplied as a pair.

    Must be called from the un-jitted entry point, where "not supplied" is still
    visible as ``None``; inside the kernels it has already been replaced by a
    sentinel and the values are tracers.
    """
    a_missing = power_law_opacity_350nm is None
    b_missing = power_law_opacity_coefficient is None

    if a_missing ^ b_missing:
        none_parameter = (
            "power_law_opacity_350nm" if a_missing else "power_law_opacity_coefficient"
        )
        raise RuntimeError(
            f"to include a power law opacity, both parameters 'power_law_opacity_350nm' "
            f"and 'power_law_opacity_coefficient' must be provided; missing '{none_parameter}'"
        )


def _sanitize_frequency_surface_input(
    values, frequencies, default_value, quantity_name
):
    if values is None:
        return jnp.full_like(
            frequencies, default_value, dtype=WORKING_JAX_DTYPE
        )

    values = jnp.array(values, dtype=WORKING_JAX_DTYPE)

    if jnp.ndim(values) == 0:
        return values * jnp.ones_like(frequencies, dtype=WORKING_JAX_DTYPE)

    # In the past a debug.callback was used here.
    # But since _sanitize is called before the kernels, in the non-jitted part,
    # this error raising will always only happen at compile time,
    # since if the emissivity shape is changed a new compile is triggered.
    if jnp.size(values) != frequencies.size:
        raise ValueError(
            f"{quantity_name} must be a scalar or of the same size than frequencies "
            f"({frequencies.size}), but is of size {jnp.size(values)}"
        )

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
        stellar_intensities = jnp.zeros_like(
            frequencies, dtype=FEAUTRIER_JAX_DTYPE
        )

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
        stellar_intensities = jnp.array(
            stellar_intensities, dtype=FEAUTRIER_JAX_DTYPE
        )

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


def _normalize_optional_mapping(mapping, dtype=WORKING_JAX_DTYPE):
    if mapping is None or isinstance(mapping, (int, float)):
        return mapping

    if isinstance(mapping, (np.ndarray, jnp.ndarray)):
        return jnp.array(mapping, dtype=dtype)

    normalized_mapping = {}

    for key, value in list(mapping.items()):
        normalized_mapping[key] = jnp.array(value, dtype=dtype)

    return normalized_mapping


def _normalize_optional_scalar(value, missing_value, dtype=WORKING_JAX_DTYPE):
    if value is None:
        return missing_value

    return jnp.array(value, dtype=dtype)


def _normalize_optional_mapping_or_scalar(
    value,
    missing_value,
    dtype=WORKING_JAX_DTYPE,
):
    if value is None:
        return missing_value

    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return jnp.array(value, dtype=dtype)

    if isinstance(value, dict):
        normalized_mapping = {}

        for key, item in list(value.items()):
            normalized_mapping[key] = jnp.array(item, dtype=dtype)

        return normalized_mapping

    return jnp.array(value, dtype=dtype)


def _normalize_cloud_species_selection(selection, default_species):
    if selection is None:
        return tuple(default_species)

    if isinstance(selection, (str, bytes)):
        selection = (selection,)

    selection = tuple(selection)

    return tuple(species for species in default_species if species in selection)


def _sanitize_flux_opacity_inputs(
    patchy_clouds,
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
    additional_opacities_placeholder_fn: Callable,
):
    if cloud_f_sed is None:
        cloud_f_sed_norm = {}
    elif isinstance(cloud_f_sed, dict):
        cloud_f_sed_norm = {
            key: jnp.array(value, dtype=FEAUTRIER_JAX_DTYPE)
            for key, value in cloud_f_sed.items()
        }
    else:
        cloud_f_sed_norm = {
            species: jnp.array(cloud_f_sed, dtype=FEAUTRIER_JAX_DTYPE)
            for species in cloud_species
        }

    clouds_particle_porosity_factor_norm = {
        species: missing_neg_inf for species in cloud_species
    }
    # NOTE: this sanitiser runs inside the jitted spectral kernels, so values are
    # tracers here and cannot be range-checked. The [0, 1) check lives in
    # _check_clouds_particle_porosity_factors, called from the un-jitted
    # Radtrans entry points.
    if clouds_particle_porosity_factor is not None:
        for key, value in list(clouds_particle_porosity_factor.items()):
            clouds_particle_porosity_factor_norm[key] = jnp.array(
                value, dtype=FEAUTRIER_JAX_DTYPE
            )

    if eddy_diffusion_coefficients is None:
        eddy_diffusion_coefficients_norm = None
    elif isinstance(eddy_diffusion_coefficients, dict):
        eddy_diffusion_coefficients_norm = {
            key: jnp.array(value, dtype=FEAUTRIER_JAX_DTYPE)
            for key, value in eddy_diffusion_coefficients.items()
        }
    else:
        eddy_diffusion_coefficients_norm = jnp.array(
            eddy_diffusion_coefficients, dtype=FEAUTRIER_JAX_DTYPE
        )

    return {
        "patchy_clouds": _normalize_cloud_species_selection(
            selection=patchy_clouds,
            default_species=cloud_species,
        ),
        "cloud_particle_mean_radii": _normalize_optional_mapping(
            cloud_particle_mean_radii,
            dtype=FEAUTRIER_JAX_DTYPE,
        ),
        "cloud_hansen_a": _normalize_optional_mapping(
            cloud_hansen_a, dtype=FEAUTRIER_JAX_DTYPE
        ),
        "cloud_hansen_b": _normalize_optional_mapping(
            cloud_hansen_b, dtype=FEAUTRIER_JAX_DTYPE
        ),
        "clouds_particle_porosity_factor": clouds_particle_porosity_factor_norm,
        "cloud_f_sed": cloud_f_sed_norm,
        "eddy_diffusion_coefficients": eddy_diffusion_coefficients_norm,
        # Kept in the pressure grid's precision rather than the default working
        # precision: it is compared against that grid to place the cloud deck, and
        # a float32 round-trip moves the comparison by ~1e-4 relative, which is
        # enough to flip a layer that lies exactly at the cloud top.
        "opaque_cloud_top_pressure": _normalize_optional_scalar(
            opaque_cloud_top_pressure, missing_float, dtype=FEAUTRIER_JAX_DTYPE
        ),
        "cloud_particle_radius_distribution_std": _normalize_optional_mapping_or_scalar(
            cloud_particle_radius_distribution_std,
            None,
            dtype=FEAUTRIER_JAX_DTYPE,
        ),
        "power_law_opacity_350nm": _normalize_optional_scalar(
            power_law_opacity_350nm, missing_float
        ),
        "power_law_opacity_coefficient": _normalize_optional_scalar(
            power_law_opacity_coefficient,
            missing_float,
        ),
        "gray_opacity": (
            jnp.array(0.0, dtype=WORKING_JAX_DTYPE)
            if gray_opacity is None
            else jnp.array(gray_opacity, dtype=WORKING_JAX_DTYPE)
        ),
        "cloud_photosphere_median_optical_depth": _normalize_optional_scalar(
            cloud_photosphere_median_optical_depth,
            missing_float,
        ),
        "cloud_particle_number_density_grid": _normalize_optional_mapping(
            cloud_particle_number_density_grid,
            dtype=FEAUTRIER_JAX_DTYPE,
        ),
        "additional_absorption_opacities_function": (
            additional_absorption_opacities_function
            if additional_absorption_opacities_function is not None
            else additional_opacities_placeholder_fn
        ),
        "additional_scattering_opacities_function": (
            additional_scattering_opacities_function
            if additional_scattering_opacities_function is not None
            else additional_opacities_placeholder_fn
        ),
        "target_particle_radii": (
            None
            if target_particle_radii is None
            else jnp.array(target_particle_radii, dtype=FEAUTRIER_JAX_DTYPE)
        ),
        "target_pressure": (
            None
            if target_pressure is None
            else jnp.array(target_pressure, dtype=FEAUTRIER_JAX_DTYPE)
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
