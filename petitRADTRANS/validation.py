"""
This module provides sanity checks on inputs for the Radtrans class.
"""

import os
from typing import Union, Optional

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import lax, debug
from jax.typing import ArrayLike

import petitRADTRANS.physical_constants as cst


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
