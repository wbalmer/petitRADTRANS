import os
import jax.numpy as jnp
from jax import lax, debug
from jax.typing import ArrayLike
import numpy as np
import numpy.typing as npt
import petitRADTRANS.physical_constants as cst
from typing import Union, Tuple, Optional, List, Dict, Any, Callable, TypeVar


def _check_anisotropic_cloud_scattering(mode):
    if mode not in ['auto', True, False]:
        raise ValueError(f"anisotropic cloud scattering must be 'auto'|True|False, but was '{mode}'")


def _check_line_opacity_mode(mode):
    if mode not in ['c-k', 'lbl']:
        raise ValueError(f"opacity mode must be 'c-k'|'lbl', but was '{mode}'")


def _check_input_data_file_existence(path):
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


def _check_path_input_data(path):
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


def _check_pressures(pressures):
    if not jnp.all(jnp.diff(pressures) > 0):
        raise ValueError("pressures must be an array in increasing order")


def _check_return_clear_spectrum_relevance(
    return_clear_spectrum, clouds_have_effect,
    opaque_cloud_top_pressure, cloud_fraction
):
    lax.cond(  # TODO: revert to clearer pythonic statements
        jnp.logical_and(
            jnp.logical_and(return_clear_spectrum,
                            jnp.logical_not(clouds_have_effect)),
            opaque_cloud_top_pressure is None),
        lambda: debug.print(
            "requested to return a clear spectrum, but clouds have no effect with this setup\n"
            "To remove this warning, add non-zero cloud mass fractions, set 'cloud_fraction' to a value > 0, "
            "add an opaque cloud layer, or set 'return_clear_spectrum' to False."),
        lambda: None
        )

    lax.cond(
        return_clear_spectrum & (cloud_fraction < 1),
        lambda: debug.print(
            "both the clear and cloudy spectra will be returned, but the returned cloudy spectrum will have "
            "a partial cloud coverage\n"
            "To remove this warning, set 'cloud_fraction' to 1, or set 'return_clear_spectrum' to False."),
        lambda: None
    )


def _check_wavelength_boundaries(boundaries):
    if jnp.size(boundaries) != 2:
        raise ValueError(f"wavelengths boundaries must be an array of 2 floats, but was {boundaries}")

    if not boundaries[0] < boundaries[1]:
        raise ValueError(f"wavelengths boundaries must be an array of 2 floats in increasing order, "
                            f"but was {boundaries}")


def _handle_grid_misalignment(
    indices: npt.NDArray[bool],
    frequencies_reference: ArrayLike,
    frequencies_test: ArrayLike
    ) -> npt.NDArray[bool]:
    _indices: npt.NDArray[np.signedinteger] = np.nonzero(indices)[0]

    if (
            np.abs(frequencies_test[indices][0] - frequencies_reference[0])
            < np.abs(frequencies_test[indices][0] - frequencies_reference[1])
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

    if index_include < 0 or index_include == frequencies_test.size:  # handle out-of-bounds corrections
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
