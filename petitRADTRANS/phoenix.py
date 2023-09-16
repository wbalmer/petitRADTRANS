"""
"""
import os

import h5py
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.config import petitradtrans_config


def __load_stellar_spectra():
    spec_file = os.path.join(spec_path, "stellar_spectra.h5")

    if not os.path.isfile(spec_file):
        raise FileNotFoundError(
            f"No such file: '{spec_file}'\n"
            f"This may be caused by an incorrect input_data path, outdated file formatting, or a missing file\n\n"
            f"To set the input_data path, execute: \n"
            f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
            f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
            f"replacing 'path/to/' with the path to the input_data directory\n\n"
            f"To update the outdated files, execute:\n"
            f">>> from petitRADTRANS.__file_conversion import convert_all\n"
            f">>> convert_all()\n\n"
            f"To download the missing file, "
            f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html")

    with h5py.File(spec_file, "r") as f:
        log_temp_grid = f['log10_effective_temperature'][()]
        star_rad_grid = f['radius'][()]
        spec_dats = f['spectral_radiosity'][()]
        wavelength = f['wavelength'][()]

    return log_temp_grid, star_rad_grid, spec_dats, wavelength


spec_path = os.path.join(petitradtrans_config['Paths']['pRT_input_data_path'], 'stellar_specs')

logTempGrid, StarRadGrid, specDats, wavelength_stellar = __load_stellar_spectra()


def compute_phoenix_spectrum(temperature):
    """
    Returns a matrix where the first column is the wavelength in cm
    and the second is the stellar flux :math:`F_\\nu` in units of
    :math:`\\rm erg/cm^2/s/Hz`, at the surface of the star.
    The spectra are PHOENIX models from (Husser et al. 2013), the spectral
    grid used here was described in van Boekel et al. (2012).

    Args:
        temperature (float):
            stellar effective temperature in K.
    """
    log_temp = np.log10(temperature)
    interpolation_index = np.searchsorted(logTempGrid, log_temp)

    if interpolation_index == 0:
        spec_dat = specDats[0]
        radius = StarRadGrid[0]
        print('Warning, input temperature is lower than minimum grid temperature.')
        print('Taking F = F_grid(minimum grid temperature), normalized to desired')
        print('input temperature.')

    elif interpolation_index == len(logTempGrid):
        spec_dat = specDats[int(len(logTempGrid) - 1)]
        radius = StarRadGrid[int(len(logTempGrid) - 1)]
        print('Warning, input temperature is higher than maximum grid temperature.')
        print('Taking F = F_grid(maximum grid temperature), normalized to desired')
        print('input temperature.')

    else:
        weight_high = (log_temp - logTempGrid[interpolation_index - 1]) / \
                     (logTempGrid[interpolation_index] - logTempGrid[interpolation_index - 1])

        weight_low = 1. - weight_high

        spec_dat_low = specDats[int(interpolation_index - 1)]

        spec_dat_high = specDats[int(interpolation_index)]

        spec_dat = weight_low * spec_dat_low \
            + weight_high * spec_dat_high

        radius = weight_low * StarRadGrid[int(interpolation_index - 1)] \
            + weight_high * StarRadGrid[int(interpolation_index)]

    freq = cst.c / wavelength_stellar
    flux = spec_dat
    norm = -np.sum((flux[1:] + flux[:-1]) * np.diff(freq)) / 2.

    spec_dat = flux / norm * cst.sigma * temperature ** 4.

    spec_dat = np.transpose(np.stack((wavelength_stellar, spec_dat)))

    return spec_dat, radius
