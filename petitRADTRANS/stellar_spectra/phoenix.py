"""
"""
import os

import h5py
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.config.configuration import petitradtrans_config_parser, get_input_data_subpaths
from petitRADTRANS._input_data_loader import get_input_data_file_not_found_error_message


def __load_stellar_spectra(file=None):
    if file is None:
        file = get_default_phoenix_file()

    if not os.path.isfile(file):
        raise FileNotFoundError(get_input_data_file_not_found_error_message(file))

    with h5py.File(file, "r") as f:
        log10_effective_temperature_grid = f['log10_effective_temperatures'][()]
        radius_grid = f['radii'][()]
        flux_grid = f['fluxes'][()]
        wavelength_grid = f['wavelengths'][()]

    return log10_effective_temperature_grid, radius_grid, flux_grid, wavelength_grid


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
    interpolation_index = np.searchsorted(phoenix_log10_effective_temperature_grid, log_temp)

    if interpolation_index == 0:
        spec_dat = phoenix_flux_grid[0]
        radius = phoenix_radius_grid[0]
        print('Warning, input temperature is lower than minimum grid temperature.')
        print('Taking F = F_grid(minimum grid temperature), normalized to desired')
        print('input temperature.')

    elif interpolation_index == len(phoenix_log10_effective_temperature_grid):
        spec_dat = phoenix_flux_grid[int(len(phoenix_log10_effective_temperature_grid) - 1)]
        radius = phoenix_radius_grid[int(len(phoenix_log10_effective_temperature_grid) - 1)]
        print('Warning, input temperature is higher than maximum grid temperature.')
        print('Taking F = F_grid(maximum grid temperature), normalized to desired')
        print('input temperature.')

    else:
        weight_high = (
            (log_temp - phoenix_log10_effective_temperature_grid[interpolation_index - 1])
            / (phoenix_log10_effective_temperature_grid[interpolation_index]
               - phoenix_log10_effective_temperature_grid[interpolation_index - 1])
        )

        weight_low = 1. - weight_high

        spec_dat_low = phoenix_flux_grid[int(interpolation_index - 1)]

        spec_dat_high = phoenix_flux_grid[int(interpolation_index)]

        spec_dat = weight_low * spec_dat_low \
            + weight_high * spec_dat_high

        radius = weight_low * phoenix_radius_grid[int(interpolation_index - 1)] \
            + weight_high * phoenix_radius_grid[int(interpolation_index)]

    freq = cst.c / phoenix_wavelength_grid
    flux = spec_dat
    norm = -np.sum((flux[1:] + flux[:-1]) * np.diff(freq)) / 2.

    spec_dat = flux / norm * cst.sigma * temperature ** 4.

    spec_dat = np.transpose(np.stack((phoenix_wavelength_grid, spec_dat)))

    return spec_dat, radius


def get_default_phoenix_file():
    return os.path.join(
        petitradtrans_config_parser.get_input_data_path(),
        get_input_data_subpaths()["stellar_spectra"],
        "phoenix",
        "phoenix.startable.petitRADTRANS.h5"
    )


(phoenix_log10_effective_temperature_grid,
 phoenix_radius_grid, phoenix_flux_grid, phoenix_wavelength_grid) = __load_stellar_spectra()
