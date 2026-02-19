"""Manages the Phoenix star table.
"""
import os

import h5py
import jax.numpy as jnp

import petitRADTRANS.physical_constants as cst
from petitRADTRANS._input_data import find_input_file
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.physics import wavelength2frequency, frequency2wavelength, rebin_spectrum

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64


class PhoenixStarTable:
    """Used to store petitRADTRANS's PHOENIX star spectrum models.

    The compute_spectrum function can be used to get a star spectrum at a given temperature.
    """
    def __init__(self):
        self._loaded = False

        self.log10_effective_temperature_grid = None
        self.radius_grid = None
        self.flux_grid = None
        self.wavelength_grid = None

    # TODO: Jit this function - update conds and use jnp?
    def compute_spectrum(self, temperature):
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
        if not self._loaded:
            self.load()

        log_temp = jnp.log10(temperature)
        interpolation_index = jnp.searchsorted(self.log10_effective_temperature_grid, log_temp)

        if interpolation_index == 0:
            spec_dat = self.flux_grid[0]
            radius = self.radius_grid[0]
            print('Warning, input temperature is lower than minimum grid temperature.')
            print('Taking F = F_grid(minimum grid temperature), normalized to desired')
            print('input temperature.')

        elif interpolation_index == len(self.log10_effective_temperature_grid):
            spec_dat = self.flux_grid[int(len(self.log10_effective_temperature_grid) - 1)]
            radius = self.radius_grid[int(len(self.log10_effective_temperature_grid) - 1)]
            print('Warning, input temperature is higher than maximum grid temperature.')
            print('Taking F = F_grid(maximum grid temperature), normalized to desired')
            print('input temperature.')

        else:
            weight_high = (
                    (log_temp - self.log10_effective_temperature_grid[interpolation_index - 1])
                    / (self.log10_effective_temperature_grid[interpolation_index]
                       - self.log10_effective_temperature_grid[interpolation_index - 1])
            )

            weight_low = 1. - weight_high

            spec_dat_low = self.flux_grid[int(interpolation_index - 1)]

            spec_dat_high = self.flux_grid[int(interpolation_index)]

            spec_dat = (
                weight_low * spec_dat_low
                + weight_high * spec_dat_high
            )

            radius = (
                weight_low * self.radius_grid[int(interpolation_index - 1)]
                + weight_high * self.radius_grid[int(interpolation_index)]
            )

        frequency_grid = wavelength2frequency(self.wavelength_grid)
        flux = spec_dat
        norm = -jnp.sum((flux[1:] + flux[:-1]) * jnp.diff(frequency_grid)) / 2.

        spec_dat = flux / norm * cst.sigma * temperature ** 4.

        spec_dat = jnp.transpose(jnp.stack((self.wavelength_grid, spec_dat)))

        return spec_dat, radius

    @staticmethod
    def get_default_file(path_input_data=None):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        return os.path.join(
            path_input_data,
            get_input_data_subpaths()["stellar_spectra"],
            "phoenix",
            "phoenix.startable.petitRADTRANS.h5"
        )

    def load(self, file=None, path_input_data=None, search_online=True):
        if file is None:
            file = self.get_default_file(path_input_data)

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        if not os.path.isfile(file):
            file = find_input_file(
                file=file,
                path_input_data=path_input_data,
                sub_path=None,
                find_all=False,
                search_online=search_online
            )

        print(f"Loading PHOENIX star table in file '{file}'... ", end='')

        with h5py.File(file, "r") as f:
            self.log10_effective_temperature_grid = f['log10_effective_temperatures'][()]
            self.radius_grid = f['radii'][()]
            self.flux_grid = f['fluxes'][()]
            self.wavelength_grid = f['wavelengths'][()]

        self._loaded = True

        print("Done.")


def compute_star_spectrum(star_effective_temperature, orbit_semi_major_axis, frequencies, star_radius=None):
    """Method to get the PHOENIX spectrum of the star and rebin it
    to the wavelength points. If t_star is not explicitly written, the
    spectrum will be 0. If the distance is not explicitly written,
    the code will raise an error and break to urge the user to
    specify the value.

        Args:
            star_effective_temperature (float):
                the stellar temperature in K.
            orbit_semi_major_axis (float):
                the semi-major axis of the planet in cm.
            frequencies (float):
                the frequencies on which to interpolate the spectrum in Hz.
            star_radius (float):
                if specified, uses this radius in cm
                to scale the flux, otherwise it uses PHOENIX radius.
    """
    phoenix_star_table = PhoenixStarTable()
    if star_radius is not None:
        star_spectrum, _ = phoenix_star_table.compute_spectrum(star_effective_temperature)
        _star_radius = star_radius
    else:
        star_spectrum, _star_radius = phoenix_star_table.compute_spectrum(star_effective_temperature)

    stellar_intensity = rebin_star_spectrum(
        star_spectrum=star_spectrum[:, 1],
        star_wavelengths=star_spectrum[:, 0],
        wavelengths=frequency2wavelength(frequencies)
    )

    stellar_intensity = stellar_intensity / jnp.pi * (star_radius / orbit_semi_major_axis) ** 2

    return stellar_intensity


def rebin_star_spectrum(star_spectrum, star_wavelengths, wavelengths):
    add_stellar_flux = jnp.zeros(100, dtype=JAX_DTYPE)
    add_wavelengths = jnp.logspace(jnp.log10(1.0000002e-02), 2, 100)

    wavelengths_interp = jnp.append(star_wavelengths, add_wavelengths)
    fluxes_interp = jnp.append(star_spectrum, add_stellar_flux)

    stellar_intensity = rebin_spectrum(wavelengths_interp, fluxes_interp, wavelengths)

    return stellar_intensity