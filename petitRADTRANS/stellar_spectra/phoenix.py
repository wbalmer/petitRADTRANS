"""Manages the Phoenix star table."""

import os

import h5py
import jax.numpy as jnp
import numpy as np
from jax import debug, lax
import petitRADTRANS.physical_constants as cst
from petitRADTRANS._input_data import find_input_file
from petitRADTRANS.config.configuration import (
    get_input_data_subpaths,
    petitradtrans_config_parser,
)
from petitRADTRANS.physics import wavelength2frequency, frequency2wavelength
from petitRADTRANS.radtrans_core.spectrum_utils import rebin_spectrum

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE


def _print_min_temperature_warning(_):
    debug.print(
        "Warning, input temperature is lower than minimum grid temperature. "
        "Taking F = F_grid(minimum grid temperature), normalized to desired input temperature."
    )
    return 0


def _print_max_temperature_warning(_):
    debug.print(
        "Warning, input temperature is higher than maximum grid temperature. "
        "Taking F = F_grid(maximum grid temperature), normalized to desired input temperature."
    )
    return 0


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

        assert self.log10_effective_temperature_grid is not None
        assert self.radius_grid is not None
        assert self.flux_grid is not None
        assert self.wavelength_grid is not None

        log10_effective_temperature_grid = jnp.asarray(
            self.log10_effective_temperature_grid, dtype=FEAUTRIER_JAX_DTYPE
        )
        radius_grid = jnp.asarray(self.radius_grid, dtype=FEAUTRIER_JAX_DTYPE)
        flux_grid = jnp.asarray(self.flux_grid, dtype=FEAUTRIER_JAX_DTYPE)
        wavelength_grid = jnp.asarray(self.wavelength_grid, dtype=FEAUTRIER_JAX_DTYPE)

        log_temp = jnp.log10(temperature)
        grid_size = log10_effective_temperature_grid.shape[0]
        interpolation_index = jnp.searchsorted(
            log10_effective_temperature_grid, log_temp
        )

        is_below_grid = interpolation_index == 0
        is_above_grid = interpolation_index == grid_size

        interpolation_index = jnp.clip(interpolation_index, 1, grid_size - 1)
        lower_index = interpolation_index - 1
        upper_index = interpolation_index

        lower_temperature = log10_effective_temperature_grid[lower_index]
        upper_temperature = log10_effective_temperature_grid[upper_index]
        weight_high = (log_temp - lower_temperature) / (
            upper_temperature - lower_temperature
        )
        weight_high = jnp.where(is_below_grid, 0.0, weight_high)
        weight_high = jnp.where(is_above_grid, 1.0, weight_high)
        weight_low = 1.0 - weight_high

        spec_dat_low = flux_grid[lower_index]
        spec_dat_high = flux_grid[upper_index]

        spec_dat = weight_low * spec_dat_low + weight_high * spec_dat_high
        radius = (
            weight_low * radius_grid[lower_index]
            + weight_high * radius_grid[upper_index]
        )

        lax.cond(is_below_grid, _print_min_temperature_warning, lambda _: 0, operand=0)
        lax.cond(is_above_grid, _print_max_temperature_warning, lambda _: 0, operand=0)

        frequency_grid = wavelength2frequency(wavelength_grid)
        flux = spec_dat
        norm = -jnp.sum((flux[1:] + flux[:-1]) * jnp.diff(frequency_grid)) / 2.0

        spec_dat = flux / norm * cst.sigma * temperature**4.0

        spec_dat = jnp.transpose(jnp.stack((wavelength_grid, spec_dat)))

        return spec_dat, radius

    @staticmethod
    def get_default_file(path_input_data=None):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        return os.path.join(
            path_input_data,
            get_input_data_subpaths()["stellar_spectra"],
            "phoenix",
            "phoenix.startable.petitRADTRANS.h5",
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
                search_online=search_online,
            )

        print(f"Loading PHOENIX star table in file '{file}'... ", end="")

        with h5py.File(file, "r") as f:
            self.log10_effective_temperature_grid = np.asarray(
                f["log10_effective_temperatures"],
                dtype=np.dtype(FEAUTRIER_JAX_DTYPE),
            )
            self.radius_grid = np.asarray(
                f["radii"], dtype=np.dtype(FEAUTRIER_JAX_DTYPE)
            )
            self.flux_grid = np.asarray(
                f["fluxes"], dtype=np.dtype(FEAUTRIER_JAX_DTYPE)
            )
            self.wavelength_grid = np.asarray(
                f["wavelengths"], dtype=np.dtype(FEAUTRIER_JAX_DTYPE)
            )

        self._loaded = True

        print("Done.")


_PHOENIX_STAR_TABLE = PhoenixStarTable()
_STAR_SPECTRUM_CACHE = {}


def _build_star_spectrum_cache_key(
    star_effective_temperature, orbit_semi_major_axis, frequencies, star_radius
):
    frequencies_array = np.asarray(frequencies)

    return (
        float(star_effective_temperature),
        float(orbit_semi_major_axis),
        None if star_radius is None else float(star_radius),
        str(frequencies_array.dtype),
        tuple(frequencies_array.shape),
        frequencies_array.tobytes(),
    )


def compute_star_spectrum(
    star_effective_temperature, orbit_semi_major_axis, frequencies, star_radius=None
):
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
    cache_key = None

    try:
        cache_key = _build_star_spectrum_cache_key(
            star_effective_temperature=star_effective_temperature,
            orbit_semi_major_axis=orbit_semi_major_axis,
            frequencies=frequencies,
            star_radius=star_radius,
        )
    except Exception:
        cache_key = None

    if cache_key is not None and cache_key in _STAR_SPECTRUM_CACHE:
        return _STAR_SPECTRUM_CACHE[cache_key]

    if star_radius is not None:
        star_spectrum, _ = _PHOENIX_STAR_TABLE.compute_spectrum(
            star_effective_temperature
        )
        _star_radius = star_radius
    else:
        star_spectrum, _star_radius = _PHOENIX_STAR_TABLE.compute_spectrum(
            star_effective_temperature
        )

    stellar_intensity = rebin_star_spectrum(
        star_spectrum=star_spectrum[:, 1],
        star_wavelengths=star_spectrum[:, 0],
        wavelengths=frequency2wavelength(frequencies),
    )

    stellar_intensity = (
        stellar_intensity / jnp.pi * (_star_radius / orbit_semi_major_axis) ** 2
    )

    if cache_key is not None:
        _STAR_SPECTRUM_CACHE[cache_key] = stellar_intensity

    return stellar_intensity


def rebin_star_spectrum(star_spectrum, star_wavelengths, wavelengths):
    add_stellar_flux = jnp.zeros(100, dtype=FEAUTRIER_JAX_DTYPE)
    add_wavelengths = jnp.asarray(
        jnp.logspace(jnp.log10(1.0000002e-02), 2, 100),
        dtype=FEAUTRIER_JAX_DTYPE,
    )

    wavelengths_interp = jnp.append(star_wavelengths, add_wavelengths)
    fluxes_interp = jnp.append(star_spectrum, add_stellar_flux)

    stellar_intensity = rebin_spectrum(wavelengths_interp, fluxes_interp, wavelengths)

    return stellar_intensity
