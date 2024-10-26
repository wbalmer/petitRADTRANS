import numpy as np
import jax.numpy as jnp
from numba import njit
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback
import petitRADTRANS.physical_constants as cst
import petitRADTRANS.fortran_src.fortran_core.math as fmath

"""
JAX Implementation of fortran_convolve and fortran_rebin
"""

def rebinning_interpolation():
    # TODO:
    return 

def rebin_spectrum():
    # TODO
    return

def rebin_spectrum_bin():
    # TODO
    return

@jit
def variable_width_convolution(input_wavelength, input_flux, resolutions):
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

    input_spectrum_length = input_flux.shape[0]

    # TODO probably need vmap or something to be more efficient
    for i in range(input_spectrum_length):
        #sigma is FWHM / 2.355
        sigma = (input_wavelength(i) / resolutions(i)) / 2.355
        gauss = jnp.exp(-(input_wavelength-input_wavelength(i))**2/(2*sigma**2))
        gauss = gauss/ jnp.sum(gauss)
        convolved_spectrum = convolved_spectrum.at[i].set(jnp.sum(gauss*input_flux))

    return convolved_spectrum