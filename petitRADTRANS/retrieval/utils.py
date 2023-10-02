"""
This module contains a set of useful functions that don't really fit anywhere
else. This includes flux conversions, prior functions, mean molecular weight
calculations, transforms from mass to number fractions, and fits file output.
"""
import os

import numpy as np
from scipy.special import erfcinv, gamma

SQRT2 = np.sqrt(2)


#################
# Prior Functions
#################
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
    return ((b ** a) / gamma(a)) * (1 / cube) ** (a + 1) * np.exp(-b / cube)


# Sanity checks on parameter ranges
def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.


def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.


########################
# File Formatting
########################
def fits_output(wavelength, spectrum, covariance, object_name, output_dir="",
                correlation=None):  # TODO arg object needs to be renamed, also is it used?
    """
    Generate a fits file that can be used as an input to a pRT retrieval.

    Args:
        wavelength : numpy.ndarray
            The wavelength bin centers in micron. dim(N)
        spectrum : numpy.ndarray
            The flux density in W/m2/micron at each wavelength bin. dim(N)
        covariance : numpy.ndarray
            The covariance of the flux in (W/m2/micron)^2 dim(N,N)
        object_name : string
            The name of the object, used for file naming.
        output_dir : string
            The parent directory of the output file.
        correlation : numpy.ndarray
            The correlation matrix of the flux points (See Brogi & Line 2018, https://arxiv.org/pdf/1811.01681.pdf)

    Returns:
        hdul : astropy.fits.HDUlist
            The HDUlist object storing the spectrum.
    """

    from astropy.io import fits
    primary_hdu = fits.PrimaryHDU([])
    primary_hdu.header['OBJECT'] = object_name
    c1 = fits.Column(name="WAVELENGTH", array=wavelength, format='D', unit="micron")
    c2 = fits.Column(name="FLUX", array=spectrum, format='D', unit="W/m2/micron")
    c3 = fits.Column(name="COVARIANCE", array=covariance, format=str(covariance.shape[0]) + 'D', unit="[W/m2/micron]^2")
    if correlation is not None:
        c4 = fits.Column(name="CORRELATION", array=correlation, format=str(correlation.shape[0]) + 'D', unit=" - ")
    columns = [c1, c2, c3, c4]
    table_hdu = fits.BinTableHDU.from_columns(columns, name='SPECTRUM')
    hdul = fits.HDUList([primary_hdu, table_hdu])
    outstring = os.path.join(output_dir, object_name + "_spectrum.fits")
    hdul.writeto(outstring, overwrite=True, checksum=True, output_verify='exception')
    return hdul
