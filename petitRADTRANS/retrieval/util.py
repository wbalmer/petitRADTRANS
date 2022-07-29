"""
This module contains a set of useful functions that don't really fit anywhere
else. This includes flux conversions, prior functions, mean molecular weight
calculations, transforms from mass to number fractions, and fits file output.
"""
import sys, os

# To not have numpy start parallelizing on its own
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.special import erfcinv
import numpy as np
import math as math
from molmass import Formula
from typing import Tuple
from petitRADTRANS import nat_cst as nc

# import threading, subprocess


SQRT2 = math.sqrt(2.)


#################
# Flux scaling
#################
def surf_to_meas(flux, p_rad, dist):
    """
    surf_to_meas
    Convert from emission flux to measured flux at earth
    Args:
        flux : numpy.ndarray
            Absolute flux value or spectrum as emitted by a source of radius p_rad
        p_rad : float
            Planet radius, in same units as dist
        dist : float
            Distance to the object, in the same units as p_rad
    Returns:
        m_flux : numpy.ndarray
            Apparent flux
    """

    m_flux = flux * p_rad ** 2 / dist ** 2
    return m_flux

def freq_to_micron(frequency):
    return nc.c/frequency/1e-4

def fnu_to_flambda(wlen,spectrum):
    f_lambda = spectrum*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    #f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    #f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7
    return f_lambda

def spectrum_cgs_to_si(frequency,spectrum):
    wlen = freq_to_micron(frequency)
    f_lambda = fnu_to_flambda(wlen*1e-4, spectrum)
    return wlen, f_lambda

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
# Mean Molecular Weights
########################

def getMM(species):
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to
    calculate the mass number for the standard
    isotope of an input species. If all_iso
    is part of the input, it will return the
    mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    e_molar_mass = 5.4857990888e-4  # (g.mol-1) e- molar mass (source: NIST CODATA)

    if species == 'e-':
        return e_molar_mass
    elif species == 'H-':
        return Formula('H').mass + e_molar_mass

    name = species.split("_")[0]
    name = name.split(',')[0]
    f = Formula(name)

    if "all_iso" in species:
        return f.mass

    return f.isotope.massnumber


def calc_MMW(abundances):
    """
    calc_MMW
    Calculate the mean molecular weight in each layer.

    Args:
        abundances : dict
            dictionary of abundance arrays, each array must have the shape of the pressure array used in pRT,
            and contain the abundance at each layer in the atmosphere.
    """
    mmw = sys.float_info.min  # prevent division by 0

    for key in abundances.keys():
        # exo_k resolution
        spec = key.split("_R_")[0]
        mmw += abundances[key] / getMM(spec)

    return 1.0 / mmw


def get_MMW_from_mfrac(m_frac):
    """
    wraps calc_MMW
    """

    return calc_MMW(m_frac)

def get_MMW_from_nfrac(n_frac):
    """
    Calculate the mean molecular weight from a number fraction

    Args:
        n_frac : dict
            A dictionary of number fractions
    """

    mass = 0.0
    for key, value in n_frac.items():
        spec = key.split("_R_")[0]
        mass += value * getMM(spec)
    return mass


def mass_to_number(m_frac):
    """
    Convert mass fractions to number fractions

    Args:
        m_frac : dict
            A dictionary of mass fractions
    """

    n_frac = {}
    mmw = get_MMW_from_mfrac(m_frac)

    for key, value in m_frac.items():
        spec = key.split("_R_")[0]
        n_frac[key] = value / getMM(spec) * mmw
    return n_frac


def number_to_mass(n_fracs):
    """
    Convert number fractions to mass fractions

    Args:
        n_fracs : dict
            A dictionary of number fractions
    """

    m_frac = {}
    mmw = get_MMW_from_nfrac(n_fracs)

    for key, value in n_fracs.items():
        spec = key.split("_R_")[0]
        m_frac[key] = value * getMM(spec) / mmw
    return m_frac

def teff_calc(waves,model,dist=1.0,r_pl=1.0):
    """
    This function takes in the wavelengths and flux of a model
    in units of W/m2/micron and calculates the effective temperature
    by integrating the model and using the stefan boltzmann law.
    Args:
        waves : numpy.ndarray
            Wavelength grid in units of micron
        model : numpy.ndarray
            Flux density grid in units of W/m2/micron
        dist : Optional(float)
            Distance to the object. Must have same units as r_pl
        r_pl : Optional(float)
            Object radius. Must have same units as dist
    """
    import astropy.units as u
    import astropy.constants as c
    def integ(waves,model):
        return np.sum(model[:-1]*((dist/r_pl)**2.)*(u.W/u.m**2/u.micron)* np.diff(waves)*u.micron)

    energy = integ(waves,model)
    #print(energy)
    summed = ((energy /c.sigma_sb))
    #print(summed)
    return (summed.value)**0.25


def bin_species_exok(species, resolution):
    """
    This function uses exo-k to bin the c-k table of a
    single species to a desired (lower) spectral resolution.

    Args:
        species : string
            The name of the species
        resolution : int
            The desired spectral resolving power.
    """
    from petitRADTRANS import Radtrans
    prt_path = os.environ.get("pRT_input_data_path")
    atmosphere = Radtrans(line_species = species,
                            wlen_bords_micron = [0.1, 251.])
    ck_path = prt_path + '/opacities/lines/corr_k/'
    print("Saving to " + ck_path)
    print("Resolution: ", resolution)

    masses = {}

    for spec in species:
        masses[spec.split('_')[0]] = getMM(spec)

    atmosphere.write_out_rebin(
        resolution=int(resolution),
        path=ck_path,
        species=species,
        masses=masses
    )

def compute_gravity(parameters):
    gravity = -np.inf
    R_pl = -np.inf
    if 'log_g' in parameters.keys() and 'mass' in parameters.keys():
        gravity = 10**parameters['log_g'].value
        R_pl = np.sqrt(nc.G*parameters['mass'].value/gravity)
    elif 'log_g' in parameters.keys():
        gravity= 10**parameters['log_g'].value
        R_pl = parameters['R_pl'].value
    elif 'mass' in parameters.keys():
        R_pl = parameters['R_pl'].value
        gravity = nc.G * parameters['mass'].value/R_pl**2
    else:
        print("Pick two of log_g, R_pl and mass priors!")
        sys.exit(5)
    return gravity, R_pl

def set_resolution(lines,abundances,resolution):
    """
    deprecated
    """
    # Set correct key names in abundances for pRT, with set resolution
    # Only needed for free chemistry retrieval
    #print(lines)
    #print(abundances)
    if resolution is None:
        return abundances
    for line in lines:
        abundances[line] = abundances[line.split("_R_"+str(resolution))[0]]
        del abundances[line.split("_R_"+str(resolution))]
    return abundances


def fixed_length_amr(p_clouds, pressures, scaling = 10, width = 3):
    r"""This function takes in the cloud base pressures for each cloud,
    and returns an array of pressures with a high resolution mesh
    in the region where the clouds are located.

    Author:  Francois Rozet.

    The output length is always
        len(pressures[::scaling]) + len(p_clouds) * width * (scaling - 1)

    Args:
        P_clouds : numpy.ndarray
            The cloud base pressures in bar
        press : np.ndarray
            The high resolution pressure array.
        scaling : int
            The factor by which the low resolution pressure array is scaled
        width : int
            The number of low resolution bins to be replaced for each cloud layer.
    """

    length = len(pressures)
    cloud_indices = np.searchsorted(pressures, np.asarray(p_clouds))

    # High resolution intervals
    def bounds(center: int, width: int) -> Tuple[int, int]:
        upper = min(center + width // 2, length)
        lower = max(upper - width, 0)
        return lower, lower + width

    intervals = [bounds(idx, scaling * width) for idx in cloud_indices]

    # Merge intervals
    while True:
        intervals, stack = sorted(intervals), []

        for interval in intervals:
            if stack and stack[-1][1] >= interval[0]:
                last = stack.pop()
                interval = bounds(
                    (last[0] + max(last[1], interval[1]) + 1) // 2,
                    last[1] - last[0] + interval[1] - interval[0],
                )

            stack.append(interval)

        if len(intervals) == len(stack):
            break
        intervals = stack

    # Intervals to indices
    indices = [np.arange(0, length, scaling)]

    for interval in intervals:
        indices.append(np.arange(*interval))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices

########################
# File Formatting
########################
def fits_output(wavelength, spectrum, covariance, object, output_dir="", correlation=None):  # TODO arg object needs to be renamed
    """
    Generate a fits file that can be used as an input to a pRT retrieval.

    Args:
        wavelength : numpy.ndarray
            The wavelength bin centers in micron. dim(N)
        spectrum : numpy.ndarray
            The flux density in W/m2/micron at each wavelength bin. dim(N)
        covariance : numpy.ndarray
            The covariance of the flux in (W/m2/micron)^2 dim(N,N)
        object : string
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
    primary_hdu.header['OBJECT'] = object
    c1 = fits.Column(name="WAVELENGTH", array=wavelength, format='D', unit="micron")
    c2 = fits.Column(name="FLUX", array=spectrum, format='D', unit="W/m2/micron")
    c3 = fits.Column(name="COVARIANCE", array=covariance, format=str(covariance.shape[0]) + 'D', unit="[W/m2/micron]^2")
    if correlation is not None:
        c4 = fits.Column(name="CORRELATION", array=correlation, format=str(correlation.shape[0]) + 'D', unit=" - ")
    columns = [c1, c2, c3, c4]
    table_hdu = fits.BinTableHDU.from_columns(columns, name='SPECTRUM')
    hdul = fits.HDUList([primary_hdu, table_hdu])
    outstring = os.path.join(output_dir, object + "_spectrum.fits")
    hdul.writeto(outstring, overwrite=True, checksum=True, output_verify='exception')
    return hdul
