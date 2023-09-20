"""
This module contains a set of useful functions that don't really fit anywhere
else. This includes flux conversions, prior functions, mean molecular weight
calculations, transforms from mass to number fractions, and fits file output.
"""
import copy
import os
import sys
from typing import Tuple

import numpy as np
from scipy.special import erfcinv

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.prt_molmass import get_species_molar_mass

SQRT2 = np.sqrt(2)


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
    return cst.c / frequency / 1e-4


def fnu_to_flambda(wlen, spectrum):
    f_lambda = spectrum * cst.c / wlen ** 2.
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7
    return f_lambda


def spectrum_cgs_to_si(frequency, spectrum):
    wlen = freq_to_micron(frequency)
    f_lambda = fnu_to_flambda(wlen * 1e-4, spectrum)
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
        mmw += abundances[key] / get_species_molar_mass(spec)

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
        mass += value * get_species_molar_mass(spec)
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
        n_frac[key] = value / get_species_molar_mass(spec) * mmw
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
        m_frac[key] = value * get_species_molar_mass(spec) / mmw
    return m_frac


def teff_calc(waves, model, dist=1.0, r_pl=1.0):
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

    def integ(w, m):
        return np.sum(m[:-1] * ((dist / r_pl) ** 2.) * (u.W / u.m ** 2 / u.micron) * np.diff(w) * u.micron)

    energy = integ(waves, model)
    summed = energy / c.sigma_sb

    return summed.value ** 0.25


def rebin_ck_line_opacities(radtrans, resolution, path='', species=None, species_molar_masses=None):
    import exo_k
    import h5py

    if species is None:
        species = []

    # Define own wavenumber grid, make sure that log spacing is constant everywhere
    n_spectral_points = int(
        resolution * np.log(radtrans.wavelengths_boundaries[1] / radtrans.wavelengths_boundaries[0]) + 1
    )
    wavenumber_grid = np.logspace(np.log10(1 / radtrans.wavelengths_boundaries[1] / 1e-4),
                                  np.log10(1. / radtrans.wavelengths_boundaries[0] / 1e-4),
                                  n_spectral_points)
    string_type = h5py.string_dtype(encoding='utf-8')

    # Do the rebinning, loop through species
    for s in species:
        print(f"Rebinning species {s}...")

        # Create hdf5 file that Exo-k can read...
        with h5py.File('temp.h5', 'w') as f:
            try:
                f.create_dataset('DOI', (1,), data="--", dtype=string_type)
            except ValueError:  # TODO check if ValueError is expected here (and why this try is needed at all)
                f.create_dataset('DOI', data=['--'])

            f.create_dataset('bin_centers', data=radtrans.frequencies[::-1] / cst.c)
            f.create_dataset('bin_edges', data=radtrans.frequency_bins_edges[::-1] / cst.c)
            opacity_grid = copy.copy(radtrans.lines_loaded_opacities['opacity_grid'][s])

            # Mass to go from opacities to cross-sections
            opacity_grid = opacity_grid * cst.amu * species_molar_masses[s.split('_')[0]]

            # Do the opposite of what I do when loading in Katy's ExoMol tables
            # To get opacities into the right format
            opacity_grid = opacity_grid[:, ::-1, :]
            opacity_grid = np.swapaxes(opacity_grid, 2, 0)
            opacity_grid = opacity_grid.reshape((
                radtrans.lines_loaded_opacities['temperature_grid_size'][s],
                radtrans.lines_loaded_opacities['pressure_grid_size'][s],
                radtrans.frequencies.size,
                len(radtrans.lines_loaded_opacities['weights_gauss'])
            ))
            opacity_grid = np.swapaxes(opacity_grid, 1, 0)
            opacity_grid[opacity_grid < 1e-60] = 1e-60

            f.create_dataset('kcoeff', data=opacity_grid)
            f['kcoeff'].attrs.create('units', 'cm^2/molecule')

            # Add the other required information
            try:
                f.create_dataset('method', (1,), data="petit_samples", dtype=string_type)
            except ValueError:  # TODO check if ValueError is expected here (and why this try is needed at all)
                f.create_dataset('method', data=['petit_samples'])

            f.create_dataset('mol_name', data=s.split('_')[0], dtype=string_type)
            f.create_dataset('mol_mass', data=[species_molar_masses[s.split('_')[0]]])
            f.create_dataset('ngauss', data=len(radtrans.lines_loaded_opacities['weights_gauss']))
            f.create_dataset('p', data=radtrans.lines_loaded_opacities['temperature_profile_grid'][s][
                                       :radtrans.lines_loaded_opacities['pressure_grid_size'][s], 1] / 1e6)
            f['p'].attrs.create('units', 'bar')
            f.create_dataset('samples', data=radtrans.lines_loaded_opacities['g_gauss'])
            f.create_dataset('t', data=radtrans.lines_loaded_opacities['temperature_profile_grid'][s][
                                       ::radtrans.lines_loaded_opacities['pressure_grid_size'][s], 0])
            f.create_dataset('weights', data=radtrans.lines_loaded_opacities['weights_gauss'])
            f.create_dataset('wlrange', data=[np.min(cst.c / radtrans.frequency_bins_edges / 1e-4),
                                              np.max(cst.c / radtrans.frequency_bins_edges / 1e-4)])
            f.create_dataset('wnrange', data=[np.min(radtrans.frequency_bins_edges / cst.c),
                                              np.max(radtrans.frequency_bins_edges / cst.c)])

        # Use Exo-k to rebin to low-res, save to desired folder
        tab = exo_k.Ktable(filename='temp.h5')
        tab.bin_down(wavenumber_grid)

        if path[-1] == '/':
            path = path[:-1]

        os.makedirs(path + '/' + s + '_R_' + str(int(resolution)), exist_ok=True)
        tab.write_hdf5(path + '/' + s + '_R_' + str(int(resolution)) + '/' + s + '_R_' + str(
            int(resolution)) + '.h5')
        os.system('rm temp.h5')


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
    from petitRADTRANS.radtrans import Radtrans
    from petitRADTRANS.config import petitradtrans_config_parser

    prt_path = petitradtrans_config_parser.get_input_data_path()
    atmosphere = Radtrans(
        line_species=species,
        wavelengths_boundaries=[0.1, 251.]
    )
    ck_path = os.path.join(prt_path, 'opacities/lines/corr_k/')

    print("Saving to " + ck_path)
    print("Resolution: ", resolution)

    masses = {}

    for spec in species:
        masses[spec.split('_')[0]] = get_species_molar_mass(spec)

    rebin_ck_line_opacities(
        radtrans=atmosphere,
        resolution=int(resolution),
        path=ck_path,
        species=species,
        species_molar_masses=masses
    )


def compute_gravity(parameters):
    if 'log_g' in parameters.keys() and 'mass' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        R_pl = np.sqrt(cst.G * parameters['mass'].value / gravity)
    elif 'log_g' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        R_pl = parameters['R_pl'].value
    elif 'mass' in parameters.keys():
        R_pl = parameters['R_pl'].value
        gravity = cst.G * parameters['mass'].value / R_pl ** 2
    else:
        print("Pick two of log_g, R_pl and mass priors!")
        sys.exit(5)

    return gravity, R_pl


def fixed_length_amr(p_clouds, pressures, scaling=10, width=3):
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
def fits_output(wavelength, spectrum, covariance, object, output_dir="",
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
