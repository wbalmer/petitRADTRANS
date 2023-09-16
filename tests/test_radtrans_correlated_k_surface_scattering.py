"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.

Due to the way scattering and correlated-k are calculated in petitRADTRANS, results using the same parameters may have
variations of <~ 1%. To take that into account, an important relative tolerance is set for the tests, and multiple tests
may be performed in order to rule out "unlucky" results.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010

relative_tolerance = 1e-6  # relative tolerance when comparing with older spectra


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=radtrans_parameters['pressures_thin_atmosphere'],
        line_species=radtrans_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(radtrans_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelengths_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        line_opacity_mode='c-k',
        scattering_in_emission=True
    )

    return atmosphere


atmosphere_ck_surface_scattering = init_radtrans_correlated_k()


def test_correlated_k_emission_spectrum_surface_scattering():
    # Copy atmosphere so that change in reflectance is not carried outside the function
    atmosphere = copy.deepcopy(atmosphere_ck_surface_scattering)

    frequencies, flux = atmosphere.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=radtrans_parameters['mass_fractions'],
        surface_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        emission_geometry='non-isotropic',
        star_effective_temperature=radtrans_parameters['stellar_parameters']['effective_temperature'],
        star_radius=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=radtrans_parameters['stellar_parameters']['incidence_angle'],
        reflectances=radtrans_parameters['planetary_parameters']['surface_reflectance']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_surface_scattering'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )
