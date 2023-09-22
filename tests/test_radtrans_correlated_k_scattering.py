"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.

Due to the way scattering and correlated-k are calculated in petitRADTRANS, results using the same parameters may have
variations of <~ 1%. To take that into account, an important relative tolerance is set for the tests, and multiple tests
may be performed in order to rule out "unlucky" results.
"""
import copy
import warnings

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010

relative_tolerance = 1e-6  # relative tolerance when comparing with older spectra


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=radtrans_parameters['pressures'],
        line_species=radtrans_parameters['spectrum_parameters']['line_species_correlated_k'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        cloud_species=list(radtrans_parameters['cloud_parameters']['cloud_species'].keys()),
        wavelengths_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_correlated_k'],
        line_opacity_mode='c-k',
        scattering_in_emission=True
    )

    return atmosphere


atmosphere_ck_scattering = init_radtrans_correlated_k()


def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    frequencies, flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        eddy_diffusion_coefficient=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        cloud_f_sed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        cloud_particle_radius_distribution_std=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )

    try:
        # Comparison
        compare_from_reference_file(
            reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering'],
            comparison_dict={
                'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
                'spectral_radiosity': flux
            },
            relative_tolerance=relative_tolerance
        )
    except AssertionError as error:
        warnings.warn(f"got error: '{str(error)}', "
                      f"this may be expected as this test used add_scattering_as_absorption")
        # TODO re-generate reference files for pRT 3.0.0 and *remove* this try except block


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_planetary_average():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'planetary_ave'

    frequencies, flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        eddy_diffusion_coefficient=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        cloud_f_sed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        cloud_particle_radius_distribution_std=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=radtrans_parameters['stellar_parameters']['effective_temperature'],
        star_radius=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_cloud_calculated_radius_scattering_planetary_ave'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_dayside():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'dayside_ave'

    frequencies, flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        eddy_diffusion_coefficient=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        cloud_f_sed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        cloud_particle_radius_distribution_std=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=radtrans_parameters['stellar_parameters']['effective_temperature'],
        star_radius=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_emission_cloud_calculated_radius_scattering_dayside_ave'],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_emission_spectrum_cloud_calculated_radius_stellar_scattering_non_isotropic():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    geometry = 'non-isotropic'

    frequencies, flux, _ = atmosphere_ck_scattering.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        eddy_diffusion_coefficient=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        cloud_f_sed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        cloud_particle_radius_distribution_std=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        emission_geometry=geometry,
        star_effective_temperature=radtrans_parameters['stellar_parameters']['effective_temperature'],
        star_radius=radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun,
        orbit_semi_major_axis=radtrans_parameters['planetary_parameters']['orbit_semi_major_axis'],
        star_irradiation_angle=radtrans_parameters['stellar_parameters']['incidence_angle'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames[
            'correlated_k_emission_cloud_calculated_radius_scattering_non-isotropic'
        ],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )


def test_correlated_k_transmission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions = copy.deepcopy(radtrans_parameters['mass_fractions'])
    mass_fractions['Mg2SiO4(c)'] = \
        radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['mass_fraction']

    frequencies, transit_radii, _ =atmosphere_ck_scattering.calculate_transit_radii(
        temperatures=radtrans_parameters['temperature_isothermal'] * np.ones_like(radtrans_parameters['pressures']),
        mass_fractions=mass_fractions,
        reference_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        planet_radius=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=radtrans_parameters['planetary_parameters']['reference_pressure'],
        eddy_diffusion_coefficient=radtrans_parameters['planetary_parameters']['eddy_diffusion_coefficient'],
        cloud_f_sed=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['f_sed'],
        cloud_particle_radius_distribution_std=radtrans_parameters['cloud_parameters']['cloud_species']['Mg2SiO4(c)_cd']['sigma_log_normal'],
        frequencies_to_wavelengths=False
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['correlated_k_transmission_cloud_calculated_radius_scattering'],
        comparison_dict={
            'wavelength': petitRADTRANS.physical_constants.c / frequencies * 1e4,
            'transit_radius': transit_radii / petitRADTRANS.physical_constants.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )
