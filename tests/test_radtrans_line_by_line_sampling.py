"""Test the radtrans module in line-by-line mode with sampling.

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_downsampled_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=radtrans_parameters['pressures'],
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        wavelengths_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        lbl_opacity_sampling=radtrans_parameters['spectrum_parameters']['line_by_line_opacity_sampling'],
        line_opacity_mode='lbl'
    )

    return atmosphere


atmosphere_lbl_downsampled = init_radtrans_downsampled_line_by_line()


def test_line_by_line_downsampled_emission_spectrum():
    # Calculate an emission spectrum
    frequencies, flux = atmosphere_lbl_downsampled.calculate_flux(
        temperatures=temperature_guillot_2010,
        mass_fractions=radtrans_parameters['mass_fractions'],
        surface_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_downsampled_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / frequencies * 1e4,
            'spectral_radiosity': flux
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_downsampled_transmission_spectrum():
    # Calculate a transmission spectrum
    frequencies, transit_radii = atmosphere_lbl_downsampled.calculate_transit_radii(
        temperatures=temperature_isothermal,
        mass_fractions=radtrans_parameters['mass_fractions'],
        surface_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=radtrans_parameters['mean_molar_mass'],
        planet_radius=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        reference_pressure=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_downsampled_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / frequencies * 1e4,
            'transit_radius': transit_radii / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )
