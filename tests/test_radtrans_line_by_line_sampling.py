"""Test the radtrans module in line-by-line mode with sampling.

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_downsampled_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters['pressures'],
        line_species=test_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=test_parameters['spectrum_parameters']['rayleigh_species'],
        gas_continuum_contributors=test_parameters['spectrum_parameters']['continuum_opacities'],
        wavelength_boundaries=test_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        line_by_line_opacity_sampling=test_parameters['spectrum_parameters']['line_by_line_opacity_sampling'],
        line_opacity_mode='lbl'
    )

    return atmosphere


atmosphere_lbl_downsampled = init_radtrans_downsampled_line_by_line()


def test_line_by_line_downsampled_emission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl_downsampled.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters['mass_fractions'],
        reference_gravity=test_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        frequencies_to_wavelengths=False
    )


def test_line_by_line_downsampled_transmission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl_downsampled.calculate_transit_radii,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters['mass_fractions'],
        reference_gravity=test_parameters['planetary_parameters']['surface_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        planet_radius=test_parameters['planetary_parameters']['radius']
                      * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters['planetary_parameters']['reference_pressure'],
        frequencies_to_wavelengths=False
    )
