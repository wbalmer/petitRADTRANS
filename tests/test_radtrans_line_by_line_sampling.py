"""Test the radtrans module in line-by-line mode with sampling.

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from petitRADTRANS.plotlib.plotlib import plot_radtrans_opacities
from .utils import test_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-4  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_downsampled_line_by_line(
    retain_unstacked_line_opacities_for_plotting=False,
):
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters["pressures"],
        line_species=test_parameters["spectrum_parameters"][
            "line_species_line_by_line"
        ],
        rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        wavelength_boundaries=test_parameters["spectrum_parameters"][
            "wavelength_range_line_by_line"
        ],
        line_by_line_opacity_sampling=test_parameters["spectrum_parameters"][
            "line_by_line_opacity_sampling"
        ],
        line_opacity_mode="lbl",
        retain_unstacked_line_opacities_for_plotting=retain_unstacked_line_opacities_for_plotting,
        scattering_in_emission=False,
    )

    return atmosphere


atmosphere_lbl_downsampled = init_radtrans_downsampled_line_by_line()


def test_line_by_line_downsampled_emission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl_downsampled.calculate_flux,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters["mass_fractions_line_by_line"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        frequencies_to_wavelengths=False,
    )


def test_line_by_line_downsampled_transmission_spectrum():
    benchmark = Benchmark(
        function=atmosphere_lbl_downsampled.calculate_transit_radii,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_line_by_line"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        frequencies_to_wavelengths=False,
    )


def test_line_opacity_debug_data_not_retained_by_default():
    assert len(atmosphere_lbl_downsampled.lines_loaded_opacities["opacity_grid"]) == 0
    assert (
        len(
            atmosphere_lbl_downsampled.lines_loaded_opacities[
                "temperature_pressure_grid"
            ]
        )
        == 0
    )
    assert atmosphere_lbl_downsampled.lines_loaded_opacities[
        "opacity_grid_stacked"
    ].shape[0] == len(atmosphere_lbl_downsampled.line_species)


def test_plot_radtrans_opacities_works_without_debug_layout():
    species = atmosphere_lbl_downsampled.line_species[0]
    opacities = plot_radtrans_opacities(
        radtrans=atmosphere_lbl_downsampled,
        species=[species],
        temperature=float(temperature_isothermal[0]),
        pressure_bar=float(test_parameters["pressures"][0]),
        return_opacities=True,
    )

    wavelengths, species_opacities = opacities[species]

    assert species_opacities.shape[0] == wavelengths.shape[0]
    assert species_opacities.shape[1] == 1
    assert np.all(np.isfinite(species_opacities))


def test_line_opacity_debug_data_retained_on_request():
    atmosphere = init_radtrans_downsampled_line_by_line(
        retain_unstacked_line_opacities_for_plotting=True,
    )

    assert set(atmosphere.lines_loaded_opacities["opacity_grid"]) == set(
        atmosphere.line_species
    )
    assert set(atmosphere.lines_loaded_opacities["temperature_pressure_grid"]) == set(
        atmosphere.line_species
    )
