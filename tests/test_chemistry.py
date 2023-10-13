"""Test petitRADTRANS chemistry module.
"""
import numpy as np
from .context import petitRADTRANS

from .utils import compare_from_reference_file, radtrans_parameters, reference_filenames, temperature_guillot_2010


relative_tolerance = 1e-6


def test_chemistry_atmosphere():
    c_o_ratios = radtrans_parameters['chemical_parameters']['c_o_ratios'][1] \
        * np.ones_like(radtrans_parameters['pressures'])
    metallicities = radtrans_parameters['chemical_parameters']['metallicities'][1] \
        * np.ones_like(radtrans_parameters['pressures'])

    mass_fractions, mean_molar_masses, nabla_adiabatic = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperature_guillot_2010,
            pressures=radtrans_parameters['pressures'],
            full=True
        )
    )

    # TODO pRT2 reference files compatibility, remove before pRT 3 update
    mass_fractions['C2H2,acetylene'] = mass_fractions['C2H2']
    del mass_fractions['C2H2']
    mass_fractions['MMW'] = mean_molar_masses
    mass_fractions['nabla_ad'] = nabla_adiabatic

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['mass_fractions_atmosphere'],
        comparison_dict=mass_fractions,
        relative_tolerance=relative_tolerance
    )


def test_chemistry_atmosphere_quench():
    c_o_ratios = radtrans_parameters['chemical_parameters']['c_o_ratios'][1] \
        * np.ones_like(radtrans_parameters['pressures'])
    metallicities = radtrans_parameters['chemical_parameters']['metallicities'][1] \
        * np.ones_like(radtrans_parameters['pressures'])

    mass_fractions, mean_molar_masses, nabla_adiabatic = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperature_guillot_2010,
            pressures=radtrans_parameters['pressures'],
            carbon_pressure_quench=radtrans_parameters['chemical_parameters']['pressure_quench_carbon'],
            full=True
        )
    )

    # TODO pRT2 reference files compatibility, remove before pRT 3 update
    mass_fractions['C2H2,acetylene'] = mass_fractions['C2H2']
    del mass_fractions['C2H2']
    mass_fractions['MMW'] = mean_molar_masses
    mass_fractions['nabla_ad'] = nabla_adiabatic

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['mass_fractions_atmosphere_quench'],
        comparison_dict=mass_fractions,
        relative_tolerance=relative_tolerance
    )


def test_chemistry_c_o_ratios():
    c_o_ratios = np.asarray(radtrans_parameters['chemical_parameters']['c_o_ratios'])
    metallicities = radtrans_parameters['chemical_parameters']['metallicities'][1] * np.ones_like(c_o_ratios)
    pressures = radtrans_parameters['chemical_parameters']['pressure'] * np.ones_like(c_o_ratios)
    temperatures = radtrans_parameters['chemical_parameters']['temperature'] * np.ones_like(c_o_ratios)

    mass_fractions, mean_molar_masses, nabla_adiabatic = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperatures,
            pressures=pressures,
            full=True
        )
    )

    # TODO pRT2 reference files compatibility, remove before pRT 3 update
    mass_fractions['C2H2,acetylene'] = mass_fractions['C2H2']
    del mass_fractions['C2H2']
    mass_fractions['MMW'] = mean_molar_masses
    mass_fractions['nabla_ad'] = nabla_adiabatic

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['mass_fractions_c_o_ratios'],
        comparison_dict=mass_fractions,
        relative_tolerance=relative_tolerance
    )
    

def test_chemistry_metallicities():
    metallicities = np.asarray(radtrans_parameters['chemical_parameters']['metallicities'])
    c_o_ratios = radtrans_parameters['chemical_parameters']['c_o_ratios'][1] * np.ones_like(metallicities)
    pressures = radtrans_parameters['chemical_parameters']['pressure'] * np.ones_like(metallicities)
    temperatures = radtrans_parameters['chemical_parameters']['temperature'] * np.ones_like(metallicities)

    mass_fractions, mean_molar_masses, nabla_adiabatic = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperatures,
            pressures=pressures,
            full=True
        )
    )

    # TODO pRT2 reference files compatibility, remove before pRT 3 update
    mass_fractions['C2H2,acetylene'] = mass_fractions['C2H2']
    del mass_fractions['C2H2']
    mass_fractions['MMW'] = mean_molar_masses
    mass_fractions['nabla_ad'] = nabla_adiabatic

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['mass_fractions_metallicities'],
        comparison_dict=mass_fractions,
        relative_tolerance=relative_tolerance
    )
