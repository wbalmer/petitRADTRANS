"""Test petitRADTRANS chemistry module.
"""
import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010

relative_tolerance = 1e-6


def test_chemistry_atmosphere():
    c_o_ratios = (
        test_parameters['chemical_parameters']['c_o_ratios'][1]
        * np.ones_like(test_parameters['pressures'])
    )
    metallicities = (
        test_parameters['chemical_parameters']['metallicities'][1]
        * np.ones_like(test_parameters['pressures'])
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters['pressures'],
        full=True
    )


def test_chemistry_atmosphere_quench():
    c_o_ratios = test_parameters['chemical_parameters']['c_o_ratios'][1] \
                 * np.ones_like(test_parameters['pressures'])
    metallicities = test_parameters['chemical_parameters']['metallicities'][1] \
                    * np.ones_like(test_parameters['pressures'])

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters['pressures'],
        carbon_pressure_quench=test_parameters['chemical_parameters']['pressure_quench_carbon'],
        full=True
    )


def test_chemistry_c_o_ratios():
    c_o_ratios = np.asarray(test_parameters['chemical_parameters']['c_o_ratios'])
    metallicities = test_parameters['chemical_parameters']['metallicities'][1] * np.ones_like(c_o_ratios)
    pressures = test_parameters['chemical_parameters']['pressure'] * np.ones_like(c_o_ratios)
    temperatures = test_parameters['chemical_parameters']['temperature'] * np.ones_like(c_o_ratios)

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True
    )


def test_chemistry_metallicities():
    metallicities = np.asarray(test_parameters['chemical_parameters']['metallicities'])
    c_o_ratios = test_parameters['chemical_parameters']['c_o_ratios'][1] * np.ones_like(metallicities)
    pressures = test_parameters['chemical_parameters']['pressure'] * np.ones_like(metallicities)
    temperatures = test_parameters['chemical_parameters']['temperature'] * np.ones_like(metallicities)

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True
    )


def test_mass_fractions2metallicities():
    c_o_ratios = (
        test_parameters['chemical_parameters']['c_o_ratios'][1]
        * np.ones_like(test_parameters['pressures'])
    )
    metallicities = (
        test_parameters['chemical_parameters']['metallicities'][1]
        * np.ones_like(test_parameters['pressures'])
    )

    mass_fractions = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperature_guillot_2010,
            pressures=test_parameters['pressures'],
            full=False
        )
    )

    sum_mass_fractions = np.sum(list(mass_fractions.values()), axis=0)

    mass_fractions = {species: mass_fraction / sum_mass_fractions for species, mass_fraction in mass_fractions.items()}

    mean_molar_masses = petitRADTRANS.chemistry.utils.compute_mean_molar_masses(
        abundances=mass_fractions
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.utils.mass_fractions2metallicity,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses
    )
