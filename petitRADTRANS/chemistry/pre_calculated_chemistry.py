"""Manage equilibrium chemistry pre-calculated table.
"""
import copy as cp
import os

import h5py
import numpy as np

from petitRADTRANS.config import petitradtrans_config
from petitRADTRANS.fortran_chemistry import fortran_chemistry as fchem

path = petitradtrans_config['Paths']['pRT_input_data_path']


def __load_mass_fractions_chemical_table():
    chemical_table_file = f'{path}{os.path.sep}abundance_files{os.path.sep}mass_mixing_ratios.h5'

    print(f"Loading chemical equilibrium mass mixing ratio table from file '{chemical_table_file}'...")

    with h5py.File(chemical_table_file, 'r') as f:
        feh = f['iron_to_hydrogen_ratios'][()]
        co = f['carbon_to_oxygen_ratios'][()]
        temperatures = f['temperatures'][()]
        pressures_ = f['pressures'][()]

        species_names = f['species_names'][()]
        species_names = np.array([name.decode('utf-8') for name in species_names])

        mass_fractions = f['mass_mixing_ratios'][()]

    return feh, co, temperatures, pressures_, species_names, mass_fractions


# Read in parameters of chemistry grid
table_log10_metallicities, table_co_ratios, table_temperatures, table_pressures, table_species, \
    mass_fractions_chemical_table = __load_mass_fractions_chemical_table()

# Change the order to column-wise (Fortran) to increase interpolation speed
mass_fractions_chemical_table = np.array(mass_fractions_chemical_table, order='F')


def interpolate_mass_fractions_chemical_table(co_ratios, log10_metallicities, temperatures, pressures,
                                              carbon_pressure_quench=None):
    """
    Interpol abundances to desired coordinates.
    """

    co_ratios_goal, fehs_goal, temps_goal, pressures_goal = \
        cp.copy(co_ratios), cp.copy(log10_metallicities), cp.copy(temperatures), cp.copy(pressures)

    co_ratios_goal, fehs_goal, temps_goal, pressures_goal = \
        np.array(co_ratios_goal).reshape(-1), \
        np.array(fehs_goal).reshape(-1), \
        np.array(temps_goal).reshape(-1), \
        np.array(pressures_goal).reshape(-1)

    # Apply boundary treatment
    co_ratios_goal[co_ratios_goal <= np.min(table_co_ratios)] = np.min(table_co_ratios) + 1e-6
    co_ratios_goal[co_ratios_goal >= np.max(table_co_ratios)] = np.max(table_co_ratios) - 1e-6

    fehs_goal[fehs_goal <= np.min(table_log10_metallicities)] = np.min(table_log10_metallicities) + 1e-6
    fehs_goal[fehs_goal >= np.max(table_log10_metallicities)] = np.max(table_log10_metallicities) - 1e-6

    temps_goal[temps_goal <= np.min(table_temperatures)] = np.min(table_temperatures) + 1e-6
    temps_goal[temps_goal >= np.max(table_temperatures)] = np.max(table_temperatures) - 1e-6

    pressures_goal[pressures_goal <= np.min(table_pressures)] = (
        np.min(table_pressures) + 1e-6)
    pressures_goal[pressures_goal >= np.max(table_pressures)] = (
        np.max(table_pressures) - 1e-6)

    # Get interpolation indices
    co_ratios_large_int = np.searchsorted(table_co_ratios, co_ratios_goal) + 1
    fehs_large_int = np.searchsorted(table_log10_metallicities, fehs_goal) + 1
    temps_large_int = np.searchsorted(table_temperatures, temps_goal) + 1
    pressures_large_int = np.searchsorted(table_pressures, pressures_goal) + 1

    # Get the interpolated values from Fortran routine
    # TODO nabla ad and MMW are not abundances, they should be returned as extra parameters
    abundances_arr = fchem.interpolate_mass_fractions_table(
        co_ratios_goal, fehs_goal, temps_goal,
        pressures_goal, co_ratios_large_int,
        fehs_large_int, temps_large_int,
        pressures_large_int, table_log10_metallicities, table_co_ratios,
        table_pressures, table_temperatures, mass_fractions_chemical_table
    )

    # Sort in output format of this function
    abundances = {}
    for id_, name in enumerate(table_species):
        abundances[name] = abundances_arr[id_, :]

    # Carbon quenching? Assumes pressures_goal is sorted in ascending order
    if carbon_pressure_quench is not None:
        if carbon_pressure_quench > np.min(pressures_goal):

            q_index = min(np.searchsorted(pressures_goal, carbon_pressure_quench),
                          int(len(pressures_goal)) - 1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < carbon_pressure_quench] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < carbon_pressure_quench] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < carbon_pressure_quench] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    return abundances
