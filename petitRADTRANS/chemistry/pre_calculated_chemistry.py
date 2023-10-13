"""Manage equilibrium chemistry pre-calculated table.
"""
import copy
import copy as cp
import os

import h5py
import numpy as np

from petitRADTRANS.config.configuration import petitradtrans_config_parser, get_input_data_subpaths
from petitRADTRANS.fortran_chemistry import fortran_chemistry as fchem
from petitRADTRANS._input_data_loader import get_input_data_file_not_found_error_message


class PreCalculatedEquilibriumChemistryTable:
    def __init__(self):
        self.log10_metallicities = None
        self.co_ratios = None
        self.temperatures = None
        self.pressures = None
        self.species = None
        self.mass_fractions = None
        self.nabla_adiabatic = None
        self.mean_molar_masses = None
        self._loaded = False

    @staticmethod
    def get_default_input_data_path():
        return os.path.join(
            petitradtrans_config_parser.get_input_data_path(),
            get_input_data_subpaths()["pre_calculated_chemistry"],
            "equilibrium_chemistry"
        )

    def interpolate_mass_fractions(self, co_ratios, log10_metallicities, temperatures, pressures,
                                   carbon_pressure_quench=None, full=False):
        """
        Interpol abundances to desired coordinates.
        """
        if not self._loaded:
            self.load()

        co_ratios_goal, fehs_goal, temps_goal, pressures_goal = \
            cp.copy(co_ratios), cp.copy(log10_metallicities), cp.copy(temperatures), cp.copy(pressures)

        co_ratios_goal, fehs_goal, temps_goal, pressures_goal = \
            np.array(co_ratios_goal).reshape(-1), \
            np.array(fehs_goal).reshape(-1), \
            np.array(temps_goal).reshape(-1), \
            np.array(pressures_goal).reshape(-1)

        # Apply boundary treatment
        co_ratios_goal[co_ratios_goal <= np.min(self.co_ratios)] = np.min(self.co_ratios) + 1e-6
        co_ratios_goal[co_ratios_goal >= np.max(self.co_ratios)] = np.max(self.co_ratios) - 1e-6

        fehs_goal[fehs_goal <= np.min(self.log10_metallicities)] = np.min(self.log10_metallicities) + 1e-6
        fehs_goal[fehs_goal >= np.max(self.log10_metallicities)] = np.max(self.log10_metallicities) - 1e-6

        temps_goal[temps_goal <= np.min(self.temperatures)] = np.min(self.temperatures) + 1e-6
        temps_goal[temps_goal >= np.max(self.temperatures)] = np.max(self.temperatures) - 1e-6

        pressures_goal[pressures_goal <= np.min(self.pressures)] = (
            np.min(self.pressures) + 1e-6)
        pressures_goal[pressures_goal >= np.max(self.pressures)] = (
            np.max(self.pressures) - 1e-6)

        # Get interpolation indices
        co_ratios_large_int = np.searchsorted(self.co_ratios, co_ratios_goal) + 1
        fehs_large_int = np.searchsorted(self.log10_metallicities, fehs_goal) + 1
        temps_large_int = np.searchsorted(self.temperatures, temps_goal) + 1
        pressures_large_int = np.searchsorted(self.pressures, pressures_goal) + 1

        if full:
            chemical_table = np.concatenate(
                (
                    self.mass_fractions,
                    self.mean_molar_masses[np.newaxis],
                    self.nabla_adiabatic[np.newaxis]
                )
            )
        else:
            chemical_table = self.mass_fractions

        chemical_table = np.array(chemical_table, dtype='d', order='F')

        # Get the interpolated values from Fortran routine
        # TODO nabla ad and MMW are not abundances, they should be returned as extra parameters
        _chemical_table = fchem.interpolate_mass_fractions_table(
            co_ratios_goal, fehs_goal, temps_goal,
            pressures_goal, co_ratios_large_int,
            fehs_large_int, temps_large_int,
            pressures_large_int, self.log10_metallicities, self.co_ratios,
            self.pressures, self.temperatures, chemical_table
        )

        # Sort in output format of this function
        mass_fractions = {}

        for id_, name in enumerate(self.species):
            mass_fractions[name] = _chemical_table[id_]

        if full:
            mean_molar_masses = _chemical_table[-2]
            nabla_adiabatic = _chemical_table[-1]
        else:
            mean_molar_masses = None
            nabla_adiabatic = None

        # Carbon quenching? Assumes pressures_goal is sorted in ascending order
        if carbon_pressure_quench is not None:
            if carbon_pressure_quench > np.min(pressures_goal):

                q_index = min(np.searchsorted(pressures_goal, carbon_pressure_quench),
                              int(len(pressures_goal)) - 1)

                methane_abb = mass_fractions['CH4']
                methane_abb[pressures_goal < carbon_pressure_quench] = \
                    mass_fractions['CH4'][q_index]
                mass_fractions['CH4'] = methane_abb

                co_abb = mass_fractions['CO']
                co_abb[pressures_goal < carbon_pressure_quench] = \
                    mass_fractions['CO'][q_index]
                mass_fractions['CO'] = co_abb

                h2o_abb = mass_fractions['H2O']
                h2o_abb[pressures_goal < carbon_pressure_quench] = \
                    mass_fractions['H2O'][q_index]
                mass_fractions['H2O'] = h2o_abb

        if full:
            return mass_fractions, mean_molar_masses, nabla_adiabatic

        return mass_fractions

    def load(self, path=None):
        if path is None:
            path = self.get_default_input_data_path()

        chemical_table_file = os.path.join(
            path, "equilibrium_chemistry.chemtable.petitRADTRANS.h5"
        )

        if not os.path.isfile(chemical_table_file):
            raise FileNotFoundError(get_input_data_file_not_found_error_message(chemical_table_file))

        print(f"Loading chemical equilibrium mass mixing ratio table from file '{chemical_table_file}'...")

        with h5py.File(chemical_table_file, 'r') as f:
            self.log10_metallicities = f['log10_metallicities'][()]
            self.co_ratios = f['co_ratios'][()]
            self.temperatures = f['temperatures'][()]
            self.pressures = f['pressures'][()]

            self.species = f['species'][()]
            self.species = np.array([name.decode('utf-8') for name in self.species])

            self.mass_fractions = f['mass_fractions'][()]
            self.mean_molar_masses = f['mean_molar_masses'][()]
            self.nabla_adiabatic = f['nabla_adiabatic'][()]

        self._loaded = True


pre_calculated_equilibrium_chemistry_table = PreCalculatedEquilibriumChemistryTable()
