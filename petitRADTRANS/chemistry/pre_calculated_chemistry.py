"""Manage equilibrium chemistry pre-calculated table.
"""
import os

import h5py
import numpy as np

from petitRADTRANS.config.configuration import petitradtrans_config_parser, get_input_data_subpaths
from petitRADTRANS.fortran_chemistry import fortran_chemistry as fchem
from petitRADTRANS._input_data_loader import get_input_data_file_not_found_error_message


class PreCalculatedEquilibriumChemistryTable:
    def __init__(self):
        self._loaded = False

        self.log10_metallicities = None
        self.co_ratios = None
        self.temperatures = None
        self.pressures = None
        self.species = None
        self.mass_fractions = None
        self.nabla_adiabatic = None
        self.mean_molar_masses = None

    @staticmethod
    def get_default_input_data_path() -> str:
        return os.path.join(
            petitradtrans_config_parser.get_input_data_path(),
            get_input_data_subpaths()["pre_calculated_chemistry"],
            "equilibrium_chemistry"
        )

    def interpolate_mass_fractions(self, co_ratios: iter, log10_metallicities: iter,
                                   temperatures: iter, pressures: iter,
                                   carbon_pressure_quench: float = None, full: bool = False):
        """Interpolate mass fractions from a pre-calculated table to the desired parameters.

        Args:
            co_ratios:
                Desired carbon to oxygen ratios, obtained by increasing the amount of oxygen.
            log10_metallicities:
                Base-10 logarithm of the desired metallitcities.
            temperatures:
                (K) desired temperatures
            pressures:
                (bar) desired pressures
            carbon_pressure_quench:
                (bar) pressure at which to put a simplistic carbon-bearing species quenching
            full:
                if True, output the pre-calculated mean molar mass and logarithmic derivative of temperature with
                respect to pressure in addition to the pre-calculated mass fractions
        """
        if not self._loaded:
            self.load()

        co_ratios, log10_metallicities, temperatures, pressures = \
            np.array(co_ratios).reshape(-1), \
            np.array(log10_metallicities).reshape(-1), \
            np.array(temperatures).reshape(-1), \
            np.array(pressures).reshape(-1)

        # Apply boundary treatment
        co_ratios[co_ratios <= np.min(self.co_ratios)] = np.min(self.co_ratios) + 1e-6
        co_ratios[co_ratios >= np.max(self.co_ratios)] = np.max(self.co_ratios) - 1e-6

        log10_metallicities[log10_metallicities <= np.min(self.log10_metallicities)] =\
            np.min(self.log10_metallicities) + 1e-6
        log10_metallicities[log10_metallicities >= np.max(self.log10_metallicities)] =\
            np.max(self.log10_metallicities) - 1e-6

        temperatures[temperatures <= np.min(self.temperatures)] = np.min(self.temperatures) + 1e-6
        temperatures[temperatures >= np.max(self.temperatures)] = np.max(self.temperatures) - 1e-6

        pressures[pressures <= np.min(self.pressures)] = (
            np.min(self.pressures) + 1e-6)
        pressures[pressures >= np.max(self.pressures)] = (
            np.max(self.pressures) - 1e-6)

        # Get interpolation indices
        co_ratios_large_int = np.searchsorted(self.co_ratios, co_ratios) + 1
        fehs_large_int = np.searchsorted(self.log10_metallicities, log10_metallicities) + 1
        temps_large_int = np.searchsorted(self.temperatures, temperatures) + 1
        pressures_large_int = np.searchsorted(self.pressures, pressures) + 1

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

        # Get the interpolated values from Fortran routine
        _chemical_table = fchem.interpolate_chemical_table(
            co_ratios, log10_metallicities, temperatures,
            pressures, co_ratios_large_int,
            fehs_large_int, temps_large_int,
            pressures_large_int, self.log10_metallicities, self.co_ratios,
            self.pressures, self.temperatures, chemical_table, full
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
            if carbon_pressure_quench > np.min(pressures):
                q_index = min(np.searchsorted(pressures, carbon_pressure_quench),
                              int(len(pressures)) - 1)

                methane_abb = mass_fractions['CH4']
                methane_abb[pressures < carbon_pressure_quench] = \
                    mass_fractions['CH4'][q_index]
                mass_fractions['CH4'] = methane_abb

                co_abb = mass_fractions['CO']
                co_abb[pressures < carbon_pressure_quench] = \
                    mass_fractions['CO'][q_index]
                mass_fractions['CO'] = co_abb

                h2o_abb = mass_fractions['H2O']
                h2o_abb[pressures < carbon_pressure_quench] = \
                    mass_fractions['H2O'][q_index]
                mass_fractions['H2O'] = h2o_abb

        if full:
            return mass_fractions, mean_molar_masses, nabla_adiabatic

        return mass_fractions

    def load(self, path: str = None):
        if path is None:
            path = self.get_default_input_data_path()

        chemical_table_file = os.path.join(
            path, "equilibrium_chemistry.chemtable.petitRADTRANS.h5"
        )

        if not os.path.isfile(chemical_table_file):
            raise FileNotFoundError(get_input_data_file_not_found_error_message(chemical_table_file))

        print(f"Loading chemical equilibrium chemistry table from file '{chemical_table_file}'...")

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

        # Change array ordering for more efficient fortran calculation (this takes time, so it is done during loading)
        self.mass_fractions = np.array(self.mass_fractions, dtype='d', order='F')
        self.mean_molar_masses = np.array(self.mean_molar_masses, dtype='d', order='F')
        self.nabla_adiabatic = np.array(self.nabla_adiabatic, dtype='d', order='F')

        self._loaded = True


pre_calculated_equilibrium_chemistry_table = PreCalculatedEquilibriumChemistryTable()
