"""Manage equilibrium chemistry pre-calculated table.
"""
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np  # For file I/O, as JAX doesn't handle it directly
from jax import lax

from jax.typing import ArrayLike

from petitRADTRANS._input_data import find_input_file
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser


def interpolate_chemical_table(
        co_ratios: jax.typing.ArrayLike,            # (n_layers,)
        log10_metallicities: jax.typing.ArrayLike,  # (n_layers,)
        temperatures: jax.typing.ArrayLike,         # (n_layers,)
        pressures: jax.typing.ArrayLike,            # (n_layers,)
        largest_co_ratios_indices: jax.typing.ArrayLike,      # (n_layers,) int, 1-based
        largest_metallicities_indices: jax.typing.ArrayLike,  # (n_layers,) int, 1-based
        largest_temperatures_indices: jax.typing.ArrayLike,   # (n_layers,) int, 1-based
        largest_pressures_indices: jax.typing.ArrayLike,      # (n_layers,) int, 1-based
        table_metallicities: jax.typing.ArrayLike,  # (n_metallicities_table_axis,)
        table_co_ratios: jax.typing.ArrayLike,      # (n_co_ratios_table_axis,)
        table_pressures: jax.typing.ArrayLike,      # (n_pressures_table_axis,)
        table_temperatures: jax.typing.ArrayLike,   # (n_temperatures_table_axis,)
        chemistry_table: jax.typing.ArrayLike,      # (n_species, n_temperatures, n_pressures, n_co_ratios, n_metallicities)
        is_mass_fraction: bool
        ) -> jax.typing.ArrayLike:
    """
    Performs 4D linear interpolation on the chemical table for n_layers.
    Input indices (largest_..._indices) are 1-based and point to the upper
    bound of the interpolation cell.
    """
    # Ensure table axes are JAX arrays
    table_T_jax = jnp.asarray(table_temperatures)
    table_P_jax = jnp.asarray(table_pressures)
    table_CO_jax = jnp.asarray(table_co_ratios)
    table_FeH_jax = jnp.asarray(table_metallicities)

    # Define the core interpolation logic for a single layer
    def _interpolate_single_layer(target_co, target_log_feh, target_T, target_P,
                                    idx_T_up, idx_P_up, idx_CO_up, idx_FeH_up):

        # Determine 0-based lower and upper indices for each dimension
        # These are scalar integer JAX arrays for a single layer computation
        t_indices   = jnp.array([idx_T_up - 1, idx_T_up])
        p_indices   = jnp.array([idx_P_up - 1, idx_P_up])
        co_indices  = jnp.array([idx_CO_up - 1, idx_CO_up])
        feh_indices = jnp.array([idx_FeH_up - 1, idx_FeH_up])

        # It's crucial that indices are within bounds.
        # Gather the 8 corner values from the chemistry table for interpolation
        # self.chemistry_table shape: (n_species, n_temp, n_press, n_co, n_metal)
        # intp_bound shape: (n_species, 2_T, 2_P, 2_CO, 2_FeH)
        # The index arrays (t_indices etc.) select along their respective dimensions.
        intp_bound = chemistry_table[
            :,
            t_indices[:, None, None, None],
            p_indices[None, :, None, None],
            co_indices[None, None, :, None],
            feh_indices[None, None, None, :]
        ]

        # Gather the grid coordinates for these corner points
        coords_T   = table_T_jax[t_indices]
        coords_P   = table_P_jax[p_indices]
        coords_CO  = table_CO_jax[co_indices]
        coords_FeH = table_FeH_jax[feh_indices]

        # Helper for 1D linear interpolation
        def _linear_interp_1d(y_corners, x_corners, x_target):
            # y_corners: (..., 2), x_corners: (2,), x_target: scalar
            y0, y1 = y_corners[..., 0], y_corners[..., 1]
            x0, x1 = x_corners[0], x_corners[1]

            delta_x = x1 - x0
            # Handle collapsed interval (delta_x == 0) to avoid NaN/Inf
            # If delta_x is 0, it implies x0=x1. Result should be y0.
            weight = jnp.where(delta_x == 0.0, 0.0, (x_target - x0) / delta_x)
            return y0 + (y1 - y0) * weight

        # Perform sequential 1D interpolations
        interp_val = _linear_interp_1d(intp_bound, coords_FeH, target_log_feh) # Along FeH
        interp_val = _linear_interp_1d(interp_val, coords_CO, target_co)       # Along C/O
        interp_val = _linear_interp_1d(interp_val, coords_P, target_P)         # Along Pressure
        interp_val = _linear_interp_1d(interp_val, coords_T, target_T)         # Along Temperature
        # interp_val now has shape (n_species,)

        # Replace NaN with -50.0 (as in Fortran)
        return jnp.where(jnp.isnan(interp_val), -50.0, interp_val)

    # Vectorize the single_layer interpolation over all layers
    # Inputs are (n_layers, ...)
    # Output will be (n_layers, n_species)
    chemical_table_interp_flat = jax.vmap(
        _interpolate_single_layer,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0)  # Map over the first axis of all 8 input arrays
    )(co_ratios, log10_metallicities, temperatures, pressures,
        largest_temperatures_indices, largest_pressures_indices, largest_co_ratios_indices, largest_metallicities_indices)

    # Transpose to Fortran's output shape: (n_species, n_layers)
    # TODO: not sure this is necessary now that we're not using fotran
    chemical_table_interp = jnp.transpose(chemical_table_interp_flat, (1, 0))

    # If is_mass_fraction, convert back from log space (for abundances)
    # The Fortran code applied 10** to all species if is_mass_fraction is true.
    # This means MMW and nabla_ad (if present, and were linear) also get 10**.
    # This is a direct translation of the provided Fortran.
    if is_mass_fraction:
        chemical_table_interp = 10.0**chemical_table_interp

    return chemical_table_interp


class PreCalculatedEquilibriumChemistryTable:
    """Used to store petitRADTRANS's pre-calculated mass fractions at chemical equilibrium.

    The interpolate_mass_fractions function can be used to get mass fractions at given thermochemical conditions.
    """
    def __init__(self):
        self._loaded: bool = False

        self.mass_fractions: jax.typing.ArrayLike = None
        self.mean_molar_masses: jax.typing.ArrayLike = None
        self.nabla_adiabatic: jax.typing.ArrayLike = None
        self.log10_metallicities: jax.typing.ArrayLike = None
        self.co_ratios: jax.typing.ArrayLike = None
        self.temperatures: jax.typing.ArrayLike = None
        self.pressures: jax.typing.ArrayLike = None

    @staticmethod
    def get_default_file(path_input_data: str = None) -> str:
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        return str(os.path.join(
            path_input_data,
            get_input_data_subpaths()["pre_calculated_chemistry"],
            "equilibrium_chemistry",
            "equilibrium_chemistry.chemtable.petitRADTRANS.h5"
        ))

    def interpolate_mass_fractions(self, co_ratios: iter, log10_metallicities: iter,
                                   temperatures: iter, pressures: iter,
                                   carbon_pressure_quench: float = None, full: bool = False,
                                   numpy_compatible=False):
        """Interpolate mass fractions from a pre-calculated table to the desired parameters.

        Args:
            co_ratios:
                Desired carbon to oxygen ratios, obtained by increasing the amount of oxygen.
            log10_metallicities:
                Base-10 logarithm of the desired metallicities.
            temperatures:
                (K) desired temperatures
            pressures:
                (bar) desired pressures
            carbon_pressure_quench:
                (bar) pressure at which to put a simplistic carbon-bearing species quenching
            full:
                if True, output the pre-calculated mean molar mass and logarithmic derivative of temperature with
                respect to pressure in the adiabatic case (nabla_ad) in addition to the pre-calculated mass fractions
        """
        if not self._loaded:
            (self.mass_fractions, self.mean_molar_masses, self.nabla_adiabatic, self.log10_metallicities,
             self.co_ratios, self.temperatures, self.pressures) = self.load()
            self._loaded = True
        co_ratios: ArrayLike = jnp.array(co_ratios).reshape(-1)
        log10_metallicities: ArrayLike = jnp.array(log10_metallicities).reshape(-1)
        temperatures: ArrayLike = jnp.array(temperatures).reshape(-1)
        pressures: ArrayLike = jnp.array(pressures).reshape(-1)

        # Apply boundary treatment
        co_ratios = jnp.clip(
            co_ratios,
            min=jnp.min(self.co_ratios) + 1e-6,
            max=jnp.max(self.co_ratios) - 1e-6,
        )

        log10_metallicities = jnp.clip(
            log10_metallicities,
            min=jnp.min(self.log10_metallicities) + 1e-6,
            max=jnp.max(self.log10_metallicities) - 1e-6,
        )

        temperatures = jnp.clip(
            temperatures,
            min=jnp.min(self.temperatures) + 1e-6,
            max=jnp.max(self.temperatures) - 1e-6,
        )

        pressures = jnp.clip(
            pressures,
            min=jnp.min(self.pressures) + 1e-6,
            max=jnp.max(self.pressures) - 1e-6
        )

        # Get interpolation indices
        co_ratios_large_int = jnp.searchsorted(self.co_ratios, co_ratios)
        fehs_large_int = jnp.searchsorted(self.log10_metallicities, log10_metallicities)
        temps_large_int = jnp.searchsorted(self.temperatures, temperatures)
        pressures_large_int = jnp.searchsorted(self.pressures, pressures)

        _mass_fractions = interpolate_chemical_table(
            co_ratios,
            log10_metallicities,
            temperatures,
            pressures,
            co_ratios_large_int,
            fehs_large_int,
            temps_large_int,
            pressures_large_int,
            self.log10_metallicities,
            self.co_ratios,
            self.pressures,
            self.temperatures,
            self.mass_fractions,
            True
        )

        # Sort in output format of this function
        mass_fractions: dict[str, ...] = {}

        for id_, name in enumerate(self.species):
            mass_fractions[name] = _mass_fractions[id_]

        if full:
            mean_molar_masses = interpolate_chemical_table(
                co_ratios,
                log10_metallicities,
                temperatures,
                pressures,
                co_ratios_large_int,
                fehs_large_int,
                temps_large_int,
                pressures_large_int,
                self.log10_metallicities,
                self.co_ratios,
                self.pressures,
                self.temperatures,
                self.mean_molar_masses[jnp.newaxis],
                False
            )[0]

            nabla_adiabatic = interpolate_chemical_table(
                co_ratios,
                log10_metallicities,
                temperatures,
                pressures,
                co_ratios_large_int,
                fehs_large_int,
                temps_large_int,
                pressures_large_int,
                self.log10_metallicities,
                self.co_ratios,
                self.pressures,
                self.temperatures,
                self.nabla_adiabatic[jnp.newaxis],
                False
            )[0]
        else:
            mean_molar_masses = None
            nabla_adiabatic = None

        # Carbon quenching, assumes pressures_goal is sorted in ascending order
        if carbon_pressure_quench is not None:
            def apply_carbon_quench(mass_fractions):
                q_index = jnp.clip(
                    jnp.searchsorted(pressures, carbon_pressure_quench),
                    0, pressures.size - 1
                )

                mass_fractions['CH4'] = jnp.where(
                    pressures < carbon_pressure_quench,
                    mass_fractions['CH4'][q_index],
                    mass_fractions['CH4']
                )

                mass_fractions['CO'] = jnp.where(
                    pressures < carbon_pressure_quench,
                    mass_fractions['CO'][q_index],
                    mass_fractions['CO']
                )

                mass_fractions['H2O'] = jnp.where(
                    pressures < carbon_pressure_quench,
                    mass_fractions['H2O'][q_index],
                    mass_fractions['H2O']
                )
                return mass_fractions

            def no_carbon_quench(mass_fractions):
                return mass_fractions

            mass_fractions = lax.cond(
                carbon_pressure_quench > jnp.min(pressures),
                apply_carbon_quench,
                no_carbon_quench,
                mass_fractions
            )
        if numpy_compatible:
            for key, value in mass_fractions.items():
                mass_fractions[key] = np.array(value)
            mean_molar_masses = np.array(mean_molar_masses)
            nabla_adiabatic = np.array(nabla_adiabatic)

        if full:
            return mass_fractions, mean_molar_masses, nabla_adiabatic

        return mass_fractions

    def load(self, path: str = None, path_input_data: str = None, file: str = ''):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        if path is None:
            file = self.get_default_file(path_input_data)

        if not os.path.isfile(file):
            file = find_input_file(
                file=file,
                path_input_data=path_input_data,
                sub_path=None,
                match_function=None,  # use the default match function
                find_all=False,
                search_online=True
            )

        print(f"Loading chemical equilibrium chemistry table from file '{file}'... ", end='')

        with h5py.File(file, 'r') as f:
            log10_metallicities = f['log10_metallicities'][()]
            co_ratios = f['co_ratios'][()]
            temperatures = f['temperatures'][()]
            pressures = f['pressures'][()]

            species = f['species'][()]
            species = tuple([name.decode('utf-8') for name in species])

            mass_fractions = f['mass_fractions'][()]
            mean_molar_masses = f['mean_molar_masses'][()]
            nabla_adiabatic = f['nabla_adiabatic'][()]

        # Change array ordering for more efficient fortran calculation (this takes time, so it is done during loading)
        mass_fractions = jnp.array(mass_fractions, dtype='d')
        mean_molar_masses = jnp.array(mean_molar_masses, dtype='d')
        nabla_adiabatic = jnp.array(nabla_adiabatic, dtype='d')
        log10_metallicities = jnp.array(log10_metallicities, dtype='d')
        co_ratios = jnp.array(co_ratios, dtype='d')
        temperatures = jnp.array(temperatures, dtype='d')
        pressures = jnp.array(pressures, dtype='d')

        print("Done.")
        return mass_fractions, mean_molar_masses, nabla_adiabatic, log10_metallicities, co_ratios, temperatures, pressures


pre_calculated_equilibrium_chemistry_table = PreCalculatedEquilibriumChemistryTable()
