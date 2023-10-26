import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.physics import temperature_profile_function_guillot_global

atmosphere = Radtrans(line_species = ['H2O_main_iso',
                                      'CO-NatAbund',
                                      'CH4_main_iso',
                                      'CO2_main_iso',
                                      'Na',
                                      'K'],
                      rayleigh_species = ['H2', 'He'],
                      gas_continuum_contributors= ['H2-H2', 'H2-He'],
                      wavelengths_boundaries= [2.2, 2.4],
                      line_opacity_mode='lbl')

pressures = np.logspace(-10, 2, 130)
atmosphere.setup_opa_structure(pressures)

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global

R_pl = 1.838*cst.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = temperature_profile_function_guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

mass_fractions = {}
mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
mass_fractions['He'] = 0.24 * np.ones_like(temperature)
mass_fractions['H2O_main_iso'] = 0.001 * np.ones_like(temperature)
mass_fractions['CO-NatAbund'] = 0.01 * np.ones_like(temperature)
mass_fractions['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
mass_fractions['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
mass_fractions['Na'] = 0.00001 * np.ones_like(temperature)
mass_fractions['K'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

atmosphere.calculate_transit_radii(
    temperatures=temperature,
    mass_fractions=mass_fractions,
    reference_gravity=gravity,
    mean_molar_masses=MMW,
    planet_radius=R_pl,
    reference_pressure=P0
)

atmosphere.calculate_flux(temperature, mass_fractions, gravity, MMW)
