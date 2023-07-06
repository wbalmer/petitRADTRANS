import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.physics import guillot_global

atmosphere = Radtrans(line_species = ['H2O_main_iso',
                                      'CO_all_iso',
                                      'CH4_main_iso',
                                      'CO2_main_iso',
                                      'Na',
                                      'K'],
                      rayleigh_species = ['H2', 'He'],
                      continuum_opacities = ['H2-H2', 'H2-He'],
                      wlen_bords_micron = [2.2, 2.4],
                      mode = 'lbl')

pressures = np.logspace(-10, 2, 130)
atmosphere.setup_opa_structure(pressures)

import petitRADTRANS.nat_cst as nc
from petitRADTRANS.physics import guillot_global

R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

mass_fractions = {}
mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
mass_fractions['He'] = 0.24 * np.ones_like(temperature)
mass_fractions['H2O_main_iso'] = 0.001 * np.ones_like(temperature)
mass_fractions['CO_all_iso'] = 0.01 * np.ones_like(temperature)
mass_fractions['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
mass_fractions['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
mass_fractions['Na'] = 0.00001 * np.ones_like(temperature)
mass_fractions['K'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW)
