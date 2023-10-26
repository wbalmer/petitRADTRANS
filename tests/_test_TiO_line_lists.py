import numpy as np
from petitRADTRANS import Radtrans

import os
os.environ["pRT_input_data_path"] = "/Users/molliere/Documents/programm_data/petitRADTRANS_public/input_data"

atmosphere = Radtrans(line_species = ['H2O_Exomol', 'CO-NatAbund_HITEMP', 'CH4', 'CO2', 'Na_allard', 'K_allard', 'TiO_all_Plez'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15])

atmosphere2 = Radtrans(line_species = ['H2O_Exomol', 'CO-NatAbund_HITEMP', 'CH4', 'CO2', 'Na_allard', 'K_allard', 'TiO_all_Exomol'], \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [0.3, 15])


pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)
atmosphere2.setup_opa_structure(pressures)

temperature = 2500. * np.ones_like(pressures)

abundances = {}
abundances['H2'] = 0.74 * np.ones_like(temperature)
abundances['He'] = 0.24 * np.ones_like(temperature)

abundances['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
abundances['H2O_Exomol'] = 0.001 * np.ones_like(temperature)
abundances['Na_allard'] = 0.00001 * np.ones_like(temperature)
abundances['K_allard'] = 0.000001 * np.ones_like(temperature)
abundances['Na_burrows'] = 0.00001 * np.ones_like(temperature)
abundances['K_burrows'] = 0.000001 * np.ones_like(temperature)
abundances['Na_lor_cut'] = 0.00001 * np.ones_like(temperature)
abundances['K_lor_cut'] = 0.000001 * np.ones_like(temperature)

abundances['TiO_all_Plez'] = 0.00001 * np.ones_like(temperature)
abundances['TiO_all_Exomol'] = 0.00001 * np.ones_like(temperature)

abundances['CO-NatAbund_HITEMP'] = 0.01 * np.ones_like(temperature)
abundances['CO-NatAbund_Chubb'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

from petitRADTRANS import physical_constants as cst
R_pl = 1.838*cst.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

atmosphere.calculate_transit_radii(temperature, abundances, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)
atmosphere2.calculate_transit_radii(temperature, abundances, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)

#import pdb
#pdb.set_trace()

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='Plez')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere2.transit_radii / cst.r_jup_mean, label ='Exomol')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend()
plt.title('T = 2500 K')
plt.show()

temperature = 200. * np.ones_like(pressures)

atmosphere.calculate_transit_radii(temperature, abundances, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)
atmosphere2.calculate_transit_radii(temperature, abundances, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)

plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='Plez')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere2.transit_radii / cst.r_jup_mean, label ='Exomol')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend()
plt.title('T = 200 K')
plt.show()
