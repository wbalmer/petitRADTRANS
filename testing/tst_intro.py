import numpy as np
from petitRADTRANS import Radtrans

import os
#os.environ["pRT_input_data_path"] = "/Applications/ownpy/petitRADTRANS/petitRADTRANS/input_data"

Chubb = False

if not Chubb:
    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'Na', 'K'],
                          rayleigh_species = ['H2', 'He'],
                          continuum_opacities = ['H2-H2', 'H2-He'],
                          wlen_bords_micron = [0.3, 15],
                          path_input_data = "/Applications/ownpy/petitRADTRANS/petitRADTRANS/input_data")

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

temperature = 1200. * np.ones_like(pressures)

abundances = {}
abundances['H2'] = 0.74 * np.ones_like(temperature)
abundances['He'] = 0.24 * np.ones_like(temperature)

if Chubb:
    abundances['H2O_Chubb_mass'] = 0.001 * np.ones_like(temperature)
    abundances['Na_Chubb_mass'] = 0.00001 * np.ones_like(temperature)
    abundances['K_Chubb_mass'] = 0.000001 * np.ones_like(temperature)

else:
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)

    
abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
abundances['CO2'] = 0.00001 * np.ones_like(temperature)
abundances['CH4'] = 0.000001 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import guillot_global
R_pl = 1.838*cst.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

atmosphere.calculate_transit_radii(temperature, abundances, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

plt.plot(cst.c / atmosphere.frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.show()
#plt.clf()

frequencies, flux = atmosphere.calculate_flux(temperature, abundances, gravity, MMW)

plt.plot(cst.c / frequencies / 1e-4, flux / 1e-6)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
#plt.clf()

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.

temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

plt.plot(temperature, pressures)
plt.yscale('log')
plt.ylim([1e2, 1e-6])
plt.xlabel('T (K)')
plt.ylabel('P (bar)')
plt.show()
#plt.clf()

frequencies, flux = atmosphere.calculate_flux(temperature, abundances, gravity, MMW)

plt.plot(cst.c / frequencies / 1e-4, flux / 1e-6)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
#plt.clf()
