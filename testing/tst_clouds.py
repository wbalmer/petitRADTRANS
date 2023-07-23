import numpy as np
from petitRADTRANS import Radtrans

low_res = False

if low_res:
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO_all_iso_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'],
                          rayleigh_species = ['H2', 'He'],
                          continuum_opacities = ['H2-H2', 'H2-He'],
                          wlen_bords_micron = [0.3, 15])
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO_all_iso',
                                        'CH4_main_iso',
                                        'CO2_main_iso',
                                        'Na',
                                        'K'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[2.2, 2.4],
                          mode='lbl')

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.physics import guillot_global

R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

if low_res:
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso_HITEMP'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K_allard'] = 0.000001 * np.ones_like(temperature)
else:
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

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Clear
atmosphere.calc_transm(temperature, mass_fractions, \
                       gravity, MMW, r_pl=R_pl, p0_bar=P0)
clear = atmosphere.transm_rad/nc.r_jup_mean

kappa_zero = 0.01
gamma_scat = -4.

atmosphere.calc_transm(temperature, mass_fractions, \
                       gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       kappa_zero = kappa_zero, gamma_scat = gamma_scat)

m4 = atmosphere.transm_rad/nc.r_jup_mean

kappa_zero = 0.01
gamma_scat = -2.

atmosphere.calc_transm(temperature, mass_fractions, \
                       gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       kappa_zero = kappa_zero, gamma_scat = gamma_scat)

m2 = atmosphere.transm_rad/nc.r_jup_mean

kappa_zero = 0.01
gamma_scat = 0.

atmosphere.calc_transm(temperature, mass_fractions, \
                       gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       kappa_zero = kappa_zero, gamma_scat = gamma_scat)

m0 = atmosphere.transm_rad/nc.r_jup_mean

kappa_zero = 0.01
gamma_scat = 1.

atmosphere.calc_transm(temperature, mass_fractions, \
                       gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       kappa_zero = kappa_zero, gamma_scat = gamma_scat)

p1 = atmosphere.transm_rad/nc.r_jup_mean

# Make plot

plt.plot(nc.c / atmosphere.frequencies / 1e-4, clear, label ='Clear')
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         m4, \
         label = r'Powerlaw cloud, $\gamma = -4$')
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         m2, \
         label = r'Powerlaw cloud, $\gamma = -2$')
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         m0, \
         label = r'Powerlaw cloud, $\gamma = 0$')
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         p1, \
         label = r'Powerlaw cloud, $\gamma = 1$')
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc = 'best')
plt.show()
plt.clf()

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Clear
atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, r_pl=R_pl, p0_bar=P0)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         atmosphere.transm_rad / nc.r_jup_mean, label = 'Clear')

# Gray cloud deck at 0.01 bar
atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       p_cloud= 0.01)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         atmosphere.transm_rad / nc.r_jup_mean, label = 'Gray cloud deck at 0.01 bar')

# Haze (10 x gas Rayleigh scattering)
atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       haze_factor = 10)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         atmosphere.transm_rad / nc.r_jup_mean, label = 'Rayleigh haze')

# Haze + cloud deck
atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, r_pl=R_pl, p0_bar=P0, \
                       haze_factor = 10, p_cloud= 0.01)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, \
         atmosphere.transm_rad / nc.r_jup_mean, label = 'Rayleigh haze + cloud deck')

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc = 'best')
plt.show()
plt.clf()

import numpy as np
from petitRADTRANS import Radtrans

if low_res:
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO_all_iso_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'], \
          cloud_species = ['Mg2SiO4(c)_cd'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = [0.3, 15])
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO_all_iso',
                                        'CH4_main_iso',
                                        'CO2_main_iso',
                                        'Na',
                                        'K'],
                          cloud_species=['Mg2SiO4(c)_cd'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[2.2, 2.4],
                          mode='lbl')

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

from petitRADTRANS import nat_cst as nc
R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

if low_res:
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso_HITEMP'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K_allard'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)
else:
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_main_iso'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

radius = {}
radius['Mg2SiO4(c)'] = 0.00005*np.ones_like(temperature) # I.e. a 0.5-micron particle size (0.00005 cm)

sigma_lnorm = 1.05



atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, \
                       r_pl=R_pl, p0_bar=P0, \
                       radius = radius, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.transm_rad / nc.r_jup_mean, label ='cloudy', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, \
                       r_pl=R_pl, p0_bar=P0, \
                       radius = radius, sigma_lnorm = sigma_lnorm)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.transm_rad / nc.r_jup_mean, label ='clear', zorder = 1)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc='best')
plt.show()
plt.clf()

Kzz = np.ones_like(temperature)*1e1**7.5
fsed = 2.
sigma_lnorm = 1.05


mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, \
                       r_pl=R_pl, p0_bar=P0, \
                       kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.transm_rad / nc.r_jup_mean, label ='cloudy', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_transm(temperature, mass_fractions, gravity, MMW, \
                       r_pl=R_pl, p0_bar=P0, \
                       kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.transm_rad / nc.r_jup_mean, label ='clear', zorder = 1)

plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit radius ($\rm R_{Jup}$)')
plt.legend(loc='best')
plt.show()
plt.clf()


plt.yscale('log')
plt.xscale('log')

plt.ylim([1e2,1e-6])

plt.ylabel('P (bar)')
plt.xlabel('Average particle size (microns)')

plt.plot(atmosphere.r_g[:,atmosphere.cloud_species.index('Mg2SiO4(c)')]/1e-4, pressures)
plt.show()
plt.clf()

mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)
Kzz = np.ones_like(temperature)*1e1**7.5
fsed = 2.
b_hans = 0.01
atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW,
                     kzz= Kzz, fsed=fsed, b_hans=b_hans, dist='hansen')

plt.yscale('log')
plt.xscale('log')

plt.ylim([1e2,1e-6])

plt.ylabel('P (bar)')
plt.xlabel('Average particle size (microns)')

plt.plot(atmosphere.r_g[:,atmosphere.cloud_species.index('Mg2SiO4(c)')]/1e-4, pressures)
plt.show()
plt.clf()

mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                     kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, \
         color = 'black', label = 'cloudy, no scattering', zorder = 1)

# Load scattering version of pRT
if low_res:
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO_all_iso_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'], \
          cloud_species = ['Mg2SiO4(c)_cd'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = [0.3, 15], \
          do_scat_emis = True)
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO_all_iso',
                                        'CH4_main_iso',
                                        'CO2_main_iso',
                                        'Na',
                                        'K'],
                          cloud_species=['Mg2SiO4(c)_cd'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[2.2, 2.4],
                          mode='lbl')

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                     kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm, \
                     add_cloud_scat_as_abs = True)
plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, \
         label = 'cloudy, including scattering', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                     kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm)

plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, '-', \
         color = 'red', label = 'clear', zorder = 0)

plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()


