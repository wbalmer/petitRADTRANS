import numpy as np
from petitRADTRANS import Radtrans

low_res = False

if low_res:
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO-NatAbund_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'],
                          rayleigh_species = ['H2', 'He'],
                          continuum_opacities = ['H2-H2', 'H2-He'],
                          wlen_bords_micron = [0.3, 15])
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO-NatAbund',
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

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global

R_pl = 1.838*cst.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = temperature_profile_function_guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

if low_res:
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO-NatAbund_HITEMP'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K_allard'] = 0.000001 * np.ones_like(temperature)
else:
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

import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Clear
atmosphere.calculate_transit_radii(temperature, mass_fractions, \
                                   gravity, MMW, planet_radius=R_pl, reference_pressure=P0)
clear = atmosphere.transit_radii / cst.r_jup_mean

kappa_zero = 0.01
gamma_scat = -4.

atmosphere.calculate_transit_radii(temperature, mass_fractions, \
                                   gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   power_law_opacity_350nm= kappa_zero, power_law_opacity_coefficient= gamma_scat)

m4 = atmosphere.transit_radii / cst.r_jup_mean

kappa_zero = 0.01
gamma_scat = -2.

atmosphere.calculate_transit_radii(temperature, mass_fractions, \
                                   gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   power_law_opacity_350nm= kappa_zero, power_law_opacity_coefficient= gamma_scat)

m2 = atmosphere.transit_radii / cst.r_jup_mean

kappa_zero = 0.01
gamma_scat = 0.

atmosphere.calculate_transit_radii(temperature, mass_fractions, \
                                   gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   power_law_opacity_350nm= kappa_zero, power_law_opacity_coefficient= gamma_scat)

m0 = atmosphere.transit_radii / cst.r_jup_mean

kappa_zero = 0.01
gamma_scat = 1.

atmosphere.calculate_transit_radii(temperature, mass_fractions, \
                                   gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   power_law_opacity_350nm= kappa_zero, power_law_opacity_coefficient= gamma_scat)

p1 = atmosphere.transit_radii / cst.r_jup_mean

# Make plot

plt.plot(cst.c / atmosphere._frequencies / 1e-4, clear, label ='Clear')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         m4, \
         label = r'Powerlaw cloud, $\gamma = -4$')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         m2, \
         label = r'Powerlaw cloud, $\gamma = -2$')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         m0, \
         label = r'Powerlaw cloud, $\gamma = 0$')
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
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
atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, planet_radius=R_pl, reference_pressure=P0)
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         atmosphere.transit_radii / cst.r_jup_mean, label ='Clear')

# Gray cloud deck at 0.01 bar
atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   opaque_cloud_top_pressure= 0.01)
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         atmosphere.transit_radii / cst.r_jup_mean, label ='Gray cloud deck at 0.01 bar')

# Haze (10 x gas Rayleigh scattering)
atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   haze_factor = 10)
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         atmosphere.transit_radii / cst.r_jup_mean, label ='Rayleigh haze')

# Haze + cloud deck
atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, planet_radius=R_pl, reference_pressure=P0, \
                                   haze_factor = 10, opaque_cloud_top_pressure= 0.01)
plt.plot(cst.c / atmosphere._frequencies / 1e-4, \
         atmosphere.transit_radii / cst.r_jup_mean, label ='Rayleigh haze + cloud deck')

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
                                          'CO-NatAbund_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'], \
          cloud_species = ['Mg2SiO4(s)_crystalline__DHS'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = [0.3, 15])
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO-NatAbund',
                                        'CH4_main_iso',
                                        'CO2_main_iso',
                                        'Na',
                                        'K'],
                          cloud_species=['Mg2SiO4(s)_crystalline__DHS'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[2.2, 2.4],
                          mode='lbl')

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

from petitRADTRANS import physical_constants as cst
R_pl = 1.838*cst.r_jup_mean
gravity = 1e1**2.45
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = temperature_profile_function_guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

if low_res:
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO-NatAbund_HITEMP'] = 0.01 * np.ones_like(temperature)
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
    mass_fractions['CO-NatAbund'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)

radius = {}
radius['Mg2SiO4(c)'] = 0.00005*np.ones_like(temperature) # I.e. a 0.5-micron particle size (0.00005 cm)

sigma_lnorm = 1.05



atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, \
                                   planet_radius=R_pl, reference_pressure=P0, \
                                   cloud_particles_mean_radii= radius, cloud_particle_radius_distribution_std= sigma_lnorm)

plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='cloudy', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, \
                                   planet_radius=R_pl, reference_pressure=P0, \
                                   cloud_particles_mean_radii= radius, cloud_particle_radius_distribution_std= sigma_lnorm)
plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='clear', zorder = 1)

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

atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, \
                                   planet_radius=R_pl, reference_pressure=P0, \
                                   eddy_diffusion_coefficient= Kzz, cloud_f_sed=fsed, cloud_particle_radius_distribution_std= sigma_lnorm)

plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='cloudy', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

atmosphere.calculate_transit_radii(temperature, mass_fractions, gravity, MMW, \
                                   planet_radius=R_pl, reference_pressure=P0, \
                                   eddy_diffusion_coefficient= Kzz, cloud_f_sed=fsed, cloud_particle_radius_distribution_std= sigma_lnorm)

plt.plot(cst.c / atmosphere._frequencies / 1e-4, atmosphere.transit_radii / cst.r_jup_mean, label ='clear', zorder = 1)

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
atmosphere.calculate_flux(temperature, mass_fractions, gravity, MMW,
                          eddy_diffusion_coefficients= Kzz, cloud_f_sed=fsed, cloud_hansen_b=b_hans, cloud_particles_radius_distribution='hansen')

plt.yscale('log')
plt.xscale('log')

plt.ylim([1e2,1e-6])

plt.ylabel('P (bar)')
plt.xlabel('Average particle size (microns)')

plt.plot(atmosphere.r_g[:,atmosphere.cloud_species.index('Mg2SiO4(c)')]/1e-4, pressures)
plt.show()
plt.clf()

mass_fractions['Mg2SiO4(c)'] = 0.0000005 * np.ones_like(temperature)

frequencies, flux, _ = atmosphere.calculate_flux(temperature, mass_fractions, gravity, MMW, \
                                                 eddy_diffusion_coefficients= Kzz, cloud_f_sed=fsed, cloud_particle_radius_distribution_std= sigma_lnorm)

plt.plot(cst.c / frequencies / 1e-4, flux / 1e-6, \
         color = 'black', label = 'cloudy, no scattering', zorder = 1)

# Load scattering version of pRT
if low_res:
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO-NatAbund_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'], \
          cloud_species = ['Mg2SiO4(s)_crystalline__DHS'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = [0.3, 15], \
          do_scat_emis = True)
else:
    atmosphere = Radtrans(line_species=['H2O_main_iso',
                                        'CO-NatAbund',
                                        'CH4_main_iso',
                                        'CO2_main_iso',
                                        'Na',
                                        'K'],
                          cloud_species=['Mg2SiO4(s)_crystalline__DHS'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[2.2, 2.4],
                          mode='lbl')

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

frequencies, flux, _ = atmosphere.calculate_flux(temperature, mass_fractions, gravity, MMW, \
                                                 eddy_diffusion_coefficients= Kzz, cloud_f_sed=fsed, cloud_particle_radius_distribution_std= sigma_lnorm, \
                                                 add_cloud_scattering_as_absorption= True)
plt.plot(cst.c / frequencies / 1e-4, flux / 1e-6, \
         label = 'cloudy, including scattering', zorder = 2)

mass_fractions['Mg2SiO4(c)'] = np.zeros_like(temperature)

frequencies, flux, _ = atmosphere.calculate_flux(temperature, mass_fractions, gravity, MMW, \
                                                 eddy_diffusion_coefficients= Kzz, cloud_f_sed=fsed, cloud_particle_radius_distribution_std= sigma_lnorm)

plt.plot(cst.c / frequencies / 1e-4, flux / 1e-6, '-', \
         color = 'red', label = 'clear', zorder = 0)

plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()


