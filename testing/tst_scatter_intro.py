import numpy as np
import pylab as plt

plt.rcParams['figure.figsize'] = (10, 6)
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from petitRADTRANS.physics import guillot_global

case1 = False
case2 = False
case3 = True

if case1:
    # Load scattering version of pRT
    atmosphere = Radtrans(line_species = ['H2O_HITEMP',
                                          'CO_all_iso_HITEMP',
                                          'CH4',
                                          'CO2',
                                          'Na_allard',
                                          'K_allard'],
                          cloud_species = ['Mg2SiO4(c)_cd'],
                          rayleigh_species = ['H2', 'He'],
                          continuum_opacities = ['H2-H2', 'H2-He'],
                          wlen_bords_micron = [0.3, 15],
                          do_scat_emis = True)

    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)

    R_pl = 1.2*nc.r_jup_mean
    gravity = 1e1**3.5

    # P-T parameters
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 1200.
    T_equ = 0.
    temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    # Cloud parameters
    Kzz = np.ones_like(temperature)*1e1**6.5
    fsed = 2.
    sigma_lnorm = 1.05

    # Absorber mass fractions
    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso_HITEMP'] = 0.005 * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.0000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K_allard'] = 0.000001 * np.ones_like(temperature)

    # Cloud mass fractions
    mfr_cloud = np.zeros_like(temperature)
    mfr_cloud[pressures<=3.] = 0.00005 * (pressures[pressures<=3.]/3.)**fsed
    mass_fractions['Mg2SiO4(c)'] = mfr_cloud

    MMW = 2.33 * np.ones_like(temperature)

    mass_fractions['Mg2SiO4(c)'] = mfr_cloud

    atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                         kzz= Kzz, fsed=fsed, sigma_lnorm = sigma_lnorm, \
                         contribution = False)
    plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, \
             label = 'cloudy, including scattering', zorder = 2)
    contribution_scat = atmosphere.contribution_emission
    plt.show()

if case2:

    atmosphere = Radtrans(line_species=['H2O_HITEMP',
                                        'CO_all_iso_HITEMP',
                                        'CH4',
                                        'CO2',
                                        'Na_allard',
                                        'K_allard'],
                          cloud_species=['Mg2SiO4(c)_cd'],
                          rayleigh_species=['H2', 'He'],
                          continuum_opacities=['H2-H2', 'H2-He'],
                          wlen_bords_micron=[0.3, 15],
                          do_scat_emis=True)

    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)
    R_pl = 1.838 * nc.r_jup_mean
    gravity = 1e1 ** 2.45
    P0 = 0.01

    kappa_IR = 0.01
    gamma = 0.4
    T_int = 200.
    T_equ = 1500.
    temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    # Cloud parameters
    Kzz = np.ones_like(temperature) * 1e1 ** 6.5
    fsed = 2.
    sigma_lnorm = 1.05

    mass_fractions = {}
    mass_fractions['H2'] = 0.74 * np.ones_like(temperature)
    mass_fractions['He'] = 0.24 * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * np.ones_like(temperature)
    mass_fractions['CO_all_iso_HITEMP'] = 0.01 * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.000001 * np.ones_like(temperature)
    mass_fractions['Na_allard'] = 0.00001 * np.ones_like(temperature)
    mass_fractions['K_allard'] = 0.000001 * np.ones_like(temperature)

    mfr_cloud = np.zeros_like(temperature)
    mfr_cloud[pressures <= 3.] = 0.00005 * (pressures[pressures <= 3.] / 3.) ** fsed
    mass_fractions['Mg2SiO4(c)'] = mfr_cloud

    MMW = 2.33 * np.ones_like(temperature)

    mass_fractions['Mg2SiO4(c)'] = mfr_cloud

    atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                         kzz=Kzz, fsed=fsed, sigma_lnorm=sigma_lnorm, \
                         emission_geometry='non-isotropic', t_star=5778, \
                         r_star=nc.r_sun, semimajoraxis=0.05 * nc.AU, \
                         star_inclination_angle=30.)
    plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, \
             label='non-isotropic, 30 degrees', zorder=0)

    plt.plot(nc.c / atmosphere.frequencies / 1e-4, atmosphere.stellar_intensity * np.cos(30. / 180. * np.pi) * np.pi / 1e-6, \
             label=r'Stellar spectrum at TOA for $\mu_*={\rm cos}(30^\circ)$', alpha=0.6, \
             color='C2', linestyle=':')

    plt.legend(loc='best')
    plt.xlim([0.3, 15])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
    plt.show()

if case3:
    atmosphere = Radtrans(line_species=['H2O_HITEMP',
                                        'CH4',
                                        'CO2',
                                        'O3'],
                          rayleigh_species=['N2', 'O2'],
                          continuum_opacities=['N2-N2', 'N2-O2', 'O2-O2', 'CO2-CO2'],
                          wlen_bords_micron=[0.3, 15],
                          do_scat_emis=True)

    pressures = np.logspace(-6, 0, 100)
    atmosphere.setup_opa_structure(pressures)
    R_pl = nc.r_earth
    gravity = nc.G * (nc.m_earth) / R_pl ** 2

    # P-T parameters
    kappa_IR = 0.0009
    gamma = 0.01
    T_int = 250.
    T_equ = 220.
    temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    Kzz = np.ones_like(temperature) * 1e1 ** 6.5
    fsed = 2.
    sigma_lnorm = 1.05

    # Mean molecular weight of Earth's atmosphere
    MMW = 28.7 * np.ones_like(temperature)

    # Absorber mass fractions
    mass_fractions = {}
    mass_fractions['N2'] = 0.78 * 28. / MMW * np.ones_like(temperature)
    mass_fractions['O2'] = 0.21 * 32. / MMW * np.ones_like(temperature)
    mass_fractions['H2O_HITEMP'] = 0.001 * 18. / MMW * np.ones_like(temperature)
    mass_fractions['O3'] = 1e-7 * 48. / MMW * np.ones_like(temperature)
    mass_fractions['CO2'] = 0.0004 * 44. / MMW * np.ones_like(temperature)
    mass_fractions['CH4'] = 0.0001 * 16. / MMW * np.ones_like(temperature)

    for r in [0, 0.5, 1]:
        atmosphere.reflectance = r * np.ones_like(atmosphere.frequencies)

        atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW, \
                             kzz=Kzz, fsed=fsed, sigma_lnorm=sigma_lnorm, \
                             emission_geometry='planetary_ave', t_star=5778, \
                             r_star=nc.r_sun, semimajoraxis=nc.AU)

        plt.semilogy(nc.c / atmosphere.frequencies / 1e-4, atmosphere.flux / 1e-6, \
                     label='Surface Reflectance = ' + str(r), zorder=2)

    plt.semilogy(nc.c / atmosphere.frequencies / 1e-4, atmosphere.stellar_intensity / 4. * np.pi / 1e-6, \
                 label='Stellar spectrum at TOA', alpha=0.6)

    plt.legend(loc='best')
    plt.xlim([0.3, 15])
    plt.xscale('log')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
    plt.show()

