"""Stores useful physical functions.
"""
import numpy as np

import petitRADTRANS.physical_constants as cst


def b(temperature, nu):
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    """

    planck_function = 2. * cst.h * nu ** 3. / cst.c ** 2. / (np.exp(cst.h * nu / cst.kB / temperature) - 1.)

    return planck_function


def d_b_d_temperature(temperature, nu):
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature:
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    Returns:

    """
    planck_function = b(temperature, nu)
    planck_function /= np.exp(cst.h * nu / cst.kB / temperature) - 1.
    planck_function *= np.exp(cst.h * nu / cst.kB / temperature) * cst.h * nu / cst.kB / temperature ** 2.

    return planck_function


def doppler_shift(wavelength_0, velocity):
    """Calculate the Doppler-shifted wavelength for electromagnetic waves.

    A negative velocity means that the source is going toward the observer. A positive velocity means the source is
    going away from the observer.

    Args:
        wavelength_0: (cm) wavelength of the wave in the referential of the source
        velocity: (cm.s-1) velocity of the source relative to the observer

    Returns:
        (cm) the wavelength of the source as measured by the observer
    """
    return wavelength_0 * np.sqrt((1 + velocity / cst.c) / (1 - velocity / cst.c))


def get_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    mu_star = 0.
    angle_use = False

    if (mode != 'p') & (mode != 'd'):
        mu_star = float(mode)
        angle_use = True

    if mode_what == 'temp':
        if angle_use:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. * mu_star) ** 0.25
        elif mode == 'p':
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 4.) ** 0.25
        else:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 2.) ** 0.25
        return t_irr
    elif mode_what == 'dist':
        if angle_use:
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. * mu_star) / cst.au
        elif mode == 'p':
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 4.) / cst.au
        else:
            dist = np.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 2.) / cst.au
        return dist


def get_guillot_2010_temperature_profile(pressure, infrared_mean_opacity, gamma, gravity, intrinsic_temperature,
                                         equilibrium_temperature, redistribution_coefficient=0.25):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar).

    For this the temperature model of Guillot (2010) is used (his Equation 29).
    Source: https://doi.org/10.1051/0004-6361/200913396

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity:
            The infrared mean opacity in units of :math:`\\rm cm^2/s`.
        gamma:
            The ratio between the visual and infrared mean opacities.
        gravity:
            The planetary gravity at the given pressures in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature:
            The planetary intrinsic temperature (in units of K).
        equilibrium_temperature:
            The planetary equilibrium temperature (in units of K).
        redistribution_coefficient:
            The redistribution coefficient of the irradiance. A value of 1 corresponds to the substellar point, 1/2 for
            the day-side average and 1/4 for the global average.
    """
    # Estimate tau from eq. 24: m is the column mass, dm = rho * dz, dP / dz = -g * rho, so m = P / g
    tau = infrared_mean_opacity * pressure * 1e6 / gravity
    t_irr = equilibrium_temperature * 2.0 ** 0.5  # from eqs. 1 and 2

    temperature = (
        0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
        + 0.75 * t_irr ** 4. * redistribution_coefficient
        * (
            2. / 3.
            + 1. / gamma / 3. ** 0.5
            + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma) * np.exp(-gamma * tau * 3. ** 0.5)
        )
    ) ** 0.25

    return temperature


# TODO remove deprecated functions
def guillot_day(pressure, kappa_ir, gamma, grav, t_int, t_equ):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the day side of the planet.

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        kappa_ir (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        t_int (float):
            The planetary internal temperature (in units of K).
        t_equ (float):
            The planetary equilibrium temperature (in units of K).
    """
    return get_guillot_2010_temperature_profile(
        pressure=pressure,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravity=grav,
        intrinsic_temperature=t_int,
        equilibrium_temperature=t_equ,
        redistribution_coefficient=0.5
    )


def guillot_global(pressure, kappa_ir, gamma, grav, t_int, t_equ):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the whole planetary surface.

    Args:
        pressure:
            numpy array of floats, containing the input pressure in bars.
        kappa_ir (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        t_int (float):
            The planetary internal temperature (in units of K).
        t_equ (float):
            The planetary equilibrium temperature (in units of K).
    """
    return get_guillot_2010_temperature_profile(
        pressure=pressure,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravity=grav,
        intrinsic_temperature=t_int,
        equilibrium_temperature=t_equ,
        redistribution_coefficient=0.25
    )


def guillot_global_ret(pressure, delta, gamma, t_int, t_equ):
    """Global Guillot P-T formula with kappa/gravity replaced by delta."""
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    t_int = np.abs(t_int)
    t_equ = np.abs(t_equ)
    tau = pressure * 1e6 * delta
    t_irr = t_equ * np.sqrt(2.)
    temperature = (0.75 * t_int ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * np.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


def guillot_metallic_temperature_profile(pressures, gamma, surface_gravity,
                                         intrinsic_temperature, equilibrium_temperature, kappa_ir_z0,
                                         metallicity=None):
    """Get a Guillot temperature profile depending on metallicity.

    Args:
        pressures: (bar) pressures of the profile
        gamma: ratio between visual and infrated opacity
        surface_gravity: (cm.s-2) surface gravity
        intrinsic_temperature: (K) intrinsic temperature
        equilibrium_temperature: (K) equilibrium temperature
        kappa_ir_z0: (cm2.s-1) infrared opacity
        metallicity: ratio of heavy elements abundance over H abundance with respect to the solar ratio

    Returns:
        temperatures: (K) the temperature at each pressures of the atmosphere
    """
    if metallicity is not None:
        kappa_ir = kappa_ir_z0 * metallicity
    else:
        kappa_ir = kappa_ir_z0

    temperatures = guillot_global(
        pressure=pressures,
        kappa_ir=kappa_ir,
        gamma=gamma,
        grav=surface_gravity,
        t_int=intrinsic_temperature,
        t_equ=equilibrium_temperature
    )

    return temperatures


def guillot_modif(pressure, delta, gamma, t_int, t_equ, ptrans, alpha):
    """Modified Guillot P-T formula"""
    return guillot_global_ret(
        pressure,
        np.abs(delta),
        np.abs(gamma),
        np.abs(t_int), np.abs(t_equ)
    ) * (1. - alpha * (1. / (1. + pressure / ptrans)))


def hz2um(frequency):
    """Convert frequencies into wavelengths

    Args:
        frequency: (Hz) the frequency to convert

    Returns:
        (um) the corresponding wavelengths
    """
    return cst.c / frequency * 1e4  # cm to um


def isothermal(pressures, temperature):
    # TODO only to temporarily fix methods, change name later
    return np.ones(pressures.size) * temperature


def pt_ret_model(rad_trans_params):
    """
    Self-luminous retrieval P-T model.
    # TODO fix docstring
    Args:
        T3 : np.array([t1, t2, t3])
            temperature points to be added on top
            radiative Eddington structure (above tau = 0.1).
            Use spline interpolation, t1 < t2 < t3 < tconnect as prior.
        delta : float
            proportionality factor in tau = delta * press_cgs**alpha
        alpha : float
            power law index in tau = delta * press_cgs**alpha
            For the tau model: use proximity to kappa_rosseland photosphere
            as prior.
        tint : float
            internal temperature of the Eddington model
        press : np.ndarray
            input pressure profile in bar
        conv : bool
            enforce convective adiabat yes/no
        CO : float
            C/O for the nabla_ad interpolation
        FeH : float
            metallicity for the nabla_ad interpolation
    Returns:
        Tret : np.ndarray
            The temperature as a function of atmospheric pressure.
    """
    import copy as cp
    from scipy.interpolate import interp1d, CubicSpline
    import petitRADTRANS.poor_mans_nonequ_chem.poor_mans_nonequ_chem as pm

    t3, delta, alpha, tint, press, feh, co_ratio, conv = rad_trans_params
    # Go grom bar to cgs
    press_cgs = press * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * tint ** 4. * (2. / 3. + tau)) ** 0.25

    ab = pm.interpol_abundances(co_ratio * np.ones_like(tedd),
                                feh * np.ones_like(tedd),
                                tedd,
                                press)

    nabla_ad = ab['nabla_ad']

    tfinal = None  # TODO tmp fix for reference before assignment
    tret = None  # TODO tmp fix for reference before assignment

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd)) / np.diff(np.log(press_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:] + nab_rad[:-1]) / 2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # TODO: Check remains convective and convergence
        for i in range(10):
            if i == 0:
                t_take = cp.copy(tedd)
            else:
                t_take = cp.copy(tfinal)  # TODO reference before assignment

            ab = pm.interpol_abundances(co_ratio * np.ones_like(t_take),
                                        feh * np.ones_like(t_take),
                                        t_take,
                                        press)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[:-1]) / 2.
            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index] * np.mean(np.diff(np.log(press_cgs)))
            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])
            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew) + tstart)

            # Add upper radiative and
            # lower conective part into one single array
            tfinal = cp.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take - tfinal) / t_take) < 0.01:
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(tau):
        # Returns the pressure at a given tau, in cgs
        return (tau / delta) ** (1. / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                             np.log10(p_bot_spline),
                                             4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = 10 ** np.arange(np.log10(p_bot_spline),
                                                  np.log10(press_cgs[-1]),
                                                  np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                             np.log10(p_bot_spline),
                                             7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal, kind='cubic')
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low)) - 1)] = t3
            # t_support[:3] = tfintp(support_points_low)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convectice solution)
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(press_cgs, tret, kind='cubic')  # TODO possible reference before assignment
            t_support[:(int(len(support_points_low)) - 1)] = \
                tfintp1(support_points[:(int(len(support_points_low)) - 1)])

            tfintp = interp1d(press_cgs, tfinal)
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(press_cgs))

    tret[tret < 0.0] = 10.0
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)


def radiosity_erg_cm2radiosity_erg_hz(radiosity_erg_cm, wavelength):
    """
    Convert a radiosity from erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2.sr-1/Hz at a given wavelength.
    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
        radiosity_erg_hz = radiosity_erg_cm * d[cm]/d[Hz]
        => radiosity_erg_hz = radiosity_erg_cm * wavelength**2 / c

    Args:
        radiosity_erg_cm: (erg.s-1.cm-2.sr-1/cm)
        wavelength: (cm)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    return radiosity_erg_cm * wavelength ** 2 / cst.c


def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
    """Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
        radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
        => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

    Args:
        radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    # TODO move to physics
    return radiosity_erg_hz * frequency ** 2 / cst.c


def radiosity2irradiance(spectral_radiosity, source_radius, target_distance):
    """Calculate the spectral irradiance of a spherical source on a target from its spectral radiosity.

    Args:
        spectral_radiosity: (M.L-1.T-3) spectral radiosity of the source
        source_radius: (L) radius of the spherical source
        target_distance: (L) distance from the source to the target

    Returns:
        The irradiance of the source on the target (M.L-1.T-3).
    """
    return spectral_radiosity * (source_radius / target_distance) ** 2
