"""Stores useful physical functions.
"""
import numpy as np

import petitRADTRANS.physical_constants as cst


def compute_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    # TODO find better name
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


def flux_cm2flux_hz(flux_cm, wavelength):
    """
    Convert a flux from [flux units]/cm to [flux units]/Hz at a given wavelength.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_hz = flux_cm * d[cm]/d[Hz]
        => flux_hz = flux_cm * wavelength**2 / c

    Args:
        flux_cm: ([flux units]/cm)
        wavelength: (cm)

    Returns:
        ([flux units]/Hz) the radiosity in converted units
    """
    return flux_cm * wavelength ** 2 / cst.c


def flux_hz2flux_cm(flux_hz, frequency):
    """Convert a flux from [flux units]/Hz to [flux units]/cm at a given frequency.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_cm = flux_hz * d[Hz]/d[cm]
        => flux_cm = flux_hz * frequency**2 / c

    Args:
        flux_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    return flux_hz * frequency ** 2 / cst.c


def flux2irradiance(flux, source_radius, target_distance):
    """Calculate the spectral irradiance of a spherical source on a target from its flux (spectral radiosity).

    Args:
        flux: (M.L-1.T-3) flux of the source
        source_radius: (L) radius of the spherical source
        target_distance: (L) distance from the source to the target

    Returns:
        The irradiance of the source on the target (M.L-1.T-3).
    """
    return flux * (source_radius / target_distance) ** 2


def hz2um(frequency):
    """Convert frequencies into wavelengths

    Args:
        frequency: (Hz) the frequency to convert

    Returns:
        (um) the corresponding wavelengths
    """
    return cst.c / frequency * 1e4  # cm to um


def planck_function(temperature, nu):
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    """

    _planck_function = 2. * cst.h * nu ** 3. / cst.c ** 2. / (np.exp(cst.h * nu / cst.kB / temperature) - 1.)

    return _planck_function


def planck_function_temperature_derivative(temperature, nu):
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.
    # TODO unused?

    Args:
        temperature:
            Temperature in K.
        nu:
            Array containing the frequency in Hz.
    Returns:

    """
    _planck_function = planck_function(temperature, nu)
    _planck_function /= np.exp(cst.h * nu / cst.kB / temperature) - 1.
    _planck_function *= np.exp(cst.h * nu / cst.kB / temperature) * cst.h * nu / cst.kB / temperature ** 2.

    return _planck_function


def temperature_profile_function_guillot(pressures, infrared_mean_opacity, gamma, gravities, intrinsic_temperature,
                                         equilibrium_temperature, redistribution_coefficient=0.25):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar).

    For this the temperature model of Guillot (2010) is used (his Equation 29).
    Source: https://doi.org/10.1051/0004-6361/200913396

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity:
            The infrared mean opacity in units of :math:`\\rm cm^2/s`.
        gamma:
            The ratio between the visual and infrared mean opacities.
        gravities:
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
    tau = infrared_mean_opacity * pressures * 1e6 / gravities
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


def temperature_profile_function_guillot_dayside(pressures, infrared_mean_opacity, gamma, gravities,
                                                 intrinsic_temperature, equilibrium_temperature):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the day side of the planet.

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        gravities (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature (float):
            The planetary internal temperature (in units of K).
        equilibrium_temperature (float):
            The planetary equilibrium temperature (in units of K).
    """
    return temperature_profile_function_guillot(
        pressures=pressures,
        infrared_mean_opacity=infrared_mean_opacity,
        gamma=gamma,
        gravities=gravities,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature,
        redistribution_coefficient=0.5
    )


def temperature_profile_function_guillot_global(pressures, infrared_mean_opacity, gamma, gravities,
                                                intrinsic_temperature, equilibrium_temperature):
    """ Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29), in the case of averaging the flux over the whole planetary surface.

    Args:
        pressures:
            numpy array of floats, containing the input pressure in bars.
        infrared_mean_opacity (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        gravities (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        intrinsic_temperature (float):
            The planetary internal temperature (in units of K).
        equilibrium_temperature (float):
            The planetary equilibrium temperature (in units of K).
    """
    return temperature_profile_function_guillot(
        pressures=pressures,
        infrared_mean_opacity=infrared_mean_opacity,
        gamma=gamma,
        gravities=gravities,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature,
        redistribution_coefficient=0.25
    )


def temperature_profile_function_guillot_global_ret(pressures, delta, gamma,
                                                    intrinsic_temperature, equilibrium_temperature):
    """Global Guillot P-T formula with kappa/gravity replaced by delta."""
    # TODO what is delta?
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    intrinsic_temperature = np.abs(intrinsic_temperature)
    equilibrium_temperature = np.abs(equilibrium_temperature)
    tau = pressures * 1e6 * delta
    t_irr = equilibrium_temperature * np.sqrt(2.)
    temperature = (0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * np.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


def temperature_profile_function_guillot_metallic(pressures, gamma, surface_gravity,
                                                  intrinsic_temperature, equilibrium_temperature,
                                                  infrared_mean_opacity_solar_matallicity, metallicity=None):
    """Get a Guillot temperature profile depending on metallicity.

    Args:
        pressures: (bar) pressures of the profile
        gamma: ratio between visual and infrated opacity
        surface_gravity: (cm.s-2) surface gravity
        intrinsic_temperature: (K) intrinsic temperature
        equilibrium_temperature: (K) equilibrium temperature
        infrared_mean_opacity_solar_matallicity:
            (cm2.s-1) infrared mean opacity for a solar metallicity (Z = 1) atmosphere
        metallicity: ratio of heavy elements abundance over H abundance with respect to the solar ratio

    Returns:
        temperatures: (K) the temperature at each pressures of the atmosphere
    """
    if metallicity is not None:
        kappa_ir = infrared_mean_opacity_solar_matallicity * metallicity
    else:
        kappa_ir = infrared_mean_opacity_solar_matallicity

    temperatures = temperature_profile_function_guillot_global(
        pressures=pressures,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravities=surface_gravity,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature
    )

    return temperatures


def temperature_profile_function_guillot_modif(pressures, delta, gamma,
                                               intrinsic_temperature, equilibrium_temperature, ptrans, alpha):
    """Modified Guillot P-T formula"""
    # TODO how is it modified? Why for?
    return temperature_profile_function_guillot_global_ret(
        pressures,
        np.abs(delta),
        np.abs(gamma),
        np.abs(intrinsic_temperature), np.abs(equilibrium_temperature)
    ) * (1. - alpha * (1. / (1. + pressures / ptrans)))


def temperature_profile_function_isothermal(pressures, temperature):
    # TODO only to temporarily fix methods, change name later
    return np.ones(pressures.size) * temperature


def temperature_profile_function_ret_model(rad_trans_params):
    """
    Self-luminous retrieval P-T model.
    # TODO fix docstring
    # TODO find better name
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
    import petitRADTRANS.chemistry.pre_calculated_chemistry as pm

    t3, delta, alpha, tint, press, feh, co_ratio, conv = rad_trans_params
    # Go grom bar to cgs
    press_cgs = press * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * tint ** 4. * (2. / 3. + tau)) ** 0.25

    ab = pm.interpolate_mass_fractions_chemical_table(co_ratio * np.ones_like(tedd),
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

            ab = pm.interpolate_mass_fractions_chemical_table(co_ratio * np.ones_like(t_take),
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
