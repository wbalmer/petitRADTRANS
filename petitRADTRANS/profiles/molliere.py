import jax.numpy as jnp
from jax import jit, config
from jax.typing import ArrayLike

def temperature_profile_function_ret_model(rad_trans_params):
    """
    Self-luminous retrieval P-T model.
    # TODO find better name
    Args:
        t3 : jnp.array([t1, t2, t3])
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
        press : jnp.ndarray
            input pressure profile in bar
        conv : bool
            enforce convective adiabat yes/no
        co_ratio : float
            C/O for the nabla_ad interpolation
        metallicity : float
            metallicity for the nabla_ad interpolation
    Returns:
        Tret : jnp.ndarray
            The temperature as a function of atmospheric pressure.
    """
    import copy
    from scipy.interpolate import CubicSpline
    from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table

    t3, delta, alpha, tint, press, feh, co_ratio, conv = rad_trans_params
    # Go grom bar to cgs
    press_cgs = press * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * tint ** 4. * (2. / 3. + tau)) ** 0.25

    ab, _, nabla_ad = (
        pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
            co_ratio * jnp.ones_like(tedd),
            feh * jnp.ones_like(tedd),
            tedd,
            press,
            full=True
        )
    )

    tfinal = None  # TODO tmp fix for reference before assignment
    tret = None  # TODO tmp fix for reference before assignment

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = jnp.diff(jnp.log(tedd)) / jnp.diff(jnp.log(press_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = jnp.ones_like(tedd)
        nabla_rad = nabla_rad.at[0].set(nab_rad[0])
        nabla_rad = nabla_rad.at[-1].set(nab_rad[-1])
        nabla_rad = nabla_rad.at[1:-1].set((nab_rad[1:] + nab_rad[:-1]) / 2.)

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # TODO: Check remains convective and convergence
        for i in range(10):
            if i == 0:
                t_take = copy.copy(tedd)
            else:
                t_take = copy.copy(tfinal)  # TODO reference before assignment

            ab, _, nabla_ad = (
                pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                    co_ratio * jnp.ones_like(t_take),
                    feh * jnp.ones_like(t_take),
                    t_take,
                    press,
                    full=True
                )
            )

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean = nabla_ad_mean.at[1:].set((nabla_ad[1:] + nabla_ad[:-1]) / 2.)
            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index] * jnp.mean(jnp.diff(jnp.log(press_cgs)))
            # What is the last radiative temperature?
            tstart = jnp.log(t_take[~conv_index][-1])
            # Integrate and translate to temperature from log(temperature)
            tnew = jnp.exp(jnp.cumsum(tnew) + tstart)

            # Add upper radiative and
            # lower convective part into one single array
            tfinal = copy.copy(t_take)
            tfinal = tfinal.at[conv_index].set(tnew)

            if jnp.max(jnp.abs(t_take - tfinal) / t_take) < 0.01:
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(_tau):
        # Returns the pressure at a given tau, in cgs
        return (_tau / delta) ** (1. / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = jnp.logspace(jnp.log10(press_cgs[0]),
                                             jnp.log10(p_bot_spline),
                                             4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = 10 ** jnp.arange(jnp.log10(p_bot_spline),
                                                  jnp.log10(press_cgs[-1]),
                                                  jnp.diff(jnp.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = jnp.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points = support_points.at[:4].set(support_points_low)
            support_points = support_points.at[4:].set(support_points_high[1:])

        else:
            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = jnp.logspace(
                jnp.log10(press_cgs[0]),
                jnp.log10(p_bot_spline),
                7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = jnp.logspace(jnp.log10(p_bot_spline), jnp.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = jnp.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points = support_points.at[:7].set(support_points_low)
            support_points = support_points.at[7:].set(support_points_high[1:])

        # Define the temperature values at the node points.
        t_support = jnp.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal, kind='cubic')
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support = t_support.at[int(len(support_points_low)) - 1].set(tfintp(p_bot_spline))
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support = t_support.at[:(int(len(support_points_low)) - 1)].set(t3)
            # t_support[:3] = tfintp(support_points_low)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convective solution)
            t_support = t_support.at[int(len(support_points_low)):].set(
                tfintp(support_points[(int(len(support_points_low))):])
            )
        else:
            tfintp1 = interp1d(press_cgs, tret, kind='cubic')  # TODO possible reference before assignment
            t_support = t_support.at[:(int(len(support_points_low)) - 1)].set(
                tfintp1(support_points[:(int(len(support_points_low)) - 1)])
            )
            tfintp = interp1d(press_cgs, tfinal)
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support - t_support.at[int(len(support_points_low)) - 1].set(tfintp(p_bot_spline))
            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support = t_support.at[int(len(support_points_low)):].set(
                tfintp(support_points[(int(len(support_points_low))):])
            )
        # Make the temperature spline interpolation to be returned to the user
        cs = CubicSpline(jnp.log10(support_points), t_support)
        tret = cs(jnp.log10(press_cgs))

    tret = jnp.where(tret < 0.0, 1.0, tret)
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)