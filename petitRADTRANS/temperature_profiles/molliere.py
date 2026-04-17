import jax.numpy as jnp
from jax import lax
from interpax import interp1d


# molliere2020
def molliere_2020_temperature_profile(rad_trans_params):
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
    from petitRADTRANS.chemistry.pre_calculated_chemistry import (
        pre_calculated_equilibrium_chemistry_table,
    )

    t3, delta, alpha, tint, press, feh, co_ratio, conv = rad_trans_params
    # Go from bar to cgs
    press_cgs = press * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs**alpha

    # This is the eddington temperature
    tedd = (3.0 / 4.0 * tint**4.0 * (2.0 / 3.0 + tau)) ** 0.25

    ab, _, nabla_ad = (
        pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
            co_ratio * jnp.ones_like(tedd),
            feh * jnp.ones_like(tedd),
            tedd,
            press,
            full=True,
        )
    )

    # Enforce convective adiabat
    def conv_branch(_):
        # Calculate the current, radiative temperature gradient
        nab_rad = jnp.diff(jnp.log(tedd)) / jnp.diff(jnp.log(press_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = jnp.ones_like(tedd)
        nabla_rad = nabla_rad.at[0].set(nab_rad[0])
        nabla_rad = nabla_rad.at[-1].set(nab_rad[-1])
        nabla_rad = nabla_rad.at[1:-1].set((nab_rad[1:] + nab_rad[:-1]) / 2.0)

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # Precompute loop constants
        dlp = jnp.mean(jnp.diff(jnp.log(press_cgs)))
        first_conv_idx = jnp.argmax(conv_index)
        boundary_idx = jnp.clip(first_conv_idx - 1, 0, tedd.shape[0] - 1)

        def cond_fn(state):
            i, t_current, converged = state
            return (i < 10) & ~converged

        def body_fn(state):
            i, t_current, _ = state

            _, _, nabla_ad_iter = (
                pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                    co_ratio * jnp.ones_like(t_current),
                    feh * jnp.ones_like(t_current),
                    t_current,
                    press,
                    full=True,
                )
            )

            # Average nabla_ad between adjacent layers
            nabla_ad_mean = nabla_ad_iter.at[1:].set(
                (nabla_ad_iter[1:] + nabla_ad_iter[:-1]) / 2.0
            )

            # Starting log-temperature at the radiative-convective boundary
            tstart = jnp.log(t_current[boundary_idx])

            # Cumulative log-temperature increment (convective layers only)
            cumsum_conv = jnp.cumsum(jnp.where(conv_index, nabla_ad_mean * dlp, 0.0))

            # New temperature: adiabatic in convective zone, unchanged elsewhere
            t_new = jnp.where(conv_index, jnp.exp(cumsum_conv + tstart), t_current)

            converged = jnp.max(jnp.abs(t_current - t_new) / t_current) < 0.01
            return i + 1, t_new, converged

        _, tfinal, _ = lax.while_loop(
            cond_fn, body_fn, (jnp.array(0), tedd, jnp.array(False))
        )
        return tfinal

    def non_conv_branch(_):
        return tedd

    tfinal = lax.cond(conv, conv_branch, non_conv_branch, None)

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(_tau):
        # Returns the pressure at a given tau, in cgs
        return (_tau / delta) ** (1.0 / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    tret = jnp.zeros_like(press_cgs)

    for i_intp in range(2):

        if i_intp == 0:
            # 4 support nodes spanning the low-pressure (above tau=0.1) region
            support_points_low = jnp.logspace(
                jnp.log10(press_cgs[0]), jnp.log10(p_bot_spline), 4
            )

            # 4 support nodes spanning the high-pressure (below tau=0.1) region
            # (fixed count avoids dynamic jnp.arange shape in JIT)
            support_points_high = jnp.logspace(
                jnp.log10(p_bot_spline), jnp.log10(press_cgs[-1]), 4
            )

        else:
            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = jnp.logspace(
                jnp.log10(press_cgs[0]), jnp.log10(p_bot_spline), 7
            )

            # Create the pressure coordinates for the spline support nodes at high pressure
            support_points_high = jnp.logspace(
                jnp.log10(p_bot_spline), jnp.log10(press_cgs[-1]), 7
            )

        # Combine into one support node array, don't add the p_bot_spline point twice.
        support_points = jnp.concatenate([support_points_low, support_points_high[1:]])

        n_low = support_points_low.shape[0]  # static: 4 or 7

        # Define the temperature values at the node points.
        t_support = jnp.zeros_like(support_points)

        if i_intp == 0:
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support = t_support.at[n_low - 1].set(
                interp1d(
                    jnp.atleast_1d(p_bot_spline), press_cgs, tfinal, method="cubic"
                )[0]
            )
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support = t_support.at[: n_low - 1].set(t3)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convective solution)
            t_support = t_support.at[n_low:].set(
                interp1d(support_points[n_low:], press_cgs, tfinal, method="cubic")
            )
        else:
            # Low-pressure nodes interpolated from the first-pass spline
            t_support = t_support.at[: n_low - 1].set(
                interp1d(support_points[: n_low - 1], press_cgs, tret, method="cubic")
            )
            # The temperature at p_bot_spline (from the radiative-convective solution)
            t_support = t_support.at[n_low - 1].set(
                interp1d(
                    jnp.atleast_1d(p_bot_spline), press_cgs, tfinal, method="linear"
                )[0]
            )
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convective solution)
            t_support = t_support.at[n_low:].set(
                interp1d(support_points[n_low:], press_cgs, tfinal, method="linear")
            )

        # Make the temperature spline interpolation to be returned to the user
        tret = interp1d(
            jnp.log10(press_cgs), jnp.log10(support_points), t_support, method="cubic"
        )

    tret = jnp.where(tret < 0.0, 1.0, tret)
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)
