import jax.numpy as jnp
from jax import jit, config
from jax.typing import ArrayLike

def temperature_profile_function_guillot(
    pressures: ArrayLike,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: ArrayLike,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    redistribution_coefficient: float = 0.25
) -> ArrayLike:
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
                    + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma) * jnp.exp(-gamma * tau * 3. ** 0.5)
            )
    ) ** 0.25

    return temperature


def temperature_profile_function_guillot_dayside(
    pressures: ArrayLike,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: ArrayLike,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> ArrayLike:
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
            The ratio between the visual and infrared opacity.
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


def temperature_profile_function_guillot_global(
    pressures: ArrayLike,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: ArrayLike,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> ArrayLike:
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
            The ratio between the visual and infrared opacity.
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


def temperature_profile_function_guillot_global_ret(
    pressures: ArrayLike,
    delta: float,
    gamma: float,
    intrinsic_temperature: float,
    equilibrium_temperature: float
) -> ArrayLike:
    """Global Guillot P-T formula with kappa/gravity replaced by delta."""
    # TODO what is delta?
    delta = jnp.abs(delta)
    gamma = jnp.abs(gamma)
    intrinsic_temperature = jnp.abs(intrinsic_temperature)
    equilibrium_temperature = jnp.abs(equilibrium_temperature)
    tau = pressures * 1e6 * delta
    t_irr = equilibrium_temperature * jnp.sqrt(2.)
    temperature = (0.75 * intrinsic_temperature ** 4. * (2. / 3. + tau)
                   + 0.75 * t_irr ** 4. / 4.
                   * (2. / 3. + 1. / gamma / 3. ** 0.5
                      + (gamma / 3. ** 0.5 - 1. / 3. ** 0.5 / gamma)
                      * jnp.exp(-gamma * tau * 3. ** 0.5))) ** 0.25
    return temperature


def temperature_profile_function_guillot_metallic(
    pressures: ArrayLike,
    gamma: float,
    reference_gravity: ArrayLike,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    infrared_mean_opacity_solar_metallicity: float,
    metallicity: [float, None] = None
) -> ArrayLike:
    """Get a Guillot temperature profile depending on metallicity.

    Args:
        pressures: (bar) pressures of the profile
        gamma: ratio between visual and infrared opacity
        reference_gravity: (cm.s-2) surface gravity
        intrinsic_temperature: (K) intrinsic temperature
        equilibrium_temperature: (K) equilibrium temperature
        infrared_mean_opacity_solar_metallicity:
            (cm2.s-1) infrared mean opacity for a solar metallicity (Z = 1) atmosphere
        metallicity: ratio of heavy elements abundance over H abundance with respect to the solar ratio

    Returns:
        temperatures: (K) the temperature at each pressures of the atmosphere
    """
    if metallicity is not None:
        kappa_ir = infrared_mean_opacity_solar_metallicity * metallicity
    else:
        kappa_ir = infrared_mean_opacity_solar_metallicity

    temperatures = temperature_profile_function_guillot_global(
        pressures=pressures,
        infrared_mean_opacity=kappa_ir,
        gamma=gamma,
        gravities=reference_gravity,
        intrinsic_temperature=intrinsic_temperature,
        equilibrium_temperature=equilibrium_temperature
    )

    return temperatures


def temperature_profile_function_guillot_modif(
    pressures: ArrayLike,
    delta: float,
    gamma: float,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    ptrans: float,
    alpha: float
) -> ArrayLike:
    """Modified Guillot P-T formula"""
    # TODO how is it modified? Why for?
    return temperature_profile_function_guillot_global_ret(
        pressures,
        jnp.abs(delta),
        jnp.abs(gamma),
        jnp.abs(intrinsic_temperature), jnp.abs(equilibrium_temperature)
    ) * (1. - alpha * (1. / (1. + pressures / ptrans)))