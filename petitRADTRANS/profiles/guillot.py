import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing import Optional


def temperature_profile_function_guillot(
    pressures: ArrayLike,
    infrared_mean_opacity: float,
    gamma: float,
    gravities: ArrayLike,
    intrinsic_temperature: float,
    equilibrium_temperature: float,
    redistribution_coefficient: float = 0.25
) -> Array:
    """ Returns a temperature array, in units of K, of the same dimensions as the pressure P (in bar).

    For this the temperature model of Guillot (2010) is used (his Equation 29).
    Source: https://doi.org/10.1051/0004-6361/200913396

    Args:
        pressures: Array of pressures in bar.
        infrared_mean_opacity: The infrared mean opacity in units of cm^2/g.
        gamma: The ratio between the visual and infrared mean opacities.
        gravities: The planetary gravity at the given pressures in units of cm/s^2.
        intrinsic_temperature: The planetary intrinsic temperature in K.
        equilibrium_temperature: The planetary equilibrium temperature in K.
        redistribution_coefficient: The redistribution coefficient of the irradiance. A value of 1 corresponds to the
            substellar point, 1/2 for the day-side average and 1/4 for the global average.
    Returns:
        The temperature profile in K.
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
) -> Array:
    """ Returns a dayside temperature array, in units of K, using the Guillot (2010) model.

    This is a wrapper for `temperature_profile_function_guillot` with `redistribution_coefficient` set to 0.5,
    which corresponds to an average over the day side of the planet.

    Args:
        pressures: Array of pressures in bar.
        infrared_mean_opacity: The infrared mean opacity in units of cm^2/g.
        gamma: The ratio between the visual and infrared mean opacities.
        gravities: The planetary gravity at the given pressures in units of cm/s^2.
        intrinsic_temperature: The planetary intrinsic temperature in K.
        equilibrium_temperature: The planetary equilibrium temperature in K.
    Returns:
        The temperature profile in K.
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
) -> Array:
    """ Returns a global average temperature array, in units of K, using the Guillot (2010) model.

    This is a wrapper for `temperature_profile_function_guillot` with `redistribution_coefficient` set to 0.25,
    which corresponds to an average over the whole planetary surface.

    Args:
        pressures: Array of pressures in bar.
        infrared_mean_opacity: The infrared mean opacity in units of cm^2/g.
        gamma: The ratio between the visual and infrared mean opacities.
        gravities: The planetary gravity at the given pressures in units of cm/s^2.
        intrinsic_temperature: The planetary intrinsic temperature in K.
        equilibrium_temperature: The planetary equilibrium temperature in K.
    Returns:
        The temperature profile in K.
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
) -> Array:
    """Global Guillot P-T formula for retrievals, with kappa/gravity replaced by delta.

    This function is a variation of the Guillot (2010) temperature profile, where the ratio of the
    mean infrared opacity to gravity (kappa/g) is replaced by a single parameter, delta.
    This is useful for retrieval applications where gravity is fixed and opacity is a parameter.

    Args:
        pressures: Array of pressures in bar.
        delta: Proportionality factor for optical depth, equal to kappa_ir / g, where kappa_ir is the
               infrared mean opacity (cm^2/g) and g is the gravity (cm/s^2).
        gamma: The ratio between the visual and infrared mean opacities.
        intrinsic_temperature: The planetary intrinsic temperature in K.
        equilibrium_temperature: The planetary equilibrium temperature in K.
    Returns:
        The temperature profile in K.
    """
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
    metallicity: Optional[float] = None
) -> Array:
    """Get a Guillot temperature profile depending on metallicity.

    This function calculates a global average Guillot temperature profile, where the infrared mean
    opacity is scaled with metallicity.

    Args:
        pressures: Array of pressures in bar.
        gamma: Ratio between visual and infrared opacity.
        reference_gravity: Surface gravity in cm/s^2.
        intrinsic_temperature: Intrinsic temperature in K.
        equilibrium_temperature: Equilibrium temperature in K.
        infrared_mean_opacity_solar_metallicity: Infrared mean opacity for a solar metallicity (Z=1)
            atmosphere, in cm^2/g.
        metallicity: Ratio of heavy elements abundance over H abundance with respect to the solar ratio.
            If None, solar metallicity is assumed.
    Returns:
        The temperature profile in K.
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
) -> Array:
    """Modified Guillot P-T formula for retrievals.

    This function uses the `temperature_profile_function_guillot_global_ret` and applies a
    modifying factor to the temperature profile. The modification is controlled by `alpha`
    and `ptrans`, which can shape the profile around a transition pressure.

    Args:
        pressures: Array of pressures in bar.
        delta: Proportionality factor for optical depth, equal to kappa_ir / g.
        gamma: The ratio between the visual and infrared mean opacities.
        intrinsic_temperature: The planetary intrinsic temperature in K.
        equilibrium_temperature: The planetary equilibrium temperature in K.
        ptrans: Transition pressure in bar, where the modification is centered.
        alpha: A parameter controlling the strength and shape of the temperature modification.
    Returns:
        The modified temperature profile in K.
    """
    return temperature_profile_function_guillot_global_ret(
            pressures,
            jnp.abs(delta),
            jnp.abs(gamma),
            jnp.abs(intrinsic_temperature),
            jnp.abs(equilibrium_temperature)
            ) * (1. - alpha * (1. / (1. + pressures / ptrans)))
