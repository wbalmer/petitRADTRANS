import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from interpax import interp1d


def dtdp_temperature_profile(
        pressures: ArrayLike,
        num_layer: int,
        layer_pt_slopes: ArrayLike,
        t_bottom: float,
        top_of_atmosphere_pressure: float = -3,
        bottom_of_atmosphere_pressure: float = 3
) -> Array:
    """
    This function takes the temperature gradient at a set number of spline points and interpolates a temperature
    profile as a function of pressure. Based on the method described in Zhang et al. (2023).

    Args:
        pressures: The pressure array in bar.
        num_layer: The number of layers. Default for covering 10^3 to 10^-3 bar as in Zhang 2023 is 6.
            To cover 10^3 to 10^-6 bar 10 is recommended.
        layer_pt_slopes: The temperature gradient (d(ln T) / d(ln P)) at the spline points.
        t_bottom: The temperature at the bottom of the atmosphere.
        top_of_atmosphere_pressure: Minimum atmospheric pressure in log bar. If the pressure array extends
            beyond this value, the temperature structure at lower pressures will be isothermal.
        bottom_of_atmosphere_pressure: Maximum atmospheric pressure in log bar.

    Returns:
        The temperature profile in K.
    """
    id_sub = jnp.where(pressures >= 10**top_of_atmosphere_pressure)
    p_use_sub = pressures[id_sub]
    num_sub = len(p_use_sub)

    # 1.3 pressures of layers
    layer_pressures = jnp.logspace(
        top_of_atmosphere_pressure,
        bottom_of_atmosphere_pressure,
        int(num_layer)
    )

    # 1.4 assemble the P-T slopes for these layers
    # for index in range(num_layer):
    #    layer_pt_slopes[index] = parameters['PTslope_%d'%(num_layer - index)].value
    # 1.5 interpolate the P-T slopes to compute slopes for all layers
    interp_func = interp1d(
        jnp.log10(layer_pressures),
        layer_pt_slopes,
        'quadratic'
    )
    pt_slopes_sub = interp_func(jnp.log10(p_use_sub))
    # 1.6 compute temperatures
    temperatures_sub = jnp.ones(num_sub) * jnp.nan
    temperatures_sub = temperatures_sub.at[-1].set(t_bottom)

    # Note: higher index is lower pressure.
    for index in range(1, num_sub):
        temperatures_sub = temperatures_sub.at[-1 - index].set(jnp.exp(
            jnp.log(temperatures_sub[-index]) - pt_slopes_sub[-index]
            * (jnp.log(p_use_sub[-index]) - jnp.log(p_use_sub[-1 - index]))
        ))

    # 1.7 isothermal in the remaining region, i.e., upper atmosphere
    temperatures = jnp.ones_like(pressures) * temperatures_sub[0]
    temperatures = temperatures.at[id_sub].set(temperatures_sub)
    return temperatures
