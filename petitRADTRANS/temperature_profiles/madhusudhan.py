import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def madhusudhan_seager_temperature_profile(
    pressures: ArrayLike,
    log_pressure_points: tuple[float],
    temperature_reference: float,
    alpha_points: tuple[float],
    beta_points: tuple[float],
) -> Array:
    """
    Calculate temperatures based on the Madhusudhan and Seager (2009) parameterization.

    This function computes temperatures using the Madhu and Seager (2009) parameterization
    for a given set of pressure values, pressure breakpoints, temperature breakpoints,
    alpha values, and beta values.

    Based off of the POSEIDON implementation:
    https://github.com/MartianColonist/POSEIDON/blob/main/POSEIDON/atmosphere.py

    Parameters:
        pressures : (numpy.ndarray)
            An array of pressure values (in bar) at which to calculate temperatures.
        log_pressure_points : (list)
            A list of log10 pressure breakpoints defining different temperature regimes.
            The zeroth element is the minimum pressure, should be log10(press[0]).
            The first element is the 1-2 boundary
            The second element is the level of the inversion
            The third element is the 2-3 boundary.
            The fourth element is the pressure at which the temperature is set (P_set).
        temperature_reference : (float)
            A temperature at log_pressure_points[4] used to constrain the temperature profile.
        alpha_points : (list)
            A list of alpha values used in the parameterization for different regimes.
            Must have length 2.
        beta_points : (list)
            A list of beta values used in the parameterization for different regimes.
            By default, b[0] == b[1] == 0.5, unclear how well this will work if these aren't used!
            Must have length 2.

    Returns:
        temperatures : (numpy.ndarray)
            An array of calculated temperatures (in K) corresponding to the input pressure values.

    Reference:
    - Madhusudhan, N., & Seager, S. (2009). A Temperature and Abundance Retrieval Method for Exoplanet Atmospheres.
      The Astrophysical Journal, 707(1), 24-39. https://doi.org/10.1088/0004-637X/707/1/24
    """
    temperatures = jnp.zeros_like(pressures)

    # Set up masks for the different temperature regions
    pressure_points = 10 ** jnp.array(log_pressure_points)

    # Change log10 pressure points to log pressure points as used in Madhusudhan and Seager (2009)
    nat_log_pressure_points = jnp.log(pressure_points)

    mask_1 = pressures < pressure_points[1]
    mask_2 = (pressures >= pressure_points[1]) & (pressures < pressure_points[3])
    mask_3 = pressures >= pressure_points[3]

    # Find index of pressure closest to the set pressure
    i_set = jnp.argmin(jnp.abs(pressures - pressure_points[4]))
    p_set_i = pressures[i_set]

    # Store logarithm of various pressure quantities
    nat_log_p = jnp.log(pressures)
    nat_log_p_min = nat_log_pressure_points[0]
    nat_log_p_set_i = jnp.log(p_set_i)

    # Branch 1: P_set is in layer 3
    t3_b1 = temperature_reference
    t2_b1 = t3_b1 - (
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t1_b1 = t2_b1 + (
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t0_b1 = t1_b1 - (
        (1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)
    ) ** (1 / beta_points[0])

    # Branch 2: P_set is in layer 2
    t2_b2 = temperature_reference - (
        (1.0 / alpha_points[1]) * (nat_log_p_set_i - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t1_b2 = t2_b2 + (
        # NOTE: there was a bug here, beta_points[0] was used instead of beta_points[1]
        # This is corrected to follow the POSEIDON implementation and the Madhusudhan & Seager (2009) paper
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t3_b2 = t2_b2 + (
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t0_b2 = t1_b2 - (
        (1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)
    ) ** (1 / beta_points[0])

    # Branch 3: P_set is in layer 1
    t0_b3 = temperature_reference - (
        (1.0 / alpha_points[0]) * (nat_log_p_set_i - nat_log_p_min)
    ) ** (1 / beta_points[0])
    t1_b3 = t0_b3 + (
        (1.0 / alpha_points[0]) * (nat_log_pressure_points[1] - nat_log_p_min)
    ) ** (1 / beta_points[0])
    t2_b3 = t1_b3 - (
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[1] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])
    t3_b3 = t2_b3 + (
        (1.0 / alpha_points[1])
        * (nat_log_pressure_points[3] - nat_log_pressure_points[2])
    ) ** (1 / beta_points[1])

    # Select branch based on P_set
    # Condition for branch 1
    cond1 = nat_log_pressure_points[4] >= nat_log_pressure_points[3]
    # Condition for branch 2
    cond2 = nat_log_pressure_points[4] >= nat_log_pressure_points[1]

    t0 = jnp.where(cond1, t0_b1, jnp.where(cond2, t0_b2, t0_b3))
    # t1 = jnp.where(cond1, t1_b1, jnp.where(cond2, t1_b2, t1_b3))
    t2 = jnp.where(cond1, t2_b1, jnp.where(cond2, t2_b2, t2_b3))
    t3 = jnp.where(cond1, t3_b1, jnp.where(cond2, t3_b2, t3_b3))

    temperatures = temperatures.at[mask_1].set(
        t0
        + ((nat_log_p[mask_1] - nat_log_pressure_points[0]) / (alpha_points[0]))
        ** (1 / beta_points[0])
    )
    temperatures = temperatures.at[mask_2].set(
        t2
        + ((nat_log_p[mask_2] - nat_log_pressure_points[2]) / (alpha_points[1]))
        ** (1 / beta_points[1])
    )
    temperatures = temperatures.at[mask_3].set(t3)

    return temperatures
