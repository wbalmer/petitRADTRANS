import jax.numpy as jnp
from jax import jit, config
from jax.typing import ArrayLike

def madhu_seager_2009(pressures, log_pressure_points, T_set, alpha_points, beta_points):
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
            A list of log pressure breakpoints defining different temperature regimes.
            The zeroth element is the minimum pressure, should be log10(press[0]).
            The first element is the 1-2 boundary
            The second element is the level of the inversion
            The third element is the 2-3 boundary.
            The fourth element is the pressure at which the temperature is set (P_set).
        T_set : (float)
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
    pressure_points = 10**jnp.array(log_pressure_points)
    mask_1 = pressures < pressure_points[1]
    mask_2 = (pressures >= pressure_points[1]) & (pressures < pressure_points[3])
    mask_3 = pressures >= pressure_points[3]

    # Find index of pressure closest to the set pressure
    i_set = jnp.argmin(jnp.abs(pressures - pressure_points[4]))
    p_set_i = pressures[i_set]

    # Store logarithm of various pressure quantities
    log_p = jnp.log10(pressures)
    log_p_min = log_pressure_points[0]
    log_p_set_i = jnp.log10(p_set_i)

    T0 = None
    T2 = None
    T3 = None

    # By default, (P_set = 10 bar), so T(P_set) should be in layer 3
    # By default (P_set = 10 bar), so T(P_set) should be in layer 3
    if log_pressure_points[4] >= log_pressure_points[3]:
        T3 = T_set  # T_deep is the isothermal deep temperature T3 here

        # Use the temperature parameter to compute boundary temperatures
        T2 = T3 - (
            (1.0 / alpha_points[1]) * (log_pressure_points[3] - log_pressure_points[2])
        ) ** (1 / beta_points[1])
        T1 = T2 + (
            (1.0 / alpha_points[1]) * (log_pressure_points[1] - log_pressure_points[2])
        ) ** (1 / beta_points[1])
        T0 = T1 - (
            (1.0 / alpha_points[0]) * (log_pressure_points[1] - log_p_min)
            ) ** (1 / beta_points[0])

    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (
        log_pressure_points[4] >= log_pressure_points[1]
    ):  # Temperature parameter in layer 2
        # Use the temperature parameter to compute the boundary temperatures
        T2 = T_set - (
            (1.0 / alpha_points[1]) * (log_p_set_i - log_pressure_points[2])
        ) ** (1 / beta_points[1])
        T1 = T2 + (
            (1.0 / alpha_points[1]) * (log_pressure_points[1] - log_pressure_points[2])
        ) ** (1 / beta_points[0])
        T3 = T2 + (
            (1.0 / alpha_points[1]) * (log_pressure_points[3] - log_pressure_points[2])
        ) ** (1 / beta_points[1])
        T0 = T1 - ((1.0 / alpha_points[0]) * (log_pressure_points[1] - log_p_min)) ** (
            1 / beta_points[0]
        )

    # ...or for layer 1
    elif (
        log_pressure_points[4] < log_pressure_points[1]
    ):  # Temperature parameter in layer 1

        # Use the temperature parameter to compute the boundary temperatures
        T0 = T_set - ((1.0 / alpha_points[0]) * (log_p_set_i - log_p_min)) ** (
            1 / beta_points[0]
        )
        T1 = T0 + ((1.0 / alpha_points[0]) * (log_pressure_points[1] - log_p_min)) ** (
            1 / beta_points[0]
        )
        T2 = T1 - (
            (1.0 / alpha_points[1]) * (log_pressure_points[1] - log_pressure_points[2])
        ) ** (1 / beta_points[1])
        T3 = T2 + (
            (1.0 / alpha_points[1]) * (log_pressure_points[3] - log_pressure_points[2])
        ) ** (1 / beta_points[1])

    temperatures = temperatures.at[mask_1].set(T0 + (
        (log_p[mask_1] - log_pressure_points[0]) / (alpha_points[0])
    ) ** (1 / beta_points[0]))
    temperatures = temperatures.at[mask_2].set(T2 + (
        (log_p[mask_2] - log_pressure_points[2]) / (alpha_points[1])
    ) ** (1 / beta_points[1]))
    temperatures = temperatures.at[mask_3].set(T3)
    return temperatures