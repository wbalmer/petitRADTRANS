"""
JAX implementation of fortran_core.physics
"""

import jax.numpy as jnp
import warnings
from jax import jit, vmap, lax, debug

import petitRADTRANS.physical_constants as cst

# import petitRADTRANS.fortran_radtrans_core import math as fmath


@jit
def compute_planck_function_integral_old(T, nul, nur):
    # TODO this might not be used anymore.
    # calculate planck function between two frequencies
    B_nu = 0.0
    # Take mean using Boole's method
    nu_large = jnp.maximum(nul, nur)
    nu_small = jnp.minimum(nul, nur)
    nu1 = nu_small
    nu2 = nu_small + 1.0 * (nu_large - nu_small) / 4.0
    nu3 = nu_small + 2.0 * (nu_large - nu_small) / 4.0
    nu4 = nu_small + 3.0 * (nu_large - nu_small) / 4.0
    nu5 = nu_large
    diff_nu = nu2 - nu1

    B_nu = B_nu + 1.0 / 90.0 * (
        7.0
        * 2.0
        * _h_over_c
        * (nu1**2 * (nu1 / cst.c))
        / jnp.expm1(cst.h * nu1 / (cst.kB * T))
        + 32.0
        * 2.0
        * _h_over_c
        * (nu2**2 * (nu2 / cst.c))
        / jnp.expm1(cst.h * nu2 / (cst.kB * T))
        + 12.0
        * 2.0
        * _h_over_c
        * (nu3**2 * (nu3 / cst.c))
        / jnp.expm1(cst.h * nu3 / (cst.kB * T))
        + 32.0
        * 2.0
        * _h_over_c
        * (nu4**2 * (nu4 / cst.c))
        / jnp.expm1(cst.h * nu4 / (cst.kB * T))
        + 7.0
        * 2.0
        * _h_over_c
        * (nu5**2 * (nu5 / cst.c))
        / jnp.expm1(cst.h * nu5 / (cst.kB * T))
    )
    return B_nu


@jit
def compute_rosseland_opacities_core(
    clouds_final_absorption_opacities, frequencies_bin_edges, temperature, weights_gauss
):
    """Calculate the Rosseland mean opacity as a function of temperature.

    Args:
        clouds_final_absorption_opacities (DeviceArray): Absorption opacities with shape (N_g, N_frequency_edges - 1, N_temperature).
        frequencies_bin_edges (DeviceArray): Frequency bin edges with shape (N_frequency_edges,).
        temperature (DeviceArray): Temperature with shape (N_temperature,).
        weights_gauss (DeviceArray): Gauss weights with shape (N_g,).

    Returns:
        DeviceArray: Rosseland opacity with shape (N_temperature,).
    """
    diff_freqs = frequencies_bin_edges[:-1] - frequencies_bin_edges[1:]
    B_nu_dT = compute_planck_function_temperature_derivative(
        temperature, frequencies_bin_edges
    )

    # Precompute for efficiency
    inv_absorption_opacities = 1.0 / clouds_final_absorption_opacities

    # Use einsum for efficient contraction
    weighted_inv_opacities = jnp.einsum(
        "g,gfn->fn", weights_gauss, inv_absorption_opacities
    )

    kappa_rosse = jnp.sum(
        B_nu_dT * weighted_inv_opacities * diff_freqs[:, None], axis=0
    )
    numerator = jnp.sum(B_nu_dT * diff_freqs[:, None], axis=0)

    return numerator / kappa_rosse


def compute_star_planck_function_integral(temperatures, frequencies):
    """
    Not clear what this function is actually doing...
    TODO check what the difference between the different integral functions is
    """
    return compute_planck_function_integral(temperatures, frequencies)


@jit(static_argnums=(4,))
def compute_rosseland_opacities(
    opacities,
    temperature,
    weights_gauss,
    frequencies_bin_edges,
    scattering_in_emission,
    continuum_opacities_scattering,
):
    """_summary_

    Args:
        total_opacity (_type_): _description_
        temperature (_type_): _description_
        w_gauss (_type_): _description_
        border_freqs (_type_): _description_
        do_scat_emis (_type_): _description_
        continuum_opacity_scattering_emis (_type_): _description_

    Returns:
        _type_: _description_
    """

    if scattering_in_emission:
        opacities_use = opacities + continuum_opacities_scattering[None, :, :]
    else:
        opacities_use = opacities

    kappa_rosse = compute_rosseland_opacities_core(
        opacities_use, frequencies_bin_edges, temperature, weights_gauss
    )
    return kappa_rosse


@jit
def compute_planck_function(T, nu):
    """
    Calculates B_ν https://en.wikipedia.org/wiki/Planck%27s_law
    For every temperature and frequency provided.

    Args:
        T (ndarray): 1D array of temperature in Kelvin
        nu (ndarray): 1D array of frequencies in Hertz

    Returns:
        B_nu (ndarray): (len(nu),len(T)) matrix of the spectral radiance (ie flux) as calculated from
                        Planck's law for each frequency and temperature.
    """
    T_use = jnp.tile(T, (nu.shape[0], 1))
    # Float32 safety: nu^3 overflows for near-IR (~3.7e43 > float32 max ~3.4e38).
    # Factor 2h*nu^3/c^2 = 2*(h/c)*(nu/c)*nu^2 keeps all intermediates in range:
    # h/c ~ 2.2e-37 (representable), nu/c ~ 1e4, nu^2 ~ 1e28.
    x = cst.h * nu[:, None] / (cst.kB * T_use)
    B_nu = (
        (2.0 * cst.h / cst.c) * (nu[:, None] / cst.c) * nu[:, None] ** 2 / jnp.expm1(x)
    )
    return B_nu


@jit
def compute_planck_function_temperature_derivative(T, frequency_bin_edges):
    """
    Calculates the gradient of B_ν (Planck's Law) with respect to temperature for
    each temperature and frequency provided using JAX.

    Args:
        T (DeviceArray): 1D JAX array of temperatures in Kelvin.
        frequency_bin_edges (DeviceArray): 1D JAX array of frequency bin edges in Hertz.

    Returns:
        B_nu_dT (DeviceArray): (len(T), len(nu)) JAX array of the derivative of the spectral radiance.
    """
    nu = (
        frequency_bin_edges[:-1] + frequency_bin_edges[1:]
    ) * 0.5  # Calculate midpoints directly

    # Constants for efficiency (avoid repeated calculations within the loop)
    h_over_kb = cst.h / cst.kB

    # Vectorized calculation of exp_terms and denominator for all T and nu
    exp_arg = h_over_kb * jnp.outer(0.5 / T, nu)

    # (exp(x) - exp(-x))^2 = (2 sinh(x))^2, numerically stable for all x
    denominator = (2.0 * jnp.sinh(exp_arg)) ** 2 * jnp.outer(
        cst.kB * T**2, jnp.ones_like(nu)
    )

    # Float32 safety: 2*h^2/c^2 ~ 9.8e-74 severely underflows, nu^4 severely overflows.
    # Rewrite 2*h^2*nu^4/c^2 = 2*(h*nu)^2*nu^2/c^2 with safe intermediates:
    # h*nu ~ 2.2e-12, (h*nu)^2 ~ 4.9e-24, *nu^2 ~ 5.4e5, 2/c^2 ~ 2.2e-21 -> ~1.2e-15.
    h_nu = cst.h * nu  # (n_freq_midpoints,) ~ 2.2e-12, safe in float32
    numerator_nu = h_nu**2 * nu**2  # ~ 5.4e5, safe in float32
    B_nu_dT = (
        2.0 / cst.c**2 * jnp.outer(jnp.ones_like(T), numerator_nu) / denominator
    ).T
    return B_nu_dT


def compute_planck_function_integral(temperatures, frequencies):
    """
    Compute mean using Boole's method
    TODO: possibly a faster integration method?
    """

    def reverse_frequencies(temperatures, frequencies):
        # warnings.warn("Frequencies are not in descending order. This may lead to incorrect results.")
        return _compute_planck_function_integral(temperatures, jnp.flip(frequencies))

    return lax.cond(
        frequencies[1] < frequencies[0],
        _compute_planck_function_integral,
        reverse_frequencies,
        temperatures,
        frequencies,
    )


@jit
def _compute_planck_function_integral(temperatures, frequencies):
    # Float32 safety: h/c^2 ~ 7.4e-48 underflows to 0 in float32 (min normal ~1.2e-38),
    # and nu^3 overflows (~3.7e43 for near-IR > float32 max ~3.4e38).  The combination
    # 0 * inf produces NaN.  Fix: use h/c (~2.2e-37, representable) as the base constant
    # and fold the remaining /c into the nu term as (nu/c), so the pattern becomes
    # factor * (h/c) * (nu/c) * nu^2, with all intermediates within float32 range.
    cst_h_over_c = cst.h / cst.c  # ~2.21e-37, representable in float32
    cst_h_k = cst.h / cst.kB
    factor1 = 1.0 / 90.0
    factor2 = 14.0 * cst_h_over_c  # (7 * 2) * (h/c); remaining /c applied per-nu below
    factor3 = 64.0 * cst_h_over_c  # (32 * 2) * (h/c)
    factor4 = 24.0 * cst_h_over_c  # (12 * 2) * (h/c)

    cst_h_kt = cst_h_k / temperatures

    nu_large = frequencies[:-1]
    nu_small = frequencies[1:]
    diff_nu = nu_large - nu_small
    nu2 = nu_small + diff_nu * 0.25
    nu3 = nu_small + diff_nu * 0.5
    nu4 = nu_small + diff_nu * 0.75

    # Each term: factor_k * (nu/c) * nu^2 / expm1(h*nu/(k*T))
    # This replaces factor_k_old * nu^3 / expm1(...) where factor_k_old = factor_k / c.
    planck_flux = factor1 * (
        factor2
        * (nu_small[None, :] / cst.c)
        * nu_small[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu_small[None, :])
        + factor3
        * (nu2[None, :] / cst.c)
        * nu2[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu2[None, :])
        + factor4
        * (nu3[None, :] / cst.c)
        * nu3[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu3[None, :])
        + factor3
        * (nu4[None, :] / cst.c)
        * nu4[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu4[None, :])
        + factor2
        * (nu_large[None, :] / cst.c)
        * nu_large[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu_large[None, :])
    )
    return planck_flux


@jit
def star_planck(T, nu):
    """
    Calculates B_ν https://en.wikipedia.org/wiki/Planck%27s_law
    For every temperature and frequency provided.
    Approximates the integral using Boole's method

    Args:
        T (ndarray): 1D array of temperature in Kelvin
        nu (ndarray): 1D array of frequencies in Hertz

    Returns:
        B_nu (ndarray): (len(nu),len(T)) matrix of the spectral radiance (ie flux) as calculated from
                        Planck's law for each frequency and temperature.
    """
    frequency_len = nu.shape[0]

    # Take mean using Boole's method implemented via lax.scan for JAX friendliness
    def _body(carry, i):
        nu_segment = nu[i : i + 1]
        nu_large = jnp.max(nu_segment)
        nu_small = jnp.min(nu_segment)
        nu1 = nu_small
        nu2 = nu_small + 1.0 * (nu_large - nu_small) / 4.0
        nu3 = nu_small + 2.0 * (nu_large - nu_small) / 4.0
        nu4 = nu_small + 3.0 * (nu_large - nu_small) / 4.0
        nu5 = nu_large
        # Float32 safety: h*nu^3/c^2 combines nu^3 overflow with h/c^2 underflow.
        # Use 2*(h/c)*(nu^2)*(nu/c) instead — all intermediates within float32 range.
        _h_over_c = cst.h / cst.c  # ~2.21e-37, representable in float32
        val = (
            1.0
            / 90.0
            * (
                (7.0 * 2.0 * _h_over_c * (nu1**2) * (nu1 / cst.c))
                / jnp.expm1(cst.h * nu1 / (cst.kB * T))
                + (32.0 * 2.0 * _h_over_c * (nu2**2) * (nu2 / cst.c))
                / jnp.expm1(cst.h * nu2 / (cst.kB * T))
                + (12.0 * 2.0 * _h_over_c * (nu3**2) * (nu3 / cst.c))
                / jnp.expm1(cst.h * nu3 / (cst.kB * T))
                + (32.0 * 2.0 * _h_over_c * (nu4**2) * (nu4 / cst.c))
                / jnp.expm1(cst.h * nu4 / (cst.kB * T))
                + (7.0 * 2.0 * _h_over_c * (nu5**2) * (nu5 / cst.c))
                / jnp.expm1(cst.h * nu5 / (cst.kB * T))
            )
        )
        return carry, val

    indices = jnp.arange(frequency_len - 1)
    _, bnus = lax.scan(_body, None, indices)
    B_nu = bnus
    return B_nu


def compute_pressure_hydrostatic_equilibrium(
    mean_molar_masses,
    reference_gravity,
    planet_radius,
    p0,
    temperature,
    radii,
    rk4=True,
):
    p0 = p0 * 1e6
    vs = 1.0 / radii
    r_pl_sq = planet_radius**2

    def integrand(press):
        temp = temperature(press)
        mu = mean_molar_masses(press / 1e6, temp)
        if not jnp.isscalar(mu):
            mu = mu[0]

        integral = mu * cst.amu * reference_gravity * r_pl_sq / cst.kB / temp
        return integral

    pressures = [p0]
    dvs = jnp.diff(vs)
    press1 = p0
    chi1 = jnp.log(p0)
    for dv in dvs:
        k1 = integrand(press1)

        if rk4:
            chi2 = chi1 + 0.5 * dv * k1
            press2 = jnp.exp(chi2)
            k2 = integrand(press2)
            chi3 = chi1 + 0.5 * dv * k2
            press3 = jnp.exp(chi3)
            k3 = integrand(press3)
            chi4 = chi1 + dv * k3
            press4 = jnp.exp(chi4)
            k4 = integrand(press4)
            chi1 = chi1 + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4) * dv
        else:
            chi1 = chi1 + dv * k1

        press1 = jnp.exp(chi1)
        pressures.append(press1)

    return jnp.array(pressures) / 1e6
