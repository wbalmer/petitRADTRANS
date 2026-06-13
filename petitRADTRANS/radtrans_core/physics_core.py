"""
JAX implementation of fortran_core.physics
"""

import jax.numpy as jnp
from jax import jit, lax

import petitRADTRANS.physical_constants as cst

# import petitRADTRANS.fortran_radtrans_core import math as fmath


def _as_planck_temperature_array(temperatures):
    return jnp.atleast_1d(jnp.asarray(temperatures))


def _compute_planck_frequency_cubed_term(frequencies):
    # Keep the nu^3 / c^2 factor in a float32-safe form: (nu / c) * nu^2 / c.
    return (frequencies / cst.c) * frequencies**2


def _compute_planck_spectral_radiance_matrix(temperatures, frequencies):
    temperatures = _as_planck_temperature_array(temperatures)
    frequencies = jnp.asarray(frequencies)

    exponent = (cst.h / cst.kB) * frequencies[..., None] / temperatures[None, :]
    # Clamp the Planck exponent so expm1 cannot overflow to +inf in the Wien
    # tail. The forward value (prefactor/inf -> 0) is harmless, but the
    # reverse-mode derivative is -prefactor * exp(x) / expm1(x)**2, which becomes
    # inf/inf**2 = NaN once expm1(x) overflows -- and because the Planck function
    # feeds the source function at every frequency/layer, a single deep-Wien
    # cell turned the entire log-posterior gradient into NaN at the prior median.
    # The threshold keeps expm1(x)**2 (the gradient denominator) finite; beyond
    # it the emitted flux is ~0 with ~0 sensitivity, so the clamp is physically
    # negligible and only flattens an already-vanishing tail.
    max_exponent = 0.5 * jnp.log(jnp.finfo(exponent.dtype).max) - 1.0
    exponent = jnp.minimum(exponent, max_exponent)
    prefactor = (2.0 * cst.h / cst.c) * _compute_planck_frequency_cubed_term(
        frequencies
    )

    return prefactor[..., None] / jnp.expm1(exponent)


def _compute_boole_sample_frequencies(frequencies):
    nu_large = frequencies[:-1]
    nu_small = frequencies[1:]
    diff_nu = nu_large - nu_small

    return jnp.stack(
        (
            nu_small,
            nu_small + diff_nu * 0.25,
            nu_small + diff_nu * 0.5,
            nu_small + diff_nu * 0.75,
            nu_large,
        ),
        axis=0,
    )


def _compute_boole_planck_flux_matrix(temperatures, frequencies):
    sample_frequencies = _compute_boole_sample_frequencies(frequencies)
    boole_weights = jnp.asarray(
        (7.0, 32.0, 12.0, 32.0, 7.0), dtype=sample_frequencies.dtype
    )
    sample_flux = _compute_planck_spectral_radiance_matrix(
        temperatures, sample_frequencies
    )

    return jnp.sum(boole_weights[:, None, None] * sample_flux, axis=0) / 90.0


@jit
def compute_rosseland_opacities_core(
    clouds_final_absorption_opacities, frequencies_bin_edges, temperature, weights_gauss
):
    """Calculate the Rosseland mean opacity as a function of temperature.

    Args:
        clouds_final_absorption_opacities (DeviceArray): Absorption opacities with shape
            (N_g, N_frequency_edges - 1, N_temperature).
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
    return _compute_planck_spectral_radiance_matrix(T, nu)


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
    temperatures = _as_planck_temperature_array(T)
    nu = (
        frequency_bin_edges[:-1] + frequency_bin_edges[1:]
    ) * 0.5  # Calculate midpoints directly

    # Float32 safety: 2*h^2/c^2 ~ 9.8e-74 severely underflows, nu^4 severely overflows.
    # Rewrite 2*h^2*nu^4/c^2 = 2*(h*nu)^2*nu^2/c^2 with safe intermediates:
    # h*nu ~ 2.2e-12, (h*nu)^2 ~ 4.9e-24, *nu^2 ~ 5.4e5, 2/c^2 ~ 2.2e-21 -> ~1.2e-15.
    h_nu = cst.h * nu  # (n_freq_midpoints,) ~ 2.2e-12, safe in float32
    numerator_nu = (2.0 / cst.c**2) * (h_nu**2 * nu**2)
    x_half = 0.5 * (cst.h / cst.kB) * nu[:, None] / temperatures[None, :]
    denominator = (2.0 * jnp.sinh(x_half)) ** 2 * (cst.kB * temperatures[None, :] ** 2)

    return numerator_nu[:, None] / denominator


def compute_planck_function_integral(temperatures, frequencies):
    """
    Compute mean using Boole's method
    TODO: possibly a faster integration method?
    """
    temperatures = _as_planck_temperature_array(temperatures)

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
    return _compute_boole_planck_flux_matrix(temperatures, frequencies).T


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
    return compute_planck_function_integral(T, nu).T


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
