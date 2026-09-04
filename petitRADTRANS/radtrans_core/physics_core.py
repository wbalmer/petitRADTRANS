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


@jit
def compute_planck_function(temperatures, frequencies):
    """Planck spectral radiance B_nu for every temperature and frequency.

    https://en.wikipedia.org/wiki/Planck%27s_law

    Args:
        temperatures: 1D array of temperatures in Kelvin
        frequencies: array of frequencies in Hertz

    Returns:
        (len(frequencies), len(temperatures)) matrix of the spectral radiance.
    """
    temperatures = _as_planck_temperature_array(temperatures)
    frequencies = jnp.asarray(frequencies)

    exponent = (cst.h / cst.kB) * frequencies[..., None] / temperatures[None, :]
    # Clamp the Planck exponent so expm1 cannot overflow to +inf in the Wien tail.
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


@jit
def _compute_planck_function_integral(temperatures, frequencies):
    """Bin-integrated Planck function, (len(temperatures), len(frequencies) - 1)."""
    sample_frequencies = _compute_boole_sample_frequencies(frequencies)
    boole_weights = jnp.asarray(
        (7.0, 32.0, 12.0, 32.0, 7.0), dtype=sample_frequencies.dtype
    )
    sample_flux = compute_planck_function(temperatures, sample_frequencies)

    return (jnp.sum(boole_weights[:, None, None] * sample_flux, axis=0) / 90.0).T


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
    """Rosseland mean opacity, optionally including continuum scattering.

    Args:
        opacities: absorption opacities, shape (n_g, n_frequencies, n_layers).
        temperature: temperature of each layer, shape (n_layers,).
        weights_gauss: correlated-k quadrature weights, shape (n_g,).
        frequencies_bin_edges: frequency bin edges, shape (n_frequencies + 1,).
        scattering_in_emission: if True, add ``continuum_opacities_scattering``
            to the total opacity before taking the mean. Static argument.
        continuum_opacities_scattering: continuum scattering opacity,
            shape (n_frequencies, n_layers).

    Returns:
        Rosseland mean opacity of each layer, shape (n_layers,).
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
