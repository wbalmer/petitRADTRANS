from jax import vmap
import jax.numpy as jnp
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.physics import frequency2wavelength

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE


def get_non_cia_gas_continuum_contributions():
    return {"H-": compute_h_minus_opacities}


def compute_h_minus_free_free_xsec(
    wavelengths, temperatures, electron_partial_pressure
):
    """Calculate the H- free-free cross-section in units of cm^2 per H per e- pressure (in cgs).
    Source: "The Observation and Analysis of Stellar Photospheres" by David F. Gray, p. 156

    Args:
        wavelengths: (angstrom)
        temperatures:
        electron_partial_pressure:

    Returns:

    """
    in_range = (wavelengths >= 2600.0) & (wavelengths <= 113900.0)
    # Guard log10 against out-of-range wavelengths
    lamb_safe = jnp.where(in_range, wavelengths, 1.0)
    log_lamb = jnp.log10(lamb_safe)

    theta = 5040.0 / jnp.maximum(temperatures, 2500.0)
    log_theta = jnp.log10(theta)

    f0 = (
        -2.2763
        - 1.6850 * log_lamb
        + 0.76661 * log_lamb ** 2.0
        - 0.053346 * log_lamb ** 3.0
    )
    f1 = (
        15.2827
        - 9.2846 * log_lamb
        + 1.99381 * log_lamb ** 2.0
        - 0.142631 * log_lamb ** 3.0
    )
    f2 = (
        -197.789
        + 190.266 * log_lamb
        - 67.9775 * log_lamb ** 2.0
        + 10.6913 * log_lamb ** 3.0
        - 0.625151 * log_lamb ** 4.0
    )

    values = (
        1e-26
        * electron_partial_pressure
        * 1e1 ** (f0 + f1 * log_theta + f2 * log_theta ** 2.0)
    )

    return jnp.where(in_range & (temperatures >= 2500.0), values, 0.0).astype(
        WORKING_JAX_DTYPE
    )


def compute_h_minus_bound_free_xsec(wavelengths_bin_edges):
    """Calculate the H- bound-free cross-section in units of cm^2 per H-, as defined on page 155 of
    "The Observation and Analysis of Stellar Photospheres" by David F. Gray

    Args:
        wavelengths_bin_edges: (angstrom)

    Returns:

    """
    left = wavelengths_bin_edges[:-1]
    right = wavelengths_bin_edges[1:]
    diff = jnp.diff(wavelengths_bin_edges)

    a = jnp.array(
        [
            1.99654,
            -1.18267e-5,
            2.64243e-6,
            -4.40524e-10,
            3.23992e-14,
            -1.39568e-18,
            2.78701e-23,
        ],
        dtype=WORKING_JAX_DTYPE,
    )
    powers = jnp.arange(1, len(a) + 1, dtype=WORKING_JAX_DTYPE)

    # Integrate polynomial from left to min(right, 1.64e4) for each bin; shape (n_bins, n_coeffs)
    right_clipped = jnp.minimum(right, 1.64e4)
    integral = jnp.sum(
        a / powers * (right_clipped[:, None] ** powers[None, :] - left[:, None] ** powers[None, :]),
        axis=-1,
    )

    midpoint = (left + right) / 2.0
    ret_val = jnp.where(midpoint <= 1.64e4, jnp.maximum(integral, 0.0), 0.0)

    return (ret_val * 1e-18 / diff).astype(WORKING_JAX_DTYPE)


def compute_h_minus_opacities(
    mass_fractions,
    pressures,
    temperatures,
    frequencies,
    frequency_bins_edges,
    mean_molar_masses,
    **kwargs
):
    """Calculate the H- opacity."""
    _ = kwargs  # kwargs is not used in this function, this is intended

    wavelengths = frequency2wavelength(frequencies) * 1e8  # cm to Angstrom
    wavelengths_bin_edges = (
        frequency2wavelength(frequency_bins_edges) * 1e8
    )  # cm to Angstrom

    m_e = cst.e_molar_mass  # (AMU)
    n_e = mean_molar_masses / m_e * mass_fractions["e-"]
    p_e = pressures * n_e

    opacities_h_minus_bf = (
        compute_h_minus_bound_free_xsec(wavelengths_bin_edges) / cst.amu
    )

    def compute_layer(temperature, p_e_i, mf_H, mf_Hminus):
        ff = (
            compute_h_minus_free_free_xsec(wavelengths, temperature, p_e_i)
            / cst.amu
            * mf_H
        )
        return opacities_h_minus_bf * mf_Hminus + ff

    # vmap over layers; result shape (n_layers, n_wavelengths) -> transpose to (n_wavelengths, n_layers)
    ret_val = vmap(compute_layer)(
        temperatures, p_e, mass_fractions["H"], mass_fractions["H-"]
    )

    return ret_val.T.astype(WORKING_JAX_DTYPE)


def compute_non_cia_gas_induced_continuum_opacities(
    gas_continuum_contributors,
    mass_fractions,
    pressures,
    temperatures,
    frequencies,
    frequency_bins_edges,
    mean_molar_masses,
    **kwargs
):
    continuum_opacities = 0

    for (
        gas_continuum_contributor,
        contribution_function,
    ) in get_non_cia_gas_continuum_contributions().items():

        if gas_continuum_contributor in gas_continuum_contributors:
            continuum_opacities += contribution_function(
                mass_fractions=mass_fractions,
                pressures=pressures,
                temperatures=temperatures,
                frequencies=frequencies,
                frequency_bins_edges=frequency_bins_edges,
                mean_molar_masses=mean_molar_masses,
                **kwargs
            )

    return continuum_opacities
