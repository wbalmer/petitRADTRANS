import jax.numpy as jnp
from jax import jit, config
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.physics import frequency2wavelength

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64


def get_non_cia_gas_continuum_contributions():
    return {
        'H-': compute_h_minus_opacities
    }


def compute_h_minus_free_free_xsec(wavelengths, temperatures, electron_partial_pressure):
    """Calculate the H- free-free cross-section in units of cm^2 per H per e- pressure (in cgs).
    Source: "The Observation and Analysis of Stellar Photospheres" by David F. Gray, p. 156
    # TODO complete docstring
    Args:
        wavelengths: (angstrom)
        temperatures:
        electron_partial_pressure:

    Returns:

    """

    index = (wavelengths >= 2600.) & (wavelengths <= 113900.)
    lamb_use = wavelengths[index]

    if temperatures >= 2500.:
        # Convert to Angstrom (from cgs)
        theta = 5040. / temperatures

        f0 = -2.2763 - 1.6850 * jnp.log10(lamb_use) \
            + 0.76661 * jnp.log10(lamb_use) ** 2. \
            - 0.053346 * jnp.log10(lamb_use) ** 3.
        f1 = 15.2827 - 9.2846 * jnp.log10(lamb_use) \
            + 1.99381 * jnp.log10(lamb_use) ** 2. \
            - 0.142631 * jnp.log10(lamb_use) ** 3.
        f2 = -197.789 + 190.266 * jnp.log10(lamb_use) - 67.9775 * jnp.log10(lamb_use) ** 2. \
            + 10.6913 * jnp.log10(lamb_use) ** 3. - 0.625151 * jnp.log10(lamb_use) ** 4.

        ret_val = jnp.zeros_like(wavelengths, dtype=JAX_DTYPE)
        ret_val = ret_val.at[index].set(1e-26 * electron_partial_pressure * 1e1 ** (
                f0 + f1 * jnp.log10(theta) + f2 * jnp.log10(theta) ** 2.))
        return ret_val

    else:

        return jnp.zeros_like(wavelengths, dtype=JAX_DTYPE)


def compute_h_minus_bound_free_xsec(wavelengths_bin_edges):
    """ Calculate the H- bound-free cross-section in units of cm^2 per H-, as defined on page 155 of
    "The Observation and Analysis of Stellar Photospheres" by David F. Gray
    # TODO complete docstring
    Args:
        wavelengths_bin_edges: (angstrom)

    Returns:

    """

    left = wavelengths_bin_edges[:-1]
    right = wavelengths_bin_edges[1:]
    diff = jnp.diff(wavelengths_bin_edges)

    a = [
        1.99654,
        -1.18267e-5,
        2.64243e-6,
        -4.40524e-10,
        3.23992e-14,
        -1.39568e-18,
        2.78701e-23
    ]

    ret_val = jnp.zeros_like(wavelengths_bin_edges[1:], dtype=JAX_DTYPE)

    index = right <= 1.64e4

    # TODO probably should vmap or something
    for i_a in range(len(a)):
        ret_val = ret_val.at[index].add(a[i_a] * (
                right[index] ** (i_a + 1) - left[index] ** (i_a + 1)
        ) / (i_a + 1))

    index_bracket = (left < 1.64e4) & (right > 1.64e4)
    for i_a in range(len(a)):
        ret_val = ret_val.at[index_bracket].add(a[i_a] * (1.64e4 ** (i_a + 1) -
                                            left[index_bracket] ** (i_a + 1)) / (i_a + 1))

    index = (left + right) / 2. > 1.64e4
    ret_val = ret_val.at[index].set(0.)
    index = ret_val < 0.
    ret_val = ret_val.at[index].set(0.)

    return ret_val * 1e-18 / diff


def compute_h_minus_opacities(
    mass_fractions, pressures, temperatures, frequencies, frequency_bins_edges,
    mean_molar_masses, **kwargs
    ):
    """Calculate the H- opacity."""
    _ = kwargs  # kwargs is not used in this function, this is intended

    wavelengths = frequency2wavelength(frequencies) * 1e8  # cm to Angstrom
    wavelengths_bin_edges = frequency2wavelength(frequency_bins_edges) * 1e8  # cm to Angstrom

    ret_val = jnp.array(jnp.zeros(len(wavelengths) * len(pressures)).reshape(
        len(wavelengths),
        len(pressures)), dtype=JAX_DTYPE)

    # Calculate electron number fraction
    m_e = cst.e_molar_mass  # (AMU)
    n_e = mean_molar_masses / m_e * mass_fractions['e-']

    # Calculate electron partial pressure
    p_e = pressures * n_e

    opacities_h_minus_bf = compute_h_minus_bound_free_xsec(wavelengths_bin_edges) / cst.amu

    for i_struct in range(len(n_e)):
        opacities_h_minus_ff = compute_h_minus_free_free_xsec(
            wavelengths,
            temperatures[i_struct],
            p_e[i_struct]
        ) / cst.amu * mass_fractions['H-'][i_struct]

        ret_val = ret_val.at[:, i_struct].set(opacities_h_minus_bf * mass_fractions['H-'][i_struct] + opacities_h_minus_ff)

    return ret_val


def compute_non_cia_gas_induced_continuum_opacities(
    gas_continuum_contributors, mass_fractions,
    pressures, temperatures, frequencies, frequency_bins_edges,
    mean_molar_masses, **kwargs
):
    continuum_opacities = 0

    for gas_continuum_contributor, contribution_function in (
            get_non_cia_gas_continuum_contributions().items()):

        # TODO JIT to lax.cond
        if gas_continuum_contributor in gas_continuum_contributors:
            continuum_opacities += (
                contribution_function(
                    mass_fractions=mass_fractions,
                    pressures=pressures,
                    temperatures=temperatures,
                    frequencies=frequencies,
                    frequency_bins_edges=frequency_bins_edges,
                    mean_molar_masses=mean_molar_masses,
                    **kwargs
                )
            )

    return continuum_opacities