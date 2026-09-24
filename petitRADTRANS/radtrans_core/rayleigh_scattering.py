import jax.numpy as jnp
from jax import jit, vmap

import petitRADTRANS.physical_constants as cst

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE

ABUNDANCE_THRESHOLD = jnp.finfo(FEAUTRIER_JAX_DTYPE).tiny

# Reference number density of the refractive-index tables (Loschmidt).
_N_REF = cst.standard_number_density_for_refractive_index


def _number_densities(mean_molar_masses, pressures, temperatures, mass_fractions):
    """Species number density per unit mass fraction, in cm^-3, per layer."""
    return mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions


def _rayleigh_opacity(lamb_inv, nm1, number_density, king_factor, molecule_mass,
                      mass_fractions):
    """Rayleigh opacity from a refractive index, for every wavelength and layer.

    .. math::
        \\kappa = \\frac{24 \\pi^3}{\\lambda^4 n^2}
                  \\left(\\frac{m^2 - 1}{m^2 + 2}\\right)^2
                  \\frac{F_k}{m_\\mathrm{mol}} X

    Parameters
    ----------
    lamb_inv : jax.Array, shape (n_wavelengths,)
        Inverse wavelength in cm^-1.
    nm1 : jax.Array, shape (n_wavelengths,) or (n_wavelengths, n_layers)
        Refractive index minus one. Species whose refractive index is scaled by
        the local number density pass the 2-D form.
    number_density : jax.Array, scalar or shape (n_layers,)
        Number density the refractive index refers to, in cm^-3.
    king_factor : jax.Array, scalar or shape (n_wavelengths,)
        King correction factor for depolarisation.
    molecule_mass : float
        Mass of one molecule, in g.
    mass_fractions : jax.Array, shape (n_layers,)
        Species mass fraction per layer.

    Returns
    -------
    jax.Array, shape (n_wavelengths, n_layers)
    """
    # The factor order is kept exactly as in the per-species scatter loops this
    # replaced. The result still differs from them by up to 1 ULP (float64):
    # multiplying by the whole mass-fraction vector at once lets XLA contract
    # the fused multiply-adds differently than the old one-column-at-a-time
    # loop did. That is ~7 orders of magnitude below the float32 working
    # precision of the radiative transfer, and the emergent spectra are
    # unchanged bit for bit.
    m_squared = (nm1 + 1.0) ** 2
    opacity = (
        24.0
        * jnp.pi**3
        * lamb_inv**4
        / number_density**2
        * ((m_squared - 1.0) / (m_squared + 2.0)) ** 2
        * king_factor
        / molecule_mass
    )
    if opacity.ndim == 1:
        opacity = opacity[:, None]

    return opacity * mass_fractions


@jit
def H2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates H2 Rayleigh scattering opacities."""
    # Constants hoisted out of the calculation loop
    c1 = 8.14e-13 / 1.66053892e-24 * 0.5
    c2 = 1.28e-6 / 1.66053892e-24 * 0.5
    c3 = 1.61 / 1.66053892e-24 * 0.5

    # Precompute inverse powers to avoid redundant calculations
    inv_wavelengths2 = 1 / wavelengths_angstroem**2
    inv_wavelengths4 = inv_wavelengths2**2
    inv_wavelengths6 = inv_wavelengths4 * inv_wavelengths2
    inv_wavelengths8 = inv_wavelengths6 * inv_wavelengths2

    # Compute the factor vectorized
    factor = c1 * inv_wavelengths4 + c2 * inv_wavelengths6 + c3 * inv_wavelengths8

    return factor[:, None] * mass_fractions


@jit
def He_rayleigh(wavelengths_angstroem, mass_fractions):
    """
    Calculates He Rayleigh scattering opacities using Chan & Dalgarno (1965) alphas.

    Args:
        wavelengths_angstroem: JAX array of wavelengths in Angstroms.
        mass_fractions: JAX array of mass fractions for each layer.

    Returns:
        JAX array of Rayleigh scattering opacities for each wavelength and layer.
    """
    lambda_cm = wavelengths_angstroem * 1e-8
    lambda_um = lambda_cm * 1e4

    # Calculate alpha_polarization
    alpha_polarization = 2.265983321 - 3.721350022 * lambda_um + 3.016150391 * lambda_um**2

    # Adjust for wavelengths above the threshold
    alpha_polarization = jnp.where(lambda_cm >= 0.9110e-4, 1.379, alpha_polarization)

    # Calculate multiplicative factor using alpha_polarization
    alpha_polarization = (
        128
        * jnp.pi**5
        / 3
        / lambda_cm**4
        * (alpha_polarization * 1.482e-25) ** 2
        / 4
        / 1.66053892e-24
    )

    return alpha_polarization[:, None] * mass_fractions[None, :]


@jit
def h2o_rayleigh(
    wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures
):
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    a0, a1, a2, a3, a4, a5, a6, a7 = (
        0.244257733,
        9.74634476e-3,
        -3.73234996e-3,
        2.68678472e-4,
        1.58920570e-3,
        2.45934259e-3,
        0.900704920,
        -1.66626219e-2,
    )
    luv, lir = 0.2292020, 5.432937
    mass_h2o = 18 * cst.amu

    lambda_norm = lambda_cm / 1e-4 / 0.589
    d = _number_densities(mean_molar_masses, pressures, temperatures, mass_fractions)
    temperatures_ = temperatures / 273.15

    @jit
    def H2O_rayleigh_single_layer(i_str):
        retVal = (
            a0
            + a1 * d[i_str]
            + a2 * temperatures_[i_str]
            + a3 * lambda_norm**2 * temperatures_[i_str]
            + a4 / lambda_norm**2
            + a5 / (lambda_norm**2 - luv**2)
            + a6 / (lambda_norm**2 - lir**2)
            + a7 * d[i_str] ** 2
        ) * d[i_str]

        retValMin = (
            a0
            + a1 * d[i_str]
            + a2 * temperatures_[i_str]
            + a3 * (0.2 / 0.589) ** 2 * temperatures_[i_str]
            + a4 / (0.2 / 0.589) ** 2
            + a5 / ((0.2 / 0.589) ** 2 - luv**2)
            + a6 / ((0.2 / 0.589) ** 2 - lir**2)
            + a7 * d[i_str] ** 2
        ) * d[i_str]

        retValMax = (
            a0
            + a1 * d[i_str]
            + a2 * temperatures_[i_str]
            + a3 * (1.1 / 0.589) ** 2 * temperatures_[i_str]
            + a4 / (1.1 / 0.589) ** 2
            + a5 / ((1.1 / 0.589) ** 2 - luv**2)
            + a6 / ((1.1 / 0.589) ** 2 - lir**2)
            + a7 * d[i_str] ** 2
        ) * d[i_str]

        nm1 = jnp.where(
            (lambda_cm / 1e-4 > 0.2) & (lambda_cm / 1e-4 < 1.1),
            jnp.sqrt((1 + 2 * retVal) / (1 - retVal)),
            jnp.where(
                lambda_cm / 1e-4 >= 1.1,
                jnp.sqrt((1 + 2 * retValMax) / (1 - retValMax)),
                jnp.sqrt((1 + 2 * retValMin) / (1 - retValMin)),
            ),
        )
        nm1 = nm1 - 1.0
        fk = 1.0

        result = (
            24
            * jnp.pi**3
            * lamb_inv**4
            / (d[i_str] / 18 / cst.amu) ** 2
            * (((nm1 + 1) ** 2 - 1) / ((nm1 + 1) ** 2 + 2)) ** 2
            * fk
            / mass_h2o
            * mass_fractions[i_str]
        )
        return jnp.where(jnp.isnan(result), 0, result).T

    return vmap(H2O_rayleigh_single_layer)(jnp.arange(mass_fractions.shape[0])).T


@jit
def co2_rayleigh(
    wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures
):
    """Calculates CO2 Rayleigh scattering opacities."""
    d = _number_densities(mean_molar_masses, pressures, temperatures, mass_fractions)
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)

    # (n_layers,): the refractive index is scaled to the local number density.
    scale = d / 44.0 / cst.amu / _N_REF

    nm1 = (
        1e-3
        * 1.1427e6
        * (
            5799.25 / jnp.maximum(20.0**2, 128908.9**2 - lamb_inv**2)
            + 120.05 / jnp.maximum(20.0**2, 89223.8**2 - lamb_inv**2)
            + 5.3334 / jnp.maximum(20.0**2, 75037.5**2 - lamb_inv**2)
            + 4.3244 / jnp.maximum(20.0**2, 67837.7**2 - lamb_inv**2)
            + 0.1218145e-4 / jnp.maximum(20.0**2, 2418.136**2 - lamb_inv**2)
        )
    )[:, None] * scale[None, :]  # (n_wavelengths, n_layers)

    fk = 1.1364 + 25.3e-12 * lamb_inv**2

    return _rayleigh_opacity(
        lamb_inv[:, None], nm1, scale[None, :] * _N_REF, fk[:, None],
        44.0 * cst.amu, mass_fractions,
    )


@jit
def o2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates O2 Rayleigh scattering opacities."""
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)

    # Use jnp.where for conditional assignment of A and B
    A = jnp.where(lamb_inv > 18315.0, 20564.8, 21351.1)
    B = jnp.where(lamb_inv > 18315.0, 2.480899e13, 2.18567e13)
    C = 4.09e9

    nm1 = 1e-8 * (A + B / (C - lamb_inv**2))
    fk = 1.096 + 1.385e-11 * lamb_inv**2 + 1.448e-20 * lamb_inv**4

    return _rayleigh_opacity(
        lamb_inv, nm1, 2.68678e19, fk, 32.0 * cst.amu, mass_fractions
    )


@jit
def n2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates N2 Rayleigh scattering opacities."""
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)

    # Nested jnp.where for conditional assignment of A, B, and C
    A = jnp.where(
        lamb_inv > 21360.0,
        5677.465,
        jnp.where(lamb_inv > 4860.0, 6498.2, 0.0),  # 0 if not in either range
    )
    B = jnp.where(
        lamb_inv > 21360.0,
        318.81874e12,
        jnp.where(lamb_inv > 4860.0, 307.43305e12, 0.0),
    )
    C = jnp.where(lamb_inv > 4860.0, 14.4e9, 0.0)

    nm1 = jnp.where(lamb_inv > 4860.0, 1e-8 * (A + B / (C - lamb_inv**2)), 0.0)
    fk = 1.034 + 3.17e-12 * lamb_inv**2

    # NOTE: 34.0 amu, not the 28.0 amu of N2, is carried over verbatim from the
    # Fortran implementation this was ported from.
    return _rayleigh_opacity(
        lamb_inv, nm1, 2.546899e19, fk, 34.0 * cst.amu, mass_fractions
    )


@jit
def co_rayleigh(
    wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures
):
    """Calculates CO Rayleigh scattering opacities."""
    d = _number_densities(mean_molar_masses, pressures, temperatures, mass_fractions)
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm

    scale = d / 28.0 / cst.amu / _N_REF  # (n_layers,)
    nfr_co = d / 28.0 / cst.amu

    # Use jnp.where for conditional adjustment of lamb_inv
    lamb_inv_use = jnp.where(lambda_cm / 1e-4 < 0.168, 1.0 / 0.168e-4, lamb_inv)

    nm1 = (
        (22851.0 + 0.456e12 / (71427.0**2 - lamb_inv_use**2)) * 1e-8
    )[:, None] * scale[None, :]  # (n_wavelengths, n_layers)
    fk = 1.016

    return _rayleigh_opacity(
        lamb_inv[:, None], nm1, nfr_co[None, :], fk, 28.0 * cst.amu, mass_fractions
    )


@jit
def ch4_rayleigh(
    wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures
):
    """Calculates CH4 Rayleigh scattering opacities."""
    d = _number_densities(mean_molar_masses, pressures, temperatures, mass_fractions)
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)

    scale = d / 16.0 / cst.amu / _N_REF  # (n_layers,)
    nfr_ch4 = d / 16.0 / cst.amu

    nm1 = ((46662.0 + 4.02e-6 * lamb_inv**2) * 1e-8)[:, None] * scale[None, :]
    fk = 1.0

    return _rayleigh_opacity(
        lamb_inv[:, None], nm1, nfr_ch4[None, :], fk, 16.0 * cst.amu, mass_fractions
    )


# Species that need only the wavelength grid and the mass fractions.
_WAVELENGTH_ONLY_RAYLEIGH = {
    "H2": H2_rayleigh,
    "He": He_rayleigh,
    "O2": o2_rayleigh,
    "N2": n2_rayleigh,
}

# Species whose refractive index depends on the local atmospheric state.
_STATE_DEPENDENT_RAYLEIGH = {
    "H2O": h2o_rayleigh,
    "CO2": co2_rayleigh,
    "CO": co_rayleigh,
    "CH4": ch4_rayleigh,
}


def compute_rayleigh_scattering(
    species,
    mass_fractions,
    wavelengths_angstroem,
    mean_molar_masses,
    temperatures,
    pressures,
):
    """Rayleigh scattering opacities of one species, shape (n_wavelengths, n_layers)."""
    if species in _WAVELENGTH_ONLY_RAYLEIGH:
        rayleigh_scattering_opacities = _WAVELENGTH_ONLY_RAYLEIGH[species](
            wavelengths_angstroem, mass_fractions
        )
    elif species in _STATE_DEPENDENT_RAYLEIGH:
        rayleigh_scattering_opacities = _STATE_DEPENDENT_RAYLEIGH[species](
            wavelengths_angstroem,
            mass_fractions,
            mean_molar_masses,
            pressures,
            temperatures,
        )
    else:
        raise ValueError(
            f"no Rayleigh scattering implementation for species '{species}'; "
            f"available species are "
            f"{sorted(_WAVELENGTH_ONLY_RAYLEIGH | _STATE_DEPENDENT_RAYLEIGH)}"
        )

    # Layers with a negligible abundance contribute no opacity. Applying the
    # mask once here replaces the per-species guards the individual kernels
    # used to carry (which were inconsistent: some thresholded the opacity,
    # others the mass fraction, and most were then masked again here anyway).
    return jnp.where(
        mass_fractions[None, :] < ABUNDANCE_THRESHOLD,
        0.0,
        rayleigh_scattering_opacities,
    )
