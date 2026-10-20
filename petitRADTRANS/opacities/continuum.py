"""Computes gas and cloud continuum opacities but also (confusingly) combines c-k opacities."""

import jax.numpy as jnp
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.opacities.cia import compute_cia_opacities
from petitRADTRANS.opacities.hminus import (
    compute_non_cia_gas_induced_continuum_opacities,
)
from petitRADTRANS.opacities.rayleigh import compute_rayleigh_scattering_opacities

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=WORKING_JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=WORKING_JAX_DTYPE)


def compute_continuum_opacities(
    frequencies,
    frequency_bins_edges,
    pressures,
    temperatures,
    mass_fractions,
    mean_molar_masses,
    cias_loaded_opacities,
    gas_continuum_contributors,
    gray_opacity,
):
    """Compute the non-cloud continuum absorption opacities.

    Args:
        frequencies (ArrayLike):
            frequencies of the Radtrans object (bin centers).
        frequency_bins_edges (ArrayLike):
            frequency bin borders
        pressures (ArrayLike):
            pressures of the atmospheric layers (cgs units). Shape (N_layers).
        temperatures (ArrayLike):
            temperature within the layers (in K). Shape (N_layers).
        mass_fractions:
            Dictionary containing mass-fraction profiles for all atmospheric species
            that have a requested opacity associated to them. Shape: dictionary with
            shape (N_layers) array per species inside.
        mean_molar_masses:
            Mean molecular mass profile in amu. Shape (N_layers).
        cias_loaded_opacities:
            Dictionary of dictionaries containing all relevant information to calculate
            collision-induced absorption (CIA) opacities. Example structure for the H2--H2
            collision:
                "H2--H2": {
                    "molecules":   ["H2", "H2"],    # Collision partners
                    "weight":      scalar,      # product of the colliders' molar masses
                    "lambda":      (n_lambda,), # wavelengths in cm
                    "temperature": (n_T,),
                    "alpha":       (n_lambda, n_T),  # Needed to calculate the CIA opacities as
                                                     # function of wavelength, pressure and temperature.
                }
        gas_continuum_contributors:
            tuple listing which continuum gas opacities to include (e.g., "H-", "H2--H2", etc.)
        gray_opacity:
            Optional gray absorption opacity added "everywhere"
            (at all pressures and wavelengths). Scalar.

    Returns:
        continuum_opacities (ArrayLike):
            Gas continuum absorption opacities (N_frequencies, N_layers) in cgs (cm^2/g).
    """

    # Reset
    continuum_opacities = jnp.zeros(
        (frequencies.size, pressures.size), dtype=WORKING_JAX_DTYPE
    )

    # Add CIA opacities
    continuum_opacities += compute_cia_opacities( # TODO Paul_review : continue from within here
        cia_dicts=cias_loaded_opacities,
        mass_fractions=mass_fractions,
        pressures=pressures,
        temperatures=temperatures,
        frequencies=frequencies,
        mean_molar_masses=mean_molar_masses,
    )

    # Add other gas continuum opacities
    continuum_opacities += compute_non_cia_gas_induced_continuum_opacities(
        gas_continuum_contributors=gas_continuum_contributors,
        mass_fractions=mass_fractions,
        pressures=pressures,
        temperatures=temperatures,
        frequencies=frequencies,
        frequency_bins_edges=frequency_bins_edges,
        mean_molar_masses=mean_molar_masses,
    )

    if gray_opacity is not None:
        continuum_opacities += gray_opacity

    return continuum_opacities


# @jit(static_argnames=['line_species', 'line_opacity_mode', 'sum_opacities'])
def compute_combined_opacities(
    opacities,
    continuum_opacities,
    line_species,
    line_species_mass_fractions,
    line_opacity_mode,
    sum_opacities,
    g_gauss,
    weights_gauss,
    g_combination_weights=None,
    g_combination_weights_sum=None,
    input_layout="ng_f_s_l",
    ck_chunk_size=radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE,
):
    if line_opacity_mode == "c-k" and sum_opacities:
        # Similar to the line-by-line case below, if sum_opacities is True, we will put the total opacity
        # into the first species slot and then carry the remaining radiative transfer steps only over that 0 index
        # In c-k mode, sum_opacities is True only in emission if scattering_in_emission is True
        return combine_ck_opacities(
            opacities=opacities,
            continuum_opacities=continuum_opacities,
            line_species_mass_fractions=line_species_mass_fractions,
            g_gauss=g_gauss,
            weights_gauss=weights_gauss,
            g_combination_weights=g_combination_weights,
            g_combination_weights_sum=g_combination_weights_sum,
            input_layout=input_layout,
            ck_chunk_size=ck_chunk_size,
        )

    # Combine line opacities with continuum opacities, apply line opacity mass fraction weighting
    opacities = combine_opacities(
        line_species_mass_fractions=line_species_mass_fractions, # for lbl this will just be ones, since
                                                                 # the mass fraction scaling has already
                                                                 # been done.
        opacities=opacities,
        continuum_opacities=continuum_opacities,  # added to the first species slot
    )

    if line_opacity_mode == "lbl" and len(line_species) > 1:
        # In the line-by-line case we can simply add the opacities of different species in frequency space
        # All opacities are stored in the first species index slot
        opacities = opacities.at[:, :, 0, :].set(jnp.sum(opacities, axis=2))

    return opacities


def combine_ck_opacities(
    opacities,
    continuum_opacities,
    line_species_mass_fractions,
    g_gauss,
    weights_gauss,
    g_combination_weights=None,
    g_combination_weights_sum=None,
    input_layout="ng_f_s_l",
    ck_chunk_size=radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE,
):
    combined_opacities = radiative_transfer_core.combine_ck_opacities(
        opacities,
        g_gauss,
        weights_gauss=weights_gauss,
        line_species_mass_fractions=line_species_mass_fractions,
        continuum_opacities=continuum_opacities,
        g_combination_weights=g_combination_weights,
        g_combination_weights_sum=g_combination_weights_sum,
        input_layout=input_layout,
        ck_chunk_size=ck_chunk_size,
    )

    return combined_opacities[:, :, None, :]


def combine_opacities(line_species_mass_fractions, opacities, continuum_opacities):
    abundance_weights = line_species_mass_fractions.T[None, None, :, :]
    opacities = opacities * abundance_weights
    opacities = opacities.at[:, :, 0, :].add(continuum_opacities[None, :, :])

    return opacities


def compute_continuum_scattering_opacities(
    frequencies,
    pressures,
    temperatures,
    mass_fractions,
    mean_molar_masses,
    rayleigh_species,
    haze_factor,
):
    # Reset
    continuum_opacities_scattering = jnp.zeros(
        (frequencies.size, pressures.size), dtype=FEAUTRIER_JAX_DTYPE
    )
    scattering_in_transmission = False

    # Add rayleigh scattering opacities
    if len(rayleigh_species) > 0:
        scattering_in_transmission = True

        continuum_opacities_scattering += compute_rayleigh_scattering_opacities(
            rayleigh_species=rayleigh_species,
            pressures=pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            frequencies=frequencies,
            haze_factor=haze_factor,
        )

    return continuum_opacities_scattering, scattering_in_transmission
