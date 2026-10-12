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
    #wavelengths_micron,
    frequency_bins_edges,
    pressures,
    temperatures,
    mass_fractions,
    mean_molar_masses,
    cias_loaded_opacities,
    gas_continuum_contributors,
    gray_opacity,
    #additional_absorption_opacities_function,
    #additional_opacity_functions_kwargs=None,
):
    # Reset
    continuum_opacities = jnp.zeros(
        (frequencies.size, pressures.size), dtype=WORKING_JAX_DTYPE
    )

    # Add CIA opacities
    continuum_opacities += compute_cia_opacities(
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

    '''
    # Add external user supplied opacities
    continuum_opacities += additional_absorption_opacities_function(
        wavelengths=wavelengths_micron,
        pressures=pressures*1e-6,  # convert to bar
        **(additional_opacity_functions_kwargs or {}),
    )
    '''

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
    #wavelengths_micron,
    pressures,
    temperatures,
    mass_fractions,
    mean_molar_masses,
    rayleigh_species,
    haze_factor,
    #additional_scattering_opacities_function,
    #additional_opacity_functions_kwargs=None,
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

    '''
    # Add external user supplied scattering opacities
    continuum_opacities_scattering += additional_scattering_opacities_function(
        wavelengths=wavelengths_micron,
        pressures=pressures*1e-6,  # convert to bar
        **(additional_opacity_functions_kwargs or {}),
    )
    '''

    return continuum_opacities_scattering, scattering_in_transmission
