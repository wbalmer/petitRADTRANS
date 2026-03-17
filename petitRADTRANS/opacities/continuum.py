import copy
import jax.numpy as jnp
from jax import debug, jit, config
from jax.typing import ArrayLike
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.opacities.cia import compute_cia_opacities
from petitRADTRANS.opacities.hminus import compute_non_cia_gas_induced_continuum_opacities
from petitRADTRANS.opacities.rayleigh import compute_rayleigh_scattering_opacities

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=JAX_DTYPE)


def compute_continuum_opacities(
    frequencies, frequency_bins_edges,
    pressures, temperatures, mass_fractions, mean_molar_masses,
    cias_loaded_opacities, gas_continuum_contributors, gray_opacity
    ):
    # Reset
    continuum_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)

    # Add CIA opacities
    continuum_opacities += (
        compute_cia_opacities(
            cia_dicts=cias_loaded_opacities,
            mass_fractions=mass_fractions,
            pressures=pressures,
            temperatures=temperatures,
            frequencies=frequencies,
            mean_molar_masses=mean_molar_masses,
        )
    )

    # Add other gas continuum opacities
    continuum_opacities += (
        compute_non_cia_gas_induced_continuum_opacities(
            gas_continuum_contributors=gas_continuum_contributors,
            mass_fractions=mass_fractions,
            pressures=pressures,
            temperatures=temperatures,
            frequencies=frequencies,
            frequency_bins_edges=frequency_bins_edges,
            mean_molar_masses=mean_molar_masses
        )
    )

    if gray_opacity is not None:
        continuum_opacities += gray_opacity

    return continuum_opacities


#@jit(static_argnames=['line_species', 'line_opacity_mode', 'sum_opacities'])
def compute_combined_opacities(
    opacities, continuum_opacities,
    line_species, line_species_mass_fractions, line_opacity_mode,
    sum_opacities, g_gauss, weights_gauss
    ):
    # Combine line opacities with continuum opacities, apply line opacity mass fraction weighting
    opacities = combine_opacities(
        line_species_mass_fractions=line_species_mass_fractions,
        opacities=opacities,
        continuum_opacities=continuum_opacities  # added to the first species slot
    )
    # TODO lax.cond - might need to check output shape
    if line_opacity_mode == 'c-k' and sum_opacities:
        # Similar to the line-by-line case below, if sum_opacities is True, we will put the total opacity
        # into the first species slot and then carry the remaining radiative transfer steps only over that 0 index
        # In c-k mode, sum_opacities is True only in emission if scattering_in_emission is True
        opacities = combine_ck_opacities(
            opacities=opacities,
            g_gauss=g_gauss,
            weights_gauss=weights_gauss
        )
    elif line_opacity_mode == 'lbl' and len(line_species) > 1:
        # In the line-by-line case we can simply add the opacities of different species in frequency space
        # All opacities are stored in the first species index slot
        opacities = opacities.at[:, :, 0, :].set(jnp.sum(opacities, axis=2))

    return opacities


def combine_ck_opacities(opacities, g_gauss, weights_gauss):
    opacities = opacities.at[:, :, 0, :].set(radiative_transfer_core.combine_ck_opacities(
        opacities,
        g_gauss,
        weights_gauss
    ))

    return opacities


def combine_opacities(line_species_mass_fractions, opacities, continuum_opacities):
    # TODO can we simplify this?
    # TODO maybe move to different file for organisation?
    opacities *= line_species_mass_fractions.T  # (g, frequencies, species, layers)
    opacities = jnp.swapaxes(opacities, 1, 2)  # (g, species, frequencies, layers)
    opacities = opacities.at[:, 0].add(continuum_opacities)  # quirkiness: put total opacities in the opacities of species 0
    opacities = jnp.swapaxes(opacities, 2, 1)  # back to default shape (g, frequencies, species, layers)

    return opacities


def compute_continuum_scattering_opacities(
    frequencies,
    pressures, temperatures, mass_fractions, mean_molar_masses,
    rayleigh_species, haze_factor
):
    # Reset
    continuum_opacities_scattering = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    scattering_in_transmission = False

    # Add rayleigh scattering opacities
    if len(rayleigh_species) > 0:
        scattering_in_transmission = True

        continuum_opacities_scattering += (
            compute_rayleigh_scattering_opacities(
                rayleigh_species=rayleigh_species,
                pressures=pressures,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                frequencies=frequencies,
                haze_factor=haze_factor,
            )
        )

    return continuum_opacities_scattering, scattering_in_transmission
