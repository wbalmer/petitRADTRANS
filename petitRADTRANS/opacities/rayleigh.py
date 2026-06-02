import jax.numpy as jnp
from petitRADTRANS.radtrans_core.rayleigh_scattering import compute_rayleigh_scattering
from petitRADTRANS.chemistry.utils import get_base_species_mass_fractions
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.physics import frequency2wavelength, hz2um

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE


def compute_power_law_opacities(
    power_law_opacity_350nm, power_law_opacity_coefficient, frequencies, n_layers
):
    wavelengths = hz2um(frequencies)
    power_law_opacities = (
        power_law_opacity_350nm * (wavelengths / 0.35) ** power_law_opacity_coefficient
    )

    return jnp.tile(power_law_opacities, (n_layers, 1)).T  # (p, wavelengths)


def compute_rayleigh_scattering_opacities(
    rayleigh_species,
    pressures,
    temperatures,
    mass_fractions,
    mean_molar_masses,
    frequencies,
    haze_factor=1.0,
):
    """Add Rayleigh scattering opacities to scattering continuum opacities.

    Args:
        temperatures: temperatures in each atmospheric layer
        mass_fractions: dictionary of the Rayleigh scattering species mass fractions
    """
    wavelengths_angstroem = jnp.array(
        frequency2wavelength(frequencies) * 1e8, dtype=FEAUTRIER_JAX_DTYPE
    )
    rayleigh_scattering_opacities = jnp.zeros(
        (frequencies.size, pressures.size), dtype=FEAUTRIER_JAX_DTYPE
    )

    # Sort keys to match JAX JIT's alphabetical ordering of dict keys
    species_basenames = []
    for species in sorted(mass_fractions.keys()):
        species_basenames.append(Opacity.get_species_base_name(species))
    species_basenames = tuple(species_basenames)

    base_species_mass_fractions = get_base_species_mass_fractions(
        mass_fractions, species_basenames
    )

    for species in rayleigh_species:
        if species not in base_species_mass_fractions:
            raise ValueError(
                f"Rayleigh species '{species}' not found in mass fractions dict "
                f"(listed species: {list(mass_fractions.keys())}, "
                f"listed species base names {list(base_species_mass_fractions.keys())}:)"
            )

        rayleigh_scattering_opacities += haze_factor * compute_rayleigh_scattering(
            species,
            base_species_mass_fractions[species],
            wavelengths_angstroem,
            mean_molar_masses,
            temperatures,
            pressures,
        )

    return rayleigh_scattering_opacities
