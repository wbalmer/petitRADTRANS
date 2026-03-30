import sys
import copy
import jax.numpy as jnp
from jax import vmap, config
from petitRADTRANS.utils import LockedDict
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.opacities.hminus import get_non_cia_gas_continuum_contributions
from petitRADTRANS.chemistry.utils import get_base_species_mass_fractions

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64


def get_cia_aliases(name: str) -> str:
    # TODO remove in version 4.0.0
    # Try to match name with aliases
    cia_aliases = {
        'H2--H2': ['H2H2', 'H2-H2', 'H2--H2'],
        'H2--He': ['H2He', 'H2-He', 'H2--He', 'HeH2', 'He-H2', 'He--H2'],
        'H2O--H2O': ['H2OH2O', 'H2O-H2O', 'H2O--H2O'],
        'H2O--N2': ['H2ON2', 'H2O-N2', 'H2O--N2', 'N2H2O', 'N2-H2O', 'N2--H2O'],
        'N2--H2': ['N2H2', 'N2-H2', 'N2--H2', 'H2N2', 'H2-N2', 'H2--N2'],
        'N2--He': ['N2He', 'N2-He', 'N2--He', 'HeN2', 'He-N2', 'He--N2'],
        'N2--N2': ['N2N2', 'N2-N2', 'N2--N2'],
        'O2--O2': ['O2O2', 'O2-O2', 'O2--O2'],
        'N2--O2': ['N2O2', 'N2-O2', 'N2--O2', 'O2N2', 'O2-N2', 'O2--N2'],
        'CO2--CO2': ['CO2CO2', 'CO2-CO2', 'CO2--CO2']
    }

    for cia, aliases in cia_aliases.items():
        if name in aliases:
            if name != cia:
                debug.print(
                    "usage of CIA aliases is deprecated and will be removed in a future update.\n"
                    "Use the CIA actual name, with pRT's CIA separator ('--') between colliding species, instead"
                )

            return cia

    # Name does not match a directory, a directory shortcut, or an alias
    return name


def init_cia_loaded_opacities(cia_contributors):
    tmp_collision_dict = LockedDict.build_and_lock({
        'molecules': None,
        'weight': None,
        'lambda': None,
        'temperature': None,
        'alpha': None
    })

    cia_loaded_opacities = LockedDict()

    for cia in cia_contributors:
        cia_loaded_opacities[cia] = copy.deepcopy(tmp_collision_dict)

    cia_loaded_opacities.lock()

    return cia_loaded_opacities


def get_cia_contributors(gas_continuum_contributors):
    cia = ()

    for gas_continuum_contributor in gas_continuum_contributors:
        if gas_continuum_contributor in get_non_cia_gas_continuum_contributions():
            continue

        cia = cia + (gas_continuum_contributor,)

    return cia


def compute_cia_opacities(cia_dicts, mass_fractions, pressures, temperatures, frequencies, mean_molar_masses):
    """Wrapper to interpolate_cia, calculating each collision's combined mass fraction."""
    # Add mass fractions sharing the same basename
    species_basenames = []
    for species in mass_fractions.keys():
        species_basenames.append(Opacity.get_species_base_name(species))
    species_basenames = tuple(species_basenames)

    base_species_mass_fractions = get_base_species_mass_fractions(mass_fractions, species_basenames)

    # Calculate CIA opacities
    cia_opacities = 0

    for collision, collision_dict in cia_dicts.items():
        # Calculate combined mass fraction
        combined_mass_fraction = 1

        for collision_species in collision_dict['molecules']:
            if collision_species in base_species_mass_fractions:
                combined_mass_fraction = combined_mass_fraction * base_species_mass_fractions[collision_species]
            else:
                raise ValueError(
                    f"species '{collision_species}' of CIA '{collision}' "
                    f"not found in mass fractions dict "
                    f"(listed species: {list(mass_fractions.keys())}, "
                    f"listed species base names {list(base_species_mass_fractions.keys())}:)"
                )

        # Add CIA opacities
        cia_opacities += (
            interpolate_cia(
                collision_dict=collision_dict,
                combined_mass_fractions=combined_mass_fraction,
                pressures=pressures,
                temperatures=temperatures,
                frequencies=frequencies,
                mean_molar_masses=mean_molar_masses
            )
        )

    return cia_opacities


def interpolate_cia(
    collision_dict, combined_mass_fractions, pressures, temperatures, frequencies, mean_molar_masses
    ):
    """Interpolate CIA cross-sections onto the Radtrans (wavelength, temperature) grid and convert it into
    opacities.

    Args:
        combined_mass_fractions: combined mass fractions of the colliding species
            e.g., for H2-He and an atmosphere with H2 and He MMR of respectively 0.74 and 0.24,
            combined_mas_fractions = 0.74 * 0.24
            combined_mas_fractions is divided by the combined weight (e.g. for H2 and He, 2 * 4 AMU^2), so there is
            no units issue.

    Returns:
        A (wavelength, temperature) array containing the CIA opacities.
    """
    weights = jnp.array(collision_dict['weight'], dtype=JAX_DTYPE)
    alphas = jnp.array(collision_dict['alpha'], dtype=JAX_DTYPE)
    binned_temperatures = jnp.array(collision_dict['temperature'], dtype=JAX_DTYPE)
    binned_wavelengths = jnp.array(collision_dict['lambda'], dtype=JAX_DTYPE)

    factor = (
        combined_mass_fractions / weights
        * mean_molar_masses / cst.amu / (cst.L0 ** 2) * pressures / cst.kB / temperatures
    )

    log10_alpha = jnp.log10(alphas)

    # if collision_dict['temperature'].shape[0] > 1:
    # Interpolation on temperatures for each wavelength point
    cia_temperature_interpolated = vmap(
        lambda alpha: jnp.interp(temperatures, binned_temperatures, alpha)
    )(log10_alpha)
    cia_temperature_interpolated = jnp.exp(cia_temperature_interpolated * jnp.log(10))

    # Interpolation on frequencies for each temperature point
    # TODO do we want left/right=0 or extrapolate from boundary values?
    cia_frequency_interpolated = vmap(
        lambda cia_opacity: jnp.interp(cst.c / frequencies, binned_wavelengths, cia_opacity, left=0.0, right=0.0)
    )(cia_temperature_interpolated.T).T

    # Ensure no values below sys.float_info.min
    cia_opacities = jnp.where(cia_frequency_interpolated < sys.float_info.min, 0, cia_frequency_interpolated)

    return cia_opacities * factor
