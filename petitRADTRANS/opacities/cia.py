import copy
import jax.numpy as jnp
from jax import vmap, debug
from petitRADTRANS.utils import LockedDict
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.opacities.hminus import get_non_cia_gas_continuum_contributions
from petitRADTRANS.chemistry.utils import get_base_species_mass_fractions

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE


# Number density factor of the CIA opacity, (1 / L0) ** 2 / amu / kB.
_CIA_NUMBER_DENSITY_FACTOR = jnp.array(
    (1.0 / cst.L0) ** 2 / cst.amu / cst.kB, dtype=WORKING_JAX_DTYPE
)
_CIA_OPACITY_CUTOFF = jnp.finfo(WORKING_JAX_DTYPE).tiny


def get_cia_aliases(name: str) -> str:
    # Try to match name with aliases
    cia_aliases = {
        "H2--H2": ["H2-H2", "H2--H2"],
        "H2--He": ["H2-He", "H2--He", "He-H2", "He--H2"],
        "H2O--H2O": ["H2O-H2O", "H2O--H2O"],
        "H2O--N2": ["H2O-N2", "H2O--N2", "N2-H2O", "N2--H2O"],
        "N2--H2": ["N2-H2", "N2--H2", "H2-N2", "H2--N2"],
        "N2--He": ["N2-He", "N2--He", "He-N2", "He--N2"],
        "N2--N2": ["N2-N2", "N2--N2"],
        "O2--O2": ["O2-O2", "O2--O2"],
        "N2--O2": ["N2-O2", "N2--O2", "O2-N2", "O2--N2"],
        "CO2--CO2": ["CO2-CO2", "CO2--CO2"],
    }

    for cia, aliases in cia_aliases.items():
        if name in aliases:
            return cia

    # Name does not match a directory, a directory shortcut, or an alias
    return name


def init_cia_loaded_opacities(cia_contributors):
    tmp_collision_dict = LockedDict.build_and_lock(
        {
            "molecules": None,
            "weight": None,
            "lambda": None,
            "temperature": None,
            "alpha": None,
        }
    )

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


def compute_cia_opacities(
    cia_dicts, mass_fractions, pressures, temperatures, frequencies, mean_molar_masses
):
    """Compute CIA opacities.
    Wrapper to interpolate_cia, calculating each collision's combined mass fraction.

    Args:
        cia_dicts:
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
        mass_fractions:
            Dictionary containing mass-fraction profiles for all atmospheric species
            that have a requested opacity associated to them. Shape: dictionary with
            shape (N_layers) array per species inside.
        pressures (ArrayLike):
            pressures of the atmospheric layers (cgs units). Shape (N_layers).
        temperatures (ArrayLike):
            temperature within the layers (in K). Shape (N_layers).
        frequencies (ArrayLike):
            frequencies of the Radtrans object (bin centers).
        mean_molar_masses:
            Mean molecular mass profile in amu. Shape (N_layers).

    Returns:
        cia_opacities (ArrayLike):
            CIA continuum absorption opacities (N_frequencies, N_layers) in cgs (cm^2/g).
    """

    # Add mass fractions sharing the same basename
    # Sort keys to match JAX JIT's alphabetical ordering of dict keys
    species_basenames = []
    for species in sorted(mass_fractions.keys()):
        # From the doc string:
        #   The species base name is the species chemical formula without isotopic information. It may contain cloud
        #   information and charge. E.g, 24Mg2-28Si-16O4(s)_crystalline_000__DHS
        #   becomes Mg2SiO4.
        species_basenames.append(Opacity.get_species_base_name(species))

    species_basenames = tuple(species_basenames)

    # How much mass fraction is associated with the base names, neglecting
    # (summing) over "finer" effects such as isotopologues.
    base_species_mass_fractions = get_base_species_mass_fractions(
        mass_fractions, species_basenames
    )

    # Calculate CIA opacities
    cia_opacities = 0

    for collision, collision_dict in cia_dicts.items():
        # Calculate combined mass fraction
        combined_mass_fraction = 1

        for collision_species in collision_dict["molecules"]:
            if collision_species in base_species_mass_fractions:
                combined_mass_fraction = (
                    combined_mass_fraction
                    * base_species_mass_fractions[collision_species]
                )
            else:
                raise ValueError( # Dictionaries keys and CIA
                                  # species names are never traced
                                  # so this error here would fire
                                  # corretly at trace time.
                    f"species '{collision_species}' of CIA '{collision}' "
                    f"not found in mass fractions dict "
                    f"(listed species: {list(mass_fractions.keys())}, "
                    f"listed species base names {list(base_species_mass_fractions.keys())}:)"
                )

        # Add CIA opacities
        cia_opacities += interpolate_cia(
            collision_dict=collision_dict,
            combined_mass_fractions=combined_mass_fraction,
            pressures=pressures,
            temperatures=temperatures,
            frequencies=frequencies,
            mean_molar_masses=mean_molar_masses,
        )

    return cia_opacities


def interpolate_cia(
    collision_dict,
    combined_mass_fractions,
    pressures,
    temperatures,
    frequencies,
    mean_molar_masses,
):
    """Interpolate CIA cross-sections onto the Radtrans (wavelength, temperature) grid and convert it into
    opacities.

    Args:
        collision_dict:
            An entry of the cia_dict of dicts looking like this for, for example, H2--H2 collisions:
            "H2--H2": {
                    "molecules":   ["H2", "H2"],    # Collision partners
                    "weight":      scalar,      # product of the colliders' molar masses
                    "lambda":      (n_lambda,), # wavelengths in cm
                    "temperature": (n_T,),
                    "alpha":       (n_lambda, n_T),  # Needed to calculate the CIA opacities as
                                                     # function of wavelength, pressure and temperature.
            }
        combined_mass_fractions: combined mass fractions of the colliding species
            e.g., for H2-He and an atmosphere with H2 and He MMR of respectively 0.74 and 0.24,
            combined_mas_fractions = 0.74 * 0.24
            combined_mas_fractions is divided by the combined weight later (e.g. for H2 and He, 2 * 4 AMU^2), so there is
            no units issue.
        pressures (ArrayLike):
            pressures of the atmospheric layers (cgs units). Shape (N_layers).
        temperatures (ArrayLike):
            temperature within the layers (in K). Shape (N_layers).
        frequencies (ArrayLike):
            frequencies of the Radtrans object (bin centers).
        mean_molar_masses:
            Mean molecular mass profile in amu. Shape (N_layers).

    Returns:
        A (wavelength, temperature) array containing the CIA opacities.
    """

    weights = jnp.asarray(collision_dict["weight"], dtype=WORKING_JAX_DTYPE)
    alphas = jnp.asarray(collision_dict["alpha"], dtype=WORKING_JAX_DTYPE)
    binned_temperatures = jnp.asarray(
        collision_dict["temperature"], dtype=WORKING_JAX_DTYPE
    )
    binned_wavelengths = jnp.asarray(
        collision_dict["lambda"], dtype=WORKING_JAX_DTYPE
    )

    # TODO Paul_review : continue from here

    factor = (
        combined_mass_fractions
        / weights
        * mean_molar_masses
        * _CIA_NUMBER_DENSITY_FACTOR
        * pressures
        / temperatures
    )

    log10_alpha = jnp.log10(alphas) # TODO: better to safe as log10(alpha) from the start?

    # if collision_dict['temperature'].shape[0] > 1:
    # Interpolation on temperatures for each wavelength point
    cia_temperature_interpolated = vmap(
        lambda alpha: jnp.interp(temperatures, binned_temperatures, alpha)
    )(log10_alpha)
    cia_temperature_interpolated = jnp.exp(cia_temperature_interpolated * jnp.log(10))

    # Interpolation on frequencies for each temperature point
    # TODO do we want left/right=0 or extrapolate from boundary values?
    cia_frequency_interpolated = vmap(
        lambda cia_opacity: jnp.interp(
            cst.c / frequencies, binned_wavelengths, cia_opacity, left=0.0, right=0.0
        )
    )(cia_temperature_interpolated.T).T

    # Ensure no values below sys.float_info.min
    cia_opacities = jnp.where(
        cia_frequency_interpolated < _CIA_OPACITY_CUTOFF, 0, cia_frequency_interpolated
    )

    return cia_opacities * factor
