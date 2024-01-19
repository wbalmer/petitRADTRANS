import copy
import sys

import numpy as np

from petitRADTRANS._input_data_loader import get_species_basename, get_species_isotopologue_name
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass


def compute_mean_molar_masses(abundances):
    """Calculate the mean molecular weight in each layer.

    Args:
        abundances : dict
            dictionary of abundance arrays, each array must have the shape of the pressure array used in pRT,
            and contain the abundance at each layer in the atmosphere.
    """
    mean_molar_masses = (sys.float_info.min
                         * np.ones_like(abundances[list(abundances.keys())[0]]))  # prevent division by 0

    for key in abundances.keys():
        if '(s)' in key or '(l)' in key:  # ignore clouds
            continue

        # exo_k resolution
        spec = key.split(".R")[0]
        mean_molar_masses += abundances[key] / get_species_molar_mass(spec)

    return 1.0 / mean_molar_masses


def fill_atmospheric_layer(mass_fractions: dict[str, float], filling_species: dict) -> dict[str, float]:
    """Fill an atmospheric layer with filling species, so that the sum of the mass fractions is 1.
    The filling species values are weights that are used to fill the atmospheric layer following:
        X_i = w_i / sum(w) * X_f,
    where X_i is the mass fraction of filling species i, w_i is the weight of species i, and X_f is the amount
    of mass fraction missing to get a sum of 1.

    The value X_f is calculated from:
        X_f = 1 - sum(X_nf),
    where X_nf is the sum of mass fractions that are not filling species.

    The filling species can be "weighted" (if the weights are all floats) or "unweighted" (if the weights are all None).
    In weighted mode, w corresponds to the values of `filling_species`.
    In unweighted mode, the mass fractions are used to calculate the weights. All the filling species must be in the
    mass fractions.
    Mixing weighted and unweighted filling species raises a ValueError.

    Args:
        mass_fractions:
            Mass fractions in the atmosphere. Must be a directory with the species as keys and the mass fraction at one
            layer.
        filling_species:
            Species to fill the atmosphere with. Must be a directory with the species as keys and the weights of the
            mass fractions. Unweighted filling species are represented with None.
    Returns:
        A dictionary of the mass fractions with the filling species. The sum of the mass fractions is 1.
    """
    # Change nothing if no filling species given
    if filling_species is None or len(filling_species) == 0:
        return mass_fractions

    # Check if all filling species mass fractions are None
    unweighted_filling = False

    if None in list(filling_species.values()):
        for species, mass_fraction in filling_species.items():
            if mass_fraction is not None:
                raise ValueError(f"cannot mix weighted and unweighted filling species\n"
                                 f"Ensure that all filling species are unweighted using None, "
                                 f"or that all filling species are weighted using floats or arrays\n"
                                 f"Filling species dict was: {filling_species}")

        unweighted_filling = True

    # Check if the use of unweighted filling species is permitted
    if unweighted_filling:
        for species in filling_species:
            if species not in mass_fractions:
                raise KeyError(f"unweighted filling species '{species}' not in mass fractions dict\n"
                               f"The list of mass fraction species was: {list(mass_fractions.keys())}")

        sum_weights = 0
    else:
        sum_weights = np.sum(list(filling_species.values()), axis=0)

    # Calculate the sums
    mass_fraction_to_fill = 1 - (
        np.sum([
            mass_fraction for species, mass_fraction in mass_fractions.items()
            if species not in filling_species
        ])
    )

    last_species = list(filling_species.keys())[-1]

    # Fill the atmosphere
    if mass_fraction_to_fill > 0:
        if unweighted_filling:  # initialize weights of unweighted filling species using mass fractions dict
            for species in filling_species:
                filling_species[species] = copy.deepcopy(mass_fractions[species])
                sum_weights += filling_species[species]

        for species, weight in filling_species.items():
            if species != last_species:
                mass_fractions[species] = weight / sum_weights * mass_fraction_to_fill
            else:  # ensure that the sum is exactly 1 in spite of numerical errors
                mass_fraction_from_weights = weight / sum_weights * mass_fraction_to_fill
                mass_fractions[species] = 0
                mass_fraction_from_subtraction = 1 - np.sum(np.array(list(mass_fractions.values())))

                # Sanity check
                if np.isclose(mass_fraction_from_weights, mass_fraction_from_subtraction, atol=1e-15, rtol=1e-12):
                    mass_fractions[species] = mass_fraction_from_subtraction
                else:
                    raise ValueError(f"unexpected results when filling atmosphere\n"
                                     f"A mass fraction of {mass_fraction_from_subtraction} is required to fill the "
                                     f"atmosphere with species {last_species}, "
                                     f"but from the parameters given, "
                                     f"the calculated mass fraction is {mass_fraction_from_weights}.\n"
                                     f"This is likely a coding error and not a user error.")

    return mass_fractions


def fixed_length_amr(p_clouds, pressures, scaling=10, width=3):
    r"""This function takes in the cloud base pressures for each cloud,
    and returns an array of pressures with a high resolution mesh
    in the region where the clouds are located.

    Author:  Francois Rozet.

    The output length is always
        len(pressures[::scaling]) + len(p_clouds) * width * (scaling - 1)

    Args:
        p_clouds : numpy.ndarray
            The cloud base pressures in bar
        pressures : np.ndarray
            The high resolution pressure array.
        scaling : int
            The factor by which the low resolution pressure array is scaled
        width : int
            The number of low resolution bins to be replaced for each cloud layer.
    """

    length = len(pressures)
    cloud_indices = np.searchsorted(pressures, np.asarray(p_clouds))

    # High resolution intervals
    def bounds(center: int, _width: int) -> [int, int]:
        upper = min(center + _width / 2, length)
        lower = max(upper - _width, 0)

        return lower, lower + _width

    intervals = [bounds(idx, scaling * width) for idx in cloud_indices]

    # Merge intervals
    while True:
        intervals, stack = sorted(intervals), []

        for interval in intervals:
            if stack and stack[-1][1] >= interval[0]:
                last = stack.pop()
                interval = bounds(
                    (last[0] + max(last[1], interval[1]) + 1) // 2,
                    last[1] - last[0] + interval[1] - interval[0],
                )

            stack.append(interval)

        if len(intervals) == len(stack):
            break
        intervals = stack

    # Intervals to indices
    indices = [np.arange(0, length, scaling)]

    for interval in intervals:
        indices.append(np.arange(*interval))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices


def mass_fractions2volume_mixing_ratios(mass_fractions, mean_molar_masses=None):
    """Convert mass fractions to volume mixing ratios.

    Args:
        mass_fractions : dict
            A dictionary of mass fractions
        mean_molar_masses : numpy.ndarray
            An array containing all mass fractions at each pressure level
    """
    if mean_molar_masses is None:
        mean_molar_masses = compute_mean_molar_masses(mass_fractions)

    volume_mixing_ratios = {}

    for species, mass_fraction in mass_fractions.items():
        volume_mixing_ratios[species] = mass_fraction / get_species_molar_mass(species) * mean_molar_masses

    return volume_mixing_ratios


def simplify_species_list(species_list: list, specify_natural_abundance: bool = False) -> list:
    species_basenames = [get_species_basename(species) for species in species_list]
    species_isotopologue_names = [get_species_isotopologue_name(species) for species in species_list]

    for i, species in enumerate(species_isotopologue_names):
        # Use isotopologue name if species is not the main isotopologue
        species_main_name = get_species_isotopologue_name(species=species_basenames[i])
        species_natural_abundance_name = get_species_isotopologue_name(species=species_basenames[i] + '-NatAbund')

        if species != species_main_name and species != species_natural_abundance_name:
            species_basenames[i] = species

    return species_basenames


def volume_mixing_ratios2mass_fractions(volume_mixing_ratios, mean_molar_masses=None):
    """Convert mass fractions to volume mixing ratios.

    Args:
        volume_mixing_ratios : dict
            A dictionary of volume mixing ratios
        mean_molar_masses : numpy.ndarray
            An array containing all mass fractions at each pressure level
    """
    if mean_molar_masses is None:
        mean_molar_masses = compute_mean_molar_masses(volume_mixing_ratios)

    mass_fractions = {}

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        mass_fractions[species] = volume_mixing_ratio * get_species_molar_mass(species) / mean_molar_masses

    return mass_fractions
