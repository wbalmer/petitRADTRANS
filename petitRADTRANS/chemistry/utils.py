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

    volume_mixing_ratios = {}

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        volume_mixing_ratios[species] = volume_mixing_ratio * get_species_molar_mass(species) / mean_molar_masses

    return volume_mixing_ratios
