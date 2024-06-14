import copy
import sys
import warnings

import numpy as np
import numpy.typing as npt

from petitRADTRANS._input_data_loader import get_species_basename, get_species_isotopologue_name
from petitRADTRANS.chemistry.prt_molmass import (element_symbol2element_number, get_species_molar_mass,
                                                 get_species_elements)


_solar_elemental_abundances = (  # Source: Lodders 2020 https://arxiv.org/abs/1912.00844
    # atomic number, log10 elemental abundance, uncertainty
    """
1 12.00 0.00
2 10.924 0.02
3 3.27 0.03
4 1.31 0.04
5 2.77 0.03
6 8.47 0.06
7 7.85 0.12
8 8.73 0.07
9 4.61 0.09
10 8.15 0.10
11 6.27 0.03
12 7.52 0.02
13 6.42 0.03
14 7.51 0.01
15 5.43 0.03
16 7.15 0.03
17 5.23 0.06
18 6.50 0.10
19 5.07 0.02
20 6.27 0.03
21 3.04 0.03
22 4.90 0.03
23 3.95 0.03
24 5.63 0.02
25 5.47 0.03
26 7.45 0.02
27 4.86 0.02
28 6.20 0.03
29 4.24 0.04
30 4.61 0.60
31 3.07 0.02
32 3.59 0.03
33 2.29 0.03
34 3.34 0.03
35 2.60 0.09
36 3.22 0.08
37 2.36 0.03
38 2.88 0.02
39 2.15 0.02
40 2.55 0.04
41 1.4 0.04
42 1.92 0.04
44 1.77 0.01
45 1.04 0.02
46 1.65 0.02
47 1.21 0.02
48 1.71 0.02
49 0.76 0.02
50 2.07 0.03
51 1.06 0.05
52 2.18 0.02
53 1.71 0.15
54 2.25 0.08
55 1.08 0.02
56 2.17 0.02
57 1.17 0.02
58 1.58 0.02
59 0.75 0.01
60 1.45 0.01
62 0.94 0.02
63 0.51 0.02
64 1.05 0.02
65 0.31 0.02
66 1.12 0.02
67 0.46 0.02
68 0.92 0.02
69 0.11 0.02
70 0.91 0.02
71 0.09 0.02
72 0.70 0.03
73 -0.16 0.02
74 0.67 0.04
75 0.23 0.03
76 1.33 0.03
77 1.31 0.02
78 1.60 0.03
79 0.80 0.03
80 1.08 0.15
81 0.76 0.04
82 2.03 0.03
83 0.66 0.03
90 0.04 0.02
92 -0.54 0.03
"""
)


def _compute_h_ratios(elemental_abundances: dict[int, float]) -> dict[int, float]:
    """Calculate the ratio over the hydrogen elemental abundance of other elements."""
    if 1 not in elemental_abundances:
        raise ValueError("cannot calculate hydrogen ratios if no hydrogen is present "
                         "(key '1' was lacking in provided dict)")

    if np.any(elemental_abundances[1] <= 0):
        raise ValueError(f"hydrogen abundance must always be > 0, but was {elemental_abundances[1]}")

    return {
        atomic_number: abundance / elemental_abundances[1]
        for atomic_number, abundance in elemental_abundances.items()
    }


def _compute_z_ratios(elemental_abundances: dict[int, float]) -> float:
    """Calculate the metal to non-metal abundances from elemental abundances."""
    if 1 not in elemental_abundances:
        elemental_abundances[1] = 0

    if 2 not in elemental_abundances:
        elemental_abundances[2] = 0

    if np.any(elemental_abundances[1] <= 0) and np.any(elemental_abundances[2] <= 0):
        raise ValueError(f"cannot calculate metallicity if no hydrogen and no helium is present "
                         f"(H = {elemental_abundances[1]}, He = {elemental_abundances[2]})")

    non_metal_abundances = elemental_abundances[1] + elemental_abundances[2]
    metal_abundances = np.sum(np.array(list(elemental_abundances.values()))[2:], axis=0)

    return metal_abundances / non_metal_abundances


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


def fill_atmospheric_layer(mass_fractions: dict[str, np.ndarray[float]], filling_species: dict
                           ) -> dict[str, np.ndarray[float]]:
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
    indices = [np.arange(0, length, scaling, dtype=int)]

    for interval in intervals:
        indices.append(np.arange(*interval, dtype=int))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices


def get_solar_elemental_abundances() -> dict[int, float]:
    """Parse the solar elemental abundances string."""
    abundances = _solar_elemental_abundances.split('\n')

    dictionary = {}

    for abundance in abundances:
        if abundance == '':
            continue

        atomic_number, abundance, uncertainty = abundance.split(' ')

        atomic_number = int(atomic_number)
        abundance = float(abundance)
        # uncertainty = float(uncertainty)

        dictionary[atomic_number] = 10 ** abundance

    return dictionary


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


def mass_fractions2metallicity(mass_fractions: dict[str, np.ndarray[float]], mean_molar_masses: npt.NDArray[float]):
    """Calculate the metallicity and element-over-hydrogen abundance ratios.

    Args:
        mass_fractions:
            Dictionary of mass fractions for all atmospheric species.
            Dictionary keys are the species names. Values are the mass fractions of each species at each layer.
        mean_molar_masses:
            The atmospheric mean molecular weight in amu, at each atmospheric layer.
    Returns:
        The atmospheric metallicity and a dictionary containing element-over-hydrogen abundance ratios.
        The dictionary keys are the elements atomic numbers. Values are dictionaries containing the plain-text H ratio,
        the atmospheric H ratio ('local' key), and its value relative to the solar H ratio ('relative to solar' key)
    """
    volume_mixing_ratios = mass_fractions2volume_mixing_ratios(
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses
    )

    return volume_mixing_ratios2metallicity(volume_mixing_ratios)


def simplify_species_list(species_list: list) -> list:
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


def volume_mixing_ratios2metallicity(volume_mixing_ratios: dict[str, np.ndarray[float]]):
    """Calculate the metallicity and element-over-hydrogen abundance ratios.

    Args:
        volume_mixing_ratios:
            Dictionary of volume mixing ratios for all atmospheric species.
            Dictionary keys are the species names. Values are the VMR of each species at each layer.
    Returns:
        The atmospheric metallicity and a dictionary containing element-over-hydrogen abundance ratios.
        The dictionary keys are the elements atomic numbers. Values are dictionaries containing the plain-text H ratio,
        the atmospheric H ratio ('local' key), and its value relative to the solar H ratio ('relative to solar' key)
    """
    elements = {}

    sum_vmrs = 0.0

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        if species == 'e-':
            _elements = {'H': 1}
        else:
            _elements = get_species_elements(species)

        sum_vmrs += volume_mixing_ratio

        for symbol, amount in _elements.items():
            if symbol not in elements:
                elements[symbol] = amount * volume_mixing_ratio
            else:
                elements[symbol] += amount * volume_mixing_ratio

    if 'H' not in elements:
        raise ValueError("cannot calculate metallicity if no hydrogen is present in atmosphere")

    if 'He' not in elements:
        elements['He'] = 0.0

    if np.ndim(sum_vmrs) == 0:
        sum_vmrs = np.array([sum_vmrs])

    for i, sum_vmr in enumerate(sum_vmrs):
        if np.abs(sum_vmr - 1.0) > 1e-6:
            warnings.warn(f"the sum of volume mixing ratios at level {i} is not 1 ({sum_vmr}), "
                          f"results may be inaccurate")

    # Convert keys from symbol to atomic number
    elemental_abundances = {
        element_symbol2element_number(symbol): amount
        for symbol, amount in elements.items()
    }

    solar_h_ratios = _compute_h_ratios(get_solar_elemental_abundances())
    h_ratios = _compute_h_ratios(elemental_abundances)

    solar_z_ratio = _compute_z_ratios(solar_h_ratios)
    z_ratio = _compute_z_ratios(h_ratios)

    metallicity = z_ratio / solar_z_ratio

    h_ratios = {
        list(h_ratios.keys())[i]: {
            'description': f"{list(elements.keys())[i]}/H",
            'local': h_ratio, 'relative to solar': h_ratio / solar_h_ratios[list(h_ratios.keys())[i]]
        }
        for i, h_ratio in enumerate(h_ratios.values())
        if i != 0
    }

    return metallicity, h_ratios
