import copy

import jax.numpy as jnp
from jax import jit
from interpax import PchipInterpolator

from numpy import argsort
from jax.typing import ArrayLike

from functools import lru_cache, partial

from petitRADTRANS.chemistry.prt_molmass import (
    element_symbol2element_number,
    get_species_elements,
    get_species_molar_mass,
)
from petitRADTRANS.opacities.opacities import Opacity


def build_adaptive_pressure_grid(
    pressures: ArrayLike,
    cloud_base_pressures: ArrayLike,
    refinement_strength: float = 4.0,
    refinement_width_in_cells: float = 3.0,
):
    """Redistribute a fixed number of pressure layers around cloud bases.

    The returned pressure grid preserves both the number of layers and the
    pressure bounds of the input grid. Instead of inserting extra layers, the
    method smoothly compresses the log-pressure coordinate near the supplied
    cloud-base pressures so more points land where cloud placement matters.

    Args:
        pressures:
            Monotonic pressure grid in bar.
        cloud_base_pressures:
            Cloud-base pressures in bar. Invalid or out-of-range values are
            ignored.
        refinement_strength:
            Dimensionless multiplier controlling how strongly the grid is
            compressed near cloud bases.
        refinement_width_in_cells:
            Approximate width of the refinement region measured in units of the
            original log-pressure cell width.

    Returns:
        Pressure grid in bar with the same shape as ``pressures``.
    """
    pressures = jnp.asarray(pressures)
    cloud_base_pressures = jnp.asarray(cloud_base_pressures)

    n_layers = pressures.shape[0]
    if n_layers <= 2:
        return pressures

    log_pressures = jnp.log10(pressures)
    log_pressure_min = log_pressures[0]
    log_pressure_span = log_pressures[-1] - log_pressure_min

    unit_coordinates = jnp.linspace(0.0, 1.0, n_layers, dtype=log_pressures.dtype)
    cell_centers = 0.5 * (unit_coordinates[:-1] + unit_coordinates[1:])

    normalized_cloud_positions = (
        jnp.log10(cloud_base_pressures) - log_pressure_min
    ) / log_pressure_span
    valid_clouds = jnp.logical_and(
        jnp.isfinite(cloud_base_pressures),
        jnp.logical_and(cloud_base_pressures > 0, jnp.isfinite(normalized_cloud_positions)),
    )
    valid_clouds = jnp.logical_and(
        valid_clouds,
        jnp.logical_and(normalized_cloud_positions >= 0.0, normalized_cloud_positions <= 1.0),
    )

    if cloud_base_pressures.ndim == 0:
        normalized_cloud_positions = normalized_cloud_positions[None]
        valid_clouds = valid_clouds[None]

    sigma = jnp.asarray(
        refinement_width_in_cells / (n_layers - 1), dtype=log_pressures.dtype
    )
    sigma = jnp.maximum(sigma, jnp.asarray(1e-6, dtype=log_pressures.dtype))

    cell_offsets = cell_centers[:, None] - normalized_cloud_positions[None, :]
    gaussian_windows = jnp.exp(-0.5 * (cell_offsets / sigma) ** 2)
    refinement_profile = jnp.sum(
        gaussian_windows * valid_clouds[None, :], axis=1
    )

    # Lower cell weights compress local spacing and therefore place more
    # pressure samples near the corresponding cloud base.
    cell_weights = 1.0 / (1.0 + refinement_strength * refinement_profile)
    cumulative = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=log_pressures.dtype),
            jnp.cumsum(cell_weights, dtype=log_pressures.dtype),
        ]
    )
    normalized = cumulative / cumulative[-1]

    refined_log_pressures = log_pressure_min + normalized * log_pressure_span
    refined_log_pressures = refined_log_pressures.at[0].set(log_pressures[0])
    refined_log_pressures = refined_log_pressures.at[-1].set(log_pressures[-1])

    return jnp.power(10.0, refined_log_pressures)


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
        raise ValueError(
            "cannot calculate hydrogen ratios if no hydrogen is present "
            "(key '1' was lacking in provided dict)"
        )

    # checkify.check(
    #        jnp.all(elemental_abundances[1] > 0),
    #        f"hydrogen abundance must always be > 0, but was {elemental_abundances[1]}"
    # )
    #        f"results may be inaccurate"
    # if jnp.any(elemental_abundances[1] <= 0):
    #    raise ValueError(f"hydrogen abundance must always be > 0, but was {elemental_abundances[1]}")

    return {
        atomic_number: abundance / elemental_abundances[1]
        for atomic_number, abundance in elemental_abundances.items()
    }


def _compute_z_ratios(
    elemental_abundances: dict[int, float], sort: bool = True, neglect_he: bool = False
) -> float:
    """Calculate the metal to non-metal abundances from elemental abundances."""
    if 1 not in elemental_abundances:
        elemental_abundances[1] = 0

    if 2 not in elemental_abundances:
        elemental_abundances[2] = 0

    # if jnp.any(elemental_abundances[1] <= 0) and jnp.any(elemental_abundances[2] <= 0):
    #    raise ValueError(f"cannot calculate metallicity if no hydrogen and no helium is present "
    #                     f"(H = {elemental_abundances[1]}, He = {elemental_abundances[2]})")

    non_metal_abundances = elemental_abundances[1]

    if not neglect_he:
        non_metal_abundances += elemental_abundances[2]

    if not sort:
        sorted_abundances = jnp.array(
            list(elemental_abundances.values())[i]
            for i in jnp.argsort(list(elemental_abundances.keys()))
        )
    else:
        sorted_abundances = jnp.array(list(elemental_abundances.values()))

    metal_abundances = jnp.sum(sorted_abundances[2:], axis=0)

    return metal_abundances / non_metal_abundances


def abundance_curvature_prior(press, abundance, gamma):
    """
    Compute a curvature prior for an abundance-pressure profile.

    This function calculates a curvature prior for an abundance-pressure profile,
    penalizing deviations from a smooth, low-curvature profile, based on Line 2015

    Args:
        press (array-like): An array or list of pressure data points.
        abundance (array-like): An array or list of abundance data points.
        gamma (float): The curvature penalization factor.

    Returns:
        float: The curvature prior value.
    """
    weighted_abund_prior = (
        -0.5
        * jnp.sum((abundance[2:] - 2 * abundance[1:-1] + abundance[:-2]) ** 2)
        / gamma
    )
    weighted_abund_prior -= 0.5 * jnp.log(2 * jnp.pi * gamma)

    return weighted_abund_prior


def get_parameter_value(parameter):
    if hasattr(parameter, "value"):
        return parameter.value

    return parameter


# @partial(jit, static_argnames=['mode', 'nnodes'])
def compute_pressure_nodes(pressure_array, mode="even", nnodes=0, points_list=None):
    """
    Compute the location of nodes for a spline or step profile in pressure space.

    If mode is even, the nodes are evenly spaced in log pressure, with a total of
    nnodes + 2 points (including the top and bottom of the atmospheres).
    If mode is relative, then points list is returned.

    Args:
        pressure_array (array-like): An array or list of pressure levels which is used to calculate the spectrum.
        mode (std): 'even' (evenly spaced), 'relative' (points list is monotonically decreasing in log P) or 'set'
            (points list is user-defined spline nodes)
        nnodes (int): Number of intermediate spline nodes (not counting top and bottom of atmosphere)
        points_list (array_like): if relative, list of differences in log P between each node. If set.
            then this is a user-defined list of spline points, and MUST include top and bottom pressure
            values.

    Returns:
        array: an array of pressure nodes.
    """
    pressure_array = jnp.asarray(pressure_array)
    log_top_pressure = jnp.log10(pressure_array[0])
    log_bottom_pressure = jnp.log10(pressure_array[-1])

    if mode == "even":
        return jnp.linspace(log_top_pressure, log_bottom_pressure, int(nnodes) + 2)

    if mode == "relative":
        if points_list is None:
            return jnp.array([log_top_pressure, log_bottom_pressure])

        relative_offsets = jnp.asarray(points_list)
        interior_nodes = log_top_pressure + jnp.cumsum(relative_offsets)
        interior_nodes = jnp.minimum(interior_nodes, log_bottom_pressure)

        return jnp.concatenate(
            (
                jnp.array([log_top_pressure]),
                interior_nodes,
                jnp.array([log_bottom_pressure]),
            )
        )

    if points_list is None:
        return jnp.array([log_top_pressure, log_bottom_pressure])

    return jnp.asarray(points_list)


# @partial(jit, static_argnames=['mode'])
@lru_cache(maxsize=None)
def _get_cached_species_molar_mass(species: str) -> float:
    return get_species_molar_mass(species.split(".R")[0])


def compute_mean_molar_masses(abundances: dict[str, ArrayLike], mode: str = "mmr"):
    """Calculate the mean molecular weight in each layer, from mass fractions or volume mixing ratios.

    The calculation is slightly different for mass fractions and volume mixing ratios. By default, this function assumes
    that the abundances are mass fractions. To calculate the mean molar masses from volume mixing ratios, set the 'mode'
    argument to 'vmr'.

    This function is JIT-compatible.

    Args:
        abundances : dict
            Dictionary of mass fraction or volume mixing ratio arrays, each array must have the shape of the pressure
            array used in pRT, and contain the abundance at each layer in the atmosphere.
        mode : str
            Can be 'mmr' or 'vmr', to calculate the mean molar masses from mass fraction or volume mixing ratios,
            respectively.
    """
    if len(abundances) == 0:
        raise ValueError(
            "abundance dictionary must contains at least one species, but is empty"
        )

    if mode not in ("mmr", "vmr"):
        raise ValueError(f"'mode' can be 'mmr' or 'vmr', but was '{mode}'")

    first_abundance = next(iter(abundances.values()))
    mean_molar_masses = jnp.zeros_like(first_abundance)
    use_mass_fractions = mode == "mmr"

    # Species names are static, so caching molar masses avoids reparsing them on every call.
    for species, abundance in abundances.items():
        if "(s)" in species or "(l)" in species:
            continue

        molar_mass = _get_cached_species_molar_mass(species)
        if use_mass_fractions:
            mean_molar_masses = mean_molar_masses + abundance / molar_mass
        else:
            mean_molar_masses = mean_molar_masses + abundance * molar_mass

    if use_mass_fractions:
        mean_molar_masses = jnp.where(
            mean_molar_masses == 0,
            jnp.finfo(mean_molar_masses.dtype).tiny,
            mean_molar_masses,
        )
        return 1.0 / mean_molar_masses

    return mean_molar_masses


def compute_mean_molar_masses_from_volume_mixing_ratios(volume_mixing_ratios):
    """Calculate the mean molecular weight in each layer from volume mixing ratios.

    Args:
        volume_mixing_ratios : dict
            dictionary of volume mixing ratios arrays, each array must have the shape of the pressure array used in pRT,
            and contain the abundance at each layer in the atmosphere.
    """
    mean_molar_masses = jnp.zeros_like(list(volume_mixing_ratios.values())[0])

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        if ("(s)" in species) or ("(l)" in species):  # ignore clouds
            continue

        spec = species.split(".R")[0]
        mean_molar_masses += volume_mixing_ratio * get_species_molar_mass(spec)

    return mean_molar_masses


def cubic_spline_profile(
    pressure_array, pressure_nodes, abundance_points, gamma
):
    """
    Compute a cubic spline profile for abundance based on pressure points.

    This function computes a cubic spline profile for abundance using
    pressure and abundance data points, along with a curvature prior.

    Args:
        pressure_array (array-like): An array or list of pressure levels which is used to calculate the spectrum.
        pressure_nodes (array-like): An array or list of log pressure points at which the spline is fixed
        abundance_points (array-like): An array or list of abundances for each pressure node.
        gamma (float): A parameter controlling the curvature of the spline.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated abundance values
              based on the cubic spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """
    # check that pressure_nodes is sorted
    # checkify.check(pressure_nodes)

    cs = PchipInterpolator(pressure_nodes, abundance_points, check=False)
    interpolated_abunds = 10 ** cs(jnp.log10(pressure_array))
    prior = abundance_curvature_prior(pressure_array, interpolated_abunds, gamma)
    return interpolated_abunds, prior


@partial(jit, static_argnames=["profile_mode"])
def build_abundance_profile(
    pressure_array,
    pressure_nodes,
    abundance_points,
    profile_mode="linear",
    gamma=0.04,
):
    if profile_mode == "stepped":
        return stepped_profile(pressure_array, pressure_nodes, abundance_points)

    if profile_mode == "linear":
        return linear_spline_profile(
            pressure_array,
            pressure_nodes,
            abundance_points,
            gamma=gamma,
        )[0]

    if profile_mode == "cubic":
        return cubic_spline_profile(
            pressure_array,
            pressure_nodes,
            abundance_points,
            gamma=gamma,
        )[0]

    raise ValueError(
        "profile_mode must be 'stepped', 'linear', or 'cubic', "
        f"but was '{profile_mode}'"
    )


def define_abundance_node_list(species_short_name, parameters):
    """

    Define the location of pressure nodes, allowing individual nodes to be
    retrieved parameters.

    If mode is even, the nodes are evenly spaced in log pressure, with a total of
    nnodes + 2 points (including the top and bottom of the atmospheres).
    If mode is relative, then points list is returned.

    Args:
        species_short_name (str): which molecular species does this apply to (each can be unique)
        parameters (dict): dictionary of parameters used in a retrieval
    Returns:
        array: an array of pressure nodes
    """
    abundance_nodes = None
    if f"{species_short_name}_n_abundance_nodes" in parameters.keys():
        nnodes = int(
            get_parameter_value(parameters[f"{species_short_name}_n_abundance_nodes"])
        )
        abundance_nodes = [
            get_parameter_value(parameters[f"{species_short_name}_abundance_node_{i}"])
            for i in range(nnodes)
        ]

    # Predefined list of pressure node locations
    elif f"{species_short_name}_abundance_nodes" in parameters.keys():
        abundance_nodes = get_parameter_value(
            parameters[f"{species_short_name}_abundance_nodes"]
        )

    if abundance_nodes is None:
        return None

    return jnp.asarray(abundance_nodes)


def define_pressure_node_list(pressure_array, species_short_name, parameters):
    """
    Define the location of pressure nodes, allowing individual nodes to be
    retrieved parameters.

    If mode is even, the nodes are evenly spaced in log pressure, with a total of
    nnodes + 2 points (including the top and bottom of the atmospheres).
    If mode is relative, then points list is returned.

    Args:
        pressure_array (array-like): An array or list of pressure levels which is used to calculate the spectrum.
        species_short_name (str): which molecular species does this apply to (each can be unique)
        parameters (dict): dictionary of parameters used in a retrieval
    Returns:
        array: an array of pressure nodes, can be used in calculate_pressure_nodes
    """
    pressure_array = jnp.asarray(pressure_array)
    log_top_pressure = jnp.log10(pressure_array[0])
    log_bottom_pressure = jnp.log10(pressure_array[-1])

    if f"{species_short_name}_n_pressure_nodes" in parameters.keys():
        nnodes = int(
            get_parameter_value(parameters[f"{species_short_name}_n_pressure_nodes"])
        )
        mode = get_parameter_value(
            parameters[f"{species_short_name}_interpolation_mode"]
        )

        # Evenly spaced in log P
        if mode == "even":
            return compute_pressure_nodes(
                pressure_array,
                mode=mode,
                nnodes=nnodes,
                points_list=None,
            )

        pnodes = [
            get_parameter_value(parameters[f"{species_short_name}_pressure_node_{i}"])
            for i in range(nnodes)
        ]

        if mode == "relative":
            return compute_pressure_nodes(
                pressure_array,
                mode=mode,
                points_list=pnodes,
            )

        return jnp.concatenate(
            (
                jnp.array([log_top_pressure]),
                jnp.asarray(pnodes),
                jnp.array([log_bottom_pressure]),
            )
        )

    # Predefined list of pressure node locations
    if f"{species_short_name}_pressure_nodes" in parameters.keys():
        return jnp.asarray(
            get_parameter_value(parameters[f"{species_short_name}_pressure_nodes"])
        )

    # Default to bottom and top of atmosphere
    return jnp.array([log_top_pressure, log_bottom_pressure])


# @jit
def fill_atmosphere(
    mass_fractions: dict[str, ArrayLike],
    filling_species: dict,
    mass_fraction_sum: ArrayLike,
) -> tuple[dict[str, ArrayLike], ArrayLike]:
    """Fill an atmosphere with filling species, so that the sum of the mass fractions in all layers is 1.

    The filling species values are weights that are used to fill each atmospheric layer following:
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

    Args:
        mass_fractions: dict
            Dictionary with the species as keys and the mass fraction in all layers as values.
        filling_species: dict
            Dictionary with the filling species as keys and the weights of the mass fractions as values. Unweighted
            filling species are represented with None.
        mass_fraction_sum: ArrayLike
            The sum of the trace species mass fractions in each layer.
    Returns:
        A dictionary of the mass fractions with the filling species. The sum of the mass fractions is 1.
    """

    filled_mass_fractions = mass_fractions.copy()

    if len(filling_species) == 0:
        return filled_mass_fractions, mass_fraction_sum

    sum_weights = jnp.sum(jnp.array(list(filling_species.values())), axis=0)
    can_fill = jnp.logical_and(
        jnp.logical_and(0.0 <= mass_fraction_sum, mass_fraction_sum < 1.0),
        sum_weights > 0,
    )
    safe_sum_weights = jnp.where(sum_weights > 0, sum_weights, 1.0)

    for species, weights in filling_species.items():
        filled_mass_fractions[species] = jnp.where(
            can_fill,
            (1.0 - mass_fraction_sum) * weights / safe_sum_weights,
            filled_mass_fractions[species],
        )

    # Re-add cloud mass fractions
    for species, mass_fraction in mass_fractions.items():
        if "(s)" in species or "(l)" in species:
            filled_mass_fractions[species] = mass_fraction

    msum = jnp.zeros_like(mass_fraction_sum)
    filling_mass_sum = jnp.zeros_like(mass_fraction_sum)
    for mass_fraction in filled_mass_fractions.values():
        msum = msum + mass_fraction

    for species in filling_species:
        filling_mass_sum = filling_mass_sum + filled_mass_fractions[species]

    target_filling_mass_sum = 1.0 - (msum - filling_mass_sum)
    can_rebalance_existing_fill = jnp.logical_and(
        sum_weights <= 0,
        jnp.logical_and(filling_mass_sum > 0, target_filling_mass_sum > 0),
    )
    safe_filling_mass_sum = jnp.where(filling_mass_sum > 0, filling_mass_sum, 1.0)

    for species in filling_species:
        filled_mass_fractions[species] = jnp.where(
            can_rebalance_existing_fill,
            filled_mass_fractions[species]
            * target_filling_mass_sum
            / safe_filling_mass_sum,
            filled_mass_fractions[species],
        )

    msum = jnp.zeros_like(mass_fraction_sum)
    for mass_fraction in filled_mass_fractions.values():
        msum = msum + mass_fraction

    return filled_mass_fractions, msum


def get_solar_elemental_abundances(keep_list: list[int] = None) -> dict[int, float]:
    """Parse the solar elemental abundances string.
    Args:
        keep_list:
            List of atomic numbers to keep in the returned dictionary. If None, all atomic numbers are kept.
    """
    abundances = _solar_elemental_abundances.split("\n")

    dictionary = {}

    for abundance in abundances:
        if abundance == "":
            continue

        atomic_number, abundance, _ = abundance.split(
            " "
        )  # 3rd column contains the uncertainties, not used yet

        atomic_number = int(atomic_number)

        if keep_list is not None:
            if atomic_number not in keep_list:
                continue

        abundance = float(abundance)

        dictionary[atomic_number] = 10**abundance

    return dictionary


def linear_spline_profile(
    pressure_array, pressure_nodes, abundance_points, gamma
):
    """
    Compute a linear spline profile for abundance based on pressure points.

    This function computes a linear spline profile for abundance using
    pressure and abundance data points, along with a curvature prior.

    Args:
        pressure_array (array-like): An array or list of pressure levels which is used to calculate the spectrum.
        pressure_nodes (array-like): An array or list of log pressure points at which the spline is fixed
        abundance_points (array-like): An array or list of abundances for each pressure node.
        gamma (float): A parameter controlling the curvature of the spline.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated abundance values
              based on the linear spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """
    interpolated_abunds = 10 ** jnp.interp(
        jnp.log10(pressure_array), pressure_nodes, abundance_points
    )
    prior = abundance_curvature_prior(pressure_array, interpolated_abunds, gamma)
    return interpolated_abunds, prior


@jit(static_argnames=("species_basenames",))
def get_base_species_mass_fractions(
    mass_fractions: dict, species_basenames: tuple
) -> dict:
    base_species_mass_fractions = {}
    for i, species in enumerate(sorted(list(mass_fractions.keys()))):
        # Using Opacity.get_species_basename to get the base species name
        species_basename = species_basenames[i]

        if species_basename not in base_species_mass_fractions:
            base_species_mass_fractions[species_basename] = copy.deepcopy(
                mass_fractions[species]
            )
        else:
            base_species_mass_fractions[species_basename] += mass_fractions[species]

    return base_species_mass_fractions


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
        volume_mixing_ratios[species] = (
            mass_fraction / get_species_molar_mass(species) * mean_molar_masses
        )

    return volume_mixing_ratios


def mass_fractions2metallicity(
    mass_fractions: dict[str, ArrayLike],
    mean_molar_masses: ArrayLike,
    neglect_he: bool = False,
    only_atmospheric_species_for_solar_metallicity: bool = False,
):
    """Calculate the metallicity and element-over-hydrogen abundance ratios.

    Args:
        mass_fractions:
            Dictionary of mass fractions for all atmospheric species.
            Dictionary keys are the species names. Values are the mass fractions of each species at each layer.
        mean_molar_masses:
            The atmospheric mean molecular weight in amu, at each atmospheric layer.
        neglect_he:
            If True, helium will be neglected as a non-metal in the metallicity calculation,
            such that everything is defined with respect to hydrogen.
        only_atmospheric_species_for_solar_metallicity:
            If True, only the species also present in the planet atmosphere will be used to calculate the solar
            metallicity. If False, all species present in the sun will be used to calculate the solar metallicity.
    Returns:
        The atmospheric metallicity and a dictionary containing element-over-hydrogen abundance ratios.
        The dictionary keys are the elements atomic numbers. Values are dictionaries containing the plain-text H ratio,
        the atmospheric H ratio ('local' key), and its value relative to the solar H ratio ('relative to solar' key)
    """
    volume_mixing_ratios = mass_fractions2volume_mixing_ratios(
        mass_fractions=mass_fractions, mean_molar_masses=mean_molar_masses
    )

    return volume_mixing_ratios2metallicity(
        volume_mixing_ratios,
        neglect_he=neglect_he,
        only_atmospheric_species_for_solar_metallicity=only_atmospheric_species_for_solar_metallicity,
    )


def simplify_species_list(species_list: list) -> list:
    species_basenames = [
        Opacity.get_species_base_name(species) for species in sorted(species_list)
    ]
    species_isotopologue_names = [
        Opacity.get_species_isotopologue_name(species)
        for species in sorted(species_list)
    ]

    for i, species in enumerate(species_isotopologue_names):
        # Use isotopologue name if species is not the main isotopologue
        species_main_name = Opacity.get_species_isotopologue_name(species_basenames[i])
        species_natural_abundance_name = Opacity.get_species_isotopologue_name(
            species_basenames[i] + "-NatAbund"
        )

        if species != species_main_name and species != species_natural_abundance_name:
            species_basenames[i] = species

    return species_basenames


# @jit
def stepped_profile(pressure_array, transition_pressures, abundance_points):
    """
    Compute a stepped profile for abundance based on pressure points.

    This function computes the steps for chemical abundance using
    pressure and abundance data points.

    Args:
        pressure_array (array-like): An array or list of pressure levels which is used to calculate the spectrum.
        transition_pressures (array-like): An array of transition pressures in log10 space, sorted from low to high
                                           pressure. It should include the top and bottom of the atmosphere.
        abundance_points (array-like): An array or list of abundances for each pressure interval. A legacy
            length matching transition_pressures is also accepted; in that case the last value is used only at the
            bottom boundary.

    Returns:
        array-like: An array of abundance values corresponding to the pressure_array.
    """

    transition_pressures = jnp.asarray(transition_pressures)
    abundance_points = jnp.asarray(abundance_points)
    n_intervals = len(transition_pressures) - 1

    if len(abundance_points) not in (n_intervals, len(transition_pressures)):
        raise ValueError(
            f"Length of abundance_points ({len(abundance_points)}) must be either {n_intervals} "
            f"or {len(transition_pressures)}"
        )

    log_pressure = jnp.log10(pressure_array)
    abundance_indices = jnp.searchsorted(
        transition_pressures[1:], log_pressure, side="right"
    )
    abundance_indices = jnp.clip(abundance_indices, 0, len(abundance_points) - 1)

    return 10 ** abundance_points[abundance_indices]


def volume_mixing_ratios2mass_fractions(volume_mixing_ratios, mean_molar_masses=None):
    """Convert volume mixing ratios to mass fractions.

    Args:
        volume_mixing_ratios : dict
            A dictionary of volume mixing ratios
        mean_molar_masses : numpy.ndarray
            An array containing the mean molecular weight at each pressure level
    """
    if mean_molar_masses is None:
        mean_molar_masses = compute_mean_molar_masses(volume_mixing_ratios, mode="vmr")

    mass_fractions = {}

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        mass_fractions[species] = (
            volume_mixing_ratio * get_species_molar_mass(species) / mean_molar_masses
        )

    return mass_fractions


def volume_mixing_ratios2metallicity(
    volume_mixing_ratios: dict[str, ArrayLike],
    neglect_he: bool = False,
    only_atmospheric_species_for_solar_metallicity: bool = False,
):
    """Calculate the metallicity and element-over-hydrogen abundance ratios.

    Args:
        volume_mixing_ratios:
            Dictionary of volume mixing ratios for all atmospheric species.
            Dictionary keys are the species names. Values are the VMR of each species at each layer.
        neglect_he:
            If True, helium will be neglected as a non-metal in the metallicity calculation,
            such that everything is defined with respect to hydrogen.
        only_atmospheric_species_for_solar_metallicity:
            If True, only the species also present in the planet atmosphere will be used to calculate the solar
            metallicity. If False, all species present in the sun will be used to calculate the solar metallicity.
    Returns:
        The atmospheric metallicity and a dictionary containing element-over-hydrogen abundance ratios.
        The dictionary keys are the elements atomic numbers. Values are dictionaries containing the plain-text H ratio,
        the atmospheric H ratio ('local' key), and its value relative to the solar H ratio ('relative to solar' key)
    """
    elements = {}

    sum_vmrs = 0.0

    for species, volume_mixing_ratio in volume_mixing_ratios.items():
        if species == "e-":
            _elements = {"H": 1}
        else:
            _elements = get_species_elements(species)

        sum_vmrs += volume_mixing_ratio

        for symbol, amount in _elements.items():
            if symbol not in elements:
                elements[symbol] = amount * volume_mixing_ratio
            else:
                elements[symbol] += amount * volume_mixing_ratio

    if "H" not in elements:
        raise ValueError(
            "cannot calculate metallicity if no hydrogen is present in atmosphere"
        )

    if "He" not in elements:
        elements["He"] = 0.0

    if jnp.ndim(sum_vmrs) == 0:
        sum_vmrs = jnp.array([sum_vmrs])

    # for i, sum_vmr in enumerate(list(sum_vmrs)):
    #    checkify.check(
    #        jnp.abs(sum_vmr - 1.0) <= 1e-6,
    #        f"the sum of volume mixing ratios at level {i} is not 1 ({sum_vmr}), "
    #        f"results may be inaccurate"
    # )

    # Convert keys from symbol to atomic number
    _elemental_abundances = {
        element_symbol2element_number(symbol): amount
        for symbol, amount in elements.items()
    }

    sorted_elements_indices = argsort(list(_elemental_abundances.keys()))

    elemental_abundances = {
        list(_elemental_abundances.keys())[i]: list(_elemental_abundances.values())[i]
        for i in sorted_elements_indices
    }

    keep_list = None

    if only_atmospheric_species_for_solar_metallicity:
        keep_list = list(elemental_abundances.keys())

    solar_h_ratios = _compute_h_ratios(
        get_solar_elemental_abundances(keep_list=keep_list)
    )
    h_ratios = _compute_h_ratios(elemental_abundances)

    solar_z_ratio = _compute_z_ratios(solar_h_ratios, sort=True, neglect_he=neglect_he)
    z_ratio = _compute_z_ratios(h_ratios, sort=True, neglect_he=neglect_he)

    metallicity = z_ratio / solar_z_ratio

    elements = [list(elements.keys())[i] for i in sorted_elements_indices]

    h_ratios = {
        list(h_ratios.keys())[i]: {
            "description": f"{elements[i]}/H",
            "local": h_ratio,
            "relative to solar": h_ratio / solar_h_ratios[list(h_ratios.keys())[i]],
        }
        for i, h_ratio in enumerate(h_ratios.values())
        if i != 0
    }

    return metallicity, h_ratios
