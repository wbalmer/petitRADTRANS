import sys
import warnings
import copy

import jax.numpy as jnp
from jax import lax, jit, typing
from jax import debug


try:
    from petitRADTRANS.chemistry.prt_easychem import get_exoatmos_abundances
except ModuleNotFoundError:
    get_exoatmos_abundances = None
from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.chemistry.utils import (
    compute_mean_molar_masses,
    fixed_length_adaptive_mesh_refinement,
    linear_spline_profile,
    cubic_spline_profile,
    stepped_profile,
    define_pressure_node_list,
    define_abundance_node_list,
    simplify_species_list,
    fill_atmosphere
)
from petitRADTRANS.chemistry import clouds as fc
from petitRADTRANS.opacities.opacities import Opacity


@jit(static_argnames=['cloud_species', 'adaptive_mesh_refinement'])
def calculate_cloud_mass_fractions_jit(pressures, temperatures, cloud_species,
                                       parameters, mass_fractions, mean_molar_mass,
                                       adaptive_mesh_refinement=False):
    clouds = {}
    p_bases = {}
    parameter_key_list = tuple(parameters.keys())  # Make it static for JIT

    simple_cloud_names = []
    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]
        simple_cloud_names.append(cloud_name)

    def get_cloud_abundance(cloud, parameters):
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        def easychem_case():
            if cloud_name in mass_fractions.keys():
                return mass_fractions[cloud_name]
            else:
                return jnp.zeros_like(temperatures)

        def eq_scaling_case():
            if 'eq_scaling_' + cloud_name in parameter_key_list:
                x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
                return 10 ** parameters['eq_scaling_' + cloud_name].value * x_cloud
            else:
                return jnp.zeros_like(temperatures)

        def free_case():
            if 'log_X_cb_' + cloud_name in parameter_key_list:
                return (10 ** parameters['log_X_cb_' + cloud_name].value) * jnp.ones_like(temperatures)
            else:
                return jnp.zeros_like(temperatures)

        return lax.cond(
            'use_easychem' in parameter_key_list,
            easychem_case,
            lambda: lax.cond(
                "eq_scaling_" + cloud_name in parameter_key_list,
                eq_scaling_case,
                free_case
            )
        )

    def get_cloud_base(cloud, parameters, pressures, temperatures, mean_molar_mass):
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        def log_pbase_case():
            if 'log_Pbase_' + cloud_name in parameter_key_list:
                return 10 ** parameters['log_Pbase_' + cloud_name].value
            else:
                return jnp.inf

        def pbase_case():
            if 'Pbase_' + cloud_name in parameter_key_list:
                return parameters['Pbase_' + cloud_name].value
            else:
                return jnp.inf

        def eq_case():
            if 'Fe/H' in parameter_key_list and 'C/O' in parameter_key_list:
                return fc.simple_cdf(
                    name=cloud_name,
                    press=pressures,
                    temp=temperatures,
                    metallicity=parameters['Fe/H'].value,
                    co_ratio=jnp.minimum(parameters['C/O'].value, 0.0001),
                    mmw=jnp.mean(mean_molar_mass)
                )
            else:
                return jnp.inf

        def free_case():
            if 'log_X_cb_' + cloud_name in parameter_key_list:
                return fc.simple_cdf_free(
                    cloud_name,
                    pressures,
                    temperatures,
                    10 ** parameters['log_X_cb_' + cloud_name].value,
                    mean_molar_mass[0]
                )
            else:
                return jnp.inf

        return lax.cond(
            'log_Pbase_' + cloud_name in parameter_key_list,
            log_pbase_case,
            lambda: lax.cond(
                'Pbase_' + cloud_name in parameter_key_list,
                pbase_case,
                lambda: lax.cond(
                    'Fe/H' in parameter_key_list,
                    eq_case,
                    free_case,
                ),
            )
        )

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]
        clouds[cloud_name] = get_cloud_abundance(cloud, parameters)
        p_bases[cloud_name] = get_cloud_base(cloud, parameters, pressures, temperatures, mean_molar_mass)

    def amr_case():
        #press_use, pressure_indices = fixed_length_adaptive_mesh_refinement(
        #    jnp.array(list(p_bases.values())),
        #    pressures)
        #    # jnp.array(parameters['pressure_width'].value))
        ##    # jnp.array(parameters['pressure_scaling'].value),
        # TODO: AMR pressure array needs to be input to function
        return jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    def no_amr_case():
        return jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    pressure_indices = lax.cond(
        adaptive_mesh_refinement,
        amr_case,
        no_amr_case
    )

    fseds = {}

    def get_fsed(cloud_name, parameters):
        def fsed_cloud_case(_):
            if 'fsed_' + cloud_name in parameter_key_list:
                return parameters['fsed_' + cloud_name].value
            else:
                return 0.0

        def fsed_default_case(_):
            if 'fsed' in parameter_key_list:
                return parameters['fsed'].value
            else:
                return 0.0

        return lax.cond(
            'fsed_' + cloud_name in parameter_key_list,
            fsed_cloud_case,
            fsed_default_case,
            operand=None
        )

    def fill_cloud_mass_fractions(cloud_name, pressures, p_bases, clouds, fseds, temperatures, pressure_indices):
        arr = jnp.zeros_like(temperatures)
        arr = jnp.where(
            pressures < p_bases[cloud_name],
            clouds[cloud_name] * (pressures / p_bases[cloud_name]) ** fseds[cloud_name],
            0.0
        )
        arr = arr[pressure_indices]
        return arr

    def skip_clouds_case(_):
        for cloud in cloud_species:
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]
            mass_fractions[cloud] = clouds[cloud_name]
        return mass_fractions

    def fill_clouds_case(_):
        for cloud in cloud_species:
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]

            fseds[cloud_name] = get_fsed(cloud_name, parameters)
            mass_fractions[cloud] = fill_cloud_mass_fractions(
                cloud_name, pressures, p_bases, clouds, fseds, temperatures, pressure_indices
            )
        return mass_fractions

    mass_fractions = lax.cond(
        'use_easychem' in parameter_key_list,
        skip_clouds_case,
        fill_clouds_case,
        operand=None
    )

    return mass_fractions, p_bases, pressure_indices


#@jit(static_argnames=['cloud_species', 'adaptive_mesh_refinement'])
def calculate_cloud_mass_fractions(pressures, temperatures, cloud_species, parameters, mass_fractions,
                             mean_molar_mass, adaptive_mesh_refinement=False):
    # Prior check all input params
    clouds = {}
    p_bases = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]
        if 'use_easychem' in parameters.keys():
            # adaptive_mesh_refinement CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cloud_name] = mass_fractions[cloud_name]
            continue

        if "eq_scaling_" + cloud_name in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cloud_name] = 10 ** parameters['eq_scaling_' + cloud_name].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cloud_name] = 10 ** parameters['log_X_cb_' + cloud_name].value

        # Free cloud bases
        if 'log_Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = 10 ** parameters['log_Pbase_' + cloud_name].value
        elif 'Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = parameters['Pbase_' + cloud_name].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cloud_name] = fc.simple_cdf(
                cloud_name,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                jnp.mean(mean_molar_mass)
            )
        else:
            p_bases[cloud_name] = fc.simple_cdf_free(
                cloud_name,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cloud_name].value,
                mean_molar_mass[0]
            )

    # Find high resolution pressure grid and indices
    if adaptive_mesh_refinement:
        press_use, pressure_indices = fixed_length_adaptive_mesh_refinement(
            jnp.array(list(p_bases.values())),
            pressures,
            parameters['pressure_scaling'].value,
            parameters['pressure_width'].value)
    else:
        pressure_indices = jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    # Decrease abundance with altitude to the power of -fsed
    fseds = {}
    if 'use_easychem' not in parameters.keys():
        cloud_name = None

        for cloud in copy.copy(cloud_species):
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]

            # Set up fseds per-cloud
            if 'fsed_' + cloud_name in parameters.keys():
                fseds[cloud_name] = parameters['fsed_' + cloud_name].value
            else:
                fseds[cloud_name] = parameters['fsed'].value

            mass_fractions[cloud_name] = jnp.zeros_like(temperatures)
            mass_fractions[cloud_name] = jnp.where(
                pressures < p_bases[cloud_name],
                clouds[cloud_name] * (pressures / p_bases[cloud_name]) ** fseds[cloud_name],
                0.0)
            mass_fractions[cloud_name] = mass_fractions[cloud_name][pressure_indices]
    return mass_fractions, p_bases, pressure_indices


def calculate_line_species_mass_fractions(pressures, temperatures=None, parameters=None, line_species=None,
                                          cloud_species=None, filling_species=None, verbose=False, **kwargs):
    """Initialize a model mass mixing ratios.
    Ensure that in any case, the sum of mass mixing ratios is equal to 1. Imposed mass mixing ratios are kept to
    their imposed value as long as the sum of the imposed values is lower or equal to 1. H2 and He are used as
    filling gases.
    The different possible cases are dealt with as follows:
        - Sum of imposed mass mixing ratios > 1: the mass mixing ratios are scaled down, conserving the ratio
        between them. Non-imposed mass mixing ratios are set to 0.
        - Sum of imposed mass mixing ratio of all imposed species < 1: if equilibrium chemistry is used or if H2 and
        He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
        Otherwise, the heh2_ratio parameter is used.
        - Sum of imposed and non-imposed mass mixing ratios > 1: the non-imposed mass mixing ratios are scaled down,
        conserving the ratios between them. Imposed mass mixing ratios are unchanged.
        - Sum of imposed and non-imposed mass mixing ratios < 1: if equilibrium chemistry is used or if H2 and
        He are imposed species, the atmosphere will be filled with H2 and He respecting the imposed H2/He ratio.
        Otherwise, the heh2_ratio parameter is used.

    When using equilibrium chemistry with imposed mass mixing ratios, imposed mass mixing ratios are set to their
    imposed value regardless of chemical equilibrium consistency.

    Args:
        pressures: (cgs) pressures of the mass mixing ratios
        line_species: list of line species, required to manage naming differences between opacities and chemistry
        cloud_species: list of cloud species, required to not count cloud species in the gas mass fractions.
        temperatures: (K) temperatures of the mass mixing ratios, used with equilibrium chemistry
        co_ratio: carbon over oxygen ratios of the model, used with equilibrium chemistry
        metallicity: ratio between heavy elements and H2 + He compared to solar, used with equilibrium chemistry
        carbon_pressure_quench: (bar) pressure where the carbon species are quenched, used with equilibrium
            chemistry
        imposed_mass_fractions: imposed mass mixing ratios
        use_equilibrium_chemistry: if True, use pRT equilibrium chemistry module
        filling_species:
            Dictionary with the filling species as keys and the weights of the mass fractions as values. Unweighted
            filling species are represented with None.
        verbose: if True, print additional information

    Returns:
        A dictionary containing the mass mixing ratios.
    """
    _ = kwargs  # kwargs is not used in this function, this is intended
    # Initialization
    if line_species is None:
        line_species = []

    # TODO: can't define as Nones,should always define filling species
    if hasattr(filling_species, '__iter__'):
        # Facilitate the building of H2 and He-only filling species dict
        if len(filling_species) == 2 and not isinstance(filling_species, dict):
            if 'H2' in filling_species and 'He' in filling_species:
                if 'Fe/H' in parameters.keys():
                    filling_species = {
                        'H2': None,
                        'He': None
                    }
                else:
                    filling_species = {
                        'H2': 74 * jnp.ones(pressures.size),  # % of atmosphere by mass for H2/He ratio
                        'He': 24 * jnp.ones(pressures.size)
                    }

    # Handle string and non-dict case
    # TODO: no longer accept strings.
    # TODO: restructure to depend on shape of  filling_species for JIT
    if not isinstance(filling_species, dict):
        #if isinstance(filling_species, str):
        #    filling_species = {
        #        filling_species: jnp.ones(pressures.size)
        #    }
        if hasattr(filling_species, '__iter__'):
            filling_species = {species: jnp.ones(pressures.size) for species in filling_species}
        elif filling_species is None:
            filling_species = {}

        if not isinstance(filling_species, dict):
            raise ValueError(f"filling_species must be a dictionary, but is {filling_species}")

    mass_fractions = {}
    simple_line_species = simplify_species_list(line_species)
    m_sum = jnp.zeros_like(pressures)

    imposed_mass_fractions = []

    # Free chemistry species
    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)

        # Vertically constant abundance
        if species_full_name in parameters.keys():
            imposed_mass_fractions.append(species_full_name)
            abund = 10 ** parameters[species].value
            mass_fractions[species] = abund * jnp.ones_like(pressures)
            m_sum += mass_fractions[species]
            continue
        elif species_basename in parameters.keys():
            imposed_mass_fractions.append(species_basename)
            abund = 10 ** parameters[species_basename].value
            mass_fractions[species_basename] = abund * jnp.ones_like(pressures)
            m_sum += mass_fractions[species_basename]
            continue

        # Non-vertically constant abundances
        pressure_interpolation_nodes = define_pressure_node_list(
            pressures,
            species,
            parameters
        )
        abundance_nodes = define_abundance_node_list(species, parameters)
        # Stepped abundance profile
        if f"{species}_stepped_abundance_profile" in parameters.keys():
            imposed_mass_fractions.append(species)
            mass_fractions[species] = stepped_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes)
            m_sum += mass_fractions[species]

        # Linear spline interpolation
        if f"{species}_linear_abundance_profile" in parameters.keys():
            imposed_mass_fractions.append(species)
            mass_fractions[species], _ = linear_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            m_sum += mass_fractions[species]

        # Cubic spline interpolation
        if f"{species}_cubic_abundance_profile" in parameters.keys():
            imposed_mass_fractions.append(species)
            mass_fractions[species], _ = cubic_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            m_sum += mass_fractions[species]

    # Remove imposed species from filling species
    for species in imposed_mass_fractions:
        if species in filling_species:
            warnings.warn(f"filling species '{species}' has also imposed mass fractions, "
                          f"the imposed mass fraction will be used")
            del filling_species[species]

    #  Ensure that the sum of imposed mass fractions is <= 1
    for i in range(jnp.size(m_sum)):
        if m_sum[i] > 1:
            if verbose:
                warnings.warn(f"sum of mass mixing ratios of imposed species ({m_sum}) is > 1, "
                              f"correcting...")

            for species in imposed_mass_fractions:
                if species not in cloud_species:  # Do not count clouds
                    mass_fractions[species][i] /= m_sum[i]

    # Initialize chemical equilibrium mass fractions
    mass_fractions_chemical_table = None

    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent w poor mans
        mass_fractions_chemical_table = get_exoatmos_abundances(
            pressures,
            temperatures,
            parameters
            )
        mean_molar_mass = mass_fractions_chemical_table['MMW']

    elif 'Fe/H' in parameters.keys():
        missing = {}

        if 'C/O' not in parameters.keys():
            missing['co_ratio'] = 0.55

        if len(missing) > 0:
            names = ', '.join(list(missing.keys()))
            values = list(missing.values())

            raise ValueError(f"equilibrium chemistry function missing {names} values (were {values})\n"
                             f"Add 'metallicity' as a model parameter, check your metallicity model function",
                             f"or disable equilibrium chemistry with 'use_equilibrium_chemistry=False'")

        # Interpolated Equilibrium chemistry
        # Make the abundance profile
        pquench_c = None

        if 'log_pquench' in parameters.keys():
            pquench_c = 10 ** parameters['log_pquench'].value

        # Interpolate chemical equilibrium
        mass_fractions_chemical_table, mean_molar_mass, _ = \
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                parameters['C/O'].value * jnp.ones_like(pressures),
                parameters['Fe/H'].value * jnp.ones_like(pressures),
                temperatures,
                pressures,
                carbon_pressure_quench=pquench_c,
                full=True
            )

    # Add non-imposed mass fractions to mass fractions
    m_sum_chemical_table_species = jnp.zeros(pressures.shape)

    if mass_fractions_chemical_table is not None:
        # Convert chemical table species names to line species names
        line_species_simple = simplify_species_list(line_species)
        for i, simple_species in enumerate(line_species_simple):
            if simple_species in mass_fractions_chemical_table:
                mass_fractions[line_species[i]] = mass_fractions_chemical_table[simple_species]
                m_sum += mass_fractions[line_species[i]]

    # Ensure that all line species are in mass_fractions
    for species in line_species:
        if species not in mass_fractions:
            warnings.warn(
                f"line species '{species}' initialised to {sys.float_info.min} ; "
                f"to remove this warning set use_equilibrium_chemistry to True "
                f"or add '{species}' and the desired mass mixing ratio to imposed_mass_fractions"
            )
            mass_fractions[species] = sys.float_info.min * jnp.ones(pressures.size)

    # Ensure that filling species are initialized as an array
    for species in filling_species:
        if species not in mass_fractions:
            mass_fractions[species] = jnp.zeros(pressures.size)

    # Ensure that the sum of mass mixing ratios of all species is = 1
    #m_sum_total = m_sum_chemical_table_species + m_sum_imposed_species

    # TODO: error checking on presence of filling_species
    mass_fractions = fill_atmosphere(
            mass_fractions=mass_fractions,
            filling_species=filling_species,
            mass_fraction_sum=m_sum,
        )

    for i, m_sum in enumerate(m_sum):
        if m_sum > 1:  # scale down the mass fractions, no filling species needed
            if verbose:
                warnings.warn(
                    f"sum of species mass fraction "
                    f"({m_sum_chemical_table_species[i]} + {m_sum[i]}) "
                    f"is > 1, correcting..."
                )

            for species in mass_fractions.keys():
                if species not in imposed_mass_fractions:
                    if m_sum_chemical_table_species[i] > 0:
                        mass_fractions[species] = mass_fractions[species].at[i].set(
                            mass_fractions[species][i]
                            * (1 - m_sum[i]) / m_sum_chemical_table_species[i]
                        )
                    else:
                        mass_fractions[species] = mass_fractions[species].at[i].set(
                            mass_fractions[species][i] / m_sum
                        )
        elif m_sum <= 0:
            # TODO: Check if ValueError is JIT compatible
            raise ValueError(
                f"invalid total mass fraction at pressure level {i}: "
                f"mass fraction must be > 0, but is {m_sum}\n"
                f"If the sum is 0, add at least one species with non-zero imposed mass fractions "
                f"or set equilibrium chemistry to True\n"
                f"If the sum is < 0, check your imposed mass fractions and chemical model")

    return mass_fractions, m_sum


#@partial(jit, static_argnames=['line_species', 'cloud_species', 'adaptive_mesh_refinement', 'filling_species'])
def get_abundances(pressures: typing.ArrayLike, temperatures: typing.ArrayLike,
                   parameters: dict, line_species: list, cloud_species: list,
                   filling_species: list = None, adaptive_mesh_refinement: bool = False,  **kwargs):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    Alternatively, a "free chemistry" approach can be used to set abundances.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <100% trace gas
    abundance by mass.

    Args:
        pressures : numpy.ndarray
            A log spaced pressure array. If adaptive_mesh_refinement is on it should be the full high resolution grid.
        temperatures : numpy.ndarray
            A temperature array with the same shape as pressures
        line_species : List(str)
            A list of gas species that will contribute to the line-by-line opacity of the pRT atmosphere.
        cloud_species : List(str)
            A list of condensate species that will contribute to the cloud opacity of the pRT atmosphere.
        parameters : dict
            A dictionary of model parameters, in particular it must contain the names C/O, Fe/H and
            log_pquench. Additionally, the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        adaptive_mesh_refinement : bool
            Turn the adaptive mesh grid on or off. See fixed_length_adaptive_mesh_refinement for implementation.

    Returns:
        abundances : dict
            Mass fraction abundances of all atmospheric species
        MMW : numpy.ndarray
            Array of the mean molecular weights in each pressure bin
        pressure_indices : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """
    if filling_species is None:
        filling_species = {
            'H2': 0.74 * jnp.ones(pressures.size),  # % of atmosphere by mass for H2/He ratio
            'He': 0.24 * jnp.ones(pressures.size)
        }

    line_species_no_resolution = tuple([species.split('.R')[0] for species in line_species])
    mass_fractions, m_sum_total = calculate_line_species_mass_fractions(
        pressures=pressures,
        temperatures=temperatures,
        parameters=parameters,
        line_species=line_species_no_resolution,
        cloud_species=cloud_species,
        filling_species=filling_species,
        **kwargs
    )
    mean_molar_mass = compute_mean_molar_masses(mass_fractions)
    mass_fractions, p_bases, pressure_indices = calculate_cloud_mass_fractions_jit(
        pressures,
        temperatures,
        tuple(cloud_species),
        parameters,
        mass_fractions,
        mean_molar_mass,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    # Adjust arrays for amr, and magic fix for FeH abundance.
    for species, value in mass_fractions.items():
        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = value / 2.
            if 'Fe(c)' in p_bases.keys() and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(c)']  # Must have iron cloud
                abunds_change_rainout = abunds_change_rainout.at[index_ro].set(0.)
            mass_fractions[species] = abunds_change_rainout[pressure_indices]
            continue
        mass_fractions[species] = value[pressure_indices]

    return mass_fractions, mean_molar_mass, pressure_indices, p_bases


def get_abundances_old(pressures, temperatures, line_species, cloud_species, parameters, adaptive_mesh_refinement=False):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    Alternatively, a "free chemistry" approach can be used to set abundances.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <100% trace gas
    abundance by mass.

    Args:
        pressures : numpy.ndarray
            A log spaced pressure array. If AMR is on it should be the full high resolution grid.
        temperatures : numpy.ndarray
            A temperature array with the same shape as pressures
        line_species : List(str)
            A list of gas species that will contribute to the line-by-line opacity of the pRT atmosphere.
        cloud_species : List(str)
            A list of condensate species that will contribute to the cloud opacity of the pRT atmosphere.
        parameters : dict
            A dictionary of model parameters, in particular it must contain the names C/O, Fe/H and
            log_pquench. Additionally, the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        amr : bool
            Turn the adaptive mesh grid on or off. See fixed_length_amr for implementation.

    Returns:
        abundances : dict
            Mass fraction abundances of all atmospheric species
        MMW : numpy.ndarray
            Array of the mean molecular weights in each pressure bin
        pressure_indices : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """
    # TODO replace by SpectralModel function
    # Free Chemistry
    abundances_interp = {}
    mmw = None
    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent with poor man's
        abundances_interp = get_exoatmos_abundances(pressures,
                                                    temperatures,
                                                    parameters)
        mmw = abundances_interp['MMW']
    elif "C/O" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor man's

        # Interpolated Equilibrium chemistry
        # Make the abundance profile
        pquench_c = None

        if 'log_pquench' in parameters.keys():
            pquench_c = 10 ** parameters['log_pquench'].value

        abundances_interp, mmw, _ = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                parameters['C/O'].value * jnp.ones_like(pressures),
                parameters['Fe/H'].value * jnp.ones_like(pressures),
                temperatures,
                pressures,
                carbon_pressure_quench=pquench_c,
                full=True
            )
        )
    # Free chemistry abundances
    msum = jnp.zeros_like(pressures)

    if abundances_interp:
        for key, val in abundances_interp.items():
            msum += val

    # Free chemistry species
    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)
        # Vertically constant abundance
        if species_full_name in abundances_interp.keys():
            msum -= abundances_interp[species_full_name]
        if species_basename in abundances_interp.keys():
            msum -= abundances_interp[species_basename]

        if species_full_name in parameters.keys():
            abund = 10 ** parameters[species_full_name].value
            abundances_interp[species_full_name] = abund * jnp.ones_like(pressures)
            msum += abundances_interp[species_full_name]
            continue
        elif species_basename in parameters.keys():
            abund = 10 ** parameters[species_basename].value
            abundances_interp[species_basename] = abund * jnp.ones_like(pressures)
            msum += abundances_interp[species_basename]
            continue

        # Non-vertically constant abundances
        pressure_interpolation_nodes = define_pressure_node_list(
            pressures,
            species_basename,
            parameters
        )
        abundance_nodes = define_abundance_node_list(species_basename, parameters)

        # Stepped abundance profile
        if f"{species_basename}_stepped_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename] = stepped_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes)
            msum += abundances_interp[species_basename]

        # Linear spline interpolation
        if f"{species_basename}_linear_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename], _ = linear_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[species_basename]

        # Cubic spline interpolation
        if f"{species_basename}_cubic_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename], _ = cubic_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[species_basename]

    # For free chemistry, need to fill with background gas (H2-He)
    # TODO use arbitrary background gas
    if "Fe/H" not in parameters.keys():
        # Check to make sure we're using free chemistry
        # Whatever's left is H2 and He
        if 'H2' in parameters.keys():
            abundances_interp['H2'] = 10 ** parameters['H2'].value * jnp.ones_like(pressures)
        else:
            abundances_interp['H2'] = 0.766 * (1.0 - msum)

        if 'He' in parameters.keys():
            abundances_interp['He'] = 10 ** parameters['He'].value * jnp.ones_like(pressures)
        else:
            abundances_interp['He'] = 0.234 * (1.0 - msum)

        # Imposing strict limit on msum to ensure H2 dominated composition
        #if jnp.max(msum) > 1.0:
        #    print(f"Abundance sum > 1.0, msum={jnp.max(msum):.2f}")
        #    return None, None, None, None

        mmw = compute_mean_molar_masses(abundances_interp)

    # Prior check all input params
    clouds = {}
    p_bases = {}
    abundances = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split('_')[0]

        if 'use_easychem' in parameters.keys():
            # AMR CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cloud_name] = abundances_interp[cloud_name]
            continue

        if "eq_scaling_" + cloud_name in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cloud_name] = 10 ** parameters['eq_scaling_' + cloud_name].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cloud_name] = 10 ** parameters['log_X_cb_' + cloud_name].value

        # Free cloud bases
        if 'log_Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = 10 ** parameters['log_Pbase_' + cloud_name].value
        elif 'Pbase_' + cloud_name in parameters.keys():
            p_bases[cloud_name] = parameters['Pbase_' + cloud_name].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cloud_name] = fc.simple_cdf(
                cloud_name,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                jnp.mean(mmw)
            )
        else:
            p_bases[cloud_name] = fc.simple_cdf_free(
                cloud_name,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cloud_name].value,
                mmw[0]
            )

    # Find high resolution pressure grid and indices
    if adaptive_mesh_refinement:
        press_use, pressure_indices = fixed_length_adaptive_mesh_refinement(
            jnp.array(list(p_bases.values())),
            pressures,
            parameters['pressure_scaling'].value,
            parameters['pressure_width'].value)
    else:
        pressure_indices = jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    fseds = {}

    if 'use_easychem' not in parameters.keys():
        for cloud in copy.copy(cloud_species):
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]

            # Set up fseds per-cloud
            if 'fsed_' + cloud_name in parameters.keys():
                fseds[cloud_name] = parameters['fsed_' + cloud_name].value
            else:
                fseds[cloud_name] = parameters['fsed'].value

            abundances[cloud_name] = jnp.zeros_like(temperatures)
            abundances[cloud_name] = jnp.where(
                pressures < p_bases[cloud_name],
                clouds[cloud_name] * (pressures / p_bases[cloud_name]) ** fseds[cloud_name],
                0.0)
            abundances[cloud_name] = abundances[cloud_name][pressure_indices]

    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)

        # Depending on easychem vs interpolated and different versions of pRT
        # C2H2 is named differently.
        if species_basename == "C2H2":
            # might be',acetylene'
            not_found = True
            for key in abundances_interp.keys():
                if species_basename in key:
                    species_basename = key
                    not_found = False
                    break

            if not_found:
                continue

        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = abundances_interp[species] / 2.
            if 'Fe(c)' in p_bases.keys() and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(c)']  # Must have iron cloud
                abunds_change_rainout = abunds_change_rainout.at[index_ro].set(0.)
            abundances[species] = abunds_change_rainout[pressure_indices]
        abundances[species] = abundances_interp[species][pressure_indices]
    abundances['H2'] = abundances_interp['H2'][pressure_indices]
    abundances['He'] = abundances_interp['He'][pressure_indices]

    return abundances, mmw, pressure_indices, p_bases

