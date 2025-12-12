import sys
import warnings
import copy

import jax.numpy as jnp
from jax import lax, jit, typing
from functools import partial


try:
    from petitRADTRANS.chemistry.prt_easychem import get_exoatmos_abundances
except ModuleNotFoundError:
    get_exoatmos_abundances = None

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

@partial(jit, static_argnames=['cloud_species', 'adaptive_mesh_refinement'])
def calculate_cloud_mass_fractions_jit(pressures, temperatures, cloud_species,
                                       parameters, mass_fractions, mean_molar_mass,
                                       adaptive_mesh_refinement=False):
    clouds = {}
    p_bases = {}
    parameter_key_list = tuple(parameters.keys())  # Make it static for JIT

    def get_cloud_abundance(cloud, parameters):
        cname = cloud.split("_")[0]

        def easychem_case(_):
            return mass_fractions[cname]

        def eq_scaling_case(_):
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            return 10 ** parameters['eq_scaling_' + cname].value * x_cloud

        def free_case(_):
            return 10 ** parameters['log_X_cb_' + cname].value

        return lax.cond(
            'use_easychem' in parameter_key_list,
            easychem_case,
            lambda _: lax.cond(
                "eq_scaling_" + cname in parameter_key_list,
                eq_scaling_case,
                free_case,
                operand=None
            ),
            operand=None
        )

    def get_cloud_base(cloud, parameters, pressures, temperatures, mean_molar_mass):
        cname = cloud.split("_")[0]

        def log_pbase_case(_):
            return 10 ** parameters['log_Pbase_' + cname].value

        def pbase_case(_):
            return parameters['Pbase_' + cname].value

        def eq_case(_):
            return fc.simple_cdf(
                cname,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                jnp.mean(mean_molar_mass)
            )

        def free_case(_):
            return fc.simple_cdf_free(
                cname,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cname].value,
                mean_molar_mass[0]
            )

        return lax.cond(
            'log_Pbase_' + cname in parameter_key_list,
            log_pbase_case,
            lambda _: lax.cond(
                'Pbase_' + cname in parameter_key_list,
                pbase_case,
                lambda _: lax.cond(
                    'Fe/H' in parameter_key_list,
                    eq_case,
                    free_case,
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )

    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        clouds[cname] = get_cloud_abundance(cloud, parameters)
        p_bases[cname] = get_cloud_base(cloud, parameters, pressures, temperatures, mean_molar_mass)

    def amr_case(_):
        #press_use, small_index = fixed_length_adaptive_mesh_refinement(
        #    jnp.array(list(p_bases.values())),
        #    pressures,
        #    parameters['pressure_scaling'].value,
        #    parameters['pressure_width'].value)
        return jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int) #small_index

    def no_amr_case(_):
        return jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    small_index = lax.cond(
        adaptive_mesh_refinement,
        amr_case,
        no_amr_case,
        operand=None
    )

    fseds = {}

    def get_fsed(cname, parameters):
        def fsed_cloud_case(_):
            return parameters['fsed_' + cname].value

        def fsed_default_case(_):
            return parameters['fsed'].value
        return lax.cond(
            'fsed_' + cname in parameter_key_list,
            fsed_cloud_case,
            fsed_default_case,
            operand=None
        )

    def fill_cloud_mass_fractions(cname, pressures, p_bases, clouds, fseds, temperatures, small_index):
        arr = jnp.zeros_like(temperatures)
        arr = jnp.where(
            pressures < p_bases[cname],
            clouds[cname] * (pressures / p_bases[cname]) ** fseds[cname],
            0.0
        )
        arr = arr[small_index]
        return arr

    def skip_clouds_case(_):
        return mass_fractions

    def fill_clouds_case(_):
        for cloud in cloud_species:
            cname = cloud.split('_')[0]
            fseds[cname] = get_fsed(cname, parameters)
            mass_fractions[cloud] = fill_cloud_mass_fractions(
                cname, pressures, p_bases, clouds, fseds, temperatures, small_index
            )
        return mass_fractions

    mass_fractions = lax.cond(
        'use_easychem' in parameter_key_list,
        skip_clouds_case,
        fill_clouds_case,
        operand=None
    )

    return mass_fractions, p_bases, small_index


def calculate_cloud_mass_fractions(pressures, temperatures, cloud_species, parameters, mass_fractions,
                             mean_molar_mass, adaptive_mesh_refinement=False):
    # Prior check all input params
    clouds = {}
    p_bases = {}

    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if 'use_easychem' in parameters.keys():
            # adaptive_mesh_refinement CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cname] = mass_fractions[cname]
            continue

        if "eq_scaling_" + cname in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10 ** parameters['eq_scaling_' + cname].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cname] = 10 ** parameters['log_X_cb_' + cname].value

        # Free cloud bases
        if 'log_Pbase_' + cname in parameters.keys():
            p_bases[cname] = 10 ** parameters['log_Pbase_' + cname].value
        elif 'Pbase_' + cname in parameters.keys():
            p_bases[cname] = parameters['Pbase_' + cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cname] = fc.simple_cdf(
                cname,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                jnp.mean(mean_molar_mass)
            )
        else:
            p_bases[cname] = fc.simple_cdf_free(
                cname,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cname].value,
                mean_molar_mass[0]
            )

    # Find high resolution pressure grid and indices
    if adaptive_mesh_refinement:
        press_use, small_index = fixed_length_adaptive_mesh_refinement(
            jnp.array(list(p_bases.values())),
            pressures,
            parameters['pressure_scaling'].value,
            parameters['pressure_width'].value)
    else:
        small_index = jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    # Decrease abundance with altitude to the power of -fsed
    fseds = {}
    if 'use_easychem' not in parameters.keys():
        cname = None

        for cloud in copy.copy(cloud_species):
            cname = cloud.split('_')[0]
            # Set up fseds per-cloud
            if 'fsed_' + cname in parameters.keys():
                fseds[cname] = parameters['fsed_' + cname].value
            else:
                fseds[cname] = parameters['fsed'].value

            mass_fractions[cloud] = jnp.zeros_like(temperatures)
            mass_fractions[cloud] = jnp.where(pressures < p_bases[cname],
                                          clouds[cname] * (pressures / p_bases[cname]) ** fseds[cname],
                                          0.0)
            mass_fractions[cloud] = mass_fractions[cloud][small_index]
    return mass_fractions, p_bases, small_index


@partial(jit, static_argnames=['line_species', 'cloud_species', 'verbose'])
def calculate_line_species_mass_fractions_jit(pressures, temperatures=None, parameters=None, line_species=None,
                                    cloud_species=None, filling_species=None,
                                    verbose=False, **kwargs):
    _ = kwargs  # kwargs is not used in this function, this is intended

    parameter_key_tuple = tuple(parameters.keys())  # Make it static for JIT
    if line_species is None:
        line_species = ()

    def build_filling_species(filling_species, parameters, pressures):
        def h2he_case(_):
            if 'Fe/H' in parameter_key_tuple:
                return {'H2': None, 'He': None}
            else:
                return {
                    'H2': 74 * jnp.ones(pressures.size),
                    'He': 24 * jnp.ones(pressures.size)
                }

        def default_case(_):
            return filling_species

        return lax.cond(
            (hasattr(filling_species, '__iter__') and len(filling_species) == 2 and not isinstance(filling_species, dict)
             and 'H2' in filling_species and 'He' in filling_species),
            h2he_case,
            default_case,
            operand=None
        )

    filling_species = build_filling_species(filling_species, parameters, pressures)

    # Handle string and non-dict case
    def handle_non_dict(fs):
        def str_case(_):
            return {fs: jnp.ones(pressures.size)}
        def iter_case(_):
            return {species: jnp.ones(pressures.size) for species in fs}
        def none_case(_):
            return {}
        def error_case(_):
            raise ValueError(f"filling_species must be a dictionary, but is {fs}")

        return lax.cond(
            isinstance(fs, str),
            str_case,
            lambda _: lax.cond(
                hasattr(fs, '__iter__'),
                iter_case,
                lambda _: lax.cond(
                    fs is None,
                    none_case,
                    error_case,
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )

    if not isinstance(filling_species, dict):
        filling_species = handle_non_dict(filling_species)

    mass_fractions = {}
    simple_line_species = simplify_species_list(line_species)
    m_sum_imposed_species = 0.0 * jnp.ones_like(pressures)
    imposed_mass_fractions = []

    # Free chemistry species
    for species in line_species:
        imposed_mass_fractions.append(species)
        if species in parameter_key_tuple:
            abund = 10 ** parameters[species].value
            mass_fractions[species] = abund * jnp.ones_like(pressures)
            m_sum_imposed_species += mass_fractions[species]
        """def not_vert_const_case(_):
            pressure_interpolation_nodes = define_pressure_node_list(
                pressures,
                species,
                parameters
            )
            abundance_nodes = define_abundance_node_list(species, parameters)
            def stepped_case(_):
                imposed_mass_fractions.append(species)
                mass_fractions[species] = stepped_profile(
                    pressures,
                    pressure_interpolation_nodes,
                    abundance_nodes)
                return m_sum_imposed_species + mass_fractions[species]
            def linear_case(_):
                imposed_mass_fractions.append(species)
                mf, _ = linear_spline_profile(
                    pressures,
                    pressure_interpolation_nodes,
                    abundance_nodes,
                    gamma=0.04,
                    nnodes=len(pressure_interpolation_nodes))
                mass_fractions[species] = mf
                return m_sum_imposed_species + mass_fractions[species]
            def cubic_case(_):
                imposed_mass_fractions.append(species)
                mf, _ = cubic_spline_profile(
                    pressures,
                    pressure_interpolation_nodes,
                    abundance_nodes,
                    gamma=0.04,
                    nnodes=len(pressure_interpolation_nodes))
                mass_fractions[species] = mf
                return m_sum_imposed_species + mass_fractions[species]
            def none_case(_):
                return m_sum_imposed_species

            m_sum_imposed_species_stepped = lax.cond(
                f"{species}_stepped_abundance_profile" in parameter_key_tuple,
                stepped_case,
                none_case,
                operand=None
            )
            m_sum_imposed_species_linear = lax.cond(
                f"{species}_linear_abundance_profile" in parameter_key_tuple,
                linear_case,
                none_case,
                operand=None
            )
            m_sum_imposed_species_cubic = lax.cond(
                f"{species}_cubic_abundance_profile" in parameter_key_tuple,
                cubic_case,
                none_case,
                operand=None
            )
            return m_sum_imposed_species_stepped + m_sum_imposed_species_linear + m_sum_imposed_species_cubic
        """
        #m_sum_imposed_species = vert_const_case()
        """lax.cond(
            species in parameter_key_tuple,
            vert_const_case,
            not_vert_const_case,
            operand=None
        )"""

    # Remove imposed species from filling species
    for species in imposed_mass_fractions:
        if species in filling_species:
            # warnings.warn(...)  # Not JIT compatible
            del filling_species[species]

    # Ensure that the sum of imposed mass fractions is <= 1
    imposed_mass_fractions = tuple(imposed_mass_fractions)  # Make it static for JIT
    cloud_species = tuple(cloud_species)  # Make it static for JIT

    @partial(jit, static_argnames=['imposed_mass_fractions', 'cloud_species'])
    def scale_down_if_needed(m_sum_imposed_species, mass_fractions, imposed_mass_fractions, cloud_species):
        """
        Scales down the mass fractions of imposed species if their sum exceeds 1.
        This function is designed to be JIT-compatible.
        """
        def scale_case(mf):
            # mf is the mass_fractions dictionary

            # Create updates for species that need scaling.
            # This is functional and JIT-compatible as loops over static strings unroll.
            updates = {}
            for species in imposed_mass_fractions:
                if species not in cloud_species:
                    updates[species] = mf[species] / m_sum_imposed_species
                else:
                    updates[species] = mf[species]
            # Return a new dictionary with the updated values
            return {**mf, **updates}

        def no_scale_case(mf):
            return mf

        # NOTE: The static lists `imposed_mass_fractions` and `cloud_species` are
        # captured from the outer scope. For this to be JIT-compatible, they must
        # be static (i.e. not change shape/content during execution).
        return lax.cond(
            jnp.any(m_sum_imposed_species > 1),
            scale_case,
            no_scale_case,
            mass_fractions
        )

    mass_fractions = scale_down_if_needed(m_sum_imposed_species, mass_fractions, imposed_mass_fractions, cloud_species)

    # Chemical equilibrium mass fractions
    mass_fractions_chemical_table = {}

    def use_full_equilibrium_case(_):
        # TODO: implement easychem call in JAX
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")
        mf = get_exoatmos_abundances(
            pressures,
            temperatures,
            parameters
        )
        mean_molar_mass = mf['MMW']
        return mf, mean_molar_mass

    @partial(jit, static_argnames=['key_list'])
    def precalculated_equilibrium_case(key_list):
        if 'Fe/H' not in key_list:
            metallicity = 0.0
        else:
            metallicity = parameters['Fe/H'].value

        if 'C/O' not in key_list:
            carbon_to_oxygen = 0.55
        else:
            carbon_to_oxygen = parameters['C/O'].value
        pquench_c = None
        if 'log_pquench' in key_list:
            pquench_c = 10 ** parameters['log_pquench'].value

        mf, mean_molar_mass, _ = \
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                carbon_to_oxygen * jnp.ones_like(pressures),
                metallicity * jnp.ones_like(pressures),
                temperatures,
                pressures,
                carbon_pressure_quench=pquench_c,
                full=True
            )
        return mf, mean_molar_mass

    @partial(jit, static_argnames=['key_list'])
    def no_chem_case(key_list):
        return {}, -jnp.inf

    if 'Fe/H' in parameter_key_tuple:
        mass_fractions_chemical_table, mean_molar_mass = precalculated_equilibrium_case(parameter_key_tuple)

    # Add non-imposed mass fractions to mass fractions
    m_sum_chemical_table_species = jnp.zeros(pressures.shape)

    if mass_fractions_chemical_table:
        line_species_simple = simplify_species_list(line_species)
        for i, simple_species in enumerate(line_species_simple):
            if simple_species in mass_fractions_chemical_table and simple_species != line_species[i]:
                mass_fractions_chemical_table[line_species[i]] = copy.deepcopy(
                    mass_fractions_chemical_table[simple_species]
                )
                del mass_fractions_chemical_table[simple_species]

        imposed_species_simple = simplify_species_list(imposed_mass_fractions)
        for simple_species in imposed_species_simple:
            if simple_species in mass_fractions_chemical_table:
                del mass_fractions_chemical_table[simple_species]

        for species in mass_fractions_chemical_table.keys():
            if species not in imposed_mass_fractions:
                if species in filling_species:
                    if filling_species[species] is not None:
                        continue
                mass_fractions[species] = mass_fractions_chemical_table[species]
                m_sum_chemical_table_species += mass_fractions_chemical_table[species]

    # Ensure that all line species are in mass_fractions
    for species in line_species:
        if species not in mass_fractions:
            mass_fractions[species] = sys.float_info.min * jnp.ones(pressures.size)

    # Ensure that filling species are initialized as an array
    for species in filling_species:
        if species not in mass_fractions:
            mass_fractions[species] = jnp.zeros(pressures.size)

    # Ensure that the sum of mass mixing ratios of all species is = 1
    m_sum_total = m_sum_chemical_table_species + m_sum_imposed_species

    mass_fractions = fill_atmosphere(
            mass_fractions=mass_fractions,
            filling_species=filling_species,
            mass_fraction_sum=m_sum_total,
        )
    for species, mf_array in mass_fractions.items():
        if species not in imposed_mass_fractions:
            mass_fractions[species] = lax.cond(
                jnp.sum(m_sum_chemical_table_species) > 0,
                lambda mf: jnp.where(
                    m_sum_total > 1, mf
                    * (1 - m_sum_imposed_species)
                    / m_sum_chemical_table_species,
                    mf
                ),
                lambda mf: mf / m_sum_total,
                mf_array
            )

    return mass_fractions


def calculate_line_species_mass_fractions(pressures, temperatures=None, parameters=None, line_species=None,
                                    cloud_species=None, filling_species=None,
                                    verbose=False, **kwargs):
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

    if hasattr(filling_species, '__iter__'):
        # Facilitate the building of H2 and He-only filling species dict
        if len(filling_species) == 2 and not isinstance(filling_species, dict):
            if 'H2' in filling_species and 'He' in filling_species:
                if 'Fe/H' in parameter_key_tuple:
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
    if not isinstance(filling_species, dict):
        if isinstance(filling_species, str):
            filling_species = {
                filling_species: jnp.ones(pressures.size)
            }
        elif hasattr(filling_species, '__iter__'):
            filling_species = {species: jnp.ones(pressures.size) for species in filling_species}
        elif filling_species is None:
            filling_species = {}

        if not isinstance(filling_species, dict):
            raise ValueError(f"filling_species must be a dictionary, but is {filling_species}")

    mass_fractions = {}
    simple_line_species = simplify_species_list(line_species)
    m_sum_imposed_species = 0.0 * jnp.ones_like(pressures)
    imposed_mass_fractions = []

    # Free chemistry species
    for species in line_species:
        # Vertically constant abundance
        if species in parameters.keys():
            imposed_mass_fractions.append(species)
            abund = 10 ** parameters[species].value
            mass_fractions[species] = abund * jnp.ones_like(pressures)
            m_sum_imposed_species += mass_fractions[species]
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
            m_sum_imposed_species += mass_fractions[species]

        # Linear spline interpolation
        if f"{species}_linear_abundance_profile" in parameters.keys():
            imposed_mass_fractions.append(species)
            mass_fractions[species], _ = linear_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            m_sum_imposed_species += mass_fractions[species]

        # Cubic spline interpolation
        if f"{species}_cubic_abundance_profile" in parameters.keys():
            imposed_mass_fractions.append(species)
            mass_fractions[species], _ = cubic_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            m_sum_imposed_species += mass_fractions[species]

    # Remove imposed species from filling species
    for species in imposed_mass_fractions:
        if species in filling_species:
            warnings.warn(f"filling species '{species}' has also imposed mass fractions, "
                          f"the imposed mass fraction will be used")
            del filling_species[species]

    #  Ensure that the sum of imposed mass fractions is <= 1
    for i in range(jnp.size(m_sum_imposed_species)):
        if m_sum_imposed_species[i] > 1:
            if verbose:
                warnings.warn(f"sum of mass mixing ratios of imposed species ({m_sum_imposed_species}) is > 1, "
                              f"correcting...")

            for species in imposed_mass_fractions:
                if species not in cloud_species:  # Do not count clouds
                    mass_fractions[species][i] /= m_sum_imposed_species[i]

    # Initialize chemical equilibrium mass fractions
    mass_fractions_chemical_table = None

    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent w poor mans
        mass_fractions = get_exoatmos_abundances(
            pressures,
            temperatures,
            parameters
            )
        mean_molar_mass = mass_fractions['MMW']

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
            if simple_species in mass_fractions_chemical_table and simple_species != line_species[i]:
                mass_fractions_chemical_table[line_species[i]] = copy.deepcopy(
                    mass_fractions_chemical_table[simple_species]
                )
                del mass_fractions_chemical_table[simple_species]

        # Remove imposed mass fractions names from chemical table species names
        imposed_species_simple = simplify_species_list(imposed_mass_fractions)
        for simple_species in imposed_species_simple:
            if simple_species in mass_fractions_chemical_table:
                del mass_fractions_chemical_table[simple_species]

        # Get the sum of mass fractions of non-imposed species
        for species in mass_fractions_chemical_table.keys():
            if species not in imposed_mass_fractions:
                if species in filling_species:
                    if filling_species[species] is not None:
                        continue

                mass_fractions[species] = mass_fractions_chemical_table[species]
                m_sum_chemical_table_species += mass_fractions_chemical_table[species]

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
    m_sum_total = m_sum_chemical_table_species + m_sum_imposed_species

    for i, m_sum in enumerate(m_sum_total):
        if 0 < m_sum < 1:
            if len(filling_species) > 0:  # fill the atmosphere using the filling species
                mass_fractions_i = fill_atmosphere(
                    mass_fractions=mass_fractions,
                    filling_species=filling_species,
                    fill_layer=i
                )

                for species in mass_fractions:

                    mass_fractions[species] = mass_fractions[species].at[i].set(
                        mass_fractions_i[species][0]
                    )
            else:
                warnings.warn(f"the sum of mass mixing ratios at level {i} is lower than 1 ({m_sum_total[i]}). "
                                f"Set filling_species to automatically fill the atmosphere "
                                f"or manually adjust the imposed mass fractions")
        elif m_sum > 1:  # scale down the mass fractions, no filling species needed
            if verbose:
                warnings.warn(
                    f"sum of species mass fraction "
                    f"({m_sum_chemical_table_species[i]} + {m_sum_imposed_species[i]}) "
                    f"is > 1, correcting..."
                )

            for species in mass_fractions.keys():
                if species not in imposed_mass_fractions:
                    if m_sum_chemical_table_species[i] > 0:
                        mass_fractions[species] = mass_fractions[species].at[i].set(
                            mass_fractions[species][i]
                            * (1 - m_sum_imposed_species[i]) / m_sum_chemical_table_species[i]
                        )
                    else:
                        mass_fractions[species] = mass_fractions[species].at[i].set(
                            mass_fractions[species][i] / m_sum
                        )
        elif m_sum <= 0:
            raise ValueError(
                f"invalid total mass fraction at pressure level {i}: "
                f"mass fraction must be > 0, but is {m_sum}\n"
                f"If the sum is 0, add at least one species with non-zero imposed mass fractions "
                f"or set equilibrium chemistry to True\n"
                f"If the sum is < 0, check your imposed mass fractions and chemical model")
    return mass_fractions, m_sum_total

@partial(jit, static_argnames=['line_species', 'cloud_species', 'adaptive_mesh_refinement'])
def get_abundances(pressures: typing.ArrayLike, temperatures:typing.ArrayLike, parameters: dict, line_species: list, cloud_species: list,
                   filling_species:list = None, adaptive_mesh_refinement: bool = False,  **kwargs):
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
        small_index : numpy.ndarray
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
    mass_fractions = calculate_line_species_mass_fractions_jit(
        pressures=pressures,
        temperatures=temperatures,
        parameters=parameters,
        line_species=line_species_no_resolution,
        cloud_species=cloud_species,
        filling_species=filling_species,
        **kwargs
    )
    mean_molar_mass = compute_mean_molar_masses(mass_fractions)
    mass_fractions, p_bases, small_index = calculate_cloud_mass_fractions_jit(
        pressures,
        temperatures,
        cloud_species,
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
            mass_fractions[species] = abunds_change_rainout[small_index]
        mass_fractions[species] = value[small_index]

    return mass_fractions, mean_molar_mass, small_index, p_bases


def get_abundances_old(pressures, temperatures, line_species, cloud_species, parameters, adaptive_mesh_refinement=False):
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
        small_index : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """
    # Free Chemistry
    abundances_interp = {}
    mmw = None

    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent w poor mans
        abundances_interp = get_exoatmos_abundances(pressures,
                                                    temperatures,
                                                    parameters)
        mmw = abundances_interp['MMW']
    elif "Fe/H" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor mans

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
    for species in simplify_species_list(line_species):
        #name, natural_abundance, charge, cloud_info, _, spectral_info = Opacity.split_species_all_info(
        #    species=species
        #)
        species_short_name = species
        easy_chem_name = species  #Opacity.get_species_scientific_name(species)

        # Vertically constant abundance
        if species_short_name in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abund = 10 ** parameters[species_short_name].value
            abundances_interp[easy_chem_name] = abund * jnp.ones_like(pressures)
            msum += abundances_interp[easy_chem_name]
            continue

        # Non-vertically constant abundances
        pressure_interpolation_nodes = define_pressure_node_list(
            pressures,
            species_short_name,
            parameters
        )
        abundance_nodes = define_abundance_node_list(species_short_name, parameters)
        # Stepped abundance profile
        if f"{species_short_name}_stepped_abundance_profile" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances_interp[easy_chem_name] = stepped_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes)
            msum += abundances_interp[easy_chem_name]

        # Linear spline interpolation
        if f"{species_short_name}_linear_abundance_profile" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances_interp[easy_chem_name], _ = linear_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[easy_chem_name]

        # Cubic spline interpolation
        if f"{species_short_name}_cubic_abundance_profile" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances_interp[easy_chem_name], _ = cubic_spline_profile(
                pressures,
                pressure_interpolation_nodes,
                abundance_nodes,
                gamma=0.04,
                nnodes=len(pressure_interpolation_nodes))
            msum += abundances_interp[easy_chem_name]

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
        if jnp.max(msum) > 1.0:
            print(f"Abundance sum > 1.0, msum={jnp.max(msum):.2f}")
            return None, None, None, None

        mmw = compute_mean_molar_masses(abundances_interp)

    # Prior check all input params
    clouds = {}
    p_bases = {}
    abundances = {}

    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if 'use_easychem' in parameters.keys():
            # adaptive_mesh_refinement CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cname] = abundances_interp[cname]
            continue

        if "eq_scaling_" + cname in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10 ** parameters['eq_scaling_' + cname].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cname] = 10 ** parameters['log_X_cb_' + cname].value

        # Free cloud bases
        if 'log_Pbase_' + cname in parameters.keys():
            p_bases[cname] = 10 ** parameters['log_Pbase_' + cname].value
        elif 'Pbase_' + cname in parameters.keys():
            p_bases[cname] = parameters['Pbase_' + cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cname] = fc.simple_cdf(
                cname,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                jnp.mean(mmw)
            )
        else:
            p_bases[cname] = fc.simple_cdf_free(
                cname,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cname].value,
                mmw[0]
            )

    # Find high resolution pressure grid and indices
    if adaptive_mesh_refinement:
        press_use, small_index = fixed_length_adaptive_mesh_refinement(jnp.array(list(p_bases.values())),
                                                  pressures,
                                                  parameters['pressure_scaling'].value,
                                                  parameters['pressure_width'].value)
    else:
        small_index = jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    fseds = {}

    if 'use_easychem' not in parameters.keys():
        cname = None

        for cloud in copy.copy(cloud_species):
            cname = cloud.split('_')[0]
            # Set up fseds per-cloud
            if 'fsed_' + cname in parameters.keys():
                fseds[cname] = parameters['fsed_' + cname].value
            else:
                fseds[cname] = parameters['fsed'].value

            abundances[cloud] = jnp.zeros_like(temperatures)
            abundances[cloud] = jnp.where(pressures < p_bases[cname],
                                          clouds[cname] * (pressures / p_bases[cname]) ** fseds[cname],
                                          0.0)
            abundances[cloud] = abundances[cloud][small_index]

    for species in line_species:
        easy_chem_name = species.split('_')[0].split('-')[0].split(".")[0]
        # Depending on easychem vs interpolated and different versions of pRT
        # C2H2 is named differently.
        if easy_chem_name == "C2H2":
            # might be',acetylene'
            not_found = True
            for key in abundances_interp.keys():
                if easy_chem_name in key:
                    easy_chem_name = key
                    not_found = False
                    break
            if not_found:
                continue
        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = abundances_interp[easy_chem_name] / 2.
            if 'Fe(c)' in p_bases.keys() and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(c)']  # Must have iron cloud
                abunds_change_rainout = abunds_change_rainout.at[index_ro].set(0.)
            abundances[species] = abunds_change_rainout[small_index]
        abundances[species] = abundances_interp[easy_chem_name][small_index]
    abundances['H2'] = abundances_interp['H2'][small_index]
    abundances['He'] = abundances_interp['He'][small_index]

    return abundances, mmw, small_index, p_bases
