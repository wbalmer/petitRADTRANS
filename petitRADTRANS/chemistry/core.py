import sys
import warnings
import copy

import jax.numpy as jnp
from jax import lax, jit, typing, config, debug

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
    stepped_profile,
    cubic_spline_profile,
    define_pressure_node_list,
    define_abundance_node_list,
    simplify_species_list,
    fill_atmosphere,
    mass_fractions2metallicity

)
from petitRADTRANS.chemistry import clouds as fc

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64

INVALID_VALUE = jnp.array(jnp.inf, JAX_DTYPE)

#@jit(static_argnames=['line_species', 'cloud_species', 'verbose'])
def calculate_gas_mass_fractions(pressures, temperatures=None, imposed_mass_fractions=None, line_species=None,
                               cloud_species=None, filling_species=None, use_equilibrium_chemistry=False,
                               metallicity=None, co_ratio=0.55, carbon_quench_pressure=None,
                               exo_atmos=None, atomic_mass_fractions=None, **kwargs):
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
        carbon_quench_pressure: (bar) pressure where the carbon species are quenched, used with equilibrium
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

    if line_species is None:
        line_species = ()

    if imposed_mass_fractions is None:
        imposed_mass_fractions = {}

    def build_filling_species(filling_species, pressures):
        pressure_template = jnp.asarray(pressures)

        def _zeros():
            return jnp.zeros_like(pressure_template)

        def _filled(value):
            return jnp.full_like(pressure_template, value)

        def h2he_case(filling_species):
            fill_species = {}
            if metallicity is not None:
                fill_species = {
                    'H2': _zeros(),
                    'He': _zeros(),
                }
            else:
                fill_species = {
                    'H2': _filled(74.0),
                    'He': _filled(24.0),
                }
            if isinstance(filling_species, dict):
                for species, value in filling_species.items():
                    if species not in fill_species:
                        fill_species[species] = _filled(value)
            elif isinstance(filling_species, tuple):
                for species in filling_species:
                    if species not in fill_species:
                        fill_species[species] = _zeros()
            return fill_species

        def default_case(filling_species):
            if isinstance(filling_species, dict):
                return {
                    key: _filled(value)
                    for key, value in filling_species.items()
                }

            return filling_species

        if (
            hasattr(filling_species, '__iter__')
            and len(filling_species) == 2
            and not isinstance(filling_species, dict)
            and 'H2' in filling_species
            and 'He' in filling_species
        ):
            return h2he_case(filling_species)

        return default_case(filling_species)

    filling_species = build_filling_species(filling_species, pressures)

    # Handle string and non-dict case
    def handle_non_dict(fs):
        def str_case(_):
            return {fs: jnp.ones_like(pressures)}
        def iter_case(_):
            return {species: jnp.ones_like(pressures) for species in fs}
        def none_case(_):
            return {}
        def error_case(_):
            raise ValueError(f"filling_species must be a dictionary, but is {fs}")

        if isinstance(fs, str):
            return str_case(None)
        if hasattr(fs, '__iter__'):
            return iter_case(None)
        if fs is None:
            return none_case(None)
        return error_case(None)

    if not isinstance(filling_species, dict):
        filling_species = handle_non_dict(filling_species)

    mass_fractions = {}
    simple_line_species = simplify_species_list(line_species)
    m_sum_imposed_species = 0.0 * jnp.ones_like(pressures)

    # Free chemistry species
    for species in imposed_mass_fractions:
        abund = 10 ** imposed_mass_fractions[species]
        mass_fractions[species] = abund * jnp.ones_like(pressures)
        m_sum_imposed_species += mass_fractions[species]

        # TODO: re-implement non-vertically constant abundances in JAX
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
    cloud_species = tuple(cloud_species)  # Make it static for JIT

    #@jit(static_argnames=['imposed_mass_fractions', 'cloud_species'])
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
                # Need
                cloud_opacity = CloudOpacity([species], natural_abundance=False)
                species_full_name = cloud_opacity.species_full_name
                cloud_name = species_full_name.split('_')[0]
                if species in line_species:
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
            exo_atmos,
            pressures,
            temperatures,
            atomic_mass_fractions,
            metallicity,
            carbon_quench_pressure
        )
        mean_molar_mass = mf['MMW']
        return mf, mean_molar_mass

    def precalculated_equilibrium_case():
        mf, mean_molar_mass, _ = \
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                co_ratio * jnp.ones_like(pressures),
                metallicity * jnp.ones_like(pressures),
                temperatures,
                pressures,
                carbon_quench_pressure=carbon_quench_pressure,
                full=True
            )
        return mf, mean_molar_mass

    #@jit(static_argnames=['key_list'])
    def no_chem_case():
        return {}, -INVALID_VALUE

    if metallicity is not None:
        mass_fractions_chemical_table, mean_molar_mass = precalculated_equilibrium_case()

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

        imposed_species_simple = simplify_species_list(list(imposed_mass_fractions.keys()))
        for simple_species in imposed_species_simple:
            if simple_species in mass_fractions_chemical_table:
                del mass_fractions_chemical_table[simple_species]

        for species in tuple(mass_fractions_chemical_table.keys()):
            is_imposed = species in imposed_mass_fractions

            def add_species(_):
                mass_fractions[species] = mass_fractions_chemical_table[species]
                return m_sum_chemical_table_species + mass_fractions_chemical_table[species]

            def skip_species(_):
                return m_sum_chemical_table_species

            # Condition: add if not imposed and not a filling species with non-zero values
            if species in filling_species and filling_species[species] is not None:
                # Species is a filling species - check if it has non-zero values
                has_filling_mass = jnp.any(filling_species[species] > 0)
                should_add = (not is_imposed) & (~has_filling_mass)
            else:
                # Species is not in filling species or is None - just check if imposed
                should_add = not is_imposed

            m_sum_chemical_table_species = lax.cond(
                should_add,
                add_species,
                skip_species,
                operand=None
            )

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

    mass_fractions, m_sum_total = fill_atmosphere(
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
    return mass_fractions, m_sum_total


#@jit(static_argnames=['cloud_species', 'adaptive_mesh_refinement'])
def calculate_cloud_mass_fractions(pressures, temperatures, metallicity, co_ratio, cloud_species,
                                    parameters, mass_fractions, mean_molar_mass,
                                    adaptive_mesh_refinement=False, use_easychem=False):
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
                x_cloud = fc.return_cloud_mass_fraction(cloud, metallicity, co_ratio)
                return 10 ** parameters['eq_scaling_' + cloud_name] * x_cloud
            else:
                return jnp.zeros_like(temperatures)

        def free_case():
            if 'log_X_cb_' + cloud_name in parameter_key_list:
                return (10 ** parameters['log_X_cb_' + cloud_name]) * jnp.ones_like(temperatures)
            else:
                return jnp.zeros_like(temperatures)

        return lax.cond(
            use_easychem,
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
                output = jnp.array(10 ** parameters['log_Pbase_' + cloud_name])
            else:
                output = INVALID_VALUE
            return output.astype(JAX_DTYPE)

        def pbase_case():
            if 'Pbase_' + cloud_name in parameter_key_list:
                output = jnp.array(parameters['Pbase_' + cloud_name])
            else:
                output = INVALID_VALUE
            return output.astype(JAX_DTYPE)

        def eq_case():
            output = jnp.array(fc.calculate_cloud_base_pressure(
                    name=cloud_name,
                    press=pressures,
                    temp=temperatures,
                    metallicity=metallicity,
                    co_ratio=co_ratio,
                    mmw=jnp.mean(mean_molar_mass).astype(JAX_DTYPE)
                ))
            return output.astype(JAX_DTYPE)

        def free_case():
            if 'log_X_cb_' + cloud_name in parameter_key_list:
                output = jnp.array(fc.calculate_cloud_base_pressure_free(
                    cloud_name,
                    pressures,
                    temperatures,
                    10 ** parameters['log_X_cb_' + cloud_name],
                    mean_molar_mass[0]
                ))
            else:
                output = INVALID_VALUE
            return output.astype(JAX_DTYPE)

        return lax.cond(
            'log_Pbase_' + cloud_name in parameter_key_list,
            log_pbase_case,
            lambda: lax.cond(
                'Pbase_' + cloud_name in parameter_key_list,
                pbase_case,
                eq_case
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
                output = jnp.array(parameters['fsed_' + cloud_name])
            else:
                output = INVALID_VALUE
            return output.astype(JAX_DTYPE)

        def fsed_default_case(_):
            if 'fsed' in parameter_key_list:
                output = jnp.array(parameters['fsed'])
            else:
                output = INVALID_VALUE
            return output.astype(JAX_DTYPE)

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

    def skip_clouds_case(mf, fseds):
        for cloud in cloud_species:
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]
            fseds[cloud_name] = get_fsed(cloud_name, parameters)
            mf[cloud] = clouds[cloud_name]
        return mf, fseds

    def fill_clouds_case(mf, fseds):
        for cloud in cloud_species:
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split('_')[0]

            fseds[cloud_name] = get_fsed(cloud_name, parameters)
            mf[cloud] = fill_cloud_mass_fractions(
                cloud_name, pressures, p_bases, clouds, fseds, temperatures, pressure_indices
            )
        return mf, fseds

    mass_fractions, fseds = lax.cond(
        use_easychem,
        skip_clouds_case,
        fill_clouds_case,
        mass_fractions,
        fseds
    )

    return mass_fractions, p_bases, pressure_indices


#@jit(static_argnames=['line_species', 'cloud_species',])
def get_abundances(pressures: typing.ArrayLike, temperatures: typing.ArrayLike,
                   parameters: dict, line_species: tuple, cloud_species: tuple,
                   filling_species: tuple | dict = None, adaptive_mesh_refinement: bool = False,  **kwargs):
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

    metallicity = None
    co_ratio = None
    carbon_quench_pressure = None
    use_equilibrium_chemistry = False
    exo_atmos = None
    atomic_mass_fractions = None

    if 'Fe/H' in parameters.keys():
        metallicity = parameters['Fe/H']

    if 'M/H' in parameters.keys():
        metallicity = parameters['M/H']

    if 'C/O' in parameters.keys():
        co_ratio = parameters['C/O']

    if 'log_Pquench' in parameters.keys():
        carbon_quench_pressure = 10 ** parameters['log_Pquench']
    elif 'Pquench' in parameters.keys():
        carbon_quench_pressure = parameters['Pquench']

    if 'use_easychem' in parameters.keys():
        use_equilibrium_chemistry = parameters['use_exoatmos']
        exo_atmos = parameters['exoatmos']
        atomic_mass_fractions = {}
        for atom_name in exo_atmos.atoms:
            if atom_name in parameters.keys():
                atomic_mass_fractions[atom_name] = parameters[atom_name]

    imposed_mass_fractions = {}
    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split('.R')[0]
        species_basename = Opacity.get_species_base_name(species)

        if species_full_name in parameters.keys():
            imposed_mass_fractions[species_full_name] = parameters[species_full_name]
        elif species_basename in parameters.keys():
            imposed_mass_fractions[species_basename] = parameters[species_basename]

    mass_fractions, m_sum_total = calculate_gas_mass_fractions(
        pressures=pressures,
        temperatures=temperatures,
        line_species=line_species_no_resolution,
        imposed_mass_fractions=imposed_mass_fractions,
        cloud_species=cloud_species,
        filling_species=filling_species,
        metallicity=metallicity,
        co_ratio=co_ratio,
        carbon_quench_pressure=carbon_quench_pressure,
        use_equilibrium_chemistry=use_equilibrium_chemistry,
        exo_atmos=exo_atmos,
        atomic_mass_fractions=atomic_mass_fractions,
        **kwargs
    )
    mean_molar_mass = compute_mean_molar_masses(mass_fractions)

    if metallicity is None:
        metallicity, elemental_ratios = mass_fractions2metallicity(mass_fractions, mean_molar_mass)
        metallicity = jnp.mean(jnp.array(metallicity))
        co_ratio = jnp.mean(elemental_ratios[6]['local'] / elemental_ratios[8]['local'])

    p_bases = {}
    pressure_indices = jnp.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)
    if cloud_species:
        mass_fractions, p_bases, pressure_indices = calculate_cloud_mass_fractions(
            pressures,
            temperatures,
            metallicity,
            co_ratio,
            tuple(cloud_species),
            parameters,
            mass_fractions,
            mean_molar_mass,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            use_easychem=use_equilibrium_chemistry
        )

    # Adjust arrays for amr, and magic fix for FeH abundance.
    for species, value in mass_fractions.items():
        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = value / 2.
            if 'Fe(s)' in p_bases and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(s)']  # Must have iron cloud
                abunds_change_rainout = jnp.where(index_ro, 0.0, abunds_change_rainout)
            mass_fractions[species] = abunds_change_rainout[pressure_indices]
            continue
        mass_fractions[species] = value[pressure_indices]
    return mass_fractions, mean_molar_mass, pressure_indices, p_bases


def get_abundances_from_legacy_dict(pressures: typing.ArrayLike, temperatures: typing.ArrayLike,
                                    parameters: dict, line_species: tuple, cloud_species: tuple,
                                    filling_species: tuple | dict = None, adaptive_mesh_refinement: bool = False,
                                    **kwargs):
    """Legacy wrapper for :func:`get_abundances`.

    Converts a legacy parameter dictionary (values accessed via ``.value``) to a
    plain-value mapping before forwarding to :func:`get_abundances`.

    Args:
        parameters : dict
            Legacy parameter dictionary where each value is an object with a ``.value`` attribute.
        All other arguments are forwarded unchanged to :func:`get_abundances`.
    """
    plain_parameters = {name: param.value for name, param in parameters.items()}
    return get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters=plain_parameters,
        line_species=line_species,
        cloud_species=cloud_species,
        filling_species=filling_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
        **kwargs,
    )


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
                carbon_quench_pressure=pquench_c,
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
        pressure_indices = jnp.array(list(range(pressures.shape[0])))

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
        if species_basename in abundances_interp.keys():
            abundances[species_full_name] = abundances_interp[species_basename][pressure_indices]
        else:
            abundances[species_full_name] = abundances_interp[species_full_name][pressure_indices]
    abundances['H2'] = abundances_interp['H2'][pressure_indices]
    abundances['He'] = abundances_interp['He'][pressure_indices]

    return abundances, mmw, pressure_indices, p_bases