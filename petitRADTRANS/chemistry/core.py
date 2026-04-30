import sys
import copy
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
from jax import typing

try:
    from petitRADTRANS.chemistry.prt_easychem import (
        get_easychem_abundances,
        get_exoatmos_abundances,
    )
except ModuleNotFoundError:
    get_easychem_abundances = None
    get_exoatmos_abundances = None

from petitRADTRANS.opacities.opacities import Opacity, CloudOpacity
from petitRADTRANS.chemistry.pre_calculated_chemistry import (
    pre_calculated_equilibrium_chemistry_table,
)
from petitRADTRANS.chemistry.utils import (
    build_abundance_profile,
    compute_mean_molar_masses,
    fixed_length_adaptive_mesh_refinement,
    linear_spline_profile,
    stepped_profile,
    cubic_spline_profile,
    define_pressure_node_list,
    define_abundance_node_list,
    get_parameter_value,
    simplify_species_list,
    fill_atmosphere,
    mass_fractions2metallicity,
)
from petitRADTRANS.chemistry import clouds as fc

from petitRADTRANS.jax_dtype import JAX_DTYPE

INVALID_VALUE = jnp.array(jnp.inf, JAX_DTYPE)


def _get_active_abundance_profile_mode(species_name, parameters):
    """Return the enabled pressure-varying abundance profile mode for a species.

    Args:
        species_name:
            Species name used to look up profile-control parameters.
        parameters:
            Mapping of model parameters.

    Returns:
        The first enabled profile mode among ``"stepped"``, ``"linear"``, and
        ``"cubic"``, or ``None`` if no mode is active.
    """
    for profile_mode in ("stepped", "linear", "cubic"):
        profile_key = f"{species_name}_{profile_mode}_abundance_profile"

        if profile_key in parameters and bool(
            get_parameter_value(parameters[profile_key])
        ):
            return profile_mode

    return None


def _build_pressure_varying_mass_fraction_profile(
    pressures,
    species_name,
    parameters,
    profile_mode,
):
    """Build a pressure-varying mass-fraction profile for one species.

    Args:
        pressures:
            Pressure grid on which to evaluate the abundance profile.
        species_name:
            Species name used to resolve interpolation nodes from ``parameters``.
        parameters:
            Mapping of model parameters.
        profile_mode:
            Profile-construction mode understood by ``build_abundance_profile``.

    Returns:
        A mass-fraction profile sampled on ``pressures``.
    """
    pressure_nodes = define_pressure_node_list(pressures, species_name, parameters)
    abundance_nodes = define_abundance_node_list(species_name, parameters)

    return build_abundance_profile(
        pressures,
        pressure_nodes,
        abundance_nodes,
        profile_mode=profile_mode,
    )


def _normalize_parameter_mapping(parameters):
    """Convert plain or legacy parameter values to a plain-value mapping."""
    return {
        name: get_parameter_value(value) for name, value in dict(parameters).items()
    }


def _get_parameter_alias_value(parameters, *names, default=None):
    """Return the first non-``None`` parameter value among the provided aliases."""
    for name in names:
        if name in parameters:
            value = get_parameter_value(parameters[name])

            if value is not None:
                return value

    return default


def _has_truthy_parameter(parameters, *names):
    """Return whether any aliased parameter is explicitly enabled."""
    for name in names:
        if name in parameters and bool(get_parameter_value(parameters[name])):
            return True

    return False


def _resolve_chemistry_mode(parameters):
    """Resolve free, precalculated-equilibrium, and full-easyChem chemistry modes.

    The parameter mapping drives chemistry selection as follows:
        - ``use_easychem`` or ``use_exoatmos`` explicitly enables the full
          easyChem calculation.
        - Otherwise, a metallicity parameter (``metallicity``, ``M/H``, or
          ``Fe/H``) selects the built-in precalculated equilibrium table by
          default.
        - If neither signal is present, the caller stays on the free-chemistry
          path.
    """
    metallicity = _get_parameter_alias_value(
        parameters,
        "metallicity",
        "M/H",
        "Fe/H",
    )
    co_ratio = _get_parameter_alias_value(parameters, "co_ratio", "C/O")
    carbon_quench_pressure = _get_parameter_alias_value(
        parameters,
        "Pquench",
        "carbon_quench_pressure",
    )
    log_carbon_quench_pressure = _get_parameter_alias_value(
        parameters,
        "log_Pquench",
        "log_pquench",
    )

    if log_carbon_quench_pressure is not None:
        carbon_quench_pressure = 10**log_carbon_quench_pressure

    use_full_easychem = _has_truthy_parameter(
        parameters,
        "use_easychem",
        "use_exoatmos",
    )
    use_precalculated_equilibrium_chemistry = bool(
        not use_full_easychem
        and (
            metallicity is not None
            or _has_truthy_parameter(parameters, "use_equilibrium_chemistry")
        )
    )

    exo_atmos = _get_parameter_alias_value(parameters, "exo_atmos", "exoatmos")
    atomic_mass_fractions = _get_parameter_alias_value(
        parameters,
        "atomic_mass_fractions",
    )

    if atomic_mass_fractions is None:
        atomic_mass_fractions = {}
    else:
        atomic_mass_fractions = dict(atomic_mass_fractions)

    if exo_atmos is not None and hasattr(exo_atmos, "atoms"):
        for atom_name in exo_atmos.atoms:
            if atom_name in parameters:
                atomic_mass_fractions[atom_name] = get_parameter_value(
                    parameters[atom_name]
                )

    return {
        "metallicity": metallicity,
        "co_ratio": co_ratio,
        "carbon_quench_pressure": carbon_quench_pressure,
        "use_full_easychem": use_full_easychem,
        "use_precalculated_equilibrium_chemistry": (
            use_precalculated_equilibrium_chemistry
        ),
        "exo_atmos": exo_atmos,
        "atomic_mass_fractions": atomic_mass_fractions,
    }


def _build_easychem_mass_fractions(
    pressures,
    temperatures,
    parameters,
    line_species,
    cloud_species,
    metallicity,
    co_ratio,
    carbon_quench_pressure,
    use_full_easychem,
    exo_atmos,
    atomic_mass_fractions,
    adaptive_mesh_refinement,
):
    """Build mass fractions from the explicit full easyChem chemistry path."""
    if not use_full_easychem:
        raise ValueError("full easyChem mass fractions require use_full_easychem=True")

    if exo_atmos is not None:
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError(
                "module 'easychem' is not installed, remove key 'use_easychem' "
                "from the parameters"
            )

        abundances_interp = get_exoatmos_abundances(
            exo_atmos=exo_atmos,
            pressures=pressures,
            temperatures=temperatures,
            atomic_mass_fractions=atomic_mass_fractions,
            metallicity=0.0 if metallicity is None else metallicity,
            carbon_quench_pressure=carbon_quench_pressure,
            AMR=adaptive_mesh_refinement,
        )
    else:
        if get_easychem_abundances is None:
            raise ModuleNotFoundError(
                "module 'easychem' is not installed, remove key 'use_easychem' "
                "from the parameters"
            )

        legacy_parameters = {
            name: SimpleNamespace(value=value) for name, value in parameters.items()
        }
        abundances_interp = get_easychem_abundances(
            pressures=pressures,
            temperatures=temperatures,
            line_species=line_species,
            cloud_species=cloud_species,
            metallicity=0.0 if metallicity is None else metallicity,
            carbon_to_oxygen=0.55 if co_ratio is None else co_ratio,
            carbon_quench_pressure=carbon_quench_pressure,
            parameters=legacy_parameters,
            AMR=adaptive_mesh_refinement,
        )

    mass_fractions = {
        species: jnp.asarray(abundance)
        for species, abundance in abundances_interp.items()
        if species != "MMW"
    }
    line_species_simple = tuple(simplify_species_list(list(line_species)))

    for i, simple_species in enumerate(line_species_simple):
        if simple_species in mass_fractions and simple_species != line_species[i]:
            mass_fractions[line_species[i]] = mass_fractions.pop(simple_species)

    mean_molar_mass = jnp.asarray(abundances_interp["MMW"])

    return mass_fractions, mean_molar_mass


# @jit(static_argnames=['line_species', 'cloud_species', 'verbose'])
def calculate_gas_mass_fractions(
    pressures,
    temperatures=None,
    imposed_mass_fractions=None,
    line_species=None,
    cloud_species=None,
    filling_species=None,
    use_equilibrium_chemistry=False,
    metallicity=None,
    co_ratio=0.55,
    carbon_quench_pressure=None,
    exo_atmos=None,
    atomic_mass_fractions=None,
    **kwargs,
):
    """Build and normalize per-layer gas mass-fraction profiles.

    The function combines explicitly imposed gas abundances, optional
    pre-calculated equilibrium chemistry, and background filling species so that
    each atmospheric layer sums to unity.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        temperatures:
            Temperature profile on the same grid as ``pressures``. Required when
            interpolating the equilibrium chemistry table.
        imposed_mass_fractions:
            Mapping of species names to imposed mass fractions. Scalar inputs are
            broadcast over the pressure grid. Negative values are interpreted as
            base-10 logarithmic mass fractions.
        line_species:
            Gas-opacity species names. These are used to reconcile opacity names
            with chemistry-table names and to discover pressure-varying profile
            controls.
        cloud_species:
            Cloud-opacity species names. Accepted for API compatibility with the
            broader chemistry pipeline; they are not used directly in the active
            gas-fraction path.
        filling_species:
            Optional fill-specification for background gases. This may be
            ``None``, a species name, an iterable of species names, or a mapping
            of species names to fill weights.
        use_equilibrium_chemistry:
            If ``True``, use the built-in precalculated equilibrium chemistry
            table together with ``metallicity`` and ``co_ratio``. The explicit
            full easyChem path is resolved upstream in :func:`get_abundances`.
        metallicity:
            Metallicity passed to the pre-calculated equilibrium chemistry table.
            When provided, equilibrium gas abundances are interpolated.
        co_ratio:
            Carbon-to-oxygen ratio used with the equilibrium chemistry table.
        carbon_quench_pressure:
            Optional quench pressure forwarded to the equilibrium chemistry
            interpolation.
        exo_atmos:
            Compatibility argument reserved for legacy/easychem callers.
        atomic_mass_fractions:
            Compatibility argument reserved for legacy/easychem callers.
        **kwargs:
            Additional keyword arguments. If a ``parameters`` mapping is present,
            it is used to detect pressure-varying abundance-profile controls.

    Returns:
        A tuple ``(mass_fractions, m_sum_total)`` containing the per-species
        mass-fraction profiles and the total mass-fraction sum in each layer.
    """

    # easychem/exo_atmos remain compatibility arguments for now.
    _ = exo_atmos, atomic_mass_fractions

    use_precalculated_equilibrium_chemistry = bool(
        use_equilibrium_chemistry and metallicity is not None
    )

    parameter_source: dict[str, Any] = kwargs.get("parameters", kwargs)
    if parameter_source is None:
        parameter_source = {}

    pressure_template = jnp.asarray(pressures)

    if line_species is None:
        line_species = ()
    else:
        line_species = tuple(line_species)

    if cloud_species is None:
        cloud_species = ()
    else:
        cloud_species = tuple(cloud_species)

    if imposed_mass_fractions is None:
        imposed_mass_fractions = {}
    else:
        imposed_mass_fractions = dict(imposed_mass_fractions)

    line_species_simple = tuple(simplify_species_list(list(line_species)))
    line_species_lookup = set(line_species)
    line_species_simple_lookup = set(line_species_simple)

    def _as_mass_fraction_profile(raw):
        raw_array = jnp.asarray(raw)
        abundance = jnp.where(raw_array < 0, 10**raw_array, raw_array)

        if raw_array.ndim == 0:
            return jnp.full_like(pressure_template, abundance)

        return jnp.broadcast_to(abundance, pressure_template.shape)

    def _merge_pressure_varying_profiles(current_imposed_mass_fractions):
        merged_mass_fractions = dict(current_imposed_mass_fractions)

        for species in line_species:
            species_name = species.split(".R")[0]
            species_basename = Opacity.get_species_base_name(species_name)

            if (
                species_name in merged_mass_fractions
                or species_basename in merged_mass_fractions
            ):
                continue

            profile_mode = _get_active_abundance_profile_mode(
                species_name, parameter_source
            )
            profile_species_name = species_name

            if profile_mode is None:
                profile_mode = _get_active_abundance_profile_mode(
                    species_basename, parameter_source
                )
                profile_species_name = species_basename

            if profile_mode is not None:
                merged_mass_fractions[profile_species_name] = (
                    _build_pressure_varying_mass_fraction_profile(
                        pressure_template,
                        profile_species_name,
                        parameter_source,
                        profile_mode,
                    )
                )

        return merged_mass_fractions

    imposed_mass_fractions = _merge_pressure_varying_profiles(imposed_mass_fractions)

    def build_filling_species(filling_species):
        def _zeros():
            return jnp.zeros_like(pressure_template)

        def _filled(value):
            return jnp.full_like(pressure_template, value)

        def h2he_case(filling_species):
            fill_species = {}
            if use_precalculated_equilibrium_chemistry:
                fill_species = {
                    "H2": _zeros(),
                    "He": _zeros(),
                }
            else:
                fill_species = {
                    "H2": _filled(74.0),
                    "He": _filled(24.0),
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
                return {key: _filled(value) for key, value in filling_species.items()}

            return filling_species

        if (
            hasattr(filling_species, "__iter__")
            and len(filling_species) == 2
            and not isinstance(filling_species, dict)
            and "H2" in filling_species
            and "He" in filling_species
        ):
            return h2he_case(filling_species)

        return default_case(filling_species)

    filling_species = build_filling_species(filling_species)

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
        if hasattr(fs, "__iter__"):
            return iter_case(None)
        if fs is None:
            return none_case(None)
        return error_case(None)

    if not isinstance(filling_species, dict):
        filling_species = handle_non_dict(filling_species)

    mass_fractions: dict[str, typing.ArrayLike] = {}
    m_sum_imposed_species = jnp.zeros_like(pressure_template)

    # Free chemistry species
    for species in imposed_mass_fractions:
        mass_fractions[species] = _as_mass_fraction_profile(
            imposed_mass_fractions[species]
        )
        m_sum_imposed_species += mass_fractions[species]

    # Remove imposed species from filling species
    for species in imposed_mass_fractions:
        if species in filling_species:
            # warnings.warn(...)  # Not JIT compatible
            del filling_species[species]

    # Ensure that the sum of imposed mass fractions is <= 1
    imposed_gas_species = tuple(
        species
        for species in imposed_mass_fractions
        if species in line_species_lookup or species in line_species_simple_lookup
    )

    if bool(jnp.any(m_sum_imposed_species > 1)):
        for species in imposed_gas_species:
            mass_fractions[species] = jnp.where(
                m_sum_imposed_species > 1,
                mass_fractions[species] / m_sum_imposed_species,
                mass_fractions[species],
            )

    # Chemical equilibrium mass fractions
    mass_fractions_chemical_table: dict[str, typing.ArrayLike] = {}

    def precalculated_equilibrium_case(
        current_metallicity: Any,
    ) -> dict[str, typing.ArrayLike]:
        mf, _, _ = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                co_ratio * jnp.ones_like(pressure_template),
                current_metallicity * jnp.ones_like(pressure_template),
                temperatures,
                pressure_template,
                carbon_quench_pressure=cast(Any, carbon_quench_pressure),
                full=True,
            )
        )
        return cast(dict[str, typing.ArrayLike], mf)

    if use_precalculated_equilibrium_chemistry:
        mass_fractions_chemical_table = precalculated_equilibrium_case(metallicity)

    # Add non-imposed mass fractions to mass fractions
    m_sum_chemical_table_species = jnp.zeros_like(pressure_template)

    if mass_fractions_chemical_table:
        for i, simple_species in enumerate(line_species_simple):
            if (
                simple_species in mass_fractions_chemical_table
                and simple_species != line_species[i]
            ):
                mass_fractions_chemical_table[line_species[i]] = (
                    mass_fractions_chemical_table[simple_species]
                )
                del mass_fractions_chemical_table[simple_species]

        imposed_species_simple = simplify_species_list(
            list(imposed_mass_fractions.keys())
        )
        for simple_species in imposed_species_simple:
            if simple_species in mass_fractions_chemical_table:
                del mass_fractions_chemical_table[simple_species]

        for species, species_mass_fraction in tuple(
            mass_fractions_chemical_table.items()
        ):
            if species in imposed_mass_fractions:
                continue

            if species in filling_species and filling_species[species] is not None:
                has_filling_mass = bool(jnp.any(filling_species[species] > 0))
                if has_filling_mass:
                    continue

            mass_fractions[species] = species_mass_fraction
            m_sum_chemical_table_species = (
                m_sum_chemical_table_species + species_mass_fraction
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
    pre_fill_mass_fraction_sum = jnp.asarray(m_sum_total)

    mass_fractions, m_sum_total = fill_atmosphere(
        mass_fractions=mass_fractions,
        filling_species=filling_species,
        mass_fraction_sum=m_sum_total,
    )
    m_sum_total = jnp.asarray(m_sum_total)
    effective_imposed_mass_fraction_sum = jnp.where(
        m_sum_imposed_species > 1,
        jnp.ones_like(m_sum_imposed_species),
        m_sum_imposed_species,
    )
    chemistry_table_oversubscribed = pre_fill_mass_fraction_sum > (1 + 1e-12)
    explicit_filling_species = {
        species
        for species, weights in filling_species.items()
        if weights is not None and bool(jnp.any(weights > 0))
    }

    has_chemical_table_species = bool(jnp.any(m_sum_chemical_table_species > 0))

    for species, mf_array in mass_fractions.items():
        if species not in imposed_mass_fractions:
            if has_chemical_table_species:
                if (
                    species in mass_fractions_chemical_table
                    and species not in explicit_filling_species
                ):
                    mass_fractions[species] = jnp.where(
                        chemistry_table_oversubscribed,
                        mf_array
                        * (1 - effective_imposed_mass_fraction_sum)
                        / m_sum_chemical_table_species,
                        mf_array,
                    )
            else:
                mass_fractions[species] = mf_array / m_sum_total

    final_mass_fraction_sum = jnp.zeros_like(pressure_template)

    for mf_array in mass_fractions.values():
        final_mass_fraction_sum = final_mass_fraction_sum + mf_array

    return mass_fractions, final_mass_fraction_sum


# @jit(static_argnames=['cloud_species', 'adaptive_mesh_refinement'])
def calculate_cloud_mass_fractions(
    pressures,
    temperatures,
    metallicity,
    co_ratio,
    cloud_species,
    parameters,
    mass_fractions,
    mean_molar_mass,
    adaptive_mesh_refinement=False,
    use_easychem=False,
):
    """Add cloud abundance profiles and cloud-base metadata to a composition.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        temperatures:
            Temperature profile on the same grid as ``pressures``.
        metallicity:
            Metallicity used when equilibrium cloud abundances or cloud bases are
            requested.
        co_ratio:
            Carbon-to-oxygen ratio used with equilibrium cloud calculations.
        cloud_species:
            Condensate-opacity species names to evaluate.
        parameters:
            Mapping of plain model parameter values controlling cloud abundances,
            cloud bases, and sedimentation efficiencies.
        mass_fractions:
            Existing species mass fractions. This mapping is updated in place with
            the computed cloud profiles.
        mean_molar_mass:
            Mean molar-mass profile used when computing cloud-base pressures.
        adaptive_mesh_refinement:
            Compatibility argument retained for the public API. The current cloud
            path returns the identity pressure grid.
        use_easychem:
            If ``True``, copy cloud abundances directly from a full easyChem
            composition instead of constructing sedimented cloud profiles.

    Returns:
        A tuple ``(mass_fractions, p_bases, pressure_indices)`` containing the
        updated mass fractions, cloud-base pressures keyed by cloud name, and the
        pressure-grid indices used by the cloud path.
    """
    _ = adaptive_mesh_refinement

    clouds = {}
    p_bases = {}
    fseds = {}
    parameter_keys = set(parameters)
    zero_profile = jnp.zeros_like(temperatures)
    one_profile = jnp.ones_like(temperatures)
    has_equilibrium_cloud_base = any(
        key in parameter_keys for key in ("Fe/H", "M/H", "metallicity")
    )
    mean_molar_mass_mean = float(jnp.mean(mean_molar_mass))
    mean_molar_mass_surface = float(jnp.asarray(mean_molar_mass[0]))

    cloud_metadata = []
    for cloud in cloud_species:
        cloud_name = CloudOpacity(
            [cloud], natural_abundance=False
        ).species_full_name.split("_")[0]
        cloud_metadata.append((cloud, cloud_name))

    def get_cloud_abundance(cloud, cloud_name):
        if use_easychem:
            return mass_fractions.get(cloud_name, zero_profile)

        eq_scaling_key = f"eq_scaling_{cloud_name}"
        if eq_scaling_key in parameter_keys:
            x_cloud = fc.return_cloud_mass_fraction(cloud, metallicity, co_ratio)
            return (10 ** parameters[eq_scaling_key]) * x_cloud

        log_x_cb_key = f"log_X_cb_{cloud_name}"
        if log_x_cb_key in parameter_keys:
            return (10 ** parameters[log_x_cb_key]) * one_profile

        return zero_profile

    def get_cloud_base(cloud_name):
        log_pbase_key = f"log_Pbase_{cloud_name}"
        if log_pbase_key in parameter_keys:
            return jnp.asarray(10 ** parameters[log_pbase_key], dtype=JAX_DTYPE)

        pbase_key = f"Pbase_{cloud_name}"
        if pbase_key in parameter_keys:
            return jnp.asarray(parameters[pbase_key], dtype=JAX_DTYPE)

        log_x_cb_key = f"log_X_cb_{cloud_name}"
        if not has_equilibrium_cloud_base and log_x_cb_key in parameter_keys:
            return jnp.asarray(
                fc.calculate_cloud_base_pressure_free(
                    cloud_name,
                    pressures,
                    temperatures,
                    10 ** parameters[log_x_cb_key],
                    mean_molar_mass_surface,
                ),
                dtype=JAX_DTYPE,
            )

        if has_equilibrium_cloud_base:
            return jnp.asarray(
                fc.calculate_cloud_base_pressure(
                    name=cloud_name,
                    press=pressures,
                    temp=temperatures,
                    metallicity=metallicity,
                    co_ratio=co_ratio,
                    mmw=mean_molar_mass_mean,
                ),
                dtype=JAX_DTYPE,
            )

        return INVALID_VALUE

    def get_fsed(cloud_name):
        cloud_fsed_key = f"fsed_{cloud_name}"
        if cloud_fsed_key in parameter_keys:
            return jnp.asarray(parameters[cloud_fsed_key], dtype=JAX_DTYPE)
        if "fsed" in parameter_keys:
            return jnp.asarray(parameters["fsed"], dtype=JAX_DTYPE)
        return INVALID_VALUE

    # AMR currently returns the identity grid until the pressure-remeshing path is restored.
    pressure_indices = jnp.arange(pressures.shape[0], dtype=int)
    fseds = {}

    for cloud, cloud_name in cloud_metadata:
        cloud_abundance = get_cloud_abundance(cloud, cloud_name)
        cloud_base = get_cloud_base(cloud_name)
        cloud_fsed = get_fsed(cloud_name)

        clouds[cloud_name] = cloud_abundance
        p_bases[cloud_name] = cloud_base
        fseds[cloud_name] = cloud_fsed

        if use_easychem:
            mass_fractions[cloud] = cloud_abundance
            continue

        cloud_profile = jnp.where(
            pressures < cloud_base,
            cloud_abundance * (pressures / cloud_base) ** cloud_fsed,
            0.0,
        )
        mass_fractions[cloud] = cloud_profile[pressure_indices]

    return mass_fractions, p_bases, pressure_indices


# @jit(static_argnames=['line_species', 'cloud_species',])
def get_abundances(
    pressures: typing.ArrayLike,
    temperatures: typing.ArrayLike,
    parameters: dict,
    line_species: tuple,
    cloud_species: tuple,
    filling_species: tuple | dict = None,
    adaptive_mesh_refinement: bool = False,
    **kwargs,
):
    """Assemble gas and cloud abundances for a pressure-temperature profile.

    This function resolves the requested chemistry mode from ``parameters``,
    computes gas mass fractions, derives the mean molar mass, and optionally adds
    condensate abundances and cloud-base information.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        temperatures:
            Temperature profile on the same grid as ``pressures``.
        parameters:
            Mapping of plain model parameter values. Supported keys depend on the
            selected chemistry and cloud mode and may include equilibrium-chemistry
            controls, directly imposed abundances, pressure-varying profile flags,
            cloud abundance parameters, and cloud-base parameters.
        line_species:
            Gas-opacity species names.
        cloud_species:
            Condensate-opacity species names.
        filling_species:
            Optional background fill-specification forwarded to
            ``calculate_gas_mass_fractions``. If omitted, a default H2/He fill is
            used.
        adaptive_mesh_refinement:
            Forwarded to ``calculate_cloud_mass_fractions``. The current cloud path
            returns the identity pressure grid.
        **kwargs:
            Additional keyword arguments forwarded to
            ``calculate_gas_mass_fractions``.

    Returns:
        A tuple ``(mass_fractions, mean_molar_mass, pressure_indices, p_bases)``
        containing the species mass fractions, the mean molar-mass profile, the
        selected pressure-grid indices, and the cloud-base pressures.

    Chemistry-mode selection:
        ``get_abundances`` distinguishes three chemistry modes from the plain
        parameter mapping:
            - free chemistry: no equilibrium selector is present;
            - precalculated equilibrium chemistry: selected by default when a
              metallicity parameter such as ``Fe/H``, ``M/H``, or
              ``metallicity`` is present, or when
              ``use_equilibrium_chemistry=True``;
            - full easyChem chemistry: selected only when ``use_easychem`` or
              ``use_exoatmos`` is explicitly truthy.

        An explicit full easyChem request takes precedence over the default
        precalculated equilibrium-table path.
    """
    if filling_species is None:
        filling_species = {
            "H2": 0.74
            * jnp.ones(pressures.size),  # % of atmosphere by mass for H2/He ratio
            "He": 0.24 * jnp.ones(pressures.size),
        }

    line_species_no_resolution = tuple(
        [species.split(".R")[0] for species in line_species]
    )
    plain_parameters = _normalize_parameter_mapping(parameters)
    chemistry_mode = _resolve_chemistry_mode(plain_parameters)
    metallicity = chemistry_mode["metallicity"]
    co_ratio = chemistry_mode["co_ratio"]
    carbon_quench_pressure = chemistry_mode["carbon_quench_pressure"]
    use_full_easychem = chemistry_mode["use_full_easychem"]
    use_precalculated_equilibrium_chemistry = chemistry_mode[
        "use_precalculated_equilibrium_chemistry"
    ]
    exo_atmos = chemistry_mode["exo_atmos"]
    atomic_mass_fractions = chemistry_mode["atomic_mass_fractions"]

    imposed_mass_fractions = {}
    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split(".R")[0]
        species_basename = Opacity.get_species_base_name(species)

        if species_full_name in plain_parameters:
            imposed_mass_fractions[species_full_name] = plain_parameters[
                species_full_name
            ]
        elif species_basename in plain_parameters:
            imposed_mass_fractions[species_basename] = plain_parameters[
                species_basename
            ]

    if use_full_easychem:
        mass_fractions, mean_molar_mass = _build_easychem_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            parameters=plain_parameters,
            line_species=line_species_no_resolution,
            cloud_species=tuple(cloud_species),
            metallicity=metallicity,
            co_ratio=co_ratio,
            carbon_quench_pressure=carbon_quench_pressure,
            use_full_easychem=use_full_easychem,
            exo_atmos=exo_atmos,
            atomic_mass_fractions=atomic_mass_fractions,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
        )
    else:
        mass_fractions, _ = calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            line_species=line_species_no_resolution,
            imposed_mass_fractions=imposed_mass_fractions,
            cloud_species=cloud_species,
            filling_species=filling_species,
            metallicity=metallicity,
            co_ratio=co_ratio,
            carbon_quench_pressure=carbon_quench_pressure,
            use_equilibrium_chemistry=use_precalculated_equilibrium_chemistry,
            exo_atmos=exo_atmos,
            atomic_mass_fractions=atomic_mass_fractions,
            parameters=plain_parameters,
            **kwargs,
        )
        mean_molar_mass = compute_mean_molar_masses(mass_fractions)

    if metallicity is None:
        metallicity, elemental_ratios = mass_fractions2metallicity(
            mass_fractions, mean_molar_mass
        )
        metallicity = jnp.mean(jnp.array(metallicity))
        # TODO: Test if elemental_ratios[6] and elemental_ratios[8] exist
        co_ratio = jnp.mean(elemental_ratios[6]["local"] / elemental_ratios[8]["local"])

    p_bases = {}
    pressure_indices = jnp.linspace(
        0, pressures.shape[0] - 1, pressures.shape[0], dtype=int
    )
    if cloud_species:
        mass_fractions, p_bases, pressure_indices = calculate_cloud_mass_fractions(
            pressures,
            temperatures,
            metallicity,
            co_ratio,
            tuple(cloud_species),
            plain_parameters,
            mass_fractions,
            mean_molar_mass,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            use_easychem=use_full_easychem,
        )

    # Adjust arrays for amr, and magic fix for FeH abundance.
    for species, value in mass_fractions.items():
        if "FeH" in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = value / 2.0
            if "Fe(s)" in p_bases and not use_full_easychem:
                index_ro = pressures < p_bases["Fe(s)"]  # Must have iron cloud
                abunds_change_rainout = jnp.where(index_ro, 0.0, abunds_change_rainout)
            mass_fractions[species] = abunds_change_rainout[pressure_indices]
            continue
        mass_fractions[species] = value[pressure_indices]
    return mass_fractions, mean_molar_mass, pressure_indices, p_bases


def get_abundances_from_legacy_dict(
    pressures: typing.ArrayLike,
    temperatures: typing.ArrayLike,
    parameters: dict,
    line_species: tuple,
    cloud_species: tuple,
    filling_species: tuple | dict = None,
    adaptive_mesh_refinement: bool = False,
    **kwargs,
):
    """Legacy wrapper for :func:`get_abundances`.

    Converts a legacy parameter dictionary (values accessed via ``.value``) to a
    plain-value mapping before forwarding to :func:`get_abundances`.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        temperatures:
            Temperature profile on the same grid as ``pressures``.
        parameters:
            Legacy parameter mapping whose values expose a ``.value`` attribute.
        line_species:
            Gas-opacity species names.
        cloud_species:
            Condensate-opacity species names.
        filling_species:
            Optional background fill-specification forwarded to
            :func:`get_abundances`.
        adaptive_mesh_refinement:
            Forwarded unchanged to :func:`get_abundances`.
        **kwargs:
            Additional keyword arguments forwarded to :func:`get_abundances`.

    Returns:
        The same tuple returned by :func:`get_abundances`.
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


def get_abundances_legacy(
    pressures,
    temperatures,
    line_species,
    cloud_species,
    parameters,
    adaptive_mesh_refinement=False,
):
    """Legacy abundance pipeline retained for compatibility.

    This historical implementation supports the older parameter objects whose
    values are accessed via ``.value`` and preserves the original free-chemistry,
    equilibrium-chemistry, and cloud-handling flow.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        temperatures:
            Temperature profile on the same grid as ``pressures``.
        line_species:
            Gas-opacity species names.
        cloud_species:
            Condensate-opacity species names.
        parameters:
            Legacy parameter mapping whose values are accessed via ``.value``.
            Supported keys depend on the selected legacy chemistry and cloud mode.
        adaptive_mesh_refinement:
            Whether to derive adaptive pressure-grid indices with
            ``fixed_length_adaptive_mesh_refinement``.

    Returns:
        A tuple ``(abundances, mmw, pressure_indices, p_bases)`` containing the
        species mass fractions, the mean molar-mass profile, the selected
        pressure-grid indices, and the cloud-base pressures.
    """
    # TODO replace by SpectralModel function
    # Free Chemistry
    abundances_interp = {}
    mmw = None
    if "use_easychem" in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError(
                "module 'easychem' is not installed, "
                "remove key 'use_easychem from the parameters'"
            )

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent with poor man's
        abundances_interp = get_exoatmos_abundances(pressures, temperatures, parameters)
        mmw = abundances_interp["MMW"]
    elif "C/O" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor man's

        # Interpolated Equilibrium chemistry
        # Make the abundance profile
        pquench_c = None

        if "log_pquench" in parameters.keys():
            pquench_c = 10 ** parameters["log_pquench"].value

        abundances_interp, mmw, _ = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                parameters["C/O"].value * jnp.ones_like(pressures),
                parameters["Fe/H"].value * jnp.ones_like(pressures),
                temperatures,
                pressures,
                carbon_quench_pressure=pquench_c,
                full=True,
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
        species_full_name = species_opacity.get_full_name().split(".R")[0]
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
            pressures, species_basename, parameters
        )
        abundance_nodes = define_abundance_node_list(species_basename, parameters)

        # Stepped abundance profile
        if f"{species_basename}_stepped_abundance_profile" in parameters.keys():
            if species_basename in abundances_interp.keys():
                msum -= abundances_interp[species_basename]
            abundances_interp[species_basename] = stepped_profile(
                pressures, pressure_interpolation_nodes, abundance_nodes
            )
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
                nnodes=len(pressure_interpolation_nodes),
            )
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
                nnodes=len(pressure_interpolation_nodes),
            )
            msum += abundances_interp[species_basename]

    # For free chemistry, need to fill with background gas (H2-He)
    # TODO use arbitrary background gas
    if "Fe/H" not in parameters.keys():
        # Check to make sure we're using free chemistry
        # Whatever's left is H2 and He
        if "H2" in parameters.keys():
            abundances_interp["H2"] = 10 ** parameters["H2"].value * jnp.ones_like(
                pressures
            )
        else:
            abundances_interp["H2"] = 0.766 * (1.0 - msum)

        if "He" in parameters.keys():
            abundances_interp["He"] = 10 ** parameters["He"].value * jnp.ones_like(
                pressures
            )
        else:
            abundances_interp["He"] = 0.234 * (1.0 - msum)

        # Imposing strict limit on msum to ensure H2 dominated composition
        # if jnp.max(msum) > 1.0:
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
        cloud_name = species_full_name.split("_")[0]

        if "use_easychem" in parameters.keys():
            # AMR CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cloud_name] = abundances_interp[cloud_name]
            continue

        if "eq_scaling_" + cloud_name in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(
                cloud, parameters["Fe/H"].value, parameters["C/O"].value
            )
            # Scaled by a constant factor
            clouds[cloud_name] = (
                10 ** parameters["eq_scaling_" + cloud_name].value * x_cloud
            )
        else:
            # Free cloud abundance
            clouds[cloud_name] = 10 ** parameters["log_X_cb_" + cloud_name].value

        # Free cloud bases
        if "log_Pbase_" + cloud_name in parameters.keys():
            p_bases[cloud_name] = 10 ** parameters["log_Pbase_" + cloud_name].value
        elif "Pbase_" + cloud_name in parameters.keys():
            p_bases[cloud_name] = parameters["Pbase_" + cloud_name].value
        # Equilibrium locations
        elif "Fe/H" in parameters.keys():
            p_bases[cloud_name] = fc.simple_cdf(
                cloud_name,
                pressures,
                temperatures,
                parameters["Fe/H"].value,
                parameters["C/O"].value,
                jnp.mean(mmw),
            )
        else:
            p_bases[cloud_name] = fc.simple_cdf_free(
                cloud_name,
                pressures,
                temperatures,
                10 ** parameters["log_X_cb_" + cloud_name].value,
                mmw[0],
            )

    # Find high resolution pressure grid and indices
    if adaptive_mesh_refinement:
        press_use, pressure_indices = fixed_length_adaptive_mesh_refinement(
            jnp.array(list(p_bases.values())),
            pressures,
            parameters["pressure_scaling"].value,
            parameters["pressure_width"].value,
        )
    else:
        pressure_indices = jnp.array(list(range(pressures.shape[0])))

    fseds = {}

    if "use_easychem" not in parameters.keys():
        for cloud in copy.copy(cloud_species):
            cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
            species_full_name = cloud_opacity.species_full_name
            cloud_name = species_full_name.split("_")[0]

            # Set up fseds per-cloud
            if "fsed_" + cloud_name in parameters.keys():
                fseds[cloud_name] = parameters["fsed_" + cloud_name].value
            else:
                fseds[cloud_name] = parameters["fsed"].value

            abundances[cloud_name] = jnp.zeros_like(temperatures)
            abundances[cloud_name] = jnp.where(
                pressures < p_bases[cloud_name],
                clouds[cloud_name]
                * (pressures / p_bases[cloud_name]) ** fseds[cloud_name],
                0.0,
            )
            abundances[cloud_name] = abundances[cloud_name][pressure_indices]

    for species in line_species:
        species_opacity = Opacity([species])
        species_full_name = species_opacity.get_full_name().split(".R")[0]
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

        if "FeH" in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = abundances_interp[species] / 2.0
            if "Fe(c)" in p_bases.keys() and "use_easychem" not in parameters.keys():
                index_ro = pressures < p_bases["Fe(c)"]  # Must have iron cloud
                abunds_change_rainout = abunds_change_rainout.at[index_ro].set(0.0)
            abundances[species] = abunds_change_rainout[pressure_indices]
        if species_basename in abundances_interp.keys():
            abundances[species_full_name] = abundances_interp[species_basename][
                pressure_indices
            ]
        else:
            abundances[species_full_name] = abundances_interp[species_full_name][
                pressure_indices
            ]
    abundances["H2"] = abundances_interp["H2"][pressure_indices]
    abundances["He"] = abundances_interp["He"][pressure_indices]

    return abundances, mmw, pressure_indices, p_bases
