import jax.numpy as jnp
from jax import lax, debug, jit
from jax.typing import ArrayLike
from petitRADTRANS.radtrans_core import radiative_transfer_core, cloud_core
from petitRADTRANS.opacities.opacities import CloudOpacity, Opacity
from petitRADTRANS.opacities.rayleigh import compute_power_law_opacities
import petitRADTRANS.physical_constants as cst

from petitRADTRANS.utils import LockedDict

from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE, WORKING_JAX_DTYPE

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=FEAUTRIER_JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=FEAUTRIER_JAX_DTYPE)


def _selected_clouds_have_effect(mass_fractions, selected_cloud_species):
    has_effect = False

    for species in selected_cloud_species:
        has_effect = jnp.logical_or(has_effect, jnp.any(mass_fractions[species] > 0))

    return has_effect


def _build_cloud_species_mass_fractions(mass_fractions, selected_cloud_species):
    cloud_species_mass_fractions = {}

    for species in selected_cloud_species:
        cloud_species_mass_fractions[species] = mass_fractions[species]

    return cloud_species_mass_fractions


def _resolve_cloud_parameter_key(parameter, cloud_name):
    if cloud_name in parameter:
        return cloud_name

    cloud_opacity = CloudOpacity([cloud_name], natural_abundance=False)
    species_full_name = cloud_opacity.species_full_name
    cloud_short_name = species_full_name.split("_")[0]

    for candidate in (species_full_name, cloud_short_name):
        if candidate in parameter:
            return candidate

    return None


def _expand_cloud_parameter_column(value, pressures, parameter_name, cloud_name):
    value = jnp.asarray(value, dtype=FEAUTRIER_JAX_DTYPE)

    if value.ndim == 0:
        return jnp.full((pressures.size,), value, dtype=FEAUTRIER_JAX_DTYPE)

    if value.ndim == 1:
        if value.shape[0] == 1:
            return jnp.full((pressures.size,), value[0], dtype=FEAUTRIER_JAX_DTYPE)

        if value.shape[0] == pressures.size:
            return value

    raise ValueError(
        f"{parameter_name} for cloud species '{cloud_name}' must be a scalar or an array "
        f"with shape {(pressures.size,)}, but was of shape {value.shape}"
    )


def _standardize_cloud_species_parameter(
    parameter,
    cloud_species,
    pressures,
    parameter_name,
):
    if parameter is None or parameter is _MISSING_FLOAT:
        return parameter

    n_layers = pressures.size
    n_clouds = len(cloud_species)

    if isinstance(parameter, dict):
        parameter_grid = jnp.zeros((n_layers, n_clouds), dtype=FEAUTRIER_JAX_DTYPE)
        missing_species = []

        for i_spec, cloud_name in enumerate(cloud_species):
            parameter_key = _resolve_cloud_parameter_key(parameter, cloud_name)

            if parameter_key is None:
                missing_species.append(cloud_name)
                continue

            parameter_grid = parameter_grid.at[:, i_spec].set(
                _expand_cloud_parameter_column(
                    parameter[parameter_key],
                    pressures=pressures,
                    parameter_name=parameter_name,
                    cloud_name=cloud_name,
                )
            )

        if len(missing_species) > 0:
            raise ValueError(
                f"{parameter_name} is missing values for cloud species: {', '.join(missing_species)}"
            )

        return parameter_grid

    parameter = jnp.asarray(parameter, dtype=FEAUTRIER_JAX_DTYPE)

    if parameter.ndim == 0:
        return jnp.full((n_layers, n_clouds), parameter, dtype=FEAUTRIER_JAX_DTYPE)

    if parameter.ndim == 1:
        if parameter.shape[0] == n_layers:
            return jnp.tile(parameter[:, None], (1, n_clouds))

        if parameter.shape[0] == n_clouds:
            return jnp.tile(parameter[None, :], (n_layers, 1))

    if parameter.ndim == 2:
        if parameter.shape == (n_layers, n_clouds):
            return parameter

        if parameter.shape == (n_clouds, n_layers):
            return parameter.T

    raise ValueError(
        f"{parameter_name} must be a scalar, a dict keyed by cloud species, or an array "
        f"with shape {(n_layers, n_clouds)}, but was of shape {parameter.shape}"
    )


def _filter_patchy_clouds(cloud_species, patchy_clouds):
    return tuple(species for species in cloud_species if species in patchy_clouds)


def _select_loaded_cloud_opacity_tables(
    clouds_loaded_opacities, cloud_species, selected_cloud_species
):
    if len(selected_cloud_species) == 0 or len(selected_cloud_species) == len(
        cloud_species
    ):
        return clouds_loaded_opacities

    selected_indices = tuple(
        i
        for i, species in enumerate(cloud_species)
        if species in selected_cloud_species
    )

    selected_loaded_opacities = LockedDict.build_and_lock(clouds_loaded_opacities)
    selected_loaded_opacities.unlock()
    selected_loaded_opacities["particle_densities"] = clouds_loaded_opacities[
        "particle_densities"
    ][jnp.array(selected_indices)]
    selected_loaded_opacities["absorption_opacities"] = clouds_loaded_opacities[
        "absorption_opacities"
    ][:, :, jnp.array(selected_indices)]
    selected_loaded_opacities["scattering_opacities"] = clouds_loaded_opacities[
        "scattering_opacities"
    ][:, :, jnp.array(selected_indices)]
    selected_loaded_opacities["particle_asymmetry_parameters"] = (
        clouds_loaded_opacities["particle_asymmetry_parameters"][
            :, :, jnp.array(selected_indices)
        ]
    )
    selected_loaded_opacities.lock()

    return selected_loaded_opacities


def _init_cloud_branch_outputs(frequencies, pressures, n_clouds):
    min_cloud_shape = max(n_clouds, 1)
    zero_2d = jnp.zeros((frequencies.size, pressures.size), dtype=FEAUTRIER_JAX_DTYPE)
    zero_radii = jnp.zeros((pressures.size, min_cloud_shape), dtype=FEAUTRIER_JAX_DTYPE)

    return zero_2d, zero_2d, zero_2d, zero_2d, zero_2d, zero_radii


def _maybe_add_opaque_cloud_deck(
    cloud_continuum_opacities, pressures, opaque_cloud_top_pressure
):
    # The transmission kernels later run in WORKING_JAX_DTYPE. Define the deck
    # in that working precision so the mixed-precision cast does not overflow to
    # inf before the shell-path integration.
    _OPAQUE_DECK_OPACITY = jnp.array(
        jnp.finfo(WORKING_JAX_DTYPE).max / 2,
        dtype=FEAUTRIER_JAX_DTYPE,
    )

    deck_active = jnp.greater_equal(opaque_cloud_top_pressure, 0.0)
    mask = jnp.greater(pressures, opaque_cloud_top_pressure * 1e6)  # bar to pascal
    deck = jnp.where(
        jnp.logical_and(deck_active, mask)[jnp.newaxis, :],
        _OPAQUE_DECK_OPACITY,
        0,
    )

    return cloud_continuum_opacities + deck


def _maybe_add_power_law_cloud_scattering(
    scattering_in_transmission,
    cloud_continuum_opacities_scattering,
    frequencies,
    pressures,
    power_law_opacity_350nm,
    power_law_opacity_coefficient,
):
    both_power_law_present = jnp.logical_and(
        jnp.not_equal(power_law_opacity_350nm, _MISSING_FLOAT),
        jnp.not_equal(power_law_opacity_coefficient, _MISSING_FLOAT),
    )

    def _add_powerlaw(args):
        _scattering_in_transmission, _cloud_continuum_opacities_scattering = args
        _scattering_in_transmission = True
        _cloud_continuum_opacities_scattering = (
            _cloud_continuum_opacities_scattering
            + compute_power_law_opacities(
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                frequencies=frequencies,
                n_layers=pressures.size,
            )
        )
        return _scattering_in_transmission, _cloud_continuum_opacities_scattering

    return lax.cond(
        both_power_law_present,
        _add_powerlaw,
        lambda args: args,
        (scattering_in_transmission, cloud_continuum_opacities_scattering),
    )


def _validate_power_law_inputs(power_law_opacity_350nm, power_law_opacity_coefficient):
    def _raise_incomplete_powerlaw(a, b):
        a_missing = bool(a == _MISSING_FLOAT)
        b_missing = bool(b == _MISSING_FLOAT)

        if a_missing ^ b_missing:
            none_parameter = (
                "power_law_opacity_350nm"
                if a_missing
                else "power_law_opacity_coefficient"
            )
            raise RuntimeError(
                f"to include a power law opacity, both parameters 'power_law_opacity_350nm' "
                f"and 'power_law_opacity_coefficient' must be provided; missing '{none_parameter}'"
            )

    debug.callback(
        _raise_incomplete_powerlaw,
        power_law_opacity_350nm,
        power_law_opacity_coefficient,
    )


def _init_cloud_background_continuum(
    frequencies,
    pressures,
    n_clouds,
    scattering_in_transmission,
    opaque_cloud_top_pressure,
    power_law_opacity_350nm,
    power_law_opacity_coefficient,
    compute_clear_opacities,
):
    # Convert None sentinels to traceable JAX arrays so that lax.cond and
    # _maybe_add_opaque_cloud_deck / _maybe_add_power_law_cloud_scattering
    # always receive array-like inputs.
    if opaque_cloud_top_pressure is None:
        opaque_cloud_top_pressure = _MISSING_FLOAT
    if power_law_opacity_350nm is None:
        power_law_opacity_350nm = _MISSING_FLOAT
    if power_law_opacity_coefficient is None:
        power_law_opacity_coefficient = _MISSING_FLOAT

    cloudy_continuum_opacities, cloudy_continuum_opacities_scattering, _, _, _, _ = (
        _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    )

    cloudy_continuum_opacities = _maybe_add_opaque_cloud_deck(
        cloud_continuum_opacities=cloudy_continuum_opacities,
        pressures=pressures,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
    )
    scattering_in_transmission, cloudy_continuum_opacities_scattering = (
        _maybe_add_power_law_cloud_scattering(
            scattering_in_transmission=scattering_in_transmission,
            cloud_continuum_opacities_scattering=cloudy_continuum_opacities_scattering,
            frequencies=frequencies,
            pressures=pressures,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
        )
    )

    if compute_clear_opacities:
        clear_continuum_opacities, clear_continuum_opacities_scattering, _, _, _, _ = (
            _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
        )
    else:
        clear_continuum_opacities = jnp.zeros(
            (frequencies.size, pressures.size), dtype=FEAUTRIER_JAX_DTYPE
        )
        clear_continuum_opacities_scattering = jnp.zeros(
            (frequencies.size, pressures.size), dtype=FEAUTRIER_JAX_DTYPE
        )

    _validate_power_law_inputs(power_law_opacity_350nm, power_law_opacity_coefficient)

    return (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        clear_continuum_opacities,
        clear_continuum_opacities_scattering,
        scattering_in_transmission,
    )


def _validate_selected_cloud_species(
    cloud_species_mass_fractions, selected_cloud_species, mass_fraction_keys
):
    def raise_error_for_inconsistent_cloud_species_length(
        _mass_fractions, _selected_cloud_species, _keys
    ):
        if len(_mass_fractions) != len(_selected_cloud_species):
            raise ValueError(
                f"the number of cloud species mass fractions ({len(_mass_fractions)}) "
                f"does not match the number of cloud species ({len(_selected_cloud_species)}); "
                f"check for inconsistencies between mass_fractions keys ({_keys}) "
                f"and cloud_species keys ({_selected_cloud_species})"
            )

    debug.callback(
        raise_error_for_inconsistent_cloud_species_length,
        cloud_species_mass_fractions,
        selected_cloud_species,
        mass_fraction_keys,
    )


def _compute_selected_cloud_branch(
    selected_cloud_species,
    continuum_seed,
    pressures,
    temperatures,
    frequency_bins_edges,
    mass_fractions,
    mean_molar_masses,
    reference_gravity,
    cloud_particle_radius_distribution_std,
    clouds_loaded_opacities,
    sum_opacities,
    anisotropic_cloud_scattering,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    cloud_particle_mean_radii,
    cloud_particle_number_density_grid,
    cloud_particle_radius_distribution,
    cloud_hansen_a,
    cloud_hansen_b,
    clouds_particle_porosity_factor,
    cloud_photosphere_median_optical_depth,
    return_cloud_contribution,
    target_particle_radii,
    target_pressure,
):
    cloud_species_mass_fractions = _build_cloud_species_mass_fractions(
        mass_fractions=mass_fractions, selected_cloud_species=selected_cloud_species
    )
    _validate_selected_cloud_species(
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        selected_cloud_species=selected_cloud_species,
        mass_fraction_keys=list(mass_fractions.keys()),
    )

    return _compute_cloud_opacities(
        pressures=pressures,
        temperatures=temperatures,
        frequency_bins_edges=frequency_bins_edges,
        cloud_species=selected_cloud_species,
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        clouds_loaded_opacities=clouds_loaded_opacities,
        sum_opacities=sum_opacities,
        anisotropic_cloud_scattering=anisotropic_cloud_scattering,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        cloud_particle_radius_distribution=cloud_particle_radius_distribution,
        cloud_hansen_a=cloud_hansen_a,
        cloud_hansen_b=cloud_hansen_b,
        clouds_particle_porosity_factor=clouds_particle_porosity_factor,
        photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
        continuum_opacities=continuum_seed,
        return_cloud_contribution=return_cloud_contribution,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )


def _maybe_compute_cloud_branch(
    selected_cloud_species,
    selected_clouds_have_effect,
    base_continuum_opacities,
    base_continuum_opacities_scattering,
    cloud_scattering_opacities,
    cloud_absorption_opacities,
    cloud_opacities,
    cloud_particle_mean_radii,
    pressures,
    temperatures,
    frequency_bins_edges,
    mass_fractions,
    mean_molar_masses,
    reference_gravity,
    cloud_particle_radius_distribution_std,
    clouds_loaded_opacities,
    sum_opacities,
    anisotropic_cloud_scattering,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    input_cloud_particle_mean_radii,
    cloud_particle_number_density_grid,
    cloud_particle_radius_distribution,
    cloud_hansen_a,
    cloud_hansen_b,
    clouds_particle_porosity_factor,
    cloud_photosphere_median_optical_depth,
    return_cloud_contribution,
    target_particle_radii,
    target_pressure,
):
    if len(selected_cloud_species) == 0:
        return (
            base_continuum_opacities,
            base_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            cloud_particle_mean_radii,
            False,
        )

    selected_cloud_particle_mean_radii = jnp.zeros(
        (pressures.size, max(len(selected_cloud_species), 1)),
        dtype=cloud_particle_mean_radii.dtype,
    )

    def _compute_branch(args):
        (
            _base_continuum_opacities,
            _base_continuum_opacities_scattering,
            _cloud_scattering_opacities,
            _cloud_absorption_opacities,
            _cloud_opacities,
            _cloud_particle_mean_radii,
        ) = args

        (
            _base_continuum_opacities,
            branch_continuum_scattering,
            _cloud_scattering_opacities,
            _cloud_absorption_opacities,
            _cloud_opacities,
            _cloud_particle_mean_radii,
        ) = _compute_selected_cloud_branch(
            selected_cloud_species=selected_cloud_species,
            continuum_seed=_base_continuum_opacities,
            pressures=pressures,
            temperatures=temperatures,
            frequency_bins_edges=frequency_bins_edges,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            clouds_loaded_opacities=clouds_loaded_opacities,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_particle_mean_radii=input_cloud_particle_mean_radii,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            return_cloud_contribution=return_cloud_contribution,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )
        _base_continuum_opacities_scattering += branch_continuum_scattering

        return (
            _base_continuum_opacities,
            _base_continuum_opacities_scattering,
            _cloud_scattering_opacities,
            _cloud_absorption_opacities,
            _cloud_opacities,
            _cloud_particle_mean_radii,
        )

    def _cast_branch_outputs(args):
        return (
            jnp.asarray(args[0], dtype=FEAUTRIER_JAX_DTYPE),
            jnp.asarray(args[1], dtype=FEAUTRIER_JAX_DTYPE),
            jnp.asarray(args[2], dtype=FEAUTRIER_JAX_DTYPE),
            jnp.asarray(args[3], dtype=FEAUTRIER_JAX_DTYPE),
            jnp.asarray(args[4], dtype=FEAUTRIER_JAX_DTYPE),
            jnp.asarray(args[5], dtype=FEAUTRIER_JAX_DTYPE),
        )

    (
        base_continuum_opacities,
        base_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        cloud_particle_mean_radii,
    ) = lax.cond(
        selected_clouds_have_effect,
        lambda args: _cast_branch_outputs(_compute_branch(args)),
        lambda args: _cast_branch_outputs(
            (
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                selected_cloud_particle_mean_radii,
            )
        ),
        (
            base_continuum_opacities,
            base_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            cloud_particle_mean_radii,
        ),
    )

    return (
        base_continuum_opacities,
        base_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        cloud_particle_mean_radii,
        selected_clouds_have_effect,
    )


def _initialize_hansen_b(
    cloud_hansen_b,
    cloud_particle_radius_distribution,
    cloud_species,
    pressures,
):
    if "hansen" not in cloud_particle_radius_distribution.lower():
        return cloud_hansen_b
    if cloud_hansen_b is None:
        raise ValueError(
            "The Hansen distribution width (cloud_hansen_b) must be set when "
            "using the Hansen particle radius distribution"
        )

    return _standardize_cloud_species_parameter(
        parameter=cloud_hansen_b,
        cloud_species=cloud_species,
        pressures=pressures,
        parameter_name="cloud_hansen_b",
    )


def _initialize_cloud_species_inputs(
    cloud_species,
    cloud_species_mass_fractions,
    pressures,
    clouds_loaded_opacities,
    clouds_particle_porosity_factor,
    cloud_particle_mean_radii,
    cloud_hansen_a,
):
    n_clouds = len(cloud_species)
    min_cloud_shape = max(n_clouds, 1)
    species_mass_fractions = jnp.zeros(
        (pressures.size, min_cloud_shape), dtype=FEAUTRIER_JAX_DTYPE
    )
    cloud_particle_mean_radii_grid = jnp.zeros(
        (pressures.size, min_cloud_shape), dtype=FEAUTRIER_JAX_DTYPE
    )
    cloud_particle_density = jnp.zeros(n_clouds, dtype=FEAUTRIER_JAX_DTYPE)

    # Iterate in explicit cloud_species order to keep i_spec aligned with
    # clouds_loaded_opacities tables. Dict iteration order may differ under JAX tracing.
    for i_spec, cloud_name in enumerate(cloud_species):
        if cloud_name not in cloud_species_mass_fractions:
            continue

        species_mass_fractions = species_mass_fractions.at[:, i_spec].set(
            cloud_species_mass_fractions[cloud_name]
        )

        if clouds_particle_porosity_factor is not None:
            cloud_particle_density = cloud_particle_density.at[i_spec].set(
                clouds_loaded_opacities["particle_densities"][i_spec]
                * (1 - clouds_particle_porosity_factor[cloud_name])
            )
        else:
            cloud_particle_density = cloud_particle_density.at[i_spec].set(
                clouds_loaded_opacities["particle_densities"][i_spec]
            )

    radii_source = cloud_particle_mean_radii
    radii_parameter_name = "cloud_particle_mean_radii"

    if radii_source is None and cloud_hansen_a is not None:
        radii_source = cloud_hansen_a
        radii_parameter_name = "cloud_hansen_a"

    if radii_source is not None:
        cloud_particle_mean_radii_grid = _standardize_cloud_species_parameter(
            parameter=radii_source,
            cloud_species=cloud_species,
            pressures=pressures,
            parameter_name=radii_parameter_name,
        )

    return (
        species_mass_fractions,
        cloud_particle_density,
        cloud_particle_mean_radii_grid,
    )


def _standardize_f_sed(cloud_f_sed, cloud_species, pressures_size):
    if isinstance(cloud_f_sed, dict) and len(cloud_f_sed) == 0:
        cloud_f_sed = None  # treat empty dict as no per-species f_sed
    f_seds = jnp.zeros((pressures_size, len(cloud_species)), dtype=FEAUTRIER_JAX_DTYPE)

    for i_spec, cloud in enumerate(cloud_species):
        if isinstance(cloud_f_sed, dict):
            f_seds = f_seds.at[:, i_spec].set(cloud_f_sed[cloud])
        elif isinstance(cloud_f_sed, jnp.ndarray):
            f_seds = f_seds.at[:, i_spec].set(cloud_f_sed)
        elif isinstance(cloud_f_sed, float):
            f_seds = f_seds.at[:, i_spec].set(
                jnp.ones_like(pressures_size) * cloud_f_sed
            )
        elif cloud_f_sed is None or cloud_f_sed is _MISSING_FLOAT:
            f_seds = f_seds.at[:, i_spec].set(jnp.zeros_like(pressures_size))

    return f_seds


def _compute_cloud_opacity_grids_from_known_radii(
    cloud_particle_radius_distribution,
    atmospheric_densities,
    cloud_particle_density,
    cloud_species_mass_fractions_grid,
    cloud_particle_mean_radii,
    cloud_hansen_b,
    cloud_particle_radius_distribution_std,
    cloud_particle_number_density_grid,
    clouds_loaded_opacities,
    cloud_species_mass_fractions,
    pressures,
    target_particle_radii,
    target_pressure,
):
    if cloud_particle_radius_distribution == "lognormal":
        return cloud_core.compute_cloud_log_normal_particle_distribution_opacities_grid(
            atmosphere_densities=atmospheric_densities,
            clouds_particle_densities=cloud_particle_density,
            clouds_mass_fractions=cloud_species_mass_fractions_grid,
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_particle_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_radii_bins=clouds_loaded_opacities["particle_radii_bins"],
            cloud_particle_radii=clouds_loaded_opacities["particle_radii"],
            clouds_absorption_opacities=clouds_loaded_opacities["absorption_opacities"],
            clouds_scattering_opacities=clouds_loaded_opacities["scattering_opacities"],
            clouds_particle_asymmetry_parameters=clouds_loaded_opacities[
                "particle_asymmetry_parameters"
            ],
            pressures=pressures,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )

    if cloud_particle_radius_distribution == "custom":
        return cloud_core._compute_cloud_opacities_using_particle_size_grid(
            atmosphere_densities=atmospheric_densities,
            clouds_mass_fractions=cloud_species_mass_fractions,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_particle_densities=cloud_particle_density,
            cloud_particle_radii_bins=clouds_loaded_opacities["particle_radii_bins"],
            cloud_particle_radii=clouds_loaded_opacities["particle_radii"],
            clouds_absorption_opacities=clouds_loaded_opacities["absorption_opacities"],
            clouds_scattering_opacities=clouds_loaded_opacities["scattering_opacities"],
            clouds_particle_asymmetry_parameters=clouds_loaded_opacities[
                "particle_asymmetry_parameters"
            ],
            pressures=pressures,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )

    return cloud_core.compute_cloud_hansen_opacities(
        atmospheric_densities,
        cloud_particle_density,
        cloud_species_mass_fractions_grid,
        cloud_particle_mean_radii,
        cloud_hansen_b,
        cloud_particle_radii_bins=clouds_loaded_opacities["particle_radii_bins"],
        cloud_particle_radii=clouds_loaded_opacities["particle_radii"],
        clouds_absorption_opacities=clouds_loaded_opacities["absorption_opacities"],
        clouds_scattering_opacities=clouds_loaded_opacities["scattering_opacities"],
        clouds_particle_asymmetry_parameters=clouds_loaded_opacities[
            "particle_asymmetry_parameters"
        ],
    )


def _validate_missing_cloud_radius_arguments(
    cloud_particle_radius_distribution_std,
    cloud_particle_radius_distribution,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    clouds_loaded_opacities,
    cloud_particle_number_density_grid,
):
    missing_arguments = []

    if cloud_particle_radius_distribution == "lognormal" and (
        cloud_particle_radius_distribution_std is None
        or cloud_particle_radius_distribution_std is _MISSING_FLOAT
    ):
        missing_arguments.append("'cloud_particle_radius_distribution_std'")

    if cloud_f_sed is None or cloud_f_sed is _MISSING_FLOAT:
        missing_arguments.append("'cloud_f_sed'")

    if (
        eddy_diffusion_coefficients is None
        or eddy_diffusion_coefficients is _MISSING_FLOAT
    ):
        missing_arguments.append("'eddy_diffusion_coefficients'")

    if clouds_loaded_opacities["particle_radii_bins"] is None:
        missing_arguments.append("'clouds_loaded_opacities.particle_radii_bins'")

    if (
        cloud_particle_radius_distribution == "custom"
        and cloud_particle_number_density_grid is None
    ):
        missing_arguments.append("'cloud_particle_number_density_grid'")

    def raise_missing_arguments_error(_missing_arguments):
        if len(_missing_arguments) > 0:
            raise ValueError(
                f"missing necessary arguments to calculate cloud particle radii: {', '.join(_missing_arguments)} "
                f"(got 'None')\n"
                f"Set the missing arguments, "
                f"or directly set cloud particle radii "
                f"through the arguments 'cloud_particle_mean_radii' or 'cloud_hansen_a', "
                f"and argument 'cloud_particle_radius_distribution_std'.\n"
                f"The cloud particle radii are necessary to calculate the cloud opacities."
            )

    debug.callback(raise_missing_arguments_error, missing_arguments)


def _compute_cloud_opacity_grids_with_derived_radii(
    cloud_particle_radius_distribution,
    reference_gravity,
    atmospheric_densities,
    cloud_particle_density,
    temperatures,
    mean_molar_masses,
    cloud_f_sed,
    cloud_species,
    pressures,
    cloud_particle_radius_distribution_std,
    eddy_diffusion_coefficients,
    cloud_hansen_b,
    cloud_species_mass_fractions_grid,
    cloud_species_mass_fractions,
    cloud_particle_number_density_grid,
    clouds_loaded_opacities,
    target_particle_radii,
    target_pressure,
):
    f_seds = _standardize_f_sed(cloud_f_sed, list(cloud_species), pressures.size)

    if cloud_particle_radius_distribution == "lognormal":
        cloud_particle_mean_radii = cloud_core.compute_cloud_particle_mean_radius(
            reference_gravity,
            atmospheric_densities,
            cloud_particle_density,
            temperatures,
            mean_molar_masses,
            f_seds,
            cloud_particle_radius_distribution_std,
            eddy_diffusion_coefficients,
        )
    elif cloud_particle_radius_distribution == "hansen":
        cloud_particle_mean_radii = (
            cloud_core.compute_cloud_particle_mean_radius_hansen(
                reference_gravity,
                atmospheric_densities,
                cloud_particle_density,
                temperatures,
                mean_molar_masses,
                f_seds,
                cloud_hansen_b,
                eddy_diffusion_coefficients,
            )
        )
    else:
        cloud_particle_mean_radii = jnp.zeros_like(cloud_species_mass_fractions_grid)

    cloud_opacity_grids = _compute_cloud_opacity_grids_from_known_radii(
        cloud_particle_radius_distribution=cloud_particle_radius_distribution,
        atmospheric_densities=atmospheric_densities,
        cloud_particle_density=cloud_particle_density,
        cloud_species_mass_fractions_grid=cloud_species_mass_fractions_grid,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        clouds_loaded_opacities=clouds_loaded_opacities,
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        pressures=pressures,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )

    return (*cloud_opacity_grids, cloud_particle_mean_radii)


def _assemble_cloud_opacity_outputs(
    clouds_total_absorption_opacities,
    clouds_total_scattering_opacities,
    cloud_scattering_reduction_factor,
    clouds_loaded_opacities,
    frequency_bins_edges,
    anisotropic_cloud_scattering,
    sum_opacities,
    photospheric_cloud_optical_depths,
    continuum_opacities,
):
    (
        cloud_final_absorption_opacities,
        cloud_anisotropic_extinctions,
        cloud_scattering_reduction_factor,
        cloud_isotropic_extinctions,
    ) = cloud_core.interpolate_cloud_opacities(
        clouds_total_absorption_opacities,
        clouds_total_scattering_opacities,
        cloud_scattering_reduction_factor,
        clouds_loaded_opacities["wavelengths"],
        frequency_bins_edges,
    )

    continuum_opacities_scattering = lax.cond(
        anisotropic_cloud_scattering,
        lambda: cloud_anisotropic_extinctions - cloud_final_absorption_opacities,
        lambda: cloud_isotropic_extinctions - cloud_final_absorption_opacities,
    )

    def sum_cloud_absorption_opacities(
        _cloud_anisotropic_extinctions, _cloud_final_absorption_opacities
    ):
        cloud_anisotropic_scattering_opacities = (
            _cloud_anisotropic_extinctions - _cloud_final_absorption_opacities
        )
        cloud_absorption_opacities = _cloud_final_absorption_opacities
        return cloud_anisotropic_scattering_opacities, cloud_absorption_opacities

    cloud_anisotropic_scattering_opacities, cloud_absorption_opacities = lax.cond(
        jnp.logical_and(
            sum_opacities, jnp.logical_not(photospheric_cloud_optical_depths is None)
        ),
        sum_cloud_absorption_opacities,
        lambda a, b: (jnp.zeros_like(a), jnp.zeros_like(b)),
        cloud_anisotropic_extinctions,
        cloud_final_absorption_opacities,
    )

    continuum_opacities = continuum_opacities + cloud_final_absorption_opacities
    cloud_opacities = cloud_anisotropic_extinctions

    return (
        continuum_opacities,
        continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
    )


@jit(static_argnames=["cloud_species"])
def init_clouds_particle_porosity_factor(
    clouds_particle_porosity_factor, cloud_species
):
    """Normalize and validate per-species cloud porosity factors.

    The input mapping is adjusted to the defaults expected by each cloud
    opacity source. DHS-based tables default to a porosity of ``0.25`` when no
    finite value is supplied, while non-DHS sources default to ``0.0``. The
    function also validates that all resulting porosity factors lie in the
    interval $[0, 1)$.

    Args:
        clouds_particle_porosity_factor:
            Mapping from cloud species name to porosity factor.
        cloud_species:
            Cloud species included in the current calculation.

    Returns:
        Updated porosity-factor mapping with defaults applied.
    """

    for species in cloud_species:
        _, _, _, _, source, _ = Opacity.split_species_all_info(
            CloudOpacity.get_aliases(species)
        )

        # Operate on a local variable so we can use lax.cond and jittable jnp ops
        _porosity = jnp.asarray(
            clouds_particle_porosity_factor.get(
                species,
                jnp.array(jnp.nan, dtype=FEAUTRIER_JAX_DTYPE),
            ),
            dtype=FEAUTRIER_JAX_DTYPE,
        )

        # For DHS sources: if porosity == 0 -> print a warning (leave value unchanged),
        # elif not finite -> set to 0.25
        def _handle_dhs(porosity):
            return lax.cond(
                jnp.equal(porosity, 0),
                # true: print warning and return original p
                lambda q: (
                    debug.print(
                        f"cloud species '{species}' opacity calculation method is using the DHS method, "
                        f"the porosity factor should be > 0 (0.25 is used by default)"
                    ),
                    q,
                )[1],
                # false: if finite -> return p else -> set to default 0.25
                lambda q: lax.cond(
                    jnp.isfinite(q),
                    lambda r: r,
                    lambda r: jnp.array(0.25, dtype=FEAUTRIER_JAX_DTYPE),
                    q,
                ),
                porosity,
            )

        # For non-DHS sources: if not finite -> set to 0.0, else keep value
        def _handle_non_dhs(porosity):
            return lax.cond(
                jnp.isfinite(porosity),
                lambda r: r,
                lambda r: jnp.array(0.0, dtype=FEAUTRIER_JAX_DTYPE),
                porosity,
            )

        # Choose handler based on source (source is static Python string)
        _porosity = lax.cond(
            source == "DHS",
            lambda p: _handle_dhs(p),
            lambda p: _handle_non_dhs(p),
            _porosity,
        )

        # Store back the (possibly updated) porosity
        clouds_particle_porosity_factor[species] = _porosity

        # Validate value with jnp logicals, raise via debug.callback so it works under jitting/tracing
        def _raise_if_invalid(porosity, species_name):
            if porosity < 0 or porosity >= 1:
                raise ValueError(
                    f"clouds particles porosity factor "
                    f"must be >= 0 (full particles) and < 1 (empty particles), "
                    f"but is {porosity} for species '{species_name}'"
                )

        # Only call the callback when invalid is True (callback will run during tracing/execution)
        debug.callback(_raise_if_invalid, _porosity, species)

    return clouds_particle_porosity_factor


@jit(
    static_argnames=[
        "cloud_species",
        "cloud_particle_radius_distribution",
        "patchy_clouds",
        "compute_clear_opacities",
    ]
)
def compute_cloud_continuum_opacities(
    frequencies: ArrayLike,
    frequency_bins_edges: ArrayLike,
    pressures: ArrayLike,
    temperatures: ArrayLike,
    mass_fractions: dict[str, ArrayLike],
    mean_molar_masses: ArrayLike,
    reference_gravity: float,
    scattering_in_transmission: bool,
    sum_opacities: bool,
    opaque_cloud_top_pressure: float,
    power_law_opacity_350nm: float,
    power_law_opacity_coefficient: float,
    cloud_species: list[str] | tuple[str],
    clouds_particle_porosity_factor: dict[str, float],
    clouds_loaded_opacities: LockedDict[str, ArrayLike],
    anisotropic_cloud_scattering: bool,
    cloud_particle_mean_radii: ArrayLike,
    cloud_particle_radius_distribution_std: float,
    cloud_particle_number_density_grid: ArrayLike,
    cloud_f_sed: dict[str, float],
    eddy_diffusion_coefficients: dict[str, float],
    cloud_particle_radius_distribution: str,
    cloud_hansen_a: float,
    cloud_hansen_b: float,
    cloud_photosphere_median_optical_depth: float,
    patchy_clouds: tuple[str],
    return_cloud_contribution: bool,
    compute_clear_opacities,
    target_particle_radii,
    target_pressure,
):
    """Compute cloud continuum opacity products for cloudy and optional clear branches.

    This is the main cloud entry point used by ``Radtrans``. It builds the
    cloud absorption and scattering continuum contributions, evaluates the
    selected cloud species, and optionally computes a second branch containing
    only the fully covering clouds for partial-cloud post-processing.

    Args:
        frequencies:
            Frequency grid used for continuum outputs.
        frequency_bins_edges:
            Frequency-bin edges used for cloud opacity interpolation.
        pressures:
            Atmospheric pressure grid in cgs units.
        temperatures:
            Atmospheric temperature profile in K.
        mass_fractions:
            Species mass-fraction mapping for the current atmosphere.
        mean_molar_masses:
            Mean molecular mass profile in amu.
        reference_gravity:
            Reference gravity in cgs units.
        scattering_in_transmission:
            Whether scattering is already active in transmission before adding
            cloud contributions.
        sum_opacities:
            Whether cloud opacities should be collapsed into a single absorber
            channel.
        opaque_cloud_top_pressure:
            Optional pressure in bar for an opaque cloud deck.
        power_law_opacity_350nm:
            Optional reference opacity for the power-law cloud scattering model.
        power_law_opacity_coefficient:
            Wavelength exponent for the power-law cloud scattering model.
        cloud_species:
            Cloud species to include in the cloudy branch.
        clouds_particle_porosity_factor:
            Mapping of cloud species to particle porosity factors.
        clouds_loaded_opacities:
            Loaded cloud opacity tables and metadata.
        anisotropic_cloud_scattering:
            If ``True``, use anisotropic scattering when forming continuum
            scattering opacities.
        cloud_particle_mean_radii:
            Optional prescribed particle mean radii.
        cloud_particle_radius_distribution_std:
            Width of the log-normal particle size distribution.
        cloud_particle_number_density_grid:
            Optional custom particle number density grid.
        cloud_f_sed:
            Cloud settling parameter, scalar or per-species mapping.
        eddy_diffusion_coefficients:
            Eddy diffusion coefficient profile or mapping.
        cloud_particle_radius_distribution:
            Particle size distribution model, such as ``"lognormal"``,
            ``"hansen"``, or ``"custom"``.
        cloud_hansen_a:
            Optional Hansen ``a`` parameter or equivalent mean-radius input.
        cloud_hansen_b:
            Optional Hansen ``b`` width parameter.
        cloud_photosphere_median_optical_depth:
            Optional target cloud optical depth near the photosphere.
        patchy_clouds:
            Cloud species affected by ``cloud_fraction`` and therefore
            excluded from the clear branch.
        return_cloud_contribution:
            Whether to propagate cloud-only products required by contribution
            diagnostics.
        compute_clear_opacities:
            If ``True``, also compute the clear branch.
        target_particle_radii:
            Optional target particle radii grid for cloud opacity evaluation.
        target_pressure:
            Optional pressure grid associated with ``target_particle_radii``.

    Returns:
        Tuple containing cloudy continuum absorption and scattering opacities,
        cloud-only opacity diagnostics, particle radii, clear-branch cloud
        products, and the updated transmission-scattering flag.
    """
    n_clouds = len(cloud_species)
    (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        _cloud_particle_mean_radii,
    ) = _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    (
        clear_continuum_opacities,
        clear_continuum_opacities_scattering,
        clear_cloud_scattering_opacities,
        clear_cloud_absorption_opacities,
        _,
        _,
    ) = _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    patchy_clouds = _filter_patchy_clouds(cloud_species, patchy_clouds)
    complete_coverage_clouds = tuple(
        species for species in cloud_species if species not in patchy_clouds
    )

    (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        clear_continuum_opacities,
        clear_continuum_opacities_scattering,
        scattering_in_transmission,
    ) = _init_cloud_background_continuum(
        frequencies=frequencies,
        pressures=pressures,
        n_clouds=n_clouds,
        scattering_in_transmission=scattering_in_transmission,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        power_law_opacity_350nm=power_law_opacity_350nm,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        compute_clear_opacities=compute_clear_opacities,
    )

    if len(cloud_species) > 0:
        if clouds_particle_porosity_factor is None:
            clouds_particle_porosity_factor = {}
        clouds_particle_porosity_factor = init_clouds_particle_porosity_factor(
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_species=cloud_species,
        )

    cloudy_clouds_have_effect = _selected_clouds_have_effect(
        mass_fractions=mass_fractions, selected_cloud_species=cloud_species
    )
    cloudy_loaded_opacities = _select_loaded_cloud_opacity_tables(
        clouds_loaded_opacities=clouds_loaded_opacities,
        cloud_species=cloud_species,
        selected_cloud_species=cloud_species,
    )

    (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        _cloud_particle_mean_radii,
        cloudy_branch_has_effect,
    ) = _maybe_compute_cloud_branch(
        selected_cloud_species=cloud_species,
        selected_clouds_have_effect=cloudy_clouds_have_effect,
        base_continuum_opacities=cloudy_continuum_opacities,
        base_continuum_opacities_scattering=cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities=cloud_scattering_opacities,
        cloud_absorption_opacities=cloud_absorption_opacities,
        cloud_opacities=cloud_opacities,
        cloud_particle_mean_radii=_cloud_particle_mean_radii,
        pressures=pressures,
        temperatures=temperatures,
        frequency_bins_edges=frequency_bins_edges,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        clouds_loaded_opacities=cloudy_loaded_opacities,
        sum_opacities=sum_opacities,
        anisotropic_cloud_scattering=anisotropic_cloud_scattering,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        input_cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        cloud_particle_radius_distribution=cloud_particle_radius_distribution,
        cloud_hansen_a=cloud_hansen_a,
        cloud_hansen_b=cloud_hansen_b,
        clouds_particle_porosity_factor=clouds_particle_porosity_factor,
        cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
        return_cloud_contribution=return_cloud_contribution,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )
    scattering_in_transmission = jnp.logical_or(
        scattering_in_transmission, cloudy_branch_has_effect
    )

    if compute_clear_opacities:
        clear_clouds_have_effect = _selected_clouds_have_effect(
            mass_fractions=mass_fractions,
            selected_cloud_species=complete_coverage_clouds,
        )
        clear_loaded_opacities = _select_loaded_cloud_opacity_tables(
            clouds_loaded_opacities=clouds_loaded_opacities,
            cloud_species=cloud_species,
            selected_cloud_species=complete_coverage_clouds,
        )

        (
            clear_continuum_opacities,
            clear_continuum_opacities_scattering,
            clear_cloud_scattering_opacities,
            clear_cloud_absorption_opacities,
            _,
            _,
            clear_branch_has_effect,
        ) = _maybe_compute_cloud_branch(
            selected_cloud_species=complete_coverage_clouds,
            selected_clouds_have_effect=clear_clouds_have_effect,
            base_continuum_opacities=clear_continuum_opacities,
            base_continuum_opacities_scattering=clear_continuum_opacities_scattering,
            cloud_scattering_opacities=clear_cloud_scattering_opacities,
            cloud_absorption_opacities=clear_cloud_absorption_opacities,
            cloud_opacities=cloud_opacities,
            cloud_particle_mean_radii=_cloud_particle_mean_radii,
            pressures=pressures,
            temperatures=temperatures,
            frequency_bins_edges=frequency_bins_edges,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            clouds_loaded_opacities=clear_loaded_opacities,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            input_cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            return_cloud_contribution=return_cloud_contribution,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )
        scattering_in_transmission = jnp.logical_or(
            scattering_in_transmission, clear_branch_has_effect
        )

    return (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        _cloud_particle_mean_radii,
        clear_continuum_opacities,
        clear_continuum_opacities_scattering,
        clear_cloud_scattering_opacities,
        clear_cloud_absorption_opacities,
        scattering_in_transmission,
    )


@jit(static_argnames=["cloud_species", "cloud_particle_radius_distribution"])
def _compute_cloud_opacities(
    pressures,
    temperatures,
    cloud_species,
    frequency_bins_edges,
    cloud_species_mass_fractions,
    mean_molar_masses,
    reference_gravity,
    cloud_particle_radius_distribution_std,
    clouds_loaded_opacities,
    sum_opacities,
    anisotropic_cloud_scattering,
    cloud_f_sed=None,
    eddy_diffusion_coefficients=None,
    cloud_particle_mean_radii=None,
    cloud_particle_radius_distribution="lognormal",
    cloud_hansen_a=None,
    cloud_hansen_b=None,
    cloud_particle_number_density_grid=None,
    clouds_particle_porosity_factor=None,
    photospheric_cloud_optical_depths=None,
    continuum_opacities=None,
    return_cloud_contribution=False,
    target_particle_radii=None,
    target_pressure=None,
):
    """Calculate cloud opacity terms for a single cloud branch.

    The function prepares cloud particle properties, evaluates the selected
    particle size distribution against the loaded cloud tables, interpolates the
    resulting opacities onto the model frequency grid, and separates the final
    absorption and scattering products used by the radiative-transfer solvers.

    Args:
        pressures:
            Atmospheric pressure grid in cgs units.
        temperatures:
            Atmospheric temperature profile in K.
        cloud_species:
            Cloud species included in this branch.
        frequency_bins_edges:
            Frequency-bin edges for interpolation of cloud opacity tables.
        cloud_species_mass_fractions:
            Mapping of cloud species to mass-fraction profiles.
        mean_molar_masses:
            Mean molecular mass profile in amu.
        reference_gravity:
            Reference gravity in cgs units.
        cloud_particle_radius_distribution_std:
            Width of the log-normal particle size distribution.
        clouds_loaded_opacities:
            Loaded cloud opacity tables and metadata.
        sum_opacities:
            Whether cloud opacity outputs should be collapsed into a single
            absorber channel.
        anisotropic_cloud_scattering:
            Whether to use anisotropic scattering in the continuum output.
        cloud_f_sed:
            Cloud settling parameter used when radii must be derived.
        eddy_diffusion_coefficients:
            Eddy diffusion coefficient profile or mapping used for derived
            particle radii.
        cloud_particle_mean_radii:
            Optional prescribed cloud particle mean radii.
        cloud_particle_radius_distribution:
            Particle size distribution model.
        cloud_hansen_a:
            Optional Hansen ``a`` parameter or equivalent mean-radius input.
        cloud_hansen_b:
            Optional Hansen ``b`` width parameter.
        cloud_particle_number_density_grid:
            Optional custom particle number density grid.
        clouds_particle_porosity_factor:
            Optional mapping of cloud species to particle porosity factors.
        photospheric_cloud_optical_depths:
            Optional target cloud optical depth near the photosphere.
        continuum_opacities:
            Seed continuum absorption opacities to which cloud absorption is
            added.
        return_cloud_contribution:
            Whether to retain cloud-only products needed for contribution
            diagnostics.
        target_particle_radii:
            Optional target particle radii grid for cloud opacity evaluation.
        target_pressure:
            Optional pressure grid associated with ``target_particle_radii``.

    Returns:
        Tuple ``(continuum_opacities, continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities, cloud_absorption_opacities,
        cloud_opacities, cloud_particle_mean_radii)``.
    """
    atmospheric_densities = (
        pressures * mean_molar_masses * cst.amu / cst.kB / temperatures
    )

    if continuum_opacities is None:
        continuum_opacities = jnp.zeros(
            (frequency_bins_edges.size - 1, pressures.size), dtype=FEAUTRIER_JAX_DTYPE
        )

    cloud_hansen_b = _initialize_hansen_b(
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_radius_distribution=cloud_particle_radius_distribution,
        cloud_species=cloud_species,
        pressures=pressures,
    )
    (
        _cloud_species_mass_fractions,
        _cloud_particle_density,
        _cloud_particle_mean_radii,
    ) = _initialize_cloud_species_inputs(
        cloud_species=cloud_species,
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        pressures=pressures,
        clouds_loaded_opacities=clouds_loaded_opacities,
        clouds_particle_porosity_factor=clouds_particle_porosity_factor,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_hansen_a=cloud_hansen_a,
    )
    cloud_particle_radius_distribution_std = _standardize_cloud_species_parameter(
        parameter=cloud_particle_radius_distribution_std,
        cloud_species=cloud_species,
        pressures=pressures,
        parameter_name="cloud_particle_radius_distribution_std",
    )

    if cloud_particle_mean_radii is not None or cloud_hansen_a is not None:
        (
            clouds_total_absorption_opacities,
            clouds_total_scattering_opacities,
            cloud_scattering_reduction_factor,
        ) = _compute_cloud_opacity_grids_from_known_radii(
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            atmospheric_densities=atmospheric_densities,
            cloud_particle_density=_cloud_particle_density,
            cloud_species_mass_fractions_grid=_cloud_species_mass_fractions,
            cloud_particle_mean_radii=_cloud_particle_mean_radii,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_loaded_opacities=clouds_loaded_opacities,
            cloud_species_mass_fractions=cloud_species_mass_fractions,
            pressures=pressures,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )
    else:
        _validate_missing_cloud_radius_arguments(
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            clouds_loaded_opacities=clouds_loaded_opacities,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        )
        # Provide traceable sentinel arrays for optional inputs that may be
        # None so that lax.cond can trace through this branch without error.
        # The _validate_missing_cloud_radius_arguments callback above will
        # still raise at runtime when these values are genuinely required.
        _eddy = (
            eddy_diffusion_coefficients
            if eddy_diffusion_coefficients is not None
            else jnp.zeros_like(pressures)
        )
        _cloud_f_sed = cloud_f_sed if cloud_f_sed is not None else {}
        _sigma = (
            cloud_particle_radius_distribution_std
            if cloud_particle_radius_distribution_std is not None
            and cloud_particle_radius_distribution_std is not _MISSING_FLOAT
            else jnp.array(1.05, dtype=FEAUTRIER_JAX_DTYPE)
        )
        (
            clouds_total_absorption_opacities,
            clouds_total_scattering_opacities,
            cloud_scattering_reduction_factor,
            _cloud_particle_mean_radii,
        ) = _compute_cloud_opacity_grids_with_derived_radii(
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            reference_gravity=reference_gravity,
            atmospheric_densities=atmospheric_densities,
            cloud_particle_density=_cloud_particle_density,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            cloud_f_sed=_cloud_f_sed,
            cloud_species=cloud_species,
            pressures=pressures,
            cloud_particle_radius_distribution_std=_sigma,
            eddy_diffusion_coefficients=_eddy,
            cloud_hansen_b=cloud_hansen_b,
            cloud_species_mass_fractions_grid=_cloud_species_mass_fractions,
            cloud_species_mass_fractions=cloud_species_mass_fractions,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_loaded_opacities=clouds_loaded_opacities,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )

    (
        continuum_opacities,
        continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
    ) = _assemble_cloud_opacity_outputs(
        clouds_total_absorption_opacities=clouds_total_absorption_opacities,
        clouds_total_scattering_opacities=clouds_total_scattering_opacities,
        cloud_scattering_reduction_factor=cloud_scattering_reduction_factor,
        clouds_loaded_opacities=clouds_loaded_opacities,
        frequency_bins_edges=frequency_bins_edges,
        anisotropic_cloud_scattering=anisotropic_cloud_scattering,
        sum_opacities=sum_opacities,
        photospheric_cloud_optical_depths=photospheric_cloud_optical_depths,
        continuum_opacities=continuum_opacities,
    )

    return (
        continuum_opacities,
        continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        _cloud_particle_mean_radii,
    )


def compute_species_optical_depths(reference_gravity, pressures, cloud_opacities):
    """Compute cloud optical depths from per-species cloud opacities.

    Optical depths are integrated from the top of the atmosphere downward using
    the atmospheric pressure grid and a vertically constant reference gravity.

    Args:
        reference_gravity:
            Surface gravity in cgs units.
        pressures:
            Atmospheric pressure grid in cgs units.
        cloud_opacities:
            Cloud opacity array with dimensions ``(..., n_layers)``.

    Returns:
        Cloud optical depths with the same shape as ``cloud_opacities``.
    """
    return radiative_transfer_core.compute_species_optical_depths(
        reference_gravity, pressures, cloud_opacities
    )
