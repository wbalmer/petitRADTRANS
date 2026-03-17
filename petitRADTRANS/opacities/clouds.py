import jax.numpy as jnp
from jax import lax, debug, jit, config
from jax.typing import ArrayLike
from petitRADTRANS.radtrans_core import radiative_transfer_core, cloud_core
from petitRADTRANS.opacities.opacities import CloudOpacity, Opacity
from petitRADTRANS.opacities.rayleigh import compute_power_law_opacities
import petitRADTRANS.physical_constants as cst

from petitRADTRANS.utils import LockedDict

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64

# JAX-friendly sentinels and helpers for optional inputs
_MISSING_FLOAT = jnp.array(-1.0, dtype=JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=JAX_DTYPE)


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


def _filter_complete_coverage_clouds(cloud_species, complete_coverage_clouds):
    return tuple(species for species in cloud_species if species in complete_coverage_clouds)


def _init_cloud_branch_outputs(frequencies, pressures, n_clouds):
    min_cloud_shape = max(n_clouds, 1)
    zero_2d = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    zero_radii = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)

    return zero_2d, zero_2d, zero_2d, zero_2d, zero_2d, zero_radii


def _maybe_add_opaque_cloud_deck(cloud_continuum_opacities, pressures, opaque_cloud_top_pressure):
    def _add_opaque_deck(_cloud_continuum_opacities, _pressures, _p_top):
        inds = jnp.where(jnp.greater_equal(_pressures, _p_top * 1e6), size=_pressures.shape[0])
        return _cloud_continuum_opacities.at[:, inds].add(1e99)

    return lax.cond(
        jnp.greater_equal(opaque_cloud_top_pressure, 0.0),
        _add_opaque_deck,
        lambda a, b, c: a,
        cloud_continuum_opacities,
        pressures,
        opaque_cloud_top_pressure
    )


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
        jnp.not_equal(power_law_opacity_coefficient, _MISSING_FLOAT)
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
                n_layers=pressures.size
            )
        )
        return _scattering_in_transmission, _cloud_continuum_opacities_scattering

    return lax.cond(
        both_power_law_present,
        _add_powerlaw,
        lambda args: args,
        (scattering_in_transmission, cloud_continuum_opacities_scattering)
    )


def _validate_power_law_inputs(power_law_opacity_350nm, power_law_opacity_coefficient):
    def _raise_incomplete_powerlaw(a, b):
        a_missing = bool(a == _MISSING_FLOAT)
        b_missing = bool(b == _MISSING_FLOAT)

        if a_missing ^ b_missing:
            none_parameter = 'power_law_opacity_350nm' if a_missing else 'power_law_opacity_coefficient'
            raise RuntimeError(
                f"to include a power law opacity, both parameters 'power_law_opacity_350nm' "
                f"and 'power_law_opacity_coefficient' must be provided; missing '{none_parameter}'"
            )

    debug.callback(_raise_incomplete_powerlaw, power_law_opacity_350nm, power_law_opacity_coefficient)


def _init_cloud_background_continuum(
    frequencies,
    pressures,
    n_clouds,
    scattering_in_transmission,
    opaque_cloud_top_pressure,
    power_law_opacity_350nm,
    power_law_opacity_coefficient,
):
    cloudy_continuum_opacities, cloudy_continuum_opacities_scattering, _, _, _, _ = (
        _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    )
    clear_continuum_opacities, clear_continuum_opacities_scattering, _, _, _, _ = (
        _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    )

    cloudy_continuum_opacities = _maybe_add_opaque_cloud_deck(
        cloud_continuum_opacities=cloudy_continuum_opacities,
        pressures=pressures,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure
    )
    clear_continuum_opacities = _maybe_add_opaque_cloud_deck(
        cloud_continuum_opacities=clear_continuum_opacities,
        pressures=pressures,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure
    )

    scattering_in_transmission, cloudy_continuum_opacities_scattering = _maybe_add_power_law_cloud_scattering(
        scattering_in_transmission=scattering_in_transmission,
        cloud_continuum_opacities_scattering=cloudy_continuum_opacities_scattering,
        frequencies=frequencies,
        pressures=pressures,
        power_law_opacity_350nm=power_law_opacity_350nm,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
    )
    _, clear_continuum_opacities_scattering = _maybe_add_power_law_cloud_scattering(
        scattering_in_transmission=scattering_in_transmission,
        cloud_continuum_opacities_scattering=clear_continuum_opacities_scattering,
        frequencies=frequencies,
        pressures=pressures,
        power_law_opacity_350nm=power_law_opacity_350nm,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
    )

    _validate_power_law_inputs(power_law_opacity_350nm, power_law_opacity_coefficient)

    return (
        cloudy_continuum_opacities,
        cloudy_continuum_opacities_scattering,
        clear_continuum_opacities,
        clear_continuum_opacities_scattering,
        scattering_in_transmission,
    )


def _validate_selected_cloud_species(cloud_species_mass_fractions, selected_cloud_species, mass_fraction_keys):
    def raise_error_for_inconsistent_cloud_species_length(_mass_fractions, _selected_cloud_species, _keys):
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
        mass_fraction_keys
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
    cloud_particles_mean_radii,
    cloud_particles_radius_distribution,
    cloud_hansen_a,
    cloud_hansen_b,
    clouds_particles_porosity_factor,
    cloud_photosphere_median_optical_depth,
    return_cloud_contribution,
    target_particle_radii,
    target_pressure,
):
    cloud_species_mass_fractions = _build_cloud_species_mass_fractions(
        mass_fractions=mass_fractions,
        selected_cloud_species=selected_cloud_species
    )
    _validate_selected_cloud_species(
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        selected_cloud_species=selected_cloud_species,
        mass_fraction_keys=list(mass_fractions.keys())
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
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_particles_radius_distribution=cloud_particles_radius_distribution,
        cloud_hansen_a=cloud_hansen_a,
        cloud_hansen_b=cloud_hansen_b,
        clouds_particles_porosity_factor=clouds_particles_porosity_factor,
        photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
        continuum_opacities=continuum_seed,
        return_cloud_contribution=return_cloud_contribution,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure
    )


def _maybe_compute_cloud_branch(
    selected_cloud_species,
    selected_clouds_have_effect,
    base_continuum_opacities,
    base_continuum_opacities_scattering,
    cloud_scattering_opacities,
    cloud_absorption_opacities,
    cloud_opacities,
    cloud_particles_mean_radii,
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
    input_cloud_particles_mean_radii,
    cloud_particles_radius_distribution,
    cloud_hansen_a,
    cloud_hansen_b,
    clouds_particles_porosity_factor,
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
            cloud_particles_mean_radii,
            False,
        )

    def _compute_branch(args):
        (_base_continuum_opacities, _base_continuum_opacities_scattering,
         _cloud_scattering_opacities, _cloud_absorption_opacities,
         _cloud_opacities, _cloud_particles_mean_radii) = args

        (_base_continuum_opacities, branch_continuum_scattering,
         _cloud_scattering_opacities, _cloud_absorption_opacities,
         _cloud_opacities, _cloud_particles_mean_radii) = _compute_selected_cloud_branch(
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
            cloud_particles_mean_radii=input_cloud_particles_mean_radii,
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
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
            _cloud_particles_mean_radii,
        )

    (base_continuum_opacities, base_continuum_opacities_scattering,
     cloud_scattering_opacities, cloud_absorption_opacities,
     cloud_opacities, cloud_particles_mean_radii) = lax.cond(
        selected_clouds_have_effect,
        _compute_branch,
        lambda args: args,
        (
            base_continuum_opacities,
            base_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            cloud_particles_mean_radii,
        )
    )

    return (
        base_continuum_opacities,
        base_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        cloud_particles_mean_radii,
        selected_clouds_have_effect,
    )


def _initialize_hansen_b(cloud_hansen_b, cloud_particles_radius_distribution, pressures, n_clouds):
    if "hansen" not in cloud_particles_radius_distribution.lower():
        return cloud_hansen_b

    if isinstance(cloud_hansen_b, jnp.ndarray):
        if cloud_hansen_b.shape != (pressures.size, n_clouds):
            raise ValueError(
                "cloud_hansen_b must be a float, a dictionary with arrays for each cloud species, "
                f"or an array with shape {(pressures.shape[0], n_clouds)}, "
                f"but was of shape {jnp.shape(cloud_hansen_b)}"
            )
        return cloud_hansen_b

    if isinstance(cloud_hansen_b, dict):
        return jnp.array(list(cloud_hansen_b.values()), dtype=JAX_DTYPE)

    if isinstance(cloud_hansen_b, float):
        return jnp.array(
            jnp.tile(cloud_hansen_b * jnp.ones_like(pressures), (n_clouds, 1)),
            dtype=JAX_DTYPE,
        ).T

    raise ValueError(
        f"The Hansen distribution width (cloud_hansen_b) must be an array, a dict, "
        f"or a float, but is of type '{type(cloud_hansen_b)}' ({cloud_hansen_b})"
    )


def _initialize_cloud_species_inputs(
    cloud_species,
    cloud_species_mass_fractions,
    pressures,
    clouds_loaded_opacities,
    clouds_particles_porosity_factor,
    cloud_particles_mean_radii,
    cloud_hansen_a,
):
    n_clouds = len(cloud_species)
    min_cloud_shape = max(n_clouds, 1)
    species_mass_fractions = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)
    cloud_particles_mean_radii_grid = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)
    cloud_particles_density = jnp.zeros(n_clouds, dtype=JAX_DTYPE)

    for i_spec, cloud_name in enumerate(cloud_species_mass_fractions):
        cloud_opacity = CloudOpacity([cloud_name], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_short_name = species_full_name.split('_')[0]

        species_mass_fractions = species_mass_fractions.at[:, i_spec].set(cloud_species_mass_fractions[cloud_name])

        if clouds_particles_porosity_factor is not None:
            cloud_particles_density = cloud_particles_density.at[i_spec].set(
                clouds_loaded_opacities['particles_densities'][i_spec]
                * (1 - clouds_particles_porosity_factor[cloud_name])
            )
        else:
            cloud_particles_density = cloud_particles_density.at[i_spec].set(
                clouds_loaded_opacities['particles_densities']
            )

        if cloud_particles_mean_radii is not None:
            if cloud_name in cloud_particles_mean_radii.keys():
                cloud_particles_mean_radii_grid = cloud_particles_mean_radii_grid.at[:, i_spec].set(
                    cloud_particles_mean_radii[cloud_name]
                )
            elif cloud_short_name in cloud_particles_mean_radii.keys():
                cloud_particles_mean_radii_grid = cloud_particles_mean_radii_grid.at[:, i_spec].set(
                    cloud_particles_mean_radii[cloud_short_name]
                )
        elif cloud_hansen_a is not None:
            if cloud_name in cloud_hansen_a.keys():
                cloud_particles_mean_radii_grid = cloud_particles_mean_radii_grid.at[:, i_spec].set(
                    cloud_hansen_a[cloud_name]
                )
            elif cloud_short_name in cloud_hansen_a.keys():
                cloud_particles_mean_radii_grid = cloud_particles_mean_radii_grid.at[:, i_spec].set(
                    cloud_hansen_a[cloud_short_name]
                )

    return species_mass_fractions, cloud_particles_density, cloud_particles_mean_radii_grid


def _standardize_f_sed(cloud_f_sed, cloud_species, pressures_size):
    f_seds = jnp.zeros((pressures_size, len(cloud_species)), dtype=JAX_DTYPE)

    for i_spec, cloud in enumerate(cloud_species):
        if isinstance(cloud_f_sed, dict):
            f_seds = f_seds.at[:, i_spec].set(cloud_f_sed[cloud])
        else:
            f_seds = f_seds.at[:, i_spec].set(cloud_f_sed)

    return f_seds


def _compute_cloud_opacity_grids_from_known_radii(
    cloud_particles_radius_distribution,
    atmospheric_densities,
    cloud_particles_density,
    cloud_species_mass_fractions_grid,
    cloud_particles_mean_radii,
    cloud_hansen_b,
    cloud_particle_radius_distribution_std,
    cloud_particle_number_density_grid,
    clouds_loaded_opacities,
    cloud_species_mass_fractions,
    pressures,
    target_particle_radii,
    target_pressure,
):
    if cloud_particles_radius_distribution == "lognormal":
        return cloud_core.compute_cloud_log_normal_particles_distribution_opacities_grid(
            atmosphere_densities=atmospheric_densities,
            clouds_particles_densities=cloud_particles_density,
            clouds_mass_fractions=cloud_species_mass_fractions_grid,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_particles_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
            cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
            clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
            clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
            clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
            pressures=pressures,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )

    if cloud_particles_radius_distribution == "custom":
        return cloud_core._compute_cloud_opacities_using_particle_size_grid(
            atmosphere_densities=atmospheric_densities,
            clouds_mass_fractions=cloud_species_mass_fractions,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_particles_densities=cloud_particles_density,
            cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
            cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
            clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
            clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
            clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
            pressures=pressures,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
        )

    return cloud_core.compute_cloud_hansen_opacities(
        atmospheric_densities,
        cloud_particles_density,
        cloud_species_mass_fractions_grid,
        cloud_particles_mean_radii,
        cloud_hansen_b,
        clouds_loaded_opacities['particles_radii_bins'],
        clouds_loaded_opacities['particles_radii'],
        clouds_loaded_opacities['absorption_opacities'],
        clouds_loaded_opacities['scattering_opacities'],
        clouds_loaded_opacities['particles_asymmetry_parameters'],
    )


def _validate_missing_cloud_radius_arguments(
    cloud_particle_radius_distribution_std,
    cloud_particles_radius_distribution,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    clouds_loaded_opacities,
    cloud_particle_number_density_grid,
):
    missing_arguments = []

    if (
        cloud_particles_radius_distribution == "lognormal"
        and (cloud_particle_radius_distribution_std is None or cloud_particle_radius_distribution_std is _MISSING_FLOAT)
    ):
        missing_arguments.append("'cloud_particle_radius_distribution_std'")

    if cloud_f_sed is None or cloud_f_sed is _MISSING_FLOAT:
        missing_arguments.append("'cloud_f_sed'")

    if eddy_diffusion_coefficients is None or eddy_diffusion_coefficients is _MISSING_FLOAT:
        missing_arguments.append("'eddy_diffusion_coefficients'")

    if clouds_loaded_opacities['particles_radii_bins'] is None:
        missing_arguments.append("'clouds_loaded_opacities.particles_radii_bins'")

    if cloud_particles_radius_distribution == "custom" and cloud_particle_number_density_grid is None:
        missing_arguments.append("'cloud_particle_number_density_grid'")

    def raise_missing_arguments_error(_missing_arguments):
        if len(_missing_arguments) > 0:
            raise ValueError(
                f"missing necessary arguments to calculate cloud particle radii: {', '.join(_missing_arguments)} "
                f"(got 'None')\n"
                f"Set the missing arguments, "
                f"or directly set cloud particle radii "
                f"through the arguments 'cloud_particles_mean_radii' or 'cloud_hansen_a', "
                f"and argument 'cloud_particle_radius_distribution_std'.\n"
                f"The cloud particle radii are necessary to calculate the cloud opacities."
            )

    debug.callback(raise_missing_arguments_error, missing_arguments)


def _compute_cloud_opacity_grids_with_derived_radii(
    cloud_particles_radius_distribution,
    reference_gravity,
    atmospheric_densities,
    cloud_particles_density,
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

    if cloud_particles_radius_distribution == "lognormal":
        cloud_particles_mean_radii = cloud_core.compute_cloud_particles_mean_radius(
            reference_gravity,
            atmospheric_densities,
            cloud_particles_density,
            temperatures,
            mean_molar_masses,
            f_seds,
            cloud_particle_radius_distribution_std,
            eddy_diffusion_coefficients,
        )
    elif cloud_particles_radius_distribution == "hansen":
        cloud_particles_mean_radii = cloud_core.compute_cloud_particles_mean_radius_hansen(
            reference_gravity,
            atmospheric_densities,
            cloud_particles_density,
            temperatures,
            mean_molar_masses,
            f_seds,
            cloud_hansen_b,
            eddy_diffusion_coefficients,
        )
    else:
        cloud_particles_mean_radii = jnp.zeros_like(cloud_species_mass_fractions_grid)

    cloud_opacity_grids = _compute_cloud_opacity_grids_from_known_radii(
        cloud_particles_radius_distribution=cloud_particles_radius_distribution,
        atmospheric_densities=atmospheric_densities,
        cloud_particles_density=cloud_particles_density,
        cloud_species_mass_fractions_grid=cloud_species_mass_fractions_grid,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        clouds_loaded_opacities=clouds_loaded_opacities,
        cloud_species_mass_fractions=cloud_species_mass_fractions,
        pressures=pressures,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )

    return (*cloud_opacity_grids, cloud_particles_mean_radii)


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
        clouds_loaded_opacities['wavelengths'],
        frequency_bins_edges,
    )

    continuum_opacities_scattering = lax.cond(
        anisotropic_cloud_scattering,
        lambda: cloud_anisotropic_extinctions - cloud_final_absorption_opacities,
        lambda: cloud_isotropic_extinctions - cloud_final_absorption_opacities,
    )

    def sum_cloud_absorption_opacities(_cloud_anisotropic_extinctions, _cloud_final_absorption_opacities):
        cloud_anisotropic_scattering_opacities = (
            _cloud_anisotropic_extinctions - _cloud_final_absorption_opacities
        )
        cloud_absorption_opacities = _cloud_final_absorption_opacities
        return cloud_anisotropic_scattering_opacities, cloud_absorption_opacities

    cloud_anisotropic_scattering_opacities, cloud_absorption_opacities = lax.cond(
        jnp.logical_and(sum_opacities, jnp.logical_not(photospheric_cloud_optical_depths is None)),
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


@jit(static_argnames=['cloud_species'])
def init_clouds_particles_porosity_factor(clouds_particles_porosity_factor, cloud_species):
    # This check is done in _compute_opacities now
    #if (clouds_particles_porosity_factor is None) or (not bool(clouds_particles_porosity_factor)):
    #    clouds_particles_porosity_factor = {species: -jnp.inf for species in cloud_species}

    for species in cloud_species:
        # TODO: check JAX compatibility for split_species_all_info
        _, _, _, _, source, _ = Opacity.split_species_all_info(CloudOpacity.get_aliases(species))

        # Operate on a local variable so we can use lax.cond and jittable jnp ops
        _porosity = clouds_particles_porosity_factor[species]

        # For DHS sources: if porosity == 0 -> print a warning (leave value unchanged),
        # elif not finite -> set to 0.25
        def _handle_dhs(porosity):
            return lax.cond(
                jnp.equal(porosity, 0),
                # true: print warning and return original p
                lambda q: (debug.print(f"cloud species '{species}' opacity calculation method is using the DHS method, "
                                        f"the porosity factor should be > 0 (0.25 is used by default)"), q)[1],
                # false: if finite -> return p else -> set to default 0.25
                lambda q: lax.cond(jnp.isfinite(q), lambda r: r, lambda r: jnp.array(0.25, dtype=JAX_DTYPE), q),
                porosity
            )

        # For non-DHS sources: if not finite -> set to 0.0, else keep value
        def _handle_non_dhs(porosity):
            return lax.cond(jnp.isfinite(porosity), lambda r: r, lambda r: jnp.array(0.0, dtype=JAX_DTYPE), porosity)

        # Choose handler based on source (source is static Python string)
        _porosity = lax.cond(source == 'DHS', lambda p: _handle_dhs(p), lambda p: _handle_non_dhs(p), _porosity)

        # Store back the (possibly updated) porosity
        clouds_particles_porosity_factor[species] = _porosity

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

    return clouds_particles_porosity_factor


@jit(static_argnames=['cloud_species', 'cloud_particles_radius_distribution', 'complete_coverage_clouds'])
def compute_cloud_continuum_opacities( # TODO: revert to previous logic
    frequencies: ArrayLike,  # NOTE: use self
    frequency_bins_edges: ArrayLike, # NOTE: use self
    pressures: ArrayLike, # NOTE: use self
    temperatures: ArrayLike,
    mass_fractions: dict[str, ArrayLike],
    mean_molar_masses: ArrayLike,
    reference_gravity: float,
    scattering_in_transmission: bool,  # NOTE: use self
    sum_opacities: bool,  # NOTE: use self
    opaque_cloud_top_pressure: float,
    power_law_opacity_350nm: float,
    power_law_opacity_coefficient: float,
    cloud_species: list[str] | tuple[str], # NOTE: use self
    clouds_particles_porosity_factor: dict[str, float],
    clouds_loaded_opacities: LockedDict[str, ArrayLike], # NOTE: use self
    anisotropic_cloud_scattering: bool, # NOTE: use self?
    cloud_particles_mean_radii: ArrayLike,
    cloud_particle_radius_distribution_std: float,
    cloud_particle_number_density_grid: ArrayLike, # TODO update to Annie's version
    cloud_f_sed: dict[str, float],
    eddy_diffusion_coefficients: dict[str, float],
    cloud_particles_radius_distribution: str,
    cloud_hansen_a: float,
    cloud_hansen_b: float,
    cloud_photosphere_median_optical_depth: float,
    complete_coverage_clouds: tuple[str],
    return_cloud_contribution: bool,
    target_particle_radii,
    target_pressure

):
    n_clouds = len(cloud_species)
    cloudy_continuum_opacities, cloudy_continuum_opacities_scattering, cloud_scattering_opacities, \
        cloud_absorption_opacities, cloud_opacities, _cloud_particles_mean_radii = (
            _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
        )
    clear_continuum_opacities, clear_continuum_opacities_scattering, clear_cloud_scattering_opacities, \
        clear_cloud_absorption_opacities, _, _ = _init_cloud_branch_outputs(frequencies, pressures, n_clouds)
    complete_coverage_clouds = _filter_complete_coverage_clouds(cloud_species, complete_coverage_clouds)

    (cloudy_continuum_opacities, cloudy_continuum_opacities_scattering,
     clear_continuum_opacities, clear_continuum_opacities_scattering,
     scattering_in_transmission) = _init_cloud_background_continuum(
        frequencies=frequencies,
        pressures=pressures,
        n_clouds=n_clouds,
        scattering_in_transmission=scattering_in_transmission,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        power_law_opacity_350nm=power_law_opacity_350nm,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
    )

    if len(cloud_species) > 0:
        clouds_particles_porosity_factor = init_clouds_particles_porosity_factor(
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_species=cloud_species
        )

    cloudy_clouds_have_effect = _selected_clouds_have_effect(
        mass_fractions=mass_fractions,
        selected_cloud_species=cloud_species
    )
    clear_clouds_have_effect = _selected_clouds_have_effect(
        mass_fractions=mass_fractions,
        selected_cloud_species=complete_coverage_clouds
    )

    (cloudy_continuum_opacities, cloudy_continuum_opacities_scattering,
     cloud_scattering_opacities, cloud_absorption_opacities,
     cloud_opacities, _cloud_particles_mean_radii,
     cloudy_branch_has_effect) = _maybe_compute_cloud_branch(
        selected_cloud_species=cloud_species,
        selected_clouds_have_effect=cloudy_clouds_have_effect,
        base_continuum_opacities=cloudy_continuum_opacities,
        base_continuum_opacities_scattering=cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities=cloud_scattering_opacities,
        cloud_absorption_opacities=cloud_absorption_opacities,
        cloud_opacities=cloud_opacities,
        cloud_particles_mean_radii=_cloud_particles_mean_radii,
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
        input_cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_particles_radius_distribution=cloud_particles_radius_distribution,
        cloud_hansen_a=cloud_hansen_a,
        cloud_hansen_b=cloud_hansen_b,
        clouds_particles_porosity_factor=clouds_particles_porosity_factor,
        cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
        return_cloud_contribution=return_cloud_contribution,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )
    scattering_in_transmission = jnp.logical_or(scattering_in_transmission, cloudy_branch_has_effect)

    (clear_continuum_opacities, clear_continuum_opacities_scattering,
     clear_cloud_scattering_opacities, clear_cloud_absorption_opacities,
     _, _, clear_branch_has_effect) = _maybe_compute_cloud_branch(
        selected_cloud_species=complete_coverage_clouds,
        selected_clouds_have_effect=clear_clouds_have_effect,
        base_continuum_opacities=clear_continuum_opacities,
        base_continuum_opacities_scattering=clear_continuum_opacities_scattering,
        cloud_scattering_opacities=clear_cloud_scattering_opacities,
        cloud_absorption_opacities=clear_cloud_absorption_opacities,
        cloud_opacities=cloud_opacities,
        cloud_particles_mean_radii=_cloud_particles_mean_radii,
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
        input_cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_particles_radius_distribution=cloud_particles_radius_distribution,
        cloud_hansen_a=cloud_hansen_a,
        cloud_hansen_b=cloud_hansen_b,
        clouds_particles_porosity_factor=clouds_particles_porosity_factor,
        cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
        return_cloud_contribution=return_cloud_contribution,
        target_particle_radii=target_particle_radii,
        target_pressure=target_pressure,
    )
    scattering_in_transmission = jnp.logical_or(scattering_in_transmission, clear_branch_has_effect)

    return (
        cloudy_continuum_opacities, cloudy_continuum_opacities_scattering,
        cloud_scattering_opacities, cloud_absorption_opacities,
        cloud_opacities, _cloud_particles_mean_radii,
        clear_continuum_opacities, clear_continuum_opacities_scattering,
        clear_cloud_scattering_opacities, clear_cloud_absorption_opacities,
        scattering_in_transmission
    )


@jit(static_argnames=['cloud_species', 'cloud_particles_radius_distribution'])
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
    sum_opacities, anisotropic_cloud_scattering,
    cloud_f_sed=None, eddy_diffusion_coefficients=None,
    cloud_particles_mean_radii=None,
    cloud_particles_radius_distribution="lognormal",
    cloud_hansen_a=None, cloud_hansen_b=None,
    cloud_particle_number_density_grid=None,
    clouds_particles_porosity_factor=None,
    photospheric_cloud_optical_depths=None,
    continuum_opacities=None,
    return_cloud_contribution=False,
    target_particle_radii=None,
    target_pressure=None

    ):
    """Calculate cloud opacities for a defined atmospheric structure.
    This specifically calculates the various cloud absorption and scattering opacities
    for a physical cloud model, typically based on an Ackerman & Marley (2001) cloud, or similar.
    # TODO complete docstring

    Args:
        temperatures:
        cloud_species_mass_fractions:
        mean_molar_masses:
        reference_gravity:
        cloud_particle_radius_distribution_std:
        cloud_f_sed:
        eddy_diffusion_coefficients:
        cloud_particles_mean_radii:
        cloud_particles_radius_distribution:
        cloud_hansen_a:
        cloud_hansen_b:
        return_cloud_contribution:

    Returns:

    """
    n_clouds = len(cloud_species)
    atmospheric_densities = pressures * mean_molar_masses * cst.amu / cst.kB / temperatures

    if continuum_opacities is None:
        continuum_opacities = jnp.zeros((frequency_bins_edges.size - 1, pressures.size), dtype=JAX_DTYPE)

    cloud_hansen_b = _initialize_hansen_b(
        cloud_hansen_b=cloud_hansen_b,
        cloud_particles_radius_distribution=cloud_particles_radius_distribution,
        pressures=pressures,
        n_clouds=n_clouds,
    )
    _cloud_species_mass_fractions, _cloud_particles_density, _cloud_particles_mean_radii = (
        _initialize_cloud_species_inputs(
            cloud_species=cloud_species,
            cloud_species_mass_fractions=cloud_species_mass_fractions,
            pressures=pressures,
            clouds_loaded_opacities=clouds_loaded_opacities,
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_particles_mean_radii=cloud_particles_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
        )
    )

    if cloud_particles_mean_radii is not None or cloud_hansen_a is not None:
        (
            clouds_total_absorption_opacities,
            clouds_total_scattering_opacities,
            cloud_scattering_reduction_factor,
        ) = _compute_cloud_opacity_grids_from_known_radii(
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            atmospheric_densities=atmospheric_densities,
            cloud_particles_density=_cloud_particles_density,
            cloud_species_mass_fractions_grid=_cloud_species_mass_fractions,
            cloud_particles_mean_radii=_cloud_particles_mean_radii,
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
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            clouds_loaded_opacities=clouds_loaded_opacities,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
        )
        (
            clouds_total_absorption_opacities,
            clouds_total_scattering_opacities,
            cloud_scattering_reduction_factor,
            _cloud_particles_mean_radii,
        ) = _compute_cloud_opacity_grids_with_derived_radii(
            cloud_particles_radius_distribution=cloud_particles_radius_distribution,
            reference_gravity=reference_gravity,
            atmospheric_densities=atmospheric_densities,
            cloud_particles_density=_cloud_particles_density,
            temperatures=temperatures,
            mean_molar_masses=mean_molar_masses,
            cloud_f_sed=cloud_f_sed,
            cloud_species=cloud_species,
            pressures=pressures,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
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
        _cloud_particles_mean_radii,
    )


def compute_species_optical_depths(reference_gravity, pressures, cloud_opacities):
    """Calculate the optical depth of the clouds as function of
    frequency and pressure. The array with the optical depths is set to the
    ``tau_cloud`` attribute. The optical depth is calculated from the top of
    the atmosphere (i.e. the smallest pressure). Therefore, below the cloud
    base, the optical depth is constant and equal to the value at the cloud
    base.
    # TODO complete docstring
    # TODO rename to _compute_cloud_species_optical_depths

        Args:
            reference_gravity (float):
                Surface gravity in cgs. Vertically constant for emission
                spectra.
    """
    return radiative_transfer_core.compute_species_optical_depths(reference_gravity, pressures, cloud_opacities)
