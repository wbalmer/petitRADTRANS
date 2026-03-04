import copy
import jax.numpy as jnp
from jax import lax, debug, jit, config
from jax.typing import ArrayLike
#from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore
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
    clouds_have_effect: bool, # NOTE: use self
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
    # Initialization
    n_clouds = len(cloud_species)
    min_cloud_shape = max(n_clouds, 1)  # to allow for no-cloud cases without breaking the code with zero-size arrays
    cloud_absorption_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    cloud_scattering_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    # TODO: figure out what shape cloud_opacities should actually have. And why we even need it...
    #cloud_opacities = jnp.zeros((min_cloud_shape, frequencies.size, min_cloud_shape, pressures.size), dtype=JAX_DTYPE)
    cloud_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    _cloud_particles_mean_radii = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)

    cloud_continuum_opacities = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)
    cloud_continuum_opacities_scattering = jnp.zeros((frequencies.size, pressures.size), dtype=JAX_DTYPE)

    # Add opaque cloud deck opacity (use numeric sentinel: present when >= 0)
    def _add_opaque_deck(_cloud_continuum_opacities: ArrayLike, _pressures: ArrayLike, _p_top: float):
        inds = jnp.where(jnp.greater_equal(_pressures, _p_top*1e6), size=_pressures.shape[0])
        _cloud_continuum_opacities = _cloud_continuum_opacities.at[:, inds].add(1e99)
        return _cloud_continuum_opacities

    cloud_continuum_opacities = lax.cond(
        jnp.greater_equal(opaque_cloud_top_pressure, 0.0),
        _add_opaque_deck,
        lambda a, b, c: a,
        cloud_continuum_opacities,
        pressures,
        opaque_cloud_top_pressure
    )

    # Add power law opacity: use sentinels to detect presence
    both_power_law_present = jnp.logical_and(
        jnp.not_equal(power_law_opacity_350nm, _MISSING_FLOAT),
        jnp.not_equal(power_law_opacity_coefficient, _MISSING_FLOAT)
    )

    def _add_powerlaw(args):
        _scat_flag, _scat_arr = args
        _scat_flag = True
        _scat_arr = _scat_arr + compute_power_law_opacities(
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            frequencies=frequencies,
            n_layers=pressures.size
        )
        return (_scat_flag, _scat_arr)

    scattering_in_transmission, cloud_continuum_opacities_scattering = lax.cond(
        both_power_law_present,
        _add_powerlaw,
        lambda args: args,
        (scattering_in_transmission, cloud_continuum_opacities_scattering)
    )

    # If only one parameter is provided (incomplete pair), raise a runtime error via debug.callback
    def _raise_incomplete_powerlaw(a, b):
        # Raise only if exactly one of the two parameters is provided
        a_missing = bool(a == _MISSING_FLOAT)
        b_missing = bool(b == _MISSING_FLOAT)
        if a_missing ^ b_missing:
            none_parameter = 'power_law_opacity_350nm' if a_missing else 'power_law_opacity_coefficient'
            raise RuntimeError(
                f"to include a power law opacity, both parameters 'power_law_opacity_350nm' "
                f"and 'power_law_opacity_coefficient' must be provided; missing '{none_parameter}'"
            )

    debug.callback(_raise_incomplete_powerlaw, power_law_opacity_350nm, power_law_opacity_coefficient)

    def process_cloud_case(
            cloud_continuum_opacities,
            cloud_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            clouds_particles_porosity_factor,
            _cloud_particles_mean_radii,
            scattering_in_transmission):
        cloud_species_mass_fractions = {}
        for species in cloud_species:
            cloud_species_mass_fractions[species] = mass_fractions[species]

        def raise_error_for_inconsistent_cloud_species_length(cloud_species_mass_fractions, cloud_species, keys):
            if len(cloud_species_mass_fractions) != len(cloud_species):
                raise ValueError(
                    f"the number of cloud species mass fractions ({len(cloud_species_mass_fractions)}) "
                    f"does not match the number of cloud species ({len(cloud_species)}); "
                    f"check for inconsistencies between mass_fractions keys ({keys}) "
                    f"and cloud_species keys ({cloud_species})"
                )
        debug.callback(
            raise_error_for_inconsistent_cloud_species_length,
            cloud_species_mass_fractions,
            cloud_species,
            list(mass_fractions.keys())
            )

        scattering_in_transmission = True

        clouds_particles_porosity_factor = init_clouds_particles_porosity_factor(
            clouds_particles_porosity_factor=clouds_particles_porosity_factor,
            cloud_species=cloud_species
        )

        # Opacities with all clouds
        (cloud_continuum_opacities, cloud_continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities,
            cloud_opacities,  # scattering + absorption, used for cloud contribution
            _cloud_particles_mean_radii) = _compute_cloud_opacities(
            pressures=pressures,
            temperatures=temperatures,
            frequency_bins_edges=frequency_bins_edges,
            cloud_species=cloud_species,
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
            continuum_opacities=cloud_continuum_opacities,
            return_cloud_contribution=return_cloud_contribution,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure
        )

        # Opacities with only the clouds with complete coverage (will be treated as standard continuum opacities)
        def complete_cloud_coverage_case(
            cloud_continuum_opacities,
            cloud_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            clouds_particles_porosity_factor
        ):
            clear_cloud_species_mass_fractions = copy.deepcopy(cloud_species_mass_fractions)

            # TODO JIT to cond
            for species in clear_cloud_species_mass_fractions:
                clear_cloud_species_mass_fractions[species] = lax.cond(
                    species not in complete_coverage_clouds,
                    lambda mf: jnp.zeros_like(mf),
                    lambda mf: mf,
                    clear_cloud_species_mass_fractions[species]
                )

            complete_coverage_cloud_continuum_opacities = lax.cond(
                jnp.greater_equal(opaque_cloud_top_pressure, 0.0),
                lambda c: c.at[:, jnp.where(pressures > opaque_cloud_top_pressure * 1e6, size=pressures.shape[0])].add(1e99),
                lambda c: c,
                jnp.zeros_like(cloud_continuum_opacities)
            )

            (complete_coverage_cloud_continuum_opacities, complete_coverage_cloud_continuum_opacities_scattering,
                complete_coverage_cloud_scattering_opacities, complete_coverage_cloud_absorption_opacities,
                _, _) = _compute_cloud_opacities(
                pressures=pressures,
                temperatures=temperatures,
                frequency_bins_edges=frequency_bins_edges,
                cloud_species=cloud_species,
                cloud_species_mass_fractions=clear_cloud_species_mass_fractions,
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
                continuum_opacities=cloud_continuum_opacities[1],
                return_cloud_contribution=return_cloud_contribution
            )
            return (complete_coverage_cloud_continuum_opacities,
                    complete_coverage_cloud_continuum_opacities_scattering,
                    complete_coverage_cloud_scattering_opacities,
                    complete_coverage_cloud_absorption_opacities)

        # Calculate cloud opacities
        def pass_through_case(
                cloud_continuum_opacities,
                cloud_continuum_opacities_scattering,
                cloud_scattering_opacities,
                cloud_absorption_opacities,
                clouds_particles_porosity_factor):
            return (jnp.zeros_like(cloud_continuum_opacities),
                    jnp.zeros_like(cloud_continuum_opacities_scattering),
                    jnp.zeros_like(cloud_scattering_opacities),
                    jnp.zeros_like(cloud_absorption_opacities))

        (complete_coverage_cloud_continuum_opacities, complete_coverage_cloud_continuum_opacities_scattering,
            complete_coverage_cloud_scattering_opacities, complete_coverage_cloud_absorption_opacities) = lax.cond(
            len(complete_coverage_clouds) > 0,
            complete_cloud_coverage_case,
            pass_through_case,
            cloud_continuum_opacities,
            cloud_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            clouds_particles_porosity_factor
        )

        return (
            cloud_continuum_opacities, cloud_continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities,
            cloud_opacities, _cloud_particles_mean_radii,
            scattering_in_transmission
        )

    def pass_through_case(
                cloud_continuum_opacities,
                cloud_continuum_opacities_scattering,
                cloud_scattering_opacities,
                cloud_absorption_opacities,
                cloud_opacities,
                clouds_particles_porosity_factor,
                _cloud_particles_mean_radii,
                scattering_in_transmission):
        return (cloud_continuum_opacities,
                cloud_continuum_opacities_scattering,
                cloud_scattering_opacities,
                cloud_absorption_opacities,
                cloud_opacities,
                jnp.zeros_like(_cloud_particles_mean_radii),
                scattering_in_transmission)

    predicate = jnp.any(jnp.array([
        clouds_have_effect,  # This is probably good enough
        len(complete_coverage_clouds) > 0,
        jnp.greater_equal(opaque_cloud_top_pressure, 0.0),
        clouds_loaded_opacities['particles_radii_bins'] is not None
    ], dtype=bool))

    (cloud_continuum_opacities, cloud_continuum_opacities_scattering,
        cloud_scattering_opacities, cloud_absorption_opacities,
        cloud_opacities, _cloud_particles_mean_radii,
        scattering_in_transmission) = lax.cond(
        predicate,
        process_cloud_case,
        pass_through_case,
        cloud_continuum_opacities,
        cloud_continuum_opacities_scattering,
        cloud_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        clouds_particles_porosity_factor,
        _cloud_particles_mean_radii,
        scattering_in_transmission
    )

    debug.print("cloud_continuum_opacities {}", cloud_continuum_opacities)
    debug.print("cloud_continuum_opacities_scattering {}", cloud_continuum_opacities_scattering)
    debug.print("cloud_scattering_opacities {}", cloud_scattering_opacities)
    debug.print("cloud_absorption_opacities {}", cloud_absorption_opacities)
    debug.print("cloud_opacities {}", cloud_opacities)
    debug.print("_cloud_particles_mean_radii {}",_cloud_particles_mean_radii)
    # TODO: fix outputs now that we aren't using smart_clear_opacities
    return (
        cloud_continuum_opacities, cloud_continuum_opacities_scattering,
        cloud_scattering_opacities, cloud_absorption_opacities,
        cloud_opacities, _cloud_particles_mean_radii,
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
    # Initialization
    n_clouds = len(cloud_species)
    min_cloud_shape = max(n_clouds, 1)

    _cloud_species_mass_fractions = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)
    cloud_absorption_opacities = None
    cloud_anisotropic_scattering_opacities = None
    _cloud_particles_mean_radii = jnp.zeros((pressures.size, min_cloud_shape), dtype=JAX_DTYPE)
    _cloud_particles_density = jnp.zeros(n_clouds, dtype=JAX_DTYPE)
    atmospheric_densities = pressures * mean_molar_masses * cst.amu / cst.kB / temperatures

    if continuum_opacities is None:
        continuum_opacities = jnp.zeros((frequency_bins_edges.size - 1, pressures.size), dtype=JAX_DTYPE)

    # TODO JIT
    # better parameter defaults
    # make sure if statements are use static properties (size etc)
    # modularise - break hansen and lognormal into separate functions
    # Initialize Hansen's b coefficient
    if "hansen" in cloud_particles_radius_distribution.lower():
        if isinstance(cloud_hansen_b, jnp.ndarray):
            if not cloud_hansen_b.shape == (pressures.size, n_clouds):
                raise ValueError(
                    "cloud_hansen_b must be a float, a dictionary with arrays for each cloud species, "
                    f"or an array with shape {(pressures.shape[0], n_clouds)}, "
                    f"but was of shape {jnp.shape(cloud_hansen_b)}"
                )
        elif isinstance(cloud_hansen_b, dict):
            cloud_hansen_b = jnp.array(list(cloud_hansen_b.values()), dtype=JAX_DTYPE)
        elif isinstance(cloud_hansen_b, float):
            cloud_hansen_b = jnp.array(
                jnp.tile(cloud_hansen_b * jnp.ones_like(pressures), (n_clouds, 1)), dtype=JAX_DTYPE).T
        else:
            raise ValueError(f"The Hansen distribution width (cloud_hansen_b) must be an array, a dict, "
                                f"or a float, but is of type '{type(cloud_hansen_b)}' ({cloud_hansen_b})")

    # Initialize cloud species mass fractions, densities, and mean radii
    for i_spec, cloud_name in enumerate(cloud_species_mass_fractions):
        # TODO: standardize cloud naming
        cloud_opacity = CloudOpacity([cloud_name], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_short_name = species_full_name.split('_')[0]

        _cloud_species_mass_fractions = _cloud_species_mass_fractions.at[:, i_spec].set(
            cloud_species_mass_fractions[cloud_name]
            )
        if clouds_particles_porosity_factor is not None:
            _cloud_particles_density = _cloud_particles_density.at[i_spec].set(
                clouds_loaded_opacities['particles_densities'][i_spec]
                * (1 - clouds_particles_porosity_factor[cloud_name])
            )
        else:
            _cloud_particles_density = _cloud_particles_density.at[i_spec].set(
                clouds_loaded_opacities['particles_densities']
                )

        if cloud_particles_mean_radii is not None:
            if cloud_name in cloud_particles_mean_radii.keys():
                _cloud_particles_mean_radii = _cloud_particles_mean_radii.at[:, i_spec].set(
                    cloud_particles_mean_radii[cloud_name]
                    )
            elif cloud_short_name in cloud_particles_mean_radii.keys():
                _cloud_particles_mean_radii = _cloud_particles_mean_radii.at[:, i_spec].set(
                    cloud_particles_mean_radii[cloud_short_name]
                    )
        elif cloud_hansen_a is not None:
            if cloud_name in cloud_hansen_a.keys():
                _cloud_particles_mean_radii = _cloud_particles_mean_radii.at[:, i_spec].set(cloud_hansen_a[cloud_name])
            elif cloud_short_name in cloud_hansen_a.keys():
                _cloud_particles_mean_radii = _cloud_particles_mean_radii.at[:, i_spec].set(cloud_hansen_a[cloud_short_name])

    # Calculate cloud opacities
    if cloud_particles_mean_radii is not None or cloud_hansen_a is not None:
        if cloud_particles_radius_distribution == "lognormal":
            (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                cloud_scattering_reduction_factor) = \
                cloud_core.compute_cloud_log_normal_particles_distribution_opacities_grid(
                        atmosphere_densities=atmospheric_densities,
                        clouds_particles_densities=_cloud_particles_density,
                        clouds_mass_fractions=_cloud_species_mass_fractions,
                        cloud_particles_mean_radii=_cloud_particles_mean_radii,
                        cloud_particles_distribution_std=cloud_particle_radius_distribution_std,
                        cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                        cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                        clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                        clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                        clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                        pressures=pressures,
                        target_particle_radii=target_particle_radii,
                        target_pressure=target_pressure
                )
        elif cloud_particles_radius_distribution == "custom":
            (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                cloud_scattering_reduction_factor) = \
                cloud_core._compute_cloud_opacities_using_particle_size_grid(
                    atmosphere_densities=atmospheric_densities,
                    clouds_mass_fractions=cloud_species_mass_fractions,
                    cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                    clouds_particles_densities=_cloud_particles_density,
                    cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                    cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                    clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                    clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                    clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                    pressures=pressures,
                    target_particle_radii=target_particle_radii,
                    target_pressure=target_pressure
                )

        else:
            #TODO: Implement hansen cloud opas in cloud_core
            (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                cloud_scattering_reduction_factor) = \
                cloud_core.compute_cloud_hansen_opacities(
                    atmospheric_densities,
                    _cloud_particles_density,
                    _cloud_species_mass_fractions,
                    _cloud_particles_mean_radii,
                    cloud_hansen_b,
                    clouds_loaded_opacities['particles_radii_bins'],
                    clouds_loaded_opacities['particles_radii'],
                    clouds_loaded_opacities['absorption_opacities'],
                    clouds_loaded_opacities['scattering_opacities'],
                    clouds_loaded_opacities['particles_asymmetry_parameters']
                )
    else:
        missing_arguments = []

        if (cloud_particle_radius_distribution_std is None or cloud_particle_radius_distribution_std is _MISSING_FLOAT) and cloud_particles_radius_distribution == "lognormal":
            missing_arguments.append("'cloud_particle_radius_distribution_std'")

        if cloud_f_sed is None or cloud_f_sed is _MISSING_FLOAT:
            missing_arguments.append("'cloud_f_sed'")

        if eddy_diffusion_coefficients is None or eddy_diffusion_coefficients is _MISSING_FLOAT:
            missing_arguments.append("'eddy_diffusion_coefficients'")

        if clouds_loaded_opacities['particles_radii_bins'] is None:
            missing_arguments.append("'clouds_loaded_opacities.particles_radii_bins'")

        if cloud_particles_radius_distribution == "custom" and cloud_particle_number_density_grid is None:
                missing_arguments.append("'cloud_particle_number_density_grid'")

        def raise_missing_arguments_error(missing_arguments):
            if len(missing_arguments) > 0:
                raise ValueError(
                    f"missing necessary arguments to calculate cloud particle radii: {', '.join(missing_arguments)} "
                    f"(got 'None')\n"
                    f"Set the missing arguments, "
                    f"or directly set cloud particle radii "
                    f"through the arguments 'cloud_particles_mean_radii' or 'cloud_hansen_a', "
                    f"and argument 'cloud_particle_radius_distribution_std'.\n"
                    f"The cloud particle radii are necessary to calculate the cloud opacities."
                )
        debug.callback(raise_missing_arguments_error, missing_arguments)

        def standardize_f_sed(cloud_f_sed, cloud_species, pressures_size):
            # Initialize f_seds
            f_seds = jnp.zeros((pressures.size, n_clouds), dtype=JAX_DTYPE)

            for i_spec, cloud in enumerate(cloud_species):
                if isinstance(cloud_f_sed, dict):
                    f_seds = f_seds.at[:, i_spec].set(cloud_f_sed[cloud])
                else:  # or array with length of number of atmospheric layers, or scalar.
                    f_seds = f_seds.at[:, i_spec].set(cloud_f_sed)
            return f_seds
        f_seds = standardize_f_sed(cloud_f_sed, list(cloud_species_mass_fractions.keys()), pressures.size)

        # TODO: better conditional for jit
        # Calculate cloud_particles_mean_radii then cloud opacities
        if cloud_particles_radius_distribution == "lognormal":
            _cloud_particles_mean_radii = cloud_core.compute_cloud_particles_mean_radius(
                reference_gravity,
                atmospheric_densities,
                _cloud_particles_density,
                temperatures,
                mean_molar_masses,
                f_seds,
                cloud_particle_radius_distribution_std,
                eddy_diffusion_coefficients
            )

            (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                 cloud_scattering_reduction_factor) = \
                    cloud_core.compute_cloud_log_normal_particles_distribution_opacities_grid(
                        atmosphere_densities=atmospheric_densities,
                        clouds_particles_densities=_cloud_particles_density,
                        clouds_mass_fractions=_cloud_species_mass_fractions,
                        cloud_particles_mean_radii=_cloud_particles_mean_radii,
                        cloud_particles_distribution_std=cloud_particle_radius_distribution_std,
                        cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                        cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                        clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                        clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                        clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                        pressures=pressures,
                        target_particle_radii=target_particle_radii,
                        target_pressure=target_pressure

                    )

        elif cloud_particles_radius_distribution == "hansen":
            _cloud_particles_mean_radii = cloud_core.compute_cloud_particles_mean_radius_hansen(
                reference_gravity,
                atmospheric_densities,
                _cloud_particles_density,
                temperatures,
                mean_molar_masses,
                f_seds,
                cloud_hansen_b,
                eddy_diffusion_coefficients
            )

            #TODO: Implement hansen cloud opas
            (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                cloud_scattering_reduction_factor) = \
                cloud_core.compute_cloud_hansen_opacities(
                    atmospheric_densities,
                    _cloud_particles_density,
                    _cloud_species_mass_fractions,
                    _cloud_particles_mean_radii,
                    cloud_hansen_b,
                    clouds_loaded_opacities['particles_radii_bins'],
                    clouds_loaded_opacities['particles_radii'],
                    clouds_loaded_opacities['absorption_opacities'],
                    clouds_loaded_opacities['scattering_opacities'],
                    clouds_loaded_opacities['particles_asymmetry_parameters']
                )
        elif cloud_particles_radius_distribution == "custom":
                (clouds_total_absorption_opacities, clouds_total_scattering_opacities,
                 cloud_scattering_reduction_factor) = cloud_core._compute_cloud_opacities_using_particle_size_grid(
                    atmosphere_densities=atmospheric_densities,
                    clouds_mass_fractions=cloud_species_mass_fractions,
                    cloud_particle_number_density_grid=cloud_particle_number_density_grid,
                    clouds_particles_densities=_cloud_particles_density,
                    cloud_particles_radii_bins=clouds_loaded_opacities['particles_radii_bins'],
                    cloud_particles_radii=clouds_loaded_opacities['particles_radii'],
                    clouds_absorption_opacities=clouds_loaded_opacities['absorption_opacities'],
                    clouds_scattering_opacities=clouds_loaded_opacities['scattering_opacities'],
                    clouds_particles_asymmetry_parameters=clouds_loaded_opacities['particles_asymmetry_parameters'],
                    pressures=pressures,
                    target_particle_radii=target_particle_radii,
                    target_pressure=target_pressure
                )


    # Take into account anisotropy (= 1 - asymmetry_parameter)
    (cloud_final_absorption_opacities, cloud_anisotropic_extinctions,
        cloud_scattering_reduction_factor, cloud_isotropic_extinctions) = \
        cloud_core.interpolate_cloud_opacities(
            clouds_total_absorption_opacities,
            clouds_total_scattering_opacities,
            cloud_scattering_reduction_factor,
            clouds_loaded_opacities['wavelengths'],
            frequency_bins_edges
        )

    continuum_opacities_scattering = lax.cond(
        anisotropic_cloud_scattering,
        lambda: cloud_anisotropic_extinctions - cloud_final_absorption_opacities,
        lambda: cloud_isotropic_extinctions - cloud_final_absorption_opacities
    )

    def sum_cloud_absorption_opacities(
            cloud_anisotropic_extinctions,
            cloud_final_absorption_opacities):
        cloud_anisotropic_scattering_opacities = (
                cloud_anisotropic_extinctions - cloud_final_absorption_opacities
        )
        cloud_absorption_opacities = cloud_final_absorption_opacities
        return cloud_anisotropic_scattering_opacities, cloud_absorption_opacities

    cloud_anisotropic_scattering_opacities, cloud_absorption_opacities = lax.cond(
        sum_opacities & jnp.logical_not(photospheric_cloud_optical_depths is None),
        sum_cloud_absorption_opacities,
        lambda a, b: (jnp.zeros_like(a), jnp.zeros_like(b)),
        cloud_anisotropic_extinctions,
        cloud_final_absorption_opacities,
    )

    # Same thing as cloud_absorption_opacities
    continuum_opacities += cloud_final_absorption_opacities

    # This included scattering plus absorption
    # TODO: figure out cloud_opacities and how to return contribution
    #if return_cloud_contribution:
    opacity_shape = (min_cloud_shape, frequency_bins_edges.size - 1, min_cloud_shape, pressures.size)
    cloud_opacities = cloud_anisotropic_extinctions#.reshape(opacity_shape)
    #else:
    #    cloud_opacities = None

    return (continuum_opacities, continuum_opacities_scattering,
            cloud_anisotropic_scattering_opacities, cloud_absorption_opacities, cloud_opacities,
            _cloud_particles_mean_radii)


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
