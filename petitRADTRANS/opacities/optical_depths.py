import jax.numpy as jnp
from jax import debug
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.opacities.continuum import combine_opacities
from petitRADTRANS.opacities.clouds import compute_species_optical_depths
from petitRADTRANS.physics import frequency2wavelength

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE


def _total_absorption_opacity(opacities):
    return jnp.asarray(opacities[:, :, :1, :], dtype=WORKING_JAX_DTYPE)


def _compute_total_optical_depths(
    pressures,
    reference_gravity,
    opacities,
    continuum_opacities_scattering,
    scattering_in_emission,
):
    optical_depths, photon_destruction_probabilities = (
        radiative_transfer_core.compute_optical_depths(
            reference_gravity=reference_gravity,
            pressures=pressures,
            opacities=_total_absorption_opacity(opacities),
            scattering_in_emission=scattering_in_emission,
            continuum_opacities_scattering=continuum_opacities_scattering,
        )
    )

    return (
        jnp.asarray(optical_depths, dtype=WORKING_JAX_DTYPE),
        jnp.asarray(photon_destruction_probabilities, dtype=WORKING_JAX_DTYPE),
    )


def compute_optical_depths(
    pressures,
    reference_gravity,
    opacities,
    continuum_opacities_scattering,
    scattering_in_emission,
    sum_opacities,
    photospheric_cloud_optical_depths=None,
    absorber_present=True,
    **custom_cloud_parameters,
):
    photon_destruction_probabilities = None
    relative_cloud_scaling_factor = None

    # Calculate optical depth for the total opacity
    if sum_opacities:
        if photospheric_cloud_optical_depths is not None:
            (
                optical_depths,
                photon_destruction_probabilities,
                relative_cloud_scaling_factor,
            ) = compute_optical_depths_with_photospheric_cloud(
                pressures=pressures,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                photospheric_cloud_optical_depths=photospheric_cloud_optical_depths,
                scattering_in_emission=scattering_in_emission,
                **custom_cloud_parameters,
            )
        else:
            if scattering_in_emission:
                _continuum_opacities_scattering = continuum_opacities_scattering
            else:
                _continuum_opacities_scattering = jnp.zeros(
                    continuum_opacities_scattering.shape, dtype=WORKING_JAX_DTYPE
                )
            optical_depths, photon_destruction_probabilities = (
                _compute_total_optical_depths(
                    pressures=pressures,
                    reference_gravity=reference_gravity,
                    opacities=opacities,
                    continuum_opacities_scattering=_continuum_opacities_scattering,
                    scattering_in_emission=scattering_in_emission,
                )
            )

        # Handle cases without any absorbers, where opacities are zero
        if not absorber_present:
            debug.print(
                "No absorbers present, setting the photon destruction probability in the atmosphere to 1."
            )
            photon_destruction_probabilities = photon_destruction_probabilities.at[
                jnp.isnan(photon_destruction_probabilities)
            ].set(1.0)

        # To handle cases when tau_cloud_at_Phot_clear = 0,
        # therefore cloud_scaling_factor = inf,
        # continuum_opacities_scattering_emission will contain nans and infs,
        # and photon_destruction_prob contains only nans
        photon_destruction_probabilities = jnp.where(
            jnp.isnan(photon_destruction_probabilities),
            1.0,
            photon_destruction_probabilities,
        )
        photon_destruction_probabilities = jnp.asarray(
            photon_destruction_probabilities, dtype=WORKING_JAX_DTYPE
        )
    else:
        optical_depths = jnp.asarray(
            compute_species_optical_depths(
                reference_gravity=reference_gravity,
                pressures=pressures,
                cloud_opacities=opacities,  # TODO David: these are opacities per (gas) species, too
                                            #    not just clouds. Fix function naming.
            ),
            dtype=WORKING_JAX_DTYPE,
        )

    return (
        optical_depths,
        photon_destruction_probabilities,
        relative_cloud_scaling_factor,
    )


def compute_optical_depths_with_photospheric_cloud(
    pressures,
    reference_gravity,
    opacities,
    continuum_opacities_scattering,
    frequencies,
    weights_gauss,
    cloud_wavelengths,
    cloud_f_sed,
    cloud_anisotropic_scattering_opacities,
    cloud_absorption_opacities,
    photospheric_cloud_optical_depths,
    scattering_in_emission,
):
    relative_cloud_scaling_factor = None
    total_absorption_opacities = _total_absorption_opacity(opacities)

    if scattering_in_emission:
        continuum_opacities_scattering_emission = jnp.asarray(
            continuum_opacities_scattering, dtype=WORKING_JAX_DTYPE
        )
    else:
        continuum_opacities_scattering_emission = jnp.zeros(
            continuum_opacities_scattering.shape, dtype=WORKING_JAX_DTYPE
        )

    _mass_fractions_1 = jnp.ones((pressures.size, 1), dtype=WORKING_JAX_DTYPE)

    if (
        cloud_anisotropic_scattering_opacities is not None
        and cloud_absorption_opacities is not None
    ):
        # Calculate continuum scattering opacity without clouds
        continuum_opacities_scattering_emission -= (
            cloud_anisotropic_scattering_opacities
        )

        total_absorption_opacities = combine_opacities(
            line_species_mass_fractions=_mass_fractions_1,
            opacities=total_absorption_opacities,
            continuum_opacities=-cloud_absorption_opacities,
        )

    # Calculate optical depths without clouds
    optical_depths, photon_destruction_probabilities = _compute_total_optical_depths(
        pressures=pressures,
        reference_gravity=reference_gravity,
        opacities=total_absorption_opacities,
        continuum_opacities_scattering=continuum_opacities_scattering_emission,
        scattering_in_emission=scattering_in_emission,
    )

    # If there are no cloud opacities, return the optical depths without clouds
    if (
        cloud_anisotropic_scattering_opacities is None
        or cloud_absorption_opacities is None
    ):
        return optical_depths, photon_destruction_probabilities, None

    cloud_only_absorption_opacities = combine_opacities(
        line_species_mass_fractions=_mass_fractions_1,
        opacities=jnp.zeros_like(total_absorption_opacities, dtype=WORKING_JAX_DTYPE),
        continuum_opacities=cloud_absorption_opacities,
    )

    # Calculate optical depth of cloud only
    total_tau_cloud, _ = _compute_total_optical_depths(
        pressures=pressures,
        reference_gravity=reference_gravity,
        opacities=cloud_only_absorption_opacities,
        continuum_opacities_scattering=cloud_anisotropic_scattering_opacities,
        scattering_in_emission=scattering_in_emission,
    )

    # Calculate photospheric position of atmo without cloud, determine cloud optical depth there, compare to
    # photospheric_cloud_optical_depths, calculate scaling ratio
    median = True

    if cloud_wavelengths is None:
        # Use the full wavelength range for calculating the median
        # optical depth of the clouds
        wavelengths_select = jnp.ones(frequencies.shape[0], dtype=bool)

    else:
        # Use a smaller wavelength range for the median optical depth
        # The units of cloud_wavelengths are converted from micron to cm
        # Unit conversion is done multiple times to preserve memory
        wavelengths_select = jnp.logical_and(
            jnp.greater_equal(
                frequency2wavelength(frequencies), cloud_wavelengths[0] * 1e-4
            ),
            jnp.less_equal(
                frequency2wavelength(frequencies), cloud_wavelengths[1] * 1e-4
            ),
        )

    def _integrate_over_frequencies(optical_depth):
        if median:
            # Masking with NaN instead of boolean indexing keeps the shapes static under jit
            return jnp.nanmedian(
                jnp.where(wavelengths_select[:, jnp.newaxis], optical_depth, jnp.nan), axis=0
            )

        return (
            jnp.sum(
                (optical_depth[1:, :] + optical_depth[:-1, :])
                * jnp.diff(frequencies)[..., jnp.newaxis],
                axis=0,
            )
            / (frequencies[-1] - frequencies[0])
            / 2.0
        )

    # Calculate the cloud-free and cloud-only optical depths per wavelength.
    # Written as an explicit contraction (einsum) rather than as
    # jnp.sum(weights[..., None, None] * tau, axis=0): the latter spelling is
    # miscompiled by XLA/CPU (jax 0.10.2, Apple silicon) in some of the
    # programs calculate_flux() generates, returning numbers that are wrong by
    # orders of magnitude. See the matching comment in
    # Radtrans._interpolate_photosphere_radius_profile().
    optical_depth_integral = _integrate_over_frequencies(
        jnp.einsum("g,gfl->fl", weights_gauss, optical_depths[:, :, 0, :])
    )
    optical_depth_cloud_integral = _integrate_over_frequencies(
        jnp.einsum("g,gfl->fl", weights_gauss, total_tau_cloud[:, :, 0, :])
    )

    # Interpolate the pressure where the optical depth of the cloud-free atmosphere is 1.0
    # (the optical depth increases with pressure, as jnp.interp requires)
    p_phot_clear = jnp.where(
        (jnp.min(optical_depth_integral) < 1) & (1 < jnp.max(optical_depth_integral)),
        jnp.interp(1.0, optical_depth_integral, pressures),
        pressures[-1],
    )

    # Interpolate the optical depth of the cloud-only atmosphere at the pressure of the cloud-free photosphere
    tau_cloud_at_phot_clear = jnp.interp(p_phot_clear, pressures, optical_depth_cloud_integral)

    # Apply cloud scaling
    cloud_scaling_factor = photospheric_cloud_optical_depths / tau_cloud_at_phot_clear

    if cloud_f_sed is not None and len(cloud_f_sed) > 0:
        max_rescaling = 1e100

        for f in cloud_f_sed.keys():
            if jnp.ndim(cloud_f_sed[f]) > 0:
                raise ValueError(
                    "Only scalar cloud_f_sed values are allowed for cloud rescaling!"
                )

            _max_rescaling = 2.0 * (cloud_f_sed[f] + 1.0)
            max_rescaling = jnp.minimum(max_rescaling, _max_rescaling)

        relative_cloud_scaling_factor = cloud_scaling_factor / max_rescaling

    # Get continuum scattering opacity, including clouds:
    continuum_opacities_scattering_emission = (
        continuum_opacities_scattering_emission
        + cloud_scaling_factor * cloud_anisotropic_scattering_opacities
    )

    total_absorption_opacities = combine_opacities(
        line_species_mass_fractions=_mass_fractions_1,
        opacities=total_absorption_opacities,
        continuum_opacities=cloud_scaling_factor * cloud_absorption_opacities,
    )

    # Calculate total optical depth, including clouds
    updated_optical_depths, photon_destruction_probabilities = _compute_total_optical_depths(
        pressures=pressures,
        reference_gravity=reference_gravity,
        opacities=total_absorption_opacities,
        continuum_opacities_scattering=continuum_opacities_scattering_emission,
        scattering_in_emission=scattering_in_emission,
    )

    return (
        updated_optical_depths,
        photon_destruction_probabilities,
        relative_cloud_scaling_factor,
    )
