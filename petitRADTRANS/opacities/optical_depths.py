_file_purpose="Calculating the optical depths and photon destruction probabilities."

from typing import Optional, Tuple, Dict

import jax.numpy as jnp
from interpax import interp1d
from jax import debug
from jax.typing import ArrayLike

from petitRADTRANS.jax_dtype import WORKING_JAX_DTYPE
from petitRADTRANS.opacities.clouds import compute_species_optical_depths
from petitRADTRANS.opacities.continuum import combine_opacities
from petitRADTRANS.physics import frequency2wavelength
from petitRADTRANS.radtrans_core import radiative_transfer_core


def _total_absorption_opacity(
        opacities: ArrayLike,
) -> ArrayLike:
    """
    In case of sum_opacities=True, the third dimension of the opacities (species) has the already summed opacities, so
    only this is taken into account.

    Args:
        opacities: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            (cm^2/g) The opacities in the atmosphere.

    Returns:
        jnp.ndarray, shape (n_g, n_frequencies, 1, n_layers)
            (cm^2/g) The opacity in the atmosphere, but with only one dimension in the species dimension
    """
    return jnp.asarray(opacities[:, :, :1, :], dtype=WORKING_JAX_DTYPE)


def _compute_total_optical_depths_and_pdp(
    pressures: ArrayLike,
    reference_gravity: float,
    opacities: ArrayLike,
    scattering_in_emission: bool,
    continuum_opacities_scattering: Optional[ArrayLike],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Wrapper function to compute the optical depths and photon destruction probabilities.

    Args:
        pressures: jnp.ndarray, shape (n_layers, )
            (barye) The pressures in the atmosphere.
        reference_gravity: float
            (cm/s^2) The reference gravity in the atmosphere.
        opacities: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            (cm^2/g) The opacities in the atmosphere.
        scattering_in_emission: bool
            Whether to include scattering in the emission calculation.
        continuum_opacities_scattering: Optional[ArrayLike]
            (cm^2/g) The continuum scattering opacity in the atmosphere.

    Returns:
        tuple[jax.Array, jax.Array]
            optical_depths: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
                The optical depths in the atmosphere.
            photon_destruction_probabilities: jnp.ndarray, shape (n_g, n_frequencies, n_layers)
                The photon destruction probabilities in the atmosphere.
    """
    optical_depths, photon_destruction_probabilities = (
        radiative_transfer_core.compute_optical_depths_and_pdp_core(
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


def compute_optical_depths_and_pdp(
    pressures: ArrayLike,
    reference_gravity: float,
    opacities: ArrayLike,
    continuum_opacities_scattering: ArrayLike,
    scattering_in_emission: bool,
    sum_opacities: bool,
    photospheric_cloud_optical_depths: Optional[float] = None,
    absorber_present: bool = True,
    **custom_cloud_parameters,
) -> tuple[ArrayLike, ArrayLike, float]:
    """
    Compute the optical depths and photon destruction probabilities.

    Args:
        pressures: jnp.ndarray, shape (n_layers, )
            (barye) The pressures in the atmosphere.
        reference_gravity: float
            (cm/s^2) The reference gravity in the atmosphere.
        opacities: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            (cm^2/g) The opacities in the atmosphere.
        continuum_opacities_scattering: jnp.ndarray, shape (n_g or 1, n_frequencies, n_layers) or (n_frequencies,
                                        n_layers)
            (cm^2/g) The continuum scattering opacities in the atmosphere.
        scattering_in_emission: bool
            Whether to include scattering in the emission calculation.
        sum_opacities: bool
            Whether to sum the opacities or not.
        photospheric_cloud_optical_depths: float, optional
            The photospheric cloud optical depths in the atmosphere. Default is None.
        absorber_present: bool
            Whether an absorber is present in the atmosphere. Default is True.
        custom_cloud_parameters: dict
            Custom cloud parameters to be passed to the cloud opacity calculation.#todo is that correct?

    Returns:
        optical_depths: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            The cumulative optical depths in the atmosphere.
        photon_destruction_probabilities: jnp.ndarray, shape (n_g, n_frequencies, n_layers)
            The photon destruction probabilities in the atmosphere.
        relative_cloud_scaling_factor: Optional[float]
            The relative cloud scaling factor in the atmosphere:
            cloud_scaling_factor / max_rescaling with max_rescaling = 2(min_f_sed + 1).
    """
    photon_destruction_probabilities = None
    relative_cloud_scaling_factor = None

    # Calculate optical depth for the total opacity
    if sum_opacities:
        if photospheric_cloud_optical_depths is not None:
            (
                optical_depths,
                photon_destruction_probabilities,
                relative_cloud_scaling_factor,
            ) = compute_optical_depths_and_pdp_with_photospheric_cloud(
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
                _compute_total_optical_depths_and_pdp(
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
                species_opacities=opacities,
            ),
            dtype=WORKING_JAX_DTYPE,
        )

    return (
        optical_depths,
        photon_destruction_probabilities,
        relative_cloud_scaling_factor,
    )

def compute_optical_depths_and_pdp_with_photospheric_cloud(
    pressures: ArrayLike,
    reference_gravity: float,
    opacities: ArrayLike,
    continuum_opacities_scattering: ArrayLike,
    frequencies: ArrayLike,
    weights_gauss: ArrayLike,
    cloud_wavelengths: ArrayLike,
    cloud_f_sed: Dict,
    cloud_anisotropic_scattering_opacities: ArrayLike,
    cloud_absorption_opacities: ArrayLike,
    photospheric_cloud_optical_depths: float,
    scattering_in_emission: bool,
) -> Tuple[ArrayLike, ArrayLike, float]:
    """
    Compute the optical depths and photon destruction probabilities, including the effect of clouds.

    Args:
        pressures: jnp.ndarray, shape (n_layers, )
            (barye) The pressures in the atmosphere.
        reference_gravity: float
            (cm/s^2) The reference gravity in the atmosphere.
        opacities: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            (cm^2/g) The opacities in the atmosphere.
        continuum_opacities_scattering: jnp.ndarray, shape (n_g or 1, n_frequencies, n_layers) or (n_frequencies,
                                        n_layers)
            (cm^2/g) The continuum scattering opacities in the atmosphere.
        frequencies: jnp.ndarray, shape (n_frequencies,)
            (Hz) The frequencies in the atmosphere.
        weights_gauss: jnp.ndarray, shape (n_gauss,)
            The Gauss-Legendre weights for integration over angles.
        cloud_wavelengths: jnp.ndarray, shape (2,)
            (micron) The wavelength range for cloud calculations.
        cloud_f_sed: dict
            The cloud sedimentation efficiency parameters for each cloud species.
        cloud_anisotropic_scattering_opacities: jnp.ndarray, shape (n_g, n_frequencies, n_layers)
            (cm^2/g) The anisotropic scattering opacities of clouds in the atmosphere.
        cloud_absorption_opacities: jnp.ndarray, shape (n_g, n_frequencies, n_layers)
            (cm^2/g) The absorption opacities of clouds in the atmosphere.
        photospheric_cloud_optical_depths: float
            The photospheric cloud optical depth.
        scattering_in_emission: bool
            Whether to include scattering in the emission calculation.

    Returns:
        tuple[jax.Array, jax.Array, float]
            optical_depths: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
                The cumulative optical depths in the atmosphere, including clouds.
            photon_destruction_probabilities: jnp.ndarray, shape (n_layers,)
                The photon destruction probabilities in the atmosphere, including clouds.
            relative_cloud_scaling_factor: Optional[float]
                The relative cloud scaling factor in the atmosphere:
                cloud_scaling_factor / max_rescaling with max_rescaling = 2(min_f_sed + 1).
    """
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
    optical_depths, photon_destruction_probabilities = _compute_total_optical_depths_and_pdp(
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
        return (optical_depths, photon_destruction_probabilities, None)

    cloud_only_absorption_opacities = combine_opacities(
        line_species_mass_fractions=_mass_fractions_1,
        opacities=jnp.zeros_like(total_absorption_opacities, dtype=WORKING_JAX_DTYPE),
        continuum_opacities=cloud_absorption_opacities,
    )

    # Calculate optical depth of cloud only
    total_tau_cloud, _ = _compute_total_optical_depths_and_pdp(
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

    # todo Evert: I changed the interpolation here because interp1d from interpax takes the value(s) where to
    #  interpolate directly. Also, it was not jitable.
    # Interpolate the pressure where the optical
    # depth of cloud-free atmosphere is 1.0
    tau_one_bracketed = jnp.logical_and(
        jnp.min(optical_depth_integral) < 1.0,
        jnp.max(optical_depth_integral) > 1.0,
    )
    p_phot_clear = jnp.where(
        tau_one_bracketed,
        jnp.interp(1.0, optical_depth_integral, pressures),
        pressures[-1],
    )

    # Interpolate the optical depth of the cloud-only atmosphere at the pressure of the cloud-free photosphere
    tau_cloud_at_phot_clear = interp1d(p_phot_clear, pressures, optical_depth_cloud_integral, method="linear")

    # Apply cloud scaling
    cloud_scaling_factor = photospheric_cloud_optical_depths / tau_cloud_at_phot_clear

    if len(cloud_f_sed) > 0:
        max_rescaling = 1e100
        # cloud_f_sed values are traced scalars under jit; scalar validation
        # happens outside the jitted kernel.
        min_f_sed = jnp.min(jnp.asarray(list(cloud_f_sed.values())))
        _max_rescaling = 2.0 * (min_f_sed + 1.0)
        max_rescaling = jnp.minimum(max_rescaling, _max_rescaling) # safety
        relative_cloud_scaling_factor = cloud_scaling_factor / max_rescaling

        # todo Evert: should we keep that? Then we have to move it outside of jit
        #if isinstance(cloud_f_sed[f], jnp.ndarray):
        #    raise ValueError(
        #        "Only scalar cloud_f_sed values are allowed for cloud rescaling!"
        #    )

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
    updated_optical_depths, photon_destruction_probabilities = _compute_total_optical_depths_and_pdp(
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
