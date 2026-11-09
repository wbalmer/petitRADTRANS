"""Calculating the optical depths and photon destruction probabilities."""

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
            If True, the opacities of the species are already summed in the first species index, and the total optical
            depths and the photon destruction probabilities are calculated. If False, the optical depths of each species
            are calculated, and the photon destruction probabilities are None.
        photospheric_cloud_optical_depths: float, optional
            If not None, the cloud opacities are scaled so that the cloud optical depth at the photosphere of the
            cloud-free atmosphere is this value (only if sum_opacities is True). Default is None.
        absorber_present: bool
            Whether an absorber is present in the atmosphere. Default is True.
        custom_cloud_parameters: dict
            Additional arguments of compute_optical_depths_and_pdp_with_photospheric_cloud, used only if
            photospheric_cloud_optical_depths is not None: frequencies, weights_gauss, cloud_wavelengths, cloud_f_sed,
            cloud_anisotropic_scattering_opacities and cloud_absorption_opacities.

    Returns:
        optical_depths: jnp.ndarray, shape (n_g, n_frequencies, n_species, n_layers)
            The cumulative optical depths in the atmosphere. If sum_opacities is True, only the first species index
            is used (n_species = 1).
        photon_destruction_probabilities: jnp.ndarray, shape (n_g, n_frequencies, n_layers), or None
            The photon destruction probabilities in the atmosphere; None if sum_opacities is False.
        relative_cloud_scaling_factor: Optional[float]
            The relative cloud scaling factor in the atmosphere:
            cloud_scaling_factor / max_rescaling with max_rescaling = 2(min_f_sed + 1).
            None if the clouds are not scaled, or if cloud_f_sed is empty.
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
        weights_gauss: jnp.ndarray, shape (n_g,)
            The weights of the correlated-k g-points, used to average the optical depths over g.
        cloud_wavelengths: jnp.ndarray, shape (2,), or None
            (micron) The wavelength range over which the median optical depths are taken. If None, the full wavelength
            range is used.
        cloud_f_sed: dict, or None
            The scalar sedimentation efficiency of each cloud species. Its minimum sets the maximum rescaling,
            2 * (min_f_sed + 1), used to calculate relative_cloud_scaling_factor.
        cloud_anisotropic_scattering_opacities: jnp.ndarray, shape (n_frequencies, n_layers)
            (cm^2/g) The anisotropic scattering opacities of clouds in the atmosphere.
        cloud_absorption_opacities: jnp.ndarray, shape (n_frequencies, n_layers)
            (cm^2/g) The absorption opacities of clouds in the atmosphere.
        photospheric_cloud_optical_depths: float
            The cloud optical depth at the photosphere (optical depth of 1) of the cloud-free atmosphere, taken as the
            median over wavelengths, to which the cloud opacities are scaled.
        scattering_in_emission: bool
            Whether to include scattering in the emission calculation.

    Returns:
        tuple[jax.Array, jax.Array, float]
            optical_depths: jnp.ndarray, shape (n_g, n_frequencies, 1, n_layers)
                The cumulative optical depths in the atmosphere, including clouds.
            photon_destruction_probabilities: jnp.ndarray, shape (n_g, n_frequencies, n_layers)
                The photon destruction probabilities in the atmosphere, including clouds.
            relative_cloud_scaling_factor: Optional[float]
                The relative cloud scaling factor in the atmosphere:
                cloud_scaling_factor / max_rescaling with max_rescaling = 2(min_f_sed + 1).
                None if cloud_f_sed is empty or None, or if there are no cloud opacities.

    Raises:
        ValueError: if a cloud_f_sed value is not a scalar.
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

    # Interpolate the pressure where the optical depth of the cloud-free atmosphere is 1.0
    # (the optical depth increases with pressure, as jnp.interp requires)
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

    if cloud_f_sed is not None and len(cloud_f_sed) > 0:
        max_rescaling = 1e100

        for species, f_sed in cloud_f_sed.items():
            # The number of dimensions is static, so this check also works under jit, where f_sed is traced
            if jnp.ndim(f_sed) > 0:
                raise ValueError(
                    f"Only scalar cloud_f_sed values are allowed for cloud rescaling, "
                    f"but cloud_f_sed of '{species}' has shape {jnp.shape(f_sed)}"
                )

            max_rescaling = jnp.minimum(max_rescaling, 2.0 * (f_sed + 1.0))

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
