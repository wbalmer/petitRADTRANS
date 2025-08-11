import numpy as np
from jax import jit, vmap, lax, debug
import jax.numpy as jnp
import jax.scipy.special as jsp # For gamma function
from jax.scipy.ndimage import map_coordinates

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.radtrans_core.radiative_transfer_core import integrate_opacities
from petitRADTRANS.radtrans_core.linalg import linear_fit


@jit
def compute_turbulent_settling_speed(
    x, reference_gravity, rho, clouds_particles_densities,
    temperatures, mean_molar_masses
):
    """
    Calculates the turbulent settling speed of cloud particles.

    Args:
        x: Particle radius (cm).
        reference_gravity: Reference surface gravity (cm.s-2).
        rho: Atmospheric density at the particle's location (g.cm-3).
        clouds_particles_densities: Density of the cloud particle material (g.cm-3).
        temperatures: Atmospheric temperature at the particle's location (K).
        mean_molar_masses: Mean molar mass of the atmosphere at the particle's location (g/mol).
                           Note: In Fortran, this was likely in amu, converted by cst_amu.
                           Here, ensure it's consistent or adjust cst_amu usage.

    Returns:
        Turbulent settling speed (cm.s-1).
    """
    # constants
    diameter = 2.827e-8  # TODO: find out what this constant is
    epsilon = 59.7 * cst.kB

    # N_Knudsen: Knudsen number
    N_Knudsen = (mean_molar_masses * cst.amu) / \
                (jnp.pi * rho * diameter**2.0 * x)

    # psi: Cunningham slip correction factor
    psi = 1.0 + N_Knudsen * (1.249 + 0.42 * jnp.exp(-0.87 * N_Knudsen))

    # eta: Dynamic viscosity of the gas (g.cm-1.s-1)
    # The original Fortran formula for eta:
    # eta = 15d0/16d0*sqrt(cst_pi*2d0*cst_amu*cst_k*temperatures)/(cst_pi*d**2d0) &
    #       *(cst_k*temperatures/epsilon)**0.16d0/1.22d0
    eta = (15.0 / 16.0) * jnp.sqrt(jnp.pi * 2.0 * cst.amu * cst.kB * temperatures) / \
          (jnp.pi * diameter**2.0) * \
          (cst.kB * temperatures / epsilon)**0.16 / 1.22

    # CdNreSq: Product of drag coefficient and Reynolds number squared
    CdNreSq = 32.0 * rho * reference_gravity * x**3.0 * \
              (clouds_particles_densities - rho) / (3.0 * eta**2.0)

    # Nre: Reynolds number (empirical fit)
    # Note: log is natural logarithm (ln) in jnp.log, equivalent to Fortran's log
    log_CdNreSq = jnp.log(CdNreSq)
    Nre = jnp.exp(-2.7905 + 0.9209 * log_CdNreSq - 0.0135 * log_CdNreSq**2.0)

    # Cd: Drag coefficient
    # Using lax.cond for conditional assignment based on Nre
    Cd = lax.cond(
        Nre < 1.0,
        lambda _: 24.0,
        lambda _: lax.cond(
            Nre > 1.0e3,
            lambda __: 0.45,
            lambda ___: CdNreSq / Nre**2.0,
            None
        ),
        None
    )

    # v_settling_visc: Settling speed in the viscous (Stokes) regime
    v_settling_visc = 2.0 * x**2.0 * (clouds_particles_densities - rho) * \
                      psi * reference_gravity / (9.0 * eta)

    # turbulent_settling_speed_ret: Settling speed in the turbulent regime
    turbulent_settling_speed_ret = psi * jnp.sqrt(
        8.0 * reference_gravity * x * (clouds_particles_densities - rho) /
        (3.0 * Cd * rho)
    )

    # Final selection of settling speed
    # if ((Nre < 1d0) .and. (v_settling_visc < turbulent_settling_speed_ret)) then
    #     turbulent_settling_speed_ret = v_settling_visc
    # end if
    turbulent_settling_speed_ret = lax.cond(
        (Nre < 1.0) & (v_settling_visc < turbulent_settling_speed_ret),
        lambda _: v_settling_visc,
        lambda _: turbulent_settling_speed_ret,
        None
    )
    return turbulent_settling_speed_ret


@jit
def compute_cloud_particles_mean_radius(
    reference_gravity,
    atmospheric_densities,
    clouds_particles_densities,
    temperatures,
    mean_molar_masses,
    f_seds,
    cloud_particle_radius_distribution_std,
    eddy_diffusion_coefficients
):
    """
    Calculates the mean radius of cloud particles for each layer and species.
    Translated from Fortran compute_cloud_particles_mean_radius.

    Args:
        reference_gravity: Reference surface gravity (scalar float).
        atmospheric_densities: Atmospheric density (n_layers, float).
        clouds_particles_densities: Density of cloud particle material (n_clouds, float).
        temperatures: Atmospheric temperature (n_layers, float).
        mean_molar_masses: Mean molar mass of atmosphere (n_layers, float).
        f_seds: Sedimentation efficiency parameter (n_layers, n_clouds, float).
        cloud_particle_radius_distribution_std: Standard deviation of log-normal radius distribution (scalar float).
        eddy_diffusion_coefficients: Eddy diffusion coefficient (n_layers, float).

    Returns:
        Mean particle radius (n_layers, n_clouds, float).
    """
    n_layers = atmospheric_densities.shape[0]
    n_clouds = clouds_particles_densities.shape[0]

    # Prepare flattened inputs for vmap
    rho_flat = jnp.repeat(atmospheric_densities, n_clouds)
    cloud_dens_flat = jnp.tile(clouds_particles_densities, n_layers)
    temp_flat = jnp.repeat(temperatures, n_clouds)
    mean_molar_mass_flat = jnp.repeat(mean_molar_masses, n_clouds)

    f_sed_flat = f_seds.flatten()
    eddy_diff_flat = jnp.repeat(eddy_diffusion_coefficients, n_clouds)

    # Vmap the core function over the flattened dimension
    mean_radius_flat = vmap(
        _compute_mean_radius_single,
        in_axes=(None, 0, 0, 0, 0, 0, None, 0)
    )(
        reference_gravity, rho_flat, cloud_dens_flat, temp_flat,
        mean_molar_mass_flat, f_sed_flat, cloud_particle_radius_distribution_std,
        eddy_diff_flat
    )

    # Reshape the result back to (n_layers, n_clouds)
    cloud_particles_mean_radii = mean_radius_flat.reshape((n_layers, n_clouds))

    return cloud_particles_mean_radii


@jit
def compute_cloud_particles_mean_radius_hansen(
    reference_gravity, rho, clouds_particles_densities, temperatures,
    mean_molar_masses, f_seds, clouds_b_hansen, eddy_diffusion_coefficients
):
    """
    Calculates the 'a' parameter for the Hansen distribution for cloud particles.
    Translated from Fortran compute_cloud_particles_mean_radius_hansen.

    Args:
        reference_gravity: Reference surface gravity (scalar float).
        rho: Atmospheric density (n_layers, float).
        clouds_particles_densities: Density of cloud particle material (n_clouds, float).
        temperatures: Atmospheric temperature (n_layers, float).
        mean_molar_masses: Mean molar mass of atmosphere (n_layers, float).
        f_seds: Sedimentation efficiency parameter (n_layers, n_clouds, float).
        clouds_b_hansen: 'b' parameter for the Hansen distribution (n_layers, n_clouds, float).
        eddy_diffusion_coefficients: Eddy diffusion coefficient (n_layers, float).

    Returns:
        'a' parameter for the Hansen distribution (n_layers, n_clouds, float).
    """
    n_layers = rho.shape[0]
    n_clouds = clouds_particles_densities.shape[0]

    # Prepare flattened inputs for vmap
    rho_flat = jnp.repeat(rho, n_clouds)
    cloud_dens_flat = jnp.tile(clouds_particles_densities, n_layers)
    temp_flat = jnp.repeat(temperatures, n_clouds)
    mean_molar_mass_flat = jnp.repeat(mean_molar_masses, n_clouds)
    f_sed_flat = f_seds.flatten()
    b_hansen_flat = clouds_b_hansen.flatten()
    eddy_diff_flat = jnp.repeat(eddy_diffusion_coefficients, n_clouds)

    # Vmap the core function over the flattened dimension
    a_hansen_flat = vmap(
        _compute_mean_radius_hansen_single,
        in_axes=(None, 0, 0, 0, 0, 0, 0, 0)
    )(
        reference_gravity, rho_flat, cloud_dens_flat, temp_flat,
        mean_molar_mass_flat, f_sed_flat, b_hansen_flat, eddy_diff_flat
    )

    # Reshape the result back to (n_layers, n_clouds)
    clouds_a_hansen = a_hansen_flat.reshape((n_layers, n_clouds))

    return clouds_a_hansen


@jit
def compute_cloud_log_normal_particles_distribution_opacities(
    atmosphere_densities,
    clouds_particles_densities,
    clouds_mass_fractions,
    cloud_particles_mean_radii,
    cloud_particles_distribution_std,
    cloud_particles_radii_bins,
    cloud_particles_radii,
    clouds_absorption_opacities,
    clouds_scattering_opacities,
    clouds_particles_asymmetry_parameters
):
    r"""This function reimplements calc_cloud_opas from fortran_radiative_transfer_core.f90.
    For some reason it runs faster in python than in fortran, so we'll use this from now on.
    This function integrates the cloud opacity through the different layers of the atmosphere to get the total
    optical depth, scattering and anisotropic fraction.
    # TODO optical depth or opacity?

    Author: Francois Rozet

    Args:
        atmosphere_densities:
            Density of the atmosphere at each of its layer
        clouds_particles_densities:
            Density of each cloud particles
        clouds_mass_fractions:
            Mass fractions of each cloud at each atmospheric layer
        cloud_particles_mean_radii:
            Mean radius of each cloud particles at each atmospheric layer
        cloud_particles_distribution_std:
            Standard deviation of the log-normal cloud particles distribution
        cloud_particles_radii_bins:
            Bins of the particles cloud radii grid
        cloud_particles_radii:
            Particles cloud radii grid
        clouds_absorption_opacities:
            Cloud absorption opacities (radius grid, wavelength grid, clouds)
        clouds_scattering_opacities:
            Cloud scattering opacities (radius grid, wavelength grid, clouds)
        clouds_particles_asymmetry_parameters:
            Cloud particles asymmetry parameters (radius grid, wavelength grid, clouds)

    Returns:

    """
    n = (  # (n_layers, n_clouds)
            3.0
            * clouds_mass_fractions
            * atmosphere_densities[:, None]
            / (4.0 * jnp.pi * clouds_particles_densities * (cloud_particles_mean_radii ** 3))
            * jnp.exp(-4.5 * jnp.log(cloud_particles_distribution_std) ** 2)
    )

    diff = jnp.log(cloud_particles_radii[:, None, None]) - jnp.log(cloud_particles_mean_radii)
    dn_dr = (  # (n_radii, n_layers, n_clouds)
        n / (
            cloud_particles_radii[:, None, None] * jnp.sqrt(2.0 * jnp.pi)
            * jnp.log(cloud_particles_distribution_std)
        )
        * jnp.exp(-diff ** 2 / (2.0 * jnp.log(cloud_particles_distribution_std) ** 2))
    )

    integrand_scale = (  # (n_radii, n_layers, n_clouds)
            (4.0 * jnp.pi / 3.0)
            * cloud_particles_radii[:, None, None] ** 3
            * clouds_particles_densities
            * dn_dr
    )

    integrand_absorption = integrand_scale[:, None] * clouds_absorption_opacities[:, :, None]
    integrand_scattering = integrand_scale[:, None] * clouds_scattering_opacities[:, :, None]
    integrand_anisotropy = integrand_scattering * (1.0 - clouds_particles_asymmetry_parameters[:, :, None])

    widths = jnp.diff(cloud_particles_radii_bins)[:, None, None, None]  # (n_radii, 1, 1, 1)

    _cloud_absorption_opacities = jnp.sum(integrand_absorption * widths, axis=(0, 3))  # (n_wavelengths, n_layers)
    _cloud_scattering_opacities = jnp.sum(integrand_scattering * widths, axis=(0, 3))  # (n_wavelengths, n_layers)
    cloud_anisotropic_fraction = jnp.sum(integrand_anisotropy * widths, axis=(0, 3))  # (n_wavelengths, n_layers)
    _cloud_scattering_opacities = jnp.where(
        _cloud_scattering_opacities < 1e-200,
        1e-200,
        _cloud_scattering_opacities
        )
    cloud_anisotropic_fraction = jnp.true_divide(
        cloud_anisotropic_fraction,
        _cloud_scattering_opacities
    )

    _cloud_absorption_opacities = _cloud_absorption_opacities / atmosphere_densities
    _cloud_scattering_opacities = _cloud_scattering_opacities / atmosphere_densities

    return _cloud_absorption_opacities, _cloud_scattering_opacities, cloud_anisotropic_fraction


@jit
def interpolate_cloud_opacities(cloud_absorption_opacities, cloud_scattering_opacities,
                                cloud_scattering_reduction_factor, cloud_wavelengths,
                                frequencies_bin_edges):
    #TODO: just read in wavelength centers
    #TODO Read in pressure coords

    edges_wlen = cst.c/frequencies_bin_edges # cm
    bin_centers = edges_wlen[1:] - jnp.diff(edges_wlen)/2.0  # nm

    def interp_index(bin_center):
        idx = jnp.searchsorted(cloud_wavelengths, bin_center, side='left')
        idx = jnp.clip(idx, 1, cloud_wavelengths.shape[0] - 1)
        x0 = cloud_wavelengths[idx - 1]
        x1 = cloud_wavelengths[idx]
        # Linear interpolation for fractional index
        frac = (bin_center - x0) / (x1 - x0 + 1e-30)
        return (idx - 1) + frac

    transformed_wavelength_coords = vmap(interp_index)(bin_centers)  # shape (n_bins,)
    pressure_coords = jnp.linspace(
        0,
        cloud_absorption_opacities.shape[1]-1,
        cloud_absorption_opacities.shape[1],
        dtype=int)
    coordinate_grid = jnp.meshgrid(
        transformed_wavelength_coords,
        pressure_coords,
        indexing='ij'
        )

    cloud_absorption_opacities_output = map_coordinates(
        cloud_absorption_opacities,
        coordinate_grid,
        order=1,
        mode='nearest'
        )
    cloud_scattering_opacities_output = map_coordinates(
        cloud_scattering_opacities,
        coordinate_grid,
        order=1,
        mode='nearest'
        )
    cloud_scattering_reduction_factor_output = map_coordinates(
        cloud_scattering_reduction_factor,
        coordinate_grid,
        order=1,
        mode='nearest'
        )
    
    cloud_anisotropic_extinctions = (
        cloud_absorption_opacities_output
        + (cloud_scattering_opacities_output * cloud_scattering_reduction_factor_output)
        )
    cloud_isotropic_extinctions = cloud_anisotropic_extinctions + cloud_scattering_opacities_output

    return (cloud_absorption_opacities_output,
            cloud_anisotropic_extinctions,
            cloud_scattering_reduction_factor_output,
            cloud_isotropic_extinctions
            )


@jit
def interpolate_cloud_opacities_fortran(cloud_absorption_opacities, cloud_scattering_opacities,
                                cloud_scattering_reduction_factor, cloud_wavelengths,
                                frequencies_bin_edges):
    """Interpolate cloud opacities to actual radiative transfer wavelength grid."""

    n_cloud_wavelengths, n_layers = cloud_absorption_opacities.shape
    n_frequencies_bins = frequencies_bin_edges.shape[0]

    # frequency bin edges - 1 because we're interpolating to the bin centres
    clouds_final_absorption_opacities = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_anisotropic_extinctions = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_isotropic_extinctions = jnp.zeros((n_frequencies_bins - 1, n_layers))
    cloud_scattering_reduction_factor_out = jnp.zeros((n_frequencies_bins - 1, n_layers))

    # Wavelengths
    HIT_border_lamb = cst.c / frequencies_bin_edges

    def loop_body(HIT_i_lamb, carry):
        # Loop over every output frequency bin
        (clouds_final_absorption_opacities,
         cloud_anisotropic_extinctions,
         cloud_isotropic_extinctions,
         cloud_scattering_reduction_factor_out) = carry

        # Find the indices of the cloud wavelengths that are closest to 
        # the HIT border wavelengths for this bin
        intp_index_small_min = jnp.argmin(jnp.abs(cloud_wavelengths - HIT_border_lamb[HIT_i_lamb]))
        intp_index_small_max = jnp.argmin(jnp.abs(cloud_wavelengths - HIT_border_lamb[HIT_i_lamb+1]))

        def integrate_section(start_index, lambda_low, lambda_high, kappa_integ, kappa_scat_integ, red_fac_aniso_integ):
            start_index = start_index.astype(int)
            # One iteration of the integration for the opacities, scatetring and reduction factors.
            kappa_integ += integrate_opacities(
                start_index,
                cloud_absorption_opacities,
                cloud_wavelengths,
                lambda_low,
                lambda_high
                )
            kappa_scat_integ += integrate_opacities(
                start_index,
                cloud_scattering_opacities,
                cloud_wavelengths,
                lambda_low,
                lambda_high
                )
            red_fac_aniso_integ += integrate_opacities(
                start_index,
                cloud_scattering_reduction_factor,
                cloud_wavelengths,
                lambda_low,
                lambda_high
                )
            return kappa_integ, kappa_scat_integ, red_fac_aniso_integ

        def integrate_twice(index_min, index_max, kappa_integ, kappa_scat_integ, red_fac_aniso_integ):
            # Used for the last bins
            # Runs if (intp_index_small_max - intp_index_small_min) == 1
            kappa_integ, kappa_scat_integ, red_fac_aniso_integ = integrate_section(
                index_min,
                HIT_border_lamb[HIT_i_lamb],
                cloud_wavelengths[index_min+1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            kappa_integ, kappa_scat_integ, red_fac_aniso_integ = integrate_section(
                index_max,
                cloud_wavelengths[index_max],
                HIT_border_lamb[HIT_i_lamb + 1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            return kappa_integ, kappa_scat_integ, red_fac_aniso_integ

        def inner_loop_body(carry):
            # Used between the first and the last bins
            new_small_ind, kappa_integ, kappa_scat_integ, red_fac_aniso_integ = carry
            kappa_integ, kappa_scat_integ, red_fac_aniso_integ = integrate_section(
                new_small_ind,
                cloud_wavelengths[new_small_ind],
                cloud_wavelengths[new_small_ind+1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            return new_small_ind + 1, kappa_integ, kappa_scat_integ, red_fac_aniso_integ

        def integrate_and_loop(index_min, index_max, kappa_integ, kappa_scat_integ, red_fac_aniso_integ):
            # Used between the first and the last bins
            # Runs while (intp_index_small_max - intp_index_small_min) != (0 or 1)
            kappa_integ, kappa_scat_integ, red_fac_aniso_integ = integrate_section(
                index_min,
                HIT_border_lamb[HIT_i_lamb],
                cloud_wavelengths[index_min+1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            _, kappa_integ, kappa_scat_integ, red_fac_aniso_integ = lax.while_loop(
                            lambda val: val[0] < index_max,
                            inner_loop_body,
                            (index_min + 1,
                             kappa_integ,
                             kappa_scat_integ,
                             red_fac_aniso_integ)
                            )
            kappa_integ, kappa_scat_integ, red_fac_aniso_integ = integrate_section(
                index_max,
                cloud_wavelengths[index_max],
                HIT_border_lamb[HIT_i_lamb+1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            return kappa_integ, kappa_scat_integ, red_fac_aniso_integ

        # Initialize the arrays for the integrated values
        kappa_integ = jnp.zeros(n_layers)
        kappa_scat_integ = jnp.zeros(n_layers)
        red_fac_aniso_integ = jnp.zeros(n_layers)

        # Loop over the cloud opacity wavelength bins
        kappa_integ, kappa_scat_integ, red_fac_aniso_integ = lax.cond(
            (intp_index_small_max - intp_index_small_min) == 0,
            lambda: integrate_section(
                intp_index_small_min,
                HIT_border_lamb[HIT_i_lamb],
                HIT_border_lamb[HIT_i_lamb + 1],
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ),
            lambda: lax.cond(
                (intp_index_small_max - intp_index_small_min) == 1,
                integrate_twice,
                integrate_and_loop,
                intp_index_small_min,
                intp_index_small_max,
                kappa_integ,
                kappa_scat_integ,
                red_fac_aniso_integ)
            )

        # Normalisation to bin width
        kappa_integ = kappa_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        kappa_scat_integ = kappa_scat_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])
        red_fac_aniso_integ = red_fac_aniso_integ / (HIT_border_lamb[HIT_i_lamb + 1] - HIT_border_lamb[HIT_i_lamb])

        # Add the integrated values to the final arrays
        clouds_final_absorption_opacities = clouds_final_absorption_opacities.at[HIT_i_lamb, :].add(kappa_integ)
        cloud_anisotropic_extinctions =\
            cloud_anisotropic_extinctions.at[HIT_i_lamb, :].add(kappa_integ + (kappa_scat_integ * red_fac_aniso_integ))
        cloud_isotropic_extinctions =\
            cloud_isotropic_extinctions.at[HIT_i_lamb, :].add(kappa_integ + kappa_scat_integ)
        cloud_scattering_reduction_factor_out =\
            cloud_scattering_reduction_factor_out.at[HIT_i_lamb, :].add(red_fac_aniso_integ)

        return (clouds_final_absorption_opacities,
                cloud_anisotropic_extinctions,
                cloud_isotropic_extinctions,
                cloud_scattering_reduction_factor_out
                )

    carry = (clouds_final_absorption_opacities,
             cloud_anisotropic_extinctions,
             cloud_isotropic_extinctions,
             cloud_scattering_reduction_factor_out)

    # Loop over the output frequency bins
    (clouds_final_absorption_opacities,
     cloud_anisotropic_extinctions,
     cloud_isotropic_extinctions,
     cloud_scattering_reduction_factor_out) = lax.fori_loop(0, n_frequencies_bins - 1, loop_body, carry)

    return (clouds_final_absorption_opacities,
            cloud_anisotropic_extinctions,
            cloud_scattering_reduction_factor_out,
            cloud_isotropic_extinctions)


@jit
def particle_radius(
    x1, x2, reference_gravity, rho, clouds_particles_densities,
    temperatures, mean_molar_masses, w_star
):
    """
    Finds the particle radius where its turbulent settling speed matches w_star,
    using a bisection-like method.

    Args:
        x1: Lower bound for particle radius search (cm).
        x2: Upper bound for particle radius search (cm).
        reference_gravity: Reference surface gravity (cm.s-2).
        rho: Atmospheric density (g.cm-3).
        clouds_particles_densities: Density of cloud particle material (g.cm-3).
        temperatures: Atmospheric temperature (K).
        mean_molar_masses: Mean molar mass of atmosphere (g/mol).
        w_star: Target settling speed (cm.s-1).

    Returns:
        Particle radius (cm) or 1e-17 if root not bracketed or max iterations reached.
    """
    ITMAX = 1000  # Maximum number of iterations
    TOLERANCE = 1e-9 # Convergence tolerance for 'del'

    a = x1
    b = x2

    fa_val = compute_turbulent_settling_speed(
        a, reference_gravity, rho, clouds_particles_densities,
        temperatures, mean_molar_masses
    )
    fa = fa_val - w_star

    fb_val = compute_turbulent_settling_speed(
        b, reference_gravity, rho, clouds_particles_densities,
        temperatures, mean_molar_masses
    )
    fb = fb_val - w_star

    root_not_bracketed = ((fa > 0.0) & (fb > 0.0)) | ((fa < 0.0) & (fb < 0.0))

    # Store particle parameters for the loop
    particle_params = (reference_gravity, rho, clouds_particles_densities, temperatures, mean_molar_masses)

    # Initial state for the while loop
    # iter_count, a, b, fa, fb, w_star, del_val, max_iter, particle_params
    initial_val = (0, a, b, fa, fb, w_star, 1.0, ITMAX, particle_params)

    def condition_fun(val):
        iter_count, _, _, _, _, _, del_val, max_iter, _ = val
        return (jnp.abs(del_val) >= TOLERANCE) & (iter_count < max_iter)

    final_val = lax.while_loop(condition_fun, particle_radius_bisection_iteration, initial_val)

    iter_count_final, a_final, b_final, fa_final, fb_final, _, _, _, _ = final_val

    # Recalculate c from the final a and b, as it would be in the Fortran code's scope
    c_final = lax.cond(
        jnp.abs(jnp.log10(a_final / b_final)) > 1.0,
        lambda: 10.0**(jnp.log10(a_final * b_final) / 2.0),
        lambda: (a_final + b_final) / 2.0
    )

    # Handle the case where the root was not bracketed initially
    result_radius = lax.cond(
        root_not_bracketed,
        lambda: 1e-17,
        lambda: c_final  # c_final is the 'c' from the last iteration
    )

    return result_radius


@jit
def particle_radius_bisection_iteration(val):
    """
    Performs one iteration of the bisection method for particle_radius.
    Needed for lax.while_loop.
    """
    iter_count, a, b, fa, fb, w_star, tol, max_iter, particle_params = val
    reference_gravity, rho, clouds_particles_densities, temperatures, mean_molar_masses = particle_params

    # Choose c based on the log-midpoint or arithmetic midpoint
    c = lax.cond(
        jnp.abs(jnp.log10(a / b)) > 1.0,
        lambda: 10.0**(jnp.log10(a * b) / 2.0),
        lambda: (a + b) / 2.0
    )

    fc_val = compute_turbulent_settling_speed(
        c, reference_gravity, rho, clouds_particles_densities,
        temperatures, mean_molar_masses
    )
    fc = fc_val - w_star

    condition = jnp.logical_or(jnp.logical_and((fc > 0.0), (fa > 0.0)),
                               jnp.logical_and((fc < 0.0), (fa < 0.0)))

    new_a = lax.cond(condition, lambda: c, lambda: a)
    new_fa = lax.cond(condition, lambda: fc, lambda: fa)
    new_b = lax.cond(condition, lambda: b, lambda: c)
    new_fb = lax.cond(condition, lambda: fb, lambda: fc)
    del_val = lax.cond(
        condition,
        lambda: 2.0 * jnp.abs(a - c) / (a + b),  # a was updated
        lambda: 2.0 * jnp.abs(b - c) / (a + b)   # b was updated
    )

    return iter_count + 1, new_a, new_b, new_fa, new_fb, w_star, del_val, max_iter, particle_params


# --- Core Functions for Vmapping ---
@jit
def _compute_mean_radius_single(
    ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i, f_sed_ij, std_dev, eddy_diff_i
):
    """
    Core function for compute_cloud_particles_mean_radius, operating on single
    layer and species values.
    """
    N_fit = 100

    # H = cst_k*temperatures/(mean_molar_masses*cst_amu*reference_gravity)
    H_i = cst.kB * temp_i / (mean_molar_mass_i * cst.amu * ref_g)
    # w_star = eddy_diffusion_coefficients/H
    w_star_i = eddy_diff_i / H_i

    # r_w(i_str,i_spec) = particle_radius(...)
    r_w_ij = particle_radius(
        1e-16, 1e2, ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i, w_star_i
    )

    # if (r_w(i_str,i_spec) > 1d-16) then
    def compute_valid_radius(r_w_val):
        # if (f_seds(i_str,i_spec) > 1d0) then
        def f_sed_gt_1():
            std_dev_clamped = jnp.maximum(std_dev, 1.0001)
            rad_start = r_w_val / std_dev_clamped
            rad_end = r_w_val
            # Generate N_fit points linearly between rad_start and rad_end
            rad_vals = jnp.linspace(rad_start, rad_end, N_fit)
            return rad_vals

        # else
        def f_sed_le_1():
            std_dev_clamped = jnp.maximum(std_dev, 1.0001)
            rad_start = r_w_val
            rad_end = r_w_val * std_dev_clamped
            # Generate N_fit points linearly between rad_start and rad_end
            rad_vals = jnp.linspace(rad_start, rad_end, N_fit)
            return rad_vals

        rad = lax.cond(f_sed_ij > 1.0, f_sed_gt_1, f_sed_le_1)

        # call compute_turbulent_settling_speed(...)
        # This needs to be mapped over the `rad` array
        vel = vmap(
            lambda r: compute_turbulent_settling_speed(
                r, ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i
            )
        )(rad)

        # call linear_fit(log(rad), log(vel/w_star(i_str)), N_fit, a, b)
        # Handle potential log of non-positive values
        log_rad = jnp.log(jnp.maximum(rad, 1e-30)) # Clamp to avoid log(0) or log(neg)
        log_vel_ratio = jnp.log(jnp.maximum(vel / w_star_i, 1e-30)) # Clamp

        a, b = linear_fit(log_rad, log_vel_ratio)

        # alpha(i_str,i_spec) = b
        alpha_ij = b
        # r_w(i_str,i_spec) = exp(-a/b)
        # Handle potential division by zero if b is zero
        r_w_refined_ij = lax.cond(
            jnp.abs(alpha_ij) < 1e-15,
            lambda: r_w_val,
            lambda: jnp.exp(-a / alpha_ij)
        )

        # Handle potential division by zero for 1d0/alpha
        exponent = lax.cond(
             jnp.abs(alpha_ij) < 1e-15,
             lambda: jnp.inf * jnp.sign(1.0),  # 1/0 -> inf, sign depends on sign of 1.0
             lambda: 1.0 / alpha_ij
        )
        # Handle potential issues with f_sed_ij**inf
        f_sed_term = lax.cond(
            jnp.isinf(exponent),
            lambda: lax.cond(
                f_sed_ij > 1.0, lambda: jnp.inf,
                lambda: lax.cond(f_sed_ij < 1.0, lambda: 0.0, lambda: 1.0) # 0**inf is 0, 1**inf is 1
            ),
            lambda: f_sed_ij**exponent
        )

        # Handle potential log of non-positive std_dev
        log_std_dev_sq = lax.cond(
            std_dev > 0,
            lambda: jnp.log(std_dev)**2.0,
            lambda: jnp.nan  # Handle invalid std_dev
        )

        mean_radius_ij = lax.cond(
            jnp.isnan(log_std_dev_sq),
            lambda: jnp.nan,  # Propagate NaN
            lambda: r_w_refined_ij * f_sed_term * jnp.exp(-(alpha_ij + 6.0) / 2.0 * log_std_dev_sq)
        )

        return mean_radius_ij

    # else (r_w <= 1d-16)
    def handle_invalid_radius():
        # cloud_particles_mean_radii(i_str,i_spec) = 1d-17
        # alpha(i_str,i_spec) = 1d0 # alpha is not returned, but set in Fortran
        return 1e-17

    mean_radius_ij = lax.cond(
        r_w_ij > 1e-16,
        lambda: compute_valid_radius(r_w_ij),
        handle_invalid_radius
    )

    # Handle potential NaN results from valid radius calculation
    return lax.cond(
        jnp.isnan(mean_radius_ij),
        lambda: 1e-17,  # Default to small value if calculation failed
        lambda: mean_radius_ij
    )

@jit
def _compute_mean_radius_hansen_single(
    ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i, f_sed_ij, b_hansen_ij, eddy_diff_i
):
    """
    Core function for compute_cloud_particles_mean_radius_hansen, operating on single
    layer and species values.
    """
    # Constants needed here (re-defined for clarity in vmapped function)
    x_gamma_max = 170.0  # gamma(x_gamma_max) >~ huge(0d0)
    N_fit = 100

    # H = cst_k * temperatures / (mean_molar_masses * cst_amu * reference_gravity)
    H_i = cst.kB * temp_i / (mean_molar_mass_i * cst.amu * ref_g)
    # w_star = eddy_diffusion_coefficients / H
    w_star_i = eddy_diff_i / H_i
    # x_gamma = 1d0 + 1d0 / clouds_b_hansen
    # Handle potential division by zero for 1/b_hansen_ij
    x_gamma_ij = lax.cond(
        jnp.abs(b_hansen_ij) < 1e-15,
        lambda: jnp.inf * jnp.sign(1.0),  # 1/0 -> inf
        lambda: 1.0 + 1.0 / b_hansen_ij
    )

    # r_w(i_str,i_spec) = particle_radius(...)
    r_w_ij = particle_radius(
        1e-16, 1e2, ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i, w_star_i
    )

    # if (r_w(i_str,i_spec) > 1d-16) then
    def compute_valid_radius(r_w_val):
        # if (f_seds(i_str,i_spec) > 1d0) then
        def f_sed_gt_1():
            rad_start = r_w_val * b_hansen_ij
            rad_end = r_w_val
            # Generate N_fit points linearly between rad_start and rad_end
            rad_vals = jnp.linspace(rad_start, rad_end, N_fit)
            return rad_vals

        # else
        def f_sed_le_1():
            rad_end = lax.cond(
                jnp.abs(b_hansen_ij) < 1e-15,
                lambda: jnp.inf * jnp.sign(r_w_val),  # r_w / 0 -> inf
                lambda: r_w_val / b_hansen_ij
            )
            rad_start = r_w_val

            # Generate N_fit points linearly between rad_start and rad_end
            rad_vals = jnp.linspace(rad_start, rad_end, N_fit)
            return rad_vals

        rad = lax.cond(f_sed_ij > 1.0, f_sed_gt_1, f_sed_le_1)

        # call compute_turbulent_settling_speed(...)
        # This needs to be mapped over the `rad` array
        vel = vmap(
            lambda r: compute_turbulent_settling_speed(
                r, ref_g, rho_i, cloud_dens_j, temp_i, mean_molar_mass_i
            )
        )(rad)

        # call linear_fit(log(rad), log(vel/w_star(i_str)), N_fit, a, b)
        # Handle potential log of non-positive values
        log_rad = jnp.log(jnp.maximum(rad, 1e-30))  # Clamp to avoid log(0) or log(neg)
        log_vel_ratio = jnp.log(jnp.maximum(vel / w_star_i, 1e-30))  # Clamp

        a, b = linear_fit(log_rad, log_vel_ratio)

        # alpha(i_str, i_spec) = b
        alpha_ij = b
        # r_w(i_str,i_spec) = exp(-a / b)
        # Handle potential division by zero if b is zero
        r_w_refined_ij = lax.cond(
            jnp.abs(alpha_ij) < 1e-15,
            lambda: r_w_val,
            lambda: jnp.exp(-a / alpha_ij)
        )

        # Calculate clouds_a_hansen
        # if (x_gamma(i_str, i_spec) + alpha(i_str, i_spec) < x_gamma_max) then
        def use_gamma_function():
            # Handle potential division by zero for 1d0/alpha
            exponent = lax.cond(
                 jnp.abs(alpha_ij) < 1e-15,
                 lambda: jnp.inf * jnp.sign(1.0),
                 lambda: 1.0 / alpha_ij
            )

            term1 = b_hansen_ij ** (-1.0 * alpha_ij)
            term2 = r_w_refined_ij ** alpha_ij
            term3 = f_sed_ij
            term4 = (b_hansen_ij ** 3.0 * b_hansen_ij ** alpha_ij - b_hansen_ij + 1.0)

            # Handle potential issues with gamma arguments
            gamma_arg1 = x_gamma_ij
            gamma_arg2 = x_gamma_ij + alpha_ij

            term5_numerator = lax.cond(
                gamma_arg1 > 0, lambda: jsp.gamma(gamma_arg1), lambda: jnp.nan 
            )
            term5_denominator_factor1 = (b_hansen_ij * alpha_ij + 2.0 * b_hansen_ij + 1.0)
            term5_denominator_factor2 = lax.cond(
                gamma_arg2 > 0, lambda: jsp.gamma(gamma_arg2), lambda: jnp.nan 
            )
            term5_denominator = term5_denominator_factor1 * term5_denominator_factor2

            # Handle potential division by zero or NaN in term5_denominator
            term5 = lax.cond(
                jnp.isnan(term5_numerator) | jnp.isnan(term5_denominator) | (jnp.abs(term5_denominator) < 1e-15),
                lambda: jnp.nan,  # Propagate NaN or handle division by zero
                lambda: term5_numerator / term5_denominator
            )

            base = term1 * term2 * term3 * term4 * term5

            # Handle potential issues with base**exponent or NaN base
            a_hansen_ij = lax.cond(
                jnp.isnan(base) | jnp.isinf(exponent),
                lambda: lax.cond(
                    jnp.isnan(base), lambda: jnp.nan,  # Propagate NaN
                    lambda: lax.cond(  # Handle base**inf
                        base > 1.0, lambda: jnp.inf,
                        lambda: lax.cond(base < 1.0, lambda: 0.0, lambda: 1.0)
                    )
                ),
                lambda: base**exponent
            )

            return a_hansen_ij

        # else  ! to avoid overflow, approxiate gamma(x) / gamma(x + a) by x ** -a (from Stirling formula)
        def use_gamma_approximation():
            # Handle potential division by zero for 1d0/alpha
            exponent = lax.cond(
                 jnp.abs(alpha_ij) < 1e-15,
                 lambda: jnp.inf * jnp.sign(1.0),
                 lambda: 1.0 / alpha_ij
            )

            term1 = b_hansen_ij ** (-1.0 * alpha_ij)
            term2 = r_w_refined_ij ** alpha_ij
            term3 = f_sed_ij
            term4 = (b_hansen_ij ** 3.0 * b_hansen_ij ** alpha_ij - b_hansen_ij + 1.0)

            # Handle potential issues with x_gamma_ij ** (-alpha_ij)
            term5_numerator = lax.cond(
                x_gamma_ij > 0, lambda: x_gamma_ij ** (-alpha_ij), lambda: jnp.nan
            )
            term5_denominator = (b_hansen_ij * alpha_ij + 2.0 * b_hansen_ij + 1.0)

            # Handle potential division by zero or NaN in term5_denominator
            term5 = lax.cond(
                jnp.isnan(term5_numerator) | (jnp.abs(term5_denominator) < 1e-15),
                lambda: jnp.nan,
                lambda: term5_numerator / term5_denominator
            )

            base = term1 * term2 * term3 * term4 * term5

            # Handle potential issues with base**exponent or NaN base
            a_hansen_ij = lax.cond(
                jnp.isnan(base) | jnp.isinf(exponent),
                lambda: lax.cond(
                    jnp.isnan(base), lambda: jnp.nan,
                    lambda: lax.cond(
                        base > 1.0, lambda: jnp.inf,
                        lambda: lax.cond(base < 1.0, lambda: 0.0, lambda: 1.0)
                    )
                ),
                lambda: base**exponent
            )

            return a_hansen_ij

        a_hansen_ij = lax.cond(
            x_gamma_ij + alpha_ij < x_gamma_max,
            use_gamma_function,
            use_gamma_approximation
        )
        return a_hansen_ij

    # else (r_w <= 1d-16)
    def handle_invalid_radius():
        # clouds_a_hansen(i_str, i_spec) = 1d-17
        # alpha(i_str, i_spec) = 1d0 # alpha is not returned, but set in Fortran
        return 1e-17

    a_hansen_ij = lax.cond(
        r_w_ij > 1e-16,
        lambda: compute_valid_radius(r_w_ij),
        handle_invalid_radius
    )

    # Handle potential NaN results from valid radius calculation
    return lax.cond(
        jnp.isnan(a_hansen_ij),
        lambda: 1e-17,
        lambda: a_hansen_ij
    )
