"""Models Module

This module contains a set of functions that generate the spectra used
in the petitRADTRANS retrieval. This includes setting up the
pressure-temperature structure, the chemistry, and the radiative
transfer to compute the emission or transmission spectrum.

All models must take the same set of inputs:

    prt_object : petitRADTRANS.RadTrans
        This is the pRT object that is used to compute the spectrum
        It must be fully initialized prior to be used in the model function
    parameters : dict
        A dictionary of Parameter objects. The naming of the parameters
        must be consistent between the Priors and the model function you
        are using.
    PT_plot_mode : bool
        If this argument is True, the model function should return the pressure
        and temperature arrays before computing the flux.
    adaptive_mesh_refinement : bool
        If this parameter is True, your model should allow for reshaping of the
        pressure and temperature arrays based on the position of the clouds or
        the location of the photosphere, increasing the resolution where required.
        For example, using the fixed_length_adaptive_mesh_refinement function defined below.
"""
# TODO replace by SpectralModel function
import jax.numpy as jnp

from petitRADTRANS.chemistry import clouds
from petitRADTRANS.chemistry.core import get_abundances
from petitRADTRANS.physics import (
    flux2irradiance,
    planck_function_cm
)
from petitRADTRANS.profiles import (
    cubic_spline_temperature_profile,
    linear_spline_temperature_profile,
    dtdp_temperature_profile,
    madhusudhan_seager_temperature_profile,
    power_law_temperature_profile,
    temperature_profile_function_guillot_global,
    temperature_profile_function_isothermal,
    spline_eddington_adiabatic_temperature_profile
)
from petitRADTRANS.planet import Planet
from petitRADTRANS.retrieval.utils import get_calculate_flux_return_values
# Global constants to reduce calculations and initializations.
#PGLOBAL = jnp.logspace(-6, 3, 1000)


def _compute_gravity(parameters):
    if 'log_g' in parameters.keys() and 'mass' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        planet_radius, _, _ = Planet.reference_gravity2radius(
            reference_gravity=gravity,
            mass=parameters['mass'].value
        )
    elif 'log_g' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        planet_radius = parameters['planet_radius'].value
    elif 'mass' in parameters.keys() or 'planet_mass' in parameters.keys():
        planet_radius = parameters['planet_radius'].value
        gravity, _, _ = Planet.mass2reference_gravity(
            mass=parameters['mass'].value,
            radius=planet_radius
        )
    else:
        raise KeyError("Pick two of log_g, planet_radius and mass priors!")

    return gravity, planet_radius


def spline_eddington_adiabatic_emission(
        prt_object,
        parameters,
        pt_plot_mode=False,
        adaptive_mesh_refinement=True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    Many of the parameters are optional, but must be used in the correct combination
    with other parameters.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline [K]
                *  T2 : Middle temperature spline [K]
                *  T1 : Outer temperature spline [K]
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  fsed : sedimentation parameter - can be unique to each cloud type by adding _CloudName
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : ArrayLike
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : ArrayLike
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, ArrayLike
            Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    # Use this for debugging.
    # for key, val in parameters.items():
    #    print(key,val.value)

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - parameters['T3'].value)
    t2 = t3 * (1.0 - parameters['T2'].value)
    t1 = t2 * (1.0 - parameters['T1'].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta'].value)) * 1e6) ** (-parameters['alpha'].value)
    gravity, planet_radius = _compute_gravity(parameters)

    # Make the P-T profile
    temp_arr = jnp.array([t1, t2, t3])
    carbon_to_oxygen = 0

    if 'C/O' in parameters.keys():
        carbon_to_oxygen = parameters['C/O'].value
    elif "C" in parameters.keys():
        carbon_to_oxygen = parameters['C'].value / parameters['O'].value

    temperatures = spline_eddington_adiabatic_temperature_profile(  # TODO weird way of calling the function
        (
            temp_arr,
            delta,
            parameters['alpha'].value,
            parameters['T_int'].value,
            p_use,
            parameters['Fe/H'].value,
            carbon_to_oxygen,
            True  # conv
        )
    )

    if 'use_easychem' in parameters.keys():
        temperatures = jnp.clip(temperatures, 40, 40000)

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Calculate the spectrum
    if pressures.shape[0] != prt_object.pressures.shape[0]:
        print("Incorrect output shape!")
        return None, None

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def emission_model_diseq_patchy_clouds(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    This model includes patchy clouds, and requires a unique temperature profile for the
    clear atmosphere regions - ie this is a full two column model!

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline
                *  T2 : Middle temperature spline
                *  T1 : Outer temperature spline
                *  T3_clear : Innermost temperature spline for clear atmosphere
                *  T2_clear : Middle temperature spline for clear atmosphere
                *  T1_clear : Outer temperature spline for clear atmosphere
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  alpha_clear : power law index in tau = delta * press_cgs**alpha for clear atmosphere
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  log_delta_clear : proportionality factor in tau = delta * press_cgs**alpha for clear atmosphere
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two  columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, ArrayLike
            Emission contribution function, relative contributions for each wavelength and pressure level.
            Only returns the contribution of the clear atmosphere component.

    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    # Not sure how to deal with having 2 separate contribution function
    contribution = False

    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - parameters['T3'].value)
    t2 = t3 * (1.0 - parameters['T2'].value)
    t1 = t2 * (1.0 - parameters['T1'].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta'].value)) * 1e6) ** (-parameters['alpha'].value)
    temp_arr = jnp.array([t1, t2, t3])

    t3_clear = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (
                1.0 - parameters['T3_clear'].value)
    t2_clear = t3_clear * (1.0 - parameters['T2_clear'].value)
    t1_clear = t2_clear * (1.0 - parameters['T1_clear'].value)
    temps_clear = jnp.array([t1_clear, t2_clear, t3_clear])
    delta_clear = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta_clear'].value)) * 1e6) ** (
        -parameters['alpha_clear'].value)
    gravity, planet_radius = _compute_gravity(parameters)

    temperatures = spline_eddington_adiabatic_temperature_profile(  # TODO weird way of calling the function
        (
            temp_arr,
            delta,
            parameters['alpha'].value,
            parameters['T_int'].value,
            p_use,
            parameters['Fe/H'].value,
            parameters['C/O'].value,
            True  # conv
        )
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    t_clear = spline_eddington_adiabatic_temperature_profile(
        (  # TODO weird way of calling the function
            temps_clear,
            delta_clear,
            parameters['alpha_clear'].value,
            parameters['T_int'].value,
            PGLOBAL[small_index],
            parameters['Fe/H'].value,
            parameters['C/O'].value,
            True  # conv
        )
    )
    abundances_clear, mmw_clear, small_index_clear, p_bases_clear = get_abundances(
        pressures=PGLOBAL[small_index],
        temperatures=t_clear,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=False
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use
    if pressures.shape[0] != prt_object.pressures.shape[0]:
        return None, None

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    wlen_model, flux, _ = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_particles_radius_distribution=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_cloudy = flux2irradiance(
        f_lambda,
        planet_radius,
        parameters['distance_to_system'].value
    )

    # Set the cloud abundances to 0 for clear case
    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances_clear[cname] = jnp.zeros_like(temperatures)

    wlen_model, flux, additional_outputs = prt_object.calculate_flux(
        temperatures=t_clear,
        mass_fractions=abundances_clear,
        reference_gravity=gravity,
        mean_molar_masses=mmw_clear,
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_clear = flux2irradiance(
        f_lambda,
        planet_radius,
        parameters['distance_to_system'].value
    )

    # Patchiness fraction
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + \
                     ((1 - patchiness) * spectrum_model_clear)

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']

    return wlen_model, spectrum_model


def guillot_emission(prt_object,
                     parameters,
                     pt_plot_mode=False,
                     adaptive_mesh_refinement=False):
    """
    Emission spectrum calculation for the Guillot 2010 temperature profile.
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    gravity, planet_radius = _compute_gravity(parameters)

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def guillot_emission_add_gaussian_temperature(prt_object,
                                              parameters,
                                              pt_plot_mode=False,
                                              adaptive_mesh_refinement=False):
    """
    Emission spectrum calculation for the Guillot 2010 temperature profile.
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    gravity, planet_radius = _compute_gravity(parameters)

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    def gaussian(x, scale, mu, sig):
        return scale*jnp.exp(-jnp.power((x - mu)/sig, 2.)/2)
    temperature_addition = gaussian(jnp.log10(p_use),
                                    parameters["temperature_peak"].value,
                                    parameters["temperature_location_log_pressure"].value,
                                    parameters["temperature_width_log_pressure"].value)
    temperatures += temperature_addition
    safe_inds = jnp.where(temperatures < 1.0)
    temperatures[safe_inds] = 1.0

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def guillot_patchy_emission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    Deprecated, to be removed in future version
    """
    return guillot_emission(prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement)


def interpolated_profile_emission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    This model computes a emission spectrum based a spline temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  nnodes : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  T{i}  : One parameter for each temperature node
                *  gamma : weight for penalizing the profile curvature

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    gravity, planet_radius = _compute_gravity(parameters)

    temp_arr = jnp.array(
        [parameters[f"T{i}"].value for i in range(parameters['nnodes'].value + 2)]
        )
    if "linear" in parameters.keys():
        temperatures, log_prior_weight = linear_spline_temperature_profile(
            p_use,
            temp_arr,
            parameters['gamma'].value,
            parameters['nnodes'].value
        )
    else:
        temperatures, log_prior_weight = cubic_spline_temperature_profile(
            p_use,
            temp_arr,
            parameters['gamma'].value,
            parameters['nnodes'].value
        )
    parameters['log_prior_weight'].value = log_prior_weight
    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def gradient_profile_emission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  N_layers : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  T_bottom : temperature at the base of the atmosphere
                *  PTslope_* : temperature gradient for each of the n_layers between which the profile is interpolated.

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    gravity, planet_radius = _compute_gravity(parameters)

    num_layer = parameters['N_layers'].value
    # 1.4 assemble the P-T slopes for these layers
    layer_pt_slopes = jnp.ones(num_layer) * jnp.nan
    for index in range(num_layer):
        layer_pt_slopes = layer_pt_slopes.at[index].set(parameters[f'PTslope_{num_layer - index}'].value)

    top_of_atmosphere_presure = -3
    bottom_of_atmosphere_presure = 3

    if "top_of_atmosphere_pressure" in parameters.keys():
        top_of_atmosphere_presure = parameters["top_of_atmosphere_pressure"].value
    if "bottom_of_atmosphere_pressure" in parameters.keys():
        bottom_of_atmosphere_presure = parameters["bottom_of_atmosphere_pressure"].value
    temperatures = dtdp_temperature_profile(
        p_use,
        num_layer,
        layer_pt_slopes,
        parameters['T_bottom'].value,
        top_of_atmosphere_pressure=top_of_atmosphere_presure,
        bottom_of_atmosphere_pressure=bottom_of_atmosphere_presure
        )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def power_law_profile_emission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  alpha : power law slope for the temperture profile
                *  T_0 : multiplicative factor for the power law slope

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    gravity, planet_radius = _compute_gravity(parameters)

    num_layer = parameters['N_layers'].value
    # 1.4 assemble the P-T slopes for these layers
    layer_pt_slopes = jnp.ones(num_layer) * jnp.nan
    for index in range(num_layer):
        layer_pt_slopes = layer_pt_slopes.at[index].set(parameters[f'PTslope_{num_layer - index}'].value)

    temperatures = power_law_temperature_profile(
        p_use,
        parameters['alpha'].value,
        parameters['T_0'].value
        )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def guillot_transmission(prt_object,
                         parameters,
                         pt_plot_mode=False,
                         adaptive_mesh_refinement=False):
    """
    Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on the Guillot profile
    Either free or (dis)equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)
    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value
    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, \
        haze_factor, power_law_opacity_350nm = clouds.setup_simple_clouds_hazes(parameters)
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def guillot_patchy_transmission(prt_object,
                                parameters,
                                pt_plot_mode=False,
                                adaptive_mesh_refinement=False):
    """
    Deprecated
    """
    return guillot_transmission(prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement)


def madhushudhan_seager_emission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    Transmission Model, Madhusudhan Seager 2009 Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  log_P_set : Pressure value to contrain the PT profile, defaults to 10 bar.
                *  T_set : temperature at P_set to constrain the PT profile. [K]
                *  log_P3 : (log) Pressure value for the top of the deep atmospheric layer, [bar]
                *  P2 : in range (0,1), sets the pressure level of the middle atmospheric layer
                *  P1 : in range (0,1), sets the pressure level of the top atmospheric layer
                *  alpha_0 : slope of the upper atmospheric layer
                *  alpha_1 : slope of the middle atmospheric layer
                Optional :
                    *  beta : power law for the slopes, default value is 0.5. Not recommended to change!

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)

    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)

    # Set up pressure points, guaranteeing P3>P2>P1 >= P_top
    offset = jnp.log10(p_use[0])
    log_p3 = parameters['log_P3'].value - offset
    log_p2 = log_p3 * (1.0 - parameters['P2'].value)
    log_p1 = log_p2 * (1.0 - parameters['P1'].value)
    pressure_points = [p_use[0], (log_p1 + offset), (log_p2 + offset), (log_p3 + offset), parameters["log_P_set"].value]

    alpha_points = [parameters["alpha_0"].value, parameters["alpha_1"].value]
    beta_points = [0.5, 0.5]
    if "beta" in parameters.keys():
        beta_points = [parameters["beta"].value, parameters["beta"].value]

    temperatures = madhusudhan_seager_temperature_profile(
        p_use,
        pressure_points,
        parameters["T_set"].value,
        alpha_points,
        beta_points
        )

    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties
    # Calculate the spectrum.
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def madhushudhan_seager_transmission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    Transmission Model, Madhusudhan Seager 2009 Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  log_P_set : Pressure value to contrain the PT profile, defaults to 10 bar.
                *  T_set : temperature at P_set to constrain the PT profile. [K]
                *  log_P3 : (log) Pressure value for the top of the deep atmospheric layer, [bar]
                *  P2 : in range (0,1), sets the pressure level of the middle atmospheric layer
                *  P1 : in range (0,1), sets the pressure level of the top atmospheric layer
                *  alpha_0 : slope of the upper atmospheric layer
                *  alpha_1 : slope of the middle atmospheric layer
                Optional :
                    *  beta : power law for the slopes, default value is 0.5. Not recommended to change!

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)
    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value
    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)

    # Set up pressure points, guaranteeing P3>P2>P1 >= P_top
    offset = jnp.log10(p_use[0])
    log_p3 = parameters['log_P3'].value - offset
    log_p2 = log_p3 * (1.0 - parameters['P2'].value)
    log_p1 = log_p2 * (1.0 - parameters['P1'].value)
    pressure_points = [p_use[0], (log_p1 + offset), (log_p2 + offset), (log_p3 + offset), parameters["log_P_set"].value]

    alpha_points = [parameters["alpha_0"].value, parameters["alpha_1"].value]
    beta_points = [0.5, 0.5]
    if "beta" in parameters.keys():
        beta_points = [parameters["beta"].value, parameters["beta"].value]

    temperatures = madhusudhan_seager_temperature_profile(p_use,
                                     pressure_points,
                                     parameters["T_set"].value,
                                     alpha_points,
                                     beta_points)

    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, \
        haze_factor, power_law_opacity_350nm = clouds.setup_simple_clouds_hazes(parameters)
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def madhu_seager_patchy_transmission(prt_object,
                                     parameters,
                                     pt_plot_mode=False,
                                     adaptive_mesh_refinement=False):
    """
    Deprecated
    """
    return madhushudhan_seager_transmission(prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement)


def isothermal_transmission(prt_object,
                            parameters,
                            pt_plot_mode=False,
                            adaptive_mesh_refinement=False):
    """
    Equilibrium Chemistry Transmission Model, Isothermal Profile

    This model computes a transmission spectrum based on an isothermal temperature-pressure profile.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  temperature : Interior temperature of the planet [K]

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned if patchy clouds are considered.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)
    reference_pressure = 100.0

    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value

    # Make the P-T profile
    temperatures = temperature_profile_function_isothermal(p_use, parameters["temperature"].value)
    gravity, planet_radius = _compute_gravity(parameters)

    # Make the abundance profile
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, \
        haze_factor, power_law_opacity_350nm = clouds.setup_simple_clouds_hazes(parameters)
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def power_law_profile_transmission(prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  alpha : power law slope for the temperture profile
                *  T_0 : multiplicative factor for the power law slope

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/Rstar**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure(prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement)
    gravity, planet_radius = _compute_gravity(parameters)

    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value

    temperatures = power_law_temperature_profile(
        p_use,
        parameters['alpha'].value,
        parameters['T_0'].value
        )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, \
        haze_factor, power_law_opacity_350nm = clouds.setup_simple_clouds_hazes(parameters)
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds(pressures, parameters, prt_object.cloud_species)
    sigma_lnorm, cloud_f_sed, eddy_diffusion_coefficients, \
        cloud_hansen_b, cloud_particles_mean_radii, \
        cloud_fraction, complete_coverage_clouds, distribution = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        distribution=distribution
        )


def add_blackbody_cpd_model(parameters, wavelengths):
    """
    Calculates the flux of a blackbody with area 4*pi*disk_radius^2 and temperature T_disk_blackbody.
    This is in units of W/m2/micron, and can be added to a planetary spectrum to model the contribution
    of a circumplanetary disk

    Args:
        parameters (dict): dictionary of atmospheric and disk parameters
        wavelengths (ArrayLike): Wavelength grid of atmospheric model in micron

    Returns:
        blackbody_spectrum (ArrayLike): 1D Planck emission spectrum for a circular CPD.
    """
    if "T_disk_blackbody" in parameters.keys():
        blackbody_spectrum = planck_function_cm(parameters["T_disk_blackbody"].value, wavelengths*1e-4)*1e-7
        blackbody_spectrum = blackbody_spectrum * (parameters["disk_radius"].value/parameters['distance_to_system'].value)**2
        return blackbody_spectrum


def initialize_pressure(press, parameters, adaptive_mesh_refinement):
    """
    Provide the pressure array correctly sized to the prt_object in use, accounting for
    the use of Adaptive Mesh Refinement around the location of clouds.

    Args:
        press : numpy.ndarray
            Pressure array from a prt_object. Used to set the min and max values of PGLOBAL
        parameters :
            # TODO complete docstring
        adaptive_mesh_refinement :
            # TODO complete docstring
    """
    if adaptive_mesh_refinement:
        PGLOBAL = set_pglobal(press, parameters)
        p_use = PGLOBAL
    else:
        p_use = press
        PGLOBAL = press
    return p_use, PGLOBAL


def set_pglobal(press, parameters):
    """
    Check to ensure that the global pressure array has the correct length.
    Updates PGLOBAL.

    Args:
        press : numpy.ndarray
            Pressure array from a prt_object. Used to set the min and max values of PGLOBAL
        parameters : dict
            Must include the 'pressure_simple' and 'pressure_scaling' parameters,
            used to determine the size of the high resolution grid.
    """
    try:
        pglobal_check(press,
                      parameters['pressure_simple'].value,
                      parameters['pressure_scaling'].value)
    except KeyError():
        raise KeyError(
            "missing parameters 'pressure_simple' and 'pressure_scaling parameters', required when using AMR"
        )


def calculate_emission_spectrum(
        prt_object,
        parameters,
        temperatures,
        abundances,
        gravity,
        mean_molar_masses,
        planet_radius,
        sigma_lnorm,
        cloud_particles_mean_radii,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_fraction,
        complete_coverage_clouds,
        distribution):
    """
    Calls Radtrans.calculate_flux to compute the emission spectrum of an atmosphere.
    This function automatically checks if patchiness is included in the retrieval, and
    mixes the clear and cloudy columns. Patchiness can be applied to all of the cloud species,
    or individual clouds can be chosen using the remove_cloud_species parameter.
    A circumplanetary disk model is optionally included, modelled as a blackbody with some temperature
    T_disk_blackbody and a radius disk_radius.

    Args:
        prt_object (Radtrans): The Radtrans object used to calculate the spectrum
        parameters (dict): Dictionary of atmospheric parameters.
        temperatures (ArrayLike): Array of temperatures for each pressure level in the atmosphere
        abundances (dict): Dictionary of molecular mass fraction abundances for each level in the atmosphere
        gravity (ArrayLike): Gravitational acceleration at each pressure level
        mean_molar_masses (ArrayLike): Mean molecular mass at each pressure level
        planet_radius (float): Planet radius in cm
        sigma_lnorm (float): Width of the cloud particle size distribution (log-normal)
        cloud_particles_mean_radii (ArrayLike): Mean particle radius
        cloud_f_sed (ArrayLike): Sedimentation fraction
        eddy_diffusion_coefficients (ArrayLike): Vertical mixing strength (Kzz)
        cloud_hansen_b (ArrayLike): Cloud particle distribution width, hansen distsribution
        cloud_fraction (float) : fraction of planet covered by clouds
        complete_coverage_clouds (list(str)) : Which clouds are NOT patchy
        distribution (string): Which cloud particle size distribution to use
    """
    (return_contribution,
     return_opacities,
     return_photosphere_radius,
     return_rosseland_optical_depths,
     return_radius_hydrostatic_equilibrium,
     return_cloud_contribution,
     return_abundances,
     return_any) = get_calculate_flux_return_values(parameters)

    results = prt_object.simple_calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mean_molar_masses,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        cloud_particles_radius_distribution=distribution,
        #return_contribution=return_contribution,
        #return_photosphere_radius=return_photosphere_radius,
        #return_rosseland_optical_depths=return_rosseland_optical_depths,
        #return_cloud_contribution=return_cloud_contribution,
        #return_abundances=return_abundances,
        #return_opacities=return_opacities
    )

    wlen_model, flux, additional_outputs = results

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model = flux2irradiance(
        f_lambda,
        planet_radius,
        parameters['distance_to_system'].value
    )

    if "T_disk_blackbody" in parameters.keys():
        spectrum_model += add_blackbody_cpd_model(parameters, wlen_model)

    if "v_band_extinction" in parameters.keys():
        spectrum_model = apply_ism_ext(
            wlen_model,
            spectrum_model,
            parameters["v_band_extinction"].value,
            parameters["v_band_reddening"].value)
    if return_any:
        return wlen_model, spectrum_model, additional_outputs
    else:
        return wlen_model, spectrum_model


def calculate_transmission_spectrum(
        prt_object,
        parameters,
        temperatures,
        abundances,
        gravity,
        mean_molar_masses,
        planet_radius,
        reference_pressure,
        opaque_cloud_top_pressure,
        sigma_lnorm,
        cloud_particles_mean_radii,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        haze_factor,
        power_law_opacity_coefficient,
        power_law_opacity_350nm,
        cloud_hansen_b,
        cloud_fraction,
        complete_coverage_clouds,
        distribution
        ):
    """_summary_

    Args:
        prt_object (Radtrans): The Radtrans object used to calculate the spectrum
        parameters (dict): Dictionary of atmospheric parameters.
        temperatures (ArrayLike): Array of temperatures for each pressure level in the atmosphere
        abundances (dict): Dictionary of molecular mass fraction abundances for each level in the atmosphere
        gravity (ArrayLike): Gravitational acceleration at each pressure level
        mean_molar_masses (ArrayLike): Mean molecular mass at each pressure level
        planet_radius (float):Planet radius in cm
        reference_pressure (float): Pressure at which the planet radius is defined
        opaque_cloud_top_pressure (float): Pressure where an opaque grey cloud deck is placed
        sigma_lnorm (float): Width of the cloud particle size distribution (log-normal)
        cloud_particles_mean_radii (ArrayLike): Mean particle radius
        cloud_f_sed (ArrayLike): Sedimentation fraction
        eddy_diffusion_coefficients (ArrayLike): Vertical mixing strength (Kzz)
        haze_factor (float): Multiplicative factor on the strength of a power law haze slope
        power_law_opacity_coefficient (float): Exponent for the slope of a power law haze
        power_law_opacity_350nm (float): Strength of the power law scattering at 350nm
        cloud_hansen_b (ArrayLike): Cloud particle distribution width, hansen distsribution
        cloud_fraction (float) : fraction of planet covered by clouds
        complete_coverage_clouds (list(str)) : Which clouds are NOT patchy
        distribution (string): Log normal or hansen particle size distribution
    Returns:
        _type_: _description_
    """
    (return_contribution,
     return_opacities,
     return_photosphere_radius,
     return_rosseland_optical_depths,
     return_radius_hydrostatic_equilibrium,
     return_cloud_contribution,
     return_abundances,
     return_any) = get_calculate_flux_return_values(parameters)

    results = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mean_molar_masses,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_particles_mean_radii=cloud_particles_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        complete_coverage_clouds=complete_coverage_clouds,
        cloud_particles_radius_distribution=distribution,
        return_contribution=return_contribution,
        return_cloud_contribution=return_cloud_contribution,
        return_radius_hydrostatic_equilibrium=return_radius_hydrostatic_equilibrium,
        return_abundances=return_abundances,
        return_opacities=return_opacities,
    )
    wlen_model, transit_radii, additional_outputs = results

    wlen_model *= 1e4
    spectrum_model = (transit_radii / parameters['stellar_radius'].value) ** 2.
    if return_any:
        return wlen_model, spectrum_model, additional_outputs
    return wlen_model, spectrum_model


def pglobal_check(press, shape, scaling):
    """
    Check to ensure that the global pressure array has the correct length.
    Updates PGLOBAL.

    Args:
        press : numpy.ndarray
            Pressure array from a prt_object. Used to set the min and max values of PGLOBAL
        shape : int
            the shape of the pressure array if no adaptive_mesh_refinement is used
        scaling :
            The factor by which the pressure array resolution should be scaled.
    """
    #global PGLOBAL
    PGLOBAL = jnp.logspace(-6, 3, 1000)
    if PGLOBAL.shape[0] != int(scaling * shape):
        PGLOBAL = jnp.logspace(jnp.log10(press[0]),
                              jnp.log10(press[-1]),
                              int(scaling * shape))
    return PGLOBAL


def ism_extinction(
    av_mag, rv_red, wavelengths
):
    """
    Function for calculating the optical and IR extinction
    with the empirical relation from `Cardelli et al. (1989)
    <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_.
    Reimplemented from species.utils.dust_utils to avoid circular
    dependencies.

    Parameters
    ----------
    av_mag : float
        Extinction (mag) in the $V$ band.
    rv_red : float
        Reddening in the $V$ band, ``R_V = A_V / E(B-V)``.
    wavelengths : np.ndarray, list(float), float
        Array or list with the wavelengths (um) for which
        the extinction is calculated. It is also possible
        to provide a single value as float.

    Returns
    -------
    np.ndarray
        Extinction (mag) at ``wavelengths``.
    """

    wavelengths = jnp.array(wavelengths)

    x_wavel = 1.0 / wavelengths
    y_wavel = x_wavel - 1.82

    a_coeff = jnp.zeros(x_wavel.size)
    b_coeff = jnp.zeros(x_wavel.size)

    indices = jnp.where(x_wavel < 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = 0.574 * x_wavel[indices] ** 1.61
        b_coeff[indices] = -0.527 * x_wavel[indices] ** 1.61

    indices = jnp.where(x_wavel >= 1.1)[0]

    if len(indices) > 0:
        a_coeff[indices] = (
            1.0
            + 0.17699 * y_wavel[indices]
            - 0.50447 * y_wavel[indices] ** 2
            - 0.02427 * y_wavel[indices] ** 3
            + 0.72085 * y_wavel[indices] ** 4
            + 0.01979 * y_wavel[indices] ** 5
            - 0.77530 * y_wavel[indices] ** 6
            + 0.32999 * y_wavel[indices] ** 7
        )

        b_coeff[indices] = (
            1.41338 * y_wavel[indices]
            + 2.28305 * y_wavel[indices] ** 2
            + 1.07233 * y_wavel[indices] ** 3
            - 5.38434 * y_wavel[indices] ** 4
            - 0.62251 * y_wavel[indices] ** 5
            + 5.30260 * y_wavel[indices] ** 6
            - 2.09002 * y_wavel[indices] ** 7
        )

    return av_mag * (a_coeff + b_coeff / rv_red)


def apply_ism_ext(
    wavelengths, flux, v_band_ext, v_band_red
):
    """
    Function for applying ISM extinction to a spectrum.
    Reimplemented from species.utils.dust_utils to avoid circular
    dependencies.

    wavelengths : jnp.ndarray
        Wavelengths (um) of the spectrum.
    flux : jnp.ndarray
        Fluxes (W m-2 um-1) of the spectrum.
    v_band_ext : float
        Extinction (mag) in the $V$ band.
    v_band_red : float
        Reddening in the $V$ band.

    Returns
    -------
    jnp.ndarray
        Fluxes (W m-2 um-1) with the extinction applied.
    """

    ext_mag = ism_extinction(v_band_ext, v_band_red, wavelengths)

    return flux * 10.0 ** (-0.4 * ext_mag)
