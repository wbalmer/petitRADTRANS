import sys

import numpy as np

from petitRADTRANS.containers.planet import Planet
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import (
    flux2irradiance,
    temperature_profile_function_ret_model, temperature_profile_function_guillot_global,
    temperature_profile_function_isothermal,
    cubic_spline_profile,
    linear_spline_profile,
    dtdp_temperature_profile,
    madhu_seager_2009
)
from petitRADTRANS.retrieval import cloud_cond as fc
from .chemistry import get_abundances

"""
Models Module

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
    amr : bool
        If this parameter is True, your model should allow for reshaping of the
        pressure and temperature arrays based on the position of the clouds or
        the location of the photosphere, increasing the resolution where required.
        For example, using the fixed_length_amr function defined below.
"""

# Global constants to reduce calculations and initializations.
PGLOBAL = np.logspace(-6, 3, 1000)


def _compute_gravity(parameters):
    if 'log_g' in parameters.keys() and 'mass' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        r_pl, _, _ = Planet.reference_gravity2radius(
            reference_gravity=gravity,
            mass=parameters['mass'].value
        )
    elif 'log_g' in parameters.keys():
        gravity = 10 ** parameters['log_g'].value
        r_pl = parameters['R_pl'].value
    elif 'mass' in parameters.keys():
        r_pl = parameters['R_pl'].value
        gravity, _, _ = Planet.mass2reference_gravity(
            mass=parameters['mass'].value,
            radius=r_pl
        )
    else:
        raise KeyError("Pick two of log_g, R_pl and mass priors!")

    return gravity, r_pl


def emission_model_diseq(prt_object,
                         parameters,
                         pt_plot_mode=False,
                         amr=True):
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
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.ndarray
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.ndarray
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, np.ndarray
            Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Use this for debugging.
    # for key, val in parameters.items():
    #    print(key,val.value)

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - parameters['T3'].value)
    t2 = t3 * (1.0 - parameters['T2'].value)
    t1 = t2 * (1.0 - parameters['T1'].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta'].value)) * 1e6) ** (-parameters['alpha'].value)
    gravity, r_pl = _compute_gravity(parameters)

    # Make the P-T profile
    temp_arr = np.array([t1, t2, t3])
    carbon_to_oxygen = 0

    if 'C/O' in parameters.keys():
        carbon_to_oxygen = parameters['C/O'].value
    elif "C" in parameters.keys():
        carbon_to_oxygen = parameters['C'].value / parameters['O'].value

    temperatures = temperature_profile_function_ret_model(  # TODO weird way of calling the function
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
        temperatures[np.where(temperatures < 40.0)] = 40.0
        temperatures[np.where(temperatures > 42000.0)] = 42000.0

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
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

    # Hansen or log normal clouds
    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)

    # calculate the spectrum
    results = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        return_contribution=contribution,
        cloud_f_sed=fseds,
        eddy_diffusion_coefficient=kzz,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_b_hansen=b_hans,
        clouds_particles_mean_radii=radii,
        cloud_particles_radius_distribution=distribution,
        frequencies_to_wavelengths=True
    )

    if not contribution:
        wlen_model, flux, _ = results
        additional_outputs = None
    else:
        wlen_model, flux, additional_outputs = results

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']

    return wlen_model, spectrum_model


def emission_model_diseq_patchy_clouds(prt_object, parameters, pt_plot_mode=False, amr=True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    This model includes patchy clouds, and requires a unique temperature profile for the
    clear atmosphere regions.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, np.ndarray
            Emission contribution function, relative contributions for each wavelength and pressure level.
            Only returns the contribution of the clear atmosphere component.

    """
    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)

    # Not sure how to deal with having 2 separate contribution function
    contribution = False

    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - parameters['T3'].value)
    t2 = t3 * (1.0 - parameters['T2'].value)
    t1 = t2 * (1.0 - parameters['T1'].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta'].value)) * 1e6) ** (-parameters['alpha'].value)
    temp_arr = np.array([t1, t2, t3])

    t3_clear = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (
                1.0 - parameters['T3_clear'].value)
    t2_clear = t3_clear * (1.0 - parameters['T2_clear'].value)
    t1_clear = t2_clear * (1.0 - parameters['T1_clear'].value)
    temps_clear = np.array([t1_clear, t2_clear, t3_clear])
    delta_clear = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta_clear'].value)) * 1e6) ** (
        -parameters['alpha_clear'].value)
    gravity, r_pl = _compute_gravity(parameters)

    temperatures = temperature_profile_function_ret_model(  # TODO weird way of calling the function
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
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    t_clear = temperature_profile_function_ret_model(
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
        PGLOBAL[small_index],
        t_clear,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=False
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use
    if pressures.shape[0] != prt_object.pressures.shape[0]:
        return None, None

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    wlen_model, flux, _ = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        fsed=fseds,
        Kzz=kzz,
        sigma_lnorm=sigma_lnorm,
        b_hans=b_hans,
        radius=radii,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_cloudy = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    # Set the cloud abundances to 0 for clear case
    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances_clear[cname] = np.zeros_like(temperatures)

    wlen_model, flux, additional_outputs = prt_object.calculate_flux(
        temperatures=t_clear,
        mass_fractions=abundances_clear,
        reference_gravity=gravity,
        mean_molar_masses=mmw_clear,
        contribution=contribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_clear = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    # Patchiness fraction
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + \
                     ((1 - patchiness) * spectrum_model_clear)

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']

    return wlen_model, spectrum_model


def emission_model_diseq_simple_patchy_clouds(prt_object,
                                              parameters,
                                              pt_plot_mode=False,
                                              amr=True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature-pressure profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    This model includes patchy clouds, but uses a constant pressure temperature profile
    for both the clear and cloudy regions.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                Optional
                  *  contribution : return the emission contribution function
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, np.ndarray
            Emission contribution function, relative contributions for each wavelength and pressure level.
            Only returns the contribution of the clear atmosphere component.
    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Not sure how to deal with having 2 separate contribution function

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = ((3. / 4. * parameters['T_int'].value ** 4. * (0.1 + 2. / 3.)) ** 0.25) * (1.0 - parameters['T3'].value)
    t2 = t3 * (1.0 - parameters['T2'].value)
    t1 = t2 * (1.0 - parameters['T1'].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters['log_delta'].value)) * 1e6) ** (-parameters['alpha'].value)
    temp_arr = np.array([t1, t2, t3])

    gravity, r_pl = _compute_gravity(parameters)

    temperatures = temperature_profile_function_ret_model(
        (  # TODO weird way of calling the function
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
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        if contribution:
            return None, None, None
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use
    if pressures.shape[0] != prt_object.press.shape[0]:
        if contribution:
            return None, None, None
        return None, None

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    wlen_model, flux, _ = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        fsed=fseds,
        Kzz=kzz,
        sigma_lnorm=sigma_lnorm,
        b_hans=b_hans,
        radius=radii,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_cloudy = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    # Set the cloud abundances to 0 for clear case
    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)

    wlen_model, f_lambda, additional_outputs = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_clear = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
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
                     amr=False):
    """
    Emission spectrum calculation for the Guillot 2010 temperature profile.
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
        pt_plot_mode : bool
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, r_pl = _compute_gravity(parameters)

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
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    wlen_model, flux, additional_outputs = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        fsed=fseds,
        Kzz=kzz,
        sigma_lnorm=sigma_lnorm,
        b_hans=b_hans,
        radius=radii,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )
    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']
    return wlen_model, spectrum_model


def guillot_patchy_emission(prt_object, parameters, pt_plot_mode=False, amr=False):
    """
    This model computes a emission spectrum based the Guillot temperature-pressure profile and patchy clouds.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                Optional
                  *  contribution : return the emission contribution function
        pt_plot_mode : bool
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
            Only returns the contribution of the clear atmosphere component.

    """
    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, r_pl = _compute_gravity(parameters)

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
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    wlen_model, flux, _ = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        fsed=fseds,
        Kzz=kzz,
        sigma_lnorm=sigma_lnorm,
        b_hans=b_hans,
        radius=radii,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_cloudy = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    # Set the cloud abundances to 0 for clear case
    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)

    wlen_model, flux, additional_opacities = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_clear = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )

    # Patchiness fraction
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + \
                     ((1 - patchiness) * spectrum_model_clear)
    if contribution:
        return wlen_model, spectrum_model, additional_opacities['emission_contribution']

    return wlen_model, spectrum_model


def interpolated_profile_emission(prt_object, parameters, pt_plot_mode=False, amr=False):
    """
    This model computes a emission spectrum based a spline temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
                  *  mass : planet mass [g]
                *  nnodes : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  Temps : One parameter for each temperature node
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
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, r_pl = _compute_gravity(parameters)

    temp_arr = np.array([parameters[f"T{i}"].value for i in range(parameters['nnodes'].value + 2)])
    if "linear" in parameters.keys():
        temperatures, log_prior_weight = linear_spline_profile(p_use,
                                                               temp_arr,
                                                               parameters['gamma'].value,
                                                               parameters['nnodes'].value)
    else:
        temperatures, log_prior_weight = cubic_spline_profile(p_use,
                                                              temp_arr,
                                                              parameters['gamma'].value,
                                                              parameters['nnodes'].value)
    parameters['log_prior_weight'].value = log_prior_weight
    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    wlen_model, flux, additional_outputs = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        contribution=contribution,
        fsed=fseds,
        Kzz=kzz,
        sigma_lnorm=sigma_lnorm,
        b_hans=b_hans,
        radius=radii,
        dist=distribution
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )
    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']
    return wlen_model, spectrum_model


def gradient_profile_emission(prt_object, parameters, pt_plot_mode=False, amr=False):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
                  *  mass : planet mass [g]
                *  N_layers : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  T_bottom : Temperature at the base of the atmosphere
                *  PTslope_* : Temperature gradient for each of the n_layers between which the profile is interpolated.

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
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, r_pl = _compute_gravity(parameters)

    num_layer = parameters['N_layers'].value
    # 1.4 assemble the P-T slopes for these layers
    layer_pt_slopes = np.ones(num_layer) * np.nan
    for index in range(num_layer):
        layer_pt_slopes[index] = parameters[f'PTslope_{num_layer - index}'].value

    temperatures = dtdp_temperature_profile(p_use,
                                            num_layer,
                                            layer_pt_slopes,
                                            parameters['T_bottom'].value)

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    results = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        return_contribution=contribution,
        cloud_f_sed=fseds,
        eddy_diffusion_coefficient=kzz,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_b_hansen=b_hans,
        clouds_particles_mean_radii=radii,
        cloud_particles_radius_distribution=distribution,
        frequencies_to_wavelengths=True
    )

    if not contribution:
        wlen_model, flux, _ = results
        additional_outputs = None
    else:
        wlen_model, flux, additional_outputs = results

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model = flux2irradiance(
        f_lambda,
        r_pl,
        parameters['D_pl'].value
    )
    if contribution:
        return wlen_model, spectrum_model, additional_outputs['emission_contribution']

    return wlen_model, spectrum_model


def guillot_transmission(prt_object,
                         parameters,
                         pt_plot_mode=False,
                         amr=False):
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
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
    """
    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)
    p_reference = 100.0
    if "reference_pressure" in parameters.keys():
        p_reference = parameters["reference_pressure"].value
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, r_pl = _compute_gravity(parameters)

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    pcloud = None
    gamma_scat = None
    kappa_0 = None
    if 'log_Pcloud' in parameters.keys():
        pcloud = 10 ** parameters['log_Pcloud'].value
    elif 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value
    if "gamma_scat" in parameters.keys():
        gamma_scat = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        kappa_0 = 10 ** parameters["kappa_0"].value
    # Calculate the spectrum
    if len(prt_object.cloud_species) > 0:
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                               prt_object.cloud_species)
        results = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            sigma_lnorm=sigma_lnorm,
            radius=radii,
            fsed=fseds,
            Kzz=kzz,
            kappa_zero=kappa_0,
            gamma_scat=gamma_scat,
            b_hans=b_hans,
            dist=distribution,
            contribution=contribution
        )
    elif pcloud is not None:
        results = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            planet_radius=r_pl,
            reference_pressure=p_reference,
            opaque_cloud_top_pressure=pcloud,
            kappa_zero=kappa_0,
            gamma_scat=gamma_scat,
            return_contribution=contribution,
            frequencies_to_wavelengths=False
        )
    else:
        results = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            planet_radius=r_pl,
            reference_pressure=p_reference,
            return_contribution=contribution,
            frequencies_to_wavelengths=False
        )

    if not contribution:
        frequencies, transit_radii, _ = results
        additional_output = None
    else:
        frequencies, transit_radii, additional_output = results

    wlen_model = cst.c / frequencies / 1e-4
    spectrum_model = (transit_radii / parameters['Rstar'].value) ** 2.
    if contribution:
        return wlen_model, spectrum_model, additional_output['transmission_contribution']
    return wlen_model, spectrum_model


def guillot_patchy_transmission(prt_object,
                                parameters,
                                pt_plot_mode=False,
                                amr=False):
    """
    Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                Optional
                  *  contribution : return the transmission contribution function
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)
    p_reference = 100.0
    if "reference_pressure" in parameters.keys():
        p_reference = parameters["reference_pressure"].value

    contribution = False

    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Calculate the spectrum
    gravity, r_pl = _compute_gravity(parameters)

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    # Hazes
    gamma_scat = None
    kappa_0 = None
    if "gamma_scat" in parameters.keys():
        gamma_scat = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        kappa_0 = 10 ** parameters["kappa_0"].value

    # Calc cloudy spectrum
    wlen_model, transit_radii, _ = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        R_pl=r_pl,
        P0_bar=p_reference,
        sigma_lnorm=sigma_lnorm,
        radius=radii,
        fsed=fseds,
        Kzz=kzz,
        kappa_zero=kappa_0,
        gamma_scat=gamma_scat,
        b_hans=b_hans,
        dist=distribution,
        contribution=contribution
    )

    wlen_model *= 1e4
    spectrum_model_cloudy = (transit_radii / parameters['Rstar'].value) ** 2.

    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)

    wlen_model, transit_radii, additional_outputs = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        R_pl=r_pl,
        P0_bar=p_reference,
        contribution=contribution
    )

    wlen_model *= 1e4
    spectrum_model_clear = (transit_radii / parameters['Rstar'].value) ** 2.
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + \
                     ((1 - patchiness) * spectrum_model_clear)
    if contribution:
        return wlen_model, spectrum_model, additional_outputs['transmission_contribution']

    return wlen_model, spectrum_model


def madhu_seager_patchy_transmission(prt_object, parameters, pt_plot_mode=False, amr=False):
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
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                Optional
                  *  contribution : return the transmission contribution function
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)
    p_reference = 100.0
    if "reference_pressure" in parameters.keys():
        p_reference = parameters["reference_pressure"].value
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, r_pl = _compute_gravity(parameters)

    # Set up pressure points, guaranteeing P3>P2>P1 >= P_top
    offset = np.log10(p_use[0])
    log_p3 = parameters['log_P3'].value - offset
    log_p2 = log_p3 * (1.0 - parameters['P2'].value)
    log_p1 = log_p2 * (1.0 - parameters['P1'].value)
    pressure_points = [p_use[0], (log_p1 + offset), (log_p2 + offset), (log_p3 + offset), parameters["log_P_set"].value]

    alpha_points = [parameters["alpha_0"].value, parameters["alpha_1"].value]
    beta_points = [0.5, 0.5]
    if "beta" in parameters.keys():
        beta_points = [parameters["beta"].value, parameters["beta"].value]

    temperatures = madhu_seager_2009(p_use,
                                     pressure_points,
                                     parameters["T_set"].value,
                                     alpha_points,
                                     beta_points)

    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    # Hazes
    gamma_scat = None
    kappa_0 = None
    if "gamma_scat" in parameters.keys():
        gamma_scat = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        kappa_0 = 10 ** parameters["kappa_0"].value

    # Calc cloudy spectrum
    wlen_model, transit_radii, _ = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        R_pl=r_pl,
        P0_bar=p_reference,
        sigma_lnorm=sigma_lnorm,
        radius=radii,
        fsed=fseds,
        Kzz=kzz,
        kappa_zero=kappa_0,
        gamma_scat=gamma_scat,
        b_hans=b_hans,
        dist=distribution,
        contribution=contribution
    )

    wlen_model *= 1e4
    spectrum_model_cloudy = (prt_object.transm_rad / parameters['Rstar'].value) ** 2.

    if "patchiness" in parameters.key():
        for cloud in prt_object.cloud_species:
            cname = cloud.split('_')[0]
            abundances[cname] = np.zeros_like(temperatures)

        wlen_model, transit_radii, additional_outputs = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            contribution=contribution
        )

        wlen_model *= 1e4
        spectrum_model_clear = (transit_radii / parameters['Rstar'].value) ** 2.
        patchiness = parameters["patchiness"].value
        spectrum_model = (patchiness * spectrum_model_cloudy) + \
                         ((1 - patchiness) * spectrum_model_clear)
    else:
        spectrum_model = spectrum_model_cloudy
        additional_outputs = {'transmission_contribution': None}

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['transmission_contribution']

    return wlen_model, spectrum_model


def guillot_patchy_transmission_constrained_chem(prt_object, parameters, pt_plot_mode=False, amr=False):
    """
    Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
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
                *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                Optional
                  *  contribution : return the transmission contribution function
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use = initialize_pressure(prt_object.press / 1e6, parameters, amr)
    p_reference = 100.0
    if "reference_pressure" in parameters.keys():
        p_reference = parameters["reference_pressure"].value
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, r_pl = _compute_gravity(parameters)

    if parameters["H2O_Exomol"].value < parameters["CO2"].value:
        return None, None

    if parameters["CO_all_iso_HITEMP"].value < parameters["CO2"].value:
        return None, None

    temperatures = temperature_profile_function_guillot_global(
        p_use,
        10 ** parameters['log_kappa_IR'].value,
        parameters['gamma'].value,
        gravity,
        parameters['T_int'].value,
        parameters['T_equ'].value
    )

    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )

    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                           prt_object.cloud_species)
    # Hazes
    gamma_scat = None
    kappa_0 = None
    if "gamma_scat" in parameters.keys():
        gamma_scat = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        kappa_0 = 10 ** parameters["kappa_0"].value

    # Calc cloudy spectrum
    wlen_model, transit_radii, _ = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        R_pl=r_pl,
        P0_bar=p_reference,
        sigma_lnorm=sigma_lnorm,
        radius=radii,
        fsed=fseds,
        Kzz=kzz,
        kappa_zero=kappa_0,
        gamma_scat=gamma_scat,
        b_hans=b_hans,
        dist=distribution,
        contribution=contribution
    )

    wlen_model *= 1e4
    spectrum_model_cloudy = (transit_radii / parameters['Rstar'].value) ** 2.

    for cloud in prt_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)
    results = prt_object.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        R_pl=r_pl,
        P0_bar=p_reference,
        contribution=contribution
    )

    if not contribution:
        frequencies, transit_radii, _ = results
        additional_outputs = None
    else:
        frequencies, transit_radii, additional_outputs = results

    wlen_model = cst.c / frequencies / 1e-4
    spectrum_model_clear = (transit_radii / parameters['Rstar'].value) ** 2.
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + \
                     ((1 - patchiness) * spectrum_model_clear)

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['transmission_contribution']
    return wlen_model, spectrum_model


def isothermal_transmission(prt_object,
                            parameters,
                            pt_plot_mode=False,
                            amr=False):
    """
    Equilibrium Chemistry Transmission Model, Isothermal Profile

    This model computes a transmission spectrum based on an isothermal temperature-pressure profile.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  R_pl : planet radius [cm]
                  *  mass : planet mass [g]
                *  Temp : Interior temperature of the planet [K]

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

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
                Optional
                  *  patchiness : Fraction of cloud coverage, clear contribution is (1-patchiness)
                  *  contribution : return the transmission contribution function
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        amr :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
        contr-em : np.ndarray
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned if patchy clouds are considered.
    """

    p_use = initialize_pressure(prt_object.pressures / 1e6, parameters, amr)
    p_reference = 100.0

    if "reference_pressure" in parameters.keys():
        p_reference = parameters["reference_pressure"].value

    # Make the P-T profile
    temperatures = temperature_profile_function_isothermal(p_use, parameters["Temp"].value)
    gravity, r_pl = _compute_gravity(parameters)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Make the abundance profile
    abundances, mmw, small_index, p_bases = get_abundances(
        p_use,
        temperatures,
        prt_object.line_species,
        prt_object.cloud_species,
        parameters,
        amr=amr
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if amr:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Calculate the spectrum
    pcloud = None
    kappa_0 = None
    gamma_scat = None
    if 'log_Pcloud' in parameters.keys():
        pcloud = 10 ** parameters['log_Pcloud'].value
    elif 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value
    if "gamma_scat" in parameters.keys():
        gamma_scat = parameters["gamma_scat"].value
    if "kappa_0" in parameters.keys():
        kappa_0 = 10 ** parameters["kappa_0"].value
    # Calculate the spectrum
    if len(prt_object.cloud_species) > 0:
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters,
                                                                               prt_object.cloud_species)
        wlen_model, transit_radii, additional_outputs = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            sigma_lnorm=sigma_lnorm,
            radius=radii,
            fsed=fseds,
            Kzz=kzz,
            kappa_zero=kappa_0,
            gamma_scat=gamma_scat,
            b_hans=b_hans,
            dist=distribution,
            contribution=contribution
        )
    elif pcloud is not None:
        wlen_model, transit_radii, additional_outputs = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            Pcloud=pcloud,
            kappa_zero=kappa_0,
            gamma_scat=gamma_scat,
            contribution=contribution
        )
    else:
        wlen_model, transit_radii, additional_outputs = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            kappa_zero=kappa_0,
            gamma_scat=gamma_scat,
            contribution=contribution
        )

    wlen_model *= 1e4
    spectrum_model = (transit_radii / parameters['Rstar'].value) ** 2.

    if contribution:
        return wlen_model, spectrum_model, additional_outputs['transmission_contribution']

    if "patchiness" in parameters.keys():
        if len(prt_object.cloud_species) > 0:
            for cloud in prt_object.cloud_species:
                abundances[cloud.split('_')[0]] = np.zeros_like(temperatures)

        _, spectrum_model_clear, _ = prt_object.calculate_transit_radii(
            temperatures=temperatures,
            mass_fractions=abundances,
            reference_gravity=gravity,
            mean_molar_masses=mmw,
            R_pl=r_pl,
            P0_bar=p_reference,
            Pcloud=None,
            contribution=contribution
        )

        spectrum_model_clear = (spectrum_model_clear / parameters['Rstar'].value) ** 2.
        patchiness = parameters["patchiness"].value
        spectrum_model_full = (patchiness * spectrum_model) + \
                              ((1 - patchiness) * spectrum_model_clear)

        return wlen_model, spectrum_model_full

    return wlen_model, spectrum_model


def initialize_pressure(press, parameters, amr):
    """
    Provide the pressure array correctly sized to the prt_object in use, accounting for
    the use of Adaptive Mesh Refinement around the location of clouds.

    Args:
        press : numpy.ndarray
            Pressure array from a prt_object. Used to set the min and max values of PGLOBAL
        parameters :
            # TODO complete docstring
        amr :
            # TODO complete docstring
    """
    if amr:
        set_pglobal(press, parameters)
        p_use = PGLOBAL
    else:
        p_use = press

    return p_use


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
        print("You must include the pressure_simple and pressure_scaling parameters when using amr!")
        sys.exit(1)


def pglobal_check(press, shape, scaling):
    """
    Check to ensure that the global pressure array has the correct length.
    Updates PGLOBAL.

    Args:
        press : numpy.ndarray
            Pressure array from a prt_object. Used to set the min and max values of PGLOBAL
        shape : int
            the shape of the pressure array if no amr is used
        scaling :
            The factor by which the pressure array resolution should be scaled.
    """
    global PGLOBAL
    if PGLOBAL.shape[0] != int(scaling * shape):
        PGLOBAL = np.logspace(np.log10(press[0]),
                              np.log10(press[-1]),
                              int(scaling * shape))
