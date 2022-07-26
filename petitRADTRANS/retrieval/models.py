import sys, os
import copy as cp
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import interp1d,CubicSpline

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.retrieval import cloud_cond as fc
from petitRADTRANS import poor_mans_nonequ_chem as pm

from petitRADTRANS.physics import PT_ret_model, guillot_global, guillot_global_ret, guillot_modif, isothermal
from .chemistry import get_abundances
from .util import surf_to_meas, calc_MMW, compute_gravity, spectrum_cgs_to_si
"""
Models Module

This module contains a set of functions that generate the spectra used
in the petitRADTRANS retrieval. This includes setting up the
pressure-temperature structure, the chemistry, and the radiative
transfer to compute the emission or transmission spectrum.

All models must take the same set of inputs:

    pRT_object : petitRADTRANS.RadTrans
        This is the pRT object that is used to compute the spectrum
        It must be fully initialized prior to be used in the model function
    parameters : dict
        A dictionary of Parameter objects. The naming of the parameters
        must be consistent between the Priors and the model function you
        are using.
    PT_plot_mode : bool
        If this argument is True, the model function should return the pressure
        and temperature arrays before computing the flux.
    AMR : bool
        If this parameter is True, your model should allow for reshaping of the
        pressure and temperature arrays based on the position of the clouds or
        the location of the photosphere, increasing the resolution where required.
        For example, using the fixed_length_amr function defined below.
"""

# Global constants to reduce calculations and initializations.
PGLOBAL = np.logspace(-6,3,1000)

def emission_model_diseq(pRT_object,
                         parameters,
                         PT_plot_mode = False,
                         AMR = True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on disequilibrium carbon chemistry,
    equilibrium clouds and a spline temperature-pressure profile. (Molliere 2020).
    Many of the parameters are optional, but must be used in the correct combination
    with other parameters.

    Args:
        pRT_object : object
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
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed emission spectrum [W/m2/micron]
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Use this for debugging.
    #for key, val in parameters.items():
    #    print(key,val.value)

    # Priors for these parameters are implemented here, as they depend on each other
    T3 = ((3./4.*parameters['T_int'].value**4.*(0.1+2./3.))**0.25)*(1.0-parameters['T3'].value)
    T2 = T3*(1.0-parameters['T2'].value)
    T1 = T2*(1.0-parameters['T1'].value)
    delta = ((10.0**(-3.0+5.0*parameters['log_delta'].value))*1e6)**(-parameters['alpha'].value)
    gravity, R_pl =  compute_gravity(parameters)

    # Make the P-T profile
    temp_arr = np.array([T1,T2,T3])

    temperatures = PT_ret_model(temp_arr, \
                            delta,
                            parameters['alpha'].value,
                            parameters['T_int'].value,
                            p_use,
                            parameters['Fe/H'].value,
                            parameters['C/O'].value,
                            conv=True)
    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None
    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use

    # Calculate the spectrum
    if pressures.shape[0] != pRT_object.press.shape[0]:
        print("Incorrect output shape!")
        return None,None

    # Hansen or log normal clouds
    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)

    # calculate the spectrum
    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = fseds,
                        Kzz = kzz,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        radius = radii,
                        dist = distribution)

    # Getting the model into correct units (W/m2/micron)
    wlen_model, f_lambda = spectrum_cgs_to_si(pRT_object.freq, pRT_object.flux)
    spectrum_model = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def emission_model_diseq_patchy_clouds(pRT_object,
                                       parameters,
                                       PT_plot_mode = False,
                                       AMR = True):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on disequilibrium carbon chemistry,
    equilibrium clouds and a spline temperature-pressure profile. (Molliere 2020).

    Args:
        pRT_object : object
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
                *  patchiness : Fraction of cloud coverage
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed emission spectrum [W/m2/micron]
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    contribution = False # Not sure how to deal with having 2 separate contribution function

    # Priors for these parameters are implemented here, as they depend on each other
    T3 = ((3./4.*parameters['T_int'].value**4.*(0.1+2./3.))**0.25)*(1.0-parameters['T3'].value)
    T2 = T3*(1.0-parameters['T2'].value)
    T1 = T2*(1.0-parameters['T1'].value)
    temp_arr = np.array([T1,T2,T3])

    delta = ((10.0**(-3.0+5.0*parameters['log_delta'].value))*1e6)**(-parameters['alpha'].value)
    gravity, R_pl =  compute_gravity(parameters)

    temperatures = PT_ret_model(temp_arr, \
                            delta,
                            parameters['alpha'].value,
                            parameters['T_int'].value,
                            p_use,
                            parameters['Fe/H'].value,
                            parameters['C/O'].value,
                            conv=True)

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None
    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use
    if pressures.shape[0] != pRT_object.press.shape[0]:
        return None,None

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = fseds,
                        Kzz = kzz,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        radius = radii,
                        dist = distribution)
    wlen_model, f_lambda = spectrum_cgs_to_si(pRT_object.freq, pRT_object.flux)
    spectrum_model_cloudy = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)

    # Set the cloud abundances to 0 for clear case
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)
    pRT_object.calc_flux(temperatures,
                    abundances,
                    gravity,
                    MMW,
                    contribution = contribution,
                    fsed = fseds,
                    Kzz = kzz,
                    sigma_lnorm = sigma_lnorm,
                    b_hans = b_hans,
                    dist = distribution)
    wlen_model, f_lambda = spectrum_cgs_to_si(pRT_object.freq, pRT_object.flux)
    spectrum_model_clear = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)

    # Patchiness fraction
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) +\
                     ((1-patchiness)*spectrum_model_clear)

    return wlen_model, spectrum_model

def guillot_emission(pRT_object, \
                     parameters, \
                     PT_plot_mode = False,
                     AMR = False):
    """
    Equilibrium Chemistry Emission Model, Guillot Profile

    This model computes a emission spectrum based the Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        pRT_object : object
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
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, R_pl = compute_gravity(parameters)

    temperatures = guillot_global(p_use, \
                                10**parameters['log_kappa_IR'].value,
                                parameters['gamma'].value, \
                                gravity, \
                                parameters['T_int'].value, \
                                parameters['T_equ'].value)

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None

    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = fseds,
                        Kzz = kzz,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        radius = radii,
                        dist = distribution)
    wlen_model, f_lambda = spectrum_cgs_to_si(pRT_object.freq, pRT_object.flux)
    spectrum_model = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def guillot_transmission(pRT_object, \
                         parameters, \
                         PT_plot_mode = False,
                         AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on the Guillot profile
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source.

    Args:
        pRT_object : object
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
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, R_pl =  compute_gravity(parameters)

    temperatures = guillot_global(p_use, \
                                    10**parameters['log_kappa_IR'].value, \
                                    parameters['gamma'].value, \
                                    gravity, \
                                    parameters['T_int'].value, \
                                    parameters['T_equ'].value)


    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None

    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use

    pcloud = None
    if 'log_Pcloud' in parameters.keys():
        pcloud = 10**parameters['log_Pcloud'].value
    elif 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value

    # Calculate the spectrum
    if len(pRT_object.cloud_species)> 0:
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
        pRT_object.calc_transm(temperatures,
                                abundances,
                                gravity,
                                MMW,
                                R_pl=R_pl,
                                P0_bar=0.01,
                                sigma_lnorm = sigma_lnorm,
                                radius = radii,
                                fsed = fseds,
                                Kzz = kzz,
                                b_hans = b_hans,
                                distribution = distribution,
                                contribution = contribution)
    elif pcloud is not None:
        pRT_object.calc_transm(temperatures, \
                        abundances, \
                        gravity, \
                        MMW, \
                        R_pl=R_pl, \
                        P0_bar=0.01,
                        Pcloud = pcloud,
                        contribution = contribution)
    else:
        pRT_object.calc_transm(temperatures, \
                               abundances, \
                               gravity, \
                               MMW, \
                               R_pl=R_pl, \
                               P0_bar=0.01,
                               contribution = contribution)

    wlen_model = nc.c/pRT_object.freq/1e-4
    spectrum_model = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_tr
    return wlen_model, spectrum_model

def guillot_patchy_transmission(pRT_object, \
                                    parameters, \
                                    PT_plot_mode = False,
                                    AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source.

    Args:
        pRT_object : object
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
                * patchiness : Fraction of cloud coverage
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, R_pl =  compute_gravity(parameters)

    temperatures = guillot_global(p_use, \
                                    10**parameters['log_kappa_IR'].value, \
                                    parameters['gamma'].value, \
                                    gravity, \
                                    parameters['T_int'].value, \
                                    parameters['T_equ'].value)

    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None
    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use

    sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
    # Calculate the spectrum
    pRT_object.calc_transm(temperatures, \
                            abundances, \
                            gravity, \
                            MMW, \
                            R_pl=R_pl, \
                            P0_bar=0.01,
                            sigma_lnorm = sigma_lnorm,
                            b_hans = b_hans,
                            fsed = fseds,
                            Kzz = kzz,
                            radius = radii,
                            distribution = distribution,
                            contribution = contribution)

    wlen_model = nc.c/pRT_object.freq/1e-4
    spectrum_model_cloudy = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)
    pRT_object.calc_transm(temperatures, \
                            abundances, \
                            gravity, \
                            MMW, \
                            R_pl=R_pl, \
                            P0_bar=0.01,
                            sigma_lnorm = parameters['sigma_lnorm'].value,
                            radius = radii,
                            contribution = contribution)

    wlen_model = nc.c/pRT_object.freq/1e-4
    spectrum_model_clear = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) +\
                     ((1-patchiness)*spectrum_model_clear)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_tr
    return wlen_model, spectrum_model

def isothermal_transmission(pRT_object, \
                            parameters, \
                            PT_plot_mode = False,
                            AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Isothermal Profile

    This model computes a transmission spectrum based on an isothermal temperature-pressure profile.

    Args:
        pRT_object : object
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
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
        PT_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        AMR :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : np.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : np.array
            Computed transmission spectrum R_pl**2/Rstar**2
    """
    p_use = initialize_pressure(pRT_object.press/1e6, parameters, AMR)

    # Make the P-T profile
    temperatures = isothermal(p_use,parameters["Temp"].value)
    gravity, R_pl = compute_gravity(parameters)

    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Make the abundance profile
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if abundances is None:
        return None, None

    if PT_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = p_use

       # Calculate the spectrum
    pcloud = None
    if 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value
    elif 'log_Pcloud' in parameters.keys():
        pcloud = 10**parameters['log_Pcloud'].value

    if pcloud is not None:
        # P0_bar is important for low gravity transmission
        # spectrum. 100 is standard, 0.01 is good for small,
        # low gravity objects
        pRT_object.calc_transm(temperatures, \
                               abundances, \
                               gravity, \
                               MMW, \
                               R_pl=R_pl, \
                               P0_bar=0.01,
                               Pcloud = pcloud)
    elif len(pRT_object.cloud_species) > 0:
        sigma_lnorm, fseds, kzz, b_hans, radii, distribution = fc.setup_clouds(pressures, parameters, pRT_object.cloud_species)
        pRT_object.calc_transm(temperatures, \
                                abundances, \
                                gravity, \
                                MMW, \
                                R_pl=R_pl, \
                                P0_bar=0.01,
                                sigma_lnorm = sigma_lnorm,
                                b_hans = b_hans,
                                fsed = fseds,
                                Kzz = kzz,
                                radius = radii,
                                contribution = contribution)
    else:
        pRT_object.calc_transm(temperatures, \
                               abundances, \
                               gravity, \
                               MMW, \
                               R_pl=R_pl, \
                               P0_bar=0.01,
                               contribution = contribution)
    wlen_model = nc.c/pRT_object.freq/1e-4
    spectrum_model = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_tr
    return wlen_model, spectrum_model

def initialize_pressure(press, parameters, AMR):
    """
    Provide the pressure array correctly sized to the pRT_object in use, accounting for
    the use of Adaptive Mesh Refinement around the location of clouds.

    Args:
        press : numpy.ndarray
            Pressure array from a pRT_object. Used to set the min and max values of PGLOBAL
        shape : int
            the shape of the pressure array if no AMR is used
        scaling :
            The factor by which the pressure array resolution should be scaled.
    """
    if AMR:
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
            Pressure array from a pRT_object. Used to set the min and max values of PGLOBAL
        parameters : dict
            Must include the 'pressure_simple' and 'pressure_scaling' parameters,
            used to determine the size of the high resolution grid.
    """
    try:
        pglobal_check(press,
                    parameters['pressure_simple'].value,
                    parameters['pressure_scaling'].value)
    except KeyError():
        print("You must include the pressure_simple and pressure_scaling parameters when using AMR!")
        sys.exit(1)

def pglobal_check(press,shape,scaling):
    """
    Check to ensure that the global pressure array has the correct length.
    Updates PGLOBAL.

    Args:
        press : numpy.ndarray
            Pressure array from a pRT_object. Used to set the min and max values of PGLOBAL
        shape : int
            the shape of the pressure array if no AMR is used
        scaling :
            The factor by which the pressure array resolution should be scaled.
    """
    global PGLOBAL
    if PGLOBAL.shape[0] != int(scaling*shape):
        PGLOBAL = np.logspace(np.log10(press[0]),
                              np.log10(press[-1]),
                              int(scaling*shape))


