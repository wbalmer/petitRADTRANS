import sys, os
import copy as cp
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.interpolate import interp1d,CubicSpline
from typing import Tuple

from petitRADTRANS import nat_cst as nc
from petitRADTRANS.temperature_structures import guillot_global_ret, guillot_modif, PT_ret_model

from petitRADTRANS.retrieval.chemistry import get_abundances
from petitRADTRANS.retrieval import cloud_cond as fc
from petitRADTRANS import poor_mans_nonequ_chem as pm
from .util import surf_to_meas, calc_MMW, compute_gravity, fixed_length_amr
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

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline [K]
                *  T2 : Middle temperature spline [K]
                *  T1 : Outer temperature spline [K]
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  log_kzz : Vertical mixing parameter
                *  fsed : sedimentation parameter
                *  log_X_cb : Scaling factor for equilibrium cloud abundances.
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
    #start = time.time()
    pglobal_check(pRT_object.press/1e6,
                    parameters['pressure_simple'].value,
                    parameters['pressure_scaling'].value)
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    #for key, val in parameters.items():
    #    print(key,val.value)

    # Priors for these parameters are implemented here, as they depend on each other
    T3 = ((3./4.*parameters['T_int'].value**4.*(0.1+2./3.))**0.25)*(1.0-parameters['T3'].value)
    T2 = T3*(1.0-parameters['T2'].value)
    T1 = T2*(1.0-parameters['T1'].value)
    delta = ((10.0**(-3.0+5.0*parameters['log_delta'].value))*1e6)**(-parameters['alpha'].value)
    Kzz_use = (10.0**parameters['log_kzz'].value ) * np.ones_like(p_use)
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
    if PT_plot_mode:
        return p_use, temperatures

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)

    # Only include the high resolution pressure array near the cloud base.
    pressures = p_use
    #print(PGLOBAL.shape, pressures.shape, pressures[small_index].shape, pRT_object.press.shape)
    if AMR:
        #pRT_object.setup_opa_structure(PGLOBAL[small_index])
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        #pRT_object.setup_opa_structure(pressures)
    # Calculate the spectrum
    if pressures.shape[0] != pRT_object.press.shape[0]:
        return None,None
    pRT_object.press = pressures*1e6

    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"

    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = parameters['b_hans'].value
        distribution = "hansen"

    # per-cloud species fseds
    fseds = {}
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        try:
            #print(cname)
            fseds[cloud] = parameters['fsed_'+cname].value
        except:
            fseds[cloud] = parameters['fsed'].value
    # calculate the spectrum
    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = fseds,
                        Kzz = Kzz_use,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        dist = distribution)
    # Getting the model into correct units (W/m2/micron)
    wlen_model = nc.c/pRT_object.freq/1e-4
    wlen = nc.c/pRT_object.freq
    f_lambda = pRT_object.flux*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    #f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    #f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7
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
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline [K]
                *  T2 : Middle temperature spline [K]
                *  T1 : Outer temperature spline [K]
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  log_kzz : Vertical mixing parameter
                *  fsed : sedimentation parameter
                *  log_X_cb : Scaling factor for equilibrium cloud abundances.
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
    #start = time.time()
    pglobal_check(pRT_object.press/1e6,
                    parameters['pressure_simple'].value,
                    parameters['pressure_scaling'].value)
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
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
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    temperatures = PT_ret_model(temp_arr, \
                            delta,
                            parameters['alpha'].value,
                            parameters['T_int'].value,
                            p_use,
                            parameters['Fe/H'].value,
                            parameters['C/O'].value,
                            conv=True)
    if PT_plot_mode:
        return p_use, temperatures

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    Kzz_use = (10.0**parameters['log_kzz'].value ) * np.ones_like(p_use)

    # Only include the high resolution pressure array near the cloud base.
    pressures = p_use
    #print(PGLOBAL.shape, pressures.shape, pressures[small_index].shape, pRT_object.press.shape)
    if AMR:
        #pRT_object.setup_opa_structure(PGLOBAL[small_index])
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        #pRT_object.setup_opa_structure(pressures)
    # Calculate the spectrum
    if pressures.shape[0] != pRT_object.press.shape[0]:
        return None,None
    pRT_object.press = pressures*1e6

    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = parameters['b_hans'].value
        distribution = "hansen"

    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = parameters['fsed'].value,
                        Kzz = Kzz_use,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        dist = distribution)

    # Getting the model into correct units (W/m2/micron)
    wlen_model = nc.c/pRT_object.freq/1e-4
    wlen = nc.c/pRT_object.freq
    f_lambda = pRT_object.flux*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    #f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    #f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7
    spectrum_model_cloudy = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(temperatures)
    pRT_object.calc_flux(temperatures,
                    abundances,
                    gravity,
                    MMW,
                    contribution = False,
                    fsed = parameters['fsed'].value,
                    Kzz = Kzz_use,
                    sigma_lnorm = sigma_lnorm,
                    b_hans = b_hans,
                    dist = distribution)
    # Getting the model into correct units (W/m2/micron)
    wlen_model = nc.c/pRT_object.freq/1e-4
    wlen = nc.c/pRT_object.freq
    f_lambda = pRT_object.flux*nc.c/wlen**2.
    f_lambda = f_lambda * 1e-7
    spectrum_model_clear = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_clouds) +\
                     ((1-patchiness)*spectrum_model_clear)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def guillot_free_emission(pRT_object, \
                            parameters, \
                            PT_plot_mode = False,
                            AMR = False):
    """
    Free Chemistry Emission Model

    This model computes an emission spectrum based on free retrieval chemistry,
    free Ackermann-Marley clouds and a Guillot temperature-pressure profile. (Molliere 2018).

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  D_pl : Distance to the planet in [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities
                *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                *  log_kzz : Vertical mixing parameter
                *  fsed : sedimentation parameter
                *  species : Log abundances for each species in rd.line_list
                   (species stands in for the actual name)
                *  log_X_cb : Log cloud abundances.
                *  Pbase : log of cloud base pressure for each species.
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

    #for key, val in parameters.items():
    #    print(key,val.value)

    # let's start out by setting up our global pressure arrays
    # This is used for the hi res bins for AMR
    pglobal_check(pRT_object.press/1e6,
                  parameters['pressure_simple'].value,
                  parameters['pressure_scaling'].value)
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6

    # We need 2 of 3 for gravity, radius and mass
    # So check which parameters are included in the Retrieval
    # and calculate the third if necessary
    gravity, R_pl =  compute_gravity(parameters)

    # We're using a guillot profile
    temperatures = nc.guillot_global(p_use, \
                                10**parameters['log_kappa_IR'].value,
                                parameters['gamma'].value, \
                                gravity, \
                                parameters['T_int'].value, \
                                parameters['T_equ'].value)

    # Set up gas phase abundances, check to make sure that
    # the total mass fraction is < 1.0
    # We assume vertically constant abundances
    abundances = {}
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    # Imposing strict limit on msum to ensure H2 dominated composition
    if abundances is None:
        return None, None

    # Set up the adaptive pressure grid
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = pRT_object.press/1e6

    # Now that we have the pressure array, we can set up the
    # cloud abundance profile
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        abundances[cname] = np.zeros_like(pRT_object.press)
        try:
            abundances[cname][pressures < Pbases[cname]] = 10**parameters['log_X_cb_'+cname].value *\
                        ((pressures[pressures <= Pbases[cname]]/Pbases[cname])**parameters['fsed'].value)
        except:
            print(cname)
            print(f"{Pbases[cname]}")
            print(f"{10**parameters['log_X_cb_'+cname].value}")
            print(f"{(pressures[pressures <= Pbases[cname]]/Pbases[cname])**parameters['fsed'].value}\n")
            return None,None

    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return pressures, temperatures

    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = parameters['b_hans'].value
        distribution = "hansen"

    # Calculate the spectrum
    # per-cloud species fseds
    fseds = {}
    for cloud in pRT_object.cloud_species:
        cname = cloud.split('_')[0]
        try:
            fseds[cloud] = parameters['fsed_'+cname].value
        except:
            fseds[cloud] = parameters['fsed'].value
    pRT_object.calc_flux(temperatures, \
                     abundances, \
                     gravity, \
                     MMW, \
                     contribution = contribution,
                     fsed = fseds,
                     Kzz = 10**parameters['log_kzz'].value * np.ones_like(pressures),
                     sigma_lnorm = sigma_lnorm,
                     b_hans = b_hans,
                     dist = distribution)

    # Change units to W/m^2/micron
    # and wavelength units in micron
    wlen_model = nc.c/pRT_object.freq/1e-4
    wlen = nc.c/pRT_object.freq
    f_lambda = pRT_object.flux*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    #f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    #f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7

    # Scale to planet distance
    spectrum_model = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def guillot_eqchem_transmission(pRT_object, \
                                    parameters, \
                                    PT_plot_mode = False,
                                    AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on equilibrium chemistry
    and a Guillot temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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
    # Make the P-T profile
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, R_pl =  compute_gravity(parameters)

    temperatures = nc.guillot_global(p_use, \
                                    10**parameters['log_kappa_IR'].value, \
                                    parameters['gamma'].value, \
                                    gravity, \
                                    parameters['T_int'].value, \
                                    parameters['T_equ'].value)

    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return p_use, temperatures

    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = pRT_object.press/1e6

    radii = {}
    for cloud in pRT_object.cloud_species:
        radii[cloud] = 10**parameters['log_cloud_radius_' + cloud.split('_')[0]].value * np.ones_like(p_use)
    # Calculate the spectrum
    if 'sigma_lnorm' in parameters.keys():
        pRT_object.calc_transm(temperatures, \
                                abundances, \
                                gravity, \
                                MMW, \
                                R_pl=R_pl, \
                                P0_bar=0.01,
                                sigma_lnorm = parameters['sigma_lnorm'].value,
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
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def guillot_patchy_eqchem_transmission(pRT_object, \
                                    parameters, \
                                    PT_plot_mode = False,
                                    AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on equilibrium chemistry
    and a Guillot temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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
    # Make the P-T profile
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # Calculate the spectrum
    gravity, R_pl =  compute_gravity(parameters)

    temperatures = nc.guillot_global(p_use, \
                                    10**parameters['log_kappa_IR'].value, \
                                    parameters['gamma'].value, \
                                    gravity, \
                                    parameters['T_int'].value, \
                                    parameters['T_equ'].value)

    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return p_use, temperatures

    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = pRT_object.press/1e6

    radii = {}
    for cloud in pRT_object.cloud_species:
        radii[cloud] = 10**parameters['log_cloud_radius_' + cloud.split('_')[0]].value * np.ones_like(p_use)
    # Calculate the spectrum
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
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def guillot_eqchem_emission(pRT_object, \
                            parameters, \
                            PT_plot_mode = False,
                            AMR = False):
    """
    Equilibrium Chemistry Emission Model, Guillot Profile

    This model computes a transmission spectrum based on equilibrium chemistry
    and a Guillot temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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
    # Make the P-T profile
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    gravity, R_pl = compute_gravity(parameters)


    temperatures = nc.guillot_global(p_use, \
                                10**parameters['log_kappa_IR'].value,
                                parameters['gamma'].value, \
                                gravity, \
                                parameters['T_int'].value, \
                                parameters['T_equ'].value)

    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return p_use, temperatures
    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)
    Kzz_use = (10.0**parameters['log_kzz'].value ) * np.ones_like(p_use)

    # Only include the high resolution pressure array near the cloud base.
    pressures = p_use
    #print(PGLOBAL.shape, pressures.shape, pressures[small_index].shape, pRT_object.press.shape)
    if AMR:
        #pRT_object.setup_opa_structure(PGLOBAL[small_index])
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        Kzz_use = Kzz_use[small_index]
        #pRT_object.setup_opa_structure(pressures)
    # Calculate the spectrum
    #print(pressures.shape,pRT_object.press.shape[0],p_use.shape)
    #print(temperatures.shape)
    if pressures.shape[0] != pRT_object.press.shape[0]:
        return None,None
    pRT_object.press = pressures*1e6

    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = parameters['b_hans'].value
        distribution = "hansen"

    pRT_object.calc_flux(temperatures,
                        abundances,
                        gravity,
                        MMW,
                        contribution = contribution,
                        fsed = parameters['fsed'].value,
                        Kzz = Kzz_use,
                        sigma_lnorm = sigma_lnorm,
                        b_hans = b_hans,
                        dist = distribution)
    # Getting the model into correct units (W/m2/micron)
    wlen_model = nc.c/pRT_object.freq/1e-4
    wlen = nc.c/pRT_object.freq
    f_lambda = pRT_object.flux*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    #f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    #f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7
    spectrum_model = surf_to_meas(f_lambda,
                                  R_pl,
                                  parameters['D_pl'].value)
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_em
    return wlen_model, spectrum_model

def isothermal_eqchem_transmission(pRT_object, \
                                    parameters, \
                                    PT_plot_mode = False,
                                    AMR = False):
    """
    Equilibrium Chemistry Transmission Model, Isothermal Profile

    This model computes a transmission spectrum based on equilibrium chemistry
    and a Guillot temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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
    # Make the P-T profile
    pressures = pRT_object.press/1e6
    temperatures = parameters['Temp'].value * np.ones_like(pressures)
    gravity, R_pl =  compute_gravity(parameters)
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return pressures, temperatures

    # Make the abundance profile
    COs = parameters['C/O'].value * np.ones_like(pressures)
    FeHs = parameters['Fe/H'].value * np.ones_like(pressures)

    abundances_interp = pm.interpol_abundances(COs, \
                                               FeHs, \
                                               temperatures, \
                                               pressures)

    abundances = {}
    for species in pRT_object.line_species:
        abundances[species] = abundances_interp[species.split('_R_')[0]]
    abundances['H2'] = abundances_interp['H2']
    abundances['He'] = abundances_interp['He']

    MMW = abundances_interp['MMW']
    pcloud = None
    if 'log_Pcloud' in parameters.keys():
        pcloud = 10**parameters['log_Pcloud'].value
    elif 'Pcloud' in parameters.keys():
        pcloud = parameters['log_Pcloud'].value
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
                               Pcloud = pcloud,
                               contribution = contribution)
    else:
        pRT_object.calc_transm(temperatures, \
                               abundances, \
                               10**gravity, \
                               MMW, \
                               R_pl=R_pl, \
                               P0_bar=0.01,
                               contribution = contribution)
    wlen_model = nc.c/pRT_object.freq/1e-4
    spectrum_model = (pRT_object.transm_rad/parameters['Rstar'].value)**2.
    if contribution:
        return wlen_model, spectrum_model, pRT_object.contr_tr
    return wlen_model, spectrum_model


def isothermal_free_transmission(pRT_object, \
                                parameters, \
                                PT_plot_mode = False,
                                AMR = False):
    """
    Free Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on free retrieval chemistry
    and an isothermal temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  Temp : Isothermal temperature [K]
                *  species : Abundances for each species used in the retrieval
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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

    # Make the P-T profile
    pressures = pRT_object.press/1e6
    gravity, R_pl = compute_gravity(parameters)
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value
    temperatures = parameters['Temp'].value * np.ones_like(pressures)
    #for key,value in parameters.items():
    #    print(key,value.value)
    # If in evaluation mode, and PTs are supposed to be plotted
    if PT_plot_mode:
        return pressures, temperatures
    abundances = {}
    msum = 0.0
    for species in pRT_object.line_species:
        spec = species.split('_R_')[0]
        abundances[species] = 10**parameters[spec].value * np.ones_like(pressures)
        msum += 10**parameters[spec].value
    if msum > 0.1:
        return None, None
    abundances['H2'] = 0.766 * (1.0-msum) * np.ones_like(pressures)
    abundances['He'] = 0.234 * (1.0-msum) * np.ones_like(pressures)

    #MMW = abundances_interp['MMW']
    MMW = calc_MMW(abundances)

    # Calculate the spectrum
    pcloud = 100.0
    if 'Pcloud' in parameters.keys():
        pcloud = parameters['Pcloud'].value
    elif 'log_Pcloud' in parameters.keys():
        pcloud = 10**parameters['log_Pcloud'].value
    # Calculate the spectrum

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

def guillot_free_transmission(pRT_object, \
                                parameters, \
                                PT_plot_mode = False,
                                AMR = False):
    """
    Free Chemistry Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on free retrieval chemistry
    and an isothermal temperature-pressure profile.

    Args:
        pRT_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  Rstar : Radius of the host star [cm]
                *  log_g : Log of surface gravity
                *  R_pl : planet radius [cm]
                *  Temp : Isothermal temperature [K]
                *  species : Abundances for each species used in the retrieval
                *  Pcloud : optional, cloud base pressure of a grey cloud deck.
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
    # Make the P-T profile
    if AMR:
        p_use = PGLOBAL
    else:
        p_use = pRT_object.press/1e6
    contribution = False
    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Calculate the spectrum
    gravity, R_pl = compute_gravity(parameters)


    temperatures = nc.guillot_global(p_use, \
                                    10**parameters['log_kappa_IR'].value, \
                                    parameters['gamma'].value, \
                                    gravity, \
                                    parameters['T_int'].value, \
                                    parameters['T_equ'].value)
    if PT_plot_mode:
        return pressures, temperatures
    abundances, MMW, small_index, Pbases = get_abundances(p_use,
                                                  temperatures,
                                                  pRT_object.line_species,
                                                  pRT_object.cloud_species,
                                                  parameters,
                                                  AMR =AMR)

    if abundances is None:
        return None, None
    if AMR:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        MMW = MMW[small_index]
        #Kzz_use = Kzz_use[small_index]
        pRT_object.press = pressures * 1e6
    else:
        pressures = pRT_object.press/1e6

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
    elif "sigma_lnorm" in parameters.keys():
        radii = {}
        for cloud in pRT_object.cloud_species:
            radii[cloud] = 10**parameters['log_cloud_radius_' + cloud.split('_')[0]].value * np.ones_like(p_use)
        # Calculate the spectrum
        pRT_object.calc_transm(temperatures, \
                                abundances, \
                                gravity, \
                                MMW, \
                                R_pl=R_pl, \
                                P0_bar=0.01,
                                sigma_lnorm = parameters['sigma_lnorm'].value,
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


