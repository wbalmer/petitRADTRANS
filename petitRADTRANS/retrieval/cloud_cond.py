import copy as cp

import numpy as np
from scipy.interpolate import interp1d
import logging
plotting = False
if plotting:
    import pylab as plt
    from petitRADTRANS import nat_cst as nc

#############################################################
# Cloud Cond
#############################################################
# This file allows the calculation of equilibrium cloud abundances
# and base pressures
#
# TODO: Make a better cloud module.

#############################################################
# To calculate X_Fe from [Fe/H], C/O
#############################################################

# metal species
metals = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Fe', 'Ni']

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
nfracs = {
    'H': 0.9207539305,
    'He': 0.0783688694,
    'C': 0.0002478241,
    'N': 6.22506056949881e-05,
    'O': 0.0004509658,
    'Na': 1.60008694353205e-06,
    'Mg': 3.66558742055362e-05,
    'Al': 2.595e-06,
    'Si': 2.9795e-05,
    'P': 2.36670201997668e-07,
    'S': 1.2137900734604e-05,
    'Cl': 2.91167958499589e-07,
    'K': 9.86605611925677e-08,
    'Ca': 2.01439011429255e-06,
    'Ti': 8.20622804366359e-08,
    'V': 7.83688694089992e-09,
    'Fe': 2.91167958499589e-05,
    'Ni': 1.52807116806281e-06
}

# atomic masses  TODO use molmass instead
masses = {
    'H': 1.,
    'He': 4.,
    'C': 12.,
    'N': 14.,
    'O': 16.,
    'Na': 23.,
    'Mg': 24.3,
    'Al': 27.,
    'Si': 28.,
    'P': 31.,
    'S': 32.,
    'Cl': 35.45,
    'K': 39.1,
    'Ca': 40.,
    'Ti': 47.9,
    'V': 51.,
    'Fe': 55.8,
    'Ni': 58.7
}

def setup_clouds(pressures, parameters, cloud_species):
    """
    This function provides the set of cloud parameters used in
    petitRADTRANS. This will be some combination of atmospheric
    parameters (fsed and Kzz), distribution descriptions (log normal or hansen)
    and the cloud particle radius. Fsed and the particle radii can be provided
    on a per-cloud basis.

    Args:
        pressures : np.ndarray
            The pressure array used to provide the atmospheric grid
        parameters : dict
            The dictionary of parameters passed to the model function. Should contain:
                 *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
        cloud_species : list
            A list of the names of each of the cloud species used in the atmosphere.

    Returns:
        sigma_lnorm : float, None
            The width of a log normal particle size distribution
        fseds : dict, None
            The sedimentation fraction for each cloud species in the atmosphere
        kzz : np.ndarray, None
            The vertical mixing parameter
        b_hans : float, None
            The width of a hansen particle size distribution
        radii : dict, None
            The central radius of the particle size distribution
        distribution : string
            Either "lognormal" or "hansen" - tells pRT which distribution to use.
    """


    sigma_lnorm = None
    radii = None
    b_hans = None
    distribution = "lognormal"
    fseds = None
    kzz = None

    # Setup distribution shape
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = parameters['b_hans'].value
        distribution = "hansen"

    # Are we retrieving the particle radii?
    radii = {}
    for cloud in cloud_species:
        if 'log_cloud_radius_' + cloud.split('_')[0] in parameters.keys():
            radii[cloud] = 10**parameters['log_cloud_radius_' + cloud.split('_')[0]].value * np.ones_like(p_use)
    if not radii:
        radii = None

    # per-cloud species fseds
    fseds = get_fseds(parameters, cloud_species)
    if "log_kzz" in parameters.keys():
        kzz = 10**parameters["log_kzz"].value * np.ones_like(pressures)
    return sigma_lnorm, fseds, kzz, b_hans, radii, distribution

def get_fseds(parameters, cloud_species):
    """
    This function checks to see if the fsed values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """

    fseds = {}
    for cloud in cloud_species:
        cname = cloud.split('_')[0]
        if 'fsed_'+cname in parameters.keys():
            fseds[cloud] = parameters['fsed_'+cname].value
        elif 'fsed' in parameters.keys():
                fseds[cloud] = parameters['fsed'].value
    if not fseds:
        fseds = None
    return fseds

def setup_clouds(pressures, parameters, cloud_species):
    """
    This function provides the set of cloud parameters used in
    petitRADTRANS. This will be some combination of atmospheric
    parameters (fsed and Kzz), distribution descriptions (log normal or hansen)
    and the cloud particle radius. Fsed and the particle radii can be provided
    on a per-cloud basis.

    Args:
        pressures : np.ndarray
            The pressure array used to provide the atmospheric grid
        parameters : dict
            The dictionary of parameters passed to the model function. Should contain:
                 *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
        cloud_species : list
            A list of the names of each of the cloud species used in the atmosphere.

    Returns:
        sigma_lnorm : float, None
            The width of a log normal particle size distribution
        fseds : dict, None
            The sedimentation fraction for each cloud species in the atmosphere
        kzz : np.ndarray, None
            The vertical mixing parameter
        b_hans : float, None
            The width of a hansen particle size distribution
        radii : dict, None
            The central radius of the particle size distribution
        distribution : string
            Either "lognormal" or "hansen" - tells pRT which distribution to use.
    """


    sigma_lnorm = None
    radii = None
    b_hans = None
    distribution = "lognormal"
    fseds = None
    kzz = None

    # Setup distribution shape
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters['sigma_lnorm'].value
    elif "b_hans" in parameters.keys():
        b_hans = get_bhans(parameters, cloud_species, pressures.shape[0])
        distribution = "hansen"

    # Are we retrieving the particle radii?
    radii = {}
    for cloud in cloud_species:
        if 'log_cloud_radius_' + cloud.split('_')[0] in parameters.keys():
            radii[cloud] = 10**parameters['log_cloud_radius_' + cloud.split('_')[0]].value * np.ones_like(pressures)
    if not radii:
        radii = None

    # per-cloud species fseds
    fseds = get_fseds(parameters, cloud_species)
    if "log_kzz" in parameters.keys():
        kzz = 10**parameters["log_kzz"].value * np.ones_like(pressures)
    return sigma_lnorm, fseds, kzz, b_hans, radii, distribution

def cloud_dict(parameters, parameter_name, cloud_species, shape = 0):
    """
    This is a generic method to create a dictionary of
    parameters values for a given cloud parameterization, testing if
    the parameter should be filled on a per-species basis or
    if each cloud species should have the same value.
    """
    output_dictionary = {}
    for cloud in cloud_species:
        cname = cloud.split('_')[0]
        if parameter_name + "_" + cname in parameters.keys():
            output = parameters[parameter_name+"_"+cname].value
        elif parameter_name in parameters.keys():
            output = parameters[parameter_name].value
        if shape > 0:
            output = output * np.ones(shape)
        output_dictionary[cloud] = output
    if not output_dictionary:
        output_dictionary = None
    return output_dictionary

def get_fseds(parameters, cloud_species):
    """
    This function checks to see if the fsed values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters,"fsed", cloud_species)

def get_bhans(parameters, cloud_species, shape = 0):
    """
    This function checks to see if the bhans values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters, "b_hans", cloud_species, shape = shape)

def return_cloud_mass_fraction(name,FeH,CO):
    if "Fe(c)" in name:
        return return_XFe(FeH,CO)
    if "MgSiO3(c)" in name:
        return return_XMgSiO3(FeH,CO)
    if "Mg2SiO4(c)" in name:
        return return_XMg2SiO4(FeH,CO)
    if "Na2S(c)" in name:
        return return_XNa2S(FeH,CO)
    if "KCL(c)" in name:
        return return_XKCL(FeH,CO)
    else:
        logging.warn(f"The cloud {name} is not currently implemented.")
        return np.zeros_like(FeH)

def simple_cdf(name,press, temp, FeH, CO, MMW = 2.33):
        if "Fe(c)" in name:
            return simple_cdf_Fe(press, temp, FeH, CO, MMW)
        if "MgSiO3(c)" in name:
            return simple_cdf_MgSiO3(press, temp, FeH, CO, MMW)
        if "Mg2SiO4(c)" in name:
            return simple_cdf_Mg2SiO4(press, temp, FeH, CO, MMW)
        if "Na2S(c)" in name:
            return simple_cdf_Na2S(press, temp, FeH, CO, MMW)
        if "KCL(c)" in name:
            return simple_cdf_KCL(press, temp, FeH, CO, MMW)
        else:
            logging.warn(f"The cloud {name} is not currently implemented.")
            return np.zeros_like(FeH)
def simple_cdf_free(name,press, temp, mfrac, MMW = 2.33):
        if "Fe(c)" in name:
            return simple_cdf_Fe_free(press, temp, mfrac, MMW)
        if "MgSiO3(c)" in name:
            return simple_cdf_MgSiO3_free(press, temp, mfrac, MMW)
        if "Mg2SiO4(c)" in name:
            return simple_cdf_Mg2SiO4_free(press, temp, FeH, CO, MMW)
        if "Na2S(c)" in name:
            return simple_cdf_Na2S_free(press, temp, mfrac, MMW)
        if "KCL(c)" in name:
            return simple_cdf_KCL_free(press, temp, mfrac, MMW)
        else:
            logging.warn(f"The cloud {name} is not currently implemented.")
            return np.zeros_like(FeH)

def return_XFe(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    XFe = masses['Fe']*nfracs_use['Fe']
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    XFe = XFe / add

    return XFe

def return_XMgSiO3(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_mgsio3 = np.min([nfracs_use['Mg'], \
                            nfracs_use['Si'], \
                            nfracs_use['O']/3.])
    masses_mgsio3 = masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']

    Xmgsio3 = masses_mgsio3*nfracs_mgsio3
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xmgsio3 = Xmgsio3 / add

    return Xmgsio3

def return_XMg2SiO4(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_mg2sio4 = np.min([nfracs_use['Mg']/2., \
                            nfracs_use['Si'], \
                            nfracs_use['O']/4.])
    masses_mg2sio4 = 2* masses['Mg'] \
      + masses['Si'] \
      + 4. * masses['O']

    Xmg2sio4 = masses_mg2sio4*nfracs_mg2sio4
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xmg2sio4 = Xmg2sio4 / add

    return Xmg2sio4

def return_XNa2S(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_na2s = np.min([nfracs_use['Na']/2., \
                            nfracs_use['S']])
    masses_na2s = 2.*masses['Na'] \
      + masses['S']

    Xna2s = masses_na2s*nfracs_na2s
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xna2s = Xna2s / add

    return Xna2s

def return_XKCL(FeH, CO):

    nfracs_use = cp.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_kcl = np.min([nfracs_use['K'], \
                            nfracs_use['Cl']])
    masses_kcl = masses['K'] \
      + masses['Cl']

    Xkcl = masses_kcl*nfracs_kcl
    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xkcl = Xkcl / add

    return Xkcl

#############################################################
# Fe saturation pressure, from Ackerman & Marley (2001), including erratum (P_vap is in bar, not cgs!)
#############################################################

def return_T_cond_Fe(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(15.71 - 47664./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_l(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(9.86 - 37120./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_comb(FeH, CO, MMW = 2.33):

    P1, T1 = return_T_cond_Fe(FeH, CO, MMW)
    P2, T2 = return_T_cond_Fe_l(FeH, CO, MMW)

    retP = np.zeros_like(P1)
    index = P1<P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]
    return retP, T2

def return_T_cond_Fe_free(XFe, MMW = 2.33):

    T = np.linspace(100.,12000.,1200)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(15.71 - 47664./x)
    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_l_free(XFe, MMW = 2.33):

    T = np.linspace(100.,12000.,1200)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(9.86 - 37120./x)
    return P_vap(T)/(XFe*MMW/masses['Fe']), T

def return_T_cond_Fe_comb_free(XFe, MMW = 2.33):

    P1, T1 = return_T_cond_Fe_free(XFe, MMW)
    P2, T2 = return_T_cond_Fe_l_free(XFe, MMW)
    retP = np.zeros_like(P1)
    index = P1<P2
    retP[index] = P1[index]
    retP[~index] = P2[~index]
    return retP, T2

def return_T_cond_MgSiO3(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(25.37 - 58663./x)
    #P_vap = lambda x: np.exp(17.89 - 2.0*FeH - 28571.43/x)
    Xmgsio3 = return_XMgSiO3(FeH, CO)

    m_mgsio3 =  masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']
    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T

def return_T_cond_Mg2SiO4(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    # Visscher 2010 condensation curve
    P_vap = lambda x: np.exp(15.92 - 1.97*FeH - 27027.03/x)


    Xmg2sio4 = return_XMg2SiO4(FeH, CO)

    m_mg2sio4 =  2.*masses['Mg'] \
      + masses['Si'] \
      + 4. * masses['O']
    return P_vap(T), T

def return_T_cond_MgSiO3_free(Xmgsio3, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(25.37 - 58663./x)
    m_mgsio3 =  masses['Mg'] \
      + masses['Si'] \
      + 3. * masses['O']
    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T

def return_T_cond_Mg2SiO4_free(Xmg2sio4, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    # Taken from Ackerman & Marley (2001)
    # including their erratum
    P_vap = lambda x: np.exp(25.37 - 58663./x)
    m_mg2sio4 = 2* masses['Mg'] \
      + masses['Si'] \
      + 4. * masses['O']
    return P_vap(T)/(Xmg2sio4*MMW/m_mg2sio4), T

def return_T_cond_Na2S(FeH, CO, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.
    P_vap = lambda x: 1e1**(8.55 - 13889./x - 0.5*FeH)/2.

    Xna2s = return_XNa2S(FeH, CO)

    m_na2s =  2.*masses['Na'] \
      + masses['S']
    return P_vap(T)/(Xna2s*MMW/m_na2s), T

def return_T_cond_Na2S_free(Xna2s, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    # This is the partial pressure of Na, so
    # Divide by factor 2 to get the partial
    # pressure of the hypothetical Na2S gas
    # particles, this is OK: there are
    # more S than Na atoms at solar
    # abundance ratios.

    # We're also using [Na/H] as a proxy for [Fe/H]
    # Definitely not strictly correct, but should be
    # good enough for ~ solar compositions. [+- 1 for Fe/H]
    # Assumes constant vertical abundance
    m_na2s =  2.*masses['Na'] \
      + masses['S']
    P_vap = lambda x: 10**(8.55 - 13889./x - 0.5*(np.log10(2*Xna2s*MMW/m_na2s)+5.7))/2

    return P_vap(T)/(Xna2s*MMW/m_na2s), T

def return_T_cond_KCL(FeH, CO, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: 1e1**(7.611 - 11382./x)

    Xkcl = return_XKCL(FeH, CO)

    m_kcl =  masses['K'] \
      + masses['Cl']
    return P_vap(T)/(Xkcl*MMW/m_kcl), T
def return_T_cond_KCL_free(Xkcl, MMW = 2.33):

    # Taken from Charnay+2018
    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: 1e1**(7.611 - 11382./x)
    m_kcl =  masses['K'] \
      + masses['Cl']
    return P_vap(T)/(Xkcl*MMW/m_kcl), T


if plotting:

    #FeHs = np.linspace(-0.5, 2., 5)
    #COs = np.linspace(0.3, 1.2, 5)
    FeHs = [0.]
    COs = [0.55]

    for FeH in FeHs:
        for CO in COs:
            P, T = return_T_cond_Fe(FeH, CO)
            plt.plot(T,P, label = 'Fe(c), [Fe/H] = '+str(FeH)+', C/O = '+str(CO), color = 'black')
            P, T = return_T_cond_Fe_l(FeH, CO)
            plt.plot(T,P, '--', label = 'Fe(l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_Fe_comb(FeH, CO)
            plt.plot(T,P, ':', label = 'Fe(c+l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_MgSiO3(FeH, CO)
            plt.plot(T,P, label = 'MgSiO3, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_Na2S(FeH, CO)
            plt.plot(T,P, label = 'Na2S, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
            P, T = return_T_cond_KCL(FeH, CO)
            plt.plot(T,P, label = 'KCL, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))


    plt.yscale('log')
    '''
    plt.xlim([0., 5000.])
    plt.ylim([1e5,1e-10])
    '''
    plt.xlim([0., 2000.])
    plt.ylim([1e2,1e-3])
    plt.legend(loc = 'best', frameon = False)
    plt.show()

def simple_cdf_Fe(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_Fe_comb(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
def simple_cdf_Fe_free(press, temp, XFe, MMW = 2.33):
    Pc, Tc = return_T_cond_Fe_comb_free(XFe, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    try:
        Tcond_on_input_grid = tcond_p(press)
    except:
        print(Pc)
        return np.min(press)
    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)

    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_MgSiO3(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_MgSiO3(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    try:
        tcond_p = interp1d(Pc, Tc)
    except ValueError:
        logging.warn("Could not interpolate pressures and temperatures!")
        return np.min(press)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
def simple_cdf_MgSiO3_free(press, temp, Xmgsio3, MMW = 2.33):

    Pc, Tc = return_T_cond_MgSiO3_free(Xmgsio3, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud


def simple_cdf_Mg2SiO4(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_Mg2SiO4(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    try:
        tcond_p = interp1d(Pc, Tc)
    except ValueError:
        logging.warn("Could not interpolate pressures and temperatures!")
        return np.min(press)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_Mg2SiO4_free(press, temp, Xmg2sio4, MMW = 2.33):
    Pc, Tc = return_T_cond_Mg2SiO4_free(Xmg2sio4, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()
    return P_cloud

def simple_cdf_Na2S(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_Na2S(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_Na2S_free(press, temp, XNa2S, MMW = 2.33):

    Pc, Tc = return_T_cond_Na2S_free(XNa2S, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)

    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_KCL(press, temp, FeH, CO, MMW = 2.33):

    Pc, Tc = return_T_cond_KCL(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud

def simple_cdf_KCL_free(press, temp, XKCL, MMW = 2.33):

    Pc, Tc = return_T_cond_KCL_free(XKCL, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = np.min(press)

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.show()

    return P_cloud
if plotting:
    kappa_IR = 0.01
    gamma = 0.4
    T_int = 200.
    T_equ = 1550.
    gravity = 1e1**2.45

    pressures = np.logspace(-6, 2, 100)

    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    simple_cdf_Fe(pressures, temperature, 0., 0.55)
    simple_cdf_MgSiO3(pressures, temperature, 0., 0.55)

    T_int = 200.
    T_equ = 800.
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    simple_cdf_Na2S(pressures, temperature, 0., 0.55)

    T_int = 150.
    T_equ = 650.
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    simple_cdf_KCL(pressures, temperature, 0., 0.55)

