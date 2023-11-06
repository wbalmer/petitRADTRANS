import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import copy as cp
from scipy.interpolate import interp1d,CubicSpline,PchipInterpolator
from petitRADTRANS import poor_mans_nonequ_chem as pm
from scipy.interpolate import interp1d,CubicSpline

import petitRADTRANS.nat_cst as nc


def doppler_shift(wavelength_0, velocity):
    """Calculate the Doppler-shifted wavelength for electromagnetic waves.

    A negative velocity means that the source is going toward the observer. A positive velocity means the source is
    going away from the observer.

    Args:
        wavelength_0: (cm) wavelength of the wave in the referential of the source
        velocity: (cm.s-1) velocity of the source relative to the observer

    Returns:
        (cm) the wavelength of the source as measured by the observer
    """
    return wavelength_0 * np.sqrt((1 + velocity / nc.c) / (1 - velocity / nc.c))

##################################################################
### Radtrans utility for retrieval temperature model computation
##################################################################
def guillot_global(P,kappa_IR,gamma,grav,T_int,T_equ):
    ''' Returns a temperature array, in units of K,
    of the same dimensions as the pressure P
    (in bar). For this the temperature model of Guillot (2010)
    is used (his Equation 29).

    Args:
        P:
            numpy array of floats, containing the input pressure in bars.
        kappa_IR (float):
            The infrared opacity in units of :math:`\\rm cm^2/s`.
        gamma (float):
            The ratio between the visual and infrated opacity.
        grav (float):
            The planetary surface gravity in units of :math:`\\rm cm/s^2`.
        T_int (float):
            The planetary internal temperature (in units of K).
        T_equ (float):
            The planetary equilibrium temperature (in units of K).
    '''
    tau = P*1e6*kappa_IR/grav
    T_irr = T_equ*np.sqrt(2.)
    T = (0.75 * T_int**4. * (2. / 3. + tau) + \
      0.75 * T_irr**4. / 4. * (2. / 3. + 1. / gamma / 3.**0.5 + \
      (gamma / 3.**0.5 - 1. / 3.**0.5 / gamma)* \
      np.exp(-gamma * tau *3.**0.5)))**0.25
    return T

### Global Guillot P-T formula with kappa/grav replaced by delta
def guillot_global_ret(P,delta,gamma,T_int,T_equ):
    delta = np.abs(delta)
    gamma = np.abs(gamma)
    T_int = np.abs(T_int)
    T_equ = np.abs(T_equ)
    tau = P*1e6*delta
    T_irr = T_equ*np.sqrt(2.)
    T = (0.75*T_int**4.*(2./3.+tau) + \
      0.75*T_irr**4./4.*(2./3.+1./gamma/3.**0.5+ \
                         (gamma/3.**0.5-1./3.**0.5/gamma)* \
                             np.exp(-gamma*tau*3.**0.5)))**0.25
    return T


def guillot_metallic_temperature_profile(pressures, gamma, surface_gravity,
                                         intrinsic_temperature, equilibrium_temperature, kappa_ir_z0,
                                         metallicity=None):
    """Get a Guillot temperature profile depending on metallicity.

    Args:
        pressures: (bar) pressures of the profile
        gamma: ratio between visual and infrated opacity
        surface_gravity: (cm.s-2) surface gravity
        intrinsic_temperature: (K) intrinsic temperature
        equilibrium_temperature: (K) equilibrium temperature
        kappa_ir_z0: (cm2.s-1) infrared opacity
        metallicity: ratio of heavy elements abundance over H abundance with respect to the solar ratio

    Returns:
        temperatures: (K) the temperature at each pressures of the atmosphere
    """
    if metallicity is not None:
        kappa_ir = kappa_ir_z0 * metallicity
    else:
        kappa_ir = kappa_ir_z0

    temperatures = guillot_global(
        P=pressures,
        kappa_IR=kappa_ir,
        gamma=gamma,
        grav=surface_gravity,
        T_int=intrinsic_temperature,
        T_equ=equilibrium_temperature
    )

    return temperatures


### Modified Guillot P-T formula
def guillot_modif(P,delta,gamma,T_int,T_equ,ptrans,alpha):
    return guillot_global_ret(P,np.abs(delta),np.abs(gamma), \
                                  np.abs(T_int),np.abs(T_equ))* \
                                  (1.-alpha*(1./(1.+ P/ptrans)))


def hz2um(frequency):
    """Convert frequencies into wavelengths

    Args:
        frequency: (Hz) the frequency to convert

    Returns:
        (um) the corresponding wavelengths
    """
    return nc.c / frequency * 1e4  # cm to um


def isothermal(P, T):
    return T * np.ones_like(P)

### Box car conv. average, found on stackoverflow somewhere
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

### Function to make temp
def make_press_temp(rad_trans_params):

    press_many = np.logspace(-8,5,260)
    t_no_ave = guillot_modif(press_many, \
        1e1**rad_trans_params['log_delta'],1e1**rad_trans_params['log_gamma'], \
        rad_trans_params['t_int'],rad_trans_params['t_equ'], \
        1e1**rad_trans_params['log_p_trans'],rad_trans_params['alpha'])

    # new
    press_many_new = 1e1**running_mean(np.log10(press_many), 25)
    t_new          = running_mean(t_no_ave  , 25)
    index_new      = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new       = t_new[index_new][::2]
    press_new      = press_many_new[index_new][::2]

    return press_new, temp_new

### Function to make temp
def make_press_temp_iso(rad_trans_params):

    press_many = np.logspace(-8,5,260)
    t_no_ave = rad_trans_params['t_equ']  * np.ones_like(press_many)

    # new
    press_many_new = 1e1**running_mean(np.log10(press_many), 25)
    t_new          = running_mean(t_no_ave  , 25)
    index_new      = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new       = t_new[index_new][::2]
    press_new      = press_many_new[index_new][::2]

    return press_new, temp_new

### Global Guillot P-T formula with kappa/grav replaced by delta
def PT_ret_model(T3, delta, alpha, tint, press, FeH, CO, conv = True):
    """
    Self-luminous retrieval P-T model.

    Args:
        T3 : np.array([t1, t2, t3])
            temperature points to be added on top
            radiative Eddington structure (above tau = 0.1).
            Use spline interpolation, t1 < t2 < t3 < tconnect as prior.
        delta : float
            proportionality factor in tau = delta * press_cgs**alpha
        alpha : float
            power law index in tau = delta * press_cgs**alpha
            For the tau model: use proximity to kappa_rosseland photosphere
            as prior.
        tint : float
            internal temperature of the Eddington model
        press : np.ndarray
            input pressure profile in bar
        conv : bool
            enforce convective adiabat yes/no
        CO : float
            C/O for the nabla_ad interpolation
        FeH : float
            metallicity for the nabla_ad interpolation
    Returns:
        Tret : np.ndarray
            The temperature as a function of atmospheric pressure.
    """

    from petitRADTRANS import poor_mans_nonequ_chem as pm

    # Go grom bar to cgs
    press_cgs = press * 1e6

    # Calculate the optical depth
    tau = delta * press_cgs ** alpha

    # This is the eddington temperature
    tedd = (3. / 4. * tint ** 4. * (2. / 3. + tau)) ** 0.25

    ab = pm.interpol_abundances(CO * np.ones_like(tedd),
                                FeH * np.ones_like(tedd),
                                tedd,
                                press)

    nabla_ad = ab['nabla_ad']

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd)) / np.diff(np.log(press_cgs))
        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:] + nab_rad[:-1]) / 2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        # TODO: Check remains convective and convergence
        for i in range(10):
            if i == 0:
                t_take = cp.copy(tedd)
            else:
                t_take = cp.copy(tfinal)  # TODO possible reference before assignment

            ab = pm.interpol_abundances(CO * np.ones_like(t_take),
                                        FeH * np.ones_like(t_take),
                                        t_take,
                                        press)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:] + nabla_ad[:-1]) / 2.
            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index] * np.mean(np.diff(np.log(press_cgs)))
            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])
            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew) + tstart)

            # Add upper radiative and
            # lower conective part into one single array
            tfinal = cp.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take - tfinal) / t_take) < 0.01:
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(tau):
        # Returns the pressure at a given tau, in cgs
        return (tau / delta) ** (1. / alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                             np.log10(p_bot_spline),
                                             4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = 10 ** np.arange(np.log10(p_bot_spline),
                                                  np.log10(press_cgs[-1]),
                                                  np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]),
                                             np.log10(p_bot_spline),
                                             7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the
            # radiative+convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low) + len(support_points_high) - 1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal, kind='cubic')
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low)) - 1)] = T3
            # t_support[:3] = tfintp(support_points_low)
            # The temperature at pressures above p_bot_spline
            # (from the radiative-convectice solution)
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(press_cgs, tret, kind='cubic')  # TODO possible reference before assignment
            t_support[:(int(len(support_points_low)) - 1)] = \
                tfintp1(support_points[:(int(len(support_points_low)) - 1)])

            tfintp = interp1d(press_cgs, tfinal)
            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)) - 1] = tfintp(p_bot_spline)
            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = \
                tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(press_cgs))

    tret[tret < 0.0] = 1.0
    # Return the temperature, the pressure at tau = 1,
    # and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret  # , press_tau(1.)/1e6, tfintp(p_bot_spline)

def madhu_seager_2009(press, pressure_points, T_set, alpha_points, beta_points):
    """
    Calculate temperatures based on the Madhusudhan and Seager (2009) parameterization.

    This function computes temperatures using the Madhu and Seager (2009) parameterization
    for a given set of pressure values, pressure breakpoints, temperature breakpoints,
    alpha values, and beta values.

    Based off of the POSEIDON implementation:
    https://github.com/MartianColonist/POSEIDON/blob/main/POSEIDON/atmosphere.py

    Parameters:
        press : (numpy.ndarray)
            An array of pressure values (in bar) at which to calculate temperatures.
        pressure_points : (list)
            A list of pressure breakpoints defining different temperature regimes.
        T_set : (float)
            A temperature at pressure_points[4] used to constrain the temperature profile.
        alpha_points : (list)
            A list of alpha values used in the parameterization for different regimes.
        beta_points : (list)
            A list of beta values used in the parameterization for different regimes.
            By default b[0] == b[1] == 0.5, unclear how well this will work if these aren't used!

    Returns:
        temperatures : (numpy.ndarray)
            An array of calculated temperatures (in K) corresponding to the input pressure values.

    Note:
    - This function assumes that pressure_points, temperature_points, alpha_points, and beta_points
      are lists with the same length, defining different pressure-temperature regimes. The function
      uses logarithmic relationships to calculate temperatures within these regimes.

    Reference:
    - Madhusudhan, N., & Seager, S. (2009). A Temperature and Abundance Retrieval Method for Exoplanet Atmospheres. 
      The Astrophysical Journal, 707(1), 24-39. https://doi.org/10.1088/0004-637X/707/1/24
    """
    temperatures = np.zeros_like(press)
    
    # Set up masks for the different temperature regions
    mask_1 = press < pressure_points[1]
    mask_2 = (press >= pressure_points[1]) & (press < pressure_points[3])
    mask_3 = press >= pressure_points[3]

    # Find index of pressure closest to the set pressure
    i_set = np.argmin(np.abs(press - pressure_points[4]))
    P_set_i = press[i_set]
    
    # Store logarithm of various pressure quantities
    log_P = np.log10(press)
    log_P_min = pressure_points[0]
    log_P_set_i = np.log10(P_set_i)

    # By default (P_set = 10 bar), so T(P_set) should be in layer 3
    if (pressure_points[4] >= pressure_points[3]):
        T3 = T_set  # T_deep is the isothermal deep temperature T3 here
        
        # Use the temperature parameter to compute boundary temperatures
        T2 = T3 - ((1.0/alpha_points[1])*(pressure_points[3] - pressure_points[2]))**(1/beta_points[1])    
        T1 = T2 + ((1.0/alpha_points[1])*(pressure_points[1] - pressure_points[2]))**(1/beta_points[1])       
        T0 = T1 - ((1.0/alpha_points[0])*(pressure_points[1] - log_P_min))**(1/beta_points[0])     
        
    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (pressure_points[4] >= pressure_points[1]):   # Temperature parameter in layer 2
        # Use the temperature parameter to compute the boundary temperatures
        T2 = T_set - ((1.0/alpha_points[1])*(log_P_set_i - pressure_points[2]))**(1/beta_points[1])  
        T1 = T2 + ((1.0/alpha_points[1])*(pressure_points[1] - pressure_points[2]))**(1/beta_points[0])  
        T3 = T2 + ((1.0/alpha_points[1])*(pressure_points[3] - pressure_points[2]))**(1/beta_points[1]) 
        T0 = T1 - ((1.0/alpha_points[0])*(pressure_points[1] - log_P_min))**(1/beta_points[0])    
        
    # ...or for layer 1
    elif (pressure_points[4] < pressure_points[1]):  # Temperature parameter in layer 1
    
        # Use the temperature parameter to compute the boundary temperatures
        T0 = T_set - ((1.0/alpha_points[0])*(log_P_set_i - log_P_min))**(1/beta_points[0]) 
        T1 = T0 + ((1.0/alpha_points[0])*(pressure_points[1] - log_P_min))**(1/beta_points[0])    
        T2 = T1 - ((1.0/alpha_points[1])*(pressure_points[1] - pressure_points[2]))**(1/beta_points[1])   
        T3 = T2 + ((1.0/alpha_points[1])*(pressure_points[3] - pressure_points[2]))**(1/beta_points[1]) 
        

    temperatures[mask_1] = (log_P[mask_1] - pressure_points[0])**(1/beta_points[0]) / alpha_points[0] + T0
    temperatures[mask_2] = (log_P[mask_2] - pressure_points[2])**(1/beta_points[1]) / alpha_points[1] + T2
    temperatures[mask_3] = T3
    return temperatures

def cubic_spline_profile(press, temperature_points, gamma, nnodes = 0):
    """
    Compute a cubic spline profile for temperature based on pressure points.

    This function computes a cubic spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the cubic spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """

    cs = PchipInterpolator(np.linspace(np.log10(press[0]),
                             np.log10(press[-1]),
                             nnodes+2),
                 temperature_points)

    interpolated_temps = cs(np.log10(press))
    prior = temperature_curvature_prior(press,interpolated_temps,gamma)
    return interpolated_temps, prior

def linear_spline_profile(press, temperature_points, gamma, nnodes = 0):
    """
    Compute a linear spline profile for temperature based on pressure points.

    This function computes a linear spline profile for temperature using
    pressure and temperature data points, along with a curvature prior.

    Args:
        press (array-like): An array or list of pressure data points.
        temperature_points (array-like): An array or list of temperature data points.
        gamma (float): A parameter controlling the curvature of the spline.
        nnodes (int, optional): Number of nodes to use in the spline interpolation.
            Defaults to 0, which means automatic determination of nodes.

    Returns:
        tuple: A tuple containing two elements:
            - interpolated_temps (array-like): Interpolated temperature values
              based on the linear spline.
            - prior (array-like): Curvature prior values calculated for the spline.
    """
    interpolated_temps = np.interp(np.log10(press),
                    np.linspace(np.log10(press[0]),
                             np.log10(press[-1]),
                             int(nnodes)+2),
                    temperature_points)
    prior = temperature_curvature_prior(press,interpolated_temps,gamma)
    return interpolated_temps, prior

def temperature_curvature_prior(press,temps,gamma):
    """
    Compute a curvature prior for a temperature-pressure profile.

    This function calculates a curvature prior for a temperature-pressure profile,
    penalizing deviations from a smooth, low-curvature profile, based on Line 2015

    Args:
        press (array-like): An array or list of pressure data points.
        temps (array-like): An array or list of temperature data points.
        gamma (float): The curvature penalization factor.

    Returns:
        float: The curvature prior value.
    """
    weighted_temp_prior = -0.5*np.sum((temps[2:]-2*temps[1:-1]+temps[:-2])**2)/gamma
    weighted_temp_prior -= 0.5*np.log(2*np.pi*gamma)
    return weighted_temp_prior

def dTdP_temperature_profile(press,num_layer,layer_pt_slopes,T_bottom):
    """
    This function takes the temperature gradient at a set number of spline points and interpolates a temperature profile as a function of pressure.

    Args:
        press : array_like
            The pressure array.
        num_layer : int
            The number of layers.
        layer_pt_slopes : array_like
            The temperature gradient at the spline points.
        T_bottom : float
            The temperature at the bottom of the atmosphere.

    Returns:
        temperatures : array_like
            The temperature profile.
    """
    id_sub = np.where(press >= 1.0e-3)
    p_use_sub = press[id_sub]
    num_sub = len(p_use_sub)
    ## 1.3 pressures of layers
    layer_pressures = np.logspace(-3, 3, int(num_layer))
    ## 1.4 assemble the P-T slopes for these layers
    #for index in range(num_layer):
    #    layer_pt_slopes[index] = parameters['PTslope_%d'%(num_layer - index)].value
    ## 1.5 interpolate the P-T slopes to compute slopes for all layers
    interp_func = interp1d(np.log10(layer_pressures),
                           layer_pt_slopes,
                           'quadratic')
    pt_slopes_sub = interp_func( np.log10(p_use_sub) )
    ## 1.6 compute temperatures
    temperatures_sub = np.ones(num_sub) * np.nan
    temperatures_sub[-1] = T_bottom
    for index in range(1, num_sub):
        temperatures_sub[-1-index] = np.exp( np.log(temperatures_sub[-index]) - pt_slopes_sub[-index] *\
                                             (np.log(p_use_sub[-index]) - np.log(p_use_sub[-1-index])) )
    ## 1.7 isothermal in the remaining region, i.e., upper atmosphere
    temperatures = np.ones_like(press) * temperatures_sub[0]
    temperatures[id_sub] = np.copy(temperatures_sub)
    return temperatures

def radiosity_erg_cm2radiosity_erg_hz(radiosity_erg_cm, wavelength):
    """
    Convert a radiosity from erg.s-1.cm-2.sr-1/cm to erg.s-1.cm-2.sr-1/Hz at a given wavelength.
    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
        radiosity_erg_hz = radiosity_erg_cm * d[cm]/d[Hz]
        => radiosity_erg_hz = radiosity_erg_cm * wavelength**2 / c

    Args:
        radiosity_erg_cm: (erg.s-1.cm-2.sr-1/cm)
        wavelength: (cm)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    return radiosity_erg_cm * wavelength ** 2 / nc.c


def radiosity_erg_hz2radiosity_erg_cm(radiosity_erg_hz, frequency):
    """Convert a radiosity from erg.s-1.cm-2.sr-1/Hz to erg.s-1.cm-2.sr-1/cm at a given frequency.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: radiosity_erg_cm * d[cm] = radiosity_erg_hz * d[Hz]
        radiosity_erg_cm = radiosity_erg_hz * d[Hz]/d[cm]
        => radiosity_erg_cm = radiosity_erg_hz * frequency**2 / c

    Args:
        radiosity_erg_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    # TODO move to physics
    return radiosity_erg_hz * frequency ** 2 / nc.c


def radiosity2irradiance(spectral_radiosity, source_radius, target_distance):
    """Calculate the spectral irradiance of a spherical source on a target from its spectral radiosity.

    Args:
        spectral_radiosity: (M.L-1.T-3) spectral radiosity of the source
        source_radius: (L) radius of the spherical source
        target_distance: (L) distance from the source to the target

    Returns:
        The irradiance of the source on the target (M.L-1.T-3).
    """
    return spectral_radiosity * (source_radius / target_distance) ** 2
