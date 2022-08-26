import numpy as np
from config import *
from util import *
import sys
import os
import csv

from matplotlib import pyplot as plt
from petitRADTRANS import Radtrans
import master_retrieval_model as rm
from petitRADTRANS import nat_cst as nc
import rebin_give_width as rgw
from scipy.interpolate import interp1d

# Declare diagnostics
function_calls = 0
computed_spectra = 0
NaN_spectra = 0
delta_wt = write_threshold

class Priors:
    log_priors = {}
    log_priors['log_delta']      = lambda x: -((x-(-5.5))/2.5)**2./2.                           
    log_priors['log_gamma']      = lambda x: -((x-(-0.0))/2.)**2./2. 
    log_priors['t_int']          = lambda x: a_b_range(x, 0., 1500.)
    log_priors['t_equ']          = lambda x: a_b_range(x, 0., 4000.)
    log_priors['log_p_trans']    = lambda x: -((x-(-3))/3.)**2./2.
    log_priors['alpha']          = lambda x: -((x-0.25)/0.4)**2./2.
    log_priors['log_g']          = lambda x: a_b_range(x, 2.0, 3.7) 
    log_priors['log_P0']         = lambda x: a_b_range(x, -4, 2.)
    
    # Priors for log mass fractions
    log_priors['CO_all_iso']     = lambda x: a_b_range(x, -10., 0.)
    log_priors['H2O']            = lambda x: a_b_range(x, -10., 0.)
    log_priors['CH4']            = lambda x: a_b_range(x, -10., 0.)
    log_priors['NH3']            = lambda x: a_b_range(x, -10., 0.)
    log_priors['CO2']            = lambda x: a_b_range(x, -10., 0.)
    log_priors['H2S']            = lambda x: a_b_range(x, -10., 0.)
    log_priors['Na']             = lambda x: a_b_range(x, -10., 0.)
    log_priors['K']              = lambda x: a_b_range(x, -10., 0.)

    p0 = np.array([np.random.normal(loc = -5.5, scale = 2.5, size=1)[0], \
                   np.random.normal(loc = 0., scale = 2., size=1)[0], \
                   0.+1500.*np.random.uniform(size=1)[0], \
                   0.+4000.*np.random.uniform(size=1)[0], \
                   np.random.normal(loc = -3., scale = 3., size=1)[0], \
                   np.random.normal(loc = -0.25, scale = 0.4, size=1)[0], \
                   LOG_G,
                   -2., \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0], \
                   -10.+10.*np.random.uniform(size=1)[0]])
    
    def __init__(data_obj,
                 rt_obj,
                 name_in = None,
                 data_path = None,
                 output_path= None,
                 plotting = False,
                 diagnostics = False):      
        file_object = open(data_path + 'diag_' + \
                           name_in+ '.dat', 'w').close()
        self.data = data_obj
        self.rt_obj = rt_obj
        self.plotting = plotting
        self.diagnostics = diagnostics
        self.ndim = len(log_priors)
        return

    def prior(self,cube,ndim,nparam):
        for ind in range(ndim):
            cube[ind] = cube[ind]*p0[ind]

    def loglike(self,cube,ndim,nparam):
        return self.compute_log_likelihood(cube,ndim)
    
    def compute_log_likelihood(self,cube,ndim):
        params = []
        for i in range(ndim):
            params.append(cube[i])
        params = np.array(params)        
        log_delta, log_gamma, t_int, t_equ, log_p_trans, alpha, \
        log_g, log_P0 = params[:-8]

        # Make dictionary for modified Guillot parameters
        temp_params = {}
        temp_params['log_delta'] = log_delta
        temp_params['log_gamma'] = log_gamma
        temp_params['t_int'] = t_int
        temp_params['t_equ'] = t_equ
        temp_params['log_p_trans'] = log_p_trans
        temp_params['alpha'] = alpha
        
        # Make dictionary for log 'metal' abundances
        ab_metals = {}
        ab_metals['CO_all_iso']     = params[-8:][0]
        ab_metals['H2O']            = params[-8:][1]
        ab_metals['CH4']            = params[-8:][2]
        ab_metals['NH3']            = params[-8:][3]
        ab_metals['CO2']            = params[-8:][4]
        ab_metals['H2S']            = params[-8:][5]
        ab_metals['Na']             = params[-8:][6]
        ab_metals['K']              = params[-8:][7]

        if self.diagnostics:
            global function_calls
            global computed_spectra
            global NaN_spectra
            global write_threshold
        
        function_calls += 1
        
        # Prior calculation of all input parameters
        log_prior = 0.
        
        # Alpha should not be smaller than -1, this
        # would lead to negative temperatures!
        if alpha < -1:
            return -np.inf
        
        for key in temp_params.keys():
            log_prior += log_priors[key](temp_params[key])
            
            log_prior += log_priors['log_g'](log_g)
            log_prior += log_priors['log_P0'](log_P0)
            
        # Metal abundances: check that their
        # summed mass fraction is below 1.
        metal_sum = 0.
        for name in ab_metals.keys():
            log_prior += log_priors[name](ab_metals[name])
            metal_sum += 1e1**ab_metals[name]
                
        if metal_sum > 1.:
            log_prior += -np.inf
                    
        # Return -inf if parameters fall outside prior distribution
        if (log_prior == -np.inf):
            return -np.inf
    
        # Calculate the log-likelihood
        log_likelihood = 0.
        
        # Calculate the forward model, this
        # returns the wavelengths in cm and the flux F_nu
        # in erg/cm^2/s/Hz
        wlen, flux_nu = \
        rm.retrieval_model_plain(self.rt_obj, temp_params, log_g, \
                                 log_P0, R_pl, ab_metals)

        # Just to make sure that a long chain does not die
        # unexpectedly:
        # Return -inf if forward model returns NaN values
        if np.sum(np.isnan(flux_nu)) > 0:
            print("NaN spectrum encountered")
            if self.diagnostics:
                NaN_spectra += 1
            return -np.inf

        # Convert to observation for emission case
        flux_star = fstar(wlen)
        flux_sq   = flux_nu/flux_star*(R_pl/R_star)**2 
        
        # Calculate log-likelihood
        for instrument in self.data.data_wlen.keys():

        # Rebin model to observation
        flux_rebinned = rgw.rebin_give_width(wlen, flux_sq, \
                        self.data.data_wlen[instrument], self.data.data_wlen_bins[instrument])

        if self.plotting:
            plt.errorbar(self.data.data_wlen[instrument], \
                         self.data.data_flux_nu[instrument], \
                         self.data.data_flux_nu_error[instrument], \
                         fmt = 'o', \
                         zorder = -20, \
                         color = 'red')
            
            plt.plot(self.data.data_wlen[instrument], \
                     flux_rebinned, \
                     's', \
                     zorder = -20, \
                     color = 'blue')
            
        # Calculate log-likelihood
        log_likelihood += -np.sum(((flux_rebinned - self.data.data_flux_nu[instrument])/ \
                                   self.data.data_flux_nu_error[instrument])**2.)/2.

        if self.plotting:
            plt.plot(wlen, flux_sq, color = 'black')
            plt.xscale('log')
            plt.show()
            
        if self.diagnostics:
            computed_spectra += 1
            if (function_calls >= write_threshold):
                
                write_threshold += delta_wt
                hours = (time.time() - start_time)/3600.0
                info_list = [function_calls, computed_spectra, NaN_spectra, \
                             log_prior + log_likelihood, hours] 
                
                file_object = open(absolute_path + 'diag_' + retrieval_name + '.dat', 'a')

        for i in np.arange(len(info_list)):
            if (i == len(info_list) - 1):
                file_object.write(str(info_list[i]).ljust(15) + "\n")
            else:
                file_object.write(str(info_list[i]).ljust(15) + " ")
            file_object.close()
        if self.diagnostics:
            print(log_prior + log_likelihood)
            print("--> ", function_calls, " --> ", computed_spectra)
            
        if np.isnan(log_prior + log_likelihood):
            return -np.inf
        else:
            return log_prior + log_likelihood

        
