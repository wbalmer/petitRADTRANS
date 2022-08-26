import numpy as np
import spectres as spectres
class Data:
    def __init__(self, observation_files,
                 name_in=None,
                 data_path=None,
                 output_path=None):
        self.observation_files = observation_files
        
        # Read in data, convert all to cgs!
        
        self.data_wlen = {}
        self.data_flux_nu = {}
        self.data_flux_nu_error = {}
        self.data_wlen_bins = {}
        
        for name in observation_files.keys():
            
            self.dat_obs = np.genfromtxt(observation_files[name])
            self.data_wlen[name] = dat_obs[:,0]*1e-4
            self.data_flux_nu[name] = dat_obs[:,1]
            self.data_flux_nu_error[name] = dat_obs[:,2]
            
            self.data_wlen_bins[name] = np.zeros_like(data_wlen[name])
            self.data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
            self.data_wlen_bins[name][-1] = data_wlen_bins[name][-2]

        def getWlen():
            return self.data_wlen
        
        def getWnum():
            wnum = {}
            for name in self.observation_files.keys():
                wnum[name] = 1. / self.data_wlen[name]
            return wnum
        
        def rebinData(new_wlen_bins):
            for name in self.observation_files.keys():
                self.data_flux_new[name], self.data_flux_nu_error[name] = \
                    spectres(new_wlen_bins,self.data_wlen[name],\
                             self.data_flux_nu[name],self.data_flux_new_error[name])
                self.data_wlen[name] = new_wlen_bins
