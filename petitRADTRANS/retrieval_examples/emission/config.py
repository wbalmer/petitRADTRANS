from petitRADTRANS import nat_cst as nc
from scipy.interpolate import interp1d

################################################################################
################################################################################
### Hyperparameter setup, where to save things to / read things from
################################################################################
################################################################################

retrieval_name = 'JWST_emission_petitRADTRANSpaper'
absolute_path = '' # end with forward slash!
output_directory = '/home/evert/Documents/ThesisNotebooks/petitRet/'
planet_name = 'g395M'

observation_files = {}
observation_files['NIRISS SOSS'] = 'NIRISS_SOSS_flux.dat'
observation_files['NIRSpec G395M'] = 'NIRSpec_G395M_flux.dat'
observation_files['MIRI LRS'] = 'MIRI_LRS_flux.dat'

plotting = False
if plotting:
    import pylab as plt

stepsize = 1.75
live = 1000

cluster = False       # Submit to cluster
n_threads = 1         # Use threading (local = 1)
write_threshold = 200 # number of iterations after which diagnostics are updated

# Wavelength range of observations, fixed parameters that will not be retrieved
WLEN = [0.8, 14.0]
LOG_G =  2.58
R_pl =   1.84*nc.r_jup_mean
R_star = 1.81*nc.r_sun
# Get host star spectrum to calculate F_pl / F_star later.
T_star = 6295.
x = nc.get_PHOENIX_spec(T_star)
fstar = interp1d(x[:,0], x[:,1])

parameters = ['log_delta','log_gamma','t_int','t_equ','log_p_trans','alpha','log_g','log_P0','CO_all_iso','H2O','CH4','NH3','CO2','H2S','Na','K']
