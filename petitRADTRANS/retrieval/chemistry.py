import os
os.environ["OMP_NUM_THREADS"] = "1"
import copy as cp
import numpy as np
from petitRADTRANS import poor_mans_nonequ_chem as pm
from petitRADTRANS.retrieval import cloud_cond as fc
from petitRADTRANS.retrieval.util import fixed_length_amr, calc_MMW

def get_abundances(pressures, temperatures, line_species, cloud_species, parameters, AMR = False):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <10% trace gas
    abundance by mass.

    Args:
        pressures : numpy.ndarray
            A log spaced pressure array. If AMR is on it should be the full high resolution grid.
        temperatures : numpy.ndarray
            A temperature array with the same shape as pressures
        line_species : List(str)
            A list of gas species that will contribute to the line-by-line opacity of the pRT atmosphere.
        cloud_species : List(str)
            A list of condensate species that will contribute to the cloud opacity of the pRT atmosphere.
        parameters : dict
            A dictionary of model parameters, in particular it must contain the names C/O, Fe/H and
            log_pquench. Additionally the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        AMR : bool
            Turn the adaptive mesh grid on or off. See fixed_length_amr for implementation.

    Returns:
        abundances : dict
            Mass fraction abundances of all atmospheric species
        MMW : numpy.ndarray
            Array of the mean molecular weights in each pressure bin
        small_index : numpy.ndarray
            The indices of the high resolution grid to use to define the adaptive grid.
        PBases : dict
            A dictionary of the cloud base pressures, either computed from equilibrium
            condensation or set by the user.
    """
    # Free Chemistry
    abundances_interp = {}
    if "C/O" in parameters.keys():
        # Equilibrium chemistry
        # Make the abundance profile
        pquench_C = None
        if 'log_pquench' in parameters.keys():
            pquench_C = 10**parameters['log_pquench'].value
        abundances_interp = pm.interpol_abundances(parameters['C/O'].value * np.ones_like(pressures), \
                                                parameters['Fe/H'].value * np.ones_like(pressures), \
                                                temperatures, \
                                                pressures,
                                                Pquench_carbon = pquench_C)
        MMW = abundances_interp['MMW']

    # Free chemistry abundances
    msum = 0.0
    for species in line_species:
        if species.split("_R_")[0] in parameters.keys():
            # Cannot mix free and equilibrium chemistry. Maybe something to add?
            abund = 10**parameters[species.split("_R_")[0]].value
            abundances_interp[species.split('_')[0]] = abund * np.ones_like(pressures)
            msum += abund
    if not "C/O" in parameters.keys():
        # Whatever's left is H2 and
        abundances_interp['H2'] = 0.766 * (1.0-msum) * np.ones_like(pressures)
        abundances_interp['He'] = 0.234 * (1.0-msum) * np.ones_like(pressures)

        # Imposing strict limit on msum to ensure H2 dominated composition
        if msum > 0.1:
            #print(f"Abundance sum > 1.0, msum={msum}")
            return None,None,None,None
        MMW = calc_MMW(abundances_interp)

    # Prior check all input params
    clouds = {}
    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if "eq_scaling_"+cname in parameters.keys():
            # equilibrium cloud abundance
            Xcloud= fc.return_cloud_mass_fraction(cloud,parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10**parameters['eq_scaling_'+cname].value*Xcloud
        else:
            # Free cloud abundance
            clouds[cname] = 10**parameters['log_X_cb_'+cloud.split("_")[0]].value

    # Get the cloud locations
    Pbases = {}
    for cloud in cloud_species:
        cname = cloud.split('_')[0]
        # Free cloud bases
        if 'Pbase_'+cname in parameters.keys():
            Pbases[cname] = 10**parameters['log_Pbase_'+cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            Pbases[cname] = fc.simple_cdf(cname, pressures, temperatures,
                                            parameters['Fe/H'].value, parameters['C/O'].value, np.mean(MMW))
        else:
            Pbases[cname] = fc.simple_cdf_free(cname,
                                               pressures,
                                               temperatures,
                                               10**parameters['log_X_cb_'+cname].value,
                                               MMW[0])
    # Find high resolution pressure grid and indices
    if AMR:
        press_use, small_index = fixed_length_amr(np.array(list(Pbases.values())),
                                                  pressures,
                                                  parameters['pressure_scaling'].value,
                                                  parameters['pressure_width'].value)
    else :
        small_index = np.linspace(0,pressures.shape[0]-1,pressures.shape[0], dtype = int)
    fseds = {}
    abundances = {}
    for cloud in cp.copy(cloud_species):
        cname = cloud.split('_')[0]
        # Set up fseds per-cloud
        if 'fsed_'+cname in parameters.keys():
            fseds[cname] = parameters['fsed_'+cname].value
        else:
            fseds[cname] = parameters['fsed'].value
        abundances[cname] = np.zeros_like(temperatures)
        abundances[cname][pressures < Pbases[cname]] = \
                        clouds[cname] *\
                        (pressures[pressures <= Pbases[cname]]/\
                        Pbases[cname])**fseds[cname]
        abundances[cname] = abundances[cname][small_index]

    for species in line_species:
        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = cp.copy(abundances_interp[species.split('_')[0]]/2.)
            index_ro = pressures < Pbases['Fe(c)'] # Must have iron cloud
            abunds_change_rainout[index_ro] = 0.
            abundances[species] = abunds_change_rainout[small_index]
        abundances[species] = abundances_interp[species.split('_')[0]][small_index]
    abundances['H2'] = abundances_interp['H2'][small_index]
    abundances['He'] = abundances_interp['He'][small_index]
    return abundances, MMW, small_index, Pbases
