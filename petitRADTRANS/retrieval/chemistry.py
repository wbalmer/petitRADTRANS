import os
os.environ["OMP_NUM_THREADS"] = "1"
import copy as cp
import numpy as np
try:
    import easychem as ec
except ModuleNotFoundError:
    pass
from petitRADTRANS import poor_mans_nonequ_chem as pm
from petitRADTRANS.retrieval import cloud_cond as fc
from petitRADTRANS.retrieval.util import fixed_length_amr, calc_MMW

def get_abundances(pressures, temperatures, line_species, cloud_species, parameters, AMR = False):
    """
    This function takes in the C/O ratio, metallicity, and quench pressures and uses them
    to compute the gas phase and equilibrium condensate abundances from an interpolated table.
    Alternatively, a "free chemistry" approach can be used to set abundances.
    This function assumes a hydrogen-helium dominated atmosphere, and enforces <100% trace gas
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
    if 'use_easychem' in parameters.keys():
        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent w poor mans
        """abundances_interp = get_easychem_abundances(pressures,
                                                    temperatures,
                                                    line_species,
                                                    cloud_species,
                                                    parameters)"""
        abundances_interp = get_exoatmos_abundances(pressures,
                                                    temperatures,
                                                    parameters)
        MMW = abundances_interp['MMW']
    elif "C/O" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor mans

        # Interpolated Equilibrium chemistry
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
    if abundances_interp:
        for key, val in abundances_interp.items():
            msum += np.max(val)

    # Free chemistry species
    for species in line_species:
        if species.split("_R_")[0] in parameters.keys():
            if species.split('_')[0] in abundances_interp.keys():
                msum -= np.max(abundances_interp[species.split('_')[0]])
            abund = 10**parameters[species.split("_R_")[0]].value
            abundances_interp[species.split('_')[0]] = abund * np.ones_like(pressures)
            msum += abund

    # For free chemistry, need to fill with background gas (H2-He)
    # TODO use arbitrary background gas
    if not "Fe/H" in parameters.keys():
        # Check to make sure we're using free chemistry
        # Whatever's left is H2 and He
        if 'H2' in parameters.keys():
            abundances_interp['H2'] = 10**parameters['H2'].value * np.ones_like(pressures)
            msum += 10**parameters['H2'].value
        else:
            abundances_interp['H2'] = 0.766 * (1.0-msum) * np.ones_like(pressures)
            msum += abundances_interp['H2'][0]
        if 'He' in  parameters.keys():
            abundances_interp['He'] = 10**parameters['He'].value * np.ones_like(pressures)
            msum += 10**parameters['He'].value
        else:
            abundances_interp['He'] = 0.234 * (1.0-msum) * np.ones_like(pressures)
            msum += abundances_interp['He'][0]

        # Imposing strict limit on msum to ensure H2 dominated composition
        if msum > 1.0:
            print(f"Abundance sum > 1.0, msum={msum}")
            return None,None,None,None
        MMW = calc_MMW(abundances_interp)

    # Prior check all input params
    clouds = {}
    Pbases = {}
    abundances = {}
    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if 'use_easychem' in parameters.keys():
            # AMR CANNOT BE USED WITH EASYCHEM RIGHT NOW
            abundances[cname] = abundances_interp[cname]
            continue

        if "eq_scaling_"+cname in parameters.keys():
            # equilibrium cloud abundance
            Xcloud= fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10**parameters['eq_scaling_'+cname].value*Xcloud
        else:
            # Free cloud abundance
            clouds[cname] = 10**parameters['log_X_cb_'+cname].value

        # Free cloud bases
        if 'log_Pbase_'+cname in parameters.keys():
            Pbases[cname] = 10**parameters['log_Pbase_'+cname].value
        elif 'Pbase_'+cname in parameters.keys():
            Pbases[cname] = parameters['Pbase_'+cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            Pbases[cname] = fc.simple_cdf(cname, 
                                          pressures, 
                                          temperatures,
                                          parameters['Fe/H'].value, 
                                          parameters['C/O'].value, 
                                          np.mean(MMW))
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
    if not 'use_easychem' in parameters.keys():
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
        sname = species.split('_')[0]

        # Depending on easychem vs interpolated and different versions of pRT
        # C2H2 is named differently.
        if sname == "C2H2":
            # might be',acetylene'
            not_found = True
            for key in abundances_interp.keys():
                if sname in key:
                    sname = key
                    not_found = False
                    break
            if not_found:
                continue
        if 'FeH' in species:
            # Magic factor for FeH opacity - off by factor of 2
            abunds_change_rainout = cp.copy(abundances_interp[sname]/2.)
            if 'Fe(c)' in Pbases.keys() and not 'use_easychem' in parameters.keys():
                index_ro = pressures < Pbases['Fe(c)'] # Must have iron cloud
                abunds_change_rainout[index_ro] = 0.
            abundances[species] = abunds_change_rainout[small_index]
        abundances[species] = abundances_interp[sname][small_index]
    abundances['H2'] = abundances_interp['H2'][small_index]
    abundances['He'] = abundances_interp['He'][small_index]
    return abundances, MMW, small_index, Pbases

def get_easychem_abundances(pressures, temperatures, line_species, cloud_species, parameters, AMR = False):

    metallicity = None
    carbon_to_oxygen = None

    if "Fe/H" in parameters.keys():
        metallicity = parameters["Fe/H"].value
    if "C/O" in parameters.keys():
        carbon_to_oxygen = parameters["C/O"].value

    default_remove_condensates = ['Mg2SiO4(c)',
                                 'Mg2SiO4(L)',
                                 'MgAl2O4(c)',
                                 'FeO(c)',
                                 'Fe2SiO4(c)',
                                 'TiO2(c)',
                                 'TiO2(L)',
                                 'H3PO4(c)',
                                 'H3PO4(L)',
                                 'H2O(L)',
                                 'H2O(c)',]

    exo = ec.ExoAtmos(atoms=None,
                  reactants=None,
                  atomAbunds=None,
                  thermofpath=None,
                  feh=metallicity,
                  co=carbon_to_oxygen)

    reactants = exo.reactants.copy()

    for condensate in default_remove_condensates:
        reactants = np.delete(reactants, np.argwhere(reactants == condensate))
    exo.updateReactants(reactants)

    exo.feh = metallicity
    exo._updateFEH()

    # Set atomic abundances from parameters
    for key,val in parameters.items():
        if key not in exo.atoms:
            continue
        itemindex = np.where(exo.atoms == key)
        exo.atomAbunds[itemindex] = 10**val.value


    exo.solve(pressures, temperatures)
    abundances = exo.result_mass()
    pressures_goal = pressures.reshape(-1)
    if 'log_pquench' in parameters.keys():
        Pquench_carbon = 10**parameters['log_pquench'].value
        if Pquench_carbon > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, Pquench_carbon),
                            int(len(pressures_goal))-1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < Pquench_carbon] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < Pquench_carbon] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < Pquench_carbon] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb
    abundances["MMW"] = exo.mmw
    return abundances

def update_atom_abundances(atom_abundances_list,atom_name_list,parameters):
    modif_abundances = atom_abundances_list.copy()
    for i_spec, name in enumerate(atom_name_list):
            if name in parameters.keys():
                modif_abundances[i_spec] = 10**parameters[name].value
    return modif_abundances

def get_exoatmos_abundances(pressures, temperatures, parameters, AMR = False):
    exo = parameters['exoatmos'].value
    metallicity = None

    if "Fe/H" in parameters.keys():
        metallicity = parameters["Fe/H"].value
        exo.Z = metallicity
    # Set atomic abundances from parameters
    atom_abundances = exo._atomAbunds.copy()
    update_abunds = update_atom_abundances(atom_abundances, exo.atoms, parameters)
    exo.updateAtomAbunds(update_abunds)
    exo.solve(pressures, temperatures)
    abundances = exo.result_mass()

    pressures_goal = pressures.reshape(-1)
    if 'log_pquench' in parameters.keys():
        Pquench_carbon = 10**parameters['log_pquench'].value
        if Pquench_carbon > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, Pquench_carbon),
                            int(len(pressures_goal))-1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < Pquench_carbon] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < Pquench_carbon] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < Pquench_carbon] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb
    abundances["MMW"] = exo.mmw
    return abundances

def setup_exoatmos():
    exo = ec.ExoAtmos()
    # In this version, we initialize the abundances to solar
    # and fit in terms of [C/H], [O/H], and [M/H] where M is everything besides C, O, H, and He.
    exo.Z = 0.0
    exo.co = 0.55

    reactants = exo.reactants.copy()
    default_remove_condensates = ['Mg2SiO4(c)',
                                 'TiO(c)',
                                 'TiO(L)',
                                 'MgAl2O4(c)',
                                 'FeO(c)',
                                 'Fe2SiO4(c)',
                                 'TiO2(c)',
                                 'TiO2(L)',
                                 'H3PO4(c)',
                                 'H3PO4(L)']
    
    for condensate in default_remove_condensates:
        reactants = np.delete(reactants, np.argwhere(reactants == condensate))
    exo.updateReactants(reactants)

    # add FeH
    #reactants = np.append(reactants,'FeH')
    exo.updateReactants(reactants)
    return exo