import easychem as ec
import numpy as np


def get_easychem_abundances(pressures, temperatures, line_species, cloud_species,
                            metallicity, carbon_to_oxygen, carbon_quench_pressure, AMR=False):

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
                                  'H2O(c)', ]

    exo_atmos = ec.ExoAtmos(
        atoms=None,
        reactants=None,
        atomAbunds=None,
        thermofpath=None,
        feh=metallicity,
        co=carbon_to_oxygen
        )

    reactants = exo_atmos.reactants.copy()

    for condensate in default_remove_condensates:
        reactants = np.delete(reactants, np.argwhere(reactants == condensate))
    exo_atmos.updateReactants(reactants)

    exo_atmos.feh = metallicity
    exo_atmos._updateFEH()

    # Set atomic abundances from parameters
    for key, val in parameters.items():
        if key not in exo_atmos.atoms:
            continue
        itemindex = np.where(exo_atmos.atoms == key)
        exo_atmos.atomAbunds[itemindex] = 10 ** val.value

    exo_atmos.solve(pressures, temperatures)
    abundances = exo_atmos.result_mass()
    pressures_goal = pressures.reshape(-1)
    if carbon_quench_pressure is not None:
        if carbon_quench_pressure > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, carbon_quench_pressure),
                          int(len(pressures_goal)) - 1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    abundances["MMW"] = exo_atmos.mmw

    return abundances


def update_atom_abundances(atom_abundances_list, atom_name_list, atomic_mass_fractions):
    modif_abundances = atom_abundances_list.copy()
    for i_spec, name in enumerate(atom_name_list):
        if name in atomic_mass_fractions.keys():
            modif_abundances[i_spec] = 10 ** atomic_mass_fractions[name]
    return modif_abundances


def get_exoatmos_abundances(exo_atmos, pressures, temperatures, atomic_mass_fractions,
                            metallicity, carbon_quench_pressure, AMR=False):
    # TODO jit
    exo_atmos.Z = metallicity

    # Set atomic abundances from parameters
    atom_abundances = exo_atmos._atomAbunds.copy()
    update_abunds = update_atom_abundances(atom_abundances, exo_atmos.atoms, atomic_mass_fractions)
    exo_atmos.updateAtomAbunds(update_abunds)
    exo_atmos.solve(pressures, temperatures)
    abundances = exo_atmos.result_mass()

    pressures_goal = pressures.reshape(-1)
    if carbon_quench_pressure is not None:
        if carbon_quench_pressure > np.min(pressures_goal):
            q_index = min(np.searchsorted(pressures_goal, carbon_quench_pressure),
                          int(len(pressures_goal)) - 1)

            methane_abb = abundances['CH4']
            methane_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['CH4'][q_index]
            abundances['CH4'] = methane_abb

            co_abb = abundances['CO']
            co_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['CO'][q_index]
            abundances['CO'] = co_abb

            h2o_abb = abundances['H2O']
            h2o_abb[pressures_goal < carbon_quench_pressure] = \
                abundances['H2O'][q_index]
            abundances['H2O'] = h2o_abb

    abundances["MMW"] = exo_atmos.mmw

    return abundances


def setup_exoatmos():
    exo_atmos = ec.ExoAtmos()
    # In this version, we initialize the abundances to solar
    # and fit in terms of [C/H], [O/H], and [M/H] where M is everything besides C, O, H, and He.
    exo_atmos.Z = 0.0
    exo_atmos.co = 0.55

    reactants = exo_atmos.reactants.copy()
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
    exo_atmos.updateReactants(reactants)

    # add FeH
    # reactants = np.append(reactants,'FeH')
    exo_atmos.updateReactants(reactants)
    return exo_atmos
