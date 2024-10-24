import copy

import numpy as np

try:
    from petitRADTRANS.chemistry.prt_easychem import get_exoatmos_abundances
except ModuleNotFoundError:
    get_exoatmos_abundances = None

from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table
from petitRADTRANS.chemistry.utils import compute_mean_molar_masses,\
    fixed_length_amr,\
    linear_spline_profile,\
    cubic_spline_profile,\
    stepped_profile
from petitRADTRANS.chemistry import clouds as fc


def get_abundances(pressures, temperatures, line_species, cloud_species, parameters, amr=False):
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
            log_pquench. Additionally, the cloud parameters log_X_cb_Fe(c) and MgSiO3(c) must be present.
        amr : bool
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
    # TODO replace by SpectralModel function
    # Free Chemistry
    abundances_interp = {}
    mmw = None

    if 'use_easychem' in parameters.keys():
        if get_exoatmos_abundances is None:
            raise ModuleNotFoundError("module 'easychem' is not installed, "
                                      "remove key 'use_easychem from the parameters'")

        # Actual equilibrium chemistry
        # Can retrieve atomic abundances

        # Calling it abundances_interp to be consistent w poor mans
        abundances_interp = get_exoatmos_abundances(pressures,
                                                    temperatures,
                                                    parameters)
        mmw = abundances_interp['MMW']
    elif "C/O" in parameters.keys():
        # Check C/O AFTER easychem check -> need to use poor mans

        # Interpolated Equilibrium chemistry
        # Make the abundance profile
        pquench_c = None

        if 'log_pquench' in parameters.keys():
            pquench_c = 10 ** parameters['log_pquench'].value

        abundances_interp, mmw, _ = (
            pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
                parameters['C/O'].value * np.ones_like(pressures),
                parameters['Fe/H'].value * np.ones_like(pressures),
                temperatures,
                pressures,
                carbon_pressure_quench=pquench_c,
                full=True
            )
        )
    # Free chemistry abundances
    msum = np.zeros_like(pressures)

    if abundances_interp:
        for key, val in abundances_interp.items():
            msum += val

    # Free chemistry species
    for species in line_species:
        species_short_name = species.split(".R")[0]
        easy_chem_name = species.split('_')[0].split('-')[0].split(".")[0]
        # Vertically constant abundance
        if species_short_name in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abund = 10 ** parameters[species_short_name].value
            abundances_interp[easy_chem_name] = abund * np.ones_like(pressures)
            msum += abund * np.ones_like(pressures)

        # Stepped abundance profile
        if f"{species_short_name}_abundance_steps" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances[species_short_name] = stepped_profile(
                pressures,
                parameters[f"{species_short_name}_pressure_nodes"].value,
                parameters[f"{species_short_name}_abundance_steps"].value)
            msum += abundances[species_short_name]
        
        # Linear spline interpolation
        if f"{species_short_name}_linear_abundance_nodes" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances[species_short_name], prior = linear_spline_profile(
                pressures[small_index],
                parameters[f"{species_short_name}_pressure_nodes"].value,
                parameters[f"{species_short_name}_linear_abundance_nodes"].value,
                gamma = 0.04,
                nnodes = len(parameters[f"{species_short_name}_pressure_nodes"].value))
            msum += abundances[species_short_name]

        # Cubic spline interpolation
        if f"{species_short_name}_cubic_abundance_nodes" in parameters.keys():
            if easy_chem_name in abundances_interp.keys():
                msum -= abundances_interp[easy_chem_name]
            abundances[species_short_name],prior = cubic_spline_profile(
                pressures[small_index],
                parameters[f"{species_short_name}_pressure_nodes"].value,
                parameters[f"{species_short_name}_cubic_abundance_nodes"].value,
                gamma = 0.04,
                nnodes = len(parameters[f"{species_short_name}_pressure_nodes"].value))
            msum += abundances[species_short_name]
            
    # For free chemistry, need to fill with background gas (H2-He)
    # TODO use arbitrary background gas
    if "Fe/H" not in parameters.keys():
        # Check to make sure we're using free chemistry
        # Whatever's left is H2 and He
        if 'H2' in parameters.keys():
            abundances_interp['H2'] = 10 ** parameters['H2'].value

        else:
            abundances_interp['H2'] = 0.766 * (1.0 - msum)

        if 'He' in parameters.keys():
            abundances_interp['He'] = 10 ** parameters['He'].value

        else:
            abundances_interp['He'] = 0.234 * (1.0 - msum)

        # Imposing strict limit on msum to ensure H2 dominated composition
        if np.max(msum) > 1.0:
            print(f"Abundance sum > 1.0, msum={msum}")
            return None, None, None, None

        mmw = compute_mean_molar_masses(abundances_interp)

    # Prior check all input params
    clouds = {}
    p_bases = {}
    abundances = {}

    for cloud in cloud_species:
        cname = cloud.split("_")[0]
        if 'use_easychem' in parameters.keys():
            # AMR CANNOT BE USED WITH EASYCHEM RIGHT NOW
            clouds[cname] = abundances_interp[cname]
            continue

        if "eq_scaling_" + cname in parameters.keys():
            # equilibrium cloud abundance
            x_cloud = fc.return_cloud_mass_fraction(cloud, parameters['Fe/H'].value, parameters['C/O'].value)
            # Scaled by a constant factor
            clouds[cname] = 10 ** parameters['eq_scaling_' + cname].value * x_cloud
        else:
            # Free cloud abundance
            clouds[cname] = 10 ** parameters['log_X_cb_' + cname].value

        # Free cloud bases
        if 'log_Pbase_' + cname in parameters.keys():
            p_bases[cname] = 10 ** parameters['log_Pbase_' + cname].value
        elif 'Pbase_' + cname in parameters.keys():
            p_bases[cname] = parameters['Pbase_' + cname].value
        # Equilibrium locations
        elif 'Fe/H' in parameters.keys():
            p_bases[cname] = fc.simple_cdf(
                cname,
                pressures,
                temperatures,
                parameters['Fe/H'].value,
                parameters['C/O'].value,
                np.mean(mmw)
            )
        else:
            p_bases[cname] = fc.simple_cdf_free(
                cname,
                pressures,
                temperatures,
                10 ** parameters['log_X_cb_' + cname].value,
                mmw[0]
            )

    # Find high resolution pressure grid and indices
    if amr:
        press_use, small_index = fixed_length_amr(np.array(list(p_bases.values())),
                                                  pressures,
                                                  parameters['pressure_scaling'].value,
                                                  parameters['pressure_width'].value)
    else:
        small_index = np.linspace(0, pressures.shape[0] - 1, pressures.shape[0], dtype=int)

    fseds = {}

    if 'use_easychem' not in parameters.keys():
        cname = None

        for cloud in copy.copy(cloud_species):
            cname = cloud.split('_')[0]
            # Set up fseds per-cloud
            if 'fsed_' + cname in parameters.keys():
                fseds[cname] = parameters['fsed_' + cname].value
            else:
                fseds[cname] = parameters['fsed'].value

            abundances[cloud] = np.zeros_like(temperatures)
            abundances[cloud][pressures < p_bases[cname]] = \
                clouds[cname] * (
                        pressures[pressures <= p_bases[cname]] / p_bases[cname]
                ) ** fseds[cname]

            abundances[cloud] = abundances[cloud][small_index]

    for species in line_species:
        sname = species.split('_')[0].split('-')[0]
        sname = sname.split('.')[0]
        sname = sname.split('-')[0]
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
            abunds_change_rainout = copy.copy(abundances_interp[sname] / 2.)
            if 'Fe(c)' in p_bases.keys() and 'use_easychem' not in parameters.keys():
                index_ro = pressures < p_bases['Fe(c)']  # Must have iron cloud
                abunds_change_rainout[index_ro] = 0.
            abundances[species] = abunds_change_rainout[small_index]
        abundances[species] = abundances_interp[sname][small_index]
    abundances['H2'] = abundances_interp['H2'][small_index]
    abundances['He'] = abundances_interp['He'][small_index]

    return abundances, mmw, small_index, p_bases


