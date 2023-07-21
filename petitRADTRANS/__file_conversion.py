"""Stores functions that convert files from a format to another.

The functions in this module are stored for the sake of keeping trace of changes made to files. They are intended to be
used only once.
"""
import copy
import os

import h5py
import numpy as np

import datetime
import warnings
from petitRADTRANS.molar_mass import getMM
import petitRADTRANS
import petitRADTRANS.nat_cst as nc
from petitRADTRANS.config import petitradtrans_config
from petitRADTRANS.fort_input import fort_input as fi
from petitRADTRANS.radtrans import Radtrans


def continuum_clouds_opacities_dat2h5(path_input_data=petitradtrans_config['Paths']['prt_input_data_path'],
                                      rewrite=False, output_directory=None):
    """Using ExoMol units for HDF5 files."""
    # Initialize infos
    molliere2019_doi = '10.1051/0004-6361/201935470'

    doi_dict = {
        'Al2O3(c)_cm': molliere2019_doi,
        'Al2O3(c)_cd': molliere2019_doi,
        'Fe(c)_am': molliere2019_doi,
        'Fe(c)_ad': molliere2019_doi,
        'Fe(c)_cm': molliere2019_doi,
        'Fe(c)_cd': molliere2019_doi,
        'H2O(c)_cm': molliere2019_doi,
        'H2O(c)_cd': molliere2019_doi,
        'KCL(c)_cm': molliere2019_doi,
        'KCL(c)_cd': molliere2019_doi,
        'Mg05Fe05SiO3(c)_am': molliere2019_doi,
        'Mg05Fe05SiO3(c)_ad': molliere2019_doi,
        'Mg2SiO4(c)_am': molliere2019_doi,
        'Mg2SiO4(c)_ad': molliere2019_doi,
        'Mg2SiO4(c)_cm': molliere2019_doi,
        'Mg2SiO4(c)_cd': molliere2019_doi,
        'MgAl2O4(c)_cm': molliere2019_doi,
        'MgAl2O4(c)_cd': molliere2019_doi,
        'MgFeSiO4(c)_am': molliere2019_doi,
        'MgFeSiO4(c)_ad': molliere2019_doi,
        'MgSiO3(c)_am': molliere2019_doi,
        'MgSiO3(c)_ad': molliere2019_doi,
        'MgSiO3(c)_cm': molliere2019_doi,
        'MgSiO3(c)_cd': molliere2019_doi,
        'Na2S(c)_cm': molliere2019_doi,
        'Na2S(c)_cd': molliere2019_doi,
        'SiC(c)_cm': molliere2019_doi,
        'SiC(c)_cd': molliere2019_doi
    }

    description_dict = {
        'Al2O3(c)_cm': '',
        'Al2O3(c)_cd': '',
        'Fe(c)_am': '',
        'Fe(c)_ad': '',
        'Fe(c)_cm': '',
        'Fe(c)_cd': '',
        'H2O(c)_cm': '',
        'H2O(c)_cd': '',
        'KCL(c)_cm': '',
        'KCL(c)_cd': '',
        'Mg05Fe05SiO3(c)_am': '',
        'Mg05Fe05SiO3(c)_ad': '',
        'Mg2SiO4(c)_am': '',
        'Mg2SiO4(c)_ad': '',
        'Mg2SiO4(c)_cm': '',
        'Mg2SiO4(c)_cd': '',
        'MgAl2O4(c)_cm': '',
        'MgAl2O4(c)_cd': '',
        'MgFeSiO4(c)_am': '',
        'MgFeSiO4(c)_ad': '',
        'MgSiO3(c)_am': '',
        'MgSiO3(c)_ad': '',
        'MgSiO3(c)_cm': '',
        'MgSiO3(c)_cd': '',
        'Na2S(c)_cm': '',
        'Na2S(c)_cd': '',
        'SiC(c)_cm': '',
        'SiC(c)_cd': ''
    }

    for key in description_dict:
        particle_mode = key.rsplit('_', 1)[1]
        particle_mode_description = ''

        if particle_mode[0] == 'c':
            particle_mode_description += 'Crystalline, '
        elif particle_mode[0] == 'a':
            particle_mode_description += 'Amorphous, '
        else:
            raise ValueError(f"invalid particle name '{key}': "
                             f"particle internal structure must be either crystalline ('c') or amorphous ('a'), "
                             f"but was '{particle_mode[0]}'")

        if particle_mode[1] == 'm':
            particle_mode_description += 'Mie scattering (spherical)'
        elif particle_mode[1] == 'd':
            particle_mode_description += 'DHS (irregular shape)'
        else:
            raise ValueError(f"invalid particle name '{key}': "
                             f"particle shape must be either calculated using Mie scattering ('m') "
                             f"or using the distribution of hollow spheres method ('d'), "
                             f"but was '{particle_mode[1]}'")

        description_dict[key] = particle_mode_description

    molmass_dict = {
        'Al2O3(c)_cm': getMM('Al2O3'),
        'Al2O3(c)_cd': getMM('Al2O3'),
        'Fe(c)_am': getMM('Fe'),
        'Fe(c)_ad': getMM('Fe'),
        'Fe(c)_cm': getMM('Fe'),
        'Fe(c)_cd': getMM('Fe'),
        'H2O(c)_cm': getMM('H2O'),
        'H2O(c)_cd': getMM('H2O'),
        'KCL(c)_cm': getMM('H2O'),
        'KCL(c)_cd': getMM('H2O'),
        'Mg05Fe05SiO3(c)_am': 0.5 * getMM('Mg') + 0.5 * getMM('Fe') + getMM('SiO3'),
        'Mg05Fe05SiO3(c)_ad': 0.5 * getMM('Mg') + 0.5 * getMM('Fe') + getMM('SiO3'),
        'Mg2SiO4(c)_am': getMM('Mg2SiO4'),
        'Mg2SiO4(c)_ad': getMM('Mg2SiO4'),
        'Mg2SiO4(c)_cm': getMM('Mg2SiO4'),
        'Mg2SiO4(c)_cd': getMM('Mg2SiO4'),
        'MgAl2O4(c)_cm': getMM('MgAl2O4'),
        'MgAl2O4(c)_cd': getMM('MgAl2O4'),
        'MgFeSiO4(c)_am': getMM('MgFeSiO4'),
        'MgFeSiO4(c)_ad': getMM('MgFeSiO4'),
        'MgSiO3(c)_am': getMM('MgSiO3'),
        'MgSiO3(c)_ad': getMM('MgSiO3'),
        'MgSiO3(c)_cm': getMM('MgSiO3'),
        'MgSiO3(c)_cd': getMM('MgSiO3'),
        'Na2S(c)_cm': getMM('Na2S'),
        'Na2S(c)_cd': getMM('Na2S'),
        'SiC(c)_cm': getMM('SiC'),
        'SiC(c)_cd': getMM('SiC')
    }

    # Get only existing directories
    input_directory = os.path.join(path_input_data, 'opacities', 'continuum', 'clouds')
    bad_keys = []

    for key in doi_dict:
        species = key.split('(', 1)[0]
        species_dir = os.path.join(input_directory, species + '_c')

        if not os.path.isdir(species_dir):
            print(f"data for cloud '{key}' not found (path '{species_dir}' does not exist), skipping...")
            bad_keys.append(key)
            continue

        particle_mode = key.rsplit('_', 1)[1]

        particle_mode_dir = None

        if particle_mode[0] == 'c':
            particle_mode_dir = os.path.join(species_dir, 'crystalline')
        elif particle_mode[0] == 'a':
            particle_mode_dir = os.path.join(species_dir, 'amorphous')

        if not os.path.isdir(particle_mode_dir):
            print(f"data for cloud '{key}' not found (path '{particle_mode_dir}' does not exist), skipping...")
            del doi_dict[key]
            continue

        if particle_mode[1] == 'm':
            particle_mode_dir = os.path.join(particle_mode_dir, 'mie')
        elif particle_mode[1] == 'd':
            particle_mode_dir = os.path.join(particle_mode_dir, 'DHS')

        if not os.path.isdir(particle_mode_dir):
            print(f"data for cloud '{key}' not found (path '{particle_mode_dir}' does not exist), skipping...")
            del doi_dict[key]
            continue

    for key in bad_keys:
        del doi_dict[key]

    # Prepare single strings delimited by ':' which are then put into Fortran routines
    cloud_species_modes = []
    cloud_species = []

    for key in doi_dict:
        cloud_species_ = key.rsplit('_', 1)
        cloud_species_modes.append(cloud_species_[1])
        cloud_species.append(cloud_species_[0])

    tot_str_names = ''

    for cloud_species_ in cloud_species:
        tot_str_names = tot_str_names + cloud_species_ + ':'

    tot_str_modes = ''

    for cloud_species_mode in cloud_species_modes:
        tot_str_modes = tot_str_modes + cloud_species_mode + ':'

    n_cloud_wavelength_bins = int(len(np.genfromtxt(
        os.path.join(
            path_input_data, 'opacities', 'continuum', 'clouds', 'MgSiO3_c', 'amorphous', 'mie', 'opa_0001.dat'
        )
    )[:, 0]))

    # Load .dat files
    print("Loading dat files...")
    cloud_particles_densities, cloud_absorption_opacities, cloud_scattering_opacities, \
        cloud_asymmetry_parameter, cloud_wavelengths, cloud_particles_radius_bins, cloud_particles_radii \
        = fi.read_in_cloud_opacities(
            path_input_data, tot_str_names, tot_str_modes, len(doi_dict), n_cloud_wavelength_bins
        )

    wavenumbers = 1 / cloud_wavelengths[::-1]  # cm to cm-1

    cloud_modes = tot_str_modes.split(':')

    # Save each clouds data into HDF5 file
    if output_directory is None:
        output_directory_ref = input_directory
    else:
        output_directory_ref = copy.deepcopy(output_directory)

    for i, key in enumerate(doi_dict):
        # Check if current key is in all information dicts
        not_in_dict = False

        if key not in description_dict:
            warnings.warn(f"cloud '{key}' was not in contributor dict; "
                          f"add key '{key}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if key not in molmass_dict:
            warnings.warn(f"cloud '{key}' was not in molar mass dict; "
                          f"add key '{key}' to the script molmass_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            print(" Skipping due to missing species in supplementary info dict...")
            continue

        # Get HDF5 file name
        output_directory = output_directory_ref

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        hdf5_opacity_file = os.path.join(output_directory, key + '.cotable.petitRADTRANS.h5')

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            print(f"File '{hdf5_opacity_file}' already exists, skipping conversion...")
            continue

        # Write HDF5 file
        print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

        with h5py.File(hdf5_opacity_file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                data=doi_dict[key]
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['additional_description'] = description_dict[key]

            dataset = fh5.create_dataset(
                name='Date_ID',
                data=f'petitRADTRANS-v{petitRADTRANS.__version__}_{datetime.datetime.utcnow().isoformat()}'
            )
            dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                         'at which the table has been created, ' \
                                         'along with the version of petitRADTRANS'

            dataset = fh5.create_dataset(
                name='wavenumbers',
                data=wavenumbers
            )
            dataset.attrs['long_name'] = 'Opacities wavenumbers'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='absorption_opacities',
                data=cloud_absorption_opacities[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the absorption opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='scattering_opacities',
                data=cloud_scattering_opacities[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the scattering opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='asymmetry_parameters',
                data=cloud_asymmetry_parameter[:, ::-1, i]
            )
            dataset.attrs['long_name'] = 'Table of the asymmetry parameters with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'None'

            dataset = fh5.create_dataset(
                name='mol_mass',
                data=molmass_dict[key]
            )
            dataset.attrs['long_name'] = 'Mass of the species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='particles_density',
                data=cloud_particles_densities[i]
            )
            dataset.attrs['long_name'] = 'Average density of the cloud particles'
            dataset.attrs['units'] = 'g.cm^-3'

            dataset = fh5.create_dataset(
                name='mol_name',
                data=cloud_species[i]
            )
            dataset.attrs['long_name'] = 'Name of the species described, "(c)" indicates that it has condensed'

            dataset = fh5.create_dataset(
                name='particles_radii',
                data=cloud_particles_radii
            )
            dataset.attrs['long_name'] = 'Particles average radius grid'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='particle_radius_bins',
                data=cloud_particles_radius_bins
            )
            dataset.attrs['long_name'] = 'Particles average radius grid bins'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([cloud_wavelengths.min(), cloud_wavelengths.max()]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumbers.min(), wavenumbers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'

        print("Done.")

    print("Conversions successful.")


def line_by_line_opacities_dat2h5(path_input_data=petitradtrans_config['Paths']['prt_input_data_path'],
                                  rewrite=False, output_directory=None):
    """Using ExoMol units for HDF5 files."""
    # Initialize infos
    kurucz_website = 'http://kurucz.harvard.edu/'
    molliere2019_doi = '10.1051/0004-6361/201935470'
    # hitran_doi = '10.1016/j.jqsrt.2013.07.002'
    # hitemp_doi = '10.1016/j.jqsrt.2010.05.001'

    molaverdikhani_email = 'karan.molaverdikhani@colorado.edu'

    kurucz_description = 'gamma_nat + V dW, sigma_therm'

    doi_dict = {
        'Al': kurucz_website,
        'B': kurucz_website,
        'Be': kurucz_website,
        'C2H2_main_iso': molliere2019_doi,
        'Ca': kurucz_website,
        'Ca+': kurucz_website,
        'CaH': 'unknown',  # TODO not in the docs
        'CH4_212': molliere2019_doi,  # TODO not in the referenced paper
        'CH4_hargreaves_main_iso': '10.3847/1538-4365/ab7a1a',
        'CH4_Hargreaves_main_iso': '10.3847/1538-4365/ab7a1a',
        'CH4_main_iso': 'unknown',  # TODO not in the docs (outdated)
        'CO2_main_iso': molliere2019_doi,
        'CO_27': molliere2019_doi,  # TODO not in the referenced paper
        'CO_28': molliere2019_doi,  # TODO not in the referenced paper
        'CO_36': molliere2019_doi,  # TODO not in the referenced paper
        'CO_37': molliere2019_doi,  # TODO not in the referenced paper
        'CO_38': molliere2019_doi,  # TODO not in the referenced paper
        'CO_all_iso': molliere2019_doi,
        'CO_main_iso': molliere2019_doi,  # TODO not clear in the referenced paper nor in the docs
        'Cr': kurucz_website,
        'Fe': kurucz_website,
        'Fe+': kurucz_website,
        'FeH_main_iso': molliere2019_doi,  # TODO not in the docs
        'H2_12': molliere2019_doi,  # TODO not in the referenced paper
        'H2_main_iso': molliere2019_doi,
        'H2O_162': molliere2019_doi,  # TODO not in the referenced paper
        'H2O_171': molliere2019_doi,  # TODO not in the referenced paper
        'H2O_172': molliere2019_doi,  # TODO not in the referenced paper
        'H2O_181': molliere2019_doi,  # TODO not in the referenced paper
        'H2O_182': molliere2019_doi,  # TODO not in the referenced paper
        'H2O_main_iso': molliere2019_doi,
        'H2O_pokazatel_main_iso': '10.1093/mnras/sty1877',
        'H2S_main_iso': molliere2019_doi,
        'HCN_main_iso': molliere2019_doi,
        'K': molliere2019_doi,
        'K_allard_cold': 'unknown',  # TODO not in the referenced paper nor in the docs
        'Li': kurucz_website,
        'Mg': kurucz_website,
        'Mg+': kurucz_website,
        'N': kurucz_website,
        'Na_allard': molliere2019_doi,
        'Na_allard_new': 'unknown',  # TODO difference unclear with "old" version
        'NH3_main_iso': 'unknown',  # TODO referenced twice in the docs! Which one is it?
        'NH3_Coles_main_iso': '10.1093/mnras/stz2778',
        'O3_main_iso': molliere2019_doi,
        'OH_main_iso': molliere2019_doi,
        'PH3_main_iso': '10.1093/mnras/stu2246',
        'Si': kurucz_website,
        'SiO_main_iso': '10.1093/mnras/stt1105',
        'SiO_main_iso_new_incl_UV': 'unknown',  # TODO not in the docs
        'Ti': kurucz_website,
        'TiO_46_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_46_Plez': molliere2019_doi,
        'TiO_47_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_47_Plez': molliere2019_doi,
        'TiO_48_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_48_Plez': molliere2019_doi,
        'TiO_49_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_49_Plez': molliere2019_doi,
        'TiO_50_Exomol_McKemmish': '10.1093/mnras/stz1818',
        'TiO_50_Plez': molliere2019_doi,
        'TiO_all_iso_Plez': molliere2019_doi,
        'TiO_all_iso_exo': molliere2019_doi,
        'V': kurucz_website,
        'V+': kurucz_website,
        'VO': molliere2019_doi,
        'VO_ExoMol_McKemmish': '10.1093/mnras/stw1969',
        'VO_ExoMol_Specific_Transitions': '10.1093/mnras/stw1969',  # TODO difference unclear with "default" version
        'Y': kurucz_website
    }
    contributor_dict = {
        'Al': molaverdikhani_email,
        'B': molaverdikhani_email,
        'Be': molaverdikhani_email,
        'C2H2_main_iso': 'None',
        'Ca': molaverdikhani_email,
        'Ca+': molaverdikhani_email,
        'CaH': 'None',
        'CH4_212': 'None',
        'CH4_hargreaves_main_iso': 'None',
        'CH4_Hargreaves_main_iso': 'None',
        'CH4_main_iso': 'None',
        'CO2_main_iso': 'None',
        'CO_27': 'None',
        'CO_28': 'None',
        'CO_36': 'None',
        'CO_37': 'None',
        'CO_38': 'None',
        'CO_all_iso': 'None',
        'CO_main_iso': 'None',
        'Cr': molaverdikhani_email,
        'Fe': molaverdikhani_email,
        'Fe+': molaverdikhani_email,
        'FeH_main_iso': 'None',
        'H2_12': 'None',
        'H2_main_iso': 'None',
        'H2O_162': 'None',
        'H2O_171': 'None',
        'H2O_172': 'None',
        'H2O_181': 'None',
        'H2O_182': 'None',
        'H2O_main_iso': 'None',
        'H2O_pokazatel_main_iso': 'gandhi@strw.leidenuniv.nl',
        'H2S_main_iso': 'None',
        'HCN_main_iso': 'None',
        'K': 'None',
        'K_allard_cold': 'None',
        'Li': molaverdikhani_email,
        'Mg': molaverdikhani_email,
        'Mg+': molaverdikhani_email,
        'N': molaverdikhani_email,
        'Na_allard': 'None',
        'Na_allard_new': 'None',
        'NH3_main_iso': 'None',
        'NH3_Coles_main_iso': 'gandhi@strw.leidenuniv.nl',
        'O3_main_iso': 'None',
        'OH_main_iso': 'None',
        'PH3_main_iso': 'adriano.miceli@stud.unifi.it',
        'Si': molaverdikhani_email,
        'SiO_main_iso': 'None',
        'SiO_main_iso_new_incl_UV': 'None',
        'Ti': molaverdikhani_email,
        'TiO_46_Exomol_McKemmish': 'None',
        'TiO_46_Plez': 'None',
        'TiO_47_Exomol_McKemmish': 'None',
        'TiO_47_Plez': 'None',
        'TiO_48_Exomol_McKemmish': 'None',
        'TiO_48_Plez': 'None',
        'TiO_49_Exomol_McKemmish': 'None',
        'TiO_49_Plez': 'None',
        'TiO_50_Exomol_McKemmish': 'None',
        'TiO_50_Plez': 'None',
        'TiO_all_iso_Plez': 'None',
        'TiO_all_iso_exo': 'None',
        'V': molaverdikhani_email,
        'V+': molaverdikhani_email,
        'VO': 'None',
        'VO_ExoMol_McKemmish': 'regt@strw.leidenuniv.nl',
        'VO_ExoMol_Specific_Transitions': 'regt@strw.leidenuniv.nl',
        'Y': molaverdikhani_email
    }
    description_dict = {
        'Al': kurucz_description,
        'B': kurucz_description,
        'Be': kurucz_description,
        'C2H2_main_iso': 'None',
        'Ca': kurucz_description,
        'Ca+': kurucz_description,
        'CaH': 'None',
        'CH4_212': 'None',
        'CH4_hargreaves_main_iso': 'None',
        'CH4_Hargreaves_main_iso': 'None',
        'CH4_main_iso': 'None',
        'CO2_main_iso': 'None',
        'CO_27': 'None',
        'CO_28': 'None',
        'CO_36': 'None',
        'CO_37': 'None',
        'CO_38': 'None',
        'CO_all_iso': 'None',
        'CO_main_iso': 'None',
        'Cr': kurucz_description,
        'Fe': kurucz_description,
        'Fe+': kurucz_description,
        'FeH_main_iso': 'None',
        'H2_12': 'None',
        'H2_main_iso': 'None',
        'H2O_162': 'None',
        'H2O_171': 'None',
        'H2O_172': 'None',
        'H2O_181': 'None',
        'H2O_182': 'None',
        'H2O_main_iso': 'None',
        'H2O_pokazatel_main_iso': 'None',
        'H2S_main_iso': 'None',
        'HCN_main_iso': 'None',
        'K': 'None',
        'K_allard_cold': 'None',
        'Li': kurucz_description,
        'Mg': kurucz_description,
        'Mg+': kurucz_description,
        'N': kurucz_description,
        'Na_allard': 'None',
        'Na_allard_new': 'None',
        'NH3_main_iso': 'None',
        'O3_main_iso': 'None',
        'OH_main_iso': 'None',
        'PH3_main_iso': 'None',
        'Si': kurucz_description,
        'SiO_main_iso': 'None',
        'SiO_main_iso_new_incl_UV': 'None',
        'Ti': kurucz_description,
        'TiO_46_Exomol_McKemmish': 'None',
        'TiO_46_Plez': 'None',
        'TiO_47_Exomol_McKemmish': 'None',
        'TiO_47_Plez': 'None',
        'TiO_48_Exomol_McKemmish': 'None',
        'TiO_48_Plez': 'None',
        'TiO_49_Exomol_McKemmish': 'None',
        'TiO_49_Plez': 'None',
        'TiO_50_Exomol_McKemmish': 'None',
        'TiO_50_Plez': 'None',
        'TiO_all_iso_Plez': 'None',
        'TiO_all_iso_exo': 'None',
        'V': kurucz_description,
        'V+': kurucz_description,
        'VO': 'None',
        'VO_ExoMol_McKemmish': 'None',
        'VO_ExoMol_Specific_Transitions': 'Most accurate transitions from McKemmish et al. (2016)',
        'Y': kurucz_description
    }
    molmass_dict = {
        'Al': getMM('Al'),
        'B': getMM('B'),
        'Be': getMM('Be'),
        'C2H2_main_iso': getMM('C2H2'),
        'Ca': getMM('Ca'),
        'Ca+': getMM('Ca') - getMM('e-'),
        'CaH': getMM('CaH'),
        'CH4_212': getMM('CH3D'),
        'CH4_hargreaves_main_iso': getMM('CH4'),
        'CH4_Hargreaves_main_iso': getMM('CH4'),
        'CH4_main_iso': getMM('CH4'),
        'CO2_main_iso': getMM('CO2'),
        'CO_27': getMM('12C') + getMM('17O'),
        'CO_28': getMM('12C') + getMM('18O'),
        'CO_36': getMM('13C') + getMM('16O'),
        'CO_37': getMM('13C') + getMM('17O'),
        'CO_38': getMM('13C') + getMM('18O'),
        'CO_all_iso': getMM('CO_all_iso'),
        'CO_main_iso': getMM('CO'),
        'Cr': getMM('Cr'),
        'Fe': getMM('Fe'),
        'Fe+': getMM('Fe') - getMM('e-'),
        'FeH_main_iso': getMM('FeH'),
        'H2_12': getMM('HD'),
        'H2_main_iso': getMM('H2'),
        'H2O_162': getMM('1H') + getMM('16O') + getMM('2H'),
        'H2O_171': getMM('1H') + getMM('17O') + getMM('1H'),
        'H2O_172': getMM('1H') + getMM('17O') + getMM('2H'),
        'H2O_181': getMM('1H') + getMM('18O') + getMM('1H'),
        'H2O_182': getMM('1H') + getMM('18O') + getMM('2H'),
        'H2O_main_iso': getMM('1H') + getMM('16O') + getMM('2H'),
        'H2O_pokazatel_main_iso': getMM('H2O'),
        'H2S_main_iso': getMM('H2S'),
        'HCN_main_iso': getMM('HCN'),
        'K': getMM('K'),
        'K_allard_cold': getMM('K'),
        'Li': getMM('Li'),
        'Mg': getMM('Mg'),
        'Mg+': getMM('Mg') - getMM('e-'),
        'N': getMM('N'),
        'Na_allard': getMM('Na'),
        'Na_allard_new': getMM('Na'),
        'NH3_main_iso': getMM('NH3'),
        'O3_main_iso': getMM('O3'),
        'OH_main_iso': getMM('OH'),
        'PH3_main_iso': getMM('PH3'),
        'Si': getMM('Si'),
        'SiO_main_iso': getMM('SiO'),
        'SiO_main_iso_new_incl_UV': getMM('SiO'),
        'Ti': getMM('Ti'),
        'TiO_46_Exomol_McKemmish': getMM('46Ti') + getMM('16O'),
        'TiO_46_Plez': getMM('46Ti') + getMM('16O'),
        'TiO_47_Exomol_McKemmish': getMM('47Ti') + getMM('16O'),
        'TiO_47_Plez': getMM('47Ti') + getMM('16O'),
        'TiO_48_Exomol_McKemmish': getMM('48Ti') + getMM('16O'),
        'TiO_48_Plez': getMM('48Ti') + getMM('16O'),
        'TiO_49_Exomol_McKemmish': getMM('49Ti') + getMM('16O'),
        'TiO_49_Plez': getMM('49Ti') + getMM('16O'),
        'TiO_50_Exomol_McKemmish': getMM('50Ti') + getMM('16O'),
        'TiO_50_Plez': getMM('50Ti') + getMM('16O'),
        'TiO_all_iso_Plez': getMM('TiO_all_iso'),
        'TiO_all_iso_exo': getMM('TiO_all_iso'),
        'V': getMM('V'),
        'V+': getMM('V') - getMM('e-'),
        'VO': getMM('VO'),
        'VO_ExoMol_McKemmish': getMM('VO'),
        'VO_ExoMol_Specific_Transitions': getMM('VO'),
        'Y': getMM('Y')
    }

    # Loading
    print("Loading default PT grid...")

    opacities_temperature_profile_grid = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
    )

    opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)

    opacities_temperatures = np.unique(opacities_temperature_profile_grid[:, 0])
    opacities_pressures = np.unique(opacities_temperature_profile_grid[:, 1])  # grid is already in bar

    print("Loading default files names...")
    line_paths = np.loadtxt(os.path.join(path_input_data, 'opa_input_files', 'opa_filenames.txt'), dtype=str)

    input_directory = os.path.join(path_input_data, 'opacities', 'lines', 'line_by_line')

    if output_directory is None:
        output_directory_ref = input_directory
    else:
        output_directory_ref = copy.deepcopy(output_directory)

    for f in os.scandir(input_directory):
        if f.is_dir():
            directory = f.path

            species = directory.rsplit(os.path.sep, 1)[1]
            not_in_dict = False

            if species not in doi_dict:
                warnings.warn(f"species '{species}' was not in DOI dict; "
                              f"add key '{species}' to the script doi_ict to run this conversion")
                not_in_dict = True

            if species not in contributor_dict:
                warnings.warn(f"species '{species}' was not in contributor dict; "
                              f"add key '{species}' to the script contributor_dict to run this conversion")
                not_in_dict = True

            if species not in description_dict:
                warnings.warn(f"species '{species}' was not in contributor dict; "
                              f"add key '{species}' to the script contributor_dict to run this conversion")
                not_in_dict = True

            if species not in molmass_dict:
                warnings.warn(f"species '{species}' was not in molar mass dict; "
                              f"add key '{species}' to the script molmass_dict to run this conversion")
                not_in_dict = True

            if not_in_dict:
                print(" Skipping due to missing species in supplementary info dict...")
                continue

            output_directory = os.path.join(output_directory_ref)

            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

            hdf5_opacity_file = os.path.join(output_directory, species + '.otable.petitRADTRANS.h5')

            if os.path.isfile(hdf5_opacity_file) and not rewrite:
                print(f"File '{hdf5_opacity_file}' already exists, skipping conversion...")
                continue

            print(f"Converting opacities in '{directory}'...")

            custom_pt_grid_file = os.path.join(directory, 'PTpaths.ls')

            if os.path.isfile(custom_pt_grid_file):
                print(" Found custom PT grid")

                # _sort_opa_pt_grid converts bar into cgs
                custom_grid_data = Radtrans._sort_opa_pt_grid(custom_pt_grid_file)

                opacities_temperature_profile_grid_ = custom_grid_data[0]
                opacities_temperatures_ = np.unique(opacities_temperature_profile_grid_[:, 0])
                opacities_pressures_ = np.unique(opacities_temperature_profile_grid_[:, 1])
                opacities_pressures_ *= 1e-6  # cgs to bar
                line_paths_ = custom_grid_data[1]

                for i, line_path in enumerate(line_paths_):
                    line_paths_[i] = directory + os.path.sep + line_path

                line_paths_ = np.array(line_paths_)
            else:
                print(" Using default PT grid")

                opacities_temperatures_ = copy.deepcopy(opacities_temperatures)
                opacities_pressures_ = copy.deepcopy(opacities_pressures)

                line_paths_ = []

                for f_ in os.scandir(directory):
                    if f_.is_file():
                        line_paths_.append(f_.path)

                f_ = []

                for ref_path in line_paths:
                    for i, line_path in enumerate(line_paths_):
                        if ref_path in line_path:
                            f_.append(line_paths_.pop(i))

                            break

                line_paths_ = np.array(f_)

                if line_paths_.size != line_paths.size:
                    warnings.warn(f"number of opacity files founds in '{directory}' ({line_paths_.size}) "
                                  f"does not match the expected number of files ({line_paths.size})")

            molparam_file = os.path.join(directory, 'molparam_id.txt')

            if os.path.isfile(molparam_file):
                print(" Loading isotopic ratio...")
                with open(molparam_file, 'r') as f2:
                    isotopic_ratio = float(f2.readlines()[-1])
            else:
                raise FileNotFoundError(f"file '{molparam_file}' not found: unable to load isotopic ratio")

            n_items = fi.get_file_size(os.path.join(directory, 'wlen.dat'))
            wavelengths = fi.read_all_kappa(os.path.join(directory, 'wlen.dat'), n_items)
            wavenumbers = 1 / wavelengths[::-1]  # cm to cm-1

            opacities = np.zeros((line_paths_.size, wavelengths.size))

            for i, line_path in enumerate(line_paths_):
                if not os.path.isfile(line_path):
                    raise FileNotFoundError(f"file '{line_path}' does not exists")

                print(f" Loading file '{line_path}' ({i + 1}/{line_paths_.size})...")

                opacities[i] = fi.read_all_kappa(line_path, n_items)

            print(" Reshaping...")
            opacities = opacities.reshape((opacities_temperatures_.size, opacities_pressures_.size, wavelengths.size))
            opacities = np.moveaxis(opacities, 0, 1)  # Exo-Mol axis order (pressures, temperatures, wavenumbers)
            opacities = opacities[:, :, ::-1]  # match the wavenumber order

            print(f" Writing file '{hdf5_opacity_file}'...", end=' ')

            with h5py.File(hdf5_opacity_file, "w") as fh5:
                dataset = fh5.create_dataset(
                    name='DOI',
                    data=doi_dict[species]
                )
                dataset.attrs['long_name'] = 'Data object identifier linked to the data'
                dataset.attrs['contributor'] = contributor_dict[species]
                dataset.attrs['additional_description'] = description_dict[species]

                dataset = fh5.create_dataset(
                    name='Date_ID',
                    data=f'petitRADTRANS-v{petitRADTRANS.__version__}_{datetime.datetime.utcnow().isoformat()}'
                )
                dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                             'at which the table has been created, ' \
                                             'along with the version of petitRADTRANS'

                dataset = fh5.create_dataset(
                    name='wavenumbers',
                    data=wavenumbers
                )
                dataset.attrs['long_name'] = 'Opacities wavenumbers'
                dataset.attrs['units'] = 'cm^-1'

                dataset = fh5.create_dataset(
                    name='opacities',
                    data=opacities
                )
                dataset.attrs['long_name'] = 'Table of the opacities with axes (pressure, temperature, wavenumber)'
                dataset.attrs['units'] = 'cm^2.g^-1'

                dataset = fh5.create_dataset(
                    name='mol_mass',
                    data=molmass_dict[species]
                )
                dataset.attrs['long_name'] = 'Mass of the species'
                dataset.attrs['units'] = 'AMU'

                dataset = fh5.create_dataset(
                    name='mol_name',
                    data=species
                )
                dataset.attrs['long_name'] = 'Name of the species described'

                dataset = fh5.create_dataset(
                    name='isotopic_ratio',
                    data=isotopic_ratio
                )
                dataset.attrs['long_name'] = 'Isotopologue occurence rate on Earth'

                dataset = fh5.create_dataset(
                    name='p',
                    data=opacities_pressures_
                )
                dataset.attrs['long_name'] = 'Pressure grid'
                dataset.attrs['units'] = 'bar'

                dataset = fh5.create_dataset(
                    name='t',
                    data=opacities_temperatures_
                )
                dataset.attrs['long_name'] = 'Temperature grid'
                dataset.attrs['units'] = 'K'

                dataset = fh5.create_dataset(
                    name='temperature_grid_type',
                    data='regular'
                )
                dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                             '(same temperatures for all pressures) or "pressure-dependent"'

                dataset = fh5.create_dataset(
                    name='wlrange',
                    data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
                )
                dataset.attrs['long_name'] = 'Wavelength range covered'
                dataset.attrs['units'] = 'µm'

                dataset = fh5.create_dataset(
                    name='wnrange',
                    data=np.array([wavenumbers.min(), wavenumbers.max()])
                )
                dataset.attrs['long_name'] = 'Wavenumber range covered'
                dataset.attrs['units'] = 'cm^-1'

            print("Done.")

    print("Conversions successful.")


def phoenix_spec_dat2h5():
    """
    Convert a PHOENIX stellar spectrum in .dat format to HDF5 format.
    """
    # Load the stellar parameters
    description = np.genfromtxt(nc.spec_path + os.path.sep + 'stellar_params.dat')

    # Initialize the grids
    log_temp_grid = description[:, 0]
    star_rad_grid = description[:, 1]

    # Load the corresponding numbered spectral files
    spec_dats = []

    for spec_num in range(len(log_temp_grid)):
        spec_dats.append(np.genfromtxt(nc.spec_path + '/spec_'
                                       + str(int(spec_num)).zfill(2) + '.dat'))

    # Write the HDF5 file
    with h5py.File("stellar_spectra.h5", "w") as f:
        t_eff = f.create_dataset(
            name='log10_effective_temperature',
            data=log_temp_grid
        )
        t_eff.attrs['units'] = 'log10(K)'

        radius = f.create_dataset(
            name='radius',
            data=star_rad_grid
        )
        radius.attrs['units'] = 'R_sun'

        mass = f.create_dataset(
            name='mass',
            data=description[:, 2]
        )
        mass.attrs['units'] = 'M_sun'

        spectral_type = f.create_dataset(
            name='spectral_type',
            data=description[:, -1]
        )
        spectral_type.attrs['units'] = 'None'

        wavelength = f.create_dataset(
            name='wavelength',
            data=np.asarray(spec_dats)[0, :, 0]
        )
        wavelength.attrs['units'] = 'cm'

        spectral_radiosity = f.create_dataset(
            name='spectral_radiosity',
            data=np.asarray(spec_dats)[:, :, 1]
        )
        spectral_radiosity.attrs['units'] = 'erg/s/cm^2/Hz'

    print("Conversion successful.")
