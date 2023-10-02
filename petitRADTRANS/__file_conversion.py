"""Stores functions that convert files from a format to another.

The functions in this module are stored for the sake of keeping trace of changes made to files. They are intended to be
used only once.
"""
import copy
import datetime
import os
import warnings

import h5py
import numpy as np

import petitRADTRANS
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass


def __print_missing_data_file_message(object, object_name, directory):
    print(f"Data for {object} '{object_name}' not found (path '{directory}' does not exist), skipping...")


def __print_skipping_message(hdf5_opacity_file):
    print(f"File '{hdf5_opacity_file}' already exists, skipping conversion...")


def bin_species_exok(species, resolution):
    """
    This function uses exo-k to bin the c-k table of a
    single species to a desired (lower) spectral resolution.

    Args:
        species : string
            The name of the species
        resolution : int
            The desired spectral resolving power.
    """
    from petitRADTRANS.radtrans import Radtrans
    from petitRADTRANS.config import petitradtrans_config_parser

    prt_path = petitradtrans_config_parser.get_input_data_path()
    atmosphere = Radtrans(
        line_species=species,
        wavelengths_boundaries=[0.1, 251.]
    )
    ck_path = os.path.join(prt_path, 'opacities', 'lines', 'corr_k')

    print(f"Saving re-binned opacities to directory '{ck_path}'")
    print(f" Resolving power: {resolution}")

    masses = {}

    for spec in species:
        masses[spec.split('_')[0]] = get_species_molar_mass(spec)

    rebin_ck_line_opacities(
        radtrans=atmosphere,
        resolution=int(resolution),
        path=ck_path,
        species=species,
        species_molar_masses=masses
    )


def chemical_table_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(), rewrite=False):
    from petitRADTRANS.fortran_chemistry import fortran_chemistry as fchem
    # Read in parameters of chemistry grid
    path = os.path.join(path_input_data, "abundance_files")
    hdf5_file = os.path.join(path, 'mass_mixing_ratios.h5')

    if os.path.isfile(hdf5_file) and not rewrite:
        __print_skipping_message(hdf5_file)
        return

    log10_metallicities = np.genfromtxt(os.path.join(path, "FEHs.dat"))
    co_ratios = np.genfromtxt(os.path.join(path, "COs.dat"))
    temperature = np.genfromtxt(os.path.join(path, "temps.dat"))
    pressure = np.genfromtxt(os.path.join(path, "pressures.dat"))

    with open(os.path.join(path, "species.dat"), 'r') as f:
        species_name = f.readlines()

    for i in range(len(species_name)):
        species_name[i] = species_name[i][:-1]

    chemistry_table = fchem.read_dat_chemical_table(
        int(len(log10_metallicities)), int(len(co_ratios)), int(len(temperature)), int(len(pressure)),
        int(len(species_name)),
        path_input_data + os.path.sep  # the complete path is defined in the function
    )

    chemistry_table = np.array(chemistry_table, dtype='d', order='F')

    with h5py.File(hdf5_file, 'w') as f:
        log10_metallicities = f.create_dataset(
            name='iron_to_hydrogen_ratios',
            data=log10_metallicities
        )
        log10_metallicities.attrs['units'] = 'dex'

        co = f.create_dataset(
            name='carbon_to_oxygen_ratios',
            data=co_ratios
        )
        co.attrs['units'] = 'None'

        temp = f.create_dataset(
            name='temperatures',
            data=temperature
        )
        temp.attrs['units'] = 'K'

        p = f.create_dataset(
            name='pressures',
            data=pressure
        )
        p.attrs['units'] = 'bar'

        name = f.create_dataset(
            name='species_names',
            data=species_name
        )
        name.attrs['units'] = 'N/A'

        table = f.create_dataset(
            name='mass_mixing_ratios',
            data=chemistry_table
        )
        table.attrs['units'] = 'None'

    print("Successfully converted chemical tables")


def continuum_cia_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                         rewrite=False, output_directory=None):
    """Using ExoMol units for HDF5 files."""
    from petitRADTRANS.fortran_inputs import fortran_inputs as finput

    # Initialize infos
    molliere2019_doi = '10.1051/0004-6361/201935470'

    doi_dict = {
        'H2-H2': molliere2019_doi,
        'H2-He': molliere2019_doi,
        'H2O-H2O': 'unknown',
        'H2O-N2': 'unknown',
        'N2-H2': 'unknown',
        'N2-He': 'unknown',
        'N2-N2': 'unknown',
        'O2-O2': 'unknown',
        'N2-O2': 'unknown',
        'CO2-CO2': 'unknown'
    }

    description_dict = {
        'H2-H2': 'None',
        'H2-He': 'None',
        'H2O-H2O': 'None',
        'H2O-N2': 'None',
        'N2-H2': 'None',
        'N2-He': 'None',
        'N2-N2': 'None',
        'O2-O2': 'None',
        'N2-O2': 'None',
        'CO2-CO2': 'None'
    }

    # Get only existing directories
    input_directory = os.path.join(path_input_data, 'opacities', 'continuum', 'CIA')

    # Save each clouds data into HDF5 file
    if output_directory is None:
        output_directory_ref = input_directory
    else:
        output_directory_ref = copy.deepcopy(output_directory)

    # Loop over CIAs
    for i, key in enumerate(doi_dict):
        # Check if data directory exists
        cia_dir = os.path.join(input_directory, key)

        if not os.path.isdir(cia_dir):
            __print_missing_data_file_message('CIA', key, cia_dir)
            continue

        # Get HDF5 file name
        output_directory = output_directory_ref
        hdf5_cia_file = os.path.join(output_directory, key + '.ciatable.petitRADTRANS.h5')

        if os.path.isfile(hdf5_cia_file) and not rewrite:
            __print_skipping_message(hdf5_cia_file)
            continue

        # Check if current key is in all information dicts
        not_in_dict = False

        if key not in description_dict:
            warnings.warn(f"CIA '{key}' was not in contributor dict; "
                          f"add key '{key}' to the script contributor_dict to run this conversion")
            not_in_dict = True

        if not_in_dict:
            print(" Skipping due to missing species in supplementary info dict...")
            continue

        # Read the dat files
        colliding_species = key.split('-')

        print(f"  Read CIA opacities for {key}...")
        cia_directory = os.path.join(path_input_data, 'opacities', 'continuum', 'CIA', key)

        if os.path.isdir(cia_directory) is False:
            raise FileNotFoundError(f"CIA directory '{cia_directory}' do not exists")

        cia_wavelength_grid, cia_temperature_grid, cia_alpha_grid, \
            cia_temp_dims, cia_lambda_dims = finput.load_cia_opacities(key, path_input_data)
        cia_alpha_grid = np.array(cia_alpha_grid, dtype='d', order='F')
        cia_temperature_grid = cia_temperature_grid[:cia_temp_dims]
        cia_wavelength_grid = cia_wavelength_grid[:cia_lambda_dims]
        cia_alpha_grid = cia_alpha_grid[:cia_lambda_dims, :cia_temp_dims]

        weight = 1

        for species in colliding_species:
            weight = weight * get_species_molar_mass(species)

        cia_dict = {
            'id': key,
            'molecules': colliding_species,
            'weight': weight,
            'lambda': cia_wavelength_grid,
            'temperature': cia_temperature_grid,
            'alpha': cia_alpha_grid
        }

        wavenumbers = 1 / cia_dict['lambda'][::-1]  # cm to cm-1, with correct ordering

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # Write HDF5 file
        print(f" Writing file '{hdf5_cia_file}'...", end=' ')

        with h5py.File(hdf5_cia_file, "w") as fh5:
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
            dataset.attrs['long_name'] = 'CIA wavenumbers'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='cross_sections',
                data=np.transpose(cia_dict['alpha'])[:, ::-1]  # (temperature, wavenumber) wavenumbers ordering
            )
            dataset.attrs['long_name'] = 'Table of the cross-sections with axes (temperature, wavenumber)'
            dataset.attrs['units'] = 'cm^-1.mol^-2.cm^6'

            dataset = fh5.create_dataset(
                name='t',
                data=cia_dict['temperature']
            )
            dataset.attrs['long_name'] = 'Temperature grid'
            dataset.attrs['units'] = 'K'

            dataset = fh5.create_dataset(
                name='mol_mass',
                data=np.array([get_species_molar_mass(species) for species in cia_dict['molecules']])
            )
            dataset.attrs['long_name'] = 'Masses of the colliding species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='mol_name',
                data=cia_dict['molecules']
            )
            dataset.attrs['long_name'] = 'Names of the colliding species described'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([cia_dict['lambda'].min(), cia_dict['lambda'].max()]) * 1e4  # cm to um
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

    print("Successfully converted CIA opacities")


def continuum_clouds_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                      rewrite=False, output_directory=None):
    from petitRADTRANS.fortran_inputs import fortran_inputs as finput

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
        'Al2O3(c)_cm': get_species_molar_mass('Al2O3'),
        'Al2O3(c)_cd': get_species_molar_mass('Al2O3'),
        'Fe(c)_am': get_species_molar_mass('Fe'),
        'Fe(c)_ad': get_species_molar_mass('Fe'),
        'Fe(c)_cm': get_species_molar_mass('Fe'),
        'Fe(c)_cd': get_species_molar_mass('Fe'),
        'H2O(c)_cm': get_species_molar_mass('H2O'),
        'H2O(c)_cd': get_species_molar_mass('H2O'),
        'KCL(c)_cm': get_species_molar_mass('H2O'),
        'KCL(c)_cd': get_species_molar_mass('H2O'),
        'Mg05Fe05SiO3(c)_am': (
            0.5 * get_species_molar_mass('Mg') + 0.5 * get_species_molar_mass('Fe') + get_species_molar_mass('SiO3')
        ),
        'Mg05Fe05SiO3(c)_ad': (
            0.5 * get_species_molar_mass('Mg') + 0.5 * get_species_molar_mass('Fe') + get_species_molar_mass('SiO3')
        ),
        'Mg2SiO4(c)_am': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_ad': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_cm': get_species_molar_mass('Mg2SiO4'),
        'Mg2SiO4(c)_cd': get_species_molar_mass('Mg2SiO4'),
        'MgAl2O4(c)_cm': get_species_molar_mass('MgAl2O4'),
        'MgAl2O4(c)_cd': get_species_molar_mass('MgAl2O4'),
        'MgFeSiO4(c)_am': get_species_molar_mass('MgFeSiO4'),
        'MgFeSiO4(c)_ad': get_species_molar_mass('MgFeSiO4'),
        'MgSiO3(c)_am': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_ad': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_cm': get_species_molar_mass('MgSiO3'),
        'MgSiO3(c)_cd': get_species_molar_mass('MgSiO3'),
        'Na2S(c)_cm': get_species_molar_mass('Na2S'),
        'Na2S(c)_cd': get_species_molar_mass('Na2S'),
        'SiC(c)_cm': get_species_molar_mass('SiC'),
        'SiC(c)_cd': get_species_molar_mass('SiC')
    }

    # Get only existing directories
    input_directory = os.path.join(path_input_data, 'opacities', 'continuum', 'clouds')
    bad_keys = []

    for key in doi_dict:
        species = key.split('(', 1)[0]
        species_dir = os.path.join(input_directory, species + '_c')

        if not os.path.isdir(species_dir):
            __print_missing_data_file_message('cloud', key, species_dir)
            bad_keys.append(key)
            continue

        particle_mode = key.rsplit('_', 1)[1]

        particle_mode_dir = None

        if particle_mode[0] == 'c':
            particle_mode_dir = os.path.join(species_dir, 'crystalline')
        elif particle_mode[0] == 'a':
            particle_mode_dir = os.path.join(species_dir, 'amorphous')

        if not os.path.isdir(particle_mode_dir):
            __print_missing_data_file_message('cloud', key, particle_mode_dir)
            del doi_dict[key]
            continue

        if particle_mode[1] == 'm':
            particle_mode_dir = os.path.join(particle_mode_dir, 'mie')
        elif particle_mode[1] == 'd':
            particle_mode_dir = os.path.join(particle_mode_dir, 'DHS')

        if not os.path.isdir(particle_mode_dir):
            print(__print_missing_data_file_message('cloud', key, particle_mode_dir))
            del doi_dict[key]
            continue

    for key in bad_keys:
        del doi_dict[key]

    if len(doi_dict) == 0:
        print("No cloud opacities conversion is necessary or possible")
        return

    # Prepare single strings delimited by ':' which are then put into Fortran routines
    cloud_species_modes = []
    cloud_species = []

    for key in doi_dict:
        cloud_species_ = key.rsplit('_', 1)
        cloud_species_modes.append(cloud_species_[1])
        cloud_species.append(cloud_species_[0])

    all_cloud_species = ''

    for cloud_species_ in cloud_species:
        all_cloud_species = all_cloud_species + cloud_species_ + ':'

    all_cloud_species_mode = ''

    for cloud_species_mode in cloud_species_modes:
        all_cloud_species_mode = all_cloud_species_mode + cloud_species_mode + ':'

    reference_file = os.path.join(
        path_input_data, 'opacities', 'continuum', 'clouds', 'MgSiO3_c', 'amorphous', 'mie', 'opa_0001.dat'
    )

    if not os.path.isfile(reference_file):
        raise FileNotFoundError(
            f"reference file for loading .dat cloud opacities ('{reference_file}') not found, "
            f"it must be downloaded "
            f"(see https://petitradtrans.readthedocs.io/en/latest/content/available_opacities.html)"
        )

    n_cloud_wavelength_bins = int(len(np.genfromtxt(reference_file)[:, 0]))

    # Load .dat files
    print("Loading dat files...")
    cloud_particles_densities, cloud_absorption_opacities, cloud_scattering_opacities, \
        cloud_asymmetry_parameter, cloud_wavelengths, cloud_particles_radius_bins, cloud_particles_radii \
        = finput.load_cloud_opacities(
            path_input_data, all_cloud_species, all_cloud_species_mode, len(doi_dict), n_cloud_wavelength_bins
        )

    wavenumbers = 1 / cloud_wavelengths[::-1]  # cm to cm-1

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
            __print_skipping_message(hdf5_opacity_file)
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

    print("Successfully converted cloud opacities")


def correlated_k_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                  rewrite=False, output_directory=None):
    from petitRADTRANS.fortran_inputs import fortran_inputs as finput
    import petitRADTRANS.physical_constants as cst
    from petitRADTRANS.radtrans import Radtrans

    # Initialize infos
    kurucz_website = 'http://kurucz.harvard.edu/'
    molliere2019_doi = '10.1051/0004-6361/201935470'
    burrows2003_doi = '10.1086/345412'
    mckemmish2019_doi = '10.1093/mnras/stz1818'

    molaverdikhani_email = 'karan.molaverdikhani@colorado.edu'

    kurucz_description = 'gamma_nat + V dW, sigma_therm'
    k_chubb_email = 'klc20@st-andrews.ac.uk'

    # None is used for already referenced HDF5 files
    doi_dict = {
        'Al': kurucz_website,
        'Al+': kurucz_website,
        'AlH': None,
        'AlO': None,
        'C2H2': None,
        'C2H4': None,
        'Ca': kurucz_website,
        'Ca+': kurucz_website,
        'CaH': None,
        'CH4': None,
        'CO_12_HITEMP': molliere2019_doi,
        'CO_13_HITEMP': molliere2019_doi,
        'CO_13_Chubb': None,
        'CO_all_iso_Chubb': None,
        'CO_all_iso_HITEMP': molliere2019_doi,
        'CO2': None,
        'CrH': None,
        'Fe': kurucz_website,
        'Fe+': kurucz_website,
        'FeH': None,
        'H2O_Exomol': None,
        'H2O_HITEMP': molliere2019_doi,
        'H2S': None,
        'HCN': None,
        'K_allard': molliere2019_doi,
        'K_burrows': burrows2003_doi,
        'K_lor_cut': molliere2019_doi,
        'Li': kurucz_website,
        'Mg': kurucz_website,
        'Mg+': kurucz_website,
        'MgH': None,
        'MgO': None,
        'Na_allard': molliere2019_doi,
        'Na_burrows': burrows2003_doi,
        'Na_lor_cut': molliere2019_doi,
        'NaH': None,
        'NH3': None,
        'O': kurucz_website,
        'O+': kurucz_website,  # TODO not in the docs
        'O2': None,
        'O3': molliere2019_doi,
        'OH': None,
        'PH3': None,
        'SH': None,
        'Si': kurucz_website,
        'Si+': kurucz_website,
        'SiO': None,
        'SiO2': None,
        'Ti': kurucz_website,
        'Ti+': kurucz_website,
        'TiO_48_Exomol': mckemmish2019_doi,
        'TiO_48_Plez': molliere2019_doi,
        'TiO_all_Exomol': mckemmish2019_doi,
        'TiO_all_Plez': molliere2019_doi,
        'V': kurucz_website,
        'V+': kurucz_website,
        'VO': None,
        'VO_Plez': molliere2019_doi
    }
    contributor_dict = {
        'Al': kurucz_description,
        'Al+': kurucz_description,
        'AlH': k_chubb_email,
        'AlO': k_chubb_email,
        'C2H2': k_chubb_email,
        'C2H4': k_chubb_email,
        'Ca': molaverdikhani_email,
        'Ca+': molaverdikhani_email,
        'CaH': k_chubb_email,
        'CH4': k_chubb_email,
        'CO_12_HITEMP': 'None',
        'CO_13_HITEMP': 'None',
        'CO_13_Chubb': k_chubb_email,
        'CO_all_iso_Chubb': k_chubb_email,
        'CO_all_iso_HITEMP': 'None',
        'CO2': k_chubb_email,
        'CrH': k_chubb_email,
        'Fe': molaverdikhani_email,
        'Fe+': molaverdikhani_email,
        'FeH': k_chubb_email,
        'H2O_Exomol': k_chubb_email,
        'H2O_HITEMP': 'None',
        'H2S': k_chubb_email,
        'HCN': k_chubb_email,
        'K_allard': 'None',
        'K_burrows': 'None',
        'K_lor_cut': 'None',
        'Li': molaverdikhani_email,
        'Mg': molaverdikhani_email,
        'Mg+': molaverdikhani_email,
        'MgH': k_chubb_email,
        'MgO': k_chubb_email,
        'Na_allard': 'None',
        'Na_burrows': 'None',
        'Na_lor_cut': 'None',
        'NaH': k_chubb_email,
        'NH3': k_chubb_email,
        'O': molaverdikhani_email,
        'O+': molaverdikhani_email,  # TODO not in the docs
        'O2': k_chubb_email,
        'O3': 'None',
        'OH': k_chubb_email,
        'PH3': k_chubb_email,
        'SH': k_chubb_email,
        'Si': molaverdikhani_email,
        'Si+': molaverdikhani_email,
        'SiO': k_chubb_email,
        'SiO2': k_chubb_email,
        'Ti': molaverdikhani_email,
        'Ti+': molaverdikhani_email,
        'TiO_48_Exomol': k_chubb_email,
        'TiO_48_Plez': 'None',
        'TiO_all_Exomol': k_chubb_email,
        'TiO_all_Plez': 'None',
        'V': molaverdikhani_email,
        'V+': molaverdikhani_email,
        'VO': k_chubb_email,
        'VO_Plez': 'None'
    }
    description_dict = {
        'Al': kurucz_description,
        'Al+': kurucz_description,
        'AlH': 'None',
        'AlO': 'None',
        'C2H2': 'None',
        'C2H4': 'None',
        'Ca': kurucz_description,
        'Ca+': kurucz_description,
        'CaH': 'None',
        'CH4': 'None',
        'CO_12_HITEMP': 'None',
        'CO_13_HITEMP': 'None',
        'CO_13_Chubb': 'None',
        'CO_all_iso_Chubb': 'None',
        'CO_all_iso_HITEMP': 'None',
        'CO2': 'None',
        'CrH': 'None',
        'Fe': kurucz_description,
        'Fe+': kurucz_description,
        'FeH': 'None',
        'H2O_Exomol': 'None',
        'H2O_HITEMP': 'None',
        'H2S': 'None',
        'HCN': 'None',
        'K_allard': 'Allard wings',
        'K_burrows': 'Burrows wings',
        'K_lor_cut': 'Lorentzian wings',
        'Li': kurucz_description,
        'Mg': kurucz_description,
        'Mg+': kurucz_description,
        'MgH': 'None',
        'MgO': 'None',
        'Na_allard': 'new Allard wings',  # TODO difference with "old" Allard wings?
        'Na_burrows': 'Burrows wings',
        'Na_lor_cut': 'Lorentzian wings',
        'NaH': 'None',
        'NH3': 'None',
        'O': kurucz_description,
        'O+': kurucz_description,  # TODO not in the docs
        'O2': 'None',
        'O3': 'None',
        'OH': 'None',
        'PH3': 'None',
        'SH': 'None',
        'Si': kurucz_description,
        'Si+': kurucz_description,
        'SiO': 'None',
        'SiO2': 'None',
        'Ti': kurucz_description,
        'Ti+': kurucz_description,
        'TiO_48_Exomol': 'None',
        'TiO_48_Plez': 'None',
        'TiO_all_Exomol': 'None',
        'TiO_all_Plez': 'None',
        'V': kurucz_description,
        'V+': kurucz_description,
        'VO': 'None',
        'VO_Plez': 'None'
    }
    molmass_dict = {
        'Al': get_species_molar_mass('Al'),
        'Al+': get_species_molar_mass('Al') - get_species_molar_mass('e-'),
        'AlH': get_species_molar_mass('AlH'),
        'AlO': get_species_molar_mass('AlO'),
        'C2H2': get_species_molar_mass('C2H2'),
        'C2H4': get_species_molar_mass('C2H4'),
        'Ca': get_species_molar_mass('Ca'),
        'Ca+': get_species_molar_mass('Ca') - get_species_molar_mass('e-'),
        'CaH': get_species_molar_mass('CaH'),
        'CH4': get_species_molar_mass('CH4'),
        'CO_12_HITEMP': get_species_molar_mass('12CO'),
        'CO_13_HITEMP': get_species_molar_mass('13CO'),
        'CO_13_Chubb': get_species_molar_mass('13CO'),
        'CO_all_iso_Chubb': get_species_molar_mass('CO'),
        'CO_all_iso_HITEMP': get_species_molar_mass('CO'),
        'CO2': get_species_molar_mass('CO2'),
        'CrH': get_species_molar_mass('CrH'),
        'Fe': get_species_molar_mass('Fe'),
        'Fe+': get_species_molar_mass('Fe+'),
        'FeH': get_species_molar_mass('FeH'),
        'H2O_Exomol': get_species_molar_mass('H2O'),
        'H2O_HITEMP': get_species_molar_mass('H2O'),
        'H2S': get_species_molar_mass('H2S'),
        'HCN': get_species_molar_mass('HCN'),
        'K_allard': get_species_molar_mass('K'),
        'K_burrows': get_species_molar_mass('K'),
        'K_lor_cut': get_species_molar_mass('K'),
        'Li': get_species_molar_mass('Li'),
        'Mg': get_species_molar_mass('Mg'),
        'Mg+': get_species_molar_mass('Mg') - get_species_molar_mass('e-'),
        'MgH':  get_species_molar_mass('MgH'),
        'MgO':  get_species_molar_mass('MgO'),
        'Na_allard':  get_species_molar_mass('Na'),
        'Na_burrows': get_species_molar_mass('Na'),
        'Na_lor_cut': get_species_molar_mass('Na'),
        'NaH': get_species_molar_mass('NaH'),
        'NH3': get_species_molar_mass('NH3'),
        'O': get_species_molar_mass('O'),
        'O+': get_species_molar_mass('O') - get_species_molar_mass('e-'),  # TODO not in the docs
        'O2': get_species_molar_mass('O2'),
        'O3': get_species_molar_mass('O3'),
        'OH': get_species_molar_mass('OH'),
        'PH3': get_species_molar_mass('PH3'),
        'SH': get_species_molar_mass('SH'),
        'Si': get_species_molar_mass('Si'),
        'Si+': get_species_molar_mass('Si') - get_species_molar_mass('e-'),
        'SiO': get_species_molar_mass('SiO'),
        'SiO2': get_species_molar_mass('SiO2'),
        'Ti': get_species_molar_mass('Ti'),
        'Ti+': get_species_molar_mass('Ti') - get_species_molar_mass('e-'),
        'TiO_48_Exomol': get_species_molar_mass('48TiO'),
        'TiO_48_Plez': get_species_molar_mass('48TiO'),
        'TiO_all_Exomol': get_species_molar_mass('TiO'),
        'TiO_all_Plez': get_species_molar_mass('TiO'),
        'V': get_species_molar_mass('V'),
        'V+': get_species_molar_mass('V') - get_species_molar_mass('e-'),
        'VO': get_species_molar_mass('VO'),
        'VO_Plez': get_species_molar_mass('VO')
    }

    # Loading
    print("Loading default files names...")
    line_paths = np.loadtxt(os.path.join(path_input_data, 'opa_input_files', 'opa_filenames.txt'), dtype=str)

    print("Loading default PT grid...")

    opacities_temperature_profile_grid = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'opa_PT_grid.dat')
    )

    opacities_temperature_profile_grid = np.flip(opacities_temperature_profile_grid, axis=1)

    opacities_temperatures = np.unique(opacities_temperature_profile_grid[:, 0])
    opacities_pressures = np.unique(opacities_temperature_profile_grid[:, 1])  # grid is already in bar

    print("Loading correlated-k g grid...")
    buffer = np.genfromtxt(
        os.path.join(path_input_data, 'opa_input_files', 'g_comb_grid.dat')
    )
    g_gauss = np.array(buffer[:, 0], dtype='d', order='F')
    weights_gauss = np.array(buffer[:, 1], dtype='d', order='F')

    input_directory = os.path.join(path_input_data, 'opacities', 'lines', 'corr_k')

    if output_directory is None:
        output_directory_ref = input_directory
    else:
        output_directory_ref = copy.deepcopy(output_directory)

    for f in os.scandir(input_directory):
        if f.is_dir():
            directory = f.path

            species = directory.rsplit(os.path.sep, 1)[1]
            not_in_dict = False

            # Check information availability
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
                print(f" Skipping species '{species}' due to missing species in supplementary info dict...")
                continue

            if doi_dict[species] is None:
                print(f"Skipping species '{species}' due species already in HDF5...")
                continue

            # Check output directory
            output_directory = os.path.join(output_directory_ref, species)

            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

            # Check HDF5 file existence
            hdf5_opacity_file = os.path.join(output_directory, species + '.ktable.petitRADTRANS.h5')

            if os.path.isfile(hdf5_opacity_file) and not rewrite:
                __print_skipping_message(hdf5_opacity_file)
                continue

            # Read dat file
            print(f"Converting opacities in '{directory}'...")

            custom_pt_grid_file = os.path.join(directory, 'PTpaths.ls')
            has_custom_grid = False
            opacities_temperature_profile_grid_ = None

            if os.path.isfile(custom_pt_grid_file):
                print(" Found custom PT grid")
                has_custom_grid = True

                # _sort_opa_pt_grid converts bar into cgs
                custom_grid_data = Radtrans._sort_pressure_temperature_grid(custom_pt_grid_file)

                opacities_temperature_profile_grid_ = custom_grid_data[0]
                opacities_temperatures_ = np.unique(opacities_temperature_profile_grid_[:, 0])
                opacities_pressures_ = np.unique(opacities_temperature_profile_grid_[:, 1])
                opacities_pressures_ *= 1e-6  # cgs to bar
                line_paths_ = custom_grid_data[1]

                for i, line_path in enumerate(line_paths_):
                    line_paths_[i] = line_path

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

            custom_file_names = ''

            if has_custom_grid:
                size_tp_grid = opacities_temperature_profile_grid_.shape[0]

                for i_TP in range(size_tp_grid):
                    custom_file_names = custom_file_names + line_paths_[i_TP] + ':'
            else:
                size_tp_grid = opacities_pressures_.size * opacities_temperatures_.size

            # Convert units and shape
            _n_frequencies, _n_g = finput.load_frequencies_g_sizes(path_input_data, species)
            _frequencies, frequency_bins_edges = finput.load_frequencies(path_input_data, species, _n_frequencies)
            wavenumbers = _frequencies[::-1] / cst.c  # Hz to cm-1
            wavenumbers_bins_edges = frequency_bins_edges[::-1] / cst.c  # Hz to cm-1
            wavelengths = 1 / wavenumbers

            opacities = finput.load_line_opacity_grid(
                path_input_data,
                species + ':',
                _n_frequencies,
                _n_g,
                1,
                size_tp_grid,
                'c-k',
                1,  # lbl start index, unused in this case
                has_custom_grid,
                custom_file_names
            )
            # Opacities are divided by isotopic ratio in loading function, there is no need to store it or use it
            cross_sections = opacities * molmass_dict[species] * cst.amu  # opacities to cross-sections

            print(" Reshaping...")
            # Exo-Mol axis order (pressures, temperatures, wavenumbers, g)
            cross_sections = cross_sections[:, :, 0, :]  # get rid of useless dimension
            cross_sections = cross_sections.reshape(
                (_n_g, _n_frequencies, opacities_temperatures_.size, opacities_pressures_.size)
            )
            cross_sections = np.swapaxes(cross_sections, 0, -1)
            cross_sections = np.swapaxes(cross_sections, 1, -2)
            cross_sections = cross_sections[:, :, ::-1, :]  # match the wavenumber order

            # Write converted file
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
                    name='bin_centers',
                    data=wavenumbers
                )
                dataset.attrs['long_name'] = 'Centers of the wavenumber bins'
                dataset.attrs['units'] = 'cm^-1'

                dataset = fh5.create_dataset(
                    name='bin_edges',
                    data=wavenumbers_bins_edges
                )
                dataset.attrs['long_name'] = 'Separations between the wavenumber bins'
                dataset.attrs['units'] = 'cm^-1'

                dataset = fh5.create_dataset(
                    name='kcoeff',
                    data=cross_sections
                )
                dataset.attrs['long_name'] = ('Table of the k-coefficients with axes '
                                              '(pressure, temperature, wavenumber, g space)')
                dataset.attrs['units'] = 'cm^2/molecule'

                dataset = fh5.create_dataset(
                    name='method',
                    data='petit_samples'
                )
                dataset.attrs['long_name'] = 'Name of the method used to sample g-space'

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
                    name='ngauss',
                    data=_n_g
                )
                dataset.attrs['long_name'] = 'Number of points used to sample the g-space'

                dataset = fh5.create_dataset(
                    name='p',
                    data=opacities_pressures_
                )
                dataset.attrs['long_name'] = 'Pressure grid'
                dataset.attrs['units'] = 'bar'

                dataset = fh5.create_dataset(
                    name='samples',
                    data=g_gauss
                )
                dataset.attrs['long_name'] = 'Abscissas used to sample the k-coefficients in g-space'

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
                    name='weights',
                    data=weights_gauss
                )
                dataset.attrs['long_name'] = 'Weights used in the g-space quadrature'

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

    print("Successfully converted correlated-k line opacities")


def line_by_line_opacities_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(),
                                  rewrite=False, output_directory=None):
    """Using ExoMol units for HDF5 files."""
    from petitRADTRANS.fortran_inputs import fortran_inputs as finput
    from petitRADTRANS.radtrans import Radtrans

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
        'Al': get_species_molar_mass('Al'),
        'B': get_species_molar_mass('B'),
        'Be': get_species_molar_mass('Be'),
        'C2H2_main_iso': get_species_molar_mass('C2H2'),
        'Ca': get_species_molar_mass('Ca'),
        'Ca+': get_species_molar_mass('Ca') - get_species_molar_mass('e-'),
        'CaH': get_species_molar_mass('CaH'),
        'CH4_212': get_species_molar_mass('CH3D'),
        'CH4_hargreaves_main_iso': get_species_molar_mass('CH4'),
        'CH4_Hargreaves_main_iso': get_species_molar_mass('CH4'),
        'CH4_main_iso': get_species_molar_mass('CH4'),
        'CO2_main_iso': get_species_molar_mass('CO2'),
        'CO_27': get_species_molar_mass('12C') + get_species_molar_mass('17O'),
        'CO_28': get_species_molar_mass('12C') + get_species_molar_mass('18O'),
        'CO_36': get_species_molar_mass('13C') + get_species_molar_mass('16O'),
        'CO_37': get_species_molar_mass('13C') + get_species_molar_mass('17O'),
        'CO_38': get_species_molar_mass('13C') + get_species_molar_mass('18O'),
        'CO_all_iso': get_species_molar_mass('CO_all_iso'),
        'CO_main_iso': get_species_molar_mass('CO'),
        'Cr': get_species_molar_mass('Cr'),
        'Fe': get_species_molar_mass('Fe'),
        'Fe+': get_species_molar_mass('Fe') - get_species_molar_mass('e-'),
        'FeH_main_iso': get_species_molar_mass('FeH'),
        'H2_12': get_species_molar_mass('HD'),
        'H2_main_iso': get_species_molar_mass('H2'),
        'H2O_162': get_species_molar_mass('1H') + get_species_molar_mass('16O') + get_species_molar_mass('2H'),
        'H2O_171': get_species_molar_mass('1H') + get_species_molar_mass('17O') + get_species_molar_mass('1H'),
        'H2O_172': get_species_molar_mass('1H') + get_species_molar_mass('17O') + get_species_molar_mass('2H'),
        'H2O_181': get_species_molar_mass('1H') + get_species_molar_mass('18O') + get_species_molar_mass('1H'),
        'H2O_182': get_species_molar_mass('1H') + get_species_molar_mass('18O') + get_species_molar_mass('2H'),
        'H2O_main_iso': get_species_molar_mass('1H') + get_species_molar_mass('16O') + get_species_molar_mass('2H'),
        'H2O_pokazatel_main_iso': get_species_molar_mass('H2O'),
        'H2S_main_iso': get_species_molar_mass('H2S'),
        'HCN_main_iso': get_species_molar_mass('HCN'),
        'K': get_species_molar_mass('K'),
        'K_allard_cold': get_species_molar_mass('K'),
        'Li': get_species_molar_mass('Li'),
        'Mg': get_species_molar_mass('Mg'),
        'Mg+': get_species_molar_mass('Mg') - get_species_molar_mass('e-'),
        'N': get_species_molar_mass('N'),
        'Na_allard': get_species_molar_mass('Na'),
        'Na_allard_new': get_species_molar_mass('Na'),
        'NH3_main_iso': get_species_molar_mass('NH3'),
        'O3_main_iso': get_species_molar_mass('O3'),
        'OH_main_iso': get_species_molar_mass('OH'),
        'PH3_main_iso': get_species_molar_mass('PH3'),
        'Si': get_species_molar_mass('Si'),
        'SiO_main_iso': get_species_molar_mass('SiO'),
        'SiO_main_iso_new_incl_UV': get_species_molar_mass('SiO'),
        'Ti': get_species_molar_mass('Ti'),
        'TiO_46_Exomol_McKemmish': get_species_molar_mass('46Ti') + get_species_molar_mass('16O'),
        'TiO_46_Plez': get_species_molar_mass('46Ti') + get_species_molar_mass('16O'),
        'TiO_47_Exomol_McKemmish': get_species_molar_mass('47Ti') + get_species_molar_mass('16O'),
        'TiO_47_Plez': get_species_molar_mass('47Ti') + get_species_molar_mass('16O'),
        'TiO_48_Exomol_McKemmish': get_species_molar_mass('48Ti') + get_species_molar_mass('16O'),
        'TiO_48_Plez': get_species_molar_mass('48Ti') + get_species_molar_mass('16O'),
        'TiO_49_Exomol_McKemmish': get_species_molar_mass('49Ti') + get_species_molar_mass('16O'),
        'TiO_49_Plez': get_species_molar_mass('49Ti') + get_species_molar_mass('16O'),
        'TiO_50_Exomol_McKemmish': get_species_molar_mass('50Ti') + get_species_molar_mass('16O'),
        'TiO_50_Plez': get_species_molar_mass('50Ti') + get_species_molar_mass('16O'),
        'TiO_all_iso_Plez': get_species_molar_mass('TiO_all_iso'),
        'TiO_all_iso_exo': get_species_molar_mass('TiO_all_iso'),
        'V': get_species_molar_mass('V'),
        'V+': get_species_molar_mass('V') - get_species_molar_mass('e-'),
        'VO': get_species_molar_mass('VO'),
        'VO_ExoMol_McKemmish': get_species_molar_mass('VO'),
        'VO_ExoMol_Specific_Transitions': get_species_molar_mass('VO'),
        'Y': get_species_molar_mass('Y')
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

    # Conversion
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

            # Check information availability
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
                print(f" Skipping species '{species}' due to missing species in supplementary info dict...")
                continue

            # Check output directory
            output_directory = os.path.join(output_directory_ref)

            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

            # Check HDF5 file existence
            hdf5_opacity_file = os.path.join(output_directory, species + '.otable.petitRADTRANS.h5')

            if os.path.isfile(hdf5_opacity_file) and not rewrite:
                __print_skipping_message(hdf5_opacity_file)
                continue

            # Read dat file
            print(f"Converting opacities in '{directory}'...")

            custom_pt_grid_file = os.path.join(directory, 'PTpaths.ls')

            if os.path.isfile(custom_pt_grid_file):
                print(" Found custom PT grid")

                # _sort_opa_pt_grid converts bar into cgs
                custom_grid_data = Radtrans._sort_pressure_temperature_grid(custom_pt_grid_file)

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

            # Convert units and shape
            if os.path.isfile(molparam_file):
                print(" Loading isotopic ratio...")
                with open(molparam_file, 'r') as f2:
                    isotopic_ratio = float(f2.readlines()[-1])
            else:
                raise FileNotFoundError(f"file '{molparam_file}' not found: unable to load isotopic ratio")

            n_lines = finput.count_file_line_number(os.path.join(directory, 'wlen.dat'))
            wavelengths = finput.load_all_line_by_line_opacities(os.path.join(directory, 'wlen.dat'), n_lines)
            wavenumbers = 1 / wavelengths[::-1]  # cm to cm-1

            opacities = np.zeros((line_paths_.size, wavelengths.size))

            for i, line_path in enumerate(line_paths_):
                if not os.path.isfile(line_path):
                    raise FileNotFoundError(f"file '{line_path}' does not exists")

                print(f" Loading file '{line_path}' ({i + 1}/{line_paths_.size})...")

                opacities[i] = finput.load_all_line_by_line_opacities(line_path, n_lines)

            print(" Reshaping...")
            opacities = opacities.reshape((opacities_temperatures_.size, opacities_pressures_.size, wavelengths.size))
            # Exo-Mol axis order (pressures, temperatures, wavenumbers, g)
            opacities = np.moveaxis(opacities, 0, 1)
            opacities = opacities[:, :, ::-1]  # match the wavenumber order

            # Write converted file
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

    print("Successfully converted line-by-line line opacities")


def phoenix_spec_dat2h5(path_input_data=petitradtrans_config_parser.get_input_data_path(), rewrite=False):
    """
    Convert a PHOENIX stellar spectrum in .dat format to HDF5 format.
    """
    # Load the stellar parameters
    path = os.path.join(path_input_data, 'stellar_specs')
    hdf5_file = os.path.join(path, "stellar_spectra.h5")

    if os.path.isfile(hdf5_file) and not rewrite:
        __print_skipping_message(hdf5_file)
        return

    dat_file = os.path.join(path, 'stellar_params.dat')

    if not os.path.isfile(dat_file):
        __print_missing_data_file_message(
            'stellar spectrum', 'stellar_params.dat', dat_file.rsplit(os.path.sep, 1)[0]
        )
        return

    description = np.genfromtxt(os.path.join(path, 'stellar_params.dat'))

    # Initialize the grids
    log_temp_grid = description[:, 0]
    star_rad_grid = description[:, 1]

    # Load the corresponding numbered spectral files
    spec_dats = []

    for spec_num in range(len(log_temp_grid)):
        spec_dats.append(np.genfromtxt(os.path.join(
            path_input_data,
            'spec_' + str(int(spec_num)).zfill(2) + '.dat')
        ))

    # Write the HDF5 file
    with h5py.File(hdf5_file) as f:
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

    print("Successfully converted stellar spectra")


def rebin_ck_line_opacities(radtrans, resolution, path='', species=None, species_molar_masses=None, rewrite=False):
    import exo_k
    import petitRADTRANS.physical_constants as cst
    from petitRADTRANS.retrieval.data import Data

    if species is None:
        species = []

    # Define own wavenumber grid, make sure that log spacing is constant everywhere
    n_spectral_points = int(
        resolution * np.log(radtrans.wavelengths_boundaries[1] / radtrans.wavelengths_boundaries[0]) + 1
    )
    wavenumber_grid = np.logspace(
        np.log10(1 / radtrans.wavelengths_boundaries[1] * 1e4),
        np.log10(1 / radtrans.wavelengths_boundaries[0] * 1e4),
        n_spectral_points
    )

    wavenumbers = radtrans.frequencies[::-1] / cst.c  # Hz to cm-1
    wavenumbers_bins_edges = radtrans.frequency_bins_edges[::-1] / cst.c  # Hz to cm-1

    # Do the rebinning, loop through species
    for s in species:
        # Output files
        base_name = Data.get_ck_line_species_directory(
            species=s,
            model_resolution=resolution
        )
        output_directory = os.path.join(path, base_name)
        hdf5_opacity_file = os.path.join(output_directory, base_name + '.ktable.petitRADTRANS.h5')
        hdf5_opacity_file_tmp = os.path.join(output_directory, base_name + '_tmp.ktable.petitRADTRANS.h5')

        if os.path.isfile(hdf5_opacity_file) and not rewrite:
            print(f"Skipping already re-binned species '{s}' (file '{hdf5_opacity_file}' already exists)...")
            continue

        print(f"Rebinning species {s}...")

        # Mass to go from opacities to cross-sections
        cross_sections = (
                copy.copy(radtrans.lines_loaded_opacities['opacity_grid'][s])
                * species_molar_masses[s.split('_')[0]] * cst.amu
        )

        print(" Reshaping...")
        # Exo-Mol axis order (pressures, temperatures, wavenumbers, g)
        cross_sections = cross_sections[:, ::-1, :]
        cross_sections = np.swapaxes(cross_sections, 2, 0)
        cross_sections = cross_sections.reshape((
            radtrans.lines_loaded_opacities['temperature_grid_size'][s],
            radtrans.lines_loaded_opacities['pressure_grid_size'][s],
            radtrans.frequencies.size,
            len(radtrans.lines_loaded_opacities['weights_gauss'])
        ))
        cross_sections = np.swapaxes(cross_sections, 1, 0)
        cross_sections[cross_sections < 1e-60] = 1e-60

        print(f" Writing temporary file in '{output_directory}'...")
        os.makedirs(output_directory, exist_ok=True)

        # Create hdf5 file that Exo-k can read
        with h5py.File(hdf5_opacity_file_tmp, 'w') as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                data=['None']  # use list to avoid exo_k error
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'

            dataset = fh5.create_dataset(
                name='Date_ID',
                data=f'petitRADTRANS-v{petitRADTRANS.__version__}_{datetime.datetime.utcnow().isoformat()}'
            )
            dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                         'at which the table has been created, ' \
                                         'along with the version of petitRADTRANS'

            dataset = fh5.create_dataset(
                name='bin_centers',
                data=wavenumbers
            )
            dataset.attrs['long_name'] = 'Centers of the wavenumber bins'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='bin_edges',
                data=wavenumbers_bins_edges
            )
            dataset.attrs['long_name'] = 'Separations between the wavenumber bins'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='kcoeff',
                data=cross_sections
            )
            dataset.attrs['long_name'] = ('Table of the k-coefficients with axes '
                                          '(pressure, temperature, wavenumber, g space)')
            dataset.attrs['units'] = 'cm^2/molecule'

            dataset = fh5.create_dataset(
                name='method',
                data=['petit_samples']
            )
            dataset.attrs['long_name'] = 'Name of the method used to sample g-space'

            dataset = fh5.create_dataset(
                name='mol_mass',
                data=species_molar_masses[s.split('_', 1)[0]]
            )
            dataset.attrs['long_name'] = 'Mass of the species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='mol_name',
                data=s.split('_', 1)[0]
            )
            dataset.attrs['long_name'] = 'Name of the species described'

            dataset = fh5.create_dataset(
                name='ngauss',
                data=len(radtrans.lines_loaded_opacities['weights_gauss'])
            )
            dataset.attrs['long_name'] = 'Number of points used to sample the g-space'

            dataset = fh5.create_dataset(
                name='p',
                data=radtrans.lines_loaded_opacities['temperature_pressure_grid'][s][
                                       :radtrans.lines_loaded_opacities['pressure_grid_size'][s], 1] * 1e-6
            )
            dataset.attrs['long_name'] = 'Pressure grid'
            dataset.attrs['units'] = 'bar'

            dataset = fh5.create_dataset(
                name='samples',
                data=radtrans.lines_loaded_opacities['g_gauss']
            )
            dataset.attrs['long_name'] = 'Abscissas used to sample the k-coefficients in g-space'

            dataset = fh5.create_dataset(
                name='t',
                data=radtrans.lines_loaded_opacities['temperature_pressure_grid'][s][
                                       ::radtrans.lines_loaded_opacities['pressure_grid_size'][s], 0]
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
                name='weights',
                data=radtrans.lines_loaded_opacities['weights_gauss']
            )
            dataset.attrs['long_name'] = 'Weights used in the g-space quadrature'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=[
                    np.min(cst.c / radtrans.frequency_bins_edges * 1e4),
                    np.max(cst.c / radtrans.frequency_bins_edges * 1e4)
                ]
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumbers.min(), wavenumbers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'

        # Use Exo-k to rebin to low-res
        print(f" Binning down to R = '{resolution}'...", end=' ')
        tab = exo_k.Ktable(filename=hdf5_opacity_file_tmp)
        tab.bin_down(wavenumber_grid)

        print(f" Writing binned down file '{hdf5_opacity_file}'...")
        tab.write_hdf5(hdf5_opacity_file)

        print(" Removing temporary file...")
        os.remove(hdf5_opacity_file_tmp)

        print(f" Successfully binned down k-table of species '{s}' \n")

    print("Successfully binned down all k-tables\n")


def convert_all(path_input_data=petitradtrans_config_parser.get_input_data_path(), rewrite=False):
    print("Starting all conversions...")

    print("Stellar spectra...")
    phoenix_spec_dat2h5(path_input_data=path_input_data, rewrite=rewrite)

    print("CIA...")
    continuum_cia_dat2h5(path_input_data=path_input_data, rewrite=rewrite)

    print("Line-by-line opacities...")
    line_by_line_opacities_dat2h5(path_input_data=path_input_data, rewrite=rewrite)

    print("Clouds continuum...")
    continuum_clouds_opacities_dat2h5(path_input_data=path_input_data, rewrite=rewrite)

    print("Chemical tables...")
    chemical_table_dat2h5(path_input_data=path_input_data, rewrite=rewrite)

    print("Successfully converted all .dat files into HDF5")
