import copy
import os
import re
import warnings

from molmass import Formula

from petitRADTRANS.cli.prt_cli import download_input_data, get_keeper_files_url_paths
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.utils import LockedDict, user_input


def __build_cia_aliases_dict():
    cias = _get_base_cia_names()
    cias.update({
        "H2--H2": ['H2', 'H2'],
        "H2--He": ['H2', 'He'],
        "H2O--H2O": ['H2O', 'H2O'],
        "H2O--N2": ['H2O', 'N2'],
        "N2--H2": ['N2', 'H2'],
        "N2--He": ['N2', 'He'],
        "N2--N2": ['N2', 'N2'],
        "O2--O2": ['O2', 'O2'],
        "N2--O2": ['N2', 'O2'],
        "CO2--CO2": ['CO2', 'CO2'],
    })

    cia_aliases = {}

    for cia, species in cias.items():
        cia_aliases[cia] = [''.join(species), '-'.join(species), '--'.join(species)]

        if species[0] != species[1]:
            species[0], species[1] = species[1], species[0]  # flip CIA
            cia_aliases[cia].extend([''.join(species), '-'.join(species), '--'.join(species)])

    return cia_aliases


def __default_file_selection(files, full_path, sub_path):
    files_str = [f" {i + 1}: {file}" for i, file in enumerate(files)]
    files_str = "\n".join(files_str)

    introduction_message = (
        f"More than one file detected in '{full_path}', and no default file set for this path "
        f"in petitRADTRANS' configuration\n"
        f"Please select one of the files in the list below by typing the corresponding integer:\n"
        f"{files_str}"
    )

    new_default_file = user_input(
        introduction_message=introduction_message,
        input_message=f"Select which file to set as the default file for '{sub_path}'",
        failure_message=f"failure to enter new default file for '{sub_path}'",
        cancel_message="Cancelling default file selection...",
        mode='list',
        list_length=len(files)
    )

    if new_default_file is None:
        raise ValueError(f"no default file selected for path '{sub_path}'")

    new_default_file -= 1
    new_default_file = files[new_default_file]

    return new_default_file


def __recursive_merge_contiguous_isotopes(isotope_groups, i, index_merge=None):
    if index_merge is None:
        index_merge = i - 1

    if i > 0:
        element, number = isotope_groups[i]
        element_merge, number_merge = isotope_groups[index_merge]

        if element == element_merge:
            if number_merge == '':
                number_merge = 1
            elif number_merge == 0:
                if index_merge > 0:
                    isotope_groups = __recursive_merge_contiguous_isotopes(isotope_groups, i, index_merge - 1)
                    return isotope_groups
                else:
                    return isotope_groups
            else:
                number_merge = int(number_merge)

            if number == '':
                number = 1
            else:
                number = int(number)

            isotope_groups[index_merge][1] = number_merge + number
            isotope_groups[i][1] = 0

    return isotope_groups


def _get_base_cia_names():
    return LockedDict.build_and_lock({
        'H2--H2': 'H2--H2-NatAbund__BoRi.R831_0.6-250mu',
        'H2--He': 'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu',
        'H2O--H2O': 'H2O--H2O-NatAbund',
        'H2O--N2': 'H2O--N2-NatAbund',
        'N2--H2': 'N2--H2-NatAbund.DeltaWavenumber1_5.3-909mu',
        'N2--He': 'N2--He-NatAbund.DeltaWavenumber1_10-909mu',
        'N2--N2': 'N2--N2-NatAbund.DeltaWavelength1e-6_2-100mu',
        'O2--O2': 'O2--O2-NatAbund.DeltaWavelength1e-6_0.34-8.7mu',
        'N2--O2': 'N2--O2-NatAbund.DeltaWavelength1e-6_0.72-5.4mu',
        'CO2--CO2': 'CO2--CO2-NatAbund.DeltaWavelength1e-6_3-100mu',
    })


def _get_base_cloud_names():
    return LockedDict.build_and_lock({
        'Al2O3(s)_crystalline': 'Al2-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Fe(s)_amorphous': 'Fe-NatAbund(s)_amorphous.R39_0.1-250mu',
        'Fe(s)_crystalline': 'Fe-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'H2O(s)_crystalline': 'H2-O-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'H2O(l)': 'H2-O-NatAbund(l).R39_0.1-250mu',
        'H2OSO4(l)': 'H2-O-S-O4-NatAbund(l).R39_0.1-250mu',
        'KCl(s)_crystalline': 'K-Cl-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Mg2SiO4(s)_amorphous': 'Mg2-Si-O4-NatAbund(s)_amorphous.R39_0.1-250mu',
        'Mg2SiO4(s)_crystalline': 'Mg2-Si-O4-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Mg05Fe05SiO3(s)_amorphous': 'Mg05-Fe05-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgAl2O4(s)_crystalline': 'Mg-Al2-O4-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgFeSiO4(s)_amorphous': 'Mg-Fe-Si-O4-NatAbund(s)_amorphous.R39_0.2-250mu',
        'MgSiO3(s)_amorphous': 'Mg-Si-O3-NatAbund(s)_amorphous.R39_0.1-250mu',
        'MgSiO3(s)_crystalline': 'Mg-Si-O3-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'Na2S(s)_crystalline': 'Na2-S-NatAbund(s)_crystalline_000.R39_0.1-250mu',
        'SiC(s)_crystalline': 'Si-C-NatAbund(s)_crystalline_000.R39_0.1-250mu'
    })


def _get_base_correlated_k_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock({
        'Al': '27Al__Kurucz.R1000_0.1-250mu',
        'Al+': '27Al_p__Kurucz.R1000_0.1-250mu',
        'AlH': '27Al-1H__AlHambra.R1000_0.3-50mu',
        'AlO': '27Al-16O__ATP.R1000_0.3-50mu',
        'C2H2': '12C2-1H2__aCeTY.R1000_0.3-50mu',
        'C2H4': '12C2-1H4__MaYTY.R1000_0.3-50mu',
        'Ca': '40Ca__Kurucz.R1000_0.1-250mu',
        'Ca+': '40Ca_p__Kurucz.R1000_0.1-250mu',
        'CaH': '40Ca-1H__MoLLIST.R1000_0.3-50mu',
        'CH4': '12C-1H4__YT34to10.R1000_0.3-50mu',
        'CO': '12C-16O__HITEMP.R1000_0.1-250mu',
        '13CO': '13C-16O__HITEMP.R1000_0.1-250mu',
        'CO_all_iso': 'C-O-NatAbund__HITEMP.R1000_0.1-250mu',
        'CO2': '12C-16O2__UCL-4000.R1000_0.3-50mu',
        'CrH': '52Cr-1H__MoLLIST.R1000_0.3-50mu',
        'Fe': '56Fe__Kurucz.R1000_0.1-250mu',
        'Fe+': '56Fe_p__Kurucz.R1000_0.1-250mu',
        'FeH': '56Fe-1H__MoLLIST.R1000_0.3-50mu',
        'H2O': '1H2-16O__HITEMP.R1000_0.1-250mu',
        'H2S': '1H2-32S__AYT2.R1000_0.3-50mu',
        'HCN': '1H-12C-14N__Harris.R1000_0.3-50mu',
        'K': '39K__Allard.R1000_0.1-250mu',
        'Li': '7Li__Kurucz.R1000_0.1-250mu',
        'Mg': '24Mg__Kurucz.R1000_0.1-250mu',
        'Mg+': '24Mg_p__Kurucz.R1000_0.1-250mu',
        'MgH': '24Mg-1H__MoLLIST.R1000_0.3-50mu',
        'MgO': '24Mg-16O__LiTY.R1000_0.3-50mu',
        'Na': '23Na__Allard.R1000_0.1-250mu',
        'NaH': '23Na-1H__Rivlin.R1000_0.3-50mu',
        'NH3': '14N-1H3__CoYuTe.R1000_0.3-50mu',
        'O': '16O__Kurucz.R1000_0.1-250mu',
        'O+': '16O_p__Kurucz.R1000_0.1-250mu',  # TODO not in the docs
        'O2': '16O2__HITRAN.R1000_0.3-50mu',
        'O3': '16O3__HITRAN.R1000_0.1-250mu',
        'OH': '16O-1H__HITEMP.R1000_0.3-50mu',
        'PH3': '31P-1H3__SAlTY.R1000_0.3-50mu',
        'SH': '32S-1H__GYT.R1000_0.3-50mu',
        'Si': '28Si__Kurucz.R1000_0.1-250mu',
        'Si+': '28Si_p__Kurucz.R1000_0.1-250mu',
        'SiO': '28Si-16O__EBJT.R1000_0.3-50mu',
        'SiO2': '28Si-16O2__OYT3.R1000_0.3-50mu',
        'Ti': '48Ti__Kurucz.R1000_0.1-250mu',
        'Ti+': '48Ti_p__Kurucz.R1000_0.1-250mu',
        '48TiO': '48Ti-16O__Plez.R1000_0.1-250mu',
        'TiO_all_iso': 'Ti-O-NatAbund__Plez.R1000_0.1-250mu',
        'V': '51V__Kurucz.R1000_0.1-250mu',
        'V+': '51V_p__Kurucz.R1000_0.1-250mu',
        'VO': '51V-16O__VOMYT.R1000_0.3-50mu'
    })


def _get_base_line_by_line_names():
    """Only used for file conversion from pRT2 to pRT3."""
    return LockedDict.build_and_lock({
        'Al': '27Al__Kurucz.R1e6_0.3-28mu',
        'B': '11B__Kurucz.R1e6_0.3-28mu',
        'Be': '9Be__Kurucz.R1e6_0.3-28mu',
        'C2H2': '12C2-1H2__HITRAN.R1e6_0.3-28mu',
        'Ca': '40Ca__Kurucz.R1e6_0.3-28mu',
        'Ca+': '40Ca_p__Kurucz.R1e6_0.3-28mu',
        'CaH': '40Ca-1H__MoLLIST.R1e6_0.3-28mu',  # TODO not in docs
        'CH3D': '12C-1H3-2H__HITRAN.R1e6_0.3-28mu',
        'CH4': '12C-1H4__Hargreaves.R1e6_0.3-28mu',
        'CO2': '12C-16O2__HITEMP.R1e6_0.3-28mu',
        'C-17O': '12C-17O__HITRAN.R1e6_0.3-28mu',
        'C-18O': '12C-18O__HITRAN.R1e6_0.3-28mu',
        '13CO': '13C-16O__HITRAN.R1e6_0.3-28mu',
        '13C-17O': '13C-17O__HITRAN.R1e6_0.3-28mu',
        '13C-18O': '13C-18O__HITRAN.R1e6_0.3-28mu',
        'CO_all_iso': 'C-O-NatAbund__HITEMP.R1e6_0.3-28mu',
        'CO': '12C-16O__HITEMP.R1e6_0.3-28mu',
        'Cr': '52Cr__Kurucz.R1e6_0.3-28mu',
        'Fe': '56Fe__Kurucz.R1e6_0.3-28mu',
        'Fe+': '56Fe_p__Kurucz.R1e6_0.3-28mu',
        'FeH': '56Fe-1H__MoLLIST.R1e6_0.3-28mu',
        'HD': '1H-2H__HITRAN.R1e6_0.3-28mu',
        'H2': '1H2__HITRAN.R1e6_0.3-28mu',
        'HDO': '1H-2H-16O__HITEMP.R1e6_0.3-28mu',
        'H2-17O': '1H2-17O__HITEMP.R1e6_0.3-28mu',
        'HD-17O': '1H-2H-17O__HITEMP.R1e6_0.3-28mu',
        'H2-18O': '1H2-18O__HITEMP.R1e6_0.3-28mu',
        'HD-18O': '1H-2H-18O__HITEMP.R1e6_0.3-28mu',
        'H2O': '1H2-16O__HITEMP.R1e6_0.3-28mu',
        'H2S_main_iso': '1H2-32S__HITRAN.R1e6_0.3-28mu',
        'HCN_main_iso': '1H-12C-14N__Harris.R1e6_0.3-28mu',
        'K': '39K__Allard.R1e6_0.3-28mu',
        'Li': '7Li__Kurucz.R1e6_0.3-28mu',
        'Mg': '24Mg__Kurucz.R1e6_0.3-28mu',
        'Mg+': '24Mg_p__Kurucz.R1e6_0.3-28mu',
        'N': '14N__Kurucz.R1e6_0.3-28mu',
        'Na': '23Na__Allard.R1e6_0.3-28mu',
        'NH3': '14N-1H3__BYTe.R1e6_0.3-28mu',
        'O3': '16O3__HITRAN.R1e6_0.3-28mu',
        'OH': '16O-1H__HITEMP.R1e6_0.3-28mu',
        'PH3': '31P-1H3__SAlTY.R1e6_0.3-28mu',
        'Si': '28Si__Kurucz.R1e6_0.3-28mu',
        'SiO': '28Si-16O__EBJT.R1e6_0.3-28mu',
        'Ti': '48Ti__Kurucz.R1e6_0.3-28mu',
        '46TiO': '46Ti-16O__Plez.R1e6_0.3-28mu',
        '47TiO': '47Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO': '48Ti-16O__Plez.R1e6_0.3-28mu',
        '49TiO': '49Ti-16O__Plez.R1e6_0.3-28mu',
        '50TiO': '50Ti-16O__Plez.R1e6_0.3-28mu',
        'TiO_all_iso': 'Ti-O-NatAbund__Plez.R1e6_0.3-28mu',
        'V': '51V__Kurucz.R1e6_0.3-28mu',
        'V+': '51V_p__Kurucz.R1e6_0.3-28mu',
        'VO': '51V-16O__Plez.R1e6_0.3-28mu',
        'Y': '89Y__Kurucz.R1e6_0.3-28mu'
    })


def _get_input_file(path_input_data, sub_path, files=None, filename=None, expect_spectral_information=False,
                    find_all=False, display_other_files=False):
    full_path = os.path.join(path_input_data, sub_path)

    if files is None:
        files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]

    if len(files) == 0:  # no file in path, return empty list
        return []
    else:  # at least one file detected in path
        if filename is not None:  # check if one of the files matches the given filename
            matching_files = []
            resolution_filename = ''
            _filename = filename.split('.', 1)[0]

            if expect_spectral_information:
                resolution_filename, range_filename = _get_spectral_information(filename)

                # First pass, try to use default resolution
                for file in files:
                    _file, spectral_info = file.split('.', 1)

                    resolution_file, range_file = _get_spectral_information(spectral_info)

                    if resolution_file == '' or range_file == '':
                        warnings.warn(f"file '{file}' lack spectral information "
                                      f"(resolution: {resolution_file}, range: {range_file})")

                    _file = file.split('.', 1)[0]

                    if _filename in _file:
                        if resolution_filename != '':
                            range_match = False

                            if range_filename != '':
                                if range_filename == range_file:
                                    range_match = True
                            else:
                                range_match = True

                            if resolution_filename == resolution_file and range_match:
                                matching_files.append(file)
                        else:
                            if get_default_correlated_k_resolution() == resolution_file:
                                matching_files.append(file)
                            elif get_default_line_by_line_resolution() == resolution_file:
                                matching_files.append(file)
                            elif get_default_cloud_resolution() == resolution_file:
                                matching_files.append(file)

            # Second pass, take any matching file regardless of resolution
            if len(matching_files) == 0 and resolution_filename == '':
                for file in files:
                    _file = file.split('.', 1)[0]

                    if _filename in _file:
                        matching_files.append(file)

            if len(matching_files) == 0:
                if display_other_files:
                    files_str = "\n".join(files)
                    warnings.warn(f"no file matching name '{filename}' found in directory '{full_path}'\n"
                                  f"Available files are:\n"
                                  f"{files_str}")

                return []
            elif len(matching_files) == 1:
                return matching_files[0]
            elif find_all:
                return matching_files

        # No filename given and only one file is in path, return it
        if len(files) == 1:
            return files[0]

        # More than one file detected
        if sub_path in petitradtrans_config_parser['Default files']:  # check for a default file in configuration
            default_file = os.path.join(
                path_input_data,
                sub_path,
                petitradtrans_config_parser['Default files'][sub_path]
            )

            if os.path.isfile(default_file):
                return default_file
            else:
                raise FileNotFoundError(
                    f"no such file: '{default_file}'\n"
                    f"Update the 'Default file' entry for '{sub_path}' in petitRADTRANS' configuration by executing:\n"
                    f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
                    f">>> petitradtrans_config_parser.set_default_file(<new_default_file>)\n"
                    f"Or download the missing file."
                )
        else:  # make the user enter the default file
            new_default_file = __default_file_selection(
                files=files,
                full_path=full_path,
                sub_path=sub_path
            )

            petitradtrans_config_parser.set_default_file(
                file=os.path.join(full_path, new_default_file),
                path_input_data=path_input_data
            )

            return new_default_file


def _get_input_file_from_keeper(full_path, path_input_data=None, sub_path=None, filename=None,
                                expect_spectral_information=False, find_all=False,
                                ext='h5', timeout=3, url_input_data=None):
    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()

        if path_input_data not in full_path:
            raise ValueError(f"full path '{full_path}' not within default input_data path '{path_input_data}'\n "
                             f"Set the path_input_data argument in accordance with full_path, "
                             f"or correct full_path")

    if sub_path is None:
        sub_path = full_path.split(path_input_data, 1)[1]

    url_paths = get_keeper_files_url_paths(
        path=full_path,
        ext=ext,
        timeout=timeout,
        path_input_data=path_input_data,
        url_input_data=url_input_data
    )

    matches = _get_input_file(
        path_input_data=path_input_data,
        sub_path=sub_path,
        files=list(url_paths.keys()),
        filename=filename,
        expect_spectral_information=expect_spectral_information,
        find_all=find_all,
        display_other_files=True
    )

    if len(matches) == 0 and not isinstance(matches, str):
        _ = _get_input_file(
            path_input_data=path_input_data,
            sub_path=sub_path,
            files=None,
            filename=filename,
            expect_spectral_information=expect_spectral_information,
            find_all=find_all,
            display_other_files=True
        )
    elif len(matches) == 1 or isinstance(matches, str):
        download_input_data(
            destination=os.path.join(full_path, matches),
            source=url_paths[matches],
            rewrite=False,
            path_input_data=path_input_data,
            url_input_data=url_input_data
        )
    else:
        files_str = [f" {i + 1}: {file}" for i, file in enumerate(matches)]
        files_str = "\n".join(files_str)

        introduction_message = (
            f"Multiple matching files found in the Keeper library, and no default file set for this path "
            f"in petitRADTRANS' configuration\n"
            f"List of matching files:\n"
            f"{files_str}"
        )

        download_all = user_input(
            introduction_message=introduction_message,
            input_message=f"Download all of the {len(matches)} matching files?",
            failure_message="unclear answer",
            cancel_message="Cancelling...",
            mode='y/n',
            list_length=len(url_paths)
        )

        if download_all is None:
            raise ValueError("Keeper file download cancelled")

        if download_all:
            for match in matches:
                download_input_data(
                    destination=os.path.join(full_path, match),
                    source=url_paths[match],
                    rewrite=False,
                    path_input_data=path_input_data,
                    url_input_data=url_input_data
                )
        else:
            new_default_file = __default_file_selection(
                files=list(url_paths.keys()),
                full_path="the Keeper library",
                sub_path=sub_path
            )

            petitradtrans_config_parser.set_default_file(
                file=os.path.join(full_path, new_default_file),
                path_input_data=path_input_data
            )

            download_input_data(
                destination=os.path.join(full_path, new_default_file),
                source=url_paths[new_default_file],
                rewrite=False,
                path_input_data=path_input_data,
                url_input_data=url_input_data
            )

    return matches


def _get_spectral_information(filename):
    spectral_info_pattern = (
        r'('
        r'('
        r'(R|DeltaWavenumber|DeltaWavelength)'  # resolution type
        r'\d{1,9}'  # resolution value
        r'(e([+|-])?\d{1,3})?'
        r')'  # manage exponent notation
        r'(_\d+(\.\d+)?-\d+(\.\d+)?mu)?'  # wavelength range
        r')'
    )

    spectral_info = re.findall(spectral_info_pattern, filename)

    if len(spectral_info) == 0:
        resolution_filename = ''
        range_filename = ''
    elif len(spectral_info) == 1:
        spectral_info = spectral_info[0]
        range_filename = spectral_info[5][1:]  # remove starting '_'
        resolution_filename = spectral_info[1]
    else:
        raise ValueError(f"found multiple spectral information pattern in file '{filename}' ({spectral_info})")

    return resolution_filename, range_filename


def _has_isotope(string):
    if (len(re.findall(r'(\d{1,3})?([A-Z][a-z]?)(\d{0,3})?-(\d{1,3})([A-Z][a-z]?)', string)) > 0
            or len(re.findall(r'^(\d{1,3})([A-Z][a-z]?)(\d{0,3})?', string)) > 0):
        return True
    else:
        return False


def _merge_contiguous_isotopes(species, isotope_separator):
    isotope_groups = re.findall(r'([A-Z][a-z]?)(\d{1,3})?', species)
    isotope_groups = [list(isotope_group) for isotope_group in isotope_groups]

    # Merge the isotopes, numbers of contiguous isotopes are converted to int, merged groups numbers are set to 0
    for i in range(len(isotope_groups)):
        isotope_groups = __recursive_merge_contiguous_isotopes(isotope_groups, i)

    # Convert back numbers from int to str
    for i in range(len(isotope_groups)):
        isotope_groups[i][1] = str(isotope_groups[i][1])

    # Rebuild the species with merged isotopes
    return isotope_separator.join([
        isotope_separator.join(isotope_group)
        for isotope_group in isotope_groups
        if isotope_group[1] != '0'  # ignore groups that has been merged
    ])


def _rebuild_isotope_numbers(species, mode='add'):
    """Add or remove isotope numbers from a species.
    Note that using improper isotope separation can lead to incorrect results (e.g. H218O -> 1H218-16O).

    Args:
        species:
            Species name. Can also be a species collision (e.g. H2--He).
        mode:
            Can be 'add' or 'remove'.
            In 'add' mode, add the isotope number of each of the species will be added to the species name, and each
            isotope is separated with a '-'. By default, the main isotope number is used. If partial isotope
            information is provided (e.g. 13C2H2, H2-18O, ...), the information is used (e.g. 13C2-1H2, 1H2-18O, ...)
            In 'remove' mode, remove all isotope numbers (e.g. 13C2-1H2 -> C2H2).

    Returns:
        The species name with added or removed isotope information.
    """
    # Set isotope explicit separator
    if mode == 'add':
        isotope_separator = '-'
    elif mode == 'remove':
        isotope_separator = ''  # no separator in remove mode
    else:
        raise ValueError(f"iter isotopes mode must be 'add'|'remove', but was '{mode}'")

    species_pattern = r'(\d{1,3})?([A-Z][a-z]?|e)(\d{1,3})?'  # isotope number, element symbol, stoichiometric number

    # Handle natural abundance case
    if '-NatAbund' in species:
        if mode == 'add':
            species = species.rsplit('-NatAbund', 1)[0]
            _species = species.split('--')  # CIA case

            for i, __species in enumerate(_species):  # for each colliding species
                matches = re.findall(species_pattern, __species)

                __species = [group[1] + group[2] for group in matches]
                _species[i] = isotope_separator.join(__species)

            species = '--'.join(_species)

            return species + '-NatAbund'
        elif mode == 'remove':
            return species.replace('-NatAbund', '')
        else:
            raise ValueError(f"iter isotopes mode must be 'add'|'remove', but was '{mode}'")

    # Handle regular case
    _species = species.split('--')  # CIA case

    for i, __species in enumerate(_species):  # for each colliding species
        if isotope_separator in __species:
            isotopes = __species.split('-')
        else:
            isotopes = [__species]

        for j, isotope in enumerate(isotopes):  # for each separated isotope in the species
            # Match isotope pattern in order to handle the case in which not all isotopes are separated (e.g. "13C2H2")
            matches = re.findall(species_pattern, isotope)

            if len(matches) == 0:
                raise ValueError(f"invalid isotope name '{isotope}' in species '{__species}'")

            for k, groups in enumerate(matches):  # for each non-separated isotope in the "separated isotope"
                groups = list(groups)  # contains isotope number, element symbol and element count (e.g. 13, C, 2)

                # Handle deuterium
                if groups[1] == 'D':
                    if groups[0] == '':
                        groups[0] = '2'

                    groups[1] = 'H'
                elif groups[1] == 'T':
                    if groups[0] == '':
                        groups[0] = '3'

                    groups[1] = 'H'

                # Update isotope number
                if mode == 'add':
                    if groups[0] == '':
                        groups[0] = f"{Formula(groups[1]).isotope.massnumber}"
                elif mode == 'remove':
                    if groups[0] != '':
                        groups[0] = ''  # remove isotope number
                else:
                    raise ValueError(f"iter isotopes mode must be 'add'|'remove', but was '{mode}'")

                matches[k] = ''.join(groups)  # rebuild isotope string (e.g. "13C2")

            isotopes[j] = isotope_separator.join(matches)  # join non-separated isotopes with the explicit separator

        _species_tmp = isotope_separator.join(isotopes)  # join isotopes to rebuild species

        # Merge contiguous matches containing the same element (but presumably different isotopes)
        if mode == 'remove':
            _species_tmp = _merge_contiguous_isotopes(
                species=_species_tmp,
                isotope_separator=isotope_separator
            )

        _species[i] = _species_tmp

    __species = '--'.join(_species)  # join species to rebuild collision

    return __species


def _split_species_cloud_info(species):
    cloud_info = ''
    name_split = species.split('(', 1)

    if len(name_split) == 1:
        name = name_split[0]
    else:
        name, cloud_info = name_split
        cloud_info = '(' + cloud_info  # re-add parenthesis lost during split

    return name, cloud_info


def _split_species_charge(species, final_charge_format='+-'):
    # Check for repeated charge symbol, function can work without, but it eases the user's understanding of future error
    if len(re.findall(r'^.*([+\-pm])([+\-pm])$', species)) > 0:
        raise ValueError(f"invalid species formula '{species}', "
                         f"multiple consecutive charge symbols found (+, -, p, m)")

    # Extract charge symbol
    charge_pattern_match = re.findall(r'^.+(_(\d{0,3})[+\-pm])$', species)
    len_charge = 0
    charge = ''
    name = copy.deepcopy(species)

    if len(charge_pattern_match) == 1:
        charge = charge_pattern_match[0][0]
        len_charge = len(charge)
    else:
        charge_pattern_match = re.findall(r'^.+([+\-pm])$', species)

        if len(charge_pattern_match) == 1:  # positive or negative charge
            charge = charge_pattern_match[0][0]
            len_charge = len(charge)

            charge = '_' + charge
        elif species[-1] in ['-', '+']:
            raise ValueError(f"invalid species formula '{species}', "
                             f"either a symbol used is unknown, or the charge formula "
                             f"does not respects the pattern '<element_symbol><charge_number><+|-|p|m>' "
                             f"(e.g., 'Ca2+', 'H-', '7Li-1H_p', 'SO4_2m')")

    # Rewrite charge symbol with +/-
    if final_charge_format == '+-':
        charge = charge.replace('p', '+')
        charge = charge.replace('m', '-')
    elif final_charge_format == 'pm':
        charge = charge.replace('+', 'p')
        charge = charge.replace('-', 'm')
    else:
        raise ValueError(f"Final charge format must be '+-'|'pm', but was '{final_charge_format}'")

    # Temporarily remove charge symbol to remove isotopic numbers
    if len_charge > 0:
        name = species[:-len_charge]

    return name, charge


def _split_species_source(species):
    split = species.split('__', 1)

    if len(split) == 1:
        name = split[0]
        source = ''
    else:
        name, source = split

    return name, source


def _split_species_spectral_info(species):
    split = species.split('.', 1)

    if len(split) == 1:
        name = split[0]
        spectral_info = ''
    else:
        name, spectral_info = split

    return name, spectral_info


def check_opacity_name(opacity_name: str):
    """Check opacity name, based on the ExoMol format.

    The name, in this order:
        - must begin with a number (up to 3 digits) or an uppercase letter
        - must contains a "valid" chemical formula (N1237He15 is considered valid)
        - can have isotopes, that should be separated with '-' (H218O is a working ex., but corresponds to 1H218-16O)
        - can contains '-NatAbund' to signal a mix of isotopes (incompatible with providing isotopic information)
        - can contains '+', '-', 'p' or 'm', (optionally starting with '_' and a up to 3 digits number) to
            signal a ion
        - can contains '(l)' for clouds of liquid particles
        - can contains '(s)' for clouds of solid particles
            * must contains 'crystalline' or 'amorphous' for clouds with solid particles
                - 'crystalline' can be followed by a 3 digit number referring to the crystal space group number
                - 'amorphous' can be followed by up to 5 characters referring to the amorphous state name
        - can contains a source or method, starting with '__'
        - can contains spectral information, starting with '.'
            * spectral information must start with 'R', 'DeltaWavenumber' or 'DeltaWavelength', indicating
                respectively opacities evenly spectrally spaced in resolving power, wavenumber or wavelength
            * spectral spacing must end with a number (integers with or without an exponent format)
            * can contains the spectral range in micron in the format '_<float>-<float>mu', following spectral
                spacing

    Valid examples:
        - 'H' (simplest)
        - 'H2O'
        - '2H2O' (D2O)
        - '1H2-16O'
        - '1H-18O-2H+'
        - 'H2O_m'
        - 'H2O__HITEMP'
        - 'H2O.R120'
        - 'H2O(l)__Mie'
        - 'H2O(s)_amorphous__Mie'
        - 'H2O-NatAbund(s)_crystalline_194__DHS.R39_0.1-250mu'
        - '24Mg2-28Si-16O4(s)_crystalline_068__DHS.R39_0.1-250mu'  (most complex)

    Args:
        opacity_name:

    Returns:

    """
    if len(re.findall(r'^((?!--(-)+).)*$', opacity_name)) == 0:
        raise ValueError(f"invalid opacity name '{opacity_name}'\n"
                         f"Valid separators are '-'|'--', "
                         f"but a separator '---' has been used")

    if len(re.findall('--', opacity_name)) > 1:
        raise ValueError(f"invalid opacity name '{opacity_name}'\n"
                         f"CIA separator '--' must be used at most once, "
                         f"but has been used {len(re.findall('--', opacity_name))} times")

    if len(re.findall(
            r'^'
            r'(\d{0,3}[A-Z])'  # must start with up to 3 digits or an uppercase character
            r'(\d|[A-Z]|[a-z]|--(?!-)|-(?!-)|\[|])*'  # list of isotopes and their number, can be separated by "-"
            r'(-NatAbund)?'  # indicate if a mix of isotopologues has been used to make the opacities
            r'(_?(\d{1,3})?[+\-pm])?'  # charge (+ or p, - or m), can be separated from the isotopes with a "_"
            r'('  # begin clouds formatting
            r'(\(l\))'  # liquid state, no additional information required
            r'|(\(s\))_'  # solid state, it must be specified if the solid is crystalline or amorphous
            r'(crystalline(_\d{3})?'  # crystalline form, can be followed by the space group number (from 001 to 230)
            r'|amorphous(_[A-Z]{1,5})?'  # amorphous form, can be followed by the amorphous phase name
            r'|structureUnclear)'  # exceptionally used when the form is not specified by the opacity provider
            r')?'  # end clouds formatting
            r'(__(\d|[A-Z]|[a-z]|-)+)?'  # source or method
            r'(\.(R|DeltaWavenumber|DeltaWavelength)\d{1,9}(e([+|-])?\d{1,3})?)?'  # spectral sampling mode and value
            r'(_\d+(\.\d+)?-\d+(\.\d+)?mu)?'  # spectral range (min-max) in um
            r'$',
            opacity_name
    )) == 0:
        raise ValueError(
            f"invalid opacity name '{opacity_name}'\n"
            f"The name, in this order:\n"
            f"\t- must begin with a number (up to 3 digits) or an uppercase letter\n"
            f"\t- must contains a valid chemical formula\n"
            f"\t- can have isotopes, that should be separated with '-'\n"
            f"\t- can contain '-NatAbund' to signal a mix of isotopes "
            f"(incompatible with providing isotopic information)\n"
            f"\t- can contain '+', '-', 'p' or 'm', (optionally starting with '_' and a up to 3 digits number) to "
            f"signal a ion \n"
            f"\t- can contain '(l)' for clouds of liquid particles\n"
            f"\t- can contain '(s)' for clouds of solid particles\n"
            f"\t\t* must contains 'crystalline' or 'amorphous' for clouds with solid particles\n"
            f"\t\t\t- 'crystalline' can be followed by a 3 digit number referring to the crystal space group number\n"
            f"\t\t\t- 'amorphous' can be followed by up to 5 characters referring to the amorphous state name\n"
            f"\t- can contain a source or method, starting with '__'\n"
            f"\t- can contain spectral information, starting with '.'\n"
            f"\t\t* spectral information must start with 'R', 'DeltaWavenumber' or 'DeltaWavelength', indicating "
            f"respectively opacities evenly spectrally spaced in resolving power, wavenumber or wavelength\n"
            f"\t\t* spectral spacing must end with a number (integers with or without an exponent format) \n"
            f"\t\t* can contain the spectral range in micron in the format '_<float>-<float>mu', following spectral "
            f"spacing\n"
            f"Valid examples:\n"
            f"\t- 'H2O'\n"
            f"\t- '2H2O'\n"
            f"\t- '1H2-16O'\n"
            f"\t- '1H-18O-2H+'\n"
            f"\t- 'H2O_m'\n"
            f"\t- 'H2O__HITEMP'\n"
            f"\t- 'H2O.R120'\n"
            f"\t- 'H2O(l)__Mie'\n"
            f"\t- 'H2O(s)_crystalline__Mie'\n"
            f"\t- 'H2O-NatAbund(s)_crystalline_194__DHS.R39_0.1-250mu'\n"
            f"\t- '24Mg2-28Si-16O4(s)_crystalline_068__DHS.R39_0.1-250mu'\n"
        )

    if '-NatAbund' in opacity_name and _has_isotope(opacity_name):
        raise ValueError(f"invalid opacity name '{opacity_name}'\n"
                         f"Opacity cannot have one of the species isotopologue (e.g. '1H2-16O') "
                         f"and all of the species isotopologues ('-NatAbund') at the same time")


def get_cia_aliases(name: str) -> str:
    # Try to match name with full directory and directory shortcut
    cia_filenames = _get_base_cia_names()

    for cia, cia_filename in cia_filenames.items():
        if name in [cia_filename, cia_filename.rsplit('_', 1)[0]]:
            return cia_filename

    # Try to match name with aliases
    cia_aliases = copy.deepcopy(cia_filenames)

    cia_aliases.update({
        'H2--H2': ['H2H2', 'H2-H2', 'H2--H2'],
        'H2--He': ['H2He', 'H2-He', 'H2--He', 'HeH2', 'He-H2', 'He--H2'],
        'H2O--H2O': ['H2OH2O', 'H2O-H2O', 'H2O--H2O'],
        'H2O--N2': ['H2ON2', 'H2O-N2', 'H2O--N2', 'N2H2O', 'N2-H2O', 'N2--H2O'],
        'N2--H2': ['N2H2', 'N2-H2', 'N2--H2', 'H2N2', 'H2-N2', 'H2--N2'],
        'N2--He': ['N2He', 'N2-He', 'N2--He', 'HeN2', 'He-N2', 'He--N2'],
        'N2--N2': ['N2N2', 'N2-N2', 'N2--N2'],
        'O2--O2': ['O2O2', 'O2-O2', 'O2--O2'],
        'N2--O2': ['N2O2', 'N2-O2', 'N2--O2', 'O2N2', 'O2-N2', 'O2--N2'],
        'CO2--CO2': ['CO2CO2', 'CO2-CO2', 'CO2--CO2']
    })

    for cia, aliases in cia_aliases.items():
        if name in aliases:
            return cia_filenames[cia]

    # Name does not match a directory, a directory shortcut, or an alias
    return name


def get_cloud_aliases(name: str) -> str:
    cloud_directories = _get_base_cloud_names()

    if '.' in name:
        name, spectral_info = name.split('.', 1)
    else:
        spectral_info = None

    if '__' in name:
        name, method = name.split('__', 1)
    else:
        method = None

    if name in cloud_directories:
        filename = cloud_directories[name]
    else:
        if spectral_info is None:
            spectral_info = ''
        else:
            spectral_info = '.' + spectral_info

        if method is None:
            method = ''
        else:
            method = '__' + method

        # Return NatAbund if no isotope information has been provided (override def. case returning main isotopologue)
        if 'NatAbund' not in name and not _has_isotope(name):
            if '(s)' not in name and '(l)' not in name:
                raise ValueError(f"cloud species name '{name}' lacks condensed matter state information\n"
                                 f"For liquid particles, cloud species names must include '(l)' (e.g. 'H2O(l)').\n"
                                 f"For solid particles, the cloud species must include '(s)_crystalline' for crystals "
                                 f"or '(s)_amorphous' for amorphous solids (e.g. 'MgSiO3(s)_crystalline').")

            species, info = name.split('(', 1)

            name = species + '-NatAbund(' + info + method + spectral_info
        else:
            name = name + method + spectral_info

        return name

    _filename, _spectral_info = filename.split('.', 1)

    # Default method
    if '(s)' in _filename:
        _method = 'DHS'
    elif '(l)' in _filename:
        _method = 'Mie'
    else:
        raise ValueError(f"invalid cloud file name: '{filename}'")

    if method is None:
        method = _method

    if spectral_info is None:
        spectral_info = _spectral_info

    return _filename + '__' + method + '.' + spectral_info


def get_default_cloud_resolution() -> str:
    return get_resolving_power_string(39)


def get_default_correlated_k_resolution() -> str:
    return get_resolving_power_string(1000)


def get_default_line_by_line_resolution() -> str:
    return get_resolving_power_string(1e6)


def get_input_data_file_not_found_error_message(file: str) -> str:
    return (
        f"no matching file found in path '{file}'\n"
        f"This may be caused by an incorrect input_data path, outdated file formatting, or a missing file\n\n"
        f"To set the input_data path, execute: \n"
        f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
        f">>> petitradtrans_config_parser.set_input_data_path('path/to/input_data')\n"
        f"replacing 'path/to/' with the path to the input_data directory\n\n"
        f"To update the outdated files, execute:\n"
        f">>> from petitRADTRANS.__file_conversion import convert_all\n"
        f">>> convert_all()\n\n"
        f"To download the missing file, "
        f"see https://petitradtrans.readthedocs.io/en/latest/content/installation.html"
    )


def get_input_file(file: str, path_input_data: str, sub_path: str = None, expect_spectral_information: bool = False,
                   find_all: bool = False, search_online: bool = True):
    if sub_path is None:
        full_path = os.path.dirname(file)
        _, sub_path, file = split_input_data_path(file, path_input_data)
    else:
        full_path = os.path.abspath(os.path.join(path_input_data, sub_path))

    if not os.path.isdir(full_path):  # search even if search_online is False
        print(f"No such directory '{full_path}'\n"
              f"Searching in the Keeper library...")

        matches = _get_input_file_from_keeper(
            full_path=full_path,
            path_input_data=path_input_data,
            sub_path=sub_path,
            filename=file,
            find_all=find_all
        )
    else:
        matches = _get_input_file(
            path_input_data=path_input_data,
            sub_path=sub_path,
            filename=file,
            expect_spectral_information=expect_spectral_information,
            find_all=find_all
        )

        if len(matches) == 0 and search_online:
            print(f"No file matching name '{file}' found in directory '{full_path}'\n"
                  f"Searching in the Keeper library...")

            matches = _get_input_file_from_keeper(
                full_path=full_path,
                path_input_data=path_input_data,
                sub_path=sub_path,
                filename=file,
                find_all=find_all
            )

    if len(matches) == 0 and not find_all:
        raise FileNotFoundError(get_input_data_file_not_found_error_message(full_path))

    if hasattr(matches, '__iter__') and not isinstance(matches, str):
        matches = [os.path.join(full_path, m) for m in matches]
    else:
        matches = os.path.join(full_path, matches)

    return matches


def get_opacity_directory(species: str, category: str,
                          path_input_data: str = None, full: bool = False):
    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()

    check_opacity_name(species)

    basename = get_species_basename(species, join=True)
    istopologue_name = get_species_isotopologue_name(species, join=False)

    _, natural_abundance, charge, cloud_info, source, spectral_info = (
        split_species_all_info(species, final_charge_format='pm'))
    filename = join_species_all_info(
        name=istopologue_name,
        charge=charge,
        cloud_info=cloud_info,
        source=source,
        spectral_info=spectral_info
    )
    directory = join_species_all_info(
        name=istopologue_name,
        charge=charge.replace('p', '+').replace('m', '-'),
        cloud_info=cloud_info
    )

    sub_paths = get_input_data_subpaths()

    if category not in sub_paths:
        keys = list(sub_paths.keys())
        raise KeyError(f"category must be {'|'.join(keys)}, but was '{category}'")

    sub_path = os.path.join(sub_paths[category], basename, directory)
    full_path = os.path.abspath(os.path.join(path_input_data, sub_path))

    if full:
        return full_path, sub_path, filename

    return full_path


def get_opacity_input_file(path_input_data: str, category: str, species: str, find_all: bool = False,
                           search_online: bool = True) -> str:
    """Return the absolute filename of a species opacity.
    The validity of the given species name is checked.

    Automatically infer the species base and isotopologue directories from the species name.
    Then, try to match the species name with the files in the folder. If only one file is matched, it is returned.
    If multiple files match, the configured default file is used if it exists. If not, ask the user to configure a
    default file.

    Information given in the species name are decomposed for the match. For example:
        - "H2O.R120" will match e.g. the file "1H2-16O__HITEMP.R120_0.1-250mu.ktable.petitRADTRANS.h5"
        - "H2O__POKAZATEL" will match e.g. the file "1H2-16O__POKAZATEL.R1000_0.1-250mu.ktable.petitRADTRANS.h5"

    If no information on the resolution is given, the default resolution for correlated-k or line-by-line is assumed.
    If no or partial isotopic information is given:
        - for line opacities: the main isotope is assumed
        - for continuum opacities: "NatAbund" (a standard mix of all isotopes) is assumed

    Args:
        path_input_data:
            Path to the input data directory
        category:
            Input data category
        species:
            Species to get the opacity filename. The species name must be valid.
        find_all:
            If True, return all the matched files. If False, raise an error if no file is found, and only one file
            is returned.
        search_online:
            If True, search online for the opacity file
    Returns:
        The absolute opacity filename of the species
    """
    _, sub_path, filename = get_opacity_directory(
        species=species,
        category=category,
        path_input_data=path_input_data,
        full=True
    )

    return get_input_file(
        file=filename,
        path_input_data=path_input_data,
        sub_path=sub_path,
        expect_spectral_information=True,
        find_all=find_all,
        search_online=search_online
    )


def get_resolving_power_from_string(string: str) -> int:
    return int(string.split('R', 1)[1])


def get_resolving_power_string(resolving_power: [int, float]) -> str:
    if isinstance(resolving_power, int):
        return f"R{resolving_power}"
    elif isinstance(resolving_power, float):
        return f"R{resolving_power:.0e}".replace('e+', 'e').replace('e0', 'e')


def get_species_basename(species: str, join: bool = False) -> str:
    name, natural_abundance, charge, cloud_info, _, _ = split_species_all_info(species, final_charge_format='+-')

    # Remove isotopic numbers
    name = _rebuild_isotope_numbers(name, mode='remove')

    if join:
        return join_species_all_info(name, charge=charge, cloud_info=cloud_info)
    else:
        return name


def get_species_isotopologue_name(species: str, join: bool = False) -> str:
    name, natural_abundance, charge, cloud_info, _, _ = split_species_all_info(species, final_charge_format='+-')

    name = join_species_all_info(name, natural_abundance)
    name = _rebuild_isotope_numbers(name, mode='add')

    if join:
        return join_species_all_info(name, charge=charge, cloud_info=cloud_info)
    else:
        return name


def join_species_all_info(name, natural_abundance='', charge='', cloud_info='', source='', spectral_info=''):
    if natural_abundance != '':
        name += '-' + natural_abundance

    name += charge + cloud_info

    if source != '':
        name += '__' + source

    if spectral_info != '':
        name += '.' + spectral_info

    return name


def split_input_data_path(path: str, path_input_data: str):
    if path_input_data not in path:
        raise ValueError(f"path '{path}' does not contains the input data path ('{path_input_data}')")

    sub_path = path.split(path_input_data + os.path.sep, 1)[-1]
    file = os.path.basename(sub_path)
    sub_path = os.path.dirname(sub_path)

    return path_input_data, sub_path, file


def split_species_all_info(species, final_charge_format='+-'):
    name, spectral_info = _split_species_spectral_info(species)
    name, source = _split_species_source(name)  # remove resolving power or opacity source information

    # Remove cloud info
    name, cloud_info = _split_species_cloud_info(name)

    if '-NatAbund' in name:
        name = name.replace('-NatAbund', '')
        natural_abundance = 'NatAbund'
    else:
        natural_abundance = ''

    # Check for repeated ion symbol, function can work without, but it eases the user's understanding of future error
    if len(re.findall(r'^.*([+\-pm])([+\-pm])$', name)) > 0:
        raise ValueError(f"invalid species formula '{name}', "
                         f"multiple consecutive charge symbols found (+, -, p, m)")

    # Extract ion symbol
    name, charge = _split_species_charge(name, final_charge_format=final_charge_format)

    return name, natural_abundance, charge, cloud_info, source, spectral_info
