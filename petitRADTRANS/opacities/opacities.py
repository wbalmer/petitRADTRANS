"""Manage the opacity files.
"""
import copy
import datetime
import os
import re
import warnings

from molmass import Formula
import numpy as np
import numpy.typing as npt
import h5py

from petitRADTRANS import __version__ as prt_version
from petitRADTRANS._input_data import default_file_selection, find_input_file
from petitRADTRANS.cli.prt_cli import get_keeper_files_url_paths
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.utils import list_str2str, LockedDict

if os.environ.get("pRT_emcee_mode") == 'True':  # TODO make use of config_parser instead
    pass
else:
    # MPI Multiprocessing
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        MPI = None
        comm = None
        rank = 0


class Opacity:
    # Categories
    _default_category: str = 'unknown'
    _temperature_grid_types: set[str, str] = {'regular', 'pressure-dependent'}

    # Charge strings
    _minus_char: str = 'm'
    _plus_char: str = 'p'
    _minus_symbol: str = '-'
    _plus_symbol: str = '+'

    _charges: dict[str, str] = {
        _minus_symbol: _minus_char,
        _plus_symbol: _plus_char
    }
    _charge_symbols: set[str] = set(_charges.keys())
    _charge_chars: set[str] = set(_charges.values())

    # Matter states strings
    _gas_matter_state: str = '(g)'
    _liquid_matter_state: str = '(l)'
    _solid_matter_state: str = '(s)'
    _condensed_matter_states: set[str] = {
        _liquid_matter_state,
        _solid_matter_state
    }
    _matter_states = copy.copy(_condensed_matter_states)
    _matter_states.add(_gas_matter_state)

    # Solid structures strings
    _amorphous_structure: str = 'amorphous'
    _crystalline_structure: str = 'crystalline'
    _unclear_structure: str = 'structureUnclear'
    _solid_structures: set[str] = {
        _amorphous_structure,
        _crystalline_structure,
        _unclear_structure
    }

    # Spectral information strings
    _constant_delta_wavelength: str = 'DeltaWavelength'
    _constant_delta_wavenumber: str = 'DeltaWavenumber'
    _constant_resolving_power: str = 'R'
    _spectral_sampling_types: set[str] = {
        _constant_delta_wavelength,
        _constant_delta_wavenumber,
        _constant_resolving_power
    }
    _wavelength_units: str = 'mu'

    # File name elements
    _charge_separator: str = '_'
    _default_extension: str = 'unknown'
    _extension_opacity: str = 'petitRADTRANS.h5'
    _isotope_separator: str = '-'
    _colliding_species_separator: str = '--'
    _natural_abundance_string: str = "NatAbund"
    _solid_structure_separator: str = '_'
    _spectral_information_separator: str = '.'
    _source_separator: str = '__'
    _wavelength_range_separator: str = '_'
    _wavelength_separator: str = '-'

    # Patterns
    _amorphous_structure_id_pattern: re.Pattern = re.compile(r'[A-Z]{1,5}')  # up to 5 capital letters
    _charge_pattern: re.Pattern = re.compile(
        r'.+(' + rf'{_charge_separator}' + r'(\d{0,3})'
        + r'['
        + '\\' + rf'{"///".join(_charge_symbols)}'.replace('///', '\\')
        + rf'{"".join(_charge_chars)}'
        + r'])'
    )
    _isotope_pattern: re.Pattern = re.compile(r'(\d{1,3})?([A-Z][a-z]?|e)(\d{1,3})?')
    _space_group_pattern: re.Pattern = re.compile(r'\d{3}')  # 3 digits
    _spectral_sampling_pattern: re.Pattern = re.compile(
        r'(' + '\\' + rf'{_spectral_information_separator}'
        + r'(' + r'|'.join(_spectral_sampling_types) + r')' + r'\d{1,9}(e([+|-])?\d{1,3})?)?'
    )
    _wavelength_range_pattern: re.Pattern = re.compile(
        r'(' + rf'{_wavelength_range_separator}' + r'\d+(\.\d+)?' + rf'{_wavelength_separator}'
        + rf'\d+(\.\d+)?{_wavelength_units})?'
    )

    _name_pattern = re.compile(
        r'^'  # must start ...
        r'(\d{0,3}[A-Z])'  # ... with up to 3 digits or an uppercase character
        r'(\d|[A-Z]|[a-z]|'  # isotope number and element symbol (note: there is no opacities for e-)
        + _colliding_species_separator +
        r'(?!' + _isotope_separator + r')|'  # ensure that the separator is not repeated
        + _isotope_separator +
        r'(?!' + _isotope_separator + r')|'  # ensure that the separator is not repeated
                                      r'\[|])*'  # list of isotopes and their number, can be separated by "-"
                                      r'('  # "[" and "]" can be used as well
        + _isotope_separator + _natural_abundance_string +  # indicate a mix of isotopologues
        r')?'  # (natural abundance string is optional)
        r'('  # begin charge formatting
        + _charge_separator + r'?'  # can be separated with a "_"
                              r'(\d{1,3})?'  # can have a charge number of up to 3 digits
                              r'[' + _minus_char + _plus_char + '\\' + _minus_symbol + '\\' + _plus_symbol + r']'
                                                                                       r')?'  # end charge formatting
                                                                                       r'('  # begin cloud formatting
                                                                                       r'('
        + _liquid_matter_state.replace('(', r'\(').replace(')', r'\)') +  # liquid state
        r')'  # no additional information required for liquid state
        r'|('
        + _solid_matter_state.replace('(', r'\(').replace(')', r'\)') +  # solid state
        r')' + _solid_structure_separator +
        r'('  # for solid states, it must be specified if the solid is crystalline or amorphous
        + _crystalline_structure +  # crystalline form ...
        r'(' + _solid_structure_separator + r'\d{3})?'  # ... can be followed by the space group num. (from 001 to 230)
                                            r'|'
        + _amorphous_structure +  # amorphous form ...
        r'(' + _solid_structure_separator + r'[A-Z]{1,5})?'  # ... can be followed by the amorphous phase name
                                            r'|'
        + _unclear_structure +  # exceptionally used when the structure is not specified by the opacity provider
        r')'  # end solid state extra info
        r')?'  # end cloud formatting (optional)
        r'(' + _source_separator + r'(\d|[A-Z]|[a-z]|-)+)?'  # source or method (optional)
                                   r'(' + '\\' + _spectral_information_separator + '('  # begin spectral sampling type
        + _constant_resolving_power +
        r'|'
        + _constant_delta_wavelength +
        r'|'
        + _constant_delta_wavenumber +
        r')'  # end spectral sampling type
        r'\d{1,9}(e([+|-])?\d{1,3})?'  # spectral sampling value
        r')?'  # (sampling type+value is optional)
        r'('  # begin wavelength range formatting
        + _wavelength_range_separator +
        r'\d+(\.\d+)?' + _wavelength_separator + rf'\d+(\.\d+)?{_wavelength_units}'  # spectral range (min-max) in um
                                                 r')?'  # end wavelength range formatting (optional)
                                                 r'$'  # all the string must match this pattern
    )

    # Default resolving powers
    _default_cia_resolving_power: float = 831.0
    _default_cloud_resolving_power: float = 39.0
    _default_correlated_k_resolving_power: float = 1000.0
    _default_line_by_line_resolving_power: float = 1e6
    _default_resolving_power: float = 1000.0

    def __init__(
            self,
            species_list,
            natural_abundance: bool = False,
            charge: int = 0,
            source: str = 'unknown',
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = 0.0,
            wavelength_min: float = 0.0,
            wavelength_max: float = 0.0,
            matter_state: str = _gas_matter_state,
            solid_structure: str = None,
            solid_structure_id: str = None,
            path_input_data: str = None,
            category: str = _default_category,
            species_full_name: str = None,
            species_cloud_info: str = None,
            species_base_name: str = None,
            species_isotopologue_name: str = None,
            extension: str = _default_extension,
            full_extension: str = None,
            file_name: str = None,
            sub_path: str = None,
            directory: str = None,
            absolute_path: str = None
    ):
        self.species_list = species_list
        self.natural_abundance = natural_abundance
        self.charge = charge
        self.source = source
        self.spectral_sampling_type = spectral_sampling_type
        self.spectral_sampling = spectral_sampling
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.matter_state = matter_state
        self.solid_structure = solid_structure
        self.solid_structure_id = solid_structure_id

        self.category = category
        self.extension = extension

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        self.path_input_data = path_input_data

        if full_extension is None:
            full_extension = self.get_full_extension()

        self.full_extension = full_extension

        if species_full_name is None:
            species_full_name = self.get_full_name()

        self.species_full_name = species_full_name

        if species_cloud_info is None:
            species_cloud_info = self.get_cloud_info()

        self.species_cloud_info = species_cloud_info

        if species_base_name is None:
            species_base_name = self.get_base_name(join=True)

        self.species_base_name = species_base_name

        if species_isotopologue_name is None:
            species_isotopologue_name = self.get_isotopologue_name(join=True)

        self.species_isotopologue_name = species_isotopologue_name

        if file_name is None:
            file_name = self.get_file_name()

        self.file_name = file_name

        if sub_path is None:
            sub_path = self._get_sub_path(self.category)

        self.sub_path = sub_path

        if directory is None:
            self.directory = self.get_directory()

        if absolute_path is None:
            absolute_path = self.get_absolute_path()

        self.absolute_path = absolute_path

    @property
    def has_colliding_species(self) -> bool:
        for item in self.species_list:
            if not isinstance(item, str):
                if hasattr(item, '__iter__'):
                    if isinstance(item[0], str) and len(item) > 1:
                        return True

                raise ValueError(
                    f"'species_list' must be a list of strings of size > 1 or a list of lists of strings, "
                    f"but was '{self.species_list}'"
                )
            else:
                break

        return False

    @staticmethod
    def __modify_isotope_string(isotope: str, mode: str, isotope_pattern: str) -> list[str]:
        # Match isotope pattern in order to handle the case in which not all isotopes are separated (e.g. "13C2H2")
        matches = re.findall(isotope_pattern, isotope)

        if len(matches) == 0:
            raise ValueError(f"invalid isotope name '{isotope}', no match found for the given pattern")

        for k, groups in enumerate(matches):  # for each non-separated isotope in the "separated isotope"
            groups = list(groups)  # contains isotope number, element symbol and element count (e.g. 13, C, 2)

            if groups[1] == 'D':  # handle deuterium
                if groups[0] == '':
                    groups[0] = '2'

                groups[1] = 'H'
            elif groups[1] == 'T':  # handle tritium
                if groups[0] == '':
                    groups[0] = '3'

                groups[1] = 'H'

            # Update isotope number
            if mode == 'add':
                if groups[0] == '' and groups[1] != 'e':
                    groups[0] = f"{Formula(groups[1]).isotope.massnumber}"
            elif mode == 'remove':
                if groups[0] != '':
                    groups[0] = ''  # remove isotope number
            elif mode == 'scientific':
                if groups[0] != '':  # isotope number
                    groups[0] = '$^{' + groups[0] + '}$'

                if groups[2] != '':  # stoichiometric number
                    groups[2] = '$_{' + groups[2] + '}$'
            else:
                raise ValueError(f"iter isotopes mode must be 'add'|'remove'|'scientific', but was '{mode}'")

            matches[k] = ''.join(groups)  # rebuild isotope string (e.g. "13C2")

        return matches

    @classmethod
    def __recursive_merge_contiguous_isotopes(cls, isotope_groups, i, index_merge=None):
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
                        isotope_groups = cls.__recursive_merge_contiguous_isotopes(
                            isotope_groups=isotope_groups,
                            i=i,
                            index_merge=index_merge - 1
                        )
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

    @staticmethod
    def __single_separator_split(string: str, separator: str, value_error_message: str = None) -> (str, str):
        _split = string.split(separator, 1)

        if len(_split) == 1:
            if value_error_message is None:
                left_split = _split[0]
                right_split = ''
            else:
                raise ValueError(value_error_message)
        else:
            left_split, right_split = _split

        return left_split, right_split

    @classmethod
    def _before_write(cls, temperature_grid_type: str, molar_mass: float, species_name: [str, tuple[str, ...]],
                      date_id: str) -> ([float, npt.NDArray[float]], str):
        if temperature_grid_type not in cls._temperature_grid_types:
            raise ValueError(
                f"temperature grid type must be "
                f"{list_str2str(cls._temperature_grid_types, '|')}, "
                f"but is '{temperature_grid_type}'"
            )

        if molar_mass is None:
            if not isinstance(species_name, str):
                molar_mass = np.array([float(get_species_molar_mass(species)) for species in species_name])
            else:
                molar_mass = get_species_molar_mass(species_name)

        if date_id is None:
            date_id = (
                f'petitRADTRANS-v{prt_version}'
                f'_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
            )

        return molar_mass, date_id

    @classmethod
    def _init_species_name_elements(
            cls, name: str, species: str, natural_abundance: str, charge: str
    ) -> (str, bool, int):
        name = cls.modify_isotope_numbers(
            species=name,
            mode='add',
            isotope_separator=cls._isotope_separator,
            isotope_pattern=cls._isotope_pattern.pattern,
            natural_abundance_string=cls._natural_abundance_string,
            colliding_species_separator=cls._colliding_species_separator
        )

        if natural_abundance == cls._natural_abundance_string:
            natural_abundance = True
        elif natural_abundance == '':
            natural_abundance = False
        else:
            raise ValueError(
                f"natural abundance opacities must be flagged with '{cls._natural_abundance_string}', "
                f"but was '{natural_abundance}' (species was '{species}')"
            )

        if len(charge) > 0:
            if charge[-1] in {cls._minus_symbol, cls._minus_char}:
                charge = -int(charge[:-1])
            elif charge[-1] in {cls._plus_symbol, cls._plus_char}:
                charge = int(charge[:-1])
            else:
                raise ValueError(
                    f"unknown charge symbol '{charge[-1]}', "
                    f"charge symbol must be {list_str2str(cls._charge_symbols, '|')}"
                    f"or must be characters {list_str2str(cls._charge_chars)}"
                )
        else:
            charge = 0

        return name, natural_abundance, charge

    @staticmethod
    def _get_sub_path(category: str) -> str:
        sub_paths = get_input_data_subpaths()

        if category not in sub_paths:
            keys = list(sub_paths.keys())
            raise KeyError(f"category must be {'|'.join(keys)}, but was '{category}'")

        return str(sub_paths[category])

    @staticmethod
    def _has_isotope(string: str) -> bool:
        if (len(re.findall(r'(\d{1,3})?([A-Z][a-z]?)(\d{0,3})?-(\d{1,3})([A-Z][a-z]?)', string)) > 0
                or len(re.findall(r'^(\d{1,3})([A-Z][a-z]?)(\d{0,3})?', string)) > 0):
            return True
        else:
            return False

    @classmethod
    def _join_spectral_information(cls, spectral_sampling, wavelength_range):
        return f"{spectral_sampling}{cls._wavelength_range_separator}{wavelength_range}"

    @classmethod
    def _match_function(cls, path_input_data, sub_path, files=None, filename=None,
                        expect_default_file_exists=True,
                        find_all=False, display_other_files=False):
        full_path = str(os.path.join(path_input_data, sub_path))

        if files is None:
            files = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]

        if len(files) == 0:  # no file in path, return empty list
            return []
        else:  # at least one file detected in path
            if filename is not None:  # check if one of the files matches the given filename
                matching_files = []
                _filename = cls.split_species_spectral_info(filename)[0]
                filename_source = cls.split_species_source(_filename)[1]

                filename_sampling, filename_range = cls.find_spectral_information(filename)

                if filename_sampling != '' or filename_range != '':
                    # First pass, try to use default resolution
                    for file in files:
                        # Extract source and spectral info
                        _file, spectral_info = cls.split_species_spectral_info(file)

                        file_source = cls.split_species_source(_file)[1]
                        file_sampling, file_range = cls.find_spectral_information(
                            cls._spectral_information_separator + spectral_info
                        )

                        # Check if current file has a source and spectral info
                        if file_source == '':
                            warnings.warn(f"file '{file}' lacks a source")

                        if file_sampling == '' or file_range == '':
                            warnings.warn(f"file '{file}' lacks spectral information "
                                          f"(sampling: {file_sampling}, range: {file_range})")

                        # Add the current file to the filename matches if the conditions apply
                        if _filename in _file:
                            if filename_source != '':
                                if filename_source != file_source:
                                    continue

                            if filename_sampling != '':
                                range_match = False

                                if filename_range != '':
                                    if filename_range == file_range:
                                        range_match = True
                                else:
                                    range_match = True

                                if filename_sampling == file_sampling and range_match:
                                    matching_files.append(file)
                            else:
                                default_resolving_powers = set(
                                    cls.get_resolving_power_string(r)
                                    for r in (
                                        cls._default_cloud_resolving_power,
                                        cls._default_correlated_k_resolving_power,
                                        cls._default_line_by_line_resolving_power
                                    )
                                )

                                if file_sampling in default_resolving_powers:
                                    matching_files.append(file)

                # Second pass, take any matching file regardless of resolution
                if len(matching_files) == 0 and filename_sampling == '':
                    for file in files:
                        _file = cls.split_species_spectral_info(file)[0]
                        file_source = cls.split_species_source(_file)[1]

                        if _filename in _file:
                            if filename_source != '':
                                if filename_source != file_source:
                                    continue

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

                # Check if spectral info of default file is consistent with requested default file
                if filename is not None:
                    # Get spectral info
                    filename_sampling, filename_range = cls.find_spectral_information(filename)
                    resolution_default_file, range_default_file = cls.find_spectral_information(
                        petitradtrans_config_parser['Default files'][sub_path]
                    )

                    # Check spectral info consistency
                    inconsistent_spectral_info = False

                    if filename_sampling != '' and filename_sampling != resolution_default_file:
                        inconsistent_spectral_info = True

                    if filename_range != '' and filename_range != range_default_file:
                        inconsistent_spectral_info = True

                    # Raise error if there is an inconsistency
                    if inconsistent_spectral_info:
                        # Get all info
                        name, natural_abundance, charge, cloud_info, _, spectral_info = cls.split_species_all_info(
                            species=filename
                        )
                        _, _, _, _, source, spectral_info_default_file = cls.split_species_all_info(
                            species=petitradtrans_config_parser['Default files'][sub_path]
                        )

                        # Remove default file extension to get a clean default file spectral info to display
                        spectral_info_default_file = spectral_info_default_file.rsplit('.', 3)[0]

                        # Make an example species with added source for the error message
                        example_species = cls.join_species_all_info(
                            species_name=name,
                            natural_abundance=natural_abundance,
                            charge=charge,
                            cloud_info=cloud_info,
                            source=source,
                            spectral_info=spectral_info
                        )

                        raise FileExistsError(
                            f"More than one file detected in '{full_path}' "
                            f"with spectral information '{spectral_info}'\n"
                            f"A default file is already set for species '{name}' "
                            f"('{petitradtrans_config_parser['Default files'][sub_path]}'), "
                            f"but with different spectral info ('{spectral_info_default_file}')\n"
                            f"Add a source to your species name (e.g. '{example_species}'), or update your default file"
                        )

                # Check if the default file exists
                if os.path.isfile(default_file):
                    return default_file
                elif not expect_default_file_exists:
                    return os.path.split(default_file)[-1]
                else:
                    raise FileNotFoundError(
                        f"no such file: '{default_file}'\n"
                        f"Update the 'Default file' entry for '{sub_path}' in petitRADTRANS' configuration "
                        f"by executing:\n"
                        f">>> from petitRADTRANS.config.configuration import petitradtrans_config_parser\n"
                        f">>> petitradtrans_config_parser.set_default_file(<new_default_file>)\n"
                        f"Or download the missing file."
                    )
            else:  # make the user enter the default file
                new_default_file = default_file_selection(
                    files=files,
                    full_path=full_path,
                    sub_path=sub_path
                )

                petitradtrans_config_parser.set_default_file(
                    file=os.path.join(full_path, new_default_file),
                    path_input_data=path_input_data
                )

                return new_default_file

    @classmethod
    def _merge_contiguous_isotopes(cls, species, isotope_separator):
        isotope_groups = re.findall(r'([A-Z][a-z]?|e)(\d{1,3})?', species)
        isotope_groups = [list(isotope_group) for isotope_group in isotope_groups]

        # Merge the isotopes, numbers of contiguous isotopes are converted to int, merged groups numbers are set to 0
        for i in range(len(isotope_groups)):
            isotope_groups = cls.__recursive_merge_contiguous_isotopes(isotope_groups, i)

        # Convert back numbers from int to str
        for i in range(len(isotope_groups)):
            isotope_groups[i][1] = str(isotope_groups[i][1])

        # Rebuild the species with merged isotopes
        return isotope_separator.join([
            isotope_separator.join(isotope_group)
            for isotope_group in isotope_groups
            if isotope_group[1] != '0'  # ignore groups that has been merged
        ])

    @classmethod
    def check_name(cls, opacity_name: str):
        """Check opacity name, based on the ExoMol format.

        The name, in this order:
            - must begin with a number (up to 3 digits) or an uppercase letter
            - must contains a "valid" chemical formula (N1237He15 is considered valid)
            - can have isotopes, that should be separated with '-' (e.g. H218O works, but corresponds to 1H218-16O)
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
            raise ValueError(
                f"invalid opacity name '{opacity_name}'\n"
                f"Valid separators are "
                f"'{cls._isotope_separator}'|'{cls._colliding_species_separator}', "
                f"but a separator '---' has been used"
            )

        if len(re.findall(cls._colliding_species_separator, opacity_name)) > 1:
            raise ValueError(
                f"invalid opacity name '{opacity_name}'\n"
                f"CIA separator '{cls._colliding_species_separator}' must be used at most once, "
                f"but has been used "
                f"{len(re.findall(f'{cls._colliding_species_separator}', opacity_name))} times"
            )

        # Check if name respect the convention
        if len(re.findall(cls._name_pattern, opacity_name)) == 0:
            raise ValueError(
                f"invalid opacity name '{opacity_name}'\n"
                f"The name, in this order:\n"
                f"\t- must begin with a number (up to 3 digits) or an uppercase letter\n"
                f"\t- must contains a valid chemical formula\n"
                f"\t- can have isotopes, that should be separated with '{cls._isotope_separator}'\n"
                f"\t- can contain '{cls._natural_abundance_string}' to signal a mix of isotopes "
                f"(incompatible with providing isotopic information)\n"
                f"\t- can contain "
                f"'{cls._plus_symbol}', '{cls._minus_symbol}', "
                f"'{cls._plus_char}' or '{cls._minus_char}', "
                f"(optionally starting with '{cls._charge_separator}' "
                f"and a up to 3 digits number) to signal a ion \n"
                f"\t- can contain '{cls._liquid_matter_state}' for clouds of liquid particles\n"
                f"\t- can contain '{cls._solid_matter_state}' for clouds of solid particles\n"
                f"\t\t* must contains '{cls._crystalline_structure}' or '{cls._amorphous_structure}' "
                f"for clouds with solid particles\n"
                f"\t\t\t- '{cls._crystalline_structure}' can be followed by a 3 digit number referring to "
                f"the crystal space group number\n"
                f"\t\t\t- '{cls._amorphous_structure}' can be followed by up to 5 characters referring to "
                f"the amorphous state name\n"
                f"\t- can contain a source or method, starting with '{cls._source_separator}'\n"
                f"\t- can contain spectral information, starting with '{cls._spectral_information_separator}'\n"
                f"\t\t* spectral information must start with "
                f"'{cls._constant_resolving_power}', "
                f"'{cls._constant_delta_wavelength}' "
                f"or '{cls._constant_delta_wavenumber}', indicating "
                f"respectively opacities evenly spectrally spaced in resolving power, wavelength or wavenumber\n"
                f"\t\t* spectral spacing must end with a number (integers with or without an exponent format) \n"
                f"\t\t* can contain the spectral range in micron in the format "
                f"'{cls._wavelength_range_separator}"
                f"<float>{cls._wavelength_separator}"
                f"<float>{cls._wavelength_units}', "
                f"following spectral spacing\n"
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

        if cls._natural_abundance_string in opacity_name and cls._has_isotope(opacity_name):
            raise ValueError(
                f"invalid opacity name '{opacity_name}'\n"
                f"Opacity cannot have one of the species isotopologue (e.g. '1H2-16O') "
                f"and all of the species isotopologues ('{cls._natural_abundance_string}') at the same time"
            )

    @classmethod
    def find(cls, species: str, category: str = None, path_input_data: str = None, find_all: bool = False,
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

        If no information on the resolution is given, the default resolution for corr.-k or line-by-line is assumed.
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
        if category is None:
            category = cls._default_category

        cls.check_name(species)

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        basename = cls.get_species_base_name(species, join=True)
        istopologue_name = cls.get_species_isotopologue_name(species, join=False)
        sub_path = cls._get_sub_path(category)

        _, _, charge, cloud_info, source, spectral_info = (
            cls.split_species_all_info(
                species=species,
                replace_charge_symbol_with_char=False
            )
        )

        species_filename = cls.join_species_all_info(
            species_name=istopologue_name,
            natural_abundance='',  # no need to add the natural abundance
            charge=charge,
            cloud_info=cloud_info,
            source=source,
            spectral_info=spectral_info,
            spectral_sampling=None,
            wavelength_range=None
        )

        for charge_symbol in cls._charge_symbols:
            charge = charge.replace(charge_symbol, cls._charges[charge_symbol])

        species_directory = cls.join_species_all_info(
            species_name=istopologue_name,
            natural_abundance='',  # no need to add the natural abundance
            charge=charge,  # charge with symbols
            cloud_info=cloud_info,
            source='',
            spectral_info='',
            spectral_sampling=None,
            wavelength_range=None
        )

        sub_path = os.path.join(sub_path, basename, species_directory)

        return find_input_file(
            file=species_filename,
            path_input_data=path_input_data,
            sub_path=sub_path,
            match_function=cls._match_function,
            find_all=find_all,
            search_online=search_online
        )

    @classmethod
    def find_spectral_information(cls, filename):
        filename_sampling = [
            match[0]
            for match in re.findall(cls._spectral_sampling_pattern, filename)
            if match[0] != ''
        ]
        filename_range = [
            match[0]
            for match in re.findall(cls._wavelength_range_pattern, filename)
            if match[0] != ''
        ]

        multiple_samplings = False
        multiple_ranges = False

        if len(filename_sampling) == 0:
            filename_sampling = ''
        elif len(filename_sampling) == 1:
            filename_sampling = filename_sampling[0].replace(cls._spectral_information_separator, '')
        else:
            multiple_samplings = True

        if len(filename_range) == 0:
            filename_range = ''
        elif len(filename_range) == 1:
            filename_range = filename_range[0].replace(cls._wavelength_range_separator, '')
        else:
            multiple_ranges = True

        if multiple_samplings or multiple_ranges:
            raise ValueError(f"found multiple spectral information patterns in file '{filename}' "
                             f"({filename_sampling}, {filename_range})")

        return filename_sampling, filename_range

    @classmethod
    def from_species(
            cls,
            species: str,
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = 0.0,
            wavelength_min: float = 0.0,
            wavelength_max: float = 0.0,
            path_input_data: str = None
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            matter_state, solid_structure, solid_structure_id,
            source,
            _, _, _, _  # spectral info
        ) = cls.split_species_all_info(
            species=species,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=matter_state,
            solid_structure=solid_structure,
            solid_structure_id=solid_structure_id,
            path_input_data=path_input_data,
            category=cls._default_category
        )

        return new_opacity

    @classmethod
    def from_species_fullname(
            cls,
            species_fullname: str,
            path_input_data: str = None,
            category: str = 'unknown_opacities'
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            matter_state, solid_structure, solid_structure_id,
            source,
            spectral_sampling_type, spectral_sampling,
            wavelength_min, wavelength_max
        ) = cls.split_species_all_info(
            species=species_fullname,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species_fullname,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=matter_state,
            solid_structure=solid_structure,
            solid_structure_id=solid_structure_id,
            path_input_data=path_input_data,
            category=category
        )

        return new_opacity

    def get_absolute_path(self):
        return os.path.abspath(
            os.path.join(
                self.directory,
                '.'.join((self.file_name, self.full_extension))
            )
        )

    def get_base_name(self, join: bool = False) -> str:
        return self.get_species_base_name(
            species_full_name=self.species_full_name,
            join=join
        )

    def get_cloud_info(self) -> str:
        cloud_info: str = ''

        if self.matter_state in self._matter_states:
            if self.matter_state == self._gas_matter_state:
                return cloud_info
            else:
                matter_state = self.matter_state

            cloud_info += matter_state

            # Solid structures, liquids do not have a structure
            if self.matter_state == self._solid_matter_state:
                # Add separator
                cloud_info += self._solid_structure_separator

                # Add solid structure string
                if self.solid_structure not in self._solid_structures:
                    raise ValueError(
                        f"solid structure for solid species "
                        f"(matter_state is '{self.matter_state}') "
                        f"must be {list_str2str(self._solid_structures)}, but was '{self.solid_structure}'"
                    )

                cloud_info += self.solid_structure

                # Add solid structure ID
                if self.solid_structure_id is None:
                    raise ValueError(
                        "solid structure ID for solid species cannot be None"
                    )

                respects_space_group_pattern = self._space_group_pattern.fullmatch(self.solid_structure_id) is not None

                if self.solid_structure == self._crystalline_structure:
                    if (
                            respects_space_group_pattern
                            and 0 <= int(self.solid_structure_id) <= 230
                    ):
                        structure_id = self.solid_structure_id
                    else:
                        raise ValueError(
                            f"crystalline structure space group must be a 3-digits integer "
                            f"between '000' (unknown space group) and '230' "
                            f"(see https://en.wikipedia.org/wiki/List_of_space_groups), "
                            f"but was '{self.solid_structure_id}'"
                        )
                elif self.solid_structure == self._amorphous_structure:
                    if self._amorphous_structure_id_pattern.findall(self.solid_structure_id) is not None:
                        structure_id = self.solid_structure_id
                    else:
                        raise ValueError(
                            f"amorphous structure id group must be up to 5 capital letters, "
                            f"but was '{self.solid_structure_id}'"
                        )
                elif self.solid_structure == self._unclear_structure:
                    if self.solid_structure_id is None:
                        structure_id = ''
                    else:
                        raise ValueError(
                            "unclear structure must have no structure Id"
                        )
                else:
                    raise NotImplementedError(
                        f"solid structure '{self.solid_structure}' has no implemented structure ID"
                    )

                cloud_info += self._solid_structure_separator + structure_id
        else:
            raise ValueError(f"matter state must be {list_str2str(self._matter_states)}, but was '{self.matter_state}'")

        return cloud_info

    def get_charge_string(self, replace_symbol_with_char=True) -> str:
        charge_string: str = ''

        if replace_symbol_with_char:
            charges: dict[str, str] = copy.deepcopy(self._charges)
        else:
            charges: dict[str, str] = {
                key: key for key in self._charges  # use the symbols (stored in the keys) as values
            }

        if self.charge != 0:
            if self.charge < -1:
                charge_string += str(-self.charge) + charges['-']
            elif self.charge == -1:
                charge_string += charges['-']
            elif self.charge == 1:
                charge_string += charges['+']
            elif self.charge > 1:
                charge_string += str(self.charge) + charges['+']
            else:
                raise NotImplementedError(f"a charge of {self.charge} is not handled, this is likely a code error")

        return charge_string

    @classmethod
    def get_default_category(cls) -> str:
        return cls._default_category

    @classmethod
    def get_default_extension(cls) -> str:
        return cls._default_extension

    @classmethod
    def get_default_resolving_power(cls):
        return cls._default_resolving_power

    def get_directory(self) -> str:
        return self.get_species_directory(
            species='',  # not used
            category=self.category,
            path_input_data=self.path_input_data,
            base_name=self.species_base_name,
            isotopologue_name=self.species_isotopologue_name,
            sub_path=self.sub_path
        )

    def get_file_name(self) -> str:
        if self.species_full_name is None:
            self.species_full_name = self.get_full_name()

        return self.join_species_all_info(
            species_name=self.species_full_name,
            natural_abundance='',  # already added in previous step
            charge='',  # already added in previous step
            cloud_info=self.get_cloud_info(),
            source=self.source,
            spectral_info=self.get_spectral_info()
        )

    @classmethod
    def get_file_name_elements(
            cls,
            isotope_separator: str = None,
            isotope_pattern: str = None,
            natural_abundance_string: str = None,
            colliding_species_separator: str = None
    ) -> (str, str, str, str):
        if isotope_separator is None:
            isotope_separator = cls._isotope_separator

        if isotope_pattern is None:
            isotope_pattern = cls._isotope_pattern

        if natural_abundance_string is None:
            natural_abundance_string = cls._natural_abundance_string

        natural_abundance_string = isotope_separator + natural_abundance_string

        if colliding_species_separator is None:
            colliding_species_separator = cls._colliding_species_separator

        return isotope_separator, isotope_pattern, natural_abundance_string, colliding_species_separator

    def get_full_extension(self) -> str:
        return f"{self.extension}.{self._extension_opacity}"

    def get_full_name(self) -> str:
        # Convert list of species into species name
        if self.has_colliding_species:
            colliding_species: list[str] = [''] * len(self.species_list)

            for i, colliding_isotopes in enumerate(self.species_list):
                colliding_species[i] = self._isotope_separator.join(colliding_isotopes)

            species_full_name: str = self._colliding_species_separator.join(colliding_species)
        else:
            species_full_name: str = self._isotope_separator.join(self.species_list)

        if self.natural_abundance:
            natural_abundance = self._natural_abundance_string
        else:
            natural_abundance = ''

        # Add natural abundance string
        species_full_name = self.join_species_all_info(
            species_name=species_full_name,
            natural_abundance=natural_abundance,
            charge='',
            cloud_info='',
            source='',
            spectral_info='',
            spectral_sampling=None,
            wavelength_range=None
        )

        # Add charge string
        species_full_name = self.join_species_all_info(
            species_name=species_full_name,
            natural_abundance='',  # already added
            charge=self.get_charge_string(replace_symbol_with_char=True),
            cloud_info='',
            source='',
            spectral_info='',
            spectral_sampling=None,
            wavelength_range=None
        )

        return species_full_name

    def get_isotopologue_name(self, join: bool = False) -> str:
        return self.get_species_isotopologue_name(
            species_name=self.species_full_name,
            join=join
        )

    @classmethod
    def get_resolving_power_from_string(cls, string: str) -> int:
        return int(string.split(cls._constant_resolving_power, 1)[1])

    @classmethod
    def get_resolving_power_string(cls, resolving_power: [int, float, str]) -> str:
        if isinstance(resolving_power, int) or isinstance(resolving_power, str):
            return f"{cls._constant_resolving_power}{resolving_power}"
        elif isinstance(resolving_power, float):
            return f"{cls._constant_resolving_power}{resolving_power:.0e}".replace(
                'e+', 'e'
            ).replace('e0', 'e')

    @classmethod
    def get_species_base_name(cls, species_full_name: str, join: bool = False) -> str:
        name, natural_abundance, charge, cloud_info, _, _ = cls.split_species_all_info(
            species=species_full_name,
            replace_charge_symbol_with_char=False
        )

        # Remove isotopic numbers
        species_base_name = cls.modify_isotope_numbers(
            species=name,
            mode='remove',
            isotope_separator=cls._isotope_separator,
            isotope_pattern=cls._isotope_pattern.pattern,
            natural_abundance_string=cls._natural_abundance_string,
            colliding_species_separator=cls._colliding_species_separator
        )

        if join:
            return cls.join_species_all_info(
                species_name=species_base_name,
                natural_abundance='',
                charge=charge,
                cloud_info=cloud_info,
                source='',
                spectral_info='',
                spectral_sampling=None,
                wavelength_range=None
            )
        else:
            return species_base_name

    @classmethod
    def get_species_directory(cls, species: str, category: str = None, path_input_data: str = None,
                              base_name: str = None, isotopologue_name: str = None, sub_path: str = None) -> str:
        if category is None:
            category = cls._default_category

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        if base_name is None:
            base_name = cls.get_species_base_name(
                species_full_name=species,
                join=True
            )

        if isotopologue_name is None:
            isotopologue_name = cls.get_species_isotopologue_name(
                species_name=species,
                join=True
            )

        if sub_path is None:
            sub_path = cls._get_sub_path(category)

        return os.path.abspath(
            os.path.join(
                path_input_data,
                sub_path,
                base_name,
                isotopologue_name
            )
        )

    @classmethod
    def get_species_isotopologue_name(cls, species_name: str, join: bool = False) -> str:
        species_isotopologue_name, natural_abundance, charge, cloud_info, _, _ = cls.split_species_all_info(
            species=species_name,
            replace_charge_symbol_with_char=False,
            full=False
        )

        # Re-add natural abundance string to get correct isotopologue name
        species_isotopologue_name = cls.join_species_all_info(
            species_name=species_isotopologue_name,
            natural_abundance=natural_abundance,
            charge='',
            cloud_info='',
            source='',
            spectral_info='',
            spectral_sampling=None,
            wavelength_range=None
        )

        # Add isotopic numbers
        species_isotopologue_name = cls.modify_isotope_numbers(
            species=species_isotopologue_name,
            mode='add'
        )

        if join:
            return cls.join_species_all_info(
                species_name=species_isotopologue_name,
                natural_abundance='',  # already added in previous step
                charge=charge,
                cloud_info=cloud_info,
                source='',
                spectral_info='',
                spectral_sampling=None,
                wavelength_range=None
            )
        else:
            return species_isotopologue_name

    @classmethod
    def get_species_scientific_name(cls, species: str) -> str:
        name = cls.split_species_all_info(species, replace_charge_symbol_with_char=False)[0]
        name = cls.modify_isotope_numbers(
            species=name,
            mode='scientific',
            isotope_separator=cls._isotope_separator,
            isotope_pattern=cls._isotope_pattern.pattern,
            natural_abundance_string=cls._natural_abundance_string,
            colliding_species_separator=cls._colliding_species_separator
        )

        # Remove isotopic numbers
        return rf"{name}"

    def get_spectral_info(self) -> str:
        spectral_info: str = ''

        if self.spectral_sampling_type not in self._spectral_sampling_types:
            raise ValueError(
                f"spectral sampling type must be {list_str2str(self._spectral_sampling_types)}, "
                f"but was '{self.spectral_sampling_type}'"
            )

        if self.spectral_sampling_type == self._constant_resolving_power:
            spectral_info += self.get_resolving_power_string(self.spectral_sampling)
        else:
            spectral_info += (
                f"{self.spectral_sampling_type}{self.spectral_sampling}"
            )

        spectral_info += (
            f"{self._wavelength_range_separator}"
            f"{self.wavelength_min}{self._wavelength_separator}{self.wavelength_max}{self._wavelength_units}"
        )

        return spectral_info

    @classmethod
    def join_species_all_info(
            cls,
            species_name: str,
            natural_abundance: str = '',
            charge: str = '',
            cloud_info: str = '',
            source: str = '',
            spectral_info: str = '',
            spectral_sampling: str = None,
            wavelength_range: str = None
    ) -> str:
        if natural_abundance != '':
            species_name += cls._isotope_separator + natural_abundance

        if charge != '':
            species_name += cls._charge_separator + charge

        species_name += cloud_info  # cloud info has no separator

        if source != '':
            species_name += cls._source_separator + source

        if spectral_info != '':
            if spectral_sampling is not None or wavelength_range is not None:
                raise ValueError(
                    f"cannot give both complete spectral info ('{spectral_info}'), "
                    f"and resolution ('{spectral_sampling}') + range ('{wavelength_range}')\n"
                    f"Set resolution_filename and range_filename to None, or set spectral_info to an empty string"
                )

            species_name += cls._spectral_information_separator + spectral_info
        elif spectral_sampling is not None or wavelength_range is not None:
            if spectral_sampling is None:
                raise ValueError("both resolution_filename and range_filename must be not None, "
                                 "but resolution_filename is None")

            if wavelength_range is None:
                raise ValueError("both resolution_filename and range_filename must be not None, "
                                 "but range_filename is None")

            species_name += cls._spectral_information_separator + cls._join_spectral_information(
                spectral_sampling=spectral_sampling,
                wavelength_range=wavelength_range
            )

        return species_name

    def load(self, file: str):
        warnings.warn(
            "loading an opacity is not implemented yet.\n"
            "Nothing to be done."
        )

    @classmethod
    def modify_isotope_numbers(cls, species: str, mode: str, isotope_separator: str = None, isotope_pattern: str = None,
                               natural_abundance_string: str = None, colliding_species_separator: str = None) -> str:
        """Add or remove isotope numbers from a species.
        Note that using improper isotope separation can lead to incorrect results (e.g. H218O -> 1H218-16O).

        Args:
            species:
                Species name. Can also be a species collision (e.g. H2--He).
            mode:
                Can be 'add', 'remove', or 'scientific'.
                In 'add' mode, add the isotope number of each of the species will be added to the species name, and each
                isotope is separated with a '-'. By default, the main isotope number is used. If partial isotope
                information is provided (e.g. 13C2H2, H2-18O, ...), use the main isotope number on the isotopes for
                which no information is given (e.g. 13C2-1H2, 1H2-18O, ...).
                In 'remove' mode, remove all isotope numbers (e.g. 13C2-1H2 -> C2H2).
                In 'scientific' mode, convert the species into scientific notation using LaTeX formatting.
            isotope_separator:
                Isotope separator. If None, use the default value.
            isotope_pattern:
                Regex pattern string to detect isotopes in a species name. If None, use the default value.
            natural_abundance_string:
                Natural abundance string. If None, use the default value.
            colliding_species_separator:
                Colliding species separator. If None, use the default value.

        Returns:
            The species name with added or removed isotope information.
        """
        isotope_separator, isotope_pattern, natural_abundance_string, colliding_species_separator = (
            cls.get_file_name_elements(
                isotope_separator=isotope_separator,
                isotope_pattern=isotope_pattern,
                natural_abundance_string=natural_abundance_string,
                colliding_species_separator=colliding_species_separator,
            )
        )

        if colliding_species_separator is None:
            colliding_species_separator = cls._colliding_species_separator

        if mode == 'add':
            _isotope_separator = isotope_separator
        elif mode == 'remove' or mode == 'scientific':
            _isotope_separator = ''  # no separator in remove mode
        else:
            raise ValueError(f"iter isotopes mode must be 'add'|'remove'|'scientific', but was '{mode}'")

        # Handle natural abundance case
        if natural_abundance_string in species:
            species = species.rsplit(natural_abundance_string, 1)[0]
            colliding_species = species.split(colliding_species_separator)  # CIA case

            for i, __species in enumerate(colliding_species):  # for each colliding species
                matches = re.findall(isotope_pattern, __species)

                __species = [group[1] + group[2] for group in matches]
                colliding_species[i] = _isotope_separator.join(__species)

            species = colliding_species_separator.join(colliding_species)

            if mode == 'add':
                return species + natural_abundance_string
            elif mode == 'remove' or mode == 'scientific':
                species.replace(natural_abundance_string, '')

                if mode == 'remove':
                    return species
            else:
                raise ValueError(f"iter isotopes mode must be 'add'|'remove', but was '{mode}'")

        # Handle regular case
        colliding_species = species.split(colliding_species_separator)  # CIA case

        for i, __species in enumerate(colliding_species):  # for each colliding species
            if isotope_separator in __species:
                isotopes = __species.split(isotope_separator)
            else:
                isotopes = [__species]

            for j, isotope in enumerate(isotopes):  # for each explicitly separated isotope in the species
                matches = cls.__modify_isotope_string(
                    isotope=isotope,
                    mode=mode,
                    isotope_pattern=isotope_pattern
                )

                isotopes[j] = _isotope_separator.join(matches)  # join non-separated isotopes with the separator

            _species = _isotope_separator.join(isotopes)  # join isotopes to rebuild species

            # Merge contiguous matches containing the same element (but presumably different isotopes)
            if mode == 'remove':
                _species = cls._merge_contiguous_isotopes(
                    species=_species,
                    isotope_separator=_isotope_separator
                )

            colliding_species[i] = _species

        if mode == 'scientific':
            cia_separator = f"${isotope_separator}$"
        else:
            cia_separator = colliding_species_separator

        return cia_separator.join(colliding_species)  # join species to rebuild collision

    def save(self, file: str):
        warnings.warn(
            "saving an opacity is not implemented yet.\n"
            "Nothing to be done."
        )

    @classmethod
    def split_cloud_info(cls, cloud_info: str) -> (str, str, str):
        matter_state_string: str = cloud_info[:3]  # "(s)" or "(l)"

        matter_state: str = ''
        solid_structure: str = ''
        solid_structure_id: str = ''

        for _matter_state in cls._condensed_matter_states:
            if _matter_state == matter_state_string:
                _, solid_structure = cloud_info.split(cls._solid_structure_separator, 1)
                matter_state = _matter_state
                break

        if solid_structure != '':
            solid_structure, solid_structure_id = solid_structure.split(cls._solid_structure_separator, 1)

        return matter_state, solid_structure, solid_structure_id

    @classmethod
    def split_species_all_info(cls, species: str, replace_charge_symbol_with_char: bool = False, full: bool = False):
        name, spectral_info = cls.split_species_spectral_info(species)
        name, source = cls.split_species_source(name)
        name, cloud_info = cls.split_species_cloud_info(name)

        # Check for repeated ion symbol, eases the user's understanding of future error
        if len(re.findall(r'^.*([+\-pm])([+\-pm])$', name)) > 0:
            raise ValueError(f"invalid species formula '{name}', "
                             f"multiple consecutive charge symbols found (+, -, p, m)")

        # Extract charge string
        name, charge = cls.split_species_charge(name, replace_symbol_with_char=replace_charge_symbol_with_char)

        natural_abundance_string = cls._isotope_separator + cls._natural_abundance_string

        if f"{natural_abundance_string}" in name:
            name = name.replace(f'{natural_abundance_string}', '')
            natural_abundance = natural_abundance_string
        else:
            natural_abundance = ''

        if full:
            matter_state: str = ''
            solid_structure: str = ''
            solid_structure_id: str = ''
            spectral_sampling_info: str = ''
            wavelength_range_info: str = ''
            spectral_sampling_type: str = ''
            spectral_sampling_value: str = ''
            wavelength_min: str = ''
            wavelength_max: str = ''

            if len(cloud_info) > 0:
                matter_state, solid_structure, solid_structure_id = cls.split_cloud_info(cloud_info)

            if matter_state == '':
                matter_state = cls._gas_matter_state

            if len(spectral_info) > 0:
                spectral_sampling_info, wavelength_range_info = cls.split_spectral_info(spectral_info)

            if len(spectral_sampling_info) > 0:
                spectral_sampling_type, spectral_sampling_value = cls.split_spectral_sampling_info(
                    spectral_sampling_info
                )

            if len(wavelength_range_info) > 0:
                wavelength_min, wavelength_max = cls.split_wavelength_range_info(wavelength_range_info)

            return (
                name, natural_abundance, charge,
                matter_state, solid_structure, solid_structure_id,
                source,
                spectral_sampling_type, spectral_sampling_value,
                wavelength_min, wavelength_max
            )

        return name, natural_abundance, charge, cloud_info, source, spectral_info

    @classmethod
    def split_species_charge(cls, species: str, replace_symbol_with_char: bool = False) -> (str, str):
        # Extract charge symbol
        charge_pattern_match = re.findall(cls._charge_pattern, species)
        charge = ''
        name = copy.deepcopy(species)

        if species[-2:] in {'Tm', 'Am', 'Cm',
                            'Fm'}:  # prevent matches with atomic Thulium, Americium, Curium and Fermium
            return name, ''

        if len(charge_pattern_match) == 1:
            charge = charge_pattern_match[0][0]

        # Change charge format
        if replace_symbol_with_char:  # replace symbols (+/-) with characters (p/m)
            for symbol, char in cls._charges.items():
                charge = charge.replace(symbol, char)
        else:  # replace characters (p/m) with symbols (+/-)
            for symbol, char in cls._charges.items():
                charge = charge.replace(char, symbol)

        charge = charge[len(cls._charge_separator):]  # remove leading separator

        # Temporarily remove charge symbol to remove isotopic numbers
        if len(charge) > 0:
            name = species.split(charge, 1)[0]

        return name, charge

    @staticmethod
    def split_species_cloud_info(species: str) -> (str, str):
        cloud_info = ''
        _split = species.split('(', 1)

        if len(_split) == 1:
            name = _split[0]
        else:
            name, cloud_info = _split
            cloud_info = '(' + cloud_info  # re-add parenthesis lost during split

        return name, cloud_info

    @classmethod
    def split_species_source(cls, species: str) -> (str, str):
        _split = species.split(cls._source_separator, 1)

        if len(_split) == 1:
            name = _split[0]
            source = ''
        else:
            name, source = _split

        return name, source

    @classmethod
    def split_species_spectral_info(cls, species: str) -> (str, str):
        _split = species.split(cls._spectral_information_separator, 1)

        if len(_split) == 1:
            name = _split[0]
            spectral_info = ''
        else:
            name, spectral_info = _split

        return name, spectral_info

    @classmethod
    def split_spectral_sampling_info(cls, spectral_sampling_info: str) -> (str, str):
        spectral_sampling_type = None
        spectral_sampling_value = None

        for spectral_sampling_type in cls._spectral_sampling_types:
            if spectral_sampling_type in spectral_sampling_info:
                spectral_sampling_value = spectral_sampling_info.split(spectral_sampling_type, 1)[1]
                break

        if spectral_sampling_value is None:
            raise ValueError(
                f"spectral sampling type must be {list_str2str(cls._spectral_sampling_types)}, "
                f"but spectral sampling info was '{spectral_sampling_info}'"
            )

        return spectral_sampling_type, spectral_sampling_value

    @classmethod
    def split_spectral_info(cls, spectral_info: str) -> (str, str):
        _split = spectral_info.split(cls._wavelength_range_separator, 1)

        if len(_split) == 1:
            raise ValueError(
                f"no wavelength range information detected in spectral information '{spectral_info}', "
                f"wavelength range information must be separated from spectral sampling information using "
                f"'{cls._wavelength_range_separator}'"
            )
        else:
            spectral_sampling_info, wavelength_range_info = _split

        return spectral_sampling_info, wavelength_range_info

    @classmethod
    def split_wavelength_range_info(cls, wavelength_range_info: str) -> (str, str):
        _split = wavelength_range_info.split(cls._wavelength_separator, 1)

        if len(_split) == 1:
            raise ValueError(
                f"no wavelength separator detected in wavelength range '{wavelength_range_info}', "
                f"wavelength boundaries must be separated from spectral sampling information using "
                f"'{cls._wavelength_separator}'"
            )
        else:
            wavelength_min, wavelength_max = _split

        wavelength_max = wavelength_max.replace(cls._wavelength_units, '')

        return wavelength_min, wavelength_max

    @staticmethod
    def write(**kwargs):
        warnings.warn(
            "writing function for this Opacity object has not been implemented.\n"
            "Nothing to be done."
        )


class CIAOpacity(Opacity):
    _default_category: str = 'cia_opacities'
    _default_extension: str = 'ciatable'
    _default_wavelength_range: tuple[float, float] = (0.1, 250.0)
    _default_resolving_power: float = Opacity._default_cia_resolving_power

    def __init__(
            self,
            species_list,
            natural_abundance: bool = False,
            charge: int = 0,
            source: str = 'unknown',
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = 830,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            species_full_name: str = None,
            species_base_name: str = None,
            species_isotopologue_name: str = None,
            full_extension: str = None,
            file_name: str = None,
            sub_path: str = None,
            absolute_path: str = None
    ):
        super().__init__(
            species_list=species_list,
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=super()._gas_matter_state,
            solid_structure=None,
            solid_structure_id=None,
            path_input_data=path_input_data,
            category=self._default_category,
            species_full_name=species_full_name,
            species_cloud_info=None,
            species_base_name=species_base_name,
            species_isotopologue_name=species_isotopologue_name,
            extension=self._default_extension,
            full_extension=full_extension,
            file_name=file_name,
            sub_path=sub_path,
            absolute_path=absolute_path
        )

    @classmethod
    def from_species(
            cls,
            species: str,
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_line_by_line_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            _, _, _, _  # spectral info
        ) = cls.split_species_all_info(
            species=species,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def from_species_fullname(
            cls,
            species_fullname: str,
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            spectral_sampling_type, spectral_sampling,
            wavelength_min, wavelength_max
        ) = cls.split_species_all_info(
            species=species_fullname,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species_fullname,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def write(
            cls,
            file: str,
            colliding_species: tuple[str, ...],
            alphas: npt.NDArray[float],
            wavenumbers: npt.NDArray[float],
            pressures: npt.NDArray[float],
            temperatures: npt.NDArray[float],
            molar_mass: float = None,
            doi: str = '',
            contributor: str = '',
            description: str = '',
            date_id: str = None,
            n_g: int = None,
            wavelength_range: tuple[float] = None
    ):
        molar_mass, date_id = cls._before_write(
            temperature_grid_type='regular',
            molar_mass=molar_mass,
            species_name=colliding_species,
            date_id=date_id
        )

        expected_shape = (temperatures.size, wavenumbers.size)

        if alphas.shape != expected_shape:
            raise ValueError(
                f"k-coefficients must be of shape (pressures, temperatures, wavenumbers bin centers, g-space), "
                f"i.e. {expected_shape}, but is of shape {alphas.shape}"
            )

        with h5py.File(file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                shape=(1,),
                data=doi
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['additional_description'] = description

            dataset = fh5.create_dataset(
                name='Date_ID',
                shape=(1,),
                data=date_id
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
                name='alpha',
                data=alphas
            )
            dataset.attrs['long_name'] = 'Table of monochromatic absorption with axes (temperature, wavenumber)'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='t',
                data=temperatures
            )
            dataset.attrs['long_name'] = 'Temperature grid'
            dataset.attrs['units'] = 'K'

            dataset = fh5.create_dataset(
                name='mol_mass',
                data=molar_mass
            )
            dataset.attrs['long_name'] = 'Masses of the colliding species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='mol_name',
                data=colliding_species
            )
            dataset.attrs['long_name'] = 'Names of the colliding species described'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([wavelength_range[0], wavelength_range[1]]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumbers.min(), wavenumbers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'


class CloudOpacity(Opacity):
    _default_category: str = 'clouds_opacities'
    _default_extension: str = 'cotable'
    _default_resolving_power: float = Opacity._default_cloud_resolving_power
    _default_wavelength_range: tuple[float, float] = (0.1, 250.0)

    _default_file_names: LockedDict[str, str] = LockedDict.build_and_lock({
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

    def __init__(
            self,
            species_list,
            natural_abundance: bool = False,
            charge: int = 0,
            source: str = 'unknown',
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_cloud_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            matter_state: str = Opacity._solid_matter_state,
            solid_structure: str = 'crystalline',
            solid_structure_id: str = '000',
            path_input_data: str = None,
            species_full_name: str = None,
            species_cloud_info: str = None,
            species_base_name: str = None,
            species_isotopologue_name: str = None,
            full_extension: str = None,
            file_name: str = None,
            sub_path: str = None,
            absolute_path: str = None
    ):
        super().__init__(
            species_list=species_list,
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=matter_state,
            solid_structure=solid_structure,
            solid_structure_id=solid_structure_id,
            path_input_data=path_input_data,
            category=self._default_category,
            species_full_name=species_full_name,
            species_cloud_info=species_cloud_info,
            species_base_name=species_base_name,
            species_isotopologue_name=species_isotopologue_name,
            extension=self._default_extension,
            full_extension=full_extension,
            file_name=file_name,
            sub_path=sub_path,
            absolute_path=absolute_path
        )

    @classmethod
    def from_species(
            cls,
            species: str,
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_cloud_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            matter_state, solid_structure, solid_structure_id,
            source,
            _, _, _, _  # spectral info
        ) = cls.split_species_all_info(
            species=species,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=matter_state,
            solid_structure=solid_structure,
            solid_structure_id=solid_structure_id,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def from_species_fullname(
            cls,
            species_fullname: str,
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            matter_state, solid_structure, solid_structure_id,
            source,
            spectral_sampling_type, spectral_sampling,
            wavelength_min, wavelength_max
        ) = cls.split_species_all_info(
            species=species_fullname,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species_fullname,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=matter_state,
            solid_structure=solid_structure,
            solid_structure_id=solid_structure_id,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def get_aliases(cls, name: str) -> str:
        cloud_opacities_path = str(os.path.join(
            petitradtrans_config_parser.get_input_data_path(),
            get_input_data_subpaths()['clouds_opacities']
        ))

        cloud_directories = []

        if os.path.isdir(cloud_opacities_path):
            cloud_directories = [
                f.path.rsplit(os.path.sep, 1)[1] for f in os.scandir(cloud_opacities_path) if f.is_dir()
            ]

        _name, spectral_info = cls.split_species_spectral_info(name)
        _name, method = cls.split_species_source(_name)
        _name, cloud_info = cls.split_species_cloud_info(_name)

        matter_state, structure, space_group = cls.split_cloud_info(cloud_info)

        if matter_state == cls._solid_matter_state and structure == cls._crystalline_structure:
            # No space group in name, try to find relevant one in the cloud opacities directory
            if space_group == '' and structure != '':
                matches = []

                for cloud_directory in cloud_directories:
                    cloud_directory_name, cloud_directory_info = cls.split_species_cloud_info(cloud_directory)

                    if cloud_directory_name == _name and cloud_info in cloud_directory_info:
                        matches.append(cloud_directory)

                # Try to look into the Keeper library if nothing was found locally
                if len(matches) == 0:
                    keeper_directory_string = '&mode=list'  # directories elements on Keeper end with this string

                    keeper_cloud_directories = get_keeper_files_url_paths(
                        path=cloud_opacities_path,
                        ext=keeper_directory_string
                    )

                    # Remove the Keeper directory string
                    keeper_cloud_directories = {
                        key.rsplit(keeper_directory_string, 1)[0]
                        for key in keeper_cloud_directories
                    }

                    matches = [
                        cloud_directory
                        for cloud_directory in keeper_cloud_directories
                        if _name + cloud_info in cloud_directory
                    ]

                if len(matches) == 1:
                    _, _cloud_info = cls.split_species_cloud_info(matches[0])
                    _, _, space_group = cls.split_cloud_info(_cloud_info)
                    cloud_info += cls._solid_structure_separator + space_group
                elif len(matches) > 1:
                    space_group_example = ''

                    valid_opacities = set()

                    for match in matches:
                        space_group = match.rsplit(cls._solid_structure_separator, 1)[1]

                        if 'crystalline' in match:
                            space_group_match = re.match(r'^\d{3}$', space_group)

                            if not space_group_match:
                                warnings.warn(
                                    f"crystalline cloud opacity '{os.path.join(cloud_opacities_path, match)}' "
                                    f"does not seem to contain "
                                    f"a valid space group (should be 3 digits, was '{space_group}')\n"
                                    f"If you have legacy or custom opacities, ensure to add a space group to them "
                                    f"(see "
                                    f"https://petitradtrans.readthedocs.io/en/dev/content/available_opacities.html#id79"
                                    f")."
                                )
                            else:
                                space_group_example = space_group
                                valid_opacities.add(match)
                        else:
                            valid_opacities.add(match)

                    available = list_str2str(
                        [
                            match.rsplit(cls._solid_structure_separator, 1)[1]
                            for match in valid_opacities
                        ]
                    )

                    if space_group_example != '':
                        cloud_info += cls._solid_structure_separator + space_group_example

                    if spectral_info != '':
                        spectral_info = cls._spectral_information_separator + spectral_info

                    if method != '':
                        method = cls._source_separator + method

                    _name = _name + cloud_info + method + spectral_info

                    raise FileExistsError(
                        f"more than one solid condensate cloud with name '{name}'\n"
                        f"Space groups are not mandatory only if a unique cloud opacity with this name exists in your "
                        f"input_data directory.\n"
                        f"Add a space group to your cloud name (e.g., '{_name}')\n"
                        f"Available space groups with this name: {available}"
                    )

        if spectral_info != '':
            spectral_info = cls._spectral_information_separator + spectral_info

        if method != '':
            method = cls._source_separator + method

        # Return NatAbund if no isotope information has been provided (override def. case returning main isotopologue)
        if cls._natural_abundance_string not in name and not cls._has_isotope(name):
            if not any([
                condensed_matter_state in name
                for condensed_matter_state in cls._condensed_matter_states
            ]):
                raise ValueError(f"cloud species name '{name}' lacks condensed matter state information\n"
                                 f"For liquid particles, cloud species names must include '(l)' (e.g. 'H2O(l)').\n"
                                 f"For solid particles, the cloud species must include '(s)_crystalline' for crystals "
                                 f"or '(s)_amorphous' for amorphous solids (e.g. 'MgSiO3(s)_crystalline').")

            _name = cls.modify_isotope_numbers(
                species=_name + '-' + cls._natural_abundance_string,
                mode='add',
                isotope_separator=cls._isotope_separator,
                isotope_pattern=cls._isotope_pattern.pattern,
                natural_abundance_string=cls._natural_abundance_string,
                colliding_species_separator=cls._colliding_species_separator
            )
            name = _name + cloud_info + method + spectral_info
        else:
            name = _name + cloud_info + method + spectral_info

        return name

    @classmethod
    def write(
            cls,
            file: str,
            species_name: str,
            absorption_opacities: npt.NDArray[float],
            scattering_opacities: npt.NDArray[float],
            asymmetry_parameters: npt.NDArray[float],
            particles_densities: npt.NDArray[float],
            particles_radius_bin_centers: npt.NDArray[float],
            particles_radius_bin_edges: npt.NDArray[float],
            wavenumbers: npt.NDArray[float],
            doi: str = '',
            contributor: str = '',
            description: str = '',
            date_id: str = None,
            wavelength_range: tuple[float] = None
    ):
        _, date_id = cls._before_write(
            temperature_grid_type='regular',  # no temperature dependency
            molar_mass=0,  # molar mass is not used
            species_name=species_name,
            date_id=date_id
        )

        expected_shape = (particles_radius_bin_centers.size, wavenumbers.size)

        if particles_radius_bin_edges.size != particles_radius_bin_centers.size + 1:
            raise ValueError(
                f"particles radius bin edges must be of size (particles radius bin centers + 1), "
                f"i.e. ({particles_radius_bin_centers.size + 1}), "
                f"but is of size ({particles_radius_bin_edges.size})"
            )

        if absorption_opacities.shape != expected_shape:
            raise ValueError(
                f"absorption opacities must be of shape (particles radius bin centers, wavenumbers), "
                f"i.e. {expected_shape}, but is of shape {absorption_opacities.shape}"
            )

        if scattering_opacities.shape != expected_shape:
            raise ValueError(
                f"scattering opacities must be of shape (particles radius bin centers, wavenumbers), "
                f"i.e. {expected_shape}, but is of shape {absorption_opacities.shape}"
            )

        if asymmetry_parameters.shape != expected_shape:
            raise ValueError(
                f"asymmetry parameters must be of shape (particles radius bin centers, wavenumbers), "
                f"i.e. {expected_shape}, but is of shape {absorption_opacities.shape}"
            )

        with h5py.File(file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                shape=(1,),
                data=doi
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['additional_description'] = str(description)

            dataset = fh5.create_dataset(
                name='Date_ID',
                shape=(1,),
                data=date_id
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
                data=absorption_opacities
            )
            dataset.attrs['long_name'] = 'Table of the absorption opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='scattering_opacities',
                data=scattering_opacities
            )
            dataset.attrs['long_name'] = 'Table of the scattering opacities with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'cm^2.g^-1'

            dataset = fh5.create_dataset(
                name='asymmetry_parameters',
                data=asymmetry_parameters
            )
            dataset.attrs['long_name'] = 'Table of the asymmetry parameters with axes (particle radius, wavenumber)'
            dataset.attrs['units'] = 'None'

            dataset = fh5.create_dataset(
                name='particles_density',
                data=particles_densities
            )
            dataset.attrs['long_name'] = 'Average density of the cloud particles'
            dataset.attrs['units'] = 'g.cm^-3'

            dataset = fh5.create_dataset(
                name='mol_name',
                shape=(1,),
                data=species_name
            )
            dataset.attrs['long_name'] = 'Name of the species described, "(c)" indicates that it has condensed'

            dataset = fh5.create_dataset(
                name='particles_radii',
                data=particles_radius_bin_centers
            )
            dataset.attrs['long_name'] = 'Particles average radius grid'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='particle_radius_bins',
                data=particles_radius_bin_edges
            )
            dataset.attrs['long_name'] = 'Particles average radius grid bins'
            dataset.attrs['units'] = 'cm'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([wavelength_range[0], wavelength_range[1]]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumbers.min(), wavenumbers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'


class CorrelatedKOpacity(Opacity):
    _default_category: str = 'correlated_k_opacities'
    _default_extension: str = 'ktable'
    _default_rebinning_wavelength_range: tuple[float, float] = (0.1, 251.0)  # microns
    _default_resolving_power: float = Opacity._default_correlated_k_resolving_power
    _default_wavelength_range: tuple[float, float] = (0.1, 250.0)  # microns

    def __init__(
            self,
            species_list,
            natural_abundance: bool = False,
            charge: int = 0,
            source: str = 'unknown',
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_correlated_k_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            species_full_name: str = None,
            species_base_name: str = None,
            species_isotopologue_name: str = None,
            full_extension: str = None,
            file_name: str = None,
            sub_path: str = None,
            absolute_path: str = None
    ):
        super().__init__(
            species_list=species_list,
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=super()._gas_matter_state,
            solid_structure=None,
            solid_structure_id=None,
            path_input_data=path_input_data,
            category=self._default_category,
            species_full_name=species_full_name,
            species_cloud_info=None,
            species_base_name=species_base_name,
            species_isotopologue_name=species_isotopologue_name,
            extension=self._default_extension,
            full_extension=full_extension,
            file_name=file_name,
            sub_path=sub_path,
            absolute_path=absolute_path
        )

    @classmethod
    def _get_default_rebinning_wavenumber_grid(cls, resolving_power: float) -> npt.NDArray[float]:
        wavelengths_boundaries = cls._default_rebinning_wavelength_range
        n_spectral_points = int(
            resolving_power * np.log(
                cls._default_rebinning_wavelength_range[1]
                / cls._default_rebinning_wavelength_range[0]
            ) + 1
        )

        return np.logspace(
            np.log10(1e4 / wavelengths_boundaries[1]),  # um to cm-1
            np.log10(1e4 / wavelengths_boundaries[0]),  # um to cm-1
            n_spectral_points
        )

    @classmethod
    def exo_k_multiple_rebin_from_species(cls, species: list[str], resolving_power: float, rewrite: bool = False):
        """
        This function uses exo-k to bin the c-k table of a
        multiple species to a desired (lower) spectral resolution.

        Args:
            species : string
                The name of the species
            resolving_power : int
                The desired spectral resolving power.
            rewrite : bool
                If True, rewrite the rebinned files even if they already exist.
        """
        ck_paths: list = []

        print(f"Resolving power: {resolving_power}")

        for s in species:
            ck_paths.append(cls.find(
                species=s,
                category=cls._default_category,
                find_all=False,
                search_online=True
            ))

            print(f" Re-binned opacities: '{ck_paths[-1]}'")

        cls.exo_k_multiple_rebin(
            resolving_power=int(resolving_power),
            input_files=ck_paths,
            rewrite=rewrite
        )

    @classmethod
    def exo_k_multiple_rebin(cls, input_files: list[str], resolving_power: float, rewrite=False):
        if input_files is None:
            input_files = []

        wavenumber_grid = cls._get_default_rebinning_wavenumber_grid(resolving_power)

        success = False

        # Do the rebinning, loop through species
        for input_file in input_files:
            # Output files
            state = cls.exo_k_rebin(
                input_file=input_file,
                resolving_power=resolving_power,
                wavenumber_grid=wavenumber_grid,
                rewrite=rewrite
            )

            if state is None:
                success = True
            elif state == -1:
                success = False
                break
            else:
                raise NotImplementedError(
                    f"exo-k rebinning function returned state '{state}', "
                    f"but 'None' (standard) or '1' (error) was expected"
                )

        if success:
            print("Successfully binned down all k-tables\n")

    @classmethod
    def exo_k_rebin(cls, input_file: str, resolving_power: float, wavenumber_grid: npt.NDArray[float] = None,
                    rewrite: bool = False) -> [int, None]:
        try:
            import exo_k
        except ImportError:
            # Only raise a warning to give a chance to download the binned
            warnings.warn(
                "binning down of opacities requires exo_k to be installed, no binning down has been performed")
            return -1

        species = input_file.rsplit('.' + cls._default_extension, 1)[0]
        species = species.rsplit(os.path.sep, 1)[1]
        species, spectral_info = cls.split_species_spectral_info(species)

        sampling_info, wavelength_range_info = cls.split_spectral_info(spectral_info)
        file_sampling_type = cls.split_spectral_sampling_info(sampling_info)[0]

        if file_sampling_type != cls._constant_resolving_power:
            raise ValueError(
                f"rebinning with exo-k must be done with opacities sampled at constant resolving power "
                f"(flag: '{cls._constant_resolving_power}' in input file '{input_file}'), "
                f"but the sampling type flag was '{file_sampling_type}'\n"
                f"Verify if the input file name flag is correct, or use an opacity sampled at constant resolving power."
            )

        if rank == 0:  # prevent race condition when writing the binned down file during multi-processes execution
            if wavenumber_grid is None:
                wavenumber_grid = cls._get_default_rebinning_wavenumber_grid(resolving_power)

            wavelength_min, wavelength_max = cls.split_wavelength_range_info(wavelength_range_info)

            new_opacity = cls.from_species(
                species=species,
                spectral_sampling_type='R',
                spectral_sampling=resolving_power,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
                category=cls._default_category
            )

            # Output files
            output_file = new_opacity.absolute_path

            if os.path.isfile(output_file) and not rewrite:
                print(f"File '{output_file}' already exists, skipping re-binning...")
            else:
                print(f"Rebinning file '{input_file}' to R = {resolving_power}... ", end=' ')
                # Use Exo-k to rebin to low-res
                tab = exo_k.Ktable(filename=input_file, remove_zeros=True)
                tab.bin_down(wavenumber_grid)
                print('Done.')

                print(f" Writing binned down file '{output_file}'... ", end=' ')
                tab.write_hdf5(output_file)
                print('Done.')

                print(f"Successfully binned down k-table into '{output_file}' (R = {resolving_power})")

        if comm is not None:  # wait for the main process to finish the binning down
            comm.barrier()

    @classmethod
    def from_species(
            cls,
            species: str,
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_correlated_k_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            _, _, _, _  # spectral info
        ) = cls.split_species_all_info(
            species=species,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def from_species_fullname(
            cls,
            species_fullname: str,
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            spectral_sampling_type, spectral_sampling,
            wavelength_min, wavelength_max
        ) = cls.split_species_all_info(
            species=species_fullname,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species_fullname,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def write(
            cls,
            file: str,
            species_name: str,
            k_coefficients: npt.NDArray[float],
            wavenumber_bin_centers: npt.NDArray[float],
            wavenumber_bin_edges: npt.NDArray[float],
            pressures: npt.NDArray[float],
            temperatures: npt.NDArray[float],
            temperature_grid_type: str,
            g_weights: npt.NDArray[float],
            g_samples: npt.NDArray[float],
            molar_mass: float = None,
            doi: str = '',
            contributor: str = '',
            description: str = '',
            date_id: str = None,
            wavelength_range: tuple[float] = None
    ):
        molar_mass, date_id = cls._before_write(
            temperature_grid_type=temperature_grid_type,
            molar_mass=molar_mass,
            species_name=species_name,
            date_id=date_id
        )

        if wavenumber_bin_edges.size != wavenumber_bin_centers.size + 1:
            raise ValueError(
                f"wavenumber bin edges must be of size (wavenumber bin centers + 1), "
                f"i.e. ({wavenumber_bin_centers.size + 1}), but is of size ({wavenumber_bin_edges.size})"
            )

        expected_shape = (pressures.size, temperatures.size, wavenumber_bin_centers.size, g_samples.size)

        if k_coefficients.shape != expected_shape:
            raise ValueError(
                f"k-coefficients must be of shape (pressures, temperatures, wavenumbers bin centers, g-space), "
                f"i.e. {expected_shape}, but is of shape {k_coefficients.shape}"
            )

        with h5py.File(file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                shape=(1,),
                data=doi
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['contributor'] = str(contributor)
            dataset.attrs['additional_description'] = str(description)

            dataset = fh5.create_dataset(
                name='Date_ID',
                shape=(1,),
                data=date_id
            )
            dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                         'at which the table has been created, ' \
                                         'along with the version of petitRADTRANS'

            dataset = fh5.create_dataset(
                name='bin_centers',
                data=wavenumber_bin_centers
            )
            dataset.attrs['long_name'] = 'Centers of the wavenumber bins'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='bin_edges',
                data=wavenumber_bin_edges
            )
            dataset.attrs['long_name'] = 'Separations between the wavenumber bins'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='kcoeff',
                data=k_coefficients
            )
            dataset.attrs['long_name'] = ('Table of the k-coefficients with axes '
                                          '(pressure, temperature, wavenumber, g-space)')
            dataset.attrs['units'] = 'cm^2/molecule'

            dataset = fh5.create_dataset(
                name='method',
                shape=(1,),
                data='petit_samples'
            )
            dataset.attrs['long_name'] = 'Name of the method used to sample g-space'

            dataset = fh5.create_dataset(
                name='mol_mass',
                shape=(1,),
                data=float(molar_mass)
            )
            dataset.attrs['long_name'] = 'Mass of the species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='mol_name',
                shape=(1,),
                data=cls.get_species_base_name(species_full_name=species_name, join=True)
            )
            dataset.attrs['long_name'] = 'Name of the species described'

            dataset = fh5.create_dataset(
                name='ngauss',
                data=g_samples.size
            )
            dataset.attrs['long_name'] = 'Number of points used to sample the g-space'

            dataset = fh5.create_dataset(
                name='p',
                data=pressures
            )
            dataset.attrs['long_name'] = 'Pressure grid'
            dataset.attrs['units'] = 'bar'

            dataset = fh5.create_dataset(
                name='samples',
                data=g_samples
            )
            dataset.attrs['long_name'] = 'Abscissas used to sample the k-coefficients in g-space'

            dataset = fh5.create_dataset(
                name='t',
                data=temperatures
            )
            dataset.attrs['long_name'] = 'Temperature grid'
            dataset.attrs['units'] = 'K'

            dataset = fh5.create_dataset(
                name='temperature_grid_type',
                shape=(1,),
                data=temperature_grid_type
            )
            dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                         '(same temperatures for all pressures) or "pressure-dependent"'

            dataset = fh5.create_dataset(
                name='weights',
                data=g_weights
            )
            dataset.attrs['long_name'] = 'Weights used in the g-space quadrature'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([wavelength_range[0], wavelength_range[1]]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumber_bin_centers.min(), wavenumber_bin_centers.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'


class LineByLineOpacity(Opacity):
    _default_category: str = 'line_by_line_opacities'
    _default_extension: str = 'xsec'
    _default_resolving_power: float = Opacity._default_line_by_line_resolving_power
    _default_wavelength_range: tuple[float, float] = (0.3, 28)

    def __init__(
            self,
            species_list,
            natural_abundance: bool = False,
            charge: int = 0,
            source: str = 'unknown',
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_line_by_line_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            species_full_name: str = None,
            species_base_name: str = None,
            species_isotopologue_name: str = None,
            full_extension: str = None,
            file_name: str = None,
            sub_path: str = None,
            absolute_path: str = None
    ):
        super().__init__(
            species_list=species_list,
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            matter_state=super()._gas_matter_state,
            solid_structure=None,
            solid_structure_id=None,
            path_input_data=path_input_data,
            category=self._default_category,
            species_full_name=species_full_name,
            species_cloud_info=None,
            species_base_name=species_base_name,
            species_isotopologue_name=species_isotopologue_name,
            extension=self._default_extension,
            full_extension=full_extension,
            file_name=file_name,
            sub_path=sub_path,
            absolute_path=absolute_path
        )

    @classmethod
    def from_species(
            cls,
            species: str,
            spectral_sampling_type: str = 'R',
            spectral_sampling: [int, float] = Opacity._default_line_by_line_resolving_power,
            wavelength_min: float = _default_wavelength_range[0],
            wavelength_max: float = _default_wavelength_range[1],
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            _, _, _, _  # spectral info
        ) = cls.split_species_all_info(
            species=species,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def from_species_fullname(
            cls,
            species_fullname: str,
            path_input_data: str = None,
            category: str = _default_category
    ):
        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()

        (
            name, natural_abundance, charge,
            _, _, _,  # cloud info
            source,
            spectral_sampling_type, spectral_sampling,
            wavelength_min, wavelength_max
        ) = cls.split_species_all_info(
            species=species_fullname,
            replace_charge_symbol_with_char=False,
            full=True
        )

        name, natural_abundance, charge = cls._init_species_name_elements(
            name=name,
            species=species_fullname,
            natural_abundance=natural_abundance,
            charge=charge
        )

        new_opacity = cls(
            species_list=[name],
            natural_abundance=natural_abundance,
            charge=charge,
            source=source,
            spectral_sampling_type=spectral_sampling_type,
            spectral_sampling=spectral_sampling,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            path_input_data=path_input_data
        )

        return new_opacity

    @classmethod
    def write(
            cls,
            file: str,
            species_name: str,
            cross_sections: npt.NDArray[float],
            wavenumber_bin_edges: npt.NDArray[float],
            pressures: npt.NDArray[float],
            temperatures: npt.NDArray[float],
            temperature_grid_type: str,
            molar_mass: float = None,
            doi: str = '',
            contributor: str = '',
            description: str = '',
            date_id: str = None,
            n_g: int = None,
            wavelength_range: tuple[float] = None
    ):
        molar_mass, date_id = cls._before_write(
            temperature_grid_type=temperature_grid_type,
            molar_mass=molar_mass,
            species_name=species_name,
            date_id=date_id
        )

        expected_shape = (pressures.size, temperatures.size, wavenumber_bin_edges.size)

        if cross_sections.shape != expected_shape:
            raise ValueError(
                f"k-coefficients must be of shape (pressures, temperatures, wavenumbers bin centers, g-space), "
                f"i.e. {expected_shape}, but is of shape {cross_sections.shape}"
            )

        with h5py.File(file, "w") as fh5:
            dataset = fh5.create_dataset(
                name='DOI',
                shape=(1,),
                data=doi
            )
            dataset.attrs['long_name'] = 'Data object identifier linked to the data'
            dataset.attrs['contributor'] = str(contributor)
            dataset.attrs['additional_description'] = str(description)

            dataset = fh5.create_dataset(
                name='Date_ID',
                shape=(1,),
                data=date_id
            )
            dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                         'at which the table has been created, ' \
                                         'along with the version of petitRADTRANS'

            dataset = fh5.create_dataset(
                name='bin_edges',
                data=wavenumber_bin_edges
            )
            dataset.attrs['long_name'] = 'Wavenumber grid'
            dataset.attrs['units'] = 'cm^-1'

            dataset = fh5.create_dataset(
                name='xsecarr',
                data=cross_sections
            )
            dataset.attrs['long_name'] = 'Table of the cross-sections with axes (pressure, temperature, wavenumber)'
            dataset.attrs['units'] = 'cm^2/molecule'

            dataset = fh5.create_dataset(
                name='mol_mass',
                shape=(1,),
                data=float(molar_mass)
            )
            dataset.attrs['long_name'] = 'Mass of the species'
            dataset.attrs['units'] = 'AMU'

            dataset = fh5.create_dataset(
                name='mol_name',
                shape=(1,),
                data=cls.get_species_base_name(species_full_name=species_name, join=True)
            )
            dataset.attrs['long_name'] = 'Name of the species described'

            dataset = fh5.create_dataset(
                name='p',
                data=pressures
            )
            dataset.attrs['long_name'] = 'Pressure grid'
            dataset.attrs['units'] = 'bar'

            dataset = fh5.create_dataset(
                name='t',
                data=temperatures
            )
            dataset.attrs['long_name'] = 'Temperature grid'
            dataset.attrs['units'] = 'K'

            dataset = fh5.create_dataset(
                name='temperature_grid_type',
                shape=(1,),
                data='regular'
            )
            dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                         '(same temperatures for all pressures) or "pressure-dependent"'

            dataset = fh5.create_dataset(
                name='wlrange',
                data=np.array([wavelength_range[0], wavelength_range[1]]) * 1e4  # cm to um
            )
            dataset.attrs['long_name'] = 'Wavelength range covered'
            dataset.attrs['units'] = 'µm'

            dataset = fh5.create_dataset(
                name='wnrange',
                data=np.array([wavenumber_bin_edges.min(), wavenumber_bin_edges.max()])
            )
            dataset.attrs['long_name'] = 'Wavenumber range covered'
            dataset.attrs['units'] = 'cm^-1'
