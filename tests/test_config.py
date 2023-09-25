"""Test petitRADTRANS config module.
"""
import copy
import os

from .context import petitRADTRANS


def test_configuration_file_generation():
    input_data_path_option = 'pRT_input_data_path'
    path_section = 'Paths'

    # Save old input data path
    path_input_data = petitRADTRANS.config.petitradtrans_config_parser.get_input_data_path()

    # Check if default input data path is used
    config = petitRADTRANS.config.configuration.PetitradtransConfigParser()

    using_default_directory = False

    default_directory = config.default_config[path_section][input_data_path_option]
    modified_default_directory = default_directory + '_'

    if os.path.isdir(config.default_config[path_section][input_data_path_option]):
        # Rename input_data path to raise the FileNotFound error
        print(f"Temporarily renaming input data directory to '{modified_default_directory}'")
        os.rename(default_directory, modified_default_directory)
        using_default_directory = True

    # Remove config file directory
    os.remove(config.config_file)  # prevent rmdir from complaining that the directory is not empty
    os.rmdir(config.directory)

    # Create a new instance of PetitradtransConfigParser, this should generate the default config file
    config = petitRADTRANS.config.configuration.PetitradtransConfigParser()
    config.power_load()

    # Test if the config file was generated, but the input_data path is incorrect
    print(f"Testing for incorrect input_data path behaviour...")

    try:
        petitRADTRANS.phoenix.__load_stellar_spectra()

        if os.path.isdir(config.get_input_data_path()):
            raise FileExistsError(f"directory '{config.get_input_data_path()}' exists, while it should not for this test")
    except FileNotFoundError as error:
        if not os.path.isdir(config.directory):
            raise FileNotFoundError(f"configuration directory '{config.directory}' is expected to be generated")
        elif not os.path.isdir(config.get_input_data_path()):
            print(f"directory '{config.get_input_data_path()}' is incorrect, as expected")
        else:
            raise FileExistsError(f"a file was unexpectedly not found: '{str(error)}'")
    finally:
        petitRADTRANS.config.petitradtrans_config_parser = config

        if using_default_directory:
            print(f"Renaming '{modified_default_directory}' back to '{default_directory}'")
            os.rename(modified_default_directory, default_directory)

        print(f"Set back input_data path to '{path_input_data}'")
        config.set_input_data_path(path_input_data)

    # Now test if the file is correctly loaded once everything is set properly
    print(f"Testing for correct input_data path behaviour...")
    petitRADTRANS.phoenix.__load_stellar_spectra()


def test_configuration_file_repair():
    input_data_path_option = 'pRT_input_data_path'
    path_section = 'Paths'
    url_section = 'URLs'

    config = copy.deepcopy(petitRADTRANS.config.petitradtrans_config_parser)
    config.repair()  # ensure we start with a correct configuration
    broken_config = copy.deepcopy(config)

    try:
        # Break the configuration by removing sections and options
        section_removed = broken_config.remove_section(url_section)

        if not section_removed:
            raise KeyError(f"section '{url_section}' was expected to be in config file, but was not; "
                           f"there may be an inconsistency between the default config_file and this test")

        option_removed = broken_config.remove_option(path_section, input_data_path_option)

        if not option_removed:
            raise KeyError(f"option '{input_data_path_option}' of section '{path_section}' "
                           f"was expected to be in config file, but was not; "
                           f"there may be an inconsistency between the default config_file and this test")

        # Power load should auto-repair the file
        broken_config.repair()
        broken_config.load()
        repair_is_broken = False
        broken_str = []

        # Test if the configuration is properly repaired
        if not broken_config.has_section(url_section):
            repair_is_broken = True
            broken_str.append(f"section '{url_section}' was not properly repaired")

        if not broken_config.has_option(path_section, input_data_path_option):
            repair_is_broken = True
            broken_str.append(f"option '{input_data_path_option}' of section '{path_section}' "
                              f"was not properly repaired")

        if repair_is_broken:
            raise KeyError("repair is broken for the following reason(s):\n" + "\n".join(broken_str))
    except KeyError as error:
        raise KeyError(f"something went wrong during repair testing, the error was:\n{str(error)}")
    finally:
        # Restore configuration to its previous state, even if it was broken
        config.power_update(petitRADTRANS.config.petitradtrans_config_parser, repair_update=False)

    # Now test if the file is correctly loaded once everything is set properly
    petitRADTRANS.config.petitradtrans_config_parser.load()

    print(f"Testing for correct behaviour...")
    petitRADTRANS.phoenix.__load_stellar_spectra()
