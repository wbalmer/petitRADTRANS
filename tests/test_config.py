"""Test petitRADTRANS config module.
"""
import os

from .context import petitRADTRANS


def test_configuration_file_generation():
    # Save old input data path
    path_input_data = petitRADTRANS.config.petitradtrans_config_parser.get_input_data_path()

    # Check if default input data path is used
    config = petitRADTRANS.config.configuration.PetitradtransConfigParser()

    using_default_directory = False

    default_directory = config.default_config['Paths']['pRT_input_data_path']
    modified_default_directory = default_directory + '_'

    if os.path.isdir(config.default_config['Paths']['pRT_input_data_path']):
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
