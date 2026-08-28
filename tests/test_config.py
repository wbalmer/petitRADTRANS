"""Test petitRADTRANS config module.

The tests in this module deliberately break, delete and regenerate the petitRADTRANS configuration
file. To guarantee that the user's actual configuration is never affected, every test in this module
uses the ``isolated_configuration`` fixture, which redirects the configuration onto a temporary test
configuration file that is removed once the test is over.
"""
import copy
import os
import shutil
from types import SimpleNamespace

import pytest

from .context import petitRADTRANS
phoenix_star_table = petitRADTRANS.stellar_spectra.phoenix.PhoenixStarTable()

TEST_CONFIG_FILENAME = 'petitradtrans_config_file_test.ini'

_config_parser_class = petitRADTRANS.config.configuration.PetitradtransConfigParser
_config_parser = petitRADTRANS.config.configuration.petitradtrans_config_parser


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path):
    """Redirect the petitRADTRANS configuration onto a temporary test configuration file.

    The configuration directory, configuration filename and default configuration are class
    attributes of PetitradtransConfigParser: ``load`` and ``power_load`` read them from the class,
    while ``save`` and ``repair`` read them from the instance. Patching the class attributes is
    therefore the only way to redirect both reading and writing.

    The default input data path is redirected as well, onto a path within the temporary directory
    that is never created. Tests can thus check for the "incorrect input data path" behaviour
    without having to rename the actual input data directory.

    On teardown, the class attributes are restored, the configuration parser is reloaded from the
    (untouched) actual configuration file, the temporary files are removed, and the actual
    configuration file is checked to be exactly as it was before the test.
    """
    # Save the actual configuration state
    actual_directory = _config_parser_class._directory
    actual_config_file = _config_parser_class._config_file
    actual_default_config = _config_parser_class._default_config
    actual_input_data_path = _config_parser.get_input_data_path()

    with open(actual_config_file, 'rb') as f:
        actual_config_file_content = f.read()  # used to check that the actual file is left untouched

    # Build the temporary configuration
    test_path = tmp_path / 'petitradtrans_config_test'
    test_directory = str(test_path / 'config')
    test_config_file = os.path.join(test_directory, TEST_CONFIG_FILENAME)
    test_input_data_path = str(test_path / 'default_input_data')  # deliberately never created

    # The default configuration is a mutable class attribute, so it must be copied before editing
    test_default_config = copy.deepcopy(actual_default_config)
    test_default_config['Paths']['pRT_input_data_path'] = test_input_data_path
    test_default_config['Paths']['pRT_outputs_path'] = str(test_path / 'default_outputs')
    test_default_config['Filters']['database'] = os.path.join(
        test_input_data_path, 'filters', 'filter_database.hdf5'
    )

    _config_parser_class._directory = test_directory
    _config_parser_class._config_file = test_config_file
    _config_parser_class._default_config = test_default_config

    # Copy the actual configuration into the test configuration file, so that the tests checking for
    # the correct behaviour still use the actual input data path
    os.makedirs(test_directory)

    with open(test_config_file, 'wb') as f:
        f.write(actual_config_file_content)

    _config_parser.clear()
    _config_parser.load()

    try:
        yield SimpleNamespace(
            directory=test_directory,
            config_file=test_config_file,
            default_input_data_path=test_input_data_path,
            input_data_path=actual_input_data_path
        )
    finally:
        # Restore the actual configuration state, then reload the untouched actual configuration file
        _config_parser_class._directory = actual_directory
        _config_parser_class._config_file = actual_config_file
        _config_parser_class._default_config = actual_default_config

        _config_parser.clear()
        _config_parser.load()

        shutil.rmtree(test_path, ignore_errors=True)

        # Ensure that the actual configuration file was not affected by the test
        if not os.path.isfile(actual_config_file):
            raise FileNotFoundError(f"configuration file '{actual_config_file}' was removed by this test, "
                                    f"while it should have been left untouched")

        with open(actual_config_file, 'rb') as f:
            if f.read() != actual_config_file_content:
                raise AssertionError(f"configuration file '{actual_config_file}' was modified by this test, "
                                     f"while it should have been left untouched")


def test_configuration_file_generation(isolated_configuration):
    input_data_path_option = 'pRT_input_data_path'
    path_section = 'Paths'

    # Check that the (temporary) default input data path does not exist, so that the FileNotFound
    # error can be raised without having to touch the actual input data directory
    config = petitRADTRANS.config.configuration.PetitradtransConfigParser()
    default_directory = config.default_config[path_section][input_data_path_option]

    if os.path.isdir(default_directory):
        raise FileExistsError(f"cannot test for incorrect input data path behaviour: "
                              f"test default input data directory '{default_directory}' already exists")

    # Save old config
    config = copy.deepcopy(petitRADTRANS.config.petitradtrans_config_parser)

    # Remove the test config file directory
    os.remove(config.config_file)  # prevent rmdir from complaining that the directory is not empty
    os.rmdir(config.directory)

    # Create a new instance of PetitradtransConfigParser, this should generate the default config file
    new_config = petitRADTRANS.config.configuration.PetitradtransConfigParser()
    new_config.power_load()
    petitRADTRANS.config.petitradtrans_config_parser.power_update(new_config)

    # Test if the config file was generated, but the input_data path is incorrect
    print(f"Testing for incorrect input_data path behaviour...")
    error_raised = False

    # Ensure the directory exists
    star_table_directory = os.path.dirname(phoenix_star_table.get_default_file())

    if not os.path.isdir(star_table_directory):
        os.makedirs(star_table_directory)

    try:
        phoenix_star_table.load(search_online=False)

        if os.path.isfile(phoenix_star_table.get_default_file()):
            raise FileExistsError(
                f"file '{phoenix_star_table.get_default_file()}' exists, "
                f"while it should not for this test"
            )
    except FileNotFoundError as error:
        error_raised = True

        if not os.path.isdir(new_config.directory):
            raise FileNotFoundError(f"configuration directory '{new_config.directory}' is expected to be generated")
        elif not os.path.isfile(phoenix_star_table.get_default_file()):
            print(f"directory '{new_config.get_input_data_path()}' is incorrect, as expected")
        else:
            raise FileExistsError(f"a file was unexpectedly not found: '{str(error)}'")
    finally:
        petitRADTRANS.config.petitradtrans_config_parser.power_update(config)

    if not error_raised:
        raise FileExistsError(f"stellar spectrum was unexpectedly loaded, maybe 'petitradtrans_config_parser' was "
                              f"not properly reset")

    # Now test if the file is correctly loaded once everything is set properly
    print(f"Testing for correct input_data path behaviour...")
    phoenix_star_table.load(search_online=False)


def test_configuration_file_repair(isolated_configuration):
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
        petitRADTRANS.config.petitradtrans_config_parser.power_update(config, repair_update=False)

    # Now test if the file is correctly loaded once everything is set properly
    petitRADTRANS.config.petitradtrans_config_parser.load()

    if not petitRADTRANS.config.petitradtrans_config_parser.has_section(url_section):
        raise KeyError(f"section '{url_section}' was not restored after the repair test")

    print(f"Testing for correct behaviour...")
    phoenix_star_table.load(search_online=False)
