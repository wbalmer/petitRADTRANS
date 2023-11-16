"""
Manages the configuration files.
"""
import configparser
import os
from pathlib import Path

from petitRADTRANS.utils import LockedDict


class PetitradtransConfigParser(configparser.ConfigParser):
    _instance = None

    _directory = os.path.join(str(Path.home()), '.petitradtrans')
    _config_file = os.path.join(_directory, 'petitradtrans_config_file.ini')
    _default_config = {
        'Default files': {

        },
        'Paths': {
            'pRT_input_data_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'input_data'),
            'pRT_outputs_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'outputs')
        },
        'URLs': {
            'pRT_input_data_url': 'https://keeper.mpdl.mpg.de/d/fb79812e3a694468bcda/?p='  # TODO change input_data_v3 to just input_data  # noqa E501
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(PetitradtransConfigParser, cls).__new__(cls)

        return cls._instance

    @classmethod
    def _make(cls):
        """
        Get the petitRADTRANS configuration directory, where the configuration file is stored.

        Returns:
            The configuration directory.
        """
        print(f"Generating configuration file '{cls._config_file}'...")

        config = cls.init_default()

        if not os.path.isdir(cls._directory):
            print(f"Creating directory '{cls._directory}'...")
            os.makedirs(cls._directory)

        config.save()

    def check_input_data_path(self, path):
        if self.get_input_data_path() not in path:
            raise ValueError(f"path '{path}' must be within the input_data path ('{self.get_input_data_path()}')")

        path = path.rsplit(self.get_input_data_path())[1][1:]  # remove leading '/'

        if len(path) == 0:
            raise ValueError(f"path must be within the input_data path, but is the input_data path itself ('{path}')")

        is_subpath = False

        for subpath in get_input_data_subpaths().values():
            if subpath in path:
                is_subpath = True
                break

        if not is_subpath:
            subpaths_str = "\n".join(get_input_data_subpaths().values())
            raise ValueError(f"path '{path}' must be within an input_data subpath\n"
                             f"Valid subpaths are:\n"
                             f"{subpaths_str}")

        return path

    @property
    def config_file(self):
        """
        Get the full path to the petitRADTRANS configuration file.

        Returns:
            The configuration filename.
        """
        return self._config_file

    @property
    def default_config(self):
        return self._default_config

    @property
    def directory(self):
        return self._directory

    def get_input_data_path(self):
        return self['Paths']['prt_input_data_path']

    @classmethod
    def init_default(cls):
        config = cls()
        config.update(cls._default_config)

        return config

    def load(self):
        """
        Load the petitRADTRANS configuration file.
        """
        self.read(PetitradtransConfigParser._config_file)

    def power_load(self):
        """
        Load the petitRADTRANS configuration file. Generate it if necessary.
        """
        if not os.path.isfile(PetitradtransConfigParser._config_file):
            PetitradtransConfigParser._make()  # TODO find a better, safer way to do generate the configuration file?

        self.read(PetitradtransConfigParser._config_file)
        self.repair()

    def power_set(self, section, option, value):
        self.set(section, option, value)
        self.repair()
        self.save()

    def power_update(self, new_config, repair_update=True):
        self.update(new_config)

        if repair_update:
            self.repair()

        self.save()

        print("Configuration updated")

    def save(self):
        with open(self._config_file, 'w') as configfile:
            self.write(configfile)

    def set_input_data_path(self, path: str):
        """
        Update the configuration file of petitRADTRANS.

        Args:
            path: path to the petitRADTRANS input data file
        """
        path_tail = path.rsplit(os.path.sep, 1)[1]

        if path_tail != 'input_data':
            path = os.path.join(path, 'input_data')

        self.power_set('Paths', 'pRT_input_data_path', os.path.abspath(path))

        print(f"Input data path changed to '{path}'")

        self.load()

    def set_default_file(self, file: str):
        file = os.path.abspath(file)
        path, filename = file.rsplit(os.path.sep, 1)

        sub_path = self.check_input_data_path(path)

        print(f"Setting new default file '{file}'")
        self.power_set('Default files', sub_path, filename)

        self.load()

    def repair(self):
        repaired = False

        for section in self._default_config:
            if not self.has_section(section):
                print(f"Adding missing section '{section}'...")
                repaired = True
                self.add_section(section)

                for option, value in self._default_config[section].items():
                    self.set(section, option, value)
            else:
                for option, value in self._default_config[section].items():
                    if option not in self[section]:
                        print(f"Adding missing option '{option}' in section '{section}'...")
                        repaired = True
                        self.set(section, option, value)

        if repaired:
            print(f"Repairing configuration file ('{self._config_file}')...")
            self.save()


def get_input_data_subpaths():
    return LockedDict.build_and_lock({
        "cia_opacities": os.path.join("opacities", "continuum", "collision_induced_absorptions"),
        "clouds_opacities": os.path.join("opacities", "continuum", "clouds"),
        "correlated_k_opacities": os.path.join("opacities", "lines", "correlated_k"),
        "line_by_line_opacities": os.path.join("opacities", "lines", "line_by_line"),
        "planet_data": "planet_data",
        "pre_calculated_chemistry": "pre_calculated_chemistry",
        "stellar_spectra": "stellar_spectra"
    })


petitradtrans_config_parser = PetitradtransConfigParser()
petitradtrans_config_parser.power_load()
