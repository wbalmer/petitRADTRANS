"""
Manages the configuration files.
"""
import configparser
import os
from pathlib import Path


class PetitradtransConfigParser(configparser.ConfigParser):
    _directory = os.path.join(str(Path.home()), '.petitradtrans')
    _config_file = os.path.join(_directory, 'petitradtrans_config_file.ini')
    _default_config = {
        'Paths': {
            'pRT_input_data_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'input_data'),
            'pRT_outputs_path': os.path.join(str(Path.home()), 'petitRADTRANS', 'outputs')
        },
        'URLs': {
            'pRT_input_data_url': 'https://keeper.mpdl.mpg.de/d/f48c13424cd34d2d9b47/?p='
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


petitradtrans_config_parser = PetitradtransConfigParser()
petitradtrans_config_parser.power_load()
