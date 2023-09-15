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
        print('Generating configuration file...')

        config = cls.get_default_config()

        with open(cls._config_file, 'w') as configfile:
            config.write(configfile)

    @property
    def directory(self):
        return self._config_directory

    @property
    def config_file(self):
        """
        Get the full path to the petitRADTRANS configuration file.

        Returns:
            The configuration filename.
        """
        return self._config_file

    @classmethod
    def init_default(cls):
        config = cls()
        config.update(cls._default_config)

        return config

    @classmethod
    def load(cls):
        """
        Load the petitRADTRANS configuration file. Generate it if necessary.

        Returns:
           The petitRADTRANS configuration.
        """
        if not os.path.isfile(cls._config_file):
            cls._make()  # TODO find a better, safer way to do generate the configuration file?

        config = cls()
        config.read(cls._config_file)

        return config

    def set_input_data_path(self, path: str):
        """
        Update the configuration file of petitRADTRANS.

        Args:
            path: path to the petitRADTRANS input data file
        """
        path_tail = path.rsplit(os.path.sep, 1)[1]

        if path_tail != 'input_data':
            path = os.path.join(path, 'input_data')

        config = self.load()
        config.set('Paths', 'pRT_input_data_path', os.path.abspath(path))

        with open(self._config_file, 'w') as configfile:
            config.write(configfile)

        print(f"Input data path updated ('{path}'), restart environment for the change to take effect")


petitradtrans_config_parser = PetitradtransConfigParser()
petitradtrans_config = petitradtrans_config_parser.load()
