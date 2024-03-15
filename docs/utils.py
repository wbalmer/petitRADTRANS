"""Stores useful functions to make the docs."""
import glob
import h5py
import os

from petitRADTRANS._input_data_loader import (
    _get_base_cloud_names, _split_species_source,
    get_input_data_subpaths, get_species_basename, join_species_all_info, split_species_all_info
)
from petitRADTRANS.config import petitradtrans_config_parser


def make_table_cloud_species_reference_documentation(path_to_cloud_opacities=None):
    """Display the list of cloud references in rST table format.
    Intended to be used in docs/content/available_opacities.rst

    Args:
        path_to_cloud_opacities: path to folder containing the cloud opacities.
    """
    if path_to_cloud_opacities is None:
        path_to_cloud_opacities = os.path.join(
            petitradtrans_config_parser.get_input_data_path(),
            get_input_data_subpaths()['clouds_opacities']
        )

    species_list = sorted(glob.glob(path_to_cloud_opacities + '/*/*/*'))
    return_string = ""

    for species in species_list:
        file_name = species.rsplit(os.path.sep, 1)[-1]
        _file_name = file_name.rsplit('.cotable.petitRADTRANS.h5', 1)[0]  # discard file extension

        name, natural_abundance, _, cloud_info, source, spectral_info = split_species_all_info(_file_name)

        _file_name = join_species_all_info(
            name=name,
            natural_abundance=natural_abundance,
            cloud_info=cloud_info,
            spectral_info=spectral_info
        )

        # Check if file name is in the cloud aliases dictionary
        if _file_name in _get_base_cloud_names().values():
            call_name = None

            for key, value in _get_base_cloud_names().items():
                if value == _file_name:
                    call_name = key

                    break

            call_name = join_species_all_info(
                name=call_name,
                source=source
            )
        else:
            name = get_species_basename(name)  # remove isotope separators

            call_name = join_species_all_info(
                name=name,
                cloud_info=cloud_info,
                source=source
            )

        with h5py.File(species, 'r') as f:
            doi = f['DOI'][0].decode('utf-8')

        return_string += (
            f"\n"
            f"    * - {call_name}\n"
            f"      - {file_name}\n"
            f"      - {doi}"
        )

    print(return_string)
