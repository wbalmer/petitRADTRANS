"""Stores useful functions to make the docs."""
import glob
import os

from petitRADTRANS._input_data_loader import get_input_data_subpaths, join_species_all_info, split_species_all_info
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

        name, _, _, cloud_info, source, _ = split_species_all_info(file_name)
        cloud_info = cloud_info.rsplit('_', 1)[0]  # discard space group or amorphous name info

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
