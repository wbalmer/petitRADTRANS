import os
import shutil
import tempfile

from .context import petitRADTRANS
from .utils import radtrans_parameters


def test_planet_get():
    # No need to compare the results (they can evolve), just checking if the function works

    # Remove the existing file if it exists
    filename = petitRADTRANS.containers.planet.Planet.generate_filename(
        name=radtrans_parameters['planetary_parameters']['name'],
        directory=petitRADTRANS.containers.planet.Planet.default_planet_models_directory
    )

    file_exists = False

    if os.path.isfile(filename):
        print(f"Temporarily removing file '{filename}' for testing")
        file_exists = True

        tmp_dir = tempfile.gettempdir()
        file = os.path.split(filename)[1]
        filename_tmp = os.path.join(tmp_dir, file)
        shutil.move(filename, filename_tmp)
    else:
        filename_tmp = None

    try:
        _ = petitRADTRANS.containers.planet.Planet.get(radtrans_parameters['planetary_parameters']['name'])

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no new HDF5 file '{filename}' generated")
    finally:
        if file_exists:
            print(f"Copying back file '{filename}'")
            shutil.move(filename_tmp, filename)

    vot_filename = filename.rsplit('.', 1)[0] + '.vot'

    if os.path.isfile(vot_filename):
        raise FileExistsError(f"temporary VOT file '{vot_filename}' should have been removed")
