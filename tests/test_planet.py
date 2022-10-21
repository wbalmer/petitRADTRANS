import os

from .context import petitRADTRANS
from .utils import radtrans_parameters


def test_planet_get():
    # No need to compare the results (they can evolve), just checking if the function works

    # Remove the existing file if it exists
    filename = petitRADTRANS.containers.planet.Planet.generate_filename(
        name=radtrans_parameters['planetary_parameters']['name'],
        directory=petitRADTRANS.containers.planet.Planet.default_planet_models_directory
    )

    if os.path.isfile(filename):
        os.remove(filename)

    _ = petitRADTRANS.containers.planet.Planet.get(radtrans_parameters['planetary_parameters']['name'])

    if not os.path.isfile(filename):
        raise RuntimeError(f"no HDF5 file generated ('{filename}' do not exist)")

    vot_filename = filename.rsplit('.', 1)[0] + '.vot'

    if os.path.isfile(vot_filename):
        raise RuntimeError(f"Temporary VOT file '{vot_filename}' should have been removed")
