import os
import shutil
import tempfile

from .context import petitRADTRANS
from .utils import radtrans_parameters, reference_filenames


def init_planet():
    return petitRADTRANS.planet.Planet.get(radtrans_parameters['planetary_parameters']['name'])


planet = init_planet()


def test_planet_get():
    # Try to download the planet data
    # Remove the planet file if it exists
    filename = petitRADTRANS.planet.Planet.generate_filename(
        name=radtrans_parameters['planetary_parameters']['name'],
        directory=petitRADTRANS.planet.Planet.get_default_directory()
    )

    file_exists = False
    filename_tmp = None

    if os.path.isfile(filename):
        print(f"Temporarily removing file '{filename}' for testing")
        file_exists = True

        tmp_dir = tempfile.gettempdir()
        file = os.path.split(filename)[1]
        filename_tmp = os.path.join(tmp_dir, file)
        shutil.move(filename, filename_tmp)

    try:
        _ = petitRADTRANS.planet.Planet.get(radtrans_parameters['planetary_parameters']['name'])

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no new HDF5 file '{filename}' generated")
    finally:
        if file_exists:
            print(f"Copying back file '{filename}'")
            shutil.move(filename_tmp, filename)

    vot_filename = filename.rsplit('.', 1)[0] + '.vot'

    if os.path.isfile(vot_filename):
        raise FileExistsError(f"temporary VOT file '{vot_filename}' should have been removed")

    # Planet data downloaded, now try to load the planet
    _ = petitRADTRANS.planet.Planet.get(radtrans_parameters['planetary_parameters']['name'])


def test_planet_from_tab():
    _ = petitRADTRANS.planet.Planet.from_tab_file(
        reference_filenames['NASA_exoplanet_archive_test']
    )


def test_planet_calculate_equilibrium_temperature():
    # Override planet parameters to ensure constant values are used
    planet.star_effective_temperature = radtrans_parameters['stellar_parameters']['effective_temperature']
    planet.star_radius = radtrans_parameters['stellar_parameters']['radius'] * petitRADTRANS.physical_constants.r_sun
    planet.orbit_semi_major_axis = radtrans_parameters['planetary_parameters']['orbit_semi_major_axis']
    planet.bond_albedo = radtrans_parameters['planetary_parameters']['surface_reflectance']
