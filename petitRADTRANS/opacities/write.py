import datetime
import numpy as np
import h5py

import petitRADTRANS
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass


def write_h2_rayleigh_opacities(
    hdf5_file,
    wavenumbers,
    elastic_reduced_cross_sections,
    raman_reduced_cross_sections,
    rotational_quantum_numbers,
    vibrational_quantum_numbers,
    excitation_energies,
    anti_stokes_hold_energy,
    channel_count,
    source_archive_md5,
    doi="",
    description="",
):
    with h5py.File(hdf5_file, "w") as fh5:
        dataset = fh5.create_dataset(name="DOI", shape=(1,), data=doi)
        dataset.attrs["long_name"] = "Data object identifier linked to the data"
        dataset.attrs["additional_description"] = description

        dataset = fh5.create_dataset(
            name="Date_ID",
            shape=(1,),
            data=f"petitRADTRANS-v{petitRADTRANS.__version__}_"
            f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        )
        dataset.attrs["long_name"] = (
            "ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) "
            "at which the table has been created, "
            "along with the version of petitRADTRANS"
        )

        dataset = fh5.create_dataset(name="wavenumbers", data=wavenumbers)
        dataset.attrs["long_name"] = (
            "Scattering table wavenumbers; the first node, 0, holds the zero-energy limit"
        )
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(
            name="elastic_reduced_cross_sections", data=elastic_reduced_cross_sections
        )
        dataset.attrs["long_name"] = (
            "Table of elastic cross sections divided by the photon energy to the power 4, "
            "with axes (initial state, wavenumber)"
        )
        dataset.attrs["units"] = "Mb.eV^-4"

        dataset = fh5.create_dataset(
            name="raman_reduced_cross_sections", data=raman_reduced_cross_sections
        )
        dataset.attrs["long_name"] = (
            "Table of Raman cross sections summed over final states, divided by the photon "
            "energy to the power 4, with axes (initial state, wavenumber)"
        )
        dataset.attrs["units"] = "Mb.eV^-4"

        dataset = fh5.create_dataset(
            name="rotational_quantum_numbers", data=rotational_quantum_numbers
        )
        dataset.attrs["long_name"] = "Rotational quantum number of the initial states"

        dataset = fh5.create_dataset(
            name="vibrational_quantum_numbers", data=vibrational_quantum_numbers
        )
        dataset.attrs["long_name"] = "Vibrational quantum number of the initial states"

        dataset = fh5.create_dataset(name="excitation_energies", data=excitation_energies)
        dataset.attrs["long_name"] = "Energy of the initial states relative to the lowest state"
        dataset.attrs["units"] = "eV"

        dataset = fh5.create_dataset(
            name="anti_stokes_hold_energy", data=anti_stokes_hold_energy
        )
        dataset.attrs["long_name"] = (
            "Photon energy below which the reduced cross sections of anti-Stokes channels are held"
        )
        dataset.attrs["units"] = "eV"

        dataset = fh5.create_dataset(name="channel_count", data=channel_count)
        dataset.attrs["long_name"] = "Number of scattering channels in the table"

        dataset = fh5.create_dataset(
            name="source_archive_md5", shape=(1,), data=source_archive_md5
        )
        dataset.attrs["long_name"] = "MD5 checksum of the source data archive"

        dataset = fh5.create_dataset(
            name="mol_mass", data=np.array([float(get_species_molar_mass("H2"))])
        )
        dataset.attrs["long_name"] = "Mass of the species"
        dataset.attrs["units"] = "AMU"

        dataset = fh5.create_dataset(name="mol_name", shape=(1,), data="H2")
        dataset.attrs["long_name"] = "Name of the species described"

        positive_wavenumbers = wavenumbers[wavenumbers > 0]

        dataset = fh5.create_dataset(
            name="wlrange",
            data=np.array([1 / positive_wavenumbers.max(), 1 / positive_wavenumbers.min()]) * 1e4,  # cm to um
        )
        dataset.attrs["long_name"] = "Wavelength range covered"
        dataset.attrs["units"] = "µm"

        dataset = fh5.create_dataset(
            name="wnrange", data=np.array([positive_wavenumbers.min(), positive_wavenumbers.max()])
        )
        dataset.attrs["long_name"] = "Wavenumber range covered"
        dataset.attrs["units"] = "cm^-1"


def write_cia_opacities(
    hdf5_cia_file,
    molecules,
    wavenumbers,
    wavelengths,
    alpha,
    temperatures,
    doi="",
    description="",
):
    with h5py.File(hdf5_cia_file, "w") as fh5:
        dataset = fh5.create_dataset(name="DOI", shape=(1,), data=doi)
        dataset.attrs["long_name"] = "Data object identifier linked to the data"
        dataset.attrs["additional_description"] = description

        dataset = fh5.create_dataset(
            name="Date_ID",
            shape=(1,),
            data=f"petitRADTRANS-v{petitRADTRANS.__version__}_"
            f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        )
        dataset.attrs["long_name"] = (
            "ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) "
            "at which the table has been created, "
            "along with the version of petitRADTRANS"
        )

        dataset = fh5.create_dataset(name="wavenumbers", data=wavenumbers)
        dataset.attrs["long_name"] = "CIA wavenumbers"
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(name="alpha", data=alpha)
        dataset.attrs["long_name"] = (
            "Table of monochromatic absorption with axes (temperature, wavenumber)"
        )
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(name="t", data=temperatures)
        dataset.attrs["long_name"] = "Temperature grid"
        dataset.attrs["units"] = "K"

        dataset = fh5.create_dataset(
            name="mol_mass",
            data=np.array(
                [float(get_species_molar_mass(species)) for species in molecules]
            ),
        )
        dataset.attrs["long_name"] = "Masses of the colliding species"
        dataset.attrs["units"] = "AMU"

        dataset = fh5.create_dataset(name="mol_name", data=molecules)
        dataset.attrs["long_name"] = "Names of the colliding species described"

        dataset = fh5.create_dataset(
            name="wlrange",
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4,  # cm to um
        )
        dataset.attrs["long_name"] = "Wavelength range covered"
        dataset.attrs["units"] = "µm"

        dataset = fh5.create_dataset(
            name="wnrange", data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs["long_name"] = "Wavenumber range covered"
        dataset.attrs["units"] = "cm^-1"


def write_cloud_opacities(
    hdf5_opacity_file,
    cloud_species,
    wavenumbers,
    cloud_wavelengths,
    cloud_absorption_opacities,
    cloud_scattering_opacities,
    cloud_asymmetry_parameter,
    cloud_particle_densities,
    cloud_particle_radii,
    cloud_particle_radius_bins,
    doi="",
    description="",
):
    with h5py.File(hdf5_opacity_file, "w") as fh5:
        dataset = fh5.create_dataset(name="DOI", shape=(1,), data=doi)
        dataset.attrs["long_name"] = "Data object identifier linked to the data"
        dataset.attrs["additional_description"] = str(description)

        dataset = fh5.create_dataset(
            name="Date_ID",
            shape=(1,),
            data=f"petitRADTRANS-v{petitRADTRANS.__version__}"
            f"_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        )
        dataset.attrs["long_name"] = (
            "ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) "
            "at which the table has been created, "
            "along with the version of petitRADTRANS"
        )

        dataset = fh5.create_dataset(name="wavenumbers", data=wavenumbers)
        dataset.attrs["long_name"] = "Opacities wavenumbers"
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(
            name="absorption_opacities", data=cloud_absorption_opacities
        )
        dataset.attrs["long_name"] = (
            "Table of the absorption opacities with axes (particle radius, wavenumber)"
        )
        dataset.attrs["units"] = "cm^2.g^-1"

        dataset = fh5.create_dataset(
            name="scattering_opacities", data=cloud_scattering_opacities
        )
        dataset.attrs["long_name"] = (
            "Table of the scattering opacities with axes (particle radius, wavenumber)"
        )
        dataset.attrs["units"] = "cm^2.g^-1"

        dataset = fh5.create_dataset(
            name="asymmetry_parameters", data=cloud_asymmetry_parameter
        )
        dataset.attrs["long_name"] = (
            "Table of the asymmetry parameters with axes (particle radius, wavenumber)"
        )
        dataset.attrs["units"] = "None"

        dataset = fh5.create_dataset(
            name="particle_density", data=cloud_particle_densities
        )
        dataset.attrs["long_name"] = "Average density of the cloud particles"
        dataset.attrs["units"] = "g.cm^-3"

        dataset = fh5.create_dataset(name="mol_name", shape=(1,), data=cloud_species)
        dataset.attrs["long_name"] = (
            'Name of the species described, "(c)" indicates that it has condensed'
        )

        dataset = fh5.create_dataset(name="particle_radii", data=cloud_particle_radii)
        dataset.attrs["long_name"] = "Particles average radius grid"
        dataset.attrs["units"] = "cm"

        dataset = fh5.create_dataset(
            name="particle_radius_bins", data=cloud_particle_radius_bins
        )
        dataset.attrs["long_name"] = "Particles average radius grid bins"
        dataset.attrs["units"] = "cm"

        dataset = fh5.create_dataset(
            name="wlrange",
            data=np.array([cloud_wavelengths.min(), cloud_wavelengths.max()])
            * 1e4,  # cm to um
        )
        dataset.attrs["long_name"] = "Wavelength range covered"
        dataset.attrs["units"] = "µm"

        dataset = fh5.create_dataset(
            name="wnrange", data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs["long_name"] = "Wavenumber range covered"
        dataset.attrs["units"] = "cm^-1"


def write_correlated_k(
    file,
    doi,
    wavenumbers,
    wavenumbers_bins_edges,
    cross_sections,
    mol_mass,
    species,
    opacities_pressures,
    opacities_temperatures,
    g_gauss,
    weights_gauss,
    wavelengths=None,
    n_g=None,
    contributor=None,
    description=None,
):
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    if n_g is None:
        n_g = g_gauss.size

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(name="DOI", shape=(1,), data=doi)
        dataset.attrs["long_name"] = "Data object identifier linked to the data"
        dataset.attrs["contributor"] = str(contributor)
        dataset.attrs["additional_description"] = str(description)

        dataset = fh5.create_dataset(
            name="Date_ID",
            shape=(1,),
            data=f"petitRADTRANS-v{petitRADTRANS.__version__}"
            f"_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        )
        dataset.attrs["long_name"] = (
            "ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) "
            "at which the table has been created, "
            "along with the version of petitRADTRANS"
        )

        dataset = fh5.create_dataset(name="bin_centers", data=wavenumbers)
        dataset.attrs["long_name"] = "Centers of the wavenumber bins"
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(name="bin_edges", data=wavenumbers_bins_edges)
        dataset.attrs["long_name"] = "Separations between the wavenumber bins"
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(name="kcoeff", data=cross_sections)
        dataset.attrs["long_name"] = (
            "Table of the k-coefficients with axes "
            "(pressure, temperature, wavenumber, g space)"
        )
        dataset.attrs["units"] = "cm^2/molecule"

        dataset = fh5.create_dataset(name="method", shape=(1,), data="petit_samples")
        dataset.attrs["long_name"] = "Name of the method used to sample g-space"

        dataset = fh5.create_dataset(name="mol_mass", shape=(1,), data=float(mol_mass))
        dataset.attrs["long_name"] = "Mass of the species"
        dataset.attrs["units"] = "AMU"

        dataset = fh5.create_dataset(
            name="mol_name", shape=(1,), data=species.split("_", 1)[0]
        )
        dataset.attrs["long_name"] = "Name of the species described"

        dataset = fh5.create_dataset(name="ngauss", data=n_g)
        dataset.attrs["long_name"] = "Number of points used to sample the g-space"

        dataset = fh5.create_dataset(name="p", data=opacities_pressures)
        dataset.attrs["long_name"] = "Pressure grid"
        dataset.attrs["units"] = "bar"

        dataset = fh5.create_dataset(name="samples", data=g_gauss)
        dataset.attrs["long_name"] = (
            "Abscissas used to sample the k-coefficients in g-space"
        )

        dataset = fh5.create_dataset(name="t", data=opacities_temperatures)
        dataset.attrs["long_name"] = "Temperature grid"
        dataset.attrs["units"] = "K"

        dataset = fh5.create_dataset(
            name="temperature_grid_type", shape=(1,), data="regular"
        )
        dataset.attrs["long_name"] = (
            'Whether the temperature grid is "regular" '
            '(same temperatures for all pressures) or "pressure-dependent"'
        )

        dataset = fh5.create_dataset(name="weights", data=weights_gauss)
        dataset.attrs["long_name"] = "Weights used in the g-space quadrature"

        dataset = fh5.create_dataset(
            name="wlrange",
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4,  # cm to um
        )
        dataset.attrs["long_name"] = "Wavelength range covered"
        dataset.attrs["units"] = "µm"

        dataset = fh5.create_dataset(
            name="wnrange", data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs["long_name"] = "Wavenumber range covered"
        dataset.attrs["units"] = "cm^-1"


def write_line_by_line(
    file,
    doi,
    wavenumbers,
    opacities,
    mol_mass,
    species,
    opacities_pressures,
    opacities_temperatures,
    wavelengths=None,
    contributor=None,
    description=None,
):
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(name="DOI", shape=(1,), data=doi)
        dataset.attrs["long_name"] = "Data object identifier linked to the data"
        dataset.attrs["contributor"] = str(contributor)
        dataset.attrs["additional_description"] = str(description)

        dataset = fh5.create_dataset(
            name="Date_ID",
            shape=(1,),
            data=f"petitRADTRANS-v{petitRADTRANS.__version__}"
            f"_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        )
        dataset.attrs["long_name"] = (
            "ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) "
            "at which the table has been created, "
            "along with the version of petitRADTRANS"
        )

        dataset = fh5.create_dataset(name="bin_edges", data=wavenumbers)
        dataset.attrs["long_name"] = "Wavenumber grid"
        dataset.attrs["units"] = "cm^-1"

        dataset = fh5.create_dataset(name="xsecarr", data=opacities)
        dataset.attrs["long_name"] = (
            "Table of the cross-sections with axes (pressure, temperature, wavenumber)"
        )
        dataset.attrs["units"] = "cm^2/molecule"

        dataset = fh5.create_dataset(name="mol_mass", shape=(1,), data=float(mol_mass))
        dataset.attrs["long_name"] = "Mass of the species"
        dataset.attrs["units"] = "AMU"

        dataset = fh5.create_dataset(
            name="mol_name", shape=(1,), data=species.split("_", 1)[0]
        )
        dataset.attrs["long_name"] = "Name of the species described"

        dataset = fh5.create_dataset(name="p", data=opacities_pressures)
        dataset.attrs["long_name"] = "Pressure grid"
        dataset.attrs["units"] = "bar"

        dataset = fh5.create_dataset(name="t", data=opacities_temperatures)
        dataset.attrs["long_name"] = "Temperature grid"
        dataset.attrs["units"] = "K"

        dataset = fh5.create_dataset(
            name="temperature_grid_type", shape=(1,), data="regular"
        )
        dataset.attrs["long_name"] = (
            'Whether the temperature grid is "regular" '
            '(same temperatures for all pressures) or "pressure-dependent"'
        )

        dataset = fh5.create_dataset(
            name="wlrange",
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4,  # cm to um
        )
        dataset.attrs["long_name"] = "Wavelength range covered"
        dataset.attrs["units"] = "µm"

        dataset = fh5.create_dataset(
            name="wnrange", data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs["long_name"] = "Wavenumber range covered"
        dataset.attrs["units"] = "cm^-1"
