"""Test the correlated-k scattering part of the radtrans module.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""

import copy

import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import (
    check_cloud_complete_coverage_consistency,
    check_partial_cloud_coverage_full_consistency,
    get_cloud_parameters,
    test_parameters,
    temperature_guillot_2010,
)

relative_tolerance = 1e-5  # relative tolerance when comparing with older spectra


# Initializations
def init_radtrans_correlated_k():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters["pressures"],
        line_species=test_parameters["spectrum_parameters"][
            "line_species_correlated_k"
        ],
        rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        cloud_species=list(test_parameters["cloud_parameters"]["cloud_species"].keys()),
        wavelength_boundaries=test_parameters["spectrum_parameters"][
            "wavelength_range_correlated_k"
        ],
        line_opacity_mode="c-k",
        scattering_in_emission=True,
    )

    return atmosphere


atmosphere_ck_scattering = init_radtrans_correlated_k()


def test_correlated_k_photospheric_radius_calculation():
    import copy
    from .benchmark import ReferenceFile

    (
        mass_fractions,
        cloud_particle_mean_radii,
        _,
        cloud_particle_radius_distribution_std,
        _,
    ) = get_cloud_parameters("mass_fractions_correlated_k")

    # Relax tolerance due to test pipeline inconsistencies
    _parameters_relative_tolerance = copy.deepcopy(
        ReferenceFile.parameters_relative_tolerance
    )
    ReferenceFile.parameters_relative_tolerance = 1e-4

    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_photosphere_radius,
        relative_tolerance=relative_tolerance,
    )

    # atmosphere_ck_scattering._Radtrans__set_sum_opacities(emission=True)
    # atmosphere_ck_scattering._Radtrans__use_smart_clear_opacities = False
    # Calculate opacities
    (
        opacities,
        continuum_opacities_scattering,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        cloud_opacities,
        cloud_particle_mean_radii,
        _,
        _,
        _,
        _,
        _,
    ) = atmosphere_ck_scattering._calculate_opacities(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        mean_molar_masses=test_parameters["mean_molar_mass"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mean_molar_masses=test_parameters["mean_molar_mass"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup,
        opacities=opacities,
        continuum_opacities_scattering=continuum_opacities_scattering,
        cloud_f_sed=np.array(
            list(test_parameters["cloud_parameters"]["cloud_species"].values())[0][
                "f_sed"
            ]
        ),
        cloud_photosphere_median_optical_depth=None,
        cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities=cloud_absorption_opacities,
    )

    # Put back default parameter relative tolerance
    ReferenceFile.parameters_relative_tolerance = _parameters_relative_tolerance


"""
def test_correlated_k_emission_spectrum_cloud_calculated_radius_scattering():
    mass_fractions, _, cloud_f_sed, cloud_particle_radius_distribution_std, _ = get_cloud_parameters(
        'mass_fractions_correlated_k'
    )
    print("In Test Function")
    print("mass_fractions: ", mass_fractions)
    print("cloud_f_sed: ", cloud_f_sed)
    print("cloud_particle_radius_distribution_std: ", cloud_particle_radius_distribution_std)
    print("\n\n")
    benchmark = Benchmark(
        function=atmosphere_ck_scattering.calculate_flux,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        temperatures=temperature_guillot_2010,
        mass_fractions=mass_fractions,
        reference_gravity=test_parameters['planetary_parameters']['reference_gravity'],
        mean_molar_masses=test_parameters['mean_molar_mass'],
        eddy_diffusion_coefficients=test_parameters['planetary_parameters']['eddy_diffusion_coefficients'],
        cloud_f_sed=cloud_f_sed,
        cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
        frequencies_to_wavelengths=False
    )
"""
