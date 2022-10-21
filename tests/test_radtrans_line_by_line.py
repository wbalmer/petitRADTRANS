"""Test the radtrans module in line-by-line mode.

Essentially go through a simplified version of the tutorial, and compare the results with previous ones.
C.f. (https://petitradtrans.readthedocs.io/en/latest/content/notebooks/getting_started.html).

Do not change the parameters used to generate the comparison files, including input_data files, when running the tests.
"""
import copy

import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, \
    reference_filenames, radtrans_parameters, temperature_guillot_2010, temperature_isothermal

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


# Initializations
def init_radtrans_line_by_line():
    atmosphere = petitRADTRANS.radtrans.Radtrans(
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        continuum_opacities=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        wlen_bords_micron=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        mode='lbl'
    )

    atmosphere.setup_opa_structure(radtrans_parameters['pressures'])

    return atmosphere


def init_spectral_model_line_by_line():
    spectral_model = petitRADTRANS.containers.spectral_model.SpectralModel(
        # Radtrans object parameters
        pressures=radtrans_parameters['pressures'],  # bar
        line_species=radtrans_parameters['spectrum_parameters']['line_species_line_by_line'],
        rayleigh_species=radtrans_parameters['spectrum_parameters']['rayleigh_species'],
        continuum_opacities=radtrans_parameters['spectrum_parameters']['continuum_opacities'],
        do_scat_emis=False,  # is False by default on Radtrans but True by default on SpectralModel
        opacity_mode='lbl',
        # Temperature profile parameters: generate a Guillot temperature profile
        temperature_profile_mode='guillot',
        temperature=radtrans_parameters['temperature_guillot_2010_parameters']['equilibrium_temperature'],  # K
        intrinsic_temperature=radtrans_parameters['temperature_guillot_2010_parameters']['intrinsic_temperature'],
        # metallicity= ,
        guillot_temperature_profile_gamma=radtrans_parameters['temperature_guillot_2010_parameters']['gamma'],
        guillot_temperature_profile_kappa_ir_z0=radtrans_parameters['temperature_guillot_2010_parameters'][
            'infrared_mean_opacity'],
        # Chemical parameters
        use_equilibrium_chemistry=False,
        imposed_mass_mixing_ratios=radtrans_parameters['mass_fractions'],
        # Transmission spectrum parameters (radtrans.calc_transm)
        planet_radius=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,  # cm
        planet_surface_gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        reference_pressure=radtrans_parameters['planetary_parameters']['reference_pressure'],  # bar
        # cloud_pressure=1e-1,
        # Instrument parameters
        # new_resolving_power=8.04e4,
        # output_wavelengths=wavelengths_instrument,  # um
        wavelengths_boundaries=radtrans_parameters['spectrum_parameters']['wavelength_range_line_by_line'],
        # Scaling parameters
        # star_radius=planet.star_radius,  # cm
        # Orbital parameters
        # star_mass=planet.star_mass,  # g
        # semi_major_axis=planet.orbit_semi_major_axis,  # cm
        # orbital_phases=orbital_phases,
        # system_observer_radial_velocities=planet.star_radial_velocity - barycentric_velocities * 1e5,  # cm.s-1
        # planet_rest_frame_shift=0.0,  # cm.s-1
        # Reprocessing parameters
        # uncertainties=model_uncertainties,
        # airmass=airmasses,
        # tellurics_mask_threshold=tellurics_mask_threshold,
        # polynomial_fit_degree=2,
        # apply_throughput_removal=True,
        # apply_telluric_lines_removal=True,
        # Special parameters
        # Test addition of a useless parameter
        yet_another_useless_parameter42="irrelevant string"
    )

    def calculate_mean_molar_masses(pressures, **kwargs):
        return radtrans_parameters['mean_molar_mass'] * np.ones(pressures.size)

    # Test custom function
    spectral_model.calculate_mass_mixing_ratios = \
        petitRADTRANS.containers.spectral_model.BaseSpectralModel.calculate_mass_mixing_ratios
    spectral_model.calculate_mean_molar_masses = \
        calculate_mean_molar_masses

    radtrans = spectral_model.get_radtrans()

    return spectral_model, radtrans


atmosphere_lbl = init_radtrans_line_by_line()
spectral_model_lbl, radtrans_spectral_model_lbl = init_spectral_model_line_by_line()


def test_line_by_line_emission_spectrum():
    # Calculate an emission spectrum
    atmosphere_lbl.calc_flux(
        temp=temperature_guillot_2010,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_emission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl.freq * 1e4,
            'spectral_radiosity': atmosphere_lbl.flux
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_transmission_spectrum():
    # Calculate a transmission spectrum
    atmosphere_lbl.calc_transm(
        temp=temperature_isothermal,
        abunds=radtrans_parameters['mass_fractions'],
        gravity=radtrans_parameters['planetary_parameters']['surface_gravity'],
        mmw=radtrans_parameters['mean_molar_mass'],
        R_pl=radtrans_parameters['planetary_parameters']['radius'] * petitRADTRANS.nat_cst.r_jup_mean,
        P0_bar=radtrans_parameters['planetary_parameters']['reference_pressure']
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission'],
        comparison_dict={
            'wavelength': petitRADTRANS.nat_cst.c / atmosphere_lbl.freq * 1e4,
            'transit_radius': atmosphere_lbl.transm_rad / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_spectral_model_emission():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    wavelengths, spectral_radiosities = spectral_model.get_spectrum_model(
        radtrans=radtrans_spectral_model_lbl,
        mode='emission',
        parameters=None,
        update_parameters=True,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=False,
        shift=False,
        convolve=False,
        rebin=False,
        reduce=False
    )

    assert np.allclose(
        spectral_model.temperatures,
        temperature_guillot_2010,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_mixing_ratios[species],
            radtrans_parameters['mass_fractions'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        radtrans_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_emission'],
        comparison_dict={
            'wavelength': wavelengths,
            'spectral_radiosity': petitRADTRANS.physics.radiosity_erg_cm2radiosity_erg_hz(
                radiosity_erg_cm=spectral_radiosities * 1e7,  # W to erg
                wavelength=wavelengths * 1e-4  # um to cm
            )
        },
        relative_tolerance=relative_tolerance
    )


def test_line_by_line_spectral_model_transmission():
    spectral_model = copy.deepcopy(spectral_model_lbl)

    spectral_model.model_parameters['temperature_profile_mode'] = 'isothermal'
    spectral_model.model_parameters['temperature'] = temperature_isothermal

    wavelengths, transit_radii = spectral_model.get_spectrum_model(
        radtrans=radtrans_spectral_model_lbl,
        mode='transmission',
        parameters=None,
        update_parameters=True,
        telluric_transmittances=None,
        instrumental_deformations=None,
        noise_matrix=None,
        scale=False,
        shift=False,
        convolve=False,
        rebin=False,
        reduce=False
    )

    assert np.allclose(
        spectral_model.temperatures,
        temperature_isothermal,
        atol=0,
        rtol=relative_tolerance
    )

    for species in spectral_model.line_species:
        assert np.allclose(
            spectral_model.mass_mixing_ratios[species],
            radtrans_parameters['mass_fractions'][species],
            atol=0,
            rtol=relative_tolerance
        )

    assert np.allclose(
        spectral_model.mean_molar_masses,
        radtrans_parameters['mean_molar_mass'],
        atol=0,
        rtol=relative_tolerance
    )

    # Comparison
    compare_from_reference_file(
        reference_file=reference_filenames['line_by_line_transmission'],
        comparison_dict={
            'wavelength': wavelengths,
            'transit_radius': transit_radii / petitRADTRANS.nat_cst.r_jup_mean
        },
        relative_tolerance=relative_tolerance
    )
