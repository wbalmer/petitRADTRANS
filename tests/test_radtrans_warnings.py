import warnings

import numpy as np

from .context import petitRADTRANS


def _make_minimal_radtrans(cloud_species=(), scattering_in_emission=False):
    atmosphere = petitRADTRANS.radtrans.Radtrans.__new__(
        petitRADTRANS.radtrans.Radtrans
    )
    atmosphere._cloud_species = tuple(cloud_species)
    atmosphere._line_opacity_mode = "c-k"
    atmosphere._scattering_in_emission = scattering_in_emission
    atmosphere._anisotropic_cloud_scattering = "auto"

    return atmosphere


def test_setting_scattering_in_emission_warns_without_cloud_species():
    atmosphere = _make_minimal_radtrans(cloud_species=())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.scattering_in_emission.fset(atmosphere, True)

    assert any(
        "scattering_in_emission=True, but 'cloud_species' is empty" in str(warning.message)
        for warning in caught
    )


def test_partial_cloud_fraction_warns_without_patchy_clouds():
    atmosphere = _make_minimal_radtrans(cloud_species=("Mg2SiO4",))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, patchy_clouds, _, _ = atmosphere._init_spectral_function(
            is_emission=False,
            reference_gravity=1.0,
            patchy_clouds=[],
            return_clear_spectrum=False,
            mass_fractions={"Mg2SiO4": np.array([1e-6])},
            cloud_fraction=0.5,
            opaque_cloud_top_pressure=None,
        )

    assert patchy_clouds == ()
    assert any(
        "there are no patchy clouds" in str(warning.message)
        for warning in caught
    )


def test_partial_cloud_fraction_with_patchy_clouds_does_not_warn():
    atmosphere = _make_minimal_radtrans(cloud_species=("Mg2SiO4",))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, patchy_clouds, _, _ = atmosphere._init_spectral_function(
            is_emission=False,
            reference_gravity=1.0,
            patchy_clouds=["Mg2SiO4"],
            return_clear_spectrum=False,
            mass_fractions={"Mg2SiO4": np.array([1e-6])},
            cloud_fraction=0.5,
            opaque_cloud_top_pressure=None,
        )

    assert patchy_clouds == ("Mg2SiO4",)
    assert not any(
        "there are no patchy clouds" in str(warning.message)
        for warning in caught
    )


def test_partial_cloud_fraction_with_power_law_clouds_does_not_warn():
    atmosphere = _make_minimal_radtrans(cloud_species=("Mg2SiO4",))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, patchy_clouds, _, _ = atmosphere._init_spectral_function(
            is_emission=False,
            reference_gravity=1.0,
            patchy_clouds=[],
            return_clear_spectrum=False,
            mass_fractions={"Mg2SiO4": np.array([1e-6])},
            cloud_fraction=0.5,
            opaque_cloud_top_pressure=None,
            has_partial_coverage_background_cloud=True,
        )

    assert patchy_clouds == ()
    assert not any(
        "there are no patchy clouds" in str(warning.message)
        for warning in caught
    )