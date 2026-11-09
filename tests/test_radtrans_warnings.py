import warnings

from .context import petitRADTRANS


def _make_minimal_radtrans(cloud_species=(),
                           rayleigh_species=(),
                           scattering_in_emission=False):
    atmosphere = petitRADTRANS.radtrans.Radtrans.__new__(
        petitRADTRANS.radtrans.Radtrans
    )
    atmosphere._cloud_species = tuple(cloud_species)
    atmosphere._rayleigh_species = tuple(rayleigh_species)
    atmosphere._line_opacity_mode = "c-k"
    atmosphere._scattering_in_emission = scattering_in_emission
    atmosphere._anisotropic_cloud_scattering = "auto"

    return atmosphere


def test_setting_scattering_in_emission_warns_without_scattering_species():
    atmosphere = _make_minimal_radtrans(cloud_species=(), rayleigh_species=())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.scattering_in_emission.fset(atmosphere, True)

    assert any(
        "scattering_in_emission=True, but 'cloud_species' and" in str(warning.message)
        for warning in caught
    )


def test_removing_all_scattering_species_warns_when_scattering_in_emission():
    # The species setters used to raise NameError instead of checking the new species
    atmosphere = _make_minimal_radtrans(cloud_species=("MgSiO3(s)",), scattering_in_emission=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.cloud_species.fset(atmosphere, [])

    assert atmosphere.cloud_species == []
    assert any(
        "scattering_in_emission=True, but 'cloud_species' and" in str(warning.message)
        for warning in caught
    )

    atmosphere = _make_minimal_radtrans(rayleigh_species=("H2",), scattering_in_emission=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.rayleigh_species.fset(atmosphere, [])

    assert atmosphere.rayleigh_species == []
    assert any(
        "scattering_in_emission=True, but 'cloud_species' and" in str(warning.message)
        for warning in caught
    )


def test_setting_a_scattering_species_does_not_warn():
    atmosphere = _make_minimal_radtrans(scattering_in_emission=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.rayleigh_species.fset(atmosphere, ["H2"])

    assert not any("'cloud_species' and" in str(warning.message) for warning in caught)
