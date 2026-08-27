import warnings

import numpy as np

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
    atmosphere = _make_minimal_radtrans(cloud_species=(),rayleigh_species=())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.radtrans.Radtrans.scattering_in_emission.fset(atmosphere, True)

    assert any(
        "scattering_in_emission=True, but 'cloud_species' and" in str(warning.message)
        for warning in caught
    )
