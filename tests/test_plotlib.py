import matplotlib.pyplot as plt
import numpy as np

from .context import petitRADTRANS
from petitRADTRANS.plotlib import plot_opacity_contributions


def _make_minimal_radtrans_stub():
    atmosphere = object.__new__(petitRADTRANS.radtrans.Radtrans)
    atmosphere._pressures = np.array([1e-6, 1e-3, 1.0])
    atmosphere._wavelength_boundaries = np.array([0.3, 1.0])
    atmosphere._line_species = ("H2O",)
    atmosphere._gas_continuum_contributors = ()
    atmosphere._cloud_species = ()
    atmosphere._rayleigh_species = ()
    atmosphere._line_opacity_mode = "c-k"
    atmosphere._line_by_line_opacity_sampling = 1
    atmosphere._scattering_in_emission = False
    atmosphere._emission_angle_grid = None
    atmosphere._anisotropic_cloud_scattering = "auto"
    atmosphere._path_input_data = ""
    atmosphere.calculate_flux = lambda **_: (
        np.array([1e-4, 2e-4]),
        np.array([1.0, 1.1]),
    )
    atmosphere.calculate_transit_radii = lambda **_: (
        np.array([1e-4, 2e-4]),
        np.array([1.0, 1.1]),
    )
    return atmosphere


class _FakeContributionRadtrans:
    def __init__(self, **kwargs):
        self.line_species = tuple(kwargs.get("line_species", ()))
        self.calculate_flux = lambda **_: (
            np.array([1e-4, 2e-4]),
            np.array([1.0, 1.1]),
        )
        self.calculate_transit_radii = lambda **_: (
            np.array([1e-4, 2e-4]),
            np.array([1.0, 1.1]),
        )

    def __getattr__(self, name):
        return super().__getattribute__(name)


def test_calculate_opacity_contribution_spectra_skips_incomplete_power_law(monkeypatch):
    original_radtrans_class = petitRADTRANS.radtrans.Radtrans
    atmosphere = _make_minimal_radtrans_stub()

    monkeypatch.setattr(
        petitRADTRANS.radtrans,
        "Radtrans",
        _FakeContributionRadtrans,
    )

    contributions = original_radtrans_class.calculate_opacity_contribution_spectra(
        atmosphere,
        mode="transmission",
        temperatures=np.ones(atmosphere._pressures.size),
        mass_fractions={"H2O": np.ones(atmosphere._pressures.size)},
        mean_molar_masses=np.ones(atmosphere._pressures.size),
        reference_gravity=1e3,
        planet_radius=7e9,
        reference_pressure=1e-2,
        power_law_opacity_350nm=1e-5,
        power_law_opacity_coefficient=None,
    )

    assert contributions["power_law"] is None
    assert "Total" in contributions
    assert "line_species" in contributions


def test_plot_opacity_contributions_preserves_existing_spectral_model_values(
    monkeypatch,
):
    spectral_model = object.__new__(petitRADTRANS.spectral_model.SpectralModel)
    spectral_model._wavelength_boundaries = np.array([0.3, 1.0])
    spectral_model.model_parameters = {"gray_opacity": 3.14}

    recorded_gray_opacity = []

    def fake_calculate_opacity_contribution_spectra(self, mode=None, **kwargs):
        recorded_gray_opacity.append(self.model_parameters["gray_opacity"])
        return {
            "Total": (
                np.array([1e-4, 2e-4]),
                np.array([1.0, 1.1]),
            )
        }

    monkeypatch.setattr(
        petitRADTRANS.spectral_model.SpectralModel,
        "calculate_opacity_contribution_spectra",
        fake_calculate_opacity_contribution_spectra,
    )

    plot_opacity_contributions(
        spectral_model,
        mode="transmission",
    )

    assert recorded_gray_opacity == [3.14]
    plt.close("all")
