import warnings

from .context import petitRADTRANS


def test_spectral_parameters_exclude_deprecated_cloud_particle_aliases():
    spectral_model = petitRADTRANS.spectral_model.SpectralModel.__new__(
        petitRADTRANS.spectral_model.SpectralModel
    )

    spectral_parameters = spectral_model._get_spectral_parameters(
        remove_flag=True, remove_branching=True
    )

    assert "cloud_particles_mean_radii" not in spectral_parameters
    assert "cloud_particles_radius_distribution_std" not in spectral_parameters
    assert "cloud_particles_radius_distribution" not in spectral_parameters
    assert "cloud_particles_number_density_grid" not in spectral_parameters

    assert "cloud_particle_mean_radii" in spectral_parameters
    assert "cloud_particle_radius_distribution_std" in spectral_parameters
    assert "cloud_particle_number_density_grid" in spectral_parameters

    assert "cloud_particle_radius_distribution" not in spectral_parameters


def test_emission_wrapper_normalizes_deprecated_cloud_particle_kwargs():
    class DummySpectralModel:
        temperatures = [1000.0]
        mass_fractions = {}
        mean_molar_masses = [2.3]
        wavelengths = None
        fluxes = None

        def calculate_flux(self, **kwargs):
            self.kwargs = kwargs
            return "wavelengths", "fluxes", {}

    spectral_model = DummySpectralModel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.spectral_model.SpectralModel.calculate_emission_spectrum(
            spectral_model,
            reference_gravity=1.0,
            cloud_particles_number_density_grid={"Mg2SiO4": [1.0]},
        )

    assert spectral_model.kwargs["cloud_particle_number_density_grid"] == {
        "Mg2SiO4": [1.0]
    }
    assert any(
        "cloud_particles_number_density_grid" in str(warning.message)
        for warning in caught
    )


def test_transmission_wrapper_normalizes_deprecated_cloud_particle_kwargs():
    class DummySpectralModel:
        temperatures = [1000.0]
        mass_fractions = {}
        mean_molar_masses = [2.3]
        wavelengths = None
        transit_radii = None

        def calculate_transit_radii(self, **kwargs):
            self.kwargs = kwargs
            return "wavelengths", "transit_radii", {}

    spectral_model = DummySpectralModel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        petitRADTRANS.spectral_model.SpectralModel.calculate_transmission_spectrum(
            spectral_model,
            reference_gravity=1.0,
            reference_pressure=0.1,
            planet_radius=1.0,
            cloud_particles_radius_distribution="hansen",
        )

    assert spectral_model.kwargs["cloud_particle_radius_distribution"] == "hansen"
    assert any(
        "cloud_particles_radius_distribution" in str(warning.message)
        for warning in caught
    )