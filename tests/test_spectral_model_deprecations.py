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
