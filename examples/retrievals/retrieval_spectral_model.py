"""High-resolution retrieval workflow with SpectralModel.

Based on the setup of Blain et al. (2024). Generates simulated high-resolution
transmission observations of HD 189733 b (with tellurics and instrumental
deformations), cleans them with the Polyfit preparing pipeline, then retrieves
temperature, abundances, cloud-top pressure, and velocity parameters (Kp,
V_rest, T0) using line-by-line opacities and a velocity-range forward model.

Equivalent to the ``retrieval_spectral_model`` documentation notebook.
"""
import os

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
from petitRADTRANS.plotlib import plot_result_corner
from petitRADTRANS.retrieval.preparing import polyfit
from petitRADTRANS.retrieval.retrieval import Retrieval
from petitRADTRANS.spectral_model import SpectralModel


def generate_simulated_observation(planet):
    """Build mock high-resolution transit observations with tellurics and deformations."""
    n_exposures = 19

    # Wavelengths at constant resolving power; S/N ~ 15000 uncertainties.
    data_wavelengths = resolving_space(1.519, 1.522, 2e5) * 1e-4  # (cm)
    data_uncertainties = 1 / 15000 * np.ones((1, n_exposures, data_wavelengths.size))

    times = 0.85 * planet.transit_duration * (np.linspace(0, 1, n_exposures) - 0.5)
    mid_transit_time = 2458004.424877  # (JD)
    dates = mid_transit_time + times / cst.s_cst.day
    orbital_phases = times / planet.orbital_period

    star_radial_velocity = planet.star_radial_velocity  # V_sys
    barycentric_velocities = planet.calculate_barycentric_velocities(time=dates, site_name="CAHA", time_format="jd")

    # Mock telluric transmittances (normally downloaded from SKYCALC).
    telluric_wavelengths = resolving_space(1.515, 1.525, 1e6) * 1e-4
    telluric_transmittances = np.ones(telluric_wavelengths.size)
    telluric_transmittances[2850:2870] = 0.85
    telluric_transmittances[2900:2930] = 0.7
    telluric_transmittances[3400:3500] = 0.1
    telluric_transmittances[3900:3950] = 0.5
    telluric_transmittances[4400:4475] = 0.3

    # Uncertainties are usually smaller within telluric/stellar lines.
    telluric_interp = np.interp(data_wavelengths, telluric_wavelengths, telluric_transmittances)
    data_uncertainties *= telluric_interp

    airmass = planet.calculate_airmass(time=dates, site_name="CAHA", time_format="jd")

    # Random wavelength-constant instrumental deformations (seeded for reproducibility).
    instrumental_deformations = (0.4 - 0.2) * np.random.default_rng(seed=12345).random(n_exposures) + 0.2
    instrumental_deformations = np.repeat(
        instrumental_deformations[:, np.newaxis], data_wavelengths.size, axis=-1
    )

    data_model = SpectralModel(
        pressures=jnp.logspace(-6, 2, 100),
        line_species=("CO-NatAbund", "H2O"),
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        line_opacity_mode="lbl",
        line_by_line_opacity_sampling=4,
        planet_radius=planet.radius,
        reference_gravity=planet.reference_gravity,
        reference_pressure=1e-2,
        star_radius=planet.star_radius,
        transit_duration=planet.transit_duration,
        orbital_period=planet.orbital_period,
        star_mass=planet.star_mass,
        orbit_semi_major_axis=planet.orbit_semi_major_axis,
        orbital_inclination=planet.orbital_inclination,
        rest_frame_velocity_shift=-5e5,  # (cm/s) V_rest
        system_observer_radial_velocities=star_radial_velocity - barycentric_velocities,
        temperature_profile_mode="isothermal",
        temperature=planet.equilibrium_temperature,
        opaque_cloud_top_pressure=1e-2,
        use_equilibrium_chemistry=False,
        imposed_mass_fractions={"CO-NatAbund": 1e-2, "H2O": 1e-3},
        filling_species={"H2": 37, "He": 12},
        rebinned_wavelengths=data_wavelengths,
        rebin_range_margin_power=4,
        convolve_resolving_power=8e4,
        mid_transit_time=0,
        times=times,
        airmass=airmass,
        tellurics_mask_threshold=0.8,
        polynomial_fit_degree=2,
        uncertainties=data_uncertainties,
    )

    # Generate noiseless data so we can check retrieval accuracy.
    data_wavelengths, data = data_model.calculate_spectrum(
        mode="transmission",
        update_parameters=True,
        telluric_transmittances_wavelengths=telluric_wavelengths,
        telluric_transmittances=telluric_transmittances,
        instrumental_deformations=instrumental_deformations,
        noise_matrix=None,
        scale=True,
        shift=True,
        use_transit_light_loss=True,
        convolve=True,
        rebin=True,
        prepare=False,
    )

    # Data and uncertainties must be masked arrays (nothing masked here).
    data = np.ma.masked_array(data)
    data_uncertainties = np.ma.masked_array(data_uncertainties)
    data.mask = np.zeros(data.shape, dtype=bool)
    data_uncertainties.mask = np.zeros(data_uncertainties.shape, dtype=bool)

    return {
        "data_model": data_model,
        "data_wavelengths": data_wavelengths,
        "data": data,
        "data_uncertainties": data_uncertainties,
        "airmass": airmass,
        "times": times,
        "orbital_phases": orbital_phases,
        "system_observer_radial_velocities": star_radial_velocity - barycentric_velocities,
    }


def retrieved_parameters_dict():
    return {
        "temperature": {
            "prior_parameters": [100, 4000], "prior_type": "uniform",
            "figure_title": r"T", "figure_label": r"T (K)",
        },
        "CO-NatAbund": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[CO]", "figure_label": r"$\log_{10}$(MMR) CO",
        },
        "H2O": {
            "prior_parameters": [-12, 0], "prior_type": "uniform",
            "figure_title": r"[H$_2$O]", "figure_label": r"$\log_{10}$(MMR) H$_2$O",
        },
        "log10_opaque_cloud_top_pressure": {
            "prior_parameters": [-10, 2], "prior_type": "uniform",
            "figure_title": r"[$P_c$]", "figure_label": r"$\log_{10}(P_c)$ ([Pa])",
            "figure_offset": 5,
        },
        "radial_velocity_semi_amplitude": {
            "prior_parameters": np.array([100e5, 200e5]), "prior_type": "uniform",
            "figure_title": r"$K_p$", "figure_label": r"$K_p$ (km$\cdot$s$^{-1}$)",
            "figure_coefficient": 1e-5,
        },
        "rest_frame_velocity_shift": {
            "prior_parameters": np.array([-10e5, 10e5]), "prior_type": "uniform",
            "figure_title": r"$V_\mathrm{rest}$", "figure_label": r"$V_\mathrm{rest}$ (km$\cdot$s$^{-1}$)",
            "figure_coefficient": 1e-5,
        },
        "mid_transit_time": {
            "prior_parameters": np.array([-150, 150]), "prior_type": "uniform",
            "figure_title": r"$T_0$", "figure_label": r"$T_0$)",
        },
    }


def build_forward_model(planet, sim, retrieved_parameters):
    """SpectralModel built over a velocity range so Kp/V_rest/T0 can be explored."""
    return SpectralModel.with_velocity_range(
        radial_velocity_semi_amplitude_range=retrieved_parameters["radial_velocity_semi_amplitude"]["prior_parameters"],
        rest_frame_velocity_shift_range=retrieved_parameters["rest_frame_velocity_shift"]["prior_parameters"],
        mid_transit_times_range=retrieved_parameters["mid_transit_time"]["prior_parameters"],
        pressures=jnp.logspace(-6, 2, 100),
        line_species=("CO-NatAbund", "H2O"),
        rayleigh_species=("H2", "He"),
        gas_continuum_contributors=("H2--H2", "H2--He"),
        line_opacity_mode="lbl",
        line_by_line_opacity_sampling=4,
        planet_radius=planet.radius,
        reference_gravity=planet.reference_gravity,
        reference_pressure=1e-2,
        star_radius=planet.star_radius,
        transit_duration=planet.transit_duration,
        orbital_period=planet.orbital_period,
        star_mass=planet.star_mass,
        orbit_semi_major_axis=planet.orbit_semi_major_axis,
        orbital_inclination=planet.orbital_inclination,
        rest_frame_velocity_shift=0,
        system_observer_radial_velocities=sim["system_observer_radial_velocities"],
        temperature_profile_mode="isothermal",
        temperature=planet.equilibrium_temperature,
        opaque_cloud_top_pressure=1e-2,
        use_equilibrium_chemistry=False,
        imposed_mass_fractions={"CO-NatAbund": 1e-12, "H2O": 1e-12},
        filling_species={"H2": 37, "He": 12},
        rebinned_wavelengths=sim["data_wavelengths"],
        rebin_range_margin_power=4,
        convolve_resolving_power=8e4,
        mid_transit_time=0,
        times=sim["times"],
        airmass=sim["airmass"],
        tellurics_mask_threshold=0.8,
        polynomial_fit_degree=2,
        uncertainties=sim["data_uncertainties"],
    )


def main():
    # Planet provides system parameters (not required, but convenient).
    planet = Planet.get("HD 189733 b")

    # In a real analysis you would load data here instead of simulating it.
    sim = generate_simulated_observation(planet)
    data_wavelengths = sim["data_wavelengths"]
    orbital_phases = sim["orbital_phases"]

    # Prepare (detrend) the data with the Polyfit pipeline to remove tellurics,
    # stellar lines, and instrumental deformations.
    prepared_data, _preparation_matrix, prepared_data_uncertainties = polyfit(
        spectrum=sim["data"],
        uncertainties=sim["data_uncertainties"],
        wavelengths=data_wavelengths,
        airmass=sim["airmass"],
        tellurics_mask_threshold=0.8,
        polynomial_fit_degree=2,
        full=True,
        apply_throughput_removal=True,
        apply_telluric_lines_removal=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pcolormesh(data_wavelengths[0] * 1e4, orbital_phases, prepared_data[0])
    ax.set_xlabel("Wavelength [microns]")
    ax.set_ylabel("Orbital phase")
    ax.set_title("Prepared noiseless data")

    retrieved_parameters = retrieved_parameters_dict()
    forward_model = build_forward_model(planet, sim, retrieved_parameters)

    # The preparing step alters uncertainties, so use the prepared uncertainties.
    data = {
        "data_1": forward_model.init_data(
            data_spectrum=prepared_data,
            data_wavelengths=data_wavelengths,
            data_uncertainties=prepared_data_uncertainties,
            data_name="data_1",
            retrieved_parameters=retrieved_parameters,
            mode="transmission",
            update_parameters=True,
            scale=True,
            shift=True,
            use_transit_light_loss=True,
            convolve=True,
            rebin=True,
            prepare=True,
        )
    }

    retrieval_name = "spectral_model_example"
    retrieval_directory = os.path.join(".", "results", retrieval_name)
    retrieval = Retrieval.from_data(
        data=data,
        retrieved_parameters=retrieved_parameters,
        retrieval_name=retrieval_name,
        output_directory=retrieval_directory,
        run_mode="retrieval",
    )

    # Few live points for speed (minutes on multiple processes). Use ~100 for real runs.
    retrieval.run(n_live_points=30, resume=False, seed=12345)

    # Corner plot. With simulated data we know the truth, so pass true_values.
    true_parameters = sim["data_model"].get_true_parameters(retrieved_parameters)
    plot_result_corner(
        retrieval_directory=[retrieval_directory],
        retrieved_parameters=retrieved_parameters,
        figure_name="corner_plot",
        label_kwargs={"fontsize": 16},
        title_kwargs={"fontsize": 14},
        figure_font_size=12,
        true_values=true_parameters,
        save=True,
        figure_directory=retrieval_directory,
        image_format="png",
    )

    # Best fit, median, and true models compared with the data.
    wavelengths, best_fit_spectra, _ = retrieval.calculate_forward_model(parameters="best fit", data="data_1")
    _, median_spectra, _ = retrieval.calculate_forward_model(parameters="median", data="data_1")
    _, true_spectra, _ = retrieval.calculate_forward_model(parameters=true_parameters, data="data_1")

    i_t0 = int(np.round(orbital_phases.size / 2))
    fig, axes = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
    axes[0].plot(wavelengths[0] * 1e4, true_spectra[0, i_t0], ls="-", color="C2", label='"True" spectrum')
    axes[0].plot(wavelengths[0] * 1e4, median_spectra[0, i_t0], ls="-", color="C0", label="Median spectrum")
    axes[0].plot(wavelengths[0] * 1e4, best_fit_spectra[0, i_t0], ls="-", color="r", label="Best-fit spectrum")
    axes[0].plot(wavelengths[0] * 1e4, prepared_data[0, i_t0], ls="--", color="k", label="Data")
    for spectra, color in ((true_spectra, "C2"), (median_spectra, "C0"), (best_fit_spectra, "r")):
        axes[1].plot(
            wavelengths[0] * 1e4,
            (spectra[0, i_t0] - prepared_data[0, i_t0]) / prepared_data_uncertainties[0, i_t0],
            ls="-", color=color,
        )
    axes[0].set_title(rf"Comparison with best fit model ($\phi = {orbital_phases[i_t0]:.2f}$)")
    axes[0].set_ylabel("Spectrum [A.U.]")
    axes[1].set_xlabel("Wavelength [microns]")
    axes[1].set_ylabel(r"Residuals ($\sigma$)")
    fig.legend()

    # Best fit parameters vs truth.
    best_fit_parameters, max_likelihood, chi2 = retrieval.get_best_fit_parameters(return_max_likelihood=True)
    print(
        f"Best fit reduced chi2 against the prepared data: "
        f"{chi2 / (prepared_data.size - len(retrieved_parameters)):.2e} (from Log(L) = {max_likelihood:.2f})"
    )
    print("\nBest fit parameters (truth):")
    for parameter_name, value in best_fit_parameters.items():
        print(f"    - '{parameter_name}': {value:.3f} ({true_parameters[parameter_name]:.3f})")

    # Posterior samples.
    samples = retrieval.get_samples_dict()
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(samples["H2O"], density=True, histtype="step")
    ax.set_xlabel(retrieved_parameters["H2O"]["figure_label"])
    ax.set_ylabel("Probability density")

    plt.show()


if __name__ == "__main__":
    main()
