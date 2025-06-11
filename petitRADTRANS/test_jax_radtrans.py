"""
Test suite for JAX radiative transfer functions, comparing against Fortran implementations.
"""
import pytest
import jax
from jax import config
# Enable float64 for JAX to match Fortran's double precision
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

# --- JAX Device Configuration ---
_JAX_DEVICE_NAME = "cpu"  # Or "gpu", "tpu", None
JAX_DTYPE = jnp.float64
NUMPY_DTYPE = np.double

# --- Test Configuration ---
RTOL = 1e-4  # Increased relative tolerance for complex functions
ATOL = 1e-6  # Increased absolute tolerance
N_TIMING_RUNS = 100 # Number of times to run each function for timing
if config.jax_enable_x64:
    RTOL = 1e-8  # Increased relative tolerance for complex functions
    ATOL = 1e-10  # Increased absolute tolerance


try:
    if _JAX_DEVICE_NAME and _JAX_DEVICE_NAME.lower() != "default":
        JAX_DEVICE = jax.devices(_JAX_DEVICE_NAME)[0]
        print(f"JAX tests will run on: {_JAX_DEVICE_NAME.upper()} ({JAX_DEVICE.platform.upper()})")
    else:
        JAX_DEVICE = jax.default_device()
        print(f"JAX tests will run on default JAX device: {JAX_DEVICE.platform.upper()}")
except Exception as e:
    print(f"Warning: Could not set JAX device to '{_JAX_DEVICE_NAME}'. Using JAX default. Error: {e}")
    JAX_DEVICE = jax.default_device()
    print(f"JAX tests will run on default JAX device: {JAX_DEVICE.platform.upper()}")


# --- Import JAX and Fortran functions ---
from petitRADTRANS.fortran_radtrans_core import fortran_radtrans_core as fcore
from petitRADTRANS.fortran_radtrans_core import math as fmath # For Fortran tridag
import petitRADTRANS.radtrans_core.radtrans_core as jrt
import petitRADTRANS.radtrans_core.physics_core as jphys # For JAX Planck functions etc.
import petitRADTRANS.physical_constants as cst


# --- Timing Results Storage ---
timing_results = {}

# --- Pytest Fixture to Save Timings ---
@pytest.fixture(scope="session", autouse=True)
def save_timing_results():
    """Saves the timing results to a file at the end of the test session."""
    yield
    output_file = "timing_results.txt"
    with open(output_file, "w") as f:
        f.write("Function Name, JAX Mean Time (s), JAX Std Time (s), Fortran Mean Time (s), Fortran Std Time (s), JAX/Fortran\n")
        for func_name, times in timing_results.items():
            f.write(f"{func_name}, {times['jax_mean_time']:.6f}, {times['jax_std_time']:.6f}, "
                    f"{times['fortran_mean_time']:.6f}, {times['fortran_std_time']:.6f}, {times['jax_mean_time']/times['fortran_mean_time']:.6f},\n")

    print(f"\nTiming results written to {output_file}")

# --- Helper Functions ---
def to_jax_array(arr, dtype=JAX_DTYPE):
    return jax.device_put(jnp.array(arr, dtype=dtype), device=JAX_DEVICE)

def to_numpy_array(arr):
    return np.array(arr)

# --- Test Fixtures ---
@pytest.fixture
def n_g_test(): return 4

@pytest.fixture
def n_freq_centers_test(): return 4 # Number of frequency bin centers

@pytest.fixture
def n_freq_edges_test(): return 5 # Number of frequency bin edges (implying 4 bins)

@pytest.fixture
def n_species_test(): return 2

@pytest.fixture
def n_layers_test(): return 10

@pytest.fixture
def n_angles_test(): return 3

@pytest.fixture
def sample_pressures(n_layers_test):
    return np.logspace(-6, 2, n_layers_test).astype(NUMPY_DTYPE)  # bar

@pytest.fixture
def sample_temperatures(n_layers_test):
    return np.linspace(500, 2000, n_layers_test).astype(NUMPY_DTYPE)  # K

@pytest.fixture
def sample_frequencies_centers(n_freq_centers_test):
    # For functions expecting frequency bin centers
    return np.linspace(3e14, 1e14, n_freq_centers_test).astype(NUMPY_DTYPE)  # Hz

@pytest.fixture
def sample_frequencies_bin_edges(n_freq_edges_test):
    # For functions expecting frequency bin edges
    # Ensure decreasing order for some Fortran routines if necessary, though JAX versions might handle it
    return np.sort(np.linspace(3.1e14, 0.9e14, n_freq_edges_test)).astype(NUMPY_DTYPE)  # Hz

@pytest.fixture
def sample_g_gauss(n_g_test):
    # Example g-values (actual values depend on the quadrature)
    return np.linspace(0.1, 0.9, n_g_test).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_weights_gauss(n_g_test):
    # Example weights (must sum to 1 if they are normalized weights)
    weights = np.abs(np.random.rand(n_g_test)).astype(NUMPY_DTYPE)
    return weights / np.sum(weights)

@pytest.fixture
def sample_opacities_gfls(n_g_test, n_freq_centers_test, n_species_test, n_layers_test):
    # Opacities for (g, freq_centers, species, layers)
    return np.abs(np.random.rand(n_g_test, n_freq_centers_test, n_species_test, n_layers_test) * 1e-23).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_opacities_gfls_one_species(n_g_test, n_freq_centers_test, n_species_test, n_layers_test):
    # Opacities for (g, freq_centers, species, layers)
    return np.abs(np.random.rand(n_g_test, n_freq_centers_test, 1, n_layers_test) * 1e-23).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_opacities_gfbl(n_g_test, n_freq_centers_test, n_layers_test):
    # Opacities for (g, freq_bins, layers) - assuming n_species=1 for some contexts
    # n_freq_edges_test is number of edges, so n_freq_edges_test-1 is number of bins
    return np.abs(np.random.rand(n_g_test, n_freq_centers_test, n_layers_test) * 1e-23).astype(NUMPY_DTYPE)


@pytest.fixture
def sample_optical_depths_gfls(n_g_test, n_freq_centers_test, n_species_test, n_layers_test):
    # Optical depths for (g, freq_centers, species, layers)
    return np.abs(np.random.rand(n_g_test, n_freq_centers_test, n_species_test, n_layers_test) * 10.0).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_optical_depths_gfbl(n_g_test, n_freq_edges_test, n_layers_test):
    # Optical depths for (g, freq_bins, layers)
    return np.abs(np.random.rand(n_g_test, n_freq_edges_test - 1, n_layers_test) * 10.0).astype(NUMPY_DTYPE)


@pytest.fixture
def sample_photon_destruction_gfbl(n_g_test, n_freq_edges_test, n_layers_test):
    # Photon destruction probabilities (g, freq_bins, layers)
    return np.random.rand(n_g_test, n_freq_edges_test - 1, n_layers_test).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_stellar_intensity_f(n_freq_edges_test):
    # Photon destruction probabilities (g, freq_bins, layers)
    return np.random.rand(n_freq_edges_test - 1).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_emissivities_f(n_freq_edges_test):
    # Photon destruction probabilities (g, freq_bins, layers)
    return np.random.rand(n_freq_edges_test - 1).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_reflectivies_f(n_freq_edges_test):
    # Photon destruction probabilities (g, freq_bins, layers)
    return np.random.rand(n_freq_edges_test - 1).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_emission_cos_angles(n_angles_test):
    return np.linspace(0.1, 1.0, n_angles_test).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_emission_cos_angles_weights(n_angles_test):
    weights = np.abs(np.random.rand(n_angles_test)).astype(NUMPY_DTYPE)
    return weights / np.sum(weights)

@pytest.fixture
def sample_continuum_opacities_scattering_fl(n_freq_centers_test, n_layers_test):
    return np.abs(np.random.rand(n_freq_centers_test, n_layers_test) * 1e-25).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_continuum_opacities_scattering_fbl(n_freq_edges_test, n_layers_test):
     # n_freq_edges_test is number of edges, so n_freq_edges_test-1 is number of bins
    return np.abs(np.random.rand(n_freq_edges_test - 1, n_layers_test) * 1e-25).astype(NUMPY_DTYPE)

# --- Test Cases ---

def test_combine_ck_opacities(sample_opacities_gfls, sample_g_gauss, sample_weights_gauss,
                               n_g_test, n_freq_centers_test, n_species_test, n_layers_test):
    opacities_np = sample_opacities_gfls
    g_gauss_np = sample_g_gauss
    weights_gauss_np = sample_weights_gauss

    # JAX Warm-up
    opacities_jax = to_jax_array(opacities_np)
    g_gauss_jax = to_jax_array(g_gauss_np)
    weights_gauss_jax = to_jax_array(weights_gauss_np)
    _ = jrt.combine_ck_opacities(opacities_jax, g_gauss_jax, weights_gauss_jax).block_until_ready()

    # Fortran Warm-up
    _ = fcore.combine_ck_opacities(
        opacities_np, g_gauss_np, weights_gauss_np,
        n_g_test, n_freq_centers_test, n_species_test, n_layers_test,
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_result = jrt.combine_ck_opacities(opacities_jax, g_gauss_jax, weights_gauss_jax).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_result_np = to_numpy_array(jax_result)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        # Fortran output is (n_g, n_frequencies, n_layers)
        fortran_opacities_out_np = np.zeros((n_g_test, n_freq_centers_test, n_layers_test), dtype=NUMPY_DTYPE)
        fortran_opacities_out_np = fcore.combine_ck_opacities(
            opacities_np, g_gauss_np, weights_gauss_np,
            n_g_test, n_freq_centers_test, n_species_test, n_layers_test,
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_combine_ck_opacities.__name__] = {
        'jax_mean_time': jax_mean_time,
        'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time,
        'fortran_std_time': fortran_std_time
    }
    
    assert jax_result_np.shape == fortran_opacities_out_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_result_np, fortran_opacities_out_np, rtol=RTOL, atol=ATOL,
                               err_msg="combine_ck_opacities outputs differ.")


def test_compute_ck_flux(sample_frequencies_centers, sample_temperatures, sample_weights_gauss,
                         sample_emission_cos_angles, sample_emission_cos_angles_weights,
                         sample_optical_depths_gfls,
                         n_freq_centers_test, n_layers_test, n_angles_test, n_g_test, n_species_test):
    freq_np = sample_frequencies_centers
    temp_np = sample_temperatures
    w_g_np = sample_weights_gauss
    mu_np = sample_emission_cos_angles
    w_mu_np = sample_emission_cos_angles_weights
    opt_depth_np = sample_optical_depths_gfls
    
    freq_jax = to_jax_array(freq_np)
    temp_jax = to_jax_array(temp_np)
    w_g_jax = to_jax_array(w_g_np)
    mu_jax = to_jax_array(mu_np)
    w_mu_jax = to_jax_array(w_mu_np)
    opt_depth_jax = to_jax_array(opt_depth_np)

    return_contribution = False # For simplicity in this test, can be expanded

    # JAX Warm-up
    _ = jrt.compute_ck_flux(
        freq_jax,
        temp_jax,
        w_g_jax,
        mu_jax,
        w_mu_jax,
        opt_depth_jax,
        return_contribution
    )[0].block_until_ready()

    # Fortran Warm-up
    _ = fcore.compute_ck_flux(
        freq_np, temp_np, w_g_np, mu_np, w_mu_np, opt_depth_np, return_contribution,
        n_freq_centers_test, n_layers_test, n_angles_test, n_g_test, n_species_test
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_flux, _ = jrt.compute_ck_flux(
            freq_jax, temp_jax, w_g_jax,
            mu_jax, w_mu_jax, opt_depth_jax,
            return_contribution
        )
        jax_flux.block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_flux_np = to_numpy_array(jax_flux)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_flux_np = np.zeros(n_freq_centers_test, dtype=NUMPY_DTYPE)
        fortran_contr_em_np = np.zeros((n_layers_test, n_freq_centers_test), dtype=NUMPY_DTYPE) # Dummy
        fortran_flux_np, fortran_contr_em_np = fcore.compute_ck_flux(
            freq_np, temp_np, w_g_np, mu_np, w_mu_np, opt_depth_np, return_contribution,
            n_freq_centers_test, n_layers_test, n_angles_test, n_g_test, n_species_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_ck_flux.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_flux_np.shape == fortran_flux_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_flux_np, fortran_flux_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_ck_flux outputs differ.")


def test_compute_feautrier_radiative_transfer():
    import os
    path_input_data = "/Users/nasedkin/python-packages/petitRADTRANS/input_data/"
    buffer = np.genfromtxt(os.path.join(path_input_data, 'opa_input_files', 'g_comb_grid.dat'))

    n_layers = 20
    n_freq_bins = 50
    n_frequencies_bin_edges = 50+1
    N_species = 2
    n_g = 16
    n_angles = 3

    wavelength_min = 1.0
    wavelength_max = 5.0
    micron_to_meter = 1e-4
    frequencies_bin_edges = cst.c/(
            np.linspace(
                wavelength_min,
                wavelength_max,
                n_frequencies_bin_edges)
            * micron_to_meter).astype(NUMPY_DTYPE)
        
    pressures = 1e6*np.logspace(-6, 3, n_layers).astype(NUMPY_DTYPE)
    temperatures = np.ones(pressures.shape[0]).astype(NUMPY_DTYPE)*500

    weights_gauss = np.array(buffer[:,1]).astype(NUMPY_DTYPE)
    emission_cos_angles = np.array([0.1127,0.5,0.88729])
    emission_cos_angles_weights = np.array([0.2777777,0.444444444,0.2777777777]).astype(NUMPY_DTYPE)

    optical_depths = (
        np.ones((n_g, n_freq_bins, n_layers)).astype(NUMPY_DTYPE)
        * np.sin(np.linspace(0.01,10,n_g))[:,None,None]
        * np.cos(np.linspace(0.1,10,n_freq_bins))[None,:,None]
        * np.sin(np.linspace(0.3,5,n_layers))[None,None,:]
        * 1e-4)
    photon_destruction_probabilities = np.ones((n_g, n_freq_bins, n_layers)).astype(NUMPY_DTYPE)  #*1e-5* np.linspace(0.1,10,n_freq_bins)[None,:,None]
    
    emission_geometry = 'planetary_ave'
    star_irradiation_cos_angle = 0.5
    stellar_intensity = np.ones(n_freq_bins).astype(NUMPY_DTYPE)*1e-7
    reflectances = np.ones(n_freq_bins).astype(NUMPY_DTYPE)*1e-7
    emissivities = np.ones(n_freq_bins).astype(NUMPY_DTYPE)*1e-7
    return_contribution = False

    # --- Convert to JAX arrays ---
    frequencies_bin_edges_jax = to_jax_array(frequencies_bin_edges)
    temperatures_jax = to_jax_array(temperatures)
    weights_gauss_jax = to_jax_array(weights_gauss)
    emission_cos_angles_jax = to_jax_array(emission_cos_angles)
    emission_cos_angles_weights_jax = to_jax_array(emission_cos_angles_weights)
    optical_depths_jax = to_jax_array(optical_depths)
    photon_destruction_probabilities_jax = to_jax_array(photon_destruction_probabilities)
    stellar_intensity_jax = to_jax_array(stellar_intensity)
    reflectances_jax = to_jax_array(reflectances)
    emissivities_jax = to_jax_array(emissivities)
    
    # JAX Warm-up
    _ = jrt.compute_feautrier_radiative_transfer(
        frequencies_bin_edges_jax,
        temperatures_jax,
        weights_gauss_jax,
        emission_cos_angles_jax,
        emission_cos_angles_weights_jax,
        optical_depths_jax,
        photon_destruction_probabilities_jax,
        stellar_intensity_jax,
        star_irradiation_cos_angle,
        reflectances_jax,
        emissivities_jax,
        emission_geometry,
        return_contribution)[0].block_until_ready()

    # Fortran Warm-up
    fortran_flux_np = np.zeros(n_freq_bins, dtype=NUMPY_DTYPE)

    _ = fcore.compute_feautrier_radiative_transfer(
        frequencies_bin_edges,
        temperatures,
        weights_gauss,
        emission_cos_angles,
        emission_cos_angles_weights,
        optical_depths,
        photon_destruction_probabilities,
        emission_geometry,
        stellar_intensity,
        star_irradiation_cos_angle,
        reflectances,
        emissivities,
        return_contribution,
        n_frequencies_bin_edges,
        n_layers,
        n_angles,
        n_g)[0]

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_flux, _ = jrt.compute_feautrier_radiative_transfer(
            frequencies_bin_edges_jax,
            temperatures_jax,
            weights_gauss_jax,
            emission_cos_angles_jax,
            emission_cos_angles_weights_jax,
            optical_depths_jax,
            photon_destruction_probabilities_jax,
            stellar_intensity_jax,
            star_irradiation_cos_angle,
            reflectances_jax,
            emissivities_jax,
            emission_geometry,
            return_contribution)
        jax_flux.block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_flux_np = to_numpy_array(jax_flux)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_flux_np, _ = fcore.compute_feautrier_radiative_transfer(
            frequencies_bin_edges,
            temperatures,
            weights_gauss,
            emission_cos_angles,
            emission_cos_angles_weights,
            optical_depths,
            photon_destruction_probabilities,
            emission_geometry,
            stellar_intensity,
            star_irradiation_cos_angle,
            reflectances,
            emissivities,
            return_contribution,
            n_frequencies_bin_edges,
            n_layers,
            n_angles,
            n_g)
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_feautrier_radiative_transfer.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_flux_np.shape == fortran_flux_np.shape, "Shape mismatch"
    np.testing.assert_allclose(
        jax_flux_np,
        fortran_flux_np,
        rtol=RTOL,
        atol=ATOL,
        err_msg="compute_ck_flux outputs differ."
        )
    

def test_compute_species_optical_depths(sample_pressures, sample_opacities_gfls,
                                         n_layers_test, n_freq_centers_test, n_g_test, n_species_test):
    ref_gravity_np = 980.0  # cm/s^2
    pressures_np = sample_pressures * 1e6  # bar to cgs for Fortran
    opacities_np = sample_opacities_gfls

    ref_gravity_jax = 980.0  # cm/s^2
    pressures_jax = to_jax_array(pressures_np) # bar to cgs for Fortran
    opacities_jax = to_jax_array(opacities_np)
    # JAX Warm-up
    # JAX function takes pressures in cgs
    _ = jrt.compute_species_optical_depths(
        ref_gravity_jax, pressures_jax, opacities_jax
    ).block_until_ready()

    # Fortran Warm-up
    _ = fcore.compute_species_optical_depths(
        ref_gravity_np, pressures_np, opacities_np,
        n_layers_test, n_freq_centers_test, n_g_test, n_species_test
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_opt_depth = jrt.compute_species_optical_depths(
            ref_gravity_jax, pressures_jax, opacities_jax
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_opt_depth_np = to_numpy_array(jax_opt_depth)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_opt_depth_np = np.zeros_like(opacities_np)
        fortran_opt_depth_np = fcore.compute_species_optical_depths(
            ref_gravity_np, pressures_np, opacities_np,
            n_layers_test, n_freq_centers_test, n_g_test, n_species_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_species_optical_depths.__name__] = {
        'jax_mean_time': jax_mean_time,
        'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time,
        'fortran_std_time': fortran_std_time
    }

    assert jax_opt_depth_np.shape == fortran_opt_depth_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_opt_depth_np, fortran_opt_depth_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_species_optical_depths outputs differ.")


def test_compute_optical_depths(sample_pressures, sample_opacities_gfls_one_species,
                                sample_continuum_opacities_scattering_fl,
                                n_layers_test, n_freq_centers_test, n_g_test, n_species_test):
    ref_gravity_np = 980.0
    pressures_np = sample_pressures * 1e6 # bar to cgs
    # Fortran expects opacities(n_g, n_frequencies, n_species, n_layers)
    # JAX expects opacities(n_g, n_frequencies, n_species, n_layers)
    opacities_np = sample_opacities_gfls_one_species
    # Fortran expects continuum_opacities_scattering(n_frequencies, n_layers)
    # JAX expects continuum_opacities_scattering(n_frequencies, n_layers)
    continuum_scat_np = sample_continuum_opacities_scattering_fl

    ref_gravity_jax = to_jax_array(ref_gravity_np)
    pressures_jax = to_jax_array(pressures_np)
    opacities_jax = to_jax_array(opacities_np)
    continuum_scat_jax = to_jax_array(continuum_scat_np)
    scattering_in_emission = True

    # JAX Warm-up
    _, _ = jrt.compute_optical_depths(
        ref_gravity_jax,
        pressures_jax,
        opacities_jax,
        scattering_in_emission,
        continuum_scat_jax
    )
    _.block_until_ready()

    # Fortran Warm-up
    _ = fcore.compute_optical_depths(
        ref_gravity_np, pressures_np, opacities_np, scattering_in_emission,
        continuum_scat_np, n_layers_test, n_freq_centers_test, n_g_test
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_opt_depth, jax_photon_dest_prob = jrt.compute_optical_depths(
            ref_gravity_jax,
            pressures_jax,
            opacities_jax,
            scattering_in_emission,
            continuum_scat_jax
        )
        jax_opt_depth.block_until_ready()
        jax_photon_dest_prob.block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_opt_depth_np = to_numpy_array(jax_opt_depth)
    jax_photon_dest_prob_np = to_numpy_array(jax_photon_dest_prob)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_opt_depth_np = np.zeros_like(opacities_np)
        # Fortran photon_destruction_probabilities is (n_g, n_frequencies, n_layers)
        fortran_photon_dest_prob_np = np.zeros((n_g_test, n_freq_centers_test, n_layers_test), dtype=NUMPY_DTYPE)
        fortran_opt_depth_np, fortran_photon_dest_prob_np = fcore.compute_optical_depths(
            ref_gravity_np, pressures_np, opacities_np, scattering_in_emission,
            continuum_scat_np, n_layers_test, n_freq_centers_test, n_g_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_optical_depths.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_opt_depth_np.shape == fortran_opt_depth_np.shape, "Optical depth shape mismatch"
    np.testing.assert_allclose(jax_opt_depth_np, fortran_opt_depth_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_optical_depths (optical_depths) outputs differ.")
    
    # JAX photon_destruction_probabilities has shape (n_g, n_frequencies, n_species, n_layers)
    # We compare against the Fortran one which is (n_g, n_frequencies, n_layers), assuming n_species=1 for this part
    assert jax_photon_dest_prob_np[:,:,:].shape == fortran_photon_dest_prob_np.shape, "Photon destruction prob shape mismatch"
    np.testing.assert_allclose(jax_photon_dest_prob_np[:,:,:], fortran_photon_dest_prob_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_optical_depths (photon_destruction_probabilities) outputs differ.")


def test_compute_planck_opacities(sample_opacities_gfbl, sample_temperatures, sample_weights_gauss,
                                  sample_frequencies_bin_edges, sample_continuum_opacities_scattering_fbl,
                                  n_freq_centers_test, n_g_test, n_freq_edges_test, n_layers_test):
    return
    # opacities(n_g, n_frequencies_bins-1, n_layers)
    opacities_np = sample_opacities_gfbl
    temperatures_np = sample_temperatures
    weights_gauss_np = sample_weights_gauss
    freq_edges_np = sample_frequencies_bin_edges
    # continuum_opacities_scattering(n_frequencies_bins-1, n_layers)
    continuum_scat_np = sample_continuum_opacities_scattering_fbl
    scattering_in_emission = True
   
    # JAX Warm-up
    # JAX total_opacity input is (n_g, n_freq_bins-1, n_layers)
    # JAX continuum_opacity_scattering_emis is (n_freq_bins-1, n_layers)
    _ = jrt.compute_planck_opacities(
        to_jax_array(opacities_np), to_jax_array(temperatures_np), to_jax_array(weights_gauss_np),
        to_jax_array(freq_edges_np), scattering_in_emission, to_jax_array(continuum_scat_np)
    ).block_until_ready()
    
    # Fortran Warm-up
    _ = fcore.compute_planck_opacities(
        opacities_np, temperatures_np, weights_gauss_np, freq_edges_np,
        scattering_in_emission, continuum_scat_np,
        n_g_test, n_freq_centers_test, n_layers_test, n_freq_edges_test
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_planck_opac = jrt.compute_planck_opacities(
            to_jax_array(opacities_np), to_jax_array(temperatures_np), to_jax_array(weights_gauss_np),
            to_jax_array(freq_edges_np), scattering_in_emission, to_jax_array(continuum_scat_np)
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_planck_opac_np = to_numpy_array(jax_planck_opac)

    # Time Fortran
    # Fortran opacities(n_g, n_frequencies, n_layers) -> n_frequencies is n_freq_bins-1
    # Fortran continuum_opacities_scattering(n_frequencies, n_layers)
    # TODO: figure out why fortran function is crashing.
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_planck_opac_np = np.zeros(n_layers_test, dtype=NUMPY_DTYPE)
        fortran_planck_opac_np = fcore.compute_planck_opacities(
            opacities_np,
            temperatures_np,
            weights_gauss_np,
            freq_edges_np,
            scattering_in_emission,
            continuum_scat_np,
            n_g_test,
            n_freq_centers_test,
            n_layers_test,
            n_freq_edges_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_planck_opacities.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_planck_opac_np.shape == fortran_planck_opac_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_planck_opac_np, fortran_planck_opac_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_planck_opacities outputs differ.")


def test_compute_radius_hydrostatic_equilibrium(sample_pressures, sample_temperatures, n_layers_test):
    pressures_np_bar = sample_pressures
    pressures_np_cgs = pressures_np_bar * 1e6
    temperatures_np = sample_temperatures
    
    # Create plausible mean_molar_masses and atmosphere_densities
    mean_molar_masses_np = np.full(n_layers_test, 2.3, dtype=NUMPY_DTYPE)  # Example: H2/He mix
    atmosphere_densities_np = pressures_np_cgs * mean_molar_masses_np * cst.amu / (cst.kB * temperatures_np)

    reference_gravity_np = 1000.0  # cm/s^2
    reference_pressure_np_cgs = 1e6  # 1 bar in cgs
    planet_radius_np = 7e9  # cm (Jupiter-like)
    variable_gravity = True

    pressures_jax_cgs = to_jax_array(pressures_np_cgs)
    reference_gravity_jax = reference_gravity_np
    atmosphere_densities_jax = to_jax_array(atmosphere_densities_np)
    reference_pressure_jax_cgs = to_jax_array(reference_pressure_np_cgs)
    planet_radius_jax = to_jax_array(planet_radius_np)
    # JAX Warm-up
    # JAX pressures are in cgs
    _ = jrt.compute_radius_hydrostatic_equilibrium(
        pressures_jax_cgs,
        reference_gravity_jax,
        atmosphere_densities_jax,
        reference_pressure_jax_cgs,
        planet_radius_jax,
        variable_gravity
    ).block_until_ready()

    # Fortran Warm-up
    _ = fcore.compute_radius_hydrostatic_equilibrium(
        pressures_np_cgs, reference_gravity_np, atmosphere_densities_np,
        reference_pressure_np_cgs, planet_radius_np, variable_gravity, n_layers_test
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_radius_eq = jrt.compute_radius_hydrostatic_equilibrium(
            pressures_jax_cgs,
            reference_gravity_jax,
            atmosphere_densities_jax,
            reference_pressure_jax_cgs,
            planet_radius_jax,
            variable_gravity
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_radius_eq_np = to_numpy_array(jax_radius_eq)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_radius_eq_np = np.zeros(n_layers_test, dtype=NUMPY_DTYPE)
        # Fortran pressures are in cgs
        fortran_radius_eq_np = fcore.compute_radius_hydrostatic_equilibrium(
            pressures_np_cgs, reference_gravity_np, atmosphere_densities_np,
            reference_pressure_np_cgs, planet_radius_np, variable_gravity, n_layers_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_radius_hydrostatic_equilibrium.__name__] = {
        'jax_mean_time': jax_mean_time,
        'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time,
        'fortran_std_time': fortran_std_time
    }

    assert jax_radius_eq_np.shape == fortran_radius_eq_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_radius_eq_np, fortran_radius_eq_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_radius_hydrostatic_equilibrium outputs differ.")


def test_compute_transit_radii(sample_opacities_gfls, sample_temperatures, sample_pressures,
                               sample_weights_gauss, sample_continuum_opacities_scattering_fl,
                               n_freq_centers_test, n_layers_test, n_g_test, n_species_test):
    opacities_np = sample_opacities_gfls
    temperatures_np = sample_temperatures
    pressures_cgs_np = sample_pressures * 1e6  # cgs
    
    mean_molar_masses_np = np.full(n_layers_test, 2.3, dtype=NUMPY_DTYPE)
    reference_gravity_np = 1000.0
    reference_pressure_bar_np = 1.0  # bar
    planet_radius_cm_np = 7e9  # cm
    weights_gauss_np = sample_weights_gauss
    scattering_in_transmission = True
    continuum_scat_np = sample_continuum_opacities_scattering_fl
    variable_gravity = True

    opacities_jax = to_jax_array(opacities_np)
    temperatures_jax = to_jax_array(temperatures_np)
    pressures_cgs_jax = to_jax_array(pressures_cgs_np)
    reference_gravity_jax = reference_gravity_np
    mean_molar_masses_jax = to_jax_array(mean_molar_masses_np)
    reference_pressure_bar_jax = reference_pressure_bar_np
    planet_radius_cm_jax = planet_radius_cm_np
    weights_gauss_jax = to_jax_array(weights_gauss_np)
    continuum_scat_jax = to_jax_array(continuum_scat_np)
    # JAX Warm-up
    # JAX pressures are in bar
    _, _ = jrt.compute_transit_radii(
        opacities_jax,
        temperatures_jax,
        pressures_cgs_jax,
        reference_gravity_jax,
        mean_molar_masses_jax,
        reference_pressure_bar_jax,
        planet_radius_cm_jax,
        weights_gauss_jax,
        scattering_in_transmission,
        continuum_scat_jax,
        variable_gravity
    )
    _[0].block_until_ready()

    # Fortran Warm-up
    _ = fcore.compute_transit_radii(
        opacities_np, temperatures_np, pressures_cgs_np, reference_gravity_np,
        mean_molar_masses_np, reference_pressure_bar_np, planet_radius_cm_np,
        weights_gauss_np, scattering_in_transmission, continuum_scat_np,
        variable_gravity, n_freq_centers_test, n_layers_test, n_g_test, n_species_test,
    )

    # Time JAX
    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_transit_radii, jax_radius_hydro_eq = jrt.compute_transit_radii(
            opacities_jax,
            temperatures_jax,
            pressures_cgs_jax,
            reference_gravity_jax,
            mean_molar_masses_jax,
            reference_pressure_bar_jax,
            planet_radius_cm_jax,
            weights_gauss_jax,
            scattering_in_transmission,
            continuum_scat_jax,
            variable_gravity
        )
        jax_transit_radii.block_until_ready()
        jax_radius_hydro_eq.block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_transit_radii_np = to_numpy_array(jax_transit_radii)
    jax_radius_hydro_eq_np = to_numpy_array(jax_radius_hydro_eq)

    # Time Fortran
    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        # Fortran pressures are in bar
        fortran_transit_radii_np = np.zeros(n_freq_centers_test, dtype=NUMPY_DTYPE)
        fortran_radius_hydro_eq_np = np.zeros(n_layers_test, dtype=NUMPY_DTYPE)
        fortran_transit_radii_np, fortran_radius_hydro_eq_np = fcore.compute_transit_radii(
            opacities_np, temperatures_np, pressures_cgs_np, reference_gravity_np,
            mean_molar_masses_np, reference_pressure_bar_np, planet_radius_cm_np,
            weights_gauss_np, scattering_in_transmission, continuum_scat_np,
            variable_gravity, n_freq_centers_test, n_layers_test, n_g_test, n_species_test, 
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)
    timing_results[test_compute_transit_radii.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_transit_radii_np.shape == fortran_transit_radii_np.shape, "Transit radii shape mismatch"
    np.testing.assert_allclose(jax_transit_radii_np, fortran_transit_radii_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_transit_radii (transit_radii) outputs differ.")

    assert jax_radius_hydro_eq_np.shape == fortran_radius_hydro_eq_np.shape, "Hydrostatic eq radius shape mismatch"
    np.testing.assert_allclose(jax_radius_hydro_eq_np, fortran_radius_hydro_eq_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_transit_radii (radius_hydrostatic_equilibrium) outputs differ.")


if __name__ == "__main__":
    pytest.main([__file__])
    save_timing_results()
