import pytest
import jax
from jax import config
# Enable float64 for JAX to match Fortran's double precision
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import time
import os

# --- JAX Device Configuration ---
_JAX_DEVICE_NAME = "cpu"  # Or "gpu", "tpu", None
JAX_DTYPE = jnp.float64
NUMPY_DTYPE = np.double

# --- Test Configuration ---
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance
N_TIMING_RUNS = 100 # Number of times to run each function for timing

if config.jax_enable_x64:
    RTOL = 1e-8
    ATOL = 1e-10

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
from petitRADTRANS.fortran_radtrans_core import math as fmath
from petitRADTRANS.fortran_radtrans_core import cloud_utils as fcloud_utils
import petitRADTRANS.radtrans_core.cloud_core as jcloud_core
import petitRADTRANS.physical_constants as cst

# --- Timing Results Storage ---
timing_results = {}

# --- Pytest Fixture to Save Timings ---
@pytest.fixture(scope="session", autouse=True)
def save_timing_results():
    """Saves the timing results to a file at the end of the test session."""
    yield
    output_file = "timing_results_cloud_core.txt"
    with open(output_file, "w") as f:
        f.write("Function Name, JAX Mean Time (s), JAX Std Time (s), Fortran Mean Time (s), Fortran Std Time (s), JAX/Fortran Ratio\n")
        for func_name, times in timing_results.items():
            ratio = times['jax_mean_time'] / times['fortran_mean_time'] if times['fortran_mean_time'] > 0 else float('inf')
            f.write(f"{func_name}, {times['jax_mean_time']:.6f}, {times['jax_std_time']:.6f}, "
                    f"{times['fortran_mean_time']:.6f}, {times['fortran_std_time']:.6f}, {ratio:.6f}\n")
    print(f"\nTiming results for cloud_core.py written to {output_file}")

# --- Helper Functions ---
def to_jax_array(arr, dtype=JAX_DTYPE):
    return jax.device_put(jnp.array(arr, dtype=dtype), device=JAX_DEVICE)

def to_numpy_array(arr):
    return np.array(arr, dtype=NUMPY_DTYPE)

# --- Test Fixtures ---
@pytest.fixture
def n_layers_test(): return 5

@pytest.fixture
def n_clouds_test(): return 2

@pytest.fixture
def n_cloud_wavelengths_test(): return 10

@pytest.fixture
def n_freq_edges_test(): return 11 # n_freq_bins = 10

@pytest.fixture
def sample_particle_radius_x():
    return np.double(1e-4)  # cm

@pytest.fixture
def sample_reference_gravity():
    return np.double(1000.0)  # cm/s^2

@pytest.fixture
def sample_rho_atmosphere(n_layers_test):
    return np.logspace(-10, -3, n_layers_test).astype(NUMPY_DTYPE) # g/cm^3

@pytest.fixture
def sample_clouds_particles_densities(n_clouds_test):
    return np.array([1.0, 2.0] * (n_clouds_test // 2) + [1.0] * (n_clouds_test % 2), dtype=NUMPY_DTYPE) # g/cm^3

@pytest.fixture
def sample_temperatures_layers(n_layers_test):
    return np.linspace(500.0, 1500.0, n_layers_test).astype(NUMPY_DTYPE) # K

@pytest.fixture
def sample_mean_molar_masses_layers(n_layers_test):
    return np.full(n_layers_test, 2.3, dtype=NUMPY_DTYPE) # amu

@pytest.fixture
def sample_w_star_target_settling_speed():
    return np.double(10.0) # cm/s

@pytest.fixture
def sample_f_seds(n_layers_test, n_clouds_test):
    return np.abs(np.random.rand(n_layers_test, n_clouds_test) * 2.0).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_cloud_particle_radius_distribution_std():
    return np.double(2.0)

@pytest.fixture
def sample_eddy_diffusion_coefficients(n_layers_test):
    return np.logspace(7, 9, n_layers_test).astype(NUMPY_DTYPE) # cm^2/s

@pytest.fixture
def sample_clouds_b_hansen(n_layers_test, n_clouds_test):
    return np.abs(np.random.rand(n_layers_test, n_clouds_test) * 0.5 + 0.1).astype(NUMPY_DTYPE) # typically 0.1 to 0.5

@pytest.fixture
def sample_cloud_abs_opacities(n_cloud_wavelengths_test, n_layers_test):
    return np.abs(np.random.rand(n_cloud_wavelengths_test, n_layers_test) * 1e-3).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_cloud_scat_opacities(n_cloud_wavelengths_test, n_layers_test):
    return np.abs(np.random.rand(n_cloud_wavelengths_test, n_layers_test) * 1e-2).astype(NUMPY_DTYPE)

@pytest.fixture
def sample_cloud_scat_reduction_factor(n_cloud_wavelengths_test, n_layers_test):
    return np.random.rand(n_cloud_wavelengths_test, n_layers_test).astype(NUMPY_DTYPE) # 0 to 1

@pytest.fixture
def sample_cloud_wavelengths(n_cloud_wavelengths_test):
    return np.sort(np.random.rand(n_cloud_wavelengths_test) * 10.0 + 0.5).astype(NUMPY_DTYPE) # microns, ensure sorted

@pytest.fixture
def sample_frequencies_bin_edges(n_freq_edges_test):
    # Ensure decreasing order for Fortran routines if necessary, though JAX versions might handle it
    return np.sort(np.linspace(3.1e14, 0.9e14, n_freq_edges_test)).astype(NUMPY_DTYPE)[::-1]  # Hz, decreasing


# --- Test Cases ---

def test_compute_turbulent_settling_speed(
    sample_particle_radius_x, sample_reference_gravity,
    sample_rho_atmosphere, sample_clouds_particles_densities,
    sample_temperatures_layers, sample_mean_molar_masses_layers):

    # Test for a single layer, single cloud species scenario for simplicity
    x_np = sample_particle_radius_x
    ref_g_np = sample_reference_gravity
    rho_np = sample_rho_atmosphere[0] # Single layer
    cloud_dens_np = sample_clouds_particles_densities[0] # Single cloud
    temp_np = sample_temperatures_layers[0] # Single layer
    mmm_np = sample_mean_molar_masses_layers[0] # Single layer

    # JAX
    x_jax = to_jax_array(x_np)
    ref_g_jax = to_jax_array(ref_g_np)
    rho_jax = to_jax_array(rho_np)
    cloud_dens_jax = to_jax_array(cloud_dens_np)
    temp_jax = to_jax_array(temp_np)
    mmm_jax = to_jax_array(mmm_np)

    _ = jcloud_core.compute_turbulent_settling_speed(
        x_jax, ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax
    ).block_until_ready()

    # Fortran
    _ = fcloud_utils.compute_turbulent_settling_speed(
        x_np, ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np
    )

    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_result = jcloud_core.compute_turbulent_settling_speed(
            x_jax, ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_result_np = to_numpy_array(jax_result)

    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_result_np = fcloud_utils.compute_turbulent_settling_speed(
            x_np, ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_compute_turbulent_settling_speed.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    np.testing.assert_allclose(jax_result_np, fortran_result_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_turbulent_settling_speed outputs differ.")


def test_particle_radius(
    sample_reference_gravity, sample_rho_atmosphere, sample_clouds_particles_densities,
    sample_temperatures_layers, sample_mean_molar_masses_layers, sample_w_star_target_settling_speed):

    # Test for a single layer, single cloud species scenario
    ref_g_np = sample_reference_gravity
    rho_np = sample_rho_atmosphere[0]
    cloud_dens_np = sample_clouds_particles_densities[0]
    temp_np = sample_temperatures_layers[0]
    mmm_np = sample_mean_molar_masses_layers[0]
    w_star_np = sample_w_star_target_settling_speed
    x1_np, x2_np = np.double(1e-7), np.double(1e-1) # Search bounds

    # JAX
    ref_g_jax = to_jax_array(ref_g_np)
    rho_jax = to_jax_array(rho_np)
    cloud_dens_jax = to_jax_array(cloud_dens_np)
    temp_jax = to_jax_array(temp_np)
    mmm_jax = to_jax_array(mmm_np)
    w_star_jax = to_jax_array(w_star_np)
    x1_jax, x2_jax = to_jax_array(x1_np), to_jax_array(x2_np)

    _ = jcloud_core.particle_radius(
        x1_jax, x2_jax, ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, w_star_jax
    ).block_until_ready()

    # Fortran
    _ = fcloud_utils.particle_radius(
        x1_np, x2_np, ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, w_star_np
    )

    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_result = jcloud_core.particle_radius(
            x1_jax, x2_jax, ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, w_star_jax
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_result_np = to_numpy_array(jax_result)

    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_result_np = fcloud_utils.particle_radius(
            x1_np, x2_np, ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, w_star_np
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_particle_radius.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }
    np.testing.assert_allclose(jax_result_np, fortran_result_np, rtol=RTOL, atol=ATOL,
                               err_msg="particle_radius outputs differ.")


def test_particle_radius_reverse_mode_differentiable(
    sample_reference_gravity, sample_rho_atmosphere, sample_clouds_particles_densities,
    sample_temperatures_layers, sample_mean_molar_masses_layers, sample_w_star_target_settling_speed):

    ref_g_jax = to_jax_array(sample_reference_gravity)
    rho_jax = to_jax_array(sample_rho_atmosphere[0])
    cloud_dens_jax = to_jax_array(sample_clouds_particles_densities[0])
    temp_jax = to_jax_array(sample_temperatures_layers[0])
    mmm_jax = to_jax_array(sample_mean_molar_masses_layers[0])
    w_star_jax = to_jax_array(sample_w_star_target_settling_speed)
    x1_jax = to_jax_array(np.double(1e-7))
    x2_jax = to_jax_array(np.double(1e-1))

    def loss_fn(w_star):
        radius = jcloud_core.particle_radius(
            x1_jax, x2_jax, ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, w_star
        )
        return radius ** 2

    gradient = jax.grad(loss_fn)(w_star_jax)

    assert jnp.isfinite(gradient)


def test_compute_cloud_particles_mean_radius(
    sample_reference_gravity, sample_rho_atmosphere, sample_clouds_particles_densities,
    sample_temperatures_layers, sample_mean_molar_masses_layers, sample_f_seds,
    sample_cloud_particle_radius_distribution_std, sample_eddy_diffusion_coefficients,
    n_layers_test, n_clouds_test):

    ref_g_np = sample_reference_gravity
    rho_np = sample_rho_atmosphere
    cloud_dens_np = sample_clouds_particles_densities
    temp_np = sample_temperatures_layers
    mmm_np = sample_mean_molar_masses_layers
    f_seds_np = sample_f_seds
    std_dev_np = sample_cloud_particle_radius_distribution_std
    eddy_diff_np = sample_eddy_diffusion_coefficients

    # JAX
    ref_g_jax = to_jax_array(ref_g_np)
    rho_jax = to_jax_array(rho_np)
    cloud_dens_jax = to_jax_array(cloud_dens_np)
    temp_jax = to_jax_array(temp_np)
    mmm_jax = to_jax_array(mmm_np)
    f_seds_jax = to_jax_array(f_seds_np)
    std_dev_jax = to_jax_array(std_dev_np)
    eddy_diff_jax = to_jax_array(eddy_diff_np)

    _ = jcloud_core.compute_cloud_particles_mean_radius(
        ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, f_seds_jax, std_dev_jax, eddy_diff_jax
    ).block_until_ready()

    # Fortran
    _ = fcore.compute_cloud_particles_mean_radius(
        ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, f_seds_np, std_dev_np, eddy_diff_np,
        n_layers_test, n_clouds_test
    )

    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_result = jcloud_core.compute_cloud_particles_mean_radius(
            ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, f_seds_jax, std_dev_jax, eddy_diff_jax
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_result_np = to_numpy_array(jax_result)

    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_result_np = fcore.compute_cloud_particles_mean_radius(
            ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, f_seds_np, std_dev_np, eddy_diff_np,
            n_layers_test, n_clouds_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_compute_cloud_particles_mean_radius.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_result_np.shape == fortran_result_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_result_np, fortran_result_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_cloud_particles_mean_radius outputs differ.")


def test_compute_cloud_particles_mean_radius_hansen(
    sample_reference_gravity, sample_rho_atmosphere, sample_clouds_particles_densities,
    sample_temperatures_layers, sample_mean_molar_masses_layers, sample_f_seds,
    sample_clouds_b_hansen, sample_eddy_diffusion_coefficients,
    n_layers_test, n_clouds_test):

    ref_g_np = sample_reference_gravity
    rho_np = sample_rho_atmosphere
    cloud_dens_np = sample_clouds_particles_densities
    temp_np = sample_temperatures_layers
    mmm_np = sample_mean_molar_masses_layers
    f_seds_np = sample_f_seds
    b_hansen_np = sample_clouds_b_hansen
    eddy_diff_np = sample_eddy_diffusion_coefficients

    # JAX
    ref_g_jax = to_jax_array(ref_g_np)
    rho_jax = to_jax_array(rho_np)
    cloud_dens_jax = to_jax_array(cloud_dens_np)
    temp_jax = to_jax_array(temp_np)
    mmm_jax = to_jax_array(mmm_np)
    f_seds_jax = to_jax_array(f_seds_np)
    b_hansen_jax = to_jax_array(b_hansen_np)
    eddy_diff_jax = to_jax_array(eddy_diff_np)

    _ = jcloud_core.compute_cloud_particles_mean_radius_hansen_jax(
        ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, f_seds_jax, b_hansen_jax, eddy_diff_jax
    ).block_until_ready()

    # Fortran
    _ = fcore.compute_cloud_particles_mean_radius_hansen(
        ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, f_seds_np, b_hansen_np, eddy_diff_np,
        n_layers_test, n_clouds_test
    )

    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_result = jcloud_core.compute_cloud_particles_mean_radius_hansen_jax(
            ref_g_jax, rho_jax, cloud_dens_jax, temp_jax, mmm_jax, f_seds_jax, b_hansen_jax, eddy_diff_jax
        ).block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_result_np = to_numpy_array(jax_result)

    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        fortran_result_np = fcore.compute_cloud_particles_mean_radius_hansen(
            ref_g_np, rho_np, cloud_dens_np, temp_np, mmm_np, f_seds_np, b_hansen_np, eddy_diff_np,
            n_layers_test, n_clouds_test
        )
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_compute_cloud_particles_mean_radius_hansen.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    assert jax_result_np.shape == fortran_result_np.shape, "Shape mismatch"
    np.testing.assert_allclose(jax_result_np, fortran_result_np, rtol=RTOL, atol=ATOL,
                               err_msg="compute_cloud_particles_mean_radius_hansen outputs differ.")


def test_interpolate_cloud_opacities(
    sample_cloud_abs_opacities, sample_cloud_scat_opacities, sample_cloud_scat_reduction_factor,
    sample_cloud_wavelengths, sample_frequencies_bin_edges,
    n_cloud_wavelengths_test, n_layers_test, n_freq_edges_test):

    abs_op_np = sample_cloud_abs_opacities
    scat_op_np = sample_cloud_scat_opacities
    scat_reduc_np = sample_cloud_scat_reduction_factor
    cloud_wl_np = sample_cloud_wavelengths
    freq_edges_np = sample_frequencies_bin_edges
    n_freq_bins = n_freq_edges_test # Fortran uses n_frequencies_bins for number of edges

    # JAX
    abs_op_jax = to_jax_array(abs_op_np)
    scat_op_jax = to_jax_array(scat_op_np)
    scat_reduc_jax = to_jax_array(scat_reduc_np)
    cloud_wl_jax = to_jax_array(cloud_wl_np)
    freq_edges_jax = to_jax_array(freq_edges_np)

    _ = jcloud_core.interpolate_cloud_opacities(
        abs_op_jax, scat_op_jax, scat_reduc_jax, cloud_wl_jax, freq_edges_jax,
        n_cloud_wavelengths_test, n_layers_test, n_freq_bins
    )[0].block_until_ready() # Check one output

    # Fortran
    _ = fcore.interpolate_cloud_opacities(
        abs_op_np, scat_op_np, scat_reduc_np, cloud_wl_np, freq_edges_np,
        n_cloud_wavelengths_test, n_layers_test, n_freq_bins
    )

    jax_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        jax_results = jcloud_core.interpolate_cloud_opacities(
            abs_op_jax, scat_op_jax, scat_reduc_jax, cloud_wl_jax, freq_edges_jax,
            n_cloud_wavelengths_test, n_layers_test, n_freq_bins
        )
        for res in jax_results: res.block_until_ready()
        jax_times.append(time.perf_counter() - start_time)
    jax_mean_time = np.mean(jax_times)
    jax_std_time = np.std(jax_times)
    jax_results_np = [to_numpy_array(res) for res in jax_results]

    fortran_times = []
    for _ in range(N_TIMING_RUNS):
        start_time = time.perf_counter()
        # Fortran returns: clouds_final_absorption_opacities, cloud_anisotropic_extinctions,
        #                  cloud_scattering_reduction_factor_out, cloud_isotropic_extinctions
        # Shapes: (n_freq_bins-1, n_layers) for all
        fortran_results_np = fcore.interpolate_cloud_opacities(
            abs_op_np, scat_op_np, scat_reduc_np, cloud_wl_np, freq_edges_np,
            n_cloud_wavelengths_test, n_layers_test, n_freq_bins
        ) # This returns a tuple of 4 arrays
        fortran_times.append(time.perf_counter() - start_time)
    fortran_mean_time = np.mean(fortran_times)
    fortran_std_time = np.std(fortran_times)

    timing_results[test_interpolate_cloud_opacities.__name__] = {
        'jax_mean_time': jax_mean_time, 'jax_std_time': jax_std_time,
        'fortran_mean_time': fortran_mean_time, 'fortran_std_time': fortran_std_time
    }

    output_names = ["clouds_final_absorption_opacities", "cloud_anisotropic_extinctions",
                    "cloud_scattering_reduction_factor_out", "cloud_isotropic_extinctions"]

    for i, name in enumerate(output_names):
        assert jax_results_np[i].shape == fortran_results_np[i].shape, f"Shape mismatch for {name}"
        np.testing.assert_allclose(jax_results_np[i], fortran_results_np[i], rtol=RTOL, atol=ATOL,
                                   err_msg=f"interpolate_cloud_opacities ({name}) outputs differ.")

if __name__ == "__main__":
    pytest.main([__file__])

