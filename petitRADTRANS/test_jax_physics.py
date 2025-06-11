# test_jphys.py

import pytest
import jax
from jax import config
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# --- JAX Device Configuration ---
# User can set this to "cpu", "gpu", or "tpu" (if available and JAX is configured for it).
# Set to None or empty string to use JAX's default device.
_JAX_DEVICE_NAME = "cpu"  # Or "gpu", "tpu", None

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

# Enable float64 for JAX to match Fortran's double precision

# --- Import JAX functions to be tested ---
# This assumes jphys.py is in a location where it can be imported.
# For example, in the same directory or in PYTHONPATH.
FORTRAN_AVAILABLE = True

from petitRADTRANS.fortran_radtrans_core import physics as fphys
import petitRADTRANS.radtrans_core.physics_core as jphys

# --- Import Fortran wrappers ---
# IMPORTANT: Replace this with the actual import mechanism for your compiled Fortran code.
# This is a placeholder. The structure depends on how petitRADTRANS compiles and exposes its Fortran code.
# It might be part of the petitRADTRANS package, e.g., petitRADTRANS.lib.fortran_module.physics

# --- Test Configuration ---
# --- Test Configuration ---
RTOL = 1e-4  # Increased relative tolerance for complex functions
ATOL = 1e-6  # Increased absolute tolerance
N_TIMING_RUNS = 100 # Number of times to run each function for timing
if config.jax_enable_x64:
    RTOL = 1e-8  # Increased relative tolerance for complex functions
    ATOL = 1e-10  # Increased absolute tolerance

# --- Helper Functions ---
def to_jax_array(arr, dtype=jnp.float64):
    """Converts a NumPy array or scalar to a JAX array on the configured device."""
    return jax.device_put(jnp.array(arr, dtype=dtype), device=JAX_DEVICE)


def to_numpy_array(arr):
    """Converts a JAX array to a NumPy array."""
    return np.array(arr)


# --- Test Fixtures (for generating common test data) ---
@pytest.fixture
def sample_temperatures_layers() -> np.ndarray:
    return np.array([1000.0, 1500.0, 2000.0, 2500.0], dtype=np.double)

@pytest.fixture
def sample_scalar_temperature() -> np.double:
    """Provides a single scalar temperature value."""
    return np.double(1750.0)

@pytest.fixture
def sample_frequencies_array() -> np.ndarray:
    # Frequencies should be in decreasing order for some Fortran routines
    return np.array([3.0e14, 2.5e14, 2.0e14, 1.5e14, 1.0e14], dtype=np.double)


@pytest.fixture
def sample_weights_gauss() -> np.ndarray:
    # Example for n_g = 4, ensure sum to 1 if they are weights
    return np.array([0.15, 0.35, 0.35, 0.15], dtype=np.double)


@pytest.fixture
def sample_clouds_opacities(sample_weights_gauss, sample_frequencies_array, sample_temperatures_layers) -> np.ndarray:
    n_g = len(sample_weights_gauss)
    # Assuming frequencies are bin edges for cloud opacities context
    n_freq_bins_minus_1 = len(sample_frequencies_array) - 1
    n_temperature_layers = len(sample_temperatures_layers)
    if n_freq_bins_minus_1 <= 0: # if freqs are bin centers (less likely for this context)
        n_freq_bins_minus_1 = len(sample_frequencies_array)
    return np.abs(np.random.rand(n_g, n_freq_bins_minus_1,n_temperature_layers) * 1e-2).astype(np.double)


# --- Test Cases for functions with Fortran counterparts ---

@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrappers not available for comparison.")
def test_compute_planck_function(sample_temperatures_layers, sample_frequencies_array):
    temperatures_np = sample_temperatures_layers
    frequencies_np = sample_frequencies_array
    n_layers = len(temperatures_np)
    n_frequencies = len(frequencies_np)

    # JAX execution
    temperatures_jax = to_jax_array(temperatures_np)
    frequencies_jax = to_jax_array(frequencies_np)

    # JAX function jphys.compute_planck_function(T_array, nu_array)
    # returns (len(nu), len(T))
    jax_result = jphys.compute_planck_function(temperatures_jax, frequencies_jax)
    jax_result_np = to_numpy_array(jax_result)

    # Fortran execution
    # Fortran: subroutine compute_planck_function(temperatures, frequencies, n_layers, planck_flux)
    # intent(in) :: temperatures(n_layers), frequencies (scalar), n_layers
    # intent(out) :: planck_flux(n_layers)
    # We need to call the Fortran function for each frequency and stack the results.
    fortran_results_list = []
    for freq_val in frequencies_np:
        # Fortran function takes a scalar frequency
        fortran_layer_fluxes = fphys.compute_planck_function(temperatures_np, freq_val, n_layers)
        fortran_results_list.append(fortran_layer_fluxes)
    # Stack along axis 0 to get (n_frequencies, n_layers) to match JAX output
    fortran_result_stacked = np.stack(fortran_results_list, axis=0)

    assert jax_result_np.shape == (n_frequencies, n_layers), "JAX output shape mismatch"
    np.testing.assert_allclose(jax_result_np, fortran_result_stacked, rtol=RTOL, atol=ATOL,
                               err_msg="compute_planck_function outputs differ.")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrappers not available for comparison.")
def test_compute_planck_function_temperature_derivative(sample_temperatures_layers, sample_frequencies_array):
    temperatures_np = sample_temperatures_layers
    frequencies_np = sample_frequencies_array
    n_layers = len(temperatures_np)
    n_frequencies = len(frequencies_np)
    n_nu_bins = n_frequencies -1


    # JAX execution
    # JAX signature: func(T_array_1D, frequencies_array_1D) -> (n_nu_bins, n_T)
    temperatures_jax = to_jax_array(temperatures_np)
    frequencies_jax = to_jax_array(frequencies_np)
    jax_result = jphys.compute_planck_function_temperature_derivative(temperatures_jax, frequencies_jax)
    jax_result_np = to_numpy_array(jax_result)

    # Fortran execution
    # Fortran: subroutine compute_planck_function_temperature_derivative(temperature_scalar, frequencies_array, n_frequencies, B_nu_dT_array_out)
    # intent(out) :: B_nu_dT(n_frequencies-1)
    # We loop Fortran calls for each temperature and stack results.
    fortran_results_list = []
    for temp_val in temperatures_np:
        fortran_b_nu_dt = fphys.compute_planck_function_temperature_derivative(
            temp_val, frequencies_np, n_frequencies
        )
        fortran_results_list.append(fortran_b_nu_dt)
    # Stack along axis 1 to get (n_nu_bins, n_layers) to match JAX output
    fortran_result_stacked = np.stack(fortran_results_list, axis=1)

    assert jax_result_np.shape == (n_nu_bins, n_layers), \
        f"JAX output shape mismatch. Expected ({n_nu_bins}, {n_layers}), Got {jax_result_np.shape}"
    np.testing.assert_allclose(jax_result_np, fortran_result_stacked, rtol=RTOL, atol=ATOL,
                               err_msg="compute_planck_function_temperature_derivative outputs differ.")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrappers not available for comparison.")
def test_compute_planck_function_integral(sample_temperatures_layers, sample_frequencies_array):
    temperatures_np = sample_temperatures_layers
    # Ensure frequencies are in decreasing order for Fortran routine
    frequencies_np = np.sort(sample_frequencies_array)[::-1]
    n_layers = len(temperatures_np)
    n_frequencies = len(frequencies_np)
    n_nu_bins = n_frequencies - 1

    # JAX execution
    # JAX signature: func(temperatures_array_1D, frequencies_array_1D) -> (n_layers, n_nu_bins)
    temperatures_jax = to_jax_array(temperatures_np)
    frequencies_jax = to_jax_array(frequencies_np)
    jax_result = jphys.compute_planck_function_integral(temperatures_jax, frequencies_jax)
    jax_result_np = to_numpy_array(jax_result)

    # Fortran execution
    # Fortran: subroutine compute_planck_function_integral(temperatures_array, frequencies_array, n_layers, n_frequencies, planck_flux_out)
    # intent(out) :: planck_flux(n_layers, n_frequencies - 1)
    fortran_result = fphys.compute_planck_function_integral(
        temperatures_np, frequencies_np, n_layers, n_frequencies
    )

    assert jax_result_np.shape == (n_layers, n_nu_bins), "JAX output shape mismatch"
    np.testing.assert_allclose(jax_result_np, fortran_result, rtol=RTOL, atol=ATOL,
                               err_msg="compute_planck_function_integral outputs differ.")


@pytest.mark.skipif(not FORTRAN_AVAILABLE, reason="Fortran wrappers not available for comparison.")
def test_compute_rosseland_opacities_core(sample_clouds_opacities, sample_frequencies_array, sample_temperatures_layers, sample_weights_gauss):
    temperatures_np = sample_temperatures_layers # Using array of temperatures
    n_layers = len(temperatures_np)

    # Frequencies are bin edges for this routine
    frequencies_bin_edges_np = sample_frequencies_array
    clouds_opacities_np = sample_clouds_opacities # Shape (n_g, n_frequencies_bins-1, N_temperature)
    weights_gauss_np = sample_weights_gauss

    n_g = clouds_opacities_np.shape[0]
    n_frequencies_bins = frequencies_bin_edges_np.shape[0]
    n_temperatures = sample_temperatures_layers.shape[0]
    assert clouds_opacities_np.shape == (n_g, n_frequencies_bins - 1,n_temperatures), "Input clouds_opacities_np shape mismatch"
    assert len(weights_gauss_np) == n_g, "Input weights_gauss_np shape mismatch"

    # JAX execution
    # JAX signature: func(opacities_2D_g_fn, freq_edges_1D_f, temp_array_1D_n, weights_1D_g) -> (n_temp,)
    clouds_opacities_jax = to_jax_array(clouds_opacities_np)
    frequencies_bin_edges_jax = to_jax_array(frequencies_bin_edges_np)
    temperatures_jax = to_jax_array(temperatures_np) # JAX array of temperatures
    weights_gauss_jax = to_jax_array(weights_gauss_np)

    jax_result = jphys.compute_rosseland_opacities_core(
        clouds_opacities_jax, frequencies_bin_edges_jax, temperatures_jax, weights_gauss_jax
    )
    jax_result_np = to_numpy_array(jax_result)

    # Fortran execution
    # Fortran: subroutine compute_rosseland_opacities_core(clouds_op, freq_edges, temp_scalar, weights, n_freq_bins, n_g, op_ross_scalar_out)
    # We loop Fortran calls for each temperature and stack results.
    fortran_results_list = []
    for i,temp_val in enumerate(temperatures_np):
        fortran_op_ross = fphys.compute_rosseland_opacities_core(
            clouds_opacities_np[:,:,i],
            frequencies_bin_edges_np,
            temp_val,
            weights_gauss_np,
            n_frequencies_bins,
            n_g
        )
        fortran_results_list.append(fortran_op_ross)
    fortran_result_stacked = np.array(fortran_results_list) # Shape (n_layers,)

    assert jax_result_np.shape == (n_layers,), \
        f"JAX output shape mismatch. Expected ({n_layers},), Got {jax_result_np.shape}"
    np.testing.assert_allclose(jax_result_np, fortran_result_stacked, rtol=RTOL, atol=ATOL,
                               err_msg="compute_rosseland_opacities_core outputs differ.")


# --- Test Cases for JAX functions without direct Fortran counterparts (Example) ---

def test_example_jax_only_function():
    """
    This is a placeholder for testing a JAX function that does not have
    a direct Fortran counterpart in the physics module.
    You would define expected inputs and outputs based on the function's specification.
    """
    pytest.skip("No JAX-only functions specified for testing in this example.")

if __name__ == "__main__":
    pytest.main([__file__])

