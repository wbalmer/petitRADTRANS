"""
See Greco & Brandt 2016, Nasedkin et al 2023 and Yoav et al 2025 for more information on
covariance matrices and fitting.

Yoav et al. 2025: 10.3847/1538-4357/adef04
"""

import jax.numpy as jnp
from jax.typing import ArrayLike


_GLOBAL_KERNEL_ALIASES = {
    "se": "squared_exponential",
    "squared_exponential": "squared_exponential",
    "squared-exponential": "squared_exponential",
    "rbf": "squared_exponential",
    "gaussian": "squared_exponential",
    "matern32": "matern32",
    "matern_32": "matern32",
    "matern-32": "matern32",
    "matern_3_2": "matern32",
    "matern-3-2": "matern32",
    "matern3/2": "matern32",
}

_LOCAL_KERNEL_ALIASES = {
    "se": "squared_exponential",
    "squared_exponential": "squared_exponential",
    "squared-exponential": "squared_exponential",
    "gaussian": "squared_exponential",
}

_MODE_ALIASES = {
    "none": "none",
    "traditional": "none",
    "diagonal": "none",
    "local": "local",
    "local_only": "local",
    "local-only": "local",
    "global": "global",
    "global_only": "global",
    "global-only": "global",
    "global_local": "global_local",
    "global+local": "global_local",
    "global_plus_local": "global_local",
    "combined": "global_local",
}


def normalize_covariance_mode(mode: str | None) -> str:
    if mode is None:
        return "none"

    normalized = _MODE_ALIASES.get(str(mode).strip().lower())
    if normalized is None:
        raise ValueError(
            "covariance_mode must be one of 'none'|'local'|'global'|'global_local'"
        )

    return normalized


def normalize_global_kernel_name(kernel_name: str | None) -> str:
    if kernel_name is None:
        return "squared_exponential"

    normalized = _GLOBAL_KERNEL_ALIASES.get(str(kernel_name).strip().lower())
    if normalized is None:
        raise ValueError(
            "global_covariance_kernel must be one of 'squared_exponential'|'matern32'"
        )

    return normalized


def normalize_local_kernel_name(kernel_name: str | None) -> str:
    if kernel_name is None:
        return "squared_exponential"

    normalized = _LOCAL_KERNEL_ALIASES.get(str(kernel_name).strip().lower())
    if normalized is None:
        raise ValueError(
            "local_covariance_kernel must be 'squared_exponential' for now"
        )

    return normalized


def _as_wavelength_vector(wavelengths: ArrayLike) -> jnp.ndarray:
    wavelength_vector = jnp.asarray(wavelengths)
    if wavelength_vector.ndim != 1:
        raise ValueError("Covariance kernels currently require a 1D wavelength grid")
    return wavelength_vector


def _pairwise_separations(wavelengths: ArrayLike) -> jnp.ndarray:
    wavelength_vector = _as_wavelength_vector(wavelengths)
    return wavelength_vector[:, jnp.newaxis] - wavelength_vector[jnp.newaxis, :]


def squared_exponential_kernel(
    wavelengths: ArrayLike,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
) -> jnp.ndarray:
    delta = _pairwise_separations(wavelengths)
    safe_length_scale = jnp.maximum(jnp.asarray(length_scale), 1e-12)
    safe_amplitude = jnp.maximum(jnp.asarray(amplitude), 0.0)
    exponent = -0.5 * (delta / safe_length_scale) ** 2
    return safe_amplitude**2 * jnp.exp(exponent)


def matern32_kernel(
    wavelengths: ArrayLike,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
) -> jnp.ndarray:
    distance = jnp.abs(_pairwise_separations(wavelengths))
    safe_length_scale = jnp.maximum(jnp.asarray(length_scale), 1e-12)
    safe_amplitude = jnp.maximum(jnp.asarray(amplitude), 0.0)
    scaled_distance = jnp.sqrt(3.0) * distance / safe_length_scale
    return safe_amplitude**2 * (1.0 + scaled_distance) * jnp.exp(-scaled_distance)


def localized_squared_exponential_kernel(
    wavelengths: ArrayLike,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    center: ArrayLike,
) -> jnp.ndarray:
    wavelength_vector = _as_wavelength_vector(wavelengths)
    safe_length_scale = jnp.maximum(jnp.asarray(length_scale), 1e-12)
    safe_amplitude = jnp.maximum(jnp.asarray(amplitude), 0.0)
    window = jnp.exp(-0.5 * ((wavelength_vector - center) / safe_length_scale) ** 2)
    return safe_amplitude**2 * jnp.outer(window, window)


def build_global_covariance_kernel(
    wavelengths: ArrayLike,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    kernel_name: str = "squared_exponential",
) -> jnp.ndarray:
    normalized_kernel = normalize_global_kernel_name(kernel_name)

    if normalized_kernel == "squared_exponential":
        return squared_exponential_kernel(
            wavelengths, amplitude=amplitude, length_scale=length_scale
        )

    if normalized_kernel == "matern32":
        return matern32_kernel(
            wavelengths, amplitude=amplitude, length_scale=length_scale
        )

    raise ValueError(f"Unsupported global covariance kernel '{kernel_name}'")


def build_local_covariance_kernel(
    wavelengths: ArrayLike,
    amplitude: ArrayLike,
    length_scale: ArrayLike,
    center: ArrayLike,
    kernel_name: str = "squared_exponential",
) -> jnp.ndarray:
    normalized_kernel = normalize_local_kernel_name(kernel_name)

    if normalized_kernel == "squared_exponential":
        return localized_squared_exponential_kernel(
            wavelengths,
            amplitude=amplitude,
            length_scale=length_scale,
            center=center,
        )

    raise ValueError(f"Unsupported local covariance kernel '{kernel_name}'")
