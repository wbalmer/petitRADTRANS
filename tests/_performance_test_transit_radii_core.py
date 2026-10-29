"""Benchmark the JAX transit-radii core against the previous dense reference path.

This is a manual performance test. It compares the current
``compute_transit_radii`` implementation against a frozen reference variant
that matches the previous dense path-length construction so performance and
gradient behavior can be re-evaluated after local changes.
"""

import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, lax

from .context import petitRADTRANS
from petitRADTRANS.radtrans_core import radiative_transfer_core as rtc

DEFAULT_CASES = (
    (8, 1024, 6, 128),
    (4, 1024, 6, 256),
    (8, 2048, 4, 128),
    (4, 2048, 4, 192),
)


def _safe_sqrt_positive(values):
    positive_values = values > 0.0
    safe_values = jnp.where(
        positive_values,
        values,
        lax.stop_gradient(jnp.ones_like(values)),
    )
    return jnp.where(positive_values, jnp.sqrt(safe_values), 0.0)


@jit(static_argnames=("variable_gravity",))
def _reference_compute_transit_radii(
    opacities,
    continuum_opacities_scattering,
    pressures,
    temperatures,
    weights_gauss,
    mean_molar_masses,
    reference_gravity,
    reference_pressure,
    planet_radius,
    variable_gravity=True,
):
    pressures_cgs = pressures * 1e6
    reference_pressure_cgs = reference_pressure * 1e6

    atmospheric_densities = (
        pressures_cgs * mean_molar_masses * rtc.cst.amu / rtc.cst.kB / temperatures
    )
    radius_hydrostatic_equilibrium = rtc.compute_radius_hydrostatic_equilibrium(
        pressures=pressures_cgs,
        reference_gravity=reference_gravity,
        atmosphere_densities=atmospheric_densities,
        reference_pressure=reference_pressure_cgs,
        planet_radius=planet_radius,
        variable_gravity=variable_gravity,
    )

    first_non_negative_idx = jnp.argmax(radius_hydrostatic_equilibrium >= 0.0)
    replacement_value = radius_hydrostatic_equilibrium[first_non_negative_idx]
    radius_hydrostatic_equilibrium = jnp.where(
        radius_hydrostatic_equilibrium < 0.0,
        replacement_value,
        radius_hydrostatic_equilibrium,
    )

    rho = pressures_cgs * mean_molar_masses * rtc.cst.amu / rtc.cst.kB / temperatures
    continuum_opa_scat_reshaped = continuum_opacities_scattering.reshape(
        (1, continuum_opacities_scattering.shape[0], 1, pressures.size)
    )

    transparencies = opacities * rho
    transparencies = transparencies.at[:, :, :1, :].add(
        continuum_opa_scat_reshaped * rho
    )
    transparencies = transparencies.at[:, :, :, 1:].add(transparencies[:, :, :, :-1])

    diff_s = jnp.zeros((pressures.size, pressures.size))
    r_ik = (
        radius_hydrostatic_equilibrium.reshape(1, pressures.size) ** 2.0
        - radius_hydrostatic_equilibrium.reshape(pressures.size, 1) ** 2.0
    )
    r_ik = _safe_sqrt_positive(r_ik)
    diff_s = diff_s.at[1:, 1:].set(-r_ik[1:, 1:] + r_ik[1:, :-1])

    transparencies = jnp.einsum("ijkl,ml", transparencies, diff_s)
    transparencies = jnp.exp(-1.0 * transparencies)

    if weights_gauss.size == 1:
        transparencies = transparencies[0]
    else:
        transparencies = jnp.einsum("ijkl,i", transparencies, weights_gauss)

    transparencies = jnp.swapaxes(transparencies, 1, 0)

    if transparencies.shape[0] == 1:
        transparencies = transparencies[0]
    else:
        transparencies = jnp.prod(transparencies, axis=0)

    transparencies = 1.0 - transparencies
    annulus_radius_increments = -jnp.diff(radius_hydrostatic_equilibrium)
    transparencies = transparencies * radius_hydrostatic_equilibrium
    transparencies = transparencies.at[:, 1:].add(transparencies[:, :-1])
    transparencies = transparencies[:, 1:]
    transparencies = transparencies * annulus_radius_increments
    transit_radii = jnp.sum(transparencies, axis=1)
    transit_radii = jnp.sqrt(transit_radii + radius_hydrostatic_equilibrium[-1] ** 2.0)

    return transit_radii, radius_hydrostatic_equilibrium


def _build_inputs(seed, n_g, n_frequencies, n_species, n_layers):
    key = jax.random.key(seed)

    def split():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    pressures = jnp.geomspace(1e-6, 1e2, n_layers, dtype=jnp.float64)
    temperatures = 700.0 + 900.0 * jax.random.uniform(
        split(), (n_layers,), dtype=jnp.float64
    )
    weights_gauss = jax.random.uniform(split(), (n_g,), dtype=jnp.float64)
    weights_gauss = weights_gauss / jnp.sum(weights_gauss)
    mean_molar_masses = 2.0 + 0.5 * jax.random.uniform(
        split(), (n_layers,), dtype=jnp.float64
    )
    opacities = 1e-4 + jax.random.uniform(
        split(), (n_g, n_frequencies, n_species, n_layers), dtype=jnp.float64
    )
    continuum = 1e-5 + jax.random.uniform(
        split(), (n_frequencies, n_layers), dtype=jnp.float64
    )

    return (
        opacities,
        continuum,
        pressures,
        temperatures,
        weights_gauss,
        mean_molar_masses,
        jnp.array(1e3, dtype=jnp.float64),
        jnp.array(1e-2, dtype=jnp.float64),
        jnp.array(7e9, dtype=jnp.float64),
    )


def _benchmark_pair(optimized_function, reference_function, args, loops):
    optimized_output = optimized_function(*args)
    optimized_output[0].block_until_ready()

    reference_output = reference_function(*args)
    reference_output[0].block_until_ready()

    optimized_samples_ms = []
    reference_samples_ms = []

    for iteration in range(loops):
        if iteration % 2 == 0:
            timed_functions = (
                (optimized_function, optimized_samples_ms),
                (reference_function, reference_samples_ms),
            )
        else:
            timed_functions = (
                (reference_function, reference_samples_ms),
                (optimized_function, optimized_samples_ms),
            )

        for function, samples_ms in timed_functions:
            start = time.perf_counter()
            output = function(*args)
            output[0].block_until_ready()
            samples_ms.append((time.perf_counter() - start) * 1e3)

    return {
        "optimized_mean_ms": float(np.mean(optimized_samples_ms)),
        "reference_mean_ms": float(np.mean(reference_samples_ms)),
        "optimized_median_ms": float(np.median(optimized_samples_ms)),
        "reference_median_ms": float(np.median(reference_samples_ms)),
        "optimized_samples_ms": optimized_samples_ms,
        "reference_samples_ms": reference_samples_ms,
    }


def _collect_case_result(case, loops):
    n_g, n_frequencies, n_species, n_layers = case
    seed = int(n_g * 10_000_000 + n_frequencies * 10_000 + n_species * 100 + n_layers)
    args = _build_inputs(seed, n_g, n_frequencies, n_species, n_layers)

    optimized_output = rtc.compute_transit_radii(*args)
    optimized_output[0].block_until_ready()

    reference_output = _reference_compute_transit_radii(*args)
    reference_output[0].block_until_ready()

    radius_diff = jnp.abs(optimized_output[0] - reference_output[0])
    radius_grad = jax.grad(
        lambda temps: jnp.sum(
            rtc.compute_transit_radii(
                args[0],
                args[1],
                args[2],
                temps,
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
            )[0]
        )
    )(args[3])
    radius_grad.block_until_ready()

    reference_grad = jax.grad(
        lambda temps: jnp.sum(
            _reference_compute_transit_radii(
                args[0],
                args[1],
                args[2],
                temps,
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
            )[0]
        )
    )(args[3])
    reference_grad.block_until_ready()

    grad_diff = jnp.abs(radius_grad - reference_grad)
    benchmark = _benchmark_pair(
        rtc.compute_transit_radii, _reference_compute_transit_radii, args, loops
    )

    return {
        "case": case,
        "max_radius_diff": float(jnp.max(radius_diff)),
        "max_radius_rel_diff": float(
            jnp.max(radius_diff / jnp.maximum(jnp.abs(reference_output[0]), 1.0))
        ),
        "max_profile_diff": float(
            jnp.max(jnp.abs(optimized_output[1] - reference_output[1]))
        ),
        "max_grad_diff": float(jnp.max(grad_diff)),
        "max_grad_rel_diff": float(
            jnp.max(grad_diff / jnp.maximum(jnp.abs(reference_grad), 1.0))
        ),
        **benchmark,
        "mean_speedup_ratio": (
            benchmark["reference_mean_ms"] / benchmark["optimized_mean_ms"]
        ),
        "speedup_ratio": (
            benchmark["reference_median_ms"] / benchmark["optimized_median_ms"]
        ),
    }


def run(cases=DEFAULT_CASES, loops=8):
    return [_collect_case_result(case, loops) for case in cases]


def main():
    print(f"petitRADTRANS {petitRADTRANS.__version__}")
    for result in run():
        print(result)


if __name__ == "__main__":
    main()
