"""Benchmark and verify Planck-function dtype behavior in physics_core.

This is a manual benchmark. It compares the current float32-safe Planck
implementations against frozen reference formulas for float64 stability and
checks that float32 outputs agree with the float64 result cast to float32.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.radtrans_core import physics_core as pc

DEFAULT_CASES = (
    (32, 256, 1e10, 3e14),
    (48, 1024, 1e10, 3e14),
)

FLOAT32_RTOL = 128.0 * np.finfo(np.float32).eps
FLOAT32_ATOL = 32.0 * np.finfo(np.float32).tiny
FLOAT64_RTOL = 1e-13


def _reference_compute_planck_function(temperatures, frequencies):
    temperatures = jnp.atleast_1d(jnp.asarray(temperatures))
    temperature_matrix = jnp.tile(temperatures, (frequencies.shape[0], 1))
    exponent = cst.h * frequencies[:, None] / (cst.kB * temperature_matrix)
    return (
        (2.0 * cst.h / cst.c)
        * (frequencies[:, None] / cst.c)
        * frequencies[:, None] ** 2
        / jnp.expm1(exponent)
    )


def _reference_compute_planck_function_temperature_derivative(
    temperatures, frequency_bin_edges
):
    temperatures = jnp.atleast_1d(jnp.asarray(temperatures))
    frequencies = 0.5 * (frequency_bin_edges[:-1] + frequency_bin_edges[1:])
    h_over_kb = cst.h / cst.kB
    exp_arg = h_over_kb * jnp.outer(0.5 / temperatures, frequencies)
    denominator = (2.0 * jnp.sinh(exp_arg)) ** 2 * jnp.outer(
        cst.kB * temperatures**2, jnp.ones_like(frequencies)
    )
    h_nu = cst.h * frequencies
    numerator_nu = h_nu**2 * frequencies**2
    return (
        2.0
        / cst.c**2
        * jnp.outer(jnp.ones_like(temperatures), numerator_nu)
        / denominator
    ).T


def _reference_compute_planck_function_integral(temperatures, frequencies):
    temperatures = jnp.atleast_1d(jnp.asarray(temperatures))

    if bool(frequencies[1] >= frequencies[0]):
        frequencies = jnp.flip(frequencies)

    cst_h_over_c = cst.h / cst.c
    cst_h_k = cst.h / cst.kB
    factor1 = 1.0 / 90.0
    factor2 = 14.0 * cst_h_over_c
    factor3 = 64.0 * cst_h_over_c
    factor4 = 24.0 * cst_h_over_c

    cst_h_kt = cst_h_k / temperatures

    nu_large = frequencies[:-1]
    nu_small = frequencies[1:]
    diff_nu = nu_large - nu_small
    nu2 = nu_small + diff_nu * 0.25
    nu3 = nu_small + diff_nu * 0.5
    nu4 = nu_small + diff_nu * 0.75

    return factor1 * (
        factor2
        * (nu_small[None, :] / cst.c)
        * nu_small[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu_small[None, :])
        + factor3
        * (nu2[None, :] / cst.c)
        * nu2[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu2[None, :])
        + factor4
        * (nu3[None, :] / cst.c)
        * nu3[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu3[None, :])
        + factor3
        * (nu4[None, :] / cst.c)
        * nu4[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu4[None, :])
        + factor2
        * (nu_large[None, :] / cst.c)
        * nu_large[None, :] ** 2
        / jnp.expm1(cst_h_kt[:, None] * nu_large[None, :])
    )


def _reference_star_planck(temperatures, frequencies):
    return _reference_compute_planck_function_integral(temperatures, frequencies).T


def _reference_compute_planck_function_integral_old(temperature, nu_left, nu_right):
    frequency_edges = jnp.stack(
        (jnp.maximum(nu_left, nu_right), jnp.minimum(nu_left, nu_right))
    )
    return _reference_compute_planck_function_integral(
        jnp.atleast_1d(jnp.asarray(temperature)), frequency_edges
    )[0, 0]


def _build_case_inputs(n_temperatures, n_frequency_edges, frequency_min, frequency_max):
    temperatures64 = jnp.geomspace(150.0, 3500.0, n_temperatures, dtype=jnp.float64)
    frequency_edges64 = jnp.geomspace(
        frequency_min,
        frequency_max,
        n_frequency_edges,
        dtype=jnp.float64,
    )
    frequencies64 = 0.5 * (frequency_edges64[:-1] + frequency_edges64[1:])

    return {
        "temperatures64": temperatures64,
        "temperatures32": temperatures64.astype(jnp.float32),
        "frequency_edges64": frequency_edges64,
        "frequency_edges32": frequency_edges64.astype(jnp.float32),
        "frequencies64": frequencies64,
        "frequencies32": frequencies64.astype(jnp.float32),
    }


def _assert_dtype_compatibility(name, current64, reference64, current32):
    current64_np = np.asarray(current64)
    reference64_np = np.asarray(reference64)
    current32_np = np.asarray(current32)
    reference32_np = reference64_np.astype(np.float32)

    np.testing.assert_allclose(
        current64_np,
        reference64_np,
        rtol=FLOAT64_RTOL,
        atol=0.0,
        err_msg=f"{name} changed its float64 output",
    )
    np.testing.assert_allclose(
        current32_np,
        reference32_np,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
        err_msg=f"{name} float32 output drifted beyond float32 precision",
    )

    diff32 = np.abs(current32_np - reference32_np)

    return {
        "all_finite_32": bool(np.isfinite(current32_np).all()),
        "max_abs_diff_64": float(np.max(np.abs(current64_np - reference64_np))),
        "max_abs_diff_32": float(np.max(diff32)),
        "max_rel_diff_32": float(
            np.max(diff32 / np.maximum(np.abs(reference32_np), np.float32(1.0)))
        ),
    }


def _benchmark_current(function, args, loops):
    output = function(*args)
    if isinstance(output, tuple):
        output = output[0]
    output.block_until_ready()

    samples_ms = []
    for _ in range(loops):
        start = time.perf_counter()
        output = function(*args)
        if isinstance(output, tuple):
            output = output[0]
        output.block_until_ready()
        samples_ms.append((time.perf_counter() - start) * 1e3)

    return {
        "mean_ms": float(np.mean(samples_ms)),
        "median_ms": float(np.median(samples_ms)),
    }


def _collect_case_result(case, loops):
    inputs = _build_case_inputs(*case)

    functions = (
        (
            "compute_planck_function",
            pc.compute_planck_function,
            _reference_compute_planck_function,
            (inputs["temperatures64"], inputs["frequencies64"]),
            (inputs["temperatures32"], inputs["frequencies32"]),
        ),
        (
            "compute_planck_function_temperature_derivative",
            pc.compute_planck_function_temperature_derivative,
            _reference_compute_planck_function_temperature_derivative,
            (inputs["temperatures64"], inputs["frequency_edges64"]),
            (inputs["temperatures32"], inputs["frequency_edges32"]),
        ),
        (
            "compute_planck_function_integral",
            pc.compute_planck_function_integral,
            _reference_compute_planck_function_integral,
            (inputs["temperatures64"], inputs["frequency_edges64"]),
            (inputs["temperatures32"], inputs["frequency_edges32"]),
        ),
        (
            "star_planck",
            pc.star_planck,
            _reference_star_planck,
            (inputs["temperatures64"], inputs["frequency_edges64"]),
            (inputs["temperatures32"], inputs["frequency_edges32"]),
        ),
        (
            "compute_planck_function_integral_old",
            pc.compute_planck_function_integral_old,
            _reference_compute_planck_function_integral_old,
            (
                inputs["temperatures64"][0],
                inputs["frequency_edges64"][0],
                inputs["frequency_edges64"][1],
            ),
            (
                inputs["temperatures32"][0],
                inputs["frequency_edges32"][0],
                inputs["frequency_edges32"][1],
            ),
        ),
    )

    result = {"case": case, "functions": {}}

    for name, current_function, reference_function, args64, args32 in functions:
        current64 = current_function(*args64)
        reference64 = reference_function(*args64)
        current32 = current_function(*args32)

        metrics = _assert_dtype_compatibility(name, current64, reference64, current32)
        metrics["float64_timing_ms"] = _benchmark_current(
            current_function, args64, loops
        )
        metrics["float32_timing_ms"] = _benchmark_current(
            current_function, args32, loops
        )
        result["functions"][name] = metrics

    return result


def run(cases=DEFAULT_CASES, loops=8):
    return [_collect_case_result(case, loops) for case in cases]


def main():
    results = run()

    for entry in results:
        print(f"case={entry['case']}")
        for name, metrics in entry["functions"].items():
            print(
                f"  {name}: "
                f"finite32={metrics['all_finite_32']} "
                f"max_abs_diff_64={metrics['max_abs_diff_64']:.3e} "
                f"max_abs_diff_32={metrics['max_abs_diff_32']:.3e} "
                f"max_rel_diff_32={metrics['max_rel_diff_32']:.3e} "
                f"f64_median_ms={metrics['float64_timing_ms']['median_ms']:.3f} "
                f"f32_median_ms={metrics['float32_timing_ms']['median_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
