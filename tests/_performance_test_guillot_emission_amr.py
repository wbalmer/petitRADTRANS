"""Steady-state AMR benchmark for the runtime ``guillot_emission`` model.

Run from the repository root with:

    python -m tests._performance_test_guillot_emission_amr

This reproduces the 27-layer, 1 correlated-k H2O line, 1 Mg2SiO4 cloud case
used to track AMR regressions in the differentiable runtime retrieval path.
"""

import argparse
import json
import statistics
import time

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from .context import petitRADTRANS
from .utils import test_parameters

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.opacities.opacities import CloudOpacity
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.retrieval.models import guillot_emission
from petitRADTRANS.retrieval.runtime import ModelContext, PhysicalParams


def _block_tree(tree):
    jax.tree_util.tree_map(
        lambda x: jax.block_until_ready(x) if hasattr(x, "shape") else x,
        tree,
    )
    return tree


def _build_benchmark_case():
    pressures = jnp.asarray(test_parameters["pressures"])
    line_species = tuple(
        test_parameters["spectrum_parameters"]["line_species_correlated_k"][:1]
    )
    cloud_species = tuple(
        list(test_parameters["cloud_parameters"]["cloud_species"].keys())[:1]
    )
    rayleigh_species = tuple(
        test_parameters["spectrum_parameters"]["rayleigh_species"]
    )
    continuum_opacities = tuple(
        test_parameters["spectrum_parameters"]["continuum_opacities"]
    )
    wavelength_boundaries = tuple(
        test_parameters["spectrum_parameters"]["wavelength_range_correlated_k"]
    )
    cloud_info = test_parameters["cloud_parameters"]["cloud_species"][cloud_species[0]]
    cloud_name = CloudOpacity(
        [cloud_species[0]], natural_abundance=False
    ).species_full_name.split("_")[0]

    radtrans = Radtrans(
        pressures=pressures,
        line_species=line_species,
        rayleigh_species=rayleigh_species,
        gas_continuum_contributors=continuum_opacities,
        cloud_species=cloud_species,
        wavelength_boundaries=wavelength_boundaries,
        line_opacity_mode="c-k",
        scattering_in_emission=True,
    )

    physical_params = PhysicalParams.from_mapping(
        {
            "distance_to_system": jnp.asarray(10.0 * cst.pc),
            "log_g": jnp.asarray(
                jnp.log10(test_parameters["planetary_parameters"]["reference_gravity"])
            ),
            "planet_radius": jnp.asarray(
                test_parameters["planetary_parameters"]["radius"] * cst.r_jup_mean
            ),
            "T_int": jnp.asarray(
                test_parameters["temperature_guillot_2010_parameters"][
                    "intrinsic_temperature"
                ]
            ),
            "T_equ": jnp.asarray(
                test_parameters["temperature_guillot_2010_parameters"][
                    "equilibrium_temperature"
                ]
            ),
            "gamma": jnp.asarray(
                test_parameters["temperature_guillot_2010_parameters"]["gamma"]
            ),
            "log_kappa_IR": jnp.asarray(
                jnp.log10(
                    test_parameters["temperature_guillot_2010_parameters"][
                        "infrared_mean_opacity"
                    ]
                )
            ),
            "Fe/H": jnp.asarray(
                test_parameters["chemical_parameters"]["metallicities"][1]
            ),
            "C/O": jnp.asarray(
                test_parameters["chemical_parameters"]["c_o_ratios"][1]
            ),
            "sigma_lnorm": jnp.asarray(cloud_info["sigma_log_normal"]),
            "fsed": jnp.asarray(cloud_info["f_sed"]),
            f"log_cloud_radius_{cloud_name}": jnp.asarray(
                jnp.log10(cloud_info["radius"])
            ),
            f"log_X_cb_{cloud_name}": jnp.asarray(
                jnp.log10(cloud_info["mass_fraction"])
            ),
            f"log_Pbase_{cloud_name}": jnp.asarray(
                jnp.log10(test_parameters["cloud_parameters"]["cloud_pressure"])
            ),
        }
    )

    baseline_context = ModelContext(
        name="baseline",
        mode="emission",
        radtrans=radtrans,
        adaptive_mesh_refinement=False,
    )
    amr_context = ModelContext(
        name="amr",
        mode="emission",
        radtrans=radtrans,
        adaptive_mesh_refinement=True,
    )

    return {
        "pressures": pressures,
        "line_species": line_species,
        "cloud_species": cloud_species,
        "wavelength_boundaries": wavelength_boundaries,
        "baseline_fn": lambda: guillot_emission(
            baseline_context,
            physical_params,
            pt_plot_mode=False,
        ),
        "amr_fn": lambda: guillot_emission(
            amr_context,
            physical_params,
            pt_plot_mode=False,
        ),
    }


def _time_interleaved(baseline_fn, amr_fn, warmup_runs, timed_runs):
    for _ in range(warmup_runs):
        _block_tree(baseline_fn())
        _block_tree(amr_fn())

    baseline_samples = []
    amr_samples = []

    for _ in range(timed_runs):
        start = time.perf_counter()
        _block_tree(baseline_fn())
        baseline_samples.append((time.perf_counter() - start) * 1e3)

        start = time.perf_counter()
        _block_tree(amr_fn())
        amr_samples.append((time.perf_counter() - start) * 1e3)

    return baseline_samples, amr_samples


def run(warmup_runs=6, timed_runs=20):
    case = _build_benchmark_case()
    baseline_samples, amr_samples = _time_interleaved(
        case["baseline_fn"],
        case["amr_fn"],
        warmup_runs,
        timed_runs,
    )

    baseline_mean = statistics.mean(baseline_samples)
    baseline_median = statistics.median(baseline_samples)
    amr_mean = statistics.mean(amr_samples)
    amr_median = statistics.median(amr_samples)

    return {
        "pressure_layers": int(case["pressures"].shape[0]),
        "line_species": case["line_species"],
        "cloud_species": case["cloud_species"],
        "wavelength_boundaries": case["wavelength_boundaries"],
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "baseline_mean_ms": baseline_mean,
        "baseline_median_ms": baseline_median,
        "baseline_min_ms": min(baseline_samples),
        "baseline_max_ms": max(baseline_samples),
        "amr_mean_ms": amr_mean,
        "amr_median_ms": amr_median,
        "amr_min_ms": min(amr_samples),
        "amr_max_ms": max(amr_samples),
        "amr_over_baseline_mean_ratio": amr_mean / baseline_mean,
        "amr_over_baseline_median_ratio": amr_median / baseline_median,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runtime guillot_emission with and without AMR."
    )
    parser.add_argument("--warmup-runs", type=int, default=6)
    parser.add_argument("--timed-runs", type=int, default=20)
    args = parser.parse_args()

    print(json.dumps(run(args.warmup_runs, args.timed_runs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()