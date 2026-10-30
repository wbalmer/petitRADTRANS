"""Module-level helpers for the Nautilus tests.

These live in an importable module because the MPI test runs them as a script
in other processes, and because ``spawn`` pool workers must be able to re-import
what the retrieval runtime refers to.
"""

import json
import sys
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tensorflow_probability.substrates.jax import distributions as tfpd  # noqa: E402

from petitRADTRANS.retrieval.data import Data  # noqa: E402
from petitRADTRANS.retrieval.parameter import Parameter  # noqa: E402
from petitRADTRANS.retrieval.runtime import (  # noqa: E402
    ModelGroupState,
    ModelResult,
    ParameterLayout,
    RetrievalRuntime,
)
from petitRADTRANS.retrieval.samplers import SamplerContext  # noqa: E402

WAVELENGTHS = np.linspace(1.0, 2.0, 12)
# Finer than the data, as for a real retrieval. On the data's own grid, an eager
# evaluation skips the rebinning that a jitted one applies, so the two would differ.
MODEL_WAVELENGTHS = np.linspace(0.9, 2.1, 300)
UNCERTAINTY = 0.3
TRUE_ALPHA = 0.4
TRUE_BETA = 1.5


def linear_model(context, params, pt_plot_mode=False):
    """A straight line, whose posterior is cheap to integrate on a grid."""
    wavelengths = jnp.asarray(MODEL_WAVELENGTHS)
    spectrum = params.get("alpha") + params.get("beta") * wavelengths
    return ModelResult(kind="spectrum", wavelengths=wavelengths, spectrum=spectrum)


def walled_linear_model(context, params, pt_plot_mode=False):
    """The same line, but invalid for ``alpha > TRUE_ALPHA``."""
    result = linear_model(context, params, pt_plot_mode)
    spectrum = jnp.where(params.get("alpha") > TRUE_ALPHA, jnp.nan, result.spectrum)
    return ModelResult(kind="spectrum", wavelengths=result.wavelengths, spectrum=spectrum)


def invalid_model(context, params, pt_plot_mode=False):
    """A model with no valid point anywhere."""
    wavelengths = jnp.asarray(MODEL_WAVELENGTHS)
    return ModelResult(
        kind="spectrum",
        wavelengths=wavelengths,
        spectrum=jnp.full_like(wavelengths, jnp.nan) * params.get("alpha"),
    )


def build_context(
    output_directory,
    retrieval_name="nautilus",
    model_function=linear_model,
    n_extra_parameters=0,
    observed_offset=0.0,
    model_contract="differentiable",
):
    """A differentiable retrieval of a line, with uniform priors on its two coefficients."""
    parameters = {
        "alpha": Parameter(
            name="alpha", is_free_parameter=True, distribution=tfpd.Uniform(low=0.0, high=1.0)
        ),
        "beta": Parameter(
            name="beta", is_free_parameter=True, distribution=tfpd.Uniform(low=1.0, high=2.0)
        ),
    }
    for index in range(n_extra_parameters):
        name = f"extra_{index}"
        parameters[name] = Parameter(
            name=name, is_free_parameter=True, distribution=tfpd.Uniform(low=0.0, high=1.0)
        )

    wavelengths = jnp.asarray(WAVELENGTHS)
    data = Data(
        name="simple",
        model_generating_function=model_function,
        radtrans_object=SimpleNamespace(line_species=[], cloud_species=[]),
        wavelengths=wavelengths,
        spectrum=TRUE_ALPHA + TRUE_BETA * wavelengths + observed_offset,
        uncertainties=UNCERTAINTY * jnp.ones_like(wavelengths),
    )
    configuration = SimpleNamespace(
        parameters=parameters,
        data={"simple": data},
        adaptive_mesh_refinement=False,
        sampler_type="nautilus",
        retrieval_name=retrieval_name,
        run_mode="retrieval",
    )
    parameter_layout = ParameterLayout.from_parameters(parameters)
    runtime = RetrievalRuntime(
        parameter_layout=parameter_layout,
        observations={"simple": data.to_observation_state(model_contract=model_contract)},
        likelihoods={
            "simple": data.to_likelihood_state(
                n_free_parameters=parameter_layout.n_free_parameters
            )
        },
        model_groups=(
            ModelGroupState(
                source_name="simple",
                observation_names=("simple",),
                model_contract=model_contract,
                model_generating_function=model_function,
                radtrans_object=data.radtrans_object,
                metadata={
                    "line_opacity_mode": data.line_opacity_mode,
                    "has_external_references": False,
                },
            ),
        ),
    )

    return SamplerContext(
        parameter_layout=parameter_layout,
        runtime=runtime,
        configuration=configuration,
        output_directory=str(output_directory),
        retrieval_name=retrieval_name,
        free_parameter_names=list(parameter_layout.free_parameter_names),
        n_free_parameters=parameter_layout.n_free_parameters,
    )


def log_evidence_by_quadrature(context, n_grid=1201):
    """``ln Z`` of a two-parameter retrieval with unit-volume uniform priors."""
    cube = (np.arange(n_grid) + 0.5) / n_grid
    alpha, beta = np.meshgrid(cube, cube + 1.0, indexing="ij")
    points = np.column_stack((alpha.ravel(), beta.ravel()))

    evaluate = jax.jit(
        lambda chunk: context.runtime.evaluate_likelihood_jax_batched(
            free_parameter_values=chunk
        )
    )
    log_likelihoods = np.concatenate(
        [np.asarray(evaluate(chunk)) for chunk in np.array_split(points, n_grid)]
    )
    log_likelihoods = np.where(log_likelihoods > -1e30, log_likelihoods, -np.inf)
    largest = np.max(log_likelihoods)

    return largest + np.log(np.mean(np.exp(log_likelihoods - largest)))


def _run_under_mpi(output_directory):
    """Run a retrieval through the sampler lifecycle, as every MPI rank does."""
    from petitRADTRANS.retrieval._mpi import get_mpi_context
    from petitRADTRANS.retrieval.samplers.nautilus import NautilusSampler

    _, comm, rank = get_mpi_context()
    context = build_context(output_directory, retrieval_name="nautilus_mpi")

    # Retrieval.run() holds a barrier before sampling, and this must not hang.
    comm.barrier()
    sampler, run_kwargs, summary_parameters = NautilusSampler.prepare(
        context, n_live=200, n_eff=500, verbose=False, seed=3
    )
    results = sampler.run_sampler(**run_kwargs)
    results, _ = sampler.post_run(results)
    comm.barrier()

    if rank == 0:
        print(
            "NAUTILUS_MPI_RESULT "
            + json.dumps(
                {
                    "mpi_workers": summary_parameters["mpi_workers"],
                    "n_like": results["n_like"],
                    "logz": results["logz"],
                    "logzerr": results["logzerr"],
                    "n_samples": int(results["samples"].shape[0]),
                    "mean": np.mean(results["samples"], axis=0).tolist(),
                }
            ),
            flush=True,
        )
    elif results is not None:
        raise RuntimeError(f"Rank {rank} was given results; only rank 0 should be.")


if __name__ == "__main__":
    _run_under_mpi(sys.argv[1])
