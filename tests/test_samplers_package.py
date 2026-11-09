"""Structural guards for the petitRADTRANS.retrieval.samplers package.

These tests protect the three invariants that the split of the old monolithic
``sampler.py`` depends on, and that ordinary backend tests would not catch:

1. the ``sampler_type`` registry stays consistent with the classes it names;
2. importing the package does not drag in any optional backend;
3. no module binds a lazy third-party handle by value at import time.

They also run each backend briefly against the installed library versions.
"""

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import warnings
from types import SimpleNamespace

import jax

# pRT fixes its floating-point precision when first imported, so enable x64
# before importing it, as the other test modules do through tests/context.py.
# Otherwise the precision would depend on which test module pytest collects first.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfpd

from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.parameter import Parameter
from petitRADTRANS.retrieval.runtime import (
    ModelGroupState,
    ModelResult,
    ParameterLayout,
    RetrievalRuntime,
)
from petitRADTRANS.retrieval.samplers import (
    _SAMPLER_REGISTRY,
    SAMPLER_DISPATCH,
    Sampler,
    SamplerContext,
    resolve_sampler_class,
)
from petitRADTRANS.retrieval.samplers import _optional_imports

SAMPLERS_DIRECTORY = pathlib.Path(_optional_imports.__file__).parent

#: Backends that must not be imported just by importing the package.
OPTIONAL_BACKENDS = (
    "jaxns",
    "numpyro",
    "blackjax",
    "dynesty",
    "nautilus",
    "pymultinest",
    "ultranest",
    "optax",
    "arviz",
)

#: Names in _optional_imports that start life as None and are rebound by a
#: require_*() call. Importing one of these by value freezes it at None.
LAZY_HANDLE_NAMES = frozenset(
    name
    for name, value in vars(_optional_imports).items()
    if not name.startswith("__") and value is None
)


# --------------------------------------------------------------------------- #
# 1. Registry consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alias", sorted(_SAMPLER_REGISTRY))
def test_every_registry_alias_resolves_to_a_class_that_declares_it(alias):
    """A wrong module/class name in the table would only fail at run time."""
    try:
        sampler_class = resolve_sampler_class(alias)
    except ImportError:
        pytest.skip(f"backend for sampler_type={alias!r} is not installed")

    assert issubclass(sampler_class, Sampler)
    assert alias in {sampler_class.sampler_type, *sampler_class.aliases}, (
        f"{sampler_class.__name__} does not declare the alias {alias!r} that the "
        "registry maps to it"
    )


def test_registry_and_dispatch_agree_on_canonical_names():
    for alias, (module_basename, class_name) in _SAMPLER_REGISTRY.items():
        assert alias == alias.lower(), "aliases are looked up lower-cased"
        assert module_basename.isidentifier()
        assert class_name.isidentifier()


def test_resolve_sampler_class_is_lenient_about_retired_names_when_asked():
    retired = "jaxnsshardedstaticnestedsampler"

    with pytest.raises(ValueError, match="no longer available"):
        resolve_sampler_class(retired)

    assert resolve_sampler_class(retired, strict=False) is None
    assert resolve_sampler_class("not-a-sampler", strict=False) is None


def test_custom_backends_registered_in_dispatch_win():
    class _Custom:
        pass

    SAMPLER_DISPATCH["_test_custom"] = _Custom
    try:
        assert resolve_sampler_class("_test_custom") is _Custom
    finally:
        del SAMPLER_DISPATCH["_test_custom"]


# --------------------------------------------------------------------------- #
# 2. Import hygiene
# --------------------------------------------------------------------------- #
def test_importing_the_samplers_package_imports_no_backend():
    """Users install only the backends they run, so the registry must stay lazy.

    Run in a subprocess: this process has already imported backends for other
    tests, so an in-process check would prove nothing.
    """
    script = (
        "import sys\n"
        "import petitRADTRANS.retrieval.retrieval\n"
        f"leaked = sorted({OPTIONAL_BACKENDS!r} and "
        f"set({list(OPTIONAL_BACKENDS)!r}) & set(sys.modules))\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    leaked = completed.stdout.strip()
    assert leaked == "", f"importing the retrieval package pulled in: {leaked}"


# --------------------------------------------------------------------------- #
# 3. Lazy handles are never bound by value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module_path",
    sorted(SAMPLERS_DIRECTORY.glob("*.py")),
    ids=lambda path: path.name,
)
def test_module_never_imports_a_lazy_handle_by_value(module_path):
    """``from _optional_imports import jnp`` would freeze ``jnp`` at ``None``.

    The handles only become real when ``require_*()`` runs, which rebinds them
    on the module object -- an importing module keeps its own stale ``None``
    and fails much later with a confusing 'NoneType is not callable'.
    """
    tree = ast.parse(module_path.read_text())

    offenders = [
        f"{node.module}.{alias.name} (line {node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("_optional_imports")
        for alias in node.names
        if alias.name in LAZY_HANDLE_NAMES
    ]

    assert not offenders, (
        f"{module_path.name} imports lazy handles by value: {offenders}. "
        "Import the module instead and use _opt.<name>."
    )


# --------------------------------------------------------------------------- #
# 4. BlackJAX and NumPyro really do share their preparation
# --------------------------------------------------------------------------- #
def _build_minimal_context(tmp_path, retrieval_name):
    """A two-parameter differentiable retrieval, enough to call prepare()."""
    parameters = {
        "alpha": Parameter(
            name="alpha",
            is_free_parameter=True,
            distribution=tfpd.Uniform(low=0.0, high=1.0),
        ),
        "beta": Parameter(
            name="beta",
            is_free_parameter=True,
            distribution=tfpd.Uniform(low=1.0, high=2.0),
        ),
    }

    def model_function(context, params, pt_plot_mode=False):
        total = params.get("alpha") + params.get("beta")
        return ModelResult(
            kind="spectrum",
            wavelengths=jnp.array([1.0, 2.0]),
            spectrum=jnp.array([total, total]),
        )

    data = Data(
        name="simple",
        model_generating_function=model_function,
        radtrans_object=SimpleNamespace(),
        wavelengths=jnp.array([1.0, 2.0]),
        spectrum=jnp.array([2.0, 2.0]),
        uncertainties=jnp.array([0.5, 0.5]),
    )
    configuration = SimpleNamespace(
        parameters=parameters, data={"simple": data}, adaptive_mesh_refinement=False
    )

    parameter_layout = ParameterLayout.from_parameters(parameters)
    runtime = RetrievalRuntime(
        parameter_layout=parameter_layout,
        observations={
            "simple": data.to_observation_state(model_contract="differentiable")
        },
        likelihoods={
            "simple": data.to_likelihood_state(
                n_free_parameters=parameter_layout.n_free_parameters
            )
        },
        model_groups=(
            ModelGroupState(
                source_name="simple",
                observation_names=("simple",),
                model_contract="differentiable",
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
        output_directory=str(tmp_path),
        retrieval_name=retrieval_name,
        free_parameter_names=list(parameter_layout.free_parameter_names),
        n_free_parameters=parameter_layout.n_free_parameters,
    )


#: Defaults that the shared GradientSampler helpers must keep producing. These
#: are the values the two backends had before their preparation was unified.
SHARED_RUN_KWARG_DEFAULTS = {
    "num_samples": 1000,
    "num_warmup": 1000,
    "initial_step_size": 1.0,
    "target_acceptance_rate": 0.8,
    "is_mass_matrix_diagonal": True,
    "progress_bar": False,
    "seed": 42654,
    "use_jit": True,
    "num_chains": 1,
}


@pytest.mark.parametrize("sampler_type", ["blackjaxnuts", "numpyronuts"])
def test_gradient_samplers_share_the_same_prepared_defaults(
    tmp_path, sampler_type
):
    pytest.importorskip({"blackjaxnuts": "blackjax", "numpyronuts": "numpyro"}[sampler_type])

    sampler_class = resolve_sampler_class(sampler_type)
    context = _build_minimal_context(tmp_path, f"gradient_{sampler_type}")
    sampler, run_kwargs, summary_parameters = sampler_class.prepare(context)

    for key, expected in SHARED_RUN_KWARG_DEFAULTS.items():
        assert run_kwargs[key] == expected, key

    assert run_kwargs["parameter_names"] == ["alpha", "beta"]
    assert run_kwargs["log_jacobian_fn"] is not None
    assert run_kwargs["initial_position"].shape == (2,)

    assert summary_parameters["sampler_type"] == "nuts"
    # progress_bar is deliberately absent: it is a display option, not a
    # setting worth recording in the run summary.
    assert "progress_bar" not in summary_parameters
    for key in SHARED_RUN_KWARG_DEFAULTS.keys() - {"progress_bar"}:
        assert summary_parameters[key] == SHARED_RUN_KWARG_DEFAULTS[key], key

    # The backend owns its output folder, and prepare() must have created it.
    assert (tmp_path / sampler_class.output_subdirectory).is_dir()
    assert sampler.uses_distribution_priors is True


def test_numpyro_prepare_keeps_its_own_extra_defaults(tmp_path):
    pytest.importorskip("numpyro")

    sampler_class = resolve_sampler_class("numpyronuts")
    context = _build_minimal_context(tmp_path, "gradient_numpyro_extra")
    _sampler, run_kwargs, summary_parameters = sampler_class.prepare(context)

    # A single chain must stay sequential: "vectorized" multiplies per-step
    # memory by the chain count and OOMs heavy forward models.
    assert run_kwargs["chain_method"] == "sequential"
    assert run_kwargs["thinning"] == 1
    assert summary_parameters["chain_method"] == "sequential"
    assert summary_parameters["thinning"] == 1


def test_blackjax_prepare_does_not_gain_numpyro_only_keys(tmp_path):
    pytest.importorskip("blackjax")

    sampler_class = resolve_sampler_class("blackjaxnuts")
    context = _build_minimal_context(tmp_path, "gradient_blackjax_extra")
    _sampler, run_kwargs, summary_parameters = sampler_class.prepare(context)

    assert "chain_method" not in run_kwargs
    assert "thinning" not in run_kwargs
    assert "chain_method" not in summary_parameters
    assert "thinning" not in summary_parameters


# --------------------------------------------------------------------------- #
# 5. The backends really run against the installed library versions
# --------------------------------------------------------------------------- #
def test_blackjax_single_chain_runs(tmp_path):
    """BlackJAX >= 1.6 rejects window_adaptation's former progress_bar kwarg."""
    pytest.importorskip("blackjax")

    sampler_class = resolve_sampler_class("blackjaxnuts")
    context = _build_minimal_context(tmp_path, "blackjax_run")
    sampler, run_kwargs, _ = sampler_class.prepare(
        context, num_samples=50, num_warmup=50, progress_bar=True
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = sampler.run_sampler(**run_kwargs)

    assert results.samples.shape == (50, 2)
    if importlib.util.find_spec("jaxtap") is None:
        assert any("jaxtap" in str(warning.message) for warning in caught)


def test_optimal_estimation_results_survive_a_reload(tmp_path):
    pytest.importorskip("optax")
    from petitRADTRANS.retrieval.samplers.optimal_estimation import (
        OptimalEstimationResults,
    )

    sampler_class = resolve_sampler_class("optimal_estimation")
    context = _build_minimal_context(tmp_path, "optax_reload")
    sampler, run_kwargs, _ = sampler_class.prepare(
        context, n_steps=30, n_posterior_samples=40, verbose=False
    )
    ran = sampler.run_sampler(**run_kwargs)

    # A fresh sampler, as run_mode="evaluate" builds one.
    reloader, _, _ = sampler_class.prepare(context)
    reloaded = reloader.load_results()

    assert isinstance(reloaded, OptimalEstimationResults)
    assert reloaded.best_logdensity == pytest.approx(ran.best_logdensity)
    np.testing.assert_allclose(reloaded.samples, ran.samples)
    assert reloaded.parameter_names == ["alpha", "beta"]

    summary = reloader.get_summary(free_parameter_names=["alpha", "beta"])
    assert summary["parameters"]["best_logdensity"] == pytest.approx(ran.best_logdensity)
    assert len(summary["posterior"]) == 2


_MULTI_DEVICE_SCRIPT = """
import importlib.util
import sys

import jax

spec = importlib.util.spec_from_file_location("samplers_tests", {test_file!r})
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)

assert jax.local_device_count() == 2, jax.local_device_count()
sampler_class = tests.resolve_sampler_class({sampler_type!r})
for num_chains in (2, 3, 4):
    context = tests._build_minimal_context({output_directory!r}, f"chains_{{num_chains}}")
    sampler, run_kwargs, _ = sampler_class.prepare(
        context, num_samples=20, num_warmup=20, num_chains=num_chains
    )
    results = sampler.run_sampler(**run_kwargs)
    print(num_chains, results.samples.shape[0], results.samples.shape[1])
"""


@pytest.mark.parametrize("sampler_type", ["blackjaxnuts", "numpyronuts"])
def test_multi_chain_runs_shard_over_devices_without_dropping_chains(
    tmp_path, sampler_type
):
    """2 and 4 chains are sharded over two devices; 3 do not divide, so they
    are vectorised on one. Every chain must contribute its draws.

    Run in a subprocess: the host device count is fixed when JAX initialises.
    """
    pytest.importorskip({"blackjaxnuts": "blackjax", "numpyronuts": "numpyro"}[sampler_type])

    script = _MULTI_DEVICE_SCRIPT.format(
        test_file=str(pathlib.Path(__file__).resolve()),
        sampler_type=sampler_type,
        output_directory=str(tmp_path),
    )
    environment = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    shapes = [
        tuple(int(value) for value in line.split())
        for line in completed.stdout.splitlines()
        if line.strip() and line.split()[0].isdigit()
    ]
    assert shapes == [(2, 40, 2), (3, 60, 2), (4, 80, 2)]



def test_numpyro_managed_warmup_compiles_once_and_resumes_exactly(monkeypatch):
    """The managed warmup compiles NUTS once per process, not once per iteration, and a
    warmup stopped after its first checkpointed block resumes to the uninterrupted state.

    Called eagerly, kernel.sample re-traced and recompiled the tree-building loop on every
    iteration, ~60 s each for an emission retrieval on an A100.
    """
    pytest.importorskip("numpyro")
    from numpyro.infer import NUTS

    from petitRADTRANS.retrieval.samplers import numpyro as numpyro_backend

    traces = []

    def potential(position):
        traces.append(None)  # Python runs only while JAX traces
        return 0.5 * jnp.sum(position**2)

    def run(num_warmup, checkpoints, checkpoint=None):
        sampler = object.__new__(numpyro_backend.NumPyroNUTSSampler)
        sampler.retrieval_name = "managed_warmup"
        sampler._save_checkpoint = checkpoints.append
        kernel = NUTS(
            potential_fn=potential,
            adapt_step_size=False,
            adapt_mass_matrix=False,
            max_tree_depth=4,
            step_size=0.5,
        )
        return sampler._run_managed_warmup(
            kernel=kernel,
            initial_position=jnp.full(3, 0.5),
            rng_key=jax.random.PRNGKey(0),
            num_warmup=num_warmup,
            num_samples=1,
            target_accept_prob=0.8,
            warmup_block=2,
            checkpoint=checkpoint,
        )

    monkeypatch.delenv("PRT_WARMUP_ITERS_PER_PROCESS", raising=False)
    run(3, [])
    traces_short = len(traces)
    traces.clear()
    state, trace = run(9, [])
    assert len(traces) == traces_short

    # Stop after the first block, then resume from its checkpoint
    checkpoints = []
    monkeypatch.setenv("PRT_WARMUP_ITERS_PER_PROCESS", "2")
    with pytest.raises(numpyro_backend.WarmupChunkComplete):
        run(9, checkpoints)
    monkeypatch.delenv("PRT_WARMUP_ITERS_PER_PROCESS")
    resumed_state, resumed_trace = run(9, [], checkpoint=checkpoints[-1])

    assert resumed_trace == trace
    for resumed, uninterrupted in zip(jax.tree.leaves(resumed_state), jax.tree.leaves(state)):
        np.testing.assert_array_equal(np.asarray(resumed), np.asarray(uninterrupted))
