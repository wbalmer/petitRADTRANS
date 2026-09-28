"""Structural guards for the petitRADTRANS.retrieval.samplers package.

These tests protect the three invariants that the split of the old monolithic
``sampler.py`` depends on, and that ordinary backend tests would not catch:

1. the ``sampler_type`` registry stays consistent with the classes it names;
2. importing the package does not drag in any optional backend;
3. no module binds a lazy third-party handle by value at import time.
"""

import ast
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import jax.numpy as jnp
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
