"""Tests of the Nautilus backend, against the installed library where it is needed."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from tests import nautilus_helpers
from tests.nautilus_helpers import build_context, log_evidence_by_quadrature

from petitRADTRANS.retrieval.parameter import Parameter  # noqa: E402
from petitRADTRANS.retrieval.retrieval import Retrieval  # noqa: E402
from petitRADTRANS.retrieval.runtime import ParameterLayout  # noqa: E402
from petitRADTRANS.retrieval.samplers import resolve_sampler_class  # noqa: E402
from petitRADTRANS.retrieval.samplers.interfaces import (  # noqa: E402
    build_batched_prior_transform,
)
from petitRADTRANS.retrieval.samplers.nautilus import NautilusSampler  # noqa: E402
from petitRADTRANS.retrieval.samplers.nautilus_pool import (  # noqa: E402
    sanitize_log_likelihoods,
)

#: Small enough to keep these fast, large enough for ln Z to within FAST_LOGZ_TOLERANCE.
FAST_SETTINGS = {"n_live": 300, "n_eff": 1000, "verbose": False, "seed": 11}
FAST_LOGZ_TOLERANCE = 0.25


def _run(context, **kwargs):
    """Drive the lifecycle that ``Retrieval.run`` drives."""
    pytest.importorskip("nautilus")
    sampler, run_kwargs, summary_parameters = NautilusSampler.prepare(
        context, **{**FAST_SETTINGS, **kwargs}
    )
    results = sampler.run_sampler(**run_kwargs)
    results, extra = sampler.post_run(results)
    assert extra == {}
    return sampler, results, summary_parameters


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def test_registry_resolves_nautilus():
    assert resolve_sampler_class("nautilus") is NautilusSampler
    assert resolve_sampler_class("NAUTILUS") is NautilusSampler


def test_prepare_applies_the_documented_defaults(tmp_path):
    pytest.importorskip("nautilus")
    context = build_context(tmp_path)

    sampler, run_kwargs, summary_parameters = NautilusSampler.prepare(context)

    init_kwargs = run_kwargs["init_kwargs"]
    assert init_kwargs["n_live"] == 2000
    assert init_kwargs["resume"] is True
    assert run_kwargs["discard_exploration"] is True
    assert init_kwargs["filepath"] == str(
        tmp_path / "out_Nautilus" / "nautilus_checkpoint.hdf5"
    )
    assert run_kwargs["max_initial_invalid_calls"] == 200000
    assert summary_parameters["sampler_type"] == "nautilus"
    assert summary_parameters["n_live"] == 2000
    assert summary_parameters["discard_exploration"] is True
    assert summary_parameters["resume"] is True
    assert summary_parameters["vectorized"] is False
    assert sampler._get_output_directory() == str(tmp_path / "out_Nautilus")


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"pool": object()}, ValueError, "builds the Nautilus pools"),
        ({"not_an_option": 1}, TypeError, "not_an_option"),
        ({"pool_njobs": 2, "sampler_pool_njobs": 2}, ValueError, "sampler_pool_njobs"),
        ({"pool_njobs": 2, "vectorized": True}, ValueError, "vectorized"),
        ({"pool_njobs": 2, "likelihood_devices": []}, ValueError, "likelihood_devices"),
    ],
)
def test_prepare_rejects_unsupported_arguments(tmp_path, kwargs, error, match):
    pytest.importorskip("nautilus")
    with pytest.raises(error, match=match):
        NautilusSampler.prepare(build_context(tmp_path), **kwargs)


def test_prepare_rejects_a_single_free_parameter(tmp_path):
    pytest.importorskip("nautilus")
    context = build_context(tmp_path)
    context.n_free_parameters = 1
    with pytest.raises(ValueError, match="at least 2 free parameters"):
        NautilusSampler.prepare(context)


def test_vectorized_mode_is_disabled_for_legacy_model_groups(tmp_path):
    pytest.importorskip("nautilus")
    context = build_context(tmp_path, model_contract="legacy")
    with pytest.warns(RuntimeWarning, match="vectorized mode disabled because the retrieval contains legacy"):
        _, _, summary_parameters = NautilusSampler.prepare(
            context, vectorized=True, use_jit=False
        )
    assert summary_parameters["vectorized"] is False


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def test_sanitize_log_likelihoods_marks_only_invalid_values():
    values = [
        -3.0,
        0.0,
        -1e30,
        np.nan,
        np.inf,
        -np.inf,
        -1e99,
        np.finfo(np.float64).min,
        float(np.finfo(np.float32).min),
    ]
    assert sanitize_log_likelihoods(values).tolist() == [
        -3.0,
        0.0,
        -1e30,
        -np.inf,
        -np.inf,
        -np.inf,
        -np.inf,
        -np.inf,
        -np.inf,
    ]


def test_batched_prior_transform_matches_the_eager_one(tmp_path):
    context = build_context(tmp_path, n_extra_parameters=3)
    cube = np.random.default_rng(0).random((16, 5))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batched = build_batched_prior_transform(context)

    assert batched is not context.parameter_layout.prior_transform_numpy
    assert np.allclose(batched(cube), context.parameter_layout.prior_transform_numpy(cube))


def test_batched_prior_transform_tolerates_float32_rounding_near_zero():
    """Bounds given as Python numbers make a distribution float32, even under x64.

    Its eager quantile then rounds to float32, and the compiled one does not. That
    error is small compared to the prior, but not to a value close to zero, as at
    the fifth probe point of the check (0.57) here.
    """

    def transform(x):
        # The eager path hands over NumPy scalars, which do subclass float.
        return x - 0.57 + (1e-6 if isinstance(x, float) else 0.0)

    parameters = {
        name: Parameter(name=name, is_free_parameter=True, transform_prior_cube_coordinate=transform)
        for name in ("a", "b")
    }
    layout = ParameterLayout.from_parameters(parameters)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batched = build_batched_prior_transform(SimpleNamespace(parameter_layout=layout))

    assert batched != layout.prior_transform_numpy

    # This is what makes the prior a test: relative to each value, the two disagree.
    probe = np.linspace(0.01, 0.99, 8)[:, np.newaxis] * np.ones(2)
    assert not np.allclose(batched(probe), layout.prior_transform_numpy(probe), rtol=1e-5)


@pytest.mark.parametrize(
    "transform, match",
    [
        # Python's float() cannot be traced.
        (lambda x: 2.0 * float(x), "could not be compiled"),
        # Traces fine, but into something other than the eager path gives.
        (lambda x: 2.0 * x if isinstance(x, float) else 3.0 * x, None),
    ],
)
def test_batched_prior_transform_falls_back_to_the_eager_one(transform, match):
    parameters = {
        name: Parameter(name=name, is_free_parameter=True, transform_prior_cube_coordinate=transform)
        for name in ("a", "b")
    }
    layout = ParameterLayout.from_parameters(parameters)
    cube = np.array([[0.25, 0.5]])
    # The eager path hands over NumPy scalars, which do subclass float.
    with pytest.warns(RuntimeWarning, match=match or "disagrees"):
        batched = build_batched_prior_transform(SimpleNamespace(parameter_layout=layout))

    assert batched == layout.prior_transform_numpy
    assert np.allclose(batched(cube), [[0.5, 1.0]])


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_evidence_and_posterior_match_a_direct_integration(tmp_path):
    context = build_context(tmp_path)
    expected = log_evidence_by_quadrature(context)

    _, results, _ = _run(context, n_live=1000, n_eff=4000)

    assert results["logz"] == pytest.approx(expected, abs=0.1)
    assert results["logzerr"] == pytest.approx(1.0 / np.sqrt(results["n_eff"]))
    assert results["converged"] is True
    assert results["discard_exploration"] is True


def test_results_follow_the_sampler_contract(tmp_path):
    context = build_context(tmp_path)
    _, results, _ = _run(context)

    samples = results["samples"]
    assert samples.ndim == 2 and samples.shape[1] == 2
    assert results["log_likelihood"].shape == (samples.shape[0],)
    assert np.allclose(results["log_weights"], -np.log(samples.shape[0]))

    # Every equal-weight row carries the log likelihood of that very point.
    likelihood = context.runtime.evaluate_likelihood_jax_batched(free_parameter_values=samples)
    assert np.allclose(results["log_likelihood"], np.asarray(likelihood))

    weighted = results["weighted_samples"]
    assert weighted["points"].shape[0] == weighted["logl"].shape[0] == weighted["log_weights"].shape[0]
    assert np.isfinite(weighted["logl"]).all()
    assert np.exp(weighted["log_weights"]).sum() == pytest.approx(1.0)

    # The best fit is read from the equal-weight samples, so it must be among them.
    assert results["log_likelihood"][0] == np.max(weighted["logl"])
    assert np.array_equal(samples[0], weighted["points"][np.argmax(weighted["logl"])])

    # The boost aims for about one draw per effective sample.
    assert samples.shape[0] == pytest.approx(results["n_eff"], rel=0.25)


def test_results_survive_a_reload(tmp_path):
    context = build_context(tmp_path)
    sampler, results, _ = _run(context)

    reloaded_sampler = NautilusSampler(None, None, str(tmp_path), "nautilus")
    reloaded = reloaded_sampler.load_results()

    assert np.array_equal(reloaded["samples"], results["samples"])
    assert np.array_equal(reloaded["log_likelihood"], results["log_likelihood"])
    assert np.array_equal(
        reloaded["weighted_samples"]["log_weights"], results["weighted_samples"]["log_weights"]
    )
    for key in ("logz", "logzerr", "n_like", "n_eff", "eta", "converged", "discard_exploration"):
        assert reloaded[key] == results[key]

    summary = reloaded_sampler.get_summary(["alpha", "beta"])
    assert summary == sampler.get_summary(["alpha", "beta"])
    assert summary["evidence"]["ln_z"] == results["logz"]
    assert summary["diagnostics"]["converged"] is True
    assert [entry["name"] for entry in summary["posterior"]] == ["alpha", "beta"]

    samples, log_likelihood, parameter_names = NautilusSampler.load_samples_archive(
        str(tmp_path), "nautilus"
    )
    assert np.array_equal(samples, results["samples"])
    assert np.array_equal(log_likelihood, results["log_likelihood"])
    assert parameter_names == ["alpha", "beta"]


def test_vectorized_mode_reproduces_the_scalar_run(tmp_path):
    _, scalar, _ = _run(build_context(tmp_path / "scalar"))
    _, batched, summary_parameters = _run(build_context(tmp_path / "batched"), vectorized=True)

    assert summary_parameters["vectorized"] is True
    assert batched["n_like"] == scalar["n_like"]
    assert batched["logz"] == pytest.approx(scalar["logz"], abs=1e-6)


def test_vectorized_batches_all_have_one_shape(tmp_path, monkeypatch):
    """Each new batch shape would recompile the likelihood."""
    shapes = set()
    sampler, run_kwargs, _ = NautilusSampler.prepare(
        build_context(tmp_path), vectorized=True, **FAST_SETTINGS
    )
    pytest.importorskip("nautilus")
    evaluate = sampler.log_likelihood_func

    def recording_likelihood(points):
        shapes.add(points.shape)
        return evaluate(points)

    monkeypatch.setattr(sampler, "log_likelihood_func", recording_likelihood)
    sampler.run_sampler(**run_kwargs)

    assert shapes == {(100, 2)}


def test_spawn_pool_gives_the_same_answer(tmp_path):
    context = build_context(tmp_path)
    expected = log_evidence_by_quadrature(context)

    _, results, summary_parameters = _run(context, pool_njobs=2)

    assert summary_parameters["pool_njobs"] == 2
    assert results["logz"] == pytest.approx(expected, abs=FAST_LOGZ_TOLERANCE)


def test_sampler_pool_serves_a_run_without_a_likelihood_pool(tmp_path):
    context = build_context(tmp_path)
    expected = log_evidence_by_quadrature(context)

    _, results, summary_parameters = _run(context, vectorized=True, sampler_pool_njobs=2)

    assert summary_parameters["sampler_pool_njobs"] == 2
    assert results["logz"] == pytest.approx(expected, abs=FAST_LOGZ_TOLERANCE)


def test_n_batch_is_raised_to_a_multiple_of_the_pool(tmp_path):
    with pytest.warns(RuntimeWarning, match="Raising n_batch from 101 to 102"):
        _, results, _ = _run(build_context(tmp_path), pool_njobs=2, n_batch=101)
    assert results["n_like"] % 102 == 0


def test_invalid_points_are_handled_as_zero_likelihood(tmp_path):
    context = build_context(tmp_path, model_function=nautilus_helpers.walled_linear_model)
    expected = log_evidence_by_quadrature(context)

    with warnings.catch_warnings():
        # Nautilus would overflow on pRT's finite marker for invalid points.
        warnings.filterwarnings("error", message=".*overflow.*", category=RuntimeWarning)
        _, results, _ = _run(context, n_live=1000, n_eff=4000)

    assert results["logz"] == pytest.approx(expected, abs=0.15)
    assert np.max(results["samples"][:, 0]) <= nautilus_helpers.TRUE_ALPHA
    assert np.isfinite(results["log_likelihood"]).all()
    assert np.isfinite(results["weighted_samples"]["logl"]).all()


def test_a_retrieval_without_valid_points_stops_with_an_error(tmp_path):
    context = build_context(tmp_path, model_function=nautilus_helpers.invalid_model)
    with pytest.raises(RuntimeError, match="were invalid"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _run(context, max_initial_invalid_calls=500)


# --------------------------------------------------------------------------- #
# Stopping early, and resuming
# --------------------------------------------------------------------------- #
def test_stopping_during_exploration_saves_nothing_and_can_be_resumed(tmp_path):
    context = build_context(tmp_path)

    with pytest.warns(RuntimeWarning, match="before finishing its exploration phase"):
        sampler, results, _ = _run(context, n_like_max=400)

    assert results is None
    assert not sampler.sampler.explored
    assert not (tmp_path / "out_Nautilus" / "nautilus_samples.npz").exists()
    assert (tmp_path / "out_Nautilus" / "nautilus_checkpoint.hdf5").exists()

    resumed, results, _ = _run(build_context(tmp_path))

    assert resumed.sampler.explored
    assert results["converged"] is True
    assert (tmp_path / "out_Nautilus" / "nautilus_samples.npz").exists()


def test_stopping_after_exploration_is_saved_and_marked(tmp_path):
    with pytest.warns(RuntimeWarning, match="marked as not converged"):
        sampler, results, _ = _run(build_context(tmp_path), n_eff=10**9, n_like_max=30000)

    assert sampler.sampler.explored
    assert results["converged"] is False
    assert NautilusSampler(None, None, str(tmp_path), "nautilus").load_results()["converged"] is False


def test_a_finished_run_resumes_without_new_likelihood_calls(tmp_path):
    _, first, _ = _run(build_context(tmp_path))
    _, second, _ = _run(build_context(tmp_path))
    assert second["n_like"] == first["n_like"]
    assert second["logz"] == first["logz"]


def test_tightening_the_stopping_criterion_extends_the_run(tmp_path):
    _, first, _ = _run(build_context(tmp_path))
    _, second, _ = _run(build_context(tmp_path), n_eff=3 * FAST_SETTINGS["n_eff"])
    assert second["n_like"] > first["n_like"]
    assert second["n_eff"] >= 3 * FAST_SETTINGS["n_eff"]


def test_resuming_applies_the_requested_discard_exploration(tmp_path):
    _, first, _ = _run(build_context(tmp_path))
    assert first["discard_exploration"] is True

    sampler, second, _ = _run(build_context(tmp_path), discard_exploration=False)

    assert second["discard_exploration"] is False
    assert sampler.sampler.discard_exploration is False
    assert second["weighted_samples"]["points"].shape[0] >= first["weighted_samples"]["points"].shape[0]


@pytest.mark.parametrize(
    "changed_context, changed_kwargs, entry",
    [
        ({"observed_offset": 0.5}, {}, "reference_log_likelihood"),
        ({"model_function": nautilus_helpers.walled_linear_model}, {}, "reference_log_likelihood"),
        ({"n_extra_parameters": 1}, {}, "parameter_names"),
        ({}, {"n_live": 400}, "settings"),
    ],
)
def test_a_checkpoint_of_a_different_problem_is_refused(
    tmp_path, changed_context, changed_kwargs, entry
):
    with pytest.warns(RuntimeWarning, match="before finishing"):
        _run(build_context(tmp_path), n_like_max=400)

    with pytest.raises(ValueError, match=f"different problem.*{entry}"):
        _run(build_context(tmp_path, **changed_context), **changed_kwargs)


def test_resume_false_replaces_a_checkpoint_of_a_different_problem(tmp_path):
    with pytest.warns(RuntimeWarning, match="before finishing"):
        _run(build_context(tmp_path), n_like_max=400)

    context = build_context(tmp_path, observed_offset=0.5)
    _, results, _ = _run(context, resume=False)

    assert results["logz"] == pytest.approx(
        log_evidence_by_quadrature(context), abs=FAST_LOGZ_TOLERANCE
    )
    # The replacement is resumable in its turn.
    _, results_again, _ = _run(build_context(tmp_path, observed_offset=0.5))
    assert results_again["n_like"] == results["n_like"]


def test_a_checkpoint_can_be_turned_off(tmp_path):
    _, results, _ = _run(build_context(tmp_path), filepath=None)
    assert results["converged"] is True
    assert not list((tmp_path / "out_Nautilus").glob("*checkpoint*"))


# --------------------------------------------------------------------------- #
# Through Retrieval
# --------------------------------------------------------------------------- #
def _build_retrieval(tmp_path, context, run_mode="retrieval"):
    retrieval = object.__new__(Retrieval)
    retrieval.comm = None
    retrieval.rank = 0
    retrieval.configuration = context.configuration
    retrieval.configuration.run_mode = run_mode
    retrieval.parameter_layout = context.parameter_layout
    retrieval.runtime = context.runtime
    retrieval._free_parameter_names_cache = list(context.free_parameter_names)
    retrieval._n_free_params_total_retrieval = context.n_free_parameters
    retrieval.output_directory = str(tmp_path)
    retrieval.uncertainties_mode = "default"
    retrieval.print_log_likelihood_for_debugging = False
    retrieval.sampler = None
    retrieval.latest_sampler_results = None
    retrieval.analyzer = None
    retrieval.plotter = SimpleNamespace(
        get_reference_data=lambda: context.configuration.data["simple"]
    )
    return retrieval


def test_retrieval_runs_saves_and_reloads(tmp_path):
    pytest.importorskip("nautilus")
    context = build_context(tmp_path)
    retrieval = _build_retrieval(tmp_path, context)

    results = retrieval.run(error_checking=False, **FAST_SETTINGS)

    archive_path = tmp_path / "out_samples" / "nautilus_samples.npz"
    assert archive_path.exists()
    with np.load(archive_path) as archive:
        assert np.array_equal(archive["sample_matrix"], results["samples"])
        assert np.array_equal(archive["log_likelihood"], results["log_likelihood"])

    summary_file = tmp_path / "evaluate_nautilus" / "nautilus_ret_summary.txt"
    summary = summary_file.read_text()
    assert "NautilusSampler" in summary
    assert "discard_exploration = True" in summary

    # get_evidence() reports log10(Z).
    expected_evidence = (results["logz"] / np.log(10), results["logzerr"] / np.log(10))
    assert retrieval.get_evidence() == expected_evidence

    evaluating = _build_retrieval(tmp_path, build_context(tmp_path), run_mode="evaluate")
    reloaded = evaluating.run()
    assert np.array_equal(reloaded["samples"], results["samples"])
    assert evaluating.get_evidence() == expected_evidence


def test_retrieval_survives_a_run_stopped_during_exploration(tmp_path):
    pytest.importorskip("nautilus")
    retrieval = _build_retrieval(tmp_path, build_context(tmp_path))

    with pytest.warns(RuntimeWarning, match="before finishing its exploration phase"):
        results = retrieval.run(error_checking=False, **{**FAST_SETTINGS, "n_like_max": 400})

    assert results is None
    assert not (tmp_path / "out_samples" / "nautilus_samples.npz").exists()
    assert (tmp_path / "evaluate_nautilus" / "nautilus_ret_summary.txt").exists()


# --------------------------------------------------------------------------- #
# MPI
# --------------------------------------------------------------------------- #
def test_mpi_launch_runs_every_rank_through_the_lifecycle(tmp_path):
    pytest.importorskip("nautilus")
    pytest.importorskip("mpi4py")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec is not available")

    environment = dict(os.environ)
    environment.update(
        {
            "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
            "OMPI_MCA_rmaps_base_oversubscribe": "1",
            "PYTHONPATH": str(pathlib.Path(__file__).parent.parent),
        }
    )
    completed = subprocess.run(
        [mpiexec, "-n", "3", sys.executable, nautilus_helpers.__file__, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("NAUTILUS_MPI_RESULT ")
    ]
    assert len(result_lines) == 1, completed.stdout[-2000:]
    result = json.loads(result_lines[0].split(" ", 1)[1])

    context = build_context(tmp_path, retrieval_name="nautilus_mpi")
    assert result["mpi_workers"] == 2
    assert result["logz"] == pytest.approx(log_evidence_by_quadrature(context), abs=FAST_LOGZ_TOLERANCE)
    assert (tmp_path / "out_Nautilus" / "nautilus_mpi_samples.npz").exists()
