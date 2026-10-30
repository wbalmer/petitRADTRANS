"""Tests of UltraNest's vectorized path, whose batches change size on every call."""

import numpy as np
import pytest

from tests.nautilus_helpers import build_context, log_evidence_by_quadrature

from petitRADTRANS.retrieval.samplers.interfaces import (  # noqa: E402
    pad_batches_to_buckets,
)
from petitRADTRANS.retrieval.samplers.ultranest import UltranestSampler  # noqa: E402


@pytest.mark.parametrize(
    "n_points, expected_bucket",
    [(1, 8), (3, 8), (8, 8), (9, 16), (16, 16), (17, 32), (100, 128), (128, 128), (129, 256)],
)
def test_batches_are_padded_to_the_next_bucket(n_points, expected_bucket):
    shapes = []

    def batched(points):
        shapes.append(points.shape)
        return np.sum(points, axis=1)

    points = np.random.default_rng(0).random((n_points, 3))
    results = pad_batches_to_buckets(batched)(points)

    assert shapes == [(expected_bucket, 3)]
    assert np.array_equal(results, np.sum(points, axis=1))


def test_padding_preserves_the_shape_of_every_output():
    prior_like = pad_batches_to_buckets(lambda points: 2.0 * points)

    assert prior_like(np.ones((3, 2))).shape == (3, 2)
    assert prior_like(np.ones((0, 2))).shape == (0, 2)
    assert np.array_equal(prior_like(np.array([1.0, 2.0])), np.array([2.0, 4.0]))


def _prepare_vectorized(tmp_path):
    pytest.importorskip("ultranest")
    context = build_context(tmp_path)
    sampler, _, summary_parameters = UltranestSampler.prepare(context, vectorized=True)
    return context, sampler, summary_parameters


def test_vectorized_likelihood_is_only_traced_once_per_bucket(tmp_path, monkeypatch):
    pytest.importorskip("ultranest")
    context = build_context(tmp_path)
    traces = []
    evaluate = context.runtime.evaluate_likelihood_jax_batched

    def counting_evaluate(*args, **kwargs):
        # Runs on every call if eager, but only when tracing if jitted.
        traces.append(kwargs["free_parameter_values"].shape)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(
        type(context.runtime), "evaluate_likelihood_jax_batched",
        lambda self, *args, **kwargs: counting_evaluate(*args, **kwargs),
    )
    sampler, _, summary_parameters = UltranestSampler.prepare(context, vectorized=True)

    rng = np.random.default_rng(0)
    for n_points in (3, 5, 7, 8, 3, 9, 12, 16, 9):
        points = np.column_stack((rng.random(n_points), 1.0 + rng.random(n_points)))
        assert sampler.log_likelihood_func(points).shape == (n_points,)

    assert traces == [(8, 2), (16, 2)]
    assert summary_parameters["use_jit"] is True


def test_vectorized_callables_match_the_eager_path(tmp_path):
    context, sampler, _ = _prepare_vectorized(tmp_path)
    cube = np.random.default_rng(1).random((11, 2))

    points = sampler.prior_func(cube)
    assert np.allclose(points, context.parameter_layout.prior_transform_numpy(cube))

    assert np.allclose(
        sampler.log_likelihood_func(points),
        np.asarray(context.runtime.evaluate_likelihood_jax_batched(free_parameter_values=points)),
    )


def test_vectorized_run_gives_the_right_evidence(tmp_path):
    """A brief run against the installed library."""
    pytest.importorskip("ultranest")
    context = build_context(tmp_path, retrieval_name="ultranest_vectorized")
    expected = log_evidence_by_quadrature(context)

    sampler, run_kwargs, summary_parameters = UltranestSampler.prepare(
        context,
        vectorized=True,
        min_num_live_points=200,
        max_num_improvement_loops=0,
        show_status=False,
        viz_callback=False,
        resume="overwrite",
    )
    results = sampler.run_sampler(**run_kwargs)

    assert summary_parameters["vectorized"] is True
    assert results["logz"] == pytest.approx(expected, abs=max(0.3, 3 * results["logzerr"]))
    assert results["samples"].shape[1] == 2
    assert results["log_likelihood"].shape[0] == results["samples"].shape[0]
