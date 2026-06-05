from types import SimpleNamespace

import numpy as np

from petitRADTRANS.retrieval import sampler as sampler_module
from petitRADTRANS.retrieval.sampler import DynestySampler


class _FakeDynamicNestedSampler:
    def __init__(self, log_likelihood_func, prior_func, **kwargs):
        del log_likelihood_func, prior_func, kwargs
        self.results = SimpleNamespace(
            niter=3,
            ncall=np.array([2, 3, 5], dtype=np.int64),
            eff=17.5,
            logz=np.array([-2.0, -1.5, -1.25]),
            logzerr=np.array([0.3, 0.2, 0.1]),
        )
        self.n_effective = 42.0

    def run_nested(self, **kwargs):
        self.run_kwargs = kwargs


def test_dynesty_sampler_sums_vector_ncall_in_run_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(sampler_module, "_require_dynesty", lambda: None)
    monkeypatch.setattr(
        sampler_module,
        "DynamicNestedSampler",
        _FakeDynamicNestedSampler,
    )
    monkeypatch.setattr(
        DynestySampler,
        "_build_equal_weight_posterior",
        lambda self, raw_results, seed: {
            "samples": np.array([[1.0], [2.0]], dtype=float),
            "log_likelihood": np.array([-1.0, -0.5], dtype=float),
            "log_weights": np.array([-np.log(2.0), -np.log(2.0)], dtype=float),
            "weighted_samples": {
                "points": np.array([[1.0], [2.0]], dtype=float),
                "logl": np.array([-1.0, -0.5], dtype=float),
                "log_weights": np.array([-np.log(2.0), -np.log(2.0)], dtype=float),
            },
        },
    )
    monkeypatch.setattr(
        DynestySampler,
        "_save_results",
        lambda self, **kwargs: (
            str(tmp_path / "samples.npz"),
            str(tmp_path / "metadata.json"),
        ),
    )
    monkeypatch.setattr(DynestySampler, "pretty_print_results", lambda self: None)
    monkeypatch.setattr(
        DynestySampler,
        "plot_diagnostics",
        lambda self, results=None: {},
    )

    sampler = DynestySampler(
        log_likelihood_func=lambda values: -0.5,
        prior_func=lambda cube: cube,
        output_directory=str(tmp_path),
        retrieval_name="dynesty-test",
    )
    sampler.parameter_names = ["theta"]

    results = sampler.run_sampler(init_kwargs={}, print_progress=False)

    assert isinstance(sampler.raw_results.ncall, np.ndarray)
    assert sampler.raw_results.ncall.shape == (3,)
    assert sampler.raw_results.ncall.dtype == np.int64
    assert results["ncall"] == 10
