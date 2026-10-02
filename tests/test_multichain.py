"""Unit tests for the native multi-chain primitives and MCMC combiner.

These are pure-NumPy tests: they synthesise per-chain ``*_samples.npz`` archives
on disk and exercise the combine/diagnostics path without running a sampler.
"""

import json
import os

import numpy as np
import pytest

from petitRADTRANS.retrieval import multichain as mc


# ---------------------------------------------------------------------------
# Phase 0 primitives
# ---------------------------------------------------------------------------


def test_derive_chain_seed_is_deterministic_offset():
    assert mc.derive_chain_seed(100, 0) == 100
    assert mc.derive_chain_seed(100, 7) == 107
    # Same (base, id) always yields the same seed -> in-process == out-of-process.
    assert mc.derive_chain_seed(42, 3) == mc.derive_chain_seed(42, 3)


def test_chain_and_combined_names():
    assert mc.chain_retrieval_name("run", 2) == "run_chain2"
    assert mc.combined_retrieval_name("run") == "run_combined"


def test_backend_output_subdir_and_family():
    assert mc.backend_output_subdir("numpyro_nuts") == "out_NumPyro"
    assert mc.backend_output_subdir("numpyronuts") == "out_NumPyro"
    assert mc.backend_output_subdir("blackjaxnuts") == "out_BlackJAX"
    assert mc.backend_output_subdir("jaxns") == "out_JAXNS"
    assert mc.sampler_family("numpyronuts") == "mcmc"
    assert mc.sampler_family("blackjaxhmc") == "mcmc"
    assert mc.sampler_family("jaxns") == "nested"
    assert mc.sampler_family("pymultinest") == "other"
    with pytest.raises(ValueError):
        mc.backend_output_subdir("not_a_sampler")


def test_chainspec_build():
    spec = mc.ChainSpec.build("run", base_seed=1000, chain_id=4)
    assert spec.chain_id == 4
    assert spec.seed == 1004
    assert spec.retrieval_name == "run_chain4"


def test_chain_execution_validate():
    assert mc.ChainExecution.validate("processes") == "processes"
    with pytest.raises(ValueError):
        mc.ChainExecution.validate("nonsense")


# ---------------------------------------------------------------------------
# Helpers to synthesise per-chain archives
# ---------------------------------------------------------------------------


def _write_chain(directory, base_name, chain_id, samples, names, *,
                 unconstrained=None, divergent=None):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "samples": samples,
        "unconstrained_samples": samples if unconstrained is None else unconstrained,
        "logdensity": -0.5 * (samples ** 2).sum(axis=1),
        "model_log_likelihood": -0.5 * (samples ** 2).sum(axis=1),
    }
    if divergent is not None:
        arrays["is_divergent"] = divergent
    name = mc.chain_retrieval_name(base_name, chain_id)
    np.savez(directory / f"{name}_samples.npz", **arrays)
    with open(directory / f"{name}_params.json", "w", encoding="utf-8") as handle:
        json.dump(list(names), handle)


# ---------------------------------------------------------------------------
# Phase 1 combiner + diagnostics
# ---------------------------------------------------------------------------


def test_combine_well_mixed_chains_converges(tmp_path):
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    n_chains, n_draws, n_dim = 4, 2000, 3
    sub = tmp_path / "out_NumPyro"
    for c in range(n_chains):
        x = rng.standard_normal((n_draws, n_dim))
        divergent = rng.random(n_draws) < 0.01
        _write_chain(sub, "run", c, x, names, divergent=divergent)

    summary = mc.combine_mcmc_chains(tmp_path, "numpyro_nuts", "run", num_chains=n_chains)
    diag = summary.diagnostics

    assert summary.combined_name == "run_combined"
    assert summary.missing_chain_ids == []
    assert diag["n_samples_total"] == n_chains * n_draws
    assert diag["rhat_available"] is True
    # Independent draws from one distribution -> R-hat ~ 1, converged.
    assert diag["max_rhat"] < 1.01
    assert diag["converged"] is True
    assert diag["divergences"]["total"] == sum(diag["divergences"]["per_chain"])

    # Combined archive matches the on-disk format get_samples reads.
    with np.load(summary.samples_file) as combined:
        assert combined["samples"].shape == (n_chains * n_draws, n_dim)
        assert "model_log_likelihood" in combined.files
        assert combined["chain_index"].shape == (n_chains * n_draws,)
        assert set(np.unique(combined["chain_index"])) == set(range(n_chains))

    with open(summary.params_file, encoding="utf-8") as handle:
        assert json.load(handle) == names


def test_combine_offset_chains_flags_nonconvergence(tmp_path):
    rng = np.random.default_rng(0)
    names = ["a", "b"]
    n_chains, n_draws, n_dim = 4, 1500, 2
    chains = []
    for c in range(n_chains):
        x = rng.standard_normal((n_draws, n_dim)) + c * 3.0  # large per-chain offset
        chains.append(mc.MCMCChainArrays(chain_id=c, samples=x, unconstrained_samples=x))
    combined = mc.MCMCChainCombiner().combine(chains, names)
    assert combined.diagnostics["max_rhat"] > 1.1
    assert combined.diagnostics["converged"] is False


def test_missing_chain_warns_and_combines_survivors(tmp_path):
    rng = np.random.default_rng(2)
    names = ["a", "b"]
    n_chains, n_draws, n_dim = 4, 800, 2
    sub = tmp_path / "out_NumPyro"
    for c in range(n_chains):
        if c == 1:
            continue  # chain 1 "failed"
        _write_chain(sub, "run", c, rng.standard_normal((n_draws, n_dim)), names)

    with pytest.warns(RuntimeWarning, match="Missing chain"):
        summary = mc.combine_mcmc_chains(
            tmp_path, "numpyro_nuts", "run", num_chains=n_chains, on_missing="warn"
        )
    assert summary.missing_chain_ids == [1]
    assert summary.chain_ids == [0, 2, 3]
    assert summary.diagnostics["n_samples_total"] == 3 * n_draws


def test_missing_chain_raises_when_requested(tmp_path):
    rng = np.random.default_rng(3)
    names = ["a"]
    sub = tmp_path / "out_NumPyro"
    _write_chain(sub, "run", 0, rng.standard_normal((100, 1)), names)
    with pytest.raises(FileNotFoundError):
        mc.combine_mcmc_chains(
            tmp_path, "numpyro_nuts", "run", num_chains=3, on_missing="raise"
        )


def test_discovery_without_num_chains(tmp_path):
    rng = np.random.default_rng(4)
    names = ["a", "b"]
    sub = tmp_path / "out_BlackJAX"
    for c in (0, 1, 2):
        _write_chain(sub, "run", c, rng.standard_normal((500, 2)), names)
    summary = mc.combine_mcmc_chains(tmp_path, "blackjaxnuts", "run")  # discover
    assert summary.chain_ids == [0, 1, 2]
    assert summary.diagnostics["n_chains"] == 3


def test_parameter_mismatch_raises(tmp_path):
    rng = np.random.default_rng(5)
    sub = tmp_path / "out_NumPyro"
    _write_chain(sub, "run", 0, rng.standard_normal((100, 2)), ["a", "b"])
    _write_chain(sub, "run", 1, rng.standard_normal((100, 2)), ["a", "c"])
    with pytest.raises(ValueError, match="Parameter mismatch"):
        mc.combine_mcmc_chains(tmp_path, "numpyro_nuts", "run", num_chains=2)


# ---------------------------------------------------------------------------
# In-process (vectorised/sharded) pooled-archive combine
# ---------------------------------------------------------------------------


def _write_pooled(directory, base_name, per_chain_samples, names):
    """Write one pooled ``*_samples.npz`` with contiguous per-chain blocks."""
    directory.mkdir(parents=True, exist_ok=True)
    pooled = np.concatenate(per_chain_samples, axis=0)
    np.savez(
        directory / f"{base_name}_samples.npz",
        samples=pooled,
        unconstrained_samples=pooled,
        logdensity=-0.5 * (pooled ** 2).sum(axis=1),
        is_divergent=np.zeros(pooled.shape[0], dtype=bool),
    )
    with open(directory / f"{base_name}_params.json", "w", encoding="utf-8") as handle:
        json.dump(list(names), handle)


def test_split_pooled_mcmc_chains_recovers_blocks(tmp_path):
    n_chains, n_per, n_dim = 4, 50, 3
    per_chain = [np.full((n_per, n_dim), c, dtype=float) for c in range(n_chains)]
    sub = tmp_path / "out_NumPyro"
    _write_pooled(sub, "run", per_chain, ["a", "b", "c"])

    chains = mc.split_pooled_mcmc_chains(sub / "run_samples.npz", n_chains)
    assert len(chains) == n_chains
    for c, chain in enumerate(chains):
        assert chain.chain_id == c
        assert chain.samples.shape == (n_per, n_dim)
        np.testing.assert_array_equal(chain.samples, per_chain[c])


def test_split_pooled_rejects_indivisible(tmp_path):
    sub = tmp_path / "out_NumPyro"
    _write_pooled(sub, "run", [np.zeros((50, 2)), np.zeros((50, 2))], ["a", "b"])
    with pytest.raises(ValueError, match="equal chains"):
        mc.split_pooled_mcmc_chains(sub / "run_samples.npz", 3)


def test_combine_in_process_well_mixed_converges(tmp_path):
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    n_chains, n_per, n_dim = 4, 2000, 3
    per_chain = [rng.standard_normal((n_per, n_dim)) for _ in range(n_chains)]
    sub = tmp_path / "out_NumPyro"
    _write_pooled(sub, "run", per_chain, names)

    summary = mc.combine_in_process_mcmc_chains(
        tmp_path, "numpyro_nuts", "run", n_chains
    )
    diag = summary.diagnostics

    assert summary.combined_name == "run_combined"
    assert summary.chain_ids == [0, 1, 2, 3]
    assert diag["n_chains"] == n_chains
    assert diag["n_samples_total"] == n_chains * n_per
    assert diag["rhat_available"] is True
    # Independent draws from one distribution -> R-hat ~ 1, converged.
    assert diag["max_rhat"] < 1.01
    assert diag["converged"] is True

    # Combined archive matches the on-disk contract of combine_mcmc_chains.
    with np.load(summary.samples_file) as combined:
        assert combined["samples"].shape == (n_chains * n_per, n_dim)
        assert combined["chain_index"].shape == (n_chains * n_per,)
        assert set(np.unique(combined["chain_index"])) == set(range(n_chains))


def test_combine_rejects_nested_sampler(tmp_path):
    with pytest.raises(ValueError, match="only supports MCMC"):
        mc.combine_mcmc_chains(tmp_path, "jaxns", "run", num_chains=2)


def test_diagnostics_identical_chains_rhat_near_one():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1000, 2))
    grouped = np.stack([x, x, x], axis=0)  # identical chains
    diag = mc.compute_mcmc_diagnostics(grouped, ["a", "b"])
    # Identical chains -> R-hat ~ 1. It is not *exactly* 1 because split-R-hat
    # compares the two halves of each chain, whose means differ slightly.
    assert diag["max_rhat"] == pytest.approx(1.0, abs=5e-3)
    assert diag["converged"] is True


def test_combine_drops_optional_field_absent_in_one_chain(tmp_path):
    # chain 0 has divergences, chain 1 does not -> combined omits is_divergent.
    rng = np.random.default_rng(7)
    names = ["a", "b"]
    sub = tmp_path / "out_NumPyro"
    _write_chain(sub, "run", 0, rng.standard_normal((300, 2)), names,
                 divergent=np.zeros(300, bool))
    _write_chain(sub, "run", 1, rng.standard_normal((300, 2)), names)  # no divergences
    summary = mc.combine_mcmc_chains(tmp_path, "numpyro_nuts", "run", num_chains=2)
    with np.load(summary.samples_file) as combined:
        assert "is_divergent" not in combined.files
        assert combined["samples"].shape == (600, 2)
    assert "divergences" not in summary.diagnostics


# ---------------------------------------------------------------------------
# Phase 2 -- orchestration helpers
# ---------------------------------------------------------------------------

import concurrent.futures
import functools

from tests.multichain_spawn_helpers import write_synthetic_chain


def test_partition_cpus():
    assert mc.partition_cpus([0, 1, 2, 3, 4, 5, 6, 7], 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    # remainder spread over the first groups
    assert mc.partition_cpus([10, 11, 12, 13, 14], 2) == [[10, 11, 12], [13, 14]]
    # fewer cpus than groups -> no pinning
    assert mc.partition_cpus([0, 1], 4) == [None, None, None, None]
    with pytest.raises(ValueError):
        mc.partition_cpus([0, 1], 0)


def test_resolve_chain_execution_policy():
    R = mc.resolve_chain_execution
    # explicit pass-through
    assert R("processes", num_chains=4) == "processes"
    assert R("sequential", num_chains=4) == "sequential"
    # auto
    assert R("auto", num_chains=1) == "sequential"
    assert R("auto", num_chains=4, gpu_available=False) == "processes"
    assert R("auto", num_chains=2, gpu_available=True, num_devices=4) == "sharded"
    assert R("auto", num_chains=8, gpu_available=True, num_devices=4) == "vectorised"
    with pytest.raises(ValueError):
        R("bogus", num_chains=2)


def test_estimate_peak_memory():
    assert mc.estimate_peak_memory_gb("sequential", 8, 18) == 18
    assert mc.estimate_peak_memory_gb("processes", 8, 18, max_workers=4) == 72
    assert mc.estimate_peak_memory_gb("processes", 8, 18) == 8 * 18
    assert mc.estimate_peak_memory_gb("vectorised", 8, 18) == 144
    assert mc.estimate_peak_memory_gb("processes", 8, None) is None


def _thread_executor_factory(workers):
    return concurrent.futures.ThreadPoolExecutor(max_workers=workers)


def _ok_runner(chain_id, cores):
    return chain_id


def _flaky_runner(chain_id, cores):
    if chain_id == 2:
        raise ValueError("boom")
    return chain_id


def test_run_chains_in_processes_collects_failures():
    result = mc.run_chains_in_processes(
        _flaky_runner, [0, 1, 2, 3], [None] * 4,
        on_chain_failure="warn", block_mpi=False,
        executor_factory=_thread_executor_factory,
    )
    assert result.succeeded_chain_ids == [0, 1, 3]
    assert set(result.failed) == {2}
    assert "boom" in result.failed[2]
    assert result.all_succeeded is False


def test_run_chains_in_processes_raise_mode():
    with pytest.raises(RuntimeError, match="chain\\(s\\) failed"):
        mc.run_chains_in_processes(
            _flaky_runner, [0, 1, 2], [None] * 3,
            on_chain_failure="raise", block_mpi=False,
            executor_factory=_thread_executor_factory,
        )


def test_processes_then_combine_pipeline(tmp_path):
    # Runner writes synthetic per-chain archives; then combine them. Uses a thread
    # executor to exercise the orchestration + combine wiring deterministically.
    names = ["a", "b"]
    sub = tmp_path / "out_NumPyro"
    sub.mkdir()
    runner = functools.partial(write_synthetic_chain, str(sub), names, 500, "run", 1000)
    result = mc.run_chains_in_processes(
        runner, [0, 1, 2, 3], [None] * 4, block_mpi=False,
        executor_factory=_thread_executor_factory,
    )
    assert result.succeeded_chain_ids == [0, 1, 2, 3]

    summary = mc.combine_mcmc_chains(tmp_path, "numpyro_nuts", "run", num_chains=4)
    assert summary.diagnostics["n_samples_total"] == 4 * 500
    assert summary.diagnostics["rhat_available"] is True


def test_run_chains_in_processes_real_spawn(tmp_path):
    # End-to-end with the real spawn start method: confirms affinity + mpi-block
    # actually run worker-side. Skips gracefully if the environment cannot spawn.
    out = tmp_path / "out_NumPyro"
    out.mkdir()
    names = ["a", "b"]
    runner = functools.partial(write_synthetic_chain, str(out), names, 200, "run", 7)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    groups = mc.partition_cpus(cpus[: max(1, min(4, len(cpus)))], 4)
    try:
        result = mc.run_chains_in_processes(
            runner, [0, 1, 2, 3], groups, block_mpi=True,
        )
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"spawn not usable in this environment: {error}")

    assert result.succeeded_chain_ids == [0, 1, 2, 3]
    for chain_id in range(4):
        meta = json.loads((out / f"run_chain{chain_id}_worker.json").read_text())
        assert meta["mpi4py_blocked"] is True
        if meta["affinity"] is not None and groups[chain_id] is not None:
            assert meta["affinity"] == groups[chain_id]

    summary = mc.combine_mcmc_chains(tmp_path, "numpyro_nuts", "run", num_chains=4)
    assert summary.diagnostics["n_samples_total"] == 4 * 200


# ---------------------------------------------------------------------------
# Phase 3 -- nested (JAXNS) evidence-aware combiner
# ---------------------------------------------------------------------------


def _write_nested_run(directory, base_name, chain_id, samples, names, log_z, log_z_err):
    directory.mkdir(parents=True, exist_ok=True)
    name = mc.chain_retrieval_name(base_name, chain_id)
    log_likelihood = -0.5 * (samples ** 2).sum(axis=1)
    np.savez(
        directory / f"{name}_samples.npz",
        parameter_names=np.asarray(names, dtype=str),
        sample_matrix=samples,
        log_likelihood=log_likelihood,
    )
    with open(directory / f"{name}_params.json", "w", encoding="utf-8") as handle:
        json.dump(list(names), handle)
    if log_z is not None:
        evidence = {"log_Z_mean": log_z, "ESS": float(samples.shape[0]),
                    "total_num_likelihood_evaluations": 1e5, "n_samples": samples.shape[0]}
        if log_z_err is not None:
            evidence["log_Z_uncert"] = log_z_err
        with open(directory / f"{name}_evidence.json", "w", encoding="utf-8") as handle:
            json.dump(evidence, handle)


def test_combine_evidence_inverse_variance_and_scatter():
    ev = mc.combine_evidence([-10.0, -10.2, -9.8, -10.1], [0.1, 0.1, 0.1, 0.1])
    assert ev["available"] is True and ev["n_runs"] == 4
    assert ev["used_inverse_variance"] is True
    assert ev["log_Z_mean"] == pytest.approx(-10.025, abs=1e-6)  # equal weights -> mean
    # between-run scatter exceeds the formal stat error here
    assert ev["log_Z_uncert"] >= ev["log_Z_stat_uncert"]
    assert ev["log_Z_between_std"] > 0


def test_combine_evidence_unweighted_fallback():
    ev = mc.combine_evidence([-10.0, -10.4], [None, None])
    assert ev["used_inverse_variance"] is False
    assert ev["log_Z_mean"] == pytest.approx(-10.2)


def test_combine_evidence_empty():
    assert mc.combine_evidence([None], [None]) == {"available": False, "n_runs": 0}


def test_combine_nested_chains_pipeline(tmp_path):
    names = ["a", "b"]
    sub = tmp_path / "out_JAXNS"
    rng = np.random.default_rng(0)
    log_zs = [-10.0, -10.1, -9.9, -10.05]
    n_per = [300, 310, 320, 330]
    for c in range(4):
        _write_nested_run(sub, "run", c, rng.standard_normal((n_per[c], 2)), names,
                          log_zs[c], 0.1)
    summary = mc.combine_nested_chains(tmp_path, "jaxns", "run", num_chains=4, seed=7)

    assert summary.combined_name == "run_combined"
    assert summary.diagnostics["n_runs"] == 4
    assert summary.diagnostics["n_samples_total"] == sum(n_per)
    assert summary.evidence["available"] is True
    assert summary.evidence["log_Z_mean"] == pytest.approx(-10.0125, abs=1e-2)

    with np.load(summary.samples_file) as combined:
        assert combined["sample_matrix"].shape == (sum(n_per), 2)
        assert "log_likelihood" in combined.files
        assert [str(x) for x in combined["parameter_names"]] == names
        # best-fit preserved at index 0
        assert np.isfinite(combined["log_likelihood"][0])
    assert os.path.exists(summary.evidence_file)
    assert os.path.exists(summary.diagnostics_file)


def test_combine_nested_equal_evidence_resamples_from_pool(tmp_path):
    # Equal evidence -> uniform per-sample weights -> combined is a resample of the
    # pooled draws (every combined row appears in the pooled set).
    names = ["a"]
    sub = tmp_path / "out_JAXNS"
    rng = np.random.default_rng(1)
    pooled = []
    for c in range(3):
        s = rng.standard_normal((200, 1))
        pooled.append(s)
        _write_nested_run(sub, "run", c, s, names, -5.0, 0.1)
    summary = mc.combine_nested_chains(tmp_path, "jaxns", "run", num_chains=3, seed=3)
    pooled_set = {round(float(v), 10) for s in pooled for v in s[:, 0]}
    with np.load(summary.samples_file) as combined:
        drawn = {round(float(v), 10) for v in combined["sample_matrix"][:, 0]}
    assert drawn.issubset(pooled_set)


def test_combine_nested_missing_evidence_fallback(tmp_path):
    # No evidence sidecars -> uniform run weights, still combines (evidence n/a).
    names = ["a", "b"]
    sub = tmp_path / "out_JAXNS"
    rng = np.random.default_rng(2)
    for c in range(3):
        _write_nested_run(sub, "run", c, rng.standard_normal((150, 2)), names,
                          None, None)
    summary = mc.combine_nested_chains(tmp_path, "jaxns", "run", num_chains=3)
    assert summary.evidence["available"] is False
    assert summary.diagnostics["n_samples_total"] == 450


def test_combine_nested_rejects_mcmc(tmp_path):
    with pytest.raises(ValueError, match="only supports nested"):
        mc.combine_nested_chains(tmp_path, "numpyronuts", "run", num_chains=2)


def test_jaxns_evidence_sidecar_written(tmp_path):
    # Exercise JAXNSSampler._write_evidence_sidecar without constructing the full
    # sampler (no JAXNS import needed for this method).
    from petitRADTRANS.retrieval.samplers.jaxns import JAXNSSampler

    sampler = JAXNSSampler.__new__(JAXNSSampler)
    sampler.output_directory = str(tmp_path)
    sampler.retrieval_name = "run_chain0"
    (tmp_path / "out_JAXNS").mkdir(parents=True)

    sampler._write_evidence_sidecar({
        "samples": np.zeros((42, 2)),
        "log_Z_mean": -12.3,
        "log_Z_uncert": 0.2,
        "ESS": 100.0,
        "total_num_likelihood_evaluations": 5e5,
        "log_efficiency": -3.0,
    })
    path = tmp_path / "out_JAXNS" / "run_chain0_evidence.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["log_Z_mean"] == -12.3
    assert data["log_Z_uncert"] == 0.2
    assert data["n_samples"] == 42


# ---------------------------------------------------------------------------
# Phase 4 -- resume + SLURM-array launcher generator
# ---------------------------------------------------------------------------

import ast as _ast
from types import SimpleNamespace


def build_demo_config():
    """Module-level config builder for the launcher tests."""
    return SimpleNamespace(retrieval_name="demo_run", sampler_type="numpyronuts")


def test_chain_artifact_path_and_existing_ids(tmp_path):
    sub = tmp_path / "out_NumPyro"
    sub.mkdir()
    (sub / "run_chain0_samples.npz").touch()
    (sub / "run_chain2_samples.npz").touch()
    p = mc.chain_artifact_path(tmp_path, "numpyro_nuts", "run", 1)
    assert p.endswith(os.path.join("out_NumPyro", "run_chain1_samples.npz"))
    assert mc.existing_chain_ids(tmp_path, "numpyro_nuts", "run", 4) == [0, 2]


def test_render_chain_worker_script():
    worker = mc.render_chain_worker_script(
        "my_pkg.cfg", "build", "/out", num_chains=4, chain_seed=12345,
        retrieval_kwargs={"evaluate_sample_spectra": False},
        run_kwargs={"num_samples": 400}, resume=True,
    )
    assert "from my_pkg.cfg import build as config_builder" in worker
    assert "SLURM_ARRAY_TASK_ID" in worker
    assert "multichain.run_single_chain" in worker
    assert "multichain.combine_chains_from_builder" in worker
    assert 'sys.modules["mpi4py"] = None' in worker
    assert "NUM_CHAINS = 4" in worker and "CHAIN_SEED = 12345" in worker
    _ast.parse(worker)  # generated worker is valid Python


def test_render_chain_worker_rejects_main():
    with pytest.raises(ValueError, match="importable module"):
        mc.render_chain_worker_script("__main__", "b", "/o", 2, 1, None, None, True)


def test_render_slurm_array_scripts():
    scripts = mc.render_slurm_array_scripts(
        base_name="run", worker_filename="run_chain_worker.py", num_chains=8,
        python="python", cpus_per_task=8, partition="p.test", time="12:00:00",
        mem="200000", env_setup_lines=["module load anaconda/3", "conda activate jaxprt"],
    )
    assert set(scripts) == {"run_chains.sbatch", "run_combine.sbatch", "run_submit.sh"}
    arr, comb, sub = scripts["run_chains.sbatch"], scripts["run_combine.sbatch"], scripts["run_submit.sh"]
    assert "#SBATCH --array=0-7" in arr
    assert "srun python run_chain_worker.py run" in arr
    assert "--partition=p.test" in arr and "conda activate jaxprt" in arr
    assert "python run_chain_worker.py combine" in comb and "--array" not in comb
    assert "sbatch --parsable run_chains.sbatch" in sub
    assert "--dependency=afterany:$ARRAY_JOB run_combine.sbatch" in sub


def test_write_chain_launcher(tmp_path):
    from petitRADTRANS.retrieval.retrieval import Retrieval

    written = Retrieval.write_chain_launcher(
        build_demo_config, num_chains=8, output_directory="/scratch/out",
        script_dir=str(tmp_path), partition="p.vera", cpus_per_task=8,
        run_kwargs={"num_samples": 400, "num_warmup": 50},
        retrieval_kwargs={"evaluate_sample_spectra": False},
        env_setup_lines=["conda activate jaxprt"], verbose=False,
    )
    names = {os.path.basename(p) for p in written.values()}
    assert {"demo_run_chain_worker.py", "demo_run_chains.sbatch",
            "demo_run_combine.sbatch", "demo_run_submit.sh"} <= names
    worker = (tmp_path / "demo_run_chain_worker.py").read_text()
    assert "import build_demo_config as config_builder" in worker
    _ast.parse(worker)
    assert os.access(tmp_path / "demo_run_submit.sh", os.X_OK)


def test_run_single_chain_resume_skip(tmp_path):
    from petitRADTRANS.retrieval.retrieval import Retrieval

    (tmp_path / "out_NumPyro").mkdir()
    (tmp_path / "out_NumPyro" / "demo_run_chain3_samples.npz").touch()
    result = Retrieval.run_single_chain(
        build_demo_config, 3, output_directory=str(tmp_path), num_chains=4,
        resume=True, verbose=False,
    )
    assert result is None  # skipped, no Retrieval built


# ---------------------------------------------------------------------------
# Orchestration entry points (moved out of Retrieval) + the Retrieval shims
# ---------------------------------------------------------------------------


def build_nested_demo_config():
    """Module-level config builder naming a nested (JAXNS) sampler."""
    return SimpleNamespace(retrieval_name="demo_run", sampler_type="jaxns")


def _write_two_mcmc_chains(tmp_path, base_name="demo_run"):
    """Two synthetic NumPyro chains under out_NumPyro; returns the parameter names."""
    rng = np.random.default_rng(0)
    names = ["a", "b"]
    for chain_id in range(2):
        _write_chain(
            tmp_path / "out_NumPyro", base_name, chain_id,
            rng.normal(size=(64, 2)), names,
        )
    return names


def test_combine_chains_dispatches_mcmc(tmp_path):
    _write_two_mcmc_chains(tmp_path)
    summary = mc.combine_chains(
        tmp_path, "numpyronuts", "demo_run", num_chains=2, verbose=False
    )
    assert summary.combined_name == "demo_run_combined"
    assert summary.chain_ids == [0, 1]
    assert summary.diagnostics["n_samples_total"] == 128
    assert summary.diagnostics["rhat_available"] is True
    assert os.path.exists(summary.samples_file)
    assert os.path.exists(summary.diagnostics_file)


def test_combine_chains_dispatches_nested(tmp_path):
    base_dir = tmp_path / "out_JAXNS"
    base_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for chain_id in range(2):
        name = mc.chain_retrieval_name("demo_run", chain_id)
        np.savez(
            base_dir / f"{name}_samples.npz",
            sample_matrix=rng.normal(size=(32, 2)),
            log_likelihood=rng.normal(size=32),
            parameter_names=np.asarray(["a", "b"], dtype=str),
        )
        with open(base_dir / f"{name}_evidence.json", "w", encoding="utf-8") as handle:
            json.dump({"log_Z_mean": -10.0 + chain_id, "log_Z_uncert": 0.2}, handle)

    summary = mc.combine_chains(
        tmp_path, "jaxns", "demo_run", num_chains=2, verbose=False
    )
    assert summary.combined_name == "demo_run_combined"
    assert summary.evidence["available"] is True
    assert summary.diagnostics["n_runs"] == 2
    assert os.path.exists(summary.evidence_file)


def test_combine_chains_rejects_other_family(tmp_path):
    with pytest.raises(NotImplementedError, match="not supported"):
        mc.combine_chains(tmp_path, "pymultinest", "demo_run", verbose=False)


def test_combine_chains_reports_summary(tmp_path, capsys):
    _write_two_mcmc_chains(tmp_path)
    mc.combine_chains(tmp_path, "numpyronuts", "demo_run", num_chains=2, verbose=True)
    out = capsys.readouterr().out
    assert "Combined 2 chains" in out
    assert "max R-hat" in out


def test_combine_chains_from_builder_needs_no_retrieval(tmp_path):
    """The SLURM combine task must not construct a Retrieval.

    ``build_demo_config`` returns a SimpleNamespace, which a real Retrieval would
    reject outright -- so this passing is proof that only the config is built.
    """
    _write_two_mcmc_chains(tmp_path)
    summary = mc.combine_chains_from_builder(
        build_demo_config, output_directory=str(tmp_path), num_chains=2,
        verbose=False,
    )
    assert summary.combined_name == "demo_run_combined"
    assert summary.diagnostics["n_samples_total"] == 128


def test_module_run_single_chain_resume_skip(tmp_path):
    (tmp_path / "out_NumPyro").mkdir()
    (tmp_path / "out_NumPyro" / "demo_run_chain3_samples.npz").touch()
    assert mc.run_single_chain(
        build_demo_config, 3, output_directory=str(tmp_path), resume=True,
        verbose=False,
    ) is None
    # num_chains is accepted and ignored (the generated workers still pass it).
    assert mc.run_single_chain(
        build_demo_config, 3, output_directory=str(tmp_path), num_chains=4,
        resume=True, verbose=False,
    ) is None


def test_module_write_chain_launcher(tmp_path):
    written = mc.write_chain_launcher(
        build_demo_config, num_chains=8, output_directory="/scratch/out",
        script_dir=str(tmp_path), partition="p.vera", verbose=False,
    )
    names = {os.path.basename(p) for p in written.values()}
    assert {"demo_run_chain_worker.py", "demo_run_chains.sbatch",
            "demo_run_combine.sbatch", "demo_run_submit.sh"} <= names
    _ast.parse((tmp_path / "demo_run_chain_worker.py").read_text())


@pytest.mark.parametrize(
    "method_name, function_name, args",
    [
        ("run_multichain", "run_multichain", (build_demo_config,)),
        ("run_single_chain", "run_single_chain", (build_demo_config, 3)),
        ("combine_chains_from_builder", "combine_chains_from_builder", (build_demo_config,)),
        ("write_chain_launcher", "write_chain_launcher", (build_demo_config,)),
    ],
)
def test_retrieval_shims_delegate_to_multichain(
    monkeypatch, method_name, function_name, args
):
    """Each Retrieval classmethod forwards verbatim to its multichain function."""
    from petitRADTRANS.retrieval.retrieval import Retrieval

    recorded = {}

    def fake(*call_args, **call_kwargs):
        recorded["args"] = call_args
        recorded["kwargs"] = call_kwargs
        return "sentinel"

    monkeypatch.setattr(mc, function_name, fake)

    result = getattr(Retrieval, method_name)(*args, num_chains=4, verbose=False)
    assert result == "sentinel"
    assert recorded["args"] == args
    assert recorded["kwargs"] == {"num_chains": 4, "verbose": False}


class _FakeRetrieval:
    """Stand-in for Retrieval that writes a synthetic chain instead of sampling.

    Records the kwargs each chain was run with, so the seeding / ``num_chains=1``
    contract of ``_run_single_chain_on`` is checked as well as the combine.
    """

    calls = []

    def __init__(self, configuration, output_directory, **kwargs):
        self.configuration = configuration
        self.output_directory = output_directory

    def run(self, **run_kwargs):
        name = self.configuration.retrieval_name
        type(self).calls.append((name, dict(run_kwargs)))
        rng = np.random.default_rng(run_kwargs["seed"])
        directory = os.path.join(self.output_directory, "out_NumPyro")
        os.makedirs(directory, exist_ok=True)
        samples = rng.normal(size=(64, 2))
        np.savez(
            os.path.join(directory, f"{name}_samples.npz"),
            samples=samples,
            unconstrained_samples=samples,
            logdensity=-0.5 * (samples ** 2).sum(axis=1),
        )
        with open(os.path.join(directory, f"{name}_params.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(["a", "b"], handle)


def test_run_multichain_sequential_end_to_end(monkeypatch, tmp_path):
    """Sequential mode: run 3 chains, auto-combine, produce diagnostics."""
    import petitRADTRANS.retrieval.retrieval as retrieval_module

    _FakeRetrieval.calls = []
    monkeypatch.setattr(retrieval_module, "Retrieval", _FakeRetrieval)

    result = mc.run_multichain(
        build_demo_config,
        num_chains=3,
        output_directory=str(tmp_path),
        chain_execution="sequential",
        chain_seed=1000,
        verbose=False,
    )

    assert result["execution"] == "sequential"
    assert result["succeeded_chain_ids"] == [0, 1, 2]
    assert result["failed"] == {}
    assert result["combine"].combined_name == "demo_run_combined"
    assert result["diagnostics"]["n_samples_total"] == 192
    assert result["diagnostics"]["rhat_available"] is True

    # Per-chain namespacing, derived seeds, and num_chains=1 per worker.
    names = [name for name, _ in _FakeRetrieval.calls]
    assert names == ["demo_run_chain0", "demo_run_chain1", "demo_run_chain2"]
    seeds = [kwargs["seed"] for _, kwargs in _FakeRetrieval.calls]
    assert seeds == [mc.derive_chain_seed(1000, c) for c in range(3)]
    assert all(kwargs["num_chains"] == 1 for _, kwargs in _FakeRetrieval.calls)

    assert os.path.exists(tmp_path / "out_NumPyro" / "demo_run_combined_samples.npz")


def test_run_multichain_in_process_uses_the_requested_base_name(monkeypatch, tmp_path):
    """The vectorised path names its run (and combine) after base_retrieval_name,
    not after whatever name the builder happens to give the configuration."""
    import petitRADTRANS.retrieval.retrieval as retrieval_module

    _FakeRetrieval.calls = []
    monkeypatch.setattr(retrieval_module, "Retrieval", _FakeRetrieval)

    result = mc.run_multichain(
        build_demo_config,
        num_chains=2,
        output_directory=str(tmp_path),
        chain_execution="vectorised",
        chain_seed=1000,
        base_retrieval_name="renamed_run",
        verbose=False,
    )

    assert [name for name, _ in _FakeRetrieval.calls] == ["renamed_run"]
    assert result["combine"].combined_name == "renamed_run_combined"
    assert os.path.exists(tmp_path / "out_NumPyro" / "renamed_run_combined_samples.npz")


def test_run_multichain_sequential_resume_skips_existing(monkeypatch, tmp_path):
    """resume=True re-runs only the chains whose artifact is missing."""
    import petitRADTRANS.retrieval.retrieval as retrieval_module

    _FakeRetrieval.calls = []
    monkeypatch.setattr(retrieval_module, "Retrieval", _FakeRetrieval)

    # Pre-create chains 0 and 2 exactly as a previous run would have.
    seeded = _FakeRetrieval(build_demo_config(), str(tmp_path))
    for chain_id in (0, 2):
        seeded.configuration.retrieval_name = mc.chain_retrieval_name("demo_run", chain_id)
        seeded.run(seed=mc.derive_chain_seed(1000, chain_id), num_chains=1)
    _FakeRetrieval.calls = []

    result = mc.run_multichain(
        build_demo_config, num_chains=3, output_directory=str(tmp_path),
        chain_execution="sequential", chain_seed=1000, resume=True, verbose=False,
    )
    assert [name for name, _ in _FakeRetrieval.calls] == ["demo_run_chain1"]
    assert result["succeeded_chain_ids"] == [0, 1, 2]


def test_run_multichain_sequential_chain_failure_is_collected(monkeypatch, tmp_path):
    """on_chain_failure='warn' keeps going and combines the survivors."""
    import petitRADTRANS.retrieval.retrieval as retrieval_module

    class _FlakyRetrieval(_FakeRetrieval):
        calls = []

        def run(self, **run_kwargs):
            if self.configuration.retrieval_name.endswith("chain1"):
                raise RuntimeError("chain 1 blew up")
            super().run(**run_kwargs)

    _FlakyRetrieval.calls = []
    monkeypatch.setattr(retrieval_module, "Retrieval", _FlakyRetrieval)

    with pytest.warns(RuntimeWarning, match="Missing chain outputs"):
        result = mc.run_multichain(
            build_demo_config, num_chains=3, output_directory=str(tmp_path),
            chain_execution="sequential", chain_seed=1000,
            on_chain_failure="warn", verbose=False,
        )
    assert result["succeeded_chain_ids"] == [0, 2]
    assert "RuntimeError: chain 1 blew up" in result["failed"][1]
    assert result["diagnostics"]["n_samples_total"] == 128  # survivors only

    # ... and 'raise' propagates instead.
    _FlakyRetrieval.calls = []
    with pytest.raises(RuntimeError, match="chain 1 blew up"):
        mc.run_multichain(
            build_demo_config, num_chains=3, output_directory=str(tmp_path),
            chain_execution="sequential", chain_seed=1000,
            on_chain_failure="raise", verbose=False,
        )


def test_retrieval_combine_chains_shim_caches_diagnostics(monkeypatch, tmp_path):
    """The instance method delegates and caches into _chain_diagnostics."""
    from petitRADTRANS.retrieval.retrieval import Retrieval

    recorded = {}
    summary = SimpleNamespace(diagnostics={"max_rhat": 1.0, "converged": True})

    def fake_combine(output_directory, sampler_type, base_name, **kwargs):
        recorded.update(
            output_directory=output_directory,
            sampler_type=sampler_type,
            base_name=base_name,
            kwargs=kwargs,
        )
        return summary

    monkeypatch.setattr(mc, "combine_chains", fake_combine)

    # A bare instance: combine_chains only reads three attributes, so we can skip
    # the (very heavy) __init__ entirely.
    retrieval = Retrieval.__new__(Retrieval)
    retrieval.configuration = build_demo_config()
    retrieval.output_directory = str(tmp_path)
    retrieval._chain_diagnostics = None

    assert retrieval.get_convergence_diagnostics() is None
    assert retrieval.combine_chains(num_chains=2, verbose=False) is summary
    assert recorded["output_directory"] == str(tmp_path)
    assert recorded["sampler_type"] == "numpyronuts"
    assert recorded["base_name"] == "demo_run"
    assert recorded["kwargs"]["num_chains"] == 2
    assert retrieval.get_convergence_diagnostics() == summary.diagnostics
