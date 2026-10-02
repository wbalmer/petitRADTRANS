"""Module-level helpers for the multi-chain spawn tests.

These must live in an importable module (not a test-local closure) so that the
``spawn`` start method can pickle them by reference and re-import them in the
worker process.
"""

import json
import os

import numpy as np


def write_synthetic_chain(out_dir, names, n_draws, base_name, base_seed, chain_id, cores):
    """Write one synthetic per-chain ``*_samples.npz`` (NumPyro/BlackJAX format).

    Records the worker affinity and whether mpi4py was blocked, so the test can
    assert the worker-side setup actually ran.
    """
    import sys

    rng = np.random.default_rng(base_seed + chain_id)
    samples = rng.standard_normal((n_draws, len(names)))
    name = f"{base_name}_chain{chain_id}"
    np.savez(
        os.path.join(out_dir, f"{name}_samples.npz"),
        samples=samples,
        unconstrained_samples=samples,
        logdensity=-0.5 * (samples ** 2).sum(axis=1),
    )
    with open(os.path.join(out_dir, f"{name}_params.json"), "w", encoding="utf-8") as handle:
        json.dump(list(names), handle)

    meta = {
        "chain_id": chain_id,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "mpi4py_blocked": sys.modules.get("mpi4py", "absent") is None,
    }
    with open(os.path.join(out_dir, f"{name}_worker.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    return name


def worker_affinity_and_pid(_):
    """Report this pool worker's pid and CPU affinity (for the pinning test)."""
    import time

    # Linger so the pool hands tasks to every worker, not just the first.
    time.sleep(0.2)
    return os.getpid(), sorted(os.sched_getaffinity(0))
