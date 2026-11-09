"""Batched likelihood callables and worker plumbing for Nautilus.

pRT always runs Nautilus in its vectorized mode, so that the prior transform is
compiled over whole batches. Nautilus then hands each pool worker one chunk of
``(n_points, n_free)`` physical points, and the callables here turn the
retrieval's scalar likelihood into something that accepts such a chunk.

Tasks never carry the likelihood: each worker receives it once, through
:func:`initialize_worker`, and :func:`evaluate_in_worker` is what gets pickled
for every task. Under MPI each rank builds the likelihood itself, so the
retrieval runtime is not pickled at all.
"""

import numpy as np

from petitRADTRANS.retrieval.samplers.dynesty_pool import DynestyPoolLogLikelihood

# The likelihood of this worker process, set once by initialize_worker().
_WORKER_LIKELIHOOD = None


def initialize_worker(likelihood):
    """Give this process the likelihood that :func:`evaluate_in_worker` calls."""
    global _WORKER_LIKELIHOOD
    _WORKER_LIKELIHOOD = likelihood


def evaluate_in_worker(points):
    """Evaluate this process' likelihood; picklable by reference, for every task."""
    return _WORKER_LIKELIHOOD(points)


def sanitize_log_likelihoods(log_likelihoods):
    """Replace invalid log-likelihoods with ``-inf``, which Nautilus handles natively.

    pRT marks an invalid point with the most negative finite float, and Nautilus
    never checks for NaN. Its statistics multiply log-likelihoods by two, so the
    finite marker would overflow on its way to the same answer as ``-inf``.
    """
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)
    # Compare against the float32 limit too: 64-bit values below it can only be
    # invalid markers, and that way this does not depend on the JAX precision.
    return np.where(
        np.isfinite(log_likelihoods)
        & (log_likelihoods > float(np.finfo(np.float32).min)),
        log_likelihoods,
        -np.inf,
    )


class WorkerPool:
    """What Nautilus needs of a pool, around a ``concurrent.futures`` executor.

    Nautilus takes the size of a pool from the first of several attributes it
    finds, and on a ``ProcessPoolExecutor`` that is a dict of its processes.
    """

    def __init__(self, executor, size):
        self.executor = executor
        self.size = size

    def map(self, function, iterable):
        return self.executor.map(function, iterable)


class ScalarChunkLogLikelihood:
    """Evaluate a scalar log-likelihood over a chunk of points, one at a time."""

    def __init__(self, scalar_log_likelihood):
        self._scalar_log_likelihood = scalar_log_likelihood

    def __call__(self, points):
        return sanitize_log_likelihoods(
            [self._scalar_log_likelihood(point) for point in points]
        )


class BatchedChunkLogLikelihood:
    """Evaluate a batched log-likelihood over a chunk of points in one call."""

    def __init__(self, batched_log_likelihood):
        self._batched_log_likelihood = batched_log_likelihood

    def __call__(self, points):
        return sanitize_log_likelihoods(self._batched_log_likelihood(points))


class NautilusPoolLogLikelihood(DynestyPoolLogLikelihood):
    """The chunked form of the picklable pool likelihood, for ``spawn`` workers."""

    def __call__(self, points):
        return sanitize_log_likelihoods(
            [super(NautilusPoolLogLikelihood, self).__call__(point) for point in points]
        )
