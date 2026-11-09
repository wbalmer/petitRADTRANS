"""Picklable likelihood and prior callables for Dynesty.

Dynesty pickles whatever it is given: to its worker pool, and into its
checkpoint file. A closure over the retrieval runtime survives neither, so the
two callables here carry the runtime as a **dill payload** instead and
rehydrate it lazily, once per process, on first call.

That indirection is also what keeps the payload small: the parent pickles the
bytes once at construction, and ``__getstate__`` ships only those bytes, never
the live ``Radtrans`` objects or their loaded opacity tables.
"""

import multiprocessing
import os

import dill
import numpy as np

# Core this process pinned itself to, so pinning happens once per worker.
_PINNED_CORE = None


def pin_pool_worker_to_core():
    """Pin the calling pool worker to one core of its inherited CPU set.

    Worker ``k`` of a pool (``multiprocessing`` numbers them from 1) takes the
    ``k``-th core of the affinity set it inherited, so ``njobs`` workers under
    an ``njobs``-core allocation land on distinct cores. Without this, every
    worker sizes its XLA intra-op thread pool to the whole allocation and
    ``njobs`` workers oversubscribe it ``njobs``-fold.

    It must run before the worker initialises JAX, because XLA sizes its thread
    pool from the affinity at backend start-up. Does nothing outside a pool
    worker or where CPU affinity is unsupported. Returns the pinned core, or
    ``None``.
    """
    global _PINNED_CORE

    if _PINNED_CORE is not None:
        return _PINNED_CORE

    identity = multiprocessing.current_process()._identity
    if not identity or not hasattr(os, "sched_getaffinity"):
        return None

    cores = sorted(os.sched_getaffinity(0))
    core = cores[(identity[0] - 1) % len(cores)]
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        return None

    _PINNED_CORE = core
    return core


class DynestyPoolPriorTransform:
    def __init__(self, parameter_layout, pin_worker: bool = False):
        self._parameter_layout_payload = dill.dumps(parameter_layout)
        self._parameter_layout = None
        self._pin_worker = pin_worker

    def __getstate__(self):
        return {
            "_parameter_layout_payload": self._parameter_layout_payload,
            "_parameter_layout": None,
            "_pin_worker": self._pin_worker,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_parameter_layout(self):
        if self._parameter_layout is None:
            # Checkpoints written before pinning existed carry no flag.
            if getattr(self, "_pin_worker", False):
                pin_pool_worker_to_core()
            self._parameter_layout = dill.loads(self._parameter_layout_payload)

        return self._parameter_layout

    def __call__(self, cube):
        return self._get_parameter_layout().prior_transform_numpy(cube)


class DynestyPoolLogLikelihood:
    def __init__(
        self,
        *,
        parameter_layout,
        runtime,
        uncertainties_mode: str,
        print_log_likelihood_for_debugging: bool,
        use_jit: bool,
        pin_worker: bool = False,
    ):
        self._parameter_layout_payload = dill.dumps(parameter_layout)
        self._runtime_payload = dill.dumps(runtime)
        self._uncertainties_mode = uncertainties_mode
        self._print_log_likelihood_for_debugging = (
            print_log_likelihood_for_debugging
        )
        self._use_jit = use_jit
        self._pin_worker = pin_worker
        self._parameter_layout = None
        self._runtime = None
        self._jitted_log_likelihood = None
        self._jax_module = None
        self._jax_dtype = None

    def __getstate__(self):
        return {
            "_parameter_layout_payload": self._parameter_layout_payload,
            "_runtime_payload": self._runtime_payload,
            "_uncertainties_mode": self._uncertainties_mode,
            "_print_log_likelihood_for_debugging": (
                self._print_log_likelihood_for_debugging
            ),
            "_use_jit": self._use_jit,
            "_pin_worker": self._pin_worker,
            "_parameter_layout": None,
            "_runtime": None,
            "_jitted_log_likelihood": None,
            "_jax_module": None,
            "_jax_dtype": None,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _load_state(self):
        # Pin before rehydrating the runtime: that is what first touches JAX in
        # a worker, and XLA sizes its thread pool from the affinity at that point.
        if getattr(self, "_pin_worker", False):
            pin_pool_worker_to_core()

        if self._parameter_layout is None:
            self._parameter_layout = dill.loads(self._parameter_layout_payload)

        if self._runtime is None:
            self._runtime = dill.loads(self._runtime_payload)

    def _build_jitted_log_likelihood(self):
        from jax import jit
        import jax.numpy as jnp

        from petitRADTRANS.retrieval.parameter import FEAUTRIER_JAX_DTYPE

        self._load_state()
        parameter_layout = self._parameter_layout
        runtime = self._runtime
        uncertainties_mode = self._uncertainties_mode
        print_log_likelihood = self._print_log_likelihood_for_debugging

        @jit
        def jitted_log_likelihood(free_parameter_values):
            physical_params = parameter_layout.physical_params_from_vector(
                free_parameter_values
            )
            return runtime.evaluate_likelihood_jax(
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=print_log_likelihood,
            )

        self._jitted_log_likelihood = jitted_log_likelihood
        self._jax_module = jnp
        self._jax_dtype = FEAUTRIER_JAX_DTYPE

    def __call__(self, free_parameter_values):
        self._load_state()

        if self._use_jit:
            if self._jitted_log_likelihood is None:
                self._build_jitted_log_likelihood()

            return float(
                np.asarray(
                    self._jitted_log_likelihood(
                        self._jax_module.asarray(
                            free_parameter_values,
                            dtype=self._jax_dtype,
                        )
                    )
                )
            )

        physical_params = self._parameter_layout.physical_params_from_vector(
            free_parameter_values
        )
        return self._runtime.evaluate_likelihood_scalar(
            physical_params=physical_params,
            uncertainties_mode=self._uncertainties_mode,
            print_log_likelihood_for_debugging=self._print_log_likelihood_for_debugging,
        )
