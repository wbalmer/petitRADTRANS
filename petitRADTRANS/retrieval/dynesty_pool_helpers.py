from __future__ import annotations

import dill
import numpy as np


class DynestyPoolPriorTransform:
    def __init__(self, parameter_layout):
        self._parameter_layout_payload = dill.dumps(parameter_layout)
        self._parameter_layout = None

    def __getstate__(self):
        return {
            "_parameter_layout_payload": self._parameter_layout_payload,
            "_parameter_layout": None,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_parameter_layout(self):
        if self._parameter_layout is None:
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
    ):
        self._parameter_layout_payload = dill.dumps(parameter_layout)
        self._runtime_payload = dill.dumps(runtime)
        self._uncertainties_mode = uncertainties_mode
        self._print_log_likelihood_for_debugging = (
            print_log_likelihood_for_debugging
        )
        self._use_jit = use_jit
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
            "_parameter_layout": None,
            "_runtime": None,
            "_jitted_log_likelihood": None,
            "_jax_module": None,
            "_jax_dtype": None,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _load_state(self):
        if self._parameter_layout is None:
            self._parameter_layout = dill.loads(self._parameter_layout_payload)

        if self._runtime is None:
            self._runtime = dill.loads(self._runtime_payload)

    def _build_jitted_log_likelihood(self):
        from jax import jit
        import jax.numpy as jnp

        from petitRADTRANS.retrieval.parameter import JAX_DTYPE

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
            return runtime.evaluate_jax(
                physical_params=physical_params,
                uncertainties_mode=uncertainties_mode,
                print_log_likelihood_for_debugging=print_log_likelihood,
            )

        self._jitted_log_likelihood = jitted_log_likelihood
        self._jax_module = jnp
        self._jax_dtype = JAX_DTYPE

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
        return self._runtime.evaluate_scalar(
            physical_params=physical_params,
            uncertainties_mode=self._uncertainties_mode,
            print_log_likelihood_for_debugging=self._print_log_likelihood_for_debugging,
        )
