"""Likelihood and prior interfaces shared by the sampler backends.

Each backend wants the retrieval's posterior in a different shape:

* PyMultiNest and non-vectorized UltraNest want scalar callbacks over the unit
  hypercube (optionally dispatched to a specific JAX device).
* BlackJAX, NumPyro and the optax optimiser want a differentiable
  log-posterior over an *unconstrained* space, plus the transform back to
  physical parameters and the change-of-variables Jacobian.
* JAXNS 3 wants a single callable that realises its JAXCTX priors and returns
  the scalar log likelihood.

The builders here derive all three from a :class:`~petitRADTRANS.retrieval.samplers.base.SamplerContext`,
so no backend needs to know how a ``RetrievalRuntime`` or a ``ParameterLayout``
is put together.
"""

import pickle

from petitRADTRANS.retrieval._mpi import get_mpi_context
from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import SamplerContext


def check_differentiable_support(context: SamplerContext, sampler_label: str) -> None:
    """Raise if the retrieval has only legacy (non-differentiable) model groups."""
    runtime = context.runtime
    if all(group.model_contract != "differentiable" for group in runtime.model_groups):
        raise ValueError(
            f"{sampler_label} requires at least one differentiable model group; "
            "legacy-only retrievals are not supported."
        )


def build_device_likelihood_wrapper(context: SamplerContext):
    """Return a PyMultiNest-compatible log-likelihood that dispatches to a JAX device.

    Each MPI rank r selects context.likelihood_devices[r % len(likelihood_devices)],
    so N ranks stripe naturally across M devices. Requires differentiable model groups.

    Returns ``(device_likelihood_func, target_device)``.

    Notes
    -----
    The user is responsible for ensuring that JAX processes do not unexpectedly contend
    on the same device (e.g. set ``JAX_PLATFORMS=cuda,cpu`` or ``CUDA_VISIBLE_DEVICES``
    per MPI process for true isolation when using multiple GPUs).
    """
    _opt.require_jax()
    check_differentiable_support(
        context, "Device-dispatched likelihood (likelihood_devices)"
    )

    likelihood_devices = context.likelihood_devices
    _, _, rank = get_mpi_context()
    target_device = likelihood_devices[rank % len(likelihood_devices)]

    parameter_layout = context.parameter_layout
    runtime = context.runtime
    uncertainties_mode = context.uncertainties_mode
    print_debug = context.print_log_likelihood_for_debugging
    default_device = _opt.default_device
    device_get = _opt.device_get

    _jit_evaluate = _opt.jit(
        lambda physical_params: runtime.evaluate_likelihood_jax(
            physical_params=physical_params,
            uncertainties_mode=uncertainties_mode,
            print_log_likelihood_for_debugging=print_debug,
        )
    )

    def device_likelihood(cube, ndim=0, nparam=0):
        physical_params = parameter_layout.physical_params_from_vector(cube)
        with default_device(target_device):
            result = _jit_evaluate(physical_params)
        return float(device_get(result))

    return device_likelihood, target_device


def build_unconstrained_mcmc_interface(context: SamplerContext):
    """Build the unconstrained-space logdensity and transforms for gradient-based samplers.

    Returns ``(free_parameter_names, initial_position, logdensity, transform_positions, log_jacobian)``.
    """
    _opt.require_jax()
    from petitRADTRANS.retrieval.parameter import FEAUTRIER_JAX_DTYPE

    check_differentiable_support(context, "Gradient-based JAX samplers")
    parameter_layout = context.parameter_layout
    runtime = context.runtime
    free_parameter_names = list(parameter_layout.free_parameter_names)

    if len(free_parameter_names) < 1:
        raise ValueError("Gradient-based samplers require at least one free parameter.")

    free_parameters = [
        context.configuration.parameters[name] for name in free_parameter_names
    ]
    for parameter_name, parameter in zip(free_parameter_names, free_parameters):
        if (
            parameter.distribution is None
            and parameter.transform_prior_cube_coordinate is None
        ):
            raise ValueError(
                f"Free parameter '{parameter_name}' has neither a JAX distribution nor a prior transform."
            )

    jnp = _opt.jnp
    vmap = _opt.vmap

    def transform_single(unconstrained_position):
        return parameter_layout.transform_unconstrained(unconstrained_position)

    def transform_positions(unconstrained_positions):
        if jnp.ndim(unconstrained_positions) == 1:
            return transform_single(unconstrained_positions)
        return vmap(transform_single)(unconstrained_positions)

    def log_jacobian_single(unconstrained_position):
        return parameter_layout.log_jacobian_single(unconstrained_position)

    def log_jacobian(unconstrained_positions):
        return parameter_layout.log_jacobian(unconstrained_positions)

    def logdensity(unconstrained_position):
        physical_position = transform_single(unconstrained_position)
        physical_params = parameter_layout.physical_params_from_free_values(
            tuple(physical_position)
        )
        return runtime.evaluate_likelihood_jax(
            physical_params=physical_params,
            uncertainties_mode=context.uncertainties_mode,
            print_log_likelihood_for_debugging=context.print_log_likelihood_for_debugging,
        ) + log_jacobian_single(unconstrained_position)

    initial_position = jnp.zeros(
        parameter_layout.n_free_parameters, dtype=FEAUTRIER_JAX_DTYPE
    )

    return (
        free_parameter_names,
        initial_position,
        logdensity,
        transform_positions,
        log_jacobian,
    )


def collect_jaxns_priors(context: SamplerContext):
    """Return ``(free_parameter_names, priors)`` for a JAXNS retrieval.

    ``priors`` is a tuple of ``(jaxctx Prior, periodic)`` pairs aligned with
    ``free_parameter_names``.
    """
    check_differentiable_support(context, "JAXNS")
    free_parameter_names = list(context.parameter_layout.free_parameter_names)

    if len(free_parameter_names) < 1:
        raise ValueError("JAXNS samplers require at least one free parameter.")

    priors = []
    for parameter_name in free_parameter_names:
        parameter = context.configuration.parameters[parameter_name]
        if parameter.distribution is None:
            raise ValueError(
                f"JAXNS samplers require distribution-backed free parameters; "
                f"'{parameter_name}' has no distribution."
            )
        priors.append(parameter.set_prior())

    return free_parameter_names, tuple(priors)


def prior_model_is_picklable(prior_model) -> bool:
    """Whether a JAXNS prior model survives ``pickle``.

    JAXNS stores the ``Model`` on its ``State``, so checkpointing pickles the
    prior-model callable. A closure defined inside a function cannot be
    pickled, and JAXNS' ``Pytree.save`` swallows that error and leaves a
    truncated file behind, so pRT checks up front instead.
    """
    try:
        pickle.dumps(prior_model)
    except Exception:
        return False
    return True


def build_jaxns_prior_model(context: SamplerContext):
    """Build the JAXNS 3 prior model for a retrieval.

    JAXNS 3 has no separate ``log_likelihood`` argument: a model is a single
    callable that realises its priors through JAXCTX and *returns the scalar
    log likelihood*. This replaces the JAXNS 2 pairing of a ``yield``-based
    prior generator with a separate likelihood function.

    If the configuration names a ``jaxns_config_factory``, the returned model
    is the picklable, worker-rebuildable one from
    :mod:`petitRADTRANS.retrieval.samplers.jaxns_worker`. That form is required
    for checkpoint/resume (and for the distributed backend), because JAXNS
    pickles the model as part of its checkpointed state.

    Returns:
        ``(free_parameter_names, prior_model)``.
    """
    _opt.require_jaxns()

    if getattr(context.configuration, "jaxns_config_factory", None) is not None:
        from petitRADTRANS.retrieval.samplers.jaxns_worker import (
            build_distributed_prior_model,
        )

        # This process evaluates the model itself, so reuse its runtime rather
        # than rebuilding a second copy of every opacity table from the factory.
        free_parameter_names, prior_model, _spec_path = build_distributed_prior_model(
            context, reuse_runtime=True
        )
        return free_parameter_names, prior_model

    free_parameter_names, priors = collect_jaxns_priors(context)
    parameter_layout = context.parameter_layout
    runtime = context.runtime
    uncertainties_mode = context.uncertainties_mode

    def prior_model():
        # realise() draws each parameter from the unit hypercube and registers
        # it in the JAXCTX 'X' collection under its name, which is what makes
        # results.X_samples a name-keyed tree downstream.
        values = tuple(
            prior.realise(periodic=periodic) for prior, periodic in priors
        )
        physical_params = parameter_layout.physical_params_from_free_values(values)
        return runtime.evaluate_likelihood_jax(
            physical_params=physical_params,
            uncertainties_mode=uncertainties_mode,
            print_log_likelihood_for_debugging=False,
        )

    return free_parameter_names, prior_model
