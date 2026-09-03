"""BlackJAX backend: HMC and NUTS with window adaptation.

BlackJAX exposes sampling as explicit, composable steps rather than a driver
loop, so this module owns the loop itself: one ``lax.scan`` per chain, and --
for multi-chain runs -- a ``shard_map`` that spreads chains across devices with
``vmap`` inside each one. Window adaptation runs once and its adapted step size
and mass matrix are shared by every chain.

Everything the NumPyro backend also does (the optax MAP warm-up, saving,
reloading, summaries) lives in
:mod:`~petitRADTRANS.retrieval.samplers.gradient`.
"""

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import SamplerContext
from petitRADTRANS.retrieval.samplers.gradient import (
    GradientSampler,
    MCMCSamplingResults,
)
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_unconstrained_mcmc_interface,
)


class BlackJAXSamplingResults(MCMCSamplingResults):
    """Results of a BlackJAX run. See :class:`MCMCSamplingResults`."""


class BlackJAXSampler(GradientSampler):
    """Hamiltonian sampling with BlackJAX."""

    output_subdirectory = "out_BlackJAX"
    backend_label = "BlackJAX"
    results_class = BlackJAXSamplingResults

    def __init__(
        self, log_likelihood_func, prior_func, output_directory, retrieval_name
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_blackjax()

    @property
    def algorithm_factory(self):
        """The BlackJAX algorithm constructor, e.g. ``blackjax.nuts``."""
        raise NotImplementedError

    @classmethod
    def _prepare_blackjax(
        cls, context: SamplerContext, algorithm_name: str, **sampler_kwargs
    ):
        """Shared preparation logic for all BlackJAX sampler variants."""
        (
            free_parameter_names,
            initial_position,
            logdensity,
            transform_positions,
            log_jacobian,
        ) = build_unconstrained_mcmc_interface(context)

        optimize_init = sampler_kwargs.pop("optimize_init", False)
        warmup_optimizer_kwargs = dict(sampler_kwargs.pop("optimizer_kwargs", {}))
        warmup_best_logdensity = None

        if optimize_init and "initial_position" not in sampler_kwargs:
            initial_position, warmup_best_logdensity = cls._run_optax_warmup(
                context=context,
                sampler_kwargs=sampler_kwargs,
                warmup_optimizer_kwargs=warmup_optimizer_kwargs,
                logdensity=logdensity,
                transform_positions=transform_positions,
                free_parameter_names=free_parameter_names,
                initial_position=initial_position,
            )

        run_kwargs = cls._common_run_kwargs(
            sampler_kwargs,
            initial_position=initial_position,
            free_parameter_names=free_parameter_names,
            log_jacobian=log_jacobian,
            num_chains=sampler_kwargs.pop("num_chains", 1),
        )
        if algorithm_name == "hmc":
            run_kwargs["num_integration_steps"] = sampler_kwargs.pop(
                "num_integration_steps", 20
            )
        run_kwargs.update(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=logdensity,
            prior_func=transform_positions,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )

        summary_parameters = cls._common_summary_parameters(algorithm_name, run_kwargs)
        if "num_integration_steps" in run_kwargs:
            summary_parameters["num_integration_steps"] = run_kwargs[
                "num_integration_steps"
            ]
        if optimize_init:
            summary_parameters["optimize_init"] = True
            summary_parameters["warmup_best_logdensity"] = warmup_best_logdensity

        return sampler, run_kwargs, summary_parameters

    def run_sampler(
        self,
        initial_position,
        parameter_names,
        num_samples=1000,
        num_warmup=1000,
        initial_step_size=1.0,
        target_acceptance_rate=0.8,
        is_mass_matrix_diagonal=True,
        progress_bar=False,
        seed=42654,
        use_jit=True,
        log_jacobian_fn=None,
        num_chains=1,
        **blackjax_kwargs,
    ):
        # --- Single warmup run; adapted parameters are shared across all chains. ---
        blackjax_kwargs.pop("jit", None)
        warmup = _opt.blackjax.window_adaptation(
            self.algorithm_factory,
            self.log_likelihood_func,
            is_mass_matrix_diagonal=is_mass_matrix_diagonal,
            initial_step_size=initial_step_size,
            target_acceptance_rate=target_acceptance_rate,
            progress_bar=progress_bar,
            **blackjax_kwargs,
        )

        rng_key = _opt.random.PRNGKey(seed)
        rng_key, warmup_key, sample_key = _opt.random.split(rng_key, 3)
        (initial_state, warmup_parameters), _ = warmup.run(
            warmup_key,
            initial_position,
            num_steps=num_warmup,
        )

        algorithm = self.algorithm_factory(
            self.log_likelihood_func, **warmup_parameters
        )
        kernel = algorithm.step

        # --- Per-chain sampling loop (captures initial_state and kernel). ---
        def run_one_chain(chain_sample_key):
            def one_step(current_state, rng_k):
                next_state, info = kernel(rng_k, current_state)
                return next_state, (next_state, info)

            chain_keys = _opt.random.split(chain_sample_key, num_samples)
            _, (chain_states, chain_infos) = _opt.lax.scan(
                one_step, initial_state, chain_keys
            )
            return chain_states, chain_infos

        if num_chains == 1:
            # Original single-chain path.
            if use_jit:
                run_one_chain = _opt.jit(run_one_chain)
            states, infos = run_one_chain(sample_key)
        else:
            # Multi-chain path: distribute chains across devices using shard_map + vmap.
            # Each device runs chains_per_device independent chains using the same adapted
            # warmup parameters but different RNG seeds.
            n_devices = _opt.local_device_count()
            effective_devices = min(n_devices, num_chains)
            chains_per_device = max(1, num_chains // effective_devices)

            if effective_devices > 1:
                mesh = _opt.make_mesh((effective_devices,), ("devices",))

                def run_device_chains(base_key):
                    # Derive a unique sample key per device from the shared base key,
                    # then vmap over chains_per_device independent chains.
                    device_idx = _opt.lax.axis_index("devices")
                    device_key = _opt.random.fold_in(base_key, device_idx)
                    per_chain_keys = _opt.random.split(device_key, chains_per_device)
                    return _opt.vmap(run_one_chain)(per_chain_keys)

                run_all = _opt.shard_map(
                    run_device_chains,
                    mesh=mesh,
                    in_specs=_opt.P(),  # base_key replicated to every device
                    out_specs=_opt.P(
                        "devices"
                    ),
                    check_vma=False,
                )
                if use_jit:
                    run_all = _opt.jit(run_all)
                with _opt.set_mesh(mesh):
                    all_states, all_infos = run_all(sample_key)
            else:
                # Single device: vmap over all chains.
                chain_keys = _opt.random.split(sample_key, num_chains)
                run_all_single = _opt.vmap(run_one_chain)
                if use_jit:
                    run_all_single = _opt.jit(run_all_single)
                all_states, all_infos = run_all_single(chain_keys)

            # Flatten [num_chains, num_samples, ...] → [num_chains * num_samples, ...]
            def _flatten(x):
                return x.reshape(-1, *x.shape[2:]) if x.ndim >= 2 else x

            states = _opt.tree_util.tree_map(_flatten, all_states)
            infos = _opt.tree_util.tree_map(_flatten, all_infos)

        unconstrained_samples = np.asarray(states.position)
        transformed_samples = (
            states.position
            if self.prior_func is None
            else self.prior_func(states.position)
        )
        samples = np.asarray(transformed_samples)
        logdensity = np.asarray(states.logdensity)
        acceptance_rate = (
            np.asarray(infos.acceptance_rate)
            if hasattr(infos, "acceptance_rate")
            else None
        )
        is_divergent = (
            np.asarray(infos.is_divergent) if hasattr(infos, "is_divergent") else None
        )
        num_integration_steps = (
            np.asarray(infos.num_integration_steps)
            if hasattr(infos, "num_integration_steps")
            else None
        )

        model_log_likelihood = None
        if log_jacobian_fn is not None:
            model_log_likelihood = np.asarray(
                states.logdensity - log_jacobian_fn(states.position)
            )

        warmup_parameters_np = _opt.tree.map(np.asarray, warmup_parameters)
        metadata = {
            "algorithm": self.algorithm_name,
            "num_samples": num_samples,
            "num_warmup": num_warmup,
            "num_chains": num_chains,
            "seed": seed,
            "parameter_names": parameter_names,
        }
        samples_file, metadata_file = self._save_results(
            parameter_names=parameter_names,
            samples=samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            warmup_parameters=warmup_parameters_np,
            metadata=metadata,
        )

        self.results = BlackJAXSamplingResults(
            algorithm=self.algorithm_name,
            samples=samples,
            unconstrained_samples=unconstrained_samples,
            logdensity=logdensity,
            model_log_likelihood=model_log_likelihood,
            warmup_parameters=self._to_serializable(warmup_parameters_np),
            acceptance_rate=acceptance_rate,
            is_divergent=is_divergent,
            num_integration_steps=num_integration_steps,
            output_file=samples_file,
            metadata_file=metadata_file,
        )

        return self.results


class BlackJAXHMCSampler(BlackJAXSampler):
    """BlackJAX Hamiltonian Monte Carlo with a fixed number of integration steps."""

    sampler_type = "blackjax_hmc"
    aliases = ("blackjaxhmc", "hmc")

    @property
    def algorithm_name(self):
        return "hmc"

    @property
    def algorithm_factory(self):
        return _opt.blackjax.hmc

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_blackjax(context, algorithm_name="hmc", **sampler_kwargs)


class BlackJAXNUTSSampler(BlackJAXSampler):
    """BlackJAX No-U-Turn Sampler, which chooses its trajectory length itself."""

    sampler_type = "blackjax_nuts"
    aliases = ("blackjaxnuts", "nuts")

    @property
    def algorithm_name(self):
        return "nuts"

    @property
    def algorithm_factory(self):
        return _opt.blackjax.nuts

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        return cls._prepare_blackjax(context, algorithm_name="nuts", **sampler_kwargs)
