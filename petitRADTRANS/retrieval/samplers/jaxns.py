"""JAXNS 3 backend: nested sampling entirely inside JAX.

JAXNS 3 restructured its public API relative to JAXNS 2, and this module
absorbs that difference so pRT run scripts do not have to:

* A model is one callable that realises its JAXCTX priors and returns the
  scalar log likelihood -- there is no separate likelihood function.
* The sampler owns its own JIT; ``run``/``run_until_goal`` return an immutable
  ``State`` that is converted to results afterwards.
* Stopping is split in two. JAXNS itself evaluates only ``dlogZ`` and
  ``cummax_XL_frac`` (a ``DepthCondition``); every other criterion is a Python
  goal the caller checks between depth epochs. This module still accepts them
  all as sampler kwargs and synthesises that goal, so run scripts keep saying
  "stop when ..." declaratively.
* The authoritative evidence is a Monte Carlo shrinkage estimate that can be
  conditioned on retained phantom states, which is calibrated tighter than the
  cheap expectation values used for depth decisions.

Kwargs that JAXNS 2 accepted and JAXNS 3 removed are rejected with migration
guidance (see :data:`_JAXNS_REMOVED_SAMPLER_KWARGS`) rather than dropped
silently, so an old run script fails loudly instead of running a different
configuration.

Phantom collection and hourly checkpointing are on by default: phantom
conditioning measurably tightens ``log_Z``, and checkpointing turns a SLURM
wall-clock timeout into a resume rather than a total loss.

This module was developed by Claude to interface with JAXNS.
"""

import dataclasses
import gc
import json
import os
import time
import warnings
from typing import Any

import numpy as np

from petitRADTRANS.retrieval.samplers import _optional_imports as _opt
from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext
from petitRADTRANS.retrieval.samplers.interfaces import (
    build_jaxns_prior_model,
    prior_model_is_picklable,
)

# Sentinel distinguishing "user did not mention checkpointing" (-> use the
# default out_JAXNS checkpoint directory) from an explicit checkpoint_dir=None
# (-> checkpointing disabled).
_CHECKPOINT_DEFAULT = object()

# Fields of jaxns.depth_condition.DepthCondition. JAXNS 3 split the old
# TerminationCondition in two: these two fields bound how far one allocation
# epoch extends inside the compiled depth loop, and everything else became the
# caller's Python goal over the immutable State (see _JAXNS_GOAL_FIELDS).
_JAXNS_DEPTH_FIELDS = (
    "dlogZ",
    "cummax_XL_frac",
)

# dlogZ above this is almost certainly a MultiNest-style ln Z tolerance passed
# by mistake; see _build_stopping_conditions.
_JAXNS_DLOGZ_WARNING_THRESHOLD = 0.01

# Stopping criteria JAXNS moved out of its condition dataclass and into the
# Python goal loop. pRT still accepts them as sampler kwargs and synthesises
# the goal, so run scripts keep expressing "stop when ..." declaratively.
#
# Each entry is (State attribute, comparison, needs_full_result). The
# comparisons reproduce the removed TerminationRegister.is_done() exactly:
# "at_least" was `register >= value`, "at_most" was `register <= value`.
_JAXNS_GOAL_FIELDS = {
    "evidence_uncert": ("expected_log_Z_uncert", "at_most", False),
    "max_samples": ("num_samples", "at_least", False),
    "max_num_likelihood_evaluations": (
        "total_num_likelihood_evaluations",
        "at_least",
        False,
    ),
    "log_L_target": ("log_L_supremum", "at_least", False),
    # State has no cheap ESS, so this one materialises the full result at each
    # goal boundary. That is a whole allocation round apart, not per sample,
    # and it is the pattern JAXNS' own run-pattern design doc uses.
    "ess": ("ess", "at_least", True),
}

# Every field pRT accepts as a stopping criterion. Used for kwarg discovery
# and error messages.
_JAXNS_TERMINATION_FIELDS = _JAXNS_DEPTH_FIELDS + tuple(_JAXNS_GOAL_FIELDS)

# JAXNS 2 spellings that JAXNS 3 renamed. Accepted with a DeprecationWarning so
# existing pRT run scripts keep working instead of failing on an unknown field.
# TODO: remove in version 5.0.0
_JAXNS_TERMINATION_RENAMES = {
    "peak_XL_frac": "cummax_XL_frac",
}

# Termination fields with no equivalent in current JAXNS 3. The register-backed
# ones were dropped together with TerminationRegister: their quantities (the
# maximum likelihood *contour*, shrinkage efficiency, and the relative and
# absolute spread of the live likelihoods) are not exposed on State, and
# approximating them would silently stop runs at a different point than
# before. Reject them rather than guess.
# TODO: remove in version 5.0.0
_JAXNS_TERMINATION_REMOVED = {
    "live_evidence_frac": (
        "JAXNS 3 removed 'live_evidence_frac'. Use 'dlogZ', which bounds the "
        "same remaining-evidence fraction, or 'cummax_XL_frac'."
    ),
    "log_L_contour": (
        "JAXNS 3 removed the likelihood-contour register. Use 'log_L_target', "
        "which stops on the maximum likelihood reached."
    ),
    "log_L_contour_target": (
        "JAXNS 3 removed the likelihood-contour register. Use 'log_L_target', "
        "which stops on the maximum likelihood reached."
    ),
    "efficiency_threshold": (
        "JAXNS 3 removed the shrinkage-efficiency register. Use "
        "'max_num_likelihood_evaluations' to bound likelihood work instead."
    ),
    "rtol": (
        "JAXNS 3 removed the live-likelihood spread registers. Use "
        "'evidence_uncert' to target evidence precision instead."
    ),
    "atol": (
        "JAXNS 3 removed the live-likelihood spread registers. Use "
        "'evidence_uncert' to target evidence precision instead."
    ),
}

# NestedSampler kwargs removed in JAXNS 3, mapped to what replaces them. These
# are rejected loudly rather than dropped: silently ignoring, say, 'devices'
# would change the execution model without telling the user.
# TODO: remove in version 5.0.0
_JAXNS_REMOVED_SAMPLER_KWARGS = {
    "s": "JAXNS 3 removed 's'. Slice count is now 'num_slices' on the sampler.",
    "k": (
        "JAXNS 3 removed 'k' (phantom count). Phantom collection now keeps every "
        "intermediate slice state; 'num_phantoms' selects a prefix for the evidence."
    ),
    "max_phantom_samples": (
        "JAXNS removed 'max_phantom_samples': phantom collection now keeps every "
        "intermediate slice state. Use 'num_phantoms' to condition the evidence on "
        "a shorter prefix."
    ),
    "c": (
        "JAXNS 3 removed 'c'. Live-point count is 'num_live_points' "
        "(alias of 'root_allocation_degree')."
    ),
    "shell_fraction": (
        "JAXNS 3 removed 'shell_fraction'. The replacement vmap width is the "
        "explicit 'replacement_width'."
    ),
    "batch_size": (
        "JAXNS removed 'batch_size'. The local replacement vmap width is "
        "'replacement_width'; distributed worker batches are set in the jaxns-cli TOML."
    ),
    "gradient_guided": (
        "JAXNS removed gradient-guided slice sampling; its slice sampler now "
        "takes only 'num_slices'."
    ),
    "gradient_guided_slices": (
        "JAXNS removed gradient-guided slice sampling; its slice sampler now "
        "takes only 'num_slices'."
    ),
    "no_step_out": (
        "JAXNS removed 'no_step_out': its slice sampler always brackets with the "
        "unit-hypercube bounds and never steps out."
    ),
    "difficult_model": (
        "JAXNS 3 removed 'difficult_model'. Raise 'num_slices' and/or "
        "'num_live_points' instead."
    ),
    "parameter_estimation": (
        "JAXNS 3 removed 'parameter_estimation'. Use a termination condition "
        "on 'ess' to target posterior quality."
    ),
    "init_efficiency_threshold": (
        "JAXNS 3 removed 'init_efficiency_threshold'; initialisation no longer "
        "has a separate efficiency phase."
    ),
    "devices": (
        "JAXNS 3 removed device sharding. Use sampler_type='jaxns_distributed' "
        "with a jaxns-cli worker pool for multi-device runs."
    ),
}


def _log_goal_boundary(state, observed, started_at, *, reached):
    """Print one progress line per completed goal boundary.

    Goal boundaries are also the only points where JAXNS may checkpoint, so this
    line is the run's heartbeat: without it a multi-day run is silent until it
    finishes. Reads only cheap State scalars (plus whatever the goal already
    computed), and tolerates DistributedState lacking any of them.
    """

    def scalar(name):
        value = getattr(state, name, None)
        if value is None:
            return float("nan")
        try:
            return float(np.asarray(value))
        except (TypeError, ValueError):
            return float("nan")

    elapsed_h = (time.monotonic() - started_at) / 3600.0
    parts = [
        f"iter {int(state.goal_loop_iter)}",
        f"samples={scalar('num_samples'):.0f}",
        f"L-evals={scalar('total_num_likelihood_evaluations'):.0f}",
        f"max logL={scalar('log_L_supremum'):.2f}",
        f"expected dlogZ={scalar('expected_log_Z_uncert'):.3f}",
    ]
    parts += [
        f"{name}={current:.4g} (goal {target:.4g})"
        for name, (current, target) in observed.items()
        if name not in ("num_samples", "total_num_likelihood_evaluations",
                        "log_L_supremum", "expected_log_Z_uncert")
    ]
    parts.append(f"elapsed={elapsed_h:.2f} h")
    if reached:
        parts.append("goal reached")
    print("[pRT-JAXNS] goal boundary: " + ", ".join(parts), flush=True)


#: Defaults for the ``gmm_directions`` sampler kwarg. ``min_ess=None`` means
#: 4 x the number of free parameters.
GMM_DIRECTION_DEFAULTS = {
    "min_ess": None,
    "refit_every": 1,
    "num_components": 1,
    "num_iterations": 10,
    "iso_prob": 0.05,
    "regularisation": 1e-6,
}


def _resolve_gmm_directions(value):
    """Validate the ``gmm_directions`` kwarg: None/False (off), True, or a dict."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(
            "gmm_directions must be True, False/None, or a dict of options "
            f"{sorted(GMM_DIRECTION_DEFAULTS)}; got {type(value).__name__}."
        )
    unknown = set(value) - set(GMM_DIRECTION_DEFAULTS)
    if unknown:
        raise TypeError(
            f"Unknown gmm_directions option(s) {sorted(unknown)}; supported: "
            f"{sorted(GMM_DIRECTION_DEFAULTS)}."
        )
    options = {**GMM_DIRECTION_DEFAULTS, **value}
    if options["min_ess"] is not None and float(options["min_ess"]) < 0:
        raise ValueError("gmm_directions.min_ess must be non-negative.")
    if int(options["refit_every"]) < 1 or int(options["num_components"]) < 1:
        raise ValueError("gmm_directions.refit_every and num_components must be >= 1.")
    if not 0.0 <= float(options["iso_prob"]) <= 1.0:
        raise ValueError("gmm_directions.iso_prob must be between 0 and 1.")
    return options


def _scientific_state(state):
    """The State inside a DistributedState, or the State itself."""
    return state.state if hasattr(state, "pending") else state


def run_with_direction_refits(
    sampler,
    goal_cond,
    *,
    key,
    checkpoint_dir,
    checkpoint_cadence,
    options,
    n_free_parameters,
    log=None,
    checkpoint_manager_cls=None,
):
    """Run the goal loop one round at a time, refitting slice directions between rounds.

    JAXNS 3 draws slice directions isotropically unless a direction law is
    fitted explicitly at a drained boundary (``fit_gmm_directions``). Its goal
    loop has no hook for that, so this drives it: one ``resume_until_goal``
    call per round, then -- once the posterior is resolved well enough (ESS >=
    ``min_ess``; the fit is posterior-weighted and degenerate before that) -- a
    refit. A failed fit keeps the current direction law.

    Checkpointing is owned here rather than by JAXNS: a JAXNS run/resume call
    given a checkpoint directory reloads the latest checkpoint, which "takes
    precedence over the explicit state", so a refit made between calls would be
    silently discarded. JAXNS is therefore called with ``checkpoint_dir=None``
    and this function loads, saves (at most every ``checkpoint_cadence``
    seconds, after any refit) and finally saves through JAXNS'
    CheckpointManager -- so a resumed run keeps its fitted directions.

    Works for both the local NestedSampler and DistributedNestedSampler.
    """
    from contextlib import nullcontext

    if checkpoint_manager_cls is None:
        from jaxns.checkpoint import CheckpointManager as checkpoint_manager_cls
    if log is None:
        def log(message):
            print(message, flush=True)

    min_ess = options["min_ess"]
    if min_ess is None:
        min_ess = 4 * int(n_free_parameters)
    min_ess = float(min_ess)

    manager_context = (
        checkpoint_manager_cls(checkpoint_dir, checkpoint_cadence)
        if checkpoint_dir is not None
        else nullcontext()
    )
    with manager_context as manager:
        state = manager.load() if manager is not None else None
        if state is None:
            state = sampler.initialise(key)
        rounds_since_fit = 0

        while True:
            start_iteration = int(_scientific_state(state).goal_loop_iter)
            outcome = {"goal": False}

            def one_round(candidate, start_iteration=start_iteration, outcome=outcome):
                # The boundary we start from was already judged (before a
                # refit, or before the checkpoint was written); judge only the
                # next one, then hand control back.
                if int(_scientific_state(candidate).goal_loop_iter) <= start_iteration:
                    return False
                outcome["goal"] = bool(goal_cond(candidate))
                return True

            state = sampler.resume_until_goal(state, one_round, checkpoint_dir=None)
            iteration = int(_scientific_state(state).goal_loop_iter)
            if outcome["goal"] or int(_scientific_state(state).termination_reason) != 0:
                break

            rounds_since_fit += 1
            if rounds_since_fit >= int(options["refit_every"]):
                ess = float(np.asarray(state.to_result().ess))
                if ess < min_ess:
                    log(f"[pRT-JAXNS] slice directions: isotropic after iter {iteration} "
                        f"(ESS {ess:.4g} < {min_ess:g})")
                else:
                    try:
                        state = state.fit_gmm_directions(
                            num_components=int(options["num_components"]),
                            num_iterations=int(options["num_iterations"]),
                            iso_prob=float(options["iso_prob"]),
                            regularisation=float(options["regularisation"]),
                        )
                    except ValueError as problem:
                        log(f"[pRT-JAXNS] slice directions: GMM fit failed after iter "
                            f"{iteration} ({problem}); keeping the current law")
                    else:
                        rounds_since_fit = 0
                        log(f"[pRT-JAXNS] slice directions: fitted "
                            f"{int(options['num_components'])}-component GMM after iter "
                            f"{iteration} (ESS {ess:.4g})")

            if manager is not None:
                manager.maybe_save(state)

        if manager is not None:
            manager.save_if_changed(state)
    return state


class JAXNSSampler(Sampler):
    """Nested sampling with JAXNS 3.

    A :class:`jaxns.model.Model` is a single callable that realises JAXCTX
    priors and returns a scalar log likelihood, the sampler owns its own JIT
    (``run``/``run_until_goal`` return an immutable ``State``), and the final
    evidence is a Monte Carlo shrinkage estimate that can be conditioned on
    retained phantom states. See https://jaxns.readthedocs.io/en/latest/
    for more information.

    This backend enables phantom collection and hourly checkpointing by
    default: phantom conditioning measurably tightens ``log_Z``, and
    checkpointing turns a SLURM wall-clock timeout into a resume rather than a
    total loss.
    """

    sampler_type = "jaxns"
    aliases = ("jaxnestedsampler",)
    output_subdirectory = "out_JAXNS"
    uses_distribution_priors = True
    archive_log_likelihood_keys = ("log_likelihood", "weighted_log_likelihood")
    archive_sample_keys = ("sample_matrix", "samples")

    def __init__(
        self,
        log_likelihood_func,  # unused for JAXNS 3; kept for the Sampler ABC
        prior_func,  # the JAXNS 3 prior model (returns the log likelihood)
        output_directory,
        retrieval_name,
    ):
        super().__init__(
            log_likelihood_func, prior_func, output_directory, retrieval_name
        )
        _opt.require_jax()
        _opt.require_jaxns()
        self.jaxns_parameters = {}

        self.seed = -1
        self.sampler = None
        self.state = None
        self.raw_results = None
        self.evidence_samples = None
        self.parameter_names = []
        self.offload_results_to_cpu = True

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_termination_condition_mapping(values, source_name: str) -> dict[str, Any]:
        if values is None:
            return {}

        if dataclasses.is_dataclass(values) and not isinstance(values, type):
            values = {
                field.name: getattr(values, field.name)
                for field in dataclasses.fields(values)
                if getattr(values, field.name) is not None
            }
        elif hasattr(values, "_asdict"):
            values = values._asdict()

        if not isinstance(values, dict):
            raise TypeError(
                f"JAXNS termination condition source '{source_name}' must be a mapping "
                f"or a condition dataclass, but is '{type(values)}'"
            )

        return dict(values)

    @staticmethod
    def _normalize_termination_field(field_name: str, source_name: str) -> str:
        """Map a JAXNS 2 termination field onto its JAXNS 3 name."""
        if field_name in _JAXNS_TERMINATION_REMOVED:
            raise TypeError(
                f"JAXNS termination condition field '{field_name}' (from "
                f"'{source_name}') is not supported: "
                f"{_JAXNS_TERMINATION_REMOVED[field_name]}"
            )

        renamed = _JAXNS_TERMINATION_RENAMES.get(field_name)
        if renamed is not None:
            warnings.warn(
                f"JAXNS termination condition field '{field_name}' was renamed to "
                f"'{renamed}' in JAXNS 3; using '{renamed}'.",
                DeprecationWarning,
                stacklevel=3,
            )
            return renamed

        if field_name not in _JAXNS_TERMINATION_FIELDS:
            raise TypeError(
                f"Unknown JAXNS termination condition field '{field_name}' (from "
                f"'{source_name}'). Supported fields are {_JAXNS_TERMINATION_FIELDS}."
            )

        return field_name

    @classmethod
    def _build_stopping_conditions(
        cls, sampler_kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """Split stopping criteria into a DepthCondition and Python-goal values.

        JAXNS 3 keeps only ``dlogZ`` and ``cummax_XL_frac`` in a condition
        dataclass; every other criterion is now evaluated by the caller between
        compiled depth epochs. pRT still accepts them all as sampler kwargs and
        performs that split here, so run scripts stay declarative.

        Returns:
            ``(depth_condition, goal_values, summary)``. ``depth_condition`` is
            None when the user set neither depth field, which is what lets
            JAXNS apply its own default of ``dlogZ = log1p(1e-3)`` -- v2's
            stopping goal, and the one its benchmarks are calibrated against.
        """
        _opt.require_jaxns()

        termination_condition_values: dict[str, Any] = {}

        def _add_termination_field(field_name: str, value: Any, source_name: str) -> None:
            field_name = cls._normalize_termination_field(field_name, source_name)

            if field_name in termination_condition_values:
                raise TypeError(
                    f"got multiple values for JAXNS termination condition field '{field_name}'"
                )

            termination_condition_values[field_name] = value

        for source_name in (
            "termination_condition",
            "term_cond",
            "termination_condition_kwargs",
        ):
            values = sampler_kwargs.pop(source_name, None)
            values = cls._coerce_termination_condition_mapping(values, source_name)

            for field_name, value in values.items():
                _add_termination_field(field_name, value, source_name)

        # Direct kwargs and 'termination_<field>' aliases, for both the JAXNS 3
        # names and the JAXNS 2 names we still accept.
        known_field_names = list(_JAXNS_TERMINATION_FIELDS)
        known_field_names += list(_JAXNS_TERMINATION_RENAMES)
        known_field_names += list(_JAXNS_TERMINATION_REMOVED)

        for field_name in known_field_names:
            option_names = [f"termination_{field_name}"]

            # Plain 'max_samples' configures the nested sampler itself, so only
            # the prefixed alias may set the termination-condition field.
            if field_name != "max_samples":
                option_names.insert(0, field_name)

            present_option_names = [
                option_name for option_name in option_names if option_name in sampler_kwargs
            ]

            if len(present_option_names) > 1:
                raise TypeError(
                    "got multiple values for JAXNS termination condition field "
                    f"'{field_name}' via {present_option_names}"
                )

            if len(present_option_names) == 0:
                continue

            option_name = present_option_names[0]
            _add_termination_field(
                field_name,
                sampler_kwargs.pop(option_name),
                option_name,
            )

        depth_values = {
            name: value
            for name, value in termination_condition_values.items()
            if name in _JAXNS_DEPTH_FIELDS
        }
        goal_values = {
            name: value
            for name, value in termination_condition_values.items()
            if name in _JAXNS_GOAL_FIELDS
        }

        dlogZ = depth_values.get("dlogZ")
        if dlogZ is not None and float(dlogZ) > _JAXNS_DLOGZ_WARNING_THRESHOLD:
            # JAXNS compares dlogZ with the remaining-evidence FRACTION
            # L*X / (Z + L*X) at the last sample block, and that block carries
            # the end-of-run drain of the live points (X shrinks by ~1/n_live).
            # A MultiNest-style tolerance such as 0.5 is therefore met after the
            # first allocation round: on an 18-parameter pRT benchmark it
            # stopped runs ~30 nats of compression short of the posterior.
            warnings.warn(
                f"JAXNS dlogZ={float(dlogZ):g} is a remaining-evidence fraction, not a "
                "ln Z tolerance like MultiNest's evidence_tolerance, and values this "
                "large typically end the run after one allocation round, far from the "
                "posterior. JAXNS' default is log1p(1e-3) ~ 1e-3; omit dlogZ to use it.",
                RuntimeWarning,
                stacklevel=3,
            )

        if len(depth_values) > 0:
            depth_condition = _opt.DepthCondition(**depth_values)
        elif len(goal_values) > 0:
            # Goal criteria but no depth field. Leaving this None would let
            # JAXNS apply its own default (dlogZ = log1p(1e-3)), which would
            # then end the run at the first allocation boundary regardless of
            # the goal -- so asking for a *tighter* evidence_uncert would stop
            # early instead of running longer. A user condition replaced the
            # default outright before the split, so pass an explicitly empty
            # one: it ends each allocation epoch promptly and leaves
            # termination entirely to the goal. This is the pattern JAXNS'
            # own run_pattern.py design doc uses with a custom goal.
            depth_condition = _opt.DepthCondition()
        else:
            depth_condition = None

        return depth_condition, goal_values, dict(termination_condition_values)

    @staticmethod
    def _build_slice_sampler(sampler_kwargs: dict[str, Any]):
        """Build an explicit UniDimSliceSampler only if the user asked for one.

        Returning ``None`` lets JAXNS pick its own default
        (``num_slices = 5 * D``, isotropic directions), which is the reference
        configuration. The runner, not the sampler, owns phantom collection.
        """
        num_slices = sampler_kwargs.pop("num_slices", None)

        # JAXNS 3 removed UniDimSliceSampler's 'direction' field: ellipsoidal
        # geometry is no longer a sampler option but immutable State data,
        # fitted explicitly at a Python goal boundary with
        # state.fit_gmm_directions() and toggled with iso_directions() /
        # gmm_directions(). pRT does not drive that yet -- choosing a refit
        # cadence is a scientific decision, and JAXNS rejects a GMM fit
        # outright for periodic coordinates, which pRT priors can declare.
        if "ellipsoidal_directions" in sampler_kwargs:
            sampler_kwargs.pop("ellipsoidal_directions")
            raise TypeError(
                "JAXNSSampler got removed kwarg 'ellipsoidal_directions'. JAXNS 3 "
                "moved ellipsoidal directions off the slice sampler and onto the "
                "run state: fit them explicitly with state.fit_gmm_directions() "
                "between depth epochs. Runs default to isotropic directions."
            )

        if num_slices is None:
            return None, {}

        sampler = _opt.UniDimSliceSampler(num_slices=int(num_slices))
        return sampler, {"num_slices": int(num_slices)}

    @classmethod
    def _resolve_run_config(cls, sampler_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Turn user kwargs into the resolved JAXNS 3 run configuration.

        Rejects JAXNS 2 kwargs explicitly instead of dropping them, so a run
        script written against the old backend fails with an actionable message
        rather than silently running a different configuration.
        """
        _opt.require_jaxns()

        for removed_name, message in _JAXNS_REMOVED_SAMPLER_KWARGS.items():
            if removed_name in sampler_kwargs:
                raise TypeError(f"JAXNSSampler got removed kwarg '{removed_name}'. {message}")

        if "use_jit" in sampler_kwargs:
            sampler_kwargs.pop("use_jit")
            warnings.warn(
                "'use_jit' is ignored by the JAXNS 3 backend: JAXNS compiles its own "
                "depth epochs, and wrapping the run in jax.jit is no longer correct.",
                RuntimeWarning,
                stacklevel=3,
            )
        sampler_kwargs.pop("jit", None)
        # JAXNS 3 logs through its package logger rather than a verbosity flag.
        sampler_kwargs.pop("verbose", None)

        depth_condition, goal_values, termination_summary = (
            cls._build_stopping_conditions(sampler_kwargs)
        )

        # The root population: JAXNS' 'root_allocation_degree', with pRT's
        # 'num_live_points' and JAXNS' former 'target_num_live_points' accepted
        # as aliases.
        root_aliases = {
            name: sampler_kwargs.pop(name)
            for name in ("num_live_points", "target_num_live_points", "root_allocation_degree")
            if sampler_kwargs.get(name) is not None
        }
        sampler_kwargs.pop("num_live_points", None)
        sampler_kwargs.pop("target_num_live_points", None)
        sampler_kwargs.pop("root_allocation_degree", None)
        if len(set(int(value) for value in root_aliases.values())) > 1:
            raise TypeError(
                f"got different root populations {root_aliases}; 'num_live_points', "
                "'target_num_live_points' and 'root_allocation_degree' are aliases."
            )
        root_allocation_degree = (
            int(next(iter(root_aliases.values()))) if root_aliases else None
        )

        # The local replacement vmap width; 'shell_size' is its former name.
        width_aliases = {
            name: sampler_kwargs.pop(name)
            for name in ("shell_size", "replacement_width")
            if sampler_kwargs.get(name) is not None
        }
        sampler_kwargs.pop("shell_size", None)
        sampler_kwargs.pop("replacement_width", None)
        if len(set(int(value) for value in width_aliases.values())) > 1:
            raise TypeError(
                f"got different replacement widths {width_aliases}; 'shell_size' is the "
                "former name of 'replacement_width'."
            )

        # NestedSampler construction kwargs. None means "let JAXNS decide",
        # which is what makes the v3 defaults (30*D roots, 1000*roots samples,
        # geometric buffer growth) actually apply.
        nested_sampler_kwargs = {
            "root_allocation_degree": root_allocation_degree,
            "max_samples": (
                None
                if sampler_kwargs.get("max_samples") is None
                else int(sampler_kwargs.pop("max_samples"))
            ),
            "replacement_width": (
                int(next(iter(width_aliases.values()))) if width_aliases else None
            ),
            "delta_K": sampler_kwargs.pop("delta_K", None),
            "initial_capacity": sampler_kwargs.pop("initial_capacity", None),
            "unlimited_samples": bool(sampler_kwargs.pop("unlimited_samples", False)),
            "allocation_target": sampler_kwargs.pop("allocation_target", "uniform"),
            "collect_phantom_samples": bool(
                sampler_kwargs.pop("collect_phantom_samples", True)
            ),
        }
        sampler_kwargs.pop("max_samples", None)

        run_config = {
            "nested_sampler_kwargs": nested_sampler_kwargs,
            "depth_condition": depth_condition,
            # Whether the user set dlogZ/cummax_XL_frac themselves. Only then
            # may the depth condition also terminate a goal-driven run.
            "depth_explicit": any(
                field_name in termination_summary
                for field_name in _JAXNS_DEPTH_FIELDS
            ),
            # Criteria JAXNS no longer evaluates itself; _build_goal_condition
            # turns these into the Python goal passed to run_until_goal.
            "goal_values": goal_values,
            "termination_summary": termination_summary,
            "seed": int(sampler_kwargs.pop("seed", 42654)),
            "checkpoint_dir": sampler_kwargs.get("checkpoint_dir", _CHECKPOINT_DEFAULT),
            "checkpoint_explicit": sampler_kwargs.pop("checkpoint_dir", _CHECKPOINT_DEFAULT)
            is not _CHECKPOINT_DEFAULT,
            "checkpoint_cadence": float(sampler_kwargs.pop("checkpoint_cadence", 3600.0)),
            "goal_cond": sampler_kwargs.pop("goal_cond", None),
            "sanity_check": bool(sampler_kwargs.pop("sanity_check", False)),
            "offload_results_to_cpu": bool(
                sampler_kwargs.pop("offload_results_to_cpu", True)
            ),
            # Final evidence: Monte Carlo shrinkage, phantom-conditioned by
            # default because it is calibrated tighter than classic shrinkage.
            "evidence_num_samples": int(sampler_kwargs.pop("evidence_num_samples", 4096)),
            "evidence_conditioning": sampler_kwargs.pop("evidence_conditioning", "phantom"),
            # Posterior-aligned slice directions, refitted between goal rounds.
            "gmm_directions": _resolve_gmm_directions(sampler_kwargs.pop("gmm_directions", None)),
            "num_phantoms": sampler_kwargs.pop("num_phantoms", None),
            "evidence_batch_size": sampler_kwargs.pop("evidence_batch_size", None),
            "num_equal_weight_samples": sampler_kwargs.pop(
                "num_equal_weight_samples", None
            ),
            # Slice-sampler options are resolved later: UniDimSliceSampler needs
            # the Model, which does not exist until run time.
            "slice_sampler_kwargs": {
                key: sampler_kwargs.pop(key)
                for key in (
                    "num_slices",
                    # Removed in JAXNS 3; forwarded only so that
                    # _build_slice_sampler can reject it with guidance rather
                    # than letting it fall through to NestedSampler.
                    "ellipsoidal_directions",
                )
                if key in sampler_kwargs
            },
        }

        if run_config["evidence_conditioning"] not in ("classic", "phantom", None):
            raise ValueError(
                "evidence_conditioning must be 'classic', 'phantom', or None, got "
                f"{run_config['evidence_conditioning']!r}."
            )
        if (
            run_config["evidence_conditioning"] == "phantom"
            and not nested_sampler_kwargs["collect_phantom_samples"]
        ):
            raise ValueError(
                "evidence_conditioning='phantom' requires collect_phantom_samples=True. "
                "Either enable phantom collection or pass evidence_conditioning='classic'."
            )

        # Anything left over is forwarded verbatim to NestedSampler so new JAXNS
        # options remain reachable without a pRT release.
        nested_sampler_kwargs.update(sampler_kwargs)

        return run_config

    @classmethod
    def prepare(cls, context: SamplerContext, **sampler_kwargs):
        _opt.require_jax()
        _opt.require_jaxns()

        free_parameter_names, prior_model = build_jaxns_prior_model(context)
        run_config = cls._resolve_run_config(sampler_kwargs)

        sampler = cls(
            log_likelihood_func=None,
            prior_func=prior_model,
            output_directory=context.output_directory,
            retrieval_name=context.retrieval_name,
        )
        sampler.parameter_names = list(free_parameter_names)
        sampler.offload_results_to_cpu = run_config["offload_results_to_cpu"]

        summary_parameters = {
            "sampler_type": cls.sampler_type,
            "parameter_names": list(free_parameter_names),
            "seed": run_config["seed"],
            "evidence_conditioning": run_config["evidence_conditioning"],
            "evidence_num_samples": run_config["evidence_num_samples"],
            "offload_results_to_cpu": run_config["offload_results_to_cpu"],
            "checkpoint_cadence": run_config["checkpoint_cadence"],
            **{
                key: value
                for key, value in run_config["nested_sampler_kwargs"].items()
                if value is not None
            },
        }
        if len(run_config["termination_summary"]) > 0:
            summary_parameters["termination_condition"] = run_config["termination_summary"]

        return sampler, {"run_config": run_config}, summary_parameters

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _get_raw_results_path(self):
        # JAXNS 3 results are immutable pytree dataclasses persisted with
        # pickle (PureDataclassPytree.save), not the JAXNS 2 JSON format.
        return self._get_output_prefix() + "_jaxns_results.pkl"

    def _get_legacy_raw_results_path(self):
        return self._get_output_prefix() + "_jaxns_results.json"

    def _get_checkpoint_dir(self):
        return self._get_output_prefix() + "_checkpoints"

    def _get_equal_weight_samples_path(self):
        return self._get_output_prefix() + "_samples.npz"

    def _get_parameter_names_path(self):
        return self._get_output_prefix() + "_params.json"

    def _get_evidence_path(self):
        return self._get_output_prefix() + "_evidence.json"

    def _write_evidence_sidecar(self, processed_results):
        """Persist scalar evidence/diagnostics next to the equal-weight samples.

        JAXNS stores the evidence (log Z, ESS, ...) only in memory and in its
        native raw-results file. Writing a tiny JSON sidecar lets the dependency
        -light, JAXNS-free multi-run combiner
        (:func:`petitRADTRANS.retrieval.multichain.combine_nested_chains`) read
        each run's evidence to combine independent nested-sampling runs.

        The key names here are a stable contract with that combiner: keep them
        even though JAXNS 3 renamed some of the attributes they come from.
        """
        evidence = {}
        for key in (
            "log_Z_mean",
            "log_Z_uncert",
            "ESS",
            "H_mean",
            "total_num_samples",
            "total_num_likelihood_evaluations",
            "log_efficiency",
            # JAXNS 3 additions; the combiner ignores unknown keys.
            "log_Z_mc_mean",
            "log_Z_mc_uncert",
            "total_phantom_samples",
        ):
            value = processed_results.get(key)
            if value is None:
                continue
            value = np.asarray(value)
            if value.size == 1:
                evidence[key] = float(value.item())

        conditioning = processed_results.get("evidence_conditioning")
        if conditioning is not None:
            evidence["evidence_conditioning"] = str(conditioning)

        samples = processed_results.get("samples")
        if samples is not None:
            evidence["n_samples"] = int(np.asarray(samples).shape[0])

        path = self._get_evidence_path()
        try:
            with open(path, "w") as handle:
                json.dump(evidence, handle, indent=4)
        except OSError as error:
            warnings.warn(
                f"Could not write JAXNS evidence sidecar {path}: {error}",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------
    # Device / memory handling
    # ------------------------------------------------------------------

    def _get_cpu_device(self):
        if not self.offload_results_to_cpu:
            return None

        try:
            cpu_devices = _opt.devices("cpu")
        except Exception as error:
            warnings.warn(
                "No JAX CPU backend is available for result offload. "
                "Ensure the environment initializes the CPU backend, for example with "
                "JAX_PLATFORMS=cuda,cpu. "
                f"Original error: {error}",
                RuntimeWarning,
            )
            return None

        for device in cpu_devices:
            if getattr(device, "platform", None) == "cpu":
                return device

        return None

    def _materialize_raw_results(self, state):
        """Convert a JAXNS ``State`` into trimmed results, off the accelerator.

        ``to_result`` materialises every sample array, so on a GPU run it is
        done on the host: pulling the state across first and clearing the JAX
        caches keeps the accelerator arena free for the (much larger) forward
        model rather than the result tree.
        """
        cpu_device = self._get_cpu_device()

        if cpu_device is None:
            return state.to_result().trim()

        state = _opt.device_get(state)

        gc.collect()
        _opt.clear_caches()

        with _opt.default_device(cpu_device):
            return state.to_result().trim()

    def _to_numpy_tree(self, pytree):
        cpu_device = self._get_cpu_device()

        if cpu_device is None:
            return _opt.tree.map(np.asarray, pytree)

        with _opt.default_device(cpu_device):
            return _opt.tree.map(np.asarray, pytree)

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_named_tree(named_tree, prefix=""):
        """Flatten a JAXCTX ScopedDict/dict of X-space samples to {name: array}."""
        if named_tree is None:
            return None

        if hasattr(named_tree, "to_dict"):
            named_tree = named_tree.to_dict()

        if not isinstance(named_tree, dict):
            return None

        flattened = {}
        for key, value in named_tree.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict) or hasattr(value, "to_dict"):
                nested = JAXNSSampler._flatten_named_tree(value, prefix=f"{name}.")
                if nested:
                    flattened.update(nested)
            else:
                flattened[name] = value

        return flattened or None

    @staticmethod
    def _get_named_samples_mapping(raw_results):
        """Return the name-keyed posterior samples from a JAXNS 3 result."""
        return JAXNSSampler._flatten_named_tree(
            getattr(raw_results, "X_samples", None)
        )

    @staticmethod
    def _samples_mapping_to_matrix(samples_by_parameter, parameter_names=None):
        if not isinstance(samples_by_parameter, dict):
            return None, list(parameter_names or [])

        ordered_names = list(parameter_names or [])
        if ordered_names:
            ordered_names = [
                name for name in ordered_names if name in samples_by_parameter
            ]

        if not ordered_names:
            ordered_names = list(samples_by_parameter.keys())

        if not ordered_names:
            return None, []

        matrix = np.column_stack(
            [
                np.asarray(samples_by_parameter[name], dtype=float).reshape(-1)
                for name in ordered_names
            ]
        )
        return matrix, ordered_names

    def _save_equal_weight_samples(self, parameter_names, samples, log_likelihood):
        if samples is None:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because no posterior sample "
                "matrix is available.",
                RuntimeWarning,
            )
            return None, None

        sample_matrix = np.asarray(samples, dtype=float)
        log_likelihood = np.asarray(log_likelihood, dtype=float).reshape(-1)
        parameter_names = list(parameter_names)

        if sample_matrix.ndim == 1:
            if len(parameter_names) != 1:
                warnings.warn(
                    "Skipping JAXNS equal-weight sample export because the sample "
                    "matrix is 1-D but the parameter list is not.",
                    RuntimeWarning,
                )
                return None, None
            sample_matrix = sample_matrix.reshape(-1, 1)
        elif sample_matrix.ndim != 2:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because the sample matrix "
                "is not 2-D.",
                RuntimeWarning,
            )
            return None, None

        if len(parameter_names) == 0 or sample_matrix.shape[1] != len(parameter_names):
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because parameter names "
                "do not match the sample matrix columns.",
                RuntimeWarning,
            )
            return None, None

        if log_likelihood.shape[0] != sample_matrix.shape[0]:
            warnings.warn(
                "Skipping JAXNS equal-weight sample export because log-likelihood "
                "values do not match the number of samples.",
                RuntimeWarning,
            )
            return None, None

        samples_file = self._get_equal_weight_samples_path()
        params_file = self._get_parameter_names_path()

        np.savez(
            samples_file,
            parameter_names=np.asarray(parameter_names, dtype=str),
            sample_matrix=sample_matrix,
            log_likelihood=log_likelihood,
        )

        with open(params_file, "w") as handle:
            json.dump(list(parameter_names), handle, indent=4)

        return samples_file, params_file

    def _build_processed_results(
        self,
        raw_results,
        *,
        equal_weight_samples=None,
        equal_weight_log_likelihood=None,
        output_file=None,
        params_file=None,
        raw_results_file=None,
        evidence_samples=None,
        evidence_conditioning=None,
        write_sidecar=True,
    ):
        weighted_matrix, parameter_names = self._samples_mapping_to_matrix(
            self._get_named_samples_mapping(raw_results),
            self.parameter_names,
        )
        self.parameter_names = list(parameter_names)

        # JAXNS 3 renamed log_L_samples -> log_L and log_dp_mean -> log_dp.
        weighted_log_likelihood = np.asarray(
            getattr(raw_results, "log_L", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)
        weighted_log_weights = np.asarray(
            getattr(raw_results, "log_dp", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)

        if equal_weight_samples is None or equal_weight_log_likelihood is None:
            samples = weighted_matrix
            log_likelihood = weighted_log_likelihood
            log_weights = weighted_log_weights
        else:
            samples = np.asarray(equal_weight_samples, dtype=float)
            if samples.ndim == 1:
                target_width = 1 if len(self.parameter_names) <= 1 else len(self.parameter_names)
                samples = samples.reshape(-1, target_width)

            log_likelihood = np.asarray(
                equal_weight_log_likelihood,
                dtype=float,
            ).reshape(-1)
            sample_count = samples.shape[0]
            if sample_count > 0:
                log_weights = np.full(sample_count, -np.log(sample_count), dtype=float)
            else:
                log_weights = np.array([], dtype=float)

        processed_results = {
            "samples": samples,
            "log_likelihood": log_likelihood,
            "log_weights": log_weights,
            "weighted_samples": {
                "points": weighted_matrix,
                "logl": weighted_log_likelihood,
                "log_weights": weighted_log_weights,
            },
        }

        if output_file is not None:
            processed_results["output_file"] = output_file
        if params_file is not None:
            processed_results["params_file"] = params_file
        if raw_results_file is not None:
            processed_results["raw_results_file"] = raw_results_file

        # Scalar diagnostics. 'ESS' keeps its pRT spelling (JAXNS 3 exposes it
        # as 'ess') because the multi-run combiner reads that key.
        for public_key, attribute_name in (
            ("log_Z_mean", "log_Z_mean"),
            ("log_Z_uncert", "log_Z_uncert"),
            ("ESS", "ess"),
            ("H_mean", "H_mean"),
            ("total_num_samples", "total_num_samples"),
            ("total_phantom_samples", "total_phantom_samples"),
            ("total_num_likelihood_evaluations", "total_num_likelihood_evaluations"),
            ("log_efficiency", "log_efficiency"),
            ("termination_reason", "termination_reason"),
        ):
            value = getattr(raw_results, attribute_name, None)
            if value is None:
                continue

            value = np.asarray(value)
            if value.size == 1:
                processed_results[public_key] = value.item()
            else:
                processed_results[public_key] = value

        # The Monte Carlo (optionally phantom-conditioned) evidence is the
        # authoritative final estimate in JAXNS 3; the block-moment
        # log_Z_mean/log_Z_uncert above are the cheap expectation values used
        # for depth decisions. Promote the MC estimate when we have it.
        if evidence_samples is not None:
            log_z_mc = np.asarray(evidence_samples.log_Z_mean)
            log_z_mc_uncert = np.asarray(evidence_samples.log_Z_uncert)
            if log_z_mc.size == 1 and np.isfinite(log_z_mc.item()):
                processed_results["log_Z_mc_mean"] = log_z_mc.item()
                processed_results["log_Z_mc_uncert"] = float(log_z_mc_uncert.item())
                processed_results["log_Z_expected_mean"] = processed_results.get("log_Z_mean")
                processed_results["log_Z_expected_uncert"] = processed_results.get(
                    "log_Z_uncert"
                )
                processed_results["log_Z_mean"] = log_z_mc.item()
                processed_results["log_Z_uncert"] = float(log_z_mc_uncert.item())
        if evidence_conditioning is not None:
            processed_results["evidence_conditioning"] = evidence_conditioning

        # When a fresh run saved equal-weight samples, persist the scalar evidence
        # next to them so the multi-run nested combiner can read it without JAXNS.
        # Reload paths pass write_sidecar=False: they have no Monte Carlo
        # evidence in hand, so rewriting would replace the run's authoritative
        # log_Z with the weaker expectation estimate.
        if write_sidecar and output_file is not None:
            self._write_evidence_sidecar(processed_results)

        return processed_results

    def _apply_evidence_sidecar(self, processed_results):
        """Restore the Monte Carlo evidence recorded when the run finished.

        The raw JAXNS results only carry the cheap expectation-based log_Z, so
        without this a reloaded run would report a different (and weaker)
        evidence than the same run reported when it completed.
        """
        path = self._get_evidence_path()
        if not os.path.exists(path):
            return processed_results

        try:
            with open(path, "r") as handle:
                evidence = json.load(handle)
        except (OSError, ValueError) as error:
            warnings.warn(
                f"Could not read JAXNS evidence sidecar {path}: {error}",
                RuntimeWarning,
            )
            return processed_results

        log_z_mc = evidence.get("log_Z_mc_mean")
        if log_z_mc is None:
            return processed_results

        processed_results["log_Z_expected_mean"] = processed_results.get("log_Z_mean")
        processed_results["log_Z_expected_uncert"] = processed_results.get("log_Z_uncert")
        processed_results["log_Z_mc_mean"] = float(log_z_mc)
        processed_results["log_Z_mc_uncert"] = float(evidence["log_Z_mc_uncert"])
        processed_results["log_Z_mean"] = float(log_z_mc)
        processed_results["log_Z_uncert"] = float(evidence["log_Z_mc_uncert"])
        if evidence.get("evidence_conditioning") is not None:
            processed_results["evidence_conditioning"] = evidence["evidence_conditioning"]

        return processed_results

    def _sample_evidence(self, raw_results, run_config):
        """Draw the authoritative Monte Carlo evidence ensemble."""
        conditioning = run_config["evidence_conditioning"]
        if conditioning is None:
            return None, None

        num_phantoms = run_config["num_phantoms"]
        if conditioning == "classic":
            num_phantoms = None
        elif int(np.asarray(raw_results.log_L_phantom).shape[1]) == 0:
            warnings.warn(
                "JAXNS phantom-conditioned evidence was requested but no phantom states "
                "were retained; falling back to classic conditioning.",
                RuntimeWarning,
            )
            conditioning = "classic"
            num_phantoms = None

        evidence_kwargs = {
            "phantom_conditioning": conditioning == "phantom",
            "key": _opt.random.PRNGKey(self.seed + 1),
            "batch_size": run_config["evidence_batch_size"],
            "diagnostics": False,
        }
        if conditioning == "phantom":
            evidence_kwargs["num_phantoms"] = num_phantoms

        cpu_device = self._get_cpu_device()
        try:
            if cpu_device is None:
                evidence_samples = raw_results.sample_evidence(
                    run_config["evidence_num_samples"], **evidence_kwargs
                )
            else:
                with _opt.default_device(cpu_device):
                    evidence_samples = raw_results.sample_evidence(
                        run_config["evidence_num_samples"], **evidence_kwargs
                    )
        except Exception as error:
            warnings.warn(
                f"JAXNS Monte Carlo evidence estimation failed ({error}); falling back "
                "to the expectation-based log_Z.",
                RuntimeWarning,
            )
            return None, conditioning

        return evidence_samples, conditioning

    def _build_equal_weight_results(self, raw_results, run_config):
        num_equal_weight_samples = run_config["num_equal_weight_samples"]
        if num_equal_weight_samples is None:
            # Match the classic sample count, the usual convention for an
            # equal-weight posterior (cf. dynesty's resample_equal).
            num_equal_weight_samples = int(np.asarray(raw_results.total_num_samples).item())
        elif isinstance(num_equal_weight_samples, str):
            # "ess": size the archive to the effective sample size instead.
            #
            # The default convention stores as many rows as there were nested
            # samples, drawn WITH REPLACEMENT from a posterior whose effective
            # size is far smaller. On a SIMP 0136 stage-1 run that was 45985 rows
            # carrying a Kish ESS of 4643 -- every effective sample present about
            # six times, unevenly. A histogram or kernel density built on those
            # rows is six times over-resolved and shows spikes that are pure
            # resampling grain; they read as modes in a corner plot and are not.
            # Sizing the draw to the ESS makes the stored archive claim only the
            # resolution the run actually supports.
            if num_equal_weight_samples.lower() != "ess":
                raise ValueError(
                    "num_equal_weight_samples must be an int, None, or the "
                    f"string 'ess'; got {num_equal_weight_samples!r}."
                )
            ess = getattr(raw_results, "ess", None)
            if ess is None:
                warnings.warn(
                    "num_equal_weight_samples='ess' was requested but this "
                    "JAXNS result exposes no 'ess'; falling back to the total "
                    "sample count.",
                    RuntimeWarning,
                )
                num_equal_weight_samples = int(
                    np.asarray(raw_results.total_num_samples).item()
                )
            else:
                num_equal_weight_samples = int(round(float(np.asarray(ess).item())))
        num_equal_weight_samples = max(1, int(num_equal_weight_samples))

        cpu_device = self._get_cpu_device()
        resample_key = _opt.random.PRNGKey(self.seed)

        def _resample():
            return raw_results.resample(
                num_equal_weight_samples,
                replace=True,
                key=resample_key,
            )

        if cpu_device is None:
            resampled = _resample()
        else:
            with _opt.default_device(cpu_device):
                resampled = _resample()

        equal_weight_samples, parameter_names = self._samples_mapping_to_matrix(
            self._get_named_samples_mapping(resampled),
            self.parameter_names,
        )
        self.parameter_names = list(parameter_names)
        # np.asarray on a JAX buffer yields a read-only view; copy so the
        # best-fit row below can be written in place.
        equal_weight_log_likelihood = np.array(
            getattr(resampled, "log_L", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)
        if equal_weight_samples is not None:
            equal_weight_samples = np.array(equal_weight_samples, dtype=float)

        # Keep the best-fit point in the equal-weight archive. JAXNS 3 reports
        # it directly as X_supremum / log_L_supremum, so there is no need to
        # argmax over the weighted samples.
        supremum_mapping = self._flatten_named_tree(
            getattr(raw_results, "X_supremum", None)
        )
        if equal_weight_samples is not None and equal_weight_samples.shape[0] > 0:
            supremum_row, _ = self._samples_mapping_to_matrix(
                supremum_mapping,
                self.parameter_names,
            )
            if (
                supremum_row is not None
                and supremum_row.size == equal_weight_samples.shape[1]
            ):
                equal_weight_samples[0, :] = supremum_row.reshape(-1)
                equal_weight_log_likelihood[0] = float(
                    np.asarray(raw_results.log_L_supremum).item()
                )

        evidence_samples, evidence_conditioning = self._sample_evidence(
            raw_results, run_config
        )
        self.evidence_samples = evidence_samples

        samples_file, params_file = self._save_equal_weight_samples(
            self.parameter_names,
            equal_weight_samples,
            equal_weight_log_likelihood,
        )

        return self._build_processed_results(
            raw_results,
            equal_weight_samples=equal_weight_samples,
            equal_weight_log_likelihood=equal_weight_log_likelihood,
            output_file=samples_file,
            params_file=params_file,
            raw_results_file=self._get_raw_results_path(),
            evidence_samples=evidence_samples,
            evidence_conditioning=evidence_conditioning,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _build_nested_sampler(self, model, run_config):
        """Construct the JAXNS 3 NestedSampler for this run."""
        nested_sampler_kwargs = {
            key: value
            for key, value in run_config["nested_sampler_kwargs"].items()
            if value is not None
        }

        slice_sampler, slice_summary = self._build_slice_sampler(
            dict(run_config["slice_sampler_kwargs"])
        )
        if slice_sampler is not None:
            nested_sampler_kwargs["sampler"] = slice_sampler

        if run_config["depth_condition"] is not None:
            nested_sampler_kwargs["depth_condition"] = run_config["depth_condition"]

        self.jaxns_parameters = {
            **{
                key: value
                for key, value in nested_sampler_kwargs.items()
                if key not in ("sampler", "depth_condition")
            },
            **slice_summary,
        }

        return _opt.NestedSampler(model=model, **nested_sampler_kwargs)

    def _resolve_checkpoint_dir(self, run_config):
        checkpoint_dir = run_config["checkpoint_dir"]
        explicit = run_config.get("checkpoint_explicit", False)

        if checkpoint_dir is _CHECKPOINT_DEFAULT:
            checkpoint_dir = self._get_checkpoint_dir()
        if checkpoint_dir is None:
            return None

        # JAXNS keeps the Model on its State, so a checkpoint pickles the prior
        # model. An ordinary pRT prior model is a closure over the retrieval
        # runtime and cannot be pickled -- and JAXNS' Pytree.save swallows that
        # failure, leaving a truncated file that only surfaces as a
        # CheckpointCorruptionError on the next resume. Refuse to write a
        # checkpoint we could never resume.
        if not prior_model_is_picklable(self.prior_func):
            message = (
                "JAXNS checkpointing needs a picklable prior model, but this retrieval's "
                "model closes over the (unpicklable) RetrievalRuntime. Set "
                "`configuration.jaxns_config_factory = 'my_module:build_configuration'` to a "
                "module-level function that rebuilds this RetrievalConfig; pRT then uses a "
                "model that carries only a small spec, which both checkpoints and resumes."
            )
            if explicit:
                raise ValueError(message)
            warnings.warn(
                f"Disabling JAXNS checkpointing for this run. {message}",
                RuntimeWarning,
            )
            return None

        checkpoint_dir = str(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def _build_goal_condition(self, run_config):
        """Turn the non-depth stopping criteria into a JAXNS Python goal.

        JAXNS 3 evaluates only ``dlogZ``/``cummax_XL_frac`` itself; the rest of
        the old TerminationCondition is now the caller's job, checked between
        compiled depth epochs. Returns None when the user asked for none of
        them, so an ordinary run keeps using ``NestedSampler.run``.
        """
        goal_values = dict(run_config["goal_values"])
        if len(goal_values) == 0:
            return None

        # run_until_goal loops until the *goal* is true; the depth condition
        # only sizes one allocation epoch and never stops the loop. When the
        # user set dlogZ/cummax_XL_frac alongside goal criteria, OR it in so
        # those keep terminating the run -- the removed TerminationCondition
        # OR-ed all of its fields together the same way. When the user set no
        # depth field, _build_stopping_conditions has already passed an empty
        # DepthCondition, and the goal alone owns termination.
        depth_condition = None
        depth_reached = None
        if run_config["depth_explicit"]:
            depth_condition = self.sampler.depth_condition
            depth_reached = _opt.depth_condition_reached
            if depth_reached is None:
                raise RuntimeError(
                    "This JAXNS build does not expose the depth-condition predicate "
                    "(jaxns.algorithm.depth._depth_condition_reached), so pRT cannot "
                    f"combine {sorted(goal_values)} with dlogZ/cummax_XL_frac "
                    "termination. Drop one side, or pass an explicit 'goal_cond' "
                    "callable that owns termination entirely."
                )

        checks = []
        for field_name, value in goal_values.items():
            attribute_name, comparison, needs_result = _JAXNS_GOAL_FIELDS[field_name]
            checks.append((attribute_name, comparison, needs_result, float(value)))

        needs_result = any(check[2] for check in checks)
        started_at = time.monotonic()

        def goal_cond(state):
            # Iteration 0 has no completed allocation round to judge; JAXNS'
            # own default goal skips it for the same reason.
            if int(state.goal_loop_iter) == 0:
                return False

            if depth_reached is not None and bool(
                depth_reached(state, depth_condition)
            ):
                _log_goal_boundary(state, {}, started_at, reached=True)
                return True

            # Only materialise the result tree when a criterion needs it
            # (currently just 'ess'); the State scalars are the cheap path.
            result = state.to_result() if needs_result else None

            reached = False
            observed = {}
            for attribute_name, comparison, from_result, target in checks:
                source = result if from_result else state
                current = float(np.asarray(getattr(source, attribute_name)))
                observed[attribute_name] = (current, target)
                if comparison == "at_least":
                    reached = reached or current >= target
                else:
                    reached = reached or current <= target

            _log_goal_boundary(state, observed, started_at, reached=reached)
            return reached

        return goal_cond

    def _execute(self, run_config, checkpoint_dir):
        """Run the sampler. Split out so the distributed backend can override."""
        jax_random_key = _opt.random.PRNGKey(self.seed)

        # An explicit goal_cond callable owns termination completely, matching
        # JAXNS' own contract; otherwise pRT synthesises one from the stopping
        # kwargs, and falls back to run() when there are none.
        goal_cond = run_config["goal_cond"]
        if goal_cond is None:
            goal_cond = self._build_goal_condition(run_config)

        gmm_options = run_config.get("gmm_directions")
        if gmm_options is not None:
            if goal_cond is None:
                raise ValueError(
                    "gmm_directions refits slice directions between goal rounds, so it "
                    "needs a stopping goal (e.g. ess=1000). Without one JAXNS runs a "
                    "single depth epoch and there is no boundary to refit at."
                )
            return run_with_direction_refits(
                self.sampler,
                goal_cond,
                key=jax_random_key,
                checkpoint_dir=checkpoint_dir,
                checkpoint_cadence=run_config["checkpoint_cadence"],
                options=gmm_options,
                n_free_parameters=len(self.parameter_names),
            )

        if goal_cond is None:
            return self.sampler.run(
                key=jax_random_key,
                checkpoint_dir=checkpoint_dir,
                checkpoint_cadence=run_config["checkpoint_cadence"],
            )

        return self.sampler.run_until_goal(
            goal_cond,
            key=jax_random_key,
            checkpoint_dir=checkpoint_dir,
            checkpoint_cadence=run_config["checkpoint_cadence"],
        )

    def run_sampler(self, run_config=None, **jaxns_kwargs):
        """Run the JAXNS 3 nested sampler."""
        _opt.require_jaxns()

        if run_config is None:
            # Direct invocation (e.g. run(sampler_object=...)) bypasses prepare().
            run_config = self._resolve_run_config(jaxns_kwargs)
        elif jaxns_kwargs:
            raise TypeError(
                f"run_sampler() got unexpected keyword arguments {sorted(jaxns_kwargs)} "
                "alongside a resolved run_config."
            )

        self.seed = run_config["seed"]
        self.offload_results_to_cpu = run_config["offload_results_to_cpu"]

        # In JAXNS 3 the prior model IS the model: it realises the priors and
        # returns the scalar log likelihood, so there is no log_likelihood arg.
        model = _opt.Model(prior_model=self.prior_func)

        if run_config["sanity_check"]:
            model.sanity_check(key=_opt.random.PRNGKey(self.seed))

        if not self.parameter_names:
            named = self._flatten_named_tree(
                model.transform_to_X(model.sample_U(_opt.random.PRNGKey(self.seed)))
            )
            self.parameter_names = list(named or [])

        self.sampler = self._build_nested_sampler(model, run_config)
        checkpoint_dir = self._resolve_checkpoint_dir(run_config)

        print(f"Starting JAXNS retrieval: {self.retrieval_name}")
        if checkpoint_dir is not None:
            print(f"  checkpointing to {checkpoint_dir}")

        self.state = self._execute(run_config, checkpoint_dir)

        self.raw_results = self._materialize_raw_results(self.state)

        del self.state
        self.state = None
        gc.collect()
        _opt.clear_caches()

        results_out = self._to_numpy_tree(self.raw_results)
        results_out.save(self._get_raw_results_path())

        self.raw_results = results_out

        self.results = self._build_equal_weight_results(self.raw_results, run_config)

        self.pretty_print_results()
        self.plot_diagnostics(self.results)
        return self.results

    def post_run(self, results):
        """JAXNS post-run: get_results is already called in run_sampler."""
        results = self.get_results()
        return results, {"stats": results}

    def get_results(self):
        return self.results

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def load_results(self):
        raw_results_path = self._get_raw_results_path()

        if not os.path.exists(raw_results_path):
            legacy_path = self._get_legacy_raw_results_path()
            if os.path.exists(legacy_path):
                raise RuntimeError(
                    f"Found a JAXNS 2 raw-results file at {legacy_path}. pRT now requires "
                    "JAXNS 3, whose result schema is different and is stored as "
                    f"{os.path.basename(raw_results_path)}. Re-run the retrieval, or delete "
                    "the old file to fall back to the generic equal-weight sample archive."
                )
            return super().load_results()

        _opt.require_jaxns()
        self.raw_results = _opt.NestedSamplerResults.load(raw_results_path)

        parameter_names_path = self._get_parameter_names_path()
        if os.path.exists(parameter_names_path):
            with open(parameter_names_path, "r") as handle:
                self.parameter_names = list(json.load(handle))
        else:
            named_samples = self._get_named_samples_mapping(self.raw_results)
            self.parameter_names = list(named_samples.keys()) if named_samples else []

        samples_path = self._get_equal_weight_samples_path()
        if os.path.exists(samples_path):
            with np.load(samples_path, allow_pickle=False) as archive:
                if "parameter_names" in archive.files:
                    self.parameter_names = [
                        str(name) for name in archive["parameter_names"].tolist()
                    ]

                equal_weight_samples = np.asarray(
                    archive["sample_matrix"],
                    dtype=float,
                )
                equal_weight_log_likelihood = np.asarray(
                    archive["log_likelihood"],
                    dtype=float,
                ).reshape(-1)

            self.results = self._apply_evidence_sidecar(
                self._build_processed_results(
                    self.raw_results,
                    equal_weight_samples=equal_weight_samples,
                    equal_weight_log_likelihood=equal_weight_log_likelihood,
                    output_file=samples_path,
                    params_file=(
                        parameter_names_path
                        if os.path.exists(parameter_names_path)
                        else None
                    ),
                    raw_results_file=raw_results_path,
                    write_sidecar=False,
                )
            )
            return self.results

        self.results = self._apply_evidence_sidecar(
            self._build_processed_results(
                self.raw_results,
                output_file=None,
                params_file=(
                    parameter_names_path if os.path.exists(parameter_names_path) else None
                ),
                raw_results_file=raw_results_path,
                write_sidecar=False,
            )
        )
        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def pretty_print_results(self):
        if self.raw_results is not None:
            # JAXNS 3 moved summary() onto the results object.
            self.raw_results.summary()
            if self.evidence_samples is not None and self.results is not None:
                conditioning = self.results.get("evidence_conditioning", "classic")
                print(
                    f"logZ ({conditioning} MC)="
                    f"{float(np.asarray(self.evidence_samples.log_Z_mean)):.2f} +- "
                    f"{float(np.asarray(self.evidence_samples.log_Z_uncert)):.2f}"
                )
            return

        if self.results is None:
            return

        sample_count = np.asarray(self.results.get("samples", np.array([]))).shape[0]
        print(f"JAXNS completed: {sample_count} posterior samples")

    def plot_diagnostics(self, results=None):
        if self.raw_results is None:
            return {}

        try:
            _opt.require_matplotlib_pyplot()
        except ImportError as error:
            warnings.warn(str(error), RuntimeWarning)
            return {}

        output_directory = self._get_output_directory()
        os.makedirs(output_directory, exist_ok=True)
        plot_files = {}

        # JAXNS 3 plotting functions take an output path and return None, so
        # they save themselves rather than handing back a figure.
        def _plot(name, filename, call):
            path = os.path.join(output_directory, filename)
            try:
                call(path)
                plot_files[name] = path
            except Exception as error:
                warnings.warn(
                    f"JAXNS {name} plotting failed: {error}", RuntimeWarning
                )

        _plot(
            "diagnostics",
            f"{self.retrieval_name}_diagnostics.pdf",
            lambda path: self.raw_results.plot_diagnostics(save_file=path),
        )
        _plot(
            "cornerplot",
            f"{self.retrieval_name}_cornerplot.pdf",
            lambda path: self.raw_results.plot_cornerplot(save_name=path),
        )

        # Only meaningful when phantom states were retained; it is the visual
        # check that phantom conditioning is behaving.
        if int(np.asarray(self.raw_results.log_L_phantom).shape[1]) > 0:
            _plot(
                "evidence",
                f"{self.retrieval_name}_evidence.pdf",
                lambda path: self.raw_results.plot_evidence(
                    num_samples=1024,
                    conditionings=("classic", "phantom"),
                    key=_opt.random.PRNGKey(self.seed + 2),
                    save_name=path,
                ),
            )

        return plot_files

    def get_summary(self, free_parameter_names=None):
        summary = super().get_summary(free_parameter_names=free_parameter_names)
        if self.results is None or not isinstance(self.results, dict):
            return summary

        summary["evidence"] = {
            "ln_z": self._summary_scalar(self.results.get("log_Z_mean")),
            "ln_z_err": self._summary_scalar(self.results.get("log_Z_uncert")),
            "conditioning": self.results.get("evidence_conditioning"),
            "ln_z_expected": self._summary_scalar(
                self.results.get("log_Z_expected_mean")
            ),
            "ln_z_expected_err": self._summary_scalar(
                self.results.get("log_Z_expected_uncert")
            ),
        }
        summary["diagnostics"] = {
            "ess": self._summary_scalar(self.results.get("ESS")),
            "h_mean": self._summary_scalar(self.results.get("H_mean")),
            "total_num_samples": self._summary_scalar(
                self.results.get("total_num_samples")
            ),
            "total_phantom_samples": self._summary_scalar(
                self.results.get("total_phantom_samples")
            ),
            "total_num_likelihood_evaluations": self._summary_scalar(
                self.results.get("total_num_likelihood_evaluations")
            ),
            "log_efficiency": self._summary_scalar(self.results.get("log_efficiency")),
        }
        summary["posterior"] = self._posterior_summary_from_matrix(
            self.results.get("samples"),
            list(free_parameter_names or self.parameter_names),
        )
        if not summary["posterior"] and isinstance(
            self.results.get("weighted_samples"), dict
        ):
            summary["posterior"] = self._posterior_summary_from_matrix(
                self.results["weighted_samples"].get("points"),
                list(free_parameter_names or self.parameter_names),
            )
        return summary
