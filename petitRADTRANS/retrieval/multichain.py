"""Native multi-chain retrieval support.

This module owns running and combining multiple chains of the JAX-based retrieval
samplers (NumPyro, BlackJAX, JAXNS). Importing it is intentionally
dependency-light (pure NumPy, no JAX and no petitRADTRANS import) so that the
combine/diagnostics step can run as a lightweight task, separately from the heavy
sampler processes, and so that it is straightforward to unit test. The
orchestration entry points import ``Retrieval`` lazily, inside the function
bodies, to preserve that.

See ``multichain_plan.md`` (same directory) for the full design. This file
implements:

* shared primitives: deterministic per-chain seeding, chain/combined
  naming, sampler-family and output-directory mapping.
*  the MCMC chain combiner (NumPyro + BlackJAX share one on-disk format)
  and convergence diagnostics (split-R-hat and effective sample size), plus
  :func:`combine_mcmc_chains`.
* process orchestration helpers: CPU partitioning, the ``auto``
  execution policy, memory estimates, and :func:`run_chains_in_processes`
  (spawn pool + affinity + mpi-block).
* the nested (JAXNS) multi-run combiner: :func:`combine_evidence`
  (inverse-variance log Z + between-run scatter) and :func:`combine_nested_chains`
  (evidence-weighted posterior).
* resume helpers (:func:`existing_chain_ids`) and the SLURM job-array
  launcher generator (:func:`render_chain_worker_script`,
  :func:`render_slurm_array_scripts`).
* the high-level entry points that glue the above together:
  :func:`run_multichain`, :func:`run_single_chain`, :func:`combine_chains`,
  :func:`combine_chains_from_builder` and :func:`write_chain_launcher`.
  ``Retrieval`` exposes each of these as a thin delegating shim.
"""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "ChainExecution",
    "derive_chain_seed",
    "chain_retrieval_name",
    "combined_retrieval_name",
    "backend_output_subdir",
    "sampler_family",
    "ChainSpec",
    "MCMCChainArrays",
    "MCMCChainCombiner",
    "CombinedMCMC",
    "combine_mcmc_chains",
    "compute_mcmc_diagnostics",
    # process orchestration
    "partition_cpus",
    "available_memory_gb",
    "estimate_peak_memory_gb",
    "resolve_chain_execution",
    "ProcessRunResult",
    "run_chains_in_processes",
    # nested (JAXNS) multi-run combiner
    "NestedChainArrays",
    "NestedChainCombiner",
    "CombinedNested",
    "combine_evidence",
    "combine_nested_chains",
    # resume + SLURM-array launcher generator
    "chain_artifact_path",
    "existing_chain_ids",
    "render_chain_worker_script",
    "render_slurm_array_scripts",
    # high-level orchestration entry points
    "combine_chains",
    "run_multichain",
    "run_single_chain",
    "combine_chains_from_builder",
    "write_chain_launcher",
]


# ---------------------------------------------------------------------------
# shared primitives
# ---------------------------------------------------------------------------


class ChainExecution:
    """Allowed values for the ``chain_execution`` policy selector."""

    AUTO = "auto"
    VECTORISED = "vectorised"
    SHARDED = "sharded"
    SEQUENTIAL = "sequential"
    PROCESSES = "processes"

    ALL = frozenset({AUTO, VECTORISED, SHARDED, SEQUENTIAL, PROCESSES})

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls.ALL:
            raise ValueError(
                f"Unknown chain_execution {value!r}; expected one of "
                f"{sorted(cls.ALL)}."
            )
        return value


# sampler_type (lower-case) -> backend output sub-directory. Mirrors the paths
# used by the *Sampler._get_output_prefix methods and Retrieval._get_samples_dict.
_BACKEND_SUBDIRS = {
    "numpyronuts": "out_NumPyro",
    "numpyro_nuts": "out_NumPyro",
    "numpyrohmc": "out_NumPyro",
    "numpyro_hmc": "out_NumPyro",
    "blackjaxnuts": "out_BlackJAX",
    "blackjax_nuts": "out_BlackJAX",
    "blackjaxhmc": "out_BlackJAX",
    "blackjax_hmc": "out_BlackJAX",
    "nuts": "out_BlackJAX",
    "hmc": "out_BlackJAX",
    "jaxns": "out_JAXNS",
    "jaxnestedsampler": "out_JAXNS",
    "jaxnsshardedstaticnestedsampler": "out_JAXNS",
}

# sampler_type (lower-case) -> chain-combination family.
_MCMC_TYPES = frozenset(
    {
        "numpyronuts",
        "numpyro_nuts",
        "numpyrohmc",
        "numpyro_hmc",
        "blackjaxnuts",
        "blackjax_nuts",
        "blackjaxhmc",
        "blackjax_hmc",
        "nuts",
        "hmc",
    }
)
_NESTED_TYPES = frozenset({"jaxns", "jaxnsshardedstaticnestedsampler"})


def derive_chain_seed(base_seed: int, chain_id: int) -> int:
    """Return the seed for chain ``chain_id`` given a base seed.

    Deterministic and reproducible from an integer, and shared by *every*
    execution mode so that in-process chain ``c`` and out-of-process chain ``c``
    are identical for the same base seed. Kept deliberately simple (offset by the
    chain index); swap in ``jax.random.fold_in`` here if stronger decorrelation is
    ever required, but keep it the single source of truth.
    """
    return int(base_seed) + int(chain_id)


def chain_retrieval_name(base_name: str, chain_id: int) -> str:
    """Per-chain retrieval name used to namespace on-disk artifacts."""
    return f"{base_name}_chain{int(chain_id)}"


def combined_retrieval_name(base_name: str) -> str:
    """Retrieval name under which the combined posterior is written."""
    return f"{base_name}_combined"


def backend_output_subdir(sampler_type: str) -> str:
    """Return the ``out_<Backend>`` sub-directory for a sampler type."""
    try:
        return _BACKEND_SUBDIRS[sampler_type.lower()]
    except KeyError as error:
        raise ValueError(
            f"No known output sub-directory for sampler_type {sampler_type!r}."
        ) from error


def sampler_family(sampler_type: str) -> str:
    """Return ``'mcmc'``, ``'nested'`` or ``'other'`` for a sampler type."""
    key = sampler_type.lower()
    if key in _MCMC_TYPES:
        return "mcmc"
    if key in _NESTED_TYPES:
        return "nested"
    return "other"


@dataclass(frozen=True)
class ChainSpec:
    """Identity of a single chain within a multi-chain run."""

    chain_id: int
    seed: int
    retrieval_name: str

    @classmethod
    def build(cls, base_name: str, base_seed: int, chain_id: int) -> "ChainSpec":
        return cls(
            chain_id=int(chain_id),
            seed=derive_chain_seed(base_seed, chain_id),
            retrieval_name=chain_retrieval_name(base_name, chain_id),
        )


# ---------------------------------------------------------------------------
# MCMC chain combiner + diagnostics
# ---------------------------------------------------------------------------

# Fields stored in the NumPyro/BlackJAX ``*_samples.npz`` archives. ``samples``
# (transformed) and ``logdensity`` are always present; the rest are optional.
_MCMC_PER_DRAW_FIELDS = (
    "samples",
    "unconstrained_samples",
    "logdensity",
    "model_log_likelihood",
    "acceptance_rate",
    "is_divergent",
    "num_integration_steps",
)


@dataclass
class MCMCChainArrays:
    """One chain's arrays, loaded from a ``*_samples.npz`` archive."""

    chain_id: int
    samples: np.ndarray  # (n_draws, n_dim), transformed/physical space
    unconstrained_samples: Optional[np.ndarray] = None  # (n_draws, n_dim)
    logdensity: Optional[np.ndarray] = None  # (n_draws,)
    model_log_likelihood: Optional[np.ndarray] = None  # (n_draws,)
    acceptance_rate: Optional[np.ndarray] = None  # (n_draws,)
    is_divergent: Optional[np.ndarray] = None  # (n_draws,)
    num_integration_steps: Optional[np.ndarray] = None  # (n_draws,)

    @property
    def n_draws(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.samples.shape[1]) if self.samples.ndim == 2 else 1


def read_mcmc_chain(samples_npz_path: os.PathLike | str, chain_id: int) -> MCMCChainArrays:
    """Load one chain from its ``*_samples.npz`` archive."""
    with np.load(samples_npz_path) as chain:
        present = set(chain.files)
        kwargs = {"chain_id": int(chain_id), "samples": np.asarray(chain["samples"])}
        for name in _MCMC_PER_DRAW_FIELDS:
            if name == "samples":
                continue
            if name in present:
                kwargs[name] = np.asarray(chain[name])
    samples = kwargs["samples"]
    if samples.ndim == 1:
        kwargs["samples"] = samples[:, np.newaxis]
    return MCMCChainArrays(**kwargs)


@dataclass
class CombinedMCMC:
    """Result of combining MCMC chains."""

    parameter_names: list[str]
    pooled: dict[str, np.ndarray]  # keys are a subset of _MCMC_PER_DRAW_FIELDS
    chain_index: np.ndarray  # (n_total,) source chain id per pooled draw
    grouped_for_diagnostics: np.ndarray  # (n_chains, n_common, n_dim)
    n_per_chain: list[int]
    diagnostics: dict

    @property
    def n_total(self) -> int:
        return int(self.pooled["samples"].shape[0])


class MCMCChainCombiner:
    """Combine NumPyro/BlackJAX chains into one pooled posterior + diagnostics."""

    def __init__(self, rhat_threshold: float = 1.01):
        self.rhat_threshold = float(rhat_threshold)

    def combine(
        self,
        chains: Sequence[MCMCChainArrays],
        parameter_names: Sequence[str],
    ) -> CombinedMCMC:
        if len(chains) == 0:
            raise ValueError("No chains to combine.")
        parameter_names = list(parameter_names)
        n_dim = chains[0].n_dim
        for chain in chains:
            if chain.n_dim != n_dim:
                raise ValueError(
                    f"Chain {chain.chain_id} has n_dim={chain.n_dim}, "
                    f"expected {n_dim}."
                )
        if len(parameter_names) != n_dim:
            raise ValueError(
                f"parameter_names has length {len(parameter_names)} but samples "
                f"have n_dim={n_dim}."
            )

        # Pool per-draw fields: only include a field present in *every* chain so
        # the combined archive is internally consistent.
        pooled: dict[str, np.ndarray] = {}
        for name in _MCMC_PER_DRAW_FIELDS:
            values = [getattr(chain, name) for chain in chains]
            if all(value is not None for value in values):
                pooled[name] = np.concatenate(values, axis=0)

        n_per_chain = [chain.n_draws for chain in chains]
        chain_index = np.concatenate(
            [np.full(n, chain.chain_id, dtype=np.int64) for n, chain in zip(n_per_chain, chains)]
        )

        # Diagnostics use the unconstrained samples when available (better
        # geometry); fall back to transformed samples. Truncate all chains to the
        # common minimum length so the grouped array is rectangular.
        use_unconstrained = all(c.unconstrained_samples is not None for c in chains)
        n_common = min(n_per_chain)
        grouped = np.stack(
            [
                (c.unconstrained_samples if use_unconstrained else c.samples)[:n_common]
                for c in chains
            ],
            axis=0,
        )  # (n_chains, n_common, n_dim)

        divergences = None
        if all(c.is_divergent is not None for c in chains):
            per_chain = [int(np.sum(np.asarray(c.is_divergent) != 0)) for c in chains]
            total_div = int(sum(per_chain))
            total_draws = int(sum(n_per_chain))
            divergences = {
                "per_chain": per_chain,
                "total": total_div,
                "fraction": (total_div / total_draws) if total_draws else 0.0,
            }

        diagnostics = compute_mcmc_diagnostics(
            grouped,
            parameter_names,
            divergences=divergences,
            rhat_threshold=self.rhat_threshold,
            n_per_chain=n_per_chain,
        )

        return CombinedMCMC(
            parameter_names=parameter_names,
            pooled=pooled,
            chain_index=chain_index,
            grouped_for_diagnostics=grouped,
            n_per_chain=n_per_chain,
            diagnostics=diagnostics,
        )


# ---- diagnostics: split-R-hat and effective sample size (pure NumPy) ----


def _split_rhat_1d(chains: np.ndarray) -> float:
    """Split-R-hat for one parameter. ``chains`` has shape (n_chains, n_draws)."""
    n_chains, n_draws = chains.shape
    n = n_draws // 2
    if n < 2:
        return float("nan")
    split = np.concatenate([chains[:, :n], chains[:, n : 2 * n]], axis=0)
    m = split.shape[0]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    within = chain_vars.mean()
    if not np.isfinite(within) or within == 0.0:
        return float("nan")
    between = n * chain_means.var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def _autocovariance_1d(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance of a 1-D series via FFT."""
    n = x.shape[0]
    x = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    freq = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(freq * np.conjugate(freq), n=size)[:n]
    return acov / n


def _ess_1d(chains: np.ndarray) -> float:
    """Effective sample size for one parameter (Stan/Geyer estimator).

    ``chains`` has shape (n_chains, n_draws). Combines the autocorrelation across
    chains and truncates with Geyer's initial positive + monotone sequence rule.
    """
    n_chains, n_draws = chains.shape
    if n_draws < 4:
        return float(n_chains * n_draws)
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    within = chain_vars.mean()
    if not np.isfinite(within) or within == 0.0:
        return float(n_chains * n_draws)
    between = n_draws * chain_means.var(ddof=1) if n_chains > 1 else 0.0
    var_plus = (n_draws - 1) / n_draws * within + (
        between / n_draws if n_chains > 1 else 0.0
    )
    if var_plus == 0.0:
        return float(n_chains * n_draws)

    mean_acov = np.mean(
        np.stack([_autocovariance_1d(chains[m]) for m in range(n_chains)], axis=0),
        axis=0,
    )
    rho = 1.0 - (within - mean_acov) / var_plus
    rho[0] = 1.0

    # Geyer initial positive sequence on paired sums, then make it monotone.
    pair_sums = []
    k = 0
    while 2 * k + 1 < n_draws:
        pair = rho[2 * k] + rho[2 * k + 1]
        if k > 0 and pair <= 0.0:
            break
        pair_sums.append(pair)
        k += 1
    for i in range(1, len(pair_sums)):
        if pair_sums[i] > pair_sums[i - 1]:
            pair_sums[i] = pair_sums[i - 1]

    tau = -1.0 + 2.0 * float(np.sum(pair_sums))
    tau = max(tau, 1e-8)
    ess = n_chains * n_draws / tau
    return float(min(ess, float(n_chains * n_draws)))


def compute_mcmc_diagnostics(
    grouped: np.ndarray,
    parameter_names: Sequence[str],
    *,
    divergences: Optional[dict] = None,
    rhat_threshold: float = 1.01,
    n_per_chain: Optional[Sequence[int]] = None,
) -> dict:
    """Compute split-R-hat and ESS per parameter for grouped chains.

    Parameters
    ----------
    grouped:
        Array of shape ``(n_chains, n_draws, n_dim)``.
    parameter_names:
        Names matching the ``n_dim`` axis.
    """
    grouped = np.asarray(grouped)
    if grouped.ndim != 3:
        raise ValueError(
            f"grouped must be (n_chains, n_draws, n_dim); got shape {grouped.shape}."
        )
    n_chains, n_draws, n_dim = grouped.shape
    parameter_names = list(parameter_names)

    per_parameter: dict[str, dict] = {}
    rhats: list[float] = []
    esss: list[float] = []
    can_rhat = n_chains >= 2 and n_draws >= 4
    for i, name in enumerate(parameter_names):
        series = grouped[:, :, i]
        rhat = _split_rhat_1d(series) if can_rhat else float("nan")
        ess = _ess_1d(series)
        per_parameter[name] = {"rhat": rhat, "ess": ess}
        if np.isfinite(rhat):
            rhats.append(rhat)
        if np.isfinite(ess):
            esss.append(ess)

    max_rhat = max(rhats) if rhats else float("nan")
    min_ess = min(esss) if esss else float("nan")
    converged = bool(np.isfinite(max_rhat) and max_rhat <= rhat_threshold)

    diagnostics = {
        "n_chains": int(n_chains),
        "n_samples_diagnostic": int(n_draws),
        "n_samples_per_chain": list(map(int, n_per_chain))
        if n_per_chain is not None
        else [int(n_draws)] * int(n_chains),
        "n_samples_total": int(sum(n_per_chain))
        if n_per_chain is not None
        else int(n_chains * n_draws),
        "rhat_threshold": float(rhat_threshold),
        "max_rhat": float(max_rhat),
        "min_ess": float(min_ess),
        "converged": converged,
        "rhat_available": bool(can_rhat),
        "parameters": per_parameter,
    }
    if divergences is not None:
        diagnostics["divergences"] = divergences
    return diagnostics


# ---- high-level entry point: read per-chain files -> write combined ----


@dataclass
class MCMCCombineSummary:
    combined_name: str
    samples_file: Optional[str]
    params_file: Optional[str]
    diagnostics_file: Optional[str]
    diagnostics: dict
    chain_ids: list[int]
    missing_chain_ids: list[int]


def _discover_chain_files(
    base_dir: Path, base_name: str
) -> list[tuple[int, Path]]:
    """Find ``<base_name>_chain<c>_samples.npz`` files, sorted by chain id."""
    found: list[tuple[int, Path]] = []
    regex = re.compile(rf"{re.escape(base_name)}_chain(\d+)_samples\.npz$")
    for path in glob.glob(str(base_dir / f"{base_name}_chain*_samples.npz")):
        match = regex.search(os.path.basename(path))
        if match:
            found.append((int(match.group(1)), Path(path)))
    found.sort(key=lambda item: item[0])
    return found


def _write_and_summarize_combined_mcmc(
    base_dir: Path,
    base_name: str,
    combined: CombinedMCMC,
    *,
    write: bool,
    chain_ids: list[int],
    missing_chain_ids: list[int],
) -> MCMCCombineSummary:
    """Write the combined posterior + diagnostics and build the summary.

    Shared by the file-based (:func:`combine_mcmc_chains`) and in-process
    (:func:`combine_in_process_mcmc_chains`) combine paths so both emit the exact
    same ``<base_name>_combined`` artifacts.
    """
    combined_name = combined_retrieval_name(base_name)
    samples_file = params_file = diagnostics_file = None
    if write:
        base_dir.mkdir(parents=True, exist_ok=True)
        samples_file = str(base_dir / f"{combined_name}_samples.npz")
        params_file = str(base_dir / f"{combined_name}_params.json")
        diagnostics_file = str(base_dir / f"{combined_name}_diagnostics.json")

        save_kwargs = dict(combined.pooled)
        save_kwargs["chain_index"] = combined.chain_index
        np.savez(samples_file, **save_kwargs)
        with open(params_file, "w", encoding="utf-8") as handle:
            json.dump(list(combined.parameter_names), handle, indent=4)
        with open(diagnostics_file, "w", encoding="utf-8") as handle:
            json.dump(combined.diagnostics, handle, indent=4)

    return MCMCCombineSummary(
        combined_name=combined_name,
        samples_file=samples_file,
        params_file=params_file,
        diagnostics_file=diagnostics_file,
        diagnostics=combined.diagnostics,
        chain_ids=list(chain_ids),
        missing_chain_ids=list(missing_chain_ids),
    )


def combine_mcmc_chains(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    *,
    num_chains: Optional[int] = None,
    write: bool = True,
    on_missing: str = "warn",
    rhat_threshold: float = 1.01,
) -> MCMCCombineSummary:
    """Combine per-chain MCMC artifacts on disk into one pooled posterior.

    Reads ``out_<Backend>/<base_name>_chain<c>_samples.npz`` (+ ``_params.json``),
    pools the draws, computes diagnostics, and (if ``write``) writes the combined
    posterior under ``<base_name>_combined`` in the same on-disk format that
    ``Retrieval._get_samples_dict`` reads, plus a ``_diagnostics.json`` sidecar.

    Parameters
    ----------
    num_chains:
        If given, expect chains ``0..num_chains-1`` (and report any missing).
        If ``None``, discover whatever ``_chain<c>_`` files exist.
    on_missing:
        ``"warn"`` (default) combines the survivors and warns; ``"raise"`` raises
        if any expected chain is missing.
    """
    if sampler_family(sampler_type) != "mcmc":
        raise ValueError(
            f"combine_mcmc_chains only supports MCMC samplers; got {sampler_type!r}."
        )
    base_dir = Path(output_directory) / backend_output_subdir(sampler_type)

    if num_chains is not None:
        expected_ids = list(range(int(num_chains)))
        candidates = [
            (c, base_dir / f"{chain_retrieval_name(base_name, c)}_samples.npz")
            for c in expected_ids
        ]
        available = [(c, p) for c, p in candidates if p.exists()]
        missing = [c for c, p in candidates if not p.exists()]
    else:
        available = _discover_chain_files(base_dir, base_name)
        expected_ids = [c for c, _ in available]
        missing = []

    if missing:
        message = (
            f"Missing chain outputs for chains {missing} under {base_dir} "
            f"(found {[c for c, _ in available]})."
        )
        if on_missing == "raise":
            raise FileNotFoundError(message)
        warnings.warn(message, RuntimeWarning)

    if len(available) == 0:
        raise FileNotFoundError(
            f"No per-chain MCMC outputs found under {base_dir} for base "
            f"name {base_name!r}."
        )

    # Parameter names from the first available chain; require consistency.
    parameter_names: Optional[list[str]] = None
    chain_arrays: list[MCMCChainArrays] = []
    for chain_id, samples_path in available:
        params_path = base_dir / f"{chain_retrieval_name(base_name, chain_id)}_params.json"
        with open(params_path, "r", encoding="utf-8") as handle:
            names = json.load(handle)
        if parameter_names is None:
            parameter_names = list(names)
        elif list(names) != parameter_names:
            raise ValueError(
                f"Parameter mismatch between chain {available[0][0]} and chain "
                f"{chain_id}: {parameter_names} vs {list(names)}."
            )
        chain_arrays.append(read_mcmc_chain(samples_path, chain_id))

    combiner = MCMCChainCombiner(rhat_threshold=rhat_threshold)
    combined = combiner.combine(chain_arrays, parameter_names)

    return _write_and_summarize_combined_mcmc(
        base_dir,
        base_name,
        combined,
        write=write,
        chain_ids=[c for c, _ in available],
        missing_chain_ids=missing,
    )


def split_pooled_mcmc_chains(
    pooled_samples_path: os.PathLike | str,
    num_chains: int,
) -> list[MCMCChainArrays]:
    """Recover per-chain arrays from an in-process pooled ``*_samples.npz``.

    The vectorised/sharded samplers store the draws of all chains in one archive,
    laid out as ``num_chains`` contiguous per-chain blocks (the pooled array is a
    ``(num_chains, n_per_chain, ...) -> (-1, ...)`` reshape). This splits each
    per-draw field's leading axis back into those blocks so the standard
    :class:`MCMCChainCombiner` can compute split-R-hat / ESS across the chains.
    """
    with np.load(pooled_samples_path) as archive:
        present = set(archive.files)
        fields = {
            name: np.asarray(archive[name])
            for name in _MCMC_PER_DRAW_FIELDS
            if name in present
        }
    if "samples" not in fields:
        raise KeyError(f"{pooled_samples_path} has no 'samples' array to split.")

    total = fields["samples"].shape[0]
    if num_chains < 1 or total % num_chains != 0:
        raise ValueError(
            f"cannot split {total} pooled draws into {num_chains} equal chains."
        )
    n_per_chain = total // num_chains

    chains: list[MCMCChainArrays] = []
    for chain_id in range(num_chains):
        start = chain_id * n_per_chain
        stop = start + n_per_chain
        chain_kwargs: dict = {"chain_id": chain_id}
        for name, values in fields.items():
            chunk = values[start:stop]
            if name == "samples" and chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]
            chain_kwargs[name] = chunk
        chains.append(MCMCChainArrays(**chain_kwargs))
    return chains


def combine_in_process_mcmc_chains(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    num_chains: int,
    *,
    write: bool = True,
    rhat_threshold: float = 1.01,
) -> MCMCCombineSummary:
    """Combine an in-process (vectorised/sharded) pooled run into diagnostics.

    The in-process MCMC samplers write a single pooled archive
    ``out_<Backend>/<base_name>_samples.npz`` instead of per-chain files. This
    splits that archive back into ``num_chains`` chains and reuses the standard
    combiner to compute split-R-hat / ESS and write the ``<base_name>_combined``
    posterior + ``_diagnostics.json`` -- matching the on-disk contract of the
    per-chain (sequential/processes) combine path.
    """
    if sampler_family(sampler_type) != "mcmc":
        raise ValueError(
            f"combine_in_process_mcmc_chains only supports MCMC samplers; "
            f"got {sampler_type!r}."
        )

    base_dir = Path(output_directory) / backend_output_subdir(sampler_type)
    pooled_samples_path = base_dir / f"{base_name}_samples.npz"
    params_path = base_dir / f"{base_name}_params.json"
    if not pooled_samples_path.exists():
        raise FileNotFoundError(
            f"No pooled in-process MCMC output found at {pooled_samples_path}."
        )
    with open(params_path, "r", encoding="utf-8") as handle:
        parameter_names = list(json.load(handle))

    chain_arrays = split_pooled_mcmc_chains(pooled_samples_path, num_chains)
    combiner = MCMCChainCombiner(rhat_threshold=rhat_threshold)
    combined = combiner.combine(chain_arrays, parameter_names)

    return _write_and_summarize_combined_mcmc(
        base_dir,
        base_name,
        combined,
        write=write,
        chain_ids=list(range(num_chains)),
        missing_chain_ids=[],
    )


# ---------------------------------------------------------------------------
# process orchestration (spawn pool, affinity, mpi-block, auto)
# ---------------------------------------------------------------------------
#
# This section is intentionally free of any JAX / petitRADTRANS imports so it
# stays import-light and unit-testable. The *chain runner* callable (which does
# import the heavy stack to build and run a Retrieval) is supplied by the caller
# (:func:`run_multichain` below) and only ever executed inside a worker.

import concurrent.futures as _futures
import multiprocessing as _multiprocessing
import sys as _sys


def partition_cpus(cpus: Sequence[int], n_groups: int) -> list[Optional[list[int]]]:
    """Split a CPU id list into ``n_groups`` contiguous, disjoint groups.

    Mirrors the affinity partitioning used by the launcher scripts. The remainder
    is spread over the first groups. If there are fewer CPUs than groups, pinning
    is not meaningful, so every group is ``None`` (let the OS schedule).
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    cpus = sorted(int(c) for c in cpus)
    if len(cpus) < n_groups:
        return [None] * n_groups
    base, remainder = divmod(len(cpus), n_groups)
    groups: list[Optional[list[int]]] = []
    start = 0
    for index in range(n_groups):
        size = base + (1 if index < remainder else 0)
        groups.append(cpus[start : start + size])
        start += size
    return groups


def available_memory_gb() -> Optional[float]:
    """Best-effort available memory in GiB from ``/proc/meminfo``; else ``None``."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kib = float(line.split()[1])
                    return kib / 1024.0 / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def estimate_peak_memory_gb(
    execution: str,
    num_chains: int,
    per_chain_memory_gb: Optional[float],
    max_workers: Optional[int] = None,
) -> Optional[float]:
    """Estimate peak resident memory for a given execution mode.

    ``processes``: ``min(max_workers, num_chains) x per_chain``.
    ``sequential``: ``per_chain`` (one resident model at a time).
    ``vectorised``/``sharded``: ``num_chains x per_chain`` (chains co-resident).
    Returns ``None`` if ``per_chain_memory_gb`` is unknown.
    """
    if per_chain_memory_gb is None:
        return None
    if execution == ChainExecution.SEQUENTIAL:
        concurrent_copies = 1
    elif execution == ChainExecution.PROCESSES:
        concurrent_copies = min(max_workers or num_chains, num_chains)
    else:  # vectorised / sharded keep all chains co-resident in one process
        concurrent_copies = num_chains
    return float(concurrent_copies) * float(per_chain_memory_gb)


def resolve_chain_execution(
    requested: str,
    *,
    num_chains: int,
    num_devices: int = 1,
    gpu_available: bool = False,
    per_chain_memory_gb: Optional[float] = None,
) -> str:
    """Resolve ``auto`` to a concrete execution mode (pure policy).

    Policy (see multichain_plan.md S5):

    * one chain -> ``sequential``;
    * GPU(s) present: ``sharded`` (one chain per device) when
      ``num_chains <= num_devices``, else ``vectorised`` (batch on the devices);
    * CPU only: ``processes`` (real parallelism; forced CPU "devices" share cores
      and do not help).
    """
    requested = ChainExecution.validate(requested)
    if requested != ChainExecution.AUTO:
        return requested
    if num_chains <= 1:
        return ChainExecution.SEQUENTIAL
    if gpu_available:
        if num_chains <= max(1, num_devices):
            return ChainExecution.SHARDED
        return ChainExecution.VECTORISED
    return ChainExecution.PROCESSES


@dataclass
class ProcessRunResult:
    """Outcome of running chains across worker processes."""

    succeeded_chain_ids: list[int]
    failed: dict[int, str] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


def _process_worker_entry(chain_runner, chain_id, cores, block_mpi):
    """Worker-side entry: set affinity, block MPI, then run one chain.

    Module-level (picklable) so it can be dispatched with the ``spawn`` start
    method. ``chain_runner(chain_id, cores)`` is the heavy, caller-supplied
    callable that builds a Retrieval and runs a single chain.
    """
    if block_mpi:
        # These are not MPI ranks; block mpi4py so petitRADTRANS' get_mpi_context
        # takes its no-MPI fallback instead of aborting in MPI_Init. Safe because
        # importing petitRADTRANS does not import mpi4py (it is lazy).
        _sys.modules["mpi4py"] = None
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(int(c) for c in cores))
        except (OSError, ValueError):
            pass
    return chain_runner(chain_id, cores)


def run_chains_in_processes(
    chain_runner,
    chain_ids: Sequence[int],
    cpu_groups: Sequence[Optional[Sequence[int]]],
    *,
    on_chain_failure: str = "warn",
    max_workers: Optional[int] = None,
    block_mpi: bool = True,
    executor_factory=None,
) -> ProcessRunResult:
    """Run one chain per worker process and collect successes/failures.

    Parameters
    ----------
    chain_runner:
        Module-level callable ``chain_runner(chain_id, cores) -> Any`` that builds
        a Retrieval and runs a single chain. Must be picklable (top-level) for the
        default ``spawn`` executor.
    chain_ids, cpu_groups:
        Parallel sequences; ``cpu_groups[i]`` is the core list (or ``None``) for
        ``chain_ids[i]``.
    on_chain_failure:
        ``"warn"`` (default) records the failure and continues; ``"raise"`` raises
        after collecting results.
    executor_factory:
        Optional ``callable(max_workers) -> Executor`` used instead of a spawn
        ``ProcessPoolExecutor`` (for testing with a thread/inline executor).
    """
    if on_chain_failure not in {"warn", "raise"}:
        raise ValueError("on_chain_failure must be 'warn' or 'raise'.")
    chain_ids = list(chain_ids)
    cpu_groups = list(cpu_groups)
    if len(cpu_groups) != len(chain_ids):
        raise ValueError("cpu_groups and chain_ids must have the same length.")
    if max_workers is None:
        max_workers = len(chain_ids)
    max_workers = max(1, min(max_workers, len(chain_ids)))

    if executor_factory is None:
        context = _multiprocessing.get_context("spawn")

        def executor_factory(workers):  # noqa: E306  (local default factory)
            return _futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            )

    result = ProcessRunResult(succeeded_chain_ids=[])
    with executor_factory(max_workers) as executor:
        future_to_chain = {
            executor.submit(
                _process_worker_entry, chain_runner, chain_id, cores, block_mpi
            ): chain_id
            for chain_id, cores in zip(chain_ids, cpu_groups)
        }
        for future in _futures.as_completed(future_to_chain):
            chain_id = future_to_chain[future]
            try:
                future.result()
                result.succeeded_chain_ids.append(int(chain_id))
            except Exception as error:  # noqa: BLE001 -- record and continue/raise
                result.failed[int(chain_id)] = f"{type(error).__name__}: {error}"

    result.succeeded_chain_ids.sort()
    if result.failed and on_chain_failure == "raise":
        raise RuntimeError(
            f"{len(result.failed)} chain(s) failed: {result.failed}"
        )
    return result


# ---------------------------------------------------------------------------
# Phase 3 -- nested-sampling (JAXNS) multi-run combiner
# ---------------------------------------------------------------------------
#
# "Chains" for a nested sampler are independent nested-sampling runs with
# different seeds. Combining them is fundamentally different from MCMC pooling:
# each run is weighted by its evidence Z, the evidences are combined with their
# uncertainties, and the between-run log Z scatter is the headline convergence
# signal. Each run's equal-weight samples live in
# ``out_JAXNS/<name>_samples.npz`` (keys ``sample_matrix``, ``log_likelihood``,
# ``parameter_names``) and its scalar evidence in ``<name>_evidence.json``
# (written by JAXNSSampler._write_evidence_sidecar).

_NESTED_EVIDENCE_KEYS = (
    "log_Z_mean",
    "log_Z_uncert",
    "ESS",
    "total_num_likelihood_evaluations",
    "log_efficiency",
)


@dataclass
class NestedChainArrays:
    """One nested-sampling run: equal-weight samples + scalar evidence."""

    chain_id: int
    sample_matrix: np.ndarray  # (n_samples, n_dim)
    log_likelihood: np.ndarray  # (n_samples,)
    parameter_names: list
    log_Z_mean: Optional[float] = None
    log_Z_uncert: Optional[float] = None
    ess: Optional[float] = None
    total_num_likelihood_evaluations: Optional[float] = None
    log_efficiency: Optional[float] = None

    @property
    def n_samples(self) -> int:
        return int(self.sample_matrix.shape[0])

    @property
    def n_dim(self) -> int:
        return int(self.sample_matrix.shape[1]) if self.sample_matrix.ndim == 2 else 1


def read_nested_chain(
    samples_npz_path: os.PathLike | str,
    chain_id: int,
    evidence_json_path: Optional[os.PathLike | str] = None,
) -> NestedChainArrays:
    """Load one nested run's equal-weight samples and (optional) evidence JSON."""
    with np.load(samples_npz_path, allow_pickle=False) as archive:
        if "sample_matrix" in archive.files:
            sample_matrix = np.asarray(archive["sample_matrix"], dtype=float)
        elif "samples" in archive.files:
            sample_matrix = np.asarray(archive["samples"], dtype=float)
        else:
            raise FileNotFoundError(
                f"Nested sample file {samples_npz_path} has no posterior samples."
            )
        if sample_matrix.ndim == 1:
            sample_matrix = sample_matrix.reshape(-1, 1)
        if "log_likelihood" in archive.files:
            log_likelihood = np.asarray(archive["log_likelihood"], dtype=float).reshape(-1)
        elif "weighted_log_likelihood" in archive.files:
            log_likelihood = np.asarray(
                archive["weighted_log_likelihood"], dtype=float
            ).reshape(-1)
        else:
            log_likelihood = np.full(sample_matrix.shape[0], np.nan)
        if "parameter_names" in archive.files:
            parameter_names = [str(name) for name in archive["parameter_names"]]
        else:
            parameter_names = []

    evidence = {}
    if evidence_json_path is not None and os.path.exists(evidence_json_path):
        with open(evidence_json_path, "r", encoding="utf-8") as handle:
            evidence = json.load(handle)

    def _scalar(key):
        value = evidence.get(key)
        return float(value) if value is not None else None

    return NestedChainArrays(
        chain_id=int(chain_id),
        sample_matrix=sample_matrix,
        log_likelihood=log_likelihood,
        parameter_names=parameter_names,
        log_Z_mean=_scalar("log_Z_mean"),
        log_Z_uncert=_scalar("log_Z_uncert"),
        ess=_scalar("ESS"),
        total_num_likelihood_evaluations=_scalar("total_num_likelihood_evaluations"),
        log_efficiency=_scalar("log_efficiency"),
    )


def combine_evidence(
    log_z: Sequence[Optional[float]],
    log_z_uncert: Sequence[Optional[float]],
) -> dict:
    """Combine independent log-evidence estimates of the same posterior.

    Uses inverse-variance weighting in log space when per-run uncertainties are
    available (falling back to an unweighted mean otherwise), and reports a
    combined uncertainty that is the larger of the formal statistical error and
    the between-run scatter (standard error of the mean). The between-run scatter
    is the key signal: if runs disagree by more than their individual error bars,
    it dominates and flags non-convergence.
    """
    log_z = list(log_z)
    log_z_uncert = list(log_z_uncert)
    values = np.array(
        [z for z in log_z if z is not None and np.isfinite(z)], dtype=float
    )
    if values.size == 0:
        return {"available": False, "n_runs": 0}

    uncerts = np.array(
        [
            u
            for z, u in zip(log_z, log_z_uncert)
            if z is not None and np.isfinite(z)
        ],
        dtype=object,
    )
    have_uncert = all(
        (u is not None and np.isfinite(u) and u > 0) for u in uncerts
    )

    if have_uncert:
        sigma = np.array([float(u) for u in uncerts], dtype=float)
        weights = 1.0 / sigma ** 2
        log_z_mean = float(np.sum(weights * values) / np.sum(weights))
        sigma_stat = float(np.sqrt(1.0 / np.sum(weights)))
    else:
        log_z_mean = float(np.mean(values))
        sigma_stat = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size >= 2
            else float("nan")
        )

    if values.size >= 2:
        between_std = float(np.std(values, ddof=1))
        sem_between = between_std / np.sqrt(values.size)
    else:
        between_std = float("nan")
        sem_between = float("nan")

    candidates = [s for s in (sigma_stat, sem_between) if np.isfinite(s)]
    log_z_uncert_combined = float(max(candidates)) if candidates else float("nan")

    return {
        "available": True,
        "n_runs": int(values.size),
        "log_Z_mean": log_z_mean,
        "log_Z_uncert": log_z_uncert_combined,
        "log_Z_stat_uncert": sigma_stat,
        "log_Z_between_std": between_std,
        "per_run_log_Z": [None if z is None else float(z) for z in log_z],
        "per_run_log_Z_uncert": [
            None if u is None else float(u) for u in log_z_uncert
        ],
        "used_inverse_variance": bool(have_uncert),
    }


def _evidence_run_weights(chains: Sequence[NestedChainArrays]) -> np.ndarray:
    """Relative per-run weights proportional to evidence Z (overflow-safe).

    Runs missing a log Z fall back to equal weight among themselves.
    """
    log_z = np.array(
        [c.log_Z_mean if c.log_Z_mean is not None else np.nan for c in chains],
        dtype=float,
    )
    if np.all(np.isnan(log_z)):
        return np.ones(len(chains), dtype=float)
    # Fill missing log Z with the mean of the available ones (neutral choice).
    fill = np.nanmean(log_z)
    log_z = np.where(np.isnan(log_z), fill, log_z)
    return np.exp(log_z - np.max(log_z))


@dataclass
class CombinedNested:
    parameter_names: list
    sample_matrix: np.ndarray  # (n_total, n_dim), equal weight
    log_likelihood: np.ndarray  # (n_total,)
    evidence: dict
    diagnostics: dict


class NestedChainCombiner:
    """Combine independent JAXNS runs into one evidence-weighted posterior."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def combine(self, chains: Sequence[NestedChainArrays]) -> CombinedNested:
        chains = [c for c in chains if c.n_samples > 0]
        if len(chains) == 0:
            raise ValueError("No nested-sampling runs with samples to combine.")

        parameter_names = list(chains[0].parameter_names)
        n_dim = chains[0].n_dim
        for chain in chains:
            if chain.n_dim != n_dim:
                raise ValueError(
                    f"Run {chain.chain_id} has n_dim={chain.n_dim}, expected {n_dim}."
                )
            if parameter_names and chain.parameter_names and list(
                chain.parameter_names
            ) != parameter_names:
                raise ValueError(
                    f"Parameter mismatch between run {chains[0].chain_id} and run "
                    f"{chain.chain_id}."
                )

        # Per-sample weights: run weight (proportional to evidence) split equally
        # over that run's equal-weight draws.
        run_weights = _evidence_run_weights(chains)
        per_sample_weights = []
        for chain, run_weight in zip(chains, run_weights):
            per_sample_weights.append(
                np.full(chain.n_samples, run_weight / chain.n_samples, dtype=float)
            )
        weights = np.concatenate(per_sample_weights)
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.ones_like(weights)
            weight_sum = weights.sum()
        weights = weights / weight_sum

        pooled_samples = np.concatenate([c.sample_matrix for c in chains], axis=0)
        pooled_logl = np.concatenate([c.log_likelihood for c in chains])

        # Resample to an equal-weight posterior of the same total size.
        n_total = pooled_samples.shape[0]
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n_total, size=n_total, replace=True, p=weights)
        sample_matrix = pooled_samples[idx]
        log_likelihood = pooled_logl[idx]

        # Preserve the global best-fit sample at index 0 (JAXNS convention).
        if np.any(np.isfinite(pooled_logl)):
            best = int(np.nanargmax(pooled_logl))
            sample_matrix[0] = pooled_samples[best]
            log_likelihood[0] = pooled_logl[best]

        evidence = combine_evidence(
            [c.log_Z_mean for c in chains],
            [c.log_Z_uncert for c in chains],
        )

        diagnostics = {
            "n_runs": len(chains),
            "n_samples_per_run": [c.n_samples for c in chains],
            "n_samples_total": int(n_total),
            "evidence": evidence,
            "log_Z_between_std": evidence.get("log_Z_between_std"),
            "per_run": [
                {
                    "chain_id": c.chain_id,
                    "log_Z_mean": c.log_Z_mean,
                    "log_Z_uncert": c.log_Z_uncert,
                    "ess": c.ess,
                    "n_samples": c.n_samples,
                    "total_num_likelihood_evaluations": c.total_num_likelihood_evaluations,
                    "log_efficiency": c.log_efficiency,
                }
                for c in chains
            ],
        }
        # Heuristic consistency flag: runs agree to within their error bars.
        between = evidence.get("log_Z_between_std")
        per_run_uncert = [
            c.log_Z_uncert for c in chains if c.log_Z_uncert is not None
        ]
        if between is not None and np.isfinite(between) and per_run_uncert:
            scale = max(3.0 * float(np.mean(per_run_uncert)), 0.5)
            diagnostics["evidence_consistent"] = bool(between <= scale)
        else:
            diagnostics["evidence_consistent"] = None

        return CombinedNested(
            parameter_names=parameter_names,
            sample_matrix=sample_matrix,
            log_likelihood=log_likelihood,
            evidence=evidence,
            diagnostics=diagnostics,
        )


@dataclass
class NestedCombineSummary:
    combined_name: str
    samples_file: Optional[str]
    params_file: Optional[str]
    evidence_file: Optional[str]
    diagnostics_file: Optional[str]
    evidence: dict
    diagnostics: dict
    chain_ids: list
    missing_chain_ids: list


def combine_nested_chains(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    *,
    num_chains: Optional[int] = None,
    write: bool = True,
    on_missing: str = "warn",
    seed: int = 0,
) -> NestedCombineSummary:
    """Combine per-run JAXNS artifacts on disk into one evidence-weighted posterior.

    Reads ``out_JAXNS/<base>_chain<c>_samples.npz`` (+ ``_evidence.json``), weights
    each run by its evidence, resamples to an equal-weight posterior, and writes
    ``<base>_combined`` in the JAXNS on-disk format that ``get_samples`` reads,
    plus ``<base>_combined_evidence.json`` and ``<base>_combined_diagnostics.json``.
    """
    if sampler_family(sampler_type) != "nested":
        raise ValueError(
            f"combine_nested_chains only supports nested samplers; got {sampler_type!r}."
        )
    base_dir = Path(output_directory) / backend_output_subdir(sampler_type)

    if num_chains is not None:
        candidates = [
            (c, base_dir / f"{chain_retrieval_name(base_name, c)}_samples.npz")
            for c in range(int(num_chains))
        ]
        available = [(c, p) for c, p in candidates if p.exists()]
        missing = [c for c, p in candidates if not p.exists()]
    else:
        regex = re.compile(rf"{re.escape(base_name)}_chain(\d+)_samples\.npz$")
        available = []
        for path in glob.glob(str(base_dir / f"{base_name}_chain*_samples.npz")):
            match = regex.search(os.path.basename(path))
            if match:
                available.append((int(match.group(1)), Path(path)))
        available.sort(key=lambda item: item[0])
        missing = []

    if missing:
        message = (
            f"Missing nested-run outputs for runs {missing} under {base_dir}."
        )
        if on_missing == "raise":
            raise FileNotFoundError(message)
        warnings.warn(message, RuntimeWarning)
    if len(available) == 0:
        raise FileNotFoundError(
            f"No per-run nested outputs found under {base_dir} for base name "
            f"{base_name!r}."
        )

    chains = []
    for chain_id, samples_path in available:
        evidence_path = (
            base_dir / f"{chain_retrieval_name(base_name, chain_id)}_evidence.json"
        )
        chains.append(read_nested_chain(samples_path, chain_id, evidence_path))

    combiner = NestedChainCombiner(seed=seed)
    combined = combiner.combine(chains)

    combined_name = combined_retrieval_name(base_name)
    samples_file = params_file = evidence_file = diagnostics_file = None
    if write:
        base_dir.mkdir(parents=True, exist_ok=True)
        samples_file = str(base_dir / f"{combined_name}_samples.npz")
        params_file = str(base_dir / f"{combined_name}_params.json")
        evidence_file = str(base_dir / f"{combined_name}_evidence.json")
        diagnostics_file = str(base_dir / f"{combined_name}_diagnostics.json")

        np.savez(
            samples_file,
            parameter_names=np.asarray(combined.parameter_names, dtype=str),
            sample_matrix=combined.sample_matrix,
            log_likelihood=combined.log_likelihood,
        )
        with open(params_file, "w", encoding="utf-8") as handle:
            json.dump(list(combined.parameter_names), handle, indent=4)
        with open(evidence_file, "w", encoding="utf-8") as handle:
            json.dump(combined.evidence, handle, indent=4)
        with open(diagnostics_file, "w", encoding="utf-8") as handle:
            json.dump(combined.diagnostics, handle, indent=4)

    return NestedCombineSummary(
        combined_name=combined_name,
        samples_file=samples_file,
        params_file=params_file,
        evidence_file=evidence_file,
        diagnostics_file=diagnostics_file,
        evidence=combined.evidence,
        diagnostics=combined.diagnostics,
        chain_ids=[c for c, _ in available],
        missing_chain_ids=missing,
    )


# ---------------------------------------------------------------------------
# resume + SLURM-array launcher generator
# ---------------------------------------------------------------------------


def chain_artifact_path(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    chain_id: int,
) -> str:
    """Path of the per-chain ``*_samples.npz`` artifact (used for resume checks)."""
    subdir = backend_output_subdir(sampler_type)
    name = chain_retrieval_name(base_name, chain_id)
    return str(Path(output_directory) / subdir / f"{name}_samples.npz")


def existing_chain_ids(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    num_chains: int,
) -> list[int]:
    """Chain ids in ``0..num_chains-1`` whose per-chain artifact already exists."""
    return [
        c
        for c in range(int(num_chains))
        if os.path.exists(
            chain_artifact_path(output_directory, sampler_type, base_name, c)
        )
    ]


def render_chain_worker_script(
    builder_module: str,
    builder_function: str,
    output_directory: str,
    num_chains: int,
    chain_seed: int,
    retrieval_kwargs: Optional[dict],
    run_kwargs: Optional[dict],
    resume: bool,
) -> str:
    """Render the auto-generated per-task worker script.

    The worker runs one chain (chain id from ``SLURM_ARRAY_TASK_ID`` or argv) or,
    in ``combine`` mode, merges the chains. ``builder_module``/``builder_function``
    must be importable (not ``__main__``); ``retrieval_kwargs``/``run_kwargs`` must
    be Python literals (load arrays inside the builder, not here).
    """
    if builder_module in (None, "__main__"):
        raise ValueError(
            "The config builder must live in an importable module (not __main__) "
            "to generate a SLURM launcher. Move it to a .py module or pass "
            "builder_module/builder_function explicitly."
        )
    return f'''#!/usr/bin/env python
"""Auto-generated multi-chain worker. Do not edit by hand.

Run one chain:   SLURM_ARRAY_TASK_ID=<c> python {{this_file}} run
Combine chains:  python {{this_file}} combine
"""
import os
import sys

# JAX-sampler workers are independent processes, not MPI ranks: block mpi4py so
# petitRADTRANS' get_mpi_context() takes its no-MPI fallback.
sys.modules["mpi4py"] = None

from {builder_module} import {builder_function} as config_builder
from petitRADTRANS.retrieval import multichain

OUTPUT_DIRECTORY = {output_directory!r}
NUM_CHAINS = {int(num_chains)}
CHAIN_SEED = {int(chain_seed)}
RETRIEVAL_KWARGS = {retrieval_kwargs or {}!r}
RUN_KWARGS = {run_kwargs or {}!r}
RESUME = {bool(resume)!r}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "combine":
        multichain.combine_chains_from_builder(
            config_builder,
            output_directory=OUTPUT_DIRECTORY,
            num_chains=NUM_CHAINS,
        )
        return

    chain_id = int(
        os.environ.get(
            "SLURM_ARRAY_TASK_ID", sys.argv[2] if len(sys.argv) > 2 else 0
        )
    )
    multichain.run_single_chain(
        config_builder,
        chain_id,
        output_directory=OUTPUT_DIRECTORY,
        num_chains=NUM_CHAINS,
        chain_seed=CHAIN_SEED,
        resume=RESUME,
        retrieval_kwargs=RETRIEVAL_KWARGS,
        **RUN_KWARGS,
    )


if __name__ == "__main__":
    main()
'''


def _render_sbatch_header(
    *,
    job_name: str,
    log_dir: str,
    array: Optional[str],
    cpus_per_task: int,
    partition: Optional[str],
    time: Optional[str],
    mem: Optional[str],
    gres: Optional[str],
    account: Optional[str],
    extra_sbatch_lines: Sequence[str],
    env_setup_lines: Sequence[str],
) -> str:
    lines = ["#!/bin/bash -l", f"#SBATCH -J {job_name}"]
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
        lines.append(f"#SBATCH -o {log_dir}/%x_%A_%a.out")
        lines.append(f"#SBATCH -e {log_dir}/%x_%A_%a.err")
    else:
        lines.append(f"#SBATCH -o {log_dir}/%x_%j.out")
        lines.append(f"#SBATCH -e {log_dir}/%x_%j.err")
    lines.append("#SBATCH --nodes=1")
    lines.append("#SBATCH --ntasks-per-node=1")
    lines.append(f"#SBATCH --cpus-per-task={int(cpus_per_task)}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    else:
        lines.append("# #SBATCH --partition=<TODO: set your partition>")
    if time:
        lines.append(f"#SBATCH --time={time}")
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    for extra in extra_sbatch_lines:
        lines.append(extra if extra.startswith("#SBATCH") else f"#SBATCH {extra}")
    lines.append("")
    lines.extend(env_setup_lines)
    if env_setup_lines:
        lines.append("")
    return "\n".join(lines)


def render_slurm_array_scripts(
    *,
    base_name: str,
    worker_filename: str,
    num_chains: int,
    python: str = "python",
    cpus_per_task: int = 8,
    partition: Optional[str] = None,
    time: Optional[str] = "24:00:00",
    mem: Optional[str] = None,
    gres: Optional[str] = None,
    account: Optional[str] = None,
    log_dir: str = "out",
    job_name: Optional[str] = None,
    combine_dependency: str = "afterany",
    extra_sbatch_lines: Sequence[str] = (),
    env_setup_lines: Sequence[str] = (),
) -> dict[str, str]:
    """Render the SLURM array sbatch, the combine sbatch, and the submit driver.

    Returns a mapping ``{filename: content}``. The array job runs one chain per
    task (``SLURM_ARRAY_TASK_ID``); the combine job depends on it
    (``afterany`` by default, so survivors are merged even if some tasks fail).
    """
    job_name = job_name or f"{base_name}_chains"
    array_header = _render_sbatch_header(
        job_name=job_name,
        log_dir=log_dir,
        array=f"0-{int(num_chains) - 1}",
        cpus_per_task=cpus_per_task,
        partition=partition,
        time=time,
        mem=mem,
        gres=gres,
        account=account,
        extra_sbatch_lines=extra_sbatch_lines,
        env_setup_lines=env_setup_lines,
    )
    combine_header = _render_sbatch_header(
        job_name=f"{job_name}_combine",
        log_dir=log_dir,
        array=None,
        cpus_per_task=cpus_per_task,
        partition=partition,
        time=time,
        mem=mem,
        gres=None,  # combine is light; no GPU needed
        account=account,
        extra_sbatch_lines=extra_sbatch_lines,
        env_setup_lines=env_setup_lines,
    )

    array_sbatch = f"{array_header}srun {python} {worker_filename} run\n"
    combine_sbatch = f"{combine_header}{python} {worker_filename} combine\n"

    array_file = f"{base_name}_chains.sbatch"
    combine_file = f"{base_name}_combine.sbatch"
    submit_sh = f'''#!/bin/bash -l
# Auto-generated submit driver for '{base_name}'. Submits the per-chain array job
# and a dependent combine job. Run:  bash {base_name}_submit.sh
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$HERE"
mkdir -p {log_dir}

ARRAY_JOB=$(sbatch --parsable {array_file})
echo "submitted chain array job: $ARRAY_JOB ({int(num_chains)} chains)"

COMBINE_JOB=$(sbatch --parsable --dependency={combine_dependency}:$ARRAY_JOB {combine_file})
echo "submitted combine job: $COMBINE_JOB (dependency {combine_dependency}:$ARRAY_JOB)"
'''

    return {
        array_file: array_sbatch,
        combine_file: combine_sbatch,
        f"{base_name}_submit.sh": submit_sh,
    }


# ---------------------------------------------------------------------------
# high-level orchestration (driven by Retrieval, or called directly)
# ---------------------------------------------------------------------------
#
# These are the entry points a user actually calls. They glue the primitives
# above into complete workflows and are the only functions in this module that
# know about ``Retrieval`` -- and they only import it *inside* the function
# bodies, so importing this module stays cheap and free of JAX/petitRADTRANS.
# ``Retrieval`` mirrors each of them with a thin delegating shim.

import functools as _functools


def _detect_jax_devices() -> tuple[int, bool]:
    """Return ``(num_devices, gpu_available)`` from JAX, or ``(1, False)``."""
    try:
        import jax

        devices = jax.devices()
        gpu_available = any(
            getattr(device, "platform", "cpu") in ("gpu", "cuda")
            for device in devices
        )
        return len(devices), gpu_available
    except Exception:  # noqa: BLE001 -- JAX missing / not initialised
        return 1, False


def _validate_top_level_builder(builder) -> None:
    """Raise a clear error if ``builder`` cannot be used with the spawn pool.

    The ``processes`` execution mode dispatches the builder to ``spawn`` workers,
    which pickle it by reference. Closures and lambdas are not picklable; the
    builder must be a top-level (module-level) function.
    """
    if not callable(builder):
        raise TypeError("config_builder must be callable.")
    qualname = getattr(builder, "__qualname__", "")
    if "<locals>" in qualname or "<lambda>" in qualname or "." in qualname:
        raise ValueError(
            "config_builder must be a top-level (module-level) function so it can "
            "be sent to spawned worker processes; got a "
            f"closure/lambda/method ({qualname!r}). Define it at module scope and "
            "call run_multichain under `if __name__ == \"__main__\":`."
        )


def _run_single_chain_on(retrieval, chain_spec, run_kwargs, force_single_chain=True) -> str:
    """Run one chain on an existing Retrieval, namespaced by ``chain_spec``.

    ``force_single_chain`` adds ``num_chains=1`` (the gradient-MCMC samplers accept
    it). Nested samplers (JAXNS) do not take ``num_chains``, so it is left off for
    them -- each run is one independent nested run distinguished only by its seed.
    """
    retrieval.configuration.retrieval_name = chain_spec.retrieval_name
    per_chain_kwargs = dict(run_kwargs)
    per_chain_kwargs.pop("chain_execution", None)
    per_chain_kwargs.pop("num_chains", None)
    if force_single_chain:
        per_chain_kwargs["num_chains"] = 1
    per_chain_kwargs["seed"] = chain_spec.seed
    retrieval.run(**per_chain_kwargs)
    return chain_spec.retrieval_name


def _run_one_chain_from_builder(
    config_builder,
    output_directory,
    retrieval_kwargs,
    base_name,
    base_seed,
    run_kwargs,
    force_single_chain,
    chain_id,
    cores,
):
    """Worker entry: build a Retrieval from ``config_builder`` and run one chain.

    Module-level + only picklable arguments so it can be dispatched to a ``spawn``
    worker. ``cores`` is accepted for signature compatibility (affinity is applied
    by :func:`_process_worker_entry` before this runs).
    """
    from petitRADTRANS.retrieval.retrieval import Retrieval

    spec = ChainSpec.build(base_name, base_seed, chain_id)
    configuration = config_builder()
    retrieval = Retrieval(
        configuration,
        output_directory=output_directory,
        **(retrieval_kwargs or {}),
    )
    return _run_single_chain_on(retrieval, spec, run_kwargs, force_single_chain)


def _report_combine_summary(summary, family: str) -> None:
    """Print the human-readable summary of a :func:`combine_chains` result."""
    diagnostics = summary.diagnostics

    if family == "nested":
        evidence = summary.evidence
        print(
            f"Combined {len(summary.chain_ids)} nested runs "
            f"({summary.chain_ids}) into '{summary.combined_name}': "
            f"{diagnostics['n_samples_total']} samples."
        )
        if evidence.get("available"):
            print(
                f"  log Z = {evidence['log_Z_mean']:.3f} +/- "
                f"{evidence['log_Z_uncert']:.3f}; between-run std = "
                f"{evidence.get('log_Z_between_std', float('nan')):.3f}; "
                f"consistent = {diagnostics.get('evidence_consistent')}"
            )
        else:
            print("  evidence unavailable (no per-run evidence sidecars).")
        return

    print(
        f"Combined {len(summary.chain_ids)} chains "
        f"({summary.chain_ids}) into '{summary.combined_name}': "
        f"{diagnostics['n_samples_total']} samples."
    )
    if diagnostics.get("rhat_available"):
        print(
            f"  max R-hat = {diagnostics['max_rhat']:.4f} "
            f"(threshold {diagnostics['rhat_threshold']}); "
            f"min ESS = {diagnostics['min_ess']:.0f}; "
            f"converged = {diagnostics['converged']}"
        )
    else:
        print("  R-hat unavailable (need >= 2 chains and >= 4 draws).")
    if diagnostics.get("divergences") is not None:
        print(
            f"  divergences: {diagnostics['divergences']['total']} "
            f"({100.0 * diagnostics['divergences']['fraction']:.2f}%)"
        )


def combine_chains(
    output_directory: os.PathLike | str,
    sampler_type: str,
    base_name: str,
    *,
    num_chains: Optional[int] = None,
    on_missing: str = "warn",
    rhat_threshold: float = 1.01,
    verbose: bool = True,
):
    """Combine per-chain outputs into one posterior + diagnostics, by family.

    Dispatches to :func:`combine_mcmc_chains` (NumPyro, BlackJAX: pooled draws with
    split-R-hat / ESS / divergences) or :func:`combine_nested_chains` (JAXNS:
    evidence-weighted posterior with between-run log Z scatter), then optionally
    prints a summary. Reads
    ``out_<Backend>/<base_name>_chain<c>_samples.npz`` and writes
    ``<base_name>_combined`` in the same on-disk format ``Retrieval.get_samples``
    reads, plus a ``<base_name>_combined_diagnostics.json`` sidecar.

    Needs no ``Retrieval``: this is pure NumPy + file I/O, so it is safe to call as
    a lightweight standalone task (e.g. a dependent SLURM combine job).

    Parameters
    ----------
    num_chains:
        If given, expect chains ``0..num_chains-1`` and report any missing. If
        ``None``, combine whatever ``_chain<c>_`` files are present.
    on_missing:
        ``"warn"`` (default) combines the survivors and warns; ``"raise"`` raises
        if any expected chain is missing.
    rhat_threshold:
        R-hat value above which an MCMC run is flagged as not converged.
    """
    family = sampler_family(sampler_type)

    if family == "nested":
        summary = combine_nested_chains(
            output_directory,
            sampler_type,
            base_name,
            num_chains=num_chains,
            write=True,
            on_missing=on_missing,
        )
    elif family == "mcmc":
        summary = combine_mcmc_chains(
            output_directory,
            sampler_type,
            base_name,
            num_chains=num_chains,
            write=True,
            on_missing=on_missing,
            rhat_threshold=rhat_threshold,
        )
    else:
        raise NotImplementedError(
            f"combine_chains supports gradient-MCMC (NumPyro, BlackJAX) and "
            f"nested (JAXNS) samplers; sampler_type {sampler_type!r} (family "
            f"{family!r}) is not supported."
        )

    if verbose:
        _report_combine_summary(summary, family)
    return summary


def run_multichain(
    config_builder,
    *,
    num_chains: int,
    output_directory: str,
    retrieval_kwargs: Optional[dict] = None,
    chain_execution: str = "auto",
    chain_seed: int = 12345,
    per_chain_memory_gb: Optional[float] = None,
    on_chain_failure: str = "warn",
    combine: bool = True,
    rhat_threshold: float = 1.01,
    max_workers: Optional[int] = None,
    sampler_type: Optional[str] = None,
    base_retrieval_name: Optional[str] = None,
    resume: bool = False,
    verbose: bool = True,
    **run_kwargs,
) -> dict:
    """Run a multi-chain retrieval and (optionally) combine the chains.

    This is the native entry point for multi-chain JAX-sampler runs, such as
    when using NumPyro or BlackJAX. It keeps a single, simple parent process
    to coordinate the retrieval, while the children build the runtime object
    and perform the sampling, with one chain per process. At completion, the
    parent process collects the results and merges the per-chain posteriors
    into one combined result with convergence diagnostics.

    Parameters
    ----------
    config_builder:
        A **top-level (module-level)** callable returning a fresh
        ``RetrievalConfig``. For the ``processes`` mode it is dispatched to
        ``spawn`` workers (so it must be picklable by reference: no closures or
        lambdas), and the call site must be guarded by
        ``if __name__ == "__main__":``. It should be cheap (it is also called
        once in the parent to read the sampler type and retrieval name unless
        those are given explicitly).
    num_chains:
        Number of independent chains.
    output_directory:
        Output directory passed to each per-chain ``Retrieval``.
    retrieval_kwargs:
        Extra keyword arguments for the ``Retrieval(config, ...)`` constructor.
    chain_execution:
        ``auto`` | ``vectorised`` | ``sharded`` | ``sequential`` | ``processes``.
        ``auto`` chooses based on devices and ``num_chains`` (see
        :func:`resolve_chain_execution`).
    chain_seed:
        Base seed; chain ``c`` uses ``derive_chain_seed(chain_seed, c)``.
    per_chain_memory_gb:
        Resident memory per chain, used for the printed peak-memory estimate
        and to cap process concurrency.
    on_chain_failure:
        ``"warn"`` (combine survivors) or ``"raise"``.
    combine:
        If ``True``, combine the per-chain posteriors and compute diagnostics
        after the chains finish.

    Returns
    -------
    dict with keys ``execution``, ``num_chains``, ``succeeded_chain_ids``,
    ``failed`` (chain_id -> error), ``combine`` (the combine summary or
    ``None``), and ``diagnostics`` (or ``None``).
    """
    from petitRADTRANS.retrieval.retrieval import Retrieval

    if int(num_chains) < 1:
        raise ValueError("num_chains must be >= 1.")
    num_chains = int(num_chains)
    chain_execution = ChainExecution.validate(chain_execution)
    if on_chain_failure not in {"warn", "raise"}:
        raise ValueError("on_chain_failure must be 'warn' or 'raise'.")

    # Light introspection: read sampler type + base name without building the
    # heavy runtime (RetrievalConfig does not load opacities).
    if sampler_type is None or base_retrieval_name is None:
        probe_config = config_builder()
        if sampler_type is None:
            sampler_type = probe_config.sampler_type
        if base_retrieval_name is None:
            base_retrieval_name = probe_config.retrieval_name
    base_name = base_retrieval_name
    family = sampler_family(sampler_type)
    # Gradient-MCMC samplers take num_chains=1 per worker; nested samplers
    # (JAXNS) do not -- each worker is one independent nested run.
    force_single_chain = family == "mcmc"

    num_devices, gpu_available = _detect_jax_devices()
    execution = resolve_chain_execution(
        chain_execution,
        num_chains=num_chains,
        num_devices=num_devices,
        gpu_available=gpu_available,
        per_chain_memory_gb=per_chain_memory_gb,
    )

    peak_gb = estimate_peak_memory_gb(
        execution, num_chains, per_chain_memory_gb, max_workers=max_workers
    )
    if verbose:
        print(
            f"run_multichain: sampler='{sampler_type}' chains={num_chains} "
            f"execution='{execution}' (requested '{chain_execution}'); "
            f"devices={num_devices} gpu={gpu_available}"
        )
        if peak_gb is not None:
            available = available_memory_gb()
            suffix = (
                f"; node MemAvailable ~= {available:.0f}GB" if available else ""
            )
            print(f"  estimated peak resident memory ~= {peak_gb:.0f}GB{suffix}")
            if available is not None and peak_gb > available:
                warnings.warn(
                    f"Estimated peak memory {peak_gb:.0f}GB exceeds available "
                    f"{available:.0f}GB; reduce num_chains or per-worker load.",
                    RuntimeWarning,
                )

    succeeded_ids: list[int] = []
    failed: dict[int, str] = {}

    # Resume: chains whose per-chain artifact already exists are treated as
    # done and skipped (only meaningful for the per-chain modes).
    already_done: list[int] = []
    if resume and family in ("mcmc", "nested"):
        already_done = existing_chain_ids(
            output_directory, sampler_type, base_name, num_chains
        )
        if already_done and verbose:
            print(f"  resume: {len(already_done)} chains already done {already_done}.")
    chains_to_run = [c for c in range(num_chains) if c not in set(already_done)]

    if execution in (ChainExecution.VECTORISED, ChainExecution.SHARDED):
        # In-process device-parallel run: the sampler pools the chains itself
        # into one archive. We then split that pooled archive back into its
        # per-chain blocks to compute split-R-hat / ESS and write the same
        # <base_name>_combined artifacts as the per-chain paths.
        configuration = config_builder()
        retrieval = Retrieval(
            configuration,
            output_directory=output_directory,
            **(retrieval_kwargs or {}),
        )
        retrieval.run(num_chains=num_chains, seed=chain_seed, **run_kwargs)

        combine_summary = None
        diagnostics = None
        if combine and family == "mcmc":
            try:
                combine_summary = combine_in_process_mcmc_chains(
                    output_directory,
                    sampler_type,
                    base_name,
                    num_chains,
                    write=True,
                    rhat_threshold=rhat_threshold,
                )
                diagnostics = combine_summary.diagnostics
                retrieval._chain_diagnostics = diagnostics
            except (FileNotFoundError, ValueError, KeyError) as error:
                warnings.warn(
                    f"In-process multi-chain combine failed: {error}",
                    RuntimeWarning,
                )
        if verbose:
            if diagnostics is not None:
                print(
                    f"  ran {num_chains} chains in-process ('{execution}'); "
                    f"combined -> '{combine_summary.combined_name}': "
                    f"{diagnostics['n_samples_total']} samples; max R-hat="
                    f"{diagnostics['max_rhat']:.4f} "
                    f"converged={diagnostics['converged']}"
                )
            else:
                print(
                    f"  ran {num_chains} chains in-process ('{execution}'); "
                    "the sampler pooled them."
                )
        return {
            "execution": execution,
            "num_chains": num_chains,
            "succeeded_chain_ids": list(range(num_chains)),
            "failed": {},
            "combine": combine_summary,
            "diagnostics": diagnostics,
        }

    if execution == ChainExecution.SEQUENTIAL:
        # One resident model; run chains one after another in this process.
        succeeded_ids = list(already_done)
        configuration = config_builder()
        retrieval = Retrieval(
            configuration,
            output_directory=output_directory,
            **(retrieval_kwargs or {}),
        )
        for chain_id in chains_to_run:
            spec = ChainSpec.build(base_name, chain_seed, chain_id)
            try:
                _run_single_chain_on(
                    retrieval, spec, run_kwargs, force_single_chain
                )
                succeeded_ids.append(chain_id)
            except Exception as error:  # noqa: BLE001
                failed[chain_id] = f"{type(error).__name__}: {error}"
                if on_chain_failure == "raise":
                    raise
        succeeded_ids = sorted(set(succeeded_ids))
    else:  # processes
        succeeded_ids = list(already_done)
        if chains_to_run:
            _validate_top_level_builder(config_builder)
            cpus = (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else list(range(os.cpu_count() or 1))
            )
            cpu_groups = partition_cpus(cpus, len(chains_to_run))
            chain_runner = _functools.partial(
                _run_one_chain_from_builder,
                config_builder,
                output_directory,
                retrieval_kwargs,
                base_name,
                chain_seed,
                run_kwargs,
                force_single_chain,
            )
            run_result = run_chains_in_processes(
                chain_runner,
                chains_to_run,
                cpu_groups,
                on_chain_failure=on_chain_failure,
                max_workers=max_workers,
                block_mpi=True,
            )
            succeeded_ids = sorted(set(succeeded_ids) | set(run_result.succeeded_chain_ids))
            failed = run_result.failed

    if verbose:
        print(
            f"  chains finished: {len(succeeded_ids)} succeeded "
            f"({succeeded_ids}), {len(failed)} failed."
        )

    combine_summary = None
    diagnostics = None
    if combine and succeeded_ids:
        if family == "mcmc":
            combine_summary = combine_mcmc_chains(
                output_directory,
                sampler_type,
                base_name,
                num_chains=num_chains,
                write=True,
                on_missing=on_chain_failure,
                rhat_threshold=rhat_threshold,
            )
            diagnostics = combine_summary.diagnostics
            if verbose:
                print(
                    f"  combined -> '{combine_summary.combined_name}': "
                    f"{diagnostics['n_samples_total']} samples; "
                    f"max R-hat="
                    f"{diagnostics['max_rhat']:.4f} "
                    f"converged={diagnostics['converged']}"
                )
        elif family == "nested":
            combine_summary = combine_nested_chains(
                output_directory,
                sampler_type,
                base_name,
                num_chains=num_chains,
                write=True,
                on_missing=on_chain_failure,
                seed=chain_seed,
            )
            diagnostics = combine_summary.diagnostics
            if verbose:
                evidence = combine_summary.evidence
                if evidence.get("available"):
                    print(
                        f"  combined -> '{combine_summary.combined_name}': "
                        f"{diagnostics['n_samples_total']} samples; "
                        f"log Z = {evidence['log_Z_mean']:.3f} +/- "
                        f"{evidence['log_Z_uncert']:.3f} "
                        f"(between-run std "
                        f"{evidence.get('log_Z_between_std', float('nan')):.3f})"
                    )
                else:
                    print(
                        f"  combined -> '{combine_summary.combined_name}': "
                        f"{diagnostics['n_samples_total']} samples; "
                        "evidence unavailable."
                    )
        elif verbose:
            print(
                f"  combine skipped: sampler family '{family}' is not "
                "supported by the native combiner."
            )

    return {
        "execution": execution,
        "num_chains": num_chains,
        "succeeded_chain_ids": succeeded_ids,
        "failed": failed,
        "combine": combine_summary,
        "diagnostics": diagnostics,
    }


def run_single_chain(
    config_builder,
    chain_id: int,
    *,
    output_directory: str,
    num_chains: Optional[int] = None,
    chain_seed: int = 12345,
    sampler_type: Optional[str] = None,
    base_retrieval_name: Optional[str] = None,
    retrieval_kwargs: Optional[dict] = None,
    resume: bool = True,
    verbose: bool = True,
    **run_kwargs,
) -> Optional[str]:
    """Run exactly one chain (used by e.g. a SLURM-array worker).

    Builds a Retrieval from ``config_builder`` and runs chain ``chain_id`` into
    its per-chain namespace. With ``resume=True`` the chain is skipped (returns
    ``None``) if its ``*_samples.npz`` artifact already exists. Returns the
    per-chain retrieval name, or ``None`` if skipped.

    ``num_chains`` is accepted and ignored: a single chain does not need to know
    the size of the run, but the generated worker scripts pass it for symmetry
    with :func:`run_multichain`.
    """
    if sampler_type is None or base_retrieval_name is None:
        probe_config = config_builder()
        sampler_type = sampler_type or probe_config.sampler_type
        base_retrieval_name = base_retrieval_name or probe_config.retrieval_name
    base_name = base_retrieval_name

    if resume:
        artifact = chain_artifact_path(
            output_directory, sampler_type, base_name, chain_id
        )
        if os.path.exists(artifact):
            if verbose:
                print(f"[chain {chain_id}] resume: {artifact} exists; skipping.")
            return None

    force_single_chain = sampler_family(sampler_type) == "mcmc"
    return _run_one_chain_from_builder(
        config_builder,
        output_directory,
        retrieval_kwargs,
        base_name,
        chain_seed,
        run_kwargs,
        force_single_chain,
        chain_id,
        None,
    )


def combine_chains_from_builder(
    config_builder,
    *,
    output_directory: str,
    num_chains: Optional[int] = None,
    retrieval_kwargs: Optional[dict] = None,
    on_missing: str = "warn",
    rhat_threshold: float = 1.01,
    verbose: bool = True,
):
    """Read the sampler type and base name from a builder, then combine its chains.

    Convenience for the SLURM-array combine task. Only the configuration is built
    -- never a ``Retrieval`` -- so the combine job stays light (no opacity load, no
    JAX, no Radtrans). ``retrieval_kwargs`` is accepted and ignored, for
    compatibility with worker scripts generated by earlier versions.
    """
    configuration = config_builder()
    return combine_chains(
        output_directory,
        configuration.sampler_type,
        configuration.retrieval_name,
        num_chains=num_chains,
        on_missing=on_missing,
        rhat_threshold=rhat_threshold,
        verbose=verbose,
    )


def write_chain_launcher(
    config_builder,
    *,
    num_chains: int,
    output_directory: str,
    script_dir: str = ".",
    scheduler: str = "slurm_array",
    builder_module: Optional[str] = None,
    builder_function: Optional[str] = None,
    retrieval_kwargs: Optional[dict] = None,
    run_kwargs: Optional[dict] = None,
    chain_seed: int = 12345,
    resume: bool = True,
    base_retrieval_name: Optional[str] = None,
    python: str = "python",
    cpus_per_task: int = 8,
    partition: Optional[str] = None,
    time: str = "24:00:00",
    mem: Optional[str] = None,
    gres: Optional[str] = None,
    account: Optional[str] = None,
    log_dir: str = "out",
    job_name: Optional[str] = None,
    env_setup_lines: Sequence[str] = (),
    extra_sbatch_lines: Sequence[str] = (),
    verbose: bool = True,
) -> dict:
    """Generate a SLURM job-array launcher for a multi-node multi-chain run.

    Writes a per-task worker script plus an array sbatch, a dependent combine
    sbatch, and a submit driver into ``script_dir``. The array runs one chain
    per task (``SLURM_ARRAY_TASK_ID``); the combine task depends on it
    (``afterany``) and merges the survivors. ``config_builder`` must live in an
    importable module (not ``__main__``) and be a top-level function;
    ``retrieval_kwargs``/``run_kwargs`` must be Python literals (load arrays
    inside the builder).
    """
    if scheduler != "slurm_array":
        raise ValueError("Only scheduler='slurm_array' is currently supported.")
    _validate_top_level_builder(config_builder)

    builder_module = builder_module or getattr(config_builder, "__module__", None)
    builder_function = builder_function or getattr(
        config_builder, "__qualname__", None
    )
    if base_retrieval_name is None:
        base_retrieval_name = config_builder().retrieval_name
    base_name = base_retrieval_name

    script_path = Path(script_dir)
    script_path.mkdir(parents=True, exist_ok=True)
    worker_filename = f"{base_name}_chain_worker.py"

    worker_content = render_chain_worker_script(
        builder_module,
        builder_function,
        output_directory,
        num_chains,
        chain_seed,
        retrieval_kwargs,
        run_kwargs,
        resume,
    )
    scripts = render_slurm_array_scripts(
        base_name=base_name,
        worker_filename=worker_filename,
        num_chains=num_chains,
        python=python,
        cpus_per_task=cpus_per_task,
        partition=partition,
        time=time,
        mem=mem,
        gres=gres,
        account=account,
        log_dir=log_dir,
        job_name=job_name,
        env_setup_lines=env_setup_lines,
        extra_sbatch_lines=extra_sbatch_lines,
    )

    written: dict[str, str] = {}
    worker_path = script_path / worker_filename
    worker_path.write_text(worker_content, encoding="utf-8")
    written["worker"] = str(worker_path)
    for filename, content in scripts.items():
        path = script_path / filename
        path.write_text(content, encoding="utf-8")
        written[filename] = str(path)
        if filename.endswith((".sh", ".sbatch")):
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass

    if verbose:
        print(f"Wrote SLURM multi-chain launcher for '{base_name}':")
        for value in written.values():
            print(f"  {value}")
        print(f"Submit with:  bash {script_path / (base_name + '_submit.sh')}")
    return written
