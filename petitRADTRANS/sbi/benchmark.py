"""Benchmarking interfaces for comparing amortized and exact retrievals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .inference import AmortizedRetrievalResult
from .task import SBITask


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    """One benchmark problem used to compare inference backends."""

    name: str
    task: SBITask
    observation: Any
    reference_posterior: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Metrics summarizing agreement and predictive performance."""

    calibration: Mapping[str, float] = field(default_factory=dict)
    posterior_distance: Mapping[str, float] = field(default_factory=dict)
    predictive_checks: Mapping[str, float] = field(default_factory=dict)
    runtime: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkComparison:
    """Compare an amortized result to one or more exact retrieval baselines."""

    case_name: str
    amortized_result: AmortizedRetrievalResult
    exact_results: Mapping[str, Any] = field(default_factory=dict)
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RetrievalBenchmarkSuite:
    """Run standardized benchmark comparisons for SBI tasks."""

    def __init__(self, cases: list[RetrievalBenchmarkCase]):
        self.cases = cases

    def run_case(self, case: RetrievalBenchmarkCase) -> BenchmarkComparison:
        """Run one benchmark case and compute comparison metrics.

        Parameters
        ----------
        case:
            Benchmark case describing the task, observation, and optional exact
            reference posterior to compare against.

        Returns
        -------
        BenchmarkComparison
            Comparison payload combining amortized and exact results together
            with any derived metrics.

        Notes
        -----
        The base class is intentionally abstract. Concrete suites are expected
        to bind exact and amortized inference backends and define the metric
        computations appropriate for the comparison.
        """
        raise NotImplementedError(
            "Benchmark execution requires concrete exact and amortized inference backends."
        )

    def run_all(self) -> list[BenchmarkComparison]:
        """Run all configured benchmark cases."""
        return [self.run_case(case) for case in self.cases]
