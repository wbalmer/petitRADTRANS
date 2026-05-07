"""Lightweight plotting helpers for SBI outputs."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .calibration import (
    LocalSensitivityReport,
    PosteriorPredictiveReport,
    SimulationBasedCalibrationReport,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised via runtime environment.
    plt = None


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for SBI plotting helpers. Install it in the active environment to use plotting.py."
        )
    return plt


def plot_posterior_marginals(
    samples: Any,
    parameter_names: Sequence[str] | None = None,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot one histogram per posterior dimension.

    Parameters
    ----------
    samples:
        Posterior samples with shape ``(n_samples, n_dim)`` or a scalar-vector
        equivalent.
    parameter_names:
        Optional display names for each posterior dimension.
    bins:
        Number of histogram bins per dimension.
    figsize:
        Optional Matplotlib figure size.

    Returns
    -------
    tuple[Any, np.ndarray]
        Matplotlib figure and axes array for further customization or saving.
    """

    pyplot = _require_matplotlib()
    sample_array = np.asarray(samples, dtype=float)

    if sample_array.ndim == 1:
        sample_array = sample_array[:, None]
    if sample_array.ndim != 2:
        raise ValueError(
            f"Expected posterior samples with shape (n_samples, n_dim), got {sample_array.shape}."
        )

    parameter_dim = int(sample_array.shape[1])
    if parameter_names is None:
        parameter_names = [f"parameter_{index}" for index in range(parameter_dim)]
    if len(parameter_names) != parameter_dim:
        raise ValueError(
            "parameter_names length must match the posterior dimensionality."
        )

    n_cols = min(3, parameter_dim)
    n_rows = int(math.ceil(parameter_dim / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 3.0 * n_rows)

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for index in range(parameter_dim):
        axis = flat_axes[index]
        axis.hist(sample_array[:, index], bins=bins, color="tab:blue", alpha=0.8)
        axis.set_title(str(parameter_names[index]))
        axis.set_xlabel("Value")
        axis.set_ylabel("Count")

    for index in range(parameter_dim, len(flat_axes)):
        flat_axes[index].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_posterior_corner(
    samples: Any,
    parameter_names: Sequence[str] | None = None,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
    max_points: int = 8_192,
) -> tuple[Any, np.ndarray]:
    """Plot a lower-triangular corner view of posterior structure.

    Parameters
    ----------
    samples:
        Posterior samples with shape ``(n_samples, n_dim)`` or a scalar-vector
        equivalent.
    parameter_names:
        Optional display names for each posterior dimension.
    bins:
        Number of bins used for one- and two-dimensional histograms.
    figsize:
        Optional Matplotlib figure size.
    max_points:
        Maximum number of posterior draws plotted. Larger sample sets are
        evenly subsampled to keep rendering costs bounded.

    Returns
    -------
    tuple[Any, np.ndarray]
        Matplotlib figure and axes array for further customization or saving.
    """

    pyplot = _require_matplotlib()
    sample_array = np.asarray(samples, dtype=float)

    if sample_array.ndim == 1:
        sample_array = sample_array[:, None]
    if sample_array.ndim != 2:
        raise ValueError(
            f"Expected posterior samples with shape (n_samples, n_dim), got {sample_array.shape}."
        )

    parameter_dim = int(sample_array.shape[1])
    if parameter_names is None:
        parameter_names = [f"parameter_{index}" for index in range(parameter_dim)]
    if len(parameter_names) != parameter_dim:
        raise ValueError(
            "parameter_names length must match the posterior dimensionality."
        )

    if max_points > 0 and sample_array.shape[0] > max_points:
        indices = np.linspace(0, sample_array.shape[0] - 1, num=max_points, dtype=int)
        sample_array = sample_array[indices]

    if figsize is None:
        side = max(2.8 * parameter_dim, 3.5)
        figsize = (side, side)

    fig, axes = pyplot.subplots(
        parameter_dim,
        parameter_dim,
        figsize=figsize,
        squeeze=False,
        sharex='col',
        sharey='row',
    )

    for row in range(parameter_dim):
        for col in range(parameter_dim):
            axis = axes[row, col]
            if row < col:
                axis.axis('off')
                continue

            if row == col:
                axis.hist(
                    sample_array[:, col],
                    bins=bins,
                    color='tab:blue',
                    alpha=0.85,
                )
                axis.axvline(
                    float(np.median(sample_array[:, col])),
                    color='black',
                    linewidth=1.0,
                    alpha=0.7,
                )
            else:
                axis.hist2d(
                    sample_array[:, col],
                    sample_array[:, row],
                    bins=bins,
                    cmap='Blues',
                )

            if row == parameter_dim - 1:
                axis.set_xlabel(str(parameter_names[col]))
            else:
                axis.tick_params(labelbottom=False)

            if col == 0 and row > 0:
                axis.set_ylabel(str(parameter_names[row]))
            elif col > 0:
                axis.tick_params(labelleft=False)

    fig.tight_layout()
    return fig, axes


def plot_local_sensitivity_jacobians(
    report: LocalSensitivityReport,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot whitened Jacobian heatmaps for each representative point."""

    pyplot = _require_matplotlib()
    point_reports = list(report.representative_points)
    if not point_reports:
        raise ValueError("LocalSensitivityReport does not contain any representative points.")

    n_cols = min(3, len(point_reports))
    n_rows = int(math.ceil(len(point_reports) / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 3.6 * n_rows)

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for axis, point_report in zip(flat_axes, point_reports, strict=False):
        matrix = np.log10(np.maximum(np.abs(point_report.whitened_jacobian), 1e-12))
        image = axis.imshow(matrix, aspect='auto', cmap='viridis')
        axis.set_title(
            f"{point_report.label}\nrank={point_report.effective_rank}, cond={point_report.condition_number:.2e}"
        )
        axis.set_xlabel('Parameter')
        axis.set_ylabel('Observation index')
        axis.set_xticks(np.arange(len(report.parameter_names)))
        axis.set_xticklabels(report.parameter_names, rotation=45, ha='right')
        fig.colorbar(image, ax=axis, shrink=0.85, label='log10 |J / sigma|')

    for axis in flat_axes[len(point_reports) :]:
        axis.axis('off')

    fig.tight_layout()
    return fig, axes


def plot_local_sensitivity_fisher_correlations(
    report: LocalSensitivityReport,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot Fisher-correlation heatmaps for each representative point."""

    pyplot = _require_matplotlib()
    point_reports = list(report.representative_points)
    if not point_reports:
        raise ValueError("LocalSensitivityReport does not contain any representative points.")

    n_cols = min(3, len(point_reports))
    n_rows = int(math.ceil(len(point_reports) / n_cols))
    if figsize is None:
        figsize = (4.2 * n_cols, 3.8 * n_rows)

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for axis, point_report in zip(flat_axes, point_reports, strict=False):
        image = axis.imshow(
            point_report.fisher_correlation,
            vmin=-1.0,
            vmax=1.0,
            cmap='coolwarm',
        )
        axis.set_title(point_report.label)
        axis.set_xticks(np.arange(len(report.parameter_names)))
        axis.set_yticks(np.arange(len(report.parameter_names)))
        axis.set_xticklabels(report.parameter_names, rotation=45, ha='right')
        axis.set_yticklabels(report.parameter_names)
        fig.colorbar(image, ax=axis, shrink=0.85, label='Correlation')

    for axis in flat_axes[len(point_reports) :]:
        axis.axis('off')

    fig.tight_layout()
    return fig, axes


def plot_local_sensitivity_singular_values(
    report: LocalSensitivityReport,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot singular spectra of the whitened Jacobian for each point."""

    pyplot = _require_matplotlib()
    point_reports = list(report.representative_points)
    if not point_reports:
        raise ValueError("LocalSensitivityReport does not contain any representative points.")

    n_cols = min(3, len(point_reports))
    n_rows = int(math.ceil(len(point_reports) / n_cols))
    if figsize is None:
        figsize = (4.2 * n_cols, 3.4 * n_rows)

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for axis, point_report in zip(flat_axes, point_reports, strict=False):
        singular_values = np.asarray(point_report.singular_values, dtype=float)
        axis.semilogy(
            np.arange(1, singular_values.shape[0] + 1),
            np.maximum(singular_values, 1e-12),
            marker='o',
            color='tab:blue',
        )
        axis.set_title(point_report.label)
        axis.set_xlabel('Singular value index')
        axis.set_ylabel('Singular value')

    for axis in flat_axes[len(point_reports) :]:
        axis.axis('off')

    fig.tight_layout()
    return fig, axes


def plot_posterior_predictive_report(
    report: PosteriorPredictiveReport,
    dataset_names: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot observed values against posterior-predictive means and intervals.

    Parameters
    ----------
    report:
        Posterior-predictive report to visualize.
    dataset_names:
        Optional subset of dataset names to plot. Defaults to all datasets in
        the report.
    figsize:
        Optional Matplotlib figure size.

    Returns
    -------
    tuple[Any, np.ndarray]
        Matplotlib figure and axes array.

    Notes
    -----
    For batched reports the helper currently visualizes only the first case for
    each dataset, which keeps the plot compact for quick inspection.
    """

    pyplot = _require_matplotlib()
    selected_dataset_names = (
        list(report.observed_values.keys())
        if dataset_names is None
        else list(dataset_names)
    )
    if not selected_dataset_names:
        raise ValueError(
            "At least one dataset name is required to plot a posterior-predictive report."
        )

    n_rows = len(selected_dataset_names)
    if figsize is None:
        figsize = (8.0, 3.0 * n_rows)

    fig, axes = pyplot.subplots(n_rows, 1, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for index, dataset_name in enumerate(selected_dataset_names):
        if dataset_name not in report.observed_values:
            raise KeyError(
                f"Dataset '{dataset_name}' is not present in the posterior-predictive report."
            )

        axis = flat_axes[index]
        observed = np.asarray(report.observed_values[dataset_name], dtype=float)
        predictive_mean = np.asarray(report.predictive_mean[dataset_name], dtype=float)
        interval_lower = np.asarray(report.interval_lower[dataset_name], dtype=float)
        interval_upper = np.asarray(report.interval_upper[dataset_name], dtype=float)

        if observed.ndim == 1:
            x_coordinates = np.arange(observed.shape[0])
            axis.plot(
                x_coordinates,
                observed,
                marker="o",
                linestyle="none",
                label="Observed",
                color="black",
            )
            axis.plot(
                x_coordinates,
                predictive_mean,
                label="Predictive mean",
                color="tab:blue",
            )
            axis.fill_between(
                x_coordinates,
                interval_lower,
                interval_upper,
                alpha=0.25,
                color="tab:blue",
            )
        else:
            x_coordinates = np.arange(observed.shape[-1])
            axis.plot(
                x_coordinates,
                observed[0],
                marker="o",
                linestyle="none",
                label="Observed",
                color="black",
            )
            axis.plot(
                x_coordinates,
                predictive_mean[0],
                label="Predictive mean",
                color="tab:blue",
            )
            axis.fill_between(
                x_coordinates,
                interval_lower[0],
                interval_upper[0],
                alpha=0.25,
                color="tab:blue",
            )

        axis.set_title(
            f"{dataset_name} | coverage={report.interval_coverage.get(dataset_name, float('nan')):.3f}, "
            f"mae={report.mean_absolute_error.get(dataset_name, float('nan')):.3e}"
        )
        axis.set_xlabel("Index")
        axis.set_ylabel("Observation scale")
        axis.legend(loc="best")

    fig.tight_layout()
    return fig, axes


def plot_sbc_rank_histograms(
    report: SimulationBasedCalibrationReport,
    parameter_names: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Plot one SBC rank histogram per inferred parameter.

    Parameters
    ----------
    report:
        SBC report containing per-parameter rank histogram counts.
    parameter_names:
        Optional display names for each inferred parameter.
    figsize:
        Optional Matplotlib figure size.

    Returns
    -------
    tuple[Any, np.ndarray]
        Matplotlib figure and axes array.
    """

    pyplot = _require_matplotlib()
    counts = np.asarray(report.rank_histogram_counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError(
            f"Expected rank_histogram_counts with shape (n_dim, n_bins), got {counts.shape}."
        )

    parameter_dim = int(counts.shape[0])
    if parameter_names is None:
        parameter_names = [f"parameter_{index}" for index in range(parameter_dim)]
    if len(parameter_names) != parameter_dim:
        raise ValueError("parameter_names length must match the SBC dimensionality.")

    n_cols = min(3, parameter_dim)
    n_rows = int(math.ceil(parameter_dim / n_cols))
    if figsize is None:
        figsize = (4.5 * n_cols, 3.0 * n_rows)

    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()
    x_values = np.arange(counts.shape[1])

    for index in range(parameter_dim):
        axis = flat_axes[index]
        axis.bar(x_values, counts[index], color="tab:green", alpha=0.85)
        axis.set_title(str(parameter_names[index]))
        axis.set_xlabel("Rank")
        axis.set_ylabel("Count")

    for index in range(parameter_dim, len(flat_axes)):
        flat_axes[index].axis("off")

    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_local_sensitivity_fisher_correlations",
    "plot_local_sensitivity_jacobians",
    "plot_local_sensitivity_singular_values",
    "plot_posterior_corner",
    "plot_posterior_marginals",
    "plot_posterior_predictive_report",
    "plot_sbc_rank_histograms",
]
