"""Lightweight plotting helpers for SBI outputs."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

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


def _prt_style_module() -> Any:
    """Return the ``petitRADTRANS.plotlib.style`` module (needs matplotlib)."""
    from ..plotlib import style

    return style


def _prt_palette() -> dict:
    """Named categorical colours taken from the petitRADTRANS style sheet.

    Every SBI plot draws its colours from the shared
    ``petitRADTRANS.plotlib.style`` colour cycle so the outputs are visually
    consistent with the rest of the package.
    """

    cycle = list(_prt_style_module().prt_default_colour_cycle)
    return {
        "primary": cycle[0],  # blue
        "accent": cycle[1],  # pink
        "highlight": cycle[2],  # amber
        "positive": cycle[3],  # green
        "data": "#000000",  # observed points
    }


def _prt_density_colormap(colour: str) -> Any:
    """White-to-``colour`` sequential colormap for 2-D density panels."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("prt_density", ["white", colour])


def _smooth_histogram2d(counts: np.ndarray, passes: int = 1) -> np.ndarray:
    """Light separable ``[1, 2, 1]/4`` smoothing for cleaner density contours."""
    smoothed = np.asarray(counts, dtype=float)
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(max(0, int(passes))):
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 0, smoothed
        )
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, smoothed
        )
    return smoothed


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

    palette = _prt_palette()
    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for index in range(parameter_dim):
        axis = flat_axes[index]
        axis.hist(
            sample_array[:, index], bins=bins, color=palette["primary"], alpha=0.8
        )
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

    # Only the x-axis is shared down each column. The y-axis is deliberately
    # NOT shared per row: the diagonal panels are 1-D histograms whose y-axis is
    # a *count*, while the off-diagonal panels share the row parameter's *value*
    # range. The previous ``sharey='row'`` tied the diagonal count axis to that
    # value range, so the histogram bars (counts of order 1e2-1e3) overflowed a
    # window scaled to the parameter values and rendered as solid filled blocks
    # that did not match the standalone marginals plot. Off-diagonal y-ranges
    # are instead pinned explicitly below.
    fig, axes = pyplot.subplots(
        parameter_dim,
        parameter_dim,
        figsize=figsize,
        squeeze=False,
        sharex='col',
    )

    palette = _prt_palette()
    density_cmap = _prt_density_colormap(palette['primary'])

    # Per-parameter display ranges (with a small margin) so every column shares
    # one x-range and every off-diagonal row shares one y-range.
    data_min = sample_array.min(axis=0)
    data_max = sample_array.max(axis=0)
    span = np.where(data_max > data_min, data_max - data_min, 1.0)
    lower = data_min - 0.02 * span
    upper = data_max + 0.02 * span

    for row in range(parameter_dim):
        for col in range(parameter_dim):
            axis = axes[row, col]
            if row < col:
                axis.axis('off')
                continue

            if row == col:
                # 1-D marginal histogram, identical in construction to
                # ``plot_posterior_marginals`` so the two views agree.
                axis.hist(
                    sample_array[:, col],
                    bins=bins,
                    color=palette['primary'],
                    alpha=0.85,
                )
                axis.axvline(
                    float(np.median(sample_array[:, col])),
                    color=palette['accent'],
                    linewidth=1.2,
                    alpha=0.9,
                )
                # The diagonal owns an independent count axis; hide it so it is
                # not confused with the row parameter's value ticks.
                axis.tick_params(labelleft=False, left=False)
            else:
                # 2-D marginal as smoothed density contours.
                counts, x_edges, y_edges = np.histogram2d(
                    sample_array[:, col],
                    sample_array[:, row],
                    bins=bins,
                    range=[[lower[col], upper[col]], [lower[row], upper[row]]],
                )
                counts = _smooth_histogram2d(counts.T, passes=1)
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                peak = float(counts.max())
                if peak > 0.0:
                    levels = np.linspace(0.05 * peak, peak, 6)
                    axis.contourf(
                        x_centers,
                        y_centers,
                        counts,
                        levels=levels,
                        cmap=density_cmap,
                    )
                    axis.contour(
                        x_centers,
                        y_centers,
                        counts,
                        levels=levels,
                        colors=palette['primary'],
                        linewidths=0.5,
                        alpha=0.6,
                    )
                axis.set_ylim(lower[row], upper[row])

            axis.set_xlim(lower[col], upper[col])

            if row == parameter_dim - 1:
                axis.set_xlabel(str(parameter_names[col]))
                for tick_label in axis.get_xticklabels():
                    tick_label.set_rotation(45)
                    tick_label.set_horizontalalignment('right')
            else:
                axis.tick_params(labelbottom=False)

            if col == 0 and row > 0:
                axis.set_ylabel(str(parameter_names[row]))
            elif col > 0 and row != col:
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
            color=_prt_palette()['primary'],
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
    x_coordinates: Mapping[str, Any] | Any | None = None,
    x_label: str = r"Wavelength [$\mu$m]",
    y_label: str = "Observation scale",
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
    x_coordinates:
        Physical x-axis sample positions (e.g. wavelength in micron). May be a
        single array applied to every dataset, or a mapping from dataset name to
        array. When ``None`` -- or when a supplied array does not match the
        number of plotted samples -- the helper falls back to the integer sample
        index. Supplying wavelengths replaces the bare coordinate index on the
        x-axis.
    x_label:
        Label for the x-axis (defaults to a wavelength label). Used only when an
        ``x_coordinates`` array is actually applied.
    y_label:
        Label for the y-axis. Pass the observable-specific unit, e.g.
        ``"Flux [W m^-2 um^-1]"`` for emission or ``"Transit depth"`` for
        transmission.

    Returns
    -------
    tuple[Any, np.ndarray]
        Matplotlib figure and axes array.

    Notes
    -----
    Spectrum plots adopt the full ``petitRADTRANS.plotlib.style`` style (fonts,
    ticks and colour cycle) for the duration of the call via a temporary rc
    context, so the global Matplotlib state is left untouched on return.

    For batched reports the helper currently visualizes only the first case for
    each dataset, which keeps the plot compact for quick inspection.
    """

    import matplotlib

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

    def _resolve_x(name: str, n_samples: int) -> tuple[np.ndarray, bool]:
        if x_coordinates is None:
            return np.arange(n_samples), False
        if isinstance(x_coordinates, Mapping):
            candidate = x_coordinates.get(name)
        else:
            candidate = x_coordinates
        if candidate is None:
            return np.arange(n_samples), False
        candidate = np.asarray(candidate, dtype=float).reshape(-1)
        if candidate.shape[0] != n_samples:
            return np.arange(n_samples), False
        return candidate, True

    n_rows = len(selected_dataset_names)
    if figsize is None:
        figsize = (14.0, 6.0 * n_rows)

    palette = _prt_palette()

    with matplotlib.rc_context():
        # Spectra adopt the full pRT style (fonts/ticks/colour cycle); the rc
        # context restores global Matplotlib state when the block exits.
        _prt_style_module().set_petitradtrans_plot_style()

        fig, axes = pyplot.subplots(n_rows, 1, figsize=figsize, squeeze=False)
        flat_axes = axes.flatten()

        for index, dataset_name in enumerate(selected_dataset_names):
            if dataset_name not in report.observed_values:
                raise KeyError(
                    f"Dataset '{dataset_name}' is not present in the posterior-predictive report."
                )

            axis = flat_axes[index]
            observed = np.asarray(report.observed_values[dataset_name], dtype=float)
            predictive_mean = np.asarray(
                report.predictive_mean[dataset_name], dtype=float
            )
            interval_lower = np.asarray(
                report.interval_lower[dataset_name], dtype=float
            )
            interval_upper = np.asarray(
                report.interval_upper[dataset_name], dtype=float
            )

            # Visualize a single spectrum: the first case for batched reports.
            observed_series = observed if observed.ndim == 1 else observed[0]
            mean_series = (
                predictive_mean if predictive_mean.ndim == 1 else predictive_mean[0]
            )
            lower_series = (
                interval_lower if interval_lower.ndim == 1 else interval_lower[0]
            )
            upper_series = (
                interval_upper if interval_upper.ndim == 1 else interval_upper[0]
            )

            x_values, used_physical_x = _resolve_x(
                dataset_name, int(observed_series.shape[0])
            )
            order = np.argsort(x_values)
            x_values = x_values[order]

            axis.fill_between(
                x_values,
                lower_series[order],
                upper_series[order],
                alpha=0.25,
                color=palette["primary"],
                label="90% interval",
            )
            axis.plot(
                x_values,
                mean_series[order],
                label="Predictive mean",
                color=palette["primary"],
            )
            axis.plot(
                x_values,
                observed_series[order],
                marker="o",
                linestyle="none",
                markersize=3.0,
                label="Observed",
                color=palette["data"],
            )

            title_parts = [
                f"coverage={report.interval_coverage.get(dataset_name, float('nan')):.3f}",
                f"mae={report.mean_absolute_error.get(dataset_name, float('nan')):.3e}",
            ]
            mae_sigma = report.mean_absolute_error_sigma.get(
                dataset_name, float("nan")
            )
            if np.isfinite(mae_sigma):
                title_parts.append(f"mae_sigma={mae_sigma:.2f}")
            axis.set_title(f"{dataset_name} | " + ", ".join(title_parts))
            axis.set_xlabel(x_label if used_physical_x else "Sample index")
            axis.set_ylabel(y_label)
            axis.legend(loc="best")

        fig.tight_layout()
        # Realise the styled ticks/labels while the rc context is still active so
        # the appearance survives the caller's later ``savefig``.
        fig.canvas.draw()

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

    palette = _prt_palette()
    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()
    x_values = np.arange(counts.shape[1])

    for index in range(parameter_dim):
        axis = flat_axes[index]
        axis.bar(x_values, counts[index], color=palette["positive"], alpha=0.85)
        axis.set_title(str(parameter_names[index]))
        axis.set_xlabel("Rank")
        axis.set_ylabel("Count")

    for index in range(parameter_dim, len(flat_axes)):
        flat_axes[index].axis("off")

    fig.tight_layout()
    return fig, axes


def save_validation_plots(
    run_root: Any,
    posterior_samples: np.ndarray,
    parameter_names: Sequence[str],
    validation_sbc_report: SimulationBasedCalibrationReport,
    validation_predictive_report: PosteriorPredictiveReport,
    target_predictive_report: PosteriorPredictiveReport,
    dataset_names: Sequence[str],
    spectrum_wavelengths: np.ndarray | None = None,
    x_label: str = r"Wavelength [$\mu$m]",
    y_label: str = "Observation scale",
    dpi: int = 160,
) -> None:
    """Render and save the standard validation plot set (marginals, corner, SBC
    rank histograms, and validation/target posterior-predictive spectra) into
    ``run_root``. A no-op if matplotlib is unavailable.
    """
    try:
        marginals_figure, _ = plot_posterior_marginals(
            posterior_samples, parameter_names=parameter_names
        )
        marginals_figure.savefig(
            run_root / "target_posterior_marginals.png", dpi=dpi, bbox_inches="tight"
        )

        corner_figure, _ = plot_posterior_corner(
            posterior_samples, parameter_names=parameter_names
        )
        corner_figure.savefig(
            run_root / "target_posterior_corner.png", dpi=dpi, bbox_inches="tight"
        )

        sbc_figure, _ = plot_sbc_rank_histograms(
            validation_sbc_report, parameter_names=parameter_names
        )
        sbc_figure.savefig(
            run_root / "validation_sbc_rank_histograms.png", dpi=dpi, bbox_inches="tight"
        )

        for report, filename in (
            (validation_predictive_report, "validation_posterior_predictive.png"),
            (target_predictive_report, "target_posterior_predictive.png"),
        ):
            figure, _ = plot_posterior_predictive_report(
                report,
                dataset_names=list(dataset_names),
                x_coordinates=spectrum_wavelengths,
                x_label=x_label,
                y_label=y_label,
            )
            figure.savefig(run_root / filename, dpi=dpi, bbox_inches="tight")
        print("Saved diagnostic plots to:", run_root)
    except ImportError as error:
        print(f"Skipping plot generation because matplotlib is unavailable: {error}")


__all__ = [
    "plot_local_sensitivity_fisher_correlations",
    "plot_local_sensitivity_jacobians",
    "plot_local_sensitivity_singular_values",
    "plot_posterior_corner",
    "plot_posterior_marginals",
    "plot_posterior_predictive_report",
    "plot_sbc_rank_histograms",
    "save_validation_plots",
]
