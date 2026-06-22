"""Pre-training data-sanity guards for SBI runs.

These catch a gross mismatch between the simulated training corpus and the
observation *before* the (expensive) training loop -- the failure mode where a
units/scale bug or a stale/corrupt dataset leaves the training spectra orders of
magnitude away from the observation. Without this guard that only surfaces as a
post-training OOD score, after a multi-hour run has already been wasted.

This is deliberately an *amplitude/scale* check, complementary to (not a
replacement for) the post-training per-wavelength OOD diagnostic. A prior
predictive legitimately spans orders of magnitude in flux, so the comparison is
done on each spectrum's overall amplitude in log10 space: the question is purely
"is the observation many orders of magnitude outside the simulated range?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_VALID_ON_FAIL = ("raise", "warn", "off")


@dataclass(frozen=True)
class ObservationScaleReport:
    """Result of :func:`check_observation_in_training_range`."""

    log10_observation_scale: float
    log10_training_p1: float
    log10_training_p50: float
    log10_training_p99: float
    orders_outside: float
    max_orders_outside: float
    passed: bool


def _per_spectrum_scale(values: np.ndarray) -> np.ndarray:
    """Robust per-spectrum amplitude: the median absolute flux of each row."""
    return np.median(np.abs(values), axis=1)


def check_observation_in_training_range(
    training_values: Any,
    observation_values: Any,
    *,
    max_orders_outside: float = 2.0,
    on_fail: str = "raise",
    max_training_rows: int = 8192,
) -> ObservationScaleReport:
    """Check the observation's flux scale lies within the training distribution.

    For every simulated spectrum a robust amplitude (median absolute flux) is
    computed, and the [p1, p99] range of those amplitudes in log10 space defines
    the simulated scale band. The observation's amplitude must lie within
    ``max_orders_outside`` decades of that band. A units/scale bug -- e.g.
    training spectra at ~1e-20 while the observation is ~1e-15 -- puts the
    observation ~3+ decades outside and is rejected here, before training,
    instead of only via the post-training OOD diagnostic.

    Parameters
    ----------
    training_values:
        Simulated spectra, shape ``(n_samples, n_wavelengths)`` (raw flux scale,
        matching ``observation_values``).
    observation_values:
        The observed spectrum, shape ``(n_wavelengths,)``.
    max_orders_outside:
        How many decades the observation amplitude may fall outside the simulated
        [p1, p99] band before the check fails.
    on_fail:
        ``"raise"`` (default) aborts with a descriptive error, ``"warn"`` prints a
        warning and continues, ``"off"`` disables the check (still returns a
        report).
    max_training_rows:
        Training rows are evenly subsampled to at most this many before computing
        statistics, keeping the check cheap on a multi-hundred-thousand-row split.

    Returns
    -------
    ObservationScaleReport
        The computed log10 scale statistics and pass/fail verdict.
    """
    if on_fail not in _VALID_ON_FAIL:
        raise ValueError(f"on_fail must be one of {_VALID_ON_FAIL}, got {on_fail!r}")

    train = np.asarray(training_values, dtype=float)
    obs = np.asarray(observation_values, dtype=float).reshape(-1)
    if train.ndim != 2:
        raise ValueError(
            "training_values must be 2D (n_samples, n_wavelengths), got shape "
            f"{train.shape}."
        )
    if train.shape[1] != obs.shape[0]:
        raise ValueError(
            f"observation length {obs.shape[0]} does not match the training "
            f"wavelength axis {train.shape[1]}."
        )

    if train.shape[0] > max_training_rows:
        indices = np.linspace(0, train.shape[0] - 1, num=max_training_rows, dtype=int)
        train = train[indices]

    train_scale = _per_spectrum_scale(train)
    train_scale = train_scale[np.isfinite(train_scale) & (train_scale > 0.0)]
    if train_scale.size == 0:
        raise ValueError(
            "Training spectra have no positive amplitude; cannot establish a "
            "scale band (the simulated dataset is degenerate or empty)."
        )
    log_train = np.log10(train_scale)
    p1, p50, p99 = (float(np.percentile(log_train, q)) for q in (1.0, 50.0, 99.0))

    obs_scale = float(np.median(np.abs(obs)))
    log_obs = np.log10(obs_scale) if obs_scale > 0.0 else float("-inf")

    # Decades the observation falls below p1 or above p99 (0 when inside the band).
    orders_outside = float(max(p1 - log_obs, log_obs - p99, 0.0))
    passed = bool(orders_outside <= max_orders_outside)

    report = ObservationScaleReport(
        log10_observation_scale=log_obs,
        log10_training_p1=p1,
        log10_training_p50=p50,
        log10_training_p99=p99,
        orders_outside=orders_outside,
        max_orders_outside=float(max_orders_outside),
        passed=passed,
    )

    if not passed and on_fail != "off":
        message = (
            "Observation flux scale is out-of-distribution relative to the "
            f"training spectra: it lies {orders_outside:.1f} decades outside the "
            f"simulated [p1, p99] band. Observation amplitude ~1e{log_obs:.1f}, "
            f"simulated band ~1e{p1:.1f}..1e{p99:.1f} (median ~1e{p50:.1f}). "
            "This usually indicates a units/scale mismatch or a stale/corrupt "
            "simulation dataset; regenerate the dataset or fix the forward model "
            "before training."
        )
        if on_fail == "raise":
            raise ValueError(message)
        print("WARNING (startup scale check):", message)

    return report


__all__ = ["ObservationScaleReport", "check_observation_in_training_range"]
