"""Prior truncation for a second, sharpened round of amortized inference.

An amortized posterior trained on the full prior is limited by how many
simulations land near the true posterior. When the posterior occupies a tiny
fraction of the prior volume -- the transmission benchmark's transit depth pins
the planet radius to ~1e-5 of its prior width -- almost every simulation is
wasted, and the flow must also manufacture a density feature far sharper than
any one of its spline layers can resolve.

Truncated sequential NPE (TSNPE, Deistler et al. 2022) fixes both at once.
Round one's posterior defines a box; round two simulates only inside that box
and trains on *box-normalized* cube coordinates, so the same flow architecture
now spends its whole support on the region that matters. Because the proposal
is a truncated prior rather than the posterior itself, no importance weights
are needed: the round-two posterior is the true posterior restricted to the
box, provided the box captures essentially all of the posterior mass. That
proviso is what :func:`truncation_box_report` measures, and it is the only
thing standing between this method and silently baking round one's mistakes
into round two.

The box is built from the posterior at one observation, so a round-two
posterior is amortized only over observations near that target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


_TRUNCATION_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class TruncationBox:
    """Axis-aligned box in unit-cube coordinates, one interval per parameter.

    Attributes
    ----------
    lower, upper:
        Per-parameter cube bounds, both in ``[0, 1]`` with ``lower < upper``.
    parameter_names:
        Free-parameter names in cube order, so a stored box can be checked
        against a task rather than trusted positionally.
    metadata:
        How the box was built (coverage, safety factor, source run).
    """

    lower: np.ndarray
    upper: np.ndarray
    parameter_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the box is a usable subset of the unit cube."""
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.shape != upper.shape or lower.ndim != 1:
            raise ValueError(
                f"Truncation bounds must be matching 1-D arrays, got "
                f"{lower.shape} and {upper.shape}."
            )
        if len(self.parameter_names) != lower.shape[0]:
            raise ValueError(
                f"Got {len(self.parameter_names)} parameter names for "
                f"{lower.shape[0]} bounds."
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Truncation bounds must be finite.")
        if np.any(upper <= lower):
            bad = [
                self.parameter_names[i]
                for i in np.flatnonzero(upper <= lower)
            ]
            raise ValueError(f"Truncation box has non-positive width for: {bad}.")
        if np.any(lower < 0.0) or np.any(upper > 1.0):
            raise ValueError("Truncation bounds must lie inside the unit cube.")

    @property
    def widths(self) -> np.ndarray:
        """Per-parameter box width in cube units."""
        return np.asarray(self.upper, dtype=float) - np.asarray(self.lower, dtype=float)

    @property
    def volume_fraction(self) -> float:
        """Fraction of the prior volume retained, as ``exp(sum(log widths))``.

        Computed in log space: with 18 parameters the product underflows.
        """
        return float(np.exp(np.sum(np.log(self.widths))))

    def to_cube(self, normalized: Any) -> np.ndarray:
        """Map box-normalized coordinates in ``[0, 1]`` to full-cube coordinates."""
        values = np.asarray(normalized, dtype=float)
        return np.asarray(self.lower, dtype=float) + values * self.widths

    def from_cube(self, cube: Any) -> np.ndarray:
        """Map full-cube coordinates into box-normalized ``[0, 1]`` coordinates."""
        values = np.asarray(cube, dtype=float)
        return (values - np.asarray(self.lower, dtype=float)) / self.widths

    def contains(self, cube: Any) -> np.ndarray:
        """Boolean mask of which full-cube points lie inside the box."""
        values = np.atleast_2d(np.asarray(cube, dtype=float))
        inside = (values >= np.asarray(self.lower, dtype=float)) & (
            values <= np.asarray(self.upper, dtype=float)
        )
        return np.all(inside, axis=-1)

    def compose(self, inner: "TruncationBox") -> "TruncationBox":
        """Compose with a box expressed in this box's normalized coordinates.

        Round two's posterior lives in round one's box-normalized coordinates,
        so the box built from it is relative, not absolute. Composing gives the
        absolute cube box for round three.

        Each round can only shrink, and a single round is limited by how
        informative the previous posterior is -- on the transmission benchmark
        the first box retains 0.43 of the prior volume because 17 of 18
        marginals are still near-prior. Chaining modest, individually safe
        rounds is how the volume comes down, rather than one aggressive cut
        from a posterior whose width is not trustworthy.
        """
        if tuple(inner.parameter_names) != tuple(self.parameter_names):
            raise ValueError(
                "Cannot compose truncation boxes over different parameters: "
                f"{inner.parameter_names} vs {self.parameter_names}."
            )
        lower = self.to_cube(inner.lower)
        upper = self.to_cube(inner.upper)
        metadata = dict(inner.metadata)
        metadata["composed_with"] = dict(self.metadata)
        metadata["round"] = int(self.metadata.get("round", 1)) + 1
        return TruncationBox(
            lower=lower,
            upper=upper,
            parameter_names=self.parameter_names,
            metadata=metadata,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "schema_version": _TRUNCATION_SCHEMA_VERSION,
            "lower": np.asarray(self.lower, dtype=float).tolist(),
            "upper": np.asarray(self.upper, dtype=float).tolist(),
            "parameter_names": list(self.parameter_names),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TruncationBox":
        """Rebuild a box from :meth:`to_payload` output."""
        return cls(
            lower=np.asarray(payload["lower"], dtype=float),
            upper=np.asarray(payload["upper"], dtype=float),
            parameter_names=tuple(payload["parameter_names"]),
            metadata=dict(payload.get("metadata", {})),
        )


def build_truncation_box(
    cube_samples: Any,
    parameter_names: tuple[str, ...] | list[str],
    *,
    coverage: float = 0.9999,
    safety_factor: float = 1.5,
    min_box_fraction: float = 0.02,
    source: str | None = None,
) -> TruncationBox:
    """Build a truncation box around posterior draws in cube coordinates.

    Parameters
    ----------
    cube_samples:
        ``(n_samples, n_parameters)`` posterior draws in unit-cube coordinates.
    parameter_names:
        Free-parameter names in cube order.
    coverage:
        Central posterior mass each per-parameter interval must span before the
        safety margin is applied. The default keeps all but one draw in ten
        thousand: truncation is only safe when it is generous.
    safety_factor:
        Multiplier applied to each interval's half-width about its centre.
        Guards against round one being slightly off-centre, which the
        transmission benchmark demonstrably is (SBC mean rank 748/1024 for the
        planet radius, where 512 is unbiased).
    min_box_fraction:
        Floor on any interval's width, as a fraction of the prior. Stops a
        parameter that round one over-sharpened from being truncated to a
        sliver that round two can never escape.
    source:
        Free-form provenance string recorded in the box metadata.

    Returns
    -------
    TruncationBox
        The box, with the construction settings recorded in its metadata.
    """
    samples = np.asarray(cube_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected 2-D cube samples, got shape {samples.shape}.")
    names = tuple(str(name) for name in parameter_names)
    if samples.shape[1] != len(names):
        raise ValueError(
            f"Got {samples.shape[1]} sample columns for {len(names)} parameter names."
        )
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}.")
    if safety_factor < 1.0:
        raise ValueError(f"safety_factor must be >= 1, got {safety_factor}.")
    if not 0.0 < min_box_fraction <= 1.0:
        raise ValueError(
            f"min_box_fraction must lie in (0, 1], got {min_box_fraction}."
        )

    tail = 0.5 * (1.0 - coverage)
    lower = np.quantile(samples, tail, axis=0)
    upper = np.quantile(samples, 1.0 - tail, axis=0)

    centre = 0.5 * (lower + upper)
    half_width = np.maximum(0.5 * (upper - lower) * safety_factor, 0.5 * min_box_fraction)
    lower = centre - half_width
    upper = centre + half_width

    # Clip into the unit cube, then slide (rather than shrink) any interval
    # that a clip would have narrowed, so the requested width survives.
    width = np.minimum(upper - lower, 1.0)
    lower = np.clip(lower, 0.0, 1.0 - width)
    upper = lower + width

    return TruncationBox(
        lower=lower,
        upper=upper,
        parameter_names=names,
        metadata={
            "coverage": float(coverage),
            "safety_factor": float(safety_factor),
            "min_box_fraction": float(min_box_fraction),
            "n_samples": int(samples.shape[0]),
            "source": source,
        },
    )


def truncation_box_report(
    box: TruncationBox,
    cube_samples: Any,
    *,
    truth_cube: Any | None = None,
    min_captured_fraction: float = 0.99,
) -> dict[str, Any]:
    """Measure whether a box is safe to truncate with.

    Truncation is only valid if the box keeps essentially all of round one's
    posterior mass; anything it cuts away, round two can never recover. When
    the truth is known it is also checked explicitly, since a box that excludes
    it guarantees a confident wrong answer.

    Returns
    -------
    dict
        ``captured_fraction``, per-parameter capture, box widths, volume
        fraction, optional truth checks, and a boolean ``safe`` verdict with
        human-readable ``reasons`` when it is ``False``.
    """
    samples = np.asarray(cube_samples, dtype=float)
    captured = float(np.mean(box.contains(samples)))
    per_parameter = {
        name: float(
            np.mean(
                (samples[:, index] >= box.lower[index])
                & (samples[:, index] <= box.upper[index])
            )
        )
        for index, name in enumerate(box.parameter_names)
    }

    report: dict[str, Any] = {
        "captured_fraction": captured,
        "min_captured_fraction": float(min_captured_fraction),
        "per_parameter_captured_fraction": per_parameter,
        "widths": {
            name: float(box.widths[index])
            for index, name in enumerate(box.parameter_names)
        },
        "volume_fraction": box.volume_fraction,
        "safe": True,
        "reasons": [],
    }

    if captured < min_captured_fraction:
        report["safe"] = False
        report["reasons"].append(
            f"the box keeps only {captured:.4f} of the round-one posterior mass "
            f"(required {min_captured_fraction:.4f}); truncating here would "
            "discard posterior mass that round two cannot recover"
        )

    if truth_cube is not None:
        truth = np.asarray(truth_cube, dtype=float).reshape(-1)
        inside = bool(box.contains(truth)[0])
        # Signed margin in box widths: negative means outside.
        margin = np.minimum(
            truth - np.asarray(box.lower, dtype=float),
            np.asarray(box.upper, dtype=float) - truth,
        ) / box.widths
        report["truth_inside_box"] = inside
        report["truth_margin_in_box_widths"] = {
            name: float(margin[index])
            for index, name in enumerate(box.parameter_names)
        }
        if not inside:
            outside = [
                name
                for index, name in enumerate(box.parameter_names)
                if margin[index] < 0.0
            ]
            report["safe"] = False
            report["reasons"].append(
                f"the truth lies outside the box for: {outside}; round two would "
                "be confidently wrong for these parameters"
            )

    return report


@dataclass(frozen=True)
class TruncatedParameterLayout:
    """Parameter layout that reads cube coordinates as box-normalized.

    Round two stores and trains on coordinates in ``[0, 1]`` spanning the
    truncation box, so every consumer that turns a cube coordinate into a
    physical parameter -- the simulator, the posterior-predictive step, the
    refinement residual -- must first map back through the box. Wrapping the
    layout puts that mapping in one place instead of at each call site, and
    anything unrelated to cube coordinates is delegated untouched.

    The prior quantiles that the task fingerprint records go through
    :meth:`_transform_parameter`, so a truncated task fingerprints differently
    from its parent and cannot silently reuse the parent's dataset.
    """

    base: Any
    box: TruncationBox

    def _to_full_cube(self, cube_positions: Any) -> Any:
        """Map box-normalized cube coordinates to full-prior cube coordinates.

        Done with ``jax.numpy`` rather than ``numpy``: test-time refinement
        differentiates the forward model with respect to the cube position, so
        these coordinates arrive as JAX tracers, and any ``np.asarray`` on the
        path raises ``TracerArrayConversionError``. The arithmetic is a
        multiply-add, so it works unchanged on concrete arrays too.
        """
        values = jnp.asarray(cube_positions)
        lower = jnp.asarray(self.box.lower, dtype=values.dtype)
        widths = jnp.asarray(self.box.widths, dtype=values.dtype)
        return lower + values * widths

    def transform_cube_to_physical_batch(self, cube_positions: Any) -> Any:
        """Transform box-normalized cube draws to physical parameter values."""
        return self.base.transform_cube_to_physical_batch(
            self._to_full_cube(cube_positions)
        )

    def transform_unconstrained_batch(self, unconstrained_positions: Any) -> Any:
        """Transform unconstrained draws, treating them as logits of the box."""
        cube = jax.nn.sigmoid(jnp.asarray(unconstrained_positions))
        return self.transform_cube_to_physical_batch(cube)

    def _transform_parameter(self, parameter: Any, cube_value: Any) -> Any:
        """Transform one parameter's box-normalized quantile to physical units."""
        name = str(getattr(parameter, "name", ""))
        try:
            index = self.box.parameter_names.index(name)
        except ValueError:
            # Fixed parameters are not part of the cube and need no remapping.
            return self.base._transform_parameter(parameter, cube_value)
        lower = float(self.box.lower[index])
        width = float(self.box.widths[index])
        return self.base._transform_parameter(parameter, lower + cube_value * width)

    def __getattr__(self, name: str) -> Any:
        """Delegate everything that does not consume cube coordinates."""
        return getattr(self.base, name)
