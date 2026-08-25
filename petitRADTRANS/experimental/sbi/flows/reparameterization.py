"""Observation-conditioned reparameterisation of an amplitude-driven parameter.

A transmission spectrum does not measure the planet radius. It measures the
*total* radius ``R_p + h``, where ``h`` is the median atmospheric annulus:
``depth = (R_p + h)^2 / R_s^2``. That combination is pinned by the data to a few
parts in ``10^5``, while the split between ``R_p`` and ``h`` is not -- across the
24M test split ``h`` has a standard deviation of 0.465 R_jup, and a linear oracle
knowing every true parameter still leaves it at 0.201.

The exact-summary channel (see
:func:`~petitRADTRANS.experimental.sbi.preprocessing.exact_summary_feature_matrix`)
hands the conditioner ``log10(median depth)`` at full precision. That is
necessary but not sufficient: the conditioner would still have to *learn* the
map from that scalar to the location of the radius marginal, and the required
accuracy is around four and a half significant digits. Trained networks in
float32 do not deliver that on a learned nonlinear map.

This module removes the requirement. The flow is given a coordinate from which
the observation's amplitude has already been divided out analytically:

    r = log10(theta) - exponent * log10(amplitude(x))

so that inferring ``theta`` becomes inferring ``r`` -- the genuinely uncertain
ratio -- with the sharp, observation-driven direction handled by an exact
formula rather than by the density estimator. ``exponent`` is 0.5 because the
observable amplitude goes as the square of the radius; the same construction
covers an emission flux scale, which also goes as ``R^2``.

The map is a monotone bijection in ``theta`` for each fixed observation, so it
is a legitimate change of variables: the NPE objective is unchanged up to a
Jacobian term that does not involve the flow's parameters, and posterior draws
are mapped back exactly.

Earlier radius reparameterisations in this campaign were measured to do nothing.
They differed in the decisive respect: they reparameterised the *marginal* of
``R_p`` with no reference to the observation, which cannot help when the
deficiency is that the posterior ignores a combination the spectrum determines.
This one is conditioned on the observed amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from .cube import _clip_cube_support


_LOG10 = float(np.log(10.0))


@dataclass(frozen=True)
class AmplitudeConditionedParameter:
    """A parameter the flow infers relative to the observation's amplitude.

    Attributes
    ----------
    parameter_name, parameter_index:
        Which free parameter is reparameterised.
    prior_lower, prior_upper:
        The parameter's physical prior bounds, i.e. the affine map from its unit
        cube coordinate. Must be positive, since the coordinate is a log ratio.
    context_index:
        Position of the standardised log-amplitude scalar inside the flow's
        conditioning vector. This is the first entry of the exact-summary
        channel, so it sits at ``embedding_dim`` (see
        :attr:`~petitRADTRANS.experimental.sbi.encoders.ObservationEncoder.context_dim`).
    amplitude_center, amplitude_scale:
        The fixed constants that standardised the log-amplitude, read back so the
        raw ``log10(amplitude)`` can be recovered exactly.
    amplitude_exponent:
        ``d log10(theta) / d log10(amplitude)`` at fixed atmosphere: 0.5 for a
        radius against a transit depth or a flux.
    ratio_lower, ratio_upper:
        Fitted bounds of ``r`` over the training split, with margin. The flow's
        coordinate is ``r`` mapped affinely onto the unit interval by these.
    """

    parameter_name: str
    parameter_index: int
    prior_lower: float
    prior_upper: float
    context_index: int
    amplitude_center: float
    amplitude_scale: float
    ratio_lower: float
    ratio_upper: float
    amplitude_exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.prior_lower <= 0.0:
            raise ValueError(
                "AmplitudeConditionedParameter needs a strictly positive prior "
                f"lower bound (the coordinate is a log ratio); got {self.prior_lower}."
            )
        if self.prior_upper <= self.prior_lower:
            raise ValueError(
                "prior_upper must exceed prior_lower; got "
                f"({self.prior_lower}, {self.prior_upper})."
            )
        if self.ratio_upper <= self.ratio_lower:
            raise ValueError(
                "ratio_upper must exceed ratio_lower; got "
                f"({self.ratio_lower}, {self.ratio_upper})."
            )
        if self.parameter_index < 0 or self.context_index < 0:
            raise ValueError("parameter_index and context_index must be non-negative.")

    # -- coordinate maps ---------------------------------------------------

    def log_amplitude(self, context: Any) -> jnp.ndarray:
        """Recover ``log10(amplitude)`` from the conditioning vector.

        Exact by construction: the exact-summary channel is a fixed affine map
        of this quantity, so undoing that map returns it unchanged rather than
        estimating it.
        """
        values = jnp.asarray(context, dtype=jnp.float32)
        standardized = values[..., self.context_index]
        return self.amplitude_center + self.amplitude_scale * standardized

    def _offset(self, context: Any) -> jnp.ndarray:
        return self.amplitude_exponent * self.log_amplitude(context)

    def to_flow_cube(self, cube_parameters: Any, context: Any) -> jnp.ndarray:
        """Replace the parameter's cube coordinate by the amplitude-free ratio."""
        cube = jnp.asarray(cube_parameters, dtype=jnp.float32)
        physical = self.prior_lower + cube[..., self.parameter_index] * (
            self.prior_upper - self.prior_lower
        )
        ratio = jnp.log10(jnp.maximum(physical, 1e-30)) - self._offset(context)
        reparameterized = (ratio - self.ratio_lower) / (
            self.ratio_upper - self.ratio_lower
        )
        return cube.at[..., self.parameter_index].set(
            _clip_cube_support(reparameterized)
        )

    def to_prior_cube(self, flow_cube_parameters: Any, context: Any) -> jnp.ndarray:
        """Exact inverse of :meth:`to_flow_cube`."""
        cube = jnp.asarray(flow_cube_parameters, dtype=jnp.float32)
        ratio = self.ratio_lower + cube[..., self.parameter_index] * (
            self.ratio_upper - self.ratio_lower
        )
        physical = jnp.power(10.0, ratio + self._offset(context))
        prior_cube = (physical - self.prior_lower) / (
            self.prior_upper - self.prior_lower
        )
        return cube.at[..., self.parameter_index].set(_clip_cube_support(prior_cube))

    def log_abs_det_jacobian(self, cube_parameters: Any, context: Any) -> jnp.ndarray:
        """``log |d r_cube / d c|`` for the reparameterised coordinate.

        Needed only so a reported log-density is expressed in the original cube
        coordinates. It does not involve the flow's parameters, so it shifts the
        objective without changing its optimum -- but a validation loss that
        omitted it would not be comparable with a run that did not reparameterise.
        """
        del context
        cube = jnp.asarray(cube_parameters, dtype=jnp.float32)
        width = self.prior_upper - self.prior_lower
        physical = jnp.maximum(
            self.prior_lower + cube[..., self.parameter_index] * width, 1e-30
        )
        # r = log10(theta) - offset(x); dr/dc = width / (theta ln10), and the
        # ratio -> unit-interval map divides by (ratio_upper - ratio_lower).
        derivative = width / (
            physical * _LOG10 * (self.ratio_upper - self.ratio_lower)
        )
        return jnp.log(jnp.maximum(derivative, 1e-30))

    # -- persistence -------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_index": int(self.parameter_index),
            "prior_lower": float(self.prior_lower),
            "prior_upper": float(self.prior_upper),
            "context_index": int(self.context_index),
            "amplitude_center": float(self.amplitude_center),
            "amplitude_scale": float(self.amplitude_scale),
            "ratio_lower": float(self.ratio_lower),
            "ratio_upper": float(self.ratio_upper),
            "amplitude_exponent": float(self.amplitude_exponent),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any] | None
    ) -> "AmplitudeConditionedParameter | None":
        if not payload:
            return None
        return cls(
            parameter_name=str(payload["parameter_name"]),
            parameter_index=int(payload["parameter_index"]),
            prior_lower=float(payload["prior_lower"]),
            prior_upper=float(payload["prior_upper"]),
            context_index=int(payload["context_index"]),
            amplitude_center=float(payload["amplitude_center"]),
            amplitude_scale=float(payload["amplitude_scale"]),
            ratio_lower=float(payload["ratio_lower"]),
            ratio_upper=float(payload["ratio_upper"]),
            amplitude_exponent=float(payload.get("amplitude_exponent", 0.5)),
        )


def fit_amplitude_conditioned_parameter(
    parameter_name: str,
    parameter_index: int,
    physical_values: Any,
    log_amplitudes: Any,
    prior_lower: float,
    prior_upper: float,
    context_index: int,
    amplitude_center: float,
    amplitude_scale: float,
    amplitude_exponent: float = 0.5,
    coverage: float = 0.999,
    margin: float = 0.1,
) -> tuple[AmplitudeConditionedParameter, dict[str, Any]]:
    """Fit the ratio bounds from training draws, and report how well they hold.

    The bounds must contain essentially every training sample: a draw whose ratio
    falls outside is clipped to the cube edge, which piles probability mass at the
    boundary -- the failure the spline-bound work already had to fix once. The
    returned report carries the captured fraction so the caller can refuse a fit
    that does not.

    A tight box is what makes the reparameterisation worth doing: the point is to
    hand the flow a coordinate whose posterior occupies a useful fraction of the
    support, rather than a sliver of a range set by the full amplitude prior.
    """
    values = np.asarray(physical_values, dtype=float).reshape(-1)
    amplitudes = np.asarray(log_amplitudes, dtype=float).reshape(-1)
    if values.shape != amplitudes.shape:
        raise ValueError(
            f"physical_values {values.shape} and log_amplitudes {amplitudes.shape} "
            "must have the same length."
        )
    finite = np.isfinite(values) & np.isfinite(amplitudes) & (values > 0.0)
    if finite.sum() < 16:
        raise ValueError(
            f"Only {int(finite.sum())} usable draws for the amplitude "
            "reparameterisation fit; need at least 16."
        )

    ratio = np.log10(values[finite]) - amplitude_exponent * amplitudes[finite]
    tail = 0.5 * (1.0 - float(coverage))
    lower, upper = np.quantile(ratio, [tail, 1.0 - tail])
    span = max(float(upper - lower), 1e-6)
    lower = float(lower - margin * span)
    upper = float(upper + margin * span)

    reparameterization = AmplitudeConditionedParameter(
        parameter_name=str(parameter_name),
        parameter_index=int(parameter_index),
        prior_lower=float(prior_lower),
        prior_upper=float(prior_upper),
        context_index=int(context_index),
        amplitude_center=float(amplitude_center),
        amplitude_scale=float(amplitude_scale),
        ratio_lower=lower,
        ratio_upper=upper,
        amplitude_exponent=float(amplitude_exponent),
    )
    inside = float(np.mean((ratio >= lower) & (ratio <= upper)))
    report = {
        "parameter": str(parameter_name),
        "n_draws": int(finite.sum()),
        "ratio_lower": lower,
        "ratio_upper": upper,
        "captured_fraction": inside,
        "ratio_sd": float(np.std(ratio)),
        # How much narrower the conditioned coordinate is than the raw log
        # parameter range: the factor by which the flow's job got easier.
        "compression_vs_prior": float(
            (np.log10(prior_upper) - np.log10(prior_lower)) / span
        ),
    }
    return reparameterization, report


__all__ = [
    "AmplitudeConditionedParameter",
    "fit_amplitude_conditioned_parameter",
]
