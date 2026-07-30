"""Parameter-space / unit-cube transforms for the conditional flow posterior.

Pure, stateless helpers that map free parameters between the physical
representation, the unit hypercube (uniform-prior coordinates), and the
unconstrained latent the flow is trained in -- including the change-of-variables
Jacobian terms that make the transformed log-densities correct. Extracted from
``flows.posterior`` so the parameter-space maths can be read and tested on its
own.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import vmap


def _cube_epsilon(dtype: Any) -> float:
    """Return the clamp margin keeping cube coordinates inside ``(0, 1)``.

    The logit map diverges at the cube edges, so coordinates are kept this far
    from 0 and 1. Looser for ``float32`` than ``float64`` to stay numerically safe
    at each precision.
    """
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return 1e-7
    return 1e-12


def _clip_cube_support(values: Any) -> jnp.ndarray:
    """Clamp values into the open unit interval ``(epsilon, 1 - epsilon)``."""
    array_values = jnp.asarray(values)
    epsilon = _cube_epsilon(array_values.dtype)
    return jnp.clip(array_values, epsilon, 1.0 - epsilon)


def _cube_to_unconstrained(values: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map unit-cube coordinates to the unconstrained latent via the logit.

    Applies ``u = logit(c) = log(c) - log(1 - c)`` element-wise after clamping
    ``c`` away from the cube edges. This is the inverse of the ``sigmoid`` link
    used to turn the flow's unconstrained output back into a uniform-prior
    coordinate.

    Parameters
    ----------
    values:
        Unit-cube coordinates ``c`` (uniform-prior coordinates), any shape.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(unconstrained_values, clipped_cube_values)``. The clipped cube values
        are returned alongside so the caller can compute the change-of-variables
        Jacobian (:func:`_cube_log_abs_det_jacobian`) consistently with the exact
        coordinates that were transformed.
    """
    cube_values = _clip_cube_support(values)
    unconstrained_values = jnp.log(cube_values) - jnp.log1p(-cube_values)
    return unconstrained_values, cube_values


def _cube_log_abs_det_jacobian(cube_values: jnp.ndarray) -> jnp.ndarray:
    """Log-absolute-determinant of the cube -> unconstrained logit map.

    For ``u = logit(c)`` the per-dimension derivative is ``du/dc = 1 / (c (1 -
    c))``, so ``log|det J| = -sum_i [log(c_i) + log(1 - c_i)]`` summed over the
    parameter axis. Adding this term to a density evaluated in unconstrained
    space converts it to the correct density in cube coordinates (the change of
    variables that makes the uniform-prior log-likelihood exact).

    Parameters
    ----------
    cube_values:
        Unit-cube coordinates of shape ``(..., n_parameters)``.

    Returns
    -------
    jnp.ndarray
        The summed log-abs-det per leading element, shape ``(...,)``.
    """
    return -jnp.sum(jnp.log(cube_values) + jnp.log1p(-cube_values), axis=-1)


def _minimum_cube_spline_bound(dtype: Any = jnp.float32, margin: float = 1.0) -> float:
    """Smallest spline support bound that covers the whole unconstrained cube.

    A cube coordinate clamped to ``[epsilon, 1 - epsilon]`` maps under the logit
    to ``[-L, L]`` with ``L = log((1 - epsilon) / epsilon)``. A spline flow whose
    finite support is narrower than ``L`` would saturate at the edges and lose the
    extreme parameter values, so the spline ``bound`` must be at least this value
    (plus a safety ``margin``).
    """
    epsilon = _cube_epsilon(dtype)
    interior_edge = 1.0 - epsilon
    required = float(np.log(interior_edge) - np.log(epsilon))
    return required + float(margin)


def _parameter_space_log_prob(
    flow: Any,
    parameters: jnp.ndarray,
    embeddings: jnp.ndarray,
    parameter_space: str,
) -> jnp.ndarray:
    """Conditional flow log-density of ``parameters`` in the requested space.

    The flow always models a density in its own unconstrained latent space. For
    ``parameter_space='cube'`` the supplied parameters are uniform-prior cube
    coordinates, so they are pushed to the unconstrained space and the flow's
    log-density is corrected by the change-of-variables term
    (:func:`_cube_log_abs_det_jacobian`) to give ``log q(c | x)`` -- the quantity
    the NPE objective maximises. For any other space the parameters already live
    in the flow's coordinates and the flow log-density is returned directly.

    Parameters
    ----------
    flow:
        Conditional flow exposing ``log_prob(parameters, context)``.
    parameters:
        Parameter batch of shape ``(batch, n_parameters)``.
    embeddings:
        Per-sample conditioning vectors of shape ``(batch, embedding_dim)``.
    parameter_space:
        ``'cube'`` to apply the uniform-prior change of variables, otherwise the
        flow log-density is used unchanged.

    Returns
    -------
    jnp.ndarray
        Per-sample log-densities, shape ``(batch,)``.
    """
    if parameter_space == "cube":
        unconstrained_parameters, cube_parameters = _cube_to_unconstrained(parameters)
        log_probabilities = vmap(flow.log_prob)(unconstrained_parameters, embeddings)
        return log_probabilities + _cube_log_abs_det_jacobian(cube_parameters)

    return vmap(flow.log_prob)(parameters, embeddings)


def _flow_space_parameters(
    parameters: jnp.ndarray,
    parameter_space: str,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Map parameters into the flow's own coordinate space.

    Returns ``(flow_inputs, cube_parameters)``. For ``parameter_space='cube'`` the
    cube coordinates are pushed through the logit so ``flow_inputs`` are the
    unconstrained values the flow's transforms act on, and the (clipped) cube
    coordinates are returned for any downstream Jacobian/diagnostic use. For other
    spaces the parameters are already in flow coordinates and ``cube_parameters``
    is ``None``.
    """
    if parameter_space == "cube":
        unconstrained_parameters, cube_parameters = _cube_to_unconstrained(parameters)
        return unconstrained_parameters, cube_parameters
    return parameters, None
