"""Conditional flow components for amortized SBI posteriors."""

from __future__ import annotations

import math

import equinox as eqx
from jax import Array, vmap, random, nn
import jax.numpy as jnp


class CouplingConditioner(eqx.Module):
    """Context-conditioned network producing coupling parameters.

    Parameters
    ----------
    input_dim:
        Size of the masked parameter vector presented to the conditioner.
    context_dim:
        Size of the observation embedding appended to the masked input.
    output_dim:
        Number of coupling parameters produced by the network.
    hidden_dim:
        Hidden width of the internal MLP.
    key:
        JAX random key used to initialize network weights.

    Notes
    -----
    The conditioner concatenates masked parameter values with the observation
    context before passing them through a small MLP.
    """

    network: eqx.nn.MLP
    input_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int,
        key: Array,
    ):
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.network = eqx.nn.MLP(
            in_size=self.input_dim + self.context_dim,
            out_size=self.output_dim,
            width_size=hidden_dim,
            depth=2,
            key=key,
            dtype=jnp.float32,
        )

    def __call__(self, masked_x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        return self.network(jnp.concatenate((masked_x, context), axis=0))


class _ScalarSplineConditioner(eqx.Module):
    """Context-only network producing scalar spline parameters."""

    network: eqx.nn.MLP
    context_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(self, context_dim: int, output_dim: int, hidden_dim: int, key: Array):
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.network = eqx.nn.MLP(
            in_size=self.context_dim,
            out_size=self.output_dim,
            width_size=hidden_dim,
            depth=2,
            key=key,
            dtype=jnp.float32,
        )

    def __call__(self, context: jnp.ndarray) -> jnp.ndarray:
        return self.network(context)


def _normalize_bin_sizes(
    unnormalized: jnp.ndarray,
    min_bin_size: float,
    total_size: float,
) -> jnp.ndarray:
    probabilities = nn.softmax(unnormalized)
    n_bins = unnormalized.shape[0]
    return min_bin_size + (total_size - min_bin_size * n_bins) * probabilities


def _searchsorted(bin_locations: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    index = jnp.sum(value >= bin_locations) - 1
    return jnp.clip(index, 0, bin_locations.shape[0] - 2)


def _rational_quadratic_spline_forward(
    value: jnp.ndarray,
    unnormalized_widths: jnp.ndarray,
    unnormalized_heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = value.dtype
    bound_value = jnp.asarray(bound, dtype=dtype)
    inside_interval = jnp.logical_and(value >= -bound_value, value <= bound_value)

    widths = _normalize_bin_sizes(unnormalized_widths, min_bin_width, 2.0 * bound)
    heights = _normalize_bin_sizes(unnormalized_heights, min_bin_height, 2.0 * bound)
    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    cumwidths = jnp.concatenate(
        (jnp.asarray([-bound], dtype=dtype), -bound_value + jnp.cumsum(widths, axis=0)),
        axis=0,
    )
    cumheights = jnp.concatenate(
        (
            jnp.asarray([-bound], dtype=dtype),
            -bound_value + jnp.cumsum(heights, axis=0),
        ),
        axis=0,
    )

    bin_index = _searchsorted(cumwidths, value)
    input_cumwidth = cumwidths[bin_index]
    input_bin_width = widths[bin_index]
    input_cumheight = cumheights[bin_index]
    input_bin_height = heights[bin_index]
    input_delta = input_bin_height / input_bin_width
    input_derivative = derivatives[bin_index]
    input_derivative_plus_one = derivatives[bin_index + 1]

    theta = (value - input_cumwidth) / input_bin_width
    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = input_delta + (
        (input_derivative + input_derivative_plus_one - 2.0 * input_delta)
        * theta_one_minus_theta
    )
    denominator = jnp.maximum(denominator, jnp.asarray(1e-8, dtype=dtype))
    numerator = input_bin_height * (
        input_delta * theta**2 + input_derivative * theta_one_minus_theta
    )
    transformed = input_cumheight + numerator / denominator

    derivative_numerator = input_delta**2 * (
        input_derivative_plus_one * theta**2
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivative * (1.0 - theta) ** 2
    )
    derivative_numerator = jnp.maximum(
        derivative_numerator, jnp.asarray(1e-8, dtype=dtype)
    )
    log_det = jnp.log(derivative_numerator) - 2.0 * jnp.log(denominator)

    return (
        jnp.where(inside_interval, transformed, value),
        jnp.where(inside_interval, log_det, jnp.asarray(0.0, dtype=dtype)),
    )


def _rational_quadratic_spline_inverse(
    value: jnp.ndarray,
    unnormalized_widths: jnp.ndarray,
    unnormalized_heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = value.dtype
    bound_value = jnp.asarray(bound, dtype=dtype)
    inside_interval = jnp.logical_and(value >= -bound_value, value <= bound_value)

    widths = _normalize_bin_sizes(unnormalized_widths, min_bin_width, 2.0 * bound)
    heights = _normalize_bin_sizes(unnormalized_heights, min_bin_height, 2.0 * bound)
    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    cumwidths = jnp.concatenate(
        (jnp.asarray([-bound], dtype=dtype), -bound_value + jnp.cumsum(widths, axis=0)),
        axis=0,
    )
    cumheights = jnp.concatenate(
        (
            jnp.asarray([-bound], dtype=dtype),
            -bound_value + jnp.cumsum(heights, axis=0),
        ),
        axis=0,
    )

    bin_index = _searchsorted(cumheights, value)
    input_cumwidth = cumwidths[bin_index]
    input_bin_width = widths[bin_index]
    input_cumheight = cumheights[bin_index]
    input_bin_height = heights[bin_index]
    input_delta = input_bin_height / input_bin_width
    input_derivative = derivatives[bin_index]
    input_derivative_plus_one = derivatives[bin_index + 1]

    shifted_value = value - input_cumheight
    slope_term = input_derivative + input_derivative_plus_one - 2.0 * input_delta
    quadratic = shifted_value * slope_term + input_bin_height * (
        input_delta - input_derivative
    )
    linear = input_bin_height * input_derivative - shifted_value * slope_term
    constant = -input_delta * shifted_value
    discriminant = jnp.maximum(linear**2 - 4.0 * quadratic * constant, 1e-12)
    quadratic_root = (2.0 * constant) / (-linear - jnp.sqrt(discriminant))
    linear_root = -constant / jnp.where(
        jnp.abs(linear) < 1e-12, jnp.asarray(-1.0, dtype=dtype), linear
    )
    root = jnp.where(jnp.abs(quadratic) > 1e-8, quadratic_root, linear_root)
    theta = jnp.clip(root, 0.0, 1.0)
    theta_one_minus_theta = theta * (1.0 - theta)

    denominator = input_delta + slope_term * theta_one_minus_theta
    denominator = jnp.maximum(denominator, jnp.asarray(1e-8, dtype=dtype))
    derivative_numerator = input_delta**2 * (
        input_derivative_plus_one * theta**2
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivative * (1.0 - theta) ** 2
    )
    derivative_numerator = jnp.maximum(
        derivative_numerator, jnp.asarray(1e-8, dtype=dtype)
    )

    transformed = input_cumwidth + theta * input_bin_width
    log_det = 2.0 * jnp.log(denominator) - jnp.log(derivative_numerator)

    return (
        jnp.where(inside_interval, transformed, value),
        jnp.where(inside_interval, log_det, jnp.asarray(0.0, dtype=dtype)),
    )


class ConditionalAffineCoupling(eqx.Module):
    """Affine coupling layer conditioned on an observation embedding.

    Notes
    -----
    The mask selects which parameter dimensions remain fixed and which are
    transformed. The conditioner predicts shift and log-scale terms only from
    the masked input and context embedding.
    """

    mask: jnp.ndarray
    conditioner: CouplingConditioner

    def _parameters(
        self, masked_x: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        outputs = self.conditioner(masked_x, context)
        shift, log_scale = jnp.split(outputs, 2)
        return shift, 2.0 * jnp.tanh(log_scale / 2.0)

    def forward(
        self, x: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the forward affine transform.

        Parameters
        ----------
        x:
            Parameter vector in data space.
        context:
            Observation embedding conditioning the transform.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Transformed vector and scalar log-determinant contribution.
        """
        mask = self.mask.astype(x.dtype)
        inverse_mask = 1.0 - mask
        masked_x = x * mask
        shift, log_scale = self._parameters(masked_x, context)
        y = masked_x + inverse_mask * (x * jnp.exp(log_scale) + shift)
        log_det = jnp.sum(inverse_mask * log_scale)
        return y, log_det

    def inverse(
        self, y: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse affine transform.

        Parameters
        ----------
        y:
            Latent-space vector.
        context:
            Observation embedding conditioning the inverse transform.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Recovered data-space vector and inverse log-determinant.
        """
        mask = self.mask.astype(y.dtype)
        inverse_mask = 1.0 - mask
        masked_y = y * mask
        shift, log_scale = self._parameters(masked_y, context)
        x = masked_y + inverse_mask * ((y - shift) * jnp.exp(-log_scale))
        log_det = -jnp.sum(inverse_mask * log_scale)
        return x, log_det


class ConditionalRationalQuadraticSplineCoupling(eqx.Module):
    """Rational-quadratic spline coupling layer conditioned on an embedding.

    Notes
    -----
    This is the main expressive transform used by spline-based SBI posteriors.
    The conditioner predicts per-dimension spline widths, heights, and
    derivatives for the unmasked subset of the parameter vector.
    """

    mask: jnp.ndarray
    conditioner: CouplingConditioner
    parameter_dim: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    min_bin_width: float = eqx.field(static=True)
    min_bin_height: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)

    def _parameters(
        self,
        masked_x: jnp.ndarray,
        context: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        outputs = self.conditioner(masked_x, context)
        parameters = outputs.reshape(self.parameter_dim, 3 * self.num_bins + 1)
        unnormalized_widths = parameters[:, : self.num_bins]
        unnormalized_heights = parameters[:, self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = parameters[:, 2 * self.num_bins :]
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(
        self, x: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the forward spline transform.

        Parameters
        ----------
        x:
            Parameter vector in data space.
        context:
            Observation embedding conditioning the spline parameters.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Transformed vector and summed log-determinant contribution.
        """
        mask = self.mask.astype(x.dtype)
        inverse_mask = 1.0 - mask
        masked_x = x * mask
        widths, heights, derivatives = self._parameters(masked_x, context)
        transformed, log_det = vmap(
            lambda value, layer_widths, layer_heights, layer_derivatives: _rational_quadratic_spline_forward(
                value,
                layer_widths,
                layer_heights,
                layer_derivatives,
                self.bound,
                self.min_bin_width,
                self.min_bin_height,
                self.min_derivative,
            )
        )(x, widths, heights, derivatives)
        y = masked_x + inverse_mask * transformed
        return y, jnp.sum(inverse_mask * log_det)

    def inverse(
        self, y: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse spline transform.

        Parameters
        ----------
        y:
            Latent-space vector.
        context:
            Observation embedding conditioning the inverse transform.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Recovered parameter vector and inverse log-determinant.
        """
        mask = self.mask.astype(y.dtype)
        inverse_mask = 1.0 - mask
        masked_y = y * mask
        widths, heights, derivatives = self._parameters(masked_y, context)
        transformed, log_det = vmap(
            lambda value, layer_widths, layer_heights, layer_derivatives: _rational_quadratic_spline_inverse(
                value,
                layer_widths,
                layer_heights,
                layer_derivatives,
                self.bound,
                self.min_bin_width,
                self.min_bin_height,
                self.min_derivative,
            )
        )(y, widths, heights, derivatives)
        x = masked_y + inverse_mask * transformed
        return x, jnp.sum(inverse_mask * log_det)


class ConditionalScalarSplineTransform(eqx.Module):
    """Context-conditioned scalar spline transform used for 1D posteriors.

    Notes
    -----
    One-dimensional posterior models cannot rely on alternating coupling masks,
    so they use this dedicated context-only spline transform instead.
    """

    conditioner: _ScalarSplineConditioner
    num_bins: int = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    min_bin_width: float = eqx.field(static=True)
    min_bin_height: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)

    def _parameters(
        self, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        outputs = self.conditioner(context)
        unnormalized_widths = outputs[: self.num_bins]
        unnormalized_heights = outputs[self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = outputs[2 * self.num_bins :]
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    @staticmethod
    def _as_scalar(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(jnp.asarray(value), ())

    def forward(
        self, x: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the forward scalar spline transform.

        Parameters
        ----------
        x:
            One-dimensional parameter vector.
        context:
            Observation embedding conditioning the spline parameters.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Transformed scalar vector and scalar log-determinant.
        """
        widths, heights, derivatives = self._parameters(context)
        transformed, log_det = _rational_quadratic_spline_forward(
            self._as_scalar(x),
            widths,
            heights,
            derivatives,
            self.bound,
            self.min_bin_width,
            self.min_bin_height,
            self.min_derivative,
        )
        return jnp.asarray([transformed], dtype=jnp.asarray(x).dtype), log_det

    def inverse(
        self, y: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the inverse scalar spline transform.

        Parameters
        ----------
        y:
            One-dimensional latent vector.
        context:
            Observation embedding conditioning the inverse transform.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Recovered scalar parameter vector and inverse log-determinant.
        """
        widths, heights, derivatives = self._parameters(context)
        transformed, log_det = _rational_quadratic_spline_inverse(
            self._as_scalar(y),
            widths,
            heights,
            derivatives,
            self.bound,
            self.min_bin_width,
            self.min_bin_height,
            self.min_derivative,
        )
        return jnp.asarray([transformed], dtype=jnp.asarray(y).dtype), log_det


class _ConditionalFlowStack(eqx.Module):
    """Shared stack logic for conditional coupling flows."""

    layers: tuple[object, ...]
    parameter_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    @staticmethod
    def _reverse_vector(values: jnp.ndarray) -> jnp.ndarray:
        if values.shape[0] <= 1:
            return values
        return values[::-1]

    def forward(
        self, x: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        transformed = x
        log_det = jnp.asarray(0.0, dtype=x.dtype)

        for layer_index, layer in enumerate(self.layers):
            transformed, layer_log_det = layer.forward(transformed, context)
            log_det = log_det + layer_log_det
            if layer_index < len(self.layers) - 1:
                transformed = self._reverse_vector(transformed)

        return transformed, log_det

    def inverse(
        self, z: jnp.ndarray, context: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        transformed = z
        log_det = jnp.asarray(0.0, dtype=z.dtype)

        for reverse_index, layer in enumerate(reversed(self.layers)):
            if reverse_index > 0:
                transformed = self._reverse_vector(transformed)
            transformed, layer_log_det = layer.inverse(transformed, context)
            log_det = log_det + layer_log_det

        return transformed, log_det

    def log_prob(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        latent, log_det = self.forward(x, context)
        base_log_prob = -0.5 * jnp.sum(latent**2 + math.log(2.0 * math.pi))
        return base_log_prob + log_det

    def sample(self, context: jnp.ndarray, n_samples: int, key: Array) -> jnp.ndarray:
        latent_samples = random.normal(
            key, shape=(n_samples, self.parameter_dim), dtype=context.dtype
        )

        def sample_single(latent_sample: jnp.ndarray) -> jnp.ndarray:
            sample, _ = self.inverse(latent_sample, context)
            return sample

        return vmap(sample_single)(latent_samples)


class ConditionalAffineFlow(_ConditionalFlowStack):
    """Stack of conditional affine couplings with alternating masks.

    Parameters
    ----------
    parameter_dim:
        Number of inferred parameters transformed by the flow.
    context_dim:
        Size of the observation embedding conditioning each layer.
    num_layers:
        Number of affine coupling layers.
    hidden_dim:
        Hidden width of each conditioner MLP.
    key:
        Optional JAX random key used for layer initialization.

    Notes
    -----
    For one-dimensional problems the mask degenerates to a context-only affine
    transform. Higher-dimensional problems alternate masks across layers.
    """

    def __init__(
        self,
        parameter_dim: int,
        context_dim: int,
        num_layers: int = 4,
        hidden_dim: int = 128,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(0)

        self.parameter_dim = int(parameter_dim)
        self.context_dim = int(context_dim)
        layer_keys = random.split(key, num_layers)
        layers = []

        for index, layer_key in enumerate(layer_keys):
            if self.parameter_dim == 1:
                mask = jnp.asarray([0.0], dtype=jnp.float32)
            else:
                mask = jnp.asarray(
                    [
                        float((dimension + index) % 2 == 0)
                        for dimension in range(self.parameter_dim)
                    ],
                    dtype=jnp.float32,
                )
            layers.append(
                ConditionalAffineCoupling(
                    mask=mask,
                    conditioner=CouplingConditioner(
                        input_dim=self.parameter_dim,
                        context_dim=self.context_dim,
                        output_dim=self.parameter_dim * 2,
                        hidden_dim=hidden_dim,
                        key=layer_key,
                    ),
                )
            )

        self.layers = tuple(layers)


class ConditionalSplineFlow(_ConditionalFlowStack):
    """Stack of conditional rational-quadratic spline transforms.

    Parameters
    ----------
    parameter_dim:
        Number of inferred parameters transformed by the flow.
    context_dim:
        Size of the conditioning observation embedding.
    num_layers:
        Number of spline transform layers.
    hidden_dim:
        Hidden width of the conditioner networks.
    num_bins:
        Number of rational-quadratic bins used per transformed dimension.
    bound:
        Finite support bound of the spline transform in latent coordinates.
    min_bin_width, min_bin_height, min_derivative:
        Numerical-stability floors applied to spline parameters.
    key:
        Optional JAX random key used for initialization.

    Notes
    -----
    One-dimensional posteriors use a dedicated scalar spline stack, while
    higher-dimensional models use alternating masked coupling layers.
    """

    num_bins: int = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    min_bin_width: float = eqx.field(static=True)
    min_bin_height: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)

    def __init__(
        self,
        parameter_dim: int,
        context_dim: int,
        num_layers: int = 6,
        hidden_dim: int = 128,
        num_bins: int = 8,
        bound: float = 10.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(0)

        self.parameter_dim = int(parameter_dim)
        self.context_dim = int(context_dim)
        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        layer_keys = random.split(key, num_layers)
        layers = []

        if self.parameter_dim == 1:
            for layer_key in layer_keys:
                layers.append(
                    ConditionalScalarSplineTransform(
                        conditioner=_ScalarSplineConditioner(
                            context_dim=self.context_dim,
                            output_dim=3 * self.num_bins + 1,
                            hidden_dim=hidden_dim,
                            key=layer_key,
                        ),
                        num_bins=self.num_bins,
                        bound=self.bound,
                        min_bin_width=self.min_bin_width,
                        min_bin_height=self.min_bin_height,
                        min_derivative=self.min_derivative,
                    )
                )

            self.layers = tuple(layers)
            return

        for index, layer_key in enumerate(layer_keys):
            mask = jnp.asarray(
                [
                    float((dimension + index) % 2 == 0)
                    for dimension in range(self.parameter_dim)
                ],
                dtype=jnp.float32,
            )
            layers.append(
                ConditionalRationalQuadraticSplineCoupling(
                    mask=mask,
                    conditioner=CouplingConditioner(
                        input_dim=self.parameter_dim,
                        context_dim=self.context_dim,
                        output_dim=self.parameter_dim * (3 * self.num_bins + 1),
                        hidden_dim=hidden_dim,
                        key=layer_key,
                    ),
                    parameter_dim=self.parameter_dim,
                    num_bins=self.num_bins,
                    bound=self.bound,
                    min_bin_width=self.min_bin_width,
                    min_bin_height=self.min_bin_height,
                    min_derivative=self.min_derivative,
                )
            )

        self.layers = tuple(layers)


__all__ = [
    "ConditionalAffineCoupling",
    "ConditionalAffineFlow",
    "ConditionalRationalQuadraticSplineCoupling",
    "ConditionalScalarSplineTransform",
    "ConditionalSplineFlow",
    "CouplingConditioner",
]
