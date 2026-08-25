"""
The encoder is a learned function whose architecture directly controls what
information the flow can condition on. The default deliberately separates spectral
**amplitude** from **shape** so the posterior cannot collapse to the prior on
shape-determined parameters (composition, temperature, clouds).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jax import Array, vmap, random
import jax.numpy as jnp

from .observation import (
    EncodedObservation,
    ObservationBlock,
    AbstractObservationEncoder,
    ObservationModality,
)


# --- Helper functions for normalising and encoding observations ---------
#
def _as_vector(values: Any) -> jnp.ndarray:
    if values is None:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.ravel(jnp.asarray(values, dtype=jnp.float32))


def _safe_mask(mask: Any, size: int) -> jnp.ndarray:
    if size == 0:
        return jnp.zeros((0,), dtype=bool)
    if mask is None:
        return jnp.zeros((size,), dtype=bool)

    flattened_mask = jnp.ravel(jnp.asarray(mask, dtype=bool))
    if flattened_mask.shape[0] == size:
        return flattened_mask

    if flattened_mask.shape[0] > size:
        return flattened_mask[:size]

    padding = jnp.zeros((size - flattened_mask.shape[0],), dtype=bool)
    return jnp.concatenate((flattened_mask, padding), axis=0)


def _merge_embedding_branches(*branches: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.ravel(jnp.asarray(branch, dtype=jnp.float32)) for branch in branches],
        axis=0,
    )


def _within_spectrum_standardized(
    values: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Standardize a spectrum across its own valid wavelengths.

    Subtracting the per-spectrum mean removes the overall amplitude (which is
    a near-constant offset across wavelengths for a single transmission
    observation, set mostly by the planet radius), and dividing by the
    per-spectrum standard deviation rescales the residual molecular features to
    order unity. This exposes the wavelength-dependent *shape* that is
    otherwise a sub-percent ripple riding on top of the dominant amplitude in
    the globally normalized value channel, so the conditioner can actually see
    the information that constrains composition, temperature, and clouds.
    """
    if values.shape[0] == 0:
        return values
    valid = valid.astype(values.dtype)
    count = jnp.maximum(jnp.sum(valid), jnp.asarray(1.0, dtype=values.dtype))
    mean = jnp.sum(values * valid) / count
    centered = (values - mean) * valid
    variance = jnp.sum(jnp.square(centered)) / count
    scale = jnp.sqrt(variance + jnp.asarray(1e-12, dtype=values.dtype))
    standardized = centered / jnp.maximum(scale, jnp.asarray(1e-6, dtype=values.dtype))
    return standardized * valid


def _resize_embedding(features: jnp.ndarray, embedding_dim: int) -> jnp.ndarray:
    features = jnp.ravel(jnp.asarray(features, dtype=jnp.float32))

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")

    if features.shape[0] == embedding_dim:
        return features

    if features.shape[0] == 0:
        return jnp.zeros((embedding_dim,), dtype=jnp.float32)

    if features.shape[0] > embedding_dim:
        positions = jnp.linspace(0, features.shape[0] - 1, embedding_dim)
        return jnp.interp(positions, jnp.arange(features.shape[0]), features)

    padding = jnp.zeros((embedding_dim - features.shape[0],), dtype=jnp.float32)
    return jnp.concatenate((features, padding), axis=0)


# Position of the standardised exact-summary vector inside a pre-stacked block
# tuple; see ``PreStackedObservations`` for the full field order. Both vmapped
# encode paths build the same layout so this index means the same thing in each.
_EXACT_SUMMARY_FIELD_INDEX = 7


def _fit_vector(values: Any, size: int) -> jnp.ndarray:
    """Pad or truncate a 1-D vector to exactly ``size`` elements.

    Unlike :func:`_resize_embedding` this never interpolates: the exact-summary
    channel carries individually meaningful scalars, and blending neighbouring
    entries would destroy the very precision the channel exists to preserve.
    """
    features = jnp.ravel(jnp.asarray(values, dtype=jnp.float32))
    if size <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    if features.shape[0] == size:
        return features
    if features.shape[0] > size:
        return features[:size]
    return jnp.concatenate(
        (features, jnp.zeros((size - features.shape[0],), dtype=jnp.float32)), axis=0
    )


def _scalar_feature(value: Any) -> jnp.ndarray:
    if value is None:
        return jnp.zeros((1,), dtype=jnp.float32)

    scalar = jnp.ravel(jnp.asarray(value, dtype=jnp.float32))
    if scalar.shape[0] == 0:
        return jnp.zeros((1,), dtype=jnp.float32)
    return scalar[:1]


def _log_scale_from_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return 0.0
    if "log_spectrum_scale" in metadata:
        return metadata.get("log_spectrum_scale")
    return metadata.get("log_median_flux", 0.0)


def _metadata_vector(metadata: Any, key: str, size: int) -> jnp.ndarray:
    if not isinstance(metadata, dict) or key not in metadata:
        return jnp.zeros((size,), dtype=jnp.float32)

    values = jnp.ravel(jnp.asarray(metadata.get(key), dtype=jnp.float32))
    if values.shape[0] == size:
        return values
    if values.shape[0] > size:
        return values[:size]

    padding = jnp.zeros((size - values.shape[0],), dtype=jnp.float32)
    return jnp.concatenate((values, padding), axis=0)


def _spectral_auxiliary_vector(metadata: Any, size: int) -> jnp.ndarray:
    if isinstance(metadata, dict) and "relative_values" in metadata:
        return _metadata_vector(metadata, "relative_values", size)
    return _metadata_vector(metadata, "absolute_values", size)


# --- Robust amplitude/shape decomposition for the convolution encoder ---------
#
# These are the JAX, differentiable, vmappable analogue of the per-spectrum
# normalisation that conceptually belongs to preprocessing. They live here (next
# to ``_within_spectrum_standardized``) rather than in the numpy-only
# ``preprocessing.py`` because they must run *inside* the encoder forward pass,
# which is vmapped over the training batch and differentiated. They are
# observable-agnostic: only masked values + uncertainties + coordinates enter,
# so the same code factors a transmission depth spectrum and an emission flux
# spectrum (whose amplitude spans orders of magnitude -> captured by ``log(scale)``).
_GENERIC_AMPLITUDE_DIM = 6


def _robust_center_scale(
    values: jnp.ndarray, valid: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-spectrum robust center (median) and scale (MAD) over valid points."""
    masked = jnp.where(valid > 0.0, values, jnp.nan)
    center = jnp.nanmedian(masked)
    mad = jnp.nanmedian(jnp.abs(masked - center))
    center = jnp.where(jnp.isfinite(center), center, jnp.asarray(0.0, values.dtype))
    # 1.4826 * MAD is a consistent std estimator for Gaussian noise.
    scale = jnp.where(
        jnp.isfinite(mad) & (mad > 0.0),
        mad * jnp.asarray(1.4826, values.dtype),
        jnp.asarray(1.0, values.dtype),
    )
    scale = jnp.maximum(scale, jnp.asarray(1e-6, values.dtype))
    return center, scale


def _normalized_coordinate_channel(
    coordinates: jnp.ndarray, valid: jnp.ndarray
) -> jnp.ndarray:
    """Map the wavelength grid to [0, 1] over valid points (a positional channel)."""
    masked = jnp.where(valid > 0.0, coordinates, jnp.nan)
    lo = jnp.nanmin(masked)
    hi = jnp.nanmax(masked)
    lo = jnp.where(jnp.isfinite(lo), lo, jnp.asarray(0.0, coordinates.dtype))
    hi = jnp.where(jnp.isfinite(hi), hi, jnp.asarray(1.0, coordinates.dtype))
    span = jnp.maximum(hi - lo, jnp.asarray(1e-6, coordinates.dtype))
    norm = (coordinates - lo) / span
    return jnp.clip(norm, 0.0, 1.0).astype(jnp.float32) * valid


def _generic_shape_tensor(
    values: jnp.ndarray,
    uncertainties: jnp.ndarray,
    coordinates: jnp.ndarray,
    valid: jnp.ndarray,
    center: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """Spectral channels ``[v, (v-center)/scale, sigma/scale, lambda, mask]``.

    The per-spectrum standardized channel exposes scale-invariant shape; the
    raw (preprocessing-normalized) value channel keeps the absolute continuum
    level and absolute feature amplitude visible to the convolution stack at
    per-wavelength resolution. Without it, both quantities reach the flow only
    as two scalars of the low-dimensional amplitude branch — which is exactly
    the information that breaks the transmission radius/atmosphere-height
    degeneracy, and squeezing it through scalars left the radius posterior
    biased toward the prior edge for low-depth spectra. With per-wavelength
    robust preprocessing the raw channel is O(1), so it is safe conv input.
    """
    shape_v = ((values - center) / scale) * valid
    shape_u = (jnp.abs(uncertainties) / scale) * valid
    coordinate = _normalized_coordinate_channel(coordinates, valid)
    raw_v = values * valid
    return jnp.stack(
        [raw_v, shape_v, shape_u, coordinate, valid], axis=0
    ).astype(jnp.float32)


def _generic_amplitude_features(
    values: jnp.ndarray,
    uncertainties: jnp.ndarray,
    valid: jnp.ndarray,
    center: jnp.ndarray,
    scale: jnp.ndarray,
    log_scale: Any,
) -> jnp.ndarray:
    """Low-dim amplitude path: global log-scale + per-spectrum offset/scale/sign stats."""
    count = jnp.maximum(jnp.sum(valid), jnp.asarray(1.0, valid.dtype))
    valid_fraction = jnp.sum(valid) / jnp.asarray(max(int(valid.shape[0]), 1), valid.dtype)
    positive_fraction = (
        jnp.sum(((values > center).astype(valid.dtype)) * valid) / count
    )
    mean_uncertainty = (jnp.sum(jnp.abs(uncertainties) * valid) / count) / scale
    return jnp.stack(
        [
            _scalar_feature(log_scale)[0],
            center.astype(jnp.float32),
            jnp.log(scale).astype(jnp.float32),
            valid_fraction.astype(jnp.float32),
            positive_fraction.astype(jnp.float32),
            mean_uncertainty.astype(jnp.float32),
        ]
    ).astype(jnp.float32)


# --- Spectral encoders implementations --------------------------------------

class SpectralConvolutionEncoder(eqx.Module):
    """ The default and recommended encoder.
    Observable-agnostic spectral encoder that explicitly factors a spectrum
    into *amplitude* and *shape*, so the network can never collapse to an
    amplitude-only summary (the failure mode that left composition / temperature
    / cloud marginals at the prior).

    The spectrum is split into:

    * **amplitude features** (low-dim): the global ``log(scale)`` plus the
      per-spectrum robust center / scale / sign statistics. For emission this
      carries the orders-of-magnitude flux scale; for transmission the baseline
      transit depth.
    * a **shape tensor** ``[(v-center)/scale, sigma/scale, normalized_lambda,
      mask]`` with the amplitude divided out.

    The shape tensor is encoded by a *dilated* 1-D convolution stack
    (dilations 1, 2, 4, ...), giving a receptive field that spans the whole band.
    A learned-query attention pools the per-wavelength features into a global
    shape embedding. Amplitude enters through a separate, un-bottlenecked MLP,
    so the shape network never spends capacity re-encoding amplitude.
    The two are concatenated, fused by an MLP, and LayerNorm-normalised.

    Parameters
    ----------
    embedding_dim:
        Size of the spectral embedding returned for one observation block.
         This is the length of the summary vector handed to the flow.
         Bigger = more capacity to describe the spectrum, but more parameters
         to train. Defaults to 64.
    n_wavelengths:
        Fixed number of spectral points per observation block. Inputs are
        padded or truncated to this length (see :meth:`_pad_to_fixed`), so it
        must match the observation schema of the SBI task.
    channels:
        Width of the convolution stack and attention projections, i.e. the
        number of feature channels carried through the shape branch.
    dilations:
        Dilation factor of each residual dilated convolution, applied in order.
        The successive doubling ``(1, 2, 4, ...)`` grows the receptive field
        geometrically so the stack spans the whole band with few layers.
        Controls how far along the spectrum the network can "see" at each layer.
        A convolution normally only looks at immediate neighbors. "Dilation" spreads
        its view out with gaps: dilation 1 looks at adjacent points, dilation 2 skips
        every other point, and so on.
    key:
        Optional JAX random key used for weight initialization. Defaults to
        ``random.PRNGKey(0)`` when omitted.

    Attributes
    ----------
    input_projection:
        ``1x1`` :class:`equinox.nn.Conv1d` lifting the 5 spectral channels
        ``[(v-center)/scale, sigma/scale, normalized_lambda, mask]`` to
        ``channels`` feature channels.
    --- Shape branch ---
    dilated_convs:
        Tuple of :class:`equinox.nn.Conv1d` layers (one per entry in
        ``dilations``) applied as GELU residual blocks to build the
        wide-receptive-field shape representation.
    attention_key:
        :class:`equinox.nn.Linear` producing the per-wavelength attention keys
        used by the learned-query pooling.
    attention_value:
        :class:`equinox.nn.Linear` producing the per-wavelength attention
        values that are weighted and summed into the shape embedding.
    attention_query:
        Learned query vector of shape ``(channels,)`` scored against every
        wavelength's key to focus pooling on informative band-heads.
    --- Amplitude branch ---
    amplitude_projection:
        :class:`equinox.nn.MLP` mapping the low-dimensional amplitude features
        (global ``log(scale)`` plus per-spectrum center/scale/sign statistics)
        into a ``channels``-wide amplitude embedding, kept separate so the
        shape branch never re-encodes amplitude.
    fusion:
        :class:`equinox.nn.MLP` fusing the concatenated shape and amplitude
        embeddings (``2 * channels`` wide) down to ``embedding_dim``.
    norm:
        :class:`equinox.nn.LayerNorm` applied to the fused embedding so the
        flow conditioner receives a consistently scaled vector.
    embedding_dim:
        Static output embedding size (mirrors the constructor argument).
    n_wavelengths:
        Static fixed spectral length inputs are padded/truncated to.
    channels:
        Static convolution / attention channel width.
    """

    input_projection: eqx.nn.Conv1d
    dilated_convs: tuple
    attention_key: eqx.nn.Linear
    attention_value: eqx.nn.Linear
    attention_query: jnp.ndarray
    amplitude_projection: eqx.nn.MLP
    fusion: eqx.nn.MLP
    norm: eqx.nn.LayerNorm
    embedding_dim: int = eqx.field(static=True)
    n_wavelengths: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 64,
        n_wavelengths: int = 233,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        key: Array | None = None,
    ):
        """Initialize the encoder's convolution, attention, and fusion modules.

        See the class docstring for a description of each parameter and the
        attributes it initializes.
        """
        if key is None:
            key = random.PRNGKey(0)

        self.embedding_dim = int(embedding_dim)
        self.n_wavelengths = int(n_wavelengths)
        self.channels = int(channels)

        keys = random.split(key, 5 + len(dilations))
        # 5 spectral channels (raw normalized values + standardized shape +
        # scaled uncertainty + coordinate + validity) -> channels.
        self.input_projection = eqx.nn.Conv1d(
            5, self.channels, kernel_size=1, key=keys[0], dtype=jnp.float32
        )
        self.dilated_convs = tuple(
            eqx.nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                key=keys[1 + index],
                dtype=jnp.float32,
            )
            for index, dilation in enumerate(dilations)
        )
        attn_key_k = keys[1 + len(dilations)]
        attn_val_k = keys[2 + len(dilations)]
        amp_k = keys[3 + len(dilations)]
        fusion_k = keys[4 + len(dilations)]
        self.attention_key = eqx.nn.Linear(
            self.channels, self.channels, key=attn_key_k, dtype=jnp.float32
        )
        self.attention_value = eqx.nn.Linear(
            self.channels, self.channels, key=attn_val_k, dtype=jnp.float32
        )
        self.attention_query = (
            0.02 * random.normal(amp_k, (self.channels,), dtype=jnp.float32)
        )
        self.amplitude_projection = eqx.nn.MLP(
            _GENERIC_AMPLITUDE_DIM,
            self.channels,
            self.channels,
            1,
            key=amp_k,
            dtype=jnp.float32,
        )
        self.fusion = eqx.nn.MLP(
            2 * self.channels,
            self.embedding_dim,
            max(self.embedding_dim, self.channels),
            2,
            key=fusion_k,
            dtype=jnp.float32,
        )
        self.norm = eqx.nn.LayerNorm(self.embedding_dim)

    def _forward(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None,
    ) -> jnp.ndarray:
        return self._forward_with_stages(
            values, uncertainties, coordinates, mask, log_median_flux
        )[0]

    def _forward_with_stages(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """The forward pass, also returning each intermediate representation.

        :meth:`_forward` delegates here, so an information audit of the stages
        measures the tensors the trained encoder actually produces. Rebuilding
        the forward pass separately to inspect it is how a previous audit
        reported R^2 of exactly 0.00000 for three consecutive stages: the
        reconstruction had computed the per-spectrum centre and scale itself and
        handed the amplitude in at the wrong layer.
        """
        valid = (~mask).astype(jnp.float32)
        center, scale = _robust_center_scale(values, valid)

        shape = _generic_shape_tensor(
            values, uncertainties, coordinates, valid, center, scale
        )  # (5, n_wl)
        features = self.input_projection(shape)  # (channels, n_wl)

        # Gaussian Error Linear Unit activation function
        for conv in self.dilated_convs:
            features = features + jax.nn.gelu(conv(features))  # residual dilated conv
        features = features * valid[None, :]

        # Learned-query attention pooling over the wavelength axis: a global
        # reduction that, unlike mean/max pooling, can focus on the informative
        # band-heads anywhere in the spectrum.
        per_wavelength = features.T  # (n_wl, channels)
        keys_ = vmap(self.attention_key)(per_wavelength)  # (n_wl, channels)
        values_ = vmap(self.attention_value)(per_wavelength)  # (n_wl, channels)
        scores = (keys_ @ self.attention_query) / jnp.sqrt(
            jnp.asarray(self.channels, dtype=jnp.float32)
        )
        scores = jnp.where(valid > 0.0, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))
        weights = jax.nn.softmax(scores)
        weights = jnp.where(jnp.isfinite(weights), weights, 0.0)
        shape_embedding = jnp.sum(weights[:, None] * values_, axis=0)  # (channels,)

        amplitude_features = _generic_amplitude_features(
            values, uncertainties, valid, center, scale, log_median_flux
        )
        amplitude_embedding = jax.nn.gelu(
            self.amplitude_projection(amplitude_features)
        )

        fused = self.fusion(
            _merge_embedding_branches(shape_embedding, amplitude_embedding)
        )
        normalized = self.norm(fused)
        stages = {
            # In pipeline order, so the first stage whose R^2 drops is the one
            # that discards the information.
            "normalized_values": values * valid,
            "shape_tensor": jnp.ravel(shape),
            "conv_features": jnp.ravel(features),
            "pooled_shape": shape_embedding,
            "amplitude_features": amplitude_features,
            "amplitude_embedding": amplitude_embedding,
            "fused": fused,
            "fused_layer_normed": normalized,
        }
        return normalized, stages

    def _pad_to_fixed(self, arr: jnp.ndarray) -> jnp.ndarray:
        n = arr.shape[0]
        if n == self.n_wavelengths:
            return arr
        if n > self.n_wavelengths:
            return arr[: self.n_wavelengths]
        return jnp.concatenate(
            [arr, jnp.zeros(self.n_wavelengths - n, dtype=arr.dtype)]
        )

    def _prepare_arrays(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        n: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = self._pad_to_fixed(values)
        uncertainties = (
            jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
            if uncertainties.shape[0] == 0
            else self._pad_to_fixed(uncertainties)
        )
        coordinates = (
            jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
            if coordinates.shape[0] == 0
            else self._pad_to_fixed(coordinates)
        )
        pad_mask = jnp.arange(self.n_wavelengths) >= n
        mask = self._pad_to_fixed(mask.astype(jnp.float32)).astype(bool) | pad_mask
        return values, uncertainties, coordinates, mask

    def encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one spectral observation block."""
        values = _as_vector(block.values)
        uncertainties = _as_vector(block.uncertainties)
        coordinates = _as_vector(block.coordinates)
        n = values.shape[0]
        if n == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)
        mask = _safe_mask(block.mask, n)
        values, uncertainties, coordinates, mask = self._prepare_arrays(
            values, uncertainties, coordinates, mask, n
        )
        return self._forward(
            values,
            uncertainties,
            coordinates,
            mask,
            _log_scale_from_metadata(block.metadata),
        )

    def _encode_block_raw(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None = None,
        absolute_values: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Encode pre-processed block arrays (vmappable — no ObservationBlock).

        The robust decomposition is computed directly from ``values``.
        """
        del absolute_values
        n = values.shape[0]
        if n == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)
        values, uncertainties, coordinates, mask = self._prepare_arrays(
            values, uncertainties, coordinates, mask, n
        )
        return self._forward(values, uncertainties, coordinates, mask, log_median_flux)

    def stage_features_raw(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Flattened intermediate representations for one block (vmappable)."""
        n = values.shape[0]
        values, uncertainties, coordinates, mask = self._prepare_arrays(
            values, uncertainties, coordinates, mask, n
        )
        return self._forward_with_stages(
            values, uncertainties, coordinates, mask, log_median_flux
        )[1]


class _ResidualDenseBlock(eqx.Module):
    """One pre-activation residual MLP block: ``x + W2 elu(W1 elu(x))``."""

    first: eqx.nn.Linear
    second: eqx.nn.Linear

    def __init__(self, width: int, key: Array):
        k1, k2 = random.split(key)
        self.first = eqx.nn.Linear(width, width, key=k1, dtype=jnp.float32)
        self.second = eqx.nn.Linear(width, width, key=k2, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.second(jax.nn.elu(self.first(jax.nn.elu(x))))


class SpectralResMLPEncoder(eqx.Module):
    """Plain residual-MLP spectral encoder (the published-recipe architecture).

    A deliberately simple alternative to :class:`SpectralConvolutionEncoder`,
    following the embedding networks of the working pRT NPE setups (Vasist et
    al. 2023: a ResMLP of decreasing widths compressing the spectrum to a small
    feature vector; Gebhard et al. 2025: a deep dense ResNet). The whole
    (preprocessing-normalized) spectrum, its scaled uncertainties, and the
    per-spectrum log-amplitude scalar enter one dense stack together, so the
    first layer can represent arbitrary global linear functionals of the
    spectrum -- matched filters for molecular band contrasts -- natively,
    rather than assembling them from local convolution kernels through an
    attention bottleneck.

    The block ladder follows Vasist et al. (two blocks at full width, three at
    half, five at quarter), scaled by ``hidden_dim`` (=512 reproduces their
    2x512, 3x256, 5x128).
    """

    input_projection: eqx.nn.Linear
    blocks: tuple[_ResidualDenseBlock, ...]
    width_projections: tuple[eqx.nn.Linear | None, ...]
    output_projection: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)
    n_wavelengths: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 64,
        n_wavelengths: int = 233,
        hidden_dim: int = 512,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(0)

        self.embedding_dim = int(embedding_dim)
        self.n_wavelengths = int(n_wavelengths)

        widths = (
            [int(hidden_dim)] * 2
            + [max(int(hidden_dim) // 2, 16)] * 3
            + [max(int(hidden_dim) // 4, 16)] * 5
        )
        keys = random.split(key, 2 + 2 * len(widths))
        # values + uncertainties + log-amplitude scalar.
        input_dim = 2 * self.n_wavelengths + 1
        self.input_projection = eqx.nn.Linear(
            input_dim, widths[0], key=keys[0], dtype=jnp.float32
        )
        blocks = []
        width_projections: list[eqx.nn.Linear | None] = []
        current = widths[0]
        for i, width in enumerate(widths):
            if width != current:
                width_projections.append(
                    eqx.nn.Linear(current, width, key=keys[1 + 2 * i], dtype=jnp.float32)
                )
                current = width
            else:
                width_projections.append(None)
            blocks.append(_ResidualDenseBlock(width, keys[2 + 2 * i]))
        self.blocks = tuple(blocks)
        self.width_projections = tuple(width_projections)
        self.output_projection = eqx.nn.Linear(
            current, self.embedding_dim, key=keys[-1], dtype=jnp.float32
        )

    def _pad_to_fixed(self, arr: jnp.ndarray) -> jnp.ndarray:
        n = arr.shape[0]
        if n == self.n_wavelengths:
            return arr
        if n > self.n_wavelengths:
            return arr[: self.n_wavelengths]
        return jnp.concatenate(
            [arr, jnp.zeros(self.n_wavelengths - n, dtype=arr.dtype)]
        )

    def _forward(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: Any,
    ) -> jnp.ndarray:
        valid = (~mask).astype(jnp.float32)
        features = jnp.concatenate(
            [
                values.astype(jnp.float32) * valid,
                uncertainties.astype(jnp.float32) * valid,
                _scalar_feature(log_median_flux),
            ]
        )
        x = self.input_projection(features)
        for projection, block in zip(self.width_projections, self.blocks):
            if projection is not None:
                x = projection(x)
            x = block(x)
        return self.output_projection(jax.nn.elu(x))

    def encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one spectral observation block."""
        values = _as_vector(block.values)
        n = values.shape[0]
        if n == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)
        uncertainties = _as_vector(block.uncertainties)
        mask = _safe_mask(block.mask, n)
        values = self._pad_to_fixed(values)
        uncertainties = (
            jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
            if uncertainties.shape[0] == 0
            else self._pad_to_fixed(uncertainties)
        )
        pad_mask = jnp.arange(self.n_wavelengths) >= n
        mask = self._pad_to_fixed(mask.astype(jnp.float32)).astype(bool) | pad_mask
        return self._forward(
            values, uncertainties, mask, _log_scale_from_metadata(block.metadata)
        )

    def _encode_block_raw(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None = None,
        absolute_values: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Encode pre-processed block arrays (vmappable — no ObservationBlock)."""
        del coordinates, absolute_values
        n = values.shape[0]
        if n == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)
        values = self._pad_to_fixed(values)
        uncertainties = self._pad_to_fixed(uncertainties)
        pad_mask = jnp.arange(self.n_wavelengths) >= n
        mask = self._pad_to_fixed(mask.astype(jnp.float32)).astype(bool) | pad_mask
        return self._forward(values, uncertainties, mask, log_median_flux)


class PhotometryPointEncoder(eqx.Module):
    """Learned photometry encoder using per-point MLP features and pooling.

    Parameters
    ----------
    embedding_dim:
        Size of the returned photometric embedding.
    hidden_dim:
        Hidden width of the per-point MLP.
    key:
        Optional JAX random key used for initialization.

    Notes
    -----
    Each photometric point is represented by value, uncertainty, coordinate,
    and an inferred width feature before permutation-invariant pooling.
    """

    point_mlp: eqx.nn.MLP
    output_projection: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self, embedding_dim: int = 64, hidden_dim: int = 96, key: Array | None = None
    ):
        if key is None:
            key = random.PRNGKey(1)

        mlp_key, projection_key = random.split(key)
        self.embedding_dim = int(embedding_dim)
        self.point_mlp = eqx.nn.MLP(4, self.embedding_dim, hidden_dim, 2, key=mlp_key, dtype=jnp.float32)
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim, self.embedding_dim, key=projection_key, dtype=jnp.float32
        )

    def encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one photometric observation block.

        Parameters
        ----------
        block:
            Photometry-like observation block to encode.

        Returns
        -------
        jnp.ndarray
            Dense embedding for the supplied photometric block.
        """
        values = _as_vector(block.values)
        uncertainties = _as_vector(block.uncertainties)
        coordinates = _as_vector(block.coordinates)
        mask = _safe_mask(block.mask, values.shape[0])

        if values.shape[0] == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        if uncertainties.shape[0] == 0:
            uncertainties = jnp.zeros_like(values)
        if coordinates.shape[0] == 0:
            coordinates = jnp.zeros_like(values)

        widths = jnp.concatenate(
            (jnp.diff(coordinates), jnp.zeros((1,), dtype=coordinates.dtype)), axis=0
        )
        point_features = jnp.stack(
            (values, uncertainties, coordinates, widths), axis=-1
        )
        encoded_points = vmap(self.point_mlp)(point_features)
        valid_weights = (~mask).astype(jnp.float32)[:, None]
        pooled = jnp.sum(encoded_points * valid_weights, axis=0) / jnp.maximum(
            jnp.sum(valid_weights), 1.0
        )
        return self.output_projection(pooled)

    def _encode_block_raw(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Encode pre-processed block arrays (vmappable — no ObservationBlock input).

        Parameters
        ----------
        values, uncertainties, coordinates:
            Float32 1-D arrays already produced by ``_as_vector``.
            ``uncertainties`` and ``coordinates`` may have length 0.
        mask:
            Boolean 1-D array already produced by ``_safe_mask``.

        Returns
        -------
        jnp.ndarray
            Dense photometric embedding of shape ``(embedding_dim,)``.
        """
        if values.shape[0] == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        if uncertainties.shape[0] == 0:
            uncertainties = jnp.zeros_like(values)
        if coordinates.shape[0] == 0:
            coordinates = jnp.zeros_like(values)

        widths = jnp.concatenate(
            (jnp.diff(coordinates), jnp.zeros((1,), dtype=coordinates.dtype)), axis=0
        )
        point_features = jnp.stack(
            (values, uncertainties, coordinates, widths), axis=-1
        )
        encoded_points = vmap(self.point_mlp)(point_features)
        valid_weights = (~mask).astype(jnp.float32)[:, None]
        pooled = jnp.sum(encoded_points * valid_weights, axis=0) / jnp.maximum(
            jnp.sum(valid_weights), 1.0
        )
        return self.output_projection(pooled)


class DatasetSetAggregator(eqx.Module):
    """Learned permutation-invariant aggregator over block embeddings.

    Parameters
    ----------
    embedding_dim:
        Target dimensionality of the aggregated observation embedding.
    hidden_dim:
        Hidden width of the block projection MLP.
    key:
        Optional JAX random key used for initialization.
    """

    block_projection: eqx.nn.MLP
    output_projection: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self, embedding_dim: int = 128, hidden_dim: int = 128, key: Array | None = None
    ):
        if key is None:
            key = random.PRNGKey(2)

        mlp_key, projection_key = random.split(key)
        self.embedding_dim = int(embedding_dim)
        self.block_projection = eqx.nn.MLP(
            self.embedding_dim, self.embedding_dim, hidden_dim, 2, key=mlp_key, dtype=jnp.float32
        )
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim * 3, self.embedding_dim, key=projection_key, dtype=jnp.float32
        )

    def aggregate(self, block_embeddings: list[jnp.ndarray]) -> jnp.ndarray:
        """Aggregate a list of block embeddings into one observation embedding.

        Parameters
        ----------
        block_embeddings:
            Encoded observation blocks for one retrieval target.

        Returns
        -------
        jnp.ndarray
            Permutation-invariant aggregated embedding.
        """
        if not block_embeddings:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        # A single block still flows through the learned block/output
        # projections (mean == max == the projection, std == 0). The previous
        # len == 1 shortcut returned the raw block embedding, bypassing those
        # learned layers entirely -- the common single-spectrum case never
        # trained them.
        stacked = jnp.stack(block_embeddings, axis=0)
        projected = vmap(self.block_projection)(stacked)
        pooled = jnp.concatenate(
            (
                jnp.mean(projected, axis=0),
                jnp.max(projected, axis=0),
                jnp.sqrt(jnp.var(projected, axis=0) + 1e-8),
            ),
            axis=0,
        )
        return self.output_projection(pooled)


class ObservationEncoder(eqx.Module, AbstractObservationEncoder):
    """Learned hierarchical encoder dispatching over modalities.

    Parameters
    ----------
    embedding_dim:
        Size of the final joint observation embedding.
    spectrum_embedding_dim:
        Intermediate embedding size used by the spectral sub-encoder.
    photometry_embedding_dim:
        Intermediate embedding size used by the photometry sub-encoder.
    hidden_dim:
        Hidden width shared across the component encoders and aggregator.
    key:
        Optional JAX random key used to initialize all submodules.

    exact_summary_dim:
        Width of the *exact-summary channel* appended to the learned embedding
        (0 disables it). These are analytically computed, high-precision
        scalars — the log median transit depth / flux first — passed straight
        through to the flow's conditioning vector, bypassing the network and
        every per-observation normalization. See
        :func:`~petitRADTRANS.experimental.sbi.preprocessing.exact_summary_feature_matrix`.

    Notes
    -----
    Spectrum and photometry blocks are handled by dedicated sub-encoders and
    then merged with a permutation-invariant aggregator. Unsupported modalities
    currently fall back to resized raw-value vectors.

    The full conditioning vector is ``embedding_dim + exact_summary_dim`` wide
    (:attr:`context_dim`), not ``embedding_dim``. The learned half is a summary
    the network is free to compress; the exact half is not, because a
    transmission spectrum determines ``log(median depth)`` — and hence
    ``R_p + h`` — to a few parts in 10^5, while the trained encoder was measured
    to recover it only to ~0.44 (R^2 0.67), a 200x degradation against a plain
    median of its own input. Anything the conditioner has to *reconstruct* from
    a learned representation is reconstructed at the representation's accuracy;
    this channel hands over the constrained combination itself.
    """

    spectrum_encoder: eqx.Module
    photometry_encoder: PhotometryPointEncoder
    aggregator: DatasetSetAggregator
    embedding_dim: int = eqx.field(static=True)
    exact_summary_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 128,
        spectrum_embedding_dim: int = 64,
        photometry_embedding_dim: int = 64,
        hidden_dim: int = 128,
        spectrum_encoder_type: str = "convolution",
        n_wavelengths: int = 233,
        exact_summary_dim: int = 0,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(3)

        spectrum_key, photometry_key, aggregator_key = random.split(key, 3)
        self.embedding_dim = int(embedding_dim)
        self.exact_summary_dim = max(int(exact_summary_dim), 0)

        # In case we add more spectrum encoder types in the future, we dispatch on a string.
        if spectrum_encoder_type == "convolution":
            self.spectrum_encoder = SpectralConvolutionEncoder(
                spectrum_embedding_dim,
                n_wavelengths,
                # The conv-stack width follows the encoder hidden width so
                # EncoderConfig.hidden_dim actually controls shape-branch
                # capacity (it was previously pinned at the 64-channel
                # default regardless of config).
                channels=hidden_dim,
                key=spectrum_key,
            )
        elif spectrum_encoder_type == "resmlp":
            self.spectrum_encoder = SpectralResMLPEncoder(
                spectrum_embedding_dim,
                n_wavelengths,
                hidden_dim=hidden_dim,
                key=spectrum_key,
            )
        else:
            raise ValueError(f"Unknown spectrum_encoder_type: {spectrum_encoder_type}")

        self.photometry_encoder = PhotometryPointEncoder(
            photometry_embedding_dim, hidden_dim, photometry_key
        )
        self.aggregator = DatasetSetAggregator(
            self.embedding_dim, hidden_dim, aggregator_key
        )

    @property
    def context_dim(self) -> int:
        """Width of the conditioning vector handed to the flow."""
        return self.embedding_dim + self.exact_summary_dim

    def _with_exact_summary(
        self,
        embedding: jnp.ndarray,
        exact_features: list[jnp.ndarray],
    ) -> jnp.ndarray:
        """Concatenate the pooled exact-summary channel onto a learned embedding.

        Pooling across spectral blocks is a mean, matching the
        permutation-invariance of :class:`DatasetSetAggregator`; for the common
        single-spectrum task it is the identity, so the scalars reach the flow
        bit-for-bit as preprocessing computed them.
        """
        if self.exact_summary_dim <= 0:
            return embedding
        if not exact_features:
            channel = jnp.zeros((self.exact_summary_dim,), dtype=jnp.float32)
        else:
            channel = jnp.mean(
                jnp.stack(
                    [
                        _fit_vector(features, self.exact_summary_dim)
                        for features in exact_features
                    ],
                    axis=0,
                ),
                axis=0,
            )
        return jnp.concatenate((embedding, channel), axis=0)

    def _encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one observation block with modality-aware dispatch.

        Parameters
        ----------
        block:
            Observation block to encode.

        Returns
        -------
        jnp.ndarray
            Fixed-width embedding for the supplied block.
        """
        if block.modality is ObservationModality.SPECTRUM:
            encoded = self.spectrum_encoder.encode_block(block)
        elif block.modality is ObservationModality.PHOTOMETRY:
            encoded = self.photometry_encoder.encode_block(block)
        else:
            encoded = _resize_embedding(_as_vector(block.values), self.embedding_dim)

        return _resize_embedding(encoded, self.embedding_dim)

    def encode(self, blocks: list[ObservationBlock]) -> EncodedObservation:
        """Encode one structured observation made of multiple blocks.

        Parameters
        ----------
        blocks:
            Observation blocks associated with one target system.

        Returns
        -------
        EncodedObservation
            Aggregated embedding and light metadata describing the block set.
        """
        block_embeddings = [self._encode_block(block) for block in blocks]
        aggregated_embedding = self.aggregator.aggregate(block_embeddings)
        exact_features = [
            _metadata_vector(block.metadata, "exact_summary", self.exact_summary_dim)
            for block in blocks
            if block.modality is ObservationModality.SPECTRUM
        ]
        # No post-hoc rescaling here: the flow is trained on the embeddings
        # produced by ``encode_from_prestacked``/``encode_stacked_batch``, so
        # this single-observation path must return the identical vector.
        # Normalising it (e.g. to unit RMS) divides out the overall embedding
        # scale, which is exactly where a spectrum's amplitude -- and hence the
        # planet radius -- is encoded. See ``test_encode_paths_agree``.
        return EncodedObservation(
            embedding=self._with_exact_summary(
                _resize_embedding(aggregated_embedding, self.embedding_dim),
                exact_features,
            ),
            mask=None,
            metadata={
                "n_blocks": len(blocks),
                "modalities": [block.modality.value for block in blocks],
            },
        )

    def encode_stacked_batch(
        self, blocks_batch: list[list[ObservationBlock]]
    ) -> jnp.ndarray:
        """Encode a batch of identically-structured observations using vmap.

        All observations must share the same block structure (same number of
        blocks, same array shapes per block).  This is always the case for SBI
        tasks where every observation is produced by the same forward model.

        Parameters
        ----------
        blocks_batch:
            Outer list indexes samples; inner list holds the per-block
            observation data for one sample.

        Returns
        -------
        jnp.ndarray
            Float32 array of shape ``(n_samples, context_dim)``.
        """
        if not blocks_batch:
            return jnp.zeros((0, self.context_dim), dtype=jnp.float32)

        first_blocks = blocks_batch[0]
        n_blocks = len(first_blocks)
        modalities = [block.modality for block in first_blocks]

        # Stack pre-processed block arrays across the sample dimension, in the
        # field order documented by ``PreStackedObservations`` so that both
        # vmapped encode paths index the same field with the same constant.
        stacked_per_block = []
        for j in range(n_blocks):
            v_list = [_as_vector(obs[j].values) for obs in blocks_batch]
            u_list = [_as_vector(obs[j].uncertainties) for obs in blocks_batch]
            c_list = [_as_vector(obs[j].coordinates) for obs in blocks_batch]
            n_wl = v_list[0].shape[0]
            m_list = [_safe_mask(obs[j].mask, n_wl) for obs in blocks_batch]
            lmf_list = [
                jnp.float32(_log_scale_from_metadata(obs[j].metadata))
                for obs in blocks_batch
            ]
            av_list = [
                _spectral_auxiliary_vector(obs[j].metadata, n_wl)
                for obs in blocks_batch
            ]
            gsf_list = [
                _scalar_feature(
                    obs[j].metadata.get("global_spectrum_scale", 1.0)
                    if isinstance(obs[j].metadata, dict)
                    else 1.0
                )[0]
                for obs in blocks_batch
            ]
            es_list = [
                _metadata_vector(
                    obs[j].metadata, "exact_summary", max(self.exact_summary_dim, 1)
                )
                for obs in blocks_batch
            ]
            stacked_per_block.append(
                (
                    jnp.stack(v_list),
                    jnp.stack(u_list),
                    jnp.stack(c_list),
                    jnp.stack(m_list),
                    jnp.array(lmf_list, dtype=jnp.float32),
                    jnp.stack(av_list),
                    jnp.array(gsf_list, dtype=jnp.float32),
                    jnp.stack(es_list),
                )
            )

        # Build a vmappable single-observation encode closure.
        # modalities is captured as a static Python list (constant for the task).
        encoder_self = self
        n_fields_per_block = len(stacked_per_block[0]) if stacked_per_block else 6

        def _encode_one(*flat_arrays: jnp.ndarray) -> jnp.ndarray:
            block_embeddings = []
            exact_features = []
            for j, modality in enumerate(modalities):
                v = flat_arrays[n_fields_per_block * j]
                u = flat_arrays[n_fields_per_block * j + 1]
                c = flat_arrays[n_fields_per_block * j + 2]
                m = flat_arrays[n_fields_per_block * j + 3]
                lmf = flat_arrays[n_fields_per_block * j + 4]
                av = flat_arrays[n_fields_per_block * j + 5] if n_fields_per_block > 5 else jnp.zeros_like(v)
                if modality is ObservationModality.SPECTRUM:
                    emb = encoder_self.spectrum_encoder._encode_block_raw(v, u, c, m, lmf, av)
                    if n_fields_per_block > _EXACT_SUMMARY_FIELD_INDEX:
                        exact_features.append(
                            flat_arrays[
                                n_fields_per_block * j + _EXACT_SUMMARY_FIELD_INDEX
                            ]
                        )
                elif modality is ObservationModality.PHOTOMETRY:
                    emb = encoder_self.photometry_encoder._encode_block_raw(v, u, c, m)
                else:
                    emb = _resize_embedding(v, encoder_self.embedding_dim)
                block_embeddings.append(
                    _resize_embedding(emb, encoder_self.embedding_dim)
                )
            aggregated = encoder_self.aggregator.aggregate(block_embeddings)
            return encoder_self._with_exact_summary(
                _resize_embedding(aggregated, encoder_self.embedding_dim),
                exact_features,
            )

        # Flatten all stacked arrays into positional args for vmap.
        flat_stacked = tuple(
            arr for block in stacked_per_block for arr in block
        )
        return vmap(_encode_one)(*flat_stacked)  # (n_samples, context_dim)

    def encode_from_prestacked(self, obs: Any) -> jnp.ndarray:
        """Encode a batch of observations from pre-stacked arrays.

        Accepts a :class:`~petitRADTRANS.experimental.sbi.observation.PreStackedObservations`
        instance whose array fields have already been extracted from
        ``ObservationBlock`` objects outside the JAX JIT boundary.  Only the
        vmapped XLA computation runs here, enabling the training step to be
        compiled once and reused across all batches.

        Parameters
        ----------
        obs:
            Pre-stacked observation container with ``stacked_blocks`` (one
            ``(values, uncertainties, coordinates, mask, log_scale,
            absolute_values)`` tuple per block, each of shape ``(batch_size,
            n_wl)`` except the scalar ``log_scale`` arrays) and ``modalities``
            (static tuple of modality value strings).

        Returns
        -------
        jnp.ndarray
            Float32 array of shape ``(batch_size, context_dim)``.
        """
        from .observation import ObservationModality

        modality_enums = [ObservationModality(m) for m in obs.modalities]
        encoder_self = self
        n_fields = len(obs.stacked_blocks[0]) if obs.stacked_blocks else 5

        def _encode_one(*flat_arrays: jnp.ndarray) -> jnp.ndarray:
            block_embeddings = []
            exact_features = []
            for j, modality in enumerate(modality_enums):
                v = flat_arrays[n_fields * j]
                u = flat_arrays[n_fields * j + 1]
                c = flat_arrays[n_fields * j + 2]
                m = flat_arrays[n_fields * j + 3]
                lmf = flat_arrays[n_fields * j + 4] if n_fields > 4 else jnp.float32(0.0)
                av = flat_arrays[n_fields * j + 5] if n_fields > 5 else jnp.zeros_like(v)
                if modality is ObservationModality.SPECTRUM:
                    emb = encoder_self.spectrum_encoder._encode_block_raw(v, u, c, m, lmf, av)
                    if n_fields > _EXACT_SUMMARY_FIELD_INDEX:
                        exact_features.append(
                            flat_arrays[n_fields * j + _EXACT_SUMMARY_FIELD_INDEX]
                        )
                elif modality is ObservationModality.PHOTOMETRY:
                    emb = encoder_self.photometry_encoder._encode_block_raw(v, u, c, m)
                else:
                    emb = _resize_embedding(v, encoder_self.embedding_dim)
                block_embeddings.append(
                    _resize_embedding(emb, encoder_self.embedding_dim)
                )
            aggregated = encoder_self.aggregator.aggregate(block_embeddings)
            return encoder_self._with_exact_summary(
                _resize_embedding(aggregated, encoder_self.embedding_dim),
                exact_features,
            )

        flat_stacked = tuple(arr for block in obs.stacked_blocks for arr in block)
        return vmap(_encode_one)(*flat_stacked)  # (batch_size, context_dim)

    def stage_features_from_prestacked(self, obs: Any) -> dict[str, jnp.ndarray]:
        """Per-stage representations of the first spectral block, batched.

        Returns one ``(batch_size, d)`` array per named stage of the spectral
        encoder, in pipeline order, plus the block embedding it produces and the
        full conditioning ``context`` the flow receives. Feeding each of these to
        the same regression localises *where* an observable statistic stops
        being recoverable — the question "the embedding lost it" cannot answer
        on its own.

        Returns an empty mapping for encoders that do not expose stages (only
        :class:`SpectralConvolutionEncoder` does) or for observations with no
        spectral block.
        """
        from .observation import ObservationModality

        if not hasattr(self.spectrum_encoder, "stage_features_raw"):
            return {}

        modality_enums = [ObservationModality(m) for m in obs.modalities]
        try:
            block_index = modality_enums.index(ObservationModality.SPECTRUM)
        except ValueError:
            return {}

        block = obs.stacked_blocks[block_index]
        n_fields = len(block)
        spectrum_encoder = self.spectrum_encoder

        def _stages_one(*arrays: jnp.ndarray) -> dict[str, jnp.ndarray]:
            v, u, c, m = arrays[0], arrays[1], arrays[2], arrays[3]
            lmf = arrays[4] if n_fields > 4 else jnp.float32(0.0)
            stages = dict(spectrum_encoder.stage_features_raw(v, u, c, m, lmf))
            stages["block_embedding"] = _resize_embedding(
                spectrum_encoder._encode_block_raw(v, u, c, m, lmf), self.embedding_dim
            )
            return stages

        stages = vmap(_stages_one)(*block)
        stages["context"] = self.encode_from_prestacked(obs)
        return stages


__all__ = [
    "DatasetSetAggregator",
    "ObservationEncoder",
    "PhotometryPointEncoder",
    "SpectralConvolutionEncoder",
]
