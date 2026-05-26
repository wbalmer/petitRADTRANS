"""Learned observation encoders for SBI conditioning."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jax import Array, vmap, random
import jax.numpy as jnp

from .observation import (
    EncodedObservation,
    ObservationBlock,
    ObservationEncoder,
    ObservationModality,
)


_ABSOLUTE_PROFILE_BINS = 16
_AMPLITUDE_FEATURE_DIM = 6 + _ABSOLUTE_PROFILE_BINS + 1


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


def _summaries(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    if values.shape[0] == 0:
        return jnp.zeros((6,), dtype=jnp.float32)

    valid_mask = ~mask
    valid_count = jnp.sum(valid_mask).astype(jnp.float32)
    safe_count = jnp.maximum(valid_count, 1.0)
    valid = jnp.where(valid_mask, values, jnp.zeros_like(values))
    mean_value = jnp.sum(valid) / safe_count
    centered = jnp.where(valid_mask, values - mean_value, jnp.zeros_like(values))
    std_value = jnp.sqrt(jnp.sum(centered**2) / safe_count + 1e-8)
    min_value = jnp.min(
        jnp.where(valid_mask, values, jnp.asarray(jnp.inf, dtype=values.dtype))
    )
    max_value = jnp.max(
        jnp.where(valid_mask, values, jnp.asarray(-jnp.inf, dtype=values.dtype))
    )
    min_value = jnp.where(valid_count > 0.0, min_value, 0.0)
    max_value = jnp.where(valid_count > 0.0, max_value, 0.0)
    valid_fraction = valid_count / max(float(values.shape[0]), 1.0)

    return jnp.asarray(
        [
            mean_value,
            std_value,
            min_value,
            max_value,
            valid_fraction,
            1.0 - valid_fraction,
        ],
        dtype=jnp.float32,
    )


def _normalize_embedding(features: jnp.ndarray) -> jnp.ndarray:
    features = jnp.asarray(features, dtype=jnp.float32)
    return features / jnp.sqrt(jnp.mean(jnp.square(features)) + 1e-6)


def _merge_embedding_branches(*branches: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [jnp.ravel(jnp.asarray(branch, dtype=jnp.float32)) for branch in branches],
        axis=0,
    )


def _spectral_local_structure(
    values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    if values.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.float32)

    valid = ~mask
    next_valid = jnp.concatenate((valid[1:], jnp.zeros((1,), dtype=bool)), axis=0)
    pair_valid = valid & next_valid
    forward_difference = jnp.concatenate(
        (values[1:] - values[:-1], jnp.zeros((1,), dtype=values.dtype)),
        axis=0,
    )
    return jnp.where(pair_valid, forward_difference, 0.0).astype(jnp.float32)


def _masked_bin_means(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    n_bins: int = _ABSOLUTE_PROFILE_BINS,
) -> jnp.ndarray:
    if values.shape[0] == 0:
        return jnp.zeros((n_bins,), dtype=jnp.float32)

    valid_mask = ~mask
    if n_bins <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)

    n_values = max(int(values.shape[0]), 1)
    bin_indices = jnp.minimum(
        (jnp.arange(values.shape[0], dtype=jnp.int32) * n_bins) // n_values,
        n_bins - 1,
    )
    safe_values = jnp.where(valid_mask, values, 0.0)
    valid_counts = valid_mask.astype(jnp.float32)

    bin_sums = jnp.zeros((n_bins,), dtype=jnp.float32).at[bin_indices].add(safe_values)
    bin_counts = jnp.zeros((n_bins,), dtype=jnp.float32).at[bin_indices].add(valid_counts)
    return bin_sums / jnp.maximum(bin_counts, 1.0)


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


def _scalar_feature(value: Any) -> jnp.ndarray:
    if value is None:
        return jnp.zeros((1,), dtype=jnp.float32)

    scalar = jnp.ravel(jnp.asarray(value, dtype=jnp.float32))
    if scalar.shape[0] == 0:
        return jnp.zeros((1,), dtype=jnp.float32)
    return scalar[:1]


def _broadcast_scalar_feature(value: Any, size: int) -> jnp.ndarray:
    if size <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    scalar = _scalar_feature(value)
    return jnp.broadcast_to(scalar, (size,)).astype(jnp.float32)


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


def _amplitude_features(
    absolute_values: jnp.ndarray,
    mask: jnp.ndarray,
    log_scale: Any,
) -> jnp.ndarray:
    return jnp.concatenate(
        (
            _summaries(absolute_values, mask),
            _masked_bin_means(absolute_values, mask),
            _scalar_feature(log_scale),
        ),
        axis=0,
    )


class SpectralPatchEncoder(eqx.Module):
    """Learned spectral encoder based on patch summaries and MLP pooling.

    Parameters
    ----------
    embedding_dim:
        Size of the spectral embedding returned for one observation block.
    patch_size:
        Number of spectral points summarized together before MLP encoding.
    hidden_dim:
        Hidden width of the internal summary MLP.
    key:
        Optional JAX random key used for weight initialization.

    Notes
    -----
    The encoder summarizes values, uncertainties, and coordinates patch by
    patch instead of operating on a fully convolutional representation. This
    keeps the implementation lightweight and shape-agnostic.
    """

    patch_mlp: eqx.nn.MLP
    amplitude_projection: eqx.nn.MLP
    branch_projection: eqx.nn.Linear
    output_projection: eqx.nn.Linear
    scale_projection: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 64,
        patch_size: int = 32,
        hidden_dim: int = 96,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(0)

        mlp_key, amplitude_key, branch_key, projection_key, scale_key = random.split(key, 5)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.patch_mlp = eqx.nn.MLP(18, self.embedding_dim, hidden_dim, 2, key=mlp_key, dtype=jnp.float32)
        self.amplitude_projection = eqx.nn.MLP(_AMPLITUDE_FEATURE_DIM, self.embedding_dim, hidden_dim, 1, key=amplitude_key, dtype=jnp.float32)
        self.branch_projection = eqx.nn.Linear(
            self.embedding_dim * 3,
            self.embedding_dim,
            key=branch_key,
            dtype=jnp.float32,
        )
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim, self.embedding_dim, key=projection_key, dtype=jnp.float32
        )
        self.scale_projection = eqx.nn.Linear(1, self.embedding_dim, key=scale_key, dtype=jnp.float32)

    def encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one spectral observation block.

        Parameters
        ----------
        block:
            Spectrum-like observation block containing values and optional
            uncertainties, coordinates, and masks.

        Returns
        -------
        jnp.ndarray
            Dense fixed-width embedding for the supplied spectral block.
        """
        values = _as_vector(block.values)
        uncertainties = _as_vector(block.uncertainties)
        coordinates = _as_vector(block.coordinates)
        mask = _safe_mask(block.mask, values.shape[0])
        if values.shape[0] == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        n = values.shape[0]
        ps = self.patch_size
        n_patches = -(-n // ps)
        padded_length = n_patches * ps
        pad_size = padded_length - n

        padding_indicator = jnp.arange(padded_length) >= n

        values_padded = jnp.concatenate(
            [values, jnp.zeros(pad_size, dtype=jnp.float32)]
        )
        mask_padded = jnp.concatenate([mask, jnp.ones(pad_size, dtype=bool)])
        values_2d = values_padded.reshape(n_patches, ps)
        mask_2d = mask_padded.reshape(n_patches, ps)

        if uncertainties.shape[0] > 0:
            unc_pad = max(0, padded_length - uncertainties.shape[0])
            unc_padded = jnp.concatenate(
                [uncertainties, jnp.zeros(unc_pad, dtype=jnp.float32)]
            )[:padded_length]
            unc_mask_2d = padding_indicator.reshape(n_patches, ps)
        else:
            unc_padded = jnp.zeros(padded_length, dtype=jnp.float32)
            unc_mask_2d = jnp.ones((n_patches, ps), dtype=bool)
        unc_2d = unc_padded.reshape(n_patches, ps)

        if coordinates.shape[0] > 0:
            coord_pad = max(0, padded_length - coordinates.shape[0])
            coord_padded = jnp.concatenate(
                [coordinates, jnp.zeros(coord_pad, dtype=jnp.float32)]
            )[:padded_length]
            coord_mask_2d = padding_indicator.reshape(n_patches, ps)
        else:
            coord_padded = jnp.zeros(padded_length, dtype=jnp.float32)
            coord_mask_2d = jnp.ones((n_patches, ps), dtype=bool)
        coord_2d = coord_padded.reshape(n_patches, ps)

        def _patch_summary(v, vm, u, um, c, cm):
            return jnp.concatenate(
                (_summaries(v, vm), _summaries(u, um), _summaries(c, cm))
            )

        summaries = vmap(_patch_summary)(
            values_2d,
            mask_2d,
            unc_2d,
            unc_mask_2d,
            coord_2d,
            coord_mask_2d,
        )
        patch_embeddings = vmap(self.patch_mlp)(summaries)
        pooled = jnp.mean(patch_embeddings, axis=0)
        amplitude_embedding = jax.nn.gelu(
            self.amplitude_projection(
                _amplitude_features(
                    values,
                    mask,
                    _log_scale_from_metadata(block.metadata),
                )
            )
        )
        scale_embedding = jax.nn.gelu(
            self.scale_projection(
                _scalar_feature(_log_scale_from_metadata(block.metadata))
            )
        )
        merged = _merge_embedding_branches(
            pooled,
            amplitude_embedding,
            scale_embedding,
        )
        merged = jax.nn.gelu(self.branch_projection(merged))
        return self.output_projection(merged)

    def _encode_block_raw(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None = None,
        absolute_values: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Encode pre-processed block arrays (vmappable — no ObservationBlock input).

        Parameters
        ----------
        values, uncertainties, coordinates:
            Float32 1-D arrays already produced by ``_as_vector``.
            ``uncertainties`` and ``coordinates`` may have length 0 to indicate
            absence.
        mask:
            Boolean 1-D array already produced by ``_safe_mask``.

        Returns
        -------
        jnp.ndarray
            Dense spectral embedding of shape ``(embedding_dim,)``.
        """
        if values.shape[0] == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        n = values.shape[0]
        ps = self.patch_size
        n_patches = -(-n // ps)
        padded_length = n_patches * ps
        pad_size = padded_length - n

        padding_indicator = jnp.arange(padded_length) >= n

        values_padded = jnp.concatenate(
            [values, jnp.zeros(pad_size, dtype=jnp.float32)]
        )
        mask_padded = jnp.concatenate([mask, jnp.ones(pad_size, dtype=bool)])
        values_2d = values_padded.reshape(n_patches, ps)
        mask_2d = mask_padded.reshape(n_patches, ps)

        if uncertainties.shape[0] > 0:
            unc_pad = max(0, padded_length - uncertainties.shape[0])
            unc_padded = jnp.concatenate(
                [uncertainties, jnp.zeros(unc_pad, dtype=jnp.float32)]
            )[:padded_length]
            unc_mask_2d = padding_indicator.reshape(n_patches, ps)
        else:
            unc_padded = jnp.zeros(padded_length, dtype=jnp.float32)
            unc_mask_2d = jnp.ones((n_patches, ps), dtype=bool)
        unc_2d = unc_padded.reshape(n_patches, ps)

        if coordinates.shape[0] > 0:
            coord_pad = max(0, padded_length - coordinates.shape[0])
            coord_padded = jnp.concatenate(
                [coordinates, jnp.zeros(coord_pad, dtype=jnp.float32)]
            )[:padded_length]
            coord_mask_2d = padding_indicator.reshape(n_patches, ps)
        else:
            coord_padded = jnp.zeros(padded_length, dtype=jnp.float32)
            coord_mask_2d = jnp.ones((n_patches, ps), dtype=bool)
        coord_2d = coord_padded.reshape(n_patches, ps)

        if absolute_values is None or absolute_values.shape[0] == 0:
            absolute_values = jnp.zeros_like(values)
        elif absolute_values.shape[0] > values.shape[0]:
            absolute_values = absolute_values[: values.shape[0]]
        elif absolute_values.shape[0] < values.shape[0]:
            absolute_values = jnp.concatenate(
                (
                    absolute_values,
                    jnp.zeros((values.shape[0] - absolute_values.shape[0],), dtype=jnp.float32),
                ),
                axis=0,
            )

        def _patch_summary(v, vm, u, um, c, cm):
            return jnp.concatenate(
                (_summaries(v, vm), _summaries(u, um), _summaries(c, cm))
            )

        summaries = vmap(_patch_summary)(
            values_2d,
            mask_2d,
            unc_2d,
            unc_mask_2d,
            coord_2d,
            coord_mask_2d,
        )
        patch_embeddings = vmap(self.patch_mlp)(summaries)
        pooled = jnp.mean(patch_embeddings, axis=0)
        amplitude_embedding = jax.nn.gelu(
            self.amplitude_projection(
                _amplitude_features(values, mask, log_median_flux)
            )
        )
        scale_embedding = jax.nn.gelu(
            self.scale_projection(
                _scalar_feature(log_median_flux)
            )
        )
        merged = _merge_embedding_branches(
            pooled,
            amplitude_embedding,
            scale_embedding,
        )
        merged = jax.nn.gelu(self.branch_projection(merged))
        return self.output_projection(merged)


class SpectralConv1DEncoder(eqx.Module):
    """Learned spectral encoder using 1D convolutions to retain positional information.

    Parameters
    ----------
    embedding_dim:
        Size of the spectral embedding returned for one observation block.
    n_wavelengths:
        Fixed number of spectral points per observation block. Must match the
        observation schema of the SBI task.
    hidden_channels:
        Tuple of intermediate channel widths for conv layers.
    key:
        Optional JAX random key used for weight initialization.
    """

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    conv3: eqx.nn.Conv1d
    amplitude_projection: eqx.nn.MLP
    pool_projection: eqx.nn.Linear
    branch_projection: eqx.nn.Linear
    output_projection: eqx.nn.Linear
    scale_projection: eqx.nn.Linear
    embedding_dim: int = eqx.field(static=True)
    n_wavelengths: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 64,
        n_wavelengths: int = 233,
        hidden_channels: tuple[int, int] = (32, 64),
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(0)

        k1, k2, k3, k_amp, k_pool, k_branch, k_proj, k_scale = random.split(key, 8)
        self.embedding_dim = int(embedding_dim)
        self.n_wavelengths = int(n_wavelengths)
        ch1, ch2 = hidden_channels

        # 6 input channels: globally normalized values, uncertainties,
        # coordinates, a sample-scale contrast profile, a local-structure
        # gradient channel derived from that contrast profile, and a broadcast
        # log-scale channel.
        # Explicitly use float32 so weights are not promoted to float64
        # when jax_enable_x64 is active.
        self.conv1 = eqx.nn.Conv1d(6, ch1, kernel_size=7, padding=3, key=k1, dtype=jnp.float32)
        self.conv2 = eqx.nn.Conv1d(ch1, ch2, kernel_size=5, padding=2, key=k2, dtype=jnp.float32)
        self.conv3 = eqx.nn.Conv1d(ch2, self.embedding_dim, kernel_size=3, padding=1, key=k3, dtype=jnp.float32)
        self.amplitude_projection = eqx.nn.MLP(_AMPLITUDE_FEATURE_DIM, self.embedding_dim, max(self.embedding_dim, ch2), 1, key=k_amp, dtype=jnp.float32)
        self.pool_projection = eqx.nn.Linear(
            self.embedding_dim * 2,
            self.embedding_dim,
            key=k_pool,
            dtype=jnp.float32,
        )
        self.branch_projection = eqx.nn.Linear(
            self.embedding_dim * 3,
            self.embedding_dim,
            key=k_branch,
            dtype=jnp.float32,
        )
        self.output_projection = eqx.nn.Linear(self.embedding_dim, self.embedding_dim, key=k_proj, dtype=jnp.float32)
        self.scale_projection = eqx.nn.Linear(1, self.embedding_dim, key=k_scale, dtype=jnp.float32)

    def _forward_conv(
        self,
        values: jnp.ndarray,
        uncertainties: jnp.ndarray,
        coordinates: jnp.ndarray,
        mask: jnp.ndarray,
        log_median_flux: jnp.ndarray | None = None,
        absolute_values: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Run 1D conv stack on pre-processed arrays.

        Parameters
        ----------
        values, uncertainties, coordinates:
            Float32 1-D arrays of length ``n_wavelengths``.
        mask:
            Boolean 1-D mask (True = invalid).

        Returns
        -------
        jnp.ndarray
            Embedding of shape ``(embedding_dim,)``.
        """
        # Zero out masked wavelength bins
        valid = (~mask).astype(jnp.float32)
        values = values * valid
        uncertainties = uncertainties * valid
        coordinates = coordinates * valid
        if absolute_values is None or absolute_values.shape[0] == 0:
            absolute_values = jnp.zeros_like(values)
        absolute_values = absolute_values * valid
        local_structure = _spectral_local_structure(absolute_values, mask) * valid
        scale_channel = _broadcast_scalar_feature(log_median_flux, values.shape[0]) * valid

        # The main spectral branch sees both the global absolute spectrum and
        # a sample-scale contrast profile plus its local gradient so it can
        # learn amplitude and wavelength-local structure jointly.
        x = jnp.stack(
            [
                values,
                uncertainties,
                coordinates,
                absolute_values,
                local_structure,
                scale_channel,
            ],
            axis=0,
        )

        x = jax.nn.gelu(self.conv1(x))   # (ch1, n_wavelengths)
        x = jax.nn.gelu(self.conv2(x))   # (ch2, n_wavelengths)
        x = jax.nn.gelu(self.conv3(x))   # (embedding_dim, n_wavelengths)

        # Combine mean and max pooling so narrow spectral features survive the
        # global reduction stage instead of being averaged away.
        valid_expanded = valid[None, :]  # (1, n_wavelengths)
        x = x * valid_expanded
        pooled_mean = jnp.sum(x, axis=-1) / jnp.maximum(jnp.sum(valid), 1.0)
        masked_for_max = jnp.where(
            valid_expanded > 0.0,
            x,
            jnp.asarray(-jnp.inf, dtype=x.dtype),
        )
        pooled_max = jnp.max(masked_for_max, axis=-1)
        pooled_max = jnp.where(jnp.any(valid), pooled_max, jnp.zeros_like(pooled_max))
        pooled = jax.nn.gelu(
            self.pool_projection(
                _merge_embedding_branches(pooled_mean, pooled_max)
            )
        )
        amplitude_embedding = jax.nn.gelu(
            self.amplitude_projection(
                _amplitude_features(values, mask, log_median_flux)
            )
        )
        scale_embedding = jax.nn.gelu(
            self.scale_projection(_scalar_feature(log_median_flux))
        )
        merged = _merge_embedding_branches(
            pooled,
            amplitude_embedding,
            scale_embedding,
        )
        merged = jax.nn.gelu(self.branch_projection(merged))
        return self.output_projection(merged)

    def _pad_to_fixed(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Pad or truncate a 1-D array to ``n_wavelengths``."""
        n = arr.shape[0]
        if n == self.n_wavelengths:
            return arr
        if n > self.n_wavelengths:
            return arr[: self.n_wavelengths]
        return jnp.concatenate(
            [arr, jnp.zeros(self.n_wavelengths - n, dtype=arr.dtype)]
        )

    def encode_block(self, block: ObservationBlock) -> jnp.ndarray:
        """Encode one spectral observation block.

        Parameters
        ----------
        block:
            Spectrum-like observation block.

        Returns
        -------
        jnp.ndarray
            Dense embedding of shape ``(embedding_dim,)``.
        """
        values = _as_vector(block.values)
        uncertainties = _as_vector(block.uncertainties)
        coordinates = _as_vector(block.coordinates)
        absolute_values = _spectral_auxiliary_vector(block.metadata, values.shape[0])
        n = values.shape[0]
        mask = _safe_mask(block.mask, n)

        if n == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        values = self._pad_to_fixed(values)
        if uncertainties.shape[0] == 0:
            uncertainties = jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
        else:
            uncertainties = self._pad_to_fixed(uncertainties)
        if coordinates.shape[0] == 0:
            coordinates = jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
        else:
            coordinates = self._pad_to_fixed(coordinates)
        absolute_values = self._pad_to_fixed(absolute_values)

        pad_mask = jnp.arange(self.n_wavelengths) >= n
        mask = self._pad_to_fixed(mask.astype(jnp.float32)).astype(bool) | pad_mask

        return self._forward_conv(
            values,
            uncertainties,
            coordinates,
            mask,
            _log_scale_from_metadata(block.metadata),
            absolute_values,
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
        """Encode pre-processed block arrays (vmappable — no ObservationBlock input).

        Parameters
        ----------
        values, uncertainties, coordinates:
            Float32 1-D arrays already produced by ``_as_vector``.
        mask:
            Boolean 1-D array already produced by ``_safe_mask``.
        log_median_flux:
            Scalar absolute-flux feature derived from the raw spectrum median.

        Returns
        -------
        jnp.ndarray
            Dense spectral embedding of shape ``(embedding_dim,)``.
        """
        if values.shape[0] == 0:
            return jnp.zeros((self.embedding_dim,), dtype=jnp.float32)

        values = self._pad_to_fixed(values)
        if uncertainties.shape[0] == 0:
            uncertainties = jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
        else:
            uncertainties = self._pad_to_fixed(uncertainties)
        if coordinates.shape[0] == 0:
            coordinates = jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
        else:
            coordinates = self._pad_to_fixed(coordinates)
        if absolute_values is None or absolute_values.shape[0] == 0:
            absolute_values = jnp.zeros(self.n_wavelengths, dtype=jnp.float32)
        else:
            absolute_values = self._pad_to_fixed(absolute_values)

        mask = self._pad_to_fixed(mask.astype(jnp.float32)).astype(bool)
        return self._forward_conv(
            values,
            uncertainties,
            coordinates,
            mask,
            log_median_flux,
            absolute_values,
        )


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
        if len(block_embeddings) == 1:
            return jnp.asarray(block_embeddings[0], dtype=jnp.float32)

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


class HierarchicalObservationEncoder(eqx.Module, ObservationEncoder):
    """Learned hierarchical encoder dispatching over modalities.

    Parameters
    ----------
    embedding_dim:
        Size of the final joint observation embedding.
    spectrum_embedding_dim:
        Intermediate embedding size used by the spectral sub-encoder.
    photometry_embedding_dim:
        Intermediate embedding size used by the photometry sub-encoder.
    patch_size:
        Patch size used by the spectral encoder.
    hidden_dim:
        Hidden width shared across the component encoders and aggregator.
    key:
        Optional JAX random key used to initialize all submodules.

    Notes
    -----
    Spectrum and photometry blocks are handled by dedicated sub-encoders and
    then merged with a permutation-invariant aggregator. Unsupported modalities
    currently fall back to resized raw-value vectors.
    """

    spectrum_encoder: SpectralPatchEncoder | SpectralConv1DEncoder
    photometry_encoder: PhotometryPointEncoder
    aggregator: DatasetSetAggregator
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int = 128,
        spectrum_embedding_dim: int = 64,
        photometry_embedding_dim: int = 64,
        patch_size: int = 32,
        hidden_dim: int = 128,
        spectrum_encoder_type: str = "conv1d",
        n_wavelengths: int = 233,
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(3)

        spectrum_key, photometry_key, aggregator_key = random.split(key, 3)
        self.embedding_dim = int(embedding_dim)
        if spectrum_encoder_type == "conv1d":
            self.spectrum_encoder = SpectralConv1DEncoder(
                spectrum_embedding_dim, n_wavelengths, key=spectrum_key
            )
        else:
            self.spectrum_encoder = SpectralPatchEncoder(
                spectrum_embedding_dim, patch_size, hidden_dim, spectrum_key
            )
        self.photometry_encoder = PhotometryPointEncoder(
            photometry_embedding_dim, hidden_dim, photometry_key
        )
        self.aggregator = DatasetSetAggregator(
            self.embedding_dim, hidden_dim, aggregator_key
        )

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
        return EncodedObservation(
            embedding=_resize_embedding(aggregated_embedding, self.embedding_dim),
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
            Float32 array of shape ``(n_samples, embedding_dim)``.
        """
        if not blocks_batch:
            return jnp.zeros((0, self.embedding_dim), dtype=jnp.float32)

        n_samples = len(blocks_batch)
        first_blocks = blocks_batch[0]
        n_blocks = len(first_blocks)
        modalities = [block.modality for block in first_blocks]

        # Stack pre-processed block arrays across the sample dimension.
        # stacked_per_block[j] = (values_b, unc_b, coord_b, mask_b, lmf_b, av_b)
        # where v/u/c/m/av have shape (n_samples, n_wl_j) and lmf has shape (n_samples,).
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
            stacked_per_block.append(
                (
                    jnp.stack(v_list),
                    jnp.stack(u_list),
                    jnp.stack(c_list),
                    jnp.stack(m_list),
                    jnp.array(lmf_list, dtype=jnp.float32),
                    jnp.stack(av_list),
                )
            )

        # Build a vmappable single-observation encode closure.
        # modalities is captured as a static Python list (constant for the task).
        encoder_self = self
        n_fields_per_block = len(stacked_per_block[0]) if stacked_per_block else 6

        def _encode_one(*flat_arrays: jnp.ndarray) -> jnp.ndarray:
            block_embeddings = []
            for j, modality in enumerate(modalities):
                v = flat_arrays[n_fields_per_block * j]
                u = flat_arrays[n_fields_per_block * j + 1]
                c = flat_arrays[n_fields_per_block * j + 2]
                m = flat_arrays[n_fields_per_block * j + 3]
                lmf = flat_arrays[n_fields_per_block * j + 4]
                av = flat_arrays[n_fields_per_block * j + 5] if n_fields_per_block > 5 else jnp.zeros_like(v)
                if modality is ObservationModality.SPECTRUM:
                    emb = encoder_self.spectrum_encoder._encode_block_raw(v, u, c, m, lmf, av)
                elif modality is ObservationModality.PHOTOMETRY:
                    emb = encoder_self.photometry_encoder._encode_block_raw(v, u, c, m)
                else:
                    emb = _resize_embedding(v, encoder_self.embedding_dim)
                block_embeddings.append(
                    _resize_embedding(emb, encoder_self.embedding_dim)
                )
            aggregated = encoder_self.aggregator.aggregate(block_embeddings)
            return _resize_embedding(aggregated, encoder_self.embedding_dim)

        # Flatten all stacked arrays into positional args for vmap.
        flat_stacked = tuple(
            arr for block in stacked_per_block for arr in block
        )
        return vmap(_encode_one)(*flat_stacked)  # (n_samples, embedding_dim)

    def encode_from_prestacked(self, obs: Any) -> jnp.ndarray:
        """Encode a batch of observations from pre-stacked arrays.

        Accepts a :class:`~petitRADTRANS.sbi.observation.PreStackedObservations`
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
            Float32 array of shape ``(batch_size, embedding_dim)``.
        """
        from .observation import ObservationModality

        modality_enums = [ObservationModality(m) for m in obs.modalities]
        encoder_self = self
        n_fields = len(obs.stacked_blocks[0]) if obs.stacked_blocks else 5

        def _encode_one(*flat_arrays: jnp.ndarray) -> jnp.ndarray:
            block_embeddings = []
            for j, modality in enumerate(modality_enums):
                v = flat_arrays[n_fields * j]
                u = flat_arrays[n_fields * j + 1]
                c = flat_arrays[n_fields * j + 2]
                m = flat_arrays[n_fields * j + 3]
                lmf = flat_arrays[n_fields * j + 4] if n_fields > 4 else jnp.float32(0.0)
                av = flat_arrays[n_fields * j + 5] if n_fields > 5 else jnp.zeros_like(v)
                if modality is ObservationModality.SPECTRUM:
                    emb = encoder_self.spectrum_encoder._encode_block_raw(v, u, c, m, lmf, av)
                elif modality is ObservationModality.PHOTOMETRY:
                    emb = encoder_self.photometry_encoder._encode_block_raw(v, u, c, m)
                else:
                    emb = _resize_embedding(v, encoder_self.embedding_dim)
                block_embeddings.append(
                    _resize_embedding(emb, encoder_self.embedding_dim)
                )
            aggregated = encoder_self.aggregator.aggregate(block_embeddings)
            return _resize_embedding(aggregated, encoder_self.embedding_dim)

        flat_stacked = tuple(arr for block in obs.stacked_blocks for arr in block)
        return vmap(_encode_one)(*flat_stacked)  # (batch_size, embedding_dim)


__all__ = [
    "DatasetSetAggregator",
    "HierarchicalObservationEncoder",
    "PhotometryPointEncoder",
    "SpectralConv1DEncoder",
    "SpectralPatchEncoder",
]
