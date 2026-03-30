"""Learned observation encoders for SBI conditioning."""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import Array, vmap, random
import jax.numpy as jnp

from .observation import (
    EncodedObservation,
    ObservationBlock,
    ObservationEncoder,
    ObservationModality,
)


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

    valid = jnp.where(mask, jnp.zeros_like(values), values)
    valid_count = jnp.maximum(1.0, jnp.sum(~mask).astype(jnp.float32))
    mean_value = jnp.sum(valid) / valid_count
    centered = jnp.where(mask, jnp.zeros_like(values), values - mean_value)
    std_value = jnp.sqrt(jnp.sum(centered**2) / valid_count)

    return jnp.asarray(
        [
            mean_value,
            std_value,
            jnp.min(valid),
            jnp.max(valid),
            valid_count,
            1.0 - (valid_count / max(float(values.shape[0]), 1.0)),
        ],
        dtype=jnp.float32,
    )


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
    output_projection: eqx.nn.Linear
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

        mlp_key, projection_key = random.split(key)
        self.embedding_dim = int(embedding_dim)
        self.patch_size = int(patch_size)
        self.patch_mlp = eqx.nn.MLP(18, self.embedding_dim, hidden_dim, 2, key=mlp_key)
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim, self.embedding_dim, key=projection_key
        )

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
            values_2d, mask_2d, unc_2d, unc_mask_2d, coord_2d, coord_mask_2d
        )
        patch_embeddings = vmap(self.patch_mlp)(summaries)
        pooled = jnp.mean(patch_embeddings, axis=0)
        return self.output_projection(pooled)


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
        self.point_mlp = eqx.nn.MLP(4, self.embedding_dim, hidden_dim, 2, key=mlp_key)
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim, self.embedding_dim, key=projection_key
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
            self.embedding_dim, self.embedding_dim, hidden_dim, 2, key=mlp_key
        )
        self.output_projection = eqx.nn.Linear(
            self.embedding_dim * 3, self.embedding_dim, key=projection_key
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

        stacked = jnp.stack(block_embeddings, axis=0)
        projected = vmap(self.block_projection)(stacked)
        pooled = jnp.concatenate(
            (
                jnp.mean(projected, axis=0),
                jnp.max(projected, axis=0),
                jnp.std(projected, axis=0),
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

    spectrum_encoder: SpectralPatchEncoder
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
        key: Array | None = None,
    ):
        if key is None:
            key = random.PRNGKey(3)

        spectrum_key, photometry_key, aggregator_key = random.split(key, 3)
        self.embedding_dim = int(embedding_dim)
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


__all__ = [
    "DatasetSetAggregator",
    "HierarchicalObservationEncoder",
    "PhotometryPointEncoder",
    "SpectralPatchEncoder",
]
