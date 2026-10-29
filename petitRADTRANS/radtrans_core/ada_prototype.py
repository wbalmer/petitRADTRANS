"""Experimental scalar adding-doubling prototype inspired by CRTMv3 ADA.

This module mirrors the control flow of CRTM's ADA implementation for the
thermal-emission case used by petitRADTRANS:

- construct a thin-layer transmission/reflection/source operator,
- repeatedly double that operator for optically thicker layers,
- add heterogeneous layers from the surface to the top of atmosphere.

It intentionally omits the full CRTM phase-matrix, solar, and adjoint/TL
machinery. The goal is a JAX-native prototype that is comparable to the
current Feautrier solver on the same scalar benchmark inputs.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax
from jax.typing import ArrayLike

from petitRADTRANS.jax_dtype import JAX_DTYPE
import petitRADTRANS.radtrans_core.physics_core as physics_core


TINIEST = jnp.finfo(JAX_DTYPE).tiny
ADA_SUBLAYER_LIMIT = jnp.array(0.25, dtype=JAX_DTYPE)
ADA_MAX_DOUBLINGS = 12


def _batched_identity(batch_size: int, n_angles: int) -> ArrayLike:
    return jnp.broadcast_to(
        jnp.eye(n_angles, dtype=JAX_DTYPE),
        (batch_size, n_angles, n_angles),
    )


def _double_layer_operator(
    reflection_baa: ArrayLike,
    transmission_baa: ArrayLike,
    source_ba: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Combine two identical half-layers into one full layer."""
    batch_size, n_angles, _ = reflection_baa.shape
    identity_baa = _batched_identity(batch_size, n_angles)

    gamma_inverse_baa = jnp.linalg.inv(
        identity_baa - reflection_baa @ reflection_baa
    )
    transmission_gamma_baa = transmission_baa @ gamma_inverse_baa

    reflected_source_ba = (reflection_baa @ source_ba[..., None])[..., 0]
    source_doubled_ba = source_ba + (
        transmission_gamma_baa @ (reflected_source_ba + source_ba)[..., None]
    )[..., 0]
    reflection_doubled_baa = reflection_baa + transmission_gamma_baa @ (
        reflection_baa @ transmission_baa
    )
    transmission_doubled_baa = transmission_gamma_baa @ transmission_baa

    return reflection_doubled_baa, transmission_doubled_baa, source_doubled_ba


def _build_layer_operator(
    layer_optical_depth_b: ArrayLike,
    single_scatter_albedo_b: ArrayLike,
    thermal_source_b: ArrayLike,
    emission_cos_angles: ArrayLike,
    emission_cos_angles_weights: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Build a layer operator using thin-layer initialization plus doubling."""
    n_angles = emission_cos_angles.shape[0]
    mu_min = jnp.min(emission_cos_angles)
    safe_tau_ratio = jnp.maximum(
        layer_optical_depth_b / jnp.maximum(mu_min, TINIEST),
        1.0,
    )
    number_doublings_b = jnp.clip(
        jnp.ceil(jnp.log2(safe_tau_ratio / ADA_SUBLAYER_LIMIT)),
        0,
        ADA_MAX_DOUBLINGS,
    ).astype(jnp.int32)
    sublayer_optical_depth_b = layer_optical_depth_b / (2.0**number_doublings_b)

    attenuation_ba = jnp.exp(
        -sublayer_optical_depth_b[:, None]
        / jnp.maximum(emission_cos_angles[None, :], TINIEST)
    )
    interaction_weight_ba = (1.0 - attenuation_ba) * single_scatter_albedo_b[:, None]
    reflection_baa = (
        0.5
        * interaction_weight_ba[:, :, None]
        * emission_cos_angles_weights[None, None, :]
    )
    transmission_baa = reflection_baa + jnp.eye(n_angles, dtype=JAX_DTYPE)[None, :, :] * attenuation_ba[:, :, None]
    source_ba = (
        (1.0 - single_scatter_albedo_b)[:, None]
        * (1.0 - attenuation_ba)
        * thermal_source_b[:, None]
    )

    def _doubling_body(
        doubling_index: int,
        state: tuple[ArrayLike, ArrayLike, ArrayLike],
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        current_reflection_baa, current_transmission_baa, current_source_ba = state
        next_reflection_baa, next_transmission_baa, next_source_ba = (
            _double_layer_operator(
                current_reflection_baa,
                current_transmission_baa,
                current_source_ba,
            )
        )

        active_mask_b = doubling_index < number_doublings_b
        active_mask_baa = active_mask_b[:, None, None]
        active_mask_ba = active_mask_b[:, None]

        current_reflection_baa = jnp.where(
            active_mask_baa,
            next_reflection_baa,
            current_reflection_baa,
        )
        current_transmission_baa = jnp.where(
            active_mask_baa,
            next_transmission_baa,
            current_transmission_baa,
        )
        current_source_ba = jnp.where(
            active_mask_ba,
            next_source_ba,
            current_source_ba,
        )

        return current_reflection_baa, current_transmission_baa, current_source_ba

    return lax.fori_loop(
        0,
        ADA_MAX_DOUBLINGS,
        _doubling_body,
        (reflection_baa, transmission_baa, source_ba),
    )


@partial(jit, static_argnames=("irradiation_geometry_str",))
def compute_ada_radiative_transfer(
    frequencies_bin_edges: ArrayLike,
    temperatures: ArrayLike,
    weights_gauss: ArrayLike,
    emission_cos_angles: ArrayLike,
    emission_cos_angles_weights: ArrayLike,
    optical_depths_in: ArrayLike,
    photon_destruction_probabilities_in: ArrayLike,
    stellar_intensity: ArrayLike,
    star_irradiation_cos_angle: float,
    reflectances: ArrayLike,
    emissivities: ArrayLike,
    irradiation_geometry_str: str = "dayside_ave",
) -> tuple[ArrayLike, ArrayLike]:
    """Compute a thermal scalar ADA prototype flux.

    Parameters match :func:`compute_feautrier_radiative_transfer` where useful,
    but this prototype currently ignores the stellar source terms and focuses on
    the thermal emission benchmark slice.
    """
    del stellar_intensity
    del star_irradiation_cos_angle
    del irradiation_geometry_str

    n_g, n_freq_bins, n_nodes = optical_depths_in.shape
    n_angles = emission_cos_angles.shape[0]
    n_layers = n_nodes - 1
    batch_size = n_g * n_freq_bins

    if n_layers <= 0:
        raise ValueError("compute_ada_radiative_transfer requires at least two depth nodes")

    planck_nodes_lf = physics_core.compute_planck_function_integral(
        temperatures,
        frequencies_bin_edges,
    )
    planck_nodes_fl = planck_nodes_lf.T
    planck_layers_fl = 0.5 * (planck_nodes_fl[:, :-1] + planck_nodes_fl[:, 1:])

    layer_optical_depths_gfl = jnp.maximum(
        optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1],
        0.0,
    )
    photon_destruction_probabilities_gfl = jnp.clip(
        0.5
        * (
            photon_destruction_probabilities_in[:, :, 1:]
            + photon_destruction_probabilities_in[:, :, :-1]
        ),
        0.0,
        1.0,
    )
    single_scatter_albedo_gfl = 1.0 - photon_destruction_probabilities_gfl

    mu_weight_a = emission_cos_angles * emission_cos_angles_weights
    surface_reflection_faa = jnp.broadcast_to(
        (2.0 * reflectances)[:, None, None] * mu_weight_a[None, None, :],
        (n_freq_bins, n_angles, n_angles),
    )
    surface_source_fa = (
        emissivities * planck_nodes_fl[:, -1]
    )[:, None] * jnp.ones((1, n_angles), dtype=JAX_DTYPE)

    composite_reflection_baa = jnp.broadcast_to(
        surface_reflection_faa[None, :, :, :],
        (n_g, n_freq_bins, n_angles, n_angles),
    ).reshape(batch_size, n_angles, n_angles)
    composite_source_ba = jnp.broadcast_to(
        surface_source_fa[None, :, :],
        (n_g, n_freq_bins, n_angles),
    ).reshape(batch_size, n_angles)

    layer_optical_depths_lbf = jnp.moveaxis(layer_optical_depths_gfl, -1, 0).reshape(
        n_layers,
        batch_size,
    )
    single_scatter_albedo_lbf = jnp.moveaxis(
        single_scatter_albedo_gfl,
        -1,
        0,
    ).reshape(n_layers, batch_size)
    thermal_source_lbf = jnp.broadcast_to(
        planck_layers_fl.T[:, None, :],
        (n_layers, n_g, n_freq_bins),
    ).reshape(n_layers, batch_size)

    def _adding_step(
        carry: tuple[ArrayLike, ArrayLike],
        layer_inputs: tuple[ArrayLike, ArrayLike, ArrayLike],
    ) -> tuple[tuple[ArrayLike, ArrayLike], None]:
        current_reflection_baa, current_source_ba = carry
        (
            layer_optical_depth_b,
            layer_single_scatter_albedo_b,
            layer_thermal_source_b,
        ) = layer_inputs

        layer_reflection_baa, layer_transmission_baa, layer_source_ba = (
            _build_layer_operator(
                layer_optical_depth_b,
                layer_single_scatter_albedo_b,
                layer_thermal_source_b,
                emission_cos_angles,
                emission_cos_angles_weights,
            )
        )

        gamma_inverse_baa = jnp.linalg.inv(
            _batched_identity(batch_size, n_angles)
            - current_reflection_baa @ layer_reflection_baa
        )
        transmission_gamma_baa = layer_transmission_baa @ gamma_inverse_baa
        reflected_source_ba = (
            current_reflection_baa @ layer_source_ba[..., None]
        )[..., 0]
        current_source_ba = layer_source_ba + (
            transmission_gamma_baa
            @ (reflected_source_ba + current_source_ba)[..., None]
        )[..., 0]
        current_reflection_baa = layer_reflection_baa + transmission_gamma_baa @ (
            current_reflection_baa @ layer_transmission_baa
        )

        return (current_reflection_baa, current_source_ba), None

    (top_reflection_baa, top_source_ba), _ = lax.scan(
        _adding_step,
        (composite_reflection_baa, composite_source_ba),
        (
            layer_optical_depths_lbf[::-1],
            single_scatter_albedo_lbf[::-1],
            thermal_source_lbf[::-1],
        ),
    )
    del top_reflection_baa

    top_intensity_gfa = top_source_ba.reshape(n_g, n_freq_bins, n_angles)
    flux_gf = 2.0 * jnp.pi * jnp.sum(
        top_intensity_gfa * mu_weight_a[None, None, :],
        axis=-1,
    )
    flux_f = jnp.sum(weights_gauss[:, None] * flux_gf, axis=0)
    top_intensity_af = jnp.transpose(
        jnp.sum(weights_gauss[:, None, None] * top_intensity_gfa, axis=0),
        (1, 0),
    )

    return flux_f, top_intensity_af
