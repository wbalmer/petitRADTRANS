"""Registration and calling helpers for the native correlated-k mixing kernels.

The kernels live in ``petitRADTRANS/cpp_src/prt_ck_mixing_ffi.cc`` and are
built as the ``petitRADTRANS.prt_ck_mixing_ffi`` extension module. They
replace the former Fortran ``combine_ck_opacities`` ``pure_callback`` path:
the mixing and its forward- and reverse-mode derivatives stay inside the XLA
program as typed FFI custom calls, run in parallel on the XLA intra-op thread
pool, and consume the mass-fraction-weighted opacities in their natural
``(n_frequencies * n_layers, n_species, n_g)`` row-major layout (no
Fortran-order transpose, no host round-trip, no GIL).
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Specify data types for c++ fumctions
_CK_MIX_TARGETS = {
    "float32": "prt_combine_ck_mix_f32",
    "float64": "prt_combine_ck_mix_f64",
}
_CK_MIX_JVP_TARGETS = {
    "float32": "prt_combine_ck_mix_jvp_f32",
    "float64": "prt_combine_ck_mix_jvp_f64",
}
_CK_MIX_VJP_TARGETS = {
    "float32": "prt_combine_ck_mix_vjp_f32",
    "float64": "prt_combine_ck_mix_vjp_f64",
}

_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()


def ensure_ck_mixing_ffi_registered() -> None:
    """Register the c-k mixing FFI targets once per process.

    Raises an informative ImportError if the native extension is missing,
    since there is no fallback implementation on CPU. Should always be
    registered if pRT is installed properly.
    """
    global _REGISTERED

    if _REGISTERED:
        return

    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return

        try:
            from petitRADTRANS import prt_ck_mixing_ffi
        except ImportError as error:
            raise ImportError(
                "the native correlated-k mixing extension "
                "(petitRADTRANS.prt_ck_mixing_ffi) is not available; "
                "reinstall petitRADTRANS so its native extensions are built "
                "(e.g. 'pip install -e .' from the repository root)"
            ) from error

        for targets, capsule_prefix in (
            (_CK_MIX_TARGETS, "combine_ck_mix"),
            (_CK_MIX_JVP_TARGETS, "combine_ck_mix_jvp"),
            (_CK_MIX_VJP_TARGETS, "combine_ck_mix_vjp"),
        ):
            for dtype_name, suffix in (("float32", "f32"), ("float64", "f64")):
                capsule_function = getattr(
                    prt_ck_mixing_ffi, f"{capsule_prefix}_{suffix}_capsule", None
                )

                if capsule_function is None:
                    raise ImportError(
                        "the native correlated-k mixing extension "
                        "(petitRADTRANS.prt_ck_mixing_ffi) predates its derivative "
                        "kernels; rebuild petitRADTRANS's native extensions "
                        "(e.g. 'pip install -e .' from the repository root)"
                    )

                try:
                    jax.ffi.register_ffi_target(
                        targets[dtype_name], capsule_function(), platform="cpu"
                    )
                except ValueError as error:
                    # Tolerate duplicate registration (e.g. across reloads).
                    if "Duplicate custom call target registration" not in str(error):
                        raise

        _REGISTERED = True


def combine_ck_mixing_ffi(
    weighted_opacities: ArrayLike,  # (n_fl, n_species, n_g)
    continuum_opacities_flat: ArrayLike,  # (n_fl,)
    g_gauss: ArrayLike,  # (n_g,)
    weights_gauss: ArrayLike,  # (n_g,)
) -> ArrayLike:
    """Mix correlated-k opacities across species with the native CPU kernel.
    Thin python wrapper around the c++ functions for easy access.

    Parameters
    ----------
    weighted_opacities : jax.Array, shape (n_fl, n_species, n_g)
        Mass-fraction-weighted line opacities, flattened over
        (frequency, layer) cells, species axis second, g axis last
        (row-major). The continuum must NOT be folded in; it is passed
        separately so the kernel can include it in the mixing threshold
        exactly as the JAX implementation does.
    continuum_opacities_flat : jax.Array, shape (n_fl,)
        Continuum opacities for each flattened cell, added to species 0
        inside the kernel.
    g_gauss, weights_gauss : jax.Array, shape (n_g,)
        Gaussian quadrature nodes and weights of the g grid.

    Returns
    -------
    jax.Array, shape (n_fl, n_g)
        Mixed opacities on the input g grid, in the input dtype.
    """
    ensure_ck_mixing_ffi_registered()

    dtype = jnp.dtype(weighted_opacities.dtype)
    target_name = _CK_MIX_TARGETS.get(dtype.name)
    if target_name is None:
        raise TypeError(f"unsupported dtype for c-k mixing FFI: {dtype}")

    output_shape = jax.ShapeDtypeStruct(
        (weighted_opacities.shape[0], weighted_opacities.shape[2]), dtype
    )

    return jax.ffi.ffi_call(
        target_name,
        output_shape,
        vmap_method="sequential",
        input_layouts=[(0, 1, 2), (0,), (0,), (0,)],
        output_layouts=(0, 1),
    )(
        weighted_opacities,
        jnp.asarray(continuum_opacities_flat, dtype=dtype),
        jnp.asarray(g_gauss, dtype=dtype),
        jnp.asarray(weights_gauss, dtype=dtype),
    )


def combine_ck_mixing_jvp_ffi(
    weighted_opacities: ArrayLike,  # (n_fl, n_species, n_g)
    continuum_opacities_flat: ArrayLike,  # (n_fl,)
    g_gauss: ArrayLike,  # (n_g,)
    weights_gauss: ArrayLike,  # (n_g,)
    weighted_opacity_tangents: ArrayLike,  # (n_tangents, n_fl, n_species, n_g)
    continuum_tangents: ArrayLike,  # (n_tangents, n_fl)
) -> ArrayLike:
    """Forward-mode derivative of the native CPU mixing kernel.

    Parameters
    ----------
    weighted_opacities, continuum_opacities_flat, g_gauss, weights_gauss
        As for :func:`combine_ck_mixing_ffi`.
    weighted_opacity_tangents : jax.Array, shape (n_tangents, n_fl, n_species, n_g)
        Tangents of the weighted opacities.
    continuum_tangents : jax.Array, shape (n_tangents, n_fl)
        Tangents of the continuum opacities.

    Returns
    -------
    jax.Array, shape (n_tangents, n_fl, n_g)
        Tangents of the mixed opacities, in the input dtype.
    """
    ensure_ck_mixing_ffi_registered()

    dtype = jnp.dtype(weighted_opacities.dtype)
    target_name = _CK_MIX_JVP_TARGETS.get(dtype.name)
    if target_name is None:
        raise TypeError(f"unsupported dtype for c-k mixing FFI: {dtype}")

    n_tangents = weighted_opacity_tangents.shape[0]
    output_shape = jax.ShapeDtypeStruct(
        (n_tangents, weighted_opacities.shape[0], weighted_opacities.shape[2]), dtype
    )

    return jax.ffi.ffi_call(
        target_name,
        output_shape,
        vmap_method="sequential",
        input_layouts=[(0, 1, 2), (0,), (0,), (0,), (0, 1, 2, 3), (0, 1)],
        output_layouts=(0, 1, 2),
    )(
        weighted_opacities,
        jnp.asarray(continuum_opacities_flat, dtype=dtype),
        jnp.asarray(g_gauss, dtype=dtype),
        jnp.asarray(weights_gauss, dtype=dtype),
        jnp.asarray(weighted_opacity_tangents, dtype=dtype),
        jnp.asarray(continuum_tangents, dtype=dtype),
    )


def combine_ck_mixing_vjp_ffi(
    weighted_opacities: ArrayLike,  # (n_fl, n_species, n_g)
    continuum_opacities_flat: ArrayLike,  # (n_fl,)
    g_gauss: ArrayLike,  # (n_g,)
    weights_gauss: ArrayLike,  # (n_g,)
    cotangents: ArrayLike,  # (n_tangents, n_fl, n_g)
) -> tuple[ArrayLike, ArrayLike]:
    """Reverse-mode derivative of the native CPU mixing kernel.

    Parameters
    ----------
    weighted_opacities, continuum_opacities_flat, g_gauss, weights_gauss
        As for :func:`combine_ck_mixing_ffi`.
    cotangents : jax.Array, shape (n_tangents, n_fl, n_g)
        Cotangents of the mixed opacities.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Cotangents of the weighted opacities, shape (n_tangents, n_fl, n_species, n_g),
        and of the continuum opacities, shape (n_tangents, n_fl), in the input dtype.
    """
    ensure_ck_mixing_ffi_registered()

    dtype = jnp.dtype(weighted_opacities.dtype)
    target_name = _CK_MIX_VJP_TARGETS.get(dtype.name)
    if target_name is None:
        raise TypeError(f"unsupported dtype for c-k mixing FFI: {dtype}")

    n_tangents = cotangents.shape[0]
    output_shapes = (
        jax.ShapeDtypeStruct((n_tangents,) + tuple(weighted_opacities.shape), dtype),
        jax.ShapeDtypeStruct((n_tangents, weighted_opacities.shape[0]), dtype),
    )

    return jax.ffi.ffi_call(
        target_name,
        output_shapes,
        vmap_method="sequential",
        input_layouts=[(0, 1, 2), (0,), (0,), (0,), (0, 1, 2)],
        output_layouts=[(0, 1, 2, 3), (0, 1)],
    )(
        weighted_opacities,
        jnp.asarray(continuum_opacities_flat, dtype=dtype),
        jnp.asarray(g_gauss, dtype=dtype),
        jnp.asarray(weights_gauss, dtype=dtype),
        jnp.asarray(cotangents, dtype=dtype),
    )
