"""Registration and calling helpers for the native correlated-k mixing kernel.

The kernel lives in ``petitRADTRANS/cpp_src/prt_ck_mixing_ffi.cc`` and is
built as the ``petitRADTRANS.prt_ck_mixing_ffi`` extension module. It
replaces the former Fortran ``combine_ck_opacities`` ``pure_callback`` path:
the forward pass stays inside the XLA program as a typed FFI custom call,
runs in parallel on the XLA intra-op thread pool, and consumes the
mass-fraction-weighted opacities in their natural
``(n_frequencies * n_layers, n_species, n_g)`` row-major layout (no
Fortran-order transpose, no host round-trip, no GIL).
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

_CK_MIX_TARGETS = {
    "float32": "prt_combine_ck_mix_f32",
    "float64": "prt_combine_ck_mix_f64",
}

_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()


def ensure_ck_mixing_ffi_registered() -> None:
    """Register the c-k mixing FFI targets once per process.

    Raises an informative ImportError if the native extension is missing,
    since there is no fallback implementation on CPU.
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

        try:
            jax.ffi.register_ffi_target(
                _CK_MIX_TARGETS["float32"],
                prt_ck_mixing_ffi.combine_ck_mix_f32_capsule(),
                platform="cpu",
            )
            jax.ffi.register_ffi_target(
                _CK_MIX_TARGETS["float64"],
                prt_ck_mixing_ffi.combine_ck_mix_f64_capsule(),
                platform="cpu",
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

    Parameters
    ----------
    weighted_opacities : jax.Array, shape (n_fl, n_species, n_g)
        Mass-fraction-weighted line opacities, flattened over
        (frequency, layer) cells, species axis second, g axis last
        (row-major). The continuum must NOT be folded in; it is passed
        separately so the kernel can reproduce the Fortran threshold
        semantics exactly.
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
