import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike
from petitRADTRANS.radtrans_core import physics_core

from petitRADTRANS.jax_dtype import JAX_DTYPE


def compute_rosseland_opacities(
    frequency_bins_edges,
    temperatures,
    weights_gauss,
    opacities,
    continuum_opacities_scattering,
    scattering_in_emission,
):
    opacities_rosseland = physics_core.compute_rosseland_opacities(
        opacities[:, :, 0, :],
        temperatures,
        weights_gauss,
        frequency_bins_edges,
        scattering_in_emission,
        continuum_opacities_scattering,
    )

    return opacities_rosseland
