import jax
import jax.numpy as jnp
import numpy as np

from petitRADTRANS.radtrans_core.linalg import (
    solve_tridiagonal_pcr_batched_feautrier_fused,
    solve_tridiagonal_thomas_batched_feautrier_fused,
)


def test_feautrier_pcr_matches_thomas_solver():
    n_g = 2
    n_freq_bins = 3
    n_angles = 4
    n_layers = 5
    batch = n_g * n_freq_bins * n_angles

    key = jax.random.PRNGKey(0)
    key_a, key_b, key_c, key_inner, key_surf, key_weights = jax.random.split(key, 6)

    a = 0.05 * jax.random.normal(key_a, (batch, n_layers))
    c = 0.05 * jax.random.normal(key_c, (batch, n_layers))
    b = 2.0 + jnp.abs(jax.random.normal(key_b, (batch, n_layers)))

    a = a.at[:, 0].set(0.0)
    c = c.at[:, -1].set(0.0)

    d_inner = jax.random.normal(key_inner, (n_g, n_freq_bins, n_layers - 1))
    d_surf = jax.random.normal(key_surf, (n_g, n_freq_bins, n_angles))
    angle_weight_a = jnp.abs(jax.random.normal(key_weights, (n_angles,)))
    angle_weight_a = angle_weight_a / jnp.sum(angle_weight_a)

    thomas_outputs = solve_tridiagonal_thomas_batched_feautrier_fused(
        a,
        b,
        c,
        d_inner,
        d_surf,
        angle_weight_a,
        n_g,
        n_freq_bins,
        n_angles,
        n_layers,
    )
    pcr_outputs = solve_tridiagonal_pcr_batched_feautrier_fused(
        a,
        b,
        c,
        d_inner,
        d_surf,
        angle_weight_a,
        n_g,
        n_freq_bins,
        n_angles,
        n_layers,
    )

    for thomas_output, pcr_output in zip(thomas_outputs, pcr_outputs):
        assert np.allclose(thomas_output, pcr_output, rtol=1e-6, atol=1e-6)
