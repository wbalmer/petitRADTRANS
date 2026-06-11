import jax
import jax.numpy as jnp
import numpy as np

from petitRADTRANS.radtrans_core.linalg import (
    _prepare_feautrier_system_lm_cpu,
    _prepare_feautrier_system_lm_gpu,
    _solve_feautrier_system_lm_cpu,
    _solve_feautrier_system_lm_assoc,
    _solve_feautrier_system_lm_gpu,
)


def test_feautrier_lm_solvers_agree():
    """The three layer-major Feautrier solvers must agree on the same system.

    - Thomas factorisation (``_prepare_feautrier_system_lm_cpu``) feeds both the
      sequential scan solve (CPU path) and the associative-scan solve (GPU path).
    - PCR (``_prepare_feautrier_system_lm_gpu`` + ``_solve_feautrier_system_lm_gpu``)
      is the independent small-batch GPU path.

    All solve the same tridiagonal system, so their outputs must match.
    """
    n_g, n_freq_bins, n_angles, n_layers = 2, 3, 4, 6
    n_gf = n_g * n_freq_bins

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    dtau1_lm = 0.1 * jnp.abs(jax.random.normal(k1, (n_layers - 1, n_gf))) + 1e-3
    dtau2_lm = 0.2 * jnp.abs(jax.random.normal(k2, (n_layers - 2, n_gf))) + 2e-3
    emission_cos_angles = jnp.linspace(0.15, 0.9, n_angles)
    angle_weight_a = jnp.abs(jax.random.normal(k3, (n_angles,)))
    angle_weight_a = angle_weight_a / jnp.sum(angle_weight_a)
    source_lm = 1e-3 * jnp.abs(jax.random.normal(k4, (n_layers, n_gf)))
    r_surf_ga = 1e-4 * jnp.abs(jax.random.normal(k5, (n_gf, n_angles)))

    # Thomas-factorisation path → sequential and associative-scan solves.
    thomas_state, _, _ = _prepare_feautrier_system_lm_cpu(
        dtau1_lm, dtau2_lm, emission_cos_angles, angle_weight_a,
        n_gf, n_angles, n_layers,
    )
    cpu_outputs = _solve_feautrier_system_lm_cpu(
        thomas_state, source_lm, r_surf_ga, angle_weight_a,
        n_gf, n_angles, n_layers,
    )
    assoc_outputs = _solve_feautrier_system_lm_assoc(
        thomas_state, source_lm, r_surf_ga, angle_weight_a,
        n_gf, n_angles, n_layers,
    )

    # PCR path → independent coefficient state.
    pcr_state, _, _ = _prepare_feautrier_system_lm_gpu(
        dtau1_lm, dtau2_lm, emission_cos_angles, angle_weight_a,
        n_gf, n_angles, n_layers,
    )
    pcr_outputs = _solve_feautrier_system_lm_gpu(
        pcr_state, source_lm, r_surf_ga, angle_weight_a,
        n_gf, n_angles, n_layers,
    )

    for cpu_output, assoc_output in zip(cpu_outputs, assoc_outputs):
        assert np.allclose(cpu_output, assoc_output, rtol=1e-6, atol=1e-6)
    for cpu_output, pcr_output in zip(cpu_outputs, pcr_outputs):
        assert np.allclose(cpu_output, pcr_output, rtol=1e-6, atol=1e-6)
