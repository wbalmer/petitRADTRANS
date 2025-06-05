from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, ShapeDtypeStruct, pure_callback, debug


import numpy as np
from functools import partial
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.fortran_radtrans_core import math as fmath
import petitRADTRANS.jax_physics as jphys
import petitRADTRANS.jax_math as jmath
# Physical and mathematical constants (assuming they are available)
# e.g., from a constants module or passed as arguments
CST_PI = jnp.pi # Or the specific precision from Fortran
TINIEST = jnp.finfo(jnp.float64).tiny
HUGE_SQRT = jnp.sqrt(jnp.finfo(jnp.float64).max)


# --- Placeholder JAX helper functions (to be fully implemented) ---

def solve_tridiagonal_fortran(a, b, c, d):
    result_shape = ShapeDtypeStruct(d.shape, d.dtype)
    return pure_callback(fmath.solve_tridiagonal_system, result_shape, a, b, c, d)


@jit
def solve_tridiagonal_system_jax(a_lower_diag, b_main_diag, c_upper_diag, r_rhs):
    """
    Solves a tridiagonal system Ax = r_rhs using Thomas algorithm or jax.scipy.
    Args:
        a_lower_diag (jax.Array): Lower diagonal, shape (N-1,). a[i] is coeff of x_i for eq i+1.
        b_main_diag (jax.Array): Main diagonal, shape (N,).
        c_upper_diag (jax.Array): Upper diagonal, shape (N-1,). c[i] is coeff of x_{i+2} for eq i+1.
        r_rhs (jax.Array): Right-hand side, shape (N,).
    Returns:
        jax.Array: Solution x, shape (N,).
    """
    

    # Option 2: Implement Thomas algorithm with lax.scan (more complex)
    # This would replicate the Fortran logic more closely, including any specific numerical handling.
    # Dummy implementation for structure
    #return solve_tridiagonal_fortran(a_lower_diag, b_main_diag, c_upper_diag, r_rhs)
    return solve_tridiagonal_fortran(jnp.concatenate([jnp.array([0]), a_lower_diag]),
                                        b_main_diag,
                                        jnp.concatenate([c_upper_diag, jnp.array([0])]),
                                        r_rhs)


@partial(jit)
def m_approximate_ng_source_layer_jax(tn, tn1, tn2, tn3, current_source_layer):
    """
    Ng acceleration for a single layer's source function history.
    Inputs are all shape (n_layers,).
    """
    Q1 = tn - 2.0 * tn1 + tn2
    Q2 = tn - tn1 - tn2 + tn3
    Q3 = tn - tn1

    # Boundary condition from Fortran (Q1[0]=0 etc.)
    Q1 = Q1.at[0].set(0.0)
    Q2 = Q2.at[0].set(0.0)
    Q3 = Q3.at[0].set(0.0)

    A1 = jnp.minimum(jnp.sum(Q1 * Q1), HUGE_SQRT)
    A2 = jnp.minimum(jnp.sum(Q2 * Q1), HUGE_SQRT)
    B1 = jnp.minimum(jnp.sum(Q1 * Q2), HUGE_SQRT)
    B2 = jnp.minimum(jnp.sum(Q2 * Q2), HUGE_SQRT)
    C1 = jnp.minimum(jnp.sum(Q1 * Q3), HUGE_SQRT)
    C2 = jnp.minimum(jnp.sum(Q2 * Q3), HUGE_SQRT)

    AB_denominator = A1 * B2 - A2 * B1

    cond_skip_ng = (AB_denominator < TINIEST) | \
                   (jnp.abs(A1) < 1e-250) | (jnp.abs(A2) < 1e-250) | \
                   (jnp.abs(B1) < 1e-250) | (jnp.abs(B2) < 1e-250) | \
                   (jnp.abs(C1) < 1e-250) | (jnp.abs(C2) < 1e-250)

    def ng_update_func():
        a_ng = (C1 * B2 - C2 * B1) / AB_denominator
        b_ng = (C2 * A1 - C1 * A2) / AB_denominator
        return jnp.maximum((1.0 - a_ng - b_ng) * tn + a_ng * tn1 + b_ng * tn2, 0.0)

    def no_ng_update_func():
        return current_source_layer  # Or tn, depending on Fortran logic for `source`

    return lax.cond(cond_skip_ng, no_ng_update_func, ng_update_func)

# Vmap over (g, freq) dimensions
# Input source arrays will be (n_g, n_freq, n_layers)
vmapped_m_approximate_ng_source = vmap(vmap(m_approximate_ng_source_layer_jax,
                                                    in_axes=(0,0,0,0,0)), # vmap over freq
                                           in_axes=(0,0,0,0,0))          # vmap over g


# --- Main Feautrier Radiative Transfer JAX Function ---
@partial(jit, static_argnames=(
    "n_layers", "n_angles", "n_g", "emission_geometry_str", "return_contribution",
    "iter_scat_max", "n_iterations_min_convergence"
))
def compute_feautrier_radiative_transfer(
    frequencies_bin_edges,  # (n_frequencies_bin_edges,)
    temperatures,           # (n_layers,)
    weights_gauss,          # (n_g,)
    emission_cos_angles,    # (n_angles,)
    emission_cos_angles_weights, # (n_angles,)
    optical_depths_in,      # (n_g, n_freq_bins, n_layers)
    photon_destruction_probabilities_in, # (n_g, n_freq_bins, n_layers)
    stellar_intensity,      # (n_freq_bins,)
    star_irradiation_cos_angle, # scalar
    reflectances,           # (n_freq_bins,)
    emissivities,           # (n_freq_bins,)
    # Static parameters for JIT compilation
    n_layers: int,
    n_angles: int,
    n_g: int,
    emission_geometry_str: str,
    return_contribution: bool,
    iter_scat_max: int = 1000,
    n_iterations_min_convergence: int = 9, # Renamed from n_iterations_min
    convergence_threshold: float = 1e-3
):
    n_freq_bins = frequencies_bin_edges.shape[0] - 1
    pi_4 = 4.0 * CST_PI
    inv_del_tau_min = 1e10 # From Fortran
    inv_del_tau_min_half = inv_del_tau_min * 0.5

    # Reshape photon_destruction_probabilities: (n_g, n_freq, n_layers) -> (n_layers, n_g, n_freq)
    photon_destruction_probabilities_gfl = photon_destruction_probabilities_in # (g,f,l)

    mu_double = emission_cos_angles * 2.0
    surf_refl_2 = 2.0 * reflectances
    mu_weight_a = emission_cos_angles * emission_cos_angles_weights # Renamed for clarity (a,)

    # --- 1. Stellar Attenuation (J_star_ini) ---
    # J_star_ini: (n_g, n_freq_bins, n_layers)
    if emission_geometry_str == 'dayside_ave': geometry_coefficient = 0.5
    elif emission_geometry_str == 'planetary_ave': geometry_coefficient = 0.25
    elif emission_geometry_str == 'non-isotropic': geometry_coefficient = -1.0 # Special flag
    else: raise ValueError(f"Unknown emission_geometry: {emission_geometry_str}")

    if geometry_coefficient > 0:
        # optical_depths_in: (g, f, l), emission_cos_angles: (a)
        # stellar_intensity: (f)
        # exp_term: (g, f, a, l) = exp(- opt_depths(g,f,l) / mu(a) )
        exp_term_gfal = jnp.exp(-optical_depths_in[:, :, None, :] / emission_cos_angles[None, None, :, None])
        I_star_calc_gfal = geometry_coefficient * jnp.abs(stellar_intensity[None, :, None, None]) * exp_term_gfal
        # J_star_ini (g,f,l) = 0.5 * sum_a (I_star_calc_gfal * mu_weight(a))
        J_star_ini_gfl = 0.5 * jnp.sum(I_star_calc_gfal * emission_cos_angles_weights[None, None, :, None], axis=2)
    else: # non-isotropic
        exp_term_gfl = jnp.exp(-optical_depths_in / star_irradiation_cos_angle)
        J_star_ini_gfl = jnp.abs(stellar_intensity[None, :, None] / 4.0 * exp_term_gfl)

    # Clamp photon destruction probabilities
    photon_destruction_probabilities_gfl = jnp.maximum(photon_destruction_probabilities_gfl, 1e-10)

    # --- 2. Planck Function Integral ---
    # planck_integral_lf: (n_layers, n_freq_bins)
    planck_integral_lf = jphys.compute_planck_function_integral(temperatures, frequencies_bin_edges)
    planck_integral_fl = jnp.transpose(planck_integral_lf, (1,0)) # (f,l) for easier broadcast with (g,f,l)

    # --- 3. Surface Terms ---
    # I_surface_no_scattering_gf: (n_g, n_freq_bins)
    I_surface_emission_f = emissivities * planck_integral_lf[n_layers-1, :]  # (f,)
    
    if geometry_coefficient > 0:
        # I_star_calc_gfal was (g,f,a,l). Need I_star_calc at surface: I_star_calc_gfa_surf (g,f,a)
        I_star_calc_gfa_surf = I_star_calc_gfal[:, :, :, n_layers-1]
        I_surface_reflection_gf = surf_refl_2[None, :] * jnp.sum( # sum over angles
            I_star_calc_gfa_surf * mu_weight_a[None, None, :], axis=2
        )
    else: # non-isotropic
        I_surface_reflection_gf = reflectances[None, :] * J_star_ini_gfl[:, :, n_layers-1] * 4.0 * star_irradiation_cos_angle
    
    I_surface_no_scattering_gf = I_surface_emission_f[None, :] + I_surface_reflection_gf

    # --- 4. Initial Source Function & Precompute Feautrier Coefficients ---
    # S_gfl: (g, f, l)
    # planck_integral_fl is (f,l), broadcast to (g,f,l) using [None, :, :]
    # photon_destruction_probabilities_gfl is (g,f,l)
    initial_source_gfl = photon_destruction_probabilities_gfl * planck_integral_fl[None, :, :] + \
                         (1.0 - photon_destruction_probabilities_gfl) * J_star_ini_gfl

    # Delta taus for Feautrier coefficients
    # dtau1_gfl: (n_g, n_freq, n_layers-1), optical_depths_in(g,f,l)
    dtau1_gfl = optical_depths_in[:, :, 1:] - optical_depths_in[:, :, :-1]
    dtau1_gfl = jnp.where((jnp.abs(dtau1_gfl) < TINIEST), TINIEST, dtau1_gfl)
    
    # dtau2_gfl: (n_g, n_freq, n_layers-2)
    dtau2_gfl = optical_depths_in[:, :, 2:] - optical_depths_in[:, :, :-2]
    dtau2_gfl = jnp.where((jnp.abs(dtau2_gfl) < TINIEST), TINIEST, dtau2_gfl)
    
    # Feautrier coefficients (b_main_diags_gfal, a_lower_diags_gfalm1, c_upper_diags_gfalm1)
    # Shapes: (g, f, a, l) for main, (g, f, a, l-1) for off-diagonals
    
    # Top layer (l=0)
    f1_top_gfa = jnp.clip(emission_cos_angles[None, None, :, None] /
                          dtau1_gfl[:, :, None, 0, None],
                          -inv_del_tau_min,
                          inv_del_tau_min)
    b_gfal_0 = 1.0 + 2.0 * f1_top_gfa * (1.0 + f1_top_gfa)
    c_gfal_0 = -2.0 * f1_top_gfa**2
    # a_gfal_0 is 0

    # Mid layers (l=1 to n_layers-2)
    # f1_mid_gfalm1: (g,f,a,l-2), using dtau2 for layers 1 to n_layers-2
    f1_mid_gfalm1 = jnp.clip(mu_double[None, None, :, None] /
                             dtau2_gfl[:, :, None, :],
                             - inv_del_tau_min_half,
                             inv_del_tau_min_half)
    # f2_mid_gfalm1: (g,f,a,l-2), using dtau1 for layers 1 to n_layers-2 (dtau1 indices 1 to n_layers-2)
    f2_mid_gfalm1 = jnp.clip(emission_cos_angles[None, None, :, None] /
                             dtau1_gfl[:, :, None, 1:],
                             -inv_del_tau_min,
                             inv_del_tau_min)
    # f3_mid_gfalm1: (g,f,a,l-2), using dtau1 for layers 0 to n_layers-3 (dtau1 indices 0 to n_layers-3)
    f3_mid_gfalm1 = jnp.clip(emission_cos_angles[None, None, :, None] /
                             dtau1_gfl[:, :, None, :-1],
                             -inv_del_tau_min,
                             inv_del_tau_min)

    b_gfal_mid = 1.0 + f1_mid_gfalm1 * (f2_mid_gfalm1 + f3_mid_gfalm1)
    c_gfal_mid = -f1_mid_gfalm1 * f2_mid_gfalm1  # Upper diagonal part
    a_gfal_mid = -f1_mid_gfalm1 * f3_mid_gfalm1  # Lower diagonal part

    # Assemble full diagonal arrays for tridiagonal solver
    # b_main_diags_gfal: (g,f,a,l)
    _b = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers))
    _b = _b.at[:, :, :, 0].set(b_gfal_0[..., 0])
    _b = _b.at[:, :, :, 1:n_layers-1].set(b_gfal_mid)
    _b = _b.at[:, :, :, n_layers-1].set(1.0)  # Surface b=1
    b_main_diags_gfal = _b

    # c_upper_diags_gfalm1: (g,f,a,l-1)
    _c = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers - 1))
    _c = _c.at[:, :, :, 0].set(c_gfal_0[..., 0])
    _c = _c.at[:, :, :, 1:n_layers-1].set(c_gfal_mid)  # c_gfal_mid is shape (g,f,a,l-2)
    # This covers upper diags for rows 1 to n_layers-2
    c_upper_diags_gfalm1 = _c
    
    # a_lower_diags_gfalm1: (g,f,a,l-1)
    _a = jnp.zeros((n_g, n_freq_bins, n_angles, n_layers - 1))
    # a_gfal_mid is shape (g,f,a,l-2), covers lower diags for rows 1 to n_layers-2
    _a = _a.at[:, :, :, :n_layers-2].set(a_gfal_mid)
    a_lower_diags_gfalm1 = _a

    # Lambda_loc for ALI (Accelerated Lambda Iteration)
    # lambda_loc_gfl: (n_g, n_freq_bins, n_layers)
    lambda_loc_contrib_top_gfa = mu_weight_a[None, None, :, None] / b_gfal_0
    lambda_loc_top_gf = jnp.sum(lambda_loc_contrib_top_gfa, axis=2)  # sum over angles

    lambda_loc_contrib_mid_gfalm1 = mu_weight_a[None, None, :, None] / b_gfal_mid
    lambda_loc_mid_gflm2 = jnp.sum(lambda_loc_contrib_mid_gfalm1, axis=2)

    f1_surf_gfa = jnp.clip(emission_cos_angles[None, None, :, None] / dtau1_gfl[:, :, None, n_layers-2, None],
                           -inv_del_tau_min,
                           inv_del_tau_min)
    lambda_loc_contrib_surf_gfa = mu_weight_a[None, None, :, None] / (1.0 + 2.0 * f1_surf_gfa**2)
    lambda_loc_surf_gf = jnp.sum(lambda_loc_contrib_surf_gfa, axis=2)

    _l = jnp.zeros((n_g, n_freq_bins, n_layers))
    _l = _l.at[:, :, 0].set(lambda_loc_top_gf[..., 0])
    _l = _l.at[:, :, 1:n_layers-1].set(lambda_loc_mid_gflm2)
    _l = _l.at[:, :, n_layers-1].set(lambda_loc_surf_gf[..., 0])
    lambda_loc_gfl = _l # (g,f,l)

    # --- 5. Scattering Iteration Loop ---
    # State: (current_source_gfl, current_I_plus_surface_gfa, I_H_gfal_prev, J_planet_scat_gfl_prev,
    #         s_n_gfl, s_n1_gfl, s_n2_gfl, s_n3_gfl, iter_idx, converged_flag)
    # current_I_plus_surface_gfa: (n_g, n_freq_bins, n_angles)

    def scattering_loop_body_func(loop_state_prev):
        (current_source_gfl, current_I_plus_surface_gfa, I_H_gfal_prev, J_planet_scat_gfl_prev,
         s_n_gfl, s_n1_gfl, s_n2_gfl, s_n3_gfl, iter_idx, converged_flag) = loop_state_prev

        # --- 5a. Solve Feautrier System for each (g,f,a) ---
        # RHS for tridiagonal system: r_gfal (g,f,a,l)
        # Source part: current_source_gfl (g,f,l)
        r_gfal = jnp.repeat(current_source_gfl[:, :, None, :], n_angles, axis=2)  # (g,f,a,l)

        # Surface boundary for r_gfal at n_layers-1
        # I_minus_boundary_gf (g,f) = I_surf_no_scat_gf + refl_const_f * sum_a(I_plus_surf_gfa * mu_w_a)
        # current_I_plus_surface_gfa (g,f,a)
        
        reflection_term_gf = surf_refl_2[None, :] *\
            jnp.sum(current_I_plus_surface_gfa * mu_weight_a[None, None, :], axis=2) # sum over angles
        I_minus_boundary_gf = I_surface_no_scattering_gf + reflection_term_gf
        
        r_gfal_surface = 0.5 * (current_I_plus_surface_gfa + I_minus_boundary_gf[:, :, None])  # (g,f,a)
        r_gfal = r_gfal.at[:, :, :, n_layers-1].set(r_gfal_surface)

        # Vectorize solver: solve_tridiagonal_system_jax(a_lower(l-1), b_main(l), c_upper(l-1), r_rhs(l))
        # Inputs are (g,f,a,dims). Vmap over (g,f,a).
        # I_J_gfal: (g,f,a,l)
        I_J_gfal = vmap(vmap(vmap(solve_tridiagonal_system_jax, in_axes=(0, 0, 0, 0))))(
            a_lower_diags_gfalm1, b_main_diags_gfal, c_upper_diags_gfalm1, r_gfal
        )

        # --- 5b. Calculate I_H_gfal_current and new I_plus_surface_gfa ---
        I_H_gfal_current = jnp.zeros_like(I_J_gfal) # Renamed
        I_H_gfal_current = I_H_gfal_current.at[:, :, :, 0].set(-I_J_gfal[:, :, :, 0])  # Top layer

        # Mid layers for I_H
        # deriv1_gfalm1: (g,f,a,l-2) using f2_mid and I_J for layers 1 to n_layers-2
        deriv1_gfalm1 = f2_mid_gfalm1 * (I_J_gfal[:, :, :, 2:n_layers] - I_J_gfal[:, :, :, 1:n_layers-1])
        # deriv2_gfalm1: (g,f,a,l-2) using f3_mid and I_J for layers 0 to n_layers-3
        deriv2_gfalm1 = f3_mid_gfalm1 * (I_J_gfal[:, :, :, 1:n_layers-1] - I_J_gfal[:, :, :, 0:n_layers-2])
        I_H_gfal_current = I_H_gfal_current.at[:, :, :, 1:n_layers-1].set(-(deriv1_gfalm1 + deriv2_gfalm1) * 0.5)
        
        # New I_plus_surface_gfa (g,f,a) for next iteration
        # I_plus_surface = I_J_surface - deriv1_surface 
        #   (deriv1 for k=n_layers-1, which is last element of deriv1_gfalm1)
        new_I_plus_surface_gfa = I_J_gfal[:, :, :, n_layers-1] - deriv1_gfalm1[:, :, :, n_layers-2]

        # --- 5c. Update Source Function (ALI + Ng) ---
        # J_bol_g_gfl (g,f,l) = sum_a (I_J_gfal * mu_w_a)
        J_bol_g_gfl = jnp.sum(I_J_gfal * mu_weight_a[None, None, :, None], axis=2)
        J_planet_scat_gfl_current = J_bol_g_gfl  # (g,f,l)

        # ALI update for source_lgf
        # photon_destruction_probabilities_gfl (g,f,l), planck_integral_fl (f,l)
        # J_star_ini_gfl (g,f,l), lambda_loc_gfl (g,f,l), current_source_gfl (g,f,l)
        num_gfl = photon_destruction_probabilities_gfl * planck_integral_fl[None, :, :] + \
            (1.0 - photon_destruction_probabilities_gfl) * \
            (J_star_ini_gfl + J_planet_scat_gfl_current - lambda_loc_gfl * current_source_gfl)
        den_gfl = 1.0 - (1.0 - photon_destruction_probabilities_gfl) * lambda_loc_gfl
        source_after_ali_gfl = num_gfl / jnp.maximum(den_gfl, TINIEST)
        
        # Ng Acceleration History Update
        next_s_n3_gfl, next_s_n2_gfl, next_s_n1_gfl = s_n2_gfl, s_n1_gfl, s_n_gfl
        next_s_n_gfl = source_after_ali_gfl  # Store pre-Ng source for history
        
        # Apply Ng (iter_idx is 0-based)
        # Fortran: mod(i_iter_scat, 4) == 0. Here: (iter_idx + 1) % 4 == 0
        def ng_accel_func():
            # vmapped_m_approximate_ng_source expects (g,f,l) inputs
            accelerated_s_gfl = vmapped_m_approximate_ng_source(
                next_s_n_gfl, next_s_n1_gfl, next_s_n2_gfl, next_s_n3_gfl, source_after_ali_gfl
            )
            return accelerated_s_gfl

        final_updated_source_gfl = lax.cond(
            (iter_idx > 0) & ((iter_idx + 1) % 4 == 0),  # iter_idx is 0 to iter_scat_max-1
            ng_accel_func,
            lambda: source_after_ali_gfl  # No Ng
        )
        
        # Convergence check based on relative change in total I_H at the top boundary
        total_I_H_top_prev = jnp.sum(I_H_gfal_prev[:, :, :, 0], axis=(0, 2)) # Sum over g and angles
        total_I_H_top_current = jnp.sum(I_H_gfal_current[:, :, :, 0], axis=(0, 2))
        max_rel_change = jnp.max(1 - total_I_H_top_prev / total_I_H_top_current)
        new_converged_flag = (iter_idx > n_iterations_min_convergence) &\
            (max_rel_change < convergence_threshold)
        
        # Update loop state
        next_loop_state = (final_updated_source_gfl, new_I_plus_surface_gfa, I_H_gfal_current, J_planet_scat_gfl_current,
                           next_s_n_gfl, next_s_n1_gfl, next_s_n2_gfl, next_s_n3_gfl,
                           iter_idx + 1, new_converged_flag)

        return next_loop_state  # keep returning I_H for final flux

    def convergence_condition_func(loop_state):
        # Note: the requested condition had iter_idx > iter_scat_max for *continuation*.  
        # Here we want the loop to *stop* when converged OR max iterations reached.
        _, _, _, _, _, _, _, _, iter_idx, converged_flag = loop_state
        return (iter_idx <= iter_scat_max) & (jnp.logical_not(converged_flag))
        # Stopping condition: loop runs WHILE (not converged AND iter_idx <= iter_scat_max)

    # Initial state for while_loop (using (g,f,l) and (g,f,a) layouts)
    initial_iter_state = (initial_source_gfl,                       # current_source_gfl
                          jnp.zeros((n_g, n_freq_bins, n_angles)),  # current_I_plus_surface_gfa
                          jnp.zeros((n_g, n_freq_bins, n_angles, n_layers)),  # I_H_gfal_prev
                          jnp.zeros_like(initial_source_gfl),       # J_planet_scat_gfl_prev
                          jnp.zeros_like(initial_source_gfl),       # s_n_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n1_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n2_gfl
                          jnp.zeros_like(initial_source_gfl),       # s_n3_gfl
                          0,                                       # iter_idx
                          False)                                   # converged_flag

    final_loop_state = lax.while_loop(convergence_condition_func,
                                      scattering_loop_body_func,
                                      initial_iter_state)
    
    # Extract final values from the loop state.
    # The I_H_gfal_prev and J_planet_scat_gfl_prev in the final_loop_state
    # are the I_H_gfal_current and J_planet_scat_gfl_current from the last completed iteration.
    _, _, I_H_gfal_final, _, _, _, _, _, _, _ = final_loop_state

    # --- 6. Calculate Flux ---
    # flux_f (n_freq_bins) = sum_g sum_a (-I_H_gfal(g,f,a,l=0) * pi_4 * w_gauss_g * mu_w_a)
    # I_H_gfal_final is (g,f,a,l). We need I_H at top (l=0).
    I_H_top_gfa = I_H_gfal_final[:, :, :, 0]  # (g,f,a)
    flux_f = jnp.sum(
        -I_H_top_gfa * pi_4 * weights_gauss[:, None, None] * mu_weight_a[None, None, :],
        axis=(0, 2)  # Sum over g and angles
    )

    # --- 7. Contribution Function (if return_contribution) ---
    emission_contribution_lf = jnp.zeros((n_layers, n_freq_bins))
    if return_contribution:
        # Logic similar to Fortran:
        # transm_mu_gfl = exp(-optical_depths_in / emission_cos_angles)
        # transm_all_fl = sum_g (transm_mu_gfl * weights_gauss_g)
        # r_source_fl = sum_g (final_source_lgf (transposed to g,f,l) * weights_gauss_g)
        # Loop over angles i_mu:
        #   contrib_lf += (r_source_fl(l) + r_source_fl(l+1)) * (transm_all_fl(l) - transm_all_fl(l+1)) * mu * mu_w
        #   Handle surface term carefully.
        # Normalize contrib_lf.
        # This part needs careful translation of the loops and sums.
        pass  # Placeholder for detailed implementation

    return flux_f, emission_contribution_lf
