import jax.numpy as jnp
from jax import jit,vmap,lax

import petitRADTRANS.physical_constants as cst

ABUNDANCE_THRESHOLD = 1e-60

@jit
def H2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates H2 Rayleigh scattering opacities (optimized)."""

    # Constants hoisted out of the calculation loop
    c1 = 8.14e-13 / 1.66053892e-24 * 0.5
    c2 = 1.28e-6 / 1.66053892e-24 * 0.5
    c3 = 1.61 / 1.66053892e-24 * 0.5

    # Precompute inverse powers to avoid redundant calculations
    inv_wavelengths2 = 1 / wavelengths_angstroem**2
    inv_wavelengths4 = inv_wavelengths2**2
    inv_wavelengths6 = inv_wavelengths4 * inv_wavelengths2
    inv_wavelengths8 = inv_wavelengths6 * inv_wavelengths2

    # Compute the factor vectorized
    factor = c1 * inv_wavelengths4 + c2 * inv_wavelengths6 + c3 * inv_wavelengths8

    # Compute opacities, avoiding unnecessary zeros
    rayleigh_scattering_opacities = factor[:, None] * mass_fractions
    rayleigh_scattering_opacities = jnp.where(
        rayleigh_scattering_opacities < ABUNDANCE_THRESHOLD,
        0.0,
        rayleigh_scattering_opacities
    )

    return rayleigh_scattering_opacities

@jit
def He_rayleigh(wavelengths_angstroem,mass_fractions):
    """
    Calculates He Rayleigh scattering opacities using Chan & Dalgarno (1965) alphas.

    Args:
        wavelengths_angstroem: JAX array of wavelengths in Angstroms.
        mass_fractions: JAX array of mass fractions for each layer.
        ABUNDANCE_THRESHOLD: Threshold below which opacity is set to zero.

    Returns:
        JAX array of Rayleigh scattering opacities for each wavelength and layer.
    """
    lambda_cm = wavelengths_angstroem * 1e-8
    l = lambda_cm * 1e4

    # Calculate alpha_polarization
    alpha_polarization = (
        2.265983321 - 3.721350022 * l + 3.016150391 * l**2
    )

    # Adjust for wavelengths above the threshold
    alpha_polarization = jnp.where(
        lambda_cm >= 0.9110e-4, 1.379, alpha_polarization
    )

    # Calculate multiplicative factor using alpha_polarization
    alpha_polarization = (
        128 * jnp.pi**5 / 3 / lambda_cm**4
        * (alpha_polarization * 1.482e-25)**2 / 4 / 1.66053892e-24
    )

    # Calculate Rayleigh scattering opacities with broadcasting
    rayleigh_scattering_opacities = alpha_polarization[:, None] * mass_fractions[None, :]

    # Apply abundance threshold
    rayleigh_scattering_opacities = jnp.where(
        mass_fractions[None, :] < ABUNDANCE_THRESHOLD,
        0.0,
        rayleigh_scattering_opacities
    )

    return rayleigh_scattering_opacities

@jit
def h2o_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    a0, a1, a2, a3, a4, a5, a6, a7 = 0.244257733, 9.74634476e-3, -3.73234996e-3, 2.68678472e-4, 1.58920570e-3, 2.45934259e-3, 0.900704920, -1.66626219e-2
    luv, lir = 0.2292020, 5.432937
    mass_h2o = 18 * cst.amu

    l = lambda_cm / 1e-4 / 0.589
    d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    temperatures_ = temperatures / 273.15

    @jit
    def H2O_rayleigh_single_layer(i_str):
        #if mass_fractions[i_str] < ABUNDANCE_THRESHOLD:
        #    return jnp.zeros(n_frequencies)

        retVal = (
            a0 +
            a1 * d[i_str] +
            a2 * temperatures_[i_str] +
            a3 * l**2 * temperatures_[i_str] +
            a4 / l**2 +
            a5 / (l**2 - luv**2) +
            a6 / (l**2 - lir**2) +
            a7 * d[i_str]**2
        ) * d[i_str]

        retValMin = (
            a0 +
            a1 * d[i_str] +
            a2 * temperatures_[i_str] +
            a3 * (0.2 / 0.589)**2 * temperatures_[i_str] +
            a4 / (0.2 / 0.589)**2 +
            a5 / ((0.2 / 0.589)**2 - luv**2) +
            a6 / ((0.2 / 0.589)**2 - lir**2) +
            a7 * d[i_str]**2
        ) * d[i_str]

        retValMax = (
            a0 +
            a1 * d[i_str] +
            a2 * temperatures_[i_str] +
            a3 * (1.1 / 0.589)**2 * temperatures_[i_str] +
            a4 / (1.1 / 0.589)**2 +
            a5 / ((1.1 / 0.589)**2 - luv**2) +
            a6 / ((1.1 / 0.589)**2 - lir**2) +
            a7 * d[i_str]**2
        ) * d[i_str]

        nm1 = jnp.where((lambda_cm / 1e-4 > 0.2) & (lambda_cm / 1e-4 < 1.1),
                        jnp.sqrt((1 + 2 * retVal) / (1 - retVal)),
                        jnp.where(lambda_cm / 1e-4 >= 1.1,
                                    jnp.sqrt((1 + 2 * retValMax) / (1 - retValMax)),
                                    jnp.sqrt((1 + 2 * retValMin) / (1 - retValMin))))
        nm1 = nm1 - 1.0
        fk = 1.0

        result = (24 * jnp.pi**3 * lamb_inv**4 / (d[i_str] / 18 / cst.amu)**2 *
                    (((nm1 + 1)**2 - 1) / ((nm1 + 1)**2 + 2))**2 * fk / mass_h2o *
                    mass_fractions[i_str])
        return jnp.where(jnp.isnan(result), 0, result).T
    return vmap(H2O_rayleigh_single_layer)(jnp.arange(mass_fractions.shape[0])).T


@jit
def co2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    rayleigh_scattering_opacities = jnp.zeros((wavelengths_angstroem.size, mass_fractions.size))
    d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    mass_co2 = 44.0 * cst.amu
    # Use lax.cond for conditional execution within JIT
    def compute_opacity(i_str, opacities):
        scale = d[i_str] / 44.0 / cst.amu / cst.standard_number_density_for_refractive_index

        nm1 = 1e-3 * 1.1427e6 * (
            5799.25 / jnp.maximum(20.0**2, 128908.9**2 - lamb_inv**2) +
            120.05 / jnp.maximum(20.0**2, 89223.8**2 - lamb_inv**2) +
            5.3334 / jnp.maximum(20.0**2, 75037.5**2 - lamb_inv**2) +
            4.3244 / jnp.maximum(20.0**2, 67837.7**2 - lamb_inv**2) +
            0.1218145e-4 / jnp.maximum(20.0**2, 2418.136**2 - lamb_inv**2)
        ) * scale

        fk = 1.1364 + 25.3e-12 * lamb_inv**2

        new_opacities = opacities.at[:, i_str].set(
            opacities[:, i_str]
            + 24.0 * jnp.pi**3 * lamb_inv**4 / (scale * cst.standard_number_density_for_refractive_index)**2
            * (((nm1 + 1.0)**2 - 1.0) / ((nm1 + 1.0)**2 + 2.0))**2 * fk / mass_co2
            * mass_fractions[i_str]
        )

        return new_opacities

    rayleigh_scattering_opacities = lax.fori_loop(
        0,
        mass_fractions.size,
        lambda i_str, opacities: lax.cond(
            mass_fractions[i_str] > ABUNDANCE_THRESHOLD,
            compute_opacity,
            lambda i_str, opacities: opacities,  # No-op if condition fails
            i_str,
            opacities,
        ),
        rayleigh_scattering_opacities,
    )
    return rayleigh_scattering_opacities

@jit
def o2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    """Calculates O2 Rayleigh scattering opacities (JIT-compatible)."""
    #d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    mass_o2 = 32.0 * cst.amu

    # Preallocate output array (required for JIT)
    rayleigh_scattering_opacities = jnp.zeros((wavelengths_angstroem.size, mass_fractions.size))

    # Use lax.cond for conditional execution within JIT
    def compute_opacity(i_str, opacities):
        #scale = d[i_str] / mass_o2 / 2.68678e19

        # Use jnp.where for conditional assignment of A and B
        A = jnp.where(lamb_inv > 18315.0, 20564.8, 21351.1)
        B = jnp.where(lamb_inv > 18315.0, 2.480899e13, 2.18567e13)
        C = 4.09e9

        # Vectorized calculation of nm1 and fk
        nm1 = 1e-8 * (A + B / (C - lamb_inv**2))
        #nm1 *= scale # WHY NOT!?
        fk = 1.096 + 1.385e-11 * lamb_inv**2 + 1.448e-20 * lamb_inv**4

        # Calculate Rayleigh scattering opacities, broadcasting for layers
        new_opacities = opacities.at[:, i_str].set(
            opacities[:, i_str]
            + 24.0 * jnp.pi**3 * lamb_inv**4 / (2.68678e19)**2
            * (((nm1 + 1.0)**2 - 1.0) / ((nm1 + 1.0)**2 + 2.0))**2 * fk / mass_o2
            * mass_fractions[i_str]
        )
        return new_opacities

    # Use lax.fori_loop for JIT-compatible loop
    rayleigh_scattering_opacities = lax.fori_loop(
        0,
        mass_fractions.size,
        lambda i_str, opacities: lax.cond(
            mass_fractions[i_str] > ABUNDANCE_THRESHOLD,
            compute_opacity,
            lambda i_str, opacities: opacities,  # No-op if condition fails
            i_str,
            opacities,
        ),
        rayleigh_scattering_opacities,
    )
    return rayleigh_scattering_opacities

@jit
def n2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    """Calculates N2 Rayleigh scattering opacities (JIT-compatible)."""
    d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    mass_n2 = 34.0 * cst.amu

    # Preallocate output array (required for JIT)
    rayleigh_scattering_opacities = jnp.zeros((wavelengths_angstroem.size, mass_fractions.size))

    # Use lax.cond for conditional execution within JIT
    def compute_opacity(i_str, opacities):
        scale = d[i_str] / mass_n2 / 2.546899e19

        # Nested jnp.where for conditional assignment of A, B, and C
        A = jnp.where(
            lamb_inv > 21360.0,
            5677.465,
            jnp.where(lamb_inv > 4860.0, 6498.2, 0.0)  # 0 if not in either range
        )
        B = jnp.where(
            lamb_inv > 21360.0,
            318.81874e12,
            jnp.where(lamb_inv > 4860.0, 307.43305e12, 0.0)
        )
        C = jnp.where(lamb_inv > 4860.0, 14.4e9, 0.0)

        # Vectorized calculation of nm1 and fk (with condition applied to nm1)
        nm1 = jnp.where(lamb_inv > 4860.0, 1e-8 * (A + B / (C - lamb_inv**2)), 0.0)
        fk = 1.034 + 3.17e-12 * lamb_inv**2

        # Calculate Rayleigh scattering opacities (with broadcasting for layers)
        new_opacities = opacities.at[:, i_str].set(
            opacities[:, i_str]
            + 24.0 * jnp.pi**3 * lamb_inv**4 / (2.546899e19)**2
            * (((nm1 + 1.0)**2 - 1.0) / ((nm1 + 1.0)**2 + 2.0))**2 * fk / mass_n2
            * mass_fractions[i_str]
        )
        return new_opacities

    # Use lax.fori_loop for JIT-compatible loop
    rayleigh_scattering_opacities = lax.fori_loop(
        0,
        mass_fractions.size,
        lambda i_str, opacities: lax.cond(
            mass_fractions[i_str] > ABUNDANCE_THRESHOLD,
            compute_opacity,
            lambda i_str, opacities: opacities,  # No-op if condition fails
            i_str,
            opacities,
        ),
        rayleigh_scattering_opacities,
    )

    return rayleigh_scattering_opacities

@jit
def co_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    """Calculates CO Rayleigh scattering opacities (JIT-compatible)."""
    d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    mass_co = 28.0 * cst.amu

    # Preallocate output array (required for JIT)
    rayleigh_scattering_opacities = jnp.zeros((wavelengths_angstroem.size, mass_fractions.size))

    # Use lax.cond for conditional execution within JIT
    def compute_opacity(i_str, opacities):
        scale = d[i_str] / 28.0 / cst.amu / cst.standard_number_density_for_refractive_index
        nfr_co = d[i_str] / 28.0 / cst.amu

        # Use jnp.where for conditional adjustment of lamb_inv
        lamb_inv_use = jnp.where(lambda_cm / 1e-4 < 0.168, 1.0 / 0.168e-4, lamb_inv)

        # Vectorized calculation of nm1 and fk
        nm1 = (22851.0 + 0.456e12 / (71427.0**2 - lamb_inv_use**2)) * 1e-8 * scale
        fk = 1.016

        # Calculate Rayleigh scattering opacities, broadcasting for layers
        new_opacities = opacities.at[:, i_str].set(
            opacities[:, i_str]
            + 24.0 * jnp.pi**3 * lamb_inv**4 / nfr_co**2
            * (((nm1 + 1.0)**2 - 1.0) / ((nm1 + 1.0)**2 + 2.0))**2 * fk / mass_co
            * mass_fractions[i_str]
        )

        return new_opacities

    # Use lax.fori_loop for JIT-compatible loop
    rayleigh_scattering_opacities = lax.fori_loop(
        0,
        mass_fractions.size,
        lambda i_str, opacities: lax.cond(
            mass_fractions[i_str] > ABUNDANCE_THRESHOLD,
            compute_opacity,
            lambda i_str, opacities: opacities,  # No-op if condition fails
            i_str,
            opacities,
        ),
        rayleigh_scattering_opacities,
    )

    return rayleigh_scattering_opacities

@jit
def ch4_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures):
    """Calculates CH4 Rayleigh scattering opacities (JIT-compatible)."""

    d = mean_molar_masses * cst.amu * pressures / cst.kB / temperatures * mass_fractions
    lambda_cm = wavelengths_angstroem * 1e-8
    lamb_inv = 1.0 / lambda_cm
    mass_ch4 = 16.0 * cst.amu

    # Preallocate output array (required for JIT)
    rayleigh_scattering_opacities = jnp.zeros((wavelengths_angstroem.size, mass_fractions.size))

    # Use lax.cond for conditional execution within JIT
    def compute_opacity(i_str, opacities):
        scale = d[i_str] / 16.0 / cst.amu / cst.standard_number_density_for_refractive_index
        nfr_ch4 = d[i_str] / 16.0 / cst.amu

        # Vectorized calculation of nm1 and fk
        nm1 = (46662.0 + 4.02e-6 * lamb_inv**2) * 1e-8 * scale
        fk = 1.0

        # Calculate Rayleigh scattering opacities, broadcasting for layers
        new_opacities = opacities.at[:, i_str].set(
            opacities[:, i_str]
            + 24.0 * jnp.pi**3 * lamb_inv**4 / nfr_ch4**2
            * (((nm1 + 1.0)**2 - 1.0) / ((nm1 + 1.0)**2 + 2.0))**2 * fk / mass_ch4
            * mass_fractions[i_str]
        )
        return new_opacities

    # Use lax.fori_loop for JIT-compatible loop
    rayleigh_scattering_opacities = lax.fori_loop(
        0,
        mass_fractions.size,
        lambda i_str, opacities: lax.cond(
            mass_fractions[i_str] > ABUNDANCE_THRESHOLD,
            compute_opacity,
            lambda i_str, opacities: opacities,  # No-op if condition fails
            i_str,
            opacities,
        ),
        rayleigh_scattering_opacities,
    )

    return rayleigh_scattering_opacities

def compute_rayleigh_scattering(species, mass_fractions, wavelengths_angstroem,
                                          mean_molar_masses, temperatures, pressures):
    if species == 'H2':
        return H2_rayleigh(wavelengths_angstroem,mass_fractions)
    elif species == 'He':
        return He_rayleigh(wavelengths_angstroem,mass_fractions)


    if species == 'H2O':
        rayleigh_scattering_opacities = h2o_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)
    elif species == 'CO2':
        rayleigh_scattering_opacities = co2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)
    elif species == 'O2':
       rayleigh_scattering_opacities = o2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)
    elif species == 'N2':
       rayleigh_scattering_opacities = n2_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)
    elif species == 'CO':
        rayleigh_scattering_opacities = co_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)
    elif species == 'CH4':
        rayleigh_scattering_opacities = ch4_rayleigh(wavelengths_angstroem, mass_fractions, mean_molar_masses, pressures, temperatures)

    rayleigh_scattering_opacities = jnp.where(
            mass_fractions[None, :] < ABUNDANCE_THRESHOLD,
            0.0,
            rayleigh_scattering_opacities
        )
    return rayleigh_scattering_opacities