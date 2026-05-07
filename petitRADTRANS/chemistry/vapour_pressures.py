"""This module provides vapor pressure curves for various condensate species."""

import jax.numpy as jnp
from petitRADTRANS.jax_dtype import JAX_DTYPE


# ============================================================================
# Condensation vapor pressure curves
# ============================================================================


def vapour_pressure_fe_solid(temperature):
    return jnp.exp(15.71 - 47664.0 / temperature)


def vapour_pressure_fe_liquid(temperature):
    return jnp.exp(9.86 - 37120.0 / temperature)


def vapour_pressure_mgsio3(temperature):
    return jnp.exp(25.37 - 58663.0 / temperature)


def vapour_pressure_mg2sio4(temperature, metallicity=None):
    """Total pressure of condensation, not the species' vapor pressure like for most of the other species."""
    if metallicity is None:
        return jnp.exp(25.37 - 58663.0 / temperature)
    return 10 ** (15.92 - 1.97 * metallicity - 27027.03 / temperature)


def vapour_pressure_na2s(
    temperature, metallicity=0.0, na2s_mass_fraction=None, mmw=2.33
):
    if na2s_mass_fraction is not None:
        # We're also using [Na/H] as a proxy for [Fe/H]
        # Definitely not strictly correct, but should be good enough for ~ solar compositions.
        # [+- 1 for Fe/H]
        # Assumes constant vertical abundance
        return (
            10
            ** (
                8.55
                - 13889.0 / temperature
                - 0.5
                * (jnp.log10(2 * na2s_mass_fraction * mmw / na2s_mass_fraction) + 5.7)
            )
            / 2
        )
    return 10 ** (8.55 - 13889.0 / temperature - 0.5 * metallicity) / 2.0


def vapour_pressure_kcl(temperature):
    """
    Notes
    -----
    .. [1] Morley, C.~V., Fortney, et a.  2012.
           Neglected Clouds in T and Y Dwarf Atmospheres.
           The Astrophysical Journal 756. doi:10.1088/0004-637X/756/2/172
    .. [2] Lodders, K. 1999.
           Alkali Element Chemistry in Cool Dwarf Atmospheres.
           The Astrophysical Journal 519, 793–801. doi:10.1086/307387
    """
    return 10 ** (7.6106 - 11382.0 / temperature)


def vapour_pressure_sio(temperature):
    return jnp.exp(33.848 - 42648.0 / temperature) * 1e-6


def vapour_pressure_tio2(temp, metallicity=0):
    """Compute vapor pressure curve for TiO2.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Marley M. S., Saumon D., Visscher C., Lupu R., Freedman R.,
            Morley C., Fortney J. J., et al., 2021, ApJ, 920, 85.
            doi:10.3847/1538-4357/ac141d
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # return 1e6 * 10. ** (9.5489 - 32456.8678/temp)  # Gao 2020
    return 1e6 * 10.0 ** (13.95 - 38266 / temp - metallicity)  # Marley et al. 2021


def vapour_pressure_cr(temp, metallicity=0):
    """Compute vapor pressure curve for Cr.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf
            atmospheres." The Astrophysical Journal 756.2 (2012): 172.
    """
    # Cr vapor pressure above cloud
    pvap_cr_bars = 10.0 ** (7.49 - 20592.0 / temp)
    # Then convert from bars to dynes/cm^2
    pvap_cr = pvap_cr_bars * 1e6
    return pvap_cr


def vapour_pressure_zns(temp, metallicity=0):
    """Compute vapor pressure curve for ZnS.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf
            atmospheres." The Astrophysical Journal 756.2 (2012): 172.
        [2] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr.
            "Atmospheric chemistry in giant planets, brown dwarfs, and
            low-mass dwarf stars. II. Sulfur and phosphorus."
            The Astrophysical Journal 648.2 (2006): 1181.
        [3] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # Zn vapor pressure above cloud
    pvap_zns_bars = 10.0 ** (12.8117 - 15873.0 / temp - metallicity)
    # Then convert from bars to dynes/cm^2
    pvap_zns = pvap_zns_bars * 1e6
    return pvap_zns


def vapour_pressure_nh3(temp, metallicity=0):
    """Compute vapor pressure curve for NH3.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's
            companion / Katharina Lodders, Bruce Fegley. New York:
            Oxford University Press, 1998. QB601 .L84 1998
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # NH3 vapor pressure above cloud
    is_scalar = isinstance(temp, float)
    if is_scalar:
        temp = jnp.array([temp])

    # Use JAX-compatible operations
    low_temp_vp = 10 ** (6.900 - 1588 / temp)
    high_temp_vp = 10 ** (5.201 - 1248 / temp)

    # Combine based on temperature threshold
    pvap_nh3 = jnp.where(temp < 195.4, low_temp_vp, high_temp_vp)

    # Convert from bars to dyne/cm^2
    pvap_nh3 = pvap_nh3 * 1e6

    if is_scalar:
        pvap_nh3 = pvap_nh3[0]

    return pvap_nh3


def vapour_pressure_mns(temp, metallicity=0):
    """Compute vapor pressure curve for MnS.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Morley, Caroline V., et al. "Neglected clouds in T and Y dwarf
            atmospheres." The Astrophysical Journal 756.2 (2012): 172.
        [2] Visscher, Channon, Katharina Lodders, and Bruce Fegley Jr.
            "Atmospheric chemistry in giant planets, brown dwarfs, and
            low-mass dwarf stars. II. Sulfur and phosphorus."
            The Astrophysical Journal 648.2 (2006): 1181.
        [3] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # Mn vapor pressure above cloud
    pvap_mns_bars = 10.0 ** (11.5315 - 23810.0 / temp - metallicity)
    # Then convert from bars to dynes/cm^2
    pvap_mns = pvap_mns_bars * 1e6
    return pvap_mns


def vapour_pressure_h2o(temp, do_buck=True, metallicity=0):
    """Compute vapor pressure curve for H2O.

    Args:
        temp: float or ndarray
            Temperature (K)
        do_buck: bool
            If True, use Buck (1981) expression. If False, use Wexler's.
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's
            companion / Katharina Lodders, Bruce Fegley. New York:
            Oxford University Press, 1998. QB601 .L84 1998
        [2] Buck, Arden L. "New equations for computing vapor pressure and
            enhancement factor." Journal of Applied Meteorology and
            Climatology 20.12 (1981): 1527-1532.
        [3] Flatau, Piotr J., Robert L. Walko, and William R. Cotton.
            "Polynomial fits to saturation vapor pressure." Journal of
            Applied Meteorology 31.12 (1992): 1507-1513.
        [4] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    is_scalar = isinstance(temp, float)
    if is_scalar:
        temp = jnp.array([temp])

    # Convert to array if needed
    temp = jnp.asarray(temp)

    # Define constants used in Buck's expressions
    # Buck, 1981 (J. Atmos. Sci., 20, p. 1527)
    BAL = 6.1121e3
    BBL = 18.729
    BCL = 257.87
    BDL = 227.3
    BAI = 6.1115e3
    BBI = 23.036
    BCI = 279.82
    BDI = 333.7

    # Define constants used in Wexler formulas
    # (see Flatau et al., 1992, J. Appl. Meteor. p. 1507)
    GG0 = -0.29912729e4
    GG1 = -0.60170128e4
    GG2 = 0.1887643854e2
    GG3 = -0.28354721e-1
    GG4 = 0.17838301e-4
    GG5 = -0.84150417e-9
    GG6 = 0.44412543e-12
    GG7 = 0.28584870e1

    HH0 = -0.58653696e4
    HH1 = 0.2224103300e2
    HH2 = 0.13749042e-1
    HH3 = -0.34031775e-4
    HH4 = 0.26967687e-7
    HH5 = 0.6918651

    # Initialize output array
    pvap_h2o = jnp.zeros_like(temp)

    # Ice (T < 273.16 K)
    if do_buck:
        tc_low = temp - 273.16
        vap_low = BAI * jnp.exp((BBI - tc_low / BDI) * tc_low / (tc_low + BCI))
    else:
        vap_low = 10 * jnp.exp(
            1.0
            / temp
            * (
                HH0
                + (HH1 + HH5 * jnp.log(temp) + (HH2 + (HH3 + HH4 * temp) * temp) * temp)
                * temp
            )
        )

    # Liquid water (273.16 K <= T < 1048 K)
    if do_buck:
        tc_mid = temp - 273.16
        vap_mid = BAL * jnp.exp((BBL - tc_mid / BDL) * tc_mid / (tc_mid + BCL))
    else:
        vap_mid = 10 * jnp.exp(
            (1.0 / (temp * temp))
            * (
                GG0
                + (
                    GG1
                    + (
                        GG2
                        + GG7 * jnp.log(temp)
                        + (GG3 + (GG4 + (GG5 + GG6 * temp) * temp) * temp) * temp
                    )
                    * temp
                )
                * temp
            )
        )

    # High temperature (T >= 1048 K) - fixed at 600 bars
    vap_high = 600.0e6

    # Combine based on temperature ranges using jnp.where
    pvap_h2o = jnp.where(
        temp < 273.16, vap_low, jnp.where(temp < 1048, vap_mid, vap_high)
    )

    if is_scalar:
        pvap_h2o = pvap_h2o[0]

    return pvap_h2o


def vapour_pressure_ch4(temp, metallicity=0):
    """Compute vapor pressure curve for CH4.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Lodders, K. & Fegley, B. 1998, The planetary scientist's
            companion / Katharina Lodders, Bruce Fegley. New York:
            Oxford University Press, 1998. QB601 .L84 1998
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # Constants
    AMR = 16.043 / 8.3143
    TCRIT = 90.68
    PCRIT = 0.11719
    AS = 2.213 - 2.650  # specific heat (gas - solid)
    AL = 2.213 - 3.370  # specific heat (gas - liquid)
    ALS = 611.10  # latent heat of sublimation
    ALV = 552.36  # latent heat of vaporization

    is_scalar = isinstance(temp, float)
    temp_arr = jnp.atleast_1d(jnp.asarray(temp, dtype=JAX_DTYPE))

    # Prepare coefficients for both phases
    C_solid = -AMR * AS
    C_liquid = -AMR * AL
    B_solid = -AMR * (ALS + AS * TCRIT)
    B_liquid = -AMR * (ALV + AL * TCRIT)
    A_solid = PCRIT * TCRIT ** (-C_solid) * jnp.exp(-B_solid / TCRIT)
    A_liquid = PCRIT * TCRIT ** (-C_liquid) * jnp.exp(-B_liquid / TCRIT)

    # ic=0: temperature below triple point (solid), ic=1: temperature above triple point (liquid)
    # Calculate vapor pressure for both phases and select based on temperature
    pvap_solid = A_solid * temp_arr**C_solid * jnp.exp(B_solid / temp_arr)
    pvap_liquid = A_liquid * temp_arr**C_liquid * jnp.exp(B_liquid / temp_arr)

    # Use liquid phase for T > TCRIT, solid phase otherwise
    pvap_ch4 = jnp.where(temp_arr > TCRIT, pvap_liquid, pvap_solid)

    # Convert from bars to dyne/cm^2
    pvap_ch4 = pvap_ch4 * 1e6

    if is_scalar:
        pvap_ch4 = pvap_ch4[0]

    return pvap_ch4


def vapour_pressure_al2o3(temp, metallicity=0):
    """Compute vapor pressure curve for Al2O3.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Wakeford, Hannah R., et al. "High temperature condensate clouds
            in super-hot Jupiter atmospheres." Monthly Notices of the Royal
            Astronomical Society (2016): stw2639.
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # return jnp.exp(-73503./temp + 22.01)*1e6  # Kozasa et al. Ap J. 344 325
    # calculated from wakeford 2017
    # pvap_al2o3 = 1e6 * 10.0 ** (17.7 - 45892.6/temp - 1.66*metallicity)  # wakeford et al 2017
    pvap_al2o3 = 1e6 * 10.0 ** (
        15.24 - 41481 / temp - 1.66 * metallicity
    )  # diamondback
    return pvap_al2o3


def vapour_pressure_catio3(temp, p, metallicity=0):
    """Compute vapor pressure curve for CaTiO3.

    Args:
        temp: float or ndarray
            Temperature (K)
        p: float
            Pressure (dyne/cm^2)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Wakeford, Hannah R., et al. "High temperature condensate clouds
            in super-hot Jupiter atmospheres." Monthly Notices of the Royal
            Astronomical Society (2016): stw2639.
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # calculated from wakeford 2017
    pvap_catio3 = 1e6 * 10.0 ** (
        -72160.0 / temp + 30.24 - 1 * jnp.log10(p / 1e6) - 2 * metallicity
    )
    return pvap_catio3


def vapour_pressure_caal12o19(temp, p, metallicity=0):
    """Compute vapor pressure curve for CaAl12O19.

    Args:
        temp: float or ndarray
            Temperature (K)
        p: float
            Pressure (dyne/cm^2)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2

    References:
        [1] Wakeford, Hannah R., et al. "High temperature condensate clouds
            in super-hot Jupiter atmospheres." Monthly Notices of the Royal
            Astronomical Society (2016): stw2639.
        [2] Virga: https://github.com/natashabatalha/virga/
            Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    """
    # calculated from wakeford 2017
    pvap_caal12o19 = 1e6 * 10.0 ** (
        16.46 - 44021.0 / temp - 0.083 * jnp.log10(p / 1e6) - 1.67 * metallicity
    )
    return pvap_caal12o19


def vapour_pressure_sio2(temp, metallicity=0):
    """Compute vapor pressure curve for SiO2.

    Args:
        temp: float or ndarray
            Temperature (K)
        metallicity: float
            Metallicity relative to solar in log10 units (default: 0 = 1X solar)

    Returns:
        float or ndarray
            Vapor pressure in dyne/cm^2
    References:
        [1] Virga: https://github.com/natashabatalha/virga/
                Moran et al. 2025 https://ui.adsabs.harvard.edu/abs/2025ApJ...994..116M/abstract
    Notes:
        PLACEHOLDER: Not suited yet for public use.
    """
    # V1 undepleted gas source
    pvap_sio2 = 10.0 ** (13.168 - 28265 / temp - metallicity)

    # V1 depleted gas source
    # 13.360 - 28265/T(K) - mh

    # Convert bars -> dynes/cm^2
    pvap_sio2 = 1e6 * pvap_sio2
    return pvap_sio2
