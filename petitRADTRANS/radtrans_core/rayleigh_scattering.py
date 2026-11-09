"""Rayleigh scattering opacities, in cm^2 per gram of atmospheric mixture.

Wavelengths are in Angstrom, temperatures in K and abundances are mass fractions.
Every kernel returns shape (n_wavelengths, n_layers) and is differentiable and finite
in float32 and float64, including at zero abundance.

Each species has one continuous treatment, with no switching between models at
wavelength or temperature boundaries:

* H2: Singor et al. (2024), elastic + Raman, calibrated to Peck & Huang (1977)
* He: Puchalski et al. (2016), Table VII
* H2O: Egan (2026), Eq. 14/Table 5, temperature slope of Egan & Yang (2024)
* CH4: He et al. (2021), Eq. 10
* O2: Bates (1984), as tabulated by Sneep & Ubachs (2005) and Thalman et al. (2014)
* N2: Peck & Khanna (1966) and Bates (1984), via Sneep & Ubachs (2005)
* CO2: Bideau-Mehu et al. (1973), via Sneep & Ubachs (2005)
* CO: Smith et al. (1976), via Sneep & Ubachs (2005)

Corrections to earlier versions of this module
----------------------------------------------
Errors in the published equations, which the code had copied:

* CO, Sneep & Ubachs (2005) Eq. 17 prints the dispersion coefficient as 0.456e12. Their
  own Table 2 (n-1 = 325e-6 and sigma = 6.82e-27 cm^2 at 18788.4 cm^-1) requires 0.456e14,
  which also matches the Smith et al. (1976) data to better than 1%. The printed value
  gave half the cross section at 532 nm, below that of N2.
* CO2, Sneep & Ubachs (2005) Eq. 13 prints the fifth coefficient as 0.1218145e-4. Their
  Table 2 (n-1 = 427e-6 and sigma = 13.29e-27 cm^2) and the Bideau-Mehu formula
  require 0.1218145. The printed prefactor 1.1427e6 is also 1e3 too large.

Physics:

* He: the polynomial attributed to Chan & Dalgarno (1965) fell to 1.12 a.u. near 0.6 um,
  giving up to 36% too little opacity. The polarizability is now 1.38--1.42 a.u. in the
  optical, matching Mansfield & Peck (1969) to 0.35% in cross section.
* CH4: the Sneep & Ubachs (2005) Eq. 18 cross section, 14.69e-27 cm^2 at 532 nm, exceeds even
  their measured total extinction of 12.47(0.23)e-27 cm^2. He et al. (2021) gives
  11.35e-27 cm^2. Measured extinction also includes CH4 absorption, so it is an upper limit.
* H2O: the IAPWS R9-97 fit gave 3.4% too much opacity against the Ciddor (1996) vapour
  refractivity, and fell by 30% between 300 and 3000 K, far outside its 261--773 K range.
  The measured gas-phase slope is +1.5e-6/K, of the opposite sign and 40 times smaller.
* H2: Dalgarno & Williams (1962) has no temperature dependence, and is 3.5% below the
  cross section implied by the measured Peck & Huang (1977) refractivity before any King
  factor is applied. See ``H2_rayleigh``.
* N2: the molecular mass was 34 amu instead of 28 amu. Scattering was set to zero beyond
  2.059 um, and now continues the long-wavelength branch as a smooth extrapolation.
* CO2: a positive floor on the dispersion denominators made the optical cross section 17%
  too large.
* Density scaling: CO2, CO and CH4 scaled n-1 linearly with local density. Sneep & Ubachs
  (2005) Eqs. 2--4 show that the cross section is independent of density, so this was a
  spurious pressure dependence of -1.5% at 100 bar in pure CO2. It is removed.

Numerics:

* Float32 range: units and molecular masses are combined before squaring tiny molecular
  volumes, which previously underflowed.
* Cancellation: n^2 - 1 is factored as (n - 1) * (2 + (n - 1)) rather than evaluated directly.
* Zero abundance: no abundance mask remains. Multiplying by the mass fraction gives exactly
  zero at X = 0 while preserving the derivative there.
* Poles: dispersion formulae are held at the edge of their fit range rather than evaluated
  at their poles, which reached 40 times the local value at R = 1e6 for CO2 at 4.135 um,
  and gave NaN in float32. Where a refractivity is held, opacity keeps its lambda^-4 trend.

Not adopted:

* O2 from Egan (2026). It changes the optical cross section by 0.5% or less, and the
  retained treatment already agrees with measurement to 0.2 +/- 0.4% (Thalman et al. 2014).

Known limitations
-----------------
These are extrapolated beyond their measurements: H2O above 433 K and outside 0.3--0.9 and
1.5--1.6 um, CH4 outside 0.264--0.671 um, CO above 0.288 um, CO2 above 1.69 um and N2 above
2.059 um. Sneep & Ubachs (2005) measure CO 10% below the n-based value at 532 nm, unexplained.
The refractivities are not absorption or resonant scattering models, and all treatments are
for dilute gases. Scattering is treated as elastic, whereas Raman photons change wavelength.

Sources
-------
* Singor et al. (2024), doi:10.1063/5.0235703, data doi:10.5281/zenodo.13441472
* Peck & Huang (1977), doi:10.1364/JOSA.67.001550
* Puchalski et al. (2016), doi:10.1103/PhysRevA.93.032515
* Egan (2026), doi:10.1088/1361-6501/ae6e94
* Egan & Yang (2024), doi:10.1007/s10765-024-03380-w
* He et al. (2021), doi:10.5194/acp-21-14927-2021
* Sneep & Ubachs (2005), doi:10.1016/j.jqsrt.2004.07.025
* Thalman et al. (2014), doi:10.1016/j.jqsrt.2014.05.030
"""

from functools import lru_cache

import h5py
import jax.numpy as jnp
from jax import jit
import numpy as np

from petitRADTRANS._input_data import find_input_file
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
import petitRADTRANS.physical_constants as cst

# Reference density at 288.15 K and 1013.25 hPa (Sneep & Ubachs 2005).
_N_REF = cst.standard_number_density_for_refractive_index

HC_EV_UM = 1.2398419843320026
KB_EV_K = 8.617333262145179e-5

# Peck & Huang (1977) at 273.15 K over the isotropic part of the Singor et al. (2024) thermal
# elastic cross section, using their rotational Raman to remove the anisotropic part. The
# ratio is flat to within 4e-4 over 0.3--1.2 um, so the table's shape is retained and only
# its scale changes. Applied to all channels.
_H2_PECK_HUANG_CALIBRATION = 1.02876
_H2_RANK = 12
_H2_TEMPERATURE_NODES = 128
_H2_MINIMUM_TEMPERATURE, _H2_MAXIMUM_TEMPERATURE = 1.0, 10000.0
_H2_TABLE_FILE = "1H2__Singor.DeltaWavenumber363_0.207-27.5mu.rayleightable.petitRADTRANS.h5"


def _opacity_from_lorentz_lorenz_volume(
    wavelengths_um, volume_1e23, king_factor, molecule_mass, mass_fractions
):
    """Opacity from 1e23 * (n^2 - 1) / ((n^2 + 2) * N), in cm^3.

    Scaling the molecular volume before squaring, and combining the powers of
    ten with the molecular mass, avoids float32 underflow/overflow. The result
    is per gram of atmospheric mixture, with shape (wavelength, layer).
    """
    prefactor = 24.0 * jnp.pi**3 * (1e-30 / molecule_mass)
    opacity = prefactor * (volume_1e23 / wavelengths_um**2)**2 * king_factor
    if opacity.ndim == 1:
        opacity = opacity[:, None]
    return opacity * mass_fractions


def _rayleigh_opacity(lamb_inv, nm1, number_density, king_factor, molecule_mass,
                      mass_fractions):
    """Rayleigh opacity from a refractivity at a fixed reference number density.

    ``nm1`` is n - 1. Factoring n^2 - 1 avoids cancellation near unity.
    """
    delta = nm1 * (2.0 + nm1)
    volume_1e23 = (1e4 / (number_density / 1e19)) * delta / (3.0 + delta)
    return _opacity_from_lorentz_lorenz_volume(
        1e4 / lamb_inv, volume_1e23, king_factor, molecule_mass, mass_fractions
    )


def _h2_thermal_coefficients(states, statistical_weights, excitation_ev, rank_weights, log_temperatures):
    """Rank coefficients of the populations of ``states``, and their derivatives in log T."""
    boltzmann = excitation_ev[states, None] / (KB_EV_K * np.exp(log_temperatures)[None, :])
    log_populations = np.log(statistical_weights[states])[:, None] - boltzmann
    populations = np.exp(log_populations - log_populations.max(axis=0))
    populations /= populations.sum(axis=0)
    mean_boltzmann = (populations * boltzmann).sum(axis=0)
    return (rank_weights[states].T @ populations,
            rank_weights[states].T @ (populations * (boltzmann - mean_boltzmann)))


def _get_h2_table_file(path_input_data):
    """Path of the H2 table in the input_data directory, which is downloaded if it is missing."""
    return find_input_file(
        file=_H2_TABLE_FILE,
        path_input_data=path_input_data,
        sub_path=get_input_data_subpaths()["rayleigh_opacities"],
    )


@lru_cache(maxsize=None)
def _h2_data(path_input_data):
    """Load the table of an input_data directory once, as NumPy arrays, and reduce it to rank ``_H2_RANK``.

    The thermal sum over 293 states is a low-rank problem. Decomposing the table, and tabulating
    the thermal coefficients in temperature, is 5 times faster than summing all states at
    runtime, with a largest error of 3e-4 in the thermal cross section.
    """
    with h5py.File(_get_h2_table_file(path_input_data), "r") as table:
        wavenumbers = table["wavenumbers"][:]
        rotational_n = table["rotational_quantum_numbers"][:]
        excitation_ev = table["excitation_energies"][:]
        elastic = table["elastic_reduced_cross_sections"][:]
        raman = table["raman_reduced_cross_sections"][:]
    total = _H2_PECK_HUANG_CALIBRATION * (elastic + raman)
    left, singular_values, right = np.linalg.svd(total, full_matrices=False)
    rank_weights = left[:, :_H2_RANK] * singular_values[:_H2_RANK]
    log_temperatures = np.linspace(
        np.log(_H2_MINIMUM_TEMPERATURE), np.log(_H2_MAXIMUM_TEMPERATURE), _H2_TEMPERATURE_NODES
    )
    # Nuclear-spin weights are 1 for even and 3 for odd rotational states.
    odd = rotational_n % 2 == 1
    weights = (2.0 * rotational_n + 1.0) * np.where(odd, 3.0, 1.0)
    return dict(
        energy2=(wavenumbers * HC_EV_UM * 1e-4)**2,
        basis=right[:_H2_RANK],
        log_temperatures=log_temperatures,
        thermal=_h2_thermal_coefficients(np.arange(odd.size), weights, excitation_ev, rank_weights, log_temperatures),
        ortho=_h2_thermal_coefficients(np.flatnonzero(odd), weights, excitation_ev, rank_weights, log_temperatures),
        para=_h2_thermal_coefficients(np.flatnonzero(~odd), weights, excitation_ev, rank_weights, log_temperatures),
    )


@jit
def _h2_opacity(wavelengths_angstroem, mass_fractions, temperatures, energy2_grid, basis,
                log_temperatures, coefficients, coefficient_derivatives):
    energy2 = (HC_EV_UM * 1e4 / wavelengths_angstroem)**2
    # The reduced cross section sigma/E^4 is smooth and finite at E=0. Interpolate in E^2.
    index = jnp.clip(jnp.searchsorted(energy2_grid, energy2) - 1, 0, energy2_grid.size - 2)
    fraction = jnp.clip(
        (energy2 - energy2_grid[index]) / (energy2_grid[index + 1] - energy2_grid[index]), 0.0, 1.0
    )
    reduced = (basis[:, index] * (1.0 - fraction) + basis[:, index + 1] * fraction).T

    # The table stops below the UV resonances. Continue it with the Dalgarno & Williams shape.
    def dalgarno_williams_shape(energy2_):
        inv2 = energy2_ / HC_EV_UM**2
        return 8.14 + inv2 * (0.128 + inv2 * 0.00161)
    energy2_maximum = energy2_grid[-1]
    continuation = (dalgarno_williams_shape(jnp.maximum(energy2, energy2_maximum))
                    / dalgarno_williams_shape(energy2_maximum))

    # Cubic Hermite interpolation in log T, which keeps temperature derivatives continuous.
    log_t = jnp.log(jnp.clip(temperatures, _H2_MINIMUM_TEMPERATURE, _H2_MAXIMUM_TEMPERATURE))
    step = log_temperatures[1] - log_temperatures[0]
    node = jnp.clip(((log_t - log_temperatures[0]) / step).astype(jnp.int32), 0, log_temperatures.size - 2)
    u = (log_t - log_temperatures[node]) / step
    thermal = (
        (2.0 * u**3 - 3.0 * u**2 + 1.0) * coefficients[:, node]
        + (u**3 - 2.0 * u**2 + u) * step * coefficient_derivatives[:, node]
        + (-2.0 * u**3 + 3.0 * u**2) * coefficients[:, node + 1]
        + (u**3 - u**2) * step * coefficient_derivatives[:, node + 1]
    )
    factor = 1e-18 / (2.0 * cst.amu)  # Mb to cm^2, then per gram of H2
    return factor * (energy2**2 * continuation)[:, None] * (reduced @ thermal) * mass_fractions


def load_h2_rayleigh_table(path_input_data=None):
    """Load the H2 table of an input_data directory, by default the configured one, into the cache."""
    if path_input_data is None:
        path_input_data = petitradtrans_config_parser.get_input_data_path()
    return _h2_data(path_input_data)


def H2_rayleigh(wavelengths_angstroem, mass_fractions, temperatures, ortho_fraction=None,
                path_input_data=None):
    """H2 opacity, elastic + Raman, Singor et al. (2024), calibrated to Peck & Huang (1977).

    Populations are in LTE over all 293 bound states, with the nuclear-spin weights. A supplied
    ``ortho_fraction`` (e.g. 0.75) fixes that fraction instead, with thermal populations within
    each spin species. This does not calculate chemical dissociation.

    Raman is included because it removes photons from the beam, which is also how the King
    factors of the other species are defined. It is 3.0% of elastic at 500 nm and 300 K, and
    4.5% at 3000 K. Anti-Stokes channels are held below 1 eV, as their sigma/E^4 diverges
    towards E=0. The table covers E <= 5.99 eV (lambda >= 0.207 um), and temperatures
    are held outside 1--10000 K. It is read from ``opacities/continuum/rayleigh`` in the
    input_data directory.
    """
    table = load_h2_rayleigh_table(path_input_data)
    dtype = jnp.result_type(wavelengths_angstroem, mass_fractions, temperatures)
    if ortho_fraction is None:
        coefficients, derivatives = table["thermal"]
    else:
        coefficients = (ortho_fraction * table["ortho"][0] + (1.0 - ortho_fraction) * table["para"][0])
        derivatives = (ortho_fraction * table["ortho"][1] + (1.0 - ortho_fraction) * table["para"][1])
    return _h2_opacity(
        wavelengths_angstroem, mass_fractions, temperatures,
        jnp.asarray(table["energy2"], dtype=dtype), jnp.asarray(table["basis"], dtype=dtype),
        jnp.asarray(table["log_temperatures"], dtype=dtype),
        jnp.asarray(coefficients, dtype=dtype), jnp.asarray(derivatives, dtype=dtype),
    )


@jit
def helium_polarizability(wavelengths_angstroem):
    """4He polarizability in atomic units, Puchalski et al. (2016) Table VII through lambda^-8.

    Table VII gives contributions at 6329.908 A. These rounded contributions reproduce the
    reported reference polarizability, not the paper's full precision at arbitrary UV wavelengths.
    """
    q = (6329.908 / wavelengths_angstroem)**2
    return 1.38376077 + q * (0.00799595 + q * (0.000054840 + q * (0.000000408 + q * 3.2e-9)))


@jit
def He_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates He Rayleigh scattering opacities, holding the polarizability below 0.12 um."""
    alpha = helium_polarizability(jnp.maximum(wavelengths_angstroem, 1200.0))
    # a0^3 in units of 1e-23 cm^3; (n^2 - 1) / ((n^2 + 2) * N) = 4 * pi * alpha / 3.
    volume_1e23 = (4.0 * jnp.pi / 3.0) * 0.014818471147 * alpha
    return _opacity_from_lorentz_lorenz_volume(
        wavelengths_angstroem * 1e-4, volume_1e23, 1.0, 4.0 * cst.amu, mass_fractions
    )


@jit
def water_molar_refractivity(wavelengths_angstroem, temperatures):
    """Egan (2026) Eq. 14/Table 5 without its overtone poles, in cm^3/mol, shape (wavelength, layer).

    The 0.9--1.9 um overtone oscillators are real poles of the fit, contributing under 4e-4 of the
    cross section in the measured 0.3--0.9 and 1.5--1.6 um ranges, and are omitted. The remaining
    infrared poles lie beyond 2.6 um, so the refractivity is held beyond 2 um, and below 0.2 um.
    The reference is 293.15 K, with the 633 nm fractional temperature coefficient of Egan & Yang
    (2024), 1.5e-6/K, applied throughout. That coefficient is measured only up to 433 K.
    """
    strengths = jnp.array([
        2.945e-5, 6.583e-8, 3.273e-6, 3.094e-6, 1.862e-6, 2.788e-6, 0.26718980, 4.913464,
    ], dtype=wavelengths_angstroem.dtype)
    frequencies = jnp.array([
        0.02089, 0.06023, 0.14883, 0.17452, 0.36028, 0.38484, 7.86916, 28.84126,
    ], dtype=wavelengths_angstroem.dtype)
    nu_um2 = jnp.maximum((1e4 / jnp.maximum(wavelengths_angstroem, 2000.0))**2, 0.25)
    reference = 360.115 * jnp.sum(strengths / (frequencies**2 - nu_um2[:, None]), axis=1)
    thermal = (1.0 + 1.5e-6 * (temperatures - 303.0)) / (1.0 + 1.5e-6 * (293.15 - 303.0))
    return reference[:, None] * thermal[None, :]


@jit
def h2o_rayleigh(wavelengths_angstroem, mass_fractions, temperatures):
    """Calculates H2O Rayleigh scattering opacities, with a King factor of 1."""
    refractivity = water_molar_refractivity(wavelengths_angstroem, temperatures)
    return _opacity_from_lorentz_lorenz_volume(
        wavelengths_angstroem[:, None] * 1e-4, refractivity * (1e23 / cst.nA),
        1.0, 18.0 * cst.amu, mass_fractions,
    )


def _co2_refractivity(lamb_inv):
    """CO2 n-1 at 288.15 K and 1013.25 hPa, for an inverse wavelength in cm^-1, held outside 0.181--1.69 um.

    The fifth coefficient is printed as 0.1218145e-4 in Sneep & Ubachs (2005) Eq. 13, which does
    not reproduce their Table 2. Do not change it to the printed value.
    """
    lamb_inv_use = jnp.clip(lamb_inv, 1e4 / 1.69, 1e4 / 0.181)
    return 1142.7 * (
        5799.25 / (128908.9**2 - lamb_inv_use**2)
        + 120.05 / (89223.8**2 - lamb_inv_use**2)
        + 5.3334 / (75037.5**2 - lamb_inv_use**2)
        + 4.3244 / (67837.7**2 - lamb_inv_use**2)
        + 0.1218145 / (2418.136**2 - lamb_inv_use**2)
    )


@jit
def co2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates CO2 Rayleigh scattering opacities."""
    lamb_inv = 1e8 / wavelengths_angstroem
    fk = 1.1364 + 25.3e-12 * jnp.clip(lamb_inv, 1e4 / 1.69, 1e4 / 0.181)**2
    return _rayleigh_opacity(
        lamb_inv, _co2_refractivity(lamb_inv), _N_REF, fk, 44.0 * cst.amu, mass_fractions
    )


@jit
def o2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates O2 Rayleigh scattering opacities, holding n-1 and F_k short of the 0.156 um pole."""
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)
    lamb_inv_use = jnp.minimum(lamb_inv, 1e4 / 0.18)

    # Use jnp.where for conditional assignment of A and B
    A = jnp.where(lamb_inv_use > 18315.0, 20564.8, 21351.1)
    B = jnp.where(lamb_inv_use > 18315.0, 2.480899e13, 2.18567e13)
    C = 4.09e9

    nm1 = 1e-8 * (A + B / (C - lamb_inv_use**2))
    fk = 1.096 + 1.385e-11 * lamb_inv_use**2 + 1.448e-20 * lamb_inv_use**4

    return _rayleigh_opacity(
        lamb_inv, nm1, 2.68678e19, fk, 32.0 * cst.amu, mass_fractions
    )


@jit
def n2_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates N2 Rayleigh scattering opacities."""
    lamb_inv = 1.0 / (wavelengths_angstroem * 1e-8)

    # Continue the long-wavelength Peck--Khanna branch into the IR. This is
    # an extrapolation beyond 2.059 um, not a physical cutoff in scattering.
    A = jnp.where(lamb_inv > 21360.0, 5677.465, 6498.2)
    B = jnp.where(lamb_inv > 21360.0, 318.81874e12, 307.43305e12)
    nm1 = 1e-8 * (A + B / (14.4e9 - lamb_inv**2))
    fk = 1.034 + 3.17e-12 * lamb_inv**2
    return _rayleigh_opacity(
        lamb_inv, nm1, 2.546899e19, fk, 28.0 * cst.amu, mass_fractions
    )


def _co_refractivity(lamb_inv):
    """CO n-1 at 288.15 K and 1013.25 hPa, for an inverse wavelength in cm^-1, held below 0.168 um.

    The coefficient is printed as 0.456e12 in Sneep & Ubachs (2005) Eq. 17, which does not
    reproduce their Table 2. Do not change it to the printed value.
    """
    lamb_inv_use = jnp.minimum(lamb_inv, 1e4 / 0.168)
    return (22851.0 + 0.456e14 / (71427.0**2 - lamb_inv_use**2)) * 1e-8


@jit
def co_rayleigh(wavelengths_angstroem, mass_fractions):
    """Calculates CO Rayleigh scattering opacities."""
    lamb_inv = 1e8 / wavelengths_angstroem
    return _rayleigh_opacity(
        lamb_inv, _co_refractivity(lamb_inv), _N_REF, 1.016, 28.0 * cst.amu, mass_fractions
    )


@jit
def ch4_rayleigh(wavelengths_angstroem, mass_fractions):
    """He et al. (2021) Eq. 10, 288.15 K and 1013.25 hPa, measured over 0.264--0.671 um.

    The fit's pole is at 0.092 um, so it extends smoothly into the infrared, and
    the refractivity is held below 0.12 um.
    """
    lamb_inv = 1e8 / wavelengths_angstroem
    lamb_inv_use = jnp.minimum(lamb_inv, 1e4 / 0.12)
    nm1 = 1e-8 * (3603.09 + 4.40362e14 / (1.1741e10 - lamb_inv_use**2))
    return _rayleigh_opacity(
        lamb_inv, nm1, 2.546899e19, 1.0, 16.0 * cst.amu, mass_fractions
    )


# Species that need only the wavelength grid and the mass fractions.
_WAVELENGTH_ONLY_RAYLEIGH = {
    "He": He_rayleigh,
    "CH4": ch4_rayleigh,
    "O2": o2_rayleigh,
    "N2": n2_rayleigh,
    "CO2": co2_rayleigh,
    "CO": co_rayleigh,
}

def compute_rayleigh_scattering(
    species,
    mass_fractions,
    wavelengths_angstroem,
    mean_molar_masses,
    temperatures,
    pressures,
    path_input_data=None,
):
    """Rayleigh scattering opacities of one species, shape (n_wavelengths, n_layers).

    No treatment depends on density any more, so ``mean_molar_masses`` and ``pressures``
    are unused. They are retained so that the interface is unchanged. ``path_input_data``
    is the input_data directory holding the H2 table, by default the configured one.
    """
    del mean_molar_masses, pressures

    if species in _WAVELENGTH_ONLY_RAYLEIGH:
        rayleigh_scattering_opacities = _WAVELENGTH_ONLY_RAYLEIGH[species](
            wavelengths_angstroem, mass_fractions
        )
    elif species == "H2":
        rayleigh_scattering_opacities = H2_rayleigh(
            wavelengths_angstroem, mass_fractions, temperatures, path_input_data=path_input_data
        )
    elif species == "H2O":
        rayleigh_scattering_opacities = h2o_rayleigh(
            wavelengths_angstroem, mass_fractions, temperatures
        )
    else:
        raise ValueError(
            f"no Rayleigh scattering implementation for species '{species}'; "
            f"available species are "
            f"{sorted([*_WAVELENGTH_ONLY_RAYLEIGH, 'H2', 'H2O'])}"
        )

    # Every kernel has a finite zero-abundance limit. Multiplication by X
    # returns zero while preserving the physically meaningful derivative at X=0.
    return rayleigh_scattering_opacities
