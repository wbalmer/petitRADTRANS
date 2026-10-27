"""Stores useful physical functions."""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
import petitRADTRANS.physical_constants as cst


def calculate_distance_from_luminosity(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    mu_star = 0.0
    angle_use = False

    if (mode != "p") & (mode != "d"):
        mu_star = float(mode)
        angle_use = True

    if mode_what == "temp":
        if angle_use:
            t_irr = (
                (r_star * cst.r_sun / (dist * cst.au)) ** 2.0 * t_star**4.0 * mu_star
            ) ** 0.25
        elif mode == "p":
            t_irr = (
                (r_star * cst.r_sun / (dist * cst.au)) ** 2.0 * t_star**4.0 / 4.0
            ) ** 0.25
        else:
            t_irr = (
                (r_star * cst.r_sun / (dist * cst.au)) ** 2.0 * t_star**4.0 / 2.0
            ) ** 0.25
        return t_irr
    elif mode_what == "dist":
        if angle_use:
            dist = (
                jnp.sqrt(
                    (r_star * cst.r_sun) ** 2.0 * (t_star / t_irr) ** 4.0 * mu_star
                )
                / cst.au
            )
        elif mode == "p":
            dist = (
                jnp.sqrt((r_star * cst.r_sun) ** 2.0 * (t_star / t_irr) ** 4.0 / 4.0)
                / cst.au
            )
        else:
            dist = (
                jnp.sqrt((r_star * cst.r_sun) ** 2.0 * (t_star / t_irr) ** 4.0 / 2.0)
                / cst.au
            )
        return dist

    return None


def compute_effective_temperature(
    wavelengths: ArrayLike,
    flux: ArrayLike,
    orbit_semi_major_axis: float = 1.0,
    planet_radius: float = 1.0,
    use_si_units: bool = False,
) -> float:
    """Calculates the effective temperature by integrating the model and using the stefan boltzmann law.

    Args:
        wavelengths : numpy.ndarray
            Wavelength grid
        flux : numpy.ndarray
            Flux density grid
        orbit_semi_major_axis : Optional(float)
            Distance to the object. Must have same units as planet_radius
        planet_radius : Optional(float)
            Object radius. Must have same units as orbit_semi_major_axis
        use_si_units : Optional(bool)
            If the flux is in W/m2/micron, this should be true
    """

    def integrate_flux(_wavelengths, _flux):
        return jnp.sum(
            _flux[:-1]
            * ((orbit_semi_major_axis / planet_radius) ** 2.0)
            * jnp.diff(_wavelengths)
        )

    unit_factor = 1.0

    if use_si_units:
        unit_factor = 1e-3

    energy = integrate_flux(wavelengths, flux)

    return (energy / (cst.sigma * unit_factor)) ** 0.25


def doppler_shift(
    wavelength_0: [ArrayLike, float], velocity: [ArrayLike, float]
) -> [Array, float]:
    """Calculate the Doppler-shifted wavelength for electromagnetic waves.

    A negative velocity means that the source is going toward the observer. A positive velocity means the source is
    going away from the observer.

    Args:
        wavelength_0: (cm) wavelength of the wave in the referential of the source
        velocity: (cm.s-1) velocity of the source relative to the observer

    Returns:
        (cm) the wavelength of the source as measured by the observer
    """
    return wavelength_0 * jnp.sqrt((1 + velocity / cst.c) / (1 - velocity / cst.c))


def flux_cm2flux_hz(
    flux_cm: [ArrayLike, float], wavelength: [ArrayLike, float]
) -> [Array, float]:
    """
    Convert a flux from [flux units]/cm to [flux units]/Hz at a given wavelength.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_hz = flux_cm * d[cm]/d[Hz]
        => flux_hz = flux_cm * wavelength**2 / c

    Args:
        flux_cm: ([flux units]/cm)
        wavelength: (cm)

    Returns:
        ([flux units]/Hz) the radiosity in converted units
    """
    return flux_cm * wavelength**2 / cst.c


def flux_hz2flux_cm(
    flux_hz: [ArrayLike, float], frequency: [ArrayLike, float]
) -> [Array, float]:
    """Convert a flux from [flux units]/Hz to [flux units]/cm at a given frequency.
    Flux units can be, e.g., erg.s-1.cm-2.

    Steps:
        [cm] = c[cm.s-1] / [Hz]
        => d[cm]/d[Hz] = d(c / [Hz])/d[Hz]
        => d[cm]/d[Hz] = c / [Hz]**2
        => d[Hz]/d[cm] = [Hz]**2 / c
        integral of flux must be conserved: flux_cm * d[cm] = flux_hz * d[Hz]
        flux_cm = flux_hz * d[Hz]/d[cm]
        => flux_cm = flux_hz * frequency**2 / c

    Args:
        flux_hz: (erg.s-1.cm-2.sr-1/Hz)
        frequency: (Hz)

    Returns:
        (erg.s-1.cm-2.sr-1/cm) the radiosity in converted units
    """
    # Float32 safety: frequency**2 can overflow at optical/IR frequencies even
    # when the final flux conversion is well within range. Re-express the
    # Jacobian as c / wavelength**2 to keep intermediates near O(1e18).
    wavelength = cst.c / frequency
    return flux_hz * cst.c / wavelength**2


def flux2irradiance(
    flux: [ArrayLike, float], source_radius: float, target_distance: float
) -> [Array, float]:
    """Calculate the spectral irradiance of a spherical source on a target from its flux (spectral radiosity).

    Args:
        flux: (M.L-1.T-3) flux of the source
        source_radius: (L) radius of the spherical source
        target_distance: (L) distance from the source to the target

    Returns:
        The irradiance of the source on the target (M.L-1.T-3).
    """
    return flux * (source_radius / target_distance) ** 2


def frequency2wavelength(frequency: [ArrayLike, float]) -> [Array, float]:
    """Convert frequencies into wavelength in centimeter.

    Args:
        frequency: frequency: (Hz) the frequency to convert

    Returns:
        (cm) the corresponding wavelengths
    """
    return cst.c / frequency


def hz2um(frequency: [ArrayLike, float]) -> [Array, float]:
    """Convert frequencies into wavelengths in micrometer.

    Args:
        frequency: (Hz) the frequency to convert

    Returns:
        (um) the corresponding wavelengths
    """
    return frequency2wavelength(frequency) * 1e4  # cm to um


def ism_extinction(av_mag, rv_red, wavelengths):
    """
    Function for calculating the optical and IR extinction
    with the empirical relation from `Cardelli et al. (1989)
    <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_.
    Reimplemented from species.utils.dust_utils to avoid circular
    dependencies.

    Parameters
    ----------
    av_mag : float
        Extinction (mag) in the $V$ band.
    rv_red : float
        Reddening in the $V$ band, ``R_V = A_V / E(B-V)``.
    wavelengths : np.ndarray, list(float), float
        Array or list with the wavelengths (um) for which
        the extinction is calculated. It is also possible
        to provide a single value as float.

    Returns
    -------
    np.ndarray
        Extinction (mag) at ``wavelengths``.
    """

    wavelengths = jnp.array(wavelengths)

    x_wavel = 1.0 / wavelengths
    y_wavel = x_wavel - 1.82

    a_coeff = jnp.zeros(x_wavel.size)
    b_coeff = jnp.zeros(x_wavel.size)

    indices = jnp.where(x_wavel < 1.1)[0]

    if len(indices) > 0:
        # a_coeff[indices] = 0.574 * x_wavel[indices] ** 1.61
        a_coeff = a_coeff.at[indices].set(0.574 * x_wavel[indices] ** 1.61)
        # b_coeff[indices] = -0.527 * x_wavel[indices] ** 1.61
        b_coeff = b_coeff.at[indices].set(-0.527 * x_wavel[indices] ** 1.61)

    indices = jnp.where(x_wavel >= 1.1)[0]

    if len(indices) > 0:
        # a_coeff[indices] = (
        #     1.0
        #     + 0.17699 * y_wavel[indices]
        #     - 0.50447 * y_wavel[indices] ** 2
        #     - 0.02427 * y_wavel[indices] ** 3
        #     + 0.72085 * y_wavel[indices] ** 4
        #     + 0.01979 * y_wavel[indices] ** 5
        #     - 0.77530 * y_wavel[indices] ** 6
        #     + 0.32999 * y_wavel[indices] ** 7
        # )

        a_coeff = a_coeff.at[indices].set(
            1.0
            + 0.17699 * y_wavel[indices]
            - 0.50447 * y_wavel[indices] ** 2
            - 0.02427 * y_wavel[indices] ** 3
            + 0.72085 * y_wavel[indices] ** 4
            + 0.01979 * y_wavel[indices] ** 5
            - 0.77530 * y_wavel[indices] ** 6
            + 0.32999 * y_wavel[indices] ** 7
        )

        # b_coeff[indices] = (
        #     1.41338 * y_wavel[indices]
        #     + 2.28305 * y_wavel[indices] ** 2
        #     + 1.07233 * y_wavel[indices] ** 3
        #     - 5.38434 * y_wavel[indices] ** 4
        #     - 0.62251 * y_wavel[indices] ** 5
        #     + 5.30260 * y_wavel[indices] ** 6
        #     - 2.09002 * y_wavel[indices] ** 7
        # )
        b_coeff = b_coeff.at[indices].set(
            1.41338 * y_wavel[indices]
            + 2.28305 * y_wavel[indices] ** 2
            + 1.07233 * y_wavel[indices] ** 3
            - 5.38434 * y_wavel[indices] ** 4
            - 0.62251 * y_wavel[indices] ** 5
            + 5.30260 * y_wavel[indices] ** 6
            - 2.09002 * y_wavel[indices] ** 7
        )

    return av_mag * (a_coeff + b_coeff / rv_red)


def apply_ism_ext(wavelengths, flux, v_band_ext, v_band_red):
    """
    Function for applying ISM extinction to a spectrum.
    Reimplemented from species.utils.dust_utils to avoid circular
    dependencies.

    wavelengths : jnp.ndarray
        Wavelengths (um) of the spectrum.
    flux : jnp.ndarray
        Fluxes (W m-2 um-1) of the spectrum.
    v_band_ext : float
        Extinction (mag) in the $V$ band.
    v_band_red : float
        Reddening in the $V$ band.

    Returns
    -------
    jnp.ndarray
        Fluxes (W m-2 um-1) with the extinction applied.
    """

    ext_mag = ism_extinction(v_band_ext, v_band_red, wavelengths)

    return flux * 10.0 ** (-0.4 * ext_mag)


def planck_function_cm(
    temperature: float, wavelength: [ArrayLike, float]
) -> [Array, float]:
    """Returns the Planck function :math:`B_{\\lambda}(T)` in units of
    :math:`\\rm erg/s/cm^2/cm/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        wavelength:
            Array containing the wavelength in cm.
    """

    _planck_function = (
        2.0
        * cst.h
        * cst.c**2.0
        / wavelength**5.0
        / (jnp.exp(cst.h * cst.c / wavelength / cst.kB / temperature) - 1.0)
    )

    return _planck_function


def planck_function_hz(
    temperature: float, frequency: [ArrayLike, float]
) -> [Array, float]:
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    """

    _planck_function = (
        2.0
        * cst.h
        * frequency**3.0
        / cst.c**2.0
        / (jnp.exp(cst.h * frequency / cst.kB / temperature) - 1.0)
    )

    return _planck_function


def planck_function_hz_temperature_derivative(
    temperature: float, frequency: [ArrayLike, float]
) -> [Array, float]:
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature:
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    Returns:

    """
    _planck_function = planck_function_hz(temperature, frequency)
    _planck_function /= jnp.exp(cst.h * frequency / cst.kB / temperature) - 1.0
    _planck_function *= (
        jnp.exp(cst.h * frequency / cst.kB / temperature)
        * cst.h
        * frequency
        / cst.kB
        / temperature**2.0
    )

    return _planck_function


def shift_wavelengths_by_radial_velocity(radial_velocity):
    return jnp.sqrt((1 + radial_velocity / cst.c) / (1 - radial_velocity / cst.c))


def um2hz(wavelength: [ArrayLike, float]) -> [Array, float]:
    """Convert wavelengths in micrometer into frequencies.

    Args:
        wavelength: (um) the wavelengths to convert

    Returns:
        (Hz) the corresponding frequencies
    """
    return frequency2wavelength(
        wavelength * 1e-4
    )  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz


def wavelength2frequency(wavelength: [ArrayLike, float]) -> [Array, float]:
    """Convert wavelengths in centimeter to frequencies.

    Args:
        wavelength: (cm) the wavelengths to convert

    Returns:
        (Hz) the converted frequencies
    """
    return frequency2wavelength(
        wavelength
    )  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz
