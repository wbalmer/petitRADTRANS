"""Stores useful physical functions.
"""
# TODO: split off temperature profile functions?
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import custom_gradient, config, Array
from scipy.interpolate import interp1d, PchipInterpolator
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.math import running_mean


def compute_dist(t_irr, dist, t_star, r_star, mode, mode_what):
    # TODO rework/replace this function
    # TODO find better name -- maybe calculate_distance_from_luminosity
    mu_star = 0.
    angle_use = False

    if (mode != 'p') & (mode != 'd'):
        mu_star = float(mode)
        angle_use = True

    if mode_what == 'temp':
        if angle_use:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. * mu_star) ** 0.25
        elif mode == 'p':
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 4.) ** 0.25
        else:
            t_irr = ((r_star * cst.r_sun / (dist * cst.au)) ** 2. * t_star ** 4. / 2.) ** 0.25
        return t_irr
    elif mode_what == 'dist':
        if angle_use:
            dist = jnp.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. * mu_star) / cst.au
        elif mode == 'p':
            dist = jnp.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 4.) / cst.au
        else:
            dist = jnp.sqrt((r_star * cst.r_sun) ** 2. * (t_star / t_irr) ** 4. / 2.) / cst.au
        return dist

    return None


def compute_effective_temperature(
    wavelengths: ArrayLike,
    flux: ArrayLike,
    orbit_semi_major_axis: float = 1.0,
    planet_radius: float = 1.0,
    use_si_units: bool = False
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
            _flux[:-1] * ((orbit_semi_major_axis / planet_radius) ** 2.) * jnp.diff(_wavelengths)
        )

    unit_factor = 1.0

    if use_si_units:
        unit_factor = 1e-3

    energy = integrate_flux(wavelengths, flux)

    return (energy / (cst.sigma*unit_factor)) ** 0.25


def doppler_shift(wavelength_0: [ArrayLike, float], velocity: [ArrayLike, float]) -> [Array, float]:
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


def flux_cm2flux_hz(flux_cm: [ArrayLike, float], wavelength: [ArrayLike, float]) -> [Array, float]:
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
    return flux_cm * wavelength ** 2 / cst.c


def flux_hz2flux_cm(flux_hz: [ArrayLike, float], frequency: [ArrayLike, float]) -> [Array, float]:
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
    return flux_hz * frequency ** 2 / cst.c


def flux2irradiance(flux: [ArrayLike, float], source_radius: float, target_distance: float) -> [Array, float]:
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


def make_press_temp(rad_trans_params):
    """Function to make temp"""
    # TODO: delete, not used
    press_many = jnp.logspace(-8, 5, 260)
    t_no_ave = temperature_profile_function_guillot_modif(
        press_many,
        1e1 ** rad_trans_params['log_delta'],
        1e1 ** rad_trans_params['log_gamma'],
        rad_trans_params['t_int'], rad_trans_params['t_equ'],
        1e1 ** rad_trans_params['log_p_trans'], rad_trans_params['alpha']
    )

    # new
    press_many_new = 1e1 ** running_mean(jnp.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


def make_press_temp_iso(rad_trans_params):
    """Function to make temp"""
    # TODO: delete, not used
    press_many = jnp.logspace(-8, 5, 260)
    t_no_ave = rad_trans_params['t_equ'] * jnp.ones_like(press_many)

    # new
    press_many_new = 1e1 ** running_mean(jnp.log10(press_many), 25)
    t_new = running_mean(t_no_ave, 25)
    index_new = (press_many_new <= 1e3) & (press_many_new >= 1e-6)
    temp_new = t_new[index_new][::2]
    press_new = press_many_new[index_new][::2]

    return press_new, temp_new


def planck_function_cm(temperature: float, wavelength: [ArrayLike, float]) -> [Array, float]:
    """Returns the Planck function :math:`B_{\\lambda}(T)` in units of
    :math:`\\rm erg/s/cm^2/cm/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        wavelength:
            Array containing the wavelength in cm.
    """

    _planck_function = (
            2. * cst.h * cst.c ** 2. / wavelength ** 5.
            / (jnp.exp(cst.h * cst.c / wavelength / cst.kB / temperature) - 1.)
    )

    return _planck_function


def planck_function_hz(temperature: float, frequency: [ArrayLike, float]) -> [Array, float]:
    """Returns the Planck function :math:`B_{\\nu}(T)` in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.

    Args:
        temperature (float):
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    """

    _planck_function = (
            2. * cst.h * frequency ** 3. / cst.c ** 2. / (jnp.exp(cst.h * frequency / cst.kB / temperature) - 1.)
    )

    return _planck_function


def planck_function_hz_temperature_derivative(
    temperature: float,
    frequency: [ArrayLike, float]
) -> [Array, float]:
    """Returns the derivative of the Planck function with respect to the temperature in units of
    :math:`\\rm erg/s/cm^2/Hz/steradian`.
    # TODO unused?

    Args:
        temperature:
            Temperature in K.
        frequency:
            Array containing the frequency in Hz.
    Returns:

    """
    _planck_function = planck_function_hz(temperature, frequency)
    _planck_function /= jnp.exp(cst.h * frequency / cst.kB / temperature) - 1.
    _planck_function *= (
            jnp.exp(cst.h * frequency / cst.kB / temperature) * cst.h * frequency / cst.kB / temperature ** 2.
    )

    return _planck_function


def shift_wavelengths_by_radial_velocity(radial_velocity):
    return jnp.sqrt((1 + radial_velocity/cst.c)/(1 - radial_velocity/cst.c))


def um2hz(wavelength: [ArrayLike, float]) -> [Array, float]:
    """Convert wavelengths in micrometer into frequencies.

    Args:
        wavelength: (um) the wavelengths to convert

    Returns:
        (Hz) the corresponding frequencies
    """
    return frequency2wavelength(wavelength * 1e-4)  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz


def wavelength2frequency(wavelength: [ArrayLike, float]) -> [Array, float]:
    """Convert wavelengths in centimeter to frequencies.

    Args:
        wavelength: (cm) the wavelengths to convert

    Returns:
        (Hz) the converted frequencies
    """
    return frequency2wavelength(wavelength)  # the operation is the same: (c / Hz) -> cm, (c / cm) -> Hz
