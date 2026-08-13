"""Models Module

This module contains a set of functions that generate the spectra used
in the petitRADTRANS retrieval. This includes setting up the
pressure-temperature structure, the chemistry, and the radiative
transfer to compute the emission or transmission spectrum. See
the documentation on the Retrieval Models for full details.

All models must take the same set of inputs:

    ModelContext : ModelContext
        An object that contains the static information about the retrieval,
        including the Radtrans object.
    PhysicalParams : PhysicalParams
        An object containing all of the physical parameters of the model,
        which can be free parameters, or fixed values.
    pt_plot_mode : bool
        If this argument is True, the model function will return the pressure
        and temperature arrays instead of computing the flux.
"""

import os
import re
import inspect
from functools import wraps
from numbers import Number
from types import SimpleNamespace
import numpy as np
from jax import core as jax_core, vmap, checkpoint, lax
import jax.numpy as jnp

from petitRADTRANS.chemistry import clouds
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.chemistry.core import (
    _get_static_bool_parameter,
    get_abundances,
    get_abundances_from_legacy_dict,
    resolve_cloud_base_pressures,
)
from petitRADTRANS.chemistry.utils import build_adaptive_pressure_grid
from petitRADTRANS.physics import flux2irradiance, planck_function_cm, apply_ism_ext
from petitRADTRANS.stellar_spectra.phoenix import (
    phoenix_star_table,
    rebin_star_spectrum,
)
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.temperature_profiles import (
    cubic_spline_temperature_profile,
    linear_spline_temperature_profile,
    zhang_2023_temperature_profile,
    madhusudhan_seager_temperature_profile,
    power_law_temperature_profile,
    guillot_temperature_profile,
    guillot_global_temperature_profile,
    isothermal_temperature_profile,
    molliere_2020_temperature_profile,
)
from petitRADTRANS.planet import (
    mass2reference_gravity_jax,
    reference_gravity2radius_jax,
)
from petitRADTRANS.retrieval.runtime import (
    ModelContext,
    ModelInputs,
    ModelResult,
    PhysicalParams,
    _radtrans_wavelength_grid,
)
from petitRADTRANS.retrieval.utils import (
    get_calculate_flux_return_values,
    _get_parameter_value,
)


def _is_full_easychem_requested(parameters) -> bool:
    """Return whether the retrieval explicitly requested the full easyChem path."""
    if "use_easychem" not in parameters:
        return False

    return _get_static_bool_parameter("use_easychem", parameters["use_easychem"])


# Global constants to reduce calculations and initializations.
MODEL_CONTRACT_LEGACY = "legacy"
MODEL_CONTRACT_DIFFERENTIABLE = "differentiable"
MODEL_CONTRACT_LEGACY_ADAPTER = "runtime_legacy_adapter"
_MODEL_CONTRACT_ATTRIBUTE = "_prt_model_contract"
_WRAPPED_LEGACY_MODEL_ATTRIBUTE = "_prt_wrapped_legacy_model"

# Naming-convention patterns for automatic time-series parameter detection
_TIME_INDEXED_PATTERN = re.compile(r"^(.+)_t_(\d+)$")
_SINUSOIDAL_SUFFIXES = (
    "_amplitude",
    "_period",
    "_phase",
    "_offset",
    "_cos_amplitude",
    "_sin_amplitude",
)
_SINUSOIDAL_REQUIRED_SUFFIXES = ("_amplitude", "_phase", "_offset")
# Cartesian (phase-free) sinusoid alternative:
#   P(t) = offset + a*cos(2*pi*t/period) + b*sin(2*pi*t/period)
# with a = {base}_cos_amplitude and b = {base}_sin_amplitude. Equivalent to the
# polar form via (a, b) = (A*sin(phi), A*cos(phi)), but without the A >= 0
# boundary and the phase wrap, so gradient samplers see a nearly Gaussian
# geometry.
_CARTESIAN_REQUIRED_SUFFIXES = ("_cos_amplitude", "_sin_amplitude", "_offset")
# Optional SECOND-HARMONIC coefficients for the Cartesian form. When both are
# present the base additionally carries a component at twice the rotation
# frequency (i.e. period/2):
#   P(t) = offset + a1*cos(w t) + b1*sin(w t) + a2*cos(2 w t) + b2*sin(2 w t)
# Rotational light curves of variable brown dwarfs are generally NOT pure
# sinusoids -- a fundamental-only model leaves coherent residuals when the two
# hemispheres are not identical. Both harmonic coefficients must be supplied
# together; supplying neither reduces exactly to the fundamental-only form.
_CARTESIAN_HARMONIC_SUFFIXES = ("_cos_amplitude_h2", "_sin_amplitude_h2")
_SHARED_PERIOD_KEY = "period"


def _as_structural_int(value, parameter_name: str) -> int:
    """Return a Python int for parameters that define model structure.

    Counts such as ``nnodes``, ``N_layers``, and ``N_time`` are used in Python
    control flow and array-shape construction, so they must remain compile-time
    constants under JAX transforms.
    """
    if isinstance(value, Number):
        return int(value)

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise TypeError(
                f"Parameter '{parameter_name}' must be scalar, got shape {value.shape}"
            )

        return int(value.item())

    return jax_core.concrete_or_error(
        lambda x: int(jnp.asarray(x).item()),
        value,
        "Parameter "
        f"'{parameter_name}' must be a compile-time constant integer because "
        "it controls model structure under JAX.",
    )


def _normalize_legacy_output_value(value):
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(value.filled(0))

    return value


def _unit_to_cgs_factor(unit, target):
    """Return the float factor converting ``unit`` to the cgs ``target`` ('cm' or 'g').

    Returns ``None`` when ``unit`` is ``None`` (i.e. the value is already assumed to be in
    cgs). The factor is a static Python float derived from the astropy unit, so multiplying a
    traced JAX value by it remains JIT/grad safe.
    """
    if unit is None:
        return None

    import astropy.units as u

    target_unit = {"cm": u.cm, "g": u.g}[target]
    return float((1.0 * unit).to(target_unit).value)


def _resolve_parameter_unit(physical_params, parameter_units, name):
    """Resolve the astropy unit declared for ``name``.

    Units are looked up first in an explicit ``parameter_units`` mapping (the runtime path,
    where ``physical_params`` carries only unwrapped values), then fall back to a ``.units``
    attribute on a legacy ``Parameter`` object stored in the mapping. Returns ``None`` when no
    unit is declared.
    """
    if parameter_units:
        unit = parameter_units.get(name)
        if unit is not None:
            return unit

    entry = physical_params.get(name)
    return getattr(entry, "units", None)


def _parameter_value_in_cgs(physical_params, parameter_units, name, target):
    """Read ``name`` from ``physical_params`` and convert it to the cgs ``target`` unit."""
    value = _get_parameter_value(physical_params, name)
    unit = _resolve_parameter_unit(physical_params, parameter_units, name)
    factor = _unit_to_cgs_factor(unit, target)

    if factor is not None:
        value = value * factor

    return value


def _compute_gravity(physical_params, parameter_units=None):
    if "log_g" in physical_params.keys() and "planet_mass" in physical_params.keys():
        gravity = 10 ** _get_parameter_value(physical_params, "log_g")
        planet_radius, _, _ = reference_gravity2radius_jax(
            reference_gravity=gravity,
            mass=_parameter_value_in_cgs(
                physical_params, parameter_units, "planet_mass", "g"
            ),
        )
        planet_radius *= cst.r_jup_mean
    elif "log_g" in physical_params.keys():
        gravity = 10 ** _get_parameter_value(physical_params, "log_g")
        planet_radius = _parameter_value_in_cgs(
            physical_params, parameter_units, "planet_radius", "cm"
        )
    elif "planet_mass" in physical_params.keys() and "planet_radius" in physical_params.keys():
        planet_radius = _parameter_value_in_cgs(
            physical_params, parameter_units, "planet_radius", "cm"
        )
        gravity, _, _ = mass2reference_gravity_jax(
            mass=_parameter_value_in_cgs(
                physical_params, parameter_units, "planet_mass", "g"
            ),
            radius=planet_radius,
        )
    else:
        raise KeyError("Pick two of log_g, planet_radius and planet_mass priors!")

    # NaN-poison unphysical values instead of letting them reach the Radtrans
    # gravity validation callback, which RAISES inside jitted code and kills the
    # whole process (job 1869154/1869202: an L-BFGS line-search excursion to
    # planet_mass < 0 aborted the optimisation). NaN <= 0 is False, so the core
    # check stays silent; the NaN spectrum is then rejected cleanly through the
    # scorer's INVALID_LOG_LIKELIHOOD sentinel (a rejected sample, not a crash).
    gravity = jnp.where(gravity > 0.0, gravity, jnp.nan)
    planet_radius = jnp.where(planet_radius > 0.0, planet_radius, jnp.nan)

    return gravity, planet_radius


def _resolve_guillot_gamma(parameters):
    """Return the Guillot gamma parameter in linear units for any model contract."""
    gamma = _get_parameter_value(parameters, "gamma")
    if gamma is not None:
        return gamma

    log_gamma = _get_parameter_value(parameters, "log_gamma")
    if log_gamma is not None:
        return jnp.power(10.0, log_gamma)

    raise KeyError(
        "Missing Guillot gamma parameter: provide either 'gamma' or 'log_gamma'"
    )


def _clip_easychem_temperatures(physical_params, temperatures):
    if _is_full_easychem_requested(physical_params):
        return jnp.clip(temperatures, 40, 40000)

    return temperatures


def _wrap_runtime_temperature_builder(physical_params, temperature_builder):
    def build_temperatures(pressures):
        return _clip_easychem_temperatures(
            physical_params,
            temperature_builder(pressures),
        )

    return build_temperatures


def _stack_runtime_cloud_base_pressures(p_bases, dtype):
    if len(p_bases) == 0:
        return jnp.zeros((0,), dtype=dtype)

    return jnp.stack(
        tuple(
            jnp.asarray(cloud_base_pressure, dtype=dtype)
            for cloud_base_pressure in p_bases.values()
        )
    )


def molliere_2020_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)

    t3 = ((3.0 / 4.0 * physical_params["T_int"] ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25) * (
        1.0 - physical_params["T3"]
    )
    t2 = t3 * (1.0 - physical_params["T2"])
    t1 = t2 * (1.0 - physical_params["T1"])
    delta = ((10.0 ** (-3.0 + 5.0 * physical_params["log_delta"])) * 1e6) ** (
        -physical_params["alpha"]
    )
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    temp_arr = jnp.array([t1, t2, t3])
    carbon_to_oxygen = 0

    if "C/O" in physical_params.keys():
        carbon_to_oxygen = physical_params["C/O"]
    elif "C" in physical_params.keys():
        carbon_to_oxygen = physical_params["C"] / physical_params["O"]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: molliere_2020_temperature_profile(
            temperature_nodes=temp_arr,
            delta=delta,
            alpha=physical_params["alpha"],
            t_internal=physical_params["T_int"],
            pressures=pressures,
            metallicity=physical_params["M/H"],
            co_ratio=carbon_to_oxygen,
            conv=True,
        )
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def molliere_2020_two_column(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)

    t3 = ((3.0 / 4.0 * physical_params["T_int"] ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25) * (
        1.0 - physical_params["T3"]
    )
    t2 = t3 * (1.0 - physical_params["T2"])
    t1 = t2 * (1.0 - physical_params["T1"])
    delta = ((10.0 ** (-3.0 + 5.0 * physical_params["log_delta"])) * 1e6) ** (  # noqa: F841
        -physical_params["alpha"]
    )
    temp_arr = jnp.array([t1, t2, t3])  # noqa: F841

    t3_clear = (
        (3.0 / 4.0 * physical_params["T_int"] ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25
    ) * (1.0 - physical_params["T3_clear"])
    t2_clear = t3_clear * (1.0 - physical_params["T2_clear"])
    t1_clear = t2_clear * (1.0 - physical_params["T1_clear"])
    temps_clear = jnp.array([t1_clear, t2_clear, t3_clear])
    delta_clear = (
        (10.0 ** (-3.0 + 5.0 * physical_params["log_delta_clear"])) * 1e6
    ) ** (-physical_params["alpha_clear"])
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    cloudy_temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: molliere_2020_temperature_profile(
            temperature_nodes=temps_clear,
            delta=delta_clear,
            alpha=physical_params["alpha_clear"],
            t_internal=physical_params["T_int"],
            pressures=pressures,
            metallicity=physical_params["M/H"],
            co_ratio=physical_params["C/O"],
            conv=True,
        ),
    )
    temperatures = cloudy_temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=cloudy_temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    clear_temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: molliere_2020_temperature_profile(
            temperature_nodes=temps_clear,
            delta=delta_clear,
            alpha=physical_params["alpha_clear"],
            t_internal=physical_params["T_int_clear"],
            pressures=pressures,
            metallicity=physical_params["M/H"],
            co_ratio=physical_params["C/O"],
            conv=True,
        ),
    )
    t_clear = clear_temperature_builder(abundance_state["pressures"])
    abundances_clear, mean_molar_masses_clear, _small_index_clear, _p_bases_clear = (
        get_abundances(
            pressures=abundance_state["pressures"],
            temperatures=t_clear,
            parameters=physical_params,
            line_species=model_context.radtrans.line_species,
            cloud_species=model_context.radtrans.cloud_species,
            adaptive_mesh_refinement=False,
        )
    )

    if abundances_clear is None:
        return ModelResult.make_nan_result(
            _radtrans_wavelength_grid(model_context.radtrans)
        )

    cloudy_cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    cloudy_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloudy_cloud_properties,
    )
    cloudy_result = calculate_emission_spectrum_runtime(model_context, cloudy_inputs)
    if not cloudy_result.spectrum_is_valid():
        return cloudy_result

    clear_abundances = dict(abundances_clear)
    for cloud in model_context.radtrans.cloud_species:
        cname = cloud.split("_")[0]
        clear_abundances[cname] = jnp.zeros_like(t_clear)

    clear_state = {
        "pressures": abundance_state["pressures"],
        "temperatures": t_clear,
        "abundances": clear_abundances,
        "mean_molar_masses": mean_molar_masses_clear,
    }
    clear_inputs = ModelInputs(
        pressures=clear_state["pressures"],
        temperatures=clear_state["temperatures"],
        abundances=clear_state["abundances"],
        mean_molar_masses=clear_state["mean_molar_masses"],
        gravity=gravity,
        planet_radius=planet_radius,
        distance_to_system=physical_params["system_distance"],
    )
    clear_result = calculate_emission_spectrum_runtime(model_context, clear_inputs)
    if not clear_result.spectrum_is_valid():
        return clear_result

    patchiness = physical_params["patchiness"]
    spectrum_model = (patchiness * cloudy_result.spectrum) + (
        (1.0 - patchiness) * clear_result.spectrum
    )

    return ModelResult(
        kind="spectrum",
        wavelengths=cloudy_result.wavelengths,
        spectrum=spectrum_model,
        auxiliary_outputs=dict(clear_result.auxiliary_outputs),
    )


def guillot_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    gamma = _resolve_guillot_gamma(physical_params)

    redistribution_coefficient = _get_parameter_value(physical_params, "redistribution_coefficient")
    if redistribution_coefficient is None:
        redistribution_coefficient = 0.25

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: guillot_temperature_profile(
            pressures,
            10 ** physical_params["log_kappa_IR"],
            gamma,
            gravity,
            physical_params["T_int"],
            physical_params["T_equ"],
            redistribution_coefficient=redistribution_coefficient

        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def guillot_relative_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    """Guillot emission model normalised by a PHOENIX stellar spectrum.

    Computes the same Guillot global temperature profile and emission spectrum
    as :func:`guillot_emission`, then divides the planet flux by a PHOENIX
    stellar spectrum binned to the same wavelength grid.  The returned spectrum
    is the dimensionless planet-to-star flux ratio F_planet / F_star, suitable
    for secondary-eclipse or direct-imaging contrast retrieval.

    Required parameters (shared with :func:`guillot_emission`)
    -----------------------------------------------------------
    distance_to_system : float
        Distance from the system to the observer in cm.
    log_g, planet_radius, mass : float
        Two of these three must be supplied (see :func:`_compute_gravity`).
    T_int, T_equ, gamma, log_kappa_IR : float
        Guillot temperature-profile parameters.
    Fe/H, C/O, log_pquench : float
        Equilibrium-chemistry parameters.

    Additional stellar parameters
    -----------------------------
    star_effective_temperature : float
        Effective temperature of the host star in K.  Used to interpolate the
        PHOENIX spectral grid.
    stellar_radius : float, optional
        Stellar radius in cm.  If omitted, the radius from the PHOENIX grid is
        used.
    """
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    gamma = _resolve_guillot_gamma(physical_params)
    redistribution_coefficient = _get_parameter_value(physical_params, "redistribution_coefficient")
    if redistribution_coefficient is None:
        redistribution_coefficient = 0.25

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: guillot_temperature_profile(
            pressures,
            10 ** physical_params["log_kappa_IR"],
            gamma,
            gravity,
            physical_params["T_int"],
            physical_params["T_equ"],
            redistribution_coefficient=redistribution_coefficient
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    planet_result = calculate_emission_spectrum_runtime(model_context, model_inputs)

    if not planet_result.spectrum_is_valid():
        return planet_result

    # --- Stellar spectrum ---
    # Frequency bin centres corresponding to the model wavelength grid.
    freq_edges = model_context.radtrans.frequency_bins_edges
    freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])  # Hz
    lambda_cm = cst.c / freq_centers  # wavelengths in cm

    # Interpolate the PHOENIX grid to obtain the surface F_nu [erg/cm^2/s/Hz].
    star_spectrum, phoenix_star_radius = phoenix_star_table.compute_spectrum(
        physical_params["star_effective_temperature"]
    )

    # Use an explicitly provided stellar_radius when available.
    if "stellar_radius" in physical_params.keys():
        star_radius = physical_params["stellar_radius"]
    else:
        star_radius = phoenix_star_radius

    # Rebin surface F_nu to model wavelength grid.
    stellar_fnu_surface = rebin_star_spectrum(
        star_spectrum=star_spectrum[:, 1],
        star_wavelengths=star_spectrum[:, 0],
        wavelengths=lambda_cm,
    )  # erg/cm^2/s/Hz at stellar surface

    # Scale to the observer distance: F_nu_earth = F_nu_surface * (R_star / d)^2.
    distance = physical_params["system_distance"]
    stellar_fnu_earth = stellar_fnu_surface * (star_radius / distance) ** 2

    # Convert F_nu [erg/cm^2/s/Hz] -> F_lambda [erg/cm^2/s/cm] -> [W/cm^2/cm]
    # using F_lambda = F_nu * c / lambda^2 and 1 erg = 1e-7 J.
    # The 1e-7 matches the unit convention applied to the planet flux inside
    # calculate_emission_spectrum_runtime.
    stellar_flambda_earth = stellar_fnu_earth * cst.c / lambda_cm**2 * 1e-7

    # Dimensionless planet-to-star flux ratio.
    ratio = planet_result.spectrum / stellar_flambda_earth

    return ModelResult(
        kind="spectrum",
        wavelengths=planet_result.wavelengths,
        spectrum=ratio,
        auxiliary_outputs=planet_result.auxiliary_outputs,
    )


def guillot_emission_add_gaussian_temperature(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    gamma = _resolve_guillot_gamma(physical_params)
    redistribution_coefficient = _get_parameter_value(physical_params, "redistribution_coefficient")
    if redistribution_coefficient is None:
        redistribution_coefficient = 0.25

    def gaussian(x, scale, mu, sig):
        return scale * jnp.exp(-jnp.power((x - mu) / sig, 2.0) / 2)

    def base_temperature_builder(pressures):

        temperature_builder = _wrap_runtime_temperature_builder(
            physical_params,
            lambda pressures: guillot_temperature_profile(
                pressures,
                10 ** physical_params["log_kappa_IR"],
                gamma,
                gravity,
                physical_params["T_int"],
                physical_params["T_equ"],
                redistribution_coefficient=redistribution_coefficient

            ),
        )
        temperature_addition = gaussian(
            jnp.log10(pressures),
            physical_params["temperature_peak"],
            physical_params["temperature_location_log_pressure"],
            physical_params["temperature_width_log_pressure"],
        )
        temperatures = temperature_builder(pressures) + temperature_addition
        safe_inds = jnp.where(temperatures < 1.0)
        return temperatures.at[safe_inds].set(1.0)

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        base_temperature_builder,
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def guillot_patchy_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    return guillot_emission(model_context, physical_params, pt_plot_mode=pt_plot_mode)


def interpolated_profile_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )
    nnodes = _as_structural_int(physical_params["nnodes"], "nnodes")

    temp_arr = jnp.array(
        [physical_params[f"T{i}"] for i in range(nnodes + 2)]
    )

    def build_temperature_profile(pressures):
        if "linear" in physical_params.keys():
            return linear_spline_temperature_profile(
                pressures,
                temp_arr,
                physical_params["gamma"],
                nnodes,
            )

        return cubic_spline_temperature_profile(
            pressures,
            temp_arr,
            physical_params["gamma"],
            nnodes,
        )

    if "linear" in physical_params.keys():
        temperatures, log_prior_weight = linear_spline_temperature_profile(
            p_use,
            temp_arr,
            physical_params["gamma"],
            nnodes,
        )
    else:
        temperatures, log_prior_weight = cubic_spline_temperature_profile(
            p_use,
            temp_arr,
            physical_params["gamma"],
            nnodes,
        )

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: build_temperature_profile(pressures)[0],
    )
    temperatures = _clip_easychem_temperatures(physical_params, temperatures)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    model_result = calculate_emission_spectrum_runtime(model_context, model_inputs)

    merged_auxiliary_outputs = dict(model_result.auxiliary_outputs)

    merged_auxiliary_outputs["log_prior_weight"] = log_prior_weight
    return ModelResult(
        kind=model_result.kind,
        wavelengths=model_result.wavelengths,
        spectrum=model_result.spectrum,
        pressures=model_result.pressures,
        temperatures=model_result.temperatures,
        additional_log_likelihood=model_result.additional_log_likelihood,
        auxiliary_outputs=merged_auxiliary_outputs,
        atmospheric_column_fluxes=model_result.atmospheric_column_fluxes,
    )


def gradient_profile_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    num_layer = _as_structural_int(physical_params["N_layers"], "N_layers")
    # Build directly: an all-NaN initialiser + scatter-fill is a forward NaN
    # that poisons reverse-mode gradients via 0*NaN in the multiply's tangent.
    layer_pt_slopes = jnp.stack(
        [physical_params[f"PTslope_{num_layer - index}"] for index in range(num_layer)]
    )

    top_of_atmosphere_pressure = -3
    bottom_of_atmosphere_pressure = 3
    if "top_of_atmosphere_pressure" in physical_params.keys():
        top_of_atmosphere_pressure = physical_params["top_of_atmosphere_pressure"]
    if "bottom_of_atmosphere_pressure" in physical_params.keys():
        bottom_of_atmosphere_pressure = physical_params["bottom_of_atmosphere_pressure"]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: zhang_2023_temperature_profile(
            pressures,
            num_layer,
            layer_pt_slopes,
            physical_params["T_bottom"],
            top_of_atmosphere_pressure=top_of_atmosphere_pressure,
            bottom_of_atmosphere_pressure=bottom_of_atmosphere_pressure,
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def _build_timestep_params(physical_params, t_idx, t_value):
    """Build a :class:`PhysicalParams` snapshot for a single model timestep.

    Auto-detects time-varying parameters by scanning parameter names:

    - **Per-timestep keys** matching ``{P}_t_{int}``: the value at index
      *t_idx* is stored under the base name ``P``.
    - **Sinusoidal quartets** ``{P}_amplitude``, ``{P}_period``,
      ``{P}_phase``, ``{P}_offset``: the base name ``P`` is evaluated as
      ``offset + amplitude * sin(2π * t_value / period + phase)``.

    All detected meta-keys are stripped from the snapshot; only the resolved
    base name is included.  Parameters matching neither pattern are copied
    unchanged.  This allows *any* parameter to vary with time simply by
    following the naming convention—no pre-declaration is required.
    """
    # --- Detect per-timestep parameters: {base}_t_{int} ---
    time_varying_bases = set()
    tv_keys = set()
    for key in physical_params.keys():
        m = _TIME_INDEXED_PATTERN.match(key)
        if m:
            time_varying_bases.add(m.group(1))
            tv_keys.add(key)

    # --- Detect sinusoidal parameters ---
    # A base P is sinusoidal in one of two equivalent forms:
    #   * polar:     {P}_amplitude, {P}_phase and {P}_offset all present;
    #   * Cartesian: {P}_cos_amplitude, {P}_sin_amplitude and {P}_offset all
    #     present (phase-free; preferred for gradient samplers).
    # The period is taken from {P}_period when present, otherwise from a
    # single shared "period" key (see _SHARED_PERIOD_KEY); the per-base period
    # takes precedence when both exist.
    all_keys = set(physical_params.keys())
    shared_period_available = _SHARED_PERIOD_KEY in all_keys
    sinusoidal_bases = {}  # base -> "polar" | "cartesian"
    sin_keys = set()
    candidate_bases = set()
    for key in all_keys:
        for suf in _SINUSOIDAL_SUFFIXES:
            if key.endswith(suf):
                candidate_bases.add(key[: -len(suf)])
    for base in candidate_bases:
        if not base or base in time_varying_bases:
            continue  # per-timestep keys take precedence
        has_polar = all(
            f"{base}{suf}" in all_keys for suf in _SINUSOIDAL_REQUIRED_SUFFIXES
        )
        has_cartesian = all(
            f"{base}{suf}" in all_keys for suf in _CARTESIAN_REQUIRED_SUFFIXES
        )
        if has_polar and has_cartesian:
            raise ValueError(
                f"Time-series parameter '{base}' declares both the polar "
                f"({base}_amplitude/{base}_phase) and Cartesian "
                f"({base}_cos_amplitude/{base}_sin_amplitude) sinusoid forms; "
                "use exactly one."
            )
        if not (has_polar or has_cartesian):
            continue
        has_period = (f"{base}_period" in all_keys) or shared_period_available
        if not has_period:
            continue
        required_suffixes = (
            _SINUSOIDAL_REQUIRED_SUFFIXES if has_polar else _CARTESIAN_REQUIRED_SUFFIXES
        )
        harmonic_present = [
            f"{base}{suf}" in all_keys for suf in _CARTESIAN_HARMONIC_SUFFIXES
        ]
        if any(harmonic_present):
            if has_polar:
                raise ValueError(
                    f"Time-series parameter '{base}' declares second-harmonic "
                    "coefficients, which are only supported by the Cartesian "
                    f"form; use {base}_cos_amplitude/{base}_sin_amplitude."
                )
            if not all(harmonic_present):
                raise ValueError(
                    f"Time-series parameter '{base}' declares only one of "
                    f"{base}_cos_amplitude_h2 / {base}_sin_amplitude_h2; "
                    "supply both or neither."
                )
        has_harmonic = all(harmonic_present) and not has_polar
        sinusoidal_bases[base] = (
            "polar"
            if has_polar
            else ("cartesian_h2" if has_harmonic else "cartesian")
        )
        for suf in required_suffixes:
            sin_keys.add(f"{base}{suf}")
        if has_harmonic:
            for suf in _CARTESIAN_HARMONIC_SUFFIXES:
                sin_keys.add(f"{base}{suf}")
        if f"{base}_period" in all_keys:
            sin_keys.add(f"{base}_period")

    skip_keys = tv_keys | sin_keys
    # The shared period is a meta-parameter, not a physical model input; strip it
    # from the snapshot once at least one sinusoid consumes it.
    if shared_period_available and sinusoidal_bases:
        skip_keys = skip_keys | {_SHARED_PERIOD_KEY}

    # --- Build snapshot ---
    base_dict = {}
    for key in physical_params.keys():
        if key in skip_keys:
            continue
        base_dict[key] = physical_params[key]

    # Resolve per-timestep values
    for base in time_varying_bases:
        base_dict[base] = physical_params[f"{base}_t_{t_idx}"]

    # Evaluate sinusoidal values
    for base, form in sinusoidal_bases.items():
        if f"{base}_period" in all_keys:
            period = physical_params[f"{base}_period"]
        else:
            period = physical_params[_SHARED_PERIOD_KEY]
        offset = physical_params[f"{base}_offset"]
        angle = 2.0 * jnp.pi * t_value / period
        if form == "polar":
            amp = physical_params[f"{base}_amplitude"]
            phase = physical_params[f"{base}_phase"]
            base_dict[base] = offset + amp * jnp.sin(angle + phase)
        else:
            cos_amp = physical_params[f"{base}_cos_amplitude"]
            sin_amp = physical_params[f"{base}_sin_amplitude"]
            value = offset + cos_amp * jnp.cos(angle) + sin_amp * jnp.sin(angle)
            if form == "cartesian_h2":
                cos_amp_h2 = physical_params[f"{base}_cos_amplitude_h2"]
                sin_amp_h2 = physical_params[f"{base}_sin_amplitude_h2"]
                value = (
                    value
                    + cos_amp_h2 * jnp.cos(2.0 * angle)
                    + sin_amp_h2 * jnp.sin(2.0 * angle)
                )
            base_dict[base] = value

    return PhysicalParams.from_mapping(base_dict)


def time_series_gradient_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    """Time-variable gradient-profile emission model.

    Computes ``N_time`` emission spectra at regularly spaced model timesteps
    spanning the range of the observation times.  Spectra at intermediate
    observation times are obtained by linear interpolation along the time axis.

    Time-varying parameters are **auto-detected** from the naming convention
    by :func:`_build_timestep_params` — no explicit declaration is needed.
    Any parameter can vary with time simply by following one of the two
    supported patterns:

    Per-timestep keys
    -----------------
    Supply ``{P}_t_0`` … ``{P}_t_{N_time-1}`` as free parameters.  At each
    model timestep *i* the snapshot will contain ``P`` set to the value of
    ``{P}_t_i``.

    Sinusoidal parameterisation
    ---------------------------
    Supply ``{P}_amplitude``, ``{P}_period`` (seconds), ``{P}_phase``
    (radians), and ``{P}_offset``.  The value at time *t* is

    .. math::

        P(t) = \\text{offset} + A \\sin\\!\\left(\\frac{2\\pi t}{\\text{period}} + \\phi\\right)

    Alternatively, supply the Cartesian (phase-free) form
    ``{P}_cos_amplitude``, ``{P}_sin_amplitude`` and ``{P}_offset``:

    .. math::

        P(t) = \\text{offset} + a \\cos\\!\\left(\\frac{2\\pi t}{\\text{period}}\\right)
             + b \\sin\\!\\left(\\frac{2\\pi t}{\\text{period}}\\right)

    with :math:`(a, b) = (A \\sin\\phi, A \\cos\\phi)`.  In both forms the
    period is per-base ``{P}_period`` when present, otherwise the single
    shared free parameter ``period``.

    Parameters that do not match either pattern are treated as constant
    across all timesteps (e.g. ``distance_to_system``, ``mass``,
    ``planet_radius``, ``N_layers``).

    Returns
    -------
    ModelResult
        When ``pt_plot_mode=False``: ``spectrum`` has shape
        ``(N_time, N_wavelength)`` on the shared model time grid, and
        ``auxiliary_outputs['model_times']`` stores that grid. The runtime
        scorer projects each epoch to observation space before interpolating in
        time, so the model only needs to carry one spectral representation.
        When ``pt_plot_mode=True``: ``temperatures`` has shape
        ``(N_time, N_pressure)`` and ``auxiliary_outputs['model_times']``
        gives the corresponding model time grid.
    """
    metadata = model_context.model_metadata
    observation_times = metadata["observation_times"]
    n_time = _as_structural_int(physical_params["N_time"], "N_time")

    # Build a regular model time grid spanning the observation window
    t_min = jnp.min(observation_times)
    t_max = jnp.max(observation_times)
    model_times = jnp.linspace(t_min, t_max, n_time)

    # Pressure grid and gravity are constant across timesteps
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )
    num_layer = _as_structural_int(physical_params["N_layers"], "N_layers")

    top_of_atmosphere_pressure = -3
    bottom_of_atmosphere_pressure = 3
    if "top_of_atmosphere_pressure" in physical_params.keys():
        top_of_atmosphere_pressure = physical_params["top_of_atmosphere_pressure"]
    if "bottom_of_atmosphere_pressure" in physical_params.keys():
        bottom_of_atmosphere_pressure = physical_params["bottom_of_atmosphere_pressure"]

    def _build_timestep_temperature_builder(snapshot):
        """Gradient temperature-profile builder for one epoch's parameter snapshot."""
        layer_pt_slopes = jnp.stack(
            [snapshot[f"PTslope_{num_layer - k}"] for k in range(num_layer)]
        )

        return _wrap_runtime_temperature_builder(
            snapshot,
            lambda pressures: zhang_2023_temperature_profile(
                pressures,
                num_layer,
                layer_pt_slopes,
                snapshot["T_bottom"],
                top_of_atmosphere_pressure=top_of_atmosphere_pressure,
                bottom_of_atmosphere_pressure=bottom_of_atmosphere_pressure,
            ),
        )

    # t_idx must stay a *static* argument: _build_timestep_params indexes
    # parameter names with it (e.g. "{base}_t_{t_idx}").
    def _emission_spectrum_at_timestep(t_idx, t_val):
        snapshot = _build_timestep_params(physical_params, t_idx, t_val)
        temperature_builder = _build_timestep_temperature_builder(snapshot)
        temperatures = temperature_builder(p_use)

        # Resolve chemistry
        result, abundance_state = _resolve_runtime_abundance_state(
            model_context,
            snapshot,
            p_use,
            p_global,
            temperatures,
            temperature_builder=temperature_builder,
            pt_plot_mode=False,
        )
        if result is not None:
            return result.spectrum, result.wavelengths

        # Clouds
        cloud_properties = clouds.setup_clouds(
            abundance_state["pressures"],
            snapshot,
            model_context.radtrans.cloud_species,
        )

        # Build model inputs and compute emission spectrum
        model_inputs = _build_runtime_emission_model_inputs(
            snapshot,
            abundance_state,
            gravity,
            planet_radius,
            cloud_properties,
        )
        step_result = calculate_emission_spectrum_runtime(model_context, model_inputs)
        return step_result.spectrum, step_result.wavelengths

    _checkpointed_emission_at_timestep = checkpoint(
        _emission_spectrum_at_timestep, static_argnums=(0,)
    )

    if pt_plot_mode:
        # PT-profile-only path: cheap and not differentiated.
        model_temperatures = []
        for t_idx in range(n_time):
            snapshot = _build_timestep_params(
                physical_params, t_idx, model_times[t_idx]
            )
            temperature_builder = _build_timestep_temperature_builder(snapshot)
            model_temperatures.append(temperature_builder(p_use))
        return ModelResult(
            kind="pt_profile",
            pressures=p_use,
            temperatures=jnp.stack(model_temperatures),
            auxiliary_outputs={"model_times": model_times},
        )

    # Per-timestep-keyed parameters ("{base}_t_{i}") need a concrete Python
    # index to look up, so when they are present the epochs must stay unrolled.
    has_time_indexed_params = any(
        _TIME_INDEXED_PATTERN.match(key) for key in physical_params.keys()
    )

    if has_time_indexed_params:
        model_spectra = []
        wavelengths = None
        for t_idx in range(n_time):
            step_spectrum, wavelengths = _checkpointed_emission_at_timestep(
                t_idx, model_times[t_idx]
            )
            model_spectra.append(step_spectrum)
        model_spectra_2d = jnp.stack(model_spectra)  # (N_time, N_wavelength)
    else:

        def _emission_scan_body(_carry, t_val):
            spectrum, step_wavelengths = _emission_spectrum_at_timestep(0, t_val)
            return _carry, (spectrum, step_wavelengths)

        _, (model_spectra_2d, wavelengths_per_step) = lax.scan(
            checkpoint(_emission_scan_body), None, model_times
        )  # model_spectra_2d: (N_time, N_wavelength)
        wavelengths = wavelengths_per_step[0]

    return ModelResult(
        kind="spectrum",
        wavelengths=wavelengths,
        spectrum=model_spectra_2d,
        auxiliary_outputs={"model_times": model_times},
    )


def power_law_profile_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: power_law_temperature_profile(
            pressures,
            physical_params["alpha"],
            physical_params["T_0"],
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def guillot_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    reference_pressure = 100.0
    if "reference_pressure" in physical_params.keys():
        reference_pressure = physical_params["reference_pressure"]

    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    gamma = _resolve_guillot_gamma(physical_params)
    redistribution_coefficient = _get_parameter_value(physical_params, "redistribution_coefficient")
    if redistribution_coefficient is None:
        redistribution_coefficient = 0.25

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: guillot_temperature_profile(
            pressures,
            10 ** physical_params["log_kappa_IR"],
            gamma,
            gravity,
            physical_params["T_int"],
            physical_params["T_equ"],
            redistribution_coefficient=redistribution_coefficient
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_transmission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_properties=cloud_properties,
    )
    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


def guillot_patchy_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    return guillot_transmission(
        model_context, physical_params, pt_plot_mode=pt_plot_mode
    )


def madhusudhan_seager_emission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    offset = jnp.log10(p_use[0])
    log_p3 = physical_params["log_P3"] - offset
    log_p2 = log_p3 * (1.0 - physical_params["P2"])
    log_p1 = log_p2 * (1.0 - physical_params["P1"])
    pressure_points = [
        p_use[0],
        (log_p1 + offset),
        (log_p2 + offset),
        (log_p3 + offset),
        physical_params["log_P_set"],
    ]

    alpha_points = [physical_params["alpha_0"], physical_params["alpha_1"]]
    beta_points = [0.5, 0.5]
    if "beta" in physical_params.keys():
        beta_points = [physical_params["beta"], physical_params["beta"]]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: madhusudhan_seager_temperature_profile(
            pressures,
            pressure_points,
            physical_params["T_set"],
            alpha_points,
            beta_points,
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_emission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        cloud_properties,
    )
    return calculate_emission_spectrum_runtime(model_context, model_inputs)


def madhusudhan_seager_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    reference_pressure = 100.0
    if "reference_pressure" in physical_params.keys():
        reference_pressure = physical_params["reference_pressure"]

    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )
    offset = jnp.log10(p_use[0])
    log_p3 = physical_params["log_P3"] - offset
    log_p2 = log_p3 * (1.0 - physical_params["P2"])
    log_p1 = log_p2 * (1.0 - physical_params["P1"])
    pressure_points = [
        p_use[0],
        (log_p1 + offset),
        (log_p2 + offset),
        (log_p3 + offset),
        physical_params["log_P_set"],
    ]

    alpha_points = [physical_params["alpha_0"], physical_params["alpha_1"]]
    beta_points = [0.5, 0.5]
    if "beta" in physical_params.keys():
        beta_points = [physical_params["beta"], physical_params["beta"]]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: madhusudhan_seager_temperature_profile(
            pressures,
            pressure_points,
            physical_params["T_set"],
            alpha_points,
            beta_points,
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_transmission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_properties=cloud_properties,
    )
    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


def madhu_seager_patchy_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    return madhusudhan_seager_transmission(
        model_context,
        physical_params,
        pt_plot_mode=pt_plot_mode,
    )


def isothermal_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)

    reference_pressure = 100.0
    if "reference_pressure" in physical_params.keys():
        reference_pressure = physical_params["reference_pressure"]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: isothermal_temperature_profile(
            pressures,
            physical_params["temperature"],
        ),
    )
    temperatures = temperature_builder(p_use)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_transmission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_properties=cloud_properties,
    )
    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


def power_law_profile_transmission(
    model_context: ModelContext,
    physical_params: PhysicalParams,
    pt_plot_mode: bool = False,
) -> ModelResult:
    p_use, p_global = _initialize_runtime_pressure_grid(model_context, physical_params)
    gravity, planet_radius = _compute_gravity(
        physical_params, model_context.parameter_units
    )

    reference_pressure = 100.0
    if "reference_pressure" in physical_params.keys():
        reference_pressure = physical_params["reference_pressure"]

    temperature_builder = _wrap_runtime_temperature_builder(
        physical_params,
        lambda pressures: power_law_temperature_profile(
            pressures,
            physical_params["alpha"],
            physical_params["T_0"],
        ),
    )
    temperatures = temperature_builder(p_use)

    result, abundance_state = _resolve_runtime_abundance_state(
        model_context,
        physical_params,
        p_use,
        p_global,
        temperatures,
        temperature_builder=temperature_builder,
        pt_plot_mode=pt_plot_mode,
    )
    if result is not None:
        return result

    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes(physical_params)
    )
    cloud_properties = clouds.setup_clouds(
        abundance_state["pressures"],
        physical_params,
        model_context.radtrans.cloud_species,
    )
    model_inputs = _build_runtime_transmission_model_inputs(
        physical_params,
        abundance_state,
        gravity,
        planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_properties=cloud_properties,
    )
    return calculate_transmission_spectrum_runtime(model_context, model_inputs)


def mark_legacy_model_function(model_function):
    setattr(model_function, _MODEL_CONTRACT_ATTRIBUTE, MODEL_CONTRACT_LEGACY)
    return model_function


@mark_legacy_model_function
def legacy_molliere_2020_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True
):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    Many of the parameters are optional, but must be used in the correct combination
    with other parameters.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline [K]
                *  T2 : Middle temperature spline [K]
                *  T1 : Outer temperature spline [K]
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  fsed : sedimentation parameter - can be unique to each cloud type by adding _CloudName
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : ArrayLike
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : ArrayLike
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, ArrayLike
            Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = (
        (3.0 / 4.0 * parameters["T_int"].value ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25
    ) * (1.0 - parameters["T3"].value)
    t2 = t3 * (1.0 - parameters["T2"].value)
    t1 = t2 * (1.0 - parameters["T1"].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters["log_delta"].value)) * 1e6) ** (
        -parameters["alpha"].value
    )
    gravity, planet_radius = _compute_gravity(parameters)

    # Make the P-T profile
    temp_arr = jnp.array([t1, t2, t3])
    carbon_to_oxygen = 0

    if "C/O" in parameters.keys():
        carbon_to_oxygen = parameters["C/O"].value
    elif "C" in parameters.keys():
        carbon_to_oxygen = parameters["C"].value / parameters["O"].value

    temperatures = (
        molliere_2020_temperature_profile(
            temperature_nodes=temp_arr,
            delta=delta,
            alpha=parameters["alpha"].value,
            t_internal=parameters["T_int"].value,
            pressures=p_use,
            metallicity=parameters["M/H"].value,
            co_ratio=carbon_to_oxygen,
            conv=True,  # conv
        )
    )

    if _is_full_easychem_requested(parameters):
        temperatures = jnp.clip(temperatures, 40, 40000)

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Calculate the spectrum
    if pressures.shape[0] != prt_object.pressures.shape[0]:
        return None, None

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_molliere_2020_two_column(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=True
):
    """
    Disequilibrium Chemistry Emission Model

    This model computes an emission spectrum based on the temperature profile of (Molliere 2020).
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.
    This model includes patchy clouds, and requires a unique temperature profile for the
    clear atmosphere regions - ie this is a full two column model!

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T3 : Innermost temperature spline
                *  T2 : Middle temperature spline
                *  T1 : Outer temperature spline
                *  T3_clear : Innermost temperature spline for clear atmosphere
                *  T2_clear : Middle temperature spline for clear atmosphere
                *  T1_clear : Outer temperature spline for clear atmosphere
                *  alpha : power law index in tau = delta * press_cgs**alpha
                *  alpha_clear : power law index in tau = delta * press_cgs**alpha for clear atmosphere
                *  log_delta : proportionality factor in tau = delta * press_cgs**alpha
                *  log_delta_clear : proportionality factor in tau = delta * press_cgs**alpha for clear atmosphere
                *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                *  Fe/H : Metallicity
                *  C/O : Carbon to oxygen ratio
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two  columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement : bool
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed emission spectrum [W/m2/micron]
        contr_em : Optional, ArrayLike
            Emission contribution function, relative contributions for each wavelength and pressure level.
            Only returns the contribution of the clear atmosphere component.

    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    # Not sure how to deal with having 2 separate contribution function
    contribution = False

    if "contribution" in parameters.keys():
        contribution = parameters["contribution"].value

    # Priors for these parameters are implemented here, as they depend on each other
    t3 = (
        (3.0 / 4.0 * parameters["T_int"].value ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25
    ) * (1.0 - parameters["T3"].value)
    t2 = t3 * (1.0 - parameters["T2"].value)
    t1 = t2 * (1.0 - parameters["T1"].value)
    delta = ((10.0 ** (-3.0 + 5.0 * parameters["log_delta"].value)) * 1e6) ** (
        -parameters["alpha"].value
    )
    temp_arr = jnp.array([t1, t2, t3])

    t3_clear = (
        (3.0 / 4.0 * parameters["T_int"].value ** 4.0 * (0.1 + 2.0 / 3.0)) ** 0.25
    ) * (1.0 - parameters["T3_clear"].value)
    t2_clear = t3_clear * (1.0 - parameters["T2_clear"].value)
    t1_clear = t2_clear * (1.0 - parameters["T1_clear"].value)
    temps_clear = jnp.array([t1_clear, t2_clear, t3_clear])  # noqa: F841
    delta_clear = (  # noqa: F841
        (10.0 ** (-3.0 + 5.0 * parameters["log_delta_clear"].value)) * 1e6
    ) ** (-parameters["alpha_clear"].value)
    gravity, planet_radius = _compute_gravity(parameters)

    temperatures = (
        molliere_2020_temperature_profile(
            temperature_nodes=temp_arr,
            delta=delta,
            alpha=parameters["alpha"].value,
            t_internal=parameters["T_int"].value,
            pressures=p_use,
            metallicity=parameters["M/H"].value,
            co_ratio=parameters["C/O"].value,
            conv=True,  # conv
        )
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    t_clear = molliere_2020_temperature_profile(
            temperature_nodes=temp_arr,
            delta=delta,
            alpha=parameters["alpha"].value,
            t_internal=parameters["T_int"].value,
            pressures=p_use,
            metallicity=parameters["M/H"].value,
            co_ratio=parameters["C/O"].value,
            conv=True,  # conv
        )
    abundances_clear, mmw_clear, small_index_clear, p_bases_clear = (
        get_abundances_from_legacy_dict(
            pressures=PGLOBAL[small_index],
            temperatures=t_clear,
            parameters=parameters,
            line_species=prt_object.line_species,
            cloud_species=prt_object.cloud_species,
            adaptive_mesh_refinement=False,
        )
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use
    if pressures.shape[0] != prt_object.pressures.shape[0]:
        return None, None

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    wlen_model, flux, _ = prt_object.calculate_flux(
        temperatures=temperatures,
        mass_fractions=abundances,
        reference_gravity=gravity,
        mean_molar_masses=mmw,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_particle_radius_distribution_std=sigma_lnorm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_particle_radius_distribution=distribution,
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_cloudy = flux2irradiance(
        f_lambda, planet_radius, parameters["system_distance"].value
    )

    # Set the cloud abundances to 0 for clear case
    for cloud in prt_object.cloud_species:
        cname = cloud.split("_")[0]
        abundances_clear[cname] = jnp.zeros_like(temperatures)

    wlen_model, flux, additional_outputs = prt_object.calculate_flux(
        temperatures=t_clear,
        mass_fractions=abundances_clear,
        reference_gravity=gravity,
        mean_molar_masses=mmw_clear,
    )

    # Getting the model into correct units
    wlen_model *= 1e4  # cm to um
    f_lambda = flux * 1e-7  # erg.s-1.cm-2/cm to W.m-2/um

    spectrum_model_clear = flux2irradiance(
        f_lambda, planet_radius, parameters["system_distance"].value
    )

    # Patchiness fraction
    patchiness = parameters["patchiness"].value
    spectrum_model = (patchiness * spectrum_model_cloudy) + (
        (1 - patchiness) * spectrum_model_clear
    )

    if contribution:
        return wlen_model, spectrum_model, additional_outputs["emission_contribution"]

    return wlen_model, spectrum_model


@mark_legacy_model_function
def legacy_guillot_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Emission spectrum calculation for the Guillot 2010 temperature profile.
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    gravity, planet_radius = _compute_gravity(parameters)
    gamma = _resolve_guillot_gamma(parameters)

    temperatures = guillot_global_temperature_profile(
        p_use,
        10 ** parameters["log_kappa_IR"].value,
        gamma,
        gravity,
        parameters["T_int"].value,
        parameters["T_equ"].value,
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_guillot_emission_add_gaussian_temperature(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Emission spectrum calculation for the Guillot 2010 temperature profile.
    (Dis)equilibrium or free chemistry, can be used. The use of easychem for on-the-fly (dis)equilibrium
    chemistry calculations is supported, but is currently under development.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species

                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, Emission contribution function, relative contributions for each wavelength and pressure level.

    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    gravity, planet_radius = _compute_gravity(parameters)
    gamma = _resolve_guillot_gamma(parameters)

    temperatures = guillot_global_temperature_profile(
        p_use,
        10 ** parameters["log_kappa_IR"].value,
        gamma,
        gravity,
        parameters["T_int"].value,
        parameters["T_equ"].value,
    )

    def gaussian(x, scale, mu, sig):
        return scale * jnp.exp(-jnp.power((x - mu) / sig, 2.0) / 2)

    temperature_addition = gaussian(
        jnp.log10(p_use),
        parameters["temperature_peak"].value,
        parameters["temperature_location_log_pressure"].value,
        parameters["temperature_width_log_pressure"].value,
    )
    temperatures += temperature_addition
    safe_inds = jnp.where(temperatures < 1.0)
    temperatures[safe_inds] = 1.0

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_guillot_patchy_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Deprecated, to be removed in future version
    """
    return legacy_guillot_emission(
        prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement
    )


@mark_legacy_model_function
def legacy_interpolated_profile_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    This model computes a emission spectrum based a spline temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  nnodes : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  T{i}  : One parameter for each temperature node
                *  gamma : weight for penalizing the profile curvature

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    gravity, planet_radius = _compute_gravity(parameters)

    temp_arr = jnp.array(
        [parameters[f"T{i}"].value for i in range(parameters["nnodes"].value + 2)]
    )
    if "linear" in parameters.keys():
        temperatures, log_prior_weight = linear_spline_temperature_profile(
            p_use, temp_arr, parameters["gamma"].value, parameters["nnodes"].value
        )
    else:
        temperatures, log_prior_weight = cubic_spline_temperature_profile(
            p_use, temp_arr, parameters["gamma"].value, parameters["nnodes"].value
        )
    parameters["log_prior_weight"].value = log_prior_weight
    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )
    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_gradient_profile_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  N_layers : number of nodes to interplate, excluding the first and last points.
                            so the total number of nodes is nnodes + 2
                *  T_bottom : temperature at the base of the atmosphere
                *  PTslope_* : temperature gradient for each of the n_layers between which the profile is interpolated.

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    gravity, planet_radius = _compute_gravity(parameters)

    num_layer = parameters["N_layers"].value
    # 1.4 assemble the P-T slopes for these layers
    # Build directly: an all-NaN initialiser + scatter-fill is a forward NaN
    # that poisons reverse-mode gradients via 0*NaN in the multiply's tangent.
    layer_pt_slopes = jnp.stack(
        [parameters[f"PTslope_{num_layer - index}"].value for index in range(num_layer)]
    )

    top_of_atmosphere_presure = -3
    bottom_of_atmosphere_presure = 3

    if "top_of_atmosphere_pressure" in parameters.keys():
        top_of_atmosphere_presure = parameters["top_of_atmosphere_pressure"].value
    if "bottom_of_atmosphere_pressure" in parameters.keys():
        bottom_of_atmosphere_presure = parameters["bottom_of_atmosphere_pressure"].value
    temperatures = zhang_2023_temperature_profile(
        p_use,
        num_layer,
        layer_pt_slopes,
        parameters["T_bottom"].value,
        top_of_atmosphere_pressure=top_of_atmosphere_presure,
        bottom_of_atmosphere_pressure=bottom_of_atmosphere_presure,
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_power_law_profile_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    This model computes a emission spectrum based a gradient temperature-pressure profile (Zhang 2023).
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  alpha : power law slope for the temperture profile
                *  T_0 : multiplicative factor for the power law slope

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the emission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    gravity, planet_radius = _compute_gravity(parameters)

    num_layer = parameters["N_layers"].value
    # 1.4 assemble the P-T slopes for these layers
    # Build directly: an all-NaN initialiser + scatter-fill is a forward NaN
    # that poisons reverse-mode gradients via 0*NaN in the multiply's tangent.
    layer_pt_slopes = jnp.stack(  # noqa: F841
        [parameters[f"PTslope_{num_layer - index}"].value for index in range(num_layer)]
    )

    temperatures = power_law_temperature_profile(
        p_use, parameters["alpha"].value, parameters["T_0"].value
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Get cloud properties
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    # Calculate the spectrum, wavelength grid, and contribution function
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_guillot_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Transmission Model, Guillot Profile

    This model computes a transmission spectrum based on the Guillot profile
    Either free or (dis)equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  T_int : Interior temperature of the planet [K]
                *  T_equ : Equilibrium temperature of the planet
                *  gamma : Guillot gamma parameter
                *  log_kappa_IR : The log of the ratio between the infrared and optical opacities

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )
    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value
    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)
    gamma = _resolve_guillot_gamma(parameters)

    temperatures = guillot_global_temperature_profile(
        p_use,
        10 ** parameters["log_kappa_IR"].value,
        gamma,
        gravity,
        parameters["T_int"].value,
        parameters["T_equ"].value,
    )

    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes_from_legacy_dict(parameters)
    )
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_guillot_patchy_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Deprecated
    """
    return legacy_guillot_transmission(
        prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement
    )


@mark_legacy_model_function
def legacy_madhusudhan_seager_emission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Transmission Model, Madhusudhan Seager 2009 Profile

    This model computes a transmission spectrum based on a Guillot temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  log_P_set : Pressure value to contrain the PT profile, defaults to 10 bar.
                *  T_set : temperature at P_set to constrain the PT profile. [K]
                *  log_P3 : (log) Pressure value for the top of the deep atmospheric layer, [bar]
                *  P2 : in range (0,1), sets the pressure level of the middle atmospheric layer
                *  P1 : in range (0,1), sets the pressure level of the top atmospheric layer
                *  alpha_0 : slope of the upper atmospheric layer
                *  alpha_1 : slope of the middle atmospheric layer
                Optional :
                    *  beta : power law for the slopes, default value is 0.5. Not recommended to change!

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )

    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)

    # Set up pressure points, guaranteeing P3>P2>P1 >= P_top
    offset = jnp.log10(p_use[0])
    log_p3 = parameters["log_P3"].value - offset
    log_p2 = log_p3 * (1.0 - parameters["P2"].value)
    log_p1 = log_p2 * (1.0 - parameters["P1"].value)
    pressure_points = [
        p_use[0],
        (log_p1 + offset),
        (log_p2 + offset),
        (log_p3 + offset),
        parameters["log_P_set"].value,
    ]

    alpha_points = [parameters["alpha_0"].value, parameters["alpha_1"].value]
    beta_points = [0.5, 0.5]
    if "beta" in parameters.keys():
        beta_points = [parameters["beta"].value, parameters["beta"].value]

    temperatures = madhusudhan_seager_temperature_profile(
        p_use, pressure_points, parameters["T_set"].value, alpha_points, beta_points
    )

    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties
    # Calculate the spectrum.
    return calculate_emission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_madhusudhan_seager_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Transmission Model, Madhusudhan Seager 2009 Profile

    This model computes a transmission spectrum based on the Madhu & Seager (2009) temperature-pressure profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.
    Chemical clouds can be used, or a simple gray opacity source. This model requires patchy clouds.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  log_P_set : Pressure value to contrain the PT profile, defaults to 10 bar.
                *  T_set : temperature at P_set to constrain the PT profile. [K]
                *  log_P3 : (log) Pressure value for the top of the deep atmospheric layer, [bar]
                *  P2 : in range (0,1), sets the pressure level of the middle atmospheric layer
                *  P1 : in range (0,1), sets the pressure level of the top atmospheric layer
                *  alpha_0 : slope of the upper atmospheric layer
                *  alpha_1 : slope of the middle atmospheric layer
                Optional :
                    *  beta : power law for the slopes, default value is 0.5. Not recommended to change!

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )
    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value
    # Calculate the spectrum
    gravity, planet_radius = _compute_gravity(parameters)

    # Set up pressure points, guaranteeing P3>P2>P1 >= P_top
    offset = jnp.log10(p_use[0])
    log_p3 = parameters["log_P3"].value - offset
    log_p2 = log_p3 * (1.0 - parameters["P2"].value)
    log_p1 = log_p2 * (1.0 - parameters["P1"].value)
    pressure_points = [
        p_use[0],
        (log_p1 + offset),
        (log_p2 + offset),
        (log_p3 + offset),
        parameters["log_P_set"].value,
    ]

    alpha_points = [parameters["alpha_0"].value, parameters["alpha_1"].value]
    beta_points = [0.5, 0.5]
    if "beta" in parameters.keys():
        beta_points = [parameters["beta"].value, parameters["beta"].value]

    temperatures = madhusudhan_seager_temperature_profile(
        p_use, pressure_points, parameters["T_set"].value, alpha_points, beta_points
    )

    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]

    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes_from_legacy_dict(parameters)
    )
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_madhu_seager_patchy_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Deprecated
    """
    return legacy_madhusudhan_seager_transmission(
        prt_object, parameters, pt_plot_mode, adaptive_mesh_refinement
    )


@mark_legacy_model_function
def legacy_isothermal_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    Equilibrium Chemistry Transmission Model, Isothermal Profile

    This model computes a transmission spectrum based on an isothermal temperature-pressure profile.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  temperature : Interior temperature of the planet [K]

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][.R$RESOLUTION] : The log mass fraction abundance of the species

                Either:
                  * [log_]Pcloud : The (log) pressure at which to place the gray cloud opacity.
                Or:
                  *  fsed : sedimentation parameter - can be unique to each cloud type
                  One of:
                    *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                    *  b_hans : Width of cloud particle size distribution (hansen)
                  One of:
                    *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                    *  log_kzz : Vertical mixing parameter
                  One of
                    *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                    *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the transmission contribution function
                  *  power_law_opacity_coefficient : gamma, power law slope for a rayleigh-like haze
                  *  haze_factor : multiplicative scaling factor for the strength of the rayleigh haze
                  *  power_law_opacity_350nm : strength of the rayleigh haze at 350 nm.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure
            level.
            Only the clear atmosphere contribution is returned if patchy clouds are considered.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )
    reference_pressure = 100.0

    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value

    # Make the P-T profile
    temperatures = isothermal_temperature_profile(
        p_use, parameters["temperature"].value
    )
    gravity, planet_radius = _compute_gravity(parameters)

    # Make the abundance profile
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )
    if abundances is None:
        return None, None
    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes_from_legacy_dict(parameters)
    )
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


@mark_legacy_model_function
def legacy_power_law_profile_transmission(
    prt_object, parameters, pt_plot_mode=False, adaptive_mesh_refinement=False
):
    """
    This model computes a transmission spectrum based on a gradient power-law pressure-temperature profile.
    Either free or equilibrium chemistry can be used, together with a range of cloud parameterizations.
    It is possible to use free abundances for some species and equilibrium chemistry for the remainder.

    Args:
        prt_object : object
            An instance of the pRT class, with optical properties as defined in the RunDefinition.
        parameters : dict
            Dictionary of required parameters:
                *  distance_to_system : Distance to the planet in [cm]
                Two of
                  *  log_g : Log of surface gravity
                  *  planet_radius : planet radius [cm]
                  *  mass : planet mass [g]
                *  alpha : power law slope for the temperture profile
                *  T_0 : multiplicative factor for the power law slope

                Either:
                  *  log_pquench : Pressure at which CO, CH4 and H2O abundances become vertically constant
                  *  Fe/H : Metallicity
                  *  C/O : Carbon to oxygen ratio
                Or:
                  * $SPECIESNAME[_$DATABASE][_R_$RESOLUTION] : The log mass fraction abundance of the species
                Optional:
                *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
                One of
                  *  eq_scaling_* : Scaling factor for equilibrium cloud abundances.
                  *  log_X_cb_: cloud mass fraction abundance
                Optional
                  *  contribution : return the emission contribution function
                  *  patchiness : Cloud coverage fraction, mixes two columns with different cloud properties.
                  *  remove_cloud_species : Specifies which cloud species to remove for the clear atmosphere column.
                  *  T_disk_blackbody : Temperature of a blackbody circumplanetary disk component.
                  *  disk_radius : Radius [cm] of a blackbody circumplanetary disk component.
        pt_plot_mode : bool
            Return only the pressure-temperature profile for plotting. Evaluate mode only.
        adaptive_mesh_refinement :
            Adaptive mesh refinement. Use the high resolution pressure grid around the cloud base.

    Returns:
        wlen_model : jnp.array
            Wavlength array of computed model, not binned to data [um]
        spectrum_model : jnp.array
            Computed transmission spectrum planet_radius**2/stellar_radius**2
        contr-em : ArrayLike
            Optional, the transmission contribution function, relative contributions for each wavelength and pressure level.
    """
    p_use, PGLOBAL = initialize_pressure_from_legacy_dict(
        prt_object.pressures / 1e6, parameters, adaptive_mesh_refinement
    )
    gravity, planet_radius = _compute_gravity(parameters)

    reference_pressure = 100.0
    if "reference_pressure" in parameters.keys():
        reference_pressure = parameters["reference_pressure"].value

    temperatures = power_law_temperature_profile(
        p_use, parameters["alpha"].value, parameters["T_0"].value
    )

    # If in evaluation mode, and PTs are supposed to be plotted
    abundances, mmw, small_index, p_bases = get_abundances_from_legacy_dict(
        pressures=p_use,
        temperatures=temperatures,
        parameters=parameters,
        line_species=prt_object.line_species,
        cloud_species=prt_object.cloud_species,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
    )

    if abundances is None:
        return None, None

    if pt_plot_mode:
        return p_use[small_index], temperatures[small_index]
    if adaptive_mesh_refinement:
        temperatures = temperatures[small_index]
        pressures = PGLOBAL[small_index]
        mmw = mmw[small_index]
        prt_object.pressures = pressures * 1e6
    else:
        pressures = p_use

    # Setup transmission spectrum clouds and hazes
    pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm = (
        clouds.setup_simple_clouds_hazes_from_legacy_dict(parameters)
    )
    # Setup physical clouds (with real scattering constants)
    cloud_properties = clouds.setup_clouds_from_legacy_dict(
        pressures, parameters, prt_object.cloud_species
    )
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties
    # Calculate the spectrum.
    return calculate_transmission_spectrum(
        prt_object=prt_object,
        parameters=parameters,
        temperatures=temperatures,
        abundances=abundances,
        gravity=gravity,
        mean_molar_masses=mmw,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=pcloud,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        distribution=distribution,
    )


def add_blackbody_cpd_model(parameters, wavelengths):
    """
    Calculates the flux of a blackbody with area 4*pi*disk_radius^2 and temperature T_disk_blackbody.
    This is in units of W/m2/micron, and can be added to a planetary spectrum to model the contribution
    of a circumplanetary disk

    Args:
        parameters (dict): dictionary of atmospheric and disk parameters
        wavelengths (ArrayLike): Wavelength grid of atmospheric model in micron

    Returns:
        blackbody_spectrum (ArrayLike): 1D Planck emission spectrum for a circular CPD.
    """
    if "T_disk_blackbody" in parameters.keys():
        blackbody_spectrum = (
            planck_function_cm(parameters["T_disk_blackbody"].value, wavelengths * 1e-4)
            * 1e-7
        )
        blackbody_spectrum = (
            blackbody_spectrum
            * (parameters["disk_radius"].value / parameters["system_distance"].value)
            ** 2
        )
        return blackbody_spectrum


def initialize_pressure(press, parameters, adaptive_mesh_refinement):
    del parameters, adaptive_mesh_refinement
    return press, press


def initialize_pressure_from_legacy_dict(press, parameters, adaptive_mesh_refinement):
    """Legacy wrapper for :func:`initialize_pressure`.

    Converts a legacy parameter dictionary (values accessed via ``.value``) to a
    plain-value mapping before forwarding to :func:`initialize_pressure`.

    Args:
        parameters : dict
            Legacy parameter dictionary where each value is an object with a ``.value`` attribute.
        All other arguments are forwarded unchanged to :func:`initialize_pressure`.
    """
    plain_parameters = {name: param.value for name, param in parameters.items()}
    return initialize_pressure(press, plain_parameters, adaptive_mesh_refinement)


def _as_auxiliary_outputs(outputs):
    if outputs is None:
        return {}

    if isinstance(outputs, dict):
        return dict(outputs)

    return {"runtime_output": outputs}


_EMISSION_RUNTIME_AUXILIARY_OUTPUT_EXCLUSIONS = frozenset(
    {"stellar_intensities", "angle_dependent_intensities"}
)


def _as_filtered_auxiliary_outputs(outputs, excluded_keys=()):
    auxiliary_outputs = _as_auxiliary_outputs(outputs)

    if not excluded_keys:
        return auxiliary_outputs

    return {
        key: value
        for key, value in auxiliary_outputs.items()
        if key not in excluded_keys
    }


def _is_scalar_like(value):
    if value is None:
        return False

    if isinstance(value, Number):
        return True

    if isinstance(value, (dict, list, tuple, str, bytes)):
        return False

    try:
        return jnp.ndim(value) == 0
    except TypeError:
        return False


def _build_model_context_from_parameters(
    name,
    mode,
    prt_object,
    parameters,
    adaptive_mesh_refinement=False,
    variability_atmospheric_column_model_flux_return_mode=False,
    model_metadata=None,
):
    (
        return_contribution,
        return_opacities,
        return_photosphere_radius,
        return_rosseland_optical_depths,
        return_radius_hydrostatic_equilibrium,
        return_cloud_contribution,
        return_abundances,
        _return_any,
    ) = get_calculate_flux_return_values(parameters)

    from petitRADTRANS.retrieval.parameter_aliases import normalize_parameter_name

    parameter_units = {
        normalize_parameter_name(parameter_name): getattr(parameter, "units", None)
        for parameter_name, parameter in parameters.items()
        if getattr(parameter, "units", None) is not None
    }

    return ModelContext(
        name=name,
        mode=mode,
        radtrans=prt_object,
        adaptive_mesh_refinement=adaptive_mesh_refinement,
        scattering_in_emission=getattr(prt_object, "scattering_in_emission", False),
        parameter_units=parameter_units,
        variability_atmospheric_column_model_flux_return_mode=(
            variability_atmospheric_column_model_flux_return_mode
        ),
        return_contribution=return_contribution,
        return_opacities=return_opacities,
        return_photosphere_radius=return_photosphere_radius,
        return_rosseland_optical_depths=return_rosseland_optical_depths,
        return_radius_hydrostatic_equilibrium=return_radius_hydrostatic_equilibrium,
        return_cloud_contribution=return_cloud_contribution,
        return_abundances=return_abundances,
        model_metadata={} if model_metadata is None else dict(model_metadata),
    )


def _normalize_model_output(model_output, pt_plot_mode=False):
    if isinstance(model_output, ModelResult):
        return model_output

    if model_output is None:
        return ModelResult(kind="invalid")

    if not isinstance(model_output, (tuple, list)):
        raise TypeError(f"unsupported model output type '{type(model_output)}'")

    if len(model_output) < 2:
        raise ValueError("model output must contain at least two elements")

    first = _normalize_legacy_output_value(model_output[0])
    second = _normalize_legacy_output_value(model_output[1])

    if first is None or second is None:
        return ModelResult(kind="invalid")

    if pt_plot_mode:
        return ModelResult(kind="pt_profile", pressures=first, temperatures=second)

    additional_log_likelihood = 0.0
    atmospheric_column_fluxes = None
    auxiliary_outputs = {}

    if len(model_output) >= 3:
        third = _normalize_legacy_output_value(model_output[2])
        if third is None:
            pass
        elif _is_scalar_like(third):
            additional_log_likelihood = third
        else:
            auxiliary_outputs["legacy_output_2"] = third

    if len(model_output) >= 4:
        fourth = _normalize_legacy_output_value(model_output[3])
        if fourth is None:
            pass
        elif atmospheric_column_fluxes is None and not _is_scalar_like(fourth):
            atmospheric_column_fluxes = fourth
        else:
            auxiliary_outputs["legacy_output_3"] = fourth

    if len(model_output) > 4:
        auxiliary_outputs["legacy_extra_outputs"] = tuple(
            _normalize_legacy_output_value(value) for value in model_output[4:]
        )

    return ModelResult(
        kind="spectrum",
        wavelengths=first,
        spectrum=second,
        additional_log_likelihood=additional_log_likelihood,
        auxiliary_outputs=auxiliary_outputs,
        atmospheric_column_fluxes=atmospheric_column_fluxes,
    )


def _model_result_to_legacy_output(model_result: ModelResult):
    if model_result.kind == "invalid":
        return None, None

    if model_result.kind == "pt_profile":
        return model_result.pressures, model_result.temperatures

    if model_result.atmospheric_column_fluxes is not None:
        return (
            model_result.wavelengths,
            model_result.spectrum,
            model_result.additional_log_likelihood,
            model_result.atmospheric_column_fluxes,
        )

    if model_result.auxiliary_outputs:
        return (
            model_result.wavelengths,
            model_result.spectrum,
            model_result.auxiliary_outputs,
        )

    if model_result.additional_log_likelihood not in (0, 0.0, None):
        return (
            model_result.wavelengths,
            model_result.spectrum,
            model_result.additional_log_likelihood,
        )

    return model_result.wavelengths, model_result.spectrum


def run_legacy_model_adapter(
    model_function,
    model_context: ModelContext,
    physical_params: PhysicalParams,
    *,
    pt_plot_mode: bool = False,
) -> ModelResult:
    legacy_parameters = {
        name: SimpleNamespace(value=value)
        for name, value in physical_params.ordered_items()
    }
    _base_index: dict[str, str] = {}
    for name in legacy_parameters:
        base = name.split(".", 1)[0]
        _base_index.setdefault(base, name)

    radtrans = model_context.radtrans
    if radtrans is not None:
        for species in getattr(radtrans, "line_species", ()):
            if species not in legacy_parameters:
                base = species.split(".", 1)[0]
                original = _base_index.get(base)
                if original is not None:
                    legacy_parameters[species] = legacy_parameters[original]

    kwargs = {"pt_plot_mode": pt_plot_mode}

    sig = inspect.signature(model_function)
    if "adaptive_mesh_refinement" in sig.parameters:
        kwargs["adaptive_mesh_refinement"] = model_context.adaptive_mesh_refinement

    model_output = model_function(
        radtrans,
        legacy_parameters,
        **kwargs,
    )
    return _normalize_model_output(model_output, pt_plot_mode=pt_plot_mode)


def _infer_model_function_contract(model_function, declared_contract: str) -> str:
    """Resolve the model contract using explicit registration only.

    Contract detection follows strict precedence:
    1. Explicit ``_prt_model_contract`` attribute (set by :func:`mark_legacy_model_function`).
    2. The caller-supplied *declared_contract*.
    3. Fallback to :data:`MODEL_CONTRACT_LEGACY`.

    No signature inspection is performed — models must opt in to the
    runtime-native contract via the decorator or by setting the attribute
    directly.
    """
    if model_function is None:
        return declared_contract

    explicit_contract = getattr(model_function, _MODEL_CONTRACT_ATTRIBUTE, None)
    if explicit_contract is not None:
        return explicit_contract

    if declared_contract == MODEL_CONTRACT_LEGACY:
        return declared_contract

    return MODEL_CONTRACT_DIFFERENTIABLE


def coerce_model_generating_function(
    model_function, declared_contract: str = MODEL_CONTRACT_LEGACY
):
    resolved_contract = _infer_model_function_contract(
        model_function, declared_contract
    )

    if model_function is None:
        return resolved_contract, model_function

    if resolved_contract != MODEL_CONTRACT_LEGACY:
        return resolved_contract, model_function

    wrapped_model_function = getattr(
        model_function, _WRAPPED_LEGACY_MODEL_ATTRIBUTE, None
    )
    if wrapped_model_function is not None:
        return MODEL_CONTRACT_LEGACY_ADAPTER, wrapped_model_function

    @wraps(model_function)
    def runtime_ready_model_function(
        model_context, physical_params, pt_plot_mode=False
    ):
        return run_legacy_model_adapter(
            model_function,
            model_context,
            physical_params,
            pt_plot_mode=pt_plot_mode,
        )

    setattr(
        runtime_ready_model_function,
        _MODEL_CONTRACT_ATTRIBUTE,
        MODEL_CONTRACT_LEGACY_ADAPTER,
    )
    setattr(
        model_function, _WRAPPED_LEGACY_MODEL_ATTRIBUTE, runtime_ready_model_function
    )

    return MODEL_CONTRACT_LEGACY_ADAPTER, runtime_ready_model_function


def run_runtime_model_adapter(
    model_function,
    model_context: ModelContext,
    physical_params: PhysicalParams,
    *,
    pt_plot_mode: bool = False,
) -> ModelResult:
    model_output = model_function(
        model_context,
        physical_params,
        pt_plot_mode=pt_plot_mode,
    )
    return _normalize_model_output(model_output, pt_plot_mode=pt_plot_mode)


def make_unified_model(fn, legacy_fn=None):
    """Return a single callable that dispatches to *fn* or
    *legacy_fn* based on the type of the first positional argument.

    When the first argument is a :class:`~petitRADTRANS.retrieval.runtime.ModelContext`
    instance, the differentiable code path is taken.  Any other type routes to
    the legacy calling convention ``(prt_object, parameters, ...)``.  If no
    *legacy_fn* is supplied and the first argument is not a
    :class:`ModelContext`, a :class:`TypeError` is raised.

    The dispatcher carries ``_prt_model_contract == MODEL_CONTRACT_DIFFERENTIABLE``
    so that :func:`coerce_model_generating_function` and the retrieval runtime
    treat it as a first-class differentiable function.  The constituent
    functions remain accessible as ``.___`` and ``.__legacy__``
    attributes.
    """

    @wraps(fn)
    def dispatcher(
        first_arg,
        second_arg,
        pt_plot_mode=False,
        adaptive_mesh_refinement=False,
        **kwargs,
    ):
        if isinstance(first_arg, ModelContext):
            return fn(first_arg, second_arg, pt_plot_mode=pt_plot_mode)
        if legacy_fn is None:
            raise TypeError(
                f"{fn.__name__}: no legacy counterpart registered; "
                f"first argument must be a ModelContext, got "
                f"{type(first_arg).__name__}."
            )
        return legacy_fn(
            first_arg,
            second_arg,
            pt_plot_mode=pt_plot_mode,
            adaptive_mesh_refinement=adaptive_mesh_refinement,
            **kwargs,
        )

    setattr(dispatcher, _MODEL_CONTRACT_ATTRIBUTE, MODEL_CONTRACT_DIFFERENTIABLE)
    dispatcher.___ = fn
    dispatcher.__legacy__ = legacy_fn
    return dispatcher


def _initialize_runtime_pressure_grid(model_context: ModelContext, physical_params):
    del physical_params
    base_pressures = model_context.radtrans.pressures / 1e6
    return base_pressures, base_pressures


def _resolve_runtime_abundance_state(
    model_context: ModelContext,
    physical_params,
    p_use,
    p_global,
    temperatures,
    *,
    temperature_builder=None,
    pt_plot_mode: bool = False,
):
    del p_global

    pressures = p_use

    if model_context.adaptive_mesh_refinement:
        if temperature_builder is None:
            raise ValueError(
                "temperature_builder is required when adaptive mesh refinement is enabled"
            )

        p_bases = resolve_cloud_base_pressures(
            pressures=p_use,
            temperatures=temperatures,
            parameters=physical_params,
            line_species=model_context.radtrans.line_species,
            cloud_species=model_context.radtrans.cloud_species,
        )
        cloud_base_pressures = _stack_runtime_cloud_base_pressures(
            p_bases,
            p_use.dtype,
        )
        pressures = build_adaptive_pressure_grid(
            p_use,
            cloud_base_pressures,
        )
        temperatures = temperature_builder(pressures)

    if pt_plot_mode:
        return (
            ModelResult(
                kind="pt_profile",
                pressures=pressures,
                temperatures=temperatures,
            ),
            None,
        )

    abundances, mean_molar_masses, small_index, _ = get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters=physical_params,
        line_species=model_context.radtrans.line_species,
        cloud_species=model_context.radtrans.cloud_species,
        adaptive_mesh_refinement=False,
    )

    if abundances is None:
        wavelengths = _radtrans_wavelength_grid(model_context.radtrans)
        return ModelResult.make_nan_result(wavelengths), None

    return None, {
        "pressures": pressures,
        "temperatures": temperatures,
        "abundances": abundances,
        "mean_molar_masses": mean_molar_masses,
        "small_index": small_index,
    }


def _build_runtime_emission_model_inputs(
    physical_params,
    abundance_state,
    gravity,
    planet_radius,
    cloud_properties,
):
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    return ModelInputs(
        pressures=abundance_state["pressures"],
        temperatures=abundance_state["temperatures"],
        abundances=abundance_state["abundances"],
        mean_molar_masses=abundance_state["mean_molar_masses"],
        gravity=gravity,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        cloud_particle_radius_distribution=distribution,
        distance_to_system=physical_params["system_distance"],
        disk_blackbody_temperature=(
            physical_params["T_disk_blackbody"]
            if "T_disk_blackbody" in physical_params.keys()
            else None
        ),
        disk_radius=(
            physical_params["disk_radius"]
            if "disk_radius" in physical_params.keys()
            else None
        ),
        v_band_extinction=(
            physical_params["v_band_extinction"]
            if "v_band_extinction" in physical_params.keys()
            else None
        ),
        v_band_reddening=(
            physical_params["v_band_reddening"]
            if "v_band_reddening" in physical_params.keys()
            else None
        ),
    )


def _build_runtime_transmission_model_inputs(
    physical_params,
    abundance_state,
    gravity,
    planet_radius,
    *,
    reference_pressure,
    opaque_cloud_top_pressure,
    haze_factor,
    power_law_opacity_coefficient,
    power_law_opacity_350nm,
    cloud_properties,
):
    (
        sigma_lnorm,
        cloud_f_sed,
        eddy_diffusion_coefficients,
        cloud_hansen_b,
        cloud_particle_mean_radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    ) = cloud_properties

    return ModelInputs(
        pressures=abundance_state["pressures"],
        temperatures=abundance_state["temperatures"],
        abundances=abundance_state["abundances"],
        mean_molar_masses=abundance_state["mean_molar_masses"],
        gravity=gravity,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        cloud_particle_radius_distribution=distribution,
        stellar_radius=physical_params["stellar_radius"],
    )


def calculate_emission_spectrum_runtime(
    model_context: ModelContext,
    model_inputs: ModelInputs,
) -> ModelResult:
    # Scattering emission solves the Feautrier system on a tensor whose live
    # size is ~ n_g * feautrier_chunk_size * n_angles * n_layers (float64). With
    # the default chunk size (1536 >= the model's frequency-bin count) the whole
    # frequency grid is solved in one shot, which on GPU produced a single
    # ~48 GiB allocation that exhausted the 40 GB A100 (job 1860410). The
    # Feautrier driver chunks the frequency axis with lax.scan -- exact and
    # reverse-mode differentiable -- so a smaller chunk size cuts the per-chunk
    # peak roughly proportionally. A value supplied via RetrievalConfig (carried
    # on the ModelContext) takes precedence over the fallback below.
    _, feautrier_chunk_size = radiative_transfer_core.resolve_feautrier_chunking(
        chunk_feautrier=model_context.chunk_feautrier,
        feautrier_chunk_size=model_context.feautrier_chunk_size,
        fallback_chunk_size=128,
    )

    results = model_context.radtrans.calculate_flux(
        pressures=model_inputs.pressures,
        temperatures=model_inputs.temperatures,
        mass_fractions=model_inputs.abundances,
        reference_gravity=model_inputs.gravity,
        mean_molar_masses=model_inputs.mean_molar_masses,
        cloud_f_sed=model_inputs.cloud_f_sed,
        eddy_diffusion_coefficients=model_inputs.eddy_diffusion_coefficients,
        cloud_particle_radius_distribution_std=model_inputs.sigma_lnorm,
        cloud_hansen_b=model_inputs.cloud_hansen_b,
        cloud_particle_mean_radii=model_inputs.cloud_particle_mean_radii,
        cloud_fraction=model_inputs.cloud_fraction,
        patchy_clouds=model_inputs.patchy_clouds,
        cloud_particle_radius_distribution=model_inputs.cloud_particle_radius_distribution,
        return_contribution=model_context.return_contribution,
        return_cloud_contribution=model_context.return_cloud_contribution,
        return_abundances=model_context.return_abundances,
        return_opacities=model_context.return_opacities,
        chunk_feautrier=model_context.chunk_feautrier,
        adaptive_feautrier_iterations=model_context.adaptive_feautrier_iterations,
        feautrier_chunk_size=feautrier_chunk_size,
    )

    wavelengths_model, flux, additional_outputs = results
    wavelengths_model = wavelengths_model * 1e4
    flux_lambda = flux * 1e-7

    if model_inputs.distance_to_system is None:
        raise ValueError(
            "distance_to_system is required for emission spectrum evaluation"
        )

    # Convert the system distance to cm (matching the cgs planet_radius) before applying the
    # (planet_radius / distance)^2 flux dilution. The factor is None when no unit is declared,
    # in which case the value is assumed to already be in cm.
    distance_to_system = model_inputs.distance_to_system
    distance_factor = _unit_to_cgs_factor(
        model_context.parameter_units.get("system_distance"), "cm"
    )
    if distance_factor is not None:
        distance_to_system = distance_to_system * distance_factor

    spectrum_model = flux2irradiance(
        flux_lambda,
        model_inputs.planet_radius,
        distance_to_system,
    )

    if (
        model_inputs.disk_blackbody_temperature is not None
        and model_inputs.disk_radius is not None
    ):
        blackbody_spectrum = (
            planck_function_cm(
                model_inputs.disk_blackbody_temperature, wavelengths_model * 1e-4
            )
            * 1e-7
        )
        blackbody_spectrum = (
            blackbody_spectrum
            * (model_inputs.disk_radius / distance_to_system) ** 2
        )
        spectrum_model = spectrum_model + blackbody_spectrum

    if (
        model_inputs.v_band_extinction is not None
        and model_inputs.v_band_reddening is not None
    ):
        spectrum_model = apply_ism_ext(
            wavelengths_model,
            spectrum_model,
            model_inputs.v_band_extinction,
            model_inputs.v_band_reddening,
        )

    return ModelResult(
        kind="spectrum",
        wavelengths=wavelengths_model,
        spectrum=spectrum_model,
        auxiliary_outputs=_as_filtered_auxiliary_outputs(
            additional_outputs,
            excluded_keys=_EMISSION_RUNTIME_AUXILIARY_OUTPUT_EXCLUSIONS,
        ),
    )


def calculate_emission_spectra_batched(
    model_context: ModelContext,
    batched_model_inputs: ModelInputs,
) -> ModelResult:
    """Evaluate emission spectra for a batch of model inputs via vmap.

    ``model_context`` (including the ``Radtrans`` object and all static flags)
    is treated as static across the batch.  Only the per-sample arrays inside
    ``batched_model_inputs`` are mapped over.  Each leaf of
    ``batched_model_inputs`` must carry a leading batch dimension.

    This is the preferred entry point when the caller has already constructed
    batched ``ModelInputs`` and wants to run the RT kernel efficiently without
    wrapping the full evaluation pipeline.

    Args:
        model_context: Static evaluation context shared across all samples.
        batched_model_inputs: A ``ModelInputs`` pytree whose leaves each have
            shape ``(batch_size, ...)`` — i.e. the first axis is the batch dim.

    Returns:
        A ``ModelResult`` whose ``wavelengths`` and ``spectrum`` leaves have a
        leading batch dimension ``(batch_size, n_wavelengths)``.
    """
    return vmap(calculate_emission_spectrum_runtime, in_axes=(None, 0))(
        model_context, batched_model_inputs
    )


def calculate_transmission_spectrum_runtime(
    model_context: ModelContext,
    model_inputs: ModelInputs,
) -> ModelResult:
    results = model_context.radtrans.calculate_transit_radii(
        pressures=model_inputs.pressures,
        temperatures=model_inputs.temperatures,
        mass_fractions=model_inputs.abundances,
        reference_gravity=model_inputs.gravity,
        mean_molar_masses=model_inputs.mean_molar_masses,
        planet_radius=model_inputs.planet_radius,
        reference_pressure=model_inputs.reference_pressure,
        opaque_cloud_top_pressure=model_inputs.opaque_cloud_top_pressure,
        cloud_particle_radius_distribution_std=model_inputs.sigma_lnorm,
        cloud_particle_mean_radii=model_inputs.cloud_particle_mean_radii,
        cloud_f_sed=model_inputs.cloud_f_sed,
        eddy_diffusion_coefficients=model_inputs.eddy_diffusion_coefficients,
        haze_factor=model_inputs.haze_factor,
        power_law_opacity_coefficient=model_inputs.power_law_opacity_coefficient,
        power_law_opacity_350nm=model_inputs.power_law_opacity_350nm,
        cloud_hansen_b=model_inputs.cloud_hansen_b,
        cloud_fraction=model_inputs.cloud_fraction,
        patchy_clouds=model_inputs.patchy_clouds,
        cloud_particle_radius_distribution=model_inputs.cloud_particle_radius_distribution,
        return_contribution=model_context.return_contribution,
        return_cloud_contribution=model_context.return_cloud_contribution,
        return_radius_hydrostatic_equilibrium=model_context.return_radius_hydrostatic_equilibrium,
        return_abundances=model_context.return_abundances,
        return_opacities=model_context.return_opacities,
    )

    wavelengths_model, transit_radii, additional_outputs = results
    wavelengths_model = wavelengths_model * 1e4

    if model_inputs.stellar_radius is None:
        raise ValueError(
            "stellar_radius is required for transmission spectrum evaluation"
        )

    spectrum_model = (transit_radii / model_inputs.stellar_radius) ** 2.0

    return ModelResult(
        kind="spectrum",
        wavelengths=wavelengths_model,
        spectrum=spectrum_model,
        auxiliary_outputs=_as_auxiliary_outputs(additional_outputs),
    )


def calculate_emission_spectrum(
    prt_object,
    parameters,
    temperatures,
    abundances,
    gravity,
    mean_molar_masses,
    planet_radius,
    sigma_lnorm,
    cloud_particle_mean_radii,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    cloud_hansen_b,
    cloud_fraction,
    patchy_clouds,
    distribution,
):
    """
    Calls Radtrans.calculate_flux to compute the emission spectrum of an atmosphere.
    This function automatically checks if patchiness is included in the retrieval, and
    mixes the clear and cloudy columns. Patchiness can be applied to all of the cloud species,
    or individual clouds can be chosen using the remove_cloud_species parameter.
    A circumplanetary disk model is optionally included, modelled as a blackbody with some temperature
    T_disk_blackbody and a radius disk_radius.

    Args:
        prt_object (Radtrans): The Radtrans object used to calculate the spectrum
        parameters (dict): Dictionary of atmospheric parameters.
        temperatures (ArrayLike): Array of temperatures for each pressure level in the atmosphere
        abundances (dict): Dictionary of molecular mass fraction abundances for each level in the atmosphere
        gravity (ArrayLike): Gravitational acceleration at each pressure level
        mean_molar_masses (ArrayLike): Mean molecular mass at each pressure level
        planet_radius (float): Planet radius in cm
        sigma_lnorm (float): Width of the cloud particle size distribution (log-normal)
        cloud_particle_mean_radii (ArrayLike): Mean particle radius
        cloud_f_sed (ArrayLike): Sedimentation fraction
        eddy_diffusion_coefficients (ArrayLike): Vertical mixing strength (Kzz)
        cloud_hansen_b (ArrayLike): Cloud particle distribution width, hansen distsribution
        cloud_fraction (float) : fraction of planet covered by clouds
        patchy_clouds (list(str)) : Which clouds are patchy
        distribution (string): Which cloud particle size distribution to use
    """
    model_context = _build_model_context_from_parameters(
        name="legacy_emission_spectrum",
        mode="emission",
        prt_object=prt_object,
        parameters=parameters,
    )
    model_inputs = ModelInputs(
        pressures=getattr(prt_object, "pressures", None),
        temperatures=temperatures,
        abundances=abundances,
        mean_molar_masses=mean_molar_masses,
        gravity=gravity,
        planet_radius=planet_radius,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        cloud_particle_radius_distribution=distribution,
        distance_to_system=parameters["system_distance"].value,
        disk_blackbody_temperature=(
            parameters["T_disk_blackbody"].value
            if "T_disk_blackbody" in parameters.keys()
            else None
        ),
        disk_radius=(
            parameters["disk_radius"].value
            if "disk_radius" in parameters.keys()
            else None
        ),
        v_band_extinction=(
            parameters["v_band_extinction"].value
            if "v_band_extinction" in parameters.keys()
            else None
        ),
        v_band_reddening=(
            parameters["v_band_reddening"].value
            if "v_band_reddening" in parameters.keys()
            else None
        ),
    )

    return _model_result_to_legacy_output(
        calculate_emission_spectrum_runtime(model_context, model_inputs)
    )


def calculate_transmission_spectrum(
    prt_object,
    parameters,
    temperatures,
    abundances,
    gravity,
    mean_molar_masses,
    planet_radius,
    reference_pressure,
    opaque_cloud_top_pressure,
    sigma_lnorm,
    cloud_particle_mean_radii,
    cloud_f_sed,
    eddy_diffusion_coefficients,
    haze_factor,
    power_law_opacity_coefficient,
    power_law_opacity_350nm,
    cloud_hansen_b,
    cloud_fraction,
    patchy_clouds,
    distribution,
):
    """_summary_

    Args:
        prt_object (Radtrans): The Radtrans object used to calculate the spectrum
        parameters (dict): Dictionary of atmospheric parameters.
        temperatures (ArrayLike): Array of temperatures for each pressure level in the atmosphere
        abundances (dict): Dictionary of molecular mass fraction abundances for each level in the atmosphere
        gravity (ArrayLike): Gravitational acceleration at each pressure level
        mean_molar_masses (ArrayLike): Mean molecular mass at each pressure level
        planet_radius (float):Planet radius in cm
        reference_pressure (float): Pressure at which the planet radius is defined
        opaque_cloud_top_pressure (float): Pressure where an opaque grey cloud deck is placed
        sigma_lnorm (float): Width of the cloud particle size distribution (log-normal)
        cloud_particle_mean_radii (ArrayLike): Mean particle radius
        cloud_f_sed (ArrayLike): Sedimentation fraction
        eddy_diffusion_coefficients (ArrayLike): Vertical mixing strength (Kzz)
        haze_factor (float): Multiplicative factor on the strength of a power law haze slope
        power_law_opacity_coefficient (float): Exponent for the slope of a power law haze
        power_law_opacity_350nm (float): Strength of the power law scattering at 350nm
        cloud_hansen_b (ArrayLike): Cloud particle distribution width, hansen distsribution
        cloud_fraction (float) : fraction of planet covered by clouds
        patchy_clouds (list(str)) : Which clouds are patchy
        distribution (string): Log normal or hansen particle size distribution
    Returns:
        _type_: _description_
    """
    model_context = _build_model_context_from_parameters(
        name="legacy_transmission_spectrum",
        mode="transmission",
        prt_object=prt_object,
        parameters=parameters,
    )
    model_inputs = ModelInputs(
        pressures=getattr(prt_object, "pressures", None),
        temperatures=temperatures,
        abundances=abundances,
        mean_molar_masses=mean_molar_masses,
        gravity=gravity,
        planet_radius=planet_radius,
        reference_pressure=reference_pressure,
        opaque_cloud_top_pressure=opaque_cloud_top_pressure,
        sigma_lnorm=sigma_lnorm,
        cloud_particle_mean_radii=cloud_particle_mean_radii,
        cloud_f_sed=cloud_f_sed,
        eddy_diffusion_coefficients=eddy_diffusion_coefficients,
        haze_factor=haze_factor,
        power_law_opacity_coefficient=power_law_opacity_coefficient,
        power_law_opacity_350nm=power_law_opacity_350nm,
        cloud_hansen_b=cloud_hansen_b,
        cloud_fraction=cloud_fraction,
        patchy_clouds=patchy_clouds,
        cloud_particle_radius_distribution=distribution,
        stellar_radius=parameters["stellar_radius"].value,
    )

    return _model_result_to_legacy_output(
        calculate_transmission_spectrum_runtime(model_context, model_inputs)
    )
# ---------------------------------------------------------------------------
# Unified entry-points
# ---------------------------------------------------------------------------
# Each name below accepts *either* the JAX-native calling convention
# ``(ModelContext, PhysicalParams, pt_plot_mode=False)`` *or* the legacy
# convention ``(prt_object, parameters_dict, pt_plot_mode, adaptive_mesh_refinement)``.
# Dispatch is performed by a Python ``isinstance`` check and carries no JAX
# overhead (see :func:`make_unified_model` for the full safety analysis).
#
# The ``legacy_*`` prefixed names are preserved for callers that need the old API.
# ---------------------------------------------------------------------------


molliere_2020_emission = make_unified_model(
    molliere_2020_emission,
    legacy_molliere_2020_emission,
)
molliere_2020_two_column = make_unified_model(
    molliere_2020_two_column,
    legacy_molliere_2020_two_column,
)
guillot_emission = make_unified_model(
    guillot_emission,
    legacy_guillot_emission,
)
guillot_emission_add_gaussian_temperature = make_unified_model(
    guillot_emission_add_gaussian_temperature,
    legacy_guillot_emission_add_gaussian_temperature,
)
guillot_patchy_emission = make_unified_model(
    guillot_patchy_emission,
    legacy_guillot_patchy_emission,
)
interpolated_profile_emission = make_unified_model(
    interpolated_profile_emission,
    legacy_interpolated_profile_emission,
)
gradient_profile_emission = make_unified_model(
    gradient_profile_emission,
    legacy_gradient_profile_emission,
)
power_law_profile_emission = make_unified_model(
    power_law_profile_emission,
    legacy_power_law_profile_emission,
)
guillot_transmission = make_unified_model(
    guillot_transmission,
    legacy_guillot_transmission,
)
guillot_patchy_transmission = make_unified_model(
    guillot_patchy_transmission,
    legacy_guillot_patchy_transmission,
)
madhusudhan_seager_emission = make_unified_model(
    madhusudhan_seager_emission,
    legacy_madhusudhan_seager_emission,
)
madhusudhan_seager_transmission = make_unified_model(
    madhusudhan_seager_transmission,
    legacy_madhusudhan_seager_transmission,
)
madhu_seager_patchy_transmission = make_unified_model(
    madhu_seager_patchy_transmission,
    legacy_madhu_seager_patchy_transmission,
)
isothermal_transmission = make_unified_model(
    isothermal_transmission,
    legacy_isothermal_transmission,
)
power_law_profile_transmission = make_unified_model(
    power_law_profile_transmission,
    legacy_power_law_profile_transmission,
)
time_series_gradient_emission = make_unified_model(
    time_series_gradient_emission,
)
