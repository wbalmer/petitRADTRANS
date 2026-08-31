"""This file allows the calculation of equilibrium cloud abundances and base pressures"""

import copy
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE
from petitRADTRANS.opacities.opacities import CloudOpacity
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass
from petitRADTRANS.chemistry.vapour_pressures import (
    vapour_pressure_fe_solid,
    vapour_pressure_fe_liquid,
    vapour_pressure_mgsio3,
    vapour_pressure_mg2sio4,
    vapour_pressure_na2s,
    vapour_pressure_kcl,
    vapour_pressure_sio,
    vapour_pressure_tio2,
    vapour_pressure_cr,
    vapour_pressure_zns,
    vapour_pressure_nh3,
    vapour_pressure_mns,
    vapour_pressure_h2o,
    vapour_pressure_ch4,
    vapour_pressure_al2o3,
    vapour_pressure_catio3,
    vapour_pressure_caal12o19,
    vapour_pressure_sio2,
)

# metal species
__metals = [
    "C",
    "N",
    "O",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Ni",
    "Zn",
]

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
__elemental_abundances = (
    {  # TODO replace with chemistry.utils._solar_elemental_abundances
        "H": 0.9207539305,
        "He": 0.0783688694,
        "C": 0.0002478241,
        "N": 6.22506056949881e-05,
        "O": 0.0004509658,
        "Na": 1.60008694353205e-06,
        "Mg": 3.66558742055362e-05,
        "Al": 2.595e-06,
        "Si": 2.9795e-05,
        "P": 2.36670201997668e-07,
        "S": 1.2137900734604e-05,
        "Cl": 2.91167958499589e-07,
        "K": 9.86605611925677e-08,
        "Ca": 2.01439011429255e-06,
        "Ti": 8.20622804366359e-08,
        "V": 7.83688694089992e-09,
        "Cr": 4.018e-07,  # TODO double check value
        "Mn": 2.477e-07,  # TODO double check value
        "Fe": 2.91167958499589e-05,
        "Ni": 1.52807116806281e-06,
        "Zn": 3.34e-08,  # TODO double check value
    }
)

# Get molar masses ahead of time to gain speed
__molar_masses = {
    "H": get_species_molar_mass("H"),
    "He": get_species_molar_mass("He"),
    "C": get_species_molar_mass("C"),
    "N": get_species_molar_mass("N"),
    "O": get_species_molar_mass("O"),
    "Na": get_species_molar_mass("Na"),
    "Mg": get_species_molar_mass("Mg"),
    "Al": get_species_molar_mass("Al"),
    "Si": get_species_molar_mass("Si"),
    "P": get_species_molar_mass("P"),
    "S": get_species_molar_mass("S"),
    "Cl": get_species_molar_mass("Cl"),
    "K": get_species_molar_mass("K"),
    "Ca": get_species_molar_mass("Ca"),
    "Ti": get_species_molar_mass("Ti"),
    "V": get_species_molar_mass("V"),
    "Cr": get_species_molar_mass("Cr"),
    "Mn": get_species_molar_mass("Mn"),
    "Fe": get_species_molar_mass("Fe"),
    "Ni": get_species_molar_mass("Ni"),
    "Zn": get_species_molar_mass("Zn"),
}


def _get_species_molar_mass(species):
    """Get molar mass of a species from cache or compute it."""
    if species in __molar_masses:
        return __molar_masses[species]
    else:
        return get_species_molar_mass(species)


# ============================================================================
# Helper functions for consolidated cloud calculations
# ============================================================================


def _prepare_elemental_abundances(metallicity, co_ratio):
    """Prepare elemental abundances adjusted for metallicity and C/O ratio."""
    nfracs_use = copy.copy(__elemental_abundances)
    for spec in __elemental_abundances.keys():
        if spec not in ("H", "He"):
            nfracs_use[spec] = __elemental_abundances[spec] * 1e1**metallicity
    nfracs_use["O"] = nfracs_use["C"] / co_ratio
    return nfracs_use


def _get_total_mass(abundances):
    """Calculate total mass for mass fraction normalization."""
    total_mass = 0.0
    for spec in abundances.keys():
        total_mass += _get_species_molar_mass(spec) * abundances[spec]
    return total_mass


def _calculate_cloud_mass_fraction(element_counts, compound_mass, abundances):
    """Generic mass fraction calculator for cloud species.

    Args:
        element_counts: dict mapping element names to stoichiometric coefficients
        compound_mass: total molar mass of the compound
        abundances: elemental abundance dict (from _prepare_elemental_abundances)
    """
    # Find the limiting element
    limiting_nfracs = jnp.min(
        jnp.array([abundances[elem] / count for elem, count in element_counts.items()])
    )

    total_mass = _get_total_mass(abundances)
    return (compound_mass * limiting_nfracs) / total_mass


def _calculate_cloud_base_pressure_generic(pressures, temperatures, cond_func_args):
    """Generic cloud base pressure calculator for any species.

    The intersection between the condensation curve and the temperature profile
    is located by linear interpolation in ``log P`` across the bracketing layer
    pair, rather than being snapped to the layer midpoint. Snapping would make
    the cloud base a staircase function of temperature, metallicity and mean
    molar mass, which is a step discontinuity in every parameter those depend
    on; interpolating makes it continuous.

    Note:
        The deepest intersection is still selected discretely. When the
        temperature profile becomes tangent to the condensation curve a new pair
        of intersections appears, and the selected base jumps to it. That is a
        genuine fold in the model rather than a discretisation artefact, and it
        is not removed here.

    Args:
        press: pressure array
        temp: temperature array
        cond_func_args: tuple of (cond_callable, *args) where cond_callable returns (pc, tc)
    """
    linear_temperatures = jnp.linspace(50, 15000, 200)
    condensation_pressures = cond_func_args[0](linear_temperatures, *cond_func_args[1:])
    interpolated_condensation_temperature = jnp.interp(
        pressures, condensation_pressures, linear_temperatures
    )
    t_diff = interpolated_condensation_temperature - temperatures
    diff_vec = t_diff[1:] * t_diff[:-1]
    ind_cdf = diff_vec < 0.0

    # A bracketed sign change has t_diff of opposite signs at its two ends, so
    # the denominator cannot vanish there. It can elsewhere, and jnp.where
    # evaluates both branches, hence the guarded division.
    t_diff_drop = t_diff[:-1] - t_diff[1:]
    safe_t_diff_drop = jnp.where(ind_cdf, t_diff_drop, jnp.ones_like(t_diff_drop))
    crossing_fraction = jnp.where(
        ind_cdf, t_diff[:-1] / safe_t_diff_drop, jnp.zeros_like(t_diff_drop)
    )

    log_pressures = jnp.log(pressures)
    log_p_crossings = log_pressures[:-1] + crossing_fraction * (
        log_pressures[1:] - log_pressures[:-1]
    )

    masked_indices = jnp.where(ind_cdf, jnp.arange(log_p_crossings.shape[0]), -1)
    last_idx = jnp.max(masked_indices)
    p_cloud = jnp.where(
        last_idx != -1, jnp.exp(log_p_crossings[last_idx]), jnp.min(pressures)
    )
    return p_cloud


def _get_cloud_species_base_name(name):
    """Extract base cloud species name from full name."""
    cloud_bases = [
        "Fe(l)",
        "Fe(s)",
        "Fe",
        "MgSiO3",
        "Mg2SiO4",
        "Na2S",
        "KCL",
        "SiO",
        "TiO2",
        "Cr",
        "ZnS",
        "NH3",
        "MnS",
        "H2O",
        "CH4",
        "Al2O3",
        "CaTiO3",
        "CaAl12O19",
        "SiO2",
    ]
    for base in cloud_bases:
        if base in name:
            return base
    return None


# ============================================================================
# Cloud vertical profile
# ============================================================================

def _as_static_float(name, value):
    """Return ``value`` as a plain float, or ``None``.

    The cloud-base smoothing width selects between the smooth and the legacy
    hard-cutoff profile, so it must be a static scalar rather than a traced
    JAX value.
    """
    if value is None:
        return None

    if isinstance(value, Tracer):
        raise TypeError(
            f"'{name}' controls the cloud profile discretisation and must be a "
            f"static scalar, not a traced JAX value."
        )

    value = float(value)

    if value < 0.0:
        raise ValueError(f"'{name}' must be non-negative, but was {value}.")

    return value


def resolve_cloud_base_smoothing_width(smoothing_width=None):
    """Resolve the cloud-base smoothing width, in dex.

    Args:
        smoothing_width:
            Explicit width in dex, or ``None`` for the default below. ``0``
            selects the legacy hard cutoff.

    Returns:
        The smoothing width in dex, as a static float.
    """
    # About half a layer on a typical petitRADTRANS pressure grid (100-130
    # layers over 6-9 decades, i.e. 0.05-0.07 dex per layer), which keeps the
    # cloud base resolved by the grid it is sampled on without materially
    # widening it.
    #
    # A fixed value rather than one derived from the grid: the width selects
    # between the smooth and the hard-cutoff profile, so it must be static, and
    # inside a jit trace the pressure grid is not readable as a concrete array.
    # Pass :func:`cloud_base_smoothing_from_pressure_grid` evaluated outside the
    # traced region to tie it to a specific grid.
    default_smoothing_width = 0.03

    smoothing_width = _as_static_float("cloud_base_smoothing", smoothing_width)

    if smoothing_width is None:
        return default_smoothing_width

    return smoothing_width


def cloud_base_smoothing_from_pressure_grid(pressures, layers=0.5):
    """Return a cloud-base smoothing width tied to a pressure grid, in dex.

    The taper only conditions the likelihood well if it is resolved by the grid
    the cloud profile is sampled on, so on an unusually coarse or fine grid it is
    worth scaling the width with the layer spacing instead of taking the default
    of :func:`resolve_cloud_base_smoothing_width`.

    Call this once during model setup and pass the result as
    ``cloud_base_smoothing``: the width has to be a static value, and inside a
    jit trace the pressure grid is not readable as a concrete array.

    Args:
        pressures:
            Pressure grid the cloud profile will be sampled on. Must be
            concrete.
        layers:
            Width in units of the median layer spacing.

    Returns:
        The smoothing width in dex.
    """
    log_pressures = np.log10(np.asarray(pressures, dtype=float))

    if log_pressures.size < 2:
        return 0.0

    return layers * float(np.median(np.abs(np.diff(log_pressures))))


def cloud_base_smoothing_mass_normalisation(f_sed, smoothing_width):
    """Return the factor that makes the smoothed profile column-mass neutral.

    The hard-cutoff profile carries a column mass proportional to ``1 / a`` with
    ``a = ln(10) * (f_sed + 1)``, while the logistic taper of effective width
    ``w' = w / (1 + ln(10) f_sed w)`` carries ``pi * w' / sin(pi * a * w')``.
    Dividing by their ratio leaves the total cloud column mass unchanged by the
    smoothing.  ``smoothing_width`` is the user-facing ``w``; the effective width
    is derived here.

    The identity holds for ``a * w < 1``; beyond that the taper is wider than
    the profile's own scale height and no finite normalisation exists, so the
    factor is clamped to 1 and the smoothing is left un-normalised.

    Note:
        This is *not* applied by default, because total column mass is the
        wrong invariant to hold fixed. The mass the taper adds sits in its tail
        below the cloud base, which is optically thick and unobservable, whereas
        the rescaling it demands (15% at ``f_sed = 5``, ``w = 0.03``) applies to
        the whole profile including the observable upper cloud. Measured on a
        cloudy transmission spectrum over four cloud-base pressures, leaving the
        smoothing un-normalised moves the transit depth by at most 7.7 ppm
        relative to the hard cutoff, while normalising it moves it by up to
        86 ppm. Use it only when ``log_X_cb`` has to be calibrated against a
        total condensate mass budget.
    """
    taper_width = smoothing_width / (1.0 + jnp.log(10.0) * f_sed * smoothing_width)
    a_w = jnp.log(10.0) * (f_sed + 1.0) * taper_width
    safe_a_w = jnp.where(a_w < 0.9, a_w, 0.9)
    normalisation = jnp.sinc(safe_a_w)  # sin(pi x) / (pi x)

    return jnp.where(a_w < 0.9, normalisation, jnp.ones_like(normalisation))


def cloud_vertical_profile(
    pressures,
    cloud_base_pressure,
    cloud_abundance,
    f_sed,
    smoothing_width=None,
    conserve_column_mass=False,
):
    """Build the vertical mass-fraction profile of a sedimenting cloud.

    The Ackerman & Marley (2001) profile decays as ``(P / P_base) ** f_sed``
    above the cloud base and is zero below it.  Truncating it at the base with a
    hard cutoff makes the *sampled* profile a step function of ``P_base``: as the
    base crosses a pressure-grid point, that layer flips between zero and the
    full base abundance, which changes the spectrum discontinuously.  Any
    gradient-based method then sees an infinite derivative at every grid
    boundary, and optimisers settle exactly on one.

    This function instead tapers the profile through the base with a logistic in
    ``log10 P``:

    .. math::

        X(P) = X_\\mathrm{cb}\\,
               \\left(\\frac{P}{P_\\mathrm{base}}\\right)^{f_\\mathrm{sed}}
               \\sigma\\!\\left(
                   \\frac{\\log_{10}(P_\\mathrm{base} / P)}{w}
               \\right),

    which is infinitely differentiable in both ``P`` and ``P_base``.  The width
    ``w`` sets the physical thickness of the cloud base.  It should stay
    comparable to the pressure-grid spacing: much narrower and the sampled
    profile is smooth but still effectively a cliff, much wider and the base is
    no longer where ``P_base`` says it is.

    The taper is steepened to ``w' = w / (1 + ln(10) f_sed w)`` so that it always
    out-runs the power law, which is still growing downwards at ``ln(10) f_sed``
    per dex: the profile then decays below the base with an e-folding length of
    ``w`` dex for every ``f_sed``, and is unchanged above it.  Substituting
    ``w'`` and rearranging turns the product into the reciprocal form evaluated
    below, in which that cancellation is exact rather than a difference of large
    numbers.

    Evaluating it through ``logaddexp`` also keeps the deep layers in range,
    where the hard-cutoff form evaluates ``(P / P_base) ** f_sed`` at
    overflow-scale arguments before discarding it -- reaching ``1e87`` at
    ``f_sed = 20`` and overflowing single precision to a NaN gradient -- and it
    maps the non-finite ``f_sed`` sentinel, used when no ``f_sed`` parameter is
    given, back onto a zero cloud rather than a NaN.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        cloud_base_pressure:
            Cloud-base pressure, in the same units as ``pressures``.
        cloud_abundance:
            Cloud mass fraction at the base.
        f_sed:
            Sedimentation efficiency setting the decay above the base. Must be
            finite: a non-finite value makes ``f_sed * log10(P / P_base)`` a NaN
            on a layer sitting exactly on the base, and guarding that product
            would flatten the derivative on precisely the grid points this
            function exists to make differentiable. Callers use a zero cloud
            abundance to express "no cloud configured".
        smoothing_width:
            Width of the base taper in dex.  ``None`` takes the default of
            :func:`resolve_cloud_base_smoothing_width`; ``0`` restores the legacy
            hard cutoff, which is discontinuous in ``cloud_base_pressure`` and should
            only be used for reproducing older results.  See
            :func:`cloud_base_smoothing_from_pressure_grid` for a width tied to a
            specific pressure grid.
        conserve_column_mass:
            If ``True``, rescale the profile so the smoothing leaves the total
            cloud column mass unchanged relative to the hard cutoff. Off by
            default; see :func:`cloud_base_smoothing_mass_normalisation` for why
            that is the better choice.

    Returns:
        The cloud mass-fraction profile sampled on ``pressures``.
    """
    pressures = jnp.asarray(pressures, dtype=FEAUTRIER_JAX_DTYPE)
    smoothing_width = resolve_cloud_base_smoothing_width(smoothing_width)

    if smoothing_width == 0.0:
        return jnp.where(
            pressures < cloud_base_pressure,
            cloud_abundance * (pressures / cloud_base_pressure) ** f_sed,
            jnp.zeros_like(pressures),
        )

    # Positive below the cloud base, negative above it.
    log_pressure_ratio = jnp.log10(pressures) - jnp.log10(cloud_base_pressure)

    # Reciprocal form of the tapered profile:
    #
    #     X = X_cb / ( (P / P_base) ** -f_sed  +  10 ** (d / w) ),
    #
    # which is algebraically the power law times the logistic, with the taper
    # already steepened so that it out-runs the power law. Evaluating it through
    # logaddexp keeps both terms out of the overflow range, and makes the two
    # limits exact rather than a difference of large numbers: above the base the
    # first term dominates and gives the untouched power law, below it the second
    # dominates and gives a decay of e-folding length `smoothing_width` in dex,
    # for every f_sed. It is also what keeps a non-finite f_sed sentinel (no
    # f_sed parameter given) returning a zero cloud instead of inf - inf.
    log_profile = -jnp.logaddexp(
        -jnp.log(10.0) * f_sed * log_pressure_ratio,
        log_pressure_ratio / smoothing_width,
    )
    profile = cloud_abundance * jnp.exp(log_profile)

    if conserve_column_mass:
        profile = profile * cloud_base_smoothing_mass_normalisation(
            f_sed, smoothing_width
        )

    return profile


def cloud_base_depletion_factor(pressures, cloud_base_pressure, smoothing_width=None):
    """Return a smooth 1-to-0 factor for gas rained out above a cloud base.

    This is the complement of the taper used by :func:`cloud_vertical_profile`:
    it is one in the deep layers, one half at the base, and decays to zero above
    it.  Multiplying a gas profile by this factor depletes it above the cloud
    base without the step that ``jnp.where(pressures < cloud_base, 0, ...)``
    introduces every time the base crosses a pressure-grid point.

    Args:
        pressures:
            Pressure grid for the atmospheric layers.
        cloud_base_pressure:
            Cloud-base pressure above which the gas is depleted.
        smoothing_width:
            Width of the transition in dex.  ``None`` takes the default of
            :func:`resolve_cloud_base_smoothing_width`; ``0`` restores the legacy
            hard cutoff.

    Returns:
        A depletion factor sampled on ``pressures``, in ``[0, 1]``.
    """
    pressures = jnp.asarray(pressures, dtype=FEAUTRIER_JAX_DTYPE)
    smoothing_width = resolve_cloud_base_smoothing_width(smoothing_width)

    log_pressure_ratio = jnp.log10(pressures) - jnp.log10(cloud_base_pressure)

    if smoothing_width == 0.0:
        return jnp.where(
            log_pressure_ratio < 0,
            jnp.zeros_like(pressures),
            jnp.ones_like(pressures),
        )

    return jax.nn.sigmoid(log_pressure_ratio / smoothing_width)


def setup_clouds(pressures, parameters, cloud_species):
    """
    This function provides the set of cloud parameters used in
    petitRADTRANS. This will be some combination of atmospheric
    parameters (fsed and Kzz), distribution descriptions (log normal or hansen)
    and the cloud particle radius. Fsed and the particle radii can be provided
    on a per-cloud basis.

    Args:
        pressures : jnp.ndarray
            The pressure array used to provide the atmospheric grid
        parameters : PhysicalParams or dict
            The parameter container. Values are accessed directly (e.g.
            ``parameters['sigma_lnorm']``). Use :func:`setup_clouds_from_legacy_dict`
            if you have a legacy parameter dictionary with ``.value`` attributes.
            Should contain:
                 *  fsed : sedimentation parameter - can be unique to each cloud type
                One of:
                  *  sigma_lnorm : Width of cloud particle size distribution (log normal)
                  *  b_hans : Width of cloud particle size distribution (hansen)
                One of:
                  *  log_cloud_radius_* : Central particle radius (typically computed with fsed and Kzz)
                  *  log_kzz : Vertical mixing parameter
        cloud_species : list
            A list of the names of each of the cloud species used in the atmosphere.

    Returns:
        sigma_lnorm : float, None
            The width of a log normal particle size distribution.
        fseds : dict, None
            The sedimentation fraction for each cloud species in the atmosphere.
        kzz : jnp.ndarray, None
            The vertical mixing parameter.
        b_hans : float, None
            The width of a hansen particle size distribution.
        radii : dict, None
            The central radius of the particle size distribution.
        distribution : string
            Either "lognormal" or "hansen" - tells pRT which distribution to use.
    """
    sigma_lnorm = None
    b_hans = None
    distribution = "lognormal"
    kzz = None

    # Setup distribution shape
    if "sigma_lnorm" in parameters.keys():
        sigma_lnorm = parameters["sigma_lnorm"]
    elif "b_hans" in parameters.keys():
        b_hans = get_bhans(parameters, cloud_species, pressures.shape[0])
        distribution = "hansen"

    # Are we retrieving the particle radii?
    radii = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split("_")[0]
        if "log_cloud_radius_" + cloud_name in parameters.keys():
            radii[cloud] = 10 ** parameters[
                "log_cloud_radius_" + cloud_name
            ] * jnp.ones_like(pressures)

    if not radii:
        radii = None

    # per-cloud species fseds
    fseds = get_fseds(parameters, cloud_species)

    if "log_kzz" in parameters.keys():
        kzz = 10 ** parameters["log_kzz"] * jnp.ones_like(pressures)

    cloud_fraction = 1.0
    patchy_clouds = None

    if "patchiness" in parameters.keys():
        cloud_fraction = parameters["patchiness"]

    if "cloud_fraction" in parameters.keys():
        cloud_fraction = parameters["cloud_fraction"]

    if "patchy_clouds" in parameters.keys():
        patchy_clouds = parameters["patchy_clouds"]

    if "remove_cloud_species" in parameters.keys():
        patchy_clouds = parameters["remove_cloud_species"]
    return (
        sigma_lnorm,
        fseds,
        kzz,
        b_hans,
        radii,
        cloud_fraction,
        patchy_clouds,
        distribution,
    )


def setup_clouds_from_legacy_dict(pressures, parameters, cloud_species):
    """Wrapper for :func:`setup_clouds` that accepts a legacy parameter dictionary.

    Legacy model functions receive ``parameters`` as a dict of objects with a
    ``.value`` attribute (e.g. ``SimpleNamespace(value=...)``).  This wrapper
    extracts the plain values before forwarding to :func:`setup_clouds`.

    Args:
        pressures : jnp.ndarray
            The pressure array used to provide the atmospheric grid.
        parameters : dict
            Legacy parameter dictionary where each value exposes a ``.value``
            attribute containing the actual scalar or array.
        cloud_species : list
            A list of the names of each of the cloud species used in the
            atmosphere.

    Returns:
        Same 8-tuple as :func:`setup_clouds`.
    """
    plain_params = {name: param.value for name, param in parameters.items()}
    return setup_clouds(pressures, plain_params, cloud_species)


def setup_simple_clouds_hazes(parameters):
    """Setup cloud and haze parameters for transmission spectrum models.

    Args:
        parameters: dictionary of atmospheric parameters

    Returns:
        pcloud: cloud top pressure
        power_law_opacity_coefficient: power law exponent for opacity
        haze_factor: strength of haze opacity
        power_law_opacity_350nm: reference opacity at 350 nm
    """
    pcloud = None
    power_law_opacity_coefficient = None
    haze_factor = 1.0
    power_law_opacity_350nm = None

    # Grey clouds
    if (
        "log_pcloud" in parameters.keys()
    ):  # TODO should be log10 to be consistent with SpecModel
        pcloud = 10 ** parameters["log_pcloud"]
    elif "Pcloud" in parameters.keys():
        pcloud = parameters["Pcloud"]
    elif (
        "log_opaque_cloud_top_pressure" in parameters.keys()
    ):  # TODO should be log10 to be consistent with SpecModel
        pcloud = 10 ** parameters["log_opaque_cloud_top_pressure"]
    elif "opaque_cloud_top_pressure" in parameters.keys():
        pcloud = parameters["opaque_cloud_top_pressure"]

    # Hazes and Rayleigh
    if "power_law_opacity_350nm" in parameters.keys():
        power_law_opacity_350nm = parameters["power_law_opacity_350nm"]
    if "power_law_opacity_coefficient" in parameters.keys():
        power_law_opacity_coefficient = parameters["power_law_opacity_coefficient"]
    if (
        "log_haze_factor" in parameters.keys()
    ):  # TODO should be log10 to be consistent with SpectralModel
        haze_factor = 10 ** parameters["log_haze_factor"]
    elif "haze_factor" in parameters.keys():
        haze_factor = parameters["haze_factor"]

    # Compatibility
    if "gamma_scat" in parameters.keys():
        # TODO deprecate naming in v5.0.0
        power_law_opacity_coefficient = parameters["gamma_scat"]
    if "kappa_0" in parameters.keys():
        # TODO deprecate naming in v5.0.0
        power_law_opacity_350nm = 10 ** parameters["kappa_0"]
    return pcloud, power_law_opacity_coefficient, haze_factor, power_law_opacity_350nm


def setup_simple_clouds_hazes_from_legacy_dict(parameters):
    """Legacy wrapper for :func:`setup_simple_clouds_hazes`.

    Converts a legacy parameter dictionary (values accessed via ``.value``) to a
    plain-value mapping before forwarding to :func:`setup_simple_clouds_hazes`.

    Args:
        parameters : dict
            Legacy parameter dictionary where each value is an object with a ``.value`` attribute.
    """
    plain_parameters = {name: param.value for name, param in parameters.items()}
    return setup_simple_clouds_hazes(plain_parameters)


def cloud_dict(parameters, parameter_name, cloud_species, shape=0):
    """
    This is a generic method to create a dictionary of
    parameters values for a given cloud parameterization, testing if
    the parameter should be filled on a per-species basis or
    if each cloud species should have the same value.
    """
    output_dictionary = {}

    for cloud in cloud_species:
        cloud_opacity = CloudOpacity([cloud], natural_abundance=False)
        species_full_name = cloud_opacity.species_full_name
        cloud_name = species_full_name.split("_")[0]
        output = None

        if parameter_name + "_" + cloud_name in parameters.keys():
            output = parameters[parameter_name + "_" + cloud_name]
        elif parameter_name in parameters.keys():
            output = parameters[parameter_name]

        if shape > 0:
            output = output * jnp.ones(shape)

        output_dictionary[cloud] = output

    if not output_dictionary:
        output_dictionary = None

    return output_dictionary


def get_fseds(parameters, cloud_species):
    """
    This function checks to see if the fsed values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters, "f_sed", cloud_species)


def get_bhans(parameters, cloud_species, shape=0):
    """
    This function checks to see if the bhans values are input on a per-cloud basis
    or only as a single value, and returns the dictionary providing the fsed values
    for each cloud, or None, if no cloud is used.
    """
    return cloud_dict(parameters, "b_hans", cloud_species, shape=shape)


def return_cloud_mass_fraction(name, metallicity, co_ratio):
    """Return mass fraction for a cloud species using unified dispatch."""
    cloud_species_functions = {
        "Fe": calculate_mass_fraction_fe,
        "MgSiO3": calculate_mass_fraction_mgsio3,
        "Mg2SiO4": calculate_mass_fraction_mg2sio4,
        "Na2S": calculate_mass_fraction_na2s,
        "KCL": calculate_mass_fraction_kcl,
        "SiO": calculate_mass_fraction_sio,
        "TiO2": calculate_mass_fraction_tio2,
        "Cr": calculate_mass_fraction_cr,
        "ZnS": calculate_mass_fraction_zns,
        "NH3": calculate_mass_fraction_nh3,
        "MnS": calculate_mass_fraction_mns,
        "H2O": calculate_mass_fraction_h2o,
        "CH4": calculate_mass_fraction_ch4,
        "Al2O3": calculate_mass_fraction_al2o3,
        "CaTiO3": calculate_mass_fraction_catio3,
        "CaAl12O19": calculate_mass_fraction_caal12o19,
        "SiO2": calculate_mass_fraction_sio2,
    }

    base_name = _get_cloud_species_base_name(name)
    if base_name and base_name in cloud_species_functions:
        return cloud_species_functions[base_name](metallicity, co_ratio)

    warnings.warn(f"The cloud {name} is not currently implemented.")
    return jnp.zeros_like(metallicity)


def calculate_vapour_pressure(name, pressure, args):
    """Return mass fraction for a cloud species using unified dispatch."""
    cloud_species_functions = {
        "Fe(l)": vapour_pressure_fe_liquid,
        "Fe(s)": vapour_pressure_fe_solid,
        "MgSiO3": vapour_pressure_mgsio3,
        "Mg2SiO4": vapour_pressure_mg2sio4,
        "Na2S": vapour_pressure_na2s,
        "KCL": vapour_pressure_kcl,
        "SiO": vapour_pressure_sio,
        "TiO2": vapour_pressure_tio2,
        "Cr": vapour_pressure_cr,
        "ZnS": vapour_pressure_zns,
        "NH3": vapour_pressure_nh3,
        "MnS": vapour_pressure_mns,
        "H2O": vapour_pressure_h2o,
        "CH4": vapour_pressure_ch4,
        "Al2O3": vapour_pressure_al2o3,
        "CaTiO3": vapour_pressure_catio3,
        "CaAl12O19": vapour_pressure_caal12o19,
        "SiO2": vapour_pressure_sio2,
    }

    base_name = _get_cloud_species_base_name(name)
    if base_name and base_name in cloud_species_functions:
        return cloud_species_functions[base_name](*args)

    warnings.warn(f"The condensation curve for {name} is not currently implemented.")
    return jnp.zeros_like(args[0])


def calculate_cloud_base_pressure(
    name, press, temp, metallicity=0.0, co_ratio=0.55, mmw=2.33
):
    """Calculate cloud base pressure for a cloud species using unified dispatch."""
    cloud_species_functions = {
        "Fe": calculate_cloud_base_pressure_fe,
        "Fe(s)": calculate_cloud_base_pressure_fe,
        "Fe(l)": calculate_cloud_base_pressure_fe,
        "MgSiO3": calculate_cloud_base_pressure_mgsio3,
        "Mg2SiO4": calculate_cloud_base_pressure_mg2sio4,
        "Na2S": calculate_cloud_base_pressure_na2s,
        "KCL": calculate_cloud_base_pressure_kcl,
        "SiO": calculate_cloud_base_pressure_sio,
        "TiO2": calculate_cloud_base_pressure_tio2,
        "Cr": calculate_cloud_base_pressure_cr,
        "ZnS": calculate_cloud_base_pressure_zns,
        "NH3": calculate_cloud_base_pressure_nh3,
        "MnS": calculate_cloud_base_pressure_mns,
        "H2O": calculate_cloud_base_pressure_h2o,
        "CH4": calculate_cloud_base_pressure_ch4,
        "Al2O3": calculate_cloud_base_pressure_al2o3,
        "CaTiO3": calculate_cloud_base_pressure_catio3,
        "CaAl12O19": calculate_cloud_base_pressure_caal12o19,
        "SiO2": calculate_cloud_base_pressure_sio2,
    }

    base_name = _get_cloud_species_base_name(name)
    if base_name and base_name in cloud_species_functions:
        return cloud_species_functions[base_name](
            press, temp, metallicity, co_ratio, mmw
        )

    warnings.warn(
        f"The cloud {name} is not currently implemented, defaulting to top of atmosphere."
    )
    return press[0].astype(FEAUTRIER_JAX_DTYPE)


def calculate_cloud_base_pressure_free(name, press, temp, mfrac, mmw=2.33):
    """Calculate cloud base pressure for a cloud species (free abundance) using unified dispatch."""
    cloud_species_functions = {
        "Fe": calculate_cloud_base_pressure_fe_free,
        "MgSiO3": calculate_cloud_base_pressure_mgsio3_free,
        "Mg2SiO4": calculate_cloud_base_pressure_mg2sio4_free,
        "Na2S": calculate_cloud_base_pressure_na2s_free,
        "KCL": calculate_cloud_base_pressure_kcl_free,
        "SiO": calculate_cloud_base_pressure_sio_free,
        "TiO2": calculate_cloud_base_pressure_tio2_free,
        "Cr": calculate_cloud_base_pressure_cr_free,
        "ZnS": calculate_cloud_base_pressure_zns_free,
        "NH3": calculate_cloud_base_pressure_nh3_free,
        "MnS": calculate_cloud_base_pressure_mns_free,
        "H2O": calculate_cloud_base_pressure_h2o_free,
        "CH4": calculate_cloud_base_pressure_ch4_free,
        "Al2O3": calculate_cloud_base_pressure_al2o3_free,
        "CaTiO3": calculate_cloud_base_pressure_catio3_free,
        "CaAl12O19": calculate_cloud_base_pressure_caal12o19_free,
        "SiO2": calculate_cloud_base_pressure_sio2_free,
    }

    base_name = _get_cloud_species_base_name(name)
    if base_name and base_name in cloud_species_functions:
        return cloud_species_functions[base_name](press, temp, mfrac, mmw)

    warnings.warn(f"The cloud {name} is not currently implemented.")
    return jnp.zeros_like(temp)


def calculate_mass_fraction_fe(metallicity=0.0, co_ratio=0.55):
    """Calculate Fe mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    compound_mass = _get_species_molar_mass("Fe")
    limiting_nfracs = nfracs_use["Fe"]
    total_mass = _get_total_mass(nfracs_use)
    return (compound_mass * limiting_nfracs) / total_mass


def calculate_mass_fraction_mgsio3(metallicity, co_ratio):
    """Calculate MgSiO3 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Mg": 1, "Si": 1, "O": 3}
    compound_mass = (
        _get_species_molar_mass("Mg")
        + _get_species_molar_mass("Si")
        + 3.0 * _get_species_molar_mass("O")
    )
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_mg2sio4(metallicity, co_ratio):
    """Calculate Mg2SiO4 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Mg": 2, "Si": 1, "O": 4}
    compound_mass = (
        2 * _get_species_molar_mass("Mg")
        + _get_species_molar_mass("Si")
        + 4.0 * _get_species_molar_mass("O")
    )
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_sio(metallicity, co_ratio):
    """Calculate SiO mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Si": 1, "O": 1}
    compound_mass = _get_species_molar_mass("Si") + _get_species_molar_mass("O")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_na2s(metallicity, co_ratio):
    """Calculate Na2S mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Na": 2, "S": 1}
    compound_mass = 2.0 * _get_species_molar_mass("Na") + _get_species_molar_mass("S")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_kcl(metallicity, co_ratio):
    """Calculate KCL mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"K": 1, "Cl": 1}
    compound_mass = _get_species_molar_mass("K") + _get_species_molar_mass("Cl")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_tio2(metallicity, co_ratio):
    """Calculate TiO2 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Ti": 1, "O": 2}
    compound_mass = _get_species_molar_mass("Ti") + 2.0 * _get_species_molar_mass("O")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_cr(metallicity, co_ratio):
    """Calculate Cr mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    compound_mass = _get_species_molar_mass("Cr")
    limiting_nfracs = nfracs_use["Cr"]
    total_mass = _get_total_mass(nfracs_use)
    return (compound_mass * limiting_nfracs) / total_mass


def calculate_mass_fraction_zns(metallicity, co_ratio):
    """Calculate ZnS mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Zn": 1, "S": 1}
    compound_mass = _get_species_molar_mass("Zn") + _get_species_molar_mass("S")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_nh3(metallicity, co_ratio):
    """Calculate NH3 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"N": 1, "H": 3}
    compound_mass = _get_species_molar_mass("N") + 3.0 * _get_species_molar_mass("H")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_mns(metallicity, co_ratio):
    """Calculate MnS mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Mn": 1, "S": 1}
    compound_mass = _get_species_molar_mass("Mn") + _get_species_molar_mass("S")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_h2o(metallicity, co_ratio):
    """Calculate H2O mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"H": 2, "O": 1}
    compound_mass = 2.0 * _get_species_molar_mass("H") + _get_species_molar_mass("O")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_ch4(metallicity, co_ratio):
    """Calculate CH4 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"C": 1, "H": 4}
    compound_mass = _get_species_molar_mass("C") + 4.0 * _get_species_molar_mass("H")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_al2o3(metallicity, co_ratio):
    """Calculate Al2O3 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Al": 2, "O": 3}
    compound_mass = 2.0 * _get_species_molar_mass("Al") + 3.0 * _get_species_molar_mass(
        "O"
    )
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_catio3(metallicity, co_ratio):
    """Calculate CaTiO3 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Ca": 1, "Ti": 1, "O": 3}
    compound_mass = (
        _get_species_molar_mass("Ca")
        + _get_species_molar_mass("Ti")
        + 3.0 * _get_species_molar_mass("O")
    )
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_caal12o19(metallicity, co_ratio):
    """Calculate CaAl12O19 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Ca": 1, "Al": 12, "O": 19}
    compound_mass = (
        _get_species_molar_mass("Ca")
        + 12.0 * _get_species_molar_mass("Al")
        + 19.0 * _get_species_molar_mass("O")
    )
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


def calculate_mass_fraction_sio2(metallicity, co_ratio):
    """Calculate SiO2 mass fraction for given metallicity and C/O ratio."""
    nfracs_use = _prepare_elemental_abundances(metallicity, co_ratio)
    element_counts = {"Si": 1, "O": 2}
    compound_mass = _get_species_molar_mass("Si") + 2.0 * _get_species_molar_mass("O")
    return _calculate_cloud_mass_fraction(element_counts, compound_mass, nfracs_use)


# ============================================================================
# Vapor pressure functions are imported from vapour_pressures.py
# ============================================================================


def condensation_temperature_fe(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Fe condensation curve (solid).

    Based on Ackerman & Marley (2001) including erratum (P_vap in bar).
    """
    x_fe = calculate_mass_fraction_fe(metallicity, co_ratio)

    return vapour_pressure_fe_solid(temperatures) / (
        x_fe * mmw / _get_species_molar_mass("Fe")
    )


def condensation_temperature_fe_liquid(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Fe condensation curve (liquid).

    Based on Ackerman & Marley (2001) including erratum (P_vap in bar).
    """
    x_fe = calculate_mass_fraction_fe(metallicity, co_ratio)

    return vapour_pressure_fe_liquid(temperatures) / (
        x_fe * mmw / _get_species_molar_mass("Fe")
    )


def condensation_temperature_fe_combined(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Fe condensation curve using minimum of solid/liquid phases."""
    p1 = condensation_temperature_fe(temperatures, metallicity, co_ratio, mmw)
    p2 = condensation_temperature_fe_liquid(temperatures, metallicity, co_ratio, mmw)

    ret_p = jnp.zeros_like(p1)
    index = p1 < p2
    ret_p = jnp.where(index, p1, ret_p)
    ret_p = jnp.where(~index, p2, ret_p)
    return ret_p


def condensation_temperature_fe_free(temperatures, x_fe, mmw=2.33):
    """Calculate Fe condensation curve (solid) for prescribed abundance.

    Based on Ackerman & Marley (2001) including erratum (P_vap in bar).
    """
    return vapour_pressure_fe_solid(temperatures) / (
        x_fe * mmw / _get_species_molar_mass("Fe")
    )


def condensation_temperature_fe_l_free(temperatures, x_fe, mmw=2.33):
    """Calculate Fe condensation curve (liquid) for prescribed abundance.

    Based on Ackerman & Marley (2001) including erratum (P_vap in bar).
    """
    return vapour_pressure_fe_liquid(temperatures) / (
        x_fe * mmw / _get_species_molar_mass("Fe")
    )


def condensation_temperature_fe_comb_free(temperatures, x_fe, mmw=2.33):
    """Calculate Fe condensation curve for prescribed abundance using minimum of solid/liquid."""
    p1 = condensation_temperature_fe_free(temperatures, x_fe, mmw)
    p2 = condensation_temperature_fe_l_free(temperatures, x_fe, mmw)
    ret_p = jnp.zeros_like(p1)
    index = p1 < p2
    ret_p = jnp.where(index, p1, ret_p)
    ret_p = jnp.where(~index, p2, ret_p)
    return ret_p


def condensation_temperature_mgsio3(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate MgSiO3 condensation curve.

    Based on Ackerman & Marley (2001) including erratum.
    """
    xmgsio3 = calculate_mass_fraction_mgsio3(metallicity, co_ratio)
    m_mgsio3 = (
        _get_species_molar_mass("Mg")
        + _get_species_molar_mass("Si")
        + 3.0 * _get_species_molar_mass("O")
    )
    return vapour_pressure_mgsio3(temperatures) / (xmgsio3 * mmw / m_mgsio3)


def condensation_temperature_mgsio3_free(temperatures, x_mgsio3, mmw=2.33):
    """Calculate MgSiO3 condensation curve for prescribed abundance.

    Based on Ackerman & Marley (2001) including erratum.
    """
    m_mgsio3 = (
        _get_species_molar_mass("Mg")
        + _get_species_molar_mass("Si")
        + 3.0 * _get_species_molar_mass("O")
    )
    return vapour_pressure_mgsio3(temperatures) / (x_mgsio3 * mmw / m_mgsio3)


def condensation_temperature_mg2sio4(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Mg2SiO4 condensation curve.

    Based on Visscher 2010. Returns total pressure, not species vapor pressure.
    """
    return vapour_pressure_mg2sio4(temperatures, metallicity)


def condensation_temperature_mg2sio4_free(temperatures, x_mg2sio4, mmw=2.33):
    """Calculate Mg2SiO4 condensation curve for prescribed abundance.

    Based on Ackerman & Marley (2001) including erratum.
    """
    m_mg2sio4 = (
        2 * _get_species_molar_mass("Mg")
        + _get_species_molar_mass("Si")
        + 4.0 * _get_species_molar_mass("O")
    )

    return vapour_pressure_mg2sio4(temperatures) / (x_mg2sio4 * mmw / m_mg2sio4)


def condensation_temperature_na2s(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Na2S condensation curve.

    Based on Charnay+2018. Accounts for partial pressure of Na.
    """
    # This is the partial pressure of Na, so /2 to get the partial pressure of the hypothetical Na2S gas particles
    # This is OK: there are more S than Na atoms at solar abundance ratios

    xna2s = calculate_mass_fraction_na2s(metallicity, co_ratio)

    m_na2s = 2.0 * _get_species_molar_mass("Na") + _get_species_molar_mass("S")

    return vapour_pressure_na2s(temperatures, metallicity) / (xna2s * mmw / m_na2s)


def condensation_temperature_na2s_free(temperatures, x_na2s, mmw=2.33):
    """Calculate Na2S condensation curve for prescribed abundance.

    Based on Charnay+2018. Uses Na/H as proxy for Fe/H for solar compositions.
    """
    # This is the partial pressure of Na, so /2 to get the partial pressure of the hypothetical Na2S gas particles
    # This is OK: there are more S than Na atoms at solar abundance ratios
    m_na2s = 2.0 * _get_species_molar_mass("Na") + _get_species_molar_mass("S")

    return vapour_pressure_na2s(temperatures, na2s_mass_fraction=m_na2s, mmw=mmw) / (
        x_na2s * mmw / m_na2s
    )


def condensation_temperature_kcl(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate KCL condensation curve.

    Based on Charnay+2018.
    """
    xkcl = calculate_mass_fraction_kcl(metallicity, co_ratio)

    m_kcl = _get_species_molar_mass("K") + _get_species_molar_mass("Cl")

    return vapour_pressure_kcl(temperatures) / (xkcl * mmw / m_kcl)


def condensation_temperature_kcl_free(temperatures, x_kcl, mmw=2.33):
    """Calculate KCL condensation curve for prescribed abundance.

    Based on Charnay+2018.
    """
    m_kcl = _get_species_molar_mass("K") + _get_species_molar_mass("Cl")

    return vapour_pressure_kcl(temperatures) / (x_kcl * mmw / m_kcl)


def condensation_temperature_sio(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate SiO condensation curve.

    Based on Wetzel+2013.
    """
    xsio = calculate_mass_fraction_sio(metallicity, co_ratio)

    m_sio = _get_species_molar_mass("Si") + _get_species_molar_mass("O")

    return vapour_pressure_sio(temperatures) / (xsio * mmw / m_sio)


def condensation_temperature_sio_free(temperatures, x_sio, mmw=2.33):
    """Calculate SiO condensation curve for prescribed abundance.

    Based on Wetzel+2013.
    """
    m_sio = _get_species_molar_mass("Si") + _get_species_molar_mass("O")

    return vapour_pressure_sio(temperatures) / (x_sio * mmw / m_sio)


def condensation_temperature_tio2(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate TiO2 condensation curve."""
    xtio2 = calculate_mass_fraction_tio2(metallicity, co_ratio)
    m_tio2 = _get_species_molar_mass("Ti") + 2.0 * _get_species_molar_mass("O")
    return vapour_pressure_tio2(temperatures, metallicity) / (xtio2 * mmw / m_tio2)


def condensation_temperature_tio2_free(temperatures, x_tio2, mmw=2.33):
    """Calculate TiO2 condensation curve for prescribed abundance."""
    m_tio2 = _get_species_molar_mass("Ti") + 2.0 * _get_species_molar_mass("O")
    return vapour_pressure_tio2(temperatures) / (x_tio2 * mmw / m_tio2)


def condensation_temperature_cr(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Cr condensation curve."""
    xcr = calculate_mass_fraction_cr(metallicity, co_ratio)
    m_cr = _get_species_molar_mass("Cr")
    return vapour_pressure_cr(temperatures, metallicity) / (xcr * mmw / m_cr)


def condensation_temperature_cr_free(temperatures, x_cr, mmw=2.33):
    """Calculate Cr condensation curve for prescribed abundance."""
    m_cr = _get_species_molar_mass("Cr")
    return vapour_pressure_cr(temperatures) / (x_cr * mmw / m_cr)


def condensation_temperature_zns(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate ZnS condensation curve."""
    xzns = calculate_mass_fraction_zns(metallicity, co_ratio)
    m_zns = _get_species_molar_mass("Zn") + _get_species_molar_mass("S")
    return vapour_pressure_zns(temperatures, metallicity) / (xzns * mmw / m_zns)


def condensation_temperature_zns_free(temperatures, x_zns, mmw=2.33):
    """Calculate ZnS condensation curve for prescribed abundance."""
    m_zns = _get_species_molar_mass("Zn") + _get_species_molar_mass("S")
    return vapour_pressure_zns(temperatures) / (x_zns * mmw / m_zns)


def condensation_temperature_nh3(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate NH3 condensation curve."""
    xnh3 = calculate_mass_fraction_nh3(metallicity, co_ratio)
    m_nh3 = _get_species_molar_mass("N") + 3.0 * _get_species_molar_mass("H")
    return vapour_pressure_nh3(temperatures) / (xnh3 * mmw / m_nh3)


def condensation_temperature_nh3_free(temperatures, x_nh3, mmw=2.33):
    """Calculate NH3 condensation curve for prescribed abundance."""
    m_nh3 = _get_species_molar_mass("N") + 3.0 * _get_species_molar_mass("H")
    return vapour_pressure_nh3(temperatures) / (x_nh3 * mmw / m_nh3)


def condensation_temperature_mns(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate MnS condensation curve."""
    xmns = calculate_mass_fraction_mns(metallicity, co_ratio)
    m_mns = _get_species_molar_mass("Mn") + _get_species_molar_mass("S")
    return vapour_pressure_mns(temperatures, metallicity) / (xmns * mmw / m_mns)


def condensation_temperature_mns_free(temperatures, x_mns, mmw=2.33):
    """Calculate MnS condensation curve for prescribed abundance."""
    m_mns = _get_species_molar_mass("Mn") + _get_species_molar_mass("S")
    return vapour_pressure_mns(temperatures) / (x_mns * mmw / m_mns)


def condensation_temperature_h2o(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate H2O condensation curve."""
    xh2o = calculate_mass_fraction_h2o(metallicity, co_ratio)
    m_h2o = 2.0 * _get_species_molar_mass("H") + _get_species_molar_mass("O")
    return vapour_pressure_h2o(temperatures) / (xh2o * mmw / m_h2o)


def condensation_temperature_h2o_free(temperatures, x_h2o, mmw=2.33):
    """Calculate H2O condensation curve for prescribed abundance."""
    m_h2o = 2.0 * _get_species_molar_mass("H") + _get_species_molar_mass("O")
    return vapour_pressure_h2o(temperatures, 0) / (x_h2o * mmw / m_h2o)


def condensation_temperature_ch4(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate CH4 condensation curve."""
    xch4 = calculate_mass_fraction_ch4(metallicity, co_ratio)
    m_ch4 = _get_species_molar_mass("C") + 4.0 * _get_species_molar_mass("H")
    return vapour_pressure_ch4(temperatures, metallicity) / (xch4 * mmw / m_ch4)


def condensation_temperature_ch4_free(temperatures, x_ch4, mmw=2.33):
    """Calculate CH4 condensation curve for prescribed abundance."""
    m_ch4 = _get_species_molar_mass("C") + 4.0 * _get_species_molar_mass("H")
    return vapour_pressure_ch4(temperatures) / (x_ch4 * mmw / m_ch4)


def condensation_temperature_al2o3(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate Al2O3 condensation curve."""
    xal2o3 = calculate_mass_fraction_al2o3(metallicity, co_ratio)
    m_al2o3 = 2.0 * _get_species_molar_mass("Al") + 3.0 * _get_species_molar_mass("O")
    return vapour_pressure_al2o3(temperatures, metallicity) / (xal2o3 * mmw / m_al2o3)


def condensation_temperature_al2o3_free(temperatures, x_al2o3, mmw=2.33):
    """Calculate Al2O3 condensation curve for prescribed abundance."""
    m_al2o3 = 2.0 * _get_species_molar_mass("Al") + 3.0 * _get_species_molar_mass("O")
    return vapour_pressure_al2o3(temperatures) / (x_al2o3 * mmw / m_al2o3)


def condensation_temperature_catio3(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate CaTiO3 condensation curve."""
    xcatio3 = calculate_mass_fraction_catio3(metallicity, co_ratio)
    m_catio3 = (
        _get_species_molar_mass("Ca")
        + _get_species_molar_mass("Ti")
        + 3.0 * _get_species_molar_mass("O")
    )
    # For CaTiO3, we need to handle pressure dependence
    # The vapour_pressure_catio3 function takes pressure as an argument
    # For now, we'll use a fixed reference pressure - this should be revisited
    p_ref = 1e6  # dyne/cm^2
    return vapour_pressure_catio3(temperatures, p_ref, metallicity) / (
        xcatio3 * mmw / m_catio3
    )


def condensation_temperature_catio3_free(temperatures, x_catio3, mmw=2.33):
    """Calculate CaTiO3 condensation curve for prescribed abundance."""
    m_catio3 = (
        _get_species_molar_mass("Ca")
        + _get_species_molar_mass("Ti")
        + 3.0 * _get_species_molar_mass("O")
    )
    p_ref = 1e6  # dyne/cm^2
    return vapour_pressure_catio3(temperatures, p_ref) / (x_catio3 * mmw / m_catio3)


def condensation_temperature_caal12o19(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate CaAl12O19 condensation curve."""
    xcaal12o19 = calculate_mass_fraction_caal12o19(metallicity, co_ratio)
    m_caal12o19 = (
        _get_species_molar_mass("Ca")
        + 12.0 * _get_species_molar_mass("Al")
        + 19.0 * _get_species_molar_mass("O")
    )
    # For CaAl12O19, we need to handle pressure dependence
    p_ref = 1e6  # dyne/cm^2
    return vapour_pressure_caal12o19(temperatures, p_ref, metallicity) / (
        xcaal12o19 * mmw / m_caal12o19
    )


def condensation_temperature_caal12o19_free(temperatures, x_caal12o19, mmw=2.33):
    """Calculate CaAl12O19 condensation curve for prescribed abundance."""
    m_caal12o19 = (
        _get_species_molar_mass("Ca")
        + 12.0 * _get_species_molar_mass("Al")
        + 19.0 * _get_species_molar_mass("O")
    )
    p_ref = 1e6  # dyne/cm^2
    return vapour_pressure_caal12o19(temperatures, p_ref) / (
        x_caal12o19 * mmw / m_caal12o19
    )


def condensation_temperature_sio2(temperatures, metallicity, co_ratio, mmw=2.33):
    """Calculate SiO2 condensation curve."""
    xsio2 = calculate_mass_fraction_sio2(metallicity, co_ratio)
    m_sio2 = _get_species_molar_mass("Si") + 2.0 * _get_species_molar_mass("O")
    return vapour_pressure_sio2(temperatures, metallicity) / (xsio2 * mmw / m_sio2)


def condensation_temperature_sio2_free(temperatures, x_sio2, mmw=2.33):
    """Calculate SiO2 condensation curve for prescribed abundance."""
    m_sio2 = _get_species_molar_mass("Si") + 2.0 * _get_species_molar_mass("O")
    return vapour_pressure_sio2(temperatures) / (x_sio2 * mmw / m_sio2)


def calculate_cloud_base_pressure_fe(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_fe_combined, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_fe_free(press, temp, x_fe, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_fe_comb_free, x_fe, mmw)
    )


def calculate_cloud_base_pressure_mgsio3(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mgsio3, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_mgsio3_free(press, temp, x_mgsio3, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mgsio3_free, x_mgsio3, mmw)
    )


def calculate_cloud_base_pressure_mg2sio4(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mg2sio4, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_mg2sio4_free(press, temp, x_mg2sio4, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mg2sio4_free, x_mg2sio4, mmw)
    )


def calculate_cloud_base_pressure_na2s(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_na2s, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_na2s_free(press, temp, x_na2s, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_na2s_free, x_na2s, mmw)
    )


def calculate_cloud_base_pressure_kcl(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_kcl, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_kcl_free(press, temp, x_kcl, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_kcl_free, x_kcl, mmw)
    )


def calculate_cloud_base_pressure_sio(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_sio, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_sio_free(press, temp, x_sio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_sio_free, x_sio, mmw)
    )


def calculate_cloud_base_pressure_tio2(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_tio2, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_tio2_free(press, temp, x_tio2, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_tio2_free, x_tio2, mmw)
    )


def calculate_cloud_base_pressure_cr(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_cr, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_cr_free(press, temp, x_cr, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_cr_free, x_cr, mmw)
    )


def calculate_cloud_base_pressure_zns(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_zns, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_zns_free(press, temp, x_zns, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_zns_free, x_zns, mmw)
    )


def calculate_cloud_base_pressure_nh3(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_nh3, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_nh3_free(press, temp, x_nh3, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_nh3_free, x_nh3, mmw)
    )


def calculate_cloud_base_pressure_mns(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mns, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_mns_free(press, temp, x_mns, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_mns_free, x_mns, mmw)
    )


def calculate_cloud_base_pressure_h2o(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_h2o, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_h2o_free(press, temp, x_h2o, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_h2o_free, x_h2o, mmw)
    )


def calculate_cloud_base_pressure_ch4(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_ch4, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_ch4_free(press, temp, x_ch4, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_ch4_free, x_ch4, mmw)
    )


def calculate_cloud_base_pressure_al2o3(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_al2o3, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_al2o3_free(press, temp, x_al2o3, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_al2o3_free, x_al2o3, mmw)
    )


def calculate_cloud_base_pressure_catio3(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_catio3, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_catio3_free(press, temp, x_catio3, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_catio3_free, x_catio3, mmw)
    )


def calculate_cloud_base_pressure_caal12o19(
    press, temp, metallicity, co_ratio, mmw=2.33
):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_caal12o19, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_caal12o19_free(press, temp, x_caal12o19, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_caal12o19_free, x_caal12o19, mmw)
    )


def calculate_cloud_base_pressure_sio2(press, temp, metallicity, co_ratio, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_sio2, metallicity, co_ratio, mmw)
    )


def calculate_cloud_base_pressure_sio2_free(press, temp, x_sio2, mmw=2.33):
    return _calculate_cloud_base_pressure_generic(
        press, temp, (condensation_temperature_sio2_free, x_sio2, mmw)
    )
